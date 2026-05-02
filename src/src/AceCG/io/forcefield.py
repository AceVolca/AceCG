"""Force-field read/write helpers for LAMMPS-style inputs."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..fitters import TABLE_FITTERS
from ..potentials import POTENTIAL_REGISTRY
from ..potentials.base import BasePotential
from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey
from .tables import (
    cap_table_forces,
    estimate_table_fp,
    find_equilibrium,
    parse_lammps_table,
    write_lammps_table,
)
_TRUE_MASK_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_MASK_TOKENS = frozenset({"0", "false", "no", "off"})


def _parse_bool_mask_tokens(tokens: Sequence[str]) -> Optional[np.ndarray]:
    values: list[bool] = []
    for token in tokens:
        lowered = str(token).strip().lower()
        if lowered in _TRUE_MASK_TOKENS:
            values.append(True)
            continue
        if lowered in _FALSE_MASK_TOKENS:
            values.append(False)
            continue
        return None
    return np.asarray(values, dtype=bool)


def _parse_pair_mask_spec(
    tokens: Sequence[str],
    *,
    pair_style: str,
    pair_typ_sel: Optional[Sequence[str]],
) -> Optional[tuple[InteractionKey, list[str]]]:
    if len(tokens) < 3 or tokens[0] != "pair_coeff":
        return None
    style = str(pair_style).strip().lower()
    payload = [str(token) for token in tokens[3:]]
    if style == "hybrid":
        if len(tokens) < 4:
            return None
        style = str(tokens[3]).strip().lower()
        payload = [str(token) for token in tokens[4:]]
    elif payload and payload[0].strip().lower() == style:
        payload = payload[1:]

    if pair_typ_sel is not None:
        allowed = {str(token).strip().lower() for token in pair_typ_sel}
        if style not in allowed:
            return None

    return InteractionKey.pair(tokens[1], tokens[2]), payload


def _parse_bonded_mask_spec(
    tokens: Sequence[str],
) -> Optional[tuple[InteractionKey, list[str]]]:
    if len(tokens) < 3 or tokens[0] not in {"bond_coeff", "angle_coeff"}:
        return None
    kind = tokens[0].split("_", 1)[0]
    return InteractionKey(style=kind, types=(str(tokens[1]),)), [str(token) for token in tokens[3:]]


def _looks_like_mask_payload(payload_tokens: Sequence[str]) -> bool:
    payload = [str(token) for token in payload_tokens]
    if not payload:
        return True
    if payload[0].strip().lower() in {"mask", "unmask"}:
        return True
    return _parse_bool_mask_tokens(payload) is not None


def _read_lmpffmask_spec(
    file: str,
    pair_style: str,
    pair_typ_sel: Optional[Sequence[str]] = None,
) -> tuple[tuple[InteractionKey, tuple[str, ...]], ...]:
    entries: list[tuple[InteractionKey, tuple[str, ...]]] = []
    seen: set[InteractionKey] = set()

    with open(file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            body = raw_line.split("#", 1)[0].strip()
            if not body:
                continue
            tokens = body.split()
            if not tokens:
                continue

            parsed = _parse_pair_mask_spec(
                tokens,
                pair_style=pair_style,
                pair_typ_sel=pair_typ_sel,
            )
            if parsed is None:
                parsed = _parse_bonded_mask_spec(tokens)
            if parsed is None:
                continue

            key, payload = parsed
            if not _looks_like_mask_payload(payload):
                continue
            if key in seen:
                raise ValueError(f"Duplicate mask entry for {key.label()} in {file!r}.")
            seen.add(key)
            entries.append((key, tuple(payload)))

    return tuple(entries)


def ReadLmpFFMask(
    file: str,
    pair_style: str,
    pair_typ_sel: Optional[List[str]] = None,
) -> Any:
    """Read an AceCG forcefield-mask file.

    Parameters
    ----------
    file : str
        Path to the mask file.
    pair_style : str
        LAMMPS pair style in the source forcefield file. Use ``"hybrid"`` for
        hybrid pair coefficient lines.
    pair_typ_sel : list[str], optional
        Pair sub-styles to include when ``pair_style="hybrid"``.

    Returns
    -------
    ForcefieldMaskSpec
        Parsed mask entries grouped by :class:`InteractionKey`.
    """
    from ..configs.models import ForcefieldMaskSpec

    return ForcefieldMaskSpec(entries=_read_lmpffmask_spec(
        file,
        str(pair_style),
        pair_typ_sel=pair_typ_sel,
    ))


def ReadLmpFF(
    file: str,
    pair_style: str,
    pair_typ_sel: Optional[List[str]] = None,
    cutoff: Optional[float] = None,
    global_var: Optional[dict] = None,
    table_fit: str = "multigaussian",
    table_fit_overrides: Optional[dict] = None,
    topology_arrays: Optional[Any] = None,
):
    """Read a LAMMPS-style forcefield file.

    Parameters
    ----------
    file : str
        Input LAMMPS forcefield/include file containing ``*_coeff`` lines.
    pair_style : str
        Active pair style, or ``"hybrid"`` when pair coefficient lines include
        per-pair sub-style names.
    pair_typ_sel : list[str], optional
        Hybrid sub-styles to read. Ignored for non-hybrid pair styles.
    cutoff : float, optional
        Cutoff appended to pair potential constructors when the coefficient
        line does not carry it explicitly.
    global_var : dict, optional
        Global variables required by some LAMMPS styles, such as
        ``lj/cut/soft``.
    table_fit : str, default="multigaussian"
        Registered table fitter used when reading ``table`` coefficients.
    table_fit_overrides : dict, optional
        Extra keyword arguments passed to the table fitter.
    topology_arrays : object, optional
        Topology metadata used to translate numeric bonded/pair type ids into
        canonical :class:`InteractionKey` values.

    Returns
    -------
    Forcefield
        Unified AceCG forcefield container.
    """
    assert pair_style is not None
    if pair_style != "hybrid":
        pair_typ_sel = None
        param_offset = 3
    else:
        param_offset = 4
    if topology_arrays is None:
        warnings.warn(
            "ReadLmpFF called without topology_arrays; bonded InteractionKeys will be "
            "constructed directly from bonded type ids.",
            stacklevel=2,
        )

    base_dir = os.path.dirname(os.path.abspath(file))
    forcefield: Dict[InteractionKey, List[BasePotential]] = {}

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if "pair_coeff" in line:
            tmp = line.split()
            if pair_style == "hybrid":
                style = tmp[3]
            else:
                style = pair_style
            typ1, typ2 = tmp[1], tmp[2]
            pair_typ1 = typ1
            pair_typ2 = typ2
            atom_type_code_to_name = getattr(topology_arrays, "atom_type_code_to_name", None)
            if atom_type_code_to_name is not None and str(typ1).isdigit() and str(typ2).isdigit():
                pair_typ1 = str(atom_type_code_to_name.get(int(typ1), typ1))
                pair_typ2 = str(atom_type_code_to_name.get(int(typ2), typ2))
            pair = InteractionKey.pair(str(pair_typ1), str(pair_typ2))

            if pair_typ_sel is None or style in pair_typ_sel:
                if style == "table":
                    table_file = tmp[param_offset]
                    if not os.path.isabs(table_file):
                        table_file = os.path.join(base_dir, table_file)
                    fitter_overrides = dict(table_fit_overrides or {})
                    if table_fit == "bspline":
                        fitter_overrides["bonded"] = False
                    fitter = TABLE_FITTERS.create(table_fit, **fitter_overrides)
                    pot = fitter.fit(table_file, typ1=pair.types[0], typ2=pair.types[1])
                    forcefield[pair] = [pot]
                elif style in POTENTIAL_REGISTRY:
                    constructor = POTENTIAL_REGISTRY[style]
                    params = list(map(float, tmp[param_offset:]))
                    if style == "double/gauss":
                        if cutoff is None:
                            gauss_params, cutoff = params[:-1], params[-1]
                        else:
                            gauss_params = params[:]
                        forcefield[pair] = [
                            constructor(pair.types[0], pair.types[1], 2, cutoff, gauss_params)
                        ]
                    elif style == "lj/cut/soft":
                        if cutoff is not None:
                            params.append(cutoff)
                        if global_var is None:
                            raise ValueError("global_var is required for lj/cut/soft force fields")
                        params.append(global_var["n"])
                        params.append(global_var["alpha"])
                        forcefield[pair] = [constructor(pair.types[0], pair.types[1], *params)]
                    else:
                        if cutoff is not None:
                            params.append(cutoff)
                        forcefield[pair] = [constructor(pair.types[0], pair.types[1], *params)]

        else:
            for coeff_kw, kind in (("bond_coeff", "bond"), ("angle_coeff", "angle")):
                if coeff_kw not in line:
                    continue
                tmp = line.split()
                if not tmp or tmp[0] != coeff_kw or len(tmp) < 4:
                    continue
                type_num = tmp[1]
                bonded_style = tmp[2]
                bonded_type_id_to_key = getattr(topology_arrays, f"{kind}_type_id_to_key", None)
                key = InteractionKey(style=kind, types=(type_num,))
                if bonded_type_id_to_key is not None and str(type_num).isdigit():
                    key = bonded_type_id_to_key.get(int(type_num) - 1, key)

                if bonded_style == "table" and len(tmp) >= 5:
                    table_file = tmp[3]
                    if not os.path.isabs(table_file):
                        table_file = os.path.join(base_dir, table_file)
                    fitter_overrides = dict(table_fit_overrides or {})
                    if table_fit == "bspline":
                        fitter_overrides["bonded"] = True
                    fitter = TABLE_FITTERS.create(table_fit, **fitter_overrides)
                    pot = fitter.fit(table_file, typ1=key.types[0], typ2=key.types[-1])
                    pot.bonded = True
                    forcefield[key] = [pot]

                elif bonded_style == "harmonic" and len(tmp) >= 5:
                    from ..potentials.harmonic import HarmonicPotential

                    k_val = float(tmp[3])
                    r0_val = float(tmp[4])
                    scale = (np.pi / 180.0) ** 2 if kind == "angle" else 1.0
                    typ1 = key.types[0]
                    typ2 = key.types[-1]
                    typ3 = key.types[1] if kind == "angle" and len(key.types) == 3 else None
                    pot = HarmonicPotential(typ1, typ2, k_val, r0_val, typ3=typ3, scale=scale)
                    pot.bonded = True
                    forcefield[key] = [pot]

    return Forcefield(forcefield)


def WriteLmpFF(
    old_file: str,
    new_file: str,
    forcefield: Forcefield,
    pair_style: str,
    pair_typ_sel: Optional[List[str]] = None,
    topology_arrays: Optional[Any] = None,
):
    """Write forcefield parameters into a LAMMPS-style coefficient file.

    Parameters
    ----------
    old_file : str
        Source coefficient/include file whose structure is preserved.
    new_file : str
        Destination file path.
    forcefield : Forcefield
        Forcefield containing updated parameters.
    pair_style : str
        Active pair style, or ``"hybrid"`` for hybrid coefficient lines.
    pair_typ_sel : list[str], optional
        Hybrid sub-styles to write. Ignored for non-hybrid pair styles.
    topology_arrays : object, optional
        Topology metadata used to translate numeric type ids into canonical
        :class:`InteractionKey` values.
    """
    assert pair_style is not None
    if pair_style != "hybrid":
        pair_typ_sel = None
        param_offset = 3
    else:
        param_offset = 4
    if topology_arrays is None:
        warnings.warn(
            "WriteLmpFF called without topology_arrays; bonded InteractionKeys will be "
            "matched directly against bonded type ids.",
            stacklevel=2,
        )

    old_base_dir = os.path.dirname(os.path.abspath(old_file))
    new_base_dir = os.path.dirname(os.path.abspath(new_file))
    with open(old_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "pair_coeff" in line:
            tmp = line.split()
            if pair_style == "hybrid":
                style = tmp[3]
            else:
                style = pair_style
            lookup_key = InteractionKey.pair(tmp[1], tmp[2])
            atom_type_code_to_name = getattr(topology_arrays, "atom_type_code_to_name", None)
            if atom_type_code_to_name is not None and tmp[1].isdigit() and tmp[2].isdigit():
                lookup_key = InteractionKey.pair(
                    str(atom_type_code_to_name.get(int(tmp[1]), tmp[1])),
                    str(atom_type_code_to_name.get(int(tmp[2]), tmp[2])),
                )

            if pair_typ_sel is None or style in pair_typ_sel:
                if lookup_key in forcefield:
                    pot = forcefield[lookup_key]
                    if isinstance(pot, list):
                        pot = pot[0]
                    if style == "table":
                        table_token = tmp[param_offset]
                        source_table_file = table_token
                        dest_table_file = table_token
                        if not os.path.isabs(table_token):
                            source_table_file = os.path.join(old_base_dir, table_token)
                            dest_table_file = os.path.join(new_base_dir, table_token)

                        Path(dest_table_file).parent.mkdir(parents=True, exist_ok=True)
                        r, _, _ = parse_lammps_table(source_table_file)
                        write_lammps_table(
                            filename=dest_table_file,
                            r=r,
                            V=pot.value(r),
                            F=pot.force(r),
                            comment=f"Table {table_token}: id, r, potential, force",
                            table_name=tmp[param_offset + 1],
                            table_style="pair",
                        )
                        max_force = getattr(pot, "_acecg_max_force", None)
                        if max_force is not None:
                            cap_table_forces(dest_table_file, max_force=max_force)
                    else:
                        n_param = pot.n_params()
                        tmp[param_offset : param_offset + n_param] = map(str, pot.get_params())
                        lines[i] = "   ".join(tmp) + "\n"

        else:
            for coeff_keyword in ("bond_coeff", "angle_coeff"):
                if coeff_keyword not in line:
                    continue
                tmp = line.split()
                if tmp[0] != coeff_keyword:
                    continue
                kind = coeff_keyword.split("_")[0]
                type_num = tmp[1]
                lookup_key = InteractionKey(style=kind, types=(type_num,))
                bonded_type_id_to_key = getattr(topology_arrays, f"{kind}_type_id_to_key", None)
                if bonded_type_id_to_key is not None and type_num.isdigit():
                    lookup_key = bonded_type_id_to_key.get(int(type_num) - 1, lookup_key)
                if lookup_key not in forcefield:
                    continue
                pot = forcefield[lookup_key]
                if isinstance(pot, list):
                    pot = pot[0]
                bonded_style = tmp[2] if len(tmp) > 2 else ""
                if bonded_style == "table":
                    table_token = tmp[3]
                    source_table_file = table_token
                    dest_table_file = table_token
                    if not os.path.isabs(table_token):
                        source_table_file = os.path.join(old_base_dir, table_token)
                        dest_table_file = os.path.join(new_base_dir, table_token)
                    Path(dest_table_file).parent.mkdir(parents=True, exist_ok=True)
                    r, _, _ = parse_lammps_table(source_table_file)
                    V = pot.value(r)
                    F = pot.force(r)
                    table_name = tmp[4] if len(tmp) > 4 else Path(dest_table_file).stem
                    eq = find_equilibrium(r, F)
                    fp = estimate_table_fp(r, F)
                    write_lammps_table(
                        filename=dest_table_file,
                        r=r,
                        V=V,
                        F=F,
                        comment=f"AceCG updated {kind} {table_name}",
                        table_name=table_name,
                        table_style=kind,
                        eq=eq,
                        fp=fp,
                    )
                elif bonded_style == "harmonic":
                    params = list(pot.get_params())
                    tmp[3:] = [f"{p:.8g}" for p in params]
                    lines[i] = "   ".join(tmp) + "\n"

    Path(new_file).parent.mkdir(parents=True, exist_ok=True)
    with open(new_file, "w", encoding="utf-8") as f:
        f.writelines(lines)


def resolve_source_table_entries(
    source_forcefield_path: str,
    *,
    pair_style: str = "table",
) -> Dict[InteractionKey, Dict[str, str]]:
    """Parse a LAMMPS settings file to extract table-style interaction entries.

    Returns a mapping from ``InteractionKey`` to a dict with keys
    ``table_path``, ``table_name``, ``style``.
    """
    settings_path = Path(source_forcefield_path).resolve()
    entries: Dict[InteractionKey, Dict[str, str]] = {}

    for raw_line in settings_path.read_text(encoding="utf-8").splitlines():
        body = raw_line.split("#", 1)[0].strip()
        if not body:
            continue
        tokens = body.split()
        if not tokens:
            continue

        keyword = tokens[0].lower()
        if keyword == "pair_coeff":
            if pair_style == "hybrid":
                if len(tokens) < 6:
                    continue
                source_style = tokens[3].lower()
                table_token = tokens[4]
                table_name = tokens[5]
            else:
                if len(tokens) < 5:
                    continue
                source_style = pair_style.lower()
                table_token = tokens[3]
                table_name = tokens[4]
            if source_style != "table":
                continue
            key = InteractionKey.pair(tokens[1], tokens[2])
            table_path = _resolve_table_path(settings_path, table_token)
            entries[key] = {
                "table_path": str(table_path),
                "table_name": str(table_name),
                "style": "pair",
            }
            continue

        if keyword in {"bond_coeff", "angle_coeff"}:
            if len(tokens) < 5:
                continue
            kind = keyword.split("_", 1)[0]
            source_style = tokens[2].lower()
            if source_style != "table":
                continue
            key = InteractionKey(style=kind, types=(str(tokens[1]),))
            table_path = _resolve_table_path(settings_path, tokens[3])
            table_name = (
                tokens[4] if len(tokens) > 4 else Path(str(tokens[3])).stem
            )
            entries[key] = {
                "table_path": str(table_path),
                "table_name": str(table_name),
                "style": kind,
            }

    return entries


def _resolve_table_path(settings_path: Path, table_token: str) -> Path:
    token_path = Path(table_token)
    if token_path.is_absolute():
        return token_path.resolve()
    return (settings_path.parent / token_path).resolve()


__all__ = [
    "FFParamArray",
    "FFParamIndexMap",
    "ReadLmpFF",
    "ReadLmpFFMask",
    "WriteLmpFF",
    "resolve_source_table_entries",
]
