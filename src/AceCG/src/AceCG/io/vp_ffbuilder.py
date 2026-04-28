"""Emit the ``latent.settings`` LAMMPS include for a VP-grown topology.

The output file is consumed by CDREM/CDFM as part of ``conditioning.input``:
a hand-authored ``in.cdrem_*.lmp`` file does ``include latent.settings``
inside its ``pair_coeff`` / ``bond_coeff`` block so VP parameters are
pulled in without leaking into the trained force-field.

Pipeline
--------
1. :func:`build_vp_forcefield` — materialize a :class:`Forcefield` from a
   :class:`VPConfig`. Harmonic bonds / angles and every registered pair
   style are supported; an unsupported style raises
   :class:`NotImplementedError`.
2. :func:`render_vp_latent_template` — render the final LAMMPS include
    text directly from that :class:`Forcefield` using canonical
    :class:`InteractionKey` lookups.
3. :func:`write_latent_settings` — emit the table files required by
    ``astable=yes`` / bare ``table`` interactions and write the final
    ``latent.settings`` include alongside them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..configs.vp_config import VPConfig, VPInteractionDef
from ..potentials import POTENTIAL_REGISTRY
from ..potentials.base import BasePotential
from ..potentials.harmonic import HarmonicPotential
from ..potentials.multi_gaussian import MultiGaussianPotential
from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey
from ..topology.vpgrower import VPTopologyTemplate
from .tables import (
    estimate_table_fp,
    find_equilibrium,
    interaction_table_stem,
    write_lammps_table,
)


# ─── Public API ─────────────────────────────────────────────────────


def build_vp_forcefield(vp_config: VPConfig, template: VPTopologyTemplate) -> Forcefield:
    """Materialize the VP-only :class:`Forcefield` from a :class:`VPConfig`.

    Keys are canonical :class:`InteractionKey` instances; values are
    single-element lists carrying a :class:`BasePotential`. Bond / angle
    labels in ``vp_config`` are expected to have been resolved to
    template names during :meth:`VPGrower.from_universe`; here we trust
    :class:`VPTopologyTemplate` for the final name-level mapping.
    """
    ff: Dict[InteractionKey, List[BasePotential]] = {}

    # Bonds — HarmonicPotential only (per §A4).
    for spec in template.vp_bond_specs:
        pot = HarmonicPotential(spec.vp_name, spec.carrier_name, spec.k, spec.r0, scale=1.0)
        pot.bonded = True
        ff[InteractionKey.bond(spec.vp_name, spec.carrier_name)] = [pot]

    # Angles — HarmonicPotential with scale=(π/180)² (§A4).
    scale_angle = (np.pi / 180.0) ** 2
    for spec in template.vp_angle_specs:
        labels = spec.labels
        key = InteractionKey.angle(labels[0], labels[1], labels[2])
        pot = HarmonicPotential(
            labels[0], labels[2], spec.k, spec.theta0_deg,
            typ3=labels[1], scale=scale_angle,
        )
        pot.bonded = True
        ff[key] = [pot]

    for key, interaction in _resolve_vp_pair_definitions(vp_config, template).items():
        ff[key] = [_build_pair_potential(key, interaction)]

    return Forcefield(ff)


def render_vp_latent_template(
    vp_config: VPConfig,
    template: VPTopologyTemplate,
    *,
    default_cutoff: float,
) -> str:
    """Render the final latent-settings text.

    The emitted lines are final LAMMPS ``pair_coeff`` / ``bond_coeff`` /
    ``angle_coeff`` entries keyed by the canonical :class:`InteractionKey`
    mapping in :func:`build_vp_forcefield`.
    """
    lines: List[str] = []

    vp_names = set(vp_config.vp_names)
    pair_defs = _resolve_vp_pair_definitions(vp_config, template)
    vp_ff = build_vp_forcefield(vp_config, template)
    for key, interaction in sorted(
        pair_defs.items(),
        key=lambda item: tuple(
            sorted(
                (template.type2id[item[0].types[0]], template.type2id[item[0].types[1]])
            )
        ),
    ):
        pair_ids = tuple(
            sorted((template.type2id[key.types[0]], template.type2id[key.types[1]]))
        )
        display_types = key.types
        if key.types[0] not in vp_names and key.types[1] in vp_names:
            display_types = (key.types[1], key.types[0])
        if _is_table(interaction):
            cutoff = float(default_cutoff)
            for token in ("cutoff", "rcut", "r_c"):
                if token in interaction.pot_kwargs:
                    cutoff = float(interaction.pot_kwargs[token])
                    break
            default_filename = f"Pair_{display_types[0]}-{display_types[1]}.table"
            table_filename = str(
                default_filename
                if interaction is vp_config.default_pair
                else interaction.pot_kwargs.get("file", default_filename)
            )
            table_name = f"{display_types[0]}-{display_types[1]}"
            lines.append(
                f"   pair_coeff   {pair_ids[0]}  {pair_ids[1]} table   "
                f"{table_filename} {table_name} {cutoff:g}"
            )
            continue

        pot = vp_ff[key][0]
        pair_tokens = [
            str(interaction.pot_style).strip(),
            *[_format_lammps_token(value) for value in pot.get_params()],
        ]
        if str(interaction.pot_style).strip() != "soft":
            for token in ("cutoff", "rcut", "r_c"):
                if token in interaction.pot_kwargs:
                    pair_tokens.append(
                        _format_lammps_token(float(interaction.pot_kwargs[token]))
                    )
                    break
        lines.append(
            f"   pair_coeff   {pair_ids[0]}  {pair_ids[1]} {' '.join(pair_tokens)}"
        )
    if pair_defs:
        lines.append("")

    bond_type_ids = {
        key: int(tid) for tid, key in template.bond_type_key_by_id.items()
    }
    seen_bond_ids: set[int] = set()
    for spec in template.vp_bond_specs:
        key = InteractionKey.bond(spec.vp_name, spec.carrier_name)
        tid = bond_type_ids.get(key)
        if tid is None or tid in seen_bond_ids:
            continue
        seen_bond_ids.add(tid)
        if spec.astable:
            stem = interaction_table_stem("bond", (spec.vp_name, spec.carrier_name))
            lines.append(f"   bond_coeff   {tid}  table  {stem}.table {stem}")
        else:
            params = " ".join(_format_lammps_token(value) for value in vp_ff[key][0].get_params())
            lines.append(f"   bond_coeff   {tid}  harmonic  {params}")
    angle_type_ids = {
        key: int(tid) for tid, key in template.angle_type_key_by_id.items()
    }
    seen_angle_ids: set[int] = set()
    for spec in template.vp_angle_specs:
        key = InteractionKey.angle(*spec.labels)
        tid = angle_type_ids.get(key)
        if tid is None or tid in seen_angle_ids:
            continue
        seen_angle_ids.add(tid)
        if spec.astable:
            stem = interaction_table_stem("angle", spec.labels)
            lines.append(f"   angle_coeff  {tid}  table  {stem}.table {stem}")
        else:
            params = " ".join(_format_lammps_token(value) for value in vp_ff[key][0].get_params())
            lines.append(f"   angle_coeff  {tid}  harmonic  {params}")

    # Dihedral lines — emitted with their final coefficient values
    # directly. The model does not currently train VP dihedrals, so
    # :class:`WriteLmpFF` is not extended to rewrite ``dihedral_coeff``;
    # we instead freeze the user-supplied numbers into the template so
    # they pass through unchanged.
    for tid, spec in _vp_dihedral_tid_spec_pairs(template):
        lines.append(_render_dihedral_coeff(tid, spec))

    return "\n".join(lines) + "\n"


def write_latent_settings(
    *,
    template: VPTopologyTemplate,
    vp_config: VPConfig,
    output_dir: str | Path,
    table_points: int,
    table_rmin: float,
    table_rmax: float,
    include_name: str = "latent.settings",
) -> Path:
    """Emit ``<output_dir>/<include_name>`` plus initial VP tables.

    Returns the path to the final ``latent.settings`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_grid = np.linspace(float(table_rmin), float(table_rmax), int(table_points), dtype=float)
    angle_grid = np.linspace(0.0, 180.0, int(table_points), dtype=float)

    vp_names = set(vp_config.vp_names)
    pair_defs = _resolve_vp_pair_definitions(vp_config, template)
    vp_ff = build_vp_forcefield(vp_config, template)

    for key, interaction in sorted(
        pair_defs.items(),
        key=lambda item: tuple(
            sorted(
                (template.type2id[item[0].types[0]], template.type2id[item[0].types[1]])
            )
        ),
    ):
        if not _is_table(interaction):
            continue
        display_types = key.types
        if key.types[0] not in vp_names and key.types[1] in vp_names:
            display_types = (key.types[1], key.types[0])
        default_filename = f"Pair_{display_types[0]}-{display_types[1]}.table"
        table_filename = str(
            default_filename
            if interaction is vp_config.default_pair
            else interaction.pot_kwargs.get("file", default_filename)
        )
        table_name = f"{display_types[0]}-{display_types[1]}"
        pot = vp_ff[key][0]
        write_lammps_table(
            filename=output_dir / table_filename,
            r=pair_grid,
            V=pot.value(pair_grid),
            F=pot.force(pair_grid),
            comment=(
                f"VP initial pair table {display_types[0]}-"
                f"{display_types[1]} (style={interaction.pot_style})"
            ),
            table_name=table_name,
            table_style="pair",
        )

    bond_type_ids = {
        key: int(tid) for tid, key in template.bond_type_key_by_id.items()
    }
    seen_bond_ids: set[int] = set()
    for spec in template.vp_bond_specs:
        tid = bond_type_ids.get(InteractionKey.bond(spec.vp_name, spec.carrier_name))
        if tid is None or tid in seen_bond_ids:
            continue
        seen_bond_ids.add(tid)
        if not spec.astable:
            continue
        stem = interaction_table_stem("bond", (spec.vp_name, spec.carrier_name))
        pot = vp_ff[InteractionKey.bond(spec.vp_name, spec.carrier_name)][0]
        values = pot.value(pair_grid)
        forces = pot.force(pair_grid)
        write_lammps_table(
            filename=output_dir / f"{stem}.table",
            r=pair_grid,
            V=values,
            F=forces,
            comment=f"VP initial bond table {spec.vp_name}-{spec.carrier_name}",
            table_name=stem,
            table_style="bond",
            eq=find_equilibrium(pair_grid, forces),
            fp=estimate_table_fp(pair_grid, forces),
        )

    angle_type_ids = {
        key: int(tid) for tid, key in template.angle_type_key_by_id.items()
    }
    seen_angle_ids: set[int] = set()
    for spec in template.vp_angle_specs:
        tid = angle_type_ids.get(InteractionKey.angle(*spec.labels))
        if tid is None or tid in seen_angle_ids:
            continue
        seen_angle_ids.add(tid)
        if not spec.astable:
            continue
        stem = interaction_table_stem("angle", spec.labels)
        pot = vp_ff[InteractionKey.angle(*spec.labels)][0]
        values = pot.value(angle_grid)
        forces = pot.force(angle_grid)
        write_lammps_table(
            filename=output_dir / f"{stem}.table",
            r=angle_grid,
            V=values,
            F=forces,
            comment=f"VP initial angle table {'-'.join(spec.labels)}",
            table_name=stem,
            table_style="angle",
            eq=find_equilibrium(angle_grid, forces),
            fp=estimate_table_fp(angle_grid, forces),
        )

    # 2. Render the final settings file.
    settings_text = render_vp_latent_template(
        vp_config, template, default_cutoff=float(table_rmax),
    )
    final_path = output_dir / include_name
    final_path.write_text(settings_text)
    return final_path


# ─── Pair resolution ────────────────────────────────────────────────


def _resolve_vp_pair_definitions(
    vp_config: VPConfig,
    template: VPTopologyTemplate,
) -> Dict[InteractionKey, VPInteractionDef]:
    """Resolve explicit/default VP pair config onto canonical pair keys."""
    explicit: Dict[InteractionKey, VPInteractionDef] = {}
    for interaction in vp_config.pairs:
        if len(interaction.type_keys) != 2:
            raise ValueError(
                f"[vp_pairs] entry {interaction.type_keys!r} must have two type labels."
            )
        explicit[InteractionKey.pair(interaction.type_keys[0], interaction.type_keys[1])] = interaction

    pair_defs: Dict[InteractionKey, VPInteractionDef] = {}
    for vp_name in vp_config.vp_names:
        for other in template.type2id:
            key = InteractionKey.pair(vp_name, str(other))
            if key in pair_defs:
                continue
            interaction = explicit.get(key, vp_config.default_pair)
            if interaction is not None:
                pair_defs[key] = interaction
    return pair_defs


def _is_table(pdef: VPInteractionDef) -> bool:
    if pdef.pot_style == "table":
        return True
    astable = pdef.pot_kwargs.get("astable")
    if astable is None:
        return False
    if isinstance(astable, bool):
        return astable
    if isinstance(astable, str):
        return astable.strip().lower() in {"yes", "true", "1", "on"}
    if isinstance(astable, (int, float)):
        return bool(astable)
    return False


# ─── Pair potential construction ────────────────────────────────────


def _build_pair_potential(key: InteractionKey, interaction: VPInteractionDef) -> BasePotential:
    """Instantiate a :class:`BasePotential` from a pair entry."""
    style = str(interaction.pot_style).strip()
    a, b = key.types
    kwargs = interaction.pot_kwargs

    def pair_cutoff(*, default: Optional[float] = None) -> float:
        for token in ("cutoff", "rcut", "r_c"):
            if token in kwargs:
                return float(kwargs[token])
        if default is not None:
            return float(default)
        raise ValueError(
            f"VP pair {key.label()} with pot_style={style!r} requires an explicit cutoff/rcut/r_c."
        )

    def first_present(*tokens: str, default: object) -> object:
        for token in tokens:
            if token in kwargs:
                return kwargs[token]
        return default

    if style == "table":
        # Interpret initial table via MultiGaussianPotential with a
        # single zero-amplitude gaussian (constant-zero initial V/F).
        # Downstream training will regenerate tables from learned
        # parameters.
        return MultiGaussianPotential(a, b, n_gauss=1, cutoff=pair_cutoff(default=25.0))

    if style not in POTENTIAL_REGISTRY:
        raise NotImplementedError(
            f"VP pair pot_style={style!r} not in POTENTIAL_REGISTRY."
        )

    ctor = POTENTIAL_REGISTRY[style]
    if style in {"gauss/cut", "gauss/wall"}:
        # GaussianPotential(typ1, typ2, A, r0, sigma, cutoff)
        params = (
            float(kwargs.get("H", kwargs.get("A", 0.0))),
            float(kwargs.get("rmh", kwargs.get("r0", 0.0))),
            float(kwargs.get("sigma", 1.0)),
            pair_cutoff(),
        )
        return ctor(a, b, *params)

    if style in {"lj/cut", "lj96/cut"}:
        # LennardJonesPotential(typ1, typ2, epsilon, sigma, cutoff)
        params = (
            float(first_present("eps", "epsilon", default=0.0)),
            float(kwargs.get("sigma", 1.0)),
            pair_cutoff(),
        )
        return ctor(a, b, *params)

    if style == "soft":
        params = (
            float(first_present("A", default=0.0)),
            pair_cutoff(),
        )
        return ctor(a, b, *params)

    if style == "lj/cut/soft":
        # LennardJonesSoftPotential(typ1, typ2, epsilon, sigma, lam, cutoff, n, alpha_LJ)
        params = (
            float(first_present("eps", "epsilon", "A", default=0.0)),
            float(first_present("sigma", default=1.0)),
            float(first_present("lam", "lambda", "lambda_", default=1.0)),
            pair_cutoff(),
            int(first_present("n", default=2)),
            float(first_present("alpha_LJ", "alpha", default=0.5)),
        )
        return ctor(a, b, *params)

    if style == "harmonic":
        # Unusual as a pair style but registered; pass k, r0.
        return ctor(
            a, b,
            float(kwargs.get("k", 0.0)),
            float(kwargs.get("r0", 0.0)),
        )

    if style == "srlr_gauss":
        # SRLRGaussianPotential(typ1, typ2, A, B, C, D, cutoff)
        params = (
            float(kwargs.get("A", 0.0)),
            float(kwargs.get("B", 0.0)),
            float(kwargs.get("C", 0.0)),
            float(kwargs.get("D", 0.0)),
            pair_cutoff(),
        )
        return ctor(a, b, *params)

    if style == "double/gauss":
        # UnnormalizedMultiGaussianPotential signature differs — fall
        # back to the ``(typ1, typ2, n_gauss, cutoff, init_params)``
        # common shape used by MultiGaussianPotential.
        return ctor(a, b, 2, pair_cutoff(), None)

    raise NotImplementedError(f"VP pair pot_style={style!r} construction not implemented.")


def _vp_dihedral_tid_spec_pairs(
    template: VPTopologyTemplate,
) -> List[Tuple[int, "object"]]:
    """Pair each ``VPDihedralSpec`` with its LAMMPS dihedral type id.

    The type id is looked up by reversing
    :attr:`VPTopologyTemplate.dihedral_type_key_by_id`. A spec without a
    matching type id (e.g. one whose VP carrier atom is absent from
    every residue) is silently dropped — :meth:`VPGrower._build_template`
    will not have allocated a type id for it.
    """
    key_to_tid: Dict[InteractionKey, int] = {
        key: int(tid) for tid, key in template.dihedral_type_key_by_id.items()
    }
    pairs: List[Tuple[int, object]] = []
    seen: set[int] = set()
    for spec in template.vp_dihedral_specs:
        key = InteractionKey.dihedral(*spec.labels)
        tid = key_to_tid.get(key)
        if tid is None or tid in seen:
            continue
        seen.add(tid)
        pairs.append((tid, spec))
    pairs.sort(key=lambda p: p[0])
    return pairs


def _render_dihedral_coeff(tid: int, spec: "object") -> str:
    """Emit a LAMMPS ``dihedral_coeff <id> <style> <params>`` line.

    Parameter ordering rules (in order of precedence):

    1. ``spec.pot_kwargs["coeffs"]`` — a positional tuple parsed from
       the config value string. Used verbatim.
    2. For known styles (``harmonic`` / ``cvff`` / ``charmm`` / ``opls``
       / ``multi/harmonic``), a style-specific ordered extraction from
       named kwargs.
    3. Fallback: emit every non-``coeffs`` kwarg in insertion order.
    """
    style = spec.pot_style
    kwargs = dict(spec.pot_kwargs)
    coeffs = kwargs.pop("coeffs", None)
    if coeffs is not None:
        params = list(coeffs)
    else:
        params = _extract_dihedral_params_by_style(style, kwargs)
    tokens = [_format_lammps_token(p) for p in params]
    return f"   dihedral_coeff   {tid}  {style}  {' '.join(tokens)}"


def _extract_dihedral_params_by_style(
    style: str, kwargs: Dict[str, object]
) -> List[object]:
    """Map named kwargs to LAMMPS positional order for common dihedral styles."""
    if style == "harmonic":
        # LAMMPS: dihedral_coeff <id> harmonic K d n  (K float, d ±1, n int)
        return [
            float(kwargs.get("K", kwargs.get("k", 0.0))),
            int(kwargs.get("d", 1)),
            int(kwargs.get("n", 1)),
        ]
    if style == "cvff":
        # K d n
        return [
            float(kwargs.get("K", kwargs.get("k", 0.0))),
            int(kwargs.get("d", 1)),
            int(kwargs.get("n", 1)),
        ]
    if style == "charmm":
        # K n d weight
        return [
            float(kwargs.get("K", kwargs.get("k", 0.0))),
            int(kwargs.get("n", 1)),
            int(kwargs.get("d", 0)),
            float(kwargs.get("weight", kwargs.get("w", 0.0))),
        ]
    if style == "opls":
        # K1 K2 K3 K4
        return [
            float(kwargs.get("K1", 0.0)),
            float(kwargs.get("K2", 0.0)),
            float(kwargs.get("K3", 0.0)),
            float(kwargs.get("K4", 0.0)),
        ]
    if style == "multi/harmonic":
        # A1 A2 A3 A4 A5
        return [float(kwargs.get(f"A{i}", 0.0)) for i in range(1, 6)]
    # Generic fallback: emit every kwarg in insertion order.
    return list(kwargs.values())


def _format_lammps_token(value: object) -> str:
    """Format a single coefficient value as a LAMMPS token.

    Integers are emitted without a decimal point so that signature
    fields like the ``d`` and ``n`` in ``dihedral_style harmonic``
    remain valid for the LAMMPS parser.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)
