"""Structured VP (virtual-particle) configuration models.

Parsed from ``[vp_atoms]``, ``[vp_bonds]``, ``[vp_angles]``, ``[vp_dihedrals]``,
``[vp_pairs]`` sub-sections in a ``.acg`` file, or from a standalone VP
config file referenced by ``[vp] file = path``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class VPAtomDef:
    """One VP atom-type definition from ``[vp_atoms]``."""

    type_label: str
    mass: float


@dataclass(frozen=True)
class VPInteractionDef:
    """One VP bonded/pair interaction from ``[vp_bonds|angles|pairs]``."""

    type_keys: Tuple[str, ...]
    pot_style: str
    pot_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VPConfig:
    """Structured VP configuration.

    Attributes
    ----------
    atoms : tuple of VPAtomDef
        VP atom types and masses from ``[vp_atoms]``.
    bonds : tuple of VPInteractionDef
        VP bond interactions from ``[vp_bonds]``.
    angles : tuple of VPInteractionDef
        VP angle interactions from ``[vp_angles]``.
    dihedrals : tuple of VPInteractionDef
        VP dihedral interactions from ``[vp_dihedrals]``. Each entry
        is a 4-label key ``A-B-C-D`` with exactly one VP label. The
        current model does not use dihedrals, but the plumbing must
        flow through so a future config can add them without code
        changes upstream of :class:`VPTopologyTemplate`.
    pairs : tuple of VPInteractionDef
        VP pair interactions from ``[vp_pairs]``.
    default_pair : VPInteractionDef or None
        Default pair interaction for missing pairs (``default = ...``).
    selection : str or None
        MDAnalysis selection string identifying CG atoms that carry a VP
        (e.g. ``"resname DOPC"``). ``None`` ⇒ every CG atom is a carrier.
    atomtype_order : str
        How newly grown VP atoms are ordered in the LAMMPS data file.
        ``"front"`` places VP type ids before existing CG types;
        ``"back"`` appends them.
    clash_max_passes : int
        Max iterations of the anti-clash re-orientation loop.
    clash_min_distance : float
        Minimum inter-atom distance (Å) enforced by the anti-clash pass.
    """

    atoms: Tuple[VPAtomDef, ...] = ()
    bonds: Tuple[VPInteractionDef, ...] = ()
    angles: Tuple[VPInteractionDef, ...] = ()
    dihedrals: Tuple[VPInteractionDef, ...] = ()
    pairs: Tuple[VPInteractionDef, ...] = ()
    default_pair: Optional[VPInteractionDef] = None
    selection: Optional[str] = None
    atomtype_order: str = "front"
    clash_max_passes: int = 12
    clash_min_distance: float = 2.0

    @property
    def vp_names(self) -> Tuple[str, ...]:
        """VP atom-type labels, derived from ``atoms``."""
        return tuple(a.type_label for a in self.atoms)


def parse_vp_config(sections: Dict[str, Dict[str, str]]) -> VPConfig:
    """Build a ``VPConfig`` from parsed VP sub-sections.

    Parameters
    ----------
    sections : dict
        Keys are ``"vp_atoms"``, ``"vp_bonds"``, ``"vp_angles"``,
        ``"vp_dihedrals"``, ``"vp_pairs"``.
        Values are ``{key: raw_value}`` dicts from the ``.acg`` parser.
    """
    atoms = _parse_atoms(sections.get("vp_atoms", {}))
    bonds = _parse_interactions(sections.get("vp_bonds", {}))
    angles = _parse_interactions(sections.get("vp_angles", {}))
    dihedrals = _parse_interactions(sections.get("vp_dihedrals", {}))
    pairs_raw = dict(sections.get("vp_pairs", {}))
    default_pair = None
    if "default" in pairs_raw:
        default_pair = _parse_one_interaction(("default",), pairs_raw.pop("default"))
    pairs = _parse_interactions(pairs_raw)
    return VPConfig(
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        pairs=pairs,
        default_pair=default_pair,
    )


# ── internal helpers ──────────────────────────────────────────────────


def _parse_atoms(raw: Dict[str, str]) -> Tuple[VPAtomDef, ...]:
    result = []
    for label, mass_str in raw.items():
        result.append(VPAtomDef(type_label=label, mass=float(mass_str)))
    return tuple(result)


def _parse_interactions(
    raw: Dict[str, str],
) -> Tuple[VPInteractionDef, ...]:
    result = []
    for key_str, value_str in raw.items():
        type_keys = tuple(key_str.split("-"))
        result.append(_parse_one_interaction(type_keys, value_str))
    return tuple(result)


def _parse_one_interaction(
    type_keys: Tuple[str, ...], value_str: str
) -> VPInteractionDef:
    tokens = value_str.split()
    if not tokens:
        raise ValueError(f"Empty interaction definition for {'-'.join(type_keys)}")
    pot_style = tokens[0]
    kwargs: Dict[str, Any] = {}
    remaining = tokens[1:]
    # LAMMPS ``pair_coeff ... table <filename> ...`` carries the table
    # filename as a positional argument; accept it here and expose it as
    # the ``file`` kwarg so downstream code has a canonical name.
    if pot_style == "table" and remaining and "=" not in remaining[0]:
        kwargs["file"] = remaining.pop(0)
    # Remaining tokens: either ``key=value`` or bare positional values.
    # Positional values accumulate under ``coeffs`` (list of coerced
    # scalars) so that styles without well-defined kwarg names — in
    # particular ``dihedral_style harmonic`` (``K d n``) and other
    # LAMMPS styles with mixed int/float signatures — can be authored
    # positionally. Mixing is allowed: later ``key=value`` tokens can
    # still override specific kwargs.
    coeffs: list[Any] = []
    for token in remaining:
        if "=" in token:
            k, v = token.split("=", 1)
            kwargs[k] = _coerce_value(v)
        else:
            coeffs.append(_coerce_value(token))
    if coeffs:
        kwargs["coeffs"] = tuple(coeffs)
    return VPInteractionDef(type_keys=type_keys, pot_style=pot_style, pot_kwargs=kwargs)


def _coerce_value(s: str) -> Any:
    """Coerce a string value to int, float, or leave as str."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s
