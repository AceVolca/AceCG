"""Parser for standalone VP Growth workflow config files.

A VP Growth ``.acg`` file is a pre-flight descriptor consumed by
:mod:`AceCG.workflows.vp_growth`. It declares only what the grower needs:

- ``[aa_ref]``: CG-only (no VP) reference topology, AA-mapped trajectory,
  frame selection, and whether to persist reference forces.
- ``[vp]``: output location, runtime knobs, and static topology knobs
  (selection, atomtype_order, clash parameters).
- ``[vp_atoms | vp_bonds | vp_angles | vp_dihedrals | vp_pairs]``: VP atom
  types and bonded/pair interactions. ``[vp_dihedrals]`` is accepted
  for forward compatibility even though the current model does not
  use dihedrals.

The trainer config stack (``parse_acg_file``) is untouched; this module
reuses the low-level ``parse_acg_text`` tokenizer via its
``extra_sections`` parameter so no syntax forking occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

from .parser import (
    ACGConfigError,
    _load_dict_file,
    _parse_scalar_or_literal,
    parse_acg_text,
)
from .vp_config import VPConfig, parse_vp_config


_VP_GROWTH_SECTIONS: frozenset[str] = frozenset(
    {"aa_ref", "vp", "vp_atoms", "vp_bonds", "vp_angles", "vp_dihedrals", "vp_pairs"}
)

_AA_REF_KEYS: frozenset[str] = frozenset(
    {
        "trajectory_files",
        "trajectory_format",
        "ref_topo",
        "ref_topo_type_names",
        "skip_frames",
        "every",
        "n_frames",
        "include_forces",
    }
)

_VP_RUN_KEYS: frozenset[str] = frozenset(
    {
        "output_dir",
        "frame_ids",
        "orientation_seed_base",
        "latent_settings_name",
        "table_points",
        "table_rmin",
        "table_rmax",
        "overwrite",
    }
)

_VP_TOPOLOGY_KEYS: frozenset[str] = frozenset(
    {"selection", "atomtype_order", "clash_max_passes", "clash_min_distance"}
)


@dataclass(frozen=True)
class VPGrowthAARef:
    """AA-reference spec for the VP Growth workflow.

    Attributes
    ----------
    trajectory_files
        One or more AA-mapped CG trajectory files (no VP atoms).
    trajectory_format
        MDAnalysis-compatible format string (default ``LAMMPSDUMP``).
    ref_topo
        CG-only LAMMPS data file. Treated as the "CG (non-VP)" base;
        whatever atoms are present are treated as non-VP.
    ref_topo_type_names
        ``{lammps_type_id: bead_name}`` mapping. Required whenever any
        ``[vp_*]`` key uses bead-name aliases such as ``VP-HG``. Absent
        ⇒ ``[vp_*]`` keys must use bare stringified LAMMPS ids.
    skip_frames, every, n_frames
        Frame window: start at ``skip_frames``, stride ``every``, take
        ``n_frames`` (0 ⇒ until end of trajectory).
    include_forces
        When ``True`` and the trajectory carries forces, the grower
        writes ``frame_{fid:06d}.forces.npy`` alongside each frame data.
    """

    trajectory_files: Tuple[str, ...] = ()
    trajectory_format: str = "LAMMPSDUMP"
    ref_topo: Optional[str] = None
    ref_topo_type_names: Optional[Dict[int, str]] = None
    skip_frames: int = 0
    every: int = 1
    n_frames: int = 0
    include_forces: bool = False


@dataclass(frozen=True)
class VPGrowthRun:
    """Runtime knobs for the VP Growth workflow.

    All paths are resolved relative to the ``.acg`` file's directory when
    the workflow runs; fields stored here are raw strings as authored.
    """

    output_dir: str = ""
    frame_ids: Optional[Tuple[int, ...]] = None
    orientation_seed_base: int = 0
    latent_settings_name: str = "latent.settings"
    table_points: int = 2500
    table_rmin: float = 0.01
    table_rmax: float = 25.0
    overwrite: bool = False


@dataclass(frozen=True)
class VPGrowthConfig:
    """Top-level VP Growth workflow config.

    The three components mirror the three kinds of information the
    grower needs: what to read (``aa_ref``), what to build (``vp``),
    and where to put it (``run``).
    """

    path: Optional[Path] = None
    aa_ref: VPGrowthAARef = field(default_factory=VPGrowthAARef)
    vp: VPConfig = field(default_factory=VPConfig)
    run: VPGrowthRun = field(default_factory=VPGrowthRun)


# ─── Public API ───────────────────────────────────────────────────────


def parse_vp_growth_file(path: str | Path) -> VPGrowthConfig:
    """Load a VP Growth ``.acg`` file into a validated ``VPGrowthConfig``."""
    config_path = Path(path).expanduser().resolve()
    raw = parse_acg_text(
        config_path.read_text(encoding="utf-8"),
        source=str(config_path),
        extra_sections=_VP_GROWTH_SECTIONS,
    )
    return _build_vp_growth_config(raw, path=config_path)


def parse_vp_growth_text(
    text: str, *, source: str = "<memory>", base_dir: Optional[Path] = None
) -> VPGrowthConfig:
    """Parse VP Growth config from raw text (used by tests)."""
    raw = parse_acg_text(text, source=source, extra_sections=_VP_GROWTH_SECTIONS)
    probe_path = (base_dir or Path.cwd()) / "<inline>.acg"
    cfg = _build_vp_growth_config(raw, path=probe_path)
    # In-memory configs keep a ``None`` path so consumers know the file
    # does not exist on disk; base_dir-rooted resolution is callers' job.
    return replace(cfg, path=None)


# ─── Core builder ─────────────────────────────────────────────────────


def _build_vp_growth_config(
    raw: Mapping[str, Mapping[str, Any]], *, path: Path
) -> VPGrowthConfig:
    unknown = set(raw) - _VP_GROWTH_SECTIONS
    if unknown:
        raise ACGConfigError(
            f"VP Growth config {path} has unsupported sections: "
            f"{sorted(unknown)}. Allowed: {sorted(_VP_GROWTH_SECTIONS)}"
        )

    aa_raw: MutableMapping[str, Any] = dict(raw.get("aa_ref", {}))
    vp_raw: MutableMapping[str, Any] = dict(raw.get("vp", {}))
    vp_sub: Dict[str, Dict[str, Any]] = {
        name: dict(raw.get(name, {}))
        for name in ("vp_atoms", "vp_bonds", "vp_angles", "vp_dihedrals", "vp_pairs")
    }

    aa_ref = _build_aa_ref(aa_raw, base_dir=path.parent)
    run = _build_run(vp_raw)
    vp_core = parse_vp_config(vp_sub)
    vp = _apply_vp_topology_overrides(vp_core, vp_raw)

    # Any leftover keys in vp_raw that we didn't recognize?
    if vp_raw:
        raise ACGConfigError(
            f"Unknown [vp] keys in {path}: {sorted(vp_raw)}. "
            f"Allowed: {sorted(_VP_RUN_KEYS | _VP_TOPOLOGY_KEYS)}"
        )

    _validate_alias_consistency(vp, aa_ref)

    return VPGrowthConfig(path=path, aa_ref=aa_ref, vp=vp, run=run)


def _build_aa_ref(
    aa_raw: MutableMapping[str, Any], *, base_dir: Path
) -> VPGrowthAARef:
    unknown = set(aa_raw) - _AA_REF_KEYS
    if unknown:
        raise ACGConfigError(
            f"Unknown [aa_ref] keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_AA_REF_KEYS)}"
        )

    trajs = aa_raw.pop("trajectory_files", ())
    trajectory_files = _normalize_trajectory_files(trajs)
    ref_topo = _pop_optional_str(aa_raw, "ref_topo")
    if ref_topo is None:
        raise ACGConfigError("[aa_ref] ref_topo is required.")

    type_names = _parse_type_names_field(
        aa_raw.pop("ref_topo_type_names", None), base_dir=base_dir,
    )

    return VPGrowthAARef(
        trajectory_files=trajectory_files,
        trajectory_format=str(aa_raw.pop("trajectory_format", "LAMMPSDUMP")),
        ref_topo=ref_topo,
        ref_topo_type_names=type_names,
        skip_frames=int(aa_raw.pop("skip_frames", 0)),
        every=int(aa_raw.pop("every", 1)),
        n_frames=int(aa_raw.pop("n_frames", 0)),
        include_forces=_as_bool(aa_raw.pop("include_forces", False)),
    )


def _build_run(vp_raw: MutableMapping[str, Any]) -> VPGrowthRun:
    output_dir = _pop_optional_str(vp_raw, "output_dir")
    if output_dir is None:
        raise ACGConfigError("[vp] output_dir is required.")

    frame_ids_raw = vp_raw.pop("frame_ids", None)
    frame_ids = _parse_frame_ids(frame_ids_raw)

    return VPGrowthRun(
        output_dir=output_dir,
        frame_ids=frame_ids,
        orientation_seed_base=int(vp_raw.pop("orientation_seed_base", 0)),
        latent_settings_name=str(
            vp_raw.pop("latent_settings_name", "latent.settings")
        ),
        table_points=int(vp_raw.pop("table_points", 2500)),
        table_rmin=float(vp_raw.pop("table_rmin", 0.01)),
        table_rmax=float(vp_raw.pop("table_rmax", 25.0)),
        overwrite=_as_bool(vp_raw.pop("overwrite", False)),
    )


def _apply_vp_topology_overrides(
    vp_core: VPConfig, vp_raw: MutableMapping[str, Any]
) -> VPConfig:
    """Pull static-topology keys out of the ``[vp]`` section into ``VPConfig``.

    Consumes keys in ``_VP_TOPOLOGY_KEYS`` from ``vp_raw``; leaves
    run-time keys (``_VP_RUN_KEYS``) alone for :func:`_build_run`.
    """
    kwargs: Dict[str, Any] = {}
    if "selection" in vp_raw:
        raw_val = vp_raw.pop("selection")
        kwargs["selection"] = (
            None if raw_val is None else str(raw_val).strip() or None
        )
    if "atomtype_order" in vp_raw:
        order = str(vp_raw.pop("atomtype_order")).strip().lower()
        if order not in {"front", "back"}:
            raise ACGConfigError(
                f"[vp] atomtype_order must be 'front' or 'back', got {order!r}."
            )
        kwargs["atomtype_order"] = order
    if "clash_max_passes" in vp_raw:
        kwargs["clash_max_passes"] = int(vp_raw.pop("clash_max_passes"))
    if "clash_min_distance" in vp_raw:
        kwargs["clash_min_distance"] = float(vp_raw.pop("clash_min_distance"))
    return replace(vp_core, **kwargs) if kwargs else vp_core


# ─── Field-level helpers ──────────────────────────────────────────────


def _parse_type_names_field(
    raw: Any, *, base_dir: Path
) -> Optional[Dict[int, str]]:
    """Parse ``[aa_ref] ref_topo_type_names`` into ``{int_id: name}``.

    Mirrors ``parser._parse_type_names`` but dedicated to the VP Growth
    ``aa_ref`` section so evolution of either side stays independent.
    Accepts: ``None`` | dict | csv string ``"HG, MG, T1"`` | JSON file path.
    """
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return {int(k): str(v) for k, v in raw.items()}
    if not isinstance(raw, str):
        raise ACGConfigError(
            f"[aa_ref] ref_topo_type_names must be a dict, comma-separated "
            f"string, or file path; got {type(raw).__name__}."
        )
    text = raw.strip()
    if not text:
        return None
    loaded = _load_dict_file(text, base_dir=base_dir)
    if loaded is not None:
        return {int(k): str(v) for k, v in loaded.items()}
    names = [n.strip() for n in text.split(",") if n.strip()]
    if not names:
        return None
    return {i + 1: name for i, name in enumerate(names)}


def _parse_frame_ids(raw: Any) -> Optional[Tuple[int, ...]]:
    """Parse ``[vp] frame_ids``.

    Accepted forms:
      - absent / ``None`` / ``"all"``  → ``None`` (derive from aa_ref window)
      - tuple/list of ints             → tuple as-is
      - ``"0-499"``                    → inclusive integer range
      - ``"[0, 1, 2]"`` or ``"0, 1, 2"`` → parsed as list/tuple
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    if isinstance(raw, int):
        return (int(raw),)
    if not isinstance(raw, str):
        raise ACGConfigError(
            f"[vp] frame_ids must be a string, list, or int; got {type(raw).__name__}."
        )
    text = raw.strip()
    if not text or text.lower() == "all":
        return None
    if "-" in text and "," not in text and not text.startswith("["):
        lo_s, hi_s = text.split("-", 1)
        lo, hi = int(lo_s.strip()), int(hi_s.strip())
        if hi < lo:
            raise ACGConfigError(
                f"[vp] frame_ids range must satisfy lo <= hi, got {text!r}."
            )
        return tuple(range(lo, hi + 1))
    parsed = _parse_scalar_or_literal(text)
    if isinstance(parsed, (list, tuple)):
        return tuple(int(x) for x in parsed)
    if isinstance(parsed, int):
        return (int(parsed),)
    raise ACGConfigError(
        f"[vp] frame_ids cannot be interpreted as an integer list: {text!r}."
    )


def _normalize_trajectory_files(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if isinstance(raw, Iterable):
        return tuple(str(item) for item in raw)
    raise ACGConfigError(
        f"[aa_ref] trajectory_files must be a string or iterable of strings; "
        f"got {type(raw).__name__}."
    )


def _pop_optional_str(
    mapping: MutableMapping[str, Any], key: str
) -> Optional[str]:
    raw = mapping.pop(key, None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in {"true", "yes", "1", "on"}:
            return True
        if low in {"false", "no", "0", "off", ""}:
            return False
    raise ACGConfigError(f"Cannot interpret {raw!r} as a boolean.")


# ─── Cross-section validation ─────────────────────────────────────────


def _validate_alias_consistency(vp: VPConfig, aa_ref: VPGrowthAARef) -> None:
    """Ensure ``[vp_*]`` type labels are consistently named or numbered.

    With an alias mapping present, every type label must be either a VP
    atom name (from ``[vp_atoms]``) or a value in
    ``aa_ref.ref_topo_type_names``. Without an alias mapping, every
    non-VP label must be a bare stringified positive integer. Mixed mode
    is forbidden: a single config cannot use both aliases and bare ids.
    """
    vp_names = set(vp.vp_names)
    alias_values = (
        set(aa_ref.ref_topo_type_names.values())
        if aa_ref.ref_topo_type_names is not None
        else set()
    )

    all_labels: set[str] = set()
    for group in (vp.bonds, vp.angles, vp.pairs):
        for ix in group:
            all_labels.update(ix.type_keys)
    if vp.default_pair is not None:
        # sentinel, not a type label
        all_labels.discard("default")

    non_vp_labels = all_labels - vp_names - {"default"}
    if not non_vp_labels:
        return

    if aa_ref.ref_topo_type_names is None:
        # No alias table → every non-VP label must be a bare positive integer.
        for lbl in non_vp_labels:
            if not (lbl.isdigit() and int(lbl) > 0):
                raise ACGConfigError(
                    f"[vp_*] references non-VP type label {lbl!r}, but "
                    f"[aa_ref] ref_topo_type_names is not set. Either add "
                    f"an alias mapping or use bare LAMMPS type ids "
                    f"(e.g. '1', '2')."
                )
        return

    # Alias table present: labels must either match an alias value or be
    # a VP name. Bare integers are disallowed in this mode.
    bad = non_vp_labels - alias_values
    bare_ints = {lbl for lbl in bad if lbl.isdigit()}
    if bare_ints:
        raise ACGConfigError(
            f"[vp_*] mixes alias names with bare integer type ids "
            f"{sorted(bare_ints)}; mixed mode is not allowed. Either "
            f"replace them with alias names from ref_topo_type_names "
            f"or drop the alias table entirely."
        )
    unresolved = bad - bare_ints
    if unresolved:
        raise ACGConfigError(
            f"[vp_*] references unknown type labels {sorted(unresolved)}. "
            f"Known VP names: {sorted(vp_names)}. "
            f"Known alias values: {sorted(alias_values)}."
        )
