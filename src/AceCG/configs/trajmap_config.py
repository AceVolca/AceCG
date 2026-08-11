"""Parser for the standalone trajectory-mapping (``trajmap``) workflow config.

A trajmap ``.acg`` file describes one AA→CG mapping job for
:mod:`AceCG.workflows.trajmap`. A fourth optional section selects a fitted
force map; without it the authored ``f-weight`` behavior is unchanged::

    [aa]
    topology         = NVT.gro          # any MDAnalysis-readable topology
    trajectory_files = NVT.trr          # one path or a list of segments
    trajectory_format = TRR
    skip_frames      = 0
    every            = 1
    n_frames         = 0                # 0 → to the end
    frame_ids        =                  # explicit list/range, overrides the window
    include_forces   = true
    include_velocities = false

    [mapping]
    map_file      = martini_dppc.yaml   # OpenMSCG cgyaml-format mapping
    index_base    = 0
    mol_reference = first               # first | anchor | <local atom index>
    cg_topology   =                     # optional YAML holding only cg-topology
    resname       = DPPC                # fallback only; the YAML wins

    [force_mapping]
    method        = optimal_linear      # fixed | constraint_aware_uniform | optimal_linear
    scope         = per_template        # share one map across molecular copies
    backend       = compare             # native | osqp | compare | auto
    constraints   = h-bonds             # none | h-bonds | all-bonds | auto
    constraint_algorithm = LINCS

    [trajmap]
    output_dir       = results/cgmap
    trajectory_name  = cg.trr           # XTC (positions) or TRR (+forces/vels)
    topology_name    = cg.data          # LAMMPS data written from the first frame
    write_gro        = true
    unwrap           = molecule         # molecule | bead | none | deprecated
    wrap             = true
    triclinic        = exact            # exact | fast
    precision        = float64          # working precision of the weighted sums
    merge_segments   = true
    keep_segments    = false
    overwrite        = false

Only ``[aa] trajectory_files``, ``[mapping] map_file`` and ``[trajmap]
output_dir`` are required; ``[aa] topology`` may be omitted for XTC/TRR input. The tokenizer is the shared
:func:`~AceCG.configs.parser.parse_acg_text`, reused through its
``extra_sections`` hook, so no config syntax is forked.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Union

from .parser import ACGConfigError, _parse_scalar_or_literal, parse_acg_text
from .utils import parse_bool_token


__all__ = [
    "TrajMapForceMapping",
    "TrajMapAA",
    "TrajMapConfig",
    "TrajMapMapping",
    "TrajMapRun",
    "parse_trajmap_file",
    "parse_trajmap_text",
]


_TRAJMAP_SECTIONS: frozenset[str] = frozenset(
    {"aa", "mapping", "force_mapping", "trajmap"}
)

_AA_KEYS: frozenset[str] = frozenset(
    {
        "topology",
        "topology_format",
        "trajectory_files",
        "trajectory_format",
        "skip_frames",
        "every",
        "n_frames",
        "frame_ids",
        "include_forces",
        "include_velocities",
    }
)

_MAPPING_KEYS: frozenset[str] = frozenset(
    {
        "map_file",
        "index_base",
        "mol_reference",
        "cg_topology",
        "resname",
        "strict_weights",
        "use_topology_masses",
    }
)

_RUN_KEYS: frozenset[str] = frozenset(
    {
        "output_dir",
        "trajectory_name",
        "topology_name",
        "write_gro",
        "unwrap",
        "wrap",
        "triclinic",
        "precision",
        "merge_segments",
        "keep_segments",
        "overwrite",
        "report_name",
    }
)

_FORCE_MAPPING_KEYS: frozenset[str] = frozenset(
    {
        "method",
        "scope",
        "backend",
        "l2_regularization",
        "constraints",
        "constraint_algorithm",
        "constraint_pairs_file",
        "constraint_threshold",
        "fit_every",
        "fit_n_frames",
        "fit_frame_ids",
        "artifact_name",
        "max_covariance_bytes",
    }
)

_UNWRAP_MODES: frozenset[str] = frozenset({"molecule", "bead", "none", "deprecated"})
_PRECISIONS: Dict[str, str] = {
    "float32": "float32",
    "single": "float32",
    "float64": "float64",
    "double": "float64",
}
# Output containers we can write losslessly and MDAnalysis can read back
# without a topology hint.
_TRAJECTORY_SUFFIXES: Dict[str, str] = {".xtc": "XTC", ".trr": "TRR"}


@dataclass(frozen=True)
class TrajMapAA:
    """The all-atom input: topology, trajectory segments, frame window.

    Attributes
    ----------
    topology
        AA topology path. Anything MDAnalysis reads — ``.gro``, ``.tpr``,
        ``.pdb``, ``.data``. Only atom count, names, and masses are used, so it
        may be omitted for an XTC/TRR trajectory: those carry their own atom
        count, and a mapping whose site types all declare ``x-weight`` needs no
        masses. Omitting it skips one topology parse *per rank*, which is worth
        having when the topology is a 1.2-million-atom ``.gro``.
    topology_format
        Explicit MDAnalysis topology format; inferred from the suffix when
        omitted (``.data`` → ``DATA``).
    trajectory_files
        One or more AA trajectory segments, in time order.
    trajectory_format
        MDAnalysis format string; inferred from the first segment's suffix when
        omitted.
    skip_frames, every, n_frames
        Frame window: start at ``skip_frames``, stride ``every``, take
        ``n_frames`` frames (``0`` ⇒ to the end).
    frame_ids
        Explicit global frame ids. Takes precedence over the window.
    include_forces, include_velocities
        Map the AA forces / velocities as well. Requires a trajectory that
        carries them (TRR does, XTC does not) and a TRR output.
    """

    topology: Optional[str] = None
    topology_format: Optional[str] = None
    trajectory_files: Tuple[str, ...] = ()
    trajectory_format: Optional[str] = None
    skip_frames: int = 0
    every: int = 1
    n_frames: int = 0
    frame_ids: Optional[Tuple[int, ...]] = None
    include_forces: bool = False
    include_velocities: bool = False


@dataclass(frozen=True)
class TrajMapMapping:
    """The mapping definition and how to compile it.

    Attributes
    ----------
    map_file
        Mapping YAML in OpenMSCG ``cgyaml`` format, optionally carrying an
        AceCG ``cg-topology`` block.
    index_base
        ``0`` when the YAML's ``index``/``anchor`` values are 0-based (what
        ``cgyaml`` writes), ``1`` for 1-based files.
    mol_reference
        Which atom of a repeat unit anchors ``unwrap="molecule"``: ``"first"``
        (lowest mapped index), ``"anchor"`` (the unit origin), or an explicit
        local atom offset.
    cg_topology
        Optional separate YAML holding just a ``cg-topology`` block, for when the
        CG bonded topology is decided after the mapping file was written. An
        empty string means "ignore any block in ``map_file``".
    resname
        Fallback residue name for the emitted CG records: one name, or one per
        ``system`` group. Only used when ``map_file`` carries no residue-form
        ``cg-topology`` block. A YAML that declares residues wins, because a
        system can hold several species — two protein chains, a lipid mixture, a
        protein-lipid system — and one config-level name cannot describe that.
    strict_weights
        Reject rather than renormalize weight tables that look inconsistent.
    use_topology_masses
        Feed the AA topology's masses to the centre-of-mass fallback for site
        types that declare no ``x-weight``.
    """

    map_file: Optional[str] = None
    index_base: int = 0
    mol_reference: Union[str, int] = "first"
    cg_topology: Optional[str] = None
    resname: Union[str, Tuple[str, ...]] = "CG"
    strict_weights: bool = True
    use_topology_masses: bool = True


@dataclass(frozen=True)
class TrajMapForceMapping:
    """Optional statistically optimal AA→CG force aggregation stage."""

    method: str = "fixed"
    scope: str = "auto"
    backend: str = "auto"
    l2_regularization: float = 0.0
    constraints: str = "auto"
    constraint_algorithm: str = "LINCS"
    constraint_pairs_file: Optional[str] = None
    constraint_threshold: float = 1.0e-3
    fit_every: int = 1
    fit_n_frames: int = 0
    fit_frame_ids: Optional[Tuple[int, ...]] = None
    artifact_name: str = "force_map.npz"
    max_covariance_bytes: int = 2 * 1024**3

    @property
    def enabled(self) -> bool:
        return self.method != "fixed"


@dataclass(frozen=True)
class TrajMapRun:
    """Output location and mapping-kernel knobs.

    Attributes
    ----------
    output_dir
        Run directory. Every product lands here; relative paths resolve against
        the ``.acg`` file's directory.
    trajectory_name
        CG trajectory file name. ``.xtc`` keeps positions only; ``.trr`` also
        stores forces and velocities.
    topology_name
        CG LAMMPS data file, written from the first mapped frame.
    write_gro
        Also write a ``.gro`` of the first mapped frame, for visual inspection.
    unwrap, wrap, triclinic, precision
        Passed to :class:`~AceCG.compute.cgmap.CGMapper`.
    merge_segments
        Concatenate the per-rank segment files into ``trajectory_name`` on rank 0
        once mapping finishes. When ``False`` the segments plus a manifest are
        the output, which AceCG's own ``trajectory_files`` lists consume directly.
    keep_segments
        Keep the per-rank segments after a successful merge.
    overwrite
        Allow writing into a directory that already holds a mapped trajectory.
    report_name
        JSON run report (timings, per-rank slices, validation counters).
    """

    output_dir: str = ""
    trajectory_name: str = "cg.xtc"
    topology_name: str = "cg.data"
    write_gro: bool = True
    unwrap: str = "molecule"
    wrap: bool = True
    triclinic: str = "exact"
    precision: str = "float64"
    merge_segments: bool = True
    keep_segments: bool = False
    overwrite: bool = False
    report_name: str = "trajmap_report.json"

    @property
    def trajectory_format(self) -> str:
        """MDAnalysis writer format implied by ``trajectory_name``'s suffix."""
        suffix = Path(self.trajectory_name).suffix.lower()
        try:
            return _TRAJECTORY_SUFFIXES[suffix]
        except KeyError:
            raise ACGConfigError(
                f"[trajmap] trajectory_name must end in one of "
                f"{sorted(_TRAJECTORY_SUFFIXES)}; got {self.trajectory_name!r}."
            ) from None


@dataclass(frozen=True)
class TrajMapConfig:
    """Top-level trajmap workflow config."""

    path: Optional[Path] = None
    aa: TrajMapAA = field(default_factory=TrajMapAA)
    mapping: TrajMapMapping = field(default_factory=TrajMapMapping)
    force_mapping: TrajMapForceMapping = field(default_factory=TrajMapForceMapping)
    run: TrajMapRun = field(default_factory=TrajMapRun)


# ─── Public API ───────────────────────────────────────────────────────


def parse_trajmap_file(path: Union[str, Path]) -> TrajMapConfig:
    """Load a trajmap ``.acg`` file into a validated :class:`TrajMapConfig`."""
    config_path = Path(path).expanduser().resolve()
    raw = parse_acg_text(
        config_path.read_text(encoding="utf-8"),
        source=str(config_path),
        extra_sections=_TRAJMAP_SECTIONS,
    )
    return _build_trajmap_config(raw, path=config_path)


def parse_trajmap_text(text: str, *, source: str = "<memory>") -> TrajMapConfig:
    """Parse a trajmap config from raw text (used by tests)."""
    raw = parse_acg_text(text, source=source, extra_sections=_TRAJMAP_SECTIONS)
    cfg = _build_trajmap_config(raw, path=Path.cwd() / "<inline>.acg")
    # In-memory configs keep ``path=None`` so consumers know there is no file to
    # resolve relative paths against.
    return replace(cfg, path=None)


# ─── Core builder ─────────────────────────────────────────────────────


def _build_trajmap_config(
    raw: Mapping[str, Mapping[str, Any]], *, path: Path
) -> TrajMapConfig:
    unknown = set(raw) - _TRAJMAP_SECTIONS
    if unknown:
        raise ACGConfigError(
            f"trajmap config {path} has unsupported sections: {sorted(unknown)}. "
            f"Allowed: {sorted(_TRAJMAP_SECTIONS)}"
        )
    aa = _build_aa(dict(raw.get("aa", {})))
    mapping = _build_mapping(dict(raw.get("mapping", {})))
    force_mapping = _build_force_mapping(dict(raw.get("force_mapping", {})))
    run = _build_run(dict(raw.get("trajmap", {})))
    _validate_combination(aa, force_mapping, run)
    return TrajMapConfig(
        path=path,
        aa=aa,
        mapping=mapping,
        force_mapping=force_mapping,
        run=run,
    )


def _build_aa(aa_raw: MutableMapping[str, Any]) -> TrajMapAA:
    _reject_unknown(aa_raw, _AA_KEYS, "[aa]")

    topology = _pop_optional_str(aa_raw, "topology")
    trajectory_files = _normalize_paths(
        aa_raw.pop("trajectory_files", ()), "[aa] trajectory_files"
    )
    if not trajectory_files:
        raise ACGConfigError("[aa] trajectory_files is required.")

    every = int(aa_raw.pop("every", 1))
    if every < 1:
        raise ACGConfigError(f"[aa] every must be >= 1, got {every}.")
    skip_frames = int(aa_raw.pop("skip_frames", 0))
    if skip_frames < 0:
        raise ACGConfigError(f"[aa] skip_frames must be >= 0, got {skip_frames}.")
    n_frames = int(aa_raw.pop("n_frames", 0))
    if n_frames < 0:
        raise ACGConfigError(f"[aa] n_frames must be >= 0 (0 = all), got {n_frames}.")

    return TrajMapAA(
        topology=topology,
        topology_format=_pop_optional_str(aa_raw, "topology_format"),
        trajectory_files=trajectory_files,
        trajectory_format=_pop_optional_str(aa_raw, "trajectory_format"),
        skip_frames=skip_frames,
        every=every,
        n_frames=n_frames,
        frame_ids=_parse_frame_ids(aa_raw.pop("frame_ids", None)),
        include_forces=_as_bool(aa_raw.pop("include_forces", False), "[aa] include_forces"),
        include_velocities=_as_bool(
            aa_raw.pop("include_velocities", False), "[aa] include_velocities"
        ),
    )


def _build_mapping(map_raw: MutableMapping[str, Any]) -> TrajMapMapping:
    _reject_unknown(map_raw, _MAPPING_KEYS, "[mapping]")

    map_file = _pop_optional_str(map_raw, "map_file")
    if map_file is None:
        raise ACGConfigError("[mapping] map_file is required.")

    index_base = int(map_raw.pop("index_base", 0))
    if index_base not in (0, 1):
        raise ACGConfigError(f"[mapping] index_base must be 0 or 1, got {index_base}.")

    # ``cg_topology`` distinguishes "absent" (use whatever map_file holds) from
    # an explicit empty value (ignore the block entirely), so the raw presence of
    # the key matters and `_pop_optional_str` alone is not enough.
    if "cg_topology" in map_raw:
        cg_topology = str(map_raw.pop("cg_topology") or "").strip()
    else:
        cg_topology = None

    return TrajMapMapping(
        map_file=map_file,
        index_base=index_base,
        mol_reference=_parse_mol_reference(map_raw.pop("mol_reference", "first")),
        cg_topology=cg_topology,
        resname=_parse_resname(map_raw.pop("resname", "CG")),
        strict_weights=_as_bool(
            map_raw.pop("strict_weights", True), "[mapping] strict_weights"
        ),
        use_topology_masses=_as_bool(
            map_raw.pop("use_topology_masses", True), "[mapping] use_topology_masses"
        ),
    )


def _build_force_mapping(
    force_raw: MutableMapping[str, Any],
) -> TrajMapForceMapping:
    _reject_unknown(force_raw, _FORCE_MAPPING_KEYS, "[force_mapping]")
    method = str(force_raw.pop("method", "fixed")).strip().lower()
    if method not in {"fixed", "constraint_aware_uniform", "optimal_linear"}:
        raise ACGConfigError(
            "[force_mapping] method must be fixed, constraint_aware_uniform, "
            f"or optimal_linear; got {method!r}."
        )
    scope = str(force_raw.pop("scope", "auto")).strip().lower().replace("-", "_")
    if scope == "per_group":
        scope = "per_template"
    if scope not in {"auto", "global", "per_template"}:
        raise ACGConfigError(
            "[force_mapping] scope must be auto, global, or per_template."
        )
    backend = str(force_raw.pop("backend", "auto")).strip().lower()
    if backend not in {"auto", "native", "osqp", "compare"}:
        raise ACGConfigError(
            "[force_mapping] backend must be auto, native, osqp, or compare."
        )
    constraints = str(force_raw.pop("constraints", "auto")).strip().lower()
    if constraints not in {"auto", "none", "h-bonds", "all-bonds"}:
        raise ACGConfigError(
            "[force_mapping] constraints must be auto, none, h-bonds, or all-bonds."
        )
    algorithm = str(force_raw.pop("constraint_algorithm", "LINCS")).strip().upper()
    if algorithm not in {"LINCS", "SHAKE"}:
        raise ACGConfigError(
            "[force_mapping] constraint_algorithm must be LINCS or SHAKE."
        )
    regularization = float(force_raw.pop("l2_regularization", 0.0))
    if regularization < 0.0:
        raise ACGConfigError("[force_mapping] l2_regularization must be non-negative.")
    threshold = float(force_raw.pop("constraint_threshold", 1.0e-3))
    if threshold <= 0.0:
        raise ACGConfigError("[force_mapping] constraint_threshold must be positive.")
    fit_every = int(force_raw.pop("fit_every", 1))
    fit_n_frames = int(force_raw.pop("fit_n_frames", 0))
    if fit_every < 1 or fit_n_frames < 0:
        raise ACGConfigError(
            "[force_mapping] fit_every must be >= 1 and fit_n_frames >= 0."
        )
    max_bytes = int(force_raw.pop("max_covariance_bytes", 2 * 1024**3))
    if max_bytes <= 0:
        raise ACGConfigError("[force_mapping] max_covariance_bytes must be positive.")
    artifact_name = str(force_raw.pop("artifact_name", "force_map.npz")).strip()
    if not artifact_name.lower().endswith(".npz") or Path(artifact_name).name != artifact_name:
        raise ACGConfigError(
            "[force_mapping] artifact_name must be a local filename ending in .npz."
        )
    return TrajMapForceMapping(
        method=method,
        scope=scope,
        backend=backend,
        l2_regularization=regularization,
        constraints=constraints,
        constraint_algorithm=algorithm,
        constraint_pairs_file=_pop_optional_str(force_raw, "constraint_pairs_file"),
        constraint_threshold=threshold,
        fit_every=fit_every,
        fit_n_frames=fit_n_frames,
        fit_frame_ids=_parse_frame_ids(force_raw.pop("fit_frame_ids", None)),
        artifact_name=artifact_name,
        max_covariance_bytes=max_bytes,
    )


def _build_run(run_raw: MutableMapping[str, Any]) -> TrajMapRun:
    _reject_unknown(run_raw, _RUN_KEYS, "[trajmap]")

    output_dir = _pop_optional_str(run_raw, "output_dir")
    if output_dir is None:
        raise ACGConfigError("[trajmap] output_dir is required.")

    unwrap = str(run_raw.pop("unwrap", "molecule")).strip().lower()
    if unwrap not in _UNWRAP_MODES:
        raise ACGConfigError(
            f"[trajmap] unwrap must be one of {sorted(_UNWRAP_MODES)}, got {unwrap!r}."
        )
    triclinic = str(run_raw.pop("triclinic", "exact")).strip().lower()
    if triclinic not in ("exact", "fast"):
        raise ACGConfigError(
            f"[trajmap] triclinic must be 'exact' or 'fast', got {triclinic!r}."
        )
    precision_raw = str(run_raw.pop("precision", "float64")).strip().lower()
    if precision_raw not in _PRECISIONS:
        raise ACGConfigError(
            f"[trajmap] precision must be one of {sorted(_PRECISIONS)}, "
            f"got {precision_raw!r}."
        )

    run = TrajMapRun(
        output_dir=output_dir,
        trajectory_name=str(run_raw.pop("trajectory_name", "cg.xtc")).strip(),
        topology_name=str(run_raw.pop("topology_name", "cg.data")).strip(),
        write_gro=_as_bool(run_raw.pop("write_gro", True), "[trajmap] write_gro"),
        unwrap=unwrap,
        wrap=_as_bool(run_raw.pop("wrap", True), "[trajmap] wrap"),
        triclinic=triclinic,
        precision=_PRECISIONS[precision_raw],
        merge_segments=_as_bool(
            run_raw.pop("merge_segments", True), "[trajmap] merge_segments"
        ),
        keep_segments=_as_bool(
            run_raw.pop("keep_segments", False), "[trajmap] keep_segments"
        ),
        overwrite=_as_bool(run_raw.pop("overwrite", False), "[trajmap] overwrite"),
        report_name=str(run_raw.pop("report_name", "trajmap_report.json")).strip(),
    )
    run.trajectory_format  # validates the suffix now rather than after mapping
    return run


def _validate_combination(
    aa: TrajMapAA, force_mapping: TrajMapForceMapping, run: TrajMapRun
) -> None:
    """Reject request combinations the writer cannot honour."""
    if run.trajectory_format == "XTC":
        wanted = [
            name
            for name, on in (
                ("include_forces", aa.include_forces),
                ("include_velocities", aa.include_velocities),
            )
            if on
        ]
        if wanted:
            raise ACGConfigError(
                f"[aa] {' and '.join(wanted)} require a TRR output; "
                f"[trajmap] trajectory_name={run.trajectory_name!r} writes XTC, "
                "which stores positions only."
            )
    if run.keep_segments and not run.merge_segments:
        raise ACGConfigError(
            "[trajmap] keep_segments only applies when merge_segments = true; "
            "unmerged runs always keep their segments."
        )
    if force_mapping.enabled:
        if not aa.include_forces:
            raise ACGConfigError(
                "[force_mapping] fitted methods require [aa] include_forces = true."
            )
        if run.trajectory_format != "TRR":
            raise ACGConfigError(
                "[force_mapping] fitted methods require a force-bearing TRR output."
            )
        if force_mapping.constraints in {"h-bonds", "all-bonds"} and aa.topology is None:
            raise ACGConfigError(
                "[force_mapping] topology-derived constraints require [aa] topology."
            )


# ─── Field-level helpers ──────────────────────────────────────────────


def _parse_mol_reference(raw: Any) -> Union[str, int]:
    """Parse ``[mapping] mol_reference``: ``first``, ``anchor``, or an int."""
    if isinstance(raw, bool):
        raise ACGConfigError("[mapping] mol_reference must be 'first', 'anchor', or an int.")
    if isinstance(raw, int):
        return int(raw)
    text = str(raw).strip().lower()
    if text in ("first", "anchor"):
        return text
    try:
        return int(text)
    except ValueError:
        raise ACGConfigError(
            "[mapping] mol_reference must be 'first', 'anchor', or an integer local "
            f"atom offset; got {raw!r}."
        ) from None


def _parse_resname(raw: Any) -> Union[str, Tuple[str, ...]]:
    """One residue name, or one per ``system`` group."""
    if isinstance(raw, (list, tuple)):
        names = tuple(str(item).strip() for item in raw)
        if not names or any(not name for name in names):
            raise ACGConfigError("[mapping] resname list must not contain empty names.")
        return names
    text = str(raw).strip()
    if not text:
        raise ACGConfigError("[mapping] resname must not be empty.")
    parsed = _parse_scalar_or_literal(text)
    if isinstance(parsed, (list, tuple)):
        return tuple(str(item).strip() for item in parsed)
    return text


def _parse_frame_ids(raw: Any) -> Optional[Tuple[int, ...]]:
    """Parse ``[aa] frame_ids``: ``None``/``all``, a list, or an inclusive ``lo-hi``."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    if isinstance(raw, int):
        return (int(raw),)
    text = str(raw).strip()
    if not text or text.lower() == "all":
        return None
    if "-" in text and "," not in text and not text.startswith("["):
        lo_text, hi_text = text.split("-", 1)
        lo, hi = int(lo_text.strip()), int(hi_text.strip())
        if hi < lo:
            raise ACGConfigError(f"[aa] frame_ids range needs lo <= hi, got {text!r}.")
        return tuple(range(lo, hi + 1))
    parsed = _parse_scalar_or_literal(text)
    if isinstance(parsed, (list, tuple)):
        return tuple(int(x) for x in parsed)
    if isinstance(parsed, int):
        return (int(parsed),)
    raise ACGConfigError(f"[aa] frame_ids is not an integer list: {text!r}.")


def _normalize_paths(raw: Any, label: str) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ()
        parsed = _parse_scalar_or_literal(text)
        if isinstance(parsed, (list, tuple)):
            return tuple(str(item).strip() for item in parsed)
        return (text,)
    if isinstance(raw, Iterable):
        return tuple(str(item).strip() for item in raw)
    raise ACGConfigError(
        f"{label} must be a path or a list of paths; got {type(raw).__name__}."
    )


def _pop_optional_str(mapping: MutableMapping[str, Any], key: str) -> Optional[str]:
    raw = mapping.pop(key, None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _as_bool(raw: Any, label: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        parsed = parse_bool_token(raw)
        if parsed is not None:
            return parsed
    raise ACGConfigError(f"{label} must be a boolean; got {raw!r}.")


def _reject_unknown(
    raw: Mapping[str, Any], allowed: frozenset[str], label: str
) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ACGConfigError(
            f"Unknown {label} keys: {sorted(unknown)}. Allowed: {sorted(allowed)}"
        )
