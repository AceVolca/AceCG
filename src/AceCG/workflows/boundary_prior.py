"""One-shot endpoint-prior workflow for FM output forcefields."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import MDAnalysis as mda

from ..configs.models import ACGConfig
from ..configs.parser import parse_acg_file
from ..configs.utils import parse_pair_style_options
from ..io.forcefield import ReadLmpFF, WriteLmpFF
from ..io.logger import get_screen_logger
from ..potentials.boundary_prior import apply_boundary_prior
from ..topology.topology_array import collect_topology_arrays

logger = get_screen_logger("boundary_prior")

DEFAULT_BOUNDARY_SUMMARY = Path(
    "workspace/reports/reference-pdf-diagnostics/"
    "fm_reg_bayesian_default/support_boundaries/pmf_only_boundary_summary.md"
)


def read_boundary_summary(path: str | Path) -> dict[str, dict[str, float | None]]:
    """Read PMF boundary rows from the reference diagnostics Markdown table."""
    summary_path = Path(path)
    boundaries: dict[str, dict[str, float | None]] = {}
    for raw_line in summary_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5 or cells[0] in {"Label", "---"}:
            continue
        label = cells[0]
        if not any(label.startswith(prefix) for prefix in ("pair:", "bond:", "angle:")):
            continue
        boundaries[label] = {
            "min": _optional_markdown_float(cells[3]),
            "max": _optional_markdown_float(cells[4]),
        }
    if not boundaries:
        raise ValueError(f"No boundary rows found in {summary_path!s}.")
    return boundaries


def boundary_prior_spec_from_config(config: ACGConfig) -> dict[str, Any]:
    """Resolve boundary-prior bounds from config extras or the default report."""
    explicit = config.training.extras.get("boundary_prior")
    if explicit:
        if not isinstance(explicit, Mapping):
            raise ValueError("training.boundary_prior must be a mapping when set.")
        return {str(key): value for key, value in explicit.items()}
    raw_path = config.training.extras.get("boundary_prior_path")
    path = Path(str(raw_path)) if raw_path is not None else DEFAULT_BOUNDARY_SUMMARY
    if config.path is not None and not path.is_absolute():
        candidate = config.path.parent / path
        if candidate.exists():
            path = candidate
    return read_boundary_summary(path)


def run_boundary_prior(config: ACGConfig) -> dict[str, Any]:
    """Apply endpoint priors to the configured FM forcefield and write tables."""
    output_dir = _resolve_config_path(config, config.training.output_dir)
    if output_dir is None:
        raise ValueError("training.output_dir is required.")
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_path = _resolve_config_path(config, config.system.topology_file)
    if topology_path is None:
        raise ValueError("system.topology_file is required.")
    topology_arrays = collect_topology_arrays(
        mda.Universe(str(topology_path)),
        exclude_bonded=config.system.exclude_bonded,
        exclude_option=config.system.exclude_option,
        atom_type_name_aliases=config.system.type_names,
        vp_names=config.vp.vp_names if config.vp is not None else None,
    )

    forcefield_path = _resolve_config_path(
        config,
        config.system.forcefield_path or config.training.para_path,
    )
    if forcefield_path is None:
        raise ValueError("system.forcefield_path or training.para_path is required.")
    pair_style, sel_styles = parse_pair_style_options(config.system.pair_style)
    forcefield = ReadLmpFF(
        str(forcefield_path),
        pair_style,
        pair_typ_sel=sel_styles,
        cutoff=config.system.cutoff,
        table_fit=config.system.table_fit or "bspline",
        table_fit_overrides=config.system.table_fit_overrides,
        topology_arrays=topology_arrays,
    )
    prior_forcefield = apply_boundary_prior(
        forcefield,
        boundary_prior_spec_from_config(config),
        pair_decay=float(config.training.extras.get("boundary_prior_pair_decay", 0.25)),
        pair_strength=float(config.training.extras.get("boundary_prior_pair_strength", 50.0)),
        wall_k_min=float(config.training.extras.get("boundary_prior_wall_k_min", 1.0)),
    )
    output_settings = output_dir / forcefield_path.name
    WriteLmpFF(
        str(forcefield_path),
        str(output_settings),
        prior_forcefield,
        pair_style,
        pair_typ_sel=sel_styles,
        topology_arrays=topology_arrays,
    )
    result = {
        "forcefield_path": str(output_settings),
        "output_dir": str(output_dir),
        "n_interactions": len(prior_forcefield),
    }
    with open(output_dir / "acgreturn.pkl", "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Wrote boundary-prior forcefield to %s", output_settings)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the boundary-prior workflow CLI."""
    parser = argparse.ArgumentParser(
        prog="acg-boundary-prior",
        description="Apply one-shot endpoint priors to an AceCG FM output.",
        allow_abbrev=False,
    )
    parser.add_argument("config", type=str, help="Path to an AceCG .acg file.")
    args = parser.parse_args(argv)
    config = parse_acg_file(args.config)
    run_boundary_prior(config)
    return 0


def _optional_markdown_float(value: str) -> float | None:
    text = str(value).strip()
    if text.lower() in {"", "n/a", "na", "none", "null"}:
        return None
    return float(text)


def _resolve_config_path(config: ACGConfig, value: Any) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        base = config.path.parent if config.path is not None else Path.cwd()
        path = base / path
    return path.resolve(strict=False)


if __name__ == "__main__":
    sys.exit(main())
