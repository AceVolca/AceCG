"""FMWorkflow — Force Matching workflow.

Inherits ``BaseWorkflow`` directly (FM has no sampler / scheduler).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..configs.models import ACGConfig, FMInteractionSpec
from ..configs.parser import validate_fm_spec_domain
from ..configs.utils import parse_pair_style_options
from ..fitters import TABLE_FITTERS
from ..io.logger import get_screen_logger
from ..io.forcefield import resolve_source_table_entries
from ..io.tables import export_tables
from ..potentials.bspline import BSplinePotential
from ..schedulers.task_runner import run_post
from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey
from ..trainers import FMTrainerAnalytic
from .base import BaseWorkflow, _run_workflow_cli

logger = get_screen_logger("fm")


class FMWorkflow(BaseWorkflow):
    """Force-matching workflow.

    Attributes set after ``__init__``:
        forcefield        – trainable ``Forcefield``
        resource_pool     – discovered compute resources
        trainer_or_solver – ``FMTrainerAnalytic`` or ``FMMatrixSolver``
    """

    def __init__(self, config: ACGConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.resource_pool = self._build_resource_pool(sim_cmd=[])
        logger.info("FM resource pool: %s", self.resource_pool)
        self._fm_runtime_specs: List[Dict[str, Any]] = []
        self.forcefield = self._build_fm_forcefield()
        if self.config.system.forcefield_mask is not None:
            self.forcefield.build_mask(init_mask=self._build_forcefield_mask(self.forcefield))
        self._use_solver = self._should_use_solver()
        if self._use_solver:
            self.trainer_or_solver = self._build_solver()
        else:
            self.optimizer = self._build_optimizer(self.forcefield)
            self.trainer_or_solver = self._build_trainer()

    # ── builders ────────────────────────────────────────────────

    def _should_use_solver(self) -> bool:
        fm_method = self.config.training.fm_method
        if fm_method == "solver":
            return True
        if fm_method == "iterator":
            return False
        # auto: use solver when optimizer is Newton (linear one-shot)
        opt_spec = str(
            self.config.training.optimizer or self.config.training.trainer or "newton"
        ).strip().split()[0].lower()
        return opt_spec in {"newton", "newtonraphson", "newton_raphson"}

    def _build_trainer(self) -> FMTrainerAnalytic:
        return FMTrainerAnalytic(
            forcefield=self.forcefield,
            optimizer=self.optimizer,
        )

    def _build_solver(self) -> Any:
        from ..solvers.fm_matrix import FMMatrixSolver
        tcfg = self.config.training
        return FMMatrixSolver(
            self.forcefield,
            mode=tcfg.solver_mode,
            ridge_alpha=tcfg.solver_ridge_alpha,
        )

    def _build_fm_forcefield(
        self,
    ) -> Forcefield:
        """Build the trainable FM forcefield from ``self.config.training.fm_specs``."""
        cfg = self.config
        interactions = cfg.training.fm_specs.flattened()

        ff_data: Dict[InteractionKey, list] = {}
        runtime_specs: List[Dict[str, Any]] = []
        source_entries: Dict[InteractionKey, Dict[str, str]] = {}

        for spec in interactions:
            rd = spec.to_runtime_dict()
            canonical_key = spec.ikey

            if spec.init_mode == "authored_zero":
                potential = _build_zero_potential(spec)
                ff_data[canonical_key] = [potential]
                rd.setdefault("n_coeffs", spec.model_size)
                rd["min"] = spec.domain[0]
                rd["max"] = spec.domain[1]
                rd["table_min"] = spec.domain[0]
                rd["table_max"] = spec.domain[1]
                rd["resolution"] = spec.resolution
                rd["table_resolution"] = spec.resolution
                if "degree" not in rd and spec.model_overrides:
                    degree = spec.model_overrides.get("degree")
                    if degree is not None:
                        rd["degree"] = int(degree)
                runtime_specs.append(rd)
                continue

            # source-table-fit path: resolve source tables lazily
            if not source_entries and (cfg.system.forcefield_path or cfg.training.para_path):
                pair_style, _ = parse_pair_style_options(cfg.system.pair_style)
                ff_path_raw = cfg.system.forcefield_path or cfg.training.para_path
                ff_path = self._resolve_config_path(ff_path_raw)
                if ff_path is None:
                    raise ValueError("FM source-table path is not configured.")
                source_entries = resolve_source_table_entries(
                    str(ff_path),
                    pair_style=pair_style,
                )

            source_key = spec.ikey
            if source_key not in source_entries:
                raise ValueError(
                    f"Source table for FM spec {spec.style}:"
                    f"{':'.join(spec.types)} was not found in "
                    f"{cfg.system.forcefield_path or cfg.training.para_path}."
                )
            entry = source_entries[source_key]
            table_path = entry["table_path"]
            spec_min, spec_max, resolution = validate_fm_spec_domain(
                spec.domain, source_table_path=table_path
            )
            potential = _fit_potential(spec, source_table_path=table_path)
            ff_data[canonical_key] = [potential]
            rd["source_table_path"] = table_path
            rd["table_name"] = entry["table_name"]
            rd["min"] = spec_min
            rd["max"] = spec_max
            rd["table_min"] = spec_min
            rd["table_max"] = spec_max
            rd["resolution"] = resolution
            rd["table_resolution"] = resolution
            runtime_specs.append(rd)

        self._fm_runtime_specs = runtime_specs
        return Forcefield(ff_data)

    # ── run ─────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the FM training loop via the MPI compute engine."""
        cfg = self.config

        if self._use_solver:
            batch = self._run_post_accumulation(step_index=0)
            if batch is None:
                return {"epochs": 0, "results": []}
            batch["step_index"] = 0
            result = self.trainer_or_solver.solve(batch)
            self.forcefield.update_params(self.trainer_or_solver.get_params())
            table_manifest = self._export_table_bundle()
            logger.info("Solver result: %s", result)
            return {
                "epochs": 1,
                "results": [result],
                "table_dir": str(self.output_dir / "tables"),
                "table_manifest": table_manifest,
            }

        # Trainer (iterative) path
        results = []
        for epoch in range(cfg.training.n_epochs):
            batch = self._run_post_accumulation(step_index=epoch)
            if batch is None:
                continue
            batch["step_index"] = epoch
            out = self.trainer_or_solver.step(batch)
            self.forcefield.update_params(self.trainer_or_solver.get_params())
            results.append(out)
            logger.info("Epoch %d: %s", epoch, out)
        table_manifest = self._export_table_bundle() if results else {"tables": {}}
        return {
            "epochs": len(results),
            "results": results,
            "table_dir": str(self.output_dir / "tables"),
            "table_manifest": table_manifest,
        }

    def _export_table_bundle(self) -> Dict[str, Any]:
        """Export solved FM tables into the run root for downstream comparisons."""
        if not self._fm_runtime_specs:
            return {"tables": {}}
        return export_tables(
            {"interactions": self._fm_runtime_specs},
            self.forcefield,
            self.output_dir / "tables",
        )

    def _run_post_accumulation(
        self, *, step_index: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Serialize forcefield, invoke compute engine via scheduler, read batch."""
        cfg = self.config
        work_dir = self.output_dir / f"fm_step_{step_index:04d}"
        work_dir.mkdir(parents=True, exist_ok=True)

        ff_path = work_dir / "forcefield.pkl"
        forcefield_snapshot = self.forcefield
        if self._use_solver and not np.all(self.forcefield.param_mask):
            forcefield_snapshot = Forcefield(self.forcefield)
            forcefield_snapshot.build_mask(
                init_mask=np.ones(forcefield_snapshot.n_params(), dtype=bool)
            )
        with open(ff_path, "wb") as f:
            pickle.dump(forcefield_snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

        output_file = work_dir / "fm_batch.pkl"

        # Build one-pass spec following the no-frame-cache engine contract.
        spec: Dict[str, Any] = {
            "work_dir": str(work_dir),
            "forcefield_path": str(ff_path),
            "topology": str(self._resolve_config_path(cfg.system.topology_file)),
            "trajectory": [
                str(self._resolve_config_path(t))
                for t in cfg.aa_ref.trajectory_files
            ],
            "trajectory_format": cfg.aa_ref.trajectory_format,
            "exclude_bonded": cfg.system.exclude_bonded,
            "exclude_option": cfg.system.exclude_option,
            "cutoff": cfg.system.cutoff,
            "step_index": int(step_index),
            "steps": [
                {
                    "step_mode": "fm",
                    "name": "fm",
                    "output_file": str(output_file),
                }
            ],
        }
        if cfg.system.type_names is not None:
            spec["atom_type_name_aliases"] = cfg.system.type_names

        # Frame subsetting
        if cfg.aa_ref.every != 1:
            spec["every"] = cfg.aa_ref.every
        if cfg.aa_ref.skip_frames > 0:
            spec["frame_start"] = cfg.aa_ref.skip_frames
        if cfg.aa_ref.n_frames > 0:
            spec["frame_end"] = cfg.aa_ref.skip_frames + cfg.aa_ref.n_frames

        # Delegate MPI launch to the scheduler's run_post
        run_post(
            spec,
            self.resource_pool,
            run_dir=work_dir,
            python_exe=cfg.scheduler.python_exe or None,
        )

        if not output_file.exists():
            logger.warning("FM accumulation produced no output at step %d", step_index)
            return None

        with open(output_file, "rb") as f:
            batch = pickle.load(f)
        return batch


# ─── Module-private helpers ───────────────────────────────────────────

def _build_zero_potential(spec: FMInteractionSpec) -> BSplinePotential:
    if spec.model != "bspline":
        raise ValueError(
            f"Source-table-free FM specs only support bspline, got {spec.model!r}."
        )
    n_coeffs = spec.model_size
    degree = int(spec.model_overrides.get("degree", 3))
    minimum, maximum = spec.domain
    knots = BSplinePotential.clamped_uniform_knots(
        minimum, maximum, n_coeffs, degree
    )
    coefficients = np.zeros(n_coeffs, dtype=float)
    return BSplinePotential(
        typ1=spec.types[0],
        typ2=spec.types[-1],
        knots=knots,
        coefficients=coefficients,
        degree=degree,
        cutoff=maximum,
        bonded=(spec.style != "pair"),
    )


def _fit_potential(
    spec: FMInteractionSpec,
    *,
    source_table_path: str,
) -> Any:
    model_overrides = dict(spec.model_overrides)
    if spec.style == "pair" and spec.model == "bspline" and spec.max_force is not None:
        model_overrides["max_force"] = spec.max_force

    if spec.model == "bspline":
        _assert_no_size_override(model_overrides, "n_coeffs", spec.model_size)
        model_overrides["bonded"] = (spec.style != "pair")
        fitter = TABLE_FITTERS.create(
            "bspline", n_coeffs=spec.model_size, **model_overrides
        )
    elif spec.model == "multigaussian":
        _assert_no_size_override(model_overrides, "n_gauss", spec.model_size)
        fitter = TABLE_FITTERS.create(
            "multigaussian", n_gauss=spec.model_size, **model_overrides
        )
    else:
        raise ValueError(f"Unsupported FM model {spec.model!r}.")
    return fitter.fit(
        str(source_table_path), typ1=spec.types[0], typ2=spec.types[-1]
    )


def _assert_no_size_override(
    overrides: dict, key: str, model_size: int
) -> None:
    if key in overrides and int(overrides[key]) != model_size:
        raise ValueError(
            f"FM spec model_size={model_size} conflicts with "
            f"model_overrides['{key}']={overrides[key]!r}."
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``acg-fm`` entry point."""
    return _run_workflow_cli(
        FMWorkflow,
        prog="acg-fm",
        description="Run the AceCG FM workflow.",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
