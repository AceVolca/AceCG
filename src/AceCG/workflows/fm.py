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
from ..io.forcefield import ReadLmpFF, resolve_source_table_entries
from ..io.tables import export_tables
from ..potentials.bspline import BSplinePotential
from ..schedulers.task_runner import run_post
from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey
from ..trainers import FMTrainerAnalytic
from .base import BaseWorkflow, _run_workflow_cli
from .sampling import _BOLTZMANN_KCAL, resolve_aa_noise_sigma

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
        self._apply_forcefield_specs(self.forcefield)
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

            if spec.init_mode in {"authored_zero", "authored_direct"}:
                potential = _build_authored_potential(spec)
                ff_data.setdefault(canonical_key, []).append(potential)
                rd.setdefault("n_coeffs", spec.model_size)
                rd["min"] = spec.domain[0]
                rd["max"] = spec.domain[1]
                rd["table_min"] = spec.domain[0]
                rd["table_max"] = spec.domain[1]
                rd["resolution"] = spec.resolution
                rd["table_resolution"] = spec.resolution
                if spec.model == "bspline" and "degree" not in rd and spec.model_overrides:
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
        self._append_fixed_priors(ff_data)
        forcefield = Forcefield(ff_data)
        if cfg.system.fixed_forcefield_path:
            forcefield.key_mask = {key: True for key in forcefield}
        return forcefield

    def _append_fixed_priors(self, ff_data: Dict[InteractionKey, list]) -> None:
        """Append fixed LAMMPS priors to the FM forcefield data in-place."""
        fixed_path_raw = self.config.system.fixed_forcefield_path
        if not fixed_path_raw:
            return
        fixed_path = self._resolve_config_path(fixed_path_raw)
        if fixed_path is None:
            raise ValueError("system.fixed_forcefield_path is configured but unresolved.")
        pair_style, sel_styles = parse_pair_style_options(self.config.system.pair_style)
        fixed_ff = ReadLmpFF(
            str(fixed_path),
            pair_style,
            pair_typ_sel=sel_styles,
            cutoff=None,
            table_fit=self.config.system.table_fit or "multigaussian",
            table_fit_overrides=self.config.system.table_fit_overrides,
            topology_arrays=self.topology,
        )
        for key, pot in fixed_ff.iter_potentials():
            pot.param_mask = np.zeros(pot.n_params(), dtype=bool)
            ff_data.setdefault(key, []).append(pot)

    # ── run ─────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the FM training loop via the MPI compute engine."""
        cfg = self.config

        if self._use_solver:
            batch = self._run_post_accumulation(step_index=0)
            if batch is None:
                return {
                    "epochs": 0,
                    "results": [],
                    "validation": self._validation_result_payload(),
                }
            batch["step_index"] = 0
            result = self.trainer_or_solver.solve(batch)
            self.forcefield.update_params(self.trainer_or_solver.get_params())
            table_manifest = self._export_table_bundle()
            if self._validation_enabled():
                self._run_validation_blocking(label="final", epoch=0)
            logger.info("Solver result: %s", result)
            return {
                "epochs": 1,
                "results": [result],
                "table_dir": str(self.output_dir / "tables"),
                "table_manifest": table_manifest,
                "validation": self._validation_result_payload(),
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
            if self._validation_due_after_epoch(epoch):
                self._run_validation_blocking(
                    label=f"epoch_{epoch:04d}",
                    epoch=epoch,
                )
        if self._validation_enabled():
            last_epoch = int(cfg.training.n_epochs) - 1
            if last_epoch < 0 or not self._validation_due_after_epoch(last_epoch):
                self._run_validation_blocking(label="final", epoch=last_epoch)
        table_manifest = self._export_table_bundle() if results else {"tables": {}}
        return {
            "epochs": len(results),
            "results": results,
            "table_dir": str(self.output_dir / "tables"),
            "table_manifest": table_manifest,
            "validation": self._validation_result_payload(),
        }

    def _export_table_bundle(self) -> Dict[str, Any]:
        """Export solved FM tables into the run root for downstream comparisons."""
        if not self._fm_runtime_specs:
            return {"tables": {}}
        export_forcefield = self._trainable_export_forcefield()
        return export_tables(
            {"interactions": self._fm_runtime_specs},
            export_forcefield,
            self.output_dir / "tables",
        )

    def _trainable_export_forcefield(self) -> Forcefield:
        """Return only optimizer-active potentials for FM table export."""
        data: Dict[InteractionKey, list] = {}
        for key, pot in self.forcefield.iter_potentials():
            local_mask = getattr(pot, "param_mask", None)
            if local_mask is None:
                active = True
            else:
                local_mask = local_mask() if callable(local_mask) else local_mask
                active = bool(np.any(np.asarray(local_mask, dtype=bool)))
            if active:
                data.setdefault(key, []).append(pot)
        return Forcefield(data)

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
        noise_spec = self._fm_noise_runtime_spec(step_index)
        if noise_spec is not None:
            spec["noise"] = noise_spec
        if cfg.system.type_names is not None:
            spec["atom_type_name_aliases"] = cfg.system.type_names

        # Frame subsetting
        if cfg.aa_ref.every != 1:
            spec["every"] = cfg.aa_ref.every
        if cfg.aa_ref.skip_frames > 0:
            spec["frame_start"] = cfg.aa_ref.skip_frames
        if cfg.aa_ref.n_frames > 0:
            spec["frame_end"] = cfg.aa_ref.skip_frames + cfg.aa_ref.n_frames
        self._apply_post_runtime_options(spec)

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

    def _fm_noise_runtime_spec(self, step_index: int) -> Optional[Dict[str, Any]]:
        noise = self.config.aa_ref.noise
        if not bool(noise.enabled):
            return None
        stage = int(step_index) // int(noise.update_interval)
        runtime = noise.to_runtime_dict()
        runtime["sigma"] = resolve_aa_noise_sigma(
            noise,
            n_epochs=int(self.config.training.n_epochs),
            stage=stage,
        )
        runtime["seed"] = int(noise.seed) + int(stage)
        if int(noise.subsample_per_epoch) > 0:
            runtime["subsample_seed"] = int(noise.seed) + int(step_index)
        if float(noise.force_mix_ratio) > 0.0:
            temperature = self.config.training.temperature
            if temperature is None or float(temperature) <= 0.0:
                raise ValueError(
                    "noise_force_mix_ratio requires training.temperature or training.beta."
                )
            runtime["beta"] = 1.0 / (_BOLTZMANN_KCAL * float(temperature))
        return runtime

    def _apply_post_runtime_options(self, spec: Dict[str, Any]) -> None:
        if self.config.sampling.perf_trace:
            spec["perf_trace"] = True
            interval = self.config.sampling.extras.get("heartbeat_interval", 25)
            spec["heartbeat_interval"] = int(interval)
        if bool(self.config.sampling.extras.get("perf_trace_all_ranks", False)):
            spec["perf_trace_all_ranks"] = True


# ─── Module-private helpers ───────────────────────────────────────────

def _build_authored_potential(spec: FMInteractionSpec) -> Any:
    if spec.model in {"gaussian", "gauss/cut"}:
        from ..potentials.gaussian import GaussianPotential

        overrides = dict(spec.model_overrides)
        potential = GaussianPotential(
            spec.types[0],
            spec.types[-1],
            float(overrides["A"]),
            float(overrides["r0"]),
            float(overrides["sigma"]),
            float(overrides.get("cutoff", spec.domain[1])),
        )
        setattr(potential, "_acecg_lammps_style", "gauss/cut")
        return potential
    return _build_zero_potential(spec)


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
