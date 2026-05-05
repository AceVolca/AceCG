"""DSMWorkflow -- denoising score matching on noisy reference coordinates."""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..configs.models import ACGConfig
from ..io.logger import get_screen_logger
from ..schedulers.task_runner import run_post
from .base import _run_workflow_cli
from .fm import FMWorkflow
from .sampling import _BOLTZMANN_KCAL, resolve_aa_noise_sigma

logger = get_screen_logger("dsm")


class DSMWorkflow(FMWorkflow):
    """Iterative DSM workflow using noisy-coordinate synthetic force targets."""

    def __init__(self, config: ACGConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.beta = self._derive_beta()

    def _derive_beta(self) -> float:
        temperature = self.config.training.temperature
        if temperature is None or temperature <= 0.0:
            raise ValueError("DSM requires a positive training.temperature or training.beta.")
        return 1.0 / (_BOLTZMANN_KCAL * float(temperature))

    def _should_use_solver(self) -> bool:
        if self.config.training.fm_method == "solver":
            raise ValueError("DSM is iterative and does not support training.fm_method='solver'.")
        return False

    def run(self) -> Dict[str, Any]:
        """Execute iterative DSM training."""
        cfg = self.config
        start_epoch = int(cfg.training.start_epoch)
        n_epochs = int(cfg.training.n_epochs)
        results = []
        if start_epoch > 0:
            prev_ff_dir = self._checkpoint_dir_for_epoch(start_epoch - 1)
            prev_snapshot = prev_ff_dir / "workflow_checkpoint.pkl"
            if not prev_snapshot.exists():
                raise FileNotFoundError(
                    f"Cannot resume DSM from epoch {start_epoch}: "
                    f"checkpoint {prev_snapshot} not found."
                )
            self._load_workflow_checkpoint(prev_ff_dir)
            self.trainer_or_solver = self._build_trainer()
            logger.info(
                "Resuming DSM from epoch %d (loaded %s)",
                start_epoch,
                prev_snapshot,
            )

        for epoch in range(start_epoch, n_epochs):
            batch = self._run_post_accumulation(step_index=epoch)
            if batch is None:
                continue
            batch["step_index"] = epoch
            out = self.trainer_or_solver.step(batch)
            self.forcefield.update_params(self.trainer_or_solver.get_params())
            results.append(out)
            logger.info("Epoch %d: %s", epoch, out)
            ff_dir = self._checkpoint_dir_for_epoch(epoch)
            self._snapshot_optimizer(ff_dir)
            self._write_workflow_checkpoint(ff_dir)
        table_manifest = self._export_table_bundle() if results else {"tables": {}}
        return {
            "epochs": len(results),
            "results": results,
            "table_dir": str(self.output_dir / "tables"),
            "table_manifest": table_manifest,
        }

    def _run_post_accumulation(
        self, *, step_index: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Run DSM accumulation and return an FM-style statistics batch."""
        cfg = self.config
        work_dir = self.output_dir / f"dsm_step_{step_index:04d}"
        work_dir.mkdir(parents=True, exist_ok=True)

        ff_path = work_dir / "forcefield.pkl"
        with open(ff_path, "wb") as handle:
            pickle.dump(self.forcefield, handle, protocol=pickle.HIGHEST_PROTOCOL)

        output_file = work_dir / "dsm_batch.pkl"
        topology_path = (
            self._resolve_config_path(cfg.aa_ref.ref_topo)
            if cfg.aa_ref.ref_topo is not None
            else self._resolve_config_path(cfg.system.topology_file)
        )
        if topology_path is None:
            raise ValueError("DSM requires system.topology_file or aa_ref.ref_topo.")

        spec: Dict[str, Any] = {
            "work_dir": str(work_dir),
            "forcefield_path": str(ff_path),
            "topology": str(topology_path),
            "trajectory": [
                str(self._resolve_config_path(path))
                for path in cfg.aa_ref.trajectory_files
            ],
            "trajectory_format": cfg.aa_ref.trajectory_format,
            "exclude_bonded": cfg.system.exclude_bonded,
            "exclude_option": cfg.system.exclude_option,
            "cutoff": cfg.system.cutoff,
            "step_index": int(step_index),
            "steps": [
                {
                    "step_mode": "dsm",
                    "name": "dsm",
                    "output_file": str(output_file),
                }
            ],
            "noise": self._dsm_noise_runtime_spec(step_index),
        }
        if cfg.aa_ref.ref_resolved_aliases is not None:
            spec["atom_type_name_aliases"] = cfg.aa_ref.ref_resolved_aliases
        elif cfg.system.type_names is not None:
            spec["atom_type_name_aliases"] = cfg.system.type_names
        if cfg.aa_ref.ref_has_vp and cfg.vp is not None:
            spec["vp_names"] = list(cfg.vp.vp_names)
        if cfg.aa_ref.every != 1:
            spec["every"] = cfg.aa_ref.every
        if cfg.aa_ref.skip_frames > 0:
            spec["frame_start"] = cfg.aa_ref.skip_frames
        if cfg.aa_ref.n_frames > 0:
            spec["frame_end"] = cfg.aa_ref.skip_frames + cfg.aa_ref.n_frames
        self._apply_post_runtime_options(spec)

        run_post(
            spec,
            self.resource_pool,
            run_dir=work_dir,
            python_exe=cfg.scheduler.python_exe or None,
        )
        if not output_file.exists():
            logger.warning("DSM accumulation produced no output at step %d", step_index)
            return None

        with open(output_file, "rb") as handle:
            batch = pickle.load(handle)
        self._scale_dsm_batch(batch)
        return batch

    def _dsm_noise_runtime_spec(self, step_index: int) -> Dict[str, Any]:
        noise = self.config.aa_ref.noise
        stage = int(step_index) // int(noise.update_interval)
        runtime = noise.to_runtime_dict()
        runtime["sigma"] = resolve_aa_noise_sigma(
            noise,
            n_epochs=int(self.config.training.n_epochs),
            stage=stage,
        )
        if float(runtime["sigma"]) <= 0.0:
            raise ValueError("DSM requires resolved noise sigma > 0.")
        runtime["seed"] = int(noise.seed) + int(stage)
        runtime["target"] = "dsm"
        runtime["beta"] = float(self.beta)
        if int(noise.subsample_per_epoch) > 0:
            runtime["subsample_seed"] = int(noise.seed) + int(step_index)
        return runtime

    def _scale_dsm_batch(self, batch: Dict[str, Any]) -> None:
        scale = float(self.beta) * float(self.beta)
        for key in ("JtJ", "Jty", "Jtf"):
            if key in batch:
                arr = np.asarray(batch[key])
                dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float32
                batch[key] = arr.astype(dtype, copy=False) * np.asarray(scale, dtype=dtype)
        for key in ("y_sumsq", "f_sumsq", "fty"):
            if key in batch:
                batch[key] = float(batch[key]) * scale

    def _checkpoint_dir_for_epoch(self, epoch: int) -> Path:
        """Return the DSM per-epoch checkpoint directory."""
        return self.output_dir / f"dsm_step_{int(epoch):04d}" / "ff"

    def _active_optimizer(self):
        """Return the optimizer carrying the live DSM training state."""
        trainer_optimizer = getattr(self.trainer_or_solver, "optimizer", None)
        if trainer_optimizer is not None:
            return trainer_optimizer
        return self.optimizer

    def _snapshot_optimizer(self, ff_dir: Path) -> Path:
        """Persist optimizer state next to the DSM workflow checkpoint."""
        snapshot_path = ff_dir / "optimizer_snapshot.pkl"
        ff_dir.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "wb") as fh:
            pickle.dump(
                self._active_optimizer().state_dict(),
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return snapshot_path

    def _write_workflow_checkpoint(self, ff_dir: Path) -> Path:
        """Persist the completed-epoch state used for DSM resume."""
        snapshot_path = ff_dir / "workflow_checkpoint.pkl"
        ff_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "forcefield": copy.deepcopy(self.forcefield),
            "optimizer_state": self._active_optimizer().state_dict(),
            "workflow_rng_state": self.workflow_rng.getstate(),
        }
        with open(snapshot_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return snapshot_path

    def _load_workflow_checkpoint(self, ff_dir: Path) -> Path:
        """Restore a completed DSM epoch checkpoint."""
        snapshot_path = ff_dir / "workflow_checkpoint.pkl"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Workflow checkpoint {snapshot_path} not found.")
        with open(snapshot_path, "rb") as fh:
            payload = pickle.load(fh)
        self.forcefield = payload["forcefield"]
        self.optimizer = self._build_optimizer(self.forcefield)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if hasattr(self.optimizer, "L"):
            self.forcefield.update_params(np.asarray(self.optimizer.L, dtype=np.float64))
        self.workflow_rng.setstate(payload["workflow_rng_state"])
        return snapshot_path

    def _load_optimizer_snapshot(self, snapshot_path: Path) -> None:
        """Restore optimizer state written by :meth:`_snapshot_optimizer`."""
        if not snapshot_path.exists():
            return
        with open(snapshot_path, "rb") as fh:
            state = pickle.load(fh)
        self.optimizer.load_state_dict(state)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``acg-dsm`` entry point."""
    return _run_workflow_cli(
        DSMWorkflow,
        prog="acg-dsm",
        description="Run the AceCG DSM workflow.",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
