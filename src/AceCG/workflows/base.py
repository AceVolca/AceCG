"""BaseWorkflow — root of the workflow class hierarchy.

Every workflow takes a single ``ACGConfig`` in ``__init__`` (plus optional
``**kwargs`` forwarded to ``dataclasses.replace`` on sub-configs for overrides).

Object construction is delegated to ``_build_*`` protected methods, each
defined once at the inheritance level that introduces the object.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import pickle
import random
import re
import shlex
import shutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

import MDAnalysis as mda
import numpy as np

from ..configs.models import ACGConfig
from ..configs.parser import _parse_scalar_or_literal, parse_acg_file
from ..configs.utils import parse_pair_style_options
from ..io.forcefield import WriteLmpFF
from ..optimizers import (
    AdamMaskedOptimizer,
    AdamWMaskedOptimizer,
    BaseOptimizer,
    NewtonRaphsonOptimizer,
    RMSpropMaskedOptimizer,
)
from ..samplers.base import BaseSampler, InitConfigRecord
from ..schedulers.resource_pool import ResourcePool
from ..schedulers.task_scheduler import TaskScheduler, TaskSpec
from ..topology.forcefield import Forcefield
from ..topology.topology_array import TopologyArrays, collect_topology_arrays


def _apply_config_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    """Apply section-qualified overrides to a frozen config dataclass."""
    if not overrides:
        return config
    section_patches: dict[str, dict[str, Any]] = {}
    direct: dict[str, Any] = {}
    for key, value in overrides.items():
        if "__" in key:
            section, field = key.split("__", 1)
            section_patches.setdefault(section, {})[field] = value
        else:
            direct[key] = value
    replaced = config
    for section, patch in section_patches.items():
        sub = getattr(replaced, section, None)
        if sub is None:
            raise ValueError(
                f"Unknown config section '{section}' in override key "
                f"'{section}__{next(iter(patch))}'"
            )
        if dataclasses.is_dataclass(sub):
            replaced = dataclasses.replace(
                replaced, **{section: dataclasses.replace(sub, **patch)}
            )
    if direct:
        replaced = dataclasses.replace(replaced, **direct)
    return replaced


def _build_workflow_cli_parser(
    *, prog: str, description: str
) -> argparse.ArgumentParser:
    """Create the common AceCG workflow CLI parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        allow_abbrev=False,
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        help="Optional path to an AceCG .acg file.",
    )
    return parser


def _parse_cli_overrides(tokens: Sequence[str]) -> dict[str, Any]:
    """Parse repeated ``--section.field value`` CLI overrides."""
    overrides: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = str(tokens[index])
        if not token.startswith("--") or token == "--":
            raise ValueError(
                "Workflow CLI overrides must use --section.field value or "
                "--section.field=value syntax."
            )
        key_text = token[2:]
        if "=" in key_text:
            key, raw_value = key_text.split("=", 1)
        else:
            index += 1
            if index >= len(tokens):
                raise ValueError(f"Override {token!r} requires a value.")
            key = key_text
            raw_value = str(tokens[index])
        if "." not in key:
            raise ValueError(
                f"Override key {key!r} must use section.field syntax."
            )
        section, field = key.split(".", 1)
        if not section or not field:
            raise ValueError(
                f"Override key {key!r} must use section.field syntax."
            )
        overrides[f"{section}__{field}"] = _parse_scalar_or_literal(raw_value)
        index += 1
    return overrides


def _run_workflow_cli(
    workflow_cls: type["BaseWorkflow"],
    *,
    prog: str,
    description: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Run a training workflow from the generic CLI entry point."""
    parser = _build_workflow_cli_parser(prog=prog, description=description)
    args, unknown = parser.parse_known_args(argv)
    overrides = _parse_cli_overrides(unknown)
    config = parse_acg_file(args.config) if args.config else ACGConfig()
    config = _apply_config_overrides(config, overrides)

    workflow = workflow_cls(config)
    result = workflow.run()
    output_path = workflow.output_dir / "acgreturn.pkl"
    with open(output_path, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


class BaseWorkflow(ABC):
    """Abstract base for all AceCG workflows.

    Attributes set in ``__init__``:
        config      – frozen ``ACGConfig``
        output_dir  – resolved ``Path``, created on disk
        topology    – ``TopologyArrays`` from the CG data file
    """

    config: ACGConfig
    output_dir: Path
    topology: TopologyArrays
    workflow_rng: random.Random

    def __init__(self, config: ACGConfig, **kwargs: Any) -> None:
        self.config = self._apply_overrides(config, kwargs)
        self.workflow_rng = random.Random(int(self.config.training.seed))
        self.validation_rng = random.Random(int(self.config.training.seed) + 7919)
        self._validation_sampler: Optional[BaseSampler] = None
        self._validation_scheduler: Optional[TaskScheduler] = None
        self._validation_records: list[dict[str, Any]] = []
        self.output_dir = self._build_output_dir()
        self.topology = self._build_topology()

    # ── builders ────────────────────────────────────────────────

    @classmethod
    def _apply_overrides(cls, config: ACGConfig, overrides: dict) -> ACGConfig:
        """Apply keyword overrides to frozen sub-configs via ``dataclasses.replace``."""
        return _apply_config_overrides(config, overrides)

    # ── builders ────────────────────────────────────────────────

    def _build_output_dir(self) -> Path:
        p = self._resolve_config_path(self.config.training.output_dir)
        if p is None:
            raise ValueError("training.output_dir is required.")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _build_topology(self) -> TopologyArrays:
        cfg = self.config
        vp_names: Optional[tuple] = None
        if cfg.vp is not None:
            vp_names = cfg.vp.vp_names or None
        topology_path = self._resolve_config_path(cfg.system.topology_file)
        if topology_path is None:
            raise ValueError("system.topology_file is required.")
        u = mda.Universe(str(topology_path))
        return collect_topology_arrays(
            u,
            exclude_bonded=cfg.system.exclude_bonded,
            exclude_option=cfg.system.exclude_option,
            atom_type_name_aliases=cfg.system.type_names,
            vp_names=vp_names,
        )

    def _config_base_dir(self) -> Path:
        if self.config.path is not None:
            return self.config.path.parent
        return Path.cwd().resolve()

    def _resolve_config_path(self, value: Any) -> Optional[Path]:
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self._config_base_dir() / path
        return path.resolve(strict=False)

    def _glob_config_paths(self, pattern: Optional[str]) -> list[Path]:
        if pattern is None:
            return []
        raw = Path(pattern).expanduser()
        query = str(raw if raw.is_absolute() else self._config_base_dir() / raw)
        return [Path(match).resolve(strict=False) for match in sorted(glob.glob(query))]

    # ── validation helpers ──────────────────────────────────────

    def _validation_enabled(self) -> bool:
        return bool(getattr(self.config, "validation", None) and self.config.validation.enabled)

    def _validation_due_after_epoch(self, epoch: int) -> bool:
        if not self._validation_enabled():
            return False
        interval = self.config.validation.num_epochs_per_validation
        return interval is not None and (int(epoch) + 1) % int(interval) == 0

    def _validation_sim_cmd(self) -> list[str]:
        cmd = self.config.validation.engine_command
        if not cmd:
            raise ValueError("validation.engine_command is required.")
        return shlex.split(cmd)

    def _get_validation_sampler(self) -> BaseSampler:
        if self._validation_sampler is not None:
            return self._validation_sampler
        if not self._validation_enabled():
            raise ValueError("Validation is not enabled.")
        cfg = self.config.validation
        sim_input = self._resolve_config_path(cfg.input)
        if sim_input is None:
            raise ValueError("validation.input is required.")
        init_paths = self._glob_config_paths(cfg.init_config_pool)
        if cfg.init_config_pool is not None and not init_paths:
            raise ValueError(
                "validation.init_config_pool glob matched no files: "
                f"{cfg.init_config_pool!r}"
            )
        init_pool = [InitConfigRecord(path=p) for p in init_paths]
        self._validation_sampler = BaseSampler(
            sim_input=sim_input,
            sim_backend=cfg.sim_backend,
            init_config_pool=init_pool or None,
            replay_mode="off",
            rng=self.validation_rng,
        )
        return self._validation_sampler

    def _get_validation_scheduler(self) -> TaskScheduler:
        if self._validation_scheduler is not None:
            return self._validation_scheduler
        cfg = self.config.scheduler
        pool = self._build_resource_pool(sim_cmd=self._validation_sim_cmd())
        self._validation_scheduler = TaskScheduler(
            pool,
            task_timeout=cfg.task_timeout,
            min_success_zbx=None,
            python_exe=cfg.python_exe,
            rng_seed=self.validation_rng.randint(0, 2**31 - 1),
        )
        return self._validation_scheduler

    def _validation_forcefield_template_path(self) -> Path:
        raw = (
            self.config.validation.forcefield_template_path
            or self.config.system.forcefield_path
        )
        path = self._resolve_config_path(raw)
        if path is None:
            raise ValueError(
                "validation.forcefield_template_path is required when "
                "system.forcefield_path is absent."
            )
        return path

    def _write_lammps_forcefield_bundle(
        self,
        ff_dir: Path,
        *,
        forcefield: Optional[Forcefield] = None,
        template_path: Optional[Path] = None,
        runtime_relpath: Optional[Path] = None,
    ) -> Path:
        """Write a LAMMPS-compatible forcefield bundle under *ff_dir*.

        ``WriteLmpFF`` needs an existing settings/include file so it can
        preserve coefficient commands, table filenames, and hybrid styles.
        """
        cfg = self.config.system
        pair_style, sel_styles = parse_pair_style_options(cfg.pair_style)
        source = template_path or self._resolve_config_path(cfg.forcefield_path)
        if source is None:
            raise ValueError("A LAMMPS forcefield template path is required.")
        ff = forcefield if forcefield is not None else getattr(self, "forcefield", None)
        if ff is None:
            raise ValueError("No runtime forcefield is available to export.")
        ff_file = ff_dir / (runtime_relpath or Path(source).name)
        ff_file.parent.mkdir(parents=True, exist_ok=True)
        WriteLmpFF(
            str(source),
            str(ff_file),
            ff,
            pair_style,
            pair_typ_sel=sel_styles,
            topology_arrays=self.topology,
        )
        return ff_file

    def _validation_forcefield_runtime_relpath(self, template_path: Path) -> Path:
        sim_input = self._resolve_config_path(self.config.validation.input)
        if sim_input is not None:
            try:
                return template_path.relative_to(sim_input.parent)
            except ValueError:
                pass
        return Path(template_path.name)

    @staticmethod
    def _copy_forcefield_bundle_to_run(ff_dir: Path, run_dir: Path) -> None:
        for src in ff_dir.rglob("*"):
            if not src.is_file() or src.suffix == ".pkl":
                continue
            dst = run_dir / src.relative_to(ff_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    @staticmethod
    def _core_bounds_from_pool(
        resource_pool: ResourcePool,
        explicit_ncores: Optional[int],
    ) -> tuple[int, int, int, int]:
        hosts = resource_pool.hosts
        if not hosts:
            raise RuntimeError("resource_pool has no hosts")
        if explicit_ncores is not None:
            n = int(explicit_ncores)
            return n, n, n, n
        max_host = max(h.n_cpus for h in hosts)
        min_host = min(h.n_cpus for h in hosts)
        total = sum(h.n_cpus for h in hosts)
        return max_host, min_host, max_host, total

    def _build_validation_task(
        self,
        *,
        label: str,
        epoch: Optional[int],
        forcefield: Optional[Forcefield] = None,
        resource_pool: Optional[ResourcePool] = None,
    ) -> TaskSpec:
        """Stage one validation replica and return its optional xz task."""
        if not self._validation_enabled():
            raise ValueError("Validation is not enabled.")

        scheduler_pool = resource_pool
        if scheduler_pool is None:
            scheduler = getattr(self, "scheduler", None)
            scheduler_pool = getattr(scheduler, "pool", None)
        if scheduler_pool is None:
            scheduler_pool = self._get_validation_scheduler().pool

        validation_dir = self.output_dir / "validation" / str(label)
        ff_dir = validation_dir / "ff"
        template_path = self._validation_forcefield_template_path()
        self._write_lammps_forcefield_bundle(
            ff_dir,
            forcefield=forcefield,
            template_path=template_path,
            runtime_relpath=self._validation_forcefield_runtime_relpath(
                template_path
            ),
        )

        sampler = self._get_validation_sampler()
        state = sampler.init_epoch(
            iteration_index=-1 if epoch is None else int(epoch),
            epoch_dir=validation_dir / "epoch",
            n_runs=1,
        )
        plan = state.replica_plans[0]
        self._copy_forcefield_bundle_to_run(ff_dir, plan.run_dir)

        cpu, min_cores, pref_cores, max_cores = self._core_bounds_from_pool(
            scheduler_pool,
            self.config.validation.ncores,
        )
        trajectory_relpath = str(plan.trajectory_path.relative_to(plan.run_dir))
        return TaskSpec(
            task_class="xz",
            frame_id=None,
            run_dir=str(plan.run_dir.resolve()),
            cpu_cores=cpu,
            min_cores=min_cores,
            preferred_cores=pref_cores,
            max_cores=max_cores,
            sim_input=plan.input_script_path.name,
            sim_backend=self.config.validation.sim_backend,
            sim_var=dict(self.config.validation.sim_var),
            sim_cmd=self._validation_sim_cmd(),
            post_spec=None,
            post_exec=None,
            archive_trajectory=True,
            trajectory_files=[trajectory_relpath],
            role="validation",
            required=False,
            metadata={
                "validation_label": str(label),
                "epoch": None if epoch is None else int(epoch),
            },
            single_host_only=False,
        )

    def _run_validation_blocking(
        self,
        *,
        label: str,
        epoch: Optional[int],
        forcefield: Optional[Forcefield] = None,
        scheduler: Optional[TaskScheduler] = None,
    ) -> Any:
        if not self._validation_enabled():
            return None
        validation_scheduler = scheduler or self._get_validation_scheduler()
        task = self._build_validation_task(
            label=label,
            epoch=epoch,
            forcefield=forcefield,
            resource_pool=validation_scheduler.pool,
        )
        result = validation_scheduler.run_iteration(
            xz_tasks=[task],
            zbx_tasks=[],
            iter_dir=self.output_dir / "validation" / str(label),
        )
        self._record_validation_results(result)
        return result

    def _record_validation_results(self, iter_result: Any) -> list[dict[str, Any]]:
        results = getattr(iter_result, "results", None)
        if results is None:
            return []
        records: list[dict[str, Any]] = []
        for result in results:
            task = getattr(result, "task", None)
            if task is None or getattr(task, "role", "training") != "validation":
                continue
            metadata = dict(getattr(task, "metadata", {}) or {})
            record = {
                "label": metadata.get("validation_label"),
                "epoch": metadata.get("epoch"),
                "run_dir": task.run_dir,
                "ok": bool(getattr(result, "ok", False)),
                "returncode": int(getattr(result, "returncode", -1)),
                "elapsed": float(getattr(result, "elapsed", 0.0)),
                "timing": getattr(result, "timing", None),
            }
            self._validation_records.append(record)
            records.append(record)
        if records:
            self._write_validation_summary()
        return records

    def _validation_result_payload(self) -> dict[str, Any]:
        return {
            "enabled": self._validation_enabled(),
            "runs": list(self._validation_records),
        }

    def _write_validation_summary(self) -> None:
        if not self._validation_enabled():
            return
        path = self.output_dir / "validation_summary.json"
        path.write_text(
            json.dumps(self._validation_result_payload(), indent=2, default=str)
            + "\n",
            encoding="utf-8",
        )

    def _apply_forcefield_specs(self, forcefield: Forcefield) -> None:
        forcefield.apply_specs(
            mask_spec=self.config.system.forcefield_mask,
            bounds_spec=self.config.system.forcefield_bounds,
        )

    def _build_optimizer(
        self,
        forcefield: Forcefield,
        *,
        default_kind: str = "newton",
    ) -> BaseOptimizer:
        """Build an optimizer from ``self.config.training`` + forcefield params."""
        tcfg = self.config.training
        spec = str(
            tcfg.optimizer or tcfg.trainer or default_kind
        ).strip()
        token = spec.split()[0].lower() if spec else default_kind
        # Learning rate
        lr_raw = tcfg.lr
        if lr_raw is not None:
            lr = float(lr_raw)
        else:
            lr = 1.0 if default_kind == "newton" else 1.0e-3
        seed = tcfg.seed

        params = forcefield.param_array()
        mask = forcefield.param_mask

        # Parse weight_decay from spec string
        weight_decay = 0.0
        match = re.search(r"weight_decay\s*=?\s*([0-9.eE+-]+)", spec)
        if match is not None:
            weight_decay = float(match.group(1))
        elif token == "adamw":
            float_match = re.search(r"adamw\s+([0-9.eE+-]+)", spec.lower())
            if float_match is not None:
                weight_decay = float(float_match.group(1))

        if token in {"newton", "newtonraphson", "newton_raphson"}:
            return NewtonRaphsonOptimizer(L=params, mask=mask, lr=lr)
        if token in {"adam", "adammaskedoptimizer"}:
            return AdamMaskedOptimizer(L=params, mask=mask, lr=lr, seed=seed)
        if token in {"adamw", "adamwmaskedoptimizer"}:
            return AdamWMaskedOptimizer(
                L=params,
                mask=mask,
                lr=lr,
                seed=seed,
                weight_decay=weight_decay,
            )
        if token in {"rmsprop", "rmspropmaskedoptimizer"}:
            return RMSpropMaskedOptimizer(L=params, mask=mask, lr=lr)
        raise ValueError(f"Unsupported optimizer/trainer spec: {spec!r}")

    def _build_resource_pool(
        self,
        *,
        sim_cmd: Optional[list] = None,
    ) -> ResourcePool:
        """Discover compute resources from environment.

        Parameters
        ----------
        sim_cmd : list of str, optional
            Simulation command tokens.  Sampling workflows derive this from
            ``sampling.engine_command``; FM workflows pass ``[]``.
        """
        cfg = self.config
        if sim_cmd is None:
            cmd = cfg.sampling.engine_command
            sim_cmd = shlex.split(cmd) if cmd else []
        if cfg.scheduler.launcher not in (None, ""):
            warnings.warn(
                "SchedulerConfig.launcher is deprecated and ignored. AceCG now "
                "auto-detects the MPI backend from scheduler.mpirun_path or PATH.",
                DeprecationWarning,
                stacklevel=2,
            )
        mpirun_path = cfg.scheduler.mpirun_path or None
        raw_hosts = cfg.scheduler.extras.get("explicit_hosts")
        explicit_hosts = [
            (h["hostname"], tuple(h["cpu_ids"]))
            for h in raw_hosts
        ] if raw_hosts else None
        extra_env = cfg.scheduler.extras.get("extra_env")
        if extra_env is not None and not isinstance(extra_env, dict):
            extra_env = None
        intel_launch_mode = cfg.scheduler.extras.get(
            "intel_launch_mode", "mpmd",
        )
        return ResourcePool.discover(
            sim_cmd=sim_cmd,
            mpirun_path=mpirun_path,
            mpi_family=cfg.scheduler.mpi_family,
            explicit_hosts=explicit_hosts,
            extra_env=extra_env,
            intel_launch_mode=intel_launch_mode,
        )

    @abstractmethod
    def _build_trainer(self) -> Any:
        ...

    @abstractmethod
    def run(self) -> Any:
        """Run the workflow and return its workflow-specific result."""
        ...
