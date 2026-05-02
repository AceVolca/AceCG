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
import pickle
import random
import re
import shlex
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

import MDAnalysis as mda
import numpy as np

from ..configs.models import ACGConfig
from ..configs.parser import _parse_scalar_or_literal, parse_acg_file
from ..optimizers import (
    AdamMaskedOptimizer,
    AdamWMaskedOptimizer,
    BaseOptimizer,
    NewtonRaphsonOptimizer,
    RMSpropMaskedOptimizer,
)
from ..schedulers.resource_pool import ResourcePool
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

    def _build_forcefield_mask(self, forcefield: Forcefield) -> np.ndarray:
        mask_spec = self.config.system.forcefield_mask
        param_mask = np.ones(forcefield.n_params(), dtype=bool)
        if not mask_spec:
            return param_mask

        blocks = {}
        offset = 0
        for key, val in forcefield.items():
            pots = val if isinstance(val, list) else [val]
            block_n = sum(p.n_params() for p in pots)
            blocks[key] = slice(offset, offset + block_n)
            offset += block_n

        true_tokens = {"1", "true", "yes", "on"}
        false_tokens = {"0", "false", "no", "off"}

        def parse_local_mask(payload: tuple[str, ...], size: int, key: Any) -> np.ndarray:
            if not payload:
                return np.ones(size, dtype=bool)

            mode = payload[0].strip().lower()
            if mode in {"mask", "unmask"}:
                spec = "".join(payload[1:]).strip().lower()
                if not spec:
                    raise ValueError(
                        f"{key.label()} uses '{mode}' but provides no index range."
                    )
                if spec == "all":
                    selected = np.ones(size, dtype=bool)
                elif spec == "none":
                    selected = np.zeros(size, dtype=bool)
                else:
                    selected = np.zeros(size, dtype=bool)
                    all_idx = np.arange(size, dtype=np.int64)
                    for part in spec.split(","):
                        field = part.strip()
                        if not field:
                            continue
                        if ":" in field:
                            start_text, stop_text = field.split(":", 1)
                            start = int(start_text) if start_text else None
                            stop = int(stop_text) if stop_text else None
                            selected[all_idx[slice(start, stop)]] = True
                            continue
                        idx = int(field)
                        if idx < 0:
                            idx += size
                        if idx < 0 or idx >= size:
                            raise IndexError(
                                f"Mask index {field!r} is out of range for "
                                f"{key.label()} with {size} parameters."
                            )
                        selected[idx] = True
                local_mask = np.ones(size, dtype=bool) if mode == "mask" else np.zeros(size, dtype=bool)
                local_mask[selected] = mode == "unmask"
                return local_mask

            values = []
            for token in payload:
                lowered = token.strip().lower()
                if lowered in true_tokens:
                    values.append(True)
                    continue
                if lowered in false_tokens:
                    values.append(False)
                    continue
                raise ValueError(f"Invalid mask payload for {key.label()}: {payload!r}")
            local_mask = np.asarray(values, dtype=bool)
            if local_mask.shape != (size,):
                raise ValueError(
                    f"{key.label()} mask length {local_mask.size} does not match "
                    f"the current parameter block size {size}."
                )
            return local_mask

        for key, payload in mask_spec.entries:
            block = blocks.get(key)
            if block is None:
                raise KeyError(
                    f"Mask entry for {key.label()} does not match any interaction "
                    "in the current forcefield."
                )
            param_mask[block] = parse_local_mask(payload, block.stop - block.start, key)

        return param_mask

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
