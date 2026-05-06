#!/usr/bin/env python3
"""Benchmark Forcefield mask/bounds metadata strategies.

This script compares the optimized production ``Forcefield`` metadata cache
against the legacy eager strategy that repeatedly reassembled
``param_mask``/``param_bounds`` from potential-local metadata.

Examples
--------
Lightweight login-node smoke:

    PYTHONPATH=src python scripts/benchmark_forcefield_mask_bounds.py --preset quick

Larger runs should be launched on a compute node:

    PYTHONPATH=src python scripts/benchmark_forcefield_mask_bounds.py --preset large
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable, Sequence

import numpy as np

from AceCG.potentials.base import BasePotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey


class SyntheticPotential(BasePotential):
    """Minimal potential with many parameters and potential-local metadata."""

    def __init__(self, n_params: int, *, finite_bounds: bool = False):
        super().__init__()
        self._params = np.zeros(int(n_params), dtype=np.float64)
        self._param_names = None
        self._dparam_names = []
        self._d2param_names = []
        self._params_to_scale = []
        if finite_bounds:
            self.param_bounds = (
                np.full(int(n_params), -1.0, dtype=np.float64),
                np.full(int(n_params), 1.0, dtype=np.float64),
            )

    def value(self, r: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)

    def force(self, r: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)

    def is_param_linear(self) -> np.ndarray:
        return np.ones(self.n_params(), dtype=bool)


@dataclass
class BenchmarkResult:
    case: str
    operation: str
    strategy: str
    seconds: float
    peak_mb: float | None
    n_potentials: int
    params_per_potential: int
    n_params: int
    iterations: int


class LegacyEagerMetadata:
    """Legacy-style metadata access over a Forcefield.

    The active production ``Forcefield`` keeps a versioned dirty cache. This
    wrapper intentionally bypasses that cache and re-collects masks/bounds from
    potentials on every read so the script can quantify the improvement.
    """

    def __init__(self, forcefield: Forcefield):
        self.forcefield = forcefield

    @property
    def param_mask(self) -> np.ndarray:
        return self.forcefield._collect_potential_masks()

    @property
    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.forcefield._collect_potential_bounds()

    @param_mask.setter
    def param_mask(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        self.forcefield._assign_potential_masks(mask)

    @param_bounds.setter
    def param_bounds(self, bounds: tuple[np.ndarray, np.ndarray]) -> None:
        lb, ub = bounds
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
        self.forcefield._assign_potential_bounds(lb, ub)


def build_forcefield(
    n_potentials: int,
    params_per_potential: int,
    *,
    finite_bounds: bool,
) -> tuple[Forcefield, list[InteractionKey]]:
    data = {}
    keys = []
    for i in range(int(n_potentials)):
        key = InteractionKey.pair(f"A{i}", f"B{i}")
        keys.append(key)
        data[key] = [SyntheticPotential(params_per_potential, finite_bounds=finite_bounds)]
    return Forcefield(data), keys


def make_arrays(n_params: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones(n_params, dtype=bool)
    mask[::2] = False
    lb = np.full(n_params, -2.0, dtype=np.float64)
    ub = np.full(n_params, 2.0, dtype=np.float64)
    return mask, lb, ub


def touch_mask_bounds(mask: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> float:
    lb, ub = bounds
    if mask.size == 0:
        return 0.0
    return float(mask.shape[0]) + float(lb[0]) + float(ub[-1])


def current_read(ff: Forcefield, iterations: int) -> float:
    acc = 0.0
    for _ in range(iterations):
        acc += touch_mask_bounds(ff.param_mask, ff.param_bounds)
    return acc


def legacy_read(cache: LegacyEagerMetadata, iterations: int) -> float:
    acc = 0.0
    for _ in range(iterations):
        acc += touch_mask_bounds(cache.param_mask, cache.param_bounds)
    return acc


def current_set_then_read(ff: Forcefield, iterations: int) -> float:
    mask, lb, ub = make_arrays(ff.n_params())
    acc = 0.0
    for i in range(iterations):
        if i % 2:
            ff.param_mask = ~mask
        else:
            ff.param_mask = mask
        ff.param_bounds = (lb, ub)
        acc += touch_mask_bounds(ff.param_mask, ff.param_bounds)
    return acc


def legacy_set_then_read(cache: LegacyEagerMetadata, iterations: int) -> float:
    mask, lb, ub = make_arrays(cache.forcefield.n_params())
    acc = 0.0
    for i in range(iterations):
        if i % 2:
            cache.param_mask = ~mask
        else:
            cache.param_mask = mask
        cache.param_bounds = (lb, ub)
        acc += touch_mask_bounds(cache.param_mask, cache.param_bounds)
    return acc


def legacy_replace_eager_read(
    ff: Forcefield,
    keys: Sequence[InteractionKey],
    params_per_potential: int,
    iterations: int,
) -> float:
    mask = np.empty(0, dtype=bool)
    bounds = (
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )
    for i in range(iterations):
        key = keys[i % len(keys)]
        ff[key] = [SyntheticPotential(params_per_potential, finite_bounds=True)]
        mask = ff._collect_potential_masks()
        bounds = ff._collect_potential_bounds()
    return touch_mask_bounds(mask, bounds)


def current_replace_deferred_read(
    ff: Forcefield,
    keys: Sequence[InteractionKey],
    params_per_potential: int,
    iterations: int,
) -> float:
    for i in range(iterations):
        key = keys[i % len(keys)]
        ff[key] = [SyntheticPotential(params_per_potential, finite_bounds=True)]
    return touch_mask_bounds(ff.param_mask, ff.param_bounds)


def timed(fn, *, repeat: int, tracemalloc_enabled: bool) -> tuple[float, float | None]:
    seconds = []
    peaks = []
    for _ in range(repeat):
        gc.collect()
        if tracemalloc_enabled:
            tracemalloc.start()
        start = time.perf_counter()
        marker = fn()
        elapsed = time.perf_counter() - start
        if marker == -123456789.0:
            raise RuntimeError("unreachable marker")
        if tracemalloc_enabled:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak / (1024.0 * 1024.0))
        seconds.append(elapsed)
    peak_mb = median(peaks) if peaks else None
    return median(seconds), peak_mb


def run_case(
    case: str,
    *,
    n_potentials: int,
    params_per_potential: int,
    iterations: int,
    replace_iterations: int,
    repeat: int,
    tracemalloc_enabled: bool,
) -> list[BenchmarkResult]:
    n_params = int(n_potentials) * int(params_per_potential)
    results: list[BenchmarkResult] = []

    def record(operation: str, strategy: str, seconds: float, peak_mb: float | None) -> None:
        results.append(
            BenchmarkResult(
                case=case,
                operation=operation,
                strategy=strategy,
                seconds=seconds,
                peak_mb=peak_mb,
                n_potentials=int(n_potentials),
                params_per_potential=int(params_per_potential),
                n_params=n_params,
                iterations=int(iterations if operation != "replace_deferred_read" else replace_iterations),
            )
        )

    ff_legacy, _ = build_forcefield(n_potentials, params_per_potential, finite_bounds=True)
    legacy = LegacyEagerMetadata(ff_legacy)
    seconds, peak = timed(
        lambda: legacy_read(legacy, iterations),
        repeat=repeat,
        tracemalloc_enabled=tracemalloc_enabled,
    )
    record("read_hot_metadata", "legacy_eager_metadata", seconds, peak)

    ff_current, _ = build_forcefield(n_potentials, params_per_potential, finite_bounds=True)
    seconds, peak = timed(
        lambda: current_read(ff_current, iterations),
        repeat=repeat,
        tracemalloc_enabled=tracemalloc_enabled,
    )
    record("read_hot_metadata", "optimized_forcefield", seconds, peak)

    ff_legacy, _ = build_forcefield(n_potentials, params_per_potential, finite_bounds=True)
    legacy = LegacyEagerMetadata(ff_legacy)
    seconds, peak = timed(
        lambda: legacy_set_then_read(legacy, iterations),
        repeat=repeat,
        tracemalloc_enabled=tracemalloc_enabled,
    )
    record("set_then_read_metadata", "legacy_eager_metadata", seconds, peak)

    ff_current, _ = build_forcefield(n_potentials, params_per_potential, finite_bounds=True)
    seconds, peak = timed(
        lambda: current_set_then_read(ff_current, iterations),
        repeat=repeat,
        tracemalloc_enabled=tracemalloc_enabled,
    )
    record("set_then_read_metadata", "optimized_forcefield", seconds, peak)

    ff_legacy, keys = build_forcefield(n_potentials, params_per_potential, finite_bounds=True)
    seconds, peak = timed(
        lambda: legacy_replace_eager_read(
            ff_legacy,
            keys,
            params_per_potential,
            replace_iterations,
        ),
        repeat=repeat,
        tracemalloc_enabled=tracemalloc_enabled,
    )
    record("replace_deferred_read", "legacy_eager_metadata", seconds, peak)

    ff_current, keys = build_forcefield(n_potentials, params_per_potential, finite_bounds=True)
    seconds, peak = timed(
        lambda: current_replace_deferred_read(
            ff_current,
            keys,
            params_per_potential,
            replace_iterations,
        ),
        repeat=repeat,
        tracemalloc_enabled=tracemalloc_enabled,
    )
    record("replace_deferred_read", "optimized_forcefield", seconds, peak)

    return results


PRESETS = {
    "quick": {
        "many_potentials": (500, 4, 50, 20),
        "many_params": (2, 20_000, 50, 5),
    },
    "large": {
        "many_potentials": (25_000, 4, 100, 100),
        "many_params": (2, 500_000, 100, 10),
    },
    "stress": {
        "many_potentials": (100_000, 4, 100, 200),
        "many_params": (2, 2_000_000, 100, 10),
    },
}


def format_table(results: Sequence[BenchmarkResult]) -> str:
    by_key = {}
    for result in results:
        by_key[(result.case, result.operation, result.strategy)] = result

    rows = []
    for result in results:
        if result.strategy != "legacy_eager_metadata":
            legacy = by_key.get((result.case, result.operation, "legacy_eager_metadata"))
            speedup = legacy.seconds / result.seconds if legacy and result.seconds > 0 else np.nan
        else:
            speedup = np.nan
        peak = "" if result.peak_mb is None else f"{result.peak_mb:.2f}"
        rows.append(
            [
                result.case,
                result.operation,
                result.strategy,
                f"{result.seconds:.6f}",
                "" if np.isnan(speedup) else f"{speedup:.2f}x",
                peak,
                str(result.n_potentials),
                str(result.params_per_potential),
                str(result.n_params),
                str(result.iterations),
            ]
        )

    headers = [
        "case",
        "operation",
        "strategy",
        "seconds",
        "speedup_vs_legacy",
        "peak_mb",
        "n_pots",
        "params/pot",
        "n_params",
        "iters",
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def line(values: Iterable[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths))

    out = [line(headers), line(["-" * width for width in widths])]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--tracemalloc", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    all_results: list[BenchmarkResult] = []
    for case, (n_pots, params_per_pot, iterations, replace_iterations) in preset.items():
        all_results.extend(
            run_case(
                case,
                n_potentials=n_pots,
                params_per_potential=params_per_pot,
                iterations=iterations,
                replace_iterations=replace_iterations,
                repeat=max(1, int(args.repeat)),
                tracemalloc_enabled=bool(args.tracemalloc),
            )
        )

    if args.json:
        print(json.dumps([asdict(result) for result in all_results], indent=2))
    else:
        print(format_table(all_results))


if __name__ == "__main__":
    main()
