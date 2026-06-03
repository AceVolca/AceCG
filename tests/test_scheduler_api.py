"""Scheduler API regression tests."""

import inspect

import AceCG
import AceCG.schedulers
from AceCG.schedulers.task_scheduler import TaskScheduler


def test_run_iteration_has_no_profiler_parameter():
    params = inspect.signature(TaskScheduler.run_iteration).parameters
    assert "profiler" not in params


def test_profiler_exports_are_removed():
    assert not hasattr(AceCG.schedulers, "RuntimeProfiler")
    assert hasattr(AceCG.schedulers, "preflight_benchmark")
    assert not hasattr(AceCG, "RuntimeProfiler")
    assert hasattr(AceCG, "preflight_benchmark")
