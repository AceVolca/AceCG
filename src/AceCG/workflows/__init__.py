"""AceCG workflow classes.

Imports here are lazy (see ``AceCG/__init__.py`` for the same pattern and its
rationale). This also avoids a real hazard specific to this package: each
workflow module is also a ``python -m AceCG.workflows.<name>`` / console-script
entry point (``acg-fm`` etc., see ``pyproject.toml``). Eagerly importing a
submodule here means that when it is later run as ``__main__``, Python loads
it a *second* time under a different module identity — ``runpy`` warns
("found in sys.modules ... prior to execution ... this may result in
unpredictable behaviour") because the two copies define distinct class
objects, so an ``isinstance`` check against the package-level export would
not recognize an object built by the ``__main__``-run copy. Lazy attributes
mean the package namespace is never populated until something outside the
entry point asks for it.
"""

from importlib import import_module
from typing import Any

_LAZY: dict[str, tuple[str, str]] = {
    "BaseWorkflow": (".base", "BaseWorkflow"),
    "CDFMWorkflow": (".cdfm", "CDFMWorkflow"),
    "CDREMWorkflow": (".cdrem", "CDREMWorkflow"),
    "DSMWorkflow": (".dsm", "DSMWorkflow"),
    "FMWorkflow": (".fm", "FMWorkflow"),
    "REMWorkflow": (".rem", "REMWorkflow"),
    "SamplingWorkflow": (".sampling", "SamplingWorkflow"),
    "TrajMapResult": (".trajmap", "TrajMapResult"),
    "TrajMapWorkflow": (".trajmap", "TrajMapWorkflow"),
    "VPGrowthResult": (".vp_growth", "VPGrowthResult"),
    "VPGrowthWorkflow": (".vp_growth", "VPGrowthWorkflow"),
    "run_boundary_prior": (".boundary_prior", "run_boundary_prior"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
