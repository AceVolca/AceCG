"""Canonical compute request names and kernel keyword mapping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType


REQUEST_REGISTRY = MappingProxyType(
    {
        "energy": MappingProxyType(
            {
                "energy": "return_value",
                "energy_grad": "return_grad",
                "energy_hessian": "return_hessian",
                "energy_grad_outer": "return_grad_outer",
                "gauge_free_energy_grad": "return_gauge_free_energy_grad",
                "gauge_free_energy_grad_outer": "return_gauge_free_energy_grad_outer",
                "unmasked_energy_grad": "return_unmasked_energy_grad",
                "unmasked_energy_grad_outer": "return_unmasked_energy_grad_outer",
                "unmasked_gauge_free_energy_grad": "return_unmasked_gauge_free_energy_grad",
                "unmasked_gauge_free_energy_grad_outer": (
                    "return_unmasked_gauge_free_energy_grad_outer"
                ),
            }
        ),
        "force": MappingProxyType(
            {
                "force": "return_value",
                "force_grad": "return_grad",
                "fm_stats": "return_fm_stats",
            }
        ),
        "traj_reader": MappingProxyType(
            {
                "reference_force": "include_forces",
            }
        ),
        "geometry": MappingProxyType(
            {
                "frame_cache": "frame_cache",
            }
        ),
    }
)

REQUEST_NAMES = frozenset(
    name
    for category in REQUEST_REGISTRY.values()
    for name in category
)

def normalize_compute_request(request: Iterable[str] | None) -> frozenset[str]:
    """Return validated canonical compute request names.

    Request inputs are canonical names only. Legacy ``need_*`` boolean mapping
    requests are intentionally not accepted.
    """
    if request is None:
        return frozenset()
    if isinstance(request, Mapping):
        raise TypeError(
            "compute request must be an iterable of canonical request names; "
            "legacy need_* boolean mappings are not supported."
        )
    if isinstance(request, str):
        raise TypeError("compute request must be an iterable of names, not a string.")

    non_string = [name for name in request if not isinstance(name, str)]
    if non_string:
        bad_type = type(non_string[0]).__name__
        raise TypeError(f"compute request names must be strings, got {bad_type}.")
    names = frozenset(request)
    unknown = names - REQUEST_NAMES
    if unknown:
        unknown_text = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown compute request name(s): {unknown_text}")
    return names


def request_kwargs(category: str, request: Iterable[str]) -> dict[str, bool]:
    """Map request names in *category* to kernel keyword booleans."""
    request_names = normalize_compute_request(request)
    try:
        mapping = REQUEST_REGISTRY[category]
    except KeyError as exc:
        raise ValueError(f"Unknown compute request category: {category!r}") from exc
    return {kw: name in request_names for name, kw in mapping.items()}
