"""Topology package public API with lazy convenience exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Forcefield": (".forcefield", "Forcefield"),
    "InteractionKey": (".types", "InteractionKey"),
    "VPBondSpec": (".vpgrower", "VPBondSpec"),
    "VPAngleSpec": (".vpgrower", "VPAngleSpec"),
    "VPGrownFrame": (".vpgrower", "VPGrownFrame"),
    "VPGrower": (".vpgrower", "VPGrower"),
    "VPTopologyTemplate": (".vpgrower", "VPTopologyTemplate"),
    "write_vp_data": (".vpgrower", "write_vp_data"),
    "MSCGTopology": (".mscg", "MSCGTopology"),
    "parse_mscg_top": (".mscg", "parse_mscg_top"),
    "build_replicated_topology_arrays": (".mscg", "build_replicated_topology_arrays"),
    "attach_topology_from_mscg_top": (".mscg", "attach_topology_from_mscg_top"),
    "compute_neighbor_list": (".neighbor", "compute_neighbor_list"),
    "compute_pairs_by_type": (".neighbor", "compute_pairs_by_type"),
    "TopologyArrays": (".topology_array", "TopologyArrays"),
    "collect_topology_arrays": (".topology_array", "collect_topology_arrays"),
}


__all__ = [
    "Forcefield",
    "InteractionKey",
    "VPBondSpec",
    "VPAngleSpec",
    "VPGrownFrame",
    "VPGrower",
    "VPTopologyTemplate",
    "write_vp_data",
    "MSCGTopology",
    "parse_mscg_top",
    "build_replicated_topology_arrays",
    "attach_topology_from_mscg_top",
    "compute_neighbor_list",
    "compute_pairs_by_type",
    "TopologyArrays",
    "collect_topology_arrays",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
