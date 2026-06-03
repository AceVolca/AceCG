"""Public I/O package API with lazy convenience exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ReadLmpFF": (".forcefield", "ReadLmpFF"),
    "ReadLmpFFBounds": (".forcefield", "ReadLmpFFBounds"),
    "ReadLmpFFMask": (".forcefield", "ReadLmpFFMask"),
    "WriteLmpFF": (".forcefield", "WriteLmpFF"),
    "iter_commands": (".lammps_input", "iter_commands"),
    "resolve_include_path": (".lammps_input", "resolve_include_path"),
    "resolve_lines": (".lammps_input", "resolve_lines"),
    "strip_lines": (".lammps_input", "strip_lines"),
    "tokenize_line": (".lammps_input", "tokenize_line"),
    "tokenize_lines": (".lammps_input", "tokenize_lines"),
    "parse_lammps_table": (".tables", "parse_lammps_table"),
    "interaction_table_stem": (".tables", "interaction_table_stem"),
    "find_equilibrium": (".tables", "find_equilibrium"),
    "build_forcefield_tables": (".tables", "build_forcefield_tables"),
    "export_tables": (".tables", "export_tables"),
    "compare_table_files": (".tables", "compare_table_files"),
    "cap_table_forces": (".tables", "cap_table_forces"),
    "write_lammps_table": (".tables", "write_lammps_table"),
    "write_lammps_table_bundle": (".tables", "write_lammps_table_bundle"),
    "estimate_table_fp": (".tables", "estimate_table_fp"),
    "integrate_force_to_potential": (".tables", "integrate_force_to_potential"),
    "constant_force_extrapolate": (".tables", "constant_force_extrapolate"),
    "export_grid": (".tables", "export_grid"),
    "FrameSpec": (".trajectory", "FrameSpec"),
    "FrameMap": (".trajectory", "FrameMap"),
    "iter_frames": (".trajectory", "iter_frames"),
    "load_dump_positions": (".trajectory", "load_dump_positions"),
    "count_lammpstrj_frames_and_atoms": (".trajectory", "count_lammpstrj_frames_and_atoms"),
    "split_lammpstrj": (".trajectory", "split_lammpstrj"),
    "split_lammpstrj_mdanalysis": (".trajectory", "split_lammpstrj_mdanalysis"),
    "load_mapping_yaml": (".coordinates", "load_mapping_yaml"),
    "build_CG_coords": (".coordinates", "build_CG_coords"),
    "write_gro": (".coordinates", "write_gro"),
    "write_pdb": (".coordinates", "write_pdb"),
    "write_lammps_data": (".coordinates", "write_lammps_data"),
    "ScreenLogger": (".logger", "ScreenLogger"),
    "format_screen_message": (".logger", "format_screen_message"),
    "get_screen_logger": (".logger", "get_screen_logger"),
    "user_timestamp": (".logger", "user_timestamp"),
}


__all__ = [
    "ReadLmpFF",
    "ReadLmpFFBounds",
    "ReadLmpFFMask",
    "WriteLmpFF",
    "iter_commands",
    "resolve_include_path",
    "resolve_lines",
    "strip_lines",
    "tokenize_line",
    "tokenize_lines",
    "parse_lammps_table",
    "interaction_table_stem",
    "iter_frames",
    "find_equilibrium",
    "build_forcefield_tables",
    "export_tables",
    "compare_table_files",
    "cap_table_forces",
    "write_lammps_table",
    "write_lammps_table_bundle",
    "estimate_table_fp",
    "integrate_force_to_potential",
    "constant_force_extrapolate",
    "export_grid",
    "load_dump_positions",
    "count_lammpstrj_frames_and_atoms",
    "split_lammpstrj",
    "split_lammpstrj_mdanalysis",
    "load_mapping_yaml",
    "build_CG_coords",
    "ScreenLogger",
    "format_screen_message",
    "get_screen_logger",
    "user_timestamp",
    "write_gro",
    "write_pdb",
    "write_lammps_data",
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
