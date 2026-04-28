"""Generic script inspection protocol for simulation backends."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class ScriptInfo(Protocol):
    """Metadata extracted from a simulation input script.

    Every backend parser must return an object conforming to this protocol.
    """

    @property
    def input_path(self) -> Path:
        """Return the source input-script path."""
        ...

    @property
    def init_data_path(self) -> Path:
        """Return the initial coordinate/data file required by the script."""
        ...

    @property
    def trajectory_path(self) -> Path:
        """Return the trajectory file produced by the script."""
        ...

    @property
    def checkpoint_path(self) -> Optional[Path]:
        """Return the optional checkpoint/replay output path."""
        ...


def parse_script(script_path: Path, *, backend: str = "lammps") -> ScriptInfo:
    """Dispatch to backend-specific parser.

    Raises ``NotImplementedError`` for unknown backends.
    """
    if backend == "lammps":
        from ._lammps_script import parse_lammps_script

        return parse_lammps_script(script_path)
    raise NotImplementedError(
        f"Script inspection for backend '{backend}' is not implemented. "
        f"Supported: lammps"
    )
