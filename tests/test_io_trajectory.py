from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from AceCG.io.trajectory import (
    count_lammpstrj_frames_and_atoms,
    load_dump_positions,
    split_lammpstrj,
)


def _frame_block(timestep: int, positions: np.ndarray) -> str:
    lines = [
        "ITEM: TIMESTEP",
        str(int(timestep)),
        "ITEM: NUMBER OF ATOMS",
        str(int(len(positions))),
        "ITEM: BOX BOUNDS pp pp pp",
        "0.0 3.0",
        "0.0 3.0",
        "0.0 3.0",
        "ITEM: ATOMS id type x y z",
    ]
    for atom_id, xyz in enumerate(np.asarray(positions, dtype=np.float64), start=1):
        lines.append(f"{atom_id} 1 {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    return "\n".join(lines) + "\n"


def _write_dump(path: Path, frames: list[np.ndarray]) -> None:
    text = "".join(_frame_block(i, frame) for i, frame in enumerate(frames))
    path.write_text(text, encoding="utf-8")


def test_load_dump_positions_reads_sorted_xyz_columns(tmp_path: Path) -> None:
    dump_path = tmp_path / "traj.lammpstrj"
    frames = [
        np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=np.float64),
        np.array([[0.3, 0.4, 0.5], [1.3, 1.4, 1.5]], dtype=np.float64),
    ]
    _write_dump(dump_path, frames)

    positions = load_dump_positions(dump_path)

    assert positions.shape == (2, 2, 3)
    np.testing.assert_allclose(positions[0], frames[0])
    np.testing.assert_allclose(positions[1], frames[1])


def test_count_lammpstrj_frames_and_atoms_rejects_variable_atom_counts(tmp_path: Path) -> None:
    dump_path = tmp_path / "broken.lammpstrj"
    _write_dump(
        dump_path,
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64),
            np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
        ],
    )

    with pytest.raises(ValueError, match="natoms changes across frames"):
        count_lammpstrj_frames_and_atoms(dump_path)


def test_split_lammpstrj_balances_frame_segments(tmp_path: Path) -> None:
    dump_path = tmp_path / "traj.lammpstrj"
    frames = [
        np.array([[float(i), 0.0, 0.0], [float(i) + 0.5, 0.0, 0.0]], dtype=np.float64)
        for i in range(5)
    ]
    _write_dump(dump_path, frames)

    out_paths = split_lammpstrj(dump_path, n_parts=2, out_dir=tmp_path / "parts")

    assert len(out_paths) == 2
    assert [count_lammpstrj_frames_and_atoms(path) for path in out_paths] == [(3, 2), (2, 2)]
