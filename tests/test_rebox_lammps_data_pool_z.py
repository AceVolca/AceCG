from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_rebox_z_unwraps_crossing_bond_without_moving_ordinary_atoms(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "rebox_lammps_data_pool_z.py"
    input_dir = tmp_path / "pool"
    output_dir = tmp_path / "reboxed"
    input_dir.mkdir()
    data_path = input_dir / "frame_000000.data"
    data_path.write_text(
        """toy

4 atoms
2 bonds

1 atom types
1 bond types

0.000000 10.000000 xlo xhi
0.000000 10.000000 ylo yhi
0.000000 10.000000 zlo zhi

Masses

1 1.0

Atoms # full

1 1 1 0.0 1.000000 1.000000 9.500000 0 0 0
2 1 1 0.0 1.000000 1.000000 0.500000 0 0 0
3 2 1 0.0 2.000000 2.000000 3.000000 0 0 0
4 2 1 0.0 2.000000 2.000000 4.000000 0 0 0

Bonds

1 1 1 2
2 1 3 4
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-glob",
            str(input_dir / "frame_*.data"),
            "--output-dir",
            str(output_dir),
            "--target-z-length",
            "14",
            "--workers",
            "1",
        ],
        check=True,
        cwd=repo_root,
    )

    output = (output_dir / "frame_000000.data").read_text(encoding="utf-8")
    assert "-2.000000 12.000000 zlo zhi" in output
    assert "1 1 1 0.0 1.000000 1.000000 9.500000 0 0 0" in output
    assert "2 1 1 0.0 1.000000 1.000000 10.500000 0 0 0" in output
    assert "3 2 1 0.0 2.000000 2.000000 3.000000 0 0 0" in output
    assert "4 2 1 0.0 2.000000 2.000000 4.000000 0 0 0" in output

    summary = json.loads((output_dir / "rebox_summary.json").read_text(encoding="utf-8"))
    assert summary["n_files"] == 1
    assert summary["changed_atoms_total"] == 1
    assert summary["crossing_components_total"] == 1
    assert summary["max_raw_bond_dz"] == 1.0
