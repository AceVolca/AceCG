from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import MDAnalysis as mda

from AceCG.topology.mscg import (
    attach_topology_from_mscg_top,
    build_replicated_topology_arrays,
    parse_mscg_top,
)


def _write_top_in(path: Path) -> None:
    path.write_text(
        dedent(
            """
            cgsites 6
            cgtypes 2
            A
            B
            moltypes 1
            mol 3 1
            sitetypes
            A
            B
            A
            bonds 2
            1 2
            2 3
            system 1
            1 2
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_parse_mscg_top_and_build_replicated_arrays(tmp_path: Path) -> None:
    top_path = tmp_path / "top.in"
    _write_top_in(top_path)

    parsed = parse_mscg_top(top_path)
    arrays = build_replicated_topology_arrays(parsed)

    assert parsed.atom_type_names == ["A", "B"]
    assert parsed.n_cgsites_declared == 6
    assert parsed.system_counts == [(1, 2)]
    assert arrays["atom_types"].tolist() == ["A", "B", "A", "A", "B", "A"]
    assert arrays["bonds"].shape == (4, 2)
    assert arrays["angles"].shape == (2, 3)
    assert arrays["dihedrals"].shape == (0, 4)
    assert arrays["bond_types"].tolist() == ["A:B", "A:B", "A:B", "A:B"]
    assert arrays["angle_types"].tolist() == ["A:B:A", "A:B:A"]


def test_attach_topology_from_mscg_top_populates_universe(tmp_path: Path) -> None:
    top_path = tmp_path / "top.in"
    _write_top_in(top_path)

    u = mda.Universe.empty(
        6,
        n_residues=2,
        atom_resindex=[0, 0, 0, 1, 1, 1],
        trajectory=True,
    )

    arrays = attach_topology_from_mscg_top(u, top_path)

    assert arrays["atom_types"].tolist() == ["A", "B", "A", "A", "B", "A"]
    assert u.atoms.types.tolist() == ["A", "B", "A", "A", "B", "A"]
    assert len(u.bonds) == 4
    assert len(u.angles) == 2
