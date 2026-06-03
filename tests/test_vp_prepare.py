"""Serial integration test for :mod:`AceCG.compute.vp_prepare`.

Builds a tiny CG universe with a short synthetic trajectory and
verifies that :func:`grow_vp_frames` emits one data file per frame
with the expected filename shape and atom count. MPI is not invoked
here; that is Phase 4's concern.
"""

from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from AceCG.compute.vp_prepare import grow_vp_frames
from AceCG.configs.vp_config import (
    VPAtomDef,
    VPConfig,
    VPInteractionDef,
)
from AceCG.topology.types import InteractionKey
from AceCG.topology.vpgrower import VPGrower


def _build_universe_with_trajectory(n_frames: int = 5) -> mda.Universe:
    """2-residue × 4-bead CG universe with ``n_frames`` in-memory frames."""
    n_atoms = 8
    atom_resindex = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    u = mda.Universe.empty(
        n_atoms, n_residues=2, atom_resindex=atom_resindex, trajectory=False,
    )
    types = np.array(["1", "2", "3", "4"] * 2, dtype=object)
    u.add_TopologyAttr("types", types)
    u.add_TopologyAttr("names", types.copy())
    u.add_TopologyAttr("masses", np.full(n_atoms, 72.0))
    u.add_TopologyAttr("charges", np.zeros(n_atoms))
    u.add_TopologyAttr("resids", np.array([1, 2], dtype=np.int64))
    u.add_TopologyAttr("resnames", np.array(["DOPC", "DOPC"], dtype=object))
    u.add_TopologyAttr("bonds", np.array(
        [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)], dtype=np.int64,
    ))
    u.add_TopologyAttr("angles", np.array(
        [(0, 1, 2), (4, 5, 6)], dtype=np.int64,
    ))

    base = np.array(
        [
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0],
            [0.0, 6.0, 0.0], [2.0, 6.0, 0.0], [4.0, 6.0, 0.0], [6.0, 6.0, 0.0],
        ],
        dtype=float,
    )
    coords = np.stack([base + float(i) * 0.1 for i in range(n_frames)], axis=0)
    from MDAnalysis.coordinates.memory import MemoryReader
    dims = np.tile(
        np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0]), (n_frames, 1),
    )
    u.trajectory = MemoryReader(
        coords, order="fac", dimensions=dims, dt=1.0,
    )
    return u


def _vp_config() -> VPConfig:
    return VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=(VPInteractionDef(
            type_keys=("VP", "MG"), pot_style="harmonic",
            pot_kwargs={"k": 2.5, "r0": 1.5},
        ),),
        angles=(VPInteractionDef(
            type_keys=("VP", "MG", "HG"), pot_style="harmonic",
            pot_kwargs={"k": 2.45, "theta0": 135.0},
        ),),
        pairs=(),
        selection="resname DOPC",
        atomtype_order="back",
        clash_max_passes=8,
        clash_min_distance=1.5,
    )


def test_grow_vp_frames_serial_emits_per_frame_files(tmp_path: Path) -> None:
    u = _build_universe_with_trajectory(n_frames=5)
    grower = VPGrower.from_universe(
        u, _vp_config(), type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    manifest = grow_vp_frames(
        grower=grower, universe=u,
        frame_ids=[0, 2, 4],
        output_dir=tmp_path,
        orientation_seed_base=42,
    )
    assert manifest is not None
    assert sorted(manifest.frames.keys()) == [0, 2, 4]
    for fid, path in manifest.frames.items():
        p = Path(path)
        assert p.exists()
        assert p.name == f"frame_{fid:06d}.data"
        # Parse back via MDAnalysis to confirm atom count (8 real + 2 VP).
        u2 = mda.Universe(
            str(p), format="DATA",
            atom_style="id resid type charge x y z",
        )
        assert len(u2.atoms) == 10
    # No forces requested ⇒ all forces entries are None.
    assert all(v is None for v in manifest.forces.values())


def test_grow_vp_frames_overwrite_guard(tmp_path: Path) -> None:
    u = _build_universe_with_trajectory(n_frames=2)
    grower = VPGrower.from_universe(
        u, _vp_config(), type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    grow_vp_frames(
        grower=grower, universe=u, frame_ids=[0],
        output_dir=tmp_path, orientation_seed_base=0,
    )
    with pytest.raises(FileExistsError):
        grow_vp_frames(
            grower=grower, universe=u, frame_ids=[0],
            output_dir=tmp_path, orientation_seed_base=0,
        )
    # Overwrite=True succeeds.
    grow_vp_frames(
        grower=grower, universe=u, frame_ids=[0],
        output_dir=tmp_path, orientation_seed_base=0, overwrite=True,
    )


def test_grow_vp_frames_determinism(tmp_path: Path) -> None:
    u = _build_universe_with_trajectory(n_frames=3)
    grower = VPGrower.from_universe(
        u, _vp_config(), type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    grow_vp_frames(grower=grower, universe=u, frame_ids=[1],
                   output_dir=out_a, orientation_seed_base=7)
    grow_vp_frames(grower=grower, universe=u, frame_ids=[1],
                   output_dir=out_b, orientation_seed_base=7)
    a = (out_a / "frame_000001.data").read_text()
    b = (out_b / "frame_000001.data").read_text()
    assert a == b


def test_grow_vp_frames_include_forces_requires_force_columns(tmp_path: Path) -> None:
    u = _build_universe_with_trajectory(n_frames=1)
    grower = VPGrower.from_universe(
        u, _vp_config(), type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    with pytest.warns(UserWarning, match="continuing without force output"):
        manifest = grow_vp_frames(
            grower=grower, universe=u, frame_ids=[0],
            output_dir=tmp_path, orientation_seed_base=0, include_forces=True,
        )
    assert manifest is not None
    assert manifest.forces[0] is None


def test_grow_vp_frames_uses_global_frame_ids_with_local_frame_remap(tmp_path: Path) -> None:
    u = _build_universe_with_trajectory(n_frames=3)
    grower = VPGrower.from_universe(
        u, _vp_config(), type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    manifest = grow_vp_frames(
        grower=grower,
        universe=u,
        frame_ids=[100, 105, 110],
        local_frame_ids=[0, 1, 2],
        output_dir=tmp_path,
        orientation_seed_base=9,
    )
    assert manifest is not None
    assert sorted(manifest.frames) == [100, 105, 110]
    assert (tmp_path / "frame_000100.data").exists()
    assert (tmp_path / "frame_000105.data").exists()
    assert (tmp_path / "frame_000110.data").exists()


def test_angle_key_preserves_center_atom() -> None:
    key = InteractionKey.angle("VP", "HG", "MG")
    assert key.types[1] == "HG"
