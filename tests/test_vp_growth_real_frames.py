"""Real-frame oracle for VP growth physics (F-013).

``tests/test_vp_growth_terminal.py`` covers the terminal — preflight, staging,
manifest-last publication — against an 8-atom synthetic universe. What it
cannot cover is whether the grown virtual particles are in physically sensible
places, because an 8-atom box has no packing to clash against.

This file drives ``VPGrower.grow_frame`` (the unchanged single-frame scientific
kernel) over the five real DOPC frames in ``tests/test_data/dopc_cg6/``, using
the real DOPC VP definition from ``data/dopc_ld_Pak2019/topo/vp_growth_config.json``:
one VP per lipid, bonded to the head group at r0 = 1.5 Å with a VP-HG-MG angle
target of 130.25°, clash floor 1.5 Å.

The fixture patch is cut at the periodic corner of the real box, so the clash
resolver is genuinely working against minimum-image neighbours rather than an
empty box.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from real_frames import dopc_universe

from AceCG.configs.vp_config import VPAtomDef, VPConfig, VPInteractionDef
from AceCG.topology.vpgrower import VPGrower


SITES_PER_LIPID = 6
N_LIPIDS = 96
CLASH_MIN_DISTANCE = 1.5
VP_BOND_LENGTH = 1.5
VP_ANGLE_TARGET = 130.25
SEED_BASE = 100


def _vp_config() -> VPConfig:
    """The production DOPC VP definition, transcribed field for field.

    Source: ``data/dopc_ld_Pak2019/topo/vp_growth_config.json`` — one VP name,
    mass 100, harmonic VP-HG bond (k 2.5, r0 1.5), harmonic VP-HG-MG angle
    (k 2.45, theta0 130.25), types appended after the real sites.
    """
    return VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=100.0),),
        bonds=(
            VPInteractionDef(
                type_keys=("VP", "HG"),
                pot_style="harmonic",
                pot_kwargs={"k": 2.5, "r0": VP_BOND_LENGTH},
            ),
        ),
        angles=(
            VPInteractionDef(
                type_keys=("VP", "HG", "MG"),
                pot_style="harmonic",
                pot_kwargs={"k": 2.45, "theta0": VP_ANGLE_TARGET},
            ),
        ),
        pairs=(),
        selection=None,
        atomtype_order="back",
        clash_max_passes=8,
        clash_min_distance=CLASH_MIN_DISTANCE,
    )


@pytest.fixture(scope="module")
def grower():
    return VPGrower.from_universe(
        dopc_universe(),
        _vp_config(),
        type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )


@pytest.fixture(scope="module")
def grown(grower):
    """``[(positions, box), ...]`` for every fixture frame, seeded per frame."""
    frames = []
    for timestep in dopc_universe().trajectory:
        positions = np.asarray(timestep.positions, dtype=np.float64)
        box = np.asarray(timestep.dimensions, dtype=np.float64)
        result = grower.grow_frame(
            positions, box, orientation_seed=SEED_BASE + timestep.frame
        )
        frames.append((np.asarray(result.positions, dtype=np.float64), box, positions))
    return frames


def _vp_indices(template) -> np.ndarray:
    return np.asarray(
        sorted(int(i) for group in template.vp_indices_by_name.values() for i in group),
        dtype=np.int64,
    )


def _minimum_image(vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return vectors - lengths * np.rint(vectors / lengths)


def test_template_grows_one_virtual_particle_per_lipid(grower):
    template = grower.template
    assert template.n_real == N_LIPIDS * SITES_PER_LIPID
    assert template.n_vp == N_LIPIDS
    assert template.n_atoms == template.n_real + template.n_vp
    # `atomtype_order="back"` means the VP type is appended, so the real sites
    # keep their original type ids and any existing forcefield stays valid.
    assert max(template.type2id.values()) == template.type2id["VP"]


def test_growing_never_moves_a_real_site(grown, grower):
    """The AA-mapped coordinates must survive untouched, to the last bit."""
    real = grower.template.real_indices
    for positions, _, source in grown:
        np.testing.assert_array_equal(positions[real], source)


def test_growth_is_deterministic_in_the_seed_and_only_the_seed(grower):
    """Same seed reproduces exactly; a different seed moves only the VP sites."""
    timestep = dopc_universe().trajectory[0]
    positions = np.asarray(timestep.positions, dtype=np.float64)
    box = np.asarray(timestep.dimensions, dtype=np.float64)

    first = np.asarray(grower.grow_frame(positions, box, orientation_seed=100).positions)
    repeat = np.asarray(grower.grow_frame(positions, box, orientation_seed=100).positions)
    other = np.asarray(grower.grow_frame(positions, box, orientation_seed=101).positions)

    np.testing.assert_array_equal(first, repeat)
    real = grower.template.real_indices
    np.testing.assert_array_equal(other[real], first[real])
    assert not np.allclose(other[_vp_indices(grower.template)],
                           first[_vp_indices(grower.template)])


def test_every_virtual_particle_sits_at_the_bond_target_from_its_head_group(
    grown, grower
):
    """VP-HG distance is r0, measured under minimum image, on every real frame.

    The clash resolver is allowed to rotate a VP around its anchor but not to
    stretch the bond, so the tolerance here is tight on purpose: a drifting
    bond length would mean the resolver is translating VPs rather than
    reorienting them.
    """
    template = grower.template
    virtual = set(int(index) for index in _vp_indices(template))
    bonds = np.asarray(template.bonds, dtype=np.int64)
    vp_bonds = bonds[
        [bool({int(a), int(b)} & virtual) for a, b in bonds]
    ]
    assert vp_bonds.shape == (N_LIPIDS, 2)

    for positions, box, _ in grown:
        lengths = np.asarray(box[:3], dtype=np.float64)
        delta = _minimum_image(
            positions[vp_bonds[:, 1]] - positions[vp_bonds[:, 0]], lengths
        )
        distances = np.linalg.norm(delta, axis=1)
        np.testing.assert_allclose(distances, VP_BOND_LENGTH, atol=2.0e-3)


def test_the_angle_target_is_met_wherever_packing_allows_it(grown, grower):
    """Most VPs hit theta0 exactly; the resolver sacrifices the rest to clashes.

    Both halves matter. If every angle were exactly 130.25° the clash resolver
    would not be running at all; if none were, the angle target would not be
    reaching the placement.
    """
    template = grower.template
    virtual = set(int(index) for index in _vp_indices(template))
    angles = np.asarray(template.angles, dtype=np.int64)
    vp_angles = angles[[bool(set(map(int, row)) & virtual) for row in angles]]
    assert vp_angles.shape == (N_LIPIDS, 3)

    for positions, box, _ in grown:
        lengths = np.asarray(box[:3], dtype=np.float64)
        left = _minimum_image(
            positions[vp_angles[:, 0]] - positions[vp_angles[:, 1]], lengths
        )
        right = _minimum_image(
            positions[vp_angles[:, 2]] - positions[vp_angles[:, 1]], lengths
        )
        cosine = np.einsum("ij,ij->i", left, right) / (
            np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        )
        theta = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        deviation = np.abs(theta - VP_ANGLE_TARGET)
        # The typical VP is placed exactly on the target: the median deviation
        # is zero, and 87-92 of the 96 lipids land within float noise of it.
        assert float(np.median(deviation)) == pytest.approx(0.0, abs=1.0e-9)
        assert np.count_nonzero(deviation < 1.0e-6) >= 85
        # ... and the remainder were displaced by the clash resolver, which is
        # free to swing the VP either side of the target. If this count ever
        # reached zero the resolver would not be doing anything on real
        # packing, which is the case this fixture exists to prevent.
        assert np.count_nonzero(deviation > 1.0) >= 1


def test_no_grown_frame_leaves_a_non_bonded_pair_inside_the_clash_floor(grown, grower):
    """The clash resolver's whole purpose, checked against real packing.

    Bonded pairs are excluded because the VP-HG bond is 1.5 Å by construction,
    exactly at the floor. Everything else must clear it under minimum image.
    """
    template = grower.template
    bonded = {
        tuple(sorted((int(a), int(b)))) for a, b in np.asarray(template.bonds)
    }
    for positions, box, _ in grown:
        lengths = np.asarray(box[:3], dtype=np.float64)
        tree = cKDTree(np.mod(positions, lengths), boxsize=lengths)
        close = tree.query_pairs(CLASH_MIN_DISTANCE, output_type="ndarray")
        offenders = [
            pair
            for pair in (tuple(sorted(map(int, row))) for row in close)
            if pair not in bonded
        ]
        assert offenders == []


def test_virtual_particles_keep_real_space_clear_by_a_wide_margin(grown, grower):
    """Pin how far the nearest non-bonded neighbour of any VP actually is.

    A golden number rather than a bound: if a change to the placement or the
    resolver starts crowding VPs against real sites, the CD-REM latent
    ensemble changes and nothing else in the suite would notice.
    """
    template = grower.template
    virtual = _vp_indices(template)
    others = np.setdiff1d(np.arange(template.n_atoms, dtype=np.int64), virtual)

    margins = []
    for positions, box, _ in grown:
        lengths = np.asarray(box[:3], dtype=np.float64)
        tree = cKDTree(np.mod(positions[others], lengths), boxsize=lengths)
        # k=2 because the VP's own bonded head group is always the nearest.
        distances, _ = tree.query(np.mod(positions[virtual], lengths), k=2)
        np.testing.assert_allclose(
            distances[:, 0], VP_BOND_LENGTH, atol=2.0e-3
        )
        margins.append(float(distances[:, 1].min()))

    np.testing.assert_allclose(
        margins,
        [3.8394, 3.8742, 3.6531, 2.8603, 3.4909],
        atol=1.0e-3,
    )
