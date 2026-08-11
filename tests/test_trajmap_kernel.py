"""Pin the per-frame mapping kernel: weights, periodic images, buffer reuse.

The kernel is all whole-array NumPy over preallocated scratch, so the failure
modes are (a) an index/broadcast slip that silently mixes sites and (b) scratch
left dirty between frames. Every test here therefore checks against an explicit
per-site Python loop, and one checks that mapping the same frame twice is
bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from AceCG.compute.cgmap import CGMapper, MappedFrame
from AceCG.topology.cgmap import CGMapSpec, expand_mapping_sites, load_mapping_yaml
from real_frames import DPPC_MARTINI12

# Shipped fixture (tests/test_data/dppc_martini12/, see its README.md).
ARCHIVE_MAP = DPPC_MARTINI12 / "map.yaml"

L = 30.0
CUBIC = np.array([L, L, L, 90.0, 90.0, 90.0])


# ─── reference implementation ─────────────────────────────────────────


def slow_map(mapping, positions, *, box=None, unwrap="molecule", weight_key="x-weight"):
    """Site values by an explicit per-site loop, imaging one atom at a time.

    ``molecule`` imaging needs the compiled spec's own reference choice, so the
    bookkeeping is read off the spec while the arithmetic stays naive.
    """
    spec = CGMapSpec.from_mapping(mapping)
    sites, _, _ = expand_mapping_sites(mapping)
    mol_ref_abs = spec.atom_indices[spec.mol_ref_pos]
    lengths = None if box is None else np.asarray(box, dtype=np.float64)[:3]

    out = np.zeros((len(sites), 3), dtype=np.float64)
    for row, (type_key, anchor) in enumerate(sites):
        entry = mapping["site-types"][type_key]
        idx = anchor + np.asarray(entry["index"], dtype=np.int64)
        weight = np.asarray(
            entry.get(weight_key, np.ones(idx.size)), dtype=np.float64
        )
        pos = positions[idx].astype(np.float64)
        if lengths is not None and unwrap != "none":
            if unwrap == "molecule":
                ref = positions[mol_ref_abs[int(spec.site_mol_ids[row])]].astype(np.float64)
            else:
                ref = positions[idx[0]].astype(np.float64)
            shifts = np.rint((pos - ref) / lengths)
            if unwrap == "deprecated":
                shifts = np.clip(shifts, -1.0, 1.0)
            pos = pos - shifts * lengths
        if weight_key == "x-weight":
            weight = weight / weight.sum()
        out[row] = (pos * weight[:, None]).sum(axis=0)
    return out


def two_site_mapping(**overrides):
    """One molecule per repeat unit, two sites of two atoms each."""
    mapping = {
        "site-types": {
            "A": {"index": [0, 1], "x-weight": [1.0, 1.0], "f-weight": [1.0, 1.0]},
            "B": {"index": [0, 1], "x-weight": [3.0, 1.0], "f-weight": [2.0, 5.0]},
        },
        "system": [{"anchor": 0, "repeat": 2, "offset": 4, "sites": [["A", 0], ["B", 2]]}],
    }
    mapping.update(overrides)
    return mapping


def single_site_mapping():
    mapping = {
        "site-types": {"A": {"index": [0, 1], "x-weight": [1.0, 1.0]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["A", 0]]}],
    }
    return mapping


# ─── weights, with no periodic box ────────────────────────────────────


def test_positions_are_weighted_averages_and_match_the_slow_loop():
    mapping = two_site_mapping()
    rng = np.random.default_rng(4711)
    positions = rng.random((8, 3)) * L
    got = CGMapper(mapping_spec(mapping), unwrap="none", wrap=False).map_frame(positions)
    assert isinstance(got, MappedFrame)
    assert got.positions.shape == (4, 3)
    assert got.positions == pytest.approx(slow_map(mapping, positions, unwrap="none"))


def test_forces_are_weighted_sums_with_raw_f_weight():
    mapping = two_site_mapping()
    rng = np.random.default_rng(99)
    positions = rng.random((8, 3)) * L
    forces = rng.normal(size=(8, 3))
    cg = CGMapper(mapping_spec(mapping), wrap=False).map_frame(positions, forces=forces)
    # Site B has f-weight [2, 5]: a *sum*, so it is not bounded by the inputs.
    want = np.array([forces[2] * 2.0 + forces[3] * 5.0])
    assert cg.forces[1] == pytest.approx(want[0])
    assert cg.forces[0] == pytest.approx(forces[0] + forces[1])


def test_flat_force_layout_from_iter_frames_is_accepted():
    mapping = two_site_mapping()
    rng = np.random.default_rng(5)
    positions = rng.random((8, 3)) * L
    forces = rng.normal(size=(8, 3))
    mapper = CGMapper(mapping_spec(mapping), wrap=False)
    flat = mapper.map_frame(positions, forces=forces.astype(np.float32).ravel())
    shaped = mapper.map_frame(positions, forces=forces)
    assert flat.forces == pytest.approx(shaped.forces, rel=1e-6)


def test_wrong_length_flat_forces_are_rejected():
    mapper = CGMapper(mapping_spec(two_site_mapping()))
    with pytest.raises(ValueError, match="flat forces have"):
        mapper.map_frame(np.zeros((8, 3)), forces=np.zeros(17))


def test_velocities_use_position_weights_giving_the_com_velocity():
    mapping = two_site_mapping()
    rng = np.random.default_rng(7)
    positions = rng.random((8, 3)) * L
    velocities = rng.normal(size=(8, 3))
    cg = CGMapper(mapping_spec(mapping), wrap=False).map_frame(
        positions, velocities=velocities
    )
    # Site B x-weight [3, 1] → COM velocity of a 3:1 mass pair.
    assert cg.velocities[1] == pytest.approx(
        (3.0 * velocities[2] + velocities[3]) / 4.0
    )
    assert cg.forces is None


# ─── periodic images ──────────────────────────────────────────────────


def test_a_site_split_across_the_boundary_is_made_whole():
    """The bug this kernel exists to avoid: a site averaging to mid-box."""
    mapping = single_site_mapping()
    positions = np.array([[0.5, 5.0, 5.0], [L - 0.5, 5.0, 5.0]])
    whole = CGMapper(mapping_spec(mapping), unwrap="bead", wrap=False).map_frame(
        positions, box=CUBIC
    )
    assert whole.positions[0] == pytest.approx([0.0, 5.0, 5.0])

    naive = CGMapper(mapping_spec(mapping), unwrap="none", wrap=False).map_frame(
        positions, box=CUBIC
    )
    assert naive.positions[0] == pytest.approx([L / 2.0, 5.0, 5.0])


def test_molecule_unwrap_keeps_both_sites_of_one_molecule_on_the_same_side():
    """Sites are imaged against a shared molecule reference, not each other."""
    mapping = two_site_mapping()
    # Molecule 0 spans the boundary: atoms 0,1 near x=1, atoms 2,3 wrapped to x≈29.
    positions = np.array(
        [
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [L - 1.0, 0.0, 0.0], [L - 1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0], [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0], [12.0, 0.0, 0.0],
        ]
    )
    cg = CGMapper(mapping_spec(mapping), unwrap="molecule", wrap=False).map_frame(
        positions, box=CUBIC
    )
    # Site B of molecule 0 sits at -1, one box length below its wrapped position,
    # so the CG A–B distance is the true 2 Å and not 28 Å.
    assert cg.positions[1][0] == pytest.approx(-1.0)
    assert abs(cg.positions[0][0] - cg.positions[1][0]) == pytest.approx(2.0)
    # Molecule 1 is whole already and untouched.
    assert cg.positions[2][0] == pytest.approx(10.0)
    assert cg.positions[3][0] == pytest.approx(12.0)


def test_bead_unwrap_images_each_site_against_its_own_first_atom():
    mapping = two_site_mapping()
    positions = np.array(
        [
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [L - 1.0, 0.0, 0.0], [L - 1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0], [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0], [12.0, 0.0, 0.0],
        ]
    )
    cg = CGMapper(mapping_spec(mapping), unwrap="bead", wrap=False).map_frame(
        positions, box=CUBIC
    )
    # Each site is already whole about its own first atom, so nothing moves.
    assert cg.positions[1][0] == pytest.approx(L - 1.0)
    assert cg.positions[0][0] == pytest.approx(1.0)


def test_molecule_and_bead_unwrap_match_the_slow_loop_on_a_scrambled_frame():
    mapping = two_site_mapping()
    rng = np.random.default_rng(20260729)
    positions = rng.random((8, 3)) * L
    for unwrap in ("molecule", "bead", "none"):
        cg = CGMapper(mapping_spec(mapping), unwrap=unwrap, wrap=False).map_frame(
            positions, box=CUBIC
        )
        want = slow_map(mapping, positions, box=CUBIC, unwrap=unwrap)
        assert cg.positions == pytest.approx(want), unwrap


def test_deprecated_mode_reproduces_openmscgs_single_shift_and_is_wrong():
    """A site spanning more than one box length: one shift is not enough."""
    mapping = single_site_mapping()
    positions = np.array([[0.0, 0.0, 0.0], [2.5 * L, 0.0, 0.0]])
    spec = mapping_spec(mapping)

    exact = CGMapper(spec, unwrap="bead", wrap=False).map_frame(positions, box=CUBIC)
    legacy = CGMapper(spec, unwrap="deprecated", wrap=False).map_frame(positions, box=CUBIC)

    # d/L = 2.5 → rint 2 (wait, rint(2.5)=2 under banker's rounding) → -2L,
    # leaving +0.5L; the single shift can only remove one L, leaving +1.5L.
    assert exact.positions[0][0] == pytest.approx(0.5 * L / 2.0)
    assert legacy.positions[0][0] == pytest.approx(1.5 * L / 2.0)
    assert legacy.positions[0] == pytest.approx(
        slow_map(mapping, positions, box=CUBIC, unwrap="deprecated")[0]
    )


def test_deprecated_and_exact_agree_while_sites_are_smaller_than_the_box():
    mapping = two_site_mapping()
    rng = np.random.default_rng(31573)
    positions = rng.random((8, 3)) * L
    spec = mapping_spec(mapping)
    exact = CGMapper(spec, unwrap="bead", wrap=False).map_frame(positions, box=CUBIC)
    legacy = CGMapper(spec, unwrap="deprecated", wrap=False).map_frame(positions, box=CUBIC)
    assert exact.positions == pytest.approx(legacy.positions)


def test_a_degenerate_box_disables_imaging_and_wrapping():
    mapping = single_site_mapping()
    positions = np.array([[0.5, 5.0, 5.0], [L - 0.5, 5.0, 5.0]])
    for box in (None, np.zeros(6), np.array([L, 0.0, L, 90.0, 90.0, 90.0])):
        cg = CGMapper(mapping_spec(mapping), unwrap="bead", wrap=True).map_frame(
            positions, box=box
        )
        assert cg.positions[0] == pytest.approx([L / 2.0, 5.0, 5.0])


def test_triclinic_exact_imaging_makes_a_split_site_whole():
    mapping = single_site_mapping()
    box = np.array([L, L, L, 90.0, 90.0, 60.0])
    from MDAnalysis.lib.mdamath import triclinic_vectors

    lattice = triclinic_vectors(box)
    positions = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0] + lattice[1]])
    cg = CGMapper(mapping_spec(mapping), unwrap="bead", wrap=False).map_frame(
        positions, box=box
    )
    assert cg.positions[0] == pytest.approx([2.0, 3.0, 4.0])


def test_triclinic_fast_mode_agrees_when_displacements_are_short():
    mapping = two_site_mapping()
    box = np.array([L, L, L, 90.0, 90.0, 75.0])
    rng = np.random.default_rng(11)
    # Sites well inside the cell: both strategies must pick the same image.
    positions = rng.random((8, 3)) * 2.0 + 10.0
    spec = mapping_spec(mapping)
    exact = CGMapper(spec, unwrap="bead", triclinic="exact", wrap=False)
    fast = CGMapper(spec, unwrap="bead", triclinic="fast", wrap=False)
    assert exact.map_frame(positions, box=box).positions == pytest.approx(
        fast.map_frame(positions, box=box).positions
    )


# ─── wrapping and lattice invariance ──────────────────────────────────


def test_wrap_puts_every_site_inside_the_primary_cell():
    mapping = two_site_mapping()
    positions = np.array(
        [
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [L - 1.0, 0.0, 0.0], [L - 1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0], [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0], [12.0, 0.0, 0.0],
        ]
    )
    cg = CGMapper(mapping_spec(mapping), unwrap="molecule", wrap=True).map_frame(
        positions, box=CUBIC
    )
    assert np.all(cg.positions >= 0.0) and np.all(cg.positions < L)
    # The site that unwrapping put at -1 comes back at L-1.
    assert cg.positions[1][0] == pytest.approx(L - 1.0)


def test_translating_one_molecule_by_a_lattice_vector_leaves_cg_positions_fixed():
    mapping = two_site_mapping()
    rng = np.random.default_rng(2718)
    positions = rng.random((8, 3)) * L
    mapper = CGMapper(mapping_spec(mapping), unwrap="molecule", wrap=True)
    base = mapper.map_frame(positions, box=CUBIC).positions.copy()

    shifted = positions.copy()
    shifted[4:] += np.array([L, -2.0 * L, 0.0])  # molecule 1, whole
    got = mapper.map_frame(shifted, box=CUBIC).positions
    assert got == pytest.approx(base, abs=1e-9)


# ─── buffer discipline ────────────────────────────────────────────────


def test_mapping_the_same_frame_twice_is_bit_identical():
    """Catches scratch buffers left weighted or imaged from the previous frame."""
    mapping = two_site_mapping()
    rng = np.random.default_rng(1234)
    positions = rng.random((8, 3)) * L
    forces = rng.normal(size=(8, 3))
    mapper = CGMapper(mapping_spec(mapping), unwrap="molecule")
    first = mapper.map_frame(positions, box=CUBIC, forces=forces)
    second = mapper.map_frame(positions, box=CUBIC, forces=forces)
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.forces, second.forces)


def test_alternating_frames_do_not_leak_state():
    mapping = two_site_mapping()
    rng = np.random.default_rng(4321)
    frame_a = rng.random((8, 3)) * L
    frame_b = rng.random((8, 3)) * L
    mapper = CGMapper(mapping_spec(mapping), unwrap="molecule")
    want_a = mapper.map_frame(frame_a, box=CUBIC).positions.copy()
    mapper.map_frame(frame_b, box=CUBIC)
    assert np.array_equal(mapper.map_frame(frame_a, box=CUBIC).positions, want_a)


def test_input_arrays_are_never_modified():
    mapping = two_site_mapping()
    rng = np.random.default_rng(808)
    positions = rng.random((8, 3)) * L
    forces = rng.normal(size=(8, 3))
    frozen_positions = positions.copy()
    frozen_forces = forces.copy()
    CGMapper(mapping_spec(mapping), unwrap="molecule").map_frame(
        positions, box=CUBIC, forces=forces
    )
    assert np.array_equal(positions, frozen_positions)
    assert np.array_equal(forces, frozen_forces)


def test_float32_input_is_accepted_and_out_dtype_is_honoured():
    mapping = two_site_mapping()
    rng = np.random.default_rng(64)
    positions = (rng.random((8, 3)) * L).astype(np.float32)
    mapper = CGMapper(mapping_spec(mapping), wrap=False, out_dtype=np.float32)
    cg = mapper.map_frame(positions, box=CUBIC)
    assert cg.positions.dtype == np.float32
    exact = CGMapper(mapping_spec(mapping), wrap=False).map_frame(
        positions.astype(np.float64), box=CUBIC
    )
    assert cg.positions == pytest.approx(exact.positions, rel=1e-6)


def test_float32_working_precision_stays_close_to_float64():
    mapping = two_site_mapping()
    rng = np.random.default_rng(65)
    positions = rng.random((8, 3)) * L
    cheap = CGMapper(mapping_spec(mapping), wrap=False, dtype=np.float32)
    exact = CGMapper(mapping_spec(mapping), wrap=False, dtype=np.float64)
    assert cheap.map_frame(positions, box=CUBIC).positions == pytest.approx(
        exact.map_frame(positions, box=CUBIC).positions, rel=1e-6
    )


# ─── contracts ────────────────────────────────────────────────────────


def test_unknown_unwrap_mode_is_rejected():
    with pytest.raises(ValueError, match="unwrap must be one of"):
        CGMapper(mapping_spec(two_site_mapping()), unwrap="whole")


def test_bad_triclinic_mode_is_rejected():
    with pytest.raises(ValueError, match="triclinic must be"):
        CGMapper(mapping_spec(two_site_mapping()), triclinic="approximate")


def test_frame_shorter_than_the_mapping_is_rejected():
    mapper = CGMapper(mapping_spec(two_site_mapping()))
    with pytest.raises(ValueError, match="only 4 atoms"):
        mapper.map_frame(np.zeros((4, 3)), box=CUBIC)


def test_non_xyz_positions_are_rejected():
    mapper = CGMapper(mapping_spec(two_site_mapping()))
    with pytest.raises(ValueError, match=r"shape \(n_atoms, 3\)"):
        mapper.map_frame(np.zeros((8, 2)))


def test_frame_id_and_box_are_carried_through():
    mapper = CGMapper(mapping_spec(two_site_mapping()))
    cg = mapper.map_frame(np.zeros((8, 3)), box=CUBIC, frame_id=17)
    assert cg.frame_id == 17
    assert cg.box == pytest.approx(CUBIC)
    # A copy, so a caller reusing its frame buffer cannot corrupt the result.
    assert cg.box is not CUBIC


def test_interleaved_molecules_take_the_gather_path():
    """``mol_atom_pos`` is not None here, so the per-atom reference is scattered."""
    mapping = {
        "site-types": {"A": {"index": [0, 2], "x-weight": [1.0, 1.0]}},
        "system": [{"anchor": 0, "repeat": 2, "offset": 1, "sites": [["A", 0]]}],
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert not spec.molecules_contiguous
    positions = np.array(
        [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [L - 0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    cg = CGMapper(spec, unwrap="molecule", wrap=False).map_frame(positions, box=CUBIC)
    assert cg.positions.shape == (2, 3)
    assert cg.positions == pytest.approx(
        slow_map(mapping, positions, box=CUBIC, unwrap="molecule")
    )


def test_site_masses_and_type_ids_are_exposed():
    mapper = CGMapper(mapping_spec(two_site_mapping()))
    assert mapper.mapped_shape() == (4, 3)
    assert list(mapper.site_type_ids()) == [1, 2, 1, 2]
    # x-weight sums: A = 2, B = 4.
    assert mapper.site_masses() == pytest.approx([2.0, 4.0, 2.0, 4.0])
    assert mapper.nbytes() > 0


# ─── the real archive mapping, at scale ───────────────────────────────


def test_archive_mapping_matches_the_slow_loop_under_molecule_unwrap():
    mapping = load_mapping_yaml(ARCHIVE_MAP)
    spec = CGMapSpec.from_mapping(mapping, n_atoms=599040)
    rng = np.random.default_rng(20260729)
    box = np.array([120.0, 120.0, 140.0, 90.0, 90.0, 90.0])
    positions = rng.random((599040, 3)) * box[:3]

    cg = CGMapper(spec, unwrap="molecule", wrap=False).map_frame(positions, box=box)
    assert cg.positions.shape == (27648, 3)

    # Spot-check whole molecules against the naive loop rather than all 27,648
    # sites: the loop is ~1 ms per site.
    sites, _, _ = expand_mapping_sites(mapping)
    mol_ref_abs = spec.atom_indices[spec.mol_ref_pos]
    lengths = box[:3]
    for site in rng.choice(spec.n_sites, size=24, replace=False):
        type_key, anchor = sites[int(site)]
        entry = mapping["site-types"][type_key]
        idx = anchor + np.asarray(entry["index"], dtype=np.int64)
        weight = np.asarray(entry["x-weight"], dtype=np.float64)
        ref = positions[mol_ref_abs[int(spec.site_mol_ids[int(site)])]]
        pos = positions[idx] - lengths * np.rint((positions[idx] - ref) / lengths)
        want = (pos * weight[:, None]).sum(axis=0) / weight.sum()
        assert cg.positions[int(site)] == pytest.approx(want, abs=1e-9)


def test_archive_scratch_stays_within_a_sensible_budget():
    spec = CGMapSpec.from_mapping(load_mapping_yaml(ARCHIVE_MAP), n_atoms=599040)
    mapper = CGMapper(spec, unwrap="molecule", dtype=np.float32)
    # 599040 atoms + 599040 nnz, float32 xyz → ~14 MB per rank before imaging.
    assert mapper.nbytes() < 48 * 1024 * 1024


def test_compact_input_matches_the_full_frame_bit_for_bit():
    """`compact=True` may only skip the gather, never change the arithmetic."""
    mapping = {
        "site-types": {
            "H": {"index": [0, 1], "x-weight": [14.0, 1.0], "f-weight": [1.0, 1.0]},
            "T": {"index": [0, 1, 2], "x-weight": [12.0, 1.0, 1.0]},
        },
        # Anchored past atom 0 and strided, so the mapping ignores real atoms and
        # the compact buffer is a genuine subset rather than the whole frame.
        "system": [{"anchor": 3, "repeat": 2, "offset": 8, "sites": [["H", 0], ["T", 2]]}],
    }
    spec = CGMapSpec.from_mapping(mapping, n_atoms=24)
    assert spec.n_required_atoms < 24

    rng = np.random.default_rng(20260730)
    positions = rng.random((24, 3)) * 37.0
    forces = rng.standard_normal((24, 3))
    box = np.array([37.0, 37.0, 37.0, 90.0, 90.0, 90.0])

    for unwrap in ("molecule", "bead", "none"):
        mapper = CGMapper(spec, unwrap=unwrap)
        full = mapper.map_frame(positions, box=box, forces=forces)
        gathered = mapper.map_frame(
            positions[spec.atom_indices],
            box=box,
            forces=forces[spec.atom_indices].ravel(),
            compact=True,
        )
        np.testing.assert_array_equal(gathered.positions, full.positions)
        np.testing.assert_array_equal(gathered.forces, full.forces)

    with pytest.raises(ValueError, match="compact=True expects"):
        CGMapper(spec).map_frame(positions, box=box, compact=True)


# ─── helper ───────────────────────────────────────────────────────────


def mapping_spec(mapping):
    return CGMapSpec.from_mapping(mapping)
