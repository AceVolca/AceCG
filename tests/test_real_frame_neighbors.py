"""Real-frame oracle for the pair search and the minimum-image convention.

Review brief §3 names "periodic boundary / minimum-image handling" as part of
the numerically load-bearing surface, and `topology/neighbor.py` is where it
enters the FM/REM path. Nothing pinned it: the suite exercised minimum image
only indirectly, through quantities that would also change for a dozen other
reasons.

These tests recompute the pair list from first principles on the five real
frames of ``tests/test_data/dopc_cg6/`` — an O(N²) enumeration over all 576
sites with the minimum image taken by hand from the real 196×196×73 box — and
compare the result to ``compute_pairs_by_type`` as an exact set, per
interaction key. The fixture is cut at the periodic corner of the production
box precisely so that a dropped minimum image cannot hide: about 40% of the
sub-cutoff pairs are separated by a boundary.

A brute-force reference is the right oracle here because it shares no code
with the thing under test: ``compute_pairs_by_type`` goes through MDAnalysis'
``FastNS`` grid search, integer type codes and the cached exclusion mask, while
the reference below is three lines of numpy. They agree only if the convention
is right.
"""

from __future__ import annotations

import numpy as np
import pytest

from real_frames import (
    DOPC_CUTOFF,
    DOPC_EXCLUDE_OPTION,
    dopc_topology_arrays,
    dopc_universe,
)

from AceCG.compute.frame_geometry import compute_frame_geometry
from AceCG.topology.neighbor import compute_pairs_by_type


def _frames():
    """``(positions, box)`` for every fixture frame, in file order."""
    out = []
    for timestep in dopc_universe().trajectory:
        out.append(
            (
                np.asarray(timestep.positions, dtype=np.float64),
                np.asarray(timestep.dimensions, dtype=np.float64),
            )
        )
    return out


def _pair_keys():
    """Every unordered type pair of the four DOPC bead types."""
    from AceCG.topology.types import InteractionKey

    names = ["HG", "MG", "T1", "T2"]
    return [
        InteractionKey(style="pair", types=(names[i], names[j]))
        for i in range(len(names))
        for j in range(i, len(names))
    ]


def _brute_force_pairs(positions, box, topology, keys, cutoff):
    """``{key: set of (i, j)}`` from an O(N²) minimum-image enumeration.

    Shares no implementation with the code under test: full pair enumeration,
    displacement, explicit minimum image against the orthorhombic box lengths,
    then the same exclusion rules stated independently.
    """
    n_atoms = int(positions.shape[0])
    lengths = np.asarray(box[:3], dtype=np.float64)

    rows, cols = np.triu_indices(n_atoms, k=1)
    delta = positions[cols] - positions[rows]
    delta -= lengths * np.round(delta / lengths)
    distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    within = distances <= float(cutoff)
    rows, cols = rows[within], cols[within]

    # Exclusions, restated rather than reused: same residue, plus every
    # bonded-exclusion pair the topology carries.
    resids = np.asarray(topology.resids)[np.asarray(topology.atom_resindex)]
    keep = resids[rows] != resids[cols]
    bonded = set()
    for pairs in (topology.exclude_12, topology.exclude_13, topology.exclude_14):
        arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        for a, b in arr:
            bonded.add((int(min(a, b)), int(max(a, b))))
    if bonded:
        keep &= np.array(
            [(int(a), int(b)) not in bonded for a, b in zip(rows, cols)],
            dtype=bool,
        )
    rows, cols = rows[keep], cols[keep]

    type_names = np.asarray(
        [
            topology.atom_type_code_to_name[int(code)]
            for code in np.asarray(topology.atom_type_codes)
        ]
    )
    out = {key: set() for key in keys}
    for a, b in zip(rows, cols):
        pair = (type_names[a], type_names[b])
        for key in keys:
            if set(pair) == set(key.types) and (
                key.types[0] != key.types[1] or pair[0] == key.types[0]
            ):
                out[key].add((int(min(a, b)), int(max(a, b))))
                break
    return out


def _engine_pairs(positions, box, topology, keys, cutoff):
    """``{key: set of (i, j)}`` from the production pair search."""
    cache = compute_pairs_by_type(
        np.asarray(positions, dtype=np.float32),
        np.asarray(box, dtype=np.float32),
        keys,
        cutoff,
        topology,
        exclude_option=DOPC_EXCLUDE_OPTION,
    )
    return {
        key: {(int(min(a, b)), int(max(a, b))) for a, b in zip(*cache[key])}
        for key in keys
    }


@pytest.mark.parametrize("frame_index", range(5))
def test_pair_search_matches_a_brute_force_minimum_image_enumeration(frame_index):
    """The production pair list equals the O(N²) reference on every real frame.

    A dropped or wrong minimum image shows up as missing pairs across the box
    boundary; a wrong exclusion rule shows up as extra ones. Neither is
    survivable by an exact set comparison.
    """
    positions, box = _frames()[frame_index]
    topology = dopc_topology_arrays()
    keys = _pair_keys()

    reference = _brute_force_pairs(positions, box, topology, keys, DOPC_CUTOFF)
    produced = _engine_pairs(positions, box, topology, keys, DOPC_CUTOFF)

    total = sum(len(v) for v in reference.values())
    assert total > 10_000, f"reference found only {total} pairs; fixture is wrong"

    for key in keys:
        missing = reference[key] - produced[key]
        extra = produced[key] - reference[key]
        # FastNS works in float32 and the reference in float64, so a pair sitting
        # within one float32 ulp of the cutoff may legitimately fall either way.
        for i, j in missing | extra:
            delta = positions[j] - positions[i]
            delta -= box[:3] * np.round(delta / box[:3])
            assert abs(float(np.linalg.norm(delta)) - DOPC_CUTOFF) < 1.0e-3, (
                f"{key.label()}: pair ({i}, {j}) differs from the brute-force "
                "reference and is not at the cutoff boundary"
            )


def test_minimum_image_is_actually_exercised_by_the_fixture():
    """A large share of the kept pairs really do cross a periodic boundary.

    Without this the set comparison above could pass on a fixture where the
    minimum image never fires, which is exactly the failure mode the test is
    written to exclude.
    """
    positions, box = _frames()[0]
    topology = dopc_topology_arrays()
    keys = _pair_keys()
    produced = _engine_pairs(positions, box, topology, keys, DOPC_CUTOFF)

    crossing = 0
    total = 0
    for pairs in produced.values():
        for i, j in pairs:
            raw = positions[j] - positions[i]
            wrapped = raw - box[:3] * np.round(raw / box[:3])
            total += 1
            if not np.allclose(raw, wrapped):
                crossing += 1
    assert total > 10_000
    assert crossing / total > 0.2, (
        f"only {crossing}/{total} pairs cross a boundary; the fixture no longer "
        "exercises the minimum image"
    )


@pytest.mark.parametrize("frame_index", range(5))
def test_frame_geometry_distances_equal_the_minimum_image_distances(frame_index):
    """``compute_frame_geometry`` reports the minimum-image distance, not the raw one.

    The pair *set* can be right while the *distances* fed to the potentials are
    computed without wrapping; that would leave the neighbour list intact and
    silently corrupt every energy and force. This checks the distances directly.
    """
    positions, box = _frames()[frame_index]
    topology = dopc_topology_arrays()
    keys = _pair_keys()

    cache = compute_pairs_by_type(
        np.asarray(positions, dtype=np.float32),
        np.asarray(box, dtype=np.float32),
        keys,
        DOPC_CUTOFF,
        topology,
        exclude_option=DOPC_EXCLUDE_OPTION,
    )
    geometry = compute_frame_geometry(
        np.asarray(positions, dtype=np.float32),
        np.asarray(box, dtype=np.float32),
        topology,
        pair_cache=cache,
    )

    checked = 0
    for key in keys:
        a_idx, b_idx = geometry.pair_indices[key]
        if a_idx.size == 0:
            continue
        delta = positions[b_idx] - positions[a_idx]
        delta -= box[:3] * np.round(delta / box[:3])
        expected = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        np.testing.assert_allclose(
            np.asarray(geometry.pair_distances[key], dtype=np.float64),
            expected,
            rtol=1.0e-5,
            atol=1.0e-4,
        )
        checked += int(a_idx.size)
    assert checked > 10_000
