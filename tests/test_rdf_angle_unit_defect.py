"""Pins FINDINGS.md F-033: RDF angle/dihedral histogram units.

`compute_frame_geometry` emits `angle_values` in degrees
(`compute/frame_geometry.py:369`) and `dihedral_values` in degrees on
`[-180, 180)` (`:401`). `AceCG.analysis.rdf._binning` used to build the
corresponding histogram edges in radians, so every real sample fell outside
the bin range and `np.histogram` silently dropped it — the angle and
dihedral distributions were all but empty. This predated the decluttering
sweep (the same two lines were present in `archive/src_snapshots/AceCG_Github_0810`
and `archive/src_snapshots/AceCG_dev_0805`); it was not drift.

Fixed 2026-08-10 per the human's choice under F-033: edges are now native to
degrees, matching `compute_frame_geometry`'s output, rather than converting
the values into radians at accumulation. `angle_degrees`/`dihedral_degrees`
now mean "convert the (native-degree) edges to radians for output" instead of
"convert the (native-radian) edges to degrees for output" — flipped, but the
default (`True`) still means "report in degrees" either way, so callers who
didn't pass the flag see no change in the meaning of their output, only in
whether the histogram is finally counting anything.

Pair and bond *distance* distributions were never affected — those are in
Angstrom on both sides.
"""

from __future__ import annotations

import numpy as np

from real_frames import (
    DOPC_CUTOFF,
    dopc_forcefield,
    dopc_frame_geometries,
    dopc_topology_arrays,
)

from AceCG.analysis.rdf import (
    _binning,
    accumulate_distribution_frame,
    finalize_distribution_state,
    init_distribution_state,
)


N_COEFFS = 12


def test_angle_edges_are_native_degrees_matching_frame_geometry():
    """The two sides of the histogram agree about units, on real frames."""
    forcefield = dopc_forcefield(N_COEFFS)
    geometry = dopc_frame_geometries(forcefield)[0]
    angle_keys = [key for key in forcefield if key.style == "angle"]
    assert angle_keys, "the DOPC fixture must carry angle interactions"

    values = np.asarray(geometry.angle_values[angle_keys[0]], dtype=np.float64)
    assert values.size > 100
    # Real CG lipid angles: tens to ~180. Unambiguously degrees.
    assert values.max() > 10.0

    state = init_distribution_state(
        dopc_topology_arrays(),
        forcefield,
        interaction_keys=angle_keys,
        cutoff=DOPC_CUTOFF,
    )
    edges = np.asarray(state["angle_edges"], dtype=np.float64)
    assert edges[0] == 0.0
    assert edges[-1] == 180.0


def test_the_real_angle_distribution_counts_every_sample():
    """F-033: every real angle sample now falls inside the bins."""
    forcefield = dopc_forcefield(N_COEFFS)
    geometry = dopc_frame_geometries(forcefield)[0]
    angle_keys = [key for key in forcefield if key.style == "angle"]

    state = init_distribution_state(
        dopc_topology_arrays(),
        forcefield,
        interaction_keys=angle_keys,
        cutoff=DOPC_CUTOFF,
    )
    accumulate_distribution_frame(state, geometry, frame_weight=1.0)

    total_samples = 0
    total_counted = 0.0
    for key in angle_keys:
        total_samples += int(np.asarray(geometry.angle_values[key]).size)
        total_counted += float(np.sum(state["angle_hist_by_key"][key]))

    assert total_samples > 100
    assert total_counted == float(total_samples)

    results = finalize_distribution_state(state)
    for key in angle_keys:
        result = results[key]
        assert result.x[0] > 0.0 and result.x[-1] < 180.0
        assert np.sum(result.counts) == float(
            np.asarray(geometry.angle_values[key]).size
        )
        assert np.any(result.values > 0.0)


def test_dihedral_edges_are_native_degrees_and_count_synthetic_samples():
    """Same fix, exercised directly since no shipped fixture carries dihedrals.

    `_binning` and `accumulate_distribution_frame`/`finalize_distribution_state`
    are the real production functions; only the input values are hand-built,
    matching how `compute_frame_geometry.dihedral_values` are documented to
    come out: degrees, normalized to [-180, 180).
    """
    edges, centers = _binning(
        variable="dihedral", nbins=180, x_max=None, periodic_dihedral=True
    )
    assert edges[0] == -180.0
    assert edges[-1] == 180.0
    assert centers[0] > -180.0 and centers[-1] < 180.0

    from AceCG.topology.types import InteractionKey

    key = InteractionKey.dihedral("A", "B", "C", "D")
    state = {
        "mode_by_key": {},
        "default_pair_mode": "rdf",
        "default_bonded_mode": "pdf",
        "angle_degrees": True,
        "dihedral_degrees": True,
        "dihedral_periodic": True,
        "pair_keys": [],
        "bond_keys": [],
        "angle_keys": [],
        "dihedral_keys": [key],
        "weight_sum": 0.0,
        "n_frames": 0,
        "dihedral_edges": edges,
        "dihedral_centers": centers,
        "dihedral_hist_by_key": {key: np.zeros(180, dtype=np.float64)},
        "dihedral_meta_by_key": {key: {"n_instances": 1, "degrees": True}},
    }

    class _Geometry:
        dihedral_values = {key: np.array([-170.0, -10.0, 0.0, 45.0, 179.9], dtype=np.float64)}

    accumulate_distribution_frame(state, _Geometry(), frame_weight=1.0)
    counted = float(np.sum(state["dihedral_hist_by_key"][key]))
    assert counted == 5.0

    result = finalize_distribution_state(state)[key]
    assert result.edges[0] == -180.0 and result.edges[-1] == 180.0
    assert np.sum(result.counts) == 5.0


def test_pair_distance_distributions_are_unaffected():
    """The distance histograms are in Angstrom on both sides and do count.

    Included so this module cannot be read as "RDF is broken": the F-033
    defect was confined to the two angular variables.
    """
    forcefield = dopc_forcefield(N_COEFFS)
    geometry = dopc_frame_geometries(forcefield)[0]
    pair_keys = [key for key in forcefield if key.style == "pair"]

    state = init_distribution_state(
        dopc_topology_arrays(),
        forcefield,
        interaction_keys=pair_keys,
        cutoff=DOPC_CUTOFF,
    )
    accumulate_distribution_frame(state, geometry, frame_weight=1.0)

    counted = sum(
        float(np.sum(state["pair_hist_by_key"][key])) for key in pair_keys
    )
    samples = sum(
        int(np.asarray(geometry.pair_distances[key]).size) for key in pair_keys
    )
    assert samples > 10_000
    assert counted == float(samples)
