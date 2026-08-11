"""Real-topology oracle for the linear force-map constraint set (F-014).

The constraint set decides which atoms are rigidly coupled in the linear force
map, which sets the compression, which sets the equality block, which sets the
CG reference forces the whole force-matching campaign is fitted against. The
module's own integrity gate (``W C^T = I``) checks that the fitted operator
reproduces the coordinate map; it does **not** check the constraint set. A wrong
constraint set therefore passes every check the module performs, which is why
this file exists.

Everything runs on two real all-atom DPPC molecules with their real CHARMM36
bond graph, shipped in ``tests/test_data/dppc_aa/``. The physical fact the
tests lean on: in DPPC every hydrogen is terminal, so the number of bonds
touching a hydrogen equals the number of hydrogens exactly.

The linear force-mapping method is due to Krämer, Durumeric, Charron, Chen,
Clementi and Noé, J. Phys. Chem. Lett. 14(17), 3970-3979 (2023).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from real_frames import DPPC_AA_TOPOLOGY, dppc_aa_map_spec, dppc_aa_universe

from AceCG.compute.force_mapping import _components, _known_pairs, _layout


ATOMS_PER_MOLECULE = 130
BONDS_PER_MOLECULE = 129
HYDROGENS_PER_MOLECULE = 80


def _config(constraints: str) -> SimpleNamespace:
    """A force-mapping config with only the fields the constraint path reads."""
    return SimpleNamespace(
        path=None,
        aa=SimpleNamespace(topology_format=None),
        force_mapping=SimpleNamespace(
            method="constraint_aware_uniform",
            constraints=constraints,
            constraint_pairs_file=None,
            scope="per_template",
            fit_frame_ids=None,
            fit_every=1,
            fit_n_frames=0,
            max_covariance_bytes=10_000_000,
            constraint_threshold=1.0e-4,
            backend="native",
            l2_regularization=0.0,
            constraint_algorithm="LINCS",
        ),
    )


def _resolve(constraints: str):
    config = _config(constraints)
    spec = dppc_aa_map_spec()
    _, layout = _layout(spec, config.force_mapping)
    source, pairs = _known_pairs(config, spec, layout, DPPC_AA_TOPOLOGY)
    return layout, source, pairs


# ─────────────────────────────────────────────────────────────────────────────
# The hydrogen mask itself
# ─────────────────────────────────────────────────────────────────────────────


def test_the_three_hydrogen_heuristics_agree_on_a_real_all_atom_topology():
    """elements, names and masses must independently select the same atoms.

    ``_known_pairs`` ORs three heuristics together. That is only safe while they
    agree; if one silently stopped contributing, the OR would still look
    plausible. Here all three are available and all three must give the same
    160 atoms, so the union is not hiding a disagreement.
    """
    atoms = dppc_aa_universe().atoms
    expected = 2 * HYDROGENS_PER_MOLECULE

    by_element = np.char.upper(np.asarray(atoms.elements, dtype=str)) == "H"
    by_name = np.char.startswith(
        np.char.upper(np.asarray(atoms.names, dtype=str)), "H"
    )
    masses = np.asarray(atoms.masses, dtype=np.float64)
    by_mass = (masses >= 0.5) & (masses <= 2.0)

    assert int(by_element.sum()) == expected
    assert np.array_equal(by_element, by_name)
    assert np.array_equal(by_element, by_mass)


def test_h_bonds_selects_exactly_the_bonds_that_touch_a_hydrogen():
    """Every DPPC hydrogen is terminal, so #(h-bonds) == #(hydrogens)."""
    layout, source, pairs = _resolve("h-bonds")
    atoms = dppc_aa_universe().atoms
    hydrogen = np.char.upper(np.asarray(atoms.elements, dtype=str)) == "H"

    assert source == "h-bonds"
    assert len(pairs) == len(layout) == 1
    template = pairs[0]
    assert template.shape == (HYDROGENS_PER_MOLECULE, 2)

    # Translate the template-local pairs back to absolute atoms and confirm the
    # physical property: exactly one endpoint of each constrained bond is an H.
    absolute = dppc_aa_map_spec().atom_indices[layout[0]["atoms"][0]]
    endpoints = hydrogen[absolute[template]]
    assert np.array_equal(endpoints.sum(axis=1), np.ones(template.shape[0]))


def test_all_bonds_selects_every_bond_of_the_template():
    """The `all-bonds` mode must not filter on the hydrogen mask at all."""
    _, source, pairs = _resolve("all-bonds")
    assert source == "all-bonds"
    assert pairs[0].shape == (BONDS_PER_MOLECULE, 2)


def test_none_and_auto_defer_the_constraint_set_without_reading_the_topology():
    """Both modes must yield an empty known-pair set, not a guessed one."""
    for mode in ("none", "auto"):
        _, source, pairs = _resolve(mode)
        assert source == mode
        assert pairs[0].shape == (0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# What the constraint set does downstream
# ─────────────────────────────────────────────────────────────────────────────


def test_h_bond_constraints_collapse_each_hydrogen_onto_its_heavy_atom():
    """Constraining every X-H bond leaves exactly the heavy atoms as components.

    This is the number that propagates into the compression, the equality block
    and ultimately the CG reference forces, so it is the one worth pinning.
    """
    layout, _, pairs = _resolve("h-bonds")
    width = layout[0]["coordinate"].shape[1]
    assert width == ATOMS_PER_MOLECULE

    components = _components(width, pairs[0])
    assert components.shape == (
        ATOMS_PER_MOLECULE,
        ATOMS_PER_MOLECULE - HYDROGENS_PER_MOLECULE,
    )
    # Each atom belongs to exactly one component, and every component is
    # non-empty: a rigid group cannot lose or gain an atom.
    np.testing.assert_array_equal(components.sum(axis=1), np.ones(ATOMS_PER_MOLECULE))
    assert np.all(components.sum(axis=0) >= 1.0)

    unconstrained = _components(width, np.empty((0, 2), dtype=np.int64))
    assert unconstrained.shape == (ATOMS_PER_MOLECULE, ATOMS_PER_MOLECULE)

    every_bond = _components(width, _resolve("all-bonds")[2][0])
    assert every_bond.shape == (ATOMS_PER_MOLECULE, 1)


def test_a_broken_hydrogen_heuristic_raises_instead_of_shrinking_the_set():
    """F-014: a heuristic that fails must crash, not quietly drop out.

    Before this was fixed, each heuristic ran under ``except Exception: pass``.
    A topology where, say, ``names`` raised would still produce a mask from the
    other two, a *different* constraint set, a different force map and
    different CG reference forces — with no error and no log line. Absence of
    the attribute is still tolerated (see the ``SimpleNamespace`` case in
    ``test_force_mapping.py``); a *failure* is not.
    """
    config = _config("h-bonds")
    spec = dppc_aa_map_spec()
    _, layout = _layout(spec, config.force_mapping)

    class _ExplodingNames:
        """A topology whose element data is fine but whose names are broken."""

        n_atoms = ATOMS_PER_MOLECULE * 2
        elements = np.array(["H"] * n_atoms, dtype=object)
        masses = np.full(n_atoms, 1.008)

        @property
        def names(self):
            raise RuntimeError("topology name array is corrupt")

    universe = SimpleNamespace(
        bonds=SimpleNamespace(indices=np.array([[0, 1], [1, 2]], dtype=np.int64)),
        atoms=_ExplodingNames(),
    )
    import AceCG.io.trajectory as trajectory_module

    original = trajectory_module.open_universe
    trajectory_module.open_universe = lambda *args, **kwargs: universe
    try:
        with pytest.raises(RuntimeError, match="topology name array is corrupt"):
            _known_pairs(config, spec, layout, DPPC_AA_TOPOLOGY)
    finally:
        trajectory_module.open_universe = original


def test_h_bonds_raises_when_no_hydrogen_can_be_identified():
    """Total heuristic failure must stop the run, not silently constrain nothing."""
    config = _config("h-bonds")
    spec = dppc_aa_map_spec()
    _, layout = _layout(spec, config.force_mapping)

    heavy_only = SimpleNamespace(
        bonds=SimpleNamespace(indices=np.array([[0, 1], [1, 2]], dtype=np.int64)),
        atoms=SimpleNamespace(n_atoms=ATOMS_PER_MOLECULE * 2),
    )
    import AceCG.io.trajectory as trajectory_module

    original = trajectory_module.open_universe
    trajectory_module.open_universe = lambda *args, **kwargs: heavy_only
    try:
        with pytest.raises(ValueError, match="could not identify hydrogen atoms"):
            _known_pairs(config, spec, layout, DPPC_AA_TOPOLOGY)
    finally:
        trajectory_module.open_universe = original
