"""Shared loader for the committed real-data fixtures under ``tests/test_data``.

Every function here goes through the production path — ``MDAnalysis.Universe``
→ ``collect_topology_arrays`` → ``compute_pairs_by_type`` →
``compute_frame_geometry`` → ``compute.energy`` — so a test built on it fails
when the physics changes, not when an internal seam moves.

No test may read ``data/`` or ``experiments/``; those are repository-local and
are never published. See ``tests/test_data/dopc_cg6/README.md`` for provenance
and ``scripts/extract_test_fixtures.py`` for how the fixture was cut.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

TEST_DATA = Path(__file__).resolve().parent / "test_data"
DOPC_CG6 = TEST_DATA / "dopc_cg6"
DPPC_AA = TEST_DATA / "dppc_aa"
DPPC_MARTINI12 = TEST_DATA / "dppc_martini12"

# The DOPC CG6 model as the production configs declare it
# (``configs/templates/rem_prod.acg``).
DOPC_TYPE_ALIASES = {1: "HG", 2: "MG", 3: "T1", 4: "T2"}
DOPC_CUTOFF = 25.0
DOPC_EXCLUDE_BONDED = "100"
DOPC_EXCLUDE_OPTION = "resid"

# Boltzmann constant in LAMMPS ``real`` units, matching
# ``AceCG.workflows.sampling._BOLTZMANN_KCAL``.
BOLTZMANN_KCAL = 0.001987204


def beta_at(temperature_kelvin: float) -> float:
    """β = 1/(kB·T) in mol/kcal, the convention the REM workflows use."""
    return 1.0 / (BOLTZMANN_KCAL * float(temperature_kelvin))


@functools.lru_cache(maxsize=4)
def dopc_universe():
    """The 5-frame DOPC CG6 patch as an MDAnalysis Universe."""
    import MDAnalysis as mda

    return mda.Universe(
        str(DOPC_CG6 / "cg6_patch.data"),
        str(DOPC_CG6 / "cg6_patch.lammpstrj"),
        atom_style="id resid type charge x y z",
        format="LAMMPSDUMP",
    )


@functools.lru_cache(maxsize=4)
def dopc_topology_arrays():
    """Topology arrays for the DOPC patch, with the production exclusion rules."""
    from AceCG.topology.topology_array import collect_topology_arrays

    return collect_topology_arrays(
        dopc_universe(),
        exclude_bonded=DOPC_EXCLUDE_BONDED,
        exclude_option=DOPC_EXCLUDE_OPTION,
        atom_type_name_aliases=DOPC_TYPE_ALIASES,
    )


@functools.lru_cache(maxsize=4)
def dopc_forcefield(n_coeffs: int = 12):
    """The real REM-init tabulated forcefield, fitted onto a B-spline basis.

    ``n_coeffs`` is deliberately far below the production 128 so the fixture
    stays fast and the pinned vectors stay readable; the fit path, the tables
    and the gauge conventions are the production ones.
    """
    from AceCG.io.forcefield import ReadLmpFF

    return ReadLmpFF(
        str(DOPC_CG6 / "ff" / "system.settings"),
        pair_style="hybrid",
        pair_typ_sel=["table"],
        cutoff=DOPC_CUTOFF,
        table_fit="bspline",
        table_fit_overrides={"n_coeffs": int(n_coeffs)},
        topology_arrays=dopc_topology_arrays(),
    )


def dopc_frame_geometries(forcefield):
    """Per-frame ``FrameGeometry`` for every fixture frame, in file order."""
    from AceCG.compute.frame_geometry import compute_frame_geometry
    from AceCG.topology.neighbor import compute_pairs_by_type

    topology = dopc_topology_arrays()
    pair_keys = [key for key in forcefield if key.style == "pair"]
    geometries = []
    for timestep in dopc_universe().trajectory:
        positions = np.asarray(timestep.positions, dtype=np.float32)
        box = np.asarray(timestep.dimensions, dtype=np.float32)
        pair_cache = compute_pairs_by_type(
            positions,
            box,
            pair_keys,
            DOPC_CUTOFF,
            topology,
            exclude_option=DOPC_EXCLUDE_OPTION,
        )
        geometries.append(
            compute_frame_geometry(positions, box, topology, pair_cache=pair_cache)
        )
    return geometries


@functools.lru_cache(maxsize=4)
def dopc_energy_statistics(n_coeffs: int = 12):
    """Per-frame energy, ``dU/dλ`` and ``d²U/dλ²`` over the fixture frames.

    Returns ``(forcefield, energies, energy_grad, energy_hessian)`` with
    ``energy_grad`` shaped ``(n_frames, n_params)`` and ``energy_hessian``
    shaped ``(n_frames, n_params, n_params)``.
    """
    from AceCG.compute.energy import energy

    forcefield = dopc_forcefield(n_coeffs)
    energies = []
    grads = []
    hessians = []
    for geometry in dopc_frame_geometries(forcefield):
        result = energy(
            geometry,
            forcefield,
            return_value=True,
            return_grad=True,
            return_hessian=True,
        )
        energies.append(float(result["energy"]))
        grads.append(np.asarray(result["energy_grad"], dtype=float))
        hessians.append(np.asarray(result["energy_hessian"], dtype=float))
    return (
        forcefield,
        np.asarray(energies, dtype=float),
        np.asarray(grads, dtype=float),
        np.asarray(hessians, dtype=float),
    )


# ---------------------------------------------------------------------------
# DPPC all-atom fixture (linear force mapping)
# ---------------------------------------------------------------------------

DPPC_AA_TOPOLOGY = DPPC_AA / "dppc_2mol.pdb"
DPPC_AA_MAPPING = DPPC_AA / "martini12_2mol_map.yaml"


@functools.lru_cache(maxsize=2)
def dppc_aa_map_spec():
    """The production Martini-12 mapping narrowed to the fixture's molecules."""
    import yaml

    from AceCG.topology.cgmap import CGMapSpec

    return CGMapSpec.from_mapping(yaml.safe_load(DPPC_AA_MAPPING.read_text()))


def dppc_aa_universe():
    """The 2-molecule all-atom DPPC fixture, with bonds, elements and masses."""
    import MDAnalysis as mda

    return mda.Universe(str(DPPC_AA_TOPOLOGY))
