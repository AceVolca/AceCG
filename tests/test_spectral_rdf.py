from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import AceCG.analysis as analysis_package
import AceCG.analysis.spectral_rdf as spectral_rdf_module
from AceCG.analysis.spectral_basis import CosineBasis, ShiftedLegendreBasis
from AceCG.analysis.spectral_rdf import (
    _auto_mode_cutoff,
    accumulate_spectral_rdf_frame,
    finalize_spectral_rdf_state,
    init_spectral_rdf_state,
    spectral_rdf_local_partials,
)
from AceCG.compute.reducers import (
    geometry_requirements_spectral_rdf,
    reduce_plan_spectral_rdf,
)
from AceCG.compute.mpi_engine import build_default_engine
from AceCG.topology.neighbor import count_eligible_pairs_by_type
from AceCG.topology.neighbor import compute_pairs_by_type
from AceCG.compute.frame_geometry import compute_frame_geometry
from AceCG.io.trajectory import FrameRecord
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.topology_array import TopologyArrays
from AceCG.topology.types import InteractionKey


def test_analysis_surface_is_state_machine_only() -> None:
    assert not hasattr(spectral_rdf_module, "spectral_pair_distributions")
    assert "spectral_pair_distributions" not in spectral_rdf_module.__all__
    assert not hasattr(analysis_package, "spectral_pair_distributions")
    assert "spectral_pair_distributions" not in analysis_package.__all__

    for name in (
        "SpectralRDFResult",
        "init_spectral_rdf_state",
        "accumulate_spectral_rdf_frame",
        "spectral_rdf_local_partials",
        "finalize_spectral_rdf_state",
    ):
        assert hasattr(spectral_rdf_module, name)

    source = Path(spectral_rdf_module.__file__).read_text(encoding="utf-8")
    for forbidden_import in (
        "from MDAnalysis import Universe",
        "from ..io.trajectory import iter_frames",
        "from ..compute.frame_geometry import compute_frame_geometry",
        "compute_pairs_by_type",
    ):
        assert forbidden_import not in source


def _topology(
    type_names=("A", "A"),
    *,
    resindices=None,
    resids=None,
    molnums=None,
    exclude_12=(),
    exclude_13=(),
    exclude_14=(),
) -> TopologyArrays:
    labels = np.asarray(type_names, dtype=object)
    unique = list(dict.fromkeys(str(label) for label in labels))
    name_to_code = {name: index + 1 for index, name in enumerate(unique)}
    codes = np.asarray([name_to_code[str(label)] for label in labels], dtype=np.int32)
    n_atoms = len(labels)
    if resindices is None:
        resindices = np.arange(n_atoms, dtype=np.int32)
    else:
        resindices = np.asarray(resindices, dtype=np.int32)
    if resids is None:
        resids = np.arange(int(np.max(resindices)) + 1, dtype=np.int32)
    else:
        resids = np.asarray(resids, dtype=np.int32)
    if molnums is None:
        molnums = np.arange(n_atoms, dtype=np.int32)
    else:
        molnums = np.asarray(molnums, dtype=np.int32)

    def pairs(values):
        arr = np.asarray(values, dtype=np.int32)
        return arr.reshape(-1, 2) if arr.size else np.empty((0, 2), dtype=np.int32)

    return TopologyArrays(
        n_atoms=n_atoms,
        names=labels.copy(),
        types=labels.copy(),
        atom_type_names=np.asarray(unique, dtype=object),
        atom_type_codes=codes,
        n_residues=len(resids),
        atom_resindex=resindices,
        masses=np.ones(n_atoms, dtype=np.float32),
        charges=np.zeros(n_atoms, dtype=np.float32),
        resids=resids,
        molnums=molnums,
        bonds=np.empty((0, 2), dtype=np.int32),
        angles=np.empty((0, 3), dtype=np.int32),
        dihedrals=np.empty((0, 4), dtype=np.int32),
        exclude_12=pairs(exclude_12),
        exclude_13=pairs(exclude_13),
        exclude_14=pairs(exclude_14),
        excluded_nb=np.empty(0, dtype=np.int32),
        excluded_nb_mode="none",
        excluded_nb_all=False,
        real_site_indices=np.arange(n_atoms, dtype=np.int32),
        virtual_site_mask=np.zeros(n_atoms, dtype=bool),
        virtual_site_indices=np.empty(0, dtype=np.int32),
        bond_key_index=np.empty(0, dtype=np.int32),
        angle_key_index=np.empty(0, dtype=np.int32),
        dihedral_key_index=np.empty(0, dtype=np.int32),
        keys_bondtypes=[],
        keys_angletypes=[],
        keys_dihedraltypes=[],
        atom_type_name_to_code=name_to_code,
        atom_type_code_to_name={value: key for key, value in name_to_code.items()},
        bond_type_id_to_key={},
        angle_type_id_to_key={},
        dihedral_type_id_to_key={},
        key_to_bonded_type_id={},
    )


def _frame(key, distance, *, frame_idx=0, box_length=4.0):
    return SimpleNamespace(
        frame_idx=frame_idx,
        pair_distances={key: np.asarray([distance], dtype=np.float64)},
        box=np.asarray(
            [box_length, box_length, box_length, 90.0, 90.0, 90.0],
            dtype=np.float64,
        ),
    )


def test_basis_orthonormality_and_analytic_derivative() -> None:
    nodes, weights = np.polynomial.legendre.leggauss(240)
    left, right = 0.7, 4.3
    r = left + 0.5 * (nodes + 1.0) * (right - left)
    integration_weights = weights * 0.5 * (right - left)
    probe = np.linspace(left + 0.1, right - 0.1, 101)
    step = 1.0e-6
    for basis in (CosineBasis(), ShiftedLegendreBasis()):
        values = basis.evaluate(r, 12, (left, right))
        gram = values.T @ (integration_weights[:, None] * values)
        np.testing.assert_allclose(gram, np.eye(12), rtol=0.0, atol=1.0e-12)
        derivative = basis.derivative(probe, 12, (left, right))
        finite_difference = (
            basis.evaluate(probe + step, 12, (left, right))
            - basis.evaluate(probe - step, 12, (left, right))
        ) / (2.0 * step)
        np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-7, atol=2.0e-8)


def test_known_finite_mode_function_recovers_coefficients() -> None:
    nodes, weights = np.polynomial.legendre.leggauss(260)
    domain = (0.4, 5.7)
    r = domain[0] + 0.5 * (nodes + 1.0) * (domain[1] - domain[0])
    integration_weights = weights * 0.5 * (domain[1] - domain[0])
    expected = np.asarray([1.2, -0.4, 0.0, 0.18, -0.07, 0.03])
    for basis in (CosineBasis(), ShiftedLegendreBasis()):
        phi = basis.evaluate(r, expected.size, domain)
        values = phi @ expected
        recovered = phi.T @ (integration_weights * values)
        np.testing.assert_allclose(recovered, expected, rtol=0.0, atol=2.0e-12)
        derivative = basis.derivative(r, expected.size, domain) @ expected
        assert np.all(np.isfinite(derivative))


def test_eligible_pair_counts_share_group_and_bonded_exclusions() -> None:
    topology = _topology(
        ("A", "A", "A", "B", "B", "B"),
        resindices=(0, 0, 1, 0, 1, 2),
        resids=(10, 11, 12),
        molnums=(0, 0, 1, 0, 1, 2),
        exclude_12=((0, 1), (0, 3)),
    )
    aa = InteractionKey.pair("A", "A")
    ab = InteractionKey.pair("A", "B")
    bb = InteractionKey.pair("B", "B")
    none = count_eligible_pairs_by_type(
        topology, [aa, ab, bb], exclude_option="none"
    )
    assert none == {aa: 2, ab: 8, bb: 3}
    resid = count_eligible_pairs_by_type(
        topology, [aa, ab, bb], exclude_option="resid"
    )
    assert resid == {aa: 2, ab: 6, bb: 3}
    molid = count_eligible_pairs_by_type(
        topology, [aa, ab, bb], exclude_option="molid"
    )
    assert molid == resid
    selected = np.asarray([0, 1, 3, 4], dtype=np.int32)
    selected_none = count_eligible_pairs_by_type(
        topology,
        [aa, ab, bb],
        sel_indices=selected,
        exclude_option="none",
    )
    assert selected_none == {aa: 0, ab: 3, bb: 1}
    selected_resid = count_eligible_pairs_by_type(
        topology,
        [aa, ab, bb],
        sel_indices=selected,
        exclude_option="resid",
    )
    assert selected_resid == {aa: 0, ab: 2, bb: 1}


def test_single_frame_coefficient_uses_exact_finite_n_prefactor() -> None:
    key = InteractionKey.pair("A", "A")
    topology = _topology(("A", "A"))
    state = init_spectral_rdf_state(
        topology,
        {key: object()},
        pair_keys=[key],
        r_max=1.9,
        max_modes=4,
        mode_cutoff=4,
        grid_size=31,
        exclude_option="none",
        block_size=1,
    )
    accumulate_spectral_rdf_frame(
        state, _frame(key, 1.0), frame_weight=1.0, frame_idx=0
    )
    result = finalize_spectral_rdf_state(state)[key]
    phi = CosineBasis().evaluate(np.asarray([1.0]), 5, (0.0, 1.9))[0]
    expected = 64.0 * phi / (4.0 * np.pi)
    np.testing.assert_allclose(result.coefficients, expected, rtol=1.0e-13, atol=1.0e-13)
    assert result.meta["n_eligible_pairs"] == 1
    np.testing.assert_allclose(
        result.meta["box_validation"]["minimum_half_cell_height"],
        2.0,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_reducer_partials_contain_only_sufficient_statistics() -> None:
    key = InteractionKey.pair("A", "A")
    topology = _topology(("A", "A"))
    step = {
        "step_mode": "spectral_rdf",
        "interaction_keys": [key.label()],
        "r_max": 1.9,
    }
    requirement = geometry_requirements_spectral_rdf(step, {key: object()}, topology)
    assert requirement.keys == (key,)
    assert requirement.pair_cutoff == 1.9
    state = init_spectral_rdf_state(
        topology,
        {key: object()},
        pair_keys=[key],
        r_max=1.9,
        max_modes=3,
        mode_cutoff=3,
        exclude_option="none",
    )
    accumulate_spectral_rdf_frame(state, _frame(key, 1.0), frame_idx=0)
    partials = spectral_rdf_local_partials(state)
    assert not ({"pair_distances", "frame_cache", "trajectory_cache"} & set(partials))
    assert partials["coefficient_sum"].shape == (1, 4)
    plan = reduce_plan_spectral_rdf(step)
    assert plan["stack"] == ()
    assert "dict_update" not in plan


def test_grid_size_does_not_change_coefficients_and_volume_is_per_frame() -> None:
    key = InteractionKey.pair("A", "A")
    topology = _topology(("A", "A"))
    states = [
        init_spectral_rdf_state(
            topology,
            {key: object()},
            pair_keys=[key],
            r_max=1.9,
            max_modes=5,
            mode_cutoff=5,
            grid_size=grid_size,
            exclude_option="none",
        )
        for grid_size in (19, 113)
    ]
    for state in states:
        accumulate_spectral_rdf_frame(
            state, _frame(key, 1.0, frame_idx=0, box_length=4.0), frame_idx=0
        )
        accumulate_spectral_rdf_frame(
            state, _frame(key, 1.5, frame_idx=1, box_length=6.0), frame_idx=1
        )
    first = finalize_spectral_rdf_state(states[0])[key]
    second = finalize_spectral_rdf_state(states[1])[key]
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    phi = CosineBasis().evaluate(np.asarray([1.0, 1.5]), 6, (0.0, 1.9))
    radial = phi / (4.0 * np.pi * np.asarray([1.0, 1.5])[:, None] ** 2)
    expected = 0.5 * (4.0**3 * radial[0] + 6.0**3 * radial[1])
    wrong_mean_volume = 0.5 * (4.0**3 + 6.0**3) * np.mean(radial, axis=0)
    np.testing.assert_allclose(first.coefficients, expected, rtol=1.0e-13, atol=1.0e-13)
    assert np.max(np.abs(first.coefficients - wrong_mean_volume)) > 1.0


def test_split_frame_reduction_matches_serial_coefficients_and_uncertainty() -> None:
    key = InteractionKey.pair("A", "A")
    topology = _topology(("A", "A"))

    def new_state():
        return init_spectral_rdf_state(
            topology,
            {key: object()},
            pair_keys=[key],
            r_max=1.9,
            max_modes=5,
            mode_cutoff=5,
            grid_size=47,
            pair_chunk_size=2,
            exclude_option="none",
            block_size=2,
        )

    frames = [
        (_frame(key, distance, frame_idx=index), weight)
        for index, (distance, weight) in enumerate(
            ((0.8, 1.0), (1.0, 2.0), (1.2, 3.0), (1.4, 4.0))
        )
    ]
    serial = new_state()
    split = [new_state(), new_state()]
    for index, (frame, weight) in enumerate(frames):
        accumulate_spectral_rdf_frame(
            serial, frame, frame_weight=weight, frame_idx=index
        )
        accumulate_spectral_rdf_frame(
            split[index % 2], frame, frame_weight=weight, frame_idx=index
        )

    partials = [spectral_rdf_local_partials(state) for state in split]
    reduced = {"spectral_config": partials[0]["spectral_config"]}
    for name in (
        "coefficient_sum",
        "coefficient_square_sum",
        "weight_sum",
        "weight_square_sum",
        "n_frames",
        "volume_sum",
    ):
        reduced[name] = partials[0][name] + partials[1][name]
    for name in (
        "negative_min_half_height",
        "max_half_height_ratio",
        "max_volume",
        "negative_min_volume",
        "max_frame_idx",
        "negative_min_frame_idx",
    ):
        reduced[name] = max(partials[0][name], partials[1][name])
    for name in ("block_coefficient_sum", "block_weight_sum"):
        reduced[name] = {}
        for partial in partials:
            for block_id, value in partial[name].items():
                if block_id not in reduced[name]:
                    reduced[name][block_id] = np.zeros_like(value)
                reduced[name][block_id] += value

    serial_result = finalize_spectral_rdf_state(serial)[key]
    reduced_result = finalize_spectral_rdf_state(reduced)[key]
    for name in (
        "coefficients",
        "coefficient_stderr",
        "block_coefficient_stderr",
        "values",
        "derivative",
    ):
        np.testing.assert_allclose(
            getattr(reduced_result, name),
            getattr(serial_result, name),
            rtol=2.0e-14,
            atol=2.0e-14,
        )
    assert reduced_result.mode_cutoff == serial_result.mode_cutoff
    assert reduced_result.n_frames == serial_result.n_frames == 4
    assert reduced_result.weight_sum == serial_result.weight_sum == 10.0


def test_triclinic_cutoff_and_zero_distance_fail_loudly() -> None:
    key = InteractionKey.pair("A", "A")
    topology = _topology(("A", "A"))
    state = init_spectral_rdf_state(
        topology,
        {key: object()},
        pair_keys=[key],
        r_max=1.9,
        max_modes=2,
        mode_cutoff=2,
        exclude_option="none",
    )
    triclinic = SimpleNamespace(
        pair_distances={key: np.asarray([1.0])},
        box=np.asarray([4.0, 4.0, 4.0, 90.0, 90.0, 60.0]),
    )
    with np.testing.assert_raises_regex(ValueError, "minimum-image"):
        accumulate_spectral_rdf_frame(state, triclinic, frame_idx=7)
    zero = SimpleNamespace(
        pair_distances={key: np.asarray([0.0])},
        box=np.asarray([4.0, 4.0, 4.0, 90.0, 90.0, 90.0]),
    )
    with np.testing.assert_raises_regex(ValueError, "numerical zero"):
        accumulate_spectral_rdf_frame(state, zero, frame_idx=8)


def test_seeded_ideal_gas_partial_rdf_coefficients_match_constant_one() -> None:
    rng = np.random.default_rng(3286)
    labels = tuple(["A"] * 20 + ["B"] * 20)
    topology = _topology(labels)
    aa = InteractionKey.pair("A", "A")
    ab = InteractionKey.pair("A", "B")
    keys = [aa, ab]
    box_length = 20.0
    box = np.asarray(
        [box_length, box_length, box_length, 90.0, 90.0, 90.0],
        dtype=np.float64,
    )
    state = init_spectral_rdf_state(
        topology,
        {key: object() for key in keys},
        pair_keys=keys,
        r_max=5.0,
        max_modes=5,
        mode_cutoff=5,
        grid_size=51,
        pair_chunk_size=256,
        exclude_option="none",
    )
    for frame_idx in range(120):
        positions = rng.uniform(0.0, box_length, size=(40, 3))
        pair_cache = compute_pairs_by_type(
            positions,
            box,
            keys,
            5.0,
            topology,
            exclude_option="none",
        )
        geometry = compute_frame_geometry(
            positions,
            box,
            topology,
            pair_cache=pair_cache,
        )
        accumulate_spectral_rdf_frame(state, geometry, frame_idx=frame_idx)
    results = finalize_spectral_rdf_state(state)
    expected_mode_zero = np.sqrt(5.0)
    for result in results.values():
        assert abs(result.coefficients[0] - expected_mode_zero) <= (
            5.0 * result.coefficient_stderr[0]
        )
        assert np.all(
            np.abs(result.coefficients[1:])
            <= 5.0 * result.coefficient_stderr[1:] + 1.0e-12
        )


def test_auto_mode_cutoff_and_explicit_failure_fallback() -> None:
    modes = np.arange(41, dtype=np.float64)
    coefficients = np.exp(-0.35 * modes)
    coefficients[15:] = 4.0e-3
    cutoff, diagnostics = _auto_mode_cutoff(coefficients)
    assert diagnostics["valid"]
    assert abs(cutoff - 14) <= 2

    fallback, failure = _auto_mode_cutoff(np.ones(41, dtype=np.float64))
    assert fallback == 40
    assert not failure["valid"]


def test_one_pass_two_bases_share_one_pair_search_and_geometry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    aa = InteractionKey.pair("A", "A")
    ab = InteractionKey.pair("A", "B")
    bb = InteractionKey.pair("B", "B")
    topology = _topology(("A", "A", "B", "B"))
    frames = [
        (
            0,
            np.asarray(
                [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [4.0, 1.0, 1.0], [6.0, 1.0, 1.0]],
                dtype=np.float64,
            ),
            np.asarray([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], dtype=np.float64),
            None,
        ),
        (
            1,
            np.asarray(
                [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [4.1, 1.0, 1.0], [6.2, 1.0, 1.0]],
                dtype=np.float64,
            ),
            np.asarray([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], dtype=np.float64),
            None,
        ),
    ]

    class DummyUniverse:
        trajectory = SimpleNamespace(__len__=lambda self: 2)

        def select_atoms(self, selection):
            assert selection == "all"
            return SimpleNamespace(indices=np.arange(4, dtype=np.int32))

    class DummyTrajectory:
        def __len__(self):
            return 2

        def __getitem__(self, frame_id):
            assert 0 <= int(frame_id) < 2
            return SimpleNamespace(has_forces=False, has_velocities=False)

    DummyUniverse.trajectory = DummyTrajectory()
    monkeypatch.setattr("MDAnalysis.Universe", lambda *args, **kwargs: DummyUniverse())
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: topology,
    )
    monkeypatch.setattr(
        "AceCG.io.trajectory.iter_frames",
        lambda universe, *, frame_ids, include_forces=False: iter(
            FrameRecord(
                frame_id=int(frame_id),
                positions=frames[int(frame_id)][1],
                box=frames[int(frame_id)][2],
                forces=None,
            )
            for frame_id in frame_ids
        ),
    )

    import AceCG.compute.mpi_engine as engine_module

    actual_pairs = engine_module.compute_pairs_by_type
    actual_geometry = engine_module.compute_frame_geometry
    calls = {"pairs": 0, "geometry": 0}

    def counted_pairs(*args, **kwargs):
        calls["pairs"] += 1
        return actual_pairs(*args, **kwargs)

    def counted_geometry(*args, **kwargs):
        calls["geometry"] += 1
        return actual_geometry(*args, **kwargs)

    monkeypatch.setattr(engine_module, "compute_pairs_by_type", counted_pairs)
    monkeypatch.setattr(engine_module, "compute_frame_geometry", counted_geometry)

    forcefield_path = tmp_path / "empty_forcefield.pkl"
    with forcefield_path.open("wb") as handle:
        pickle.dump(Forcefield(), handle)

    common_step = {
        "interaction_keys": [aa.label(), ab.label(), bb.label()],
        "r_min": 0.0,
        "r_max": 4.0,
        "max_modes": 4,
        "mode_cutoff": 4,
        "grid_size": 31,
        "pair_chunk_size": 8,
        "block_size": 1,
    }
    spec = {
        "work_dir": str(tmp_path),
        "topology": "topology.data",
        "trajectory": "traj.xtc",
        "trajectory_format": "XTC",
        "forcefield_path": str(forcefield_path),
        "frame_end": 2,
        "cutoff": 4.0,
        "exclude_bonded": "000",
        "exclude_option": "none",
        "steps": [
            {
                **common_step,
                "step_mode": "spectral_rdf",
                "basis": "cosine",
                "output_file": "cosine.pkl",
            },
            {
                **common_step,
                "step_mode": "spectral_rdf",
                "basis": "shifted_legendre",
                "output_file": "legendre.pkl",
            },
            {
                "step_mode": "rdf",
                "interaction_keys": [aa.label(), ab.label(), bb.label()],
                "cutoff": 4.0,
                "nbins_pair": 4,
                "output_file": "rdf.pkl",
            },
        ],
    }
    build_default_engine().run_post(spec)
    assert calls == {"pairs": 2, "geometry": 2}
    for filename in ("cosine.pkl", "legendre.pkl"):
        with (tmp_path / filename).open("rb") as handle:
            results = pickle.load(handle)
        assert set(results) == {aa, ab, bb}
        assert {result.n_frames for result in results.values()} == {2}
    with (tmp_path / "rdf.pkl").open("rb") as handle:
        rdf_results = pickle.load(handle)
    assert set(rdf_results) == {aa, ab, bb}
    assert {result.n_frames for result in rdf_results.values()} == {2}
