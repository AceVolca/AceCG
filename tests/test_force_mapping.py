"""Characterizations for the two-stage compressed force-map path."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from AceCG.compute.cgmap import CGMapper
import AceCG.compute.cgmap as cgmap_module
import AceCG.compute.force_mapping as force_mapping_module
from AceCG.compute.force_mapping import accumulate_force_map_statistics, fit_force_map
from AceCG.io.force_operator import read_force_operator, write_force_operator
from AceCG.topology.cgmap import CGMapSpec


def _spec() -> CGMapSpec:
    return CGMapSpec.from_mapping({
        "site-types": {
            "A": {"index": [0, 1], "x-weight": [1., 1.], "f-weight": [1., 1.]},
            "B": {"index": [0, 1], "x-weight": [1., 1.], "f-weight": [1., 1.]},
        },
        "system": [{"anchor": 0, "repeat": 2, "offset": 4, "sites": [["A", 0], ["B", 2]]}],
    })


def _config(method: str, constraints: str = "none") -> SimpleNamespace:
    return SimpleNamespace(
        path=None,
        aa=SimpleNamespace(topology_format=None),
        force_mapping=SimpleNamespace(
            method=method, constraints=constraints, constraint_pairs_file=None,
            scope="per_template", fit_frame_ids=None, fit_every=1, fit_n_frames=0,
            max_covariance_bytes=10_000_000, constraint_threshold=1.e-4,
            backend="native", l2_regularization=0., constraint_algorithm="LINCS",
        ),
    )


class _Reader:
    def __init__(self, frames):
        self.plan = _Plan(tuple(range(len(frames))))
        self.frames = frames
        self.calls = []

    def iter_local(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.frames)


@dataclass(frozen=True)
class _Plan:
    frame_ids: tuple[int, ...]


def _frames(spec: CGMapSpec):
    rng = np.random.default_rng(4021)
    return [
        {"positions": rng.normal(size=(spec.n_required_atoms, 3)), "forces": rng.normal(size=(spec.n_required_atoms, 3))}
        for _ in range(12)
    ]


def test_optimal_native_attaches_consistent_shared_operator():
    spec = _spec(); reader = _Reader(_frames(spec)); config = _config("optimal_linear")
    config.force_mapping.l2_regularization = 0.3
    stats = accumulate_force_map_statistics(config=config, spec=spec, reader=reader, plan=reader.plan, topology_path=None, comm=None)
    attached, report = fit_force_map(config=config, spec=spec, statistics=stats, comm=None)
    assert attached.has_force_operator
    assert report["fit_frames"]["count"] == 12
    coordinate = np.asarray([[.5, .5, 0., 0.], [0., 0., .5, .5]])
    samples = np.asarray([frame["forces"] for frame in reader.frames]).reshape(12, 2, 4, 3)
    rows = np.swapaxes(samples, 2, 3).reshape(-1, 4)
    quadratic = rows.T @ rows
    regularized = quadratic + 0.3 * np.eye(quadratic.shape[0])
    expected = np.linalg.solve(coordinate @ np.linalg.inv(regularized) @ coordinate.T, coordinate @ np.linalg.inv(regularized))
    forces = reader.frames[0]["forces"]
    mapped = CGMapper(attached, unwrap="none").map_frame(reader.frames[0]["positions"], forces=forces, compact=True)
    assert mapped.forces == pytest.approx(np.vstack([expected @ forces[:4], expected @ forces[4:]]), abs=1.e-9)


def test_uniform_known_constraints_has_no_statistics_iteration():
    spec = _spec(); reader = _Reader(_frames(spec)); config = _config("constraint_aware_uniform")
    stats = accumulate_force_map_statistics(config=config, spec=spec, reader=reader, plan=reader.plan, topology_path=None, comm=None)
    attached, report = fit_force_map(config=config, spec=spec, statistics=stats, comm=None)
    assert reader.calls == []
    assert report["backend"] == "analytic_uniform"
    forces = np.arange(spec.n_required_atoms * 3, dtype=float).reshape(spec.n_required_atoms, 3)
    mapped = CGMapper(attached, unwrap="none").map_frame(np.zeros_like(forces), forces=forces, compact=True)
    assert mapped.forces == pytest.approx(np.vstack([forces[:2].sum(axis=0), forces[2:4].sum(axis=0), forces[4:6].sum(axis=0), forces[6:].sum(axis=0)]))


def test_optimal_known_constraints_compress_before_accumulation(tmp_path):
    spec = _spec()
    reader = _Reader(_frames(spec))
    config = _config("optimal_linear")
    config.path = tmp_path / "trajmap.acg"
    config.force_mapping.constraint_pairs_file = "pairs.yaml"
    (tmp_path / "pairs.yaml").write_text("default: [[0, 1]]\n")

    statistics = accumulate_force_map_statistics(
        config=config,
        spec=spec,
        reader=reader,
        plan=reader.plan,
        topology_path=None,
        comm=None,
    )
    assert statistics["statistics"][0]["quadratic"].shape == (3, 3)
    assert not ({"count", "sum", "sum_square"} & statistics["statistics"][0].keys())

    attached, report = fit_force_map(
        config=config,
        spec=spec,
        statistics=statistics,
        comm=None,
    )
    matrix = attached._force_operator[0][0]
    coordinate = np.asarray([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    assert matrix @ coordinate.T == pytest.approx(np.eye(2), abs=1.0e-9)
    assert report["diagnostics"][0]["n_constraints"] == 1


def test_explicit_duplicate_fit_ids_are_rejected_before_truncation():
    spec = _spec()
    reader = _Reader(_frames(spec))
    config = _config("optimal_linear")
    config.force_mapping.fit_frame_ids = (0, 1, 0)
    config.force_mapping.fit_n_frames = 2

    with pytest.raises(ValueError, match="non-empty and unique"):
        accumulate_force_map_statistics(
            config=config,
            spec=spec,
            reader=reader,
            plan=reader.plan,
            topology_path=None,
            comm=None,
        )
    assert reader.calls == []


def test_pair_moment_allocation_failure_closes_before_reader_or_gather(monkeypatch):
    spec = _spec()
    reader = _Reader(_frames(spec))
    config = _config("constraint_aware_uniform", "auto")
    real_zeros = force_mapping_module.np.zeros
    gathered_errors = []

    def fail_pair_moments(shape, *args, **kwargs):
        if shape == 6:
            raise MemoryError("pair moments did not allocate")
        return real_zeros(shape, *args, **kwargs)

    class Comm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 2

        def bcast(self, value, root=0):
            return value

        def allgather(self, error):
            gathered_errors.append(error)
            return [error, None]

        def gather(self, value, root=0):
            raise AssertionError("gather must not start after an allocation failure")

    monkeypatch.setattr(force_mapping_module.np, "zeros", fail_pair_moments)
    with pytest.raises(MemoryError, match="pair moments did not allocate"):
        accumulate_force_map_statistics(
            config=config,
            spec=spec,
            reader=reader,
            plan=reader.plan,
            topology_path=None,
            comm=Comm(),
        )
    assert len(gathered_errors) == 1
    assert isinstance(gathered_errors[0], MemoryError)
    assert reader.calls == []


def test_auto_constraints_use_one_combined_statistics_iteration():
    spec = _spec(); frames = _frames(spec)
    reader = _Reader(frames); config = _config("optimal_linear", "auto")
    stats = accumulate_force_map_statistics(config=config, spec=spec, reader=reader, plan=reader.plan, topology_path=None, comm=None)
    assert len(reader.calls) == 1 and reader.calls[0]["include_forces"]
    attached, _ = fit_force_map(config=config, spec=spec, statistics=stats, comm=None)
    assert attached.has_force_operator


def test_uniform_auto_uses_positions_once_with_minimum_image_constraints():
    spec = CGMapSpec.from_mapping({
        "site-types": {"A": {"index": [0], "x-weight": [1.]}, "B": {"index": [0], "x-weight": [1.]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 2, "sites": [["A", 0], ["B", 1]]}],
    })
    frames = [
        {"positions": np.array([[9.9, 0., 0.], [.1, 0., 0.]]), "box": np.array([10., 10., 10., 90., 90., 90.])},
        {"positions": np.array([[9.8, 0., 0.], [0., 0., 0.]]), "box": np.array([10., 10., 10., 90., 90., 90.])},
    ]
    reader = _Reader(frames); config = _config("constraint_aware_uniform", "auto")
    stats = accumulate_force_map_statistics(config=config, spec=spec, reader=reader, plan=reader.plan, topology_path=None, comm=None)
    attached, _ = fit_force_map(config=config, spec=spec, statistics=stats, comm=None)
    assert len(reader.calls) == 1 and not reader.calls[0]["include_forces"]
    mapped = CGMapper(attached, unwrap="none").map_frame(frames[0]["positions"], forces=np.array([[1., 2., 3.], [4., 5., 6.]]), compact=True)
    assert mapped.forces == pytest.approx(np.array([[5., 7., 9.], [5., 7., 9.]]))


def test_uniform_auto_accepts_a_single_atom_template_as_an_empty_constraint_set():
    spec = CGMapSpec.from_mapping({
        "site-types": {"A": {"index": [0], "x-weight": [1.], "f-weight": [1.]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 1, "sites": [["A", 0]]}],
    })
    frames = [
        {"positions": np.array([[0., 0., 0.]])},
        {"positions": np.array([[1., 0., 0.]])},
    ]
    reader = _Reader(frames)
    config = _config("constraint_aware_uniform", "auto")
    statistics = accumulate_force_map_statistics(
        config=config, spec=spec, reader=reader, plan=reader.plan,
        topology_path=None, comm=None,
    )
    assert statistics["pairs"][0].shape == (0, 2)
    attached, _ = fit_force_map(config=config, spec=spec, statistics=statistics, comm=None)
    mapped = CGMapper(attached, unwrap="none").map_frame(
        frames[0]["positions"], forces=np.array([[2., 3., 4.]]), compact=True,
    )
    assert mapped.forces == pytest.approx(np.array([[2., 3., 4.]]))


def test_all_bonds_accepts_a_topology_without_hydrogen_attributes(monkeypatch, tmp_path):
    spec = _spec(); reader = _Reader(_frames(spec)); config = _config("constraint_aware_uniform", "all-bonds")
    universe = SimpleNamespace(bonds=SimpleNamespace(indices=np.array([[0, 1], [2, 3]])), atoms=SimpleNamespace(n_atoms=8))
    monkeypatch.setattr("AceCG.io.trajectory.open_universe", lambda *args, **kwargs: universe)
    stats = accumulate_force_map_statistics(config=config, spec=spec, reader=reader, plan=reader.plan, topology_path=tmp_path / "topology", comm=None)
    attached, _ = fit_force_map(config=config, spec=spec, statistics=stats, comm=None)
    assert attached.has_force_operator and reader.calls == []


@pytest.mark.parametrize(
    "pair_file",
    ["default: [[0, 1]]\n", "pairs: [[0, 1]]\n"],
)
def test_file_constraints_are_projected_without_a_statistics_pass(tmp_path, pair_file):
    spec = _spec(); reader = _Reader(_frames(spec)); config = _config("constraint_aware_uniform")
    (tmp_path / "pairs.yaml").write_text(pair_file)
    config.path = tmp_path / "trajmap.acg"
    config.force_mapping.constraint_pairs_file = "pairs.yaml"
    statistics = accumulate_force_map_statistics(config=config, spec=spec, reader=reader, plan=reader.plan, topology_path=None, comm=None)
    attached, _ = fit_force_map(config=config, spec=spec, statistics=statistics, comm=None)
    assert attached.has_force_operator and reader.calls == []


def test_template_layout_keeps_same_shape_but_distinct_maps_separate_and_scopes_guard_memory():
    spec = CGMapSpec.from_mapping({
        "site-types": {
            "A": {"index": [0, 1], "x-weight": [1., 1.], "f-weight": [1., 1.]},
            "B": {"index": [0, 1], "x-weight": [2., 1.], "f-weight": [1., 1.]},
        },
        "system": [
            {"anchor": 0, "repeat": 2, "offset": 2, "sites": [["A", 0]]},
            {"anchor": 4, "repeat": 1, "offset": 2, "sites": [["B", 0]]},
        ],
    })
    reader = _Reader(_frames(spec))
    config = _config("constraint_aware_uniform")
    config.force_mapping.scope = "auto"
    statistics = accumulate_force_map_statistics(
        config=config, spec=spec, reader=reader, plan=reader.plan,
        topology_path=None, comm=None,
    )
    assert statistics["scope"] == "per_template"
    assert len(statistics["layout"]) == 2

    config.force_mapping.scope = "global"
    config.force_mapping.max_covariance_bytes = 1
    with pytest.raises(MemoryError, match="scope=global"):
        accumulate_force_map_statistics(
            config=config, spec=spec, reader=reader, plan=reader.plan,
            topology_path=None, comm=None,
        )


def test_explicit_global_scope_fits_one_whole_system_operator():
    spec = _spec()
    reader = _Reader(_frames(spec))
    config = _config("optimal_linear")
    config.force_mapping.scope = "global"
    statistics = accumulate_force_map_statistics(
        config=config,
        spec=spec,
        reader=reader,
        plan=reader.plan,
        topology_path=None,
        comm=None,
    )
    attached, report = fit_force_map(
        config=config,
        spec=spec,
        statistics=statistics,
        comm=None,
    )

    matrices, _, _, coordinate_maps, _, _, _ = attached._force_operator
    assert report["scope"] == "global"
    assert len(matrices) == 1
    assert matrices[0] @ coordinate_maps[0].T == pytest.approx(
        np.eye(spec.n_sites), abs=1.0e-9,
    )
    forces = reader.frames[0]["forces"]
    mapped = CGMapper(attached, unwrap="none").map_frame(
        reader.frames[0]["positions"], forces=forces, compact=True,
    )
    assert mapped.forces == pytest.approx(matrices[0] @ forces)


def test_fit_every_and_fit_n_frames_reuse_and_restore_the_scanned_plan():
    spec = _spec()
    reader = _Reader(_frames(spec))
    original_plan = reader.plan
    config = _config("optimal_linear")
    config.force_mapping.fit_every = 2
    config.force_mapping.fit_n_frames = 3

    statistics = accumulate_force_map_statistics(
        config=config,
        spec=spec,
        reader=reader,
        plan=reader.plan,
        topology_path=None,
        comm=None,
    )
    assert statistics["fit_ids"] == (0, 2, 4)
    assert reader.plan is original_plan
    assert len(reader.calls) == 1


def test_fitted_operator_batches_each_repeated_template_once(monkeypatch):
    spec = _spec(); reader = _Reader(_frames(spec)); config = _config("optimal_linear")
    statistics = accumulate_force_map_statistics(
        config=config, spec=spec, reader=reader, plan=reader.plan,
        topology_path=None, comm=None,
    )
    attached, _ = fit_force_map(config=config, spec=spec, statistics=statistics, comm=None)
    compact = reader.frames[0]["forces"]
    full = np.zeros((spec.n_required_atoms + 3, 3))
    full[spec.atom_indices] = compact
    calls = []
    real_einsum = cgmap_module.np.einsum

    def record_einsum(subscripts, *operands, **kwargs):
        if subscripts == "cf,ifd->icd":
            calls.append(tuple(operand.shape for operand in operands))
        return real_einsum(subscripts, *operands, **kwargs)

    monkeypatch.setattr(cgmap_module.np, "einsum", record_einsum)
    mapper = CGMapper(attached, unwrap="none")
    compact_mapped = mapper.map_frame(compact, forces=compact, compact=True).forces
    full_mapped = mapper.map_frame(full, forces=full, compact=False).forces
    assert full_mapped == pytest.approx(compact_mapped)
    assert calls == [((2, 4), (2, 4, 3)), ((2, 4), (2, 4, 3))]
    assert attached._force_operator[0][0].shape == (2, 4)


def test_compare_backend_reports_matrix_and_objective_agreement():
    pytest.importorskip("qpsolvers")
    pytest.importorskip("osqp")
    spec = _spec(); reader = _Reader(_frames(spec)); config = _config("optimal_linear")
    config.force_mapping.backend = "compare"
    statistics = accumulate_force_map_statistics(config=config, spec=spec, reader=reader, plan=reader.plan, topology_path=None, comm=None)
    _, report = fit_force_map(config=config, spec=spec, statistics=statistics, comm=None)
    comparison = report["diagnostics"][0]["backend_comparison"]
    assert comparison["matrix_max_abs_delta"] < 2.e-6
    assert comparison["objective_relative_delta"] <= 1.e-7


def test_auto_backend_reports_osqp_fallback_when_native_consistency_fails(monkeypatch):
    pytest.importorskip("qpsolvers")
    pytest.importorskip("osqp")
    original_solve = force_mapping_module.np.linalg.solve
    solve_calls = 0

    def invalid_native_solution(matrix, rhs):
        nonlocal solve_calls
        solve_calls += 1
        if solve_calls == 1:
            return np.zeros_like(rhs)
        return original_solve(matrix, rhs)

    monkeypatch.setattr(force_mapping_module.np.linalg, "solve", invalid_native_solution)
    coefficients, diagnostics = force_mapping_module._solve(
        np.eye(2), np.eye(2), np.eye(2), 0.0, "auto",
    )
    assert diagnostics["backend"] == "osqp"
    assert diagnostics["fallback_from"] == "native"
    assert coefficients == pytest.approx(np.eye(2), abs=1.e-7)


def test_force_operator_artifact_roundtrip_applies_and_rejects_incompatible_csr_layout(tmp_path):
    spec = _spec(); reader = _Reader(_frames(spec)); config = _config("optimal_linear")
    statistics = accumulate_force_map_statistics(config=config, spec=spec, reader=reader, plan=reader.plan, topology_path=None, comm=None)
    shared = SimpleNamespace(value=None)

    class BroadcastComm:
        def __init__(self, rank):
            self.rank = rank

        def Get_rank(self):
            return self.rank

        def Get_size(self):
            return 2

        def bcast(self, value, root=0):
            if self.rank == root:
                shared.value = value
            return shared.value

    attached, report = fit_force_map(
        config=config, spec=spec, statistics=statistics, comm=BroadcastComm(0),
    )
    peer_attached, peer_report = fit_force_map(
        config=config, spec=spec, statistics=None, comm=BroadcastComm(1),
    )
    assert peer_report == report
    path = write_force_operator(tmp_path / "force_map.npz", attached, report)
    reloaded, diagnostics = read_force_operator(path, spec)
    forces = reader.frames[0]["forces"]
    assert CGMapper(reloaded, unwrap="none").map_frame(
        reader.frames[0]["positions"], forces=forces, compact=True,
    ).forces == pytest.approx(
        CGMapper(attached, unwrap="none").map_frame(
            reader.frames[0]["positions"], forces=forces, compact=True,
        ).forces,
    )
    assert diagnostics == report
    assert CGMapper(peer_attached, unwrap="none").map_frame(
        reader.frames[0]["positions"], forces=forces, compact=True,
    ).forces == pytest.approx(
        CGMapper(attached, unwrap="none").map_frame(
            reader.frames[0]["positions"], forces=forces, compact=True,
        ).forces,
    )
    broadcast_copy = pickle.loads(pickle.dumps(reloaded))
    assert CGMapper(broadcast_copy, unwrap="none").map_frame(
        reader.frames[0]["positions"], forces=forces, compact=True,
    ).forces == pytest.approx(
        CGMapper(reloaded, unwrap="none").map_frame(
            reader.frames[0]["positions"], forces=forces, compact=True,
        ).forces,
    )
    assert not reloaded._force_operator[0][0].flags.writeable
    with pytest.raises(ValueError):
        reloaded._force_operator[0][0][0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        reloaded._force_operator = None
    incompatible = CGMapSpec.from_mapping({
        "site-types": {"A": {"index": [0, 1], "x-weight": [1., 1.]}, "B": {"index": [0, 1], "x-weight": [2., 1.]}},
        "system": [{"anchor": 0, "repeat": 2, "offset": 4, "sites": [["A", 0], ["B", 2]]}],
    })
    with pytest.raises(ValueError, match="csr_wx"):
        read_force_operator(path, incompatible)
