"""Distributed sufficient statistics and fitting for TrajMap force operators.

The linear/optimal-linear force-mapping method implemented here is due to
Kraemer, Durumeric, Charron, Chen, Clementi & Noe, "Statistically Optimal
Force Aggregation for Coarse-Graining Molecular Dynamics", J. Phys. Chem.
Lett. 14(17), 3970-3979 (2023), https://doi.org/10.1021/acs.jpclett.3c00444.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..io.trajectory import broadcast_root_outcome, raise_if_rank_failed
from ..io.coordinates import minimum_image_displacements
from ..topology.cgmap import CGMapSpec

__all__ = ["accumulate_force_map_statistics", "fit_force_map"]


def _molecule_atoms(spec: CGMapSpec, molecule: int) -> np.ndarray:
    start, stop = int(spec.mol_indptr[molecule]), int(spec.mol_indptr[molecule + 1])
    if spec.molecules_contiguous:
        return np.arange(start, stop, dtype=np.int64)
    assert spec.mol_atom_pos is not None
    return np.asarray(spec.mol_atom_pos[start:stop], dtype=np.int64)


def _layout(spec: CGMapSpec, force_cfg: Any) -> Tuple[str, Tuple[Dict[str, Any], ...]]:
    scope = str(force_cfg.scope).replace("-", "_").lower()
    global_bytes = max(spec.n_required_atoms ** 2 * 8, spec.n_required_atoms * spec.n_sites * 16)
    if scope == "auto":
        repeated = np.any(np.asarray(spec.group_repeats) > 1)
        scope = (
            "per_template"
            if repeated or global_bytes > force_cfg.max_covariance_bytes
            else "global"
        )
    if scope == "global" and global_bytes > force_cfg.max_covariance_bytes:
        raise MemoryError("scope=global exceeds [force_mapping] max_covariance_bytes.")
    if scope == "global":
        coordinate = np.zeros((spec.n_sites, spec.n_required_atoms), dtype=np.float64)
        authored = np.zeros_like(coordinate)
        for site in range(spec.n_sites):
            start, stop = int(spec.csr_indptr[site]), int(spec.csr_indptr[site + 1])
            coordinate[site, spec.csr_cols[start:stop]] += spec.csr_wx[start:stop]
            authored[site, spec.csr_cols[start:stop]] += spec.csr_wf[start:stop]
        return scope, ({
            "name": "global",
            "coordinate": coordinate,
            "authored": authored,
            "atoms": np.arange(spec.n_required_atoms, dtype=np.int64)[None, :],
            "sites": np.arange(spec.n_sites, dtype=np.int64)[None, :],
        },)
    buckets: Dict[Tuple[int, int], list[Dict[str, Any]]] = {}
    for group in range(len(spec.group_repeats)):
        repeat, mol_start = int(spec.group_repeats[group]), int(spec.group_mol_offsets[group])
        site_start, width = int(spec.group_site_offsets[group]), int(spec.group_unit_sites[group])
        atoms0 = _molecule_atoms(spec, mol_start)
        sites0 = np.arange(site_start, site_start + width, dtype=np.int64)
        coordinate = np.zeros((width, atoms0.size), dtype=np.float64)
        authored = np.zeros_like(coordinate)
        lookup = {int(atom): index for index, atom in enumerate(atoms0)}
        for local_site, site in enumerate(sites0):
            for entry in range(int(spec.csr_indptr[site]), int(spec.csr_indptr[site + 1])):
                local_atom = lookup.get(int(spec.csr_cols[entry]))
                if local_atom is None:
                    raise ValueError("per-template force mapping cannot cross molecule ownership.")
                coordinate[local_site, local_atom] += spec.csr_wx[entry]
                authored[local_site, local_atom] += spec.csr_wf[entry]
        bucket = None
        for candidate in buckets.setdefault(tuple(coordinate.shape), []):
            if (
                np.array_equal(candidate["coordinate"], coordinate)
                and np.array_equal(candidate["authored"], authored)
            ):
                bucket = candidate
                break
        if bucket is None:
            bucket = {
                "name": f"template_{sum(len(v) for v in buckets.values()):04d}",
                "coordinate": coordinate,
                "authored": authored,
                "atoms": [],
                "sites": [],
            }
            buckets[tuple(coordinate.shape)].append(bucket)
        for offset in range(repeat):
            molecule = mol_start + offset
            atoms = _molecule_atoms(spec, molecule)
            sites = np.arange(
                site_start + offset * width,
                site_start + (offset + 1) * width,
                dtype=np.int64,
            )
            if atoms.size != coordinate.shape[1]:
                raise ValueError("a repeat does not match its force-map template width.")
            bucket["atoms"].append(atoms)
            bucket["sites"].append(sites)
    layout = []
    for values in buckets.values():
        for value in values:
            layout.append({
                **value,
                "atoms": np.stack(value["atoms"]),
                "sites": np.stack(value["sites"]),
            })
    return scope, tuple(layout)


def _fit_ids(selected: Sequence[int], force_cfg: Any) -> Tuple[int, ...]:
    available = tuple(int(value) for value in selected)
    if force_cfg.fit_frame_ids is None:
        ids = available[::int(force_cfg.fit_every)]
    else:
        ids = tuple(int(value) for value in force_cfg.fit_frame_ids)
    if force_cfg.fit_frame_ids is not None:
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("force-map fitting ids must be non-empty and unique.")
        if set(ids) - set(available):
            raise ValueError("[force_mapping] fit_frame_ids must be selected mapping frames.")
    if force_cfg.fit_n_frames:
        ids = ids[:int(force_cfg.fit_n_frames)]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("force-map fitting ids must be non-empty and unique.")
    return ids


def _components(n_atoms: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(n_atoms, dtype=np.int64)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index
    for left, right in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
        left, right = root(int(left)), root(int(right))
        if left != right:
            parent[max(left, right)] = min(left, right)
    roots = np.asarray([root(index) for index in range(n_atoms)], dtype=np.int64)
    _, inverse = np.unique(roots, return_inverse=True)
    return np.eye(int(inverse.max()) + 1, dtype=np.float64)[inverse]


def _uniform(coordinate: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    blocks = _components(coordinate.shape[1], pairs)
    result = np.zeros_like(coordinate)
    for row, values in enumerate(coordinate):
        used = (
            np.any(blocks[np.flatnonzero(values)] > 0.0, axis=0)
            if np.any(values)
            else np.zeros(blocks.shape[1], dtype=bool)
        )
        if np.any(used):
            result[row, np.any(blocks[:, used] > 0.0, axis=1)] = 1.0
    return result


def _pairs_from_moments(moment: Mapping[str, np.ndarray], threshold: float) -> np.ndarray:
    count = np.asarray(moment["count"], dtype=np.float64)
    if count.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if np.any(count < 2):
        raise ValueError("automatic constraint detection needs at least two fit frames.")
    mean = moment["sum"] / count
    variance = np.maximum(moment["sum_square"] / count - mean * mean, 0.0)
    left, right = np.triu_indices(int(moment["n_atoms"]), k=1)
    constrained = np.sqrt(variance) < threshold
    return np.column_stack((left[constrained], right[constrained])).astype(np.int64)


def _known_pairs(
    config: Any,
    spec: CGMapSpec,
    layout: Sequence[Mapping[str, Any]],
    topology_path: Optional[Path],
) -> Tuple[str, Tuple[np.ndarray, ...]]:
    force_cfg = config.force_mapping
    if force_cfg.constraint_pairs_file is not None:
        import yaml
        base = config.path.parent if config.path is not None else Path.cwd()
        raw = yaml.safe_load((base / force_cfg.constraint_pairs_file).read_text(encoding="utf-8"))
        global_pairs = raw.get("pairs") if isinstance(raw, dict) else raw
        if global_pairs is not None and not isinstance(global_pairs, dict):
            pairs = np.asarray(global_pairs, dtype=np.int64).reshape(-1, 2)
            return f"file:{force_cfg.constraint_pairs_file}", tuple(
                pairs.copy() for _ in layout
            )
        table = raw.get("templates", raw) if isinstance(raw, dict) else None
        if not isinstance(table, dict):
            raise ValueError("constraint pair file must be pairs or a template mapping.")
        pairs = []
        for index, item in enumerate(layout):
            raw_pairs = table.get(item["name"], table.get(str(index), table.get("default")))
            pairs.append(np.asarray(raw_pairs, dtype=np.int64).reshape(-1, 2))
        return f"file:{force_cfg.constraint_pairs_file}", tuple(pairs)
    mode = force_cfg.constraints
    if mode == "none":
        return mode, tuple(np.empty((0, 2), dtype=np.int64) for _ in layout)
    if mode == "auto":
        return mode, tuple(np.empty((0, 2), dtype=np.int64) for _ in layout)
    if topology_path is None:
        raise ValueError("topology-derived constraints require [aa] topology.")
    from ..io.trajectory import open_universe
    universe = open_universe(topology_path, topology_format=config.aa.topology_format)
    try:
        bonds = np.asarray(universe.bonds.indices, dtype=np.int64)
        atoms = universe.atoms
        hydrogen = np.zeros(int(atoms.n_atoms), dtype=bool)
        available = []
        for attribute, predicate in (
            ("elements", lambda x: np.char.upper(np.asarray(x, dtype=str)) == "H"),
            (
                "names",
                lambda x: np.char.startswith(
                    np.char.upper(np.asarray(x, dtype=str)), "H"
                ),
            ),
            (
                "masses",
                lambda x: (np.asarray(x, dtype=np.float64) >= 0.5)
                & (np.asarray(x, dtype=np.float64) <= 2.0),
            ),
        ):
            # A topology that simply does not carry the attribute is the one
            # condition that legitimately varies here, and MDAnalysis signals it
            # with NoDataError, which subclasses AttributeError -- so getattr's
            # default covers exactly that case. Every other failure must
            # surface: these heuristics decide which bonds are constrained,
            # hence the compression, the equality block and the CG reference
            # forces, and a partially failed heuristic changes all of them with
            # no other symptom.
            values = getattr(atoms, attribute, None)
            if values is None:
                continue
            available.append(attribute)
            hydrogen |= predicate(values)
        if mode == "h-bonds" and not np.any(hydrogen):
            raise ValueError(
                "could not identify hydrogen atoms in the AA topology; "
                "the topology provided "
                + (", ".join(available) if available else "none")
                + " of elements/names/masses."
            )
        if mode == "h-bonds":
            bonds = bonds[hydrogen[bonds[:, 0]] | hydrogen[bonds[:, 1]]]
        answer = []
        for item in layout:
            absolute = spec.atom_indices[item["atoms"][0]]
            inverse = np.full(hydrogen.size, -1, dtype=np.int64)
            inverse[absolute] = np.arange(absolute.size)
            pair = np.column_stack((inverse[bonds[:, 0]], inverse[bonds[:, 1]]))
            answer.append(pair[np.all(pair >= 0, axis=1)])
        return mode, tuple(answer)
    finally:
        del universe


def accumulate_force_map_statistics(
    *,
    config: Any,
    spec: CGMapSpec,
    reader: Any,
    plan: Any,
    topology_path: Optional[Path],
    comm: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Resolve constraints and accumulate the one permitted fitting frame pass."""
    force_cfg, rank = config.force_mapping, 0 if comm is None else int(comm.Get_rank())
    setup = error = None
    if rank == 0:
        try:
            scope, layout = _layout(spec, force_cfg)
            ids = _fit_ids(plan.frame_ids, force_cfg)
            source, pairs = _known_pairs(config, spec, layout, topology_path)
            widths = [item["coordinate"].shape[1] for item in layout]
            reduced = [
                width if source == "auto" else _components(width, pair).shape[1]
                for width, pair in zip(widths, pairs)
            ]
            covariance_bytes = sum(width * width * 8 for width in reduced)
            if (
                force_cfg.method == "optimal_linear"
                and covariance_bytes > force_cfg.max_covariance_bytes
            ):
                raise MemoryError("force-map covariance exceeds [force_mapping] max_covariance_bytes.")
            setup = {"scope": scope, "layout": layout, "fit_ids": ids, "source": source, "pairs": pairs}
        except Exception as exc:
            error = exc
    setup = broadcast_root_outcome((setup, error) if rank == 0 else None, comm=comm)
    method, auto = force_cfg.method, setup["source"] == "auto"
    needs_pass = auto or method == "optimal_linear"
    if not needs_pass:
        return setup if rank == 0 else None
    local = []
    compressions = []
    uniform_maps = []
    allocation_error = None
    try:
        for item, pairs in zip(setup["layout"], setup["pairs"]):
            width = item["coordinate"].shape[1]
            compression = None if auto else _components(width, pairs)
            compressions.append(compression)
            uniform_maps.append(
                None if auto else _uniform(item["coordinate"], pairs)
            )
            quadratic_width = width if compression is None else compression.shape[1]
            payload = {
                "quadratic": (
                    np.zeros((quadratic_width, quadratic_width), dtype=np.float64)
                    if method == "optimal_linear"
                    else None
                ),
                "vectors": 0,
                "authored_force_sq_sum": 0.0,
                "uniform_force_sq_sum": 0.0,
            }
            if auto:
                moment_width = width * (width - 1) // 2
                payload.update({
                    "count": np.zeros(moment_width, dtype=np.float64),
                    "sum": np.zeros(moment_width, dtype=np.float64),
                    "sum_square": np.zeros(moment_width, dtype=np.float64),
                })
            local.append(payload)
    except Exception as exc:
        allocation_error = exc
    raise_if_rank_failed(allocation_error, comm=comm)

    original = reader.plan
    local_error = None
    try:
        reader.plan = replace(original, frame_ids=tuple(setup["fit_ids"]))
        for frame in reader.iter_local(include_forces=method == "optimal_linear", atom_indices=spec.atom_indices):
            positions = np.asarray(frame["positions"], dtype=np.float64).reshape(-1, 3)
            values = (
                None
                if method != "optimal_linear"
                else np.asarray(frame["forces"], dtype=np.float64).reshape(-1, 3)
            )
            for item, payload, compression, uniform_map in zip(
                setup["layout"], local, compressions, uniform_maps
            ):
                if auto:
                    representative = positions[item["atoms"][0]]
                    delta = representative[:, None, :] - representative[None, :, :]
                    box = frame.get("box")
                    if box is not None:
                        minimum_image_displacements(
                            delta,
                            box,
                            triclinic=getattr(
                                getattr(config, "run", None), "triclinic", "exact"
                            ),
                        )
                    upper = np.triu_indices(item["coordinate"].shape[1], k=1)
                    distances = np.linalg.norm(delta[upper], axis=1)
                    payload["count"] += 1
                    payload["sum"] += distances
                    payload["sum_square"] += distances * distances
                if values is not None:
                    blocks = values[item["atoms"]]
                    reduced = (
                        blocks
                        if compression is None
                        else np.einsum(
                            "ifd,fr->ird", blocks, compression, optimize=True
                        )
                    )
                    rows = np.swapaxes(reduced, 1, 2).reshape(-1, reduced.shape[1])
                    payload["quadratic"] += rows.T @ rows
                    payload["vectors"] += int(rows.shape[0])
                    if not auto:
                        assert uniform_map is not None
                        authored_forces = np.einsum(
                            "cf,ifd->icd",
                            item["authored"],
                            blocks,
                            optimize=True,
                        )
                        uniform_forces = np.einsum(
                            "cf,ifd->icd",
                            uniform_map,
                            blocks,
                            optimize=True,
                        )
                        payload["authored_force_sq_sum"] += float(
                            np.sum(np.square(authored_forces))
                        )
                        payload["uniform_force_sq_sum"] += float(
                            np.sum(np.square(uniform_forces))
                        )
    except Exception as exc:
        local_error = exc
    finally:
        reader.plan = original
    raise_if_rank_failed(local_error, comm=comm)
    gathered = [local] if comm is None else comm.gather(local, root=0)
    merged = root_error = None
    if rank == 0:
        try:
            merged = setup
            merged["statistics"] = []
            for index, item in enumerate(setup["layout"]):
                payload = {
                    "quadratic": (
                        None
                        if method != "optimal_linear"
                        else np.zeros_like(gathered[0][index]["quadratic"])
                    ),
                    "vectors": 0,
                    "n_atoms": item["coordinate"].shape[1],
                    "authored_force_sq_sum": 0.0,
                    "uniform_force_sq_sum": 0.0,
                }
                if auto:
                    payload.update({
                        "count": np.zeros_like(gathered[0][index]["count"]),
                        "sum": np.zeros_like(gathered[0][index]["sum"]),
                        "sum_square": np.zeros_like(gathered[0][index]["sum_square"]),
                    })
                for rank_payload in gathered:
                    incoming = rank_payload[index]
                    if payload["quadratic"] is not None:
                        payload["quadratic"] += incoming["quadratic"]
                    if auto:
                        payload["count"] += incoming["count"]
                        payload["sum"] += incoming["sum"]
                        payload["sum_square"] += incoming["sum_square"]
                    payload["vectors"] += incoming["vectors"]
                    payload["authored_force_sq_sum"] += incoming["authored_force_sq_sum"]
                    payload["uniform_force_sq_sum"] += incoming["uniform_force_sq_sum"]
                if auto:
                    merged["pairs"] = tuple(
                        _pairs_from_moments(payload, force_cfg.constraint_threshold)
                        if position == index
                        else pair
                        for position, pair in enumerate(merged["pairs"])
                    )
                merged["statistics"].append(payload)
            if method == "optimal_linear":
                for item, statistic, pair in zip(merged["layout"], merged["statistics"], merged["pairs"]):
                    compression = _components(item["coordinate"].shape[1], pair)
                    statistic["compression"] = compression
                    if auto:
                        statistic["raw_quadratic"] = statistic["quadratic"]
                        statistic["quadratic"] = compression.T @ statistic["quadratic"] @ compression
        except Exception as exc:
            root_error = exc
    broadcast_root_outcome((None, root_error) if rank == 0 else None, comm=comm)
    return merged if rank == 0 else None


def _solve(
    quadratic: np.ndarray,
    equality: np.ndarray,
    l2: np.ndarray,
    regularization: float,
    backend: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    def native() -> Tuple[np.ndarray, Dict[str, Any]]:
        pmat = quadratic + regularization * l2
        scale = max(float(np.max(np.abs(pmat))), 1.0)
        kkt = np.block([
            [pmat / scale, equality.T],
            [equality, np.zeros((equality.shape[0], equality.shape[0]))],
        ])
        rhs = np.vstack((
            np.zeros((pmat.shape[0], equality.shape[0])),
            np.eye(equality.shape[0]),
        ))
        try:
            solution = np.linalg.solve(kkt, rhs)
            solver = "solve"
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
            solver = "lstsq"
        return solution[:pmat.shape[0]].T, {
            "linear_solver": solver,
            "quadratic_condition": float(np.linalg.cond(pmat)),
        }

    def osqp() -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            from qpsolvers import solve_qp
        except ImportError as exc:
            raise RuntimeError("backend='osqp' requires qpsolvers and osqp.") from exc
        regularized = quadratic + regularization * l2
        pmat = 0.5 * (regularized + regularized.T)
        started, result = time.monotonic(), []
        for index in range(equality.shape[0]):
            target = np.zeros(equality.shape[0])
            target[index] = 1.0
            row = solve_qp(
                P=pmat,
                q=np.zeros(pmat.shape[0]),
                A=equality,
                b=target,
                solver="osqp",
                eps_abs=1e-8,
                eps_rel=1e-8,
                max_iter=100000,
                polish=True,
            )
            if row is None:
                raise RuntimeError("OSQP failed to solve a force-map row.")
            result.append(row)
        return np.stack(result), {
            "linear_solver": "qpsolvers.osqp",
            "quadratic_condition": float(np.linalg.cond(pmat)),
            "elapsed_seconds": time.monotonic() - started,
        }

    def diagnostics(coefficients: np.ndarray) -> Dict[str, float]:
        return {
            "consistency_max_abs": float(
                np.max(
                    np.abs(coefficients @ equality.T - np.eye(equality.shape[0]))
                )
            ),
            "objective_sum": float(
                np.einsum("ir,rs,is->", coefficients, quadratic, coefficients)
            ),
        }

    native_coefficients = native_info = osqp_coefficients = osqp_info = None
    if backend in {"native", "auto", "compare"}:
        native_coefficients, native_info = native()
    if backend in {"osqp", "compare"}:
        osqp_coefficients, osqp_info = osqp()
    if backend == "native":
        return native_coefficients, {**native_info, "backend": "native"}
    if backend == "osqp":
        return osqp_coefficients, {**osqp_info, "backend": "osqp"}
    if backend == "compare":
        native_diag, osqp_diag = diagnostics(native_coefficients), diagnostics(osqp_coefficients)
        comparison = {
            "matrix_max_abs_delta": float(
                np.max(np.abs(native_coefficients - osqp_coefficients))
            ),
            "objective_relative_delta": (
                abs(native_diag["objective_sum"] - osqp_diag["objective_sum"])
                / max(abs(osqp_diag["objective_sum"]), 1.0)
            ),
            "native": {**native_info, **native_diag},
            "osqp": {**osqp_info, **osqp_diag},
        }
        native_ok = (
            native_diag["consistency_max_abs"] <= 1.e-7
            and comparison["objective_relative_delta"] <= 1.e-7
        )
        chosen_coefficients = native_coefficients if native_ok else osqp_coefficients
        chosen_info = native_info if native_ok else osqp_info
        return chosen_coefficients, {
            **chosen_info,
            "backend": "native" if native_ok else "osqp",
            "backend_comparison": comparison,
        }
    if diagnostics(native_coefficients)["consistency_max_abs"] <= 1.e-7:
        return native_coefficients, {**native_info, "backend": "native"}
    osqp_coefficients, osqp_info = osqp()
    return osqp_coefficients, {**osqp_info, "backend": "osqp", "fallback_from": "native"}


def fit_force_map(
    *,
    config: Any,
    spec: CGMapSpec,
    statistics: Optional[Mapping[str, Any]],
    comm: Optional[Any],
) -> Tuple[CGMapSpec, Dict[str, Any]]:
    """Attach an analytic or statistically optimal operator without reading frames."""
    rank, outcome, error = 0 if comm is None else int(comm.Get_rank()), None, None
    if rank == 0:
        try:
            if statistics is None:
                raise ValueError("fit_force_map requires statistics on rank 0.")
            cfg, matrices, diagnostics = config.force_mapping, [], []
            empty_statistics = (None,) * len(statistics["layout"])
            for item, pairs, statistic in zip(
                statistics["layout"],
                statistics["pairs"],
                statistics.get("statistics", empty_statistics),
            ):
                coordinate = item["coordinate"]
                if cfg.method == "constraint_aware_uniform":
                    matrix = _uniform(coordinate, pairs)
                    detail = {
                        "backend": "analytic_uniform",
                        "n_constraints": int(pairs.shape[0]),
                    }
                else:
                    if statistic is None or statistic["vectors"] <= 0:
                        raise ValueError(
                            f"force-map statistic for {item['name']!r} has no "
                            "sampled vectors; cannot fit a non-uniform operator."
                        )
                    compression = statistic["compression"]
                    equality = coordinate @ compression
                    if np.linalg.matrix_rank(equality) != coordinate.shape[0]:
                        raise ValueError(
                            "coordinate/constraint system is not full row rank."
                        )
                    coefficients, detail = _solve(
                        statistic["quadratic"],
                        equality,
                        compression.T @ compression,
                        cfg.l2_regularization,
                        cfg.backend,
                    )
                    matrix = coefficients @ compression.T
                    consistency = float(
                        np.max(
                            np.abs(
                                matrix @ coordinate.T - np.eye(coordinate.shape[0])
                            )
                        )
                    )
                    if consistency > 1e-7:
                        raise RuntimeError("fitted force map violates W C^T = I.")
                    raw_quadratic = statistic.get("raw_quadratic")
                    authored_sq = (
                        statistic["authored_force_sq_sum"]
                        if raw_quadratic is None
                        else float(
                            np.einsum(
                                "ir,rs,is->",
                                item["authored"],
                                raw_quadratic,
                                item["authored"],
                            )
                        )
                    )
                    uniform = _uniform(coordinate, pairs)
                    uniform_sq = (
                        statistic["uniform_force_sq_sum"]
                        if raw_quadratic is None
                        else float(
                            np.einsum(
                                "ir,rs,is->", uniform, raw_quadratic, uniform
                            )
                        )
                    )
                    detail.update({
                        "n_vector_samples": int(statistic["vectors"]),
                        "n_force_instances": int(statistic["vectors"] // 3),
                        "n_constraints": int(pairs.shape[0]),
                        "consistency_max_abs": consistency,
                        "objective_sum": float(
                            np.einsum(
                                "ir,rs,is->",
                                coefficients,
                                statistic["quadratic"],
                                coefficients,
                            )
                        ),
                        "authored_force_sq_sum": authored_sq,
                        "uniform_force_sq_sum": uniform_sq,
                    })
                matrices.append(matrix)
                diagnostics.append({"name": item["name"], **detail})
            backends = {entry["backend"] for entry in diagnostics}
            metadata = {
                "method": cfg.method,
                "backend": diagnostics[0]["backend"] if len(backends) == 1 else "mixed",
                "constraints": statistics["source"],
                "constraint_algorithm": cfg.constraint_algorithm,
                "scope": statistics["scope"],
                "fit_frame_count": len(statistics["fit_ids"]),
                "diagnostics": diagnostics,
            }
            attached = spec.with_force_operator(
                matrices,
                [item["atoms"] for item in statistics["layout"]],
                [item["sites"] for item in statistics["layout"]],
                [item["coordinate"] for item in statistics["layout"]],
                [item["authored"] for item in statistics["layout"]],
                statistics["pairs"],
                metadata,
            )
            fit_ids = statistics["fit_ids"]
            outcome = (attached, {
                **metadata,
                "fit_frames": {
                    "count": len(fit_ids),
                    "first": int(fit_ids[0]),
                    "last": int(fit_ids[-1]),
                    "every": cfg.fit_every,
                },
            })
        except Exception as exc:
            error = exc
    return broadcast_root_outcome((outcome, error) if rank == 0 else None, comm=comm)
