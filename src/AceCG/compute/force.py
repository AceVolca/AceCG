"""Per-frame force-side observable kernel."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey
from .frame_geometry import FrameGeometry, _min_image_batch

RAD2DEG = 180.0 / np.pi


def _add_timing(bucket: Optional[Dict[str, Any]], key: str, dt: float) -> None:
    if bucket is not None:
        bucket[key] = float(bucket.get(key, 0.0)) + float(dt)


def _observed_rows(n_atoms: int, real_site_indices: Optional[np.ndarray]) -> np.ndarray:
    if real_site_indices is None:
        return np.arange(int(n_atoms) * 3, dtype=np.int32)
    atoms = np.asarray(real_site_indices, dtype=np.int32).reshape(-1)
    rows = (atoms[:, None] * 3 + np.arange(3, dtype=np.int32)[None, :]).reshape(-1)
    rows.sort()
    return rows


def _normalize_reference_force_batch(
    reference_force: np.ndarray,
    *,
    n_samples: int,
    n_atoms: Optional[int] = None,
) -> np.ndarray:
    ref = np.asarray(reference_force, dtype=np.float32)
    if ref.ndim == 1:
        return np.broadcast_to(ref.reshape(1, -1), (n_samples, ref.size)).copy()
    if ref.ndim == 2 and n_atoms is not None and ref.shape == (int(n_atoms), 3):
        flat = ref.reshape(1, -1)
        return np.broadcast_to(flat, (n_samples, flat.shape[1])).copy()
    if ref.ndim == 2 and ref.shape[0] == n_samples:
        return ref.reshape(n_samples, -1)
    if ref.ndim == 2 and ref.shape[-1] == 3:
        flat = ref.reshape(1, -1)
        return np.broadcast_to(flat, (n_samples, flat.shape[1])).copy()
    if ref.ndim == 3 and ref.shape[0] == n_samples:
        return ref.reshape(n_samples, -1)
    if ref.ndim >= 2 and int(np.prod(ref.shape[:-1], dtype=np.int64)) == n_samples:
        return ref.reshape(n_samples, -1)
    if (
        ref.ndim >= 3
        and n_atoms is not None
        and ref.shape[-2:] == (int(n_atoms), 3)
        and int(np.prod(ref.shape[:-2], dtype=np.int64)) == n_samples
    ):
        return ref.reshape(n_samples, -1)
    if (
        ref.ndim >= 3
        and ref.shape[-1] == 3
        and int(np.prod(ref.shape[:-2], dtype=np.int64)) == n_samples
    ):
        return ref.reshape(n_samples, -1)
    raise ValueError(
        "reference_force must be a force vector, force array, or batch with "
        f"leading n_samples={n_samples}; got shape {ref.shape}."
    )


def _slice_observed_force_batch(
    values: np.ndarray,
    real_site_indices: Optional[np.ndarray],
    *,
    n_atoms: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    rows = _observed_rows(n_atoms, real_site_indices)
    if arr.shape[1] == rows.size:
        return arr
    if rows.size and rows[-1] >= arr.shape[1]:
        raise ValueError(
            "reference_force batch does not match either the observed or full force-row shape."
        )
    return arr[:, rows]


def _geometry_sample_shape(geom: FrameGeometry) -> tuple[int, ...]:
    positions = np.asarray(geom.positions)
    if positions.ndim < 2:
        return ()
    return tuple(positions.shape[:-2])


def _flat_sample_count(sample_shape: tuple[int, ...]) -> int:
    return int(np.prod(sample_shape, dtype=np.int64)) if sample_shape else 1


def _batch_add_param_rows(
    mat: Optional[np.ndarray],
    sl: slice,
    sample_ids: np.ndarray,
    atom_indices: np.ndarray,
    values: np.ndarray,
) -> None:
    if mat is None or values.size == 0:
        return
    vals = np.asarray(values, dtype=np.float32)
    n_params_local = sl.stop - sl.start
    rows = (
        np.asarray(atom_indices, dtype=np.int64)[:, None] * 3
        + np.arange(3, dtype=np.int64)[None, :]
    )
    params = np.arange(sl.start, sl.stop, dtype=np.int64)
    sample_ids_arr = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    if vals.shape != (sample_ids_arr.size, 3, n_params_local):
        raise ValueError(
            f"Batch projector values have shape {vals.shape}, expected "
            f"{(sample_ids_arr.size, 3, n_params_local)}."
        )
    n_rows = mat.shape[1]
    n_params_total = mat.shape[2]
    linear = (
        (
            sample_ids_arr[:, None, None] * n_rows
            + rows[:, :, None]
        )
        * n_params_total
        + params[None, None, :]
    )
    np.add.at(mat.reshape(-1), linear.ravel(), vals.ravel())


def _batch_add_sparse_param_rows(
    mat: Optional[np.ndarray],
    sample_ids: np.ndarray,
    atom_indices: np.ndarray,
    param_cols: np.ndarray,
    values: np.ndarray,
) -> None:
    if mat is None or values.size == 0:
        return
    vals = np.asarray(values, dtype=np.float32)
    if vals.shape != (sample_ids.size, 3):
        raise ValueError(
            f"Sparse batch projector values have shape {vals.shape}, expected "
            f"{(sample_ids.size, 3)}."
        )
    rows = (
        np.asarray(atom_indices, dtype=np.int64)[:, None] * 3
        + np.arange(3, dtype=np.int64)[None, :]
    )
    sample_ids_arr = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    param_cols_arr = np.asarray(param_cols, dtype=np.int64).reshape(-1)
    n_rows = mat.shape[1]
    n_params_total = mat.shape[2]
    linear = (
        (
            sample_ids_arr[:, None] * n_rows
            + rows
        )
        * n_params_total
        + param_cols_arr[:, None]
    )
    np.add.at(mat.reshape(-1), linear.ravel(), vals.ravel())


def _batch_add_force_rows(
    fvec: Optional[np.ndarray],
    sample_ids: np.ndarray,
    atom_indices: np.ndarray,
    values: np.ndarray,
) -> None:
    if fvec is None or values.size == 0:
        return
    vals = np.asarray(values, dtype=np.float32)
    n_rows = fvec.shape[1]
    rows = (
        np.asarray(atom_indices, dtype=np.int64)[:, None] * 3
        + np.arange(3, dtype=np.int64)[None, :]
    )
    linear = np.asarray(sample_ids, dtype=np.int64)[:, None] * n_rows + rows
    if vals.shape != (sample_ids.size, 3):
        raise ValueError(
            f"Batch force values have shape {vals.shape}, expected {(sample_ids.size, 3)}."
        )
    np.add.at(fvec.reshape(-1), linear.ravel(), vals.ravel())


def _valid_flat_terms(
    values: np.ndarray,
    *,
    cutoff: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values)
    valid = arr > 1.0e-12
    if cutoff is not None and np.isfinite(float(cutoff)):
        valid &= arr <= float(cutoff)
    sample_ids, term_ids = np.nonzero(valid)
    return sample_ids.astype(np.int64, copy=False), term_ids.astype(np.int64, copy=False)


def _positions_box_as_batch(
    geom: FrameGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(geom.positions, dtype=np.float32)
    sample_shape = _geometry_sample_shape(geom)
    n_samples = _flat_sample_count(sample_shape)
    pos_batch = pos.reshape(n_samples, int(geom.n_atoms), 3)
    box = np.asarray(geom.box, dtype=np.float32)
    if box.ndim == 1:
        box_batch = np.broadcast_to(box, (n_samples, 6)).copy()
    else:
        box_batch = box.reshape(n_samples, 6)
    return pos_batch, box_batch


def _values_as_batch(values: np.ndarray, *, trailing_ndim: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == trailing_ndim + 1:
        return arr[None, ...]
    term_shape = arr.shape[-(trailing_ndim + 1):]
    leading_shape = arr.shape[:-(trailing_ndim + 1)]
    n_samples = int(np.prod(leading_shape, dtype=np.int64))
    return arr.reshape((n_samples,) + term_shape)


def force(
    frame_geometry: FrameGeometry,
    forcefield: Forcefield,
    *,
    return_value: bool = False,
    return_grad: bool = False,
    return_hessian: bool = False,
    reference_force: Optional[np.ndarray] = None,
    frame_weights: Optional[np.ndarray] = None,
    frame_weight: float = 1.0,
    return_fm_stats: bool = False,
    timing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-frame or same-frame-batch force-side observables."""
    if not (return_value or return_grad or return_hessian or return_fm_stats):
        return {}
    sample_shape = _geometry_sample_shape(frame_geometry)
    n_samples = _flat_sample_count(sample_shape)
    n_atoms = int(frame_geometry.n_atoms)
    n_rows = n_atoms * 3
    rsi = frame_geometry.real_site_indices
    interaction_mask = forcefield.key_mask
    param_mask = forcefield.param_mask
    param_blocks = forcefield.param_blocks()
    n_params = forcefield.n_params()
    if frame_weights is None:
        weights = (
            np.full(n_samples, 1.0 / float(n_samples), dtype=np.float32)
            if sample_shape
            else np.array([float(frame_weight)], dtype=np.float32)
        )
    else:
        weights = np.asarray(frame_weights, dtype=np.float32).reshape(-1)
        if weights.shape != (n_samples,):
            raise ValueError(
                "frame_weights must flatten to the geometry sample count "
                f"{n_samples}; got {weights.shape}."
            )
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("frame_weights must be nonnegative with positive sum.")
    if return_fm_stats and reference_force is None:
        raise ValueError("reference_force is required when return_fm_stats=True.")

    need_jacobian = return_grad or return_hessian or return_fm_stats
    rows = _observed_rows(n_atoms, rsi)
    mat = (
        np.zeros((n_samples, n_rows, n_params), dtype=np.float32)
        if need_jacobian
        else None
    )
    fvec = (
        np.zeros((n_samples, n_rows), dtype=np.float32)
        if (return_value or return_fm_stats)
        else None
    )

    for key, pot, sl in param_blocks:
        if interaction_mask is not None and not interaction_mask.get(key, True):
            continue
        t_project = time.monotonic()
        if key.style == "pair":
            _project_pair(mat, fvec, frame_geometry, key, pot, sl)
        elif key.style == "bond":
            _project_bond(mat, fvec, frame_geometry, key, pot, sl)
        elif key.style == "angle":
            _project_angle(mat, fvec, frame_geometry, key, pot, sl)
        elif key.style == "dihedral":
            _project_dihedral(mat, fvec, frame_geometry, key, pot, sl)
        _add_timing(timing, f"force_project_{key.style}", time.monotonic() - t_project)

    mat_obs = mat[:, rows, :] if mat is not None else None
    fvec_obs = fvec[:, rows] if fvec is not None else None
    if param_mask is not None and mat_obs is not None:
        pmask = np.asarray(param_mask, dtype=bool)
        if not np.all(pmask):
            mat_obs[:, :, ~pmask] = 0.0

    result: Dict[str, Any] = {}
    if return_grad:
        assert mat_obs is not None
        value = mat_obs.reshape(sample_shape + mat_obs.shape[1:]) if sample_shape else mat_obs[0]
        result["force_grad"] = value

    if return_value:
        assert fvec_obs is not None
        value = fvec_obs.reshape(sample_shape + fvec_obs.shape[1:]) if sample_shape else fvec_obs[0]
        result["force"] = value

    if return_hessian:
        assert mat_obs is not None
        t_hessian = time.monotonic()
        hessian = np.einsum("brp,brq->bpq", mat_obs, mat_obs, optimize=True)
        result["force_hessian"] = (
            hessian.reshape(sample_shape + hessian.shape[1:])
            if sample_shape
            else hessian[0]
        )
        _add_timing(timing, "force_hessian", time.monotonic() - t_hessian)

    if return_fm_stats:
        assert mat_obs is not None and fvec_obs is not None
        t_stats = time.monotonic()
        reference_batch = _normalize_reference_force_batch(
            reference_force,
            n_samples=n_samples,
            n_atoms=n_atoms,
        )
        reference_obs = _slice_observed_force_batch(reference_batch, rsi, n_atoms=n_atoms)
        stats = _fm_stats(
            jacobian=mat_obs,
            model_force=fvec_obs,
            reference_force=reference_obs,
            weights=weights,
            n_params=n_params,
        )
        result["fm_stats_sum" if sample_shape else "fm_stats"] = stats
        _add_timing(timing, "force_fm_stats", time.monotonic() - t_stats)

    return result


def _fm_stats(
    *,
    jacobian: np.ndarray,
    model_force: np.ndarray,
    reference_force: np.ndarray,
    weights: np.ndarray,
    n_params: int,
) -> Dict[str, Any]:
    f_obs = np.asarray(model_force, dtype=np.float32)
    y_obs = np.asarray(reference_force, dtype=np.float32)
    if f_obs.shape != y_obs.shape:
        raise ValueError(
            "Observed model/reference force shapes must match, "
            f"got {f_obs.shape} and {y_obs.shape}."
        )
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    n_samples = int(w.size)
    n_obs = int(f_obs.shape[1])
    if f_obs.shape != (n_samples, n_obs):
        raise ValueError(
            f"Observed force batch shape {f_obs.shape} does not match weights shape {w.shape}."
        )

    JtJ = np.zeros((n_params, n_params), dtype=np.float32)
    Jtf = np.zeros(n_params, dtype=np.float32)
    Jty = np.zeros(n_params, dtype=np.float32)
    J_obs = np.asarray(jacobian, dtype=np.float32)
    if J_obs.shape != (n_samples, n_obs, n_params):
        raise ValueError(
            "Force Jacobian batch shape must be "
            f"{(n_samples, n_obs, n_params)}, got {J_obs.shape}."
        )
    for sample_id in range(n_samples):
        wi = np.float32(w[sample_id])
        if wi == 0.0:
            continue
        J = J_obs[sample_id]
        JtJ += wi * (J.T @ J)
        Jtf += wi * (J.T @ f_obs[sample_id])
        Jty += wi * (J.T @ y_obs[sample_id])

    return {
        "JtJ": JtJ,
        "Jtf": Jtf,
        "Jty": Jty,
        "ftf": float(np.einsum("b,br,br->", w, f_obs, f_obs, optimize=True)),
        "fTy": float(np.einsum("b,br,br->", w, f_obs, y_obs, optimize=True)),
        "yty": float(np.einsum("b,br,br->", w, y_obs, y_obs, optimize=True)),
        "n_force_rows": n_obs,
        "n_frames": 1,
        "weight_sum": float(np.sum(w)),
        "n_atoms_obs": int(n_obs // 3),
    }


def _project_pair(
    mat: Optional[np.ndarray],
    fvec: Optional[np.ndarray],
    geom: FrameGeometry,
    key: InteractionKey,
    pot,
    sl: slice,
) -> None:
    idx_pair = geom.pair_indices.get(key)
    dr = geom.pair_vectors.get(key)
    r = geom.pair_distances.get(key)
    if idx_pair is None or r is None or r.size == 0:
        return
    a_idx, b_idx = idx_pair
    r_batch = _values_as_batch(r)
    dr_batch = _values_as_batch(dr, trailing_ndim=1)
    sample_ids, term_ids = _valid_flat_terms(
        r_batch,
        cutoff=getattr(pot, "cutoff", None),
    )
    if sample_ids.size == 0:
        return
    r_flat = r_batch[sample_ids, term_ids]
    dr_flat = dr_batch[sample_ids, term_ids]
    ia = np.asarray(a_idx, dtype=np.int64)[term_ids]
    ib = np.asarray(b_idx, dtype=np.int64)[term_ids]

    if mat is not None:
        force_grad = pot.force_grad(r_flat)
        if hasattr(force_grad, "tocoo"):
            coo = force_grad.tocoo()
            if coo.nnz > 0:
                row = np.asarray(coo.row, dtype=np.int64)
                col = np.asarray(coo.col, dtype=np.int64)
                dat = np.asarray(coo.data, dtype=np.float32)
                coeff = dat * (-1.0 / r_flat[row])
                vals = dr_flat[row] * coeff[:, None]
                param_cols = sl.start + col
                _batch_add_sparse_param_rows(
                    mat,
                    sample_ids[row],
                    ia[row],
                    param_cols,
                    vals,
                )
                _batch_add_sparse_param_rows(
                    mat,
                    sample_ids[row],
                    ib[row],
                    param_cols,
                    -vals,
                )
        else:
            B = np.asarray(force_grad, dtype=np.float32)
            coeff = B * (-1.0 / r_flat)[:, None]
            vals = dr_flat[:, :, None] * coeff[:, None, :]
            _batch_add_param_rows(mat, sl, sample_ids, ia, vals)
            _batch_add_param_rows(mat, sl, sample_ids, ib, -vals)

    if fvec is not None:
        F_scalar = np.asarray(pot.force(r_flat), dtype=np.float32).reshape(-1)
        vals = (-F_scalar / r_flat)[:, None] * dr_flat
        _batch_add_force_rows(fvec, sample_ids, ia, vals)
        _batch_add_force_rows(fvec, sample_ids, ib, -vals)


def _project_bond(
    mat: Optional[np.ndarray],
    fvec: Optional[np.ndarray],
    geom: FrameGeometry,
    key: InteractionKey,
    pot,
    sl: slice,
) -> None:
    terms = geom.bond_indices.get(key)
    dr = geom.bond_vectors.get(key)
    r = geom.bond_distances.get(key)
    if terms is None or r is None or r.size == 0:
        return
    terms = np.asarray(terms, dtype=np.int64)
    r_batch = _values_as_batch(r)
    dr_batch = _values_as_batch(dr, trailing_ndim=1)
    sample_ids, term_ids = _valid_flat_terms(r_batch)
    if sample_ids.size == 0:
        return
    r_flat = r_batch[sample_ids, term_ids]
    dr_flat = dr_batch[sample_ids, term_ids]
    ia = terms[term_ids, 0]
    ib = terms[term_ids, 1]

    if mat is not None:
        force_grad = pot.force_grad(r_flat)
        if hasattr(force_grad, "tocoo"):
            coo = force_grad.tocoo()
            if coo.nnz > 0:
                row = np.asarray(coo.row, dtype=np.int64)
                col = np.asarray(coo.col, dtype=np.int64)
                dat = np.asarray(coo.data, dtype=np.float32)
                coeff = dat * (-1.0 / r_flat[row])
                vals = dr_flat[row] * coeff[:, None]
                param_cols = sl.start + col
                _batch_add_sparse_param_rows(
                    mat,
                    sample_ids[row],
                    ia[row],
                    param_cols,
                    vals,
                )
                _batch_add_sparse_param_rows(
                    mat,
                    sample_ids[row],
                    ib[row],
                    param_cols,
                    -vals,
                )
        else:
            B = np.asarray(force_grad, dtype=np.float32)
            coeff = B * (-1.0 / r_flat)[:, None]
            vals = dr_flat[:, :, None] * coeff[:, None, :]
            _batch_add_param_rows(mat, sl, sample_ids, ia, vals)
            _batch_add_param_rows(mat, sl, sample_ids, ib, -vals)

    if fvec is not None:
        F_scalar = np.asarray(pot.force(r_flat), dtype=np.float32).reshape(-1)
        vals = (-F_scalar / r_flat)[:, None] * dr_flat
        _batch_add_force_rows(fvec, sample_ids, ia, vals)
        _batch_add_force_rows(fvec, sample_ids, ib, -vals)


def _project_angle(
    mat: Optional[np.ndarray],
    fvec: Optional[np.ndarray],
    geom: FrameGeometry,
    key: InteractionKey,
    pot,
    sl: slice,
) -> None:
    terms = geom.angle_indices.get(key)
    theta_deg = geom.angle_values.get(key)
    if terms is None or theta_deg is None or theta_deg.size == 0:
        return
    terms = np.asarray(terms, dtype=np.int64)
    ia_all, ib_all, ic_all = terms[:, 0], terms[:, 1], terms[:, 2]
    pos, box_batch = _positions_box_as_batch(geom)
    theta_batch = _values_as_batch(theta_deg)
    d1 = _min_image_batch(pos[:, ia_all, :] - pos[:, ib_all, :], box_batch)
    d2 = _min_image_batch(pos[:, ic_all, :] - pos[:, ib_all, :], box_batch)

    rsq1 = np.einsum("btj,btj->bt", d1, d1)
    rsq2 = np.einsum("btj,btj->bt", d2, d2)
    r1 = np.sqrt(rsq1)
    r2 = np.sqrt(rsq2)
    valid = (r1 > 1.0e-12) & (r2 > 1.0e-12)
    sample_ids, term_ids = np.nonzero(valid)
    if sample_ids.size == 0:
        return
    sample_ids = sample_ids.astype(np.int64, copy=False)
    term_ids = term_ids.astype(np.int64, copy=False)
    d1_flat = d1[sample_ids, term_ids]
    d2_flat = d2[sample_ids, term_ids]
    rsq1_flat = rsq1[sample_ids, term_ids]
    rsq2_flat = rsq2[sample_ids, term_ids]
    r1_flat = r1[sample_ids, term_ids]
    r2_flat = r2[sample_ids, term_ids]
    theta_flat = theta_batch[sample_ids, term_ids]

    c = np.einsum("ij,ij->i", d1_flat, d2_flat) / (r1_flat * r2_flat)
    c = np.clip(c, -1.0, 1.0)
    s = np.sqrt(np.maximum(1.0 - c * c, 1.0e-8))
    invs = 1.0 / np.maximum(s, 1.0e-4)

    a11 = invs * c / rsq1_flat
    a12 = -invs / (r1_flat * r2_flat)
    a22 = invs * c / rsq2_flat

    f1 = a11[:, None] * d1_flat + a12[:, None] * d2_flat
    f3 = a22[:, None] * d2_flat + a12[:, None] * d1_flat

    ia = ia_all[term_ids]
    ib = ib_all[term_ids]
    ic = ic_all[term_ids]
    if mat is not None:
        force_grad = pot.force_grad(theta_flat)
        if hasattr(force_grad, "tocoo"):
            coo = force_grad.tocoo()
            if coo.nnz > 0:
                row = np.asarray(coo.row, dtype=np.int64)
                col = np.asarray(coo.col, dtype=np.int64)
                dat = np.asarray(coo.data, dtype=np.float32) * np.float32(RAD2DEG)
                vals_a = f1[row] * dat[:, None]
                vals_c = f3[row] * dat[:, None]
                vals_b = -(vals_a + vals_c)
                param_cols = sl.start + col
                _batch_add_sparse_param_rows(mat, sample_ids[row], ia[row], param_cols, vals_a)
                _batch_add_sparse_param_rows(
                    mat,
                    sample_ids[row],
                    ib[row],
                    param_cols,
                    vals_b,
                )
                _batch_add_sparse_param_rows(
                    mat,
                    sample_ids[row],
                    ic[row],
                    param_cols,
                    vals_c,
                )
        else:
            B = np.asarray(force_grad, dtype=np.float32) * np.float32(RAD2DEG)
            vals_a = f1[:, :, None] * B[:, None, :]
            vals_c = f3[:, :, None] * B[:, None, :]
            vals_b = -(vals_a + vals_c)
            _batch_add_param_rows(mat, sl, sample_ids, ia, vals_a)
            _batch_add_param_rows(mat, sl, sample_ids, ib, vals_b)
            _batch_add_param_rows(mat, sl, sample_ids, ic, vals_c)

    if fvec is not None:
        F_scalar = np.asarray(pot.force(theta_flat), dtype=np.float32).reshape(-1) * np.float32(RAD2DEG)
        w_a = F_scalar[:, None] * f1
        w_c = F_scalar[:, None] * f3
        _batch_add_force_rows(fvec, sample_ids, ia, w_a)
        _batch_add_force_rows(fvec, sample_ids, ib, -(w_a + w_c))
        _batch_add_force_rows(fvec, sample_ids, ic, w_c)


def _project_dihedral(
    mat: Optional[np.ndarray],
    fvec: Optional[np.ndarray],
    geom: FrameGeometry,
    key: InteractionKey,
    pot,
    sl: slice,
) -> None:
    terms = geom.dihedral_indices.get(key)
    phi_deg = geom.dihedral_values.get(key)
    if terms is None or phi_deg is None or phi_deg.size == 0:
        return
    terms = np.asarray(terms, dtype=np.int64)
    i1_all, i2_all, i3_all, i4_all = terms[:, 0], terms[:, 1], terms[:, 2], terms[:, 3]
    pos, box_batch = _positions_box_as_batch(geom)
    phi_batch = _values_as_batch(phi_deg)

    r_ij = _min_image_batch(pos[:, i2_all, :] - pos[:, i1_all, :], box_batch)
    r_kj = _min_image_batch(pos[:, i2_all, :] - pos[:, i3_all, :], box_batch)
    r_kl = _min_image_batch(pos[:, i4_all, :] - pos[:, i3_all, :], box_batch)

    r_mj = np.cross(r_ij, r_kj)
    r_nk = np.cross(r_kj, r_kl)

    l2_kj = np.einsum("btj,btj->bt", r_kj, r_kj)
    r_mj2 = np.einsum("btj,btj->bt", r_mj, r_mj)
    r_nk2 = np.einsum("btj,btj->bt", r_nk, r_nk)

    ok = (
        np.isfinite(phi_batch)
        & (l2_kj > 1.0e-12)
        & (r_mj2 > 1.0e-12)
        & (r_nk2 > 1.0e-12)
    )
    sample_ids, term_ids = np.nonzero(ok)
    if sample_ids.size == 0:
        return
    sample_ids = sample_ids.astype(np.int64, copy=False)
    term_ids = term_ids.astype(np.int64, copy=False)
    phi_flat = phi_batch[sample_ids, term_ids]
    r_ij_flat = r_ij[sample_ids, term_ids]
    r_kj_flat = r_kj[sample_ids, term_ids]
    r_kl_flat = r_kl[sample_ids, term_ids]
    r_mj_flat = r_mj[sample_ids, term_ids]
    r_nk_flat = r_nk[sample_ids, term_ids]
    l2_kj_flat = l2_kj[sample_ids, term_ids]
    r_mj2_flat = r_mj2[sample_ids, term_ids]
    r_nk2_flat = r_nk2[sample_ids, term_ids]
    l_kj = np.sqrt(l2_kj_flat)

    f1 = r_mj_flat * (l_kj / r_mj2_flat)[:, None]
    f4 = r_nk_flat * (-l_kj / r_nk2_flat)[:, None]

    dot_ij_kj = np.einsum("ij,ij->i", r_ij_flat, r_kj_flat)
    dot_kl_kj = np.einsum("ij,ij->i", r_kl_flat, r_kj_flat)

    f2 = (
        f1 * ((dot_ij_kj / l2_kj_flat - 1.0)[:, None])
        + f4 * ((-dot_kl_kj / l2_kj_flat)[:, None])
    )
    f3 = (
        f4 * ((dot_kl_kj / l2_kj_flat - 1.0)[:, None])
        + f1 * ((-dot_ij_kj / l2_kj_flat)[:, None])
    )

    i1 = i1_all[term_ids]
    i2 = i2_all[term_ids]
    i3 = i3_all[term_ids]
    i4 = i4_all[term_ids]
    if mat is not None:
        force_grad = pot.force_grad(phi_flat)
        if hasattr(force_grad, "tocoo"):
            coo = force_grad.tocoo()
            if coo.nnz > 0:
                row = np.asarray(coo.row, dtype=np.int64)
                col = np.asarray(coo.col, dtype=np.int64)
                dat = np.asarray(coo.data, dtype=np.float32) * np.float32(RAD2DEG)
                param_cols = sl.start + col
                _batch_add_sparse_param_rows(
                    mat,
                    sample_ids[row],
                    i1[row],
                    param_cols,
                    f1[row] * dat[:, None],
                )
                _batch_add_sparse_param_rows(
                    mat,
                    sample_ids[row],
                    i2[row],
                    param_cols,
                    f2[row] * dat[:, None],
                )
                _batch_add_sparse_param_rows(
                    mat,
                    sample_ids[row],
                    i3[row],
                    param_cols,
                    f3[row] * dat[:, None],
                )
                _batch_add_sparse_param_rows(
                    mat,
                    sample_ids[row],
                    i4[row],
                    param_cols,
                    f4[row] * dat[:, None],
                )
        else:
            B = np.asarray(force_grad, dtype=np.float32) * np.float32(RAD2DEG)
            vals_1 = f1[:, :, None] * B[:, None, :]
            vals_2 = f2[:, :, None] * B[:, None, :]
            vals_3 = f3[:, :, None] * B[:, None, :]
            vals_4 = f4[:, :, None] * B[:, None, :]
            _batch_add_param_rows(mat, sl, sample_ids, i1, vals_1)
            _batch_add_param_rows(mat, sl, sample_ids, i2, vals_2)
            _batch_add_param_rows(mat, sl, sample_ids, i3, vals_3)
            _batch_add_param_rows(mat, sl, sample_ids, i4, vals_4)

    if fvec is not None:
        F_scalar = np.asarray(pot.force(phi_flat), dtype=np.float32).reshape(-1) * np.float32(RAD2DEG)
        _batch_add_force_rows(fvec, sample_ids, i1, F_scalar[:, None] * f1)
        _batch_add_force_rows(fvec, sample_ids, i2, F_scalar[:, None] * f2)
        _batch_add_force_rows(fvec, sample_ids, i3, F_scalar[:, None] * f3)
        _batch_add_force_rows(fvec, sample_ids, i4, F_scalar[:, None] * f4)
