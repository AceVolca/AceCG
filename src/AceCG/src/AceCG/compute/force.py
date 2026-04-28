"""Per-frame force-side observable kernel."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from MDAnalysis.lib.distances import minimize_vectors

from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey
from .frame_geometry import FrameGeometry


def _min_image(vec: np.ndarray, box: np.ndarray) -> np.ndarray:
    if box is None or box.size == 0:
        return vec
    return minimize_vectors(vec, box=box)


def _accumulate_rows(M: np.ndarray, rows: np.ndarray, values: np.ndarray) -> None:
    """Accumulate value matrix into selected rows of M."""
    if values.size == 0:
        return
    np.add.at(
        M,
        np.asarray(rows, dtype=np.int32).ravel(),
        np.asarray(values, dtype=np.float32),
    )


def _dense_force_grad(pot, values: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    """Return a dense force Jacobian matrix for the given scalar coordinates."""
    grad = pot.force_grad(values)
    if hasattr(grad, "toarray"):
        grad = grad.toarray()
    dense = np.asarray(grad, dtype=np.float32)
    if scale != 1.0:
        dense = dense * np.float32(scale)
    return dense


def _slice_observed_force_rows(
    values: np.ndarray,
    real_site_indices: Optional[np.ndarray],
) -> np.ndarray:
    """Return flattened force rows restricted to observed real sites."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if real_site_indices is None:
        return arr
    atoms = np.asarray(real_site_indices, dtype=np.int32).reshape(-1)
    rows = (atoms[:, None] * 3 + np.arange(3, dtype=np.int32)[None, :]).reshape(-1)
    if arr.size == rows.size:
        return arr
    if rows.size and rows[-1] >= arr.size:
        raise ValueError(
            "reference_force does not match either the observed or full force-row shape."
        )
    return arr[rows]


def _build_fm_stats(
    jacobian: np.ndarray,
    model_force: np.ndarray,
    reference_force: np.ndarray,
    *,
    frame_weight: float,
) -> Dict[str, Any]:
    """Build one-frame FM sufficient statistics from observed rows."""
    J_obs = np.asarray(jacobian, dtype=np.float32)
    f_obs = np.asarray(model_force, dtype=np.float32).reshape(-1)
    y_obs = np.asarray(reference_force, dtype=np.float32).reshape(-1)
    if f_obs.shape != y_obs.shape:
        raise ValueError(
            f"Observed model/reference force shapes must match, got {f_obs.shape} and {y_obs.shape}."
        )
    wi = np.float32(frame_weight)
    return {
        "JtJ": wi * (J_obs.T @ J_obs),
        "Jtf": wi * (J_obs.T @ f_obs),
        "Jty": wi * (J_obs.T @ y_obs),
        "ftf": wi * float(np.dot(f_obs, f_obs)),
        "fTy": wi * float(np.dot(f_obs, y_obs)),
        "yty": wi * float(np.dot(y_obs, y_obs)),
        "n_force_rows": int(f_obs.size),
        "n_frames": 1,
        "weight_sum": float(wi),
        "n_atoms_obs": int(f_obs.size // 3),
    }


RAD2DEG = 180.0 / np.pi


def force(
    frame_geometry: FrameGeometry,
    forcefield: Forcefield,
    *,
    return_value: bool = False,
    return_grad: bool = False,
    return_hessian: bool = False,
    reference_force: Optional[np.ndarray] = None,
    frame_weight: float = 1.0,
    return_fm_stats: bool = False,
) -> Dict[str, Any]:
    """Per-frame force-side observables."""
    if not (return_value or return_grad or return_hessian or return_fm_stats):
        return {}

    n_atoms = frame_geometry.n_atoms
    rsi = frame_geometry.real_site_indices

    interaction_mask = forcefield.key_mask
    param_mask = forcefield.param_mask

    param_blocks = forcefield.param_blocks()
    n_params = forcefield.n_params()

    need_jacobian = return_grad or return_hessian or return_fm_stats
    mat = np.zeros((n_atoms * 3, n_params), dtype=np.float32) if need_jacobian else None
    fvec = (
        np.zeros(n_atoms * 3, dtype=np.float32)
        if (return_value or return_fm_stats)
        else None
    )

    result: Dict[str, Any] = {}
    box = frame_geometry.box

    for key, pot, sl in param_blocks:
        if interaction_mask is not None and not interaction_mask.get(key, True):
            continue

        if key.style == "pair":
            _project_pair(mat, fvec, frame_geometry, key, pot, sl, box)
        elif key.style == "bond":
            _project_bond(mat, fvec, frame_geometry, key, pot, sl, box)
        elif key.style == "angle":
            _project_angle(mat, fvec, frame_geometry, key, pot, sl, box)
        elif key.style == "dihedral":
            _project_dihedral(mat, fvec, frame_geometry, key, pot, sl, box)

    if rsi is not None:
        rows_3d = np.concatenate([rsi * 3, rsi * 3 + 1, rsi * 3 + 2]).astype(
            np.int32, copy=False
        )
        rows_3d.sort()
        mat_obs = mat[rows_3d, :] if mat is not None else None
        fvec_obs = fvec[rows_3d] if fvec is not None else None
    else:
        mat_obs = mat
        fvec_obs = fvec

    if param_mask is not None:
        pmask = np.asarray(param_mask, dtype=bool)
        if mat_obs is not None and not np.all(pmask):
            mat_obs[:, ~pmask] = 0.0

    if return_grad:
        result["force_grad"] = mat_obs

    if return_value:
        result["force"] = (
            fvec_obs if fvec_obs is not None else np.zeros(0, dtype=np.float32)
        )

    if return_hessian:
        result["force_hessian"] = (
            mat_obs.T @ mat_obs
            if mat_obs is not None
            else np.zeros((n_params, n_params), dtype=np.float32)
        )

    if return_fm_stats:
        if mat_obs is None or fvec_obs is None:
            raise RuntimeError("FM statistics require both force values and Jacobians.")
        if reference_force is None:
            raise ValueError("reference_force is required when return_fm_stats=True.")
        reference_force_obs = _slice_observed_force_rows(reference_force, rsi)
        result["fm_stats"] = _build_fm_stats(
            mat_obs,
            fvec_obs,
            reference_force_obs,
            frame_weight=frame_weight,
        )

    return result


def _project_pair(
    mat: Optional[np.ndarray],
    fvec: Optional[np.ndarray],
    geom: FrameGeometry,
    key: InteractionKey,
    pot,
    sl: slice,
    box: np.ndarray,
) -> None:
    idx_pair = geom.pair_indices.get(key)
    dr = geom.pair_vectors.get(key)
    r = geom.pair_distances.get(key)
    if idx_pair is None or r is None or r.size == 0:
        return
    a_idx, b_idx = idx_pair

    nz = r > 1e-12
    if not np.any(nz):
        return
    a_idx, b_idx, dr, r = a_idx[nz], b_idx[nz], dr[nz], r[nz]

    if mat is not None:
        force_grad = pot.force_grad(r)
        if hasattr(force_grad, "tocoo"):
            coo = force_grad.tocoo()
            if coo.nnz > 0:
                row = np.asarray(coo.row, dtype=np.int32)
                col = np.asarray(coo.col, dtype=np.int32)
                dat = np.asarray(coo.data, dtype=np.float32)
                invr = -1.0 / r[row]
                ncol_mat = mat.shape[1]
                w_all = (dat * invr)[:, None] * dr[row]
                d_off = np.arange(3, dtype=np.int32)[None, :]
                lin_a = (
                    (a_idx[row][:, None] * 3 + d_off) * ncol_mat
                    + (sl.start + col)[:, None]
                )
                lin_b = (
                    (b_idx[row][:, None] * 3 + d_off) * ncol_mat
                    + (sl.start + col)[:, None]
                )
                mat_flat = mat.reshape(-1)
                np.add.at(mat_flat, lin_a.ravel(), w_all.ravel())
                np.add.at(mat_flat, lin_b.ravel(), -w_all.ravel())
        else:
            B = np.asarray(force_grad, dtype=np.float32)
            coeff = B * (-1.0 / r)[:, None]
            vals_3d = dr[:, :, None] * coeff[:, None, :]
            d_off = np.arange(3, dtype=np.int32)[None, :]
            rows_a = (a_idx[:, None] * 3 + d_off).ravel()
            rows_b = (b_idx[:, None] * 3 + d_off).ravel()
            _accumulate_rows(mat[:, sl], rows_a, vals_3d.reshape(-1, coeff.shape[1]))
            _accumulate_rows(mat[:, sl], rows_b, -vals_3d.reshape(-1, coeff.shape[1]))

    if fvec is not None:
        F_scalar = np.asarray(pot.force(r), dtype=np.float32).ravel()
        w_all = (-F_scalar / r)[:, None] * dr
        fvec_2d = fvec.reshape(-1, 3)
        np.add.at(fvec_2d, a_idx, w_all)
        np.add.at(fvec_2d, b_idx, -w_all)


def _project_bond(
    mat: Optional[np.ndarray],
    fvec: Optional[np.ndarray],
    geom: FrameGeometry,
    key: InteractionKey,
    pot,
    sl: slice,
    box: np.ndarray,
) -> None:
    terms = geom.bond_indices.get(key)
    dr = geom.bond_vectors.get(key)
    r = geom.bond_distances.get(key)
    if terms is None or r is None or r.size == 0:
        return
    ia, ib = terms[:, 0], terms[:, 1]

    nz = r > 1e-12
    if not np.any(nz):
        return
    ia, ib, dr, r = ia[nz], ib[nz], dr[nz], r[nz]

    if mat is not None:
        B = _dense_force_grad(pot, r)
        coeff = B * (-1.0 / r)[:, None]
        vals_3d = dr[:, :, None] * coeff[:, None, :]
        d_off = np.arange(3, dtype=np.int32)[None, :]
        rows_a = (ia[:, None] * 3 + d_off).ravel()
        rows_b = (ib[:, None] * 3 + d_off).ravel()
        _accumulate_rows(mat[:, sl], rows_a, vals_3d.reshape(-1, coeff.shape[1]))
        _accumulate_rows(mat[:, sl], rows_b, -vals_3d.reshape(-1, coeff.shape[1]))

    if fvec is not None:
        F_scalar = np.asarray(pot.force(r), dtype=np.float32).ravel()
        w_all = (-F_scalar / r)[:, None] * dr
        fvec_2d = fvec.reshape(-1, 3)
        np.add.at(fvec_2d, ia, w_all)
        np.add.at(fvec_2d, ib, -w_all)


def _project_angle(
    mat: Optional[np.ndarray],
    fvec: Optional[np.ndarray],
    geom: FrameGeometry,
    key: InteractionKey,
    pot,
    sl: slice,
    box: np.ndarray,
) -> None:
    terms = geom.angle_indices.get(key)
    theta_deg = geom.angle_values.get(key)
    if terms is None or theta_deg is None or theta_deg.size == 0:
        return
    ia, ib, ic = terms[:, 0], terms[:, 1], terms[:, 2]

    pos = geom.positions
    d1 = _min_image(pos[ia] - pos[ib], box)
    d2 = _min_image(pos[ic] - pos[ib], box)

    rsq1 = np.einsum("ij,ij->i", d1, d1)
    rsq2 = np.einsum("ij,ij->i", d2, d2)
    r1 = np.sqrt(rsq1)
    r2 = np.sqrt(rsq2)
    valid = (r1 > 1e-12) & (r2 > 1e-12)
    if not np.any(valid):
        return

    ia, ib, ic = ia[valid], ib[valid], ic[valid]
    d1, d2 = d1[valid], d2[valid]
    rsq1, rsq2 = rsq1[valid], rsq2[valid]
    r1, r2 = r1[valid], r2[valid]
    theta_deg = theta_deg[valid]

    c = np.einsum("ij,ij->i", d1, d2) / (r1 * r2)
    c = np.clip(c, -1.0, 1.0)
    s = np.sqrt(np.maximum(1.0 - c * c, 1e-8))
    invs = 1.0 / np.maximum(s, 1e-4)

    a11 = invs * c / rsq1
    a12 = -invs / (r1 * r2)
    a22 = invs * c / rsq2

    f1 = a11[:, None] * d1 + a12[:, None] * d2
    f3 = a22[:, None] * d2 + a12[:, None] * d1

    if mat is not None:
        B = _dense_force_grad(pot, theta_deg, scale=RAD2DEG)
        ncv = B.shape[1]
        vals_a = f1[:, :, None] * B[:, None, :]
        vals_c = f3[:, :, None] * B[:, None, :]
        vals_b = -(vals_a + vals_c)
        d_off = np.arange(3, dtype=np.int32)[None, :]
        rows_a = (ia[:, None] * 3 + d_off).ravel()
        rows_b = (ib[:, None] * 3 + d_off).ravel()
        rows_c = (ic[:, None] * 3 + d_off).ravel()
        _accumulate_rows(mat[:, sl], rows_a, vals_a.reshape(-1, ncv))
        _accumulate_rows(mat[:, sl], rows_b, vals_b.reshape(-1, ncv))
        _accumulate_rows(mat[:, sl], rows_c, vals_c.reshape(-1, ncv))

    if fvec is not None:
        F_scalar = np.asarray(pot.force(theta_deg), dtype=np.float32).ravel() * np.float32(RAD2DEG)
        w_a = F_scalar[:, None] * f1
        w_c = F_scalar[:, None] * f3
        fvec_2d = fvec.reshape(-1, 3)
        np.add.at(fvec_2d, ia, w_a)
        np.add.at(fvec_2d, ib, -(w_a + w_c))
        np.add.at(fvec_2d, ic, w_c)


def _project_dihedral(
    mat: Optional[np.ndarray],
    fvec: Optional[np.ndarray],
    geom: FrameGeometry,
    key: InteractionKey,
    pot,
    sl: slice,
    box: np.ndarray,
) -> None:
    terms = geom.dihedral_indices.get(key)
    phi_deg = geom.dihedral_values.get(key)
    if terms is None or phi_deg is None or phi_deg.size == 0:
        return

    pos = geom.positions
    valid = np.isfinite(phi_deg)
    if not np.any(valid):
        return

    t = terms[valid]
    phi_v = phi_deg[valid]
    i1, i2, i3, i4 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]

    r_ij = _min_image(pos[i2] - pos[i1], box)
    r_kj = _min_image(pos[i2] - pos[i3], box)
    r_kl = _min_image(pos[i4] - pos[i3], box)

    r_mj = np.cross(r_ij, r_kj)
    r_nk = np.cross(r_kj, r_kl)

    l2_kj = np.einsum("ij,ij->i", r_kj, r_kj)
    r_mj2 = np.einsum("ij,ij->i", r_mj, r_mj)
    r_nk2 = np.einsum("ij,ij->i", r_nk, r_nk)

    ok = (l2_kj > 1e-12) & (r_mj2 > 1e-12) & (r_nk2 > 1e-12)
    if not np.any(ok):
        return

    i1, i2, i3, i4 = i1[ok], i2[ok], i3[ok], i4[ok]
    phi_ok = phi_v[ok]
    r_ij = r_ij[ok]
    r_kj = r_kj[ok]
    r_kl = r_kl[ok]
    r_mj = r_mj[ok]
    r_nk = r_nk[ok]
    l2_kj = l2_kj[ok]
    r_mj2 = r_mj2[ok]
    r_nk2 = r_nk2[ok]
    l_kj = np.sqrt(l2_kj)

    f1 = r_mj * (l_kj / r_mj2)[:, None]
    f4 = r_nk * (-l_kj / r_nk2)[:, None]

    dot_ij_kj = np.einsum("ij,ij->i", r_ij, r_kj)
    dot_kl_kj = np.einsum("ij,ij->i", r_kl, r_kj)

    f2 = (
        f1 * ((dot_ij_kj / l2_kj - 1.0)[:, None])
        + f4 * ((-dot_kl_kj / l2_kj)[:, None])
    )
    f3 = (
        f4 * ((dot_kl_kj / l2_kj - 1.0)[:, None])
        + f1 * ((-dot_ij_kj / l2_kj)[:, None])
    )

    if mat is not None:
        B = _dense_force_grad(pot, phi_ok, scale=RAD2DEG)
        ncv = B.shape[1]
        vals_1 = f1[:, :, None] * B[:, None, :]
        vals_2 = f2[:, :, None] * B[:, None, :]
        vals_3 = f3[:, :, None] * B[:, None, :]
        vals_4 = f4[:, :, None] * B[:, None, :]
        d_off = np.arange(3, dtype=np.int32)[None, :]
        rows_1 = (i1[:, None] * 3 + d_off).ravel()
        rows_2 = (i2[:, None] * 3 + d_off).ravel()
        rows_3 = (i3[:, None] * 3 + d_off).ravel()
        rows_4 = (i4[:, None] * 3 + d_off).ravel()
        _accumulate_rows(mat[:, sl], rows_1, vals_1.reshape(-1, ncv))
        _accumulate_rows(mat[:, sl], rows_2, vals_2.reshape(-1, ncv))
        _accumulate_rows(mat[:, sl], rows_3, vals_3.reshape(-1, ncv))
        _accumulate_rows(mat[:, sl], rows_4, vals_4.reshape(-1, ncv))

    if fvec is not None:
        F_scalar = np.asarray(pot.force(phi_ok), dtype=np.float32).ravel() * np.float32(RAD2DEG)
        fvec_2d = fvec.reshape(-1, 3)
        np.add.at(fvec_2d, i1, F_scalar[:, None] * f1)
        np.add.at(fvec_2d, i2, F_scalar[:, None] * f2)
        np.add.at(fvec_2d, i3, F_scalar[:, None] * f3)
        np.add.at(fvec_2d, i4, F_scalar[:, None] * f4)
