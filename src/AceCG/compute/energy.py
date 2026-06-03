"""Per-frame energy-side observable kernel.

Replaces ``energy_grad_by_frame`` for single-frame usage.  Returns a dict with
the requested derivative levels (value / grad / hessian / grad_outer).

Semantics (NB-1 resolution, U4):
- ``energy_hessian``: true parameter Hessian ``Σ_samples d²U/dθ_j dθ_k``,
  computed via ``pot.d2param_names()`` method dispatch.
- ``energy_grad_outer``: gradient outer product ``Σ_samples (dU/dθ_j)(dU/dθ_k)``,
  i.e. the per-frame contribution to the Fisher information matrix.
  Used by REM for ``⟨(dU/dλ_j)(dU/dλ_k)⟩``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from ..configs.energy_mask import (
    EnergyMaskBounds,
    parse_energy_mask_runtime,
    summarize_energy_mask_counts,
)
from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey
from .frame_geometry import FrameGeometry, _geometry_sample_shape


def _interaction_values(frame_geometry: FrameGeometry, key):
    if key.style == "pair":
        return frame_geometry.pair_distances.get(key)
    if key.style == "bond":
        return frame_geometry.bond_distances.get(key)
    if key.style == "angle":
        return frame_geometry.angle_values.get(key)
    if key.style == "dihedral":
        return frame_geometry.dihedral_values.get(key)
    return None


def _sum_last_axis(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr
    return arr.sum(axis=-1)


def _coerce_term_shape(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != shape:
        try:
            arr = np.broadcast_to(arr, shape)
        except ValueError as exc:
            raise ValueError(f"{label} returned shape {arr.shape}, expected {shape}") from exc
    return np.asarray(arr, dtype=np.float64)


def _coerce_channel_shape(
    values: Any,
    term_shape: tuple[int, ...],
    n_channels: int,
    label: str,
) -> np.ndarray:
    expected = term_shape + (int(n_channels),)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != expected:
        try:
            arr = np.broadcast_to(arr, expected)
        except ValueError as exc:
            raise ValueError(f"{label} returned shape {arr.shape}, expected {expected}") from exc
    return np.asarray(arr, dtype=np.float64)


def _coerce_reduced_channel_shape(
    values: Any,
    sample_shape: tuple[int, ...],
    n_channels: int,
    label: str,
) -> np.ndarray:
    expected = sample_shape + (int(n_channels),)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != expected:
        try:
            arr = np.broadcast_to(arr, expected)
        except ValueError as exc:
            raise ValueError(f"{label} returned shape {arr.shape}, expected {expected}") from exc
    return np.asarray(arr, dtype=np.float64)


def _normalize_coordinate_mask(
    coordinate_mask: Any,
) -> dict[InteractionKey, EnergyMaskBounds]:
    """Return canonical coordinate-mask bounds for energy evaluation."""
    if coordinate_mask is None:
        return {}
    return parse_energy_mask_runtime(coordinate_mask)


def _mask_diagnostics_state() -> dict[str, Any]:
    """Return a mutable diagnostics accumulator."""
    return {"active": 0, "total": 0, "by_key": {}}


def _accumulate_mask_diagnostics(
    diagnostics: dict[str, Any],
    *,
    key: InteractionKey,
    active: np.ndarray,
    total: np.ndarray,
) -> None:
    """Accumulate coordinate-mask active/total counts for one interaction."""
    active_count = int(np.count_nonzero(active))
    total_count = int(np.count_nonzero(total))
    diagnostics["active"] = int(diagnostics["active"]) + active_count
    diagnostics["total"] = int(diagnostics["total"]) + total_count
    label = key.label()
    by_key = diagnostics["by_key"]
    if label not in by_key:
        by_key[label] = {"active": 0, "total": 0}
    by_key[label]["active"] = int(by_key[label]["active"]) + active_count
    by_key[label]["total"] = int(by_key[label]["total"]) + total_count


def _sum_energy_grad(
    pot: Any,
    r_arr: np.ndarray,
    active: np.ndarray,
    sample_shape: tuple[int, ...],
    *,
    gauge_free: bool,
) -> np.ndarray:
    """Return summed energy-gradient channels for one potential block."""
    n_pot = pot.n_params()
    if sample_shape == () and r_arr.ndim == 1:
        rv = r_arr[active]
        if gauge_free:
            return np.asarray(
                pot.gauge_free_energy_grad_sum(rv),
                dtype=np.float64,
            ).reshape(n_pot)
        return np.asarray(
            pot.energy_grad_sum(rv),
            dtype=np.float64,
        ).reshape(n_pot)
    if gauge_free:
        return _coerce_reduced_channel_shape(
            pot.gauge_free_energy_grad_sum_by_sample(r_arr, active=active),
            sample_shape,
            n_pot,
            f"{type(pot).__name__}.gauge_free_energy_grad_sum_by_sample",
        )
    return _coerce_reduced_channel_shape(
        pot.energy_grad_sum_by_sample(r_arr, active=active),
        sample_shape,
        n_pot,
        f"{type(pot).__name__}.energy_grad_sum_by_sample",
    )


def energy(
    frame_geometry: FrameGeometry,
    forcefield: Forcefield,
    *,
    return_value: bool = False,
    return_grad: bool = False,
    return_hessian: bool = False,
    return_grad_outer: bool = False,
    return_gauge_free_energy_grad: bool = False,
    return_gauge_free_energy_grad_outer: bool = False,
    return_unmasked_energy_grad: bool = False,
    return_unmasked_energy_grad_outer: bool = False,
    return_unmasked_gauge_free_energy_grad: bool = False,
    return_unmasked_gauge_free_energy_grad_outer: bool = False,
    coordinate_mask: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Energy-side observables for a frame or same-frame coordinate batch.

    Masks are read from ``forcefield.key_mask`` (L1) and
    ``forcefield.param_mask`` (L2).

    Parameters
    ----------
    frame_geometry : FrameGeometry.
    forcefield : Forcefield container.
    return_value, return_grad, return_hessian, return_grad_outer,
    return_gauge_free_energy_grad, return_gauge_free_energy_grad_outer,
    return_unmasked_energy_grad, return_unmasked_energy_grad_outer,
    return_unmasked_gauge_free_energy_grad,
    return_unmasked_gauge_free_energy_grad_outer :
        which derivative levels to compute.
    coordinate_mask : mapping, optional
        Closed coordinate bounds keyed by interaction label. The mask gates
        REM/CDREM energy statistics by geometry coordinate before potential
        evaluation; unmasked auxiliary channels ignore these bounds.

    Returns
    -------
    dict with keys ``'energy'``, ``'energy_grad'``, ``'energy_hessian'``,
    ``'energy_grad_outer'``, ``'gauge_free_energy_grad'``, and
    ``'gauge_free_energy_grad_outer'`` as requested. For batched
    ``FrameGeometry`` inputs, output arrays preserve the geometry leading
    batch dimensions.
    """
    need_gauge_free_buffer = bool(
        return_gauge_free_energy_grad
        or return_gauge_free_energy_grad_outer
    )
    need_unmasked_physical_buffer = bool(
        return_unmasked_energy_grad
        or return_unmasked_energy_grad_outer
    )
    need_unmasked_gauge_free_buffer = bool(
        return_unmasked_gauge_free_energy_grad
        or return_unmasked_gauge_free_energy_grad_outer
    )
    if not (
        return_value
        or return_grad
        or return_hessian
        or return_grad_outer
        or need_gauge_free_buffer
        or need_unmasked_physical_buffer
        or need_unmasked_gauge_free_buffer
    ):
        return {}

    result: Dict[str, Any] = {}
    coordinate_bounds = _normalize_coordinate_mask(coordinate_mask)
    mask_diagnostics = _mask_diagnostics_state() if coordinate_bounds else None

    interaction_mask = forcefield.key_mask
    param_mask = forcefield.param_mask

    param_blocks = forcefield.param_blocks()
    n_params = forcefield.n_params()
    sample_shape = _geometry_sample_shape(frame_geometry)

    if return_value:
        result["energy"] = np.zeros(sample_shape, dtype=np.float64)
    if return_grad or return_grad_outer:
        result["energy_grad"] = np.zeros(sample_shape + (n_params,), dtype=np.float64)
    gauge_free_energy_grad = (
        np.zeros(sample_shape + (n_params,), dtype=np.float64)
        if need_gauge_free_buffer
        else None
    )
    unmasked_energy_grad = (
        np.zeros(sample_shape + (n_params,), dtype=np.float64)
        if need_unmasked_physical_buffer
        else None
    )
    unmasked_gauge_free_energy_grad = (
        np.zeros(sample_shape + (n_params,), dtype=np.float64)
        if need_unmasked_gauge_free_buffer
        else None
    )
    if return_hessian:
        result["energy_hessian"] = np.zeros(
            sample_shape + (n_params, n_params),
            dtype=np.float64,
        )

    for key, pot, sl in param_blocks:
        if interaction_mask is not None and not interaction_mask.get(key, True):
            continue

        r = _interaction_values(frame_geometry, key)

        if r is None or r.size == 0:
            continue

        r_arr = np.asarray(r, dtype=np.float64)
        active = np.isfinite(r_arr)
        if hasattr(pot, "cutoff") and pot.cutoff is not None:
            active &= r_arr <= pot.cutoff
        # Keep the pre-coordinate-mask selector so hybrid/unmasked optimizer
        # modes can request an auxiliary full-gradient channel. The primary
        # ``active`` selector below remains the masked REM/CDREM statistic.
        unmasked_active = active.copy()
        bounds = coordinate_bounds.get(key)
        if bounds is not None:
            active &= bounds.active(r_arr)
            if mask_diagnostics is not None:
                _accumulate_mask_diagnostics(
                    mask_diagnostics,
                    key=key,
                    active=active,
                    total=unmasked_active,
                )
        eval_r = np.where(np.isfinite(r_arr), r_arr, 0.0)
        
        if return_value:
            values = _coerce_term_shape(
                pot.value(eval_r),
                r_arr.shape,
                f"{type(pot).__name__}.value",
            )
            result["energy"] += _sum_last_axis(np.where(active, values, 0.0))

        if return_grad or return_grad_outer:
            result["energy_grad"][..., sl] += _sum_energy_grad(
                pot,
                r_arr,
                active,
                sample_shape,
                gauge_free=False,
            )

        if gauge_free_energy_grad is not None:
            gauge_free_energy_grad[..., sl] += _sum_energy_grad(
                pot,
                r_arr,
                active,
                sample_shape,
                gauge_free=True,
            )

        if unmasked_energy_grad is not None:
            unmasked_energy_grad[..., sl] += _sum_energy_grad(
                pot,
                r_arr,
                unmasked_active,
                sample_shape,
                gauge_free=False,
            )

        if unmasked_gauge_free_energy_grad is not None:
            unmasked_gauge_free_energy_grad[..., sl] += _sum_energy_grad(
                pot,
                r_arr,
                unmasked_active,
                sample_shape,
                gauge_free=True,
            )

        if return_hessian:
            if not np.any(active):
                continue
            n_pot = pot.n_params()
            if np.all(np.asarray(pot.is_param_linear(), dtype=bool).reshape(-1)):
                continue
            d2names = pot.d2param_names()
            for j in range(n_pot):
                for k in range(j, n_pot):
                    method_name = d2names[j][k]
                    values = _coerce_term_shape(
                        getattr(pot, method_name)(eval_r),
                        r_arr.shape,
                        f"{type(pot).__name__}.{method_name}",
                    )
                    val = _sum_last_axis(np.where(active, values, 0.0))
                    result["energy_hessian"][..., sl.start + j, sl.start + k] += val
                    if j != k:
                        result["energy_hessian"][..., sl.start + k, sl.start + j] += val

    # Apply parameter trainability masks after coordinate masks. This keeps
    # the two mask concepts independent while ensuring frozen parameters
    # never contribute to trainer-visible gradients or covariances.
    if param_mask is not None:
        mask = np.asarray(param_mask, dtype=bool)
        if (return_grad or return_grad_outer) and not np.all(mask):
            result["energy_grad"][..., ~mask] = 0.0
        if gauge_free_energy_grad is not None and not np.all(mask):
            gauge_free_energy_grad[..., ~mask] = 0.0
        if unmasked_energy_grad is not None and not np.all(mask):
            unmasked_energy_grad[..., ~mask] = 0.0
        if unmasked_gauge_free_energy_grad is not None and not np.all(mask):
            unmasked_gauge_free_energy_grad[..., ~mask] = 0.0
        if return_hessian and not np.all(mask):
            result["energy_hessian"][..., ~mask, :] = 0.0
            result["energy_hessian"][..., :, ~mask] = 0.0

    # Compute outer product from the FULL accumulated gradient vector.
    # Must be outer(g, g) — NOT block-diagonal — to capture cross-interaction
    # terms needed by CDREM covariance: Cov = <g g^T> - <g><g>^T.
    if return_grad_outer:
        g = result["energy_grad"]
        result["energy_grad_outer"] = np.einsum("...i,...j->...ij", g, g)

    if gauge_free_energy_grad is not None:
        if return_gauge_free_energy_grad:
            result["gauge_free_energy_grad"] = gauge_free_energy_grad
        if return_gauge_free_energy_grad_outer:
            result["gauge_free_energy_grad_outer"] = np.einsum(
                "...i,...j->...ij",
                gauge_free_energy_grad,
                gauge_free_energy_grad,
            )

    if unmasked_energy_grad is not None:
        if return_unmasked_energy_grad:
            result["unmasked_energy_grad"] = unmasked_energy_grad
        if return_unmasked_energy_grad_outer:
            result["unmasked_energy_grad_outer"] = np.einsum(
                "...i,...j->...ij",
                unmasked_energy_grad,
                unmasked_energy_grad,
            )

    if unmasked_gauge_free_energy_grad is not None:
        if return_unmasked_gauge_free_energy_grad:
            result["unmasked_gauge_free_energy_grad"] = unmasked_gauge_free_energy_grad
        if return_unmasked_gauge_free_energy_grad_outer:
            result["unmasked_gauge_free_energy_grad_outer"] = np.einsum(
                "...i,...j->...ij",
                unmasked_gauge_free_energy_grad,
                unmasked_gauge_free_energy_grad,
            )

    if return_grad_outer and not return_grad:
        del result["energy_grad"]

    if sample_shape == () and return_value:
        result["energy"] = float(result["energy"])
    if mask_diagnostics is not None:
        result["energy_mask_diagnostics"] = summarize_energy_mask_counts(
            mask_diagnostics["active"],
            mask_diagnostics["total"],
            mask_diagnostics["by_key"],
        )

    return result
