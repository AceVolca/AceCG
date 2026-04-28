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

from typing import Any, Dict

import numpy as np

from ..topology.forcefield import Forcefield
from .frame_geometry import FrameGeometry


def energy(
    frame_geometry: FrameGeometry,
    forcefield: Forcefield,
    *,
    return_value: bool = False,
    return_grad: bool = False,
    return_hessian: bool = False,
    return_grad_outer: bool = False,
) -> Dict[str, Any]:
    """Per-frame energy-side observables.

    Masks are read from ``forcefield.key_mask`` (L1) and
    ``forcefield.param_mask`` (L2).

    Parameters
    ----------
    frame_geometry : FrameGeometry from ``compute_frame_geometry()``.
    forcefield : Forcefield container.
    return_value, return_grad, return_hessian, return_grad_outer :
        which derivative levels to compute.

    Returns
    -------
    dict with keys ``'energy'``, ``'energy_grad'``, ``'energy_hessian'``,
    ``'energy_grad_outer'`` as requested.  Unrequested keys are absent.
    """
    if not (return_value or return_grad or return_hessian or return_grad_outer):
        return {}

    result: Dict[str, Any] = {}

    # Read masks from forcefield
    # NOTICE TO AGENT: ALL callers must pass `Forcefield` objects.
    interaction_mask = forcefield.key_mask
    param_mask = forcefield.param_mask

    param_blocks = forcefield.param_blocks()
    n_params = forcefield.n_params()

    if return_value:
        result["energy"] = 0.0
    if return_grad or return_grad_outer:
        result["energy_grad"] = np.zeros(n_params, dtype=np.float64)
    if return_hessian:
        result["energy_hessian"] = np.zeros((n_params, n_params), dtype=np.float64)

    for key, pot, sl in param_blocks:
        if interaction_mask is not None and not interaction_mask.get(key, True):
            continue

        # Get distances/values for this interaction
        if key.style == "pair":
            r = frame_geometry.pair_distances.get(key)
        elif key.style == "bond":
            r = frame_geometry.bond_distances.get(key)
        elif key.style == "angle":
            r = frame_geometry.angle_values.get(key)
        elif key.style == "dihedral":
            r = frame_geometry.dihedral_values.get(key)
        else:
            continue

        if r is None or r.size == 0:
            continue

        # Filter out NaN (e.g. degenerate dihedrals)
        valid = np.isfinite(r)
        if not np.any(valid): continue
        rv = r[valid]
        # Apply cutoff here if the potential has a cutoff attribute
        if hasattr(pot, "cutoff") and pot.cutoff is not None: rv = rv[rv <= pot.cutoff]
        if rv.size == 0: continue
        
        if return_value:
            result["energy"] += float(np.sum(pot.value(rv)))

        if return_grad or return_grad_outer:
            result["energy_grad"][sl] += pot.energy_grad_sum(rv)

        if return_hessian:
            n_pot = pot.n_params()
            if np.all(np.asarray(pot.is_param_linear(), dtype=bool).reshape(-1)):
                continue
            # True parameter Hessian via pot.d2param_names() method dispatch
            d2names = pot.d2param_names()
            for j in range(n_pot):
                for k in range(j, n_pot):
                    method_name = d2names[j][k]
                    val = float(np.sum(getattr(pot, method_name)(rv)))
                    result["energy_hessian"][sl.start + j, sl.start + k] += val
                    if j != k:
                        result["energy_hessian"][sl.start + k, sl.start + j] = \
                            result["energy_hessian"][sl.start + j, sl.start + k]

    # Apply param mask (before computing outer product so mask propagates)
    if param_mask is not None:
        mask = np.asarray(param_mask, dtype=bool)
        if (return_grad or return_grad_outer) and not np.all(mask):
            result["energy_grad"][~mask] = 0.0
        if return_hessian and not np.all(mask):
            result["energy_hessian"][~mask, :] = 0.0
            result["energy_hessian"][:, ~mask] = 0.0

    # Compute outer product from the FULL accumulated gradient vector.
    # Must be outer(g, g) — NOT block-diagonal — to capture cross-interaction
    # terms needed by CDREM covariance: Cov = <g g^T> - <g><g>^T.
    if return_grad_outer:
        g = result["energy_grad"]
        result["energy_grad_outer"] = np.outer(g, g)

    # If grad was only accumulated for grad_outer, remove from result
    if return_grad_outer and not return_grad:
        del result["energy_grad"]

    return result
