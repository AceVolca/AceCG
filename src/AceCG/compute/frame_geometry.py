"""Per-frame geometry extraction — shared across REM, FM, and CDFM paths.

Provides:
- ``FrameGeometry``: Immutable dataclass holding all per-frame geometric
  quantities (distances, vectors, angles, dihedrals) keyed by InteractionKey.
- ``compute_frame_geometry()``: Unified one-pass extraction that replaces
  the scattered legacy geometry helpers and interaction-cache builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from MDAnalysis.lib.distances import calc_angles, calc_dihedrals, minimize_vectors

from ..topology.types import InteractionKey
from ..topology.topology_array import TopologyArrays


@dataclass(frozen=True)
class FrameGeometry:
    """All geometry quantities for one frame or a same-frame coordinate batch.

    ``positions`` has shape ``(..., n_atoms, 3)``. All geometry value arrays
    preserve the same leading ``...`` dimensions and place the interaction-term
    axis last, for example ``bond_distances[key].shape == (..., n_bonds)`` and
    ``bond_vectors[key].shape == (..., n_bonds, 3)``.

    Depends on: positions, box, topology_arrays, interaction_mask.
    Does NOT depend on: forcefield parameters.
    """

    pair_distances: Dict[InteractionKey, np.ndarray]
    pair_indices: Dict[InteractionKey, Tuple[np.ndarray, np.ndarray]]
    pair_vectors: Dict[InteractionKey, np.ndarray]
    bond_distances: Dict[InteractionKey, np.ndarray]
    bond_vectors: Dict[InteractionKey, np.ndarray]
    bond_indices: Dict[InteractionKey, np.ndarray]
    angle_values: Dict[InteractionKey, np.ndarray]
    angle_indices: Dict[InteractionKey, np.ndarray]
    dihedral_values: Dict[InteractionKey, np.ndarray]
    dihedral_indices: Dict[InteractionKey, np.ndarray]
    positions: np.ndarray
    box: np.ndarray
    n_atoms: int
    real_site_indices: Optional[np.ndarray]


def _geometry_sample_shape(geom: "FrameGeometry") -> Tuple[int, ...]:
    """Leading sample dims of a frame geometry (``()`` for a single frame)."""
    positions = np.asarray(geom.positions)
    if positions.ndim < 2:
        return ()
    return tuple(positions.shape[:-2])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_box(box: np.ndarray, sample_shape: tuple[int, ...]) -> np.ndarray:
    bx = np.asarray(box, dtype=np.float32)
    if bx.ndim == 1:
        if bx.shape != (6,):
            raise ValueError(
                f"box must have shape (6,) or {sample_shape + (6,)}, got {bx.shape}"
            )
        if sample_shape:
            return np.broadcast_to(bx, sample_shape + (6,)).copy()
        return bx
    if bx.shape == sample_shape + (6,):
        return bx
    raise ValueError(f"box must have shape (6,) or {sample_shape + (6,)}, got {bx.shape}")


def _shared_orthorhombic_lengths(box_batch: np.ndarray) -> np.ndarray | None:
    boxes = np.asarray(box_batch, dtype=np.float32)
    if boxes.size == 0:
        return None
    if boxes.ndim == 1:
        if boxes.shape != (6,):
            return None
        first = boxes
    elif boxes.shape[-1] == 6:
        flat = boxes.reshape(-1, 6)
        first = flat[0]
        if not np.allclose(flat, first[None, :], rtol=0.0, atol=1.0e-5):
            return None
    else:
        return None
    lengths = np.asarray(first[:3], dtype=np.float32)
    angles = np.asarray(first[3:], dtype=np.float32)
    if not np.all(np.isfinite(lengths)) or not np.all(lengths > 0.0):
        return None
    if not np.allclose(angles, 90.0, rtol=0.0, atol=1.0e-4):
        return None
    return lengths


def _orthorhombic_min_image(vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return (
        vectors - lengths * np.rint(vectors / lengths)
    ).astype(np.float32, copy=False)


def _min_image_batch(vec: np.ndarray, box_batch: np.ndarray) -> np.ndarray:
    if box_batch is None or box_batch.size == 0:
        return vec
    vectors = np.asarray(vec, dtype=np.float32)
    if vectors.size == 0:
        return vectors
    boxes = np.asarray(box_batch, dtype=np.float32)
    lengths = _shared_orthorhombic_lengths(boxes)
    if lengths is not None:
        return _orthorhombic_min_image(vectors, lengths)
    if boxes.ndim == 1:
        flat = vectors.reshape(-1, 3)
        return minimize_vectors(flat, box=boxes).reshape(vectors.shape)

    sample_shape = tuple(boxes.shape[:-1])
    if vectors.shape[:len(sample_shape)] != sample_shape:
        raise ValueError(
            f"vector leading shape {vectors.shape[:len(sample_shape)]} does not "
            f"match box batch shape {sample_shape}"
        )
    term_shape = tuple(vectors.shape[len(sample_shape):-1])
    n_batches = int(np.prod(sample_shape, dtype=np.int64))
    per_batch = int(np.prod(term_shape, dtype=np.int64)) if term_shape else 1
    flat = vectors.reshape(n_batches * per_batch, 3)
    out = np.empty_like(flat)
    boxes_flat = boxes.reshape(n_batches, 6)
    if np.allclose(boxes_flat, boxes_flat[0]):
        return minimize_vectors(flat, box=boxes_flat[0]).reshape(vectors.shape)
    unique_boxes, inverse = np.unique(boxes_flat, axis=0, return_inverse=True)
    flat_groups = np.repeat(inverse, per_batch)
    for group_index, box in enumerate(unique_boxes):
        mask = flat_groups == group_index
        out[mask] = minimize_vectors(flat[mask], box=box)
    return out.reshape(vectors.shape)


def _calc_angles_batch(
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    c_pos: np.ndarray,
    box_batch: np.ndarray,
    *,
    backend: str,
) -> np.ndarray:
    sample_shape = tuple(a_pos.shape[:-2])
    n_terms = int(a_pos.shape[-2])
    if n_terms == 0:
        return np.empty(sample_shape + (0,), dtype=np.float32)
    boxes = np.asarray(box_batch, dtype=np.float32)
    lengths = _shared_orthorhombic_lengths(boxes)
    if lengths is not None:
        v1 = _orthorhombic_min_image(
            np.asarray(a_pos, dtype=np.float32) - np.asarray(b_pos, dtype=np.float32),
            lengths,
        )
        v2 = _orthorhombic_min_image(
            np.asarray(c_pos, dtype=np.float32) - np.asarray(b_pos, dtype=np.float32),
            lengths,
        )
        dot = np.einsum("...j,...j->...", v1, v2)
        n1 = np.sqrt(np.einsum("...j,...j->...", v1, v1))
        n2 = np.sqrt(np.einsum("...j,...j->...", v2, v2))
        denom = n1 * n2
        cos_theta = np.ones_like(dot, dtype=np.float32)
        np.divide(dot, denom, out=cos_theta, where=denom > 0.0)
        np.clip(cos_theta, -1.0, 1.0, out=cos_theta)
        return np.arccos(cos_theta).astype(np.float32, copy=False)
    a_flat = np.asarray(a_pos, dtype=np.float32).reshape(-1, 3)
    b_flat = np.asarray(b_pos, dtype=np.float32).reshape(-1, 3)
    c_flat = np.asarray(c_pos, dtype=np.float32).reshape(-1, 3)
    if boxes.size == 0:
        return calc_angles(a_flat, b_flat, c_flat, backend=backend).reshape(sample_shape + (n_terms,))
    if boxes.ndim == 1:
        return calc_angles(a_flat, b_flat, c_flat, box=boxes, backend=backend).reshape(sample_shape + (n_terms,))

    n_batches = int(np.prod(sample_shape, dtype=np.int64))
    boxes_flat = boxes.reshape(n_batches, 6)
    if np.allclose(boxes_flat, boxes_flat[0]):
        return calc_angles(a_flat, b_flat, c_flat, box=boxes_flat[0], backend=backend).reshape(sample_shape + (n_terms,))
    out = np.empty(n_batches * n_terms, dtype=np.float32)
    unique_boxes, inverse = np.unique(boxes_flat, axis=0, return_inverse=True)
    flat_groups = np.repeat(inverse, n_terms)
    for group_index, box in enumerate(unique_boxes):
        mask = flat_groups == group_index
        out[mask] = calc_angles(a_flat[mask], b_flat[mask], c_flat[mask], box=box, backend=backend)
    return out.reshape(sample_shape + (n_terms,))


def _calc_dihedrals_batch(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    box_batch: np.ndarray,
    *,
    backend: str,
) -> np.ndarray:
    sample_shape = tuple(p1.shape[:-2])
    n_terms = int(p1.shape[-2])
    if n_terms == 0:
        return np.empty(sample_shape + (0,), dtype=np.float32)
    p1_flat = np.asarray(p1, dtype=np.float32).reshape(-1, 3)
    p2_flat = np.asarray(p2, dtype=np.float32).reshape(-1, 3)
    p3_flat = np.asarray(p3, dtype=np.float32).reshape(-1, 3)
    p4_flat = np.asarray(p4, dtype=np.float32).reshape(-1, 3)
    boxes = np.asarray(box_batch, dtype=np.float32)
    if boxes.size == 0:
        return calc_dihedrals(p1_flat, p2_flat, p3_flat, p4_flat, backend=backend).reshape(sample_shape + (n_terms,))
    if boxes.ndim == 1:
        return calc_dihedrals(p1_flat, p2_flat, p3_flat, p4_flat, box=boxes, backend=backend).reshape(sample_shape + (n_terms,))

    n_batches = int(np.prod(sample_shape, dtype=np.int64))
    boxes_flat = boxes.reshape(n_batches, 6)
    if np.allclose(boxes_flat, boxes_flat[0]):
        return calc_dihedrals(p1_flat, p2_flat, p3_flat, p4_flat, box=boxes_flat[0], backend=backend).reshape(sample_shape + (n_terms,))
    out = np.empty(n_batches * n_terms, dtype=np.float32)
    unique_boxes, inverse = np.unique(boxes_flat, axis=0, return_inverse=True)
    flat_groups = np.repeat(inverse, n_terms)
    for group_index, box in enumerate(unique_boxes):
        mask = flat_groups == group_index
        out[mask] = calc_dihedrals(
            p1_flat[mask], p2_flat[mask], p3_flat[mask], p4_flat[mask], box=box, backend=backend
        )
    return out.reshape(sample_shape + (n_terms,))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_frame_geometry(
    positions: np.ndarray,
    box: np.ndarray,
    topology_arrays: TopologyArrays,
    *,
    interaction_mask: Optional[Dict[InteractionKey, bool]] = None,
    pair_cache: Optional[Dict[InteractionKey, Tuple[np.ndarray, np.ndarray]]] = None,
    neighbor_backend: str = "OpenMP",
) -> FrameGeometry:
    """Compute all geometry in one pass for a frame or same-frame batch.

    Parameters
    ----------
    positions : (..., n_atoms, 3) float32 array
    box : (6,) or (..., 6) float32 array (MDAnalysis dimensions format)
    topology_arrays : dict built by ``collect_topology_arrays()``
        Required keys: ``bonds``, ``bond_key_index``, ``keys_bondtypes``,
        ``angles``, ``angle_key_index``, ``keys_angletypes``,
        ``dihedrals``, ``dihedral_key_index``, ``keys_dihedraltypes``.
    interaction_mask : optional dict mapping InteractionKey → bool.
        If provided, only compute geometry for keys where the value is True.
        If not provided, all geometry component will be calculated.
    pair_cache : Neighbor list pair cache. Direct payload from 
    neighbor_backend : backend for MDAnalysis distance calls (default OpenMP).

    Returns
    -------
    FrameGeometry
        Frozen dataclass with all per-frame geometric quantities.
    """
    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim < 2 or pos.shape[-1] != 3:
        raise ValueError(
            "positions must have shape (..., n_atoms, 3), "
            f"got {pos.shape}"
        )
    sample_shape = tuple(pos.shape[:-2])
    n_atoms = int(pos.shape[-2])
    bx = _normalize_box(box, sample_shape)

    out = {"pair_distances": {}, "pair_indices": {}, "pair_vectors": {},
           "bond_distances": {}, "bond_vectors": {}, "bond_indices": {},
           "angle_values": {}, "angle_indices": {},
           "dihedral_values": {}, "dihedral_indices": {},
           }

    # --- Pair geometry (from pre-computed pair_cache) ---
    if pair_cache is not None:
        pair_terms = []
        for key, (a_idx, b_idx) in pair_cache.items():
            # Masking out interactions that are disabled by the interaction_mask
            if interaction_mask and not interaction_mask.get(key, False):
                continue

            a_idx = np.asarray(a_idx, dtype=np.int32)
            b_idx = np.asarray(b_idx, dtype=np.int32)
            out["pair_indices"][key] = (a_idx, b_idx)
            if a_idx.size == 0:
                out["pair_distances"][key] = np.empty(sample_shape + (0,), dtype=np.float32)
                out["pair_vectors"][key] = np.empty(sample_shape + (0, 3), dtype=np.float32)
                continue
            pair_terms.append((key, a_idx, b_idx, int(a_idx.size)))

        if pair_terms:
            a_all = np.concatenate([item[1] for item in pair_terms])
            b_all = np.concatenate([item[2] for item in pair_terms])
            dr_all = _min_image_batch(pos[..., b_all, :] - pos[..., a_all, :], bx)
            r_all = np.sqrt(
                np.einsum('...j,...j->...', dr_all, dr_all),
            ).astype(np.float32, copy=False)
            offset = 0
            for key, _a_idx, _b_idx, n_terms in pair_terms:
                stop = offset + n_terms
                out["pair_distances"][key] = r_all[..., offset:stop]
                out["pair_vectors"][key] = np.asarray(
                    dr_all[..., offset:stop, :],
                    dtype=np.float32,
                )
                offset = stop
    
    # else:
    #     raise RuntimeWarning("No pair_cache (per-key neighbor pair list) provided to compute_frame_geometry(), skipping pair geometry computation. This may cause downstream errors if pair geometry is expected by the caller.")

    # --- Bond geometry ---
    bonds = topology_arrays.bonds
    bond_key_index = topology_arrays.bond_key_index
    keys_bondtypes = topology_arrays.keys_bondtypes
    
    if bonds is not None and bond_key_index is not None and keys_bondtypes is not None:
        bonds = np.asarray(bonds, dtype=np.int32)
        bond_key_index = np.asarray(bond_key_index, dtype=np.int32)
        if bonds.size > 0:
            for ki, ikey in enumerate(keys_bondtypes):
                if interaction_mask and not interaction_mask.get(ikey, False):
                    continue
                mask = bond_key_index == ki
                terms = bonds[mask]
                if terms.size == 0:
                    continue
                ia, ib = terms[:, 0], terms[:, 1]
                dr = _min_image_batch(pos[..., ib, :] - pos[..., ia, :], bx)
                r = np.sqrt(np.einsum('...j,...j->...', dr, dr)).astype(np.float32, copy=False)
                out["bond_distances"][ikey] = r
                out["bond_vectors"][ikey] = np.asarray(dr, dtype=np.float32)
                out["bond_indices"][ikey] = terms

    # --- Angle geometry ---
    angles = topology_arrays.angles
    angle_key_index = topology_arrays.angle_key_index
    keys_angletypes = topology_arrays.keys_angletypes
    
    if angles is not None and angle_key_index is not None and keys_angletypes is not None:
        angles_arr = np.asarray(angles, dtype=np.int32)
        angle_key_index = np.asarray(angle_key_index, dtype=np.int32)
        if angles_arr.size > 0:
            for ki, ikey in enumerate(keys_angletypes):
                if interaction_mask and not interaction_mask.get(ikey, False):
                    continue
                mask = angle_key_index == ki
                terms = angles_arr[mask]
                if terms.size == 0:
                    continue
                ia, ib, ic = terms[:, 0], terms[:, 1], terms[:, 2]
                theta = _calc_angles_batch(
                    pos[..., ia, :],
                    pos[..., ib, :],
                    pos[..., ic, :],
                    bx,
                    backend=neighbor_backend,
                )
                out["angle_values"][ikey] = np.degrees(theta).astype(np.float32, copy=False)
                out["angle_indices"][ikey] = terms

    # --- Dihedral geometry ---
    dihedrals = topology_arrays.dihedrals
    dihedral_key_index = topology_arrays.dihedral_key_index
    keys_dihedraltypes = topology_arrays.keys_dihedraltypes
    if dihedrals is not None and dihedral_key_index is not None and keys_dihedraltypes is not None:
        dih_arr = np.asarray(dihedrals, dtype=np.int32)
        dihedral_key_index = np.asarray(dihedral_key_index, dtype=np.int32)
        if dih_arr.size > 0:
            for ki, ikey in enumerate(keys_dihedraltypes):
                if interaction_mask and not interaction_mask.get(ikey, False):
                    continue
                mask_k = dihedral_key_index == ki
                terms = dih_arr[mask_k]
                if terms.size == 0:
                    continue
                # Dihedral angles via MDAnalysis C-accelerated calc_dihedrals
                i1, i2, i3, i4 = terms[:, 0], terms[:, 1], terms[:, 2], terms[:, 3]
                phi_rad = _calc_dihedrals_batch(
                    pos[..., i1, :],
                    pos[..., i2, :],
                    pos[..., i3, :],
                    pos[..., i4, :],
                    bx,
                    backend=neighbor_backend,
                )
                # calc_dihedrals returns [-π, π] rad; convert to [0, 360) deg
                phi_deg = np.degrees(phi_rad % (2.0 * np.pi)).astype(np.float32, copy=False)
                # Mark degenerate dihedrals as NaN
                b1 = _min_image_batch(pos[..., i2, :] - pos[..., i1, :], bx)
                b2 = _min_image_batch(pos[..., i3, :] - pos[..., i2, :], bx)
                b3 = _min_image_batch(pos[..., i4, :] - pos[..., i3, :], bx)
                n1 = np.cross(b2, b1)
                n2 = np.cross(b2, b3)
                n1_sq = np.einsum('...j,...j->...', n1, n1)
                n2_sq = np.einsum('...j,...j->...', n2, n2)
                degenerate = (n1_sq < 1e-24) | (n2_sq < 1e-24)
                if np.any(degenerate):
                    phi_deg[degenerate] = np.nan

                out["dihedral_values"][ikey] = phi_deg
                out["dihedral_indices"][ikey] = terms

    # Read real_site_indices from topology_arrays
    rsi = getattr(topology_arrays, 'real_site_indices', None)
    if rsi is not None:
        rsi = np.asarray(rsi, dtype=np.int32)

    return FrameGeometry(
        **out,
        positions=pos,
        box=bx,
        n_atoms=n_atoms,
        real_site_indices=rsi,
    )
