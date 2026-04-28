"""Per-frame geometry extraction — shared across REM, FM, and CDFM paths.

Provides:
- ``FrameGeometry``: Immutable dataclass holding all per-frame geometric
  quantities (distances, vectors, angles, dihedrals) keyed by InteractionKey.
- ``compute_frame_geometry()``: Unified one-pass extraction that replaces
  the scattered legacy geometry helpers and interaction-cache builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from MDAnalysis.lib.distances import calc_angles, calc_dihedrals, minimize_vectors

from ..topology.types import InteractionKey
from ..topology.topology_array import TopologyArrays


@dataclass(frozen=True)
class FrameGeometry:
    """All geometry quantities for one frame.  Immutable, cacheable.

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _min_image(vec: np.ndarray, box: np.ndarray) -> np.ndarray:
    if box is None or box.size == 0:
        return vec
    return minimize_vectors(vec, box=box)


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
    """Compute all per-frame geometry in a single pass.

    Parameters
    ----------
    positions : (n_atoms, 3) float32 array
    box : (6,) float32 array (MDAnalysis dimensions format)
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
    bx = np.asarray(box, dtype=np.float32)
    n_atoms = pos.shape[0]

    out = {"pair_distances": {}, "pair_indices": {}, "pair_vectors": {},
           "bond_distances": {}, "bond_vectors": {}, "bond_indices": {},
           "angle_values": {}, "angle_indices": {},
           "dihedral_values": {}, "dihedral_indices": {},
           }

    # --- Pair geometry (from pre-computed pair_cache) ---
    if pair_cache is not None:
        for key, (a_idx, b_idx) in pair_cache.items():
            
            # Masking out interactions that are disabled by the interaction_mask
            if interaction_mask and not interaction_mask.get(key, False):
                continue

            a_idx = np.asarray(a_idx, dtype=np.int32)
            b_idx = np.asarray(b_idx, dtype=np.int32)
            if a_idx.size == 0:
                out["pair_distances"][key] = np.empty(0, dtype=np.float32)
                out["pair_indices"][key] = (a_idx, b_idx)
                out["pair_vectors"][key] = np.empty((0, 3), dtype=np.float32)
                continue
            dr = _min_image(pos[b_idx] - pos[a_idx], bx)
            r = np.sqrt(np.einsum('ij,ij->i', dr, dr)).astype(np.float32, copy=False)
            out["pair_distances"][key] = r
            out["pair_indices"][key] = (a_idx, b_idx)
            out["pair_vectors"][key] = np.asarray(dr, dtype=np.float32)
    
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
                dr = _min_image(pos[ib] - pos[ia], bx)
                r = np.sqrt(np.einsum('ij,ij->i', dr, dr)).astype(np.float32, copy=False)
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
                a_pos = pos[ia]
                b_pos = pos[ib]
                c_pos = pos[ic]
                theta = calc_angles(a_pos, b_pos, c_pos, box=bx, backend=neighbor_backend)
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
                phi_rad = calc_dihedrals(
                    pos[i1], pos[i2], pos[i3], pos[i4],
                    box=bx, backend=neighbor_backend,
                )
                # calc_dihedrals returns [-π, π] rad; convert to [0, 360) deg
                phi_deg = np.degrees(phi_rad % (2.0 * np.pi)).astype(np.float32, copy=False)
                # Mark degenerate dihedrals as NaN
                b1 = _min_image(pos[i2] - pos[i1], bx)
                b2 = _min_image(pos[i3] - pos[i2], bx)
                b3 = _min_image(pos[i4] - pos[i3], bx)
                n1 = np.cross(b2, b1)
                n2 = np.cross(b2, b3)
                n1_sq = np.einsum('ij,ij->i', n1, n1)
                n2_sq = np.einsum('ij,ij->i', n2, n2)
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
