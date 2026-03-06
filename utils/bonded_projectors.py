"""Force-matching interaction projectors using MDAnalysis + AceCG neighbor helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from MDAnalysis import Universe
from MDAnalysis.lib.distances import minimize_vectors

from .neighbor import ComputeNeighborList, GetBondedInfo, NeighborList2Pair, NeighborList2PairIndices

RAD2DEG = 180.0 / np.pi


@dataclass
class FMInteraction:
    """Single interaction objective entry for FM design matrix construction."""

    style: str
    types: Tuple[str, ...]
    potential: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.style}:{':'.join(self.types)}"

    def n_params(self) -> int:
        return int(self.potential.n_params())


def _canonical_bond(typ_i: str, typ_j: str) -> Tuple[str, str]:
    return (typ_i, typ_j) if typ_i <= typ_j else (typ_j, typ_i)


def _canonical_angle(typ_i: str, typ_j: str, typ_k: str) -> Tuple[str, str, str]:
    left = (typ_i, typ_j, typ_k)
    right = (typ_k, typ_j, typ_i)
    return left if left <= right else right


def _canonical_dihedral(typ_i: str, typ_j: str, typ_k: str, typ_l: str) -> Tuple[str, str, str, str]:
    left = (typ_i, typ_j, typ_k, typ_l)
    right = (typ_l, typ_k, typ_j, typ_i)
    return left if left <= right else right


def interaction_offsets(interactions: Sequence[FMInteraction]) -> List[slice]:
    """Return parameter slices for each interaction in the provided order."""
    out: List[slice] = []
    start = 0
    for it in interactions:
        end = start + it.n_params()
        out.append(slice(start, end))
        start = end
    return out


def _basis_values(pot: Any, x: np.ndarray) -> np.ndarray:
    """Return basis matrix B with shape (n_samples, n_params)."""
    x = np.asarray(x, dtype=float)
    if hasattr(pot, "basis_values"):
        B = np.asarray(pot.basis_values(x), dtype=float)
        if B.ndim != 2 or B.shape[1] != pot.n_params():
            raise ValueError("basis_values must return shape (n_samples, n_params)")
        return B

    vals = []
    for name in pot.dparam_names():
        vals.append(np.asarray(getattr(pot, name)(x), dtype=float))
    if not vals:
        return np.empty((x.shape[0], 0), dtype=float)
    return np.vstack(vals).T


def _basis_derivatives(pot: Any, x: np.ndarray) -> np.ndarray:
    """Return dB/dx matrix with shape (n_samples, n_params)."""
    x = np.asarray(x, dtype=float)
    if hasattr(pot, "basis_derivatives"):
        D = np.asarray(pot.basis_derivatives(x), dtype=float)
        if D.ndim != 2 or D.shape[1] != pot.n_params():
            raise ValueError("basis_derivatives must return shape (n_samples, n_params)")
        return D
    eps = 1.0e-6
    return (_basis_values(pot, x + eps) - _basis_values(pot, x - eps)) / (2.0 * eps)


def _accumulate_rows(
    M: np.ndarray,
    rows: np.ndarray,
    values: np.ndarray,
) -> None:
    """Accumulate value matrix into selected rows of M."""
    if values.size == 0:
        return
    rr = np.asarray(rows, dtype=np.int64).reshape(-1)
    vv = np.asarray(values, dtype=np.float64)
    ncol = int(M.shape[1])
    base = np.arange(vv.shape[1], dtype=np.int64)
    flat_idx = (rr[:, None] * ncol + base[None, :]).ravel()
    np.add.at(M.reshape(-1), flat_idx, vv.ravel())


def _min_image(vec: np.ndarray, box: np.ndarray) -> np.ndarray:
    if box is None or box.size == 0:
        return vec
    return minimize_vectors(vec, box=box)


def _resolve_interaction_types(u: Universe, types: Sequence[str]) -> Tuple[str, ...]:
    alias = getattr(u, "_acecg_type_alias", None)
    if not isinstance(alias, dict):
        return tuple(str(t) for t in types)
    return tuple(str(alias.get(str(t), str(t))) for t in types)


def _interaction_domain(interaction: FMInteraction) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(interaction.metadata, dict):
        return None, None
    lo = interaction.metadata.get("min")
    hi = interaction.metadata.get("max")
    return (float(lo) if lo is not None else None, float(hi) if hi is not None else None)


def _domain_mask(x: np.ndarray, interaction: FMInteraction) -> np.ndarray:
    vals = np.asarray(x, dtype=float)
    lo, hi = _interaction_domain(interaction)
    mask = np.ones(vals.shape[0], dtype=bool)
    if lo is not None:
        mask &= vals >= lo
    if hi is not None:
        mask &= vals <= hi
    return mask


class PairProjector:
    """Pair projector; pair construction always uses neighbor.py helper path."""

    def compute(
        self,
        u: Universe,
        interaction: FMInteraction,
        *,
        cutoff: float,
        exclude: Any,
        sel: str = "all",
        pair_cache: Optional[Mapping[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> np.ndarray:
        n_atoms = len(u.atoms)
        n_params = interaction.n_params()
        out = np.zeros((n_atoms * 3, n_params), dtype=np.float64)

        target_types = _resolve_interaction_types(u, interaction.types)
        if pair_cache is not None and target_types in pair_cache:
            a_idx, b_idx = pair_cache[target_types]
            if a_idx.size == 0:
                return out
        else:
            pair2atom = NeighborList2Pair(
                u=u,
                pair2potential={target_types: interaction.potential},
                sel=sel,
                cutoff=float(cutoff),
                frame=None,
                exclude=exclude,
            )
            tuples = pair2atom.get(target_types, [])
            if len(tuples) == 0:
                return out

            a_idx = np.fromiter((a.index for a, _ in tuples), dtype=np.int64, count=len(tuples))
            b_idx = np.fromiter((b.index for _, b in tuples), dtype=np.int64, count=len(tuples))

        pos = np.asarray(u.atoms.positions, dtype=np.float64)
        box = np.asarray(u.dimensions, dtype=np.float64)
        dr = _min_image(pos[b_idx] - pos[a_idx], box)
        r = np.linalg.norm(dr, axis=1)
        if r.size == 0:
            return out

        nz = r > 1.0e-12
        if not np.any(nz):
            return out
        a_idx = a_idx[nz]
        b_idx = b_idx[nz]
        dr = dr[nz]
        r = r[nz]

        in_domain = _domain_mask(r, interaction)
        if not np.any(in_domain):
            return out
        a_idx = a_idx[in_domain]
        b_idx = b_idx[in_domain]
        dr = dr[in_domain]
        r = r[in_domain]

        sparse_basis = None
        if hasattr(interaction.potential, "basis_values_sparse"):
            sparse_basis = interaction.potential.basis_values_sparse(r)
        if sparse_basis is not None:
            coo = sparse_basis.tocoo()
            if coo.nnz == 0:
                return out
            row = np.asarray(coo.row, dtype=np.int64)
            col = np.asarray(coo.col, dtype=np.int64)
            dat = np.asarray(coo.data, dtype=np.float64)
            invr = -1.0 / r[row]

            for d in range(3):
                w = dat * invr * dr[row, d]
                np.add.at(out, (a_idx[row] * 3 + d, col), w)
                np.add.at(out, (b_idx[row] * 3 + d, col), -w)
            return out

        B = _basis_values(interaction.potential, r)
        coeff = B * (-1.0 / r)[:, None]

        for d in range(3):
            comp = coeff * dr[:, d : d + 1]
            _accumulate_rows(out, a_idx * 3 + d, comp)
            _accumulate_rows(out, b_idx * 3 + d, -comp)

        return out


class BondProjector:
    """Bond projector based on bonded term arrays from neighbor.py cache."""

    def compute(
        self,
        u: Universe,
        interaction: FMInteraction,
        *,
        terms_filtered: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_atoms = len(u.atoms)
        n_params = interaction.n_params()
        out = np.zeros((n_atoms * 3, n_params), dtype=np.float64)

        if terms_filtered is not None:
            terms = np.asarray(terms_filtered, dtype=np.int64)
            if terms.size == 0:
                return out
        else:
            info = GetBondedInfo(u)
            terms = np.asarray(info["bond_terms"], dtype=np.int64)
            if terms.size == 0:
                return out

            atom_types = np.asarray(u.atoms.types).astype(str)
            t0, t1 = _resolve_interaction_types(u, interaction.types[:2])

            keep = []
            target = _canonical_bond(t0, t1)
            for idx, (ia, ib) in enumerate(terms):
                key = _canonical_bond(str(atom_types[ia]), str(atom_types[ib]))
                if key == target:
                    keep.append(idx)
            if not keep:
                return out

            terms = terms[np.asarray(keep, dtype=np.int64)]
        ia = terms[:, 0]
        ib = terms[:, 1]

        pos = np.asarray(u.atoms.positions, dtype=np.float64)
        box = np.asarray(u.dimensions, dtype=np.float64)
        d = _min_image(pos[ib] - pos[ia], box)
        r = np.linalg.norm(d, axis=1)
        nz = r > 1.0e-12
        if not np.any(nz):
            return out

        ia = ia[nz]
        ib = ib[nz]
        d = d[nz]
        r = r[nz]

        in_domain = _domain_mask(r, interaction)
        if not np.any(in_domain):
            return out
        ia = ia[in_domain]
        ib = ib[in_domain]
        d = d[in_domain]
        r = r[in_domain]

        B = _basis_values(interaction.potential, r)
        coeff = B * (-1.0 / r)[:, None]

        for dim in range(3):
            comp = coeff * d[:, dim : dim + 1]
            _accumulate_rows(out, ia * 3 + dim, comp)
            _accumulate_rows(out, ib * 3 + dim, -comp)

        return out


class AngleProjector:
    """Angle projector reproducing OpenMSCG bond_list.cpp geometry factors."""

    def compute(
        self,
        u: Universe,
        interaction: FMInteraction,
        *,
        terms_filtered: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_atoms = len(u.atoms)
        n_params = interaction.n_params()
        out = np.zeros((n_atoms * 3, n_params), dtype=np.float64)

        if terms_filtered is not None:
            terms = np.asarray(terms_filtered, dtype=np.int64)
            if terms.size == 0:
                return out
        else:
            info = GetBondedInfo(u)
            terms = np.asarray(info["angle_terms"], dtype=np.int64)
            if terms.size == 0:
                return out

            atom_types = np.asarray(u.atoms.types).astype(str)
            tt = _resolve_interaction_types(u, interaction.types[:3])
            target = _canonical_angle(tt[0], tt[1], tt[2])

            keep = []
            for idx, (ia, ib, ic) in enumerate(terms):
                key = _canonical_angle(str(atom_types[ia]), str(atom_types[ib]), str(atom_types[ic]))
                if key == target:
                    keep.append(idx)
            if not keep:
                return out

            terms = terms[np.asarray(keep, dtype=np.int64)]
        ia = terms[:, 0]
        ib = terms[:, 1]
        ic = terms[:, 2]

        pos = np.asarray(u.atoms.positions, dtype=np.float64)
        box = np.asarray(u.dimensions, dtype=np.float64)
        d1 = _min_image(pos[ia] - pos[ib], box)
        d2 = _min_image(pos[ic] - pos[ib], box)

        rsq1 = np.einsum("ij,ij->i", d1, d1)
        rsq2 = np.einsum("ij,ij->i", d2, d2)
        r1 = np.sqrt(rsq1)
        r2 = np.sqrt(rsq2)
        valid = (r1 > 1.0e-12) & (r2 > 1.0e-12)
        if not np.any(valid):
            return out

        ia = ia[valid]
        ib = ib[valid]
        ic = ic[valid]
        d1 = d1[valid]
        d2 = d2[valid]
        rsq1 = rsq1[valid]
        rsq2 = rsq2[valid]
        r1 = r1[valid]
        r2 = r2[valid]

        c = np.einsum("ij,ij->i", d1, d2) / (r1 * r2)
        c = np.clip(c, -1.0, 1.0)
        s = np.sqrt(np.maximum(1.0 - c * c, 1.0e-8))
        invs = 1.0 / np.maximum(s, 1.0e-4)

        theta_rad = np.arccos(c)
        theta_deg = theta_rad * RAD2DEG

        in_domain = _domain_mask(theta_deg, interaction)
        if not np.any(in_domain):
            return out

        ia = ia[in_domain]
        ib = ib[in_domain]
        ic = ic[in_domain]
        d1 = d1[in_domain]
        d2 = d2[in_domain]
        rsq1 = rsq1[in_domain]
        rsq2 = rsq2[in_domain]
        r1 = r1[in_domain]
        r2 = r2[in_domain]
        c = c[in_domain]
        s = s[in_domain]
        invs = invs[in_domain]
        theta_deg = theta_deg[in_domain]

        a11 = invs * c / rsq1
        a12 = -invs / (r1 * r2)
        a22 = invs * c / rsq2

        f1 = a11[:, None] * d1 + a12[:, None] * d2
        f3 = a22[:, None] * d2 + a12[:, None] * d1

        B = _basis_values(interaction.potential, theta_deg) * RAD2DEG

        for dim in range(3):
            f1d = f1[:, dim : dim + 1]
            f3d = f3[:, dim : dim + 1]
            _accumulate_rows(out, ia * 3 + dim, B * f1d)
            _accumulate_rows(out, ib * 3 + dim, -B * (f1d + f3d))
            _accumulate_rows(out, ic * 3 + dim, B * f3d)

        return out


class DihedralProjector:
    """Dihedral projector matching OpenMSCG's bond_list.cpp derivative formulas."""

    def compute(
        self,
        u: Universe,
        interaction: FMInteraction,
        *,
        terms_filtered: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_atoms = len(u.atoms)
        n_params = interaction.n_params()
        out = np.zeros((n_atoms * 3, n_params), dtype=np.float64)

        if terms_filtered is not None:
            terms = np.asarray(terms_filtered, dtype=np.int64)
            if terms.size == 0:
                return out
            target = None
            atom_types = None
        else:
            info = GetBondedInfo(u)
            terms = np.asarray(info["dihedral_terms"], dtype=np.int64)
            if terms.size == 0:
                return out

            atom_types = np.asarray(u.atoms.types).astype(str)
            tt = _resolve_interaction_types(u, interaction.types[:4])
            target = _canonical_dihedral(tt[0], tt[1], tt[2], tt[3])

        pos = np.asarray(u.atoms.positions, dtype=np.float64)
        box = np.asarray(u.dimensions, dtype=np.float64)

        for i1, i2, i3, i4 in terms:
            if target is not None and atom_types is not None:
                key = _canonical_dihedral(
                    str(atom_types[i1]),
                    str(atom_types[i2]),
                    str(atom_types[i3]),
                    str(atom_types[i4]),
                )
                if key != target:
                    continue

            b1 = _min_image(pos[i2] - pos[i1], box)
            b2 = _min_image(pos[i3] - pos[i2], box)
            b3 = _min_image(pos[i4] - pos[i3], box)

            n1 = np.cross(b2, b1)
            n2 = np.cross(b2, b3)
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            if n1_norm < 1.0e-12 or n2_norm < 1.0e-12:
                continue
            n1 /= n1_norm
            n2 /= n2_norm

            cphi = -np.dot(n1, n2)
            cphi = float(np.clip(cphi, -1.0, 1.0))
            phi = float(np.arccos(cphi))
            if float(np.dot(n1, b3)) > 0.0:
                phi = 2.0 * np.pi - phi

            r_ij = b1
            r_kj = _min_image(pos[i2] - pos[i3], box)
            r_kl = b3
            r_mj = np.cross(r_ij, r_kj)
            r_nk = np.cross(r_kj, r_kl)

            l2_kj = float(np.dot(r_kj, r_kj))
            if l2_kj < 1.0e-12:
                continue
            l_kj = np.sqrt(l2_kj)

            r_mj2 = float(np.dot(r_mj, r_mj))
            r_nk2 = float(np.dot(r_nk, r_nk))
            if r_mj2 < 1.0e-12 or r_nk2 < 1.0e-12:
                continue

            f1 = r_mj * (l_kj / r_mj2)
            f4 = r_nk * (-l_kj / r_nk2)
            f2 = f1 * (np.dot(r_ij, r_kj) / l2_kj - 1.0) + f4 * (
                np.dot(r_kl, r_kj) / l2_kj * (-1.0)
            )
            f3 = f4 * (np.dot(r_kl, r_kj) / l2_kj - 1.0) + f1 * (
                np.dot(r_ij, r_kj) / l2_kj * (-1.0)
            )

            B = _basis_values(interaction.potential, np.array([phi], dtype=float))[0] * RAD2DEG
            rows = [i1, i2, i3, i4]
            grads = [f1, f2, f3, f4]
            for atom, gvec in zip(rows, grads):
                for dim in range(3):
                    out[atom * 3 + dim, :] += B * gvec[dim]

        return out


class NB3BProjector:
    """NB3B B-spline projector translated from OpenMSCG model_nb3b_bspline.cpp."""

    def compute(
        self,
        u: Universe,
        interaction: FMInteraction,
        *,
        cutoff: float,
        exclude: Any,
    ) -> np.ndarray:
        n_atoms = len(u.atoms)
        n_params = interaction.n_params()
        out = np.zeros((n_atoms * 3, n_params), dtype=np.float64)

        atom_types = np.asarray(u.atoms.types).astype(str)
        t1, tc, t2 = _resolve_interaction_types(u, interaction.types[:3])
        gamma_ij = float(interaction.metadata.get("gamma_ij", 1.2))
        a_ij = float(interaction.metadata.get("a_ij", 4.0))
        gamma_ik = float(interaction.metadata.get("gamma_ik", 1.2))
        a_ik = float(interaction.metadata.get("a_ik", 4.0))

        # Neighbor construction intentionally routes through neighbor.py helpers.
        nbr = ComputeNeighborList(u=u, cutoff=float(cutoff), frame=None, exclude=exclude)
        pos = np.asarray(u.atoms.positions, dtype=np.float64)
        box = np.asarray(u.dimensions, dtype=np.float64)

        for i in range(n_atoms):
            if str(atom_types[i]) != tc:
                continue

            cand_ij: List[Tuple[int, np.ndarray, float]] = []
            cand_ik: List[Tuple[int, np.ndarray, float]] = []
            for j in nbr[i]:
                vec = _min_image((pos[j] - pos[i])[None, :], box)[0]
                rij = float(np.linalg.norm(vec))
                if rij < 1.0e-12:
                    continue
                tj = str(atom_types[j])
                if tj == t1 and rij < a_ij:
                    cand_ij.append((j, vec, rij))
                if tj == t2 and rij < a_ik:
                    cand_ik.append((j, vec, rij))

            if not cand_ij or not cand_ik:
                continue

            for j, dr1, r1 in cand_ij:
                for k, dr2, r2 in cand_ik:
                    if j == k:
                        continue
                    if t1 == t2 and k < j:
                        continue

                    rinv1 = 1.0 / r1
                    rinv2 = 1.0 / r2
                    rinv1a = 1.0 / (r1 - a_ij)
                    rinv2a = 1.0 / (r2 - a_ik)
                    exp1 = np.exp(gamma_ij * rinv1a)
                    exp2 = np.exp(gamma_ik * rinv2a)
                    e1e2 = exp1 * exp2

                    cs = float(np.dot(dr1, dr2) / (r1 * r2))
                    cs = float(np.clip(cs, -1.0, 1.0))
                    theta = float(np.arccos(cs))
                    theta_deg = theta * RAD2DEG

                    B = _basis_values(interaction.potential, np.array([theta_deg], dtype=float))[0]
                    dB = _basis_derivatives(
                        interaction.potential,
                        np.array([theta_deg], dtype=float),
                    )[0]
                    sin_theta = np.sin(theta)
                    if abs(sin_theta) < 1.0e-8:
                        sin_theta = 1.0e-8
                    B_deriv = dB * (-1.0 / sin_theta)

                    dedxj = gamma_ij * rinv1a * rinv1a * rinv1 * e1e2
                    dedxk = gamma_ik * rinv2a * rinv2a * rinv2 * e1e2
                    dcdxj = cs * rinv1 * rinv1 * e1e2
                    dcdxk = cs * rinv2 * rinv2 * e1e2
                    dcdx2 = -1.0 / (r1 * r2) * e1e2

                    for dim in range(3):
                        fj = dedxj * dr1[dim]
                        fk = dedxk * dr2[dim]
                        fi = -fj - fk

                        fj2 = dcdxj * dr1[dim] + dcdx2 * dr2[dim]
                        fk2 = dcdxk * dr2[dim] + dcdx2 * dr1[dim]
                        fi2 = -fj2 - fk2

                        out[i * 3 + dim, :] += B * fi + B_deriv * fi2
                        out[j * 3 + dim, :] += B * fj + B_deriv * fj2
                        out[k * 3 + dim, :] += B * fk + B_deriv * fk2

        return out


PROJECTOR_MAP = {
    "pair": PairProjector(),
    "bond": BondProjector(),
    "angle": AngleProjector(),
    "dihedral": DihedralProjector(),
    "nb3b": NB3BProjector(),
}


def build_design_matrix(
    u: Universe,
    interactions: Sequence[FMInteraction],
    *,
    cutoff: float,
    exclude: Any,
    sel: str = "all",
) -> np.ndarray:
    """Build the full FM design matrix for one frame."""
    n_atoms = len(u.atoms)
    offsets = interaction_offsets(interactions)
    n_params = 0 if not offsets else offsets[-1].stop
    mat = np.zeros((n_atoms * 3, n_params), dtype=np.float64)
    resolved_types = [_resolve_interaction_types(u, it.types) for it in interactions]

    pair_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    bond_terms_cache: Dict[Tuple[str, str], np.ndarray] = {}
    angle_terms_cache: Dict[Tuple[str, str, str], np.ndarray] = {}
    dihedral_terms_cache: Dict[Tuple[str, str, str, str], np.ndarray] = {}

    pair_inters = [(it, rt) for it, rt in zip(interactions, resolved_types) if it.style == "pair"]
    if pair_inters:
        pair2potential: Dict[Tuple[str, str], Any] = {}
        for it, rt in pair_inters:
            pair2potential[rt] = it.potential
        pair2idx = NeighborList2PairIndices(
            u=u,
            pair2potential=pair2potential,
            sel=sel,
            cutoff=float(cutoff),
            frame=None,
            exclude=exclude,
        )
        for key, ab in pair2idx.items():
            ai, bi = ab
            pair_cache[key] = (np.asarray(ai, dtype=np.int64), np.asarray(bi, dtype=np.int64))

    bonded_styles = {it.style for it in interactions if it.style in {"bond", "angle", "dihedral"}}
    if bonded_styles:
        info = GetBondedInfo(u)
        atom_types = np.asarray(u.atoms.types).astype(str)

        if "bond" in bonded_styles:
            terms = np.asarray(info["bond_terms"], dtype=np.int64)
            if terms.size > 0:
                t0 = atom_types[terms[:, 0]]
                t1 = atom_types[terms[:, 1]]
                left = np.where(t0 <= t1, t0, t1)
                right = np.where(t0 <= t1, t1, t0)
                keys = np.char.add(np.char.add(left, ":"), right)
                for it, rt in zip(interactions, resolved_types):
                    if it.style != "bond":
                        continue
                    tt = rt[:2]
                    tgt = ":".join(_canonical_bond(tt[0], tt[1]))
                    mask = keys == tgt
                    bond_terms_cache[tt] = terms[mask]

        if "angle" in bonded_styles:
            terms = np.asarray(info["angle_terms"], dtype=np.int64)
            if terms.size > 0:
                ta = atom_types[terms[:, 0]]
                tb = atom_types[terms[:, 1]]
                tc = atom_types[terms[:, 2]]
                left = np.char.add(np.char.add(np.char.add(ta, ":"), tb), np.char.add(":", tc))
                right = np.char.add(np.char.add(np.char.add(tc, ":"), tb), np.char.add(":", ta))
                keys = np.where(left <= right, left, right)
                for it, rt in zip(interactions, resolved_types):
                    if it.style != "angle":
                        continue
                    tt = rt[:3]
                    c = _canonical_angle(tt[0], tt[1], tt[2])
                    tgt = ":".join(c)
                    mask = keys == tgt
                    angle_terms_cache[tt] = terms[mask]

        if "dihedral" in bonded_styles:
            terms = np.asarray(info["dihedral_terms"], dtype=np.int64)
            if terms.size > 0:
                ta = atom_types[terms[:, 0]]
                tb = atom_types[terms[:, 1]]
                tc = atom_types[terms[:, 2]]
                td = atom_types[terms[:, 3]]
                left = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(ta, ":"), tb), ":"), tc), ":"), td)
                right = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(td, ":"), tc), ":"), tb), ":"), ta)
                keys = np.where(left <= right, left, right)
                for it, rt in zip(interactions, resolved_types):
                    if it.style != "dihedral":
                        continue
                    tt = rt[:4]
                    c = _canonical_dihedral(tt[0], tt[1], tt[2], tt[3])
                    tgt = ":".join(c)
                    mask = keys == tgt
                    dihedral_terms_cache[tt] = terms[mask]

    for inter, sl, rt in zip(interactions, offsets, resolved_types):
        projector = PROJECTOR_MAP.get(inter.style)
        if projector is None:
            raise KeyError(f"Unsupported FM interaction style: {inter.style}")
        if inter.style in {"pair", "nb3b"}:
            if inter.style == "pair":
                block = projector.compute(u, inter, cutoff=cutoff, exclude=exclude, sel=sel, pair_cache=pair_cache)
            else:
                block = projector.compute(u, inter, cutoff=cutoff, exclude=exclude)
        else:
            if inter.style == "bond":
                tt = rt[:2]
                block = projector.compute(u, inter, terms_filtered=bond_terms_cache.get(tt))
            elif inter.style == "angle":
                tt = rt[:3]
                block = projector.compute(u, inter, terms_filtered=angle_terms_cache.get(tt))
            elif inter.style == "dihedral":
                tt = rt[:4]
                block = projector.compute(u, inter, terms_filtered=dihedral_terms_cache.get(tt))
            else:
                block = projector.compute(u, inter)
        mat[:, sl] = block

    return mat
