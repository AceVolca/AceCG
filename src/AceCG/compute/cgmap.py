# AceCG/compute/cgmap.py
"""Per-frame AA→CG mapping kernel.

:class:`~AceCG.topology.cgmap.CGMapSpec` compiles the mapping YAML once;
:class:`CGMapper` turns one AA frame into one CG frame using only whole-array
NumPy operations on preallocated buffers. There is no Python-level loop over
sites, molecules, or atoms, which is the whole point: OpenMSCG's ``Mapper.process``
loops over sites in the interpreter and dominates its own runtime.

The three arrays per frame
--------------------------
Positions are a **weighted average** over each site's atoms (``csr_wx`` is row
normalized), forces are a **weighted sum** (``csr_wf`` is raw)::

    X_I = Σ_i w^x_{Ii} x_i          with Σ_i w^x_{Ii} = 1
    F_I = Σ_i w^f_{Ii} f_i

Velocities follow the position weights, so mass ``x-weight`` gives the exact
centre-of-mass velocity of the site.

Periodic images
---------------
A site whose atoms straddle a periodic boundary averages to a meaningless point
somewhere in the middle of the box, so the atoms must be made whole *before*
averaging. ``unwrap`` picks the reference each atom is imaged against:

``"molecule"`` (default)
    One reference atom per molecule (``spec.mol_ref_pos``), so a whole lipid,
    protein, or repeat unit stays intact. Costs one imaging pass over the
    ``n_required_atoms`` compact buffer.
``"bead"``
    One reference atom per site — the site's own ``index[0]``, matching what
    OpenMSCG pivots on. Imaging happens in ``nnz`` space, so an atom shared by
    two sites is imaged correctly for each. Use this when molecules share atoms.
``"none"``
    No imaging. Correct only for an already-whole (unwrapped) trajectory, and
    the fastest path.
``"deprecated"``
    Bit-compatible reproduction of OpenMSCG's single-shift arithmetic, for
    reproducing old CG trajectories. Wrong whenever a site spans more than one
    box length; see :func:`~AceCG.io.coordinates.minimum_image_displacements`.

Both ``"molecule"`` and ``"bead"`` use a true minimum image
(``d -= L·rint(d/L)``) and support triclinic cells.

Forces and velocities are frame-invariant, so they are never imaged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..io.coordinates import classify_box, minimum_image_displacements, wrap_positions_in_box
from ..topology.cgmap import CGMapSpec


__all__ = ["CGMapper", "MappedFrame", "UNWRAP_MODES"]


UNWRAP_MODES = ("molecule", "bead", "none", "deprecated")


@dataclass(frozen=True)
class MappedFrame:
    """One mapped CG frame.

    Attributes
    ----------
    frame_id : int
        AA frame index this was mapped from.
    positions : np.ndarray, shape (n_sites, 3)
        CG site positions, Å, in the mapper's output dtype.
    box : np.ndarray or None
        The AA frame's unit cell, unchanged, ``[lx, ly, lz, alpha, beta, gamma]``.
    forces : np.ndarray or None
        CG site forces when AA forces were supplied.
    velocities : np.ndarray or None
        CG site velocities when AA velocities were supplied.
    """

    frame_id: int
    positions: np.ndarray
    box: Optional[np.ndarray] = None
    forces: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None


class CGMapper:
    """Map AA frames to CG frames through a compiled :class:`CGMapSpec`.

    One mapper owns its scratch buffers and is therefore **not** thread safe and
    **not** reentrant; give each thread or rank its own. Buffers are allocated
    once in ``__init__``, so a long trajectory does not churn the allocator.

    Parameters
    ----------
    spec : CGMapSpec
        Compiled mapping plan.
    unwrap : {"molecule", "bead", "none", "deprecated"}, default "molecule"
        Periodic-image policy, see the module docstring.
    wrap : bool, default True
        Wrap the mapped CG positions back into the primary cell. Mapped sites of
        an unwrapped molecule can land outside the box, which LAMMPS ``data`` and
        GRO readers dislike; XTC stores whatever it is given.
    triclinic : {"exact", "fast"}, default "exact"
        Non-orthorhombic minimum-image strategy, see
        :func:`~AceCG.io.coordinates.minimum_image_displacements`.
    dtype : np.dtype, default np.float64
        Working precision of the weighted sums. The weights are float64 either
        way; float32 halves the traffic at ~1e-7 relative accuracy.
    out_dtype : np.dtype, optional
        Dtype of the returned arrays. Defaults to ``dtype``. XTC/TRR writers take
        float32, so ``out_dtype=np.float32`` avoids a later cast.

    Examples
    --------
    >>> mapper = CGMapper(spec)                                    # doctest: +SKIP
    >>> for frame in iter_frames(universe, include_forces=True):   # doctest: +SKIP
    ...     cg = mapper.map_frame(
    ...         frame["positions"],
    ...         box=frame["box"],
    ...         forces=frame["forces"],
    ...         frame_id=frame["frame_id"],
    ...     )
    """

    def __init__(
        self,
        spec: CGMapSpec,
        *,
        unwrap: str = "molecule",
        wrap: bool = True,
        triclinic: str = "exact",
        dtype: np.dtype = np.float64,
        out_dtype: Optional[np.dtype] = None,
    ) -> None:
        unwrap = str(unwrap).strip().lower()
        if unwrap not in UNWRAP_MODES:
            raise ValueError(f"unwrap must be one of {UNWRAP_MODES}, got {unwrap!r}.")
        if triclinic not in ("exact", "fast"):
            raise ValueError(f"triclinic must be 'exact' or 'fast', got {triclinic!r}.")
        if not np.all(np.diff(spec.csr_indptr) > 0):
            raise ValueError(
                "every CG site must own at least one AA atom; this spec has an "
                "empty site row, which `np.add.reduceat` cannot express."
            )

        self.spec = spec
        self.unwrap = unwrap
        self.wrap = bool(wrap)
        self.triclinic = triclinic
        self.dtype = np.dtype(dtype)
        self.out_dtype = np.dtype(dtype if out_dtype is None else out_dtype)

        n_required = spec.n_required_atoms
        nnz = spec.nnz

        # Cast the index arrays to intp once so per-frame `np.take` does not.
        self._atom_indices = spec.atom_indices.astype(np.intp, copy=False)
        self._csr_cols = spec.csr_cols.astype(np.intp, copy=False)
        self._row_starts = spec.csr_indptr[:-1].astype(np.intp, copy=False)
        # Column vectors, so `entries *= weights` broadcasts over x/y/z.
        self._wx = spec.csr_wx.astype(self.dtype, copy=False)[:, None]
        self._wf = spec.csr_wf.astype(self.dtype, copy=False)[:, None]

        # Scratch. `_entries` is the (nnz, 3) gather of contributing atoms and is
        # reused by positions, forces, and velocities within one frame.
        self._compact = np.empty((n_required, 3), dtype=self.dtype)
        self._entries = np.empty((nnz, 3), dtype=self.dtype)
        self._image_scratch: Optional[np.ndarray] = None

        # Per-atom molecule reference, only for unwrap="molecule".
        self._atom_ref: Optional[np.ndarray] = None
        if unwrap == "molecule":
            counts = np.diff(spec.mol_indptr)
            refs = np.repeat(spec.mol_ref_pos.astype(np.intp, copy=False), counts)
            if spec.molecules_contiguous:
                self._atom_ref = refs
            else:
                atom_ref = np.empty(n_required, dtype=np.intp)
                atom_ref[spec.mol_atom_pos.astype(np.intp, copy=False)] = refs
                self._atom_ref = atom_ref
        # Per-entry site reference, for unwrap="bead"/"deprecated".
        self._entry_ref: Optional[np.ndarray] = None
        if unwrap in ("bead", "deprecated"):
            self._entry_ref = np.repeat(
                spec.site_ref_pos.astype(np.intp, copy=False), np.diff(spec.csr_indptr)
            )

    # ── introspection ─────────────────────────────────────────────

    @property
    def n_sites(self) -> int:
        """Number of CG sites produced per frame."""
        return self.spec.n_sites

    @property
    def n_required_atoms(self) -> int:
        """Number of AA atoms read per frame."""
        return self.spec.n_required_atoms

    def nbytes(self) -> int:
        """Bytes held in the per-frame scratch buffers."""
        total = self._compact.nbytes + self._entries.nbytes
        if self._image_scratch is not None:
            total += self._image_scratch.nbytes
        return int(total)

    # ── the kernel ────────────────────────────────────────────────

    def map_frame(
        self,
        positions: np.ndarray,
        *,
        box: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        frame_id: int = 0,
        compact: bool = False,
    ) -> MappedFrame:
        """Map one AA frame.

        Parameters
        ----------
        positions : np.ndarray, shape (n_atoms, 3)
            AA positions in Å. Never modified. With *compact* set, shape
            ``(spec.n_required_atoms, 3)`` instead.
        box : np.ndarray, optional
            ``[lx, ly, lz]`` or ``[lx, ly, lz, alpha, beta, gamma]``, Å/deg. When
            absent or degenerate, imaging and wrapping are both skipped.
        forces : np.ndarray, optional
            AA forces, shape ``(n_atoms, 3)`` or flat ``(3 * n_atoms,)`` — the
            flat form is what :func:`~AceCG.io.trajectory.iter_frames` yields.
        velocities : np.ndarray, optional
            AA velocities, same accepted shapes as *forces*.
        frame_id : int, default 0
            Recorded on the result; the mapping itself does not use it.
        compact : bool, default False
            The inputs already hold *only* the mapping's own atoms, in
            ``spec.atom_indices`` order — what
            ``iter_frames(..., atom_indices=spec.atom_indices)`` yields. The
            first gather is then a straight copy instead of a fancy-index over
            the whole frame, and the caller never materialized the atoms this
            mapping ignores.

        Returns
        -------
        MappedFrame
        """
        positions = np.asarray(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (n_atoms, 3), got {positions.shape}."
            )
        n_atoms = int(positions.shape[0])
        n_required = self.spec.n_required_atoms
        if compact:
            if n_atoms != n_required:
                raise ValueError(
                    f"compact=True expects the {n_required} mapped atom(s) in "
                    f"spec.atom_indices order, got {n_atoms} row(s)."
                )
        elif self._atom_indices.size and int(self._atom_indices[-1]) >= n_atoms:
            raise ValueError(
                f"mapping needs AA atom index {int(self._atom_indices[-1])} but the "
                f"frame has only {n_atoms} atoms."
            )

        box_kind, _, _ = classify_box(box)
        active_box = None if box_kind == "none" else np.asarray(box, dtype=np.float64).ravel()

        # Positions: gather → image → weighted average.
        self._compact_from(positions, compact)
        if active_box is not None and self.unwrap == "molecule":
            self._image_against(self._compact, self._atom_ref, active_box)
        self._gather(self._compact, self._csr_cols, self._entries)
        if active_box is not None and self.unwrap in ("bead", "deprecated"):
            # Image in nnz space: the reference is the site's own index[0], so an
            # atom shared by several sites is imaged once per site.
            refs = np.take(self._compact, self._entry_ref, axis=0)
            self._entries -= refs
            self._reduce_images(self._entries, active_box)
            self._entries += refs
        cg_positions = self._weighted_rows(self._wx)

        if self.wrap and active_box is not None:
            wrap_positions_in_box(cg_positions, active_box, out=cg_positions)

        cg_forces = None
        if forces is not None:
            cg_forces = (
                self._map_fitted_forces(forces, n_atoms, compact)
                if self.spec.has_force_operator
                else self._map_extensive(forces, n_atoms, self._wf, "forces", compact)
            )
        cg_velocities = None
        if velocities is not None:
            # Position weights: with mass x-weight this is the site's COM velocity.
            cg_velocities = self._map_extensive(
                velocities, n_atoms, self._wx, "velocities", compact
            )

        return MappedFrame(
            frame_id=int(frame_id),
            positions=cg_positions,
            box=None if box is None else np.asarray(box, dtype=np.float64).ravel().copy(),
            forces=cg_forces,
            velocities=cg_velocities,
        )

    # ── internals ─────────────────────────────────────────────────

    def _map_extensive(
        self,
        values: np.ndarray,
        n_atoms: int,
        weights: np.ndarray,
        what: str,
        compact: bool = False,
    ) -> np.ndarray:
        """Gather and reduce a non-positional per-atom vector field.

        No imaging: forces and velocities do not depend on the periodic image.
        """
        arr = np.asarray(values)
        if arr.ndim == 1:
            if arr.size != 3 * n_atoms:
                raise ValueError(
                    f"flat {what} have {arr.size} entries, expected {3 * n_atoms}."
                )
            arr = arr.reshape(n_atoms, 3)
        elif arr.shape != (n_atoms, 3):
            raise ValueError(
                f"{what} must have shape ({n_atoms}, 3) or ({3 * n_atoms},), "
                f"got {arr.shape}."
            )
        self._compact_from(arr, compact)
        self._gather(self._compact, self._csr_cols, self._entries)
        return self._weighted_rows(weights)

    def _map_fitted_forces(self, values: np.ndarray, n_atoms: int, compact: bool) -> np.ndarray:
        """Apply the attached compressed force operator to one AA force frame."""
        arr = np.asarray(values)
        if arr.ndim == 1:
            if arr.size != 3 * n_atoms:
                raise ValueError(f"flat forces have {arr.size} entries, expected {3 * n_atoms}.")
            arr = arr.reshape(n_atoms, 3)
        elif arr.shape != (n_atoms, 3):
            raise ValueError(f"forces must have shape ({n_atoms}, 3) or ({3 * n_atoms},), got {arr.shape}.")
        self._compact_from(arr, compact)
        assert self.spec._force_operator is not None
        matrices, atom_positions, site_positions, _, _, _, _ = self.spec._force_operator
        out = np.empty((self.spec.n_sites, 3), dtype=self.out_dtype)
        for matrix, atoms, sites in zip(
            matrices, atom_positions, site_positions,
        ):
            out[sites] = np.einsum(
                "cf,ifd->icd",
                matrix,
                self._compact[atoms],
                optimize=True,
            )
        return out

    def _compact_from(self, source: np.ndarray, compact: bool) -> None:
        """Fill the compact buffer, gathering from a full frame unless pre-gathered."""
        if compact:
            np.copyto(self._compact, source, casting="same_kind")
        else:
            self._gather(source, self._atom_indices, self._compact)

    @staticmethod
    def _gather(source: np.ndarray, indices: np.ndarray, out: np.ndarray) -> None:
        """``out[:] = source[indices]``, reusing ``out``.

        ``np.take(..., out=)`` refuses a dtype-crossing gather, and MDAnalysis
        hands out float32 frames while the default working precision is float64,
        so the mismatched case goes through an assignment that casts.
        """
        if source.dtype == out.dtype:
            np.take(source, indices, axis=0, out=out)
        else:
            np.copyto(out, np.take(source, indices, axis=0), casting="same_kind")

    def _weighted_rows(self, weights: np.ndarray) -> np.ndarray:
        """Weight ``_entries`` in place and sum each site's CSR row."""
        self._entries *= weights
        summed = np.add.reduceat(self._entries, self._row_starts, axis=0)
        return summed.astype(self.out_dtype, copy=False)

    def _image_against(
        self,
        compact: np.ndarray,
        reference: np.ndarray,
        box: np.ndarray,
    ) -> None:
        """Make ``compact`` whole in place, each atom imaged against its reference."""
        refs = np.take(compact, reference, axis=0)
        compact -= refs
        self._reduce_images(compact, box)
        compact += refs

    def _reduce_images(self, displacements: np.ndarray, box: np.ndarray) -> None:
        """Apply the configured minimum-image reduction in place."""
        if self.unwrap == "deprecated":
            self._single_shift(displacements, box)
            return
        if (
            self._image_scratch is None
            or self._image_scratch.shape != displacements.shape
            or self._image_scratch.dtype != displacements.dtype
        ):
            self._image_scratch = np.empty_like(displacements)
        minimum_image_displacements(
            displacements,
            box,
            triclinic=self.triclinic,
            scratch=self._image_scratch,
        )

    @staticmethod
    def _single_shift(displacements: np.ndarray, box: np.ndarray) -> None:
        """OpenMSCG's at-most-one-box-length shift, reproduced exactly.

        ``cgmap`` applies ``d += (d/L > 0.5) * -L`` then ``d += (d/L < -0.5) * L``,
        which is ``d -= L * clip(rint(d/L), -1, 1)`` (both leave ``|d/L| == 0.5``
        alone). Triclinic tilt is ignored, exactly as it is there.
        """
        kind, lengths, _ = classify_box(box)
        if kind == "none":
            return
        edges = lengths.astype(displacements.dtype, copy=False)
        shifts = np.rint(displacements / edges)
        np.clip(shifts, -1.0, 1.0, out=shifts)
        displacements -= shifts * edges

    # ── convenience ───────────────────────────────────────────────

    def site_masses(self) -> np.ndarray:
        """Per-site CG mass: ``cg-topology`` masses when given, else summed ``x-weight``."""
        return self.spec.site_masses_array()

    def site_type_ids(self) -> np.ndarray:
        """1-based CG site type ids, ``(n_sites,)``."""
        return self.spec.site_type_ids

    def mapped_shape(self) -> Tuple[int, int]:
        """``(n_sites, 3)``, the shape of every array this mapper returns."""
        return (self.spec.n_sites, 3)
