# AceCG/potentials/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Generator, List, Tuple
import copy as cp
import numpy as np


def IteratePotentials(
    forcefield: Dict,
) -> Generator[Tuple[object, "BasePotential"], None, None]:
    """Yield ``(key, potential)`` pairs from a forcefield dict.

    Handles both old-style ``Dict[K, BasePotential]`` and
    new-style ``Dict[K, List[BasePotential]]`` containers.
    This is a generator — no intermediate list is allocated.
    
    Note that when one only needs to iterate over the pair types (only the keys),
    one can still use the simple `for key, val in forcefield.items()` to iterate over the keys
    without invoking this generator.
    """
    for key, val in forcefield.items():
        if isinstance(val, list):
            for pot in val:
                yield key, pot
        else:
            yield key, val


class BasePotential(ABC):
    """Abstract interface implemented by all AceCG potential functions.

    Subclasses store their optimizable parameters in ``self._params`` and
    expose derivative method names through ``_dparam_names`` and
    ``_d2param_names``. Trainers use this common interface to assemble energy
    gradients, force Jacobians, and parameter metadata without knowing the
    analytic form of each potential.
    """

    def __init__(self, bonded: bool = False):
        """Initialize shared metadata slots for a potential subclass."""
        self.bonded = bool(bonded)
        self.__dict__["_params"] = None
        self._param_names = None
        self._dparam_names = None
        self._d2param_names = None
        self._params_to_scale = None
        self._df_dparam_names = None
        self._d2param_dr_names = None
        self._param_mask = None
        self._param_bounds_lb = None
        self._param_bounds_ub = None
        self._metadata_version = 0

    @abstractmethod
    def value(self, r: np.ndarray) -> np.ndarray:
        """Compute potential energy at coordinates or distances.

        Parameters
        ----------
        r : np.ndarray
            Array-like coordinates for the interaction, usually pair distances
            for pair potentials or scalar bond/angle values for bonded terms.

        Returns
        -------
        np.ndarray
            Potential energy evaluated elementwise at ``r``.
        """
        pass

    @abstractmethod
    def force(self, r: np.ndarray) -> np.ndarray:
        """Compute scalar force ``-dU/dr`` at coordinates or distances.

        Parameters
        ----------
        r : np.ndarray
            Array-like coordinates matching the convention of :meth:`value`.

        Returns
        -------
        np.ndarray
            Force values with the same broadcast shape as ``r``.
        """
        pass

    # Common method
    def param_names(self) -> List[str]:
        """Return ordered names for optimizable parameters.

        Returns
        -------
        list[str]
            Names ordered exactly like :meth:`get_params`.
        """
        assert self._param_names is not None
        return self._param_names
    
    def dparam_names(self) -> List[str]:
        """Return first-derivative method names used by :meth:`energy_grad`.

        Returns
        -------
        list[str]
            Method names whose callables evaluate ``dU/dtheta_i``.
        """
        assert self._dparam_names is not None
        return self._dparam_names
    
    def d2param_names(self) -> List[List[str]]:
        """Return second-derivative method names used for Hessian assembly.

        Returns
        -------
        list[list[str]]
            Square matrix of method names for ``d2U/dtheta_i dtheta_j``.
        """
        assert self._d2param_names is not None
        return self._d2param_names

    def df_dparam_names(self) -> List[str]:
        """Return names for d(force)/d(param) channels used by iterative FM."""
        names = getattr(self, "_df_dparam_names", None)
        if names is None:
            return []
        return names

    def d2param_dr_names(self) -> List[List[str]]:
        """Return names for d2(force)/d(param_j)d(param_k) channels."""
        if self._d2param_dr_names is None:
            return []
        return self._d2param_dr_names
    
    def n_params(self) -> int:
        """Return the number of optimizable parameters in this potential."""
        assert self._params is not None
        return len(self._params)

    def get_params(self) -> np.ndarray:
        """Return current parameter values.

        Returns
        -------
        np.ndarray
            One-dimensional copy of the potential parameter vector.
        """
        assert self._params is not None
        return self._params.copy()

    def set_params(self, new_params: np.ndarray):
        """Replace the potential parameter vector.

        Parameters
        ----------
        new_params : np.ndarray
            New parameter values ordered like :meth:`param_names`.
        """
        # if self._params is not None:
        #     assert len(new_params) == len(self._params)
        new_params = np.asarray(new_params, dtype=float).reshape(-1)
        self._params = new_params.copy()

    @property
    def metadata_version(self) -> int:
        """Monotonic metadata version for potential-local masks and bounds."""
        return int(self._metadata_version)

    def _bump_metadata_version(self) -> None:
        self._metadata_version = int(getattr(self, "_metadata_version", 0)) + 1

    @property
    def param_mask(self) -> np.ndarray:
        """Potential-local trainability mask.

        ``True`` entries are optimizer-active. ``Forcefield`` concatenates
        these local masks when constructing the global optimizer mask.
        """
        n = self.n_params()
        if self._param_mask is not None and self._param_mask.shape == (n,):
            return self._param_mask.copy()
        return np.ones(n, dtype=bool)

    @param_mask.setter
    def param_mask(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        n = self.n_params()
        if mask.shape != (n,):
            raise ValueError(f"param_mask shape must be ({n},), got {mask.shape}")
        self._param_mask = mask.copy()
        self._bump_metadata_version()

    @property
    def param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Potential-local lower/upper bounds for optimizable parameters."""
        n = self.n_params()
        lb = (
            self._param_bounds_lb.copy()
            if self._param_bounds_lb is not None and self._param_bounds_lb.shape == (n,)
            else np.full(n, -np.inf, dtype=float)
        )
        ub = (
            self._param_bounds_ub.copy()
            if self._param_bounds_ub is not None and self._param_bounds_ub.shape == (n,)
            else np.full(n, np.inf, dtype=float)
        )
        return lb, ub

    @param_bounds.setter
    def param_bounds(self, bounds: Tuple[np.ndarray, np.ndarray]) -> None:
        lb, ub = bounds
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        n = self.n_params()
        if lb.shape != (n,) or ub.shape != (n,):
            raise ValueError(
                f"param_bounds shape must be ({n},), got lb={lb.shape}, ub={ub.shape}"
            )
        if np.any(lb > ub):
            raise ValueError("param_bounds lower entries cannot exceed upper entries.")
        self._param_bounds_lb = lb.copy()
        self._param_bounds_ub = ub.copy()
        self._bump_metadata_version()

    def basis_values(self, r: np.ndarray) -> np.ndarray:
        """Return energy-side per-parameter basis values at r.

        This is not the force-side contract. Force callers must go through
        ``force_grad()`` so each potential class can decide how to optimize the
        Jacobian assembly.
        """
        names = self.dparam_names()
        r_arr = np.asarray(r, dtype=float)
        if names is None or not names:
            return np.empty(r_arr.shape + (0,), dtype=float)
        return self._stack_named_channels(names, r_arr)

    def basis_derivatives(self, r: np.ndarray) -> np.ndarray:
        """Return derivative of basis wrt r (finite-difference fallback)."""
        r = np.asarray(r, dtype=float)
        eps = 1.0e-6
        return (self.basis_values(r + eps) - self.basis_values(r - eps)) / (2.0 * eps)

    def energy_grad(self, r: np.ndarray) -> np.ndarray:
        """Return dU/dtheta evaluated at r with shape ``r.shape + (n_params,)``."""
        return self._stack_named_channels(self.dparam_names(), r)

    def energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        """Return ``dU/dtheta`` summed over the last coordinate axis.

        Subclasses can override this to avoid materializing a full
        ``(..., n_terms, n_params)`` array when the downstream caller only
        needs the reduced gradient vector.
        """
        grad = self.energy_grad(r)
        grad = np.asarray(grad, dtype=float)
        if grad.ndim == 1:
            return grad.reshape(-1)
        summed = grad.sum(axis=-2)
        return np.asarray(summed, dtype=float)

    def energy_grad_sum_by_sample(
        self,
        r: np.ndarray,
        *,
        active: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return per-sample ``dU/dtheta`` sums without requiring term gradients.

        ``r`` is interpreted as ``sample_shape + (n_terms,)``. The output has
        shape ``sample_shape + (n_params,)``. The optional ``active`` mask lets
        callers exclude nonfinite or cutoff-filtered terms before dispatching
        to a potential's optimized one-dimensional ``energy_grad_sum()``.
        """
        r_arr = np.asarray(r, dtype=float)
        n_params = int(self.n_params())
        active_arr = None
        if active is not None:
            active_arr = np.asarray(active, dtype=bool)
            if active_arr.shape != r_arr.shape:
                active_arr = np.broadcast_to(active_arr, r_arr.shape)

        if r_arr.ndim <= 1:
            rv = r_arr.reshape(-1)
            if active_arr is not None:
                rv = r_arr[active_arr]
            return np.asarray(self.energy_grad_sum(rv), dtype=float).reshape(n_params)

        flat = r_arr.reshape((-1, r_arr.shape[-1]))
        flat_active = None if active_arr is None else active_arr.reshape(flat.shape)
        out = np.zeros((flat.shape[0], n_params), dtype=float)
        for i, row in enumerate(flat):
            rv = row if flat_active is None else row[flat_active[i]]
            if rv.size:
                out[i] = np.asarray(
                    self.energy_grad_sum(rv),
                    dtype=float,
                ).reshape(n_params)
        return out.reshape(r_arr.shape[:-1] + (n_params,))

    def gauge_free_energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        """Return summed gradients with parameter-dependent gauge shifts removed.

        Most potentials have no parameter-dependent additive gauge term, so the
        gauge-free channel is identical to the physical energy gradient.
        """
        return self.energy_grad_sum(r)

    def gauge_free_energy_grad_sum_by_sample(
        self,
        r: np.ndarray,
        *,
        active: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return per-sample gauge-free gradient sums."""
        r_arr = np.asarray(r, dtype=float)
        n_params = int(self.n_params())
        active_arr = None
        if active is not None:
            active_arr = np.asarray(active, dtype=bool)
            if active_arr.shape != r_arr.shape:
                active_arr = np.broadcast_to(active_arr, r_arr.shape)

        if r_arr.ndim <= 1:
            rv = r_arr.reshape(-1)
            if active_arr is not None:
                rv = r_arr[active_arr]
            return np.asarray(
                self.gauge_free_energy_grad_sum(rv),
                dtype=float,
            ).reshape(n_params)

        flat = r_arr.reshape((-1, r_arr.shape[-1]))
        flat_active = None if active_arr is None else active_arr.reshape(flat.shape)
        out = np.zeros((flat.shape[0], n_params), dtype=float)
        for i, row in enumerate(flat):
            rv = row if flat_active is None else row[flat_active[i]]
            if rv.size:
                out[i] = np.asarray(
                    self.gauge_free_energy_grad_sum(rv),
                    dtype=float,
                ).reshape(n_params)
        return out.reshape(r_arr.shape[:-1] + (n_params,))

    def force_grad(self, r: np.ndarray) -> np.ndarray:
        """Return dF/dtheta evaluated at r.

        Subclasses may return either a dense ``ndarray`` or a sparse matrix
        object when that materially improves performance.
        """
        names = self.df_dparam_names()
        if names:
            return self._stack_named_channels(names, r)
        return self._finite_difference_param_jacobian(self.force, r)

    @abstractmethod
    def is_param_linear(self) -> np.ndarray:
        """Return a per-parameter boolean mask for linear optimization channels."""
        raise NotImplementedError

    def is_gauge_free_energy_grad_cacheable(self) -> np.ndarray:
        """Return a per-parameter mask for AA-cacheable gauge-free channels."""
        return np.asarray(self.is_param_linear(), dtype=bool)

    def _stack_named_channels(self, names: List[str], r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        if not names:
            return np.empty(r_arr.shape + (0,), dtype=float)
        cols = []
        for name in names:
            values = np.asarray(getattr(self, name)(r_arr), dtype=float)
            if values.shape != r_arr.shape:
                try:
                    values = np.broadcast_to(values, r_arr.shape)
                except ValueError as exc:
                    raise ValueError(
                        f"{type(self).__name__}.{name} returned shape {values.shape}, "
                        f"expected {r_arr.shape}"
                    ) from exc
            if values.shape != r_arr.shape:
                raise ValueError(
                    f"{type(self).__name__}.{name} returned shape {values.shape}, "
                    f"expected {r_arr.shape}"
                )
            cols.append(values)
        return np.stack(cols, axis=-1)

    def _finite_difference_param_jacobian(self, fn, r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        r_flat = r_arr.reshape(-1)
        params0 = self.get_params()
        scale = np.maximum(1.0, np.abs(params0))
        jac = np.empty((r_flat.size, params0.size), dtype=float)
        try:
            for idx in range(params0.size):
                step = 1.0e-6 * scale[idx]
                params_plus = params0.copy()
                params_minus = params0.copy()
                params_plus[idx] += step
                params_minus[idx] -= step
                self.set_params(params_plus)
                values_plus = np.asarray(fn(r_flat), dtype=float).reshape(-1)
                self.set_params(params_minus)
                values_minus = np.asarray(fn(r_flat), dtype=float).reshape(-1)
                jac[:, idx] = (values_plus - values_minus) / (2.0 * step)
        finally:
            self.set_params(params0)
        return jac.reshape(r_arr.shape + (params0.size,))

    def get_scaled_potential(self, z):  # From Ace
        """Return a deep copy whose scalable params are multiplied by ``z``.

        Potentials that expose ``self._params_to_scale`` (a list of parameter
        indices) have those entries scaled by ``z`` on the returned copy; this
        is used by the VP-growth driver to gradually turn on target
        interactions. Potentials with ``_params_to_scale is None`` return an
        unchanged deep copy.
        """
        if self._params_to_scale is None:
            return cp.deepcopy(self)

        copied = cp.deepcopy(self)

        for idx in self._params_to_scale:
            copied._params[idx] = self._params[idx] * z

        return copied
