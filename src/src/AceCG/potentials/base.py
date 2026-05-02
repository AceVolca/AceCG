"""Define staged AceCG potential interfaces."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Sequence, Tuple, Union
import copy as cp
import numpy as np

if TYPE_CHECKING:
    from ..topology.types import InteractionKey


def IteratePotentials(
    forcefield: Dict[
        "InteractionKey",
        Union["BasePotential", List["BasePotential"]],
    ],
) -> Generator[Tuple["InteractionKey", "BasePotential"], None, None]:
    """Yield ``(key, potential)`` pairs from a forcefield mapping.

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
        self._params: Optional[np.ndarray]
        vars(self)["_params"] = None
        self._param_names: Optional[List[str]] = None
        self._dparam_names: Optional[List[str]] = None
        self._d2param_names: Optional[List[List[str]]] = None
        self._params_to_scale: Optional[Sequence[int]] = None
        self._df_dparam_names: Optional[List[str]] = None
        self._d2param_dr_names: Optional[List[List[str]]] = None

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
        self._params = new_params.copy()

    def basis_values(self, r: np.ndarray) -> np.ndarray:
        """Return energy-side per-parameter basis values at r.

        This is not the force-side contract. Force callers must go through
        ``force_grad()`` so each potential class can decide how to optimize the
        Jacobian assembly.
        """
        names = self.dparam_names()
        if names is None:
            return np.empty((len(np.asarray(r)), 0), dtype=float)
        r = np.asarray(r, dtype=float)
        cols = [np.asarray(getattr(self, name)(r), dtype=float) for name in names]
        if not cols:
            return np.empty((r.size, 0), dtype=float)
        return np.vstack(cols).T

    def basis_derivatives(self, r: np.ndarray) -> np.ndarray:
        """Return derivative of basis wrt r (finite-difference fallback)."""
        r = np.asarray(r, dtype=float)
        eps = 1.0e-6
        return (self.basis_values(r + eps) - self.basis_values(r - eps)) / (2.0 * eps)

    def energy_grad(self, r: np.ndarray) -> np.ndarray:
        """Return dU/dtheta evaluated at r with shape (n_samples, n_params)."""
        return self._stack_named_channels(self.dparam_names(), r)

    def energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        """Return the summed energy gradient ``Σ_samples dU/dtheta``.

        Subclasses can override this to avoid materializing a full
        ``(n_samples, n_params)`` array when the downstream caller only needs
        the reduced gradient vector.
        """
        grad = self.energy_grad(r)
        summed = grad.sum(axis=0) if hasattr(grad, "sum") else np.sum(grad, axis=0)
        return np.asarray(summed, dtype=float).reshape(-1)

    def gauge_free_energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        """Return the summed gauge-free energy gradient ``Σ_samples dU/dtheta``.

        By default the gauge-free channel uses the same summed derivative as
        the physical energy. Potentials with parameter-dependent additive
        gauge terms can override this to drop pure-constant contributions that
        do not affect ensemble statistics.
        """
        return self.energy_grad_sum(r)

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
        """Return a per-parameter mask for AA-cacheable gauge-free channels.

        The default contract matches the physical linearity mask. Potentials
        can override this when the gauge-free training gradient is
        parameter-independent even if the physical energy gradient carries a
        gauge term.
        """
        return np.asarray(self.is_param_linear(), dtype=bool)

    def _stack_named_channels(self, names: List[str], r: np.ndarray) -> np.ndarray:
        r_flat = np.asarray(r, dtype=float).reshape(-1)
        if not names:
            return np.empty((r_flat.size, 0), dtype=float)
        cols = []
        for name in names:
            values = np.asarray(getattr(self, name)(r_flat), dtype=float).reshape(-1)
            if values.shape != (r_flat.size,):
                raise ValueError(
                    f"{type(self).__name__}.{name} returned shape {values.shape}, "
                    f"expected {(r_flat.size,)}"
                )
            cols.append(values)
        return np.column_stack(cols)

    def _finite_difference_param_jacobian(self, fn, r: np.ndarray) -> np.ndarray:
        r_flat = np.asarray(r, dtype=float).reshape(-1)
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
        return jac

    # From Ace
    def get_scaled_potential(self, z):
        """Return a deep copy whose scalable params are multiplied by ``z``.

        Potentials that expose ``self._params_to_scale`` (a list of parameter
        indices) have those entries scaled by ``z`` on the returned copy; this
        is used by the VP-growth driver to gradually turn on target
        interactions. Potentials with ``_params_to_scale is None`` return an
        unchanged deep copy.
        """
        params_to_scale = self._params_to_scale
        if params_to_scale is None:
            return cp.deepcopy(self)

        copied = cp.deepcopy(self)
        if self._params is None or copied._params is None:
            raise ValueError("Cannot scale a potential before its parameter vector is initialized.")

        for idx in params_to_scale:
            copied._params[idx] = self._params[idx] * z

        return copied
