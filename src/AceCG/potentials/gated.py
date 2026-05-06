"""Hard-concrete gates for interaction-level sparsification.

The wrapper keeps the underlying potential parameters unscaled for training
and appends one trainable ``log_alpha`` gate parameter. Runtime energy/force
queries are multiplied by the current gate value, so existing first-order
analytic trainers can consume the expanded derivatives without special cases.
"""

from __future__ import annotations

import copy
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np

from .base import BasePotential


def _sigmoid(x):
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    pos = arr >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def _potential_bounds(potential: BasePotential) -> Tuple[np.ndarray, np.ndarray]:
    bounds = getattr(potential, "param_bounds", None)
    if bounds is None:
        n = potential.n_params()
        return np.full(n, -np.inf, dtype=float), np.full(n, np.inf, dtype=float)
    if callable(bounds):
        bounds = bounds()
    lb, ub = bounds
    return np.asarray(lb, dtype=float).reshape(-1), np.asarray(ub, dtype=float).reshape(-1)


def _potential_mask(potential: BasePotential) -> np.ndarray:
    mask = getattr(potential, "param_mask", None)
    if mask is None:
        return np.ones(potential.n_params(), dtype=bool)
    if callable(mask):
        mask = mask()
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.shape != (potential.n_params(),):
        raise ValueError(
            f"param_mask shape mismatch for {type(potential).__name__}: "
            f"expected {(potential.n_params(),)}, got {mask.shape}"
        )
    return mask


class GatedPotential(BasePotential):
    """Wrap a potential with one hard-concrete interaction gate.

    Parameters
    ----------
    potential : BasePotential
        Potential whose whole interaction contribution is controlled by this
        gate.
    log_alpha : float, default=3.0
        Trainable hard-concrete location parameter. Positive values initialize
        the gate mostly open.
    beta : float, default=0.5
        Concrete temperature.
    gamma, zeta : float, default=(-0.1, 1.1)
        Stretch interval endpoints. They must satisfy ``gamma < 0 < 1 < zeta``.
    gate_mode : {"deterministic", "sample"}, default="deterministic"
        Deterministic mode uses the noise-free hard-concrete mean-path gate.
        Sample mode uses the stored uniform sample; call :meth:`sample_gate`
        before computing a batch to refresh it.
    log_alpha_bounds : tuple, default=(-20, 20)
        Bounds advertised to ``Forcefield`` for optimizer clamping.
    name : str, optional
        Human-readable label used in diagnostics.
    """

    def __init__(
        self,
        potential: BasePotential,
        *,
        log_alpha: float = 3.0,
        beta: float = 0.5,
        gamma: float = -0.1,
        zeta: float = 1.1,
        gate_mode: str = "deterministic",
        log_alpha_bounds: Tuple[float, float] = (-20.0, 20.0),
        name: Optional[str] = None,
    ) -> None:
        super().__init__(bonded=bool(getattr(potential, "bonded", False)))
        if gamma >= 0.0 or zeta <= 1.0 or not gamma < zeta:
            raise ValueError("Hard-concrete gates require gamma < 0 < 1 < zeta.")
        if beta <= 0.0:
            raise ValueError("beta must be positive.")
        if gate_mode not in {"deterministic", "sample"}:
            raise ValueError("gate_mode must be 'deterministic' or 'sample'.")

        self.potential = copy.deepcopy(potential)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.gate_mode = gate_mode
        self.log_alpha_bounds = (float(log_alpha_bounds[0]), float(log_alpha_bounds[1]))
        self.name = name
        self._sample_u: Optional[float] = None

        self._params = np.concatenate(
            [self.potential.get_params(), np.array([float(log_alpha)], dtype=float)]
        )
        self._params_to_scale = None
        self._refresh_metadata()
        lb_inner, ub_inner = _potential_bounds(self.potential)
        self.param_bounds = (
            np.concatenate([lb_inner, [self.log_alpha_bounds[0]]]),
            np.concatenate([ub_inner, [self.log_alpha_bounds[1]]]),
        )
        self.param_mask = np.concatenate(
            [_potential_mask(self.potential), np.array([True], dtype=bool)]
        )

    def _refresh_metadata(self) -> None:
        self._param_names = list(self.potential.param_names()) + ["log_alpha"]
        self._dparam_names = []
        self._d2param_names = []
        if hasattr(self.potential, "cutoff"):
            self.cutoff = getattr(self.potential, "cutoff")

    @property
    def log_alpha(self) -> float:
        return float(self._params[-1])

    def inner_params(self) -> np.ndarray:
        """Return the unscaled parameters of the wrapped potential."""
        return self.potential.get_params()

    def set_params(self, new_params: np.ndarray):
        params = np.asarray(new_params, dtype=float).reshape(-1)
        expected = self.potential.n_params() + 1
        if params.shape != (expected,):
            raise ValueError(f"Expected {expected} parameters, got {params.shape}.")
        self.potential.set_params(params[:-1])
        self._params = params.copy()
        self._refresh_metadata()

    def get_params(self) -> np.ndarray:
        self._params[:-1] = self.potential.get_params()
        return self._params.copy()

    def n_params(self) -> int:
        return self.potential.n_params() + 1

    def sample_gate(self, rng: Optional[np.random.Generator] = None) -> float:
        """Refresh the stored hard-concrete uniform sample and return ``z``."""
        if rng is None:
            rng = np.random.default_rng()
        eps = np.finfo(float).eps
        self._sample_u = float(np.clip(rng.uniform(), eps, 1.0 - eps))
        self.gate_mode = "sample"
        return self.gate_value()

    def clear_gate_sample(self, *, gate_mode: str = "deterministic") -> None:
        """Drop the stored sample and switch to deterministic gate evaluation."""
        if gate_mode not in {"deterministic", "sample"}:
            raise ValueError("gate_mode must be 'deterministic' or 'sample'.")
        self._sample_u = None
        self.gate_mode = gate_mode

    def active_probability(self) -> float:
        """Return ``P(z > 0)`` under the hard-concrete distribution."""
        shift = self.beta * np.log(-self.gamma / self.zeta)
        return float(_sigmoid(self.log_alpha - shift))

    def active_probability_grad(self) -> float:
        """Return ``d P(z > 0) / d log_alpha``."""
        p = self.active_probability()
        return float(p * (1.0 - p))

    def _raw_concrete(self) -> float:
        if self.gate_mode == "sample":
            if self._sample_u is None:
                self.sample_gate()
            assert self._sample_u is not None
            u = self._sample_u
            logit_noise = np.log(u) - np.log1p(-u)
            return float(_sigmoid((logit_noise + self.log_alpha) / self.beta))
        return float(_sigmoid(self.log_alpha))

    def _stretched_gate(self) -> Tuple[float, float]:
        s = self._raw_concrete()
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        if s_bar <= 0.0:
            return 0.0, 0.0
        if s_bar >= 1.0:
            return 1.0, 0.0
        if self.gate_mode == "sample":
            dz = (self.zeta - self.gamma) * s * (1.0 - s) / self.beta
        else:
            dz = (self.zeta - self.gamma) * s * (1.0 - s)
        return float(s_bar), float(dz)

    def gate_value(self) -> float:
        z, _ = self._stretched_gate()
        return z

    def gate_grad_log_alpha(self) -> float:
        _, dz = self._stretched_gate()
        return dz

    def value(self, r: np.ndarray) -> np.ndarray:
        return self.gate_value() * self.potential.value(r)

    def force(self, r: np.ndarray) -> np.ndarray:
        return self.gate_value() * self.potential.force(r)

    def energy_grad(self, r: np.ndarray) -> np.ndarray:
        r_shape = np.asarray(r, dtype=float).shape
        r_arr = np.asarray(r, dtype=float).reshape(-1)
        z = self.gate_value()
        dz = self.gate_grad_log_alpha()
        inner = np.asarray(self.potential.energy_grad(r_arr), dtype=float)
        inner = inner.reshape(-1, self.potential.n_params())
        gate_col = np.asarray(self.potential.value(r_arr), dtype=float).reshape(-1, 1) * dz
        out = np.column_stack([z * inner, gate_col])
        return out.reshape(r_shape + (self.n_params(),))

    def energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        r_in = np.asarray(r, dtype=float)
        if r_in.ndim > 1:
            return self.energy_grad_sum_by_sample(r_in)
        r_arr = r_in.reshape(-1)
        z = self.gate_value()
        dz = self.gate_grad_log_alpha()
        inner_sum = np.asarray(self.potential.energy_grad_sum(r_arr), dtype=float).reshape(-1)
        gate_sum = float(np.sum(self.potential.value(r_arr))) * dz
        return np.concatenate([z * inner_sum, np.array([gate_sum], dtype=float)])

    def force_grad(self, r: np.ndarray) -> np.ndarray:
        r_shape = np.asarray(r, dtype=float).shape
        r_arr = np.asarray(r, dtype=float).reshape(-1)
        z = self.gate_value()
        dz = self.gate_grad_log_alpha()
        inner = self.potential.force_grad(r_arr)
        if hasattr(inner, "toarray"):
            inner = inner.toarray()
        inner = np.asarray(inner, dtype=float)
        inner = inner.reshape(-1, self.potential.n_params())
        gate_col = np.asarray(self.potential.force(r_arr), dtype=float).reshape(-1, 1) * dz
        out = np.column_stack([z * inner, gate_col])
        return out.reshape(r_shape + (self.n_params(),))

    def is_param_linear(self) -> np.ndarray:
        inner = np.asarray(self.potential.is_param_linear(), dtype=bool).reshape(-1)
        return np.concatenate([inner, np.array([False], dtype=bool)])

    def d2param_names(self):
        raise NotImplementedError(
            "GatedPotential currently supports first-order optimization only."
        )

    def lammps_params(self) -> np.ndarray:
        """Return gate-scaled parameters for analytic LAMMPS coefficient lines."""
        scaled = self.potential.get_scaled_potential(self.gate_value())
        return scaled.get_params()

    def exported_potential(self) -> BasePotential:
        """Return a copy of the wrapped potential scaled by the current gate."""
        return self.potential.get_scaled_potential(self.gate_value())


def iter_gated_potentials(forcefield) -> Iterator[Tuple[object, GatedPotential]]:
    """Yield ``(InteractionKey, GatedPotential)`` pairs from a forcefield."""
    for key, value in forcefield.items():
        pots = value if isinstance(value, list) else [value]
        for pot in pots:
            if isinstance(pot, GatedPotential):
                yield key, pot


def sample_L0_gates(
    forcefield,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> dict:
    """Sample all L0 gates in a forcefield and return ``{key: z}`` diagnostics."""
    if rng is None:
        rng = np.random.default_rng(seed)
    return {key: pot.sample_gate(rng) for key, pot in iter_gated_potentials(forcefield)}


def set_L0_gates_deterministic(forcefield) -> dict:
    """Switch all L0 gates to deterministic mode and return ``{key: z}``."""
    values = {}
    for key, pot in iter_gated_potentials(forcefield):
        pot.clear_gate_sample(gate_mode="deterministic")
        values[key] = pot.gate_value()
    return values


def wrap_forcefield_with_L0_gates(
    forcefield,
    *,
    interaction_keys: Optional[Iterable[object]] = None,
    log_alpha: float = 3.0,
    beta: float = 0.5,
    gamma: float = -0.1,
    zeta: float = 1.1,
    gate_mode: str = "deterministic",
    inplace: bool = False,
):
    """Return a forcefield whose selected potentials are hard-concrete gated."""
    ff = forcefield if inplace else copy.deepcopy(forcefield)
    selected = None if interaction_keys is None else set(interaction_keys)
    for key in list(ff.keys()):
        if selected is not None and key not in selected:
            continue
        values = ff[key]
        pots = values if isinstance(values, list) else [values]
        wrapped = []
        for pot in pots:
            if isinstance(pot, GatedPotential):
                wrapped.append(pot)
            else:
                name = key.label() if hasattr(key, "label") else str(key)
                wrapped.append(
                    GatedPotential(
                        pot,
                        log_alpha=log_alpha,
                        beta=beta,
                        gamma=gamma,
                        zeta=zeta,
                        gate_mode=gate_mode,
                        name=name,
                    )
                )
        ff[key] = wrapped
    return ff
