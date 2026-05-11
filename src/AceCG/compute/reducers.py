"""One-pass post step helpers and shared compute math.

Frame-weight semantics
----------------------
AceCG's one-pass engine supports two distinct frame-weight modes:

1. **A-priori frame weights** — known at sampling time (e.g. Boltzmann
   reweighting with a fixed target ensemble). Weights are passed through
   ``spec['frame_weight']``; each rank scales per-frame observables by
   ``frame_weight`` inside ``consume_*_frame`` and the reducer emits
   *averaged* statistics. The trainer consumes the averages directly.

2. **A-posteriori frame weights** — known only after sampling (e.g. WHAM
   reweighting against a collection of biased simulations). In this mode the
   caller sets ``step['reduce_stack'] = True``; the reducer skips averaging
   and emits the full per-frame stack of observables under ``*_frame`` keys
   plus ``frame_ids``. The trainer then receives the stack together with an
   external weight vector and computes its own weighted average.

Every reducer mode conforms to the same uniform operator bundle, exposed via
``MODE_OPS`` and the top-level dispatch helpers below:

- ``init(step, ff, topo) -> state``
- ``request(step) -> Dict[str, bool]``
- ``consume(state, payload, *, frame_weight, reference_force) -> None``
- ``local_partials(state) -> Dict[str, Any]``
- ``reduce_plan(step) -> Dict[str, tuple[str, ...]]``
- ``finalize(state) -> Dict[str, Any]``

Step-time flags (``need_hessian``, ``reduce_stack``, ``mode``/``beta`` for
cdfm_zbx, ``y_eff`` for cdfm_zbx, ...) are resolved once inside ``init`` and
stashed in ``state``. Downstream hooks read them from ``state`` and never
re-parse ``step``; ``consume`` receives only genuine per-frame runtime
quantities (``frame_weight``, ``reference_force``, and the frame payload).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence, Tuple

import numpy as np


if TYPE_CHECKING:
    from ..topology.forcefield import Forcefield
    from ..topology.topology_array import TopologyArrays


_REQUEST_FLAGS: Tuple[str, ...] = (
    "need_energy_value",
    "need_energy_grad",
    "need_energy_hessian",
    "need_energy_grad_outer",
    "need_gauge_free_energy_grad",
    "need_gauge_free_energy_grad_outer",
    "need_force_value",
    "need_force_grad",
    "need_fm_stats",
    "need_reference_force",
    "need_frame_cache",
)


def canonical_step_mode(step_or_mode: Dict[str, Any] | str) -> str:
    """Return the registry key for ``step['step_mode']`` (e.g. ``cdrem`` -> ``rem``)."""
    if isinstance(step_or_mode, dict):
        mode = str(step_or_mode["step_mode"]).strip().lower()
    else:
        mode = str(step_or_mode).strip().lower()
    if mode == "cdrem":
        return "rem"
    return mode


# ─────────────────────────────────────────────────────────────────────────────
# Unified per-mode operator bundle + registry dispatch
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReducerOps:
    """Uniform contract each step_mode must implement."""

    init: Callable[[Dict[str, Any], "Forcefield", "TopologyArrays"], Dict[str, Any]]
    request: Callable[[Dict[str, Any]], Dict[str, bool]]
    consume: Callable[..., None]
    local_partials: Callable[[Dict[str, Any]], Dict[str, Any]]
    reduce_plan: Callable[[Dict[str, Any]], Dict[str, Tuple[str, ...]]]
    finalize: Callable[[Dict[str, Any]], Dict[str, Any]]


def _ops(step: Dict[str, Any]) -> ReducerOps:
    """Resolve the :class:`ReducerOps` bundle registered for ``step['step_mode']``."""
    mode = canonical_step_mode(step)
    ops = MODE_OPS.get(mode)
    if ops is None:
        raise NotImplementedError(f"Unsupported step_mode {mode!r}.")
    return ops


def step_request(step: Dict[str, Any]) -> Dict[str, bool]:
    """Return the compute request flags needed by ``step``."""
    request: Dict[str, bool] = {flag: False for flag in _REQUEST_FLAGS}
    request.update(_ops(step).request(step))
    return request


def init_step_state(
    step: Dict[str, Any],
    forcefield_snapshot: "Forcefield",
    topology_arrays: "TopologyArrays",
) -> Dict[str, Any]:
    """Initialise a per-step accumulator; also stashes ``step_mode`` / ``reduce_stack``."""
    state = _ops(step).init(step, forcefield_snapshot, topology_arrays)
    state.setdefault("step_mode", canonical_step_mode(step))
    state.setdefault("reduce_stack", bool(step.get("reduce_stack", False)))
    return state


def consume_step_payload(
    step: Dict[str, Any],
    state: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    frame_weight: float,
    reference_force: np.ndarray | None,
) -> None:
    """Accumulate one reducer-ready payload into ``state`` for this step."""
    _ops(step).consume(
        state,
        payload,
        frame_weight=frame_weight,
        reference_force=reference_force,
    )


def local_step_partials(step: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the per-rank partial result dict ready for MPI reduction."""
    return _ops(step).local_partials(state)


def step_reduce_plan(step: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    """Return the ``{sum, max, stack, dict_sum, dict_update}`` reduce plan."""
    return _ops(step).reduce_plan(step)


def finalize_step_root(step: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Produce the root-rank finalized output from reduced step state."""
    # Re-inject per-step inputs that the reduce plan does not carry (e.g.
    # the rank-0 broadcast y_eff vector for cdfm_zbx). These are constant
    # across ranks and already live on the step dict.
    if "y_eff" in step and "y_eff" not in state:
        state = dict(state)
        state["y_eff"] = step["y_eff"]
    if canonical_step_mode(step) == "cdfm_zbx":
        for key in ("mode", "beta"):
            if key in step and key not in state:
                state = dict(state)
                state[key] = step[key]
    return _ops(step).finalize(state)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def slice_observed_rows(A_or_f: np.ndarray, real_site_indices: Sequence[int]) -> np.ndarray:
    """Helper: Slice flattened force rows down to the observed real-site subset."""
    arr = np.asarray(A_or_f)
    atoms = np.asarray(real_site_indices, dtype=np.int32).reshape(-1)
    rows = (atoms[:, None] * 3 + np.arange(3, dtype=np.int32)[None, :]).reshape(-1)
    if arr.ndim == 1:
        return arr[rows]
    if arr.ndim >= 2:
        return arr[rows, ...]
    raise ValueError("A_or_f must be at least one-dimensional")


# ─────────────────────────────────────────────────────────────────────────────
# FM
# ─────────────────────────────────────────────────────────────────────────────


def init_fm_state(
    step: Dict[str, Any],
    forcefield_snapshot: "Forcefield",
    topology_arrays: "TopologyArrays",
) -> Dict[str, Any]:
    del topology_arrays  # unused
    n_params = forcefield_snapshot.n_params()
    reduce_stack = bool(step.get("reduce_stack", False))
    state: Dict[str, Any] = {
        "n_params": int(n_params),
        "reduce_stack": reduce_stack,
    }
    if reduce_stack:
        state["JtJ_frames"] = []
        state["Jty_frames"] = []
        state["y_sumsq_frames"] = []
        state["Jtf_frames"] = []
        state["f_sumsq_frames"] = []
        state["fty_frames"] = []
        state["weight_frames"] = []
        state["frame_ids"] = []
        state["n_atoms_obs"] = 0
    else:
        state.update(
            {
                "JtJ_sum": np.zeros((n_params, n_params), dtype=np.float32),
                "Jty_sum": np.zeros(n_params, dtype=np.float32),
                "y_sumsq_sum": 0.0,
                "Jtf_sum": np.zeros(n_params, dtype=np.float32),
                "f_sumsq_sum": 0.0,
                "fty_sum": 0.0,
                "nframe": 0,
                "weight_sum": 0.0,
                "n_atoms_obs": 0,
            }
        )
    return state


def request_fm(step: Dict[str, Any]) -> Dict[str, bool]:
    del step
    return {"need_fm_stats": True, "need_reference_force": True}


def request_dsm(step: Dict[str, Any]) -> Dict[str, bool]:
    del step
    return {"need_fm_stats": True, "need_reference_force": False}


def _require_fm_partial(payload: Dict[str, Any], key: str, label: str) -> Dict[str, Any]:
    partial = payload.get(key)
    if partial is None:
        raise ValueError(f"FM {label} payload requires payload[{key!r}].")
    return dict(partial)


def _consume_fm_partial(
    state: Dict[str, Any],
    partial: Dict[str, Any],
    *,
    frame_idx: int,
) -> None:
    if state["reduce_stack"]:
        state["JtJ_frames"].append(np.asarray(partial["JtJ"], dtype=np.float32))
        state["Jty_frames"].append(np.asarray(partial["Jty"], dtype=np.float32))
        state["y_sumsq_frames"].append(float(partial["yty"]))
        state["Jtf_frames"].append(np.asarray(partial["Jtf"], dtype=np.float32))
        state["f_sumsq_frames"].append(float(partial["ftf"]))
        state["fty_frames"].append(float(partial["fTy"]))
        state["weight_frames"].append(float(partial["weight_sum"]))
        state["frame_ids"].append(int(frame_idx))
        state["n_atoms_obs"] = max(state["n_atoms_obs"], int(partial["n_atoms_obs"]))
        return
    state["JtJ_sum"] += np.asarray(partial["JtJ"], dtype=np.float32)
    state["Jty_sum"] += np.asarray(partial["Jty"], dtype=np.float32)
    state["Jtf_sum"] += np.asarray(partial["Jtf"], dtype=np.float32)
    state["y_sumsq_sum"] += float(partial["yty"])
    state["f_sumsq_sum"] += float(partial["ftf"])
    state["fty_sum"] += float(partial["fTy"])
    state["nframe"] += int(partial.get("n_frames", 1))
    state["weight_sum"] += float(partial["weight_sum"])
    state["n_atoms_obs"] = max(state["n_atoms_obs"], int(partial["n_atoms_obs"]))


def consume_fm_frame(
    state: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    frame_weight: float,
    reference_force: np.ndarray | None,
) -> None:
    del frame_weight, reference_force  # FM carries its own weight inside fm_stats
    partial = _require_fm_partial(payload, "fm_stats", "frame")
    _consume_fm_partial(state, partial, frame_idx=int(payload["frame_idx"]))


def local_partials_fm(state: Dict[str, Any]) -> Dict[str, Any]:
    if state["reduce_stack"]:
        n_params = int(state["n_params"])
        stack_2d = lambda arrs: (
            np.stack(arrs, axis=0).astype(np.float32, copy=False)
            if arrs
            else np.empty((0, n_params, n_params), dtype=np.float32)
        )
        stack_1d = lambda arrs: (
            np.stack(arrs, axis=0).astype(np.float32, copy=False)
            if arrs
            else np.empty((0, n_params), dtype=np.float32)
        )
        stack_scalar = lambda arrs: (
            np.asarray(arrs, dtype=np.float32)
            if arrs
            else np.empty((0,), dtype=np.float32)
        )
        return {
            "reduce_stack": True,
            "JtJ_frames": stack_2d(state["JtJ_frames"]),
            "Jty_frames": stack_1d(state["Jty_frames"]),
            "y_sumsq_frames": stack_scalar(state["y_sumsq_frames"]),
            "Jtf_frames": stack_1d(state["Jtf_frames"]),
            "f_sumsq_frames": stack_scalar(state["f_sumsq_frames"]),
            "fty_frames": stack_scalar(state["fty_frames"]),
            "weight_frames": stack_scalar(state["weight_frames"]),
            "frame_ids": np.asarray(state["frame_ids"], dtype=np.int64),
            "n_atoms_obs": int(state["n_atoms_obs"]),
        }
    return {
        "JtJ_sum": state["JtJ_sum"],
        "Jty_sum": state["Jty_sum"],
        "y_sumsq_sum": float(state["y_sumsq_sum"]),
        "Jtf_sum": state["Jtf_sum"],
        "f_sumsq_sum": float(state["f_sumsq_sum"]),
        "fty_sum": float(state["fty_sum"]),
        "nframe": int(state["nframe"]),
        "weight_sum": float(state["weight_sum"]),
        "n_atoms_obs": int(state["n_atoms_obs"]),
    }


def reduce_plan_fm(step: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    if bool(step.get("reduce_stack", False)):
        return {
            "sum": (),
            "max": ("n_atoms_obs",),
            "stack": (
                "JtJ_frames",
                "Jty_frames",
                "y_sumsq_frames",
                "Jtf_frames",
                "f_sumsq_frames",
                "fty_frames",
                "weight_frames",
                "frame_ids",
            ),
        }
    return {
        "sum": (
            "JtJ_sum",
            "Jty_sum",
            "y_sumsq_sum",
            "Jtf_sum",
            "f_sumsq_sum",
            "fty_sum",
            "nframe",
            "weight_sum",
        ),
        "max": ("n_atoms_obs",),
        "stack": (),
    }


def finalize_fm_root(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("reduce_stack"):
        return {
            "reduce_stack": True,
            "JtJ_frame": np.asarray(state["JtJ_frames"], dtype=np.float64),
            "Jty_frame": np.asarray(state["Jty_frames"], dtype=np.float64),
            "y_sumsq_frame": np.asarray(state["y_sumsq_frames"], dtype=np.float64),
            "Jtf_frame": np.asarray(state["Jtf_frames"], dtype=np.float64),
            "f_sumsq_frame": np.asarray(state["f_sumsq_frames"], dtype=np.float64),
            "fty_frame": np.asarray(state["fty_frames"], dtype=np.float64),
            "weight_frame": np.asarray(state["weight_frames"], dtype=np.float64),
            "frame_ids": np.asarray(state["frame_ids"], dtype=np.int64),
            "n_atoms_obs": int(state["n_atoms_obs"]),
            "n_frames": int(np.asarray(state["frame_ids"]).size),
        }
    weight_sum = float(state["weight_sum"])
    scale = 1.0 / weight_sum if weight_sum > 0.0 else 0.0
    return {
        "JtJ": np.asarray(state["JtJ_sum"], dtype=np.float64) * scale,
        "Jty": np.asarray(state["Jty_sum"], dtype=np.float64) * scale,
        "y_sumsq": float(state["y_sumsq_sum"]) * scale,
        "Jtf": np.asarray(state["Jtf_sum"], dtype=np.float64) * scale,
        "f_sumsq": float(state["f_sumsq_sum"]) * scale,
        "fty": float(state["fty_sum"]) * scale,
        "nframe": int(state["nframe"]),
        "weight_sum": weight_sum,
        "n_atoms_obs": int(state["n_atoms_obs"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REM
# ─────────────────────────────────────────────────────────────────────────────


def init_rem_state(
    step: Dict[str, Any],
    forcefield_snapshot: "Forcefield",
    topology_arrays: "TopologyArrays",
) -> Dict[str, Any]:
    del topology_arrays  # unused
    n_params = forcefield_snapshot.n_params()
    need_hessian = bool(step.get("need_hessian", False))
    reduce_stack = bool(step.get("reduce_stack", False))
    step_mode = str(step.get("step_mode", "")).strip().lower()
    use_gauge_free_gradient = step_mode == "rem"
    state: Dict[str, Any] = {
        "n_params": int(n_params),
        "need_hessian": need_hessian,
        "reduce_stack": reduce_stack,
        "gradient_convention": "gauge_free" if use_gauge_free_gradient else "physical",
    }
    if reduce_stack:
        state["energy_grad_frames"] = []
        state["weight_frames"] = []
        state["frame_ids"] = []
        if need_hessian:
            state["d2U_frames"] = []
            state["grad_outer_frames"] = []
    else:
        state["energy_grad_sum"] = np.zeros(n_params, dtype=np.float32)
        state["weight_sum"] = 0.0
        state["n_frames"] = 0
        if need_hessian:
            state["d2U_sum"] = np.zeros((n_params, n_params), dtype=np.float32)
            state["grad_outer_sum"] = np.zeros((n_params, n_params), dtype=np.float32)
            state["energy_grad_frame"] = []
    return state


def request_rem(step: Dict[str, Any]) -> Dict[str, bool]:
    """Return observable requests required by the REM reducer."""
    need_hessian = bool(step.get("need_hessian", False))
    step_mode = str(step.get("step_mode", "")).strip().lower()
    use_gauge_free_gradient = step_mode == "rem"
    return {
        "need_energy_grad": not use_gauge_free_gradient,
        "need_energy_hessian": need_hessian,
        "need_energy_grad_outer": need_hessian and not use_gauge_free_gradient,
        "need_gauge_free_energy_grad": use_gauge_free_gradient,
        "need_gauge_free_energy_grad_outer": need_hessian and use_gauge_free_gradient,
    }


def consume_rem_frame(
    state: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    frame_weight: float,
    reference_force: np.ndarray | None,
) -> None:
    del reference_force  # unused
    gradient_convention = str(state["gradient_convention"]).strip().lower()
    if gradient_convention == "gauge_free":
        grad_key = "gauge_free_energy_grad"
        grad_outer_key = "gauge_free_energy_grad_outer"
    elif gradient_convention == "physical":
        grad_key = "energy_grad"
        grad_outer_key = "energy_grad_outer"
    else:
        raise ValueError(
            f"Unsupported REM gradient_convention {gradient_convention!r}."
        )
    need_hessian = bool(state["need_hessian"])
    grad = np.asarray(payload[grad_key], dtype=np.float32)
    if state["reduce_stack"]:
        state["energy_grad_frames"].append(grad)
        state["weight_frames"].append(float(frame_weight))
        state["frame_ids"].append(int(payload["frame_idx"]))
        if need_hessian:
            state["d2U_frames"].append(
                np.asarray(payload["energy_hessian"], dtype=np.float32)
            )
            state["grad_outer_frames"].append(
                np.asarray(payload[grad_outer_key], dtype=np.float32)
            )
        return
    wi = float(frame_weight)
    state["energy_grad_sum"] += wi * grad
    state["weight_sum"] += wi
    state["n_frames"] += 1
    if need_hessian:
        state["d2U_sum"] += wi * np.asarray(payload["energy_hessian"], dtype=np.float32)
        state["grad_outer_sum"] += wi * np.asarray(
            payload[grad_outer_key], dtype=np.float32
        )
        state["energy_grad_frame"].append(grad)


def local_partials_rem(state: Dict[str, Any]) -> Dict[str, Any]:
    n_params = int(state["n_params"])
    need_hessian = bool(state["need_hessian"])
    gradient_convention = str(state["gradient_convention"]).strip().lower()
    if gradient_convention not in {"gauge_free", "physical"}:
        raise ValueError(
            f"Unsupported REM gradient_convention {gradient_convention!r}."
        )
    metadata = {"gradient_convention": gradient_convention}
    if state["reduce_stack"]:
        stack_1d = lambda arrs: (
            np.stack(arrs, axis=0).astype(np.float32, copy=False)
            if arrs
            else np.empty((0, n_params), dtype=np.float32)
        )
        stack_2d = lambda arrs: (
            np.stack(arrs, axis=0).astype(np.float32, copy=False)
            if arrs
            else np.empty((0, n_params, n_params), dtype=np.float32)
        )
        out = {
            "reduce_stack": True,
            "need_hessian": need_hessian,
            "metadata": metadata,
            "energy_grad_frames": stack_1d(state["energy_grad_frames"]),
            "weight_frames": np.asarray(state["weight_frames"], dtype=np.float32),
            "frame_ids": np.asarray(state["frame_ids"], dtype=np.int64),
        }
        if need_hessian:
            out["d2U_frames"] = stack_2d(state["d2U_frames"])
            out["grad_outer_frames"] = stack_2d(state["grad_outer_frames"])
        return out
    out = {
        "need_hessian": need_hessian,
        "metadata": metadata,
        "energy_grad_sum": np.asarray(state["energy_grad_sum"], dtype=np.float32),
        "weight_sum": float(state["weight_sum"]),
        "n_frames": int(state["n_frames"]),
    }
    if need_hessian:
        out["d2U_sum"] = np.asarray(state["d2U_sum"], dtype=np.float32)
        out["grad_outer_sum"] = np.asarray(state["grad_outer_sum"], dtype=np.float32)
        frames = state["energy_grad_frame"]
        if frames:
            out["energy_grad_frame"] = np.vstack(frames).astype(np.float32, copy=False)
        else:
            out["energy_grad_frame"] = np.empty((0, n_params), dtype=np.float32)
    return out


def reduce_plan_rem(step: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    need_hessian = bool(step.get("need_hessian", False))
    if bool(step.get("reduce_stack", False)):
        stack_keys = ["energy_grad_frames", "weight_frames", "frame_ids"]
        if need_hessian:
            stack_keys.extend(["d2U_frames", "grad_outer_frames"])
        return {
            "sum": (),
            "max": (),
            "stack": tuple(stack_keys),
            "dict_update": ("metadata",),
        }
    sum_keys = ["energy_grad_sum", "weight_sum", "n_frames"]
    stack_keys: Tuple[str, ...] = ()
    if need_hessian:
        sum_keys.extend(["d2U_sum", "grad_outer_sum"])
        stack_keys = ("energy_grad_frame",)
    return {
        "sum": tuple(sum_keys),
        "max": (),
        "stack": stack_keys,
        "dict_update": ("metadata",),
    }


def finalize_rem_root(state: Dict[str, Any]) -> Dict[str, Any]:
    need_hessian = bool(state.get("need_hessian", False))
    gradient_convention = str(state["metadata"]["gradient_convention"]).strip().lower()
    if gradient_convention not in {"gauge_free", "physical"}:
        raise ValueError(
            f"Unsupported REM gradient_convention {gradient_convention!r}."
        )
    if state.get("reduce_stack"):
        out = {
            "reduce_stack": True,
            "energy_grad_frame": np.asarray(state["energy_grad_frames"], dtype=np.float64),
            "weight_frame": np.asarray(state["weight_frames"], dtype=np.float64),
            "frame_ids": np.asarray(state["frame_ids"], dtype=np.int64),
            "n_frames": int(np.asarray(state["frame_ids"]).size),
            "gradient_convention": gradient_convention,
        }
        if need_hessian:
            out["d2U_frame"] = np.asarray(state["d2U_frames"], dtype=np.float64)
            out["grad_outer_frame"] = np.asarray(state["grad_outer_frames"], dtype=np.float64)
        return out
    weight_sum = float(state["weight_sum"])
    if weight_sum > 0.0:
        energy_grad_avg = np.asarray(state["energy_grad_sum"], dtype=np.float64) / weight_sum
    else:
        energy_grad_avg = np.zeros_like(np.asarray(state["energy_grad_sum"], dtype=np.float64))
    out = {
        "energy_grad_avg": energy_grad_avg,
        "n_frames": int(state["n_frames"]),
        "weight_sum": weight_sum,
        "gradient_convention": gradient_convention,
    }
    if need_hessian:
        if weight_sum > 0.0:
            out["d2U_avg"] = np.asarray(state["d2U_sum"], dtype=np.float64) / weight_sum
            out["grad_outer_avg"] = np.asarray(state["grad_outer_sum"], dtype=np.float64) / weight_sum
        else:
            d2 = np.asarray(state["d2U_sum"], dtype=np.float64)
            go = np.asarray(state["grad_outer_sum"], dtype=np.float64)
            out["d2U_avg"] = np.zeros_like(d2)
            out["grad_outer_avg"] = np.zeros_like(go)
        out["energy_grad_frame"] = np.asarray(state["energy_grad_frame"], dtype=np.float64)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# rdf  (From Ace — entire quintet)
# ─────────────────────────────────────────────────────────────────────────────

def request_rdf(step: Dict[str, Any]) -> Dict[str, bool]:
    del step
    return {"need_frame_cache": True}


def init_rdf_state(
    step: Dict[str, Any],
    forcefield_snapshot: "Forcefield",
    topology_arrays: "TopologyArrays",
) -> Dict[str, Any]:
    from ..analysis.rdf import init_distribution_state
    from ..topology.types import InteractionKey

    labels = step.get("interaction_keys")
    interaction_keys = None if labels is None else [InteractionKey.from_label(str(label)) for label in labels]
    raw_mode_by_key = step.get("mode_by_key")
    mode_by_key = None
    if raw_mode_by_key is not None:
        mode_by_key = {InteractionKey.from_label(str(k)): str(v) for k, v in raw_mode_by_key.items()}

    sel_indices = step.get("sel_indices")
    if sel_indices is not None:
        sel_indices = np.asarray(sel_indices, dtype=np.int32)

    return init_distribution_state(
        topology_arrays,
        forcefield_snapshot,
        interaction_keys=interaction_keys,
        mode_by_key=mode_by_key,
        cutoff=float(step.get("cutoff", 30.0)),
        r_max=None if step.get("r_max") is None else float(step["r_max"]),
        nbins_pair=int(step.get("nbins_pair", 200)),
        nbins_bond=int(step.get("nbins_bond", 200)),
        nbins_angle=int(step.get("nbins_angle", 180)),
        nbins_dihedral=int(step.get("nbins_dihedral", 180)),
        sel_indices=sel_indices,
        angle_degrees=bool(step.get("angle_degrees", True)),
        dihedral_degrees=bool(step.get("dihedral_degrees", True)),
        dihedral_periodic=bool(step.get("dihedral_periodic", True)),
        default_pair_mode=str(step.get("default_pair_mode", "rdf")),
        default_bonded_mode=str(step.get("default_bonded_mode", "pdf")),
    )


def consume_rdf_frame(
    state: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    frame_weight: float,
    reference_force: np.ndarray | None,
) -> None:
    del reference_force
    from ..analysis.rdf import accumulate_distribution_frame

    frame_cache = payload.get("frame_cache")
    if frame_cache is None:
        raise ValueError("rdf step requires payload['frame_cache']; enable need_frame_cache in compute().")
    accumulate_distribution_frame(state, frame_cache, frame_weight=float(frame_weight))


def local_partials_rdf(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n_frames": int(state.get("n_frames", 0)),
        "weight_sum": float(state.get("weight_sum", 0.0)),
        "pair_hist_by_key": {k: np.asarray(v, dtype=np.float64) for k, v in state.get("pair_hist_by_key", {}).items()},
        "pair_expected_by_key": {k: np.asarray(v, dtype=np.float64) for k, v in state.get("pair_expected_by_key", {}).items()},
        "bond_hist_by_key": {k: np.asarray(v, dtype=np.float64) for k, v in state.get("bond_hist_by_key", {}).items()},
        "angle_hist_by_key": {k: np.asarray(v, dtype=np.float64) for k, v in state.get("angle_hist_by_key", {}).items()},
        "dihedral_hist_by_key": {k: np.asarray(v, dtype=np.float64) for k, v in state.get("dihedral_hist_by_key", {}).items()},
        "pair_edges": state.get("pair_edges"),
        "pair_centers": state.get("pair_centers"),
        "pair_shell_vol": state.get("pair_shell_vol"),
        "pair_meta_by_key": state.get("pair_meta_by_key", {}),
        "bond_edges": state.get("bond_edges"),
        "bond_centers": state.get("bond_centers"),
        "bond_meta_by_key": state.get("bond_meta_by_key", {}),
        "angle_edges": state.get("angle_edges"),
        "angle_centers": state.get("angle_centers"),
        "angle_meta_by_key": state.get("angle_meta_by_key", {}),
        "dihedral_edges": state.get("dihedral_edges"),
        "dihedral_centers": state.get("dihedral_centers"),
        "dihedral_meta_by_key": state.get("dihedral_meta_by_key", {}),
        "pair_keys": list(state.get("pair_keys", [])),
        "bond_keys": list(state.get("bond_keys", [])),
        "angle_keys": list(state.get("angle_keys", [])),
        "dihedral_keys": list(state.get("dihedral_keys", [])),
        "mode_by_key": dict(state.get("mode_by_key", {})),
        "default_pair_mode": state.get("default_pair_mode", "rdf"),
        "default_bonded_mode": state.get("default_bonded_mode", "pdf"),
        "angle_degrees": bool(state.get("angle_degrees", True)),
        "dihedral_degrees": bool(state.get("dihedral_degrees", True)),
        "dihedral_periodic": bool(state.get("dihedral_periodic", True)),
    }
    return out


def reduce_plan_rdf(step: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    del step
    return {
        "sum": ("n_frames", "weight_sum"),
        "max": (),
        "stack": (),
        "dict_sum": (
            "pair_hist_by_key",
            "pair_expected_by_key",
            "bond_hist_by_key",
            "angle_hist_by_key",
            "dihedral_hist_by_key",
        ),
    }


def finalize_rdf_root(state: Dict[str, Any]) -> Dict[str, Any]:
    from ..analysis.rdf import finalize_distribution_state
    return finalize_distribution_state(state)



# ─────────────────────────────────────────────────────────────────────────────
# cache  (From Ace — entire quintet; builds a TrajectoryCache on rank 0)
# ─────────────────────────────────────────────────────────────────────────────

def request_cache(step: Dict[str, Any]) -> Dict[str, bool]:
    del step
    return {"need_frame_cache": True}


def init_cache_state(
    step: Dict[str, Any],
    forcefield_snapshot: "Forcefield",
    topology_arrays: "TopologyArrays",
) -> Dict[str, Any]:
    del step, forcefield_snapshot, topology_arrays
    return {"frames": {}}


def consume_cache_frame(
    state: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    frame_weight: float,
    reference_force: np.ndarray | None,
) -> None:
    del frame_weight, reference_force
    frame_cache = payload.get("frame_cache")
    if frame_cache is None:
        raise ValueError("cache step requires payload['frame_cache']; enable need_frame_cache in compute().")
    state["frames"][int(payload["frame_idx"])] = frame_cache


def local_partials_cache(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"frames": dict(state.get("frames", {}))}


def reduce_plan_cache(step: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    del step
    return {
        "sum": (),
        "max": (),
        "stack": (),
        "dict_update": ("frames",),
    }


def finalize_cache_root(state: Dict[str, Any]) -> Dict[str, Any]:
    from .mpi_engine import TrajectoryCache

    return TrajectoryCache(frames=dict(state.get("frames", {})))


# ─────────────────────────────────────────────────────────────────────────────
# cdfm_zbx
# ─────────────────────────────────────────────────────────────────────────────


def init_cdfm_zbx_state(
    step: Dict[str, Any],
    forcefield_snapshot: "Forcefield",
    topology_arrays: "TopologyArrays",
) -> Dict[str, Any]:
    if bool(step.get("reduce_stack", False)):
        raise ValueError(
            "cdfm_zbx does not support reduce_stack=True: the per-frame Jacobian "
            "stack J_frame has shape (n_frames, obs_rows, n_params) and is too "
            "large to retain in memory for realistic system sizes. Use the "
            "default averaged reducer path for cdfm_zbx."
        )
    y_eff_arr = np.asarray(step["y_eff"], dtype=np.float64).ravel()
    real_site_indices = getattr(topology_arrays, "real_site_indices", None)
    n_params = forcefield_snapshot.n_params()
    return {
        "y_eff": y_eff_arr,
        "real_site_indices": None
        if real_site_indices is None
        else np.asarray(real_site_indices, dtype=np.int32).reshape(-1),
        "J_sum": np.zeros((y_eff_arr.size, n_params), dtype=np.float64),
        "f_sum": np.zeros(y_eff_arr.size, dtype=np.float64),
        "gu_sum": np.zeros(n_params, dtype=np.float64),
        "gu_f_sum": np.zeros((n_params, y_eff_arr.size), dtype=np.float64),
        "weight_sum": 0.0,
        "n_samples": 0,
        "obs_rows": int(y_eff_arr.size),
        "mode": str(step.get("mode", "direct")).strip().lower(),
        "beta": step.get("beta"),
    }


def request_cdfm_zbx(step: Dict[str, Any]) -> Dict[str, bool]:
    del step
    return {
        "need_force_value": True,
        "need_force_grad": True,
        "need_energy_grad": True,
    }


def accumulate_cdfm_zbx_stats(
    stats: Dict[str, Any] | None,
    *,
    force_value: np.ndarray,
    force_grad: np.ndarray,
    energy_grad: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, Any]:
    """Accumulate CDFM ZBX sufficient statistics from one compute chunk."""
    f = np.asarray(force_value, dtype=np.float64)
    J = np.asarray(force_grad, dtype=np.float64)
    gu = np.asarray(energy_grad, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if stats is None:
        stats = {
            "J_sum": np.zeros(J.shape[1:], dtype=np.float64),
            "f_sum": np.zeros(f.shape[1:], dtype=np.float64),
            "gu_sum": np.zeros(gu.shape[1:], dtype=np.float64),
            "gu_f_sum": np.zeros((gu.shape[-1], f.shape[-1]), dtype=np.float64),
            "weight_sum": 0.0,
            "n_samples": 1,
        }
    stats["J_sum"] += np.einsum("b,brp->rp", w, J, optimize=True)
    stats["f_sum"] += np.einsum("b,br->r", w, f, optimize=True)
    stats["gu_sum"] += np.einsum("b,bp->p", w, gu, optimize=True)
    stats["gu_f_sum"] += np.einsum("b,bp,br->pr", w, gu, f, optimize=True)
    stats["weight_sum"] = float(stats["weight_sum"]) + float(np.sum(w))
    return stats


def consume_cdfm_zbx_frame(
    state: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    frame_weight: float,
    reference_force: np.ndarray | None,
) -> None:
    del reference_force  # unused
    y_eff_arr = state["y_eff"]
    real_site_indices = state["real_site_indices"]
    stats = payload.get("cdfm_stats")
    if stats is not None:
        J_sum = np.asarray(stats["J_sum"], dtype=np.float64)
        f_sum = np.asarray(stats["f_sum"], dtype=np.float64).ravel()
        if f_sum.size != y_eff_arr.size and real_site_indices is not None:
            f_sum = slice_observed_rows(f_sum, real_site_indices)
            J_sum = slice_observed_rows(J_sum, real_site_indices)
        state["J_sum"] += J_sum
        state["f_sum"] += f_sum
        state["gu_sum"] += np.asarray(stats["gu_sum"], dtype=np.float64).ravel()
        state["gu_f_sum"] += np.asarray(stats["gu_f_sum"], dtype=np.float64)
        state["weight_sum"] += float(stats["weight_sum"])
        state["n_samples"] += int(stats.get("n_samples", 1))
        state["obs_rows"] = max(int(state["obs_rows"]), int(f_sum.size))
        return
    J_i = np.asarray(payload["force_grad"], dtype=np.float64)
    f_i = np.asarray(payload["force"], dtype=np.float64).ravel()
    if f_i.size != y_eff_arr.size and real_site_indices is not None:
        f_i = slice_observed_rows(f_i, real_site_indices)
        J_i = slice_observed_rows(J_i, real_site_indices)
    gu_i = np.asarray(payload["energy_grad"], dtype=np.float64).ravel()
    wi = float(frame_weight)
    state["J_sum"] += wi * J_i
    state["f_sum"] += wi * f_i
    state["gu_sum"] += wi * gu_i
    state["gu_f_sum"] += wi * np.outer(gu_i, f_i)
    state["weight_sum"] += wi
    state["n_samples"] += 1
    state["obs_rows"] = max(int(state["obs_rows"]), int(f_i.size))


def local_partials_cdfm_zbx(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "J_sum": np.asarray(state["J_sum"], dtype=np.float64),
        "f_sum": np.asarray(state["f_sum"], dtype=np.float64),
        "gu_sum": np.asarray(state["gu_sum"], dtype=np.float64),
        "gu_f_sum": np.asarray(state["gu_f_sum"], dtype=np.float64),
        "weight_sum": float(state["weight_sum"]),
        "n_samples": int(state["n_samples"]),
        "obs_rows": int(state["obs_rows"]),
    }


def reduce_plan_cdfm_zbx(step: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    del step
    return {
        "sum": ("J_sum", "f_sum", "gu_sum", "gu_f_sum", "weight_sum", "n_samples"),
        "max": ("obs_rows",),
        "stack": (),
    }


def finalize_cdfm_zbx_root(state: Dict[str, Any]) -> Dict[str, Any]:
    y_eff_arr = np.asarray(state["y_eff"], dtype=np.float64).ravel()
    weight_sum = float(state["weight_sum"])
    n_samples = int(state["n_samples"])
    n_params = np.asarray(state["gu_sum"], dtype=np.float64).size
    if weight_sum == 0.0:
        return {
            "grad_direct": np.zeros(n_params, dtype=np.float64),
            "grad_reinforce": np.zeros(n_params, dtype=np.float64),
            "sse": float(np.dot(y_eff_arr, y_eff_arr)),
            "obs_rows": int(y_eff_arr.size),
            "n_samples": n_samples,
            "rmse": float(np.sqrt(np.dot(y_eff_arr, y_eff_arr) / max(int(y_eff_arr.size), 1))),
        }

    f_bar = np.asarray(state["f_sum"], dtype=np.float64) / weight_sum
    error = f_bar - y_eff_arr
    grad_direct = (np.asarray(state["J_sum"], dtype=np.float64) / weight_sum).T @ error
    grad_reinforce = np.zeros_like(grad_direct)
    mode = str(state.get("mode", "direct")).strip().lower()
    beta = state.get("beta")
    if mode == "reinforce":
        if beta is None:
            raise ValueError("beta required for reinforce mode")
        gu_bar = np.asarray(state["gu_sum"], dtype=np.float64) / weight_sum
        phi_bar = float(np.dot(f_bar, error))
        grad_reinforce = -float(beta) * (
            (np.asarray(state["gu_f_sum"], dtype=np.float64) @ error) / weight_sum
            - phi_bar * gu_bar
        )
    sse = float(np.dot(error, error))
    return {
        "grad_direct": grad_direct,
        "grad_reinforce": grad_reinforce,
        "sse": sse,
        "obs_rows": int(error.size),
        "n_samples": n_samples,
        "rmse": float(np.sqrt(sse / max(int(error.size), 1))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mode registry (must live after all per-mode functions are defined)
# ─────────────────────────────────────────────────────────────────────────────


MODE_OPS: Dict[str, ReducerOps] = {
    "fm": ReducerOps(
        init=init_fm_state,
        request=request_fm,
        consume=consume_fm_frame,
        local_partials=local_partials_fm,
        reduce_plan=reduce_plan_fm,
        finalize=finalize_fm_root,
    ),
    "rem": ReducerOps(
        init=init_rem_state,
        request=request_rem,
        consume=consume_rem_frame,
        local_partials=local_partials_rem,
        reduce_plan=reduce_plan_rem,
        finalize=finalize_rem_root,
    ),
    "dsm": ReducerOps(
        init=init_fm_state,
        request=request_dsm,
        consume=consume_fm_frame,
        local_partials=local_partials_fm,
        reduce_plan=reduce_plan_fm,
        finalize=finalize_fm_root,
    ),
    "cdfm_zbx": ReducerOps(
        init=init_cdfm_zbx_state,
        request=request_cdfm_zbx,
        consume=consume_cdfm_zbx_frame,
        local_partials=local_partials_cdfm_zbx,
        reduce_plan=reduce_plan_cdfm_zbx,
        finalize=finalize_cdfm_zbx_root,
    ),
    "cache": ReducerOps(
        init=init_cache_state,
        request=request_cache,
        consume=consume_cache_frame,
        local_partials=local_partials_cache,
        reduce_plan=reduce_plan_cache,
        finalize=finalize_cache_root,
    ),
    "rdf": ReducerOps(
        init=init_rdf_state,
        request=request_rdf,
        consume=consume_rdf_frame,
        local_partials=local_partials_rdf,
        reduce_plan=reduce_plan_rdf,
        finalize=finalize_rdf_root,
    ),
    # CUSTOMIZE POINT: register new step_mode reducers here.
}
