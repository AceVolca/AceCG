"""Force-matching residual reduction and report mathematics."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "residual_sums_by_type",
    "summarize_residual_sums",
]


def _force_matrix(values: np.ndarray, *, n_atoms: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (int(n_atoms), 3):
        return array
    if array.size == int(n_atoms) * 3:
        return array.reshape(int(n_atoms), 3)
    raise ValueError(
        f"{name} must contain {int(n_atoms)} three-component forces; "
        f"got shape {array.shape}."
    )


def residual_sums_by_type(
    reference_force: np.ndarray,
    model_force: np.ndarray,
    atom_type_codes: np.ndarray,
    *,
    ordered_codes: Optional[Sequence[int]] = None,
) -> dict[str, np.ndarray]:
    """Accumulate target/model/residual force squares for one or more frames."""
    codes = np.asarray(atom_type_codes, dtype=np.int64).reshape(-1)
    n_atoms = int(codes.size)
    reference = np.asarray(reference_force, dtype=np.float64)
    model = np.asarray(model_force, dtype=np.float64)
    if reference.ndim <= 2:
        reference = _force_matrix(reference, n_atoms=n_atoms, name="reference_force")[
            None, ...
        ]
    elif reference.shape[-2:] != (n_atoms, 3):
        raise ValueError(
            f"reference_force trailing shape must be {(n_atoms, 3)}, got "
            f"{reference.shape}."
        )
    if model.ndim <= 2:
        model = _force_matrix(model, n_atoms=n_atoms, name="model_force")[None, ...]
    elif model.shape[-2:] != (n_atoms, 3):
        raise ValueError(
            f"model_force trailing shape must be {(n_atoms, 3)}, got {model.shape}."
        )
    reference = reference.reshape(-1, n_atoms, 3)
    model = model.reshape(-1, n_atoms, 3)
    if reference.shape != model.shape:
        raise ValueError(
            f"reference/model force shapes differ: {reference.shape} vs {model.shape}."
        )

    type_codes = np.asarray(
        sorted(np.unique(codes)) if ordered_codes is None else ordered_codes,
        dtype=np.int64,
    )
    if type_codes.ndim != 1 or np.unique(type_codes).size != type_codes.size:
        raise ValueError("ordered_codes must contain unique one-dimensional type codes.")
    result = {
        "type_codes": type_codes,
        "bead_instances": np.zeros(type_codes.size, dtype=np.int64),
        "target_sse": np.zeros(type_codes.size, dtype=np.float64),
        "model_sse": np.zeros(type_codes.size, dtype=np.float64),
        "residual_sse": np.zeros(type_codes.size, dtype=np.float64),
    }
    residual = model - reference
    n_frames = int(reference.shape[0])
    for index, code in enumerate(type_codes):
        mask = codes == int(code)
        if not np.any(mask):
            continue
        result["bead_instances"][index] = int(np.count_nonzero(mask)) * n_frames
        result["target_sse"][index] = float(np.sum(np.square(reference[:, mask, :])))
        result["model_sse"][index] = float(np.sum(np.square(model[:, mask, :])))
        result["residual_sse"][index] = float(np.sum(np.square(residual[:, mask, :])))
    return result


def summarize_residual_sums(
    sums: Mapping[str, np.ndarray],
    *,
    type_names: Optional[Mapping[int, str]] = None,
    frame_count: int,
    mpi_ranks: int = 1,
    elapsed_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """Convert reduced sums into per-type shares, RMSE, and explained fractions."""
    codes = np.asarray(sums["type_codes"], dtype=np.int64)
    instances = np.asarray(sums["bead_instances"], dtype=np.int64)
    target = np.asarray(sums["target_sse"], dtype=np.float64)
    model = np.asarray(sums["model_sse"], dtype=np.float64)
    residual = np.asarray(sums["residual_sse"], dtype=np.float64)
    if not (
        codes.shape
        == instances.shape
        == target.shape
        == model.shape
        == residual.shape
    ):
        raise ValueError("all reduced per-type arrays must have the same shape.")
    if int(frame_count) <= 0 or np.any(instances <= 0):
        raise ValueError("frame_count and every per-type bead-instance count must be positive.")
    if not all(np.all(np.isfinite(value)) for value in (target, model, residual)):
        raise ValueError("per-type force sums contain non-finite values.")
    target_total = float(np.sum(target))
    model_total = float(np.sum(model))
    residual_total = float(np.sum(residual))
    if target_total <= 0.0 or residual_total < 0.0:
        raise ValueError("target SSE must be positive and residual SSE non-negative.")

    names = {int(code): str(code) for code in codes}
    if type_names is not None:
        names.update({int(code): str(name) for code, name in type_names.items()})
    rows = []
    for index, code in enumerate(codes):
        component_count = 3 * int(instances[index])
        rows.append(
            {
                "type_code": int(code),
                "bead": names[int(code)],
                "bead_instances": int(instances[index]),
                "target_sse": float(target[index]),
                "target_sse_share": float(target[index] / target_total),
                "target_force_rms": float(np.sqrt(target[index] / component_count)),
                "model_sse": float(model[index]),
                "residual_sse": float(residual[index]),
                "residual_sse_share": (
                    0.0 if residual_total == 0.0 else float(residual[index] / residual_total)
                ),
                "residual_force_rmse": float(
                    np.sqrt(residual[index] / component_count)
                ),
                "unexplained_fraction": float(residual[index] / target[index]),
                "explained_fraction": float(1.0 - residual[index] / target[index]),
            }
        )
    total_components = 3 * int(np.sum(instances))
    return {
        "status": "passed",
        "definition": "sum(||F_model - F_target||^2) over Cartesian components",
        "frames": int(frame_count),
        "mpi_ranks": int(mpi_ranks),
        "elapsed_seconds": None if elapsed_seconds is None else float(elapsed_seconds),
        "global": {
            "bead_instances": int(np.sum(instances)),
            "target_sse": target_total,
            "model_sse": model_total,
            "residual_sse": residual_total,
            "target_force_rms": float(np.sqrt(target_total / total_components)),
            "residual_force_rmse": float(np.sqrt(residual_total / total_components)),
            "unexplained_fraction": float(residual_total / target_total),
            "explained_fraction": float(1.0 - residual_total / target_total),
        },
        "by_type": rows,
    }
