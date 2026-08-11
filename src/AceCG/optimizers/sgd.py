import numpy as np

from .base import BaseOptimizer


class SGDMaskedOptimizer(BaseOptimizer):
    """Stochastic gradient descent with masked, optionally clipped updates.

    ``clip_grad_norm`` applies independently to each forcefield interaction
    block.  A clipped block is scaled as a whole, preserving the relative
    gradient contributions of its parameters.
    """

    def __init__(
        self,
        L,
        mask,
        lr=1.0e-3,
        clip_grad_value=None,
        clip_grad_norm=None,
        clip_blocks=None,
    ):
        super().__init__(L, mask, lr)
        if clip_grad_value is not None and clip_grad_norm is not None:
            raise ValueError(
                "clip_grad_value and clip_grad_norm are mutually exclusive"
            )
        self._validate_positive_finite("clip_grad_value", clip_grad_value)
        self._validate_positive_finite("clip_grad_norm", clip_grad_norm)
        self.clip_grad_value = (
            None if clip_grad_value is None else float(clip_grad_value)
        )
        self.clip_grad_norm = (
            None if clip_grad_norm is None else float(clip_grad_norm)
        )
        self.clip_blocks = self._normalize_clip_blocks(clip_blocks)
        if self.clip_grad_norm is not None and not self.clip_blocks:
            raise ValueError(
                "clip_grad_norm requires forcefield interaction clip_blocks"
            )
        self.last_update = None
        self.last_clip_stats = ()

    @staticmethod
    def _validate_positive_finite(name, value):
        if value is None:
            return
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be positive and finite when provided")

    def _normalize_clip_blocks(self, blocks):
        if blocks is None:
            return ()

        normalized = []
        covered = np.zeros(self.L.size, dtype=bool)
        for block in blocks:
            if len(block) != 3:
                raise ValueError(
                    "clip_blocks entries must be (label, start, stop) triples"
                )
            label, start, stop = block
            start = int(start)
            stop = int(stop)
            if start < 0 or stop <= start or stop > self.L.size:
                raise ValueError(
                    f"Invalid clip block {label!r}: [{start}, {stop}) for "
                    f"{self.L.size} parameters"
                )
            if np.any(covered[start:stop]):
                raise ValueError(f"Overlapping clip block: {label!r}")
            covered[start:stop] = True
            normalized.append((str(label), start, stop))

        active = np.asarray(self.mask, dtype=bool).reshape(-1)
        if active.shape != covered.shape:
            raise ValueError("Optimizer mask must match the parameter vector shape")
        if self.clip_grad_norm is not None and np.any(active & ~covered):
            raise ValueError("clip_blocks must cover every active parameter")
        return tuple(normalized)

    def effective_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Return the masked gradient after the configured safety clipping."""
        g = np.asarray(grad, dtype=self.L.dtype)
        if g.shape != self.L.shape:
            raise ValueError(
                f"Gradient shape {g.shape} does not match parameters {self.L.shape}"
            )

        effective = np.zeros_like(self.L)
        effective[self.mask] = g[self.mask]
        stats = []

        if self.clip_grad_value is not None:
            effective[self.mask] = np.clip(
                effective[self.mask],
                -self.clip_grad_value,
                self.clip_grad_value,
            )
        elif self.clip_grad_norm is not None:
            for label, start, stop in self.clip_blocks:
                block_mask = self.mask[start:stop]
                block = effective[start:stop]
                raw_norm = float(np.linalg.norm(block[block_mask]))
                scale = 1.0
                if raw_norm > self.clip_grad_norm:
                    scale = self.clip_grad_norm / raw_norm
                    block[block_mask] *= scale
                stats.append(
                    {
                        "label": label,
                        "start": start,
                        "stop": stop,
                        "raw_norm": raw_norm,
                        "effective_norm": raw_norm * scale,
                        "scale": scale,
                        "clipped": scale < 1.0,
                    }
                )

        self.last_clip_stats = tuple(stats)
        return effective

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Apply ``-lr * grad`` to trainable parameters only."""
        effective_grad = self.effective_gradient(grad)
        update = self.lr * effective_grad
        self.L -= update
        self.last_update = -update
        return self.last_update

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["clip_grad_value"] = self.clip_grad_value
        state["clip_grad_norm"] = self.clip_grad_norm
        state["clip_blocks"] = [list(block) for block in self.clip_blocks]
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        value = state.get("clip_grad_value")
        self.clip_grad_value = None if value is None else float(value)
        norm = state.get("clip_grad_norm")
        self.clip_grad_norm = None if norm is None else float(norm)
        self._validate_positive_finite("clip_grad_value", self.clip_grad_value)
        self._validate_positive_finite("clip_grad_norm", self.clip_grad_norm)
        if self.clip_grad_value is not None and self.clip_grad_norm is not None:
            raise ValueError(
                "clip_grad_value and clip_grad_norm are mutually exclusive"
            )
        self.clip_blocks = self._normalize_clip_blocks(state.get("clip_blocks"))
        if self.clip_grad_norm is not None and not self.clip_blocks:
            raise ValueError(
                "clip_grad_norm checkpoint is missing forcefield interaction blocks"
            )
        self.last_update = None
        self.last_clip_stats = ()
