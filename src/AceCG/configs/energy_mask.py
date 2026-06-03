"""Coordinate-mask parsing for REM and CDREM energy statistics.

This module handles geometric coordinate masks, not forcefield parameter
trainability masks. A mask entry such as ``bond:A:B = 2.5:8.0`` decides which
observed bond lengths contribute to REM/CDREM energy-gradient statistics.
``Forcefield.param_mask`` is applied later and independently.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..topology.types import InteractionKey


@dataclass(frozen=True)
class EnergyMaskBounds:
    """Closed coordinate bounds for one interaction family."""

    minimum: float | None = None
    maximum: float | None = None

    def validate(self, *, label: str) -> "EnergyMaskBounds":
        """Return this bound after checking endpoint consistency."""
        if self.minimum is None and self.maximum is None:
            raise ValueError(f"energy_mask entry {label!r} must define at least one bound.")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(
                f"energy_mask entry {label!r} requires min <= max, "
                f"got {self.minimum} > {self.maximum}."
            )
        return self

    def active(self, values: np.ndarray) -> np.ndarray:
        """Return the boolean coordinate mask for ``values``."""
        coords = np.asarray(values, dtype=np.float64)
        mask = np.ones(coords.shape, dtype=bool)
        if self.minimum is not None:
            mask &= coords >= float(self.minimum)
        if self.maximum is not None:
            mask &= coords <= float(self.maximum)
        return mask

    def to_runtime_dict(self) -> dict[str, float | None]:
        """Return a JSON-friendly representation of these bounds."""
        return {"min": self.minimum, "max": self.maximum}


def normalize_energy_mask_spec(raw: Any) -> dict[str, dict[str, float | None]]:
    """Normalize ``training.energy_mask`` into JSON-friendly keyed bounds."""
    if _is_disabled(raw):
        return {}

    entries: list[tuple[Any, Any]] = []
    if isinstance(raw, Mapping):
        if "interactions" in raw:
            interactions = raw["interactions"]
            if not isinstance(interactions, Sequence) or isinstance(interactions, (str, bytes)):
                raise ValueError("energy_mask.interactions must be a list of entries.")
            for item in interactions:
                if not isinstance(item, Mapping):
                    raise ValueError("energy_mask.interactions entries must be mappings.")
                label = item.get("label", item.get("key"))
                if label is None:
                    style = item.get("style")
                    types = item.get("types")
                    if style is None or types is None:
                        raise ValueError(
                            "energy_mask interaction entries need label/key or style/types."
                        )
                    label = _label_from_style_types(style, types)
                entries.append((label, item))
        else:
            entries.extend(raw.items())
    else:
        raise ValueError("training.energy_mask must be a mapping or disabled value.")

    normalized: dict[str, dict[str, float | None]] = {}
    for raw_label, raw_bounds in entries:
        key = _coerce_interaction_key(raw_label)
        bounds = _parse_bounds(raw_bounds, label=key.label()).validate(
            label=key.label()
        )
        normalized[key.label()] = bounds.to_runtime_dict()
    return normalized


def parse_energy_mask_runtime(raw: Any) -> dict[InteractionKey, EnergyMaskBounds]:
    """Parse a runtime coordinate-mask spec into interaction-key bounds."""
    normalized = normalize_energy_mask_spec(raw)
    return {
        _coerce_interaction_key(label): EnergyMaskBounds(
            minimum=bounds.get("min"),
            maximum=bounds.get("max"),
        ).validate(label=label)
        for label, bounds in normalized.items()
    }


def energy_mask_active_fraction(active: int, total: int) -> float:
    """Return the active fraction with stable zero-total semantics."""
    total_int = int(total)
    if total_int <= 0:
        return 0.0
    return float(active) / float(total_int)


def summarize_energy_mask_counts(
    active: int,
    total: int,
    by_key: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Build the public active/total/fraction diagnostics payload."""
    active_int = int(active)
    total_int = int(total)
    payload: dict[str, Any] = {
        "active": active_int,
        "total": total_int,
        "fraction": energy_mask_active_fraction(active_int, total_int),
    }
    if by_key:
        payload["by_key"] = {
            str(label): {
                "active": int(counts.get("active", 0)),
                "total": int(counts.get("total", 0)),
                "fraction": energy_mask_active_fraction(
                    int(counts.get("active", 0)),
                    int(counts.get("total", 0)),
                ),
            }
            for label, counts in sorted(by_key.items())
        }
    return payload


def accumulate_mask_diagnostics(
    accumulator: MutableMapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    active_key: str = "active",
    total_key: str = "total",
    by_key_key: str = "by_key",
) -> None:
    """Accumulate one coordinate-mask diagnostics payload into *accumulator*.

    Updates *accumulator* in place; its scalar/by-key slots may use custom key
    names (e.g. a reducer's flat state) and missing entries count as zero/empty.
    """
    accumulator[active_key] = int(accumulator.get(active_key, 0)) + int(
        diagnostics.get("active", 0)
    )
    accumulator[total_key] = int(accumulator.get(total_key, 0)) + int(
        diagnostics.get("total", 0)
    )
    by_key = accumulator.setdefault(by_key_key, {})
    for label, counts in dict(diagnostics.get("by_key", {})).items():
        label = str(label)
        if label not in by_key:
            by_key[label] = {"active": 0, "total": 0}
        by_key[label]["active"] += int(counts.get("active", 0))
        by_key[label]["total"] += int(counts.get("total", 0))


def _is_disabled(raw: Any) -> bool:
    if raw is None or raw is False:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() in {"", "0", "false", "none", "off", "no"}
    return False


def _label_from_style_types(style: Any, types: Any) -> str:
    if isinstance(types, str):
        raw_types = [part.strip() for part in types.split(":") if part.strip()]
    elif isinstance(types, Sequence):
        raw_types = [str(item) for item in types]
    else:
        raise ValueError("energy_mask style/types entries require a sequence of types.")
    return ":".join([str(style).strip().lower(), *raw_types])


def _coerce_interaction_key(value: Any) -> InteractionKey:
    if isinstance(value, InteractionKey):
        key = value
    else:
        key = InteractionKey.from_label(str(value))
    style = str(key.style).strip().lower()
    types = tuple(str(item) for item in key.types)
    if style == "pair" and len(types) == 2:
        return InteractionKey.pair(types[0], types[1])
    if style == "bond" and len(types) == 2:
        return InteractionKey.bond(types[0], types[1])
    if style == "angle" and len(types) == 3:
        return InteractionKey.angle(types[0], types[1], types[2])
    if style == "dihedral" and len(types) == 4:
        return InteractionKey.dihedral(types[0], types[1], types[2], types[3])
    return InteractionKey(style=style, types=types)


def _parse_bounds(raw: Any, *, label: str) -> EnergyMaskBounds:
    if isinstance(raw, Mapping):
        if "bounds" in raw:
            return _parse_bounds(raw["bounds"], label=label)
        if "range" in raw:
            return _parse_bounds(raw["range"], label=label)
        minimum = _first_present(raw, ("min", "minimum", "low", "lower"))
        maximum = _first_present(raw, ("max", "maximum", "high", "upper"))
        return EnergyMaskBounds(
            minimum=_optional_float(minimum),
            maximum=_optional_float(maximum),
        )
    if isinstance(raw, str):
        text = raw.strip()
        delimiter = ":" if ":" in text else ","
        parts = [part.strip() for part in text.split(delimiter)]
        if len(parts) != 2:
            raise ValueError(
                f"energy_mask entry {label!r} string bounds must be 'min:max'."
            )
        return EnergyMaskBounds(
            minimum=_optional_float(parts[0]),
            maximum=_optional_float(parts[1]),
        )
    if isinstance(raw, Sequence):
        if len(raw) != 2:
            raise ValueError(f"energy_mask entry {label!r} sequence bounds need two values.")
        return EnergyMaskBounds(
            minimum=_optional_float(raw[0]),
            maximum=_optional_float(raw[1]),
        )
    raise ValueError(
        f"energy_mask entry {label!r} must be a mapping, string, or two-value sequence."
    )


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null", "na", "n/a"}:
            return None
        return float(text)
    return float(value)
