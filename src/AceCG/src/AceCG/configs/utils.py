"""Stateless config-parsing utilities.

Pure functions with no config-object dependency — safe to use in notebooks,
scripts, and workflow ``_build_*`` methods alike.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple


_FRAME_DATA_FILE_RE = re.compile(r"^frame_(\d+)\.data$")

_FORCE_FILE_DIGIT_RUN_RE = re.compile(r"\d+")


def extract_frame_id_from_data_file(path: str | Path) -> int:
    """Return the integer frame id from a ``frame_<id>.data`` filename."""
    candidate = Path(path)
    match = _FRAME_DATA_FILE_RE.fullmatch(candidate.name)
    if match is None:
        raise ValueError(
            f"Expected a frame data file named 'frame_<integer>.data', got {candidate!s}"
        )
    return int(match.group(1))


def extract_frame_id_from_force_file(path: str | Path) -> int:
    """Return the integer frame id from a reference-force npy filename.

    Force-pool files may be named ``frame_000035.forces.npy``,
    ``frame_000035_forces.npy``, ``forces_35.npy`` etc.  The only contract is
    that exactly one frame-id digit run appears in the filename stem; multiple
    *distinct* digit runs, or none, are rejected.  Duplicate digit runs with
    identical value (e.g. ``frame_35_rep35.npy``) are accepted.
    """
    candidate = Path(path)
    stem = candidate.name
    for suffix in (".npy",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    digit_runs = _FORCE_FILE_DIGIT_RUN_RE.findall(stem)
    if not digit_runs:
        raise ValueError(
            f"Force-pool filename {candidate.name!r} contains no frame-id digit run."
        )
    unique = {int(run) for run in digit_runs}
    if len(unique) != 1:
        raise ValueError(
            f"Force-pool filename {candidate.name!r} contains ambiguous frame-id "
            f"digit runs {sorted(unique)}; expected a single frame id."
        )
    return next(iter(unique))


def parse_exclude_setting(
    exclude: Optional[str],
) -> Tuple[str, str]:
    """Map an ``.acg`` ``exclude`` token to ``(exclude_bonded, exclude_option)``.

    ======== =============== ==============
    .acg     exclude_bonded  exclude_option
    ======== =============== ==============
    ``100``  ``"100"``       ``"none"``
    ``110``  ``"110"``       ``"none"``
    ``111``  ``"111"``       ``"none"``
    ``resid````"111"``       ``"resid"``
    ``molid````"111"``       ``"molid"``
    *None*   ``"111"``       ``"resid"``
    ======== =============== ==============
    """
    if exclude is None:
        return ("111", "resid")
    token = str(exclude).strip().lower()
    if token in ("100", "110", "111"):
        return (token, "none")
    if token in ("resid", "molid"):
        return ("111", token)
    raise ValueError(
        f"Unsupported exclude value {exclude!r}; "
        "expected one of '100', '110', '111', 'resid', 'molid'."
    )


def parse_pair_style_options(
    pair_style_command: Optional[str],
) -> Tuple[str, Optional[list]]:
    """Translate a user-facing ``pair_style`` command into Read/WriteLmpFF args.

    Returns ``(style, sel_styles)`` where *sel_styles* is ``None`` unless the
    pair style is ``hybrid``.
    """
    if not pair_style_command:
        return "table", None
    tokens = str(pair_style_command).split()
    if not tokens:
        return "table", None
    head = tokens[0].strip().lower()
    if head.startswith("hybrid"):
        sel_styles: list[str] = []
        if "table" in tokens[1:]:
            sel_styles.append("table")
        if not sel_styles and len(tokens) > 1:
            sel_styles.append(tokens[1])
        return "hybrid", sel_styles or None
    return head, None
