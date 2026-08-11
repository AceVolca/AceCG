"""`wrap_positions_in_box` must land every coordinate in the half-open cell [0, L).

`io/coordinates.py` documents why this matters: `np.mod` can return the box
length itself instead of a value strictly below it, and a coordinate equal to
`L` "trips downstream readers that bin by `floor(x / L)`". The function clamps
such values to 0.

The clamp has to survive the `out=` path as well as the plain one. That path is
not hypothetical: `io/trajmap.py:31-33` builds every `CGMapper` with
`out_dtype=np.float32`, and `compute/cgmap.py:290` then calls
`wrap_positions_in_box(cg_positions, active_box, out=cg_positions)` on that
float32 buffer for every mapped frame of every trajmap run that sets
`run.wrap`. A clamp evaluated only in float64 is undone by the cast back down.

See FINDINGS.md F-034.
"""

from __future__ import annotations

import numpy as np
import pytest

from AceCG.io.coordinates import wrap_positions_in_box


# The real DOPC production box edge, which is where this was found.
BOX = np.array([338.07, 338.07, 338.07, 90.0, 90.0, 90.0], dtype=np.float32)

# Coordinates a hair below zero: `np.mod` sends these to the top of the cell,
# which is precisely where the half-open convention is at risk.
NEGATIVE_EPSILONS = [-1.0e-6, -1.0e-8, -1.0e-10, -1.0e-12, -1.0e-14]


@pytest.mark.parametrize("epsilon", NEGATIVE_EPSILONS)
def test_float64_result_is_inside_the_half_open_cell(epsilon):
    positions = np.array([[epsilon, epsilon, epsilon]], dtype=np.float64)
    box = np.asarray(BOX, dtype=np.float64)
    wrapped = wrap_positions_in_box(positions, box)
    lengths = box[:3]
    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < lengths), f"{wrapped} reached the box edge {lengths}"


@pytest.mark.parametrize("epsilon", NEGATIVE_EPSILONS)
def test_float32_in_place_result_is_inside_the_half_open_cell(epsilon):
    """The `out=` path used by every trajmap run must hold the same invariant.

    Before F-034 the clamp was applied to the float64 intermediate and then
    cast down, so a value one float32 ulp below L rounded up to exactly L on
    the way into the caller's buffer.
    """
    positions = np.array([[epsilon, epsilon, epsilon]], dtype=np.float32)
    wrapped = wrap_positions_in_box(positions, BOX, out=positions)
    assert wrapped is positions
    lengths = BOX[:3]
    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < lengths), (
        f"{wrapped} reached the box edge {lengths} in the caller's float32 buffer"
    )


def test_ordinary_interior_coordinates_are_untouched_by_the_clamp():
    """The clamp must not perturb any coordinate that was already inside.

    Guards against a fix that clamps too eagerly: only values that landed on
    the edge may move, and they move to 0.
    """
    rng = np.random.default_rng(20260810)
    lengths = np.asarray(BOX[:3], dtype=np.float64)
    interior = rng.uniform(0.01, 0.99, size=(5000, 3)) * lengths
    wrapped = wrap_positions_in_box(interior.astype(np.float32), BOX)
    np.testing.assert_allclose(wrapped, interior, rtol=0.0, atol=1.0e-2)
    assert np.all(wrapped >= 0.0) and np.all(wrapped < lengths)


def test_a_whole_float32_frame_stays_inside_after_in_place_wrapping():
    """Bulk check on the shape a trajmap run actually writes."""
    rng = np.random.default_rng(11)
    lengths = np.asarray(BOX[:3], dtype=np.float32)
    # Span several images in both directions, as unwrapped CG sites do.
    frame = (rng.uniform(-3.0, 3.0, size=(20000, 3)) * lengths).astype(np.float32)
    wrap_positions_in_box(frame, BOX, out=frame)
    assert np.all(frame >= 0.0)
    assert np.all(frame < lengths), (
        f"{int(np.count_nonzero(frame >= lengths))} coordinate(s) landed on or "
        "past the box edge"
    )
