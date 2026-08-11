import numpy as np
import pytest

from AceCG.analysis.fm_residuals import (
    residual_sums_by_type,
    summarize_residual_sums,
)


def test_residual_sums_and_report_are_exact_by_type():
    reference = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [4.0, 0.0, 0.0]],
        ]
    )
    model = reference.copy()
    model[:, 0, 0] += 1.0
    model[:, 1, 1] -= 2.0
    codes = np.array([1, 2, 1])

    sums = residual_sums_by_type(reference, model, codes)
    report = summarize_residual_sums(
        sums, type_names={1: "A", 2: "B"}, frame_count=2
    )

    by_name = {row["bead"]: row for row in report["by_type"]}
    assert by_name["A"]["bead_instances"] == 4
    assert by_name["B"]["bead_instances"] == 2
    assert by_name["A"]["residual_sse"] == pytest.approx(2.0)
    assert by_name["B"]["residual_sse"] == pytest.approx(8.0)
    assert by_name["A"]["residual_sse_share"] == pytest.approx(0.2)
    assert by_name["B"]["residual_sse_share"] == pytest.approx(0.8)
    assert report["global"]["residual_force_rmse"] == pytest.approx(
        np.sqrt(10.0 / 18.0)
    )


def test_residual_sums_reject_mismatched_force_shape():
    with pytest.raises(ValueError, match="model_force"):
        residual_sums_by_type(
            np.zeros((2, 3)), np.zeros((3, 3)), np.array([1, 2])
        )
