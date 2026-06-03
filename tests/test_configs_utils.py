"""Unit tests for :mod:`AceCG.configs.utils` frame-id helpers."""

from __future__ import annotations

import pytest

from AceCG.configs.utils import (
    extract_frame_id_from_data_file,
    extract_frame_id_from_force_file,
)


def test_extract_frame_id_from_data_file_accepts_canonical_name():
    assert extract_frame_id_from_data_file("frame_000123.data") == 123
    assert extract_frame_id_from_data_file("path/to/frame_7.data") == 7


def test_extract_frame_id_from_data_file_rejects_noncanonical_name():
    with pytest.raises(ValueError, match="frame_<integer>.data"):
        extract_frame_id_from_data_file("subsample_01.data")


def test_extract_frame_id_from_force_file_accepts_single_digit_run():
    assert extract_frame_id_from_force_file("frame_000035.forces.npy") == 35
    assert extract_frame_id_from_force_file("frame_7.npy") == 7


def test_extract_frame_id_from_force_file_accepts_duplicate_consistent_digits():
    assert extract_frame_id_from_force_file("frame_35_rep35.npy") == 35


def test_extract_frame_id_from_force_file_rejects_no_digits():
    with pytest.raises(ValueError, match="no frame-id digit run"):
        extract_frame_id_from_force_file("no_digits_here.npy")


def test_extract_frame_id_from_force_file_rejects_ambiguous_digits():
    with pytest.raises(ValueError, match="ambiguous"):
        extract_frame_id_from_force_file("frame_35_rep36.npy")
