from __future__ import annotations

from pathlib import Path

import pytest

from AceCG.io.lammps_input import (
    iter_commands,
    resolve_include_path,
    resolve_lines,
    strip_lines,
    tokenize_line,
    tokenize_lines,
)


def test_strip_lines_decomments_and_joins_continuations() -> None:
    text = """
    # ignored
    pair_coeff A B & # continued coeff
      table.dat PAIR_AB # trailing comment
    bond_coeff 1 harmonic 10.0 2.0
    """

    assert list(tokenize_lines(strip_lines(text))) == [
        ("pair_coeff", "A", "B", "table.dat", "PAIR_AB"),
        ("bond_coeff", "1", "harmonic", "10.0", "2.0"),
    ]


def test_tokenize_line_strips_trailing_comments() -> None:
    assert tokenize_line("not_pair_coeff A B # pair_coeff A B") == (
        "not_pair_coeff",
        "A",
        "B",
    )
    assert tokenize_line("# pair_coeff A B") == ()


def test_resolve_lines_inlines_literal_includes(tmp_path: Path) -> None:
    coeffs = tmp_path / "coeffs.in"
    coeffs.write_text("bond_coeff 1 harmonic 10.0 2.0\n", encoding="utf-8")
    root = tmp_path / "root.in"
    root.write_text("include coeffs.in\n", encoding="utf-8")

    assert resolve_lines(root) == ["bond_coeff 1 harmonic 10.0 2.0"]
    assert list(iter_commands(root, resolve_includes=False)) == [("include", "coeffs.in")]
    assert list(iter_commands(root)) == [("bond_coeff", "1", "harmonic", "10.0", "2.0")]


def test_resolve_include_path_rejects_runtime_substitution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="variable-expanded"):
        resolve_include_path("${coeff_file}", tmp_path)
