"""Tests for AceCG.io.tables LAMMPS table parsing, writing, and utilities (§15.4 item A4)."""

import pytest
import numpy as np
from pathlib import Path

from AceCG.io.tables import (
    parse_lammps_table,
    find_equilibrium,
    compare_table_files,
    integrate_force_to_potential,
)


# ---------------------------------------------------------------------------
# Helpers to write synthetic LAMMPS table files
# ---------------------------------------------------------------------------

def _write_pair_table(path: Path, n: int = 20, k: float = 10.0, r0: float = 5.0) -> None:
    """Write a harmonic pair table: V = 0.5*k*(r-r0)^2, F = -k*(r-r0), r in [1, r0+1]."""
    r = np.linspace(1.0, r0 + 1.0, n)
    V = 0.5 * k * (r - r0) ** 2
    F = -k * (r - r0)
    with open(path, "w") as f:
        f.write("# Synthetic harmonic pair table\n")
        f.write("PAIR_AB\n")
        f.write(f"N {n} R {r[0]:.6f} {r[-1]:.6f}\n\n")
        for i, (ri, vi, fi) in enumerate(zip(r, V, F), start=1):
            f.write(f"{i:6d}  {ri:16.8f}  {vi:16.8e}  {fi:16.8e}\n")


def _write_bond_table(path: Path, n: int = 40, k: float = 10.0, center: float = 3.0) -> None:
    """Write a harmonic bond table: V = k*(r-center)^2, F = -2k*(r-center)."""
    r = np.linspace(1.0, 5.0, n)
    V = k * (r - center) ** 2
    F = -2.0 * k * (r - center)
    with open(path, "w") as f:
        f.write("# Synthetic harmonic bond table\n")
        f.write("BOND_AB\n")
        f.write(f"N {n}\n\n")
        for i, (ri, vi, fi) in enumerate(zip(r, V, F), start=1):
            f.write(f"{i:6d}  {ri:16.8f}  {vi:16.8e}  {fi:16.8e}\n")


# ---------------------------------------------------------------------------
# parse_lammps_table
# ---------------------------------------------------------------------------

def test_parse_returns_three_arrays(tmp_path):
    table_path = tmp_path / "pair.table"
    _write_pair_table(table_path)
    r, V, F = parse_lammps_table(str(table_path))
    assert isinstance(r, np.ndarray)
    assert isinstance(V, np.ndarray)
    assert F is not None
    assert isinstance(F, np.ndarray)


def test_parse_row_count(tmp_path):
    table_path = tmp_path / "pair.table"
    _write_pair_table(table_path, n=20)
    r, V, F = parse_lammps_table(str(table_path))
    assert len(r) == 20


def test_parse_r_range(tmp_path):
    table_path = tmp_path / "pair.table"
    _write_pair_table(table_path, n=20, r0=5.0)
    r, V, F = parse_lammps_table(str(table_path))
    assert r[0] == pytest.approx(1.0, abs=1e-5)
    assert r[-1] == pytest.approx(6.0, abs=1e-5)


def test_parse_vf_consistency(tmp_path):
    """F should approximate -dV/dr within finite-difference tolerance."""
    table_path = tmp_path / "pair.table"
    _write_pair_table(table_path, n=100, k=5.0, r0=4.0)
    r, V, F = parse_lammps_table(str(table_path))
    # Central finite difference of V
    dVdr = np.gradient(V, r)
    # Interior points only (avoid boundary artifacts)
    interior = slice(2, -2)
    assert np.allclose(-dVdr[interior], F[interior], atol=0.1)


def test_parse_no_force_column(tmp_path):
    """Table files with only (index, r, V) — F returns None."""
    table_path = tmp_path / "novf.table"
    n = 10
    r = np.linspace(1.0, 5.0, n)
    V = np.zeros(n)
    with open(table_path, "w") as f:
        f.write("NOVF\n")
        f.write(f"N {n}\n\n")
        for i, (ri, vi) in enumerate(zip(r, V), start=1):
            f.write(f"{i}  {ri:.6f}  {vi:.6f}\n")
    r2, V2, F2 = parse_lammps_table(str(table_path))
    assert len(r2) == n
    # F may be None for a 3-col file with no forces
    # (just check it doesn't crash and r/V are correct)
    assert r2[0] == pytest.approx(r[0], abs=1e-5)


# ---------------------------------------------------------------------------
# integrate_force_to_potential
# ---------------------------------------------------------------------------

def test_integrate_force_shape():
    r = np.linspace(1.0, 5.0, 50)
    F = -2.0 * (r - 3.0)   # harmonic: V = (r-3)^2
    V = integrate_force_to_potential(r, F)
    assert V.shape == r.shape


def test_integrate_force_minimum_at_boundary():
    """Integrated V should have V[-1] == 0.0 (by convention in integrate_force_to_potential)."""
    r = np.linspace(1.0, 5.0, 50)
    F = -2.0 * (r - 3.0)
    V = integrate_force_to_potential(r, F)
    assert V[-1] == pytest.approx(0.0, abs=1e-10)


def test_integrate_force_shape_recovery():
    """Up to a constant shift, integrated V should match the true harmonic."""
    k, r0 = 5.0, 3.0
    r = np.linspace(1.0, 5.0, 200)
    F = -2.0 * k * (r - r0)
    V_int = integrate_force_to_potential(r, F)
    V_true = k * (r - r0) ** 2
    # Shift so last point matches
    V_true_shifted = V_true - V_true[-1]
    assert np.allclose(V_int, V_true_shifted, atol=0.02)


# ---------------------------------------------------------------------------
# find_equilibrium
# ---------------------------------------------------------------------------

def test_find_equilibrium_known_minimum(tmp_path):
    """Harmonic V with minimum at r0=3.0; find_equilibrium from F should return ~3.0."""
    k, r0 = 10.0, 3.0
    r = np.linspace(1.0, 5.0, 200)
    F = -2.0 * k * (r - r0)
    eq = find_equilibrium(r, F)
    assert abs(eq - r0) < 0.05


def test_find_equilibrium_from_parsed_table(tmp_path):
    k, center = 8.0, 3.5
    table_path = tmp_path / "bond.table"
    _write_bond_table(table_path, n=80, k=k, center=center)
    r, V, F = parse_lammps_table(str(table_path))
    assert F is not None
    eq = find_equilibrium(r, F)
    assert abs(eq - center) < 0.1


# ---------------------------------------------------------------------------
# compare_table_files
# ---------------------------------------------------------------------------

def test_compare_identical_files(tmp_path):
    table_path = tmp_path / "pair.table"
    _write_pair_table(table_path)
    result = compare_table_files(table_path, table_path)
    assert result["max_abs_dV"] == pytest.approx(0.0, abs=1e-10)
    assert result["max_abs_dF"] == pytest.approx(0.0, abs=1e-10)
    assert result["abs_dEQ"] == pytest.approx(0.0, abs=1e-6)


def test_compare_different_files_nonzero_diff(tmp_path):
    path1 = tmp_path / "pair1.table"
    path2 = tmp_path / "pair2.table"
    _write_pair_table(path1, k=10.0, r0=5.0)
    _write_pair_table(path2, k=5.0, r0=5.0)  # different k
    result = compare_table_files(path1, path2)
    assert result["max_abs_dV"] > 0.0 or result["max_abs_dF"] > 0.0


def test_compare_returns_all_keys(tmp_path):
    table_path = tmp_path / "pair.table"
    _write_pair_table(table_path)
    result = compare_table_files(table_path, table_path)
    for key in ("max_abs_dV", "max_abs_dF", "eq_ref", "eq_candidate", "abs_dEQ"):
        assert key in result
