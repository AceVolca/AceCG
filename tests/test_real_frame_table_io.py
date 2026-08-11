"""Real-forcefield oracle for the tabulated-potential I/O boundary.

Review brief §3 names two things that live here: "cutoff / shift / tail
conventions, and whether energy and force paths apply them identically" and
"unit conventions at every I/O boundary". The sweep rewrote both ends of that
boundary — `parse_lammps_table` went from a 51-line lenient scraper to a
14-line delegate over a strict section reader, and `write_lammps_table` grew a
section model and a staged replace — and the existing table tests all build
their own two-column tables in `tmp_path`.

These tests instead drive the **real** REM-init DOPC forcefield shipped in
`tests/test_data/dopc_cg6/ff/` (10 pair tables, 3 bond tables, 3 angle tables,
production grids and headers) through the full `ReadLmpFF` -> `WriteLmpFF` ->
`ReadLmpFF` round trip, and assert on numbers rather than on structure. A
dropped section, a re-sorted grid, a lost `R`/`EQ`/`FP` header, or a force
column written in the wrong convention all break at least one of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from real_frames import DOPC_CG6, DOPC_CUTOFF, dopc_forcefield, dopc_topology_arrays

from AceCG.io.forcefield import ReadLmpFF, WriteLmpFF
from AceCG.io.tables import parse_lammps_table, read_lammps_table_section


N_COEFFS = 12


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    """The shipped forcefield, re-written by `WriteLmpFF` into a scratch tree."""
    source_dir = DOPC_CG6 / "ff"
    out_dir = tmp_path_factory.mktemp("ff_out")
    # WriteLmpFF resolves the relative table paths named in the settings file
    # against the *destination* directory, so every table lands in out_dir.
    forcefield = dopc_forcefield(N_COEFFS)
    WriteLmpFF(
        str(source_dir / "system.settings"),
        str(out_dir / "system.settings"),
        forcefield,
        pair_style="hybrid",
        pair_typ_sel=["table"],
        topology_arrays=dopc_topology_arrays(),
    )
    return source_dir, out_dir, forcefield


def _table_files(directory):
    return sorted(path.name for path in directory.glob("*.table"))


def test_every_shipped_table_is_rewritten_and_none_is_lost(written):
    """All 16 production tables come back, one section each, same names."""
    source_dir, out_dir, _ = written
    assert _table_files(out_dir) == _table_files(source_dir)
    assert len(_table_files(source_dir)) == 16


@pytest.mark.parametrize(
    "table_name",
    [
        "Pair_HG-HG.table",
        "Pair_MG-T1.table",
        "Pair_T2-T2.table",
        "HG_MG_bon.table",
        "MG_T1_T2_ang.table",
    ],
)
def test_the_output_grid_is_the_source_grid_exactly(written, table_name):
    """A rewritten table is evaluated on the source coordinates, unmoved.

    B's writer read the grid back through a parser that concatenated every
    section of a file and `argsort`ed the result. Anything that re-derives or
    re-orders the grid changes which coordinates LAMMPS interpolates between.
    """
    source_dir, out_dir, _ = written
    source_x, _, _ = parse_lammps_table(source_dir / table_name)
    out_x, _, _ = parse_lammps_table(out_dir / table_name)
    assert out_x.size == source_x.size
    np.testing.assert_allclose(out_x, source_x, rtol=0.0, atol=1.0e-6)


def test_pair_headers_keep_the_r_range_and_bonded_headers_keep_eq_and_fp(written):
    """Header conventions survive the rewrite, per style.

    LAMMPS reads `R rlo rhi` for a pair table and `EQ` / `FP` for bond and
    angle tables; losing either silently changes how LAMMPS interpolates or
    where it places the equilibrium.
    """
    _, out_dir, _ = written

    pair = read_lammps_table_section(out_dir / "Pair_HG-MG.table")
    tokens = [token.upper() for token in pair.header_tokens]
    assert tokens[0] == "N" and "R" in tokens
    r_index = tokens.index("R")
    assert float(pair.header_tokens[r_index + 1]) == pytest.approx(
        float(pair.x[0]), abs=1.0e-6
    )
    assert float(pair.header_tokens[r_index + 2]) == pytest.approx(
        float(pair.x[-1]), abs=1.0e-6
    )

    for name in ("HG_MG_bon.table", "MG_T1_T2_ang.table"):
        section = read_lammps_table_section(out_dir / name)
        tokens = [token.upper() for token in section.header_tokens]
        assert tokens[0] == "N"
        assert "EQ" in tokens, f"{name} lost its EQ header"
        eq = float(section.header_tokens[tokens.index("EQ") + 1])
        assert float(section.x[0]) <= eq <= float(section.x[-1])
        assert "FP" in tokens, f"{name} lost its FP header"


@pytest.mark.parametrize(
    "table_name, style, types",
    [
        ("Pair_HG-HG.table", "pair", ("HG", "HG")),
        ("Pair_MG-T1.table", "pair", ("MG", "T1")),
        ("HG_MG_bon.table", "bond", ("HG", "MG")),
        ("MG_T1_T2_ang.table", "angle", ("MG", "T1", "T2")),
    ],
)
def test_written_columns_equal_the_model_evaluated_on_that_grid(
    written, table_name, style, types
):
    """Columns 3 and 4 are the model's own `value` and `force` at column 2.

    Not tautological: both columns are recomputed from the potential object
    and compared to the bytes that landed on disk. A force column written in
    the wrong sign or a potential column taken from the source rather than the
    model both fail here.
    """
    from AceCG.topology.types import InteractionKey

    _, out_dir, forcefield = written
    key = InteractionKey(style=style, types=types)
    assert key in forcefield, f"{key.label()} missing from the shipped forcefield"
    pot = forcefield[key][0]

    section = read_lammps_table_section(out_dir / table_name)
    np.testing.assert_allclose(
        section.potential,
        np.asarray(pot.value(section.x), dtype=float),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        section.force,
        np.asarray(pot.force(section.x), dtype=float),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_reread_forcefield_reproduces_the_written_energies_and_forces(written):
    """`ReadLmpFF(WriteLmpFF(ff))` gives back the same curves on the same grids.

    The end-to-end statement of the I/O boundary: whatever the section model,
    the staged replace and the fitter do, a round trip through LAMMPS table
    text must not move the potential. Compared on the source grid so a
    re-gridding bug cannot hide behind interpolation.
    """
    source_dir, out_dir, forcefield = written
    reread = ReadLmpFF(
        str(out_dir / "system.settings"),
        pair_style="hybrid",
        pair_typ_sel=["table"],
        cutoff=DOPC_CUTOFF,
        table_fit="bspline",
        table_fit_overrides={"n_coeffs": N_COEFFS},
        topology_arrays=dopc_topology_arrays(),
    )

    assert set(reread.keys()) == set(forcefield.keys())
    compared = 0
    for key in forcefield.keys():
        original = forcefield[key][0]
        roundtrip = reread[key][0]
        lo = float(max(np.min(original.knots), np.min(roundtrip.knots)))
        hi = float(min(np.max(original.knots), np.max(roundtrip.knots)))
        probe = np.linspace(lo + 1.0e-6, hi - 1.0e-6, 200)
        # A refit through a finite table cannot be bitwise identical, but it
        # must not move the curve by more than the table's own resolution
        # implies. The scale is set by the curve itself, not by a constant.
        scale = float(np.max(np.abs(original.value(probe)))) + 1.0
        np.testing.assert_allclose(
            roundtrip.value(probe) / scale,
            original.value(probe) / scale,
            atol=2.0e-3,
            err_msg=f"{key.label()} energy moved through the table round trip",
        )
        force_scale = float(np.max(np.abs(original.force(probe)))) + 1.0
        np.testing.assert_allclose(
            roundtrip.force(probe) / force_scale,
            original.force(probe) / force_scale,
            atol=2.0e-3,
            err_msg=f"{key.label()} force moved through the table round trip",
        )
        compared += 1
    assert compared == 16


def test_max_force_capping_clips_the_force_and_reintegrates_the_energy(tmp_path):
    """`_acecg_max_force` still caps, and V is re-derived from the capped F.

    Version B did this by re-reading and rewriting the finished file
    (`cap_table_forces`, 80 lines, removed by the sweep); A does it in memory
    before the section is written (`io/forcefield.py:794-799`). The science
    must be unchanged: clip the force, then integrate the *clipped* force to
    get the energy — writing the uncapped energy beside a capped force would
    make U and F inconsistent in the file LAMMPS reads.
    """
    from AceCG.io.tables import integrate_force_to_potential
    from AceCG.topology.types import InteractionKey

    source_dir = DOPC_CG6 / "ff"
    forcefield = dopc_forcefield(N_COEFFS)
    key = InteractionKey(style="pair", types=("HG", "HG"))
    pot = forcefield[key][0]

    source_x, _, _ = parse_lammps_table(source_dir / "Pair_HG-HG.table")
    uncapped = np.asarray(pot.force(source_x), dtype=float)
    limit = float(np.percentile(np.abs(uncapped), 50.0))
    assert limit > 0.0 and np.any(np.abs(uncapped) > limit), (
        "cap would be inert on this table"
    )

    setattr(pot, "_acecg_max_force", limit)
    try:
        out_dir = tmp_path / "capped"
        out_dir.mkdir()
        WriteLmpFF(
            str(source_dir / "system.settings"),
            str(out_dir / "system.settings"),
            forcefield,
            pair_style="hybrid",
            pair_typ_sel=["table"],
            topology_arrays=dopc_topology_arrays(),
        )
        section = read_lammps_table_section(out_dir / "Pair_HG-HG.table")
    finally:
        delattr(pot, "_acecg_max_force")

    assert np.max(np.abs(section.force)) <= limit + 1.0e-9
    np.testing.assert_allclose(
        section.force, np.clip(uncapped, -limit, limit), rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        section.potential,
        integrate_force_to_potential(section.x, np.clip(uncapped, -limit, limit)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
