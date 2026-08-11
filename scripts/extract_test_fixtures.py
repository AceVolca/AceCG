#!/usr/bin/env python
"""Cut the committed real-data fixtures under ``tests/test_data/``.

``data/`` and ``experiments/`` are repository-local and are never published;
``tests/`` is. The fixtures this script writes are therefore shipped artifacts
of the test suite, and they are deliberately tiny: a handful of real frames of
a real system, enough to exercise every convention that matters (box vectors,
minimum image, type multiplicity, bonded geometry, cutoff handling) and nothing
more.

Re-run from the repository root on a machine that has the private working data:

    python scripts/extract_test_fixtures.py

Every selection rule below is deterministic, so the output is reproducible.

Fixtures produced
-----------------
``tests/test_data/dppc_aa/``
    Two whole all-atom DPPC molecules with their real bond graph, elements,
    names and masses. This is the fixture the linear force-mapping constraint
    logic needs: it is the only place in the suite where "which bonds involve a
    hydrogen" is a question with a real answer.

``tests/test_data/dopc_cg6/``
    6-site CG DOPC, the canonical REM / CD-REM working system.

    * ``cg6_patch.data`` -- LAMMPS topology for a compact patch of whole
      lipids, carrying the real box, real masses, real atom/bond/angle types.
    * ``cg6_patch.lammpstrj`` -- five real frames of that patch, with the
      mapped all-atom reference forces that the trajectory already carries.
    * ``ff/`` -- the real REM-init tabulated forcefield, resampled onto a
      coarser but still uniform grid spanning the identical coordinate range.

    The patch is the set of lipids whose head-group site lies within
    ``PATCH_RADIUS`` of the periodic corner of the box, measured under the
    minimum-image convention. Choosing the corner rather than the middle is the
    point: roughly 40% of the sub-cutoff pairs in the patch are separated by a
    box boundary, so a dropped minimum image is a test failure rather than a
    rounding difference. Whole lipids are kept, so bonds and angles are intact;
    atom types, bond types and angle types keep their original ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

DOPC_ROOT = REPO / "data" / "dopc_ld_Pak2019"
DOPC_TOPOLOGY = DOPC_ROOT / "topo" / "cg6.data"
DOPC_TRAJECTORY = DOPC_ROOT / "traj" / "cg6_1.lammpstrj"
DOPC_FF = DOPC_ROOT / "ff_init" / "rem_init"

DOPC_OUT = REPO / "tests" / "test_data" / "dopc_cg6"

SITES_PER_LIPID = 6
PATCH_RADIUS = 32.0
FRAME_INDICES = (0, 40, 80, 120, 160)

# Uniform strides that divide (N - 1), so the resampled table keeps both
# endpoints and stays uniformly spaced -- LAMMPS requires both.
TABLE_STRIDES = {2500: 7, 499: 2, 361: 2}

DOPC_TABLES = (
    "Pair_HG-HG.table", "Pair_HG-MG.table", "Pair_HG-T1.table",
    "Pair_HG-T2.table", "Pair_MG-MG.table", "Pair_MG-T1.table",
    "Pair_MG-T2.table", "Pair_T1-T1.table", "Pair_T1-T2.table",
    "Pair_T2-T2.table", "HG_MG_bon.table", "MG_T1_bon.table",
    "T1_T2_bon.table", "HG_MG_T1_ang.table", "T1_MG_T1_ang.table",
    "MG_T1_T2_ang.table",
)


# ---------------------------------------------------------------------------
# LAMMPS data file
# ---------------------------------------------------------------------------

SECTION_NAMES = ("Masses", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers",
                 "Velocities", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs")


def read_lammps_data(path: Path) -> dict[str, list[str]]:
    """Split a LAMMPS data file into its header and named sections."""
    sections: dict[str, list[str]] = {"__header__": []}
    current = "__header__"
    for line in path.read_text().splitlines():
        bare = line.split("#", 1)[0].strip()
        if bare in SECTION_NAMES:
            current = bare
            sections[current] = []
            continue
        if current == "__header__":
            sections[current].append(line)
        elif bare:
            sections[current].append(bare)
    return sections


def read_dump_frames(path: Path, frame_indices: tuple[int, ...]) -> list[dict]:
    """Read the requested frames of a LAMMPS dump without loading the file."""
    wanted = set(frame_indices)
    last = max(wanted)
    frames: list[dict] = []
    with path.open() as handle:
        index = 0
        while index <= last:
            header = [handle.readline() for _ in range(9)]
            if not header[0]:
                raise ValueError(f"{path} ended before frame {last}")
            n_atoms = int(header[3].strip())
            body = [handle.readline() for _ in range(n_atoms)]
            if index in wanted:
                frames.append(
                    {
                        "timestep": header[1].strip(),
                        "box": [header[5], header[6], header[7]],
                        "columns": header[8].strip(),
                        "rows": [line.split() for line in body],
                    }
                )
            index += 1
    return frames


def select_patch_lipids(frame: dict, n_lipids: int, radius: float) -> np.ndarray:
    """Whole lipids whose head-group site sits near the periodic box corner."""
    values = np.asarray(
        [[float(token) for token in row] for row in frame["rows"]], dtype=float
    )
    order = np.argsort(values[:, 0])
    values = values[order]
    bounds = np.asarray(
        [[float(token) for token in line.split()] for line in frame["box"]], dtype=float
    )
    origin = bounds[:, 0]
    lengths = bounds[:, 1] - bounds[:, 0]

    heads = values[:, 2:5].reshape(n_lipids, SITES_PER_LIPID, 3)[:, 0, :]
    offset = heads[:, :2] - origin[None, :2]
    offset -= lengths[None, :2] * np.rint(offset / lengths[None, :2])
    return np.where(np.hypot(offset[:, 0], offset[:, 1]) < radius)[0]


def write_dopc_topology(out_path: Path, sections: dict, keep_atoms: np.ndarray,
                        frame: dict) -> None:
    """Write the patch topology, renumbering only atom/bond/angle instance ids."""
    remap = {int(old): new + 1 for new, old in enumerate(keep_atoms)}

    atoms = [row.split() for row in sections["Atoms"]]
    atoms.sort(key=lambda row: int(row[0]))
    positions = {
        int(row[0]): row[2:5] for row in frame["rows"]
    }

    kept_atoms = []
    for new_index, old_id in enumerate(keep_atoms, start=1):
        row = atoms[old_id - 1]
        molecule = (new_index - 1) // SITES_PER_LIPID + 1
        x, y, z = positions[int(old_id)]
        kept_atoms.append(
            f"{new_index} {molecule} {row[2]} {row[3]} {x} {y} {z}"
        )

    def keep_bonded(rows: list[str], n_sites: int) -> list[str]:
        out = []
        for row in rows:
            parts = row.split()
            members = [int(token) for token in parts[2:2 + n_sites]]
            if not all(member in remap for member in members):
                continue
            mapped = " ".join(str(remap[member]) for member in members)
            out.append(f"{len(out) + 1} {parts[1]} {mapped}")
        return out

    bonds = keep_bonded(sections["Bonds"], 2)
    angles = keep_bonded(sections["Angles"], 3)

    bounds = [line.split() for line in frame["box"]]
    lines = [
        "LAMMPS Description -- AceCG test fixture, see README.md",
        "",
        f"     {len(kept_atoms)}  atoms",
        f"     {len(bonds)}  bonds",
        f"     {len(angles)}  angles",
        "     0  dihedrals",
        "     0  impropers",
        "",
        "     4  atom types",
        "     3  bond types",
        "     3  angle types",
        "",
        f"  {bounds[0][0]} {bounds[0][1]} xlo xhi",
        f"  {bounds[1][0]} {bounds[1][1]} ylo yhi",
        f"  {bounds[2][0]} {bounds[2][1]} zlo zhi",
        "",
        "Masses",
        "",
    ]
    lines.extend(sections["Masses"])
    lines.extend(["", "Atoms # full", ""])
    lines.extend(kept_atoms)
    lines.extend(["", "Bonds", ""])
    lines.extend(bonds)
    lines.extend(["", "Angles", ""])
    lines.extend(angles)
    lines.append("")
    out_path.write_text("\n".join(lines))
    return len(kept_atoms), len(bonds), len(angles)


def write_dopc_frames(out_path: Path, frames: list[dict],
                      keep_atoms: np.ndarray) -> None:
    """Write the patch frames, keeping every dumped column including forces."""
    remap = {int(old): new + 1 for new, old in enumerate(keep_atoms)}
    chunks: list[str] = []
    for frame in frames:
        rows = sorted(frame["rows"], key=lambda row: int(float(row[0])))
        kept = []
        for row in rows:
            atom_id = int(float(row[0]))
            if atom_id not in remap:
                continue
            kept.append(" ".join([str(remap[atom_id])] + row[1:]))
        kept.sort(key=lambda row: int(row.split()[0]))
        chunks.append("ITEM: TIMESTEP")
        chunks.append(frame["timestep"])
        chunks.append("ITEM: NUMBER OF ATOMS")
        chunks.append(str(len(kept)))
        chunks.append("ITEM: BOX BOUNDS pp pp pp")
        chunks.extend(line.rstrip("\n") for line in frame["box"])
        chunks.append(frame["columns"])
        chunks.extend(kept)
    out_path.write_text("\n".join(chunks) + "\n")


# ---------------------------------------------------------------------------
# Tabulated forcefield
# ---------------------------------------------------------------------------

def resample_table(text: str) -> str:
    """Thin a LAMMPS table onto a coarser grid over the identical range."""
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        tokens = stripped.split()
        if not stripped or stripped.startswith("#") or not tokens[0].upper() == "N":
            out.append(line)
            index += 1
            continue

        n_rows = int(tokens[1])
        stride = TABLE_STRIDES[n_rows]
        rows = []
        cursor = index + 1
        while cursor < len(lines) and len(rows) < n_rows:
            body = lines[cursor].strip()
            cursor += 1
            if not body:
                continue
            rows.append([float(token) for token in body.split()])
        kept = rows[::stride]
        if (n_rows - 1) % stride != 0:
            raise ValueError(f"stride {stride} does not divide N-1={n_rows - 1}")

        header = list(tokens)
        header[1] = str(len(kept))
        if "R" in [token.upper() for token in header]:
            position = [token.upper() for token in header].index("R")
            header[position + 1] = f"{kept[0][1]:.6f}"
            header[position + 2] = f"{kept[-1][1]:.6f}"
        out.append(" ".join(header))
        out.append("")
        for order, row in enumerate(kept, start=1):
            values = " ".join(f"{value:.8e}" for value in row[1:])
            out.append(f"{order} {values}")
        index = cursor
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_dopc_fixture() -> None:
    out_ff = DOPC_OUT / "ff"
    out_ff.mkdir(parents=True, exist_ok=True)

    sections = read_lammps_data(DOPC_TOPOLOGY)
    n_atoms = len(sections["Atoms"])
    n_lipids = n_atoms // SITES_PER_LIPID

    frames = read_dump_frames(DOPC_TRAJECTORY, FRAME_INDICES)
    lipids = select_patch_lipids(frames[0], n_lipids, PATCH_RADIUS)
    keep_atoms = (
        lipids[:, None] * SITES_PER_LIPID + np.arange(1, SITES_PER_LIPID + 1)[None, :]
    ).ravel()

    counts = write_dopc_topology(
        DOPC_OUT / "cg6_patch.data", sections, keep_atoms, frames[0]
    )
    write_dopc_frames(DOPC_OUT / "cg6_patch.lammpstrj", frames, keep_atoms)

    for name in ("system.init", "system.settings"):
        (out_ff / name).write_text((DOPC_FF / name).read_text())
    for name in DOPC_TABLES:
        (out_ff / name).write_text(resample_table((DOPC_FF / name).read_text()))

    print(
        f"dopc_cg6: {len(lipids)} lipids, {counts[0]} atoms, {counts[1]} bonds, "
        f"{counts[2]} angles, {len(frames)} frames"
    )


def build_dppc_aa_fixture() -> None:
    """Cut two whole all-atom DPPC molecules, with bonds, into a PDB.

    Needs the staged Pak DPPC all-atom reference (``data/catalog.yaml`` →
    ``dppc_aa_Pak2019_4608``). Parsing the 38 MB GROMACS topology costs about
    ten seconds and 1.4 GB of memory, which is why this is a maintainer-run
    extraction and not something a test does.

    PDB is the output format because it is the one MDAnalysis writes that
    carries the bond graph (``CONECT``) *and* per-atom elements, which is
    exactly the pair the force-mapping constraint logic reads.
    """
    import MDAnalysis as mda

    staged = Path("/project2/gavoth/weizhixue/data_staging/dppc_aa_Pak2019_4608")
    universe = mda.Universe(str(staged / "NVT.tpr"), str(staged / "NVT.gro"))
    molecules = universe.residues[:2].atoms
    if set(molecules.resnames) != {"DPPC"}:
        raise ValueError("expected the first two residues to be DPPC")

    out_dir = REPO / "tests" / "test_data" / "dppc_aa"
    out_dir.mkdir(parents=True, exist_ok=True)
    molecules.write(str(out_dir / "dppc_2mol.pdb"), bonds="all")

    # The production Martini-12 mapping, narrowed to the two molecules kept
    # above. Only `repeat` and the block list change; every site definition and
    # the CG topology are copied through unmodified.
    import yaml

    source_map = REPO / "experiments" / "debug_cgmap" / "inputs" / "martini_dppc_map.yaml"
    mapping = yaml.safe_load(source_map.read_text())
    first_block = dict(mapping["system"][0])
    if int(first_block["anchor"]) != 0:
        raise ValueError("expected the first mapping block to anchor at atom 0")
    first_block["repeat"] = len(molecules.residues)
    mapping["system"] = [first_block]
    (out_dir / "martini12_2mol_map.yaml").write_text(
        yaml.safe_dump(mapping, sort_keys=False)
    )

    # The bead assignment this project authored for CHARMM36 DPPC. Tiny, ours,
    # and the input the mapping builder actually consumes.
    beads = (
        REPO / "experiments" / "debug_cgmap" / "inputs"
        / "martini_dppc_charmm36_beads.yaml"
    )
    (out_dir / "martini_dppc_charmm36_beads.yaml").write_text(beads.read_text())

    print(
        f"dppc_aa: {molecules.n_atoms} atoms, "
        f"{len(molecules.bonds)} bonds, {len(molecules.residues)} molecules"
    )


def build_protein_fixture() -> None:
    """Copy the helix A/C protein acceptance inputs, whole.

    Nothing is cut here: the starting structure is 1687 atoms (133 KB) and the
    three residue-level mappings are a few tens of KB each, so the honest
    fixture is the real file rather than a slice of it. This is the suite's
    only non-lipid real-data coverage of `topology/cgmap.py`.
    """
    staged = Path("/project2/gavoth/weizhixue/data_staging/helix_AC_prot_1687")
    inputs = REPO / "experiments" / "debug_cgmap" / "inputs"
    out_dir = REPO / "tests" / "test_data" / "protein_helix_ac"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "md_start.pdb").write_text((staged / "md_start.pdb").read_text())
    for scheme in ("1res", "2res", "4site"):
        name = f"prot_{scheme}_map.yaml"
        (out_dir / name).write_text((inputs / name).read_text())
    print("protein_helix_ac: md_start.pdb + 3 residue mappings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=("dopc_cg6", "dppc_aa", "protein_helix_ac", "all"),
        default="all",
        help="Which fixture to regenerate.",
    )
    args = parser.parse_args()
    if args.system in ("dopc_cg6", "all"):
        build_dopc_fixture()
    if args.system in ("dppc_aa", "all"):
        build_dppc_aa_fixture()
    if args.system in ("protein_helix_ac", "all"):
        build_protein_fixture()


if __name__ == "__main__":
    main()
