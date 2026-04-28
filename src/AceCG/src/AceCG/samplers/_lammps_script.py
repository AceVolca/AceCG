"""LAMMPS input-script parser used by the sampler layer.

The sampler only needs a small structural understanding of the script:

- exactly one ``read_data``
- no ``read_restart``
- the last coordinate-bearing ``dump``
- the last ``write_data`` if replay mode is desired
- the relative include tree needed to run the copied script from a replica dir

It intentionally does not try to parse the whole LAMMPS language.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LammpsScriptInfo:
    """Metadata extracted from a LAMMPS input script."""

    input_path: Path
    read_data_path: Path
    last_dump_path: Path
    last_dump_style: str
    write_data_path: Optional[Path]

    @property
    def init_data_path(self) -> Path:
        """Return the initial LAMMPS data file path referenced by ``read_data``."""
        return self.read_data_path

    @property
    def trajectory_path(self) -> Path:
        """Return the trajectory dump path produced by the script."""
        return self.last_dump_path

    @property
    def checkpoint_path(self) -> Optional[Path]:
        """Return the replay/checkpoint data path produced by ``write_data``."""
        return self.write_data_path


_COORD_DUMP_STYLES = {"atom", "custom"}
_COORD_FIELD_TRIPLETS = (
    {"x", "y", "z"},
    {"xu", "yu", "zu"},
    {"xs", "ys", "zs"},
)
_STYLE_CMDS = frozenset({"pair_style", "bond_style", "angle_style", "dihedral_style"})
_COEFF_CMDS = frozenset({"pair_coeff", "bond_coeff", "angle_coeff", "dihedral_coeff"})


def _is_table_style(name: str) -> bool:
    low = name.lower()
    return low == "table" or low.startswith("table/") or low.startswith("table_")


def _strip_comments_and_continuations(text: str) -> list[str]:
    """Strip trailing ``#`` comments and join ``&`` continuations."""
    raw: list[str] = []
    for line in text.splitlines():
        idx = line.find("#")
        if idx >= 0:
            line = line[:idx]
        raw.append(line.rstrip())

    joined: list[str] = []
    buf = ""
    for line in raw:
        if line.endswith("&"):
            buf += line[:-1] + " "
        else:
            buf += line
            s = buf.strip()
            if s:
                joined.append(s)
            buf = ""
    if buf.strip():
        joined.append(buf.strip())
    return joined


def _resolve_include_path(include_token: str, base_dir: Path) -> Path:
    """Resolve one include path or fail fast if it needs runtime substitution."""
    if "$" in include_token:
        raise ValueError(
            f"Unsupported variable-expanded include path {include_token!r}. "
            "Sampler-side script inspection requires a literal include path."
        )
    inc_path = Path(include_token)
    if not inc_path.is_absolute():
        inc_path = base_dir / inc_path
    if not inc_path.exists():
        raise FileNotFoundError(f"Included LAMMPS file does not exist: {inc_path}")
    return inc_path.resolve()


def _resolve_lines(script_path: Path) -> list[str]:
    """Read *script_path*, recursively inlining ``include`` directives."""
    if not script_path.exists():
        raise FileNotFoundError(f"LAMMPS script not found: {script_path}")
    text = script_path.read_text(encoding="utf-8")
    raw_lines = _strip_comments_and_continuations(text)
    base_dir = script_path.parent

    merged: list[str] = []
    for line in raw_lines:
        tokens = line.split()
        if tokens and tokens[0] == "include" and len(tokens) >= 2:
            merged.extend(_resolve_lines(_resolve_include_path(tokens[1], base_dir)))
        else:
            merged.append(line)
    return merged


def _collect_input_tree(script_path: Path) -> list[Path]:
    """Return the script plus all recursively included files.

    Relative includes are staged into each replica directory so the copied input
    script can run there without mutating user files.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"LAMMPS script not found: {script_path}")

    files: list[Path] = []
    seen: set[Path] = set()

    def _visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        files.append(resolved)

        text = resolved.read_text(encoding="utf-8")
        for line in _strip_comments_and_continuations(text):
            tokens = line.split()
            if tokens and tokens[0] == "include" and len(tokens) >= 2:
                _visit(_resolve_include_path(tokens[1], resolved.parent))

    _visit(script_path)

    root_dir = script_path.resolve().parent
    resolved_lines = _resolve_lines(script_path)

    style_map: dict[str, str] = {}
    for line in resolved_lines:
        tokens = line.split()
        if tokens and tokens[0] in _STYLE_CMDS and len(tokens) >= 2:
            style_map[tokens[0].split("_")[0]] = tokens[1]

    for line in resolved_lines:
        tokens = line.split()
        if not tokens or tokens[0] not in _COEFF_CMDS:
            continue
        kind = tokens[0].split("_")[0]
        sty = style_map.get(kind)
        file_tok = None

        if sty in ("hybrid", "hybrid/overlay"):
            start = 3 if kind == "pair" else 2
            for i in range(start, len(tokens)):
                if _is_table_style(tokens[i]) and i + 1 < len(tokens):
                    file_tok = tokens[i + 1]
                    break
        elif sty is not None and _is_table_style(sty):
            idx = 3 if kind == "pair" else 2
            if idx < len(tokens):
                file_tok = tokens[idx]

        if file_tok and not file_tok.startswith("$"):
            raw = root_dir / file_tok
            resolved = raw.resolve()
            if resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                files.append(raw)

    return files


def stage_lammps_input_tree(script_path: Path, dest_root: Path) -> Path:
    """Copy the script and its include tree under *dest_root*.

    Only the relative layout under the root script directory is preserved.
    This keeps replica directories self-contained while still respecting the
    original script structure.
    """
    import shutil

    root = script_path.resolve()
    root_dir = root.parent
    staged_root = dest_root / root.name

    for src in _collect_input_tree(root):
        try:
            rel = src.relative_to(root_dir)
        except ValueError as exc:
            raise ValueError(
                f"Included file {src} escapes the root script directory {root_dir}. "
                "Replica staging only supports include trees under the main script directory."
            ) from exc
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return staged_root


def _has_coordinate_triplet(fields: list[str]) -> bool:
    field_set = set(fields)
    return any(triplet.issubset(field_set) for triplet in _COORD_FIELD_TRIPLETS)


def parse_lammps_script(script_path: Path) -> LammpsScriptInfo:
    """Parse a LAMMPS input script and extract sampler-relevant metadata."""
    lines = _resolve_lines(script_path)

    read_data_paths: list[str] = []
    has_read_restart = False
    dumps: list[tuple[str, str]] = []
    write_data_paths: list[str] = []

    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        cmd = tokens[0]

        if cmd == "read_data" and len(tokens) >= 2:
            read_data_paths.append(tokens[1])
        elif cmd == "read_restart":
            has_read_restart = True
        elif cmd == "dump" and len(tokens) >= 6:
            style, fpath = tokens[3], tokens[5]
            if style in _COORD_DUMP_STYLES:
                if style == "custom":
                    if _has_coordinate_triplet(tokens[6:]):
                        dumps.append((style, fpath))
                else:
                    dumps.append((style, fpath))
        elif cmd == "write_data" and len(tokens) >= 2:
            write_data_paths.append(tokens[1])

    if not read_data_paths:
        raise ValueError(
            f"LAMMPS script {script_path} has no 'read_data' command. "
            "AceCG requires exactly one read_data."
        )
    if len(read_data_paths) > 1:
        raise ValueError(
            f"LAMMPS script {script_path} has {len(read_data_paths)} 'read_data' "
            "commands. AceCG requires exactly one."
        )
    if has_read_restart:
        raise ValueError(
            f"LAMMPS script {script_path} contains 'read_restart'. "
            "AceCG only supports 'read_data'-based initialization."
        )
    if not dumps:
        raise ValueError(
            f"LAMMPS script {script_path} has no coordinate-bearing dump command. "
            "Need a final dump that writes a complete coordinate triplet."
        )

    last_style, last_dump = dumps[-1]
    return LammpsScriptInfo(
        input_path=script_path,
        read_data_path=Path(read_data_paths[0]),
        last_dump_path=Path(last_dump),
        last_dump_style=last_style,
        write_data_path=Path(write_data_paths[-1]) if write_data_paths else None,
    )
