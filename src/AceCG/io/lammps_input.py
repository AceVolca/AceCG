"""Lightweight LAMMPS input-file lexical utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator


def strip_lines(text: str) -> list[str]:
    """Return logical command lines with comments removed and continuations joined."""
    raw_lines: list[str] = []
    for line in text.splitlines():
        comment_start = line.find("#")
        if comment_start >= 0:
            line = line[:comment_start]
        raw_lines.append(line.rstrip())

    lines: list[str] = []
    buffer = ""
    for line in raw_lines:
        if line.endswith("&"):
            buffer += line[:-1] + " "
            continue
        buffer += line
        command = buffer.strip()
        if command:
            lines.append(command)
        buffer = ""

    if buffer.strip():
        lines.append(buffer.strip())
    return lines


def tokenize_line(line: str) -> tuple[str, ...]:
    """Return whitespace tokens from one decommented LAMMPS command line."""
    body = line.split("#", 1)[0].strip()
    if not body:
        return ()
    return tuple(body.split())


def tokenize_lines(lines: Iterable[str]) -> Iterator[tuple[str, ...]]:
    """Yield non-empty token tuples from LAMMPS command lines."""
    for line in lines:
        tokens = tokenize_line(line)
        if tokens:
            yield tokens


def resolve_include_path(include_token: str, base_dir: str | Path) -> Path:
    """Resolve one literal LAMMPS ``include`` path relative to *base_dir*."""
    if "$" in include_token:
        raise ValueError(
            f"Unsupported variable-expanded include path {include_token!r}. "
            "LAMMPS input inspection requires a literal include path."
        )
    include_path = Path(include_token)
    if not include_path.is_absolute():
        include_path = Path(base_dir) / include_path
    if not include_path.exists():
        raise FileNotFoundError(f"Included LAMMPS file does not exist: {include_path}")
    return include_path.resolve()


def resolve_lines(script_path: str | Path) -> list[str]:
    """Read *script_path* and recursively inline literal ``include`` directives."""
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"LAMMPS input file not found: {path}")

    lines = strip_lines(path.read_text(encoding="utf-8"))
    merged: list[str] = []
    for line in lines:
        tokens = tokenize_line(line)
        if tokens and tokens[0] == "include" and len(tokens) >= 2:
            merged.extend(resolve_lines(resolve_include_path(tokens[1], path.parent)))
        else:
            merged.append(line)
    return merged


def iter_commands(script_path: str | Path, *, resolve_includes: bool = True) -> Iterator[tuple[str, ...]]:
    """Yield tokenized LAMMPS commands from *script_path*."""
    path = Path(script_path)
    if resolve_includes:
        lines = resolve_lines(path)
    else:
        if not path.exists():
            raise FileNotFoundError(f"LAMMPS input file not found: {path}")
        lines = strip_lines(path.read_text(encoding="utf-8"))
    yield from tokenize_lines(lines)


__all__ = [
    "iter_commands",
    "resolve_include_path",
    "resolve_lines",
    "strip_lines",
    "tokenize_line",
    "tokenize_lines",
]
