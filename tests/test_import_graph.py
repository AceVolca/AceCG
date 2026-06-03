"""AST-based layer violation scanner for topology/, io/, compute/."""

from __future__ import annotations

import ast
import pathlib
from typing import List, Tuple


# Root of the source tree
SRC_ROOT = pathlib.Path(__file__).parent.parent / "src" / "AceCG"

# Current structure-allowed exceptions, keyed by (file, exact imported module).
#
# Keep this list exact. The goal is to document the specific cross-layer edges
# the repo currently accepts, not to exempt whole files from future checks.
KNOWN_EXCEPTIONS: set[Tuple[str, str]] = {
    ("compute/mpi_engine.py", "AceCG.io.logger"),
    ("compute/mpi_engine.py", "AceCG.io.trajectory"),
    # Canonical PBC wrap lives with the other coordinate/PBC helpers in
    # io.coordinates; the noise path reuses it instead of a duplicate.
    ("compute/mpi_engine.py", "AceCG.io.coordinates"),
    ("compute/vp_prepare.py", "AceCG.io.trajectory"),
}


def _absolute_import(node: ast.ImportFrom, filepath: pathlib.Path) -> str:
    """Resolve a relative import to an absolute module path string."""
    if node.level == 0:
        return node.module or ""
    parts = list(filepath.relative_to(SRC_ROOT.parent).parts)
    package_parts = parts[:-1]
    up = node.level - 1
    if up > 0:
        package_parts = package_parts[:-up] if up < len(package_parts) else []
    base = ".".join(package_parts)
    mod = node.module or ""
    return f"{base}.{mod}" if mod else base


def collect_imports(filepath: pathlib.Path) -> List[Tuple[int, str]]:
    """Return ``[(lineno, absolute_module_path)]`` for all imports in a Python file."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            abs_mod = _absolute_import(node, filepath)
            results.append((node.lineno, abs_mod))
    return results


def scan_violations() -> List[str]:
    """Walk topology/, io/, compute/ and collect layer-violation imports."""
    violations: List[str] = []

    L1_forbidden_prefixes = {
        "AceCG.compute",
        "AceCG.trainers",
        "AceCG.workflows",
        "AceCG.schedulers",
        "AceCG.samplers",
        "AceCG.solvers",
    }
    L2_forbidden_prefixes = {
        "AceCG.trainers",
        "AceCG.workflows",
        "AceCG.schedulers",
        "AceCG.samplers",
        "AceCG.solvers",
        "AceCG.io",
    }

    for layer_dir, forbidden in (
        ("topology", L1_forbidden_prefixes),
        ("io", L1_forbidden_prefixes),
        ("compute", L2_forbidden_prefixes),
    ):
        layer_root = SRC_ROOT / layer_dir
        if not layer_root.exists():
            violations.append(f"MISSING: {layer_root} does not exist")
            continue

        for pyfile in sorted(layer_root.rglob("*.py")):
            if "__pycache__" in str(pyfile):
                continue

            rel = str(pyfile.relative_to(SRC_ROOT))
            imports = collect_imports(pyfile)
            for lineno, mod in imports:
                for prefix in forbidden:
                    if mod == prefix or mod.startswith(prefix + "."):
                        key = (rel, mod)
                        if key not in KNOWN_EXCEPTIONS:
                            violations.append(
                                f"VIOLATION: {rel}:{lineno} imports {mod!r}"
                                f" (forbidden for {layer_dir}/ layer)"
                            )
                        break

    return violations


def test_no_layer_violations():
    """Assert zero layer-violation imports across topology/, io/, compute/."""
    violations = scan_violations()
    if violations:
        msg = "\n".join(violations)
        raise AssertionError(
            f"Found {len(violations)} layer violation(s):\n{msg}\n\n"
            "If a violation is intentionally allowed by the current structure, add it to "
            "KNOWN_EXCEPTIONS in tests/test_import_graph.py."
        )
    n_files = sum(
        1
        for d in ("topology", "io", "compute")
        for f in (SRC_ROOT / d).rglob("*.py")
        if "__pycache__" not in str(f)
    )
    assert n_files > 0, "No files were scanned — check SRC_ROOT path"