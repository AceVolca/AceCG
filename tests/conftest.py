"""Make the live package and repository-local test helpers importable."""
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

_tests = Path(__file__).resolve().parent
if str(_tests) not in sys.path:
    sys.path.insert(0, str(_tests))
