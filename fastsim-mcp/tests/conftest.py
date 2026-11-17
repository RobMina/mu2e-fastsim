"""Tests of the fastsim MCP server: the tool package with a stand-in kit, and the generic wrapper.

Paths: this directory's parent (the two packages) and the umbrella's core and beam (the kit's geometry rules), so
the suite also runs without PYTHONPATH from inside fastsim-mcp/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, ROOT / "core", ROOT / "beam", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
