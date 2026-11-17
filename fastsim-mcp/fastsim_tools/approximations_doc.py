"""The approximations of each workflow, read from docs/approximations.md.

The document's table is generated from the packages' declarations (``python3 -m mu2efastsim.approximations``) and a
test keeps the two in sync, so reading the table gives the declarations without importing JAX-side packages. The
"Notes per workflow" bullets come with it. Pure text parsing: no MCP, no repository imports.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

TABLE_START = "<!-- table: python3 -m mu2efastsim.approximations -->"
TABLE_END = "<!-- end table -->"
COLUMNS = ("workflow", "approximation", "levels", "fast", "fidelity", "stock_alternative", "switch")
PRESETS = ("default", "fast", "fidelity")


def _cells(line: str) -> List[str]:
    # cells are separated by " | "; a cell may contain a bare "|" (e.g. "(|p|, tan dip)")
    inner = line.strip()
    if not (inner.startswith("|") and inner.endswith("|")):
        raise ValueError(f"not a table row: {line[:60]!r}")
    return [c.strip() for c in inner[1:-1].strip().split(" | ")]


def _row(cells: List[str]) -> Dict:
    if len(cells) != len(COLUMNS):
        raise ValueError(f"approximations table row has {len(cells)} cells, expected {len(COLUMNS)}: {cells[:2]}")
    r = dict(zip(COLUMNS, cells))
    m = re.match(r"`([^`]+)`(?:: (.*)| \(no switch\))?$", r["approximation"], re.S)
    if m is None:
        raise ValueError(f"approximation cell without a `name`: {r['approximation'][:60]!r}")
    name, rest = m.group(1), (m.group(2) or "").strip()
    switchable = "(no switch)" not in r["approximation"] and r["levels"] != "always on"
    speed_only = rest.endswith("(speed only)")
    if speed_only:
        rest = rest[: -len("(speed only)")].strip()
    levels = [] if not switchable else [x.strip() for x in r["levels"].split(",")]
    alternative, _, stock = r["stock_alternative"].partition(": ")
    return {
        "workflow": r["workflow"], "name": name, "switchable": switchable,
        "description": rest if switchable else "",
        "levels": levels, "default": levels[0] if levels else None,
        "fast": r["fast"] if switchable else None,
        "fidelity": (None if r["fidelity"] in ("no path yet", "no level") else r["fidelity"]) if switchable else None,
        "stock_alternative": alternative if switchable else "none",
        "stock": stock if switchable else r["stock_alternative"],
        "switch": r["switch"] if switchable else None,
        "speed_only": speed_only,
    }


def parse_table(text: str) -> List[Dict]:
    """Rows of the generated table, one dict per approximation (switchable or fixed)."""
    try:
        body = text.split(TABLE_START, 1)[1].split(TABLE_END, 1)[0]
    except IndexError as exc:
        raise ValueError("docs/approximations.md has no generated table markers") from exc
    lines = [ln for ln in body.strip().splitlines() if ln.strip()]
    if len(lines) < 2 or not lines[1].startswith("|---"):
        raise ValueError("approximations table has no header")
    return [_row(_cells(ln)) for ln in lines[2:]]


def parse_notes(text: str) -> Dict[str, str]:
    """{bullet label: text} of the "Notes per workflow" section (labels like "W1", "W2-W4", "W6, W8")."""
    m = re.search(r"^## Notes per workflow\s*$(.*?)(?=^## |\Z)", text, re.S | re.M)
    if m is None:
        return {}
    notes: Dict[str, str] = {}
    for block in re.split(r"^- \*\*", m.group(1), flags=re.M)[1:]:
        label, _, rest = block.partition("**")
        label = label.strip().rstrip(".")
        notes[label] = re.sub(r"\s+", " ", rest).strip()
    return notes


def notes_for(notes: Dict[str, str], workflow: str) -> List[str]:
    """The note bullets whose label names ``workflow`` ("W6" matches "W6, W8" and "W6 shared reuse table")."""
    out = []
    for label, body in notes.items():
        names = re.findall(r"W\d(?:-W\d)?", label)
        if workflow in names:
            out.append((label + ". " + body) if label not in (workflow,) else body)
    return out


def presets_table(text: str) -> Dict[str, str]:
    m = re.search(r"\| preset \| each approximation at \|\n\|---\|---\|\n((?:\|.*\|\n)+)", text)
    if m is None:
        return {}
    out = {}
    for ln in m.group(1).splitlines():
        cells = _cells(ln)
        out[cells[0].strip("`")] = cells[1]
    return out


def describe(text: str, workflow: Optional[str] = None, preset: Optional[str] = None) -> Dict:
    """Approximations of one workflow (or all), with the levels a preset would select."""
    rows = parse_table(text)
    names = sorted({r["workflow"] for r in rows}, key=lambda w: [int(x) for x in re.findall(r"\d+", w)])
    if workflow is not None:
        if workflow not in names:
            raise ValueError(f"no approximations for workflow {workflow!r}; documented: {', '.join(names)}")
        rows = [r for r in rows if r["workflow"] == workflow]
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {PRESETS}")
    notes = parse_notes(text)
    workflows = {}
    for w in ([workflow] if workflow else names):
        mine = [r for r in rows if r["workflow"] == w]
        entry = {"switchable": [r for r in mine if r["switchable"]], "fixed": [r for r in mine if not r["switchable"]],
                 "notes": notes_for(notes, w)}
        if preset is not None:
            entry["selected"] = {r["name"]: (r["default"] if preset == "default" else
                                             (r[preset] or r["default"])) for r in entry["switchable"]}
        workflows[w] = entry
    return {"workflows": workflows, "presets": presets_table(text),
            "options": "--approx-preset default|fast|fidelity, --approx NAME=LEVEL, --approximations FILE"}


def read(root: Path) -> str:
    path = Path(root) / "docs" / "approximations.md"
    return path.read_text(encoding="utf-8")
