"""Audit G4BL header domains against a GDML box world without reading field payloads."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_header(path):
    text = path.read_text()
    grids = re.findall(r"^\s*grid\s+(.+)$", text, re.MULTILINE)
    require(len(grids) == 1, "Need one grid")
    fields = dict(re.findall(r"(\w+)=([-+\d.eE]+)", grids[0]))
    require(set(fields) == set("X0 Y0 Z0 nX nY nZ dX dY dZ".split()), "Unknown/incomplete grid")
    origin = re.findall(r"^\s*#\s*Origin shift for Mu2e:\s*(.+)$", text, re.MULTILINE)
    require(len(origin) <= 1, "Duplicate origin")
    shift = list(map(float, origin[0].split())) if origin else [-3904.0, 0.0, 7929.0]
    require(len(shift) == 3 and all(math.isfinite(v) for v in shift), "Bad origin shift")
    extensions = re.findall(r"^\s*(extend\S*.*)$", text, re.MULTILINE)
    require(extensions in ([], ["extendY flip=By"]), "Unsupported map extension")
    lo, hi, spacing, counts = [], [], [], []
    for i, axis in enumerate("XYZ"):
        n = float(fields["n" + axis])
        d = float(fields["d" + axis])
        lower = float(fields[axis + "0"]) - shift[i]
        require(n.is_integer() and n >= 3 and math.isfinite(d) and d > 0 and math.isfinite(lower), "Bad grid")
        lo.append(lower)
        hi.append(lower + (int(n) - 1) * d)
        spacing.append(d)
        counts.append(int(n))
    full_lo, full_counts = lo.copy(), counts.copy()
    if extensions:
        require(lo[1] == 0, "Reflection requires y=0 lower boundary")
        full_lo[1] = -hi[1]
        full_counts[1] = 2 * counts[1] - 1
    return dict(path=str(path), sha256=sha(path), origin_shift_mm=shift, origin_declared=bool(origin),
                dimensions=counts, spacing_mm=spacing, stored_lower_mu2e_mm=lo, upper_mu2e_mm=hi,
                extend_y_flip_by=bool(extensions), full_lower_mu2e_mm=full_lo,
                full_dimensions=full_counts, stored_scalar_count=3 * math.prod(counts),
                full_scalar_count=3 * math.prod(full_counts))


def world_box(path):
    root = ET.parse(path).getroot()
    worlds = root.findall("./setup/world")
    require(len(worlds) == 1, "Need exactly one GDML world")
    name = worlds[0].attrib["ref"]
    volumes = [v for v in root.findall("./structure/volume") if v.attrib["name"] == name]
    require(len(volumes) == 1, "Unresolved world")
    solid = volumes[0].find("solidref").attrib["ref"]
    boxes = [b for b in root.findall("./solids/box") if b.attrib["name"] == solid]
    require(len(boxes) == 1, "Only explicit box worlds supported")
    box = boxes[0]
    unit = box.attrib.get("lunit", "mm")
    require(unit in ("mm", "cm", "m"), "Unsupported world unit")
    scale = {"mm": 1, "cm": 10, "m": 1000}[unit]
    size = [float(box.attrib[a]) * scale for a in "xyz"]
    require(all(math.isfinite(v) and v > 0 for v in size), "Invalid world size")
    return dict(path=str(path), sha256=sha(path), volume=name, full_size_mm=size,
                lower_world_mm=[-v / 2 for v in size], upper_world_mm=[v / 2 for v in size])


def audit(headers, gdml, origin):
    require(len(origin) == 3 and all(math.isfinite(v) for v in origin), "Invalid Mu2e origin in world")
    world = world_box(gdml)
    maps = []
    for path in headers:
        row = parse_header(path)
        lo = [v + o + d for v, o, d in zip(row["full_lower_mu2e_mm"], origin, row["spacing_mm"])]
        hi = [v + o - d for v, o, d in zip(row["upper_mu2e_mm"], origin, row["spacing_mm"])]
        margins = [min(wlo - mlo, mhi - whi) for wlo, whi, mlo, mhi in
                   zip(world["lower_world_mm"], world["upper_world_mm"], lo, hi)]
        row.update(inset_lower_world_mm=lo, inset_upper_world_mm=hi, world_margin_mm=margins,
                   contains_world_with_cell_margin=all(m >= 0 for m in margins),
                   fits_adapter_scalar_cap=row["full_scalar_count"] <= 30000000)
        maps.append(row)
    return dict(format="m2fs-map-header-coverage-v1", world=world, mu2e_origin_world_mm=origin, maps=maps,
                verdict="covered" if any(r["contains_world_with_cell_margin"] for r in maps) else "uncovered",
                scope="Header/domain audit only; not payload, composite interpolation or matched-reference validation",
                trust="Header object integrity from downloader is not authenticated CVMFS root provenance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("headers", nargs="+", type=Path)
    parser.add_argument("--gdml", required=True, type=Path)
    parser.add_argument("--mu2e-origin-world-mm", required=True, nargs=3, type=float)
    args = parser.parse_args()
    print(json.dumps(audit(args.headers, args.gdml, args.mu2e_origin_world_mm), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
