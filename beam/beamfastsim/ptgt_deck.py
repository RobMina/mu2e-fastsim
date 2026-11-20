"""A production-target group of a G4beamline input file as a device Target (opt-in; Mu2e layer of ptgt_device).

Reads the solids (`tubs`, `box`) and the `place` lines of one `group ... endgroup` block of a generated target file
(e.g. the deck's Geometry/Proton_Target_Hayman2.txt, group pTarget: cores, end rings, fins) and builds the Target:
each tubs is a coaxial segment over its own z range, each box an axis-aligned box in the group frame. Only what the
device geometry represents exactly is accepted: tubs on the group axis without a phi section, and boxes rotated about
z by multiples of 90 degrees (anything else raises). Numbers are read as written (the generated files hold them to
1e-6 mm); `$param` references and expressions are refused rather than guessed.

The group's placement in the deck (Hayman2: `place pTarget ... rotation=Z-30.0,Y14.0,...` inside pTarget_group, at
z = Tposition) is the caller's: `rotation(spec)` turns a G4beamline rotation string into the matrix.
"""
import re

import numpy as np

from beamfastsim import ptgt_device as D

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _kv(text):
    """key=value pairs of one command (values: numbers or plain words)."""
    out = {}
    for k, v in re.findall(r"(\w+)=('[^']*'|\S+)", text):
        out[k] = v.strip("'")
    return out


def _num(v, what):
    if not re.fullmatch(_NUM, v):
        raise ValueError(f"{what}: {v!r} is not a plain number (parameters and expressions are not evaluated)")
    return float(v)


def _commands(path):
    """Logical lines (backslash continuations joined, comments dropped)."""
    text = open(path, encoding="latin-1").read()
    text = re.sub(r"\\\s*\n", " ", text)
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield line


def rotation(spec):
    """Matrix of a G4beamline rotation string ('Z-30,Y14,X0'): the rotations in order about the fixed axes."""
    R = np.eye(3)
    for axis, ang in re.findall(rf"([XYZxyz])\s*({_NUM})", spec or ""):
        a = np.deg2rad(float(ang))
        c, s = np.cos(a), np.sin(a)
        m = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
             "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis.lower()]
        R = np.asarray(m) @ R
    return R


def read_group(path, group="pTarget"):
    """(solids, placements) of one group: solids {name: (kind, params)}, placements [(name, params)]."""
    solids, places, inside = {}, [], False
    for line in _commands(path):
        word = line.split()[0]
        if word in ("tubs", "box"):
            name = line.split()[1]
            solids[name] = (word, _kv(line))
        elif word == "group" and line.split()[1] == group:
            inside = True
        elif word == "endgroup" and inside:
            break
        elif word == "place" and inside:
            places.append((line.split()[1], _kv(line)))
    if not places:
        raise ValueError(f"{path}: no placements in group {group!r}")
    return solids, places


def target(path, materials, group="pTarget", **placement):
    """Target of one group of a generated target file; materials: the bank's material names (index = position).
    placement: origin, rot of the group in the caller's frame."""
    solids, places = read_group(path, group)
    z_ranges, r_in, r_out, cmat, boxes, bmat = [], [], [], [], [], []
    for name, p in places:
        if name not in solids:
            raise ValueError(f"{path}: {name!r} placed in {group!r} is not a tubs or box of this file")
        kind, s = solids[name]
        x, y, z = (_num(p.get(k, "0"), f"{name} {k}") for k in ("x", "y", "z"))
        mat = s.get("material", "")
        if mat not in materials:
            raise ValueError(f"{name}: material {mat!r} not in the bank {tuple(materials)}")
        rot = p.get("rotation", "Z0")
        if kind == "tubs":
            if x or y or re.search(r"[XYxy]", rot) or "initialPhi" in s or "finalPhi" in s:
                raise ValueError(f"{name}: only tubs on the group axis without a phi section")
            h = 0.5 * _num(s["length"], f"{name} length")
            ri, ro = _num(s.get("innerRadius", "0"), f"{name} innerRadius"), _num(s["outerRadius"], f"{name} outerRadius")
            z_ranges.append([z - h, z + h]); r_in.append([ri, ri]); r_out.append([ro, ro]); cmat.append(materials.index(mat))
        else:
            ang = re.fullmatch(rf"[Zz]\s*({_NUM})", rot.replace(" ", ""))
            if not ang or abs(float(ang.group(1)) / 90.0 - round(float(ang.group(1)) / 90.0)) > 1e-9:
                raise ValueError(f"{name}: only boxes rotated about z by multiples of 90 degrees (got {rot!r})")
            hx = 0.5 * _num(s["width"], f"{name} width")
            hy = 0.5 * _num(s["height"], f"{name} height")
            hz = 0.5 * _num(s["length"], f"{name} length")
            if round(float(ang.group(1)) / 90.0) % 2:
                hx, hy = hy, hx
            boxes.append([x - hx, x + hx, y - hy, y + hy, z - hz, z + hz]); bmat.append(materials.index(mat))
    if not z_ranges:
        raise ValueError(f"{path}: group {group!r} has no coaxial solid")
    zr = np.asarray(z_ranges, float)
    return D.Target(np.array([zr[:, 0].min(), zr[:, 1].max()]), np.asarray(r_in, float), np.asarray(r_out, float),
                    np.asarray(cmat + bmat), z_ranges=zr, boxes=np.asarray(boxes, float).reshape(-1, 6), **placement)
