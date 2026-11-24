"""Production targets of the G4beamline deck as device Targets (opt-in; Mu2e layer of ptgt_device): a target group of a generated target file, and the RAL target built from its parameters.

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


def ral_target(mat=0, tlength=160.0, tradius=3.15, hub_thick=1.0, hub_pre=8.0, hub_post=4.0, hub_length=16.0,
               hub_attach=9.0, hub_angle=45.0, ball_radius=2.25, up_attach=105.0, down_attach=15.0, attach_inc=120.0,
               hub_gap=0.1, anchor_angle=65.0, anchor_length=35.0, ring_rin=176.0, ring_length=19.05, rod_radius=0.5,
               **placement):
    """The RAL radiation-cooled target of the deck (Geometry/Proton_Target_RAL.txt, Use_Proton_Target=6, defaults =
    the deck's): the rod, the two conical hubs (polycones -> coaxial frusta), the six hub balls and the six support
    rods (spokes, hub to support ring, as solid cylinders), all of one material. Not included: the support ring,
    its latches and pins and the HRS-liner support ring (other materials, at the HRS wall). The spokes' ring ends do
    not tilt with the target (the deck's convention): placement['rot'] (default the deck's Y14) maps them into the
    local frame. The beam starts at the rod face (beam_z), inside the upstream hub, as in the deck."""
    if "rot" not in placement:
        placement["rot"] = rotation("Y14")
    rot = np.asarray(placement["rot"], float)
    h, a, ca, sa = 0.5 * tlength, np.deg2rad(hub_angle), np.cos(np.deg2rad(hub_angle)), np.sin(np.deg2rad(hub_angle))
    r1, rh = tradius + hub_thick, tradius + hub_length * sa
    zf = h + hub_post + hub_length * ca                    # outer end of the hub flare
    zr = [[-h, h], [h - hub_pre, h + hub_post], [h + hub_post, zf], [-h - hub_post, -h + hub_pre], [-zf, -h - hub_post]]
    r_in = [[0, 0], [tradius, tradius], [tradius, rh], [tradius, tradius], [rh, tradius]]
    r_out = [[tradius, tradius], [r1, r1], [r1, rh + hub_thick], [r1, r1], [rh + hub_thick, r1]]
    zoff = h + hub_post + hub_attach * ca + ball_radius * sa
    yoff = r1 + hub_attach * sa - ball_radius * ca
    zpre = h + hub_post + hub_attach * ca - hub_gap * sa
    ypre = r1 + hub_attach * sa + hub_gap * ca
    zring = 0.5 * ring_length + anchor_length * np.cos(np.deg2rad(anchor_angle))
    yring = ring_rin - anchor_length * np.sin(np.deg2rad(anchor_angle))
    balls, rods = [], []
    for sgn, att in ((1.0, up_attach), (-1.0, down_attach)):
        for i in range(3):
            th = np.deg2rad(att + i * attach_inc)
            balls.append([-np.sin(th) * yoff, np.cos(th) * yoff, sgn * zoff, ball_radius])
            hub = np.array([-np.sin(th) * ypre, np.cos(th) * ypre, sgn * zpre])
            ring = rot.T @ np.array([-np.sin(th) * yring, np.cos(th) * yring, -sgn * zring])
            u = (ring - hub) / np.linalg.norm(ring - hub)
            rods.append([*(hub + rod_radius * u), *(ring - rod_radius * u), rod_radius])
    n = len(zr) + len(balls) + len(rods)
    return D.Target(np.array([-zf, zf]), np.asarray(r_in, float), np.asarray(r_out, float), np.full(n, mat),
                    z_ranges=np.asarray(zr, float), spheres=np.asarray(balls, float), cylinders=np.asarray(rods, float),
                    beam_z=h, **placement)
