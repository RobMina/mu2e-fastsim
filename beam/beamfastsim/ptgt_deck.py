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


# ---------------------------------------------------------------------------------------- Offline Stickman v1.0

_SC_LINE = re.compile(r"^\s*(int|double|bool|string|vector<double>|vector<int>|vector<string>)\s+([\w.]+)\s*=\s*(.*?)\s*;",
                      re.S | re.M)


def read_simpleconfig(path):
    """Values of an Offline SimpleConfig geometry file (`type name = value;`, vectors in braces), comments removed.
    #include lines are not followed: the caller passes the file that holds the target block (later lines win)."""
    text = re.sub(r"//[^\n]*", "", open(path).read())
    out = {}
    for typ, name, val in _SC_LINE.findall(text):
        if typ.startswith("vector"):
            items = [v.strip() for v in val.strip().strip("{}").split(",") if v.strip()]
            out[name] = [v.strip('"') for v in items] if typ == "vector<string>" else [float(v) for v in items]
        elif typ == "string":
            out[name] = val.strip().strip('"')
        elif typ == "bool":
            out[name] = val.strip() == "true"
        else:
            out[name] = float(val)
    return out


def stickman_profile(c, n):
    """Lagrange quadratic through c = (c0, c1, c2) at u = 0, 0.5, 1, sampled at u = i / (n - 1): Mu2eBO's plate
    profile (archived prodtarget/prodtarget6d modes; plate 0 is the Mu2e-upstream plate, where the protons leave)."""
    u = np.linspace(0.0, 1.0, int(n))
    c0, c1, c2 = (float(v) for v in c)
    return c0 * (1 - 2 * u) * (1 - u) + c1 * 4 * u * (1 - u) + c2 * u * (2 * u - 1)


def stickman_6d(r_ctrl, t_ctrl, n=35, lug_offset=0.75, clamp_ends=True):
    """(rOut, plateThickness, plateLugThickness) of the 6D plate study (Mu2eBO prodtarget6d): quadratic profiles of
    the core radius and the plate thickness over n plates, lug = thickness + lug_offset, and (since 2026-06-15) the
    end plates' lugs clamped to their thickness."""
    r, t = stickman_profile(r_ctrl, n), stickman_profile(t_ctrl, n)
    lug = t + lug_offset
    if clamp_ends:
        lug[0], lug[-1] = t[0], t[-1]
    return r, t, lug


def stickman_target(cfg, mat=0, r_out=None, thickness=None, lug=None, **placement):
    """The Offline Stickman production target (Mu2eG4 constructStickmanTarget, ProductionTarget_Stickman_v1_0.txt)
    as a device Target in its local frame (z along the target axis; plate 0 at the -z end, the beam enters at +z):

      * plate i: a core cylinder rOut[i] over plateThickness[i] starting at the plate's -z face, and nStickmanFins
        radial fins (boxes from the axis to plateFinOuterRadius, plateFinWidth wide, the plate's thickness, at
        plateFinAngles); the plate pitch is plateLugThickness[i]; the stack starts supportRingLength + 2
        spacerHalfLength above -halfStickmanLength;
      * per fin angle, one solid cylinder of plateLugOuterRadius at plateCenterToLugCenter over the full length:
        the plates' lugs (contiguous: lug = pitch), the rod inside them, the spacers and the support rings' lugs
        (the 0.025-0.05 mm gaps between rod, lug and spacer are filled);
      * at each end, the support ring (annulus supportRingInnerRadius..OuterRadius over supportRingLength) and its
        lug stems (boxes from the ring's inner radius to the lug centre, 2 plateLugOuterRadius wide).

    Not included: the 1 mm fillets at fin-core and fin-lug joints, the support rings' cutouts, and the bicycle-wheel
    support (Al wheel at r 178-197 mm, Inconel spokes): they do not depend on the plate design. halfStickmanLength is
    recomputed from the plates (Mu2eBO's envelope identity). One material (mat) for every solid, as the file
    (Inconel718). Heat slices: one per plate pitch, plus the two end regions (ring and spacer).
    cfg: read_simpleconfig of the geometry file; r_out, thickness, lug override its per-plate vectors."""
    g = lambda k: cfg[k]                                                                       # noqa: E731
    r = np.asarray(g("targetPS_rOut") if r_out is None else r_out, float)
    t = np.asarray(g("targetPS_plateThickness") if thickness is None else thickness, float)
    lp = np.asarray(g("targetPS_plateLugThickness") if lug is None else lug, float)
    n = int(r.size)
    if not (t.size == lp.size == n) or (r_out is None and thickness is None and n != int(g("targetPS_numberOfPlates"))):
        raise ValueError("rOut, plateThickness and plateLugThickness need one value per plate")
    if np.any(lp < t - 1e-9) or np.any(r <= 0) or np.any(t <= 0):
        raise ValueError("need rOut > 0, plateThickness > 0 and plateLugThickness >= plateThickness")
    ring, sph = float(g("targetPS_supportRingLength")), float(g("targetPS_spacerHalfLength"))
    half = ring + 2.0 * sph + 0.5 * float(lp.sum())
    z = -half + ring + 2.0 * sph + np.concatenate([[0.0], np.cumsum(lp)])                    # plate pitch edges
    angles = np.deg2rad(np.asarray(g("targetPS_plateFinAngles"), float))
    if len(angles) != int(g("targetPS_nStickmanFins")):
        raise ValueError("targetPS_plateFinAngles needs one angle per fin")
    fin_r, fin_w = float(g("targetPS_plateFinOuterRadius")), float(g("targetPS_plateFinWidth"))
    lug_c, lug_r = float(g("targetPS_plateCenterToLugCenter")), float(g("targetPS_plateLugOuterRadius"))
    rr_in, rr_out = float(g("targetPS_supportRingInnerRadius")), float(g("targetPS_supportRingOuterRadius"))
    zr = [[z[i], z[i] + t[i]] for i in range(n)] + [[-half, -half + ring], [half - ring, half]]
    r_in = [[0.0, 0.0]] * n + [[rr_in, rr_in]] * 2
    r_o = [[v, v] for v in r] + [[rr_out, rr_out]] * 2
    boxes, phi = [], []
    for i in range(n):
        for a in angles:
            boxes.append([0.0, fin_r, -0.5 * fin_w, 0.5 * fin_w, z[i], z[i] + t[i]]); phi.append(a)
    for z0, z1 in ((-half, -half + ring), (half - ring, half)):
        for a in angles:
            boxes.append([rr_in, lug_c, -lug_r, lug_r, z0, z1]); phi.append(a)
    cyl = [[lug_c * np.cos(a), lug_c * np.sin(a), -half, lug_c * np.cos(a), lug_c * np.sin(a), half, lug_r]
           for a in angles]
    heat = np.concatenate([[-half], z, [half]])
    nsol = len(zr) + len(boxes) + len(cyl)
    return D.Target(np.array([-half, half]), np.asarray(r_in, float), np.asarray(r_o, float), np.full(nsol, mat),
                    z_ranges=np.asarray(zr, float), boxes=np.asarray(boxes, float), box_phi=np.asarray(phi, float),
                    cylinders=np.asarray(cyl, float), heat_edges=heat, **placement)
