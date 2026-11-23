"""The detector envelope a CRV design is placed in: fixed obstacles and movable shield sheets, from Offline's geometry
text files, as convex pieces in Mu2e coordinates.

Sources (all read the way Offline's makers and Mu2eG4 build them):
  ExtShieldUpstream   boxes (ExtShieldUpstreamMaker + constructExternalShielding): half-lengths dims / 2 + tols / 4,
                      rotated by the orientation code (OrientationResolver); G4 names ExtShieldUpstreamBox_<i>.
                      Boxes whose material matches ``movable_material`` (default: borated polyethylene) are the
                      movable shield sheets; their label is the comment of their nBoxType line.
  ExtShieldDownstream extruded polygons (ExtShieldDownstreamMaker): vertices moved outward by tols / 2 with Offline's
                      vertex rule, half length (length + tols_w) / 2, rotated by the orientation code; each polygon is
                      split into triangles, so every piece is a convex prism. Holes and notches are taken as solid (a
                      cut-out only frees space, so the envelope is conservative there).
  floor               the half-space below yOfFloorSurface.below.mu2eOrigin (mu2eHall file).
  supports            crs.supportStructure<name> boxes of the counters file.
Not modelled: the hall services (pipes, cable trays) and the building walls beyond the shielding.

A piece is a convex polytope (vertices and face normals). Overlaps between an axis-aligned box and a piece, and the
range of translations of a box along a direction for which it overlaps a piece, follow from the separating-axis
theorem on the world axes plus the piece's face normals (complete for boxes against prisms whose extrusion axis is a
world axis, as all 90-degree orientation codes give).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .. import geometry as G

UPSTREAM = "Offline/Mu2eG4/geom/ExtShieldUpstream_v07.txt"
DOWNSTREAM = "Offline/Mu2eG4/geom/ExtShieldDownstream_v06.txt"
HALL = "Offline/Mu2eG4/geom/mu2eHall_v03.txt"
MOVABLE_MATERIAL = r"^Polyethylene\d*B"          # borated polyethylene (Polyethylene092B050d095)
K = 6                                             # max face normals per piece (box 3, triangular prism 4)


# ---------------------------------------------------------------- orientation codes (GeneralUtilities, docdb 4678)
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def orientation_matrix(code):
    """Local-to-Mu2e rotation of an orientation code: OrientationResolver builds the CLHEP rotation R (each
    rotateX/Y/Z pre-multiplies), G4PVPlacement takes it as the frame rotation, so a local point p is at R^T p."""
    R = np.eye(3)
    d = math.radians
    special = {"550": [(_ry, -13.62), (_rx, 2.71)], "040": [(_ry, 45.0)], "400": [(_rx, 45.0)], "500": [(_rx, -45.0)],
               "060": [(_ry, -32.0)], "0b0": [(_ry, -122.0)], "ll0": [(_ry, -78.0), (_rx, -2.0)],
               "ab0": [(_rx, -90.0), (_ry, -13.6)]}
    if code == "000":
        return R
    if code in special:
        for f, a in special[code]:
            R = f(d(a)) @ R
        return R.T
    if len(code) != 3:
        raise ValueError(f"orientation {code!r}: three digits expected")
    n = [int(ch) if ch in "123" else 0 for ch in code]
    u, v, w = n
    phi, theta, psi = (d(90.0 * k) for k in n)

    def rot(f, a):
        nonlocal R
        R = f(a) @ R
    if w:
        rot(_rz, -psi)
    # (v rotation, u rotation for each v) per w: OrientationResolver's table
    table = {0: ((_ry, -1), {0: (_rx, -1), 1: (_rz, -1), 2: (_rx, 1), 3: (_rz, 1)}),
             1: ((_rx, 1), {0: (_ry, -1), 1: (_rz, -1), 2: (_ry, 1), 3: (_rz, 1)}),
             2: ((_ry, 1), {0: (_rx, 1), 1: (_rz, -1), 2: (_rx, -1), 3: (_rz, 1)}),
             3: ((_rx, -1), {0: (_ry, 1), 1: (_rz, -1), 2: (_ry, -1), 3: (_rz, 1)})}
    (fv, sv), fu = table[w]
    if v:
        rot(fv, sv * theta)
    if u:
        f, su = fu[v]
        rot(f, su * phi)
    return np.rint(R.T * 1e12) / 1e12


# ---------------------------------------------------------------- convex pieces
@dataclass
class Piece:
    """One convex piece of an obstacle volume."""
    volume: str               # G4 volume name
    verts: np.ndarray         # [m, 3] Mu2e mm
    normals: np.ndarray       # [k, 3] unit face normals


def box_piece(volume, center, half, R=None):
    R = np.eye(3) if R is None else R
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return Piece(volume, (s * np.asarray(half, float)) @ R.T + np.asarray(center, float), R.T.copy())


def _area2(p):
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def triangulate(poly):
    """Ear-clipping triangulation of a simple polygon [n, 2] -> list of index triples."""
    idx = list(range(len(poly)))
    if _area2(poly) < 0:
        idx.reverse()
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, cc = poly[i0], poly[i1], poly[i2]
            if cross(a, b, cc) <= 1e-9:                    # reflex or degenerate
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = poly[j]
                if cross(a, b, p) > -1e-9 and cross(b, cc, p) > -1e-9 and cross(cc, a, p) > -1e-9:
                    inside = True
                    break
            if inside:
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:                                              # no ear (degenerate polygon): drop a collinear vertex
            idx.pop(0)
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def prism_pieces(volume, uv, hlen, center, R):
    """Convex prisms of an extruded polygon (vertices uv [n, 2] in the local u-v plane, half length hlen along w)."""
    out = []
    for tri in triangulate(np.asarray(uv, float)):
        t = np.asarray(uv, float)[list(tri)]
        if abs(_area2(t)) < 1e-6:
            continue
        loc = np.array([[x, y, z] for z in (-hlen, hlen) for x, y in t])
        nrm = [np.array([0.0, 0.0, 1.0])]
        for i in range(3):
            e = t[(i + 1) % 3] - t[i]
            n = np.array([e[1], -e[0], 0.0])
            nrm.append(n / np.linalg.norm(n))
        out.append(Piece(volume, loc @ R.T + np.asarray(center, float), np.array(nrm) @ R.T))
    return out


# ---------------------------------------------------------------- Offline files
def _comment_labels(text, prefix):
    out = {}
    for m in re.finditer(r"int\s+%s\.nBoxType(\d+)\s*=\s*\d+\s*;\s*//\s*([^\n]*)" % re.escape(prefix), text):
        out[int(m.group(1))] = m.group(2).strip()
    return out


@dataclass
class Sheet:
    """A movable shield sheet: an upstream box whose material matches the movable pattern."""
    name: str                 # G4 volume name
    label: str
    type: int
    center: np.ndarray        # stock centre (Mu2e mm)
    half: np.ndarray          # stock half lengths along Mu2e x, y, z
    axis: int                 # Mu2e axis of its thickness (the smallest half length)
    dims: np.ndarray          # Offline dims (u, v, w), w the thickness
    orientation: str


def read_upstream(path=UPSTREAM, movable=MOVABLE_MATERIAL):
    """(fixed pieces, sheets) of ExtShieldUpstream."""
    text = open(G.find_counters(path)).read()
    c = G.parse_simpleconfig(text)
    labels = _comment_labels(text, "ExtShieldUpstream")
    pieces, sheets = [], []
    i = 0
    for t in range(1, int(c["ExtShieldUpstream.numberOfBoxTypes"]) + 1):
        n = int(c.get(f"ExtShieldUpstream.nBoxType{t}", 0))
        dims = np.asarray(c.get(f"ExtShieldUpstream.dimsType{t}", [0, 0, 0]), float)
        tols = np.asarray(c.get(f"ExtShieldUpstream.tolsType{t}", [0, 0, 0]), float)
        mat = str(c.get(f"ExtShieldUpstream.materialType{t}", ""))
        for b in range(1, n + 1):
            i += 1
            cen = np.asarray(c[f"ExtShieldUpstream.centerType{t}Box{b}"], float)
            if t == int(c["ExtShieldUpstream.numberOfBoxTypes"]) and b == n:     # the door: last box of the file
                cen = cen + [float(c.get("ExtShieldUpstream.FrameGap", 0.0)), 0.0,
                             float(c.get("ExtShieldUpstream.OpenDistance", 0.0))]
            ori = str(c.get(f"ExtShieldUpstream.orientationType{t}Box{b}", "000"))
            R = orientation_matrix(ori)
            half = dims / 2 + tols / 4
            name = f"ExtShieldUpstreamBox_{i}"
            if re.search(movable, mat):
                hw = np.abs(R) @ half
                sheets.append(Sheet(name, labels.get(t, f"type{t}"), t, cen, hw, int(np.argmin(hw)), dims, ori))
            else:
                pieces.append(box_piece(name, cen, half, R))
    return pieces, sheets


def _tol_signs(v):
    """Offline's per-vertex tolerance directions (constructExternalShielding): (xCnt, yCnt) for each vertex."""
    n, eps = len(v), 1e-4
    xs, ys = np.zeros(n), np.zeros(n)
    for i in range(n):
        Q1 = Q2 = Q3 = Q4 = N = S = E = W = False
        for j in range(n):
            if j == i:
                continue
            dx, dy = v[j][0] - v[i][0], v[j][1] - v[i][1]
            if dx > eps:
                Q1, Q4, E = Q1 or dy > eps, Q4 or dy < -eps, E or abs(dy) <= eps
            elif dx < -eps:
                Q2, Q3, W = Q2 or dy > eps, Q3 or dy < -eps, W or abs(dy) <= eps
            else:
                N, S = N or dy > eps, S or dy < eps
        if (not Q1 and not Q4 and not E) or (Q2 and Q3 and W and not (Q1 and Q4 and E)) or (W and not E) or \
                (N and E and not Q1) or (S and E and not Q4):
            xs[i] = 1.0
        elif (not Q2 and not Q3 and not W) or (Q1 and Q4 and E and not (Q2 and Q3 and W)) or (E and not W) or \
                (N and W and not Q2) or (S and W and not Q3):
            xs[i] = -1.0
        if (not Q1 and not Q2 and not N) or (((S and W) or (N and E)) and Q3 and not Q1) or \
                (((S and E) or (N and W)) and Q4 and not Q2) or (S and not N):
            ys[i] = 1.0
        elif (not Q3 and not Q4 and not S) or (((S and W) or (N and E)) and Q1 and not Q3) or \
                (((S and E) or (N and W)) and Q2 and not Q4) or (N and not S):
            ys[i] = -1.0
        a, b = v[(i + 1) % n], v[i - 1]                    # Offline indexes past the end here; wrap instead
        if abs(xs[i]) < eps:
            xc = 0.5 * (a[0] + b[0])
            xs[i] = 1.0 if xc < v[i][0] else (-1.0 if xc > v[i][0] else 0.0)
        if abs(ys[i]) < eps:
            yc = 0.5 * (a[1] + b[1])
            ys[i] = 1.0 if yc < v[i][1] else (-1.0 if yc > v[i][1] else 0.0)
    return xs, ys


def read_downstream(path=DOWNSTREAM):
    """Fixed pieces of ExtShieldDownstream (convex prisms of every extruded block)."""
    c = G.parse_simpleconfig(open(G.find_counters(path)).read())
    out = []
    i = 0
    for t in range(1, int(c["ExtShieldDownstream.numberOfBoxTypes"]) + 1):
        n = int(c.get(f"ExtShieldDownstream.nBoxType{t}", 0))
        if n == 0:
            continue
        uv = np.stack([np.asarray(c[f"ExtShieldDownstream.outlineType{t}UVerts"], float),
                       np.asarray(c[f"ExtShieldDownstream.outlineType{t}VVerts"], float)], 1)
        tols = np.asarray(c[f"ExtShieldDownstream.tolsType{t}"], float) / 2
        xs, ys = _tol_signs(uv)
        uvt = uv + np.stack([xs * tols[0], ys * tols[1]], 1)
        hlen = float(c[f"ExtShieldDownstream.lengthType{t}"]) / 2 + tols[2]
        for b in range(1, n + 1):
            i += 1
            cen = np.asarray(c[f"ExtShieldDownstream.centerType{t}Box{b}"], float)
            R = orientation_matrix(str(c.get(f"ExtShieldDownstream.orientationType{t}Box{b}", "000")))
            out += prism_pieces(f"ExtShieldDownstreamBox_{i}", uvt, hlen, cen, R)
    return out


def floor_piece(path=HALL, depth=1.0e5, span=1.0e6):
    y = float(G.parse_simpleconfig(open(G.find_counters(path)).read())["yOfFloorSurface.below.mu2eOrigin"])
    return box_piece("floorSu", [0.0, y - depth / 2, 0.0], [span, depth / 2, span])


def support_pieces(counters_cfg):
    out = []
    for nm in counters_cfg.get("crs.supportStructureNames", []):
        out.append(box_piece(nm, counters_cfg[f"crs.supportStructurePosition_{nm}"],
                             counters_cfg[f"crs.supportStructureHalfLengths_{nm}"]))
    return out


# ---------------------------------------------------------------- vectorised separating-axis tests
@dataclass
class PieceSet:
    """Pieces packed for vectorised tests: AABB, face normals [n, K, 3] (zero-padded) and projections."""
    volume: list
    lo: np.ndarray
    hi: np.ndarray
    normals: np.ndarray
    plo: np.ndarray
    phi: np.ndarray
    kind: list = field(default_factory=list)

    @classmethod
    def of(cls, pieces, kind=None):
        n = len(pieces)
        lo, hi = np.zeros((n, 3)), np.zeros((n, 3))
        nr, plo, phi = np.zeros((n, K, 3)), np.zeros((n, K)), np.zeros((n, K))
        for i, p in enumerate(pieces):
            lo[i], hi[i] = p.verts.min(0), p.verts.max(0)
            k = len(p.normals)
            nr[i, :k] = p.normals
            pr = p.verts @ p.normals.T
            plo[i, :k], phi[i, :k] = pr.min(0), pr.max(0)
        return cls([p.volume for p in pieces], lo, hi, nr, plo, phi, list(kind or [""] * n))

    @classmethod
    def boxes(cls, names, lo, hi, kind=None):
        lo, hi = np.asarray(lo, float).reshape(-1, 3), np.asarray(hi, float).reshape(-1, 3)
        n = len(lo)
        return cls(list(names), lo, hi, np.zeros((n, K, 3)), np.zeros((n, K)), np.zeros((n, K)),
                   list(kind or [""] * n))

    def __len__(self):
        return len(self.volume)

    def take(self, idx):
        idx = np.asarray(idx, int)
        return PieceSet([self.volume[i] for i in idx], self.lo[idx], self.hi[idx], self.normals[idx], self.plo[idx],
                        self.phi[idx], [self.kind[i] for i in idx])

    @staticmethod
    def cat(sets):
        sets = [s for s in sets if len(s)]
        if not sets:
            return PieceSet.boxes([], np.zeros((0, 3)), np.zeros((0, 3)))
        return PieceSet(sum((s.volume for s in sets), []), np.concatenate([s.lo for s in sets]),
                        np.concatenate([s.hi for s in sets]), np.concatenate([s.normals for s in sets]),
                        np.concatenate([s.plo for s in sets]), np.concatenate([s.phi for s in sets]),
                        sum((s.kind for s in sets), []))


def _axes(blo, bhi, ps, pad):
    """Box (inflated by pad) and piece projections on every axis: (box centre, box radius, piece lo, piece hi) each
    [nb, np, 3 + K]; padded normals get an always-overlapping interval."""
    c, h = 0.5 * (blo + bhi), np.maximum(0.5 * (bhi - blo) + pad, 0.0)
    cw = np.broadcast_to(c[:, None, :], (len(c), len(ps), 3))
    rw = np.broadcast_to(h[:, None, :], (len(c), len(ps), 3))
    plw = np.broadcast_to(ps.lo[None], (len(c), len(ps), 3))
    phw = np.broadcast_to(ps.hi[None], (len(c), len(ps), 3))
    cn = np.einsum("bi,pki->bpk", c, ps.normals)
    rn = np.einsum("bi,pki->bpk", h, np.abs(ps.normals))
    live = np.any(ps.normals != 0, axis=2)[None]
    big = 1e15
    pln = np.where(live, ps.plo[None], -big)
    phn = np.where(live, ps.phi[None], big)
    return (np.concatenate([cw, np.broadcast_to(cn, rn.shape)], 2), np.concatenate([rw, rn], 2),
            np.concatenate([plw, np.broadcast_to(pln, rn.shape)], 2), np.concatenate([phw, np.broadcast_to(phn, rn.shape)], 2))


def separation(blo, bhi, ps):
    """Signed separation [nb, np] of boxes and pieces: > 0 the largest gap along a separating axis, < 0 minus the
    smallest penetration (overlap depth along the best axis)."""
    blo, bhi = np.asarray(blo, float).reshape(-1, 3), np.asarray(bhi, float).reshape(-1, 3)
    if not len(blo) or not len(ps):
        return np.zeros((len(blo), len(ps)))
    c, r, pl, ph = _axes(blo, bhi, ps, 0.0)
    gap = np.maximum(pl - (c + r), (c - r) - ph)
    return gap.max(2)


def sweep(blo, bhi, ps, d, pad):
    """Open intervals (t0, t1) [nb, np] of translations t d of each box (inflated by pad) that overlap each piece;
    t0 = t1 = nan where the box never overlaps it."""
    blo, bhi = np.asarray(blo, float).reshape(-1, 3), np.asarray(bhi, float).reshape(-1, 3)
    if not len(blo) or not len(ps):
        z = np.full((len(blo), len(ps)), np.nan)
        return z, z.copy()
    c, r, pl, ph = _axes(blo, bhi, ps, pad)
    d = np.asarray(d, float)
    dn = np.concatenate([np.broadcast_to(d, (len(ps), 3)), np.einsum("i,pki->pk", d, ps.normals)], 1)[None]
    a, b = pl - r - c, ph + r - c                       # need t dn in (a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(np.abs(dn) > 1e-12, np.minimum(a / dn, b / dn), np.where((a < 0) & (b > 0), -np.inf, np.inf))
        hi = np.where(np.abs(dn) > 1e-12, np.maximum(a / dn, b / dn), np.where((a < 0) & (b > 0), np.inf, -np.inf))
    t0, t1 = lo.max(2), hi.min(2)
    bad = ~(t1 > t0 + 1e-9)
    return np.where(bad, np.nan, t0), np.where(bad, np.nan, t1)


@dataclass
class Envelope:
    """Fixed obstacles (PieceSet) and the movable sheets of a detector envelope."""
    fixed: PieceSet
    sheets: list
    sources: dict

    @classmethod
    def load(cls, counters_cfg, upstream=UPSTREAM, downstream=DOWNSTREAM, hall=HALL, movable=MOVABLE_MATERIAL):
        """The Run 2-style envelope: any of upstream / downstream / hall may be None (left out)."""
        fixed, kinds, sheets = [], [], []
        if upstream:
            p, sheets = read_upstream(upstream, movable)
            fixed += p
            kinds += ["ExtShieldUpstream"] * len(p)
        if downstream:
            p = read_downstream(downstream)
            fixed += p
            kinds += ["ExtShieldDownstream"] * len(p)
        if hall:
            fixed.append(floor_piece(hall))
            kinds.append("floor")
        sp = support_pieces(counters_cfg)
        fixed += sp
        kinds += ["support"] * len(sp)
        return cls(PieceSet.of(fixed, kinds), sheets, dict(upstream=upstream, downstream=downstream, hall=hall,
                                                            movable_material=movable))
