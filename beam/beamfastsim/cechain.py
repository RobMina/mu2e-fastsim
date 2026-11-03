"""Conversion-electron chain from muon stops to the tracker entry plane (NumPy host stepping, opt-in, Geant4-free).

Electrons start inside a stopping-target foil (or in vacuum) and are carried through the DS field maps in helix steps
(mu2efastsim.ehelix: each step on the helix of the field at its midpoint) until they cross the tracker entry plane,
stop in material, leave the DS or exceed the path cap. Material is crossed with the option-4 slab physics of
mu2efastsim.eslab (or eslab_jax) in the local frame of the surface, with the field at the entry point:

- foils: annular planes (BeamGeometry foil arrays), slab normal along z; an electron born in a foil starts at its depth;
- inner proton absorber (IPA): cylindrical shell, radial normal; the shell curvature enters the flat slab as an
  equivalent axial field (valid for wall thickness << radius);
- IPA end rings (corrugated polypropylene inside the IPA radius at both ends): short shells crossed radially, no exit
  through their z faces;
- outer proton absorber (OPA): conical shell from its upstream to its downstream inner radius, radial normal;
- TSdA (annulus at the DS upstream end), the TS5 collimator (COL5, inside radius only), the TS5 bore and the DS wall
  radius absorb; tracks entering the TS4 bend (upstream end of TS5) are lost.

Not modelled: foil support wires, IPA wires, OPA slots and supports, field variation inside a layer, lateral escape
through a foil rim or hole within one crossing, re-crossing of a surface inside one helix step (sagitta h^2 / 8R, about
0.05 mm at h = 10 mm). Secondaries are not transported. Mu2e frame, mm, ns, MeV.
"""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from mu2efastsim import eioni, eslab, ehelix
from mu2efastsim import materials as M

from . import geometry as G

P_CE_AL = 104.9726            # MeV/c: CeEndpoint momentum on Al (Offline generator, MDC2025)
TAU_MU_AL = 864.0             # ns: mu- lifetime in Al (CeEndpoint decay-time draw)
C_MM_NS = 299.792458
POLYETHYLENE096 = M.Material("Polyethylene096", 0.96, 57.4, {"C": 0.856, "H": 0.144})
CORRUGATED_PP = M.Material("CorrugatedPolypropylene", 0.1892, 56.5, {"C": 0.856, "H": 0.144})   # IPA end rings
# Production cuts for a 0.01 mm range cut (Geant4 couple table: Al from the electron-gun reference logs; the
# polyethylenes scaled from mylar by density, gamma cuts at the 990 eV floor).
CUTS = {"G4_Al": (0.03411, 0.00099), "IPAPolyethylene": (0.0150, 0.00099), "Polyethylene096": (0.0151, 0.00099),
        "CorrugatedPolypropylene": (0.0070, 0.00099), "CarbonFiber": (0.0224, 0.00099)}

FATES = ("live", "plane", "stopped", "upstream", "wall", "absorbed", "looper")
LIVE, PLANE, STOPPED, UPSTREAM, WALL, ABSORBED, LOOPER = range(7)


@dataclass(frozen=True)
class CEGeometry:
    beam: G.BeamGeometry          # foils, IPA, TSdA (beam package reader)
    opa_rin0: float               # OPA inner radius at its upstream end [mm]
    opa_rin1: float               # ... at its downstream end
    opa_t: float                  # OPA wall thickness (taken radial)
    z_plane: float = 8539.9       # tracker entry plane (TT_FrontHollow VD) [mm]
    z_up: float = 2929.0          # lost upstream below this z (TS5 upstream end, entering the TS4 bend)
    r_wall: float = 950.0         # lost beyond this radius (DS inner wall)
    z_ds: float = 4179.2          # TS5 downstream end (TSdA upstream face): upstream of it the TS5 bore applies
    r_ts: float = 248.0           # TS5 vacuum radius (ts.ts5.rVac)
    coll5_z0: float = 2930.2      # TS5 collimator (COL5): z range and inner radius; absorbs
    coll5_z1: float = 3930.2
    coll5_rin: float = 128.0
    ring_t: float = 0.0           # IPA end rings (radial thickness inward from the IPA inner radius; 0: none)
    ring_hl: float = 0.0          # their half length in z; centred one half length downstream of each IPA end


def from_offline(geom_file, offline_root, z_plane=8539.9):
    """CEGeometry from an Offline geometry file (the beam reader plus the OPA cone)."""
    import os
    from . import simpleconfig
    beam = G.from_offline(geom_file, offline_root)
    path = geom_file if os.path.isabs(geom_file) else f"{offline_root}/{geom_file}"
    c = simpleconfig.read(path, offline_root)
    ts5_end = beam.tsda_z - beam.tsda_hl                 # TS5 ends at the TSdA upstream face
    ts5_start = ts5_end - 2 * c["ts.ts5.halfLength"]
    c5 = 0.5 * (ts5_start + ts5_end) + c["ts.coll5.sOffset"]
    rings = c.get("protonabsorber.ipa.buildSupports", False) and c.get("protonabsorber.ipa.nEndRings", 0) > 0
    return CEGeometry(beam, c["protonabsorber.outerPAInnerRadius0"], c["protonabsorber.outerPAInnerRadius1"],
                      c["protonabsorber.outerPAThickness"], z_plane, z_up=ts5_start, z_ds=ts5_end,
                      r_ts=c.get("ts.ts5.rVac", c["ts.ts5in.rOut"]), coll5_z0=c5 - c["ts.coll5.halfLength"],
                      coll5_z1=c5 + c["ts.coll5.halfLength"], coll5_rin=c["ts.coll5.Radius1"],
                      ring_t=c.get("protonabsorber.ipa.endRingRadialLength", 0.0) if rings else 0.0,
                      ring_hl=c.get("protonabsorber.ipa.endRingHalfLength", 0.0))


class Result(NamedTuple):
    fate: np.ndarray       # index into FATES
    x: np.ndarray          # (n, 3) position at the plane (or where the track ended)
    u: np.ndarray          # (n, 3) direction there
    T: np.ndarray          # kinetic energy there [MeV]
    t: np.ndarray          # global time there [ns]
    path: np.ndarray       # path length from the start [mm]
    nfoil: np.ndarray      # foil crossings (the birth foil counts)
    nipa: np.ndarray       # IPA crossings
    nopa: np.ndarray       # OPA crossings
    nring: np.ndarray = None   # IPA end-ring crossings
    rec_ok: np.ndarray = None  # with z_record: first crossing of that plane (either direction) and the state there
    rec_x: np.ndarray = None
    rec_u: np.ndarray = None
    rec_T: np.ndarray = None
    rec_t: np.ndarray = None


def field_maps(headers):
    """Callable B(x) [T] from G4BL maps, the first map whose grid contains the point wins (Offline inner maps)."""
    maps = []
    for hdr in headers:
        lo, d, nn, Bg = ehelix.read_g4bl(hdr)
        maps.append((lo, lo + d * (nn - 1), ehelix.grid_field(lo, d, nn, Bg)))

    def field(x):
        x = np.asarray(x, float)
        out = np.zeros(x.shape)
        todo = np.ones(x.shape[0], bool)
        for lo, hi, f in maps:
            inside = todo & np.all((x >= lo) & (x < hi), axis=1)
            if inside.any():
                out[inside] = f(x[inside])
            todo &= ~inside
        return out
    return field


def radial(geom, x):
    return np.hypot(x[:, 0] - geom.beam.axis_x, x[:, 1])


def opa_rin(geom, z):
    b = geom.beam
    f = np.clip((z - b.opa_z0) / (b.opa_z1 - b.opa_z0), 0.0, 1.0)
    return geom.opa_rin0 + f * (geom.opa_rin1 - geom.opa_rin0)


def _beta(T):
    g = 1.0 + T / eioni.ME
    return np.sqrt(np.maximum(1.0 - 1.0 / (g * g), 0.0))


def _momentum(T):
    return np.sqrt(T * (T + 2 * eioni.ME))


def _bisect_radius(geom, x, u, p, B, s_hi, target, n_iter=40):
    """Path s in [0, s_hi] where the helix reaches radius target(z) (sign change assumed at the ends)."""
    lo = np.zeros(x.shape[0])
    hi = np.array(s_hi, float)

    def f(s):
        xs = ehelix.helix(x, u, p, -1.0, B, s)[0]
        return radial(geom, xs) - target(xs[:, 2])
    flo = f(lo)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        same = np.sign(fm) == np.sign(flo)
        lo, flo = np.where(same, mid, lo), np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    return hi


def _frame_radial(geom, x, outward):
    """Orthonormal (e1, e2, n) per row: n radial (outward or inward), e1 = z, e2 = n x e1 (right-handed)."""
    r = np.stack([x[:, 0] - geom.beam.axis_x, x[:, 1], np.zeros(x.shape[0])], axis=1)
    n = r / np.linalg.norm(r, axis=1, keepdims=True)
    n = np.where(outward[:, None], n, -n)
    e1 = np.tile([0.0, 0.0, 1.0], (x.shape[0], 1))
    return e1, np.cross(n, e1), n


def _frame_z(sign):
    """(e1, e2, n) for a foil: n = +z, or the rotation by pi about x for tracks going to -z."""
    m = sign.size
    e1 = np.tile([1.0, 0.0, 0.0], (m, 1))
    e2 = np.stack([np.zeros(m), sign, np.zeros(m)], axis=1)
    n = np.stack([np.zeros(m), np.zeros(m), sign], axis=1)
    return e1, e2, n


def _cross_layer(rng, mat, T, x, u, e1, e2, n, thickness, B=None, z0=None, msc=True, backend="numpy"):
    """Slab crossing in the local frame (e1, e2, n) with the entry face through x (a track born inside starts at
    depth z0 below x along n), in the per-track constant field B [T] (global frame) or field-free; returns the new
    T, position, direction, stopped flag and path in the layer. backend "jax": eslab_jax on padded power-of-two
    batches with lane compaction (float64 unless MU2EFS_LAYER_FLOAT=float32)."""
    tcut, kcut = CUTS[mat.name]

    def loc(v):
        return np.stack([(v * e1).sum(1), (v * e2).sum(1), (v * n).sum(1)], axis=1)
    d, Bl = loc(u), None if B is None else loc(B)
    if backend == "jax":
        lT, lp, dn, stop, path = _layer_jax(rng, mat, T, d, thickness, tcut, kcut, Bl, z0, msc)
    else:
        r = eslab.electron_slab(rng, mat, T, d, thickness, tcut, kcut, msc=msc, z0=z0, field=Bl)
        lT, lp, dn, stop, path = r.T, r.position, r.direction, r.stopped, r.path
    xn = x + lp[:, :1] * e1 + lp[:, 1:2] * e2 + lp[:, 2:3] * n      # slab coordinates: origin on the entry face
    un = dn[:, :1] * e1 + dn[:, 1:2] * e2 + dn[:, 2:3] * n
    return lT, xn, un / np.linalg.norm(un, axis=1, keepdims=True), stop | (lT <= 0), path


def _layer_jax(rng, mat, T, d, L, tcut, kcut, B, z0, msc, min_bucket=256, chunk=64):
    import os
    import jax
    from mu2efastsim import eslab_jax
    fd = np.dtype(os.environ.get("MU2EFS_LAYER_FLOAT", "float64"))
    m = T.size
    mb = max(min_bucket, 1 << int(np.ceil(np.log2(max(m, 1)))))

    def pad(a, fill):
        a = np.asarray(a, float)
        out = np.empty((mb,) + a.shape[1:])
        out[:m] = a
        out[m:] = fill
        return out.astype(fd)
    key = jax.random.PRNGKey(int(rng.integers(0, 2 ** 31)))
    r = eslab_jax.electron_slab(key, mat, pad(T, 0.0), pad(d, [0.0, 0.0, 1.0]), L, tcut, kcut, msc=msc,
                                z0=None if z0 is None else pad(z0, 0.0), field=None if B is None else pad(B, 0.0),
                                chunk=chunk, min_bucket=min_bucket)
    g = lambda v: np.asarray(v, float)[:m]
    return g(r.T), g(r.position), g(r.direction), np.asarray(r.stopped)[:m], g(r.path)


def transport(rng, geom, field, x0, u0, p0, t0, h=10.0, s_max=30000.0, msc=True, layer_field=True, eps=1e-7,
              wait_fraction=0.05, backend="numpy", z_record=None, shell_curvature=True):
    """Carry electrons from (x0, u0, momentum p0 [MeV/c], global time t0) to the tracker entry plane (z_record: also
    keep the state at the first crossing of that plane, e.g. a virtual detector downstream of the target). A start
    inside a foil (BeamGeometry annulus, within its thickness) begins with the crossing from that depth. Tracks that
    reach a layer wait until at most wait_fraction of the live tracks are still in vacuum; the waiting crossings then
    run as one slab call per material (layer_field: the field at the entry point, constant through the layer;
    shell_curvature: the IPA / OPA curvature is carried into the flat slab as an extra axial field)."""
    b = geom.beam
    x = np.array(x0, float)
    u = np.array(u0, float)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    n = x.shape[0]
    T = np.sqrt(np.broadcast_to(np.asarray(p0, float), (n,)) ** 2 + eioni.ME ** 2) - eioni.ME
    t = np.array(np.broadcast_to(np.asarray(t0, float), (n,)), float)
    if z_record is not None:
        rec_ok = np.zeros(n, bool)
        rec_x, rec_u, rec_T, rec_t = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), np.zeros(n)
    s = np.zeros(n)
    fate = np.zeros(n, int)
    pend = np.zeros(n, int)                              # waiting layer: 0 none, 2 foil, 3 IPA, 4 OPA
    pfoil = np.full(n, -1)
    nfoil, nipa, nopa, nring = (np.zeros(n, int) for _ in range(4))
    face_up = b.foil_z - b.foil_ht                       # entry face for tracks going to +z
    face_dn = b.foil_z + b.foil_ht

    def account(idx, dpath, T0, T1):
        s[idx] += dpath
        t[idx] += dpath / (C_MM_NS * np.maximum(0.5 * (_beta(T0) + _beta(T1)), 1e-6))

    def cross(idx, mat, thick, e1, e2, nn, entry, zl, cnt, dBz=None):
        B = ehelix.field_at(field, x[idx]) if layer_field else None
        if dBz is not None:
            B = (np.zeros((idx.size, 3)) if B is None else B.copy())
            B[:, 2] += dBz
        T0 = T[idx]
        T1, xn, un, stop, path = _cross_layer(rng, mat, T0, entry, u[idx], e1, e2, nn, thick, B, zl, msc, backend)
        account(idx, path, T0, T1)
        T[idx], x[idx], u[idx] = T1, xn, un
        cnt[idx] += 1
        fate[idx[stop]] = STOPPED

    def foils(idx, k, inside=False):
        for L in np.unique(2 * b.foil_ht[k]):
            j = 2 * b.foil_ht[k] == L
            ii, kk = idx[j], k[j]
            sg = np.where(u[ii, 2] >= 0, 1.0, -1.0)
            e1, e2, nn = _frame_z(sg)
            entry = x[ii].copy()
            entry[:, 2] = np.where(sg > 0, face_up[kk], face_dn[kk])
            zl = np.clip(np.where(sg > 0, x[ii, 2] - face_up[kk], face_dn[kk] - x[ii, 2]), 0.0, L) if inside else None
            cross(ii, G.STOPPING_TARGET_AL, L, e1, e2, nn, entry, zl, nfoil)

    def shells(idx, mat, thick, cnt):
        outward = ((x[idx, 0] - b.axis_x) * u[idx, 0] + x[idx, 1] * u[idx, 1]) > 0
        e1, e2, nn = _frame_radial(geom, x[idx], outward)
        dBz = None
        if shell_curvature:
            # the flat slab misses the shell's curvature: relative to the surface a track bends outward by
            # u_perp^2 / R; for an e- that is an extra axial field (n.nn) u_e2 p / (K R) in the slab frame
            sgn = np.where(outward, 1.0, -1.0)
            dBz = sgn * (u[idx] * e2).sum(1) * _momentum(T[idx]) / (ehelix.K * radial(geom, x[idx]))
        cross(idx, mat, thick, e1, e2, nn, x[idx], None, cnt, dBz)

    k0 = G.foil_index(b, x[:, 0], x[:, 1], x[:, 2], tol=1e-3)     # births inside a foil
    if (k0 >= 0).any():
        foils(np.flatnonzero(k0 >= 0), k0[k0 >= 0], inside=True)

    live = np.flatnonzero(fate == LIVE)
    while live.size:
        act = live[pend[live] == 0]
        if act.size <= wait_fraction * live.size and (pend[live] > 0).any():
            w = live[pend[live] > 0]
            f = w[pend[w] == 2]
            if f.size:
                foils(f, pfoil[f])
            f = w[pend[w] == 3]
            if f.size:
                shells(f, G.IPA_PE, b.ipa_t, nipa)
            f = w[pend[w] == 4]
            if f.size:
                shells(f, POLYETHYLENE096, geom.opa_t, nopa)
            f = w[pend[w] == 8]
            if f.size:
                shells(f, CORRUGATED_PP, geom.ring_t, nring)
            pend[w] = 0
            live = np.flatnonzero(fate == LIVE)
            continue
        xl, ul, Tl = x[act], u[act], T[act]
        pl = _momentum(Tl)
        B0 = ehelix.field_at(field, xl)
        Bm = ehelix.field_at(field, ehelix.helix(xl, ul, pl, -1.0, B0, 0.5 * h)[0])
        xe = ehelix.helix(xl, ul, pl, -1.0, Bm, h)[0]
        m = act.size
        s_ev = np.full(m, h)
        kind = np.zeros(m, int)          # 0 none, 1 plane, 2 foil, 3 ipa, 4 opa, 5 tsda, 6 upstream, 7 wall, 8 ring
        kfoil = np.full(m, -1)
        z0, z1 = xl[:, 2], xe[:, 2]

        def take(sel, sv, kd):
            better = sel.copy()
            better[sel] = sv[sel] < s_ev[sel]
            s_ev[better] = sv[better]
            kind[better] = kd
            return better

        # tracker plane
        sel = (z0 < geom.z_plane) & (z1 >= geom.z_plane)
        if sel.any():
            sv = np.full(m, np.inf)
            sv[sel] = ehelix.to_plane(xl[sel], ul[sel], pl[sel], -1.0, Bm[sel], geom.z_plane, h)
            take(sel & np.isfinite(sv), sv, 1)
        # foil entry faces
        up = z1 > z0
        kup = np.searchsorted(face_up, z0 + eps, side="left")
        kdn = np.searchsorted(face_dn, z0 - eps, side="right") - 1
        k = np.where(up, kup, kdn)
        okk = (k >= 0) & (k < face_up.size)
        kc = np.clip(k, 0, face_up.size - 1)
        face = np.where(up, face_up[kc], face_dn[kc])
        sel = okk & np.where(up, face <= z1, face >= z1)
        if sel.any():
            sv = np.full(m, np.inf)
            sv[sel] = ehelix.to_plane(xl[sel], ul[sel], pl[sel], -1.0, Bm[sel], face[sel], h)
            fin = sel & np.isfinite(sv)
            if fin.any():
                xh = ehelix.helix(xl[fin], ul[fin], pl[fin], -1.0, Bm[fin], sv[fin])[0]
                rh = radial(geom, xh)
                hit = (rh >= b.foil_rin[kc[fin]]) & (rh <= b.foil_rout[kc[fin]])
                fin[np.flatnonzero(fin)[~hit]] = False
                better = take(fin, sv, 2)
                kfoil[better] = kc[better]
        r0, r1 = radial(geom, xl), radial(geom, xe)
        # IPA shell
        ri, ro = b.ipa_r, b.ipa_r + b.ipa_t
        out = (r0 < ri) & (r1 >= ri)
        inn = (r0 > ro) & (r1 <= ro)
        sel = (out | inn) & (np.maximum(z0, z1) >= b.ipa_z0) & (np.minimum(z0, z1) <= b.ipa_z1)
        if sel.any():
            sv = np.full(m, np.inf)
            tgt_r = np.where(out, ri, ro)[sel]
            sv[sel] = _bisect_radius(geom, xl[sel], ul[sel], pl[sel], Bm[sel], np.full(sel.sum(), h),
                                     lambda z, r=tgt_r: r)
            zh = ehelix.helix(xl[sel], ul[sel], pl[sel], -1.0, Bm[sel], sv[sel])[0][:, 2]
            ok = (zh >= b.ipa_z0) & (zh <= b.ipa_z1)
            sv[np.flatnonzero(sel)[~ok]] = np.inf
            take(sel & np.isfinite(sv), sv, 3)
        # IPA end rings (short shells inside the IPA radius at both IPA ends)
        if geom.ring_t > 0:
            rri, rro = b.ipa_r - geom.ring_t, b.ipa_r
            out = (r0 < rri) & (r1 >= rri)
            inn = (r0 > rro) & (r1 <= rro)
            zr = [(z, z + 2 * geom.ring_hl) for z in (b.ipa_z0, b.ipa_z1)]
            near = np.zeros(m, bool)
            for za, zb in zr:
                near |= (np.maximum(z0, z1) >= za) & (np.minimum(z0, z1) <= zb)
            sel = (out | inn) & near
            if sel.any():
                sv = np.full(m, np.inf)
                tgt_r = np.where(out, rri, rro)[sel]
                sv[sel] = _bisect_radius(geom, xl[sel], ul[sel], pl[sel], Bm[sel], np.full(sel.sum(), h),
                                         lambda z, r=tgt_r: r)
                zh = ehelix.helix(xl[sel], ul[sel], pl[sel], -1.0, Bm[sel], sv[sel])[0][:, 2]
                ok = np.zeros(zh.size, bool)
                for za, zb in zr:
                    ok |= (zh >= za) & (zh <= zb)
                sv[np.flatnonzero(sel)[~ok]] = np.inf
                take(sel & np.isfinite(sv), sv, 8)
        # OPA cone
        rc0, rc1 = opa_rin(geom, z0), opa_rin(geom, z1)
        out = (r0 < rc0) & (r1 >= rc1)
        inn = (r0 > rc0 + geom.opa_t) & (r1 <= rc1 + geom.opa_t)
        sel = (out | inn) & (np.maximum(z0, z1) >= b.opa_z0) & (np.minimum(z0, z1) <= b.opa_z1)
        if sel.any():
            sv = np.full(m, np.inf)
            off = np.where(out, 0.0, geom.opa_t)[sel]
            sv[sel] = _bisect_radius(geom, xl[sel], ul[sel], pl[sel], Bm[sel], np.full(sel.sum(), h),
                                     lambda z, o=off: opa_rin(geom, z) + o)
            zh = ehelix.helix(xl[sel], ul[sel], pl[sel], -1.0, Bm[sel], sv[sel])[0][:, 2]
            ok = (zh >= b.opa_z0) & (zh <= b.opa_z1)
            sv[np.flatnonzero(sel)[~ok]] = np.inf
            take(sel & np.isfinite(sv), sv, 4)
        # TSdA downstream face (tracks going upstream), upstream limit, wall
        ztsda = b.tsda_z + b.tsda_hl
        sel = (z0 > ztsda) & (z1 <= ztsda)
        if sel.any():
            sv = np.full(m, np.inf)
            sv[sel] = ehelix.to_plane(xl[sel], ul[sel], pl[sel], -1.0, Bm[sel], ztsda, h)
            fin = sel & np.isfinite(sv)
            if fin.any():
                rh = radial(geom, ehelix.helix(xl[fin], ul[fin], pl[fin], -1.0, Bm[fin], sv[fin])[0])
                fin[np.flatnonzero(fin)[~((rh >= b.tsda_rin) & (rh <= b.tsda_rout))]] = False
                take(fin, sv, 5)
        kind[(kind == 0) & (z1 < geom.z_up)] = 6           # full step taken, then lost
        kind[(kind == 0) & ((r1 > geom.r_wall) | ((z1 < geom.z_ds) & (r1 > geom.r_ts)))] = 7
        kind[(kind == 0) & (z1 >= geom.coll5_z0) & (z1 <= geom.coll5_z1) & (r1 > geom.coll5_rin)] = 5

        xn, un = ehelix.helix(xl, ul, pl, -1.0, Bm, s_ev)
        if z_record is not None:
            cand = ~rec_ok[act] & ((z0 - z_record) * (xn[:, 2] - z_record) <= 0) & (z0 != z_record)
            if cand.any():
                sr = ehelix.to_plane(xl[cand], ul[cand], pl[cand], -1.0, Bm[cand], z_record, s_ev[cand])
                fin = np.isfinite(sr)
                ic = np.flatnonzero(cand)[fin]
                xr, ur = ehelix.helix(xl[ic], ul[ic], pl[ic], -1.0, Bm[ic], sr[fin])
                j = act[ic]
                rec_ok[j], rec_x[j], rec_u[j], rec_T[j] = True, xr, ur, Tl[ic]
                rec_t[j] = t[j] + sr[fin] / (C_MM_NS * np.maximum(_beta(Tl[ic]), 1e-6))
        x[act], u[act] = xn, un
        account(act, s_ev, Tl, Tl)
        for kd, fv in ((1, PLANE), (5, ABSORBED), (6, UPSTREAM), (7, WALL)):
            fate[act[kind == kd]] = fv
        pend[act[kind == 2]] = 2
        pfoil[act[kind == 2]] = kfoil[kind == 2]
        pend[act[kind == 3]] = 3
        pend[act[kind == 4]] = 4
        pend[act[kind == 8]] = 8
        lp = act[(fate[act] == LIVE) & (pend[act] == 0) & (s[act] > s_max)]
        fate[lp] = LOOPER
        live = np.flatnonzero(fate == LIVE)
    if z_record is None:
        return Result(fate, x, u, T, t, s, nfoil, nipa, nopa, nring)
    return Result(fate, x, u, T, t, s, nfoil, nipa, nopa, nring, rec_ok, rec_x, rec_u, rec_T, rec_t)


def isotropic(rng, n):
    ct = rng.uniform(-1.0, 1.0, n)
    ph = rng.uniform(0.0, 2 * np.pi, n)
    st = np.sqrt(1 - ct * ct)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)


def from_stops(rng, x_stop, t_stop, n, p=P_CE_AL, tau=TAU_MU_AL):
    """Start states from a stop table (resampled with replacement): isotropic directions, decay time exponential
    with the mu- lifetime in Al after the stop time (as the CeEndpoint generator)."""
    i = rng.integers(0, len(t_stop), n)
    return x_stop[i], isotropic(rng, n), np.full(n, p), t_stop[i] + rng.exponential(tau, n), i


def helix_rmax(geom, x, u, p, Bz):
    """Largest radius about the DS axis of the helix through (x, u) in a uniform axial field Bz [T]."""
    R = np.hypot(u[:, 0], u[:, 1]) * p / (ehelix.K * np.abs(Bz))
    # e- (q = -1) in +Bz turns counterclockwise seen from +z: centre to the left of the transverse direction
    ut = u[:, :2] / np.maximum(np.hypot(u[:, 0], u[:, 1]), 1e-12)[:, None]
    sgn = np.sign(Bz)
    c = np.stack([x[:, 0] - geom.beam.axis_x, x[:, 1]], axis=1) + (sgn * R)[:, None] * np.stack([-ut[:, 1], ut[:, 0]],
                                                                                                 axis=1)
    return np.hypot(c[:, 0], c[:, 1]) + R, R


def _starts(args, rng):
    """Start states for one chunk: paired with a Geant4 reference table, or drawn from a stop table."""
    if args.ref:
        r = np.load(args.ref)
        i = np.arange(r["x"].size)[args.chunk::args.nchunks]
        x = np.stack([r["x"], r["y"], r["z"]], axis=1)[i].astype(float)
        pv = np.stack([r["px"], r["py"], r["pz"]], axis=1)[i].astype(float)
        p = np.linalg.norm(pv, axis=1)
        return x, pv / p[:, None], p, r["time"][i].astype(float), i
    s = np.load(args.stops)
    keep = (s["pdg"] == 13) & s["at_rest"] & (s["foil"] >= 0)
    xs = np.stack([s["xstop"], s["ystop"], s["zstop"]], axis=1)[keep].astype(float)
    x, u, p, t, i = from_stops(rng, xs, s["tstop"][keep].astype(float), args.n)
    return x, u, p, t, np.flatnonzero(keep)[i]


def main(argv=None):
    import argparse
    import json
    import time
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.cechain", description=__doc__.split("\n")[0])
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_common.txt")
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--maps", nargs="+", required=True, help="G4BL .header files, first match wins")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref", help="Geant4 reference table (cereference): transport its starts")
    src.add_argument("--stops", help="stop table (beam g4ref npz): isotropic CE from target mu- stops")
    ap.add_argument("--n", type=int, default=100_000, help="electrons per chunk with --stops")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--no-msc", action="store_true")
    ap.add_argument("--no-layer-field", action="store_true")
    ap.add_argument("--backend", choices=("numpy", "jax"), default="numpy", help="slab physics in the layers")
    ap.add_argument("--flat-shells", action="store_true", help="IPA / OPA as flat slabs (no curvature term)")
    ap.add_argument("--z-record", type=float, help="also keep the state at the first crossing of this z plane [mm]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(np.random.SeedSequence([a.seed, a.chunk]))
    geom = from_offline(a.geom, a.offline_root)
    field = field_maps(a.maps)
    x, u, p, t0, idx = _starts(a, rng)
    tic = time.time()
    res = transport(rng, geom, field, x, u, p, t0, h=a.step, msc=not a.no_msc, layer_field=not a.no_layer_field,
                    backend=a.backend, z_record=a.z_record, shell_curvature=not a.flat_shells)
    wall = time.time() - tic
    np.savez_compressed(a.out, idx=idx, x0=x, u0=u, p0=p, t0=t0, wall=wall,
                        **{k: v for k, v in res._asdict().items() if v is not None})
    print(json.dumps(dict(n=int(idx.size), wall_s=round(wall, 1), us_per_e=round(1e6 * wall / max(idx.size, 1), 1),
                          fates={f: int((res.fate == k).sum()) for k, f in enumerate(FATES)}), indent=1))


if __name__ == "__main__":
    main()
