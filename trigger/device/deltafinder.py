"""DeltaFinder (TrigFlagPH) on the device: port of the stock DeltaFinder (Offline v13_41_01, CalPatRec), with its
arithmetic and orders.

DeltaFinder flags the panel hits of low-energy (delta) electrons as Background and clears EnergySelection on the
hits of proton candidates (EnergySelection is set on every other hit when flagProtonHits is on). It writes every hit
(writeFilteredComboHits 0), so only the flags change; TZClusterFinder, CalTimePeakFinder, RobustHelixFinder and
CalHelixFinder read them, TimeClusterFinder (TestFlag false) does not.

The geometry is the nominal tracker DeltaFinder reads (GeomHandle<Tracker>, not the aligned tracker the hits are made
with): straw 0 and 1 of each panel, shipped in `data/tracker_nominal_panels.json`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .helix import TRACKER_GEOM

DATA = Path(__file__).resolve().parent / "data"
NST, NFACE, NPF = 18, 4, 3                 # kNStations, kNFaces, kNPanelsPerFace
BKG_BIT = 6                                # StrawHitFlag::bkg
ESEL_BIT = 1                               # StrawHitFlag::energysel
FINDER_KEYS = ("timeBin", "maxDeltaEDep", "maxSeedEDep", "minProtonSeedEDep", "minProtonHitEDep", "minNSeeds",
               "minDeltaNHits", "maxEleHitEnergy", "maximumTime", "maxHitSeedDt", "maxChi2Seed", "maxChi2Par",
               "maxChi2Perp", "maxChi2SeedDelta", "rCore", "maxGap", "sigmaR", "maxDriftTime", "maxSeedDt",
               "maxHitDt", "maxDtDs", "maxDtDc", "flagProtonHits", "mergePC", "pickupProtonHits")


def settings(m):
    """DeltaFinder module table -> finder parameters. The port implements the trigger configuration: all hits
    written, no hit mask tested."""
    fp = m["finderParameters"]
    if int(m.get("writeFilteredComboHits", 0)) != 0 or int(m.get("writeStrawHits", 0)) != 0:
        raise ValueError("device DeltaFinder writes all combo hits (writeFilteredComboHits 0, writeStrawHits 0)")
    if bool(fp.get("testHitMask", False)) or list(m.get("bkgHitMask", [])) or list(fp.get("bkgHitMask", [])):
        raise ValueError("device DeltaFinder implements testHitMask false and an empty bkgHitMask")
    if int(fp.get("mergePC", 0)) != 0:
        raise ValueError("device DeltaFinder implements mergePC 0")
    out = {}
    for k in FINDER_KEYS:
        v = fp[k]
        out[k] = int(v) if k in ("minNSeeds", "minDeltaNHits", "maxGap", "flagProtonHits", "mergePC",
                                 "pickupProtonHits") else float(v)
    return out


def order_id(station, plane, panel):
    """ChannelID::orderID: (station, plane in station, panel) -> (station, z-ordered face, panel in face)."""
    face = panel % 2
    if station % 2 == 0:
        of = 1 - face if plane == 0 else face + 2
    else:
        of = face if plane == 0 else 3 - face
    return station, of, panel // 2


def plane_z(plane, geom=TRACKER_GEOM):
    """Plane::origin().z() of the nominal tracker (TrackerMaker: first plane z0 plus the plane's offset)."""
    z0 = -(float(geom["nstations"] - 1) / 2.0 * geom["planeSpacing"])
    st, k = plane // 2, plane % 2
    dz = st * geom["planeSpacing"] + (geom["planeHalfSeparation"] if k else -geom["planeHalfSeparation"])
    return z0 + dz


def geometry(path=DATA / "tracker_nominal_panels.json", cos="float"):
    """Data_t::InitGeometry: per (station, z-ordered face, panel) the wire direction (wx, wy, double), the outward
    normal (nx, ny: cos/sin of the float straw-0 azimuth; `cos` picks the float or double overload), the float z;
    the face z (double, from panel 0) and the float station z."""
    pan = np.asarray(json.loads(Path(path).read_text())["panels"], np.float64).reshape(36, 6, 6)
    wx, wy, nx, ny = (np.zeros((NST, NFACE, NPF)) for _ in range(4))
    z = np.zeros((NST, NFACE, NPF), np.float32)
    fz = np.zeros((NST, NFACE))
    for ipl in range(36):
        for ipn in range(6):
            x0, y0, z0, z1, dx, dy = pan[ipl, ipn]
            os_, of, op = order_id(ipl // 2, ipl % 2, ipn)
            wx[os_, of, op], wy[os_, of, op] = dx, dy
            phi = np.float32(math.atan2(y0, x0))                 # CLHEP Hep3Vector::phi() -> float
            if cos == "float":
                nx[os_, of, op] = float(np.float32(math.cos(float(phi))))
                ny[os_, of, op] = float(np.float32(math.sin(float(phi))))
            else:
                nx[os_, of, op], ny[os_, of, op] = math.cos(float(phi)), math.sin(float(phi))
            z[os_, of, op] = np.float32((z0 + z1) / 2.)
            if op == 0:
                fz[os_, of] = float(z[os_, of, op])
    sz = np.array([np.float32((plane_z(2 * s) + plane_z(2 * s + 1)) / 2) for s in range(NST)], np.float32)
    return {"wx": wx, "wy": wy, "nx": nx, "ny": ny, "z": z, "fz": fz, "stationZ": sz}


# ======================================================================================================================
# device port
# ======================================================================================================================
# One event at a time (vmapped over the batch), with the stock containers as fixed-size tables: seeds in creation
# order (a station's list is its seeds in that order), the delta-candidate vector, the proton-candidate list, and the
# per-hit HitData_t fields. Every stock loop is a while_loop in the stock order with its break and continue
# conditions written as stock writes them (so NaN compares as in C++); float members stay float32, double members and
# promotions float64, and products are guarded against fused multiply-adds (exact.py).

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from . import exact  # noqa: E402
from .exact import g as _g, m as _m  # noqa: E402
from .mpr import _argmax, _stock_sort  # noqa: E402

F32, F64, I32 = jnp.float32, jnp.float64, jnp.int32
NTBMAX = 3000                 # kMaxNTimeBins
SPILL_OFF = 0                 # EventWindowMarker::offspill
# error bits of an event the port cannot reproduce (stock reads memory outside its arrays, or a table overflows)
ERR_BIN, ERR_SEEDS, ERR_DELTAS, ERR_PROTONS, ERR_OVERLAP, ERR_SORT, ERR_NULL, ERR_OHIT = (1 << i for i in range(8))
# seed float columns
SUMEDEP, SUMT, SUMT2, CX, CY, TMIN, TMAX, CHI21, CHI22, CPAR, CPERP = range(11)
# delta float, double columns
DCX, DCY, DNX, DNY, DSUMEDEP = range(5)
TST, TSZ, TST2, TSTZ, TSZ2, TT0, TDTDZ = range(7)
# proton float, double columns
PSUMEDEP, PTMID, PT0, PDTDZ = range(4)


def _f(x):
    return jnp.asarray(x).astype(F32)


def _d(x):
    return jnp.asarray(x).astype(F64)


def _i(x):
    return jnp.asarray(x).astype(I32)


def _sel(c, a, b):
    return jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)


def _loop(lo, hi, body, state):
    """for (i = lo; i < hi; ++i) with break: body(i, state) -> (state, brk)."""
    def cond(c):
        return (c[0] < hi) & ~c[1]

    def bd(c):
        s, b = body(c[0], c[2])
        return c[0] + 1, b, s
    return jax.lax.while_loop(cond, bd, (_i(lo), jnp.bool_(False), state))[2]


def _loop_down(hi, lo, body, state):
    """for (i = hi; i >= lo; --i) with break."""
    def cond(c):
        return (c[0] >= lo) & ~c[1]

    def bd(c):
        s, b = body(c[0], c[2])
        return c[0] - 1, b, s
    return jax.lax.while_loop(cond, bd, (_i(hi), jnp.bool_(False), state))[2]


def _first_min(v, mask, start):
    """Index kept by `if (v < best) {best = v; i = k;}` from best = start (-1 if none), and the best value."""
    ok = mask & (v < start)
    vv = jnp.where(ok, v, jnp.inf)
    m = jnp.min(vv)
    hit = ok & (vv == m)
    return jnp.where(jnp.any(hit), _argmax(hit), -1), jnp.where(jnp.any(hit), m, start)


def _sortable(t):
    """floatToSortableInt as an int32 that orders as the uint32 key."""
    u = jax.lax.bitcast_convert_type(_f(t), jnp.uint32)
    k = jnp.where((u & jnp.uint32(0x80000000)) != 0, ~u, u | jnp.uint32(0x80000000))
    return jax.lax.bitcast_convert_type(k ^ jnp.uint32(0x80000000), I32)


def _trunc_i(x):
    """C++ float -> int, with a flag where the value is not representable (stock: undefined)."""
    ok = jnp.isfinite(x) & (jnp.abs(x) < 2.0e9)
    return _i(jnp.trunc(jnp.where(ok, x, 0))), ~ok


def _const(s, geo):
    """Settings and geometry as the stock types."""
    c = {k: (np.float32(v) if isinstance(v, float) else v) for k, v in s.items()}
    c["sig2"] = np.float32(np.float32(s["sigmaR"]) * np.float32(s["sigmaR"]))
    c["max_bin"] = int(np.trunc(np.float32(s["maximumTime"]) / np.float32(s["timeBin"])))
    c["nbins_on"] = min(int(np.trunc(np.float32(2000.0) / np.float32(s["timeBin"]))), NTBMAX)
    gj = {k: jnp.asarray(v) for k, v in geo.items()}
    return c, gj


class _Ev:
    """One event's static hit data, face lists and the time-bin lookups."""

    def __init__(self, ph, c, geo):
        self.c, self.geo = c, geo
        valid = ph["valid"]
        H = valid.shape[0]
        self.H = H
        sid = _i(ph["sid"])
        plane, panel = sid >> 10, (sid >> 7) & 7
        st, pl, face = plane // 2, plane % 2, panel % 2
        zf = jnp.where(st % 2 == 0, jnp.where(pl == 0, 1 - face, face + 2), jnp.where(pl == 0, face, 3 - face))
        self.st, self.zf, self.pf = st, zf, panel // 2
        self.pid = (sid >> 7) & 0x1FF
        self.valid = valid
        self.t, self.ct = _f(ph["etime"]), _f(ph["time"])
        self.x, self.y, self.wx, self.wy = _f(ph["x"]), _f(ph["y"]), _f(ph["udx"]), _f(ph["udy"])
        sigw = jnp.sqrt(_f(ph["uvar"]))
        self.sigw2 = _m(sigw, sigw)
        self.nr = _m(self.x, self.wy) - _m(self.y, self.wx)
        self.nx2, self.nxy, self.ny2 = _m(self.wx, self.wx), _m(self.wx, self.wy), _m(self.wy, self.wy)
        self.nxr, self.nyr = _m(self.wx, self.nr), _m(self.wy, self.nr)
        self.edep, self.nsh = _f(ph["edep"]), _i(ph["ncombo"])
        g = jnp.where(valid, st * NFACE + zf, NST * NFACE)
        self.g = g
        tkey = jnp.where(valid, _sortable(self.t).astype(jnp.int64) + 2 ** 31, 2 ** 32)
        rank = jnp.zeros(H, I32).at[jnp.argsort(tkey, stable=True)].set(jnp.arange(H, dtype=I32))
        self.fl = _i(jnp.argsort(g.astype(jnp.int64) * H + rank, stable=True))       # hits by (face, time)
        self.fn = jnp.zeros(NST * NFACE + 1, I32).at[g].add(1)[:NST * NFACE]
        self.fs = jnp.concatenate([jnp.zeros(1, I32), jnp.cumsum(self.fn)[:-1]])
        self.nbins = jnp.where(ph["spill"] == SPILL_OFF, NTBMAX, c["nbins_on"])
        tb, bad = _trunc_i(self.t / c["timeBin"])
        self.tb = tb
        self.binned = valid & (tb < self.nbins)
        self.err0 = jnp.any(valid & ((tb < 0) | bad)) * ERR_BIN
        self.pos = jnp.arange(H, dtype=I32)

    def hit(self, g, ih):
        return self.fl[jnp.clip(self.fs[g] + ih, 0, self.H - 1)]

    def _bin_idx(self, g, b, last):
        h = self.fl[jnp.minimum(self.fs[g] + self.pos, self.H - 1)]
        m = (self.pos < self.fn[g]) & self.binned[h] & (self.tb[h] == b)
        v = jnp.where(jnp.any(m), (self.H - 1 - _argmax(m[::-1])) if last else _argmax(m), -1)
        inr = (b >= 0) & (b < self.nbins)
        stale = (b >= self.nbins) & (b < NTBMAX)
        if last:
            stale |= (b < 0) & (b >= self.nbins - NTBMAX)
        return jnp.where(inr, v, jnp.where(stale, 0, -1)), ~(inr | stale)

    def first(self, g, b):
        return self._bin_idx(g, b, False)

    def last(self, g, b):
        return self._bin_idx(g, b, True)

    def bins(self, tb):
        mb = self.c["max_bin"]
        return jnp.where(tb > 0, tb - 1, 0), jnp.where(tb < mb, tb + 1, mb)

    def range3(self, g, ft, lt):
        """The `while (fFirst < 0)` narrowing of [ft, lt] (at most two steps each way) -> (ok, first, last, err)."""
        err = jnp.bool_(False)
        for _ in range(2):
            v, e = self.first(g, ft)
            go = (ft < lt) & (v < 0)
            err |= (ft < lt) & e
            ft = ft + go
        for _ in range(2):
            v, e = self.first(g, lt)
            go = (lt > ft) & (v < 0)
            err |= (lt > ft) & e
            lt = lt - go
        first, e1 = self.first(g, ft)
        last, e2 = self.last(g, lt)
        ok = first >= 0
        err = err | e1 | (ok & e2) | (ok & (last >= self.fn[g]))       # stock would read past the face's hits
        return ok, first, last, err

    def pz(self, st, face, pf):
        gg = self.geo
        return gg["wx"][st, face, pf], gg["wy"][st, face, pf], gg["nx"][st, face, pf], gg["ny"][st, face, pf]


def _state(ev, NS, ND, NP):
    H = ev.H
    z = lambda shape, dt, v=0: jnp.full(shape, v, dt)
    return {
        "used": z(H, I32), "seedof": z(H, I32, -1), "chi2min": z(H, F32, np.float32(99999.0)), "hdelta": z(H, I32, -1),
        "ns": I32(0), "ns_st": z(NST, I32),
        "s_st": z(NS, I32), "s_idx": z(NS, I32), "s_good": z(NS, I32), "s_nh": z(NS, I32), "s_nsh": z(NS, I32),
        "s_sf0": z(NS, I32), "s_sf1": z(NS, I32, -1), "s_delta": z(NS, I32, -1), "s_fp": z((NS, NFACE), I32),
        "s_hd": z((NS, NFACE), I32, -1), "s_sn": z((NS, 5), F64), "s_v": z((NS, 11), F32),
        "s_comp": z(NS, bool), "s_prot": z(NS, bool),
        "nd": I32(0), "d_index": z(ND, I32, -1), "d_mask": z(ND, I32), "d_first": z(ND, I32, 999),
        "d_last": z(ND, I32, -1), "d_nseeds": z(ND, I32), "d_nh": z(ND, I32), "d_nsh": z(ND, I32),
        "d_seeds": z((ND, NST), I32, -1), "d_v": z((ND, 5), F32), "d_sn": z((ND, 5), F64), "d_t": z((ND, 7), F64),
        "npc": I32(0), "stamp": I32(0), "p_first": z(NP, I32, 999), "p_last": z(NP, I32, -1), "p_nst": z(NP, I32),
        "p_ntot": z(NP, I32), "p_nshtot": z(NP, I32), "p_v": z((NP, 4), F32), "p_nhst": z((NP, NST), I32),
        "p_sx": z((NP, NST), F32), "p_sy": z((NP, NST), F32), "p_phi": z((NP, NST), F32),
        "p_pid": z((NP, NST * NFACE), I32, -1), "p_cnt": z((NP, NST * NFACE), I32), "p_mem": z((NP, H), bool),
        "p_stamp": z((NP, H), I32), "p_t": z((NP, 5), F64),
        "err": ev.err0,
    }


# --- seeds (DeltaSeed) ------------------------------------------------------------------------------------------------

def _seed_get(S, i):
    return {k[2:]: S[k][i] for k in ("s_st", "s_idx", "s_good", "s_nh", "s_nsh", "s_sf0", "s_sf1", "s_delta", "s_fp",
                                     "s_hd", "s_sn", "s_v")}


def _seed_put(S, i, r, go=True):
    out = dict(S)
    for k, v in r.items():
        a = S["s_" + k]
        out["s_" + k] = jnp.where(go, a.at[i].set(v, mode="drop"), a)
    return out


def _hsums(ev, h):
    return jnp.stack([_d(ev.nx2[h]), _d(ev.nxy[h]), _d(ev.ny2[h]), _d(ev.nxr[h]), _d(ev.nyr[h])])


def _seed_init(ev, S, st, h0, h1, xc, yc):
    """DeltaSeed::Init (h1 < 0: a one-hit seed)."""
    two = h1 >= 0
    h1c = jnp.maximum(h1, 0)
    f0, f1 = ev.zf[h0], jnp.where(two, ev.zf[h1c], -1)
    hd = jnp.full(NFACE, -1, I32).at[f0].set(h0)
    hd = jnp.where(two, hd.at[jnp.maximum(f1, 0)].set(h1c), hd)
    fp = jnp.zeros(NFACE, I32).at[f0].set(1)
    fp = jnp.where(two, fp.at[jnp.maximum(f1, 0)].set(1), fp)
    chi21 = S["chi2min"][h0]
    chi22 = jnp.where(two, S["chi2min"][h1c], F32(0))
    ct0, ct1 = ev.ct[h0], ev.ct[h1c]
    tmin = jnp.where(two & ~(ct1 >= ct0), ct1, ct0)
    tmax = jnp.where(two & (ct1 >= ct0), ct1, ct0)
    sn = _hsums(ev, h0)
    sn = jnp.where(two, sn + _hsums(ev, h1c), sn)
    e0 = _m(ev.edep[h0], _f(ev.nsh[h0]))
    sumedep = jnp.where(two, e0 + _m(ev.edep[h1c], _f(ev.nsh[h1c])), F32(0) + e0)
    sumt = jnp.where(two, (F32(0) + ct0) + ct1, F32(0) + ct0)
    sumt2 = jnp.where(two, (F32(0) + _m(ct0, ct0)) + _m(ct1, ct1), F32(0) + _m(ct0, ct0))
    v = jnp.zeros(11, F32)
    v = v.at[SUMEDEP].set(sumedep).at[SUMT].set(sumt).at[SUMT2].set(sumt2).at[CX].set(_f(xc)).at[CY].set(_f(yc))
    v = v.at[TMIN].set(tmin).at[TMAX].set(tmax).at[CHI21].set(chi21).at[CHI22].set(chi22)
    v = v.at[CPAR].set(chi21 + chi22).at[CPERP].set(F32(0))
    return {"st": _i(st), "idx": S["ns_st"][st], "good": I32(1), "nh": jnp.where(two, 2, 1).astype(I32),
            "nsh": jnp.where(two, ev.nsh[h0] + ev.nsh[h1c], ev.nsh[h0]), "sf0": f0, "sf1": _i(f1),
            "delta": I32(-1), "fp": fp, "hd": hd, "sn": sn, "v": v}


def _seed_add_hit(ev, r, h):
    face = ev.zf[h]
    ct = ev.ct[h]
    v = r["v"]
    v = v.at[TMIN].set(jnp.where(ct < v[TMIN], ct, v[TMIN])).at[TMAX].set(jnp.where(ct > v[TMAX], ct, v[TMAX]))
    v = v.at[SUMEDEP].add(_m(ev.edep[h], _f(ev.nsh[h]))).at[SUMT].add(ct).at[SUMT2].add(_m(ct, ct))
    return {**r, "hd": r["hd"].at[face].set(h), "fp": r["fp"].at[face].set(1), "nh": r["nh"] + 1,
            "nsh": r["nsh"] + ev.nsh[h], "sn": r["sn"] + _hsums(ev, h), "v": v}


def _seed_replace_first(ev, S, r, h):
    ct = ev.ct[h]
    v = r["v"].at[CHI21].set(S["chi2min"][h]).at[TMIN].set(ct).at[TMAX].set(ct)
    v = v.at[SUMEDEP].set(_m(ev.edep[h], _f(ev.nsh[h]))).at[SUMT].set(_m(ct, _f(ev.nsh[h]))).at[SUMT2].set(_m(ct, ct))
    return {**r, "hd": r["hd"].at[r["sf0"]].set(h), "nsh": ev.nsh[h], "v": v}


def _hit_chi2(ev, c, h, xc, yc):
    """DeltaFinderAlg::hitChi2 (float)."""
    dx, dy = ev.x[h] - xc, ev.y[h] - yc
    w = _m(dx, ev.wx[h]) + _m(dy, ev.wy[h])
    cpar = _m(w, w) / (c["sig2"] + ev.sigw2[h])
    n = _m(dx, ev.wy[h]) - _m(dy, ev.wx[h])
    drr = jnp.maximum(jnp.abs(n) - c["rCore"], F32(0))
    return cpar, _m(drr, drr) / c["sig2"]


def _seed_chi2(ev, c, hd, xc, yc):
    """seedChi2 over the seed's hits hd [..., 4], face order."""
    cp = jnp.zeros(jnp.shape(xc), F32)
    cq = jnp.zeros(jnp.shape(xc), F32)
    for face in range(NFACE):
        h = hd[..., face]
        a, b = _hit_chi2(ev, c, jnp.maximum(h, 0), xc, yc)
        cp = jnp.where(h >= 0, cp + a, cp)
        cq = jnp.where(h >= 0, cq + b, cq)
    return cp, cq


def _seed_cog_chi2(ev, c, r):
    """DeltaSeed::CalculateCogAndChi2."""
    two = (r["nh"] == 2) & (r["sf0"] >= 0) & (r["sf1"] >= 0)
    sn = r["sn"]
    dd = _m(sn[0], sn[2]) - _m(sn[1], sn[1])
    dz = dd == 0.
    ddn = jnp.where(dz, 1.0, dd)
    xc = (_m(sn[4], sn[0]) - _m(sn[3], sn[1])) / ddn
    yc = (_m(sn[4], sn[1]) - _m(sn[3], sn[2])) / ddn
    cp, cq = F32(0), F32(0)
    for face in range(NFACE):
        h = r["hd"][face]
        hc = jnp.maximum(h, 0)
        dx, dy = _d(ev.x[hc]) - xc, _d(ev.y[hc]) - yc
        w = _f(_m(dx, _d(ev.wx[hc])) + _m(dy, _d(ev.wy[hc])))
        n = _f(_m(dx, _d(ev.wy[hc])) - _m(dy, _d(ev.wx[hc])))
        a = _m(w, w) / (c["sig2"] + ev.sigw2[hc])
        drr = jnp.maximum(jnp.abs(n) - c["rCore"], F32(0))
        b = _m(drr, drr) / c["sig2"]
        cp = jnp.where(h >= 0, cp + a, cp)
        cq = jnp.where(h >= 0, cq + b, cq)
    v = r["v"]
    v2 = v.at[CX].set(_f(xc)).at[CY].set(_f(yc)).at[CPAR].set(cp).at[CPERP].set(cq)
    v1 = v.at[CPAR].set(v[CHI21] + v[CHI22]).at[CPERP].set(F32(0))
    return {**r, "v": jnp.where(two, v1, jnp.where(dz, v, v2))}


def _tmean(nh, v):
    return jnp.where(nh > 0, v[..., SUMT] / _f(jnp.maximum(nh, 1)), F32(0))


def _chi2_tot_n(nh, v):
    return jnp.where(nh > 0, (v[..., CPAR] + v[..., CPERP]) / _f(jnp.maximum(nh, 1)), F32(0))


def _chi2_time(nh, v):
    n = _f(jnp.maximum(nh, 1))
    m = v[..., SUMT] / n
    return jnp.where(nh > 0, (v[..., SUMT2] / n - _m(m, m)) / F32(64), F32(0))


def _edep_mean(nsh, sumedep):
    return jnp.where(nsh > 0, sumedep / _f(jnp.maximum(nsh, 1)), F32(0))


def _t0mm(v, sign):
    return (v[TMIN] + v[TMAX]) / F32(2) + F32(20) * sign


def _new_seed(S, r, go):
    """newDeltaSeed + the seed's record, written when go."""
    i = S["ns"]
    NS = S["s_st"].shape[0]
    S = _seed_put(S, i, {**r, "idx": S["ns_st"][r["st"]]}, go)
    S["ns_st"] = jnp.where(go, S["ns_st"].at[r["st"]].add(1), S["ns_st"])
    S["err"] = S["err"] | jnp.where(go & (i >= NS), ERR_SEEDS, 0)
    S["ns"] = jnp.where(go, i + 1, i)
    return S, i


def _complete_seed(ev, c, S, i, go):
    """DeltaFinderAlg::completeSeed on seed i."""
    r = _seed_get(S, i)
    st = r["st"]
    tb, bad = _trunc_i(_tmean(r["nh"], r["v"]) / c["timeBin"])
    S["err"] = S["err"] | jnp.where(go & bad, ERR_BIN, 0)
    ft0, lt0 = ev.bins(tb)
    xs, ys = r["v"][CX], r["v"][CY]
    rho = jnp.sqrt(_m(xs, xs) + _m(ys, ys))
    snx, sny = xs / rho, ys / rho
    for face in range(NFACE):
        g = st * NFACE + face
        ok, first, last, e = ev.range3(g, ft0, lt0)
        run = go & (r["fp"][face] != 1) & ok
        S["err"] = S["err"] | jnp.where(go & (r["fp"][face] != 1) & e, ERR_BIN, 0)

        def body(ih, st_, r=r):
            S_, best, closest = st_
            h = ev.hit(g, ih)
            wx, wy, nx, ny = ev.pz(st, face, ev.pf[h])
            cont = (S_["used"][h] >= 3) | (_f(_m(_d(snx), nx) + _m(_d(sny), ny)) < F32(0.5))
            ct = ev.ct[h]
            brk = ~cont & (ct - _t0mm(r["v"], 1) > c["maxHitSeedDt"])
            cont = cont | brk | (_t0mm(r["v"], -1) - ct > c["maxHitSeedDt"])
            sn = r["sn"] + _hsums(ev, h)
            dd = _m(sn[0], sn[2]) - _m(sn[1], sn[1])
            xc = _f((_m(sn[4], sn[0]) - _m(sn[3], sn[1])) / dd)
            yc = _f((_m(sn[4], sn[1]) - _m(sn[3], sn[2])) / dd)
            dx, dy = ev.x[h] - xc, ev.y[h] - yc
            w = _f(_m(_d(dx), wx) + _m(_d(dy), wy))
            n = _f(_m(_d(dx), nx) + _m(_d(dy), ny))
            drho = jnp.maximum(jnp.abs(n) - c["rCore"], F32(0))
            cpar = _m(w, w) / (ev.sigw2[h] + c["sig2"])
            cperp = _m(drho, drho) / c["sig2"]
            sp, sq = _seed_chi2(ev, c, r["hd"], xc, yc)
            chi2 = ((cpar + sp) + (cperp + sq)) / _f(r["nh"] + 1)
            better = ~cont & (cpar < c["maxChi2Par"]) & (cperp < c["maxChi2Perp"]) & (chi2 < best)
            return (S_, jnp.where(better, chi2, best), jnp.where(better, h, closest)), brk

        S, best, closest = _loop(first, jnp.where(run, last + 1, first), body, (S, c["maxChi2Seed"], I32(-1)))
        add = run & (closest >= 0)
        cc = jnp.maximum(closest, 0)
        S["chi2min"] = jnp.where(add, S["chi2min"].at[cc].set(best), S["chi2min"])
        r = _sel(add, _seed_add_hit(ev, r, cc), r)
    r = _seed_cog_chi2(ev, c, r)
    S = _seed_put(S, i, r, go)
    for face in range(NFACE):
        h = r["hd"][face]
        S["used"] = jnp.where(go & (h >= 0), S["used"].at[jnp.maximum(h, 0)].set(r["nh"]), S["used"])
    return S


def _find_seeds_sf(ev, c, S, st, face):
    """DeltaFinderAlg::findSeeds(Station, Face)."""
    g1 = st * NFACE + face

    def body_h1(ih1, S):
        h1 = ev.hit(g1, ih1)
        go1 = S["used"][h1] < 3
        wx1, wy1, x1, y1 = ev.wx[h1], ev.wy[h1], ev.x[h1], ev.y[h1]
        _, _, nx1, ny1 = ev.pz(st, face, ev.pf[h1])
        t1 = ev.t[h1]
        it1, bad1 = _trunc_i(t1)
        tb, bad2 = _trunc_i(_f(it1) / c["timeBin"])
        S["err"] = S["err"] | jnp.where(go1 & (bad1 | bad2), ERR_BIN, 0)
        ft0, lt0 = ev.bins(tb)

        def body_f2(f2, st_):
            S, found = st_
            g2 = st * NFACE + f2
            zc = _f((ev.geo["fz"][st, face] + ev.geo["fz"][st, f2]) / 2)
            ok, first, last, e = ev.range3(g2, ft0, lt0)
            S["err"] = S["err"] | jnp.where(go1 & e, ERR_BIN, 0)
            run = go1 & ok

            def body_h2(ih2, st2):
                S, found = st2
                h2 = ev.hit(g2, ih2)
                cont = S["used"][h2] >= 3
                dt = ev.t[h2] - t1
                cont |= dt < -c["maxDriftTime"]
                brk = ~cont & (dt > c["maxDriftTime"])
                cont |= brk
                dtc = ev.ct[h1] - ev.ct[h2]
                cont |= jnp.abs(dtc) > c["maxDriftTime"]
                _, _, nx2, ny2 = ev.pz(st, f2, ev.pf[h2])
                cont |= _f(_m(nx1, nx2) + _m(ny1, ny2)) < -0.5
                x2, y2 = ev.x[h2], ev.y[h2]
                wx2, wy2 = _d(ev.wx[h2]), _d(ev.wy[h2])
                w1w2 = _m(_d(wx1), wx2) + _m(_d(wy1), wy2)
                q12 = 1. - _m(w1w2, w1w2)
                r12n1 = _d(_m(x1 - x2, wx1) + _m(y1 - y2, wy1))
                r12n2 = _m(_d(x1 - x2), wx2) + _m(_d(y1 - y2), wy2)
                wd1 = -(_m(r12n2, w1w2) - r12n1) / q12
                xc = _f(_d(x1) - _m(_d(wx1), wd1))
                yc = _f(_d(y1) - _m(_d(wy1), wd1))
                wd2 = -(r12n2 - _m(w1w2, r12n1)) / q12
                c1 = _f(_m(wd1, wd1) / _d(ev.sigw2[h1]))
                c2 = _f(_m(wd2, wd2) / _d(ev.sigw2[h2]))
                cont |= (c1 > c["maxChi2Par"]) | (c2 > c["maxChi2Par"])
                ctot = _m(dtc, dtc) / F32(64) + (c1 + c2) / F32(2)
                cont |= ctot > c["maxChi2Seed"]
                NS = S["s_st"].shape[0]
                ids = jnp.arange(NS)
                dup = jnp.any((ids < S["ns"]) & (S["s_st"] == st) & (S["s_hd"][:, face] == h1) & (S["s_hd"][:, f2] == h2))
                make = ~cont & ~dup
                chi2min = S["chi2min"]
                S["chi2min"] = jnp.where(make, chi2min.at[h1].set(c1).at[h2].set(c2), chi2min)
                r = _seed_init(ev, S, st, h1, h2, xc, yc)
                S, i = _new_seed(S, r, make)
                S["seedof"] = jnp.where(make, S["seedof"].at[h1].set(i).at[h2].set(i), S["seedof"])
                S = _complete_seed(ev, c, S, i, make)
                nh, v = S["s_nh"][i], S["s_v"][i]
                badchi = _chi2_tot_n(nh, v) > c["maxChi2Seed"]
                em = _edep_mean(S["s_nsh"][i], v[SUMEDEP])
                idx = S["s_idx"][i]
                good = jnp.where(badchi, -3000 - idx, jnp.where(em > c["maxSeedEDep"], -2000 - idx, S["s_good"][i]))
                S["s_good"] = jnp.where(make, S["s_good"].at[i].set(good, mode="drop"), S["s_good"])
                comp = make & ~badchi & ~(em > c["maxSeedEDep"])
                prot = make & ~badchi & (em > c["minProtonSeedEDep"])
                S["s_comp"] = jnp.where(comp, S["s_comp"].at[i].set(True, mode="drop"), S["s_comp"])
                S["s_prot"] = jnp.where(prot, S["s_prot"].at[i].set(True, mode="drop"), S["s_prot"])
                found = jnp.where(make & ~badchi, nh, found)
                return (S, found), brk | (make & (found >= 3))

            S, found = _loop(first, jnp.where(run, last + 1, first), body_h2, (S, found))
            return (S, found), found >= 3

        S, _ = _loop(face + 1, jnp.where(go1, NFACE, face + 1), body_f2, (S, I32(0)))
        return S, jnp.bool_(False)

    return _loop(0, ev.fn[g1], body_h1, S)


def _prune_seeds(c, S, a, b):
    """DeltaFinderAlg::pruneSeeds on the station's seeds a..b-1 (contiguous at this point)."""
    def bad(S, i):
        v, nh = S["s_v"][i], S["s_nh"][i]
        return _chi2_tot_n(nh, v) + _chi2_time(nh, v) > c["maxChi2Seed"]

    def body1(i1, S):
        skip = S["s_good"][i1] < 0
        b1 = ~skip & bad(S, i1)
        k1 = i1 - a
        S["s_good"] = jnp.where(b1, S["s_good"].at[i1].set(-1000 - k1), S["s_good"])
        run = ~skip & ~b1
        tm1 = _tmean(S["s_nh"][i1], S["s_v"][i1])

        def body2(i2, S):
            k2 = i2 - a
            sk = S["s_good"][i2] < 0
            b2 = ~sk & bad(S, i2)
            S["s_good"] = jnp.where(b2, S["s_good"].at[i2].set(-1000 - k2), S["s_good"])
            go = ~sk & ~b2 & ~(jnp.abs(tm1 - _tmean(S["s_nh"][i2], S["s_v"][i2])) > c["maxSeedDt"])
            h1, h2 = S["s_hd"][i1], S["s_hd"][i2]
            nov = jnp.sum((h1 >= 0) & (h1 == h2))
            n1, n2 = S["s_nh"][i1], S["s_nh"][i2]
            c1, c2 = _chi2_tot_n(n1, S["s_v"][i1]), _chi2_tot_n(n2, S["s_v"][i2])
            # outcome: 0 nothing, 1 ds2 loses, 2 ds1 loses (and break)
            big = jnp.where(n1 > n2, 1, jnp.where(n2 > n1, 2, jnp.where(c1 < c2, 1, 2)))
            one = jnp.where(n1 == 2, jnp.where(n2 > 2, 2, jnp.where(c1 < c2, 1, 2)), jnp.where(n2 == 2, 1, 0))
            out = jnp.where(~go, 0, jnp.where(nov > 1, big, jnp.where(nov > 0, one, 0)))
            gd = S["s_good"]
            gd = jnp.where(out == 1, gd.at[i2].set(-1000 - k1), gd)
            gd = jnp.where(out == 2, gd.at[i1].set(-1000 - k2), gd)
            S["s_good"] = gd
            return S, out == 2

        S = _loop(i1 + 1, jnp.where(run, b, i1 + 1), body2, S)
        return S, jnp.bool_(False)

    return _loop(a, b - 1, body1, S)


# --- delta candidates (DeltaCandidate) --------------------------------------------------------------------------------

def _d_get(S, j):
    return {k[2:]: S[k][j] for k in ("d_index", "d_mask", "d_first", "d_last", "d_nseeds", "d_nh", "d_nsh", "d_seeds",
                                     "d_v", "d_sn", "d_t")}


def _d_put(S, j, D, go=True):
    out = dict(S)
    for k, v in D.items():
        a = S["d_" + k]
        out["d_" + k] = jnp.where(go, a.at[j].set(v, mode="drop"), a)
    return out


def _d_centre(D):
    sn = D["sn"]
    dd = _m(sn[0], sn[2]) - _m(sn[1], sn[1])
    ok = dd != 0.
    ddn = jnp.where(ok, dd, 1.0)
    xc = (_m(sn[4], sn[0]) - _m(sn[3], sn[1])) / ddn
    yc = (_m(sn[4], sn[1]) - _m(sn[3], sn[2])) / ddn
    rho = jnp.sqrt(_m(xc, xc) + _m(yc, yc))
    rok = ok & (rho > 0.)
    rhon = jnp.where(rok, rho, 1.0)
    v = D["v"]
    v = jnp.where(ok, v.at[DCX].set(_f(xc)).at[DCY].set(_f(yc)), v)
    v = jnp.where(rok, v.at[DNX].set(_f(xc / rhon)).at[DNY].set(_f(yc / rhon)), v)
    return {**D, "v": v}


def _d_time(D, check):
    t = D["t"]
    n = _d(D["nseeds"])
    tm, zm, tzm, t2m, z2m = t[TST] / n, t[TSZ] / n, t[TSTZ] / n, t[TST2] / n, t[TSZ2] / n
    den = z2m - _m(zm, zm)
    fit = (D["nseeds"] > 1) & ((den != 0.) if check else True)
    dtdz = (tzm - _m(tm, zm)) / den
    t0 = tm - _m(dtdz, zm)
    t = t.at[TDTDZ].set(jnp.where(fit, dtdz, 0.)).at[TT0].set(jnp.where(fit, t0, tm))
    return {**D, "t": t}


def _d_add_seed(ev, S, D, i, go=True):
    """DeltaCandidate::AddSeed(seed i)."""
    st = S["s_st"][i]
    v = D["v"].at[DSUMEDEP].add(S["s_v"][i, SUMEDEP])
    tt = _d(_tmean(S["s_nh"][i], S["s_v"][i]))
    z = _d(ev.geo["stationZ"][st])
    t = D["t"]
    t = t.at[TST].add(tt).at[TST2].add(_m(tt, tt)).at[TSTZ].add(_m(tt, z)).at[TSZ].add(z).at[TSZ2].add(_m(z, z))
    N = {**D, "seeds": D["seeds"].at[st].set(i), "nseeds": D["nseeds"] + 1, "first": jnp.minimum(D["first"], st),
         "last": jnp.maximum(D["last"], st), "nh": D["nh"] + S["s_nh"][i], "nsh": D["nsh"] + S["s_nsh"][i],
         "v": v, "sn": D["sn"] + S["s_sn"][i], "t": t}
    return _sel(go, _d_time(_d_centre(N), False), D)


def _d_new(ev, S, index, i):
    D = {"index": _i(index), "mask": I32(0), "first": I32(999), "last": I32(-1), "nseeds": I32(0), "nh": I32(0),
         "nsh": I32(0), "seeds": jnp.full(NST, -1, I32), "v": jnp.zeros(5, F32), "sn": jnp.zeros(5, F64),
         "t": jnp.zeros(7, F64)}
    return _d_add_seed(ev, S, D, i)


def _d_T0(ev, D, st):
    t = D["t"]
    return jnp.where(D["nseeds"] == 1, _f(t[..., TT0]), _f(t[..., TT0] + _m(t[..., TDTDZ], _d(ev.geo["stationZ"][st]))))


def _delta_chi2(ev, c, S, D, xc, yc):
    """deltaChi2: the hits of the candidate's seeds, station then face order."""
    def body(j, acc):
        cp, cq = acc
        st, face = j // NFACE, j % NFACE
        sd = D["seeds"][st]
        h = S["s_hd"][jnp.maximum(sd, 0), face]
        on = (st >= D["first"]) & (st <= D["last"]) & (sd >= 0) & (h >= 0)
        a, b = _hit_chi2(ev, c, jnp.maximum(h, 0), xc, yc)
        return jnp.where(on, cp + a, cp), jnp.where(on, cq + b, cq)
    return jax.lax.fori_loop(0, NST * NFACE, body, (F32(0), F32(0)))


def _dts(c, base, st, last):
    return _f(_d(base) + _m(_d(c["maxDtDs"]), _d(jnp.abs(st - last))))


def _recover_seed(ev, c, S, D, last, st, go):
    """DeltaFinderAlg::recoverSeed: the closest Compton seed of the station."""
    td = _d_T0(ev, D, st)
    NS = S["s_st"].shape[0]
    ids = jnp.arange(NS)
    dt = _dts(c, c["maxSeedDt"], st, last)
    cand = (ids < S["ns"]) & S["s_comp"] & (S["s_st"] == st) & (S["s_good"] >= 0) & ~(S["s_delta"] >= 0)
    cand &= ~(jnp.abs(td - _tmean(S["s_nh"], S["s_v"])) > dt)
    a, b = _seed_chi2(ev, c, S["s_hd"], D["v"][DCX], D["v"][DCY])
    chi2 = (a + b) / _f(jnp.maximum(S["s_nh"], 1))
    k, _ = _first_min(chi2, cand, c["maxChi2SeedDelta"])
    found = go & (k >= 0)
    kk = jnp.maximum(k, 0)
    D = _d_add_seed(ev, S, D, kk, found)
    S["s_delta"] = jnp.where(found, S["s_delta"].at[kk].set(D["index"]), S["s_delta"])
    return S, D, found


def _recover_station(ev, c, S, D, last, st, use_used, recover_seeds, go):
    """DeltaFinderAlg::recoverStation -> (S, D, rc)."""
    td = _d_T0(ev, D, st)
    dnx, dny = D["v"][DNX], D["v"][DNY]
    tb, bad = _trunc_i(td / c["timeBin"])
    ft0, lt0 = ev.bins(tb)
    found = jnp.bool_(False)
    if recover_seeds:
        S, D, found = _recover_seed(ev, c, S, D, last, st, go)
    go2 = go & ~found
    S["err"] = S["err"] | jnp.where(go2 & bad, ERR_BIN, 0)
    mdt = _dts(c, c["maxHitDt"], st, last)
    new = {"on": jnp.bool_(False), "r": _seed_init(ev, S, st, I32(0), I32(-1), 0., 0.)}
    for face in range(NFACE):
        g = st * NFACE + face
        ok, first, lastb, e = ev.range3(g, ft0, lt0)
        S["err"] = S["err"] | jnp.where(go2 & e, ERR_BIN, 0)
        run = go2 & ok & ~bad

        def body(ih, st_, face=face):
            S, new = st_
            h = ev.hit(g, ih)
            cont = (S["hdelta"][h] > 0) | (S["used"][h] >= 3) | ((use_used == 0) & (S["used"][h] != 0))
            ct = ev.ct[h]
            cont |= ct < td - mdt
            brk = ~cont & (ct > td + mdt)
            cont |= brk
            wx, wy, nx, ny = ev.pz(st, face, ev.pf[h])
            cont |= _f(_m(nx, _d(dnx)) + _m(ny, _d(dny))) < F32(0.5)
            sn = D["sn"] + _hsums(ev, h)
            dd = _m(sn[0], sn[2]) - _m(sn[1], sn[1])
            xc = _f((_m(sn[4], sn[0]) - _m(sn[3], sn[1])) / dd)
            yc = _f((_m(sn[4], sn[1]) - _m(sn[3], sn[2])) / dd)
            cpar, cperp = _hit_chi2(ev, c, h, xc, yc)
            chit = cpar + cperp
            cont |= (cpar >= c["maxChi2Par"]) | (cperp >= c["maxChi2Perp"])
            sp, sq = _delta_chi2(ev, c, S, D, xc, yc)
            n1 = _f(D["nh"] + 1)
            cpar, cperp = (cpar + sp) / n1, (cperp + sq) / n1
            dtc = ct - td
            ctime = _m(dtc, dtc) / F32(100)
            cont |= (cpar >= c["maxChi2Par"]) | (cperp >= c["maxChi2Perp"])
            chi2 = cpar + cperp + ctime
            cont |= chi2 >= c["maxChi2Seed"]
            uh = S["used"][h] != 0
            so = S["seedof"][h]
            S["err"] = S["err"] | jnp.where(~cont & uh & (so < 0), ERR_NULL, 0)
            cont |= uh & (S["s_nh"][jnp.maximum(so, 0)] >= 3) & (chit > S["chi2min"][h])
            take = ~cont
            r = new["r"]
            first_hit = take & ~new["on"]
            same = take & new["on"] & (face == r["sf0"])
            rep = same & ~(chi2 >= S["chi2min"][r["hd"][jnp.maximum(r["sf0"], 0)]])
            addh = take & new["on"] & (face != r["sf0"])
            S["chi2min"] = jnp.where(first_hit | rep, S["chi2min"].at[h].set(chi2), S["chi2min"])
            r_init = _seed_init(ev, S, st, h, I32(-1), 0., 0.)
            r_rep = _seed_replace_first(ev, S, r, h)
            r_add = _seed_add_hit(ev, r, h)
            r2 = _sel(first_hit, r_init, _sel(rep, r_rep, _sel(addh, r_add, r)))
            upd = rep | addh
            v = r2["v"]
            v = jnp.where(upd, v.at[TMIN].set(jnp.where(ct < v[TMIN], ct, v[TMIN])), v)
            v = jnp.where(upd, v.at[TMAX].set(jnp.where(ct > v[TMAX], ct, v[TMAX])), v)
            return (S, {"on": new["on"] | first_hit, "r": {**r2, "v": v}}), brk

        S, new = _loop(first, jnp.where(run, lastb + 1, first), body, (S, new))
        r = new["r"]
        new = {**new, "r": _sel(new["on"], {**r, "fp": r["fp"].at[face].set(1)}, r)}
    on = new["on"]
    r = new["r"]
    S, i = _new_seed(S, r, on)
    h0 = r["hd"][jnp.maximum(r["sf0"], 0)]
    S["seedof"] = jnp.where(on, S["seedof"].at[h0].set(i), S["seedof"])
    D = _d_add_seed(ev, S, D, i, on)
    S["s_delta"] = jnp.where(on, S["s_delta"].at[i].set(D["index"], mode="drop"), S["s_delta"])
    return S, D, found | on


def _link_delta_seeds(ev, c, S, cl, cstart, ccount):
    """DeltaFinderAlg::linkDeltaSeeds (the Compton seeds of each station in list order: cl[cstart[st]:...])."""
    ND = S["d_index"].shape[0]
    ids = jnp.arange(ND)

    def station(k, S):
        st = NST - 1 - k
        ndelta = S["nd"]

        def body_seed(j, S):
            i = cl[cstart[st] + j]
            skip = ~(S["s_good"][i] >= 0) | (S["s_delta"][i] >= 0)
            first = S["d_first"]
            gap = first - st
            act = (ids < ndelta) & (S["d_index"] >= 0) & (S["d_seeds"][:, st] < 0) & (first != st) & ~(gap > c["maxGap"])
            t0 = _d_T0(ev, {"nseeds": S["d_nseeds"], "t": S["d_t"]}, st)
            dt = t0 - _tmean(S["s_nh"][i], S["s_v"][i])
            act &= ~(jnp.abs(dt) > c["maxSeedDt"] + _m(c["maxDtDs"], _f(gap)))
            sn = _f(S["d_sn"] + S["s_sn"][i][None, :])
            dd = _m(sn[:, 0], sn[:, 2]) - _m(sn[:, 1], sn[:, 1])
            xc = (_m(sn[:, 4], sn[:, 0]) - _m(sn[:, 3], sn[:, 1])) / dd
            yc = (_m(sn[:, 4], sn[:, 1]) - _m(sn[:, 3], sn[:, 2])) / dd
            a, b = _seed_chi2(ev, c, jnp.broadcast_to(S["s_hd"][i], (ND, NFACE)), xc, yc)
            chi2 = (a + b) / _f(S["s_nh"][i])
            kd, _ = _first_min(chi2, act, c["maxChi2SeedDelta"])
            attach = ~skip & (kd >= 0)
            kc = jnp.maximum(kd, 0)
            Dk = _d_add_seed(ev, S, _d_get(S, kc), i)
            S = _d_put(S, kc, Dk, attach)
            S["s_delta"] = jnp.where(attach, S["s_delta"].at[i].set(Dk["index"]), S["s_delta"])
            make = ~skip & (kd < 0)
            nd = S["nd"]
            D = _d_new(ev, S, nd, i)

            def fwd(st2, st_):
                S, D = st_
                S, D, _ = _recover_station(ev, c, S, D, D["last"], st2, 1, 1, make)
                return (S, D), st2 - D["last"] >= 2

            S, D = _loop(st + 1, jnp.where(make, NST, st + 1), fwd, (S, D))
            S, D, _ = _recover_station(ev, c, S, D, st, jnp.maximum(st - 1, 0), 1, 1, make & (st > 0))
            keep = make & (D["nseeds"] >= c["minNSeeds"])
            S = _d_put(S, nd, D, keep)
            S["err"] = S["err"] | jnp.where(keep & (nd >= ND), ERR_DELTAS, 0)
            S["nd"] = jnp.where(keep, nd + 1, nd)
            drop = make & ~keep
            sd = D["seeds"]
            onst = (jnp.arange(NST) >= D["first"]) & (jnp.arange(NST) <= D["last"]) & (sd >= 0) & drop
            S["s_delta"] = S["s_delta"].at[jnp.where(onst, sd, S["s_delta"].shape[0])].set(-1, mode="drop")
            return S, jnp.bool_(False)

        S = _loop(0, ccount[st], body_seed, S)

        def body_dc(idc, S):
            D = _d_get(S, idc)
            go = ~((D["first"] == st) | (D["first"] - st > 3))
            S, D, _ = _recover_station(ev, c, S, D, D["first"], st, 1, 0, go)
            return _d_put(S, idc, D, go), jnp.bool_(False)

        S = _loop(0, ndelta, body_dc, S)
        return S

    S = jax.lax.fori_loop(0, NST, station, S)
    return _merge_deltas(ev, c, S)


def _d_merge(ev, S, D1, D2):
    """DeltaCandidate::MergeDeltaCandidate(D2) into D1."""
    def body(st, st_):
        S, D1 = st_
        s2 = D2["seeds"][st]
        go = (st >= D2["first"]) & (st <= D2["last"]) & (s2 >= 0) & (D1["seeds"][st] < 0)
        sc = jnp.maximum(s2, 0)
        N = {**D1, "seeds": D1["seeds"].at[st].set(s2), "nseeds": D1["nseeds"] + 1, "nh": D1["nh"] + S["s_nh"][sc],
             "nsh": D1["nsh"] + S["s_nsh"][sc], "v": D1["v"].at[DSUMEDEP].add(S["s_v"][sc, SUMEDEP]),
             "first": jnp.minimum(D1["first"], st), "last": jnp.maximum(D1["last"], st)}
        S["s_delta"] = jnp.where(go, S["s_delta"].at[sc].set(D1["index"]), S["s_delta"])
        return S, _sel(go, N, D1)

    S, D1 = jax.lax.fori_loop(0, NST, body, (S, D1))
    t = D1["t"]
    for k in (TST, TST2, TSTZ, TSZ, TSZ2):
        t = t.at[k].add(D2["t"][k])
    D1 = _d_centre({**D1, "sn": D1["sn"] + D2["sn"]})
    return S, _d_time({**D1, "t": t}, True)


def _merge_deltas(ev, c, S):
    """DeltaFinderAlg::mergeDeltaCandidates."""
    nd = S["nd"]

    def body1(i1, S):
        run = S["d_index"][i1] >= 0
        x1, y1 = S["d_v"][i1, DCX], S["d_v"][i1, DCY]

        def body2(i2, S):
            D1, D2 = _d_get(S, i1), _d_get(S, i2)
            dx, dy = x1 - D2["v"][DCX], y1 - D2["v"][DCY]
            d2 = _m(dx, dx) + _m(dy, dy)
            t1, t2 = _d_T0(ev, D1, D1["last"]), _d_T0(ev, D2, D2["first"])
            dds = D2["first"] - D1["last"]
            go = (D2["index"] >= 0) & ~(d2 > F32(400)) & ~(jnp.abs(t1 - t2) > c["maxDtDc"]) & ~(dds < 0) & ~(dds > c["maxGap"])
            S2, D1m = _d_merge(ev, S, D1, D2)
            S = _sel(go, S2, S)
            S = _d_put(S, i1, D1m, go)
            S["d_index"] = jnp.where(go, S["d_index"].at[i2].set(-1000 - D1["index"]), S["d_index"])
            return S, jnp.bool_(False)

        return _loop(i1 + 1, jnp.where(run, nd, i1 + 1), body2, S), jnp.bool_(False)

    return _loop(0, nd - 1, body1, S)


def _recover_missing_hits(ev, c, S):
    """DeltaFinderAlg::recoverMissingHits."""
    def body(idc, S):
        D = _d_get(S, idc)
        s1, s2 = D["first"], D["last"] - 1

        def holes(i, st_):
            S, D, last = st_
            has = D["seeds"][i] >= 0
            S, D, _ = _recover_station(ev, c, S, D, last, i, 1, 0, ~has)
            return (S, D, jnp.where(has, i, last)), jnp.bool_(False)

        S, D, _ = _loop(s1, s2 + 1, holes, (S, D, I32(-1)))

        def up(i, st_):
            S, D = st_
            stop = D["first"] - i > c["maxGap"]
            S, D, _ = _recover_station(ev, c, S, D, D["first"], i, 1, 0, ~stop)
            return (S, D), stop

        S, D = _loop_down(D["first"] - 1, 0, up, (S, D))

        def down(i, st_):
            S, D = st_
            stop = i - D["last"] > c["maxGap"]
            S, D, _ = _recover_station(ev, c, S, D, D["last"], i, 1, 0, ~stop)
            return (S, D), stop

        S, D = _loop(D["last"] + 1, NST, down, (S, D))
        return _d_put(S, idc, D), jnp.bool_(False)

    return _loop(0, S["nd"], body, S)


# --- protons (ProtonCandidate) ----------------------------------------------------------------------------------------

def _atan2f(y, x):
    return _f(jnp.arctan2(_d(y), _d(x)))


def _p_update_time(ev, S, p, go):
    """ProtonCandidate::updateTime."""
    t = S["p_t"][p]
    n = _d(S["p_ntot"][p])
    tm, zm, tzm, z2m = t[0] / n, t[1] / n, t[3] / n, t[4] / n
    den = z2m - _m(zm, zm)
    fit = (S["p_nst"][p] > 1) & (den != 0.)
    dtdz = _f((tzm - _m(tm, zm)) / den)
    t0 = _f(tm - _m(_d(dtdz), zm))
    dtdz = jnp.where(fit, dtdz, F32(0))
    t0 = jnp.where(fit, t0, _f(tm))
    first, last = S["p_first"][p], S["p_last"][p]
    station = jnp.where(first <= 9, jnp.where(last <= 9, last, 9), first)
    zst = ev.geo["stationZ"][jnp.clip(station, 0, NST - 1)]
    S["err"] = S["err"] | jnp.where(go & ((station < 0) | (station >= NST)), ERR_PROTONS, 0)
    v = S["p_v"][p].at[PDTDZ].set(dtdz).at[PT0].set(t0).at[PTMID].set(t0 + _m(dtdz, zst))
    S["p_v"] = jnp.where(go, S["p_v"].at[p].set(v), S["p_v"])
    return S


def _p_t0(ev, S, st):
    """t0(Station) of every candidate."""
    return S["p_v"][:, PT0] + _m(S["p_v"][:, PDTDZ], ev.geo["stationZ"][st])


def _p_predict_phi(S, st):
    """ProtonCandidate::predictPhi of every candidate -> (phi, err)."""
    pi, twopi = math.pi, 2 * math.pi
    first, last, nst = S["p_first"], S["p_last"], S["p_nst"]
    fc, lc = jnp.clip(first, 0, NST - 1), jnp.clip(last, 0, NST - 1)
    take = lambda a, i: jnp.take_along_axis(a, i[:, None], axis=1)[:, 0]
    phi1, phi2 = take(S["p_phi"], lc), take(S["p_phi"], fc)
    dl = _f(last - first)
    d = phi2 - phi1
    d = jnp.where(_d(d) < pi, _f(_d(d) + twopi), d)
    d = jnp.where(_d(d) > pi, _f(_d(d) - twopi), d)
    up_multi = phi2 + d / dl
    up_one = _f(_d(phi2) + 0.15)
    up = jnp.where(nst > 1, up_multi, up_one)
    up_err = jnp.where(nst > 1, F32(0.5), F32(0.65))
    nh = S["p_nhst"][:, st]
    inner = jnp.where(nh > 0, S["p_phi"][:, st], phi2 - _m((phi2 - phi1) / dl, _f(st - first)))
    inner_err = jnp.where(nh > 0, F32(0.2), F32(0.5))
    below = st < first
    phi = jnp.where(below, jnp.where(first - st > 1, F32(-100), up), inner)
    err = jnp.where(below, jnp.where(first - st > 1, F32(0), up_err), inner_err)
    return phi, err


def _wrapf(d):
    d = jnp.where(_d(d) < -math.pi, _f(_d(d) + 2 * math.pi), d)
    return jnp.where(_d(d) > math.pi, _f(_d(d) - 2 * math.pi), d)


def _p_append(ev, S, p, h, st, go, add_mode):
    """The bookkeeping of one hit of addSeed (add_mode False) or addHit (True), on candidates p [P] where go."""
    g = st * NFACE + ev.zf[h]
    P = S["p_first"].shape[0]
    pv = jnp.arange(P)
    has = S["p_mem"][:, h]
    go = go & ~has
    nh0 = S["p_nhst"][:, st] == 0
    S["p_nst"] = jnp.where(go & nh0, S["p_nst"] + 1, S["p_nst"])
    S["p_nhst"] = S["p_nhst"].at[:, st].add(jnp.where(go, 1, 0))
    empty = S["p_cnt"][:, g] == 0
    if add_mode:
        setpid = go & (S["p_pid"][:, g] == -1)
        S["p_first"] = jnp.where(go, jnp.minimum(S["p_first"], st), S["p_first"])
        S["p_last"] = jnp.where(go, jnp.maximum(S["p_last"], st), S["p_last"])
    else:
        setpid = go & empty
    S["p_pid"] = S["p_pid"].at[:, g].set(jnp.where(setpid, ev.pid[h], S["p_pid"][:, g]))
    add = go & (S["p_pid"][:, g] == ev.pid[h]) if not add_mode else go
    S["p_mem"] = S["p_mem"].at[:, h].set(S["p_mem"][:, h] | add)
    stamp = S["stamp"] + jnp.cumsum(add.astype(I32)) - 1
    S["p_stamp"] = S["p_stamp"].at[:, h].set(jnp.where(add, stamp, S["p_stamp"][:, h]))
    S["stamp"] = S["stamp"] + jnp.sum(add.astype(I32))
    S["p_cnt"] = S["p_cnt"].at[:, g].add(jnp.where(add, 1, 0))
    S["p_ntot"] = jnp.where(add, S["p_ntot"] + 1, S["p_ntot"])
    S["p_nshtot"] = jnp.where(add, S["p_nshtot"] + ev.nsh[h], S["p_nshtot"])
    S["p_v"] = S["p_v"].at[:, PSUMEDEP].set(jnp.where(add, S["p_v"][:, PSUMEDEP] + _m(ev.edep[h], _f(ev.nsh[h])),
                                                    S["p_v"][:, PSUMEDEP]))
    S["p_sx"] = S["p_sx"].at[:, st].set(jnp.where(add, S["p_sx"][:, st] + ev.x[h], S["p_sx"][:, st]))
    S["p_sy"] = S["p_sy"].at[:, st].set(jnp.where(add, S["p_sy"][:, st] + ev.y[h], S["p_sy"][:, st]))
    zf = ev.geo["stationZ"][st]
    ct = ev.ct[h]
    if add_mode:
        t, z = _d(ct), _d(zf)
        inc = jnp.stack([t, z, _m(t, t), _m(t, z), _m(z, z)])
    else:
        inc = jnp.stack([_d(ct), _d(zf), _d(_m(ct, ct)), _d(_m(ct, zf)), _d(_m(zf, zf))])
    S["p_t"] = jnp.where(add[:, None], S["p_t"] + inc[None, :], S["p_t"])
    del pv
    return S, add


def _p_add_seed(ev, S, i, go):
    """ProtonCandidate::addSeed(seed i) on the candidates where go [P]."""
    st = S["s_st"][i]
    S["p_first"] = jnp.where(go, jnp.minimum(S["p_first"], st), S["p_first"])
    S["p_last"] = jnp.where(go, jnp.maximum(S["p_last"], st), S["p_last"])
    for face in range(NFACE):
        h = S["s_hd"][i, face]
        S, _ = _p_append(ev, S, None, jnp.maximum(h, 0), st, go & (h >= 0), False)
    phi = _atan2f(S["p_sy"][:, st], S["p_sx"][:, st])
    S["p_phi"] = S["p_phi"].at[:, st].set(jnp.where(go, phi, S["p_phi"][:, st]))
    return _p_update_all(ev, S, go)


def _p_update_all(ev, S, go):
    P = go.shape[0]

    def body(p, S):
        return _p_update_time(ev, S, p, go[p])
    return jax.lax.fori_loop(0, P, body, S)


def _p_remove_hit(ev, S, p, st, h, go):
    """ProtonCandidate::removeHit(Station, h, UpdateTime 0)."""
    g = st * NFACE + ev.zf[h]
    go = go & S["p_mem"][p, h]
    nhf = S["p_cnt"][p, g]
    nh = S["p_nhst"][p, st] - 1
    S["p_nhst"] = jnp.where(go, S["p_nhst"].at[p, st].set(nh), S["p_nhst"])
    empty = go & (nh == 0)
    S["p_nst"] = jnp.where(empty, S["p_nst"].at[p].add(-1), S["p_nst"])
    first, last = S["p_first"][p], S["p_last"][p]
    stv = jnp.arange(NST)
    nhst = S["p_nhst"][p]
    nxt = jnp.where(jnp.any((stv > st) & (stv <= last) & (nhst > 0)), _argmax((stv > st) & (stv <= last) & (nhst > 0)), first)
    prv_m = (stv < st) & (stv >= first) & (nhst > 0)
    prv = jnp.where(jnp.any(prv_m), NST - 1 - _argmax(prv_m[::-1]), last)
    S["p_first"] = jnp.where(empty & (first == st), S["p_first"].at[p].set(nxt), S["p_first"])
    S["p_last"] = jnp.where(empty & (first != st) & (st == last), S["p_last"].at[p].set(prv), S["p_last"])
    S["p_ntot"] = jnp.where(go, S["p_ntot"].at[p].add(-1), S["p_ntot"])
    S["p_nshtot"] = jnp.where(go, S["p_nshtot"].at[p].add(-ev.nsh[h]), S["p_nshtot"])
    S["p_v"] = jnp.where(go, S["p_v"].at[p, PSUMEDEP].add(-_m(ev.edep[h], _f(ev.nsh[h]))), S["p_v"])
    t, z = _d(ev.ct[h]), _d(ev.geo["stationZ"][st])
    inc = jnp.stack([t, z, _m(t, t), _m(t, z), _m(z, z)])
    S["p_t"] = jnp.where(go, S["p_t"].at[p].add(-inc), S["p_t"])
    S["p_mem"] = jnp.where(go, S["p_mem"].at[p, h].set(False), S["p_mem"])
    S["p_cnt"] = jnp.where(go, S["p_cnt"].at[p, g].add(-1), S["p_cnt"])
    S["p_pid"] = jnp.where(go & (nhf == 1), S["p_pid"].at[p, g].set(-1), S["p_pid"])
    return S


def _create_protons(ev, c, S, pl, pstart, pcount):
    """DeltaFinderAlg::createProtonCandidates."""
    P = S["p_first"].shape[0]
    pv = jnp.arange(P)

    def station(k, S):
        st = NST - 1 - k

        def body(j, S):
            i = pl[pstart[st] + j]
            gd = S["s_good"][i]
            skip = ((gd < 0) & (gd > -2000)) | (jax.lax.rem(S["s_delta"][i], 10000) >= 0)
            t = _tmean(S["s_nh"][i], S["s_v"][i])
            act = (pv < S["npc"]) & ~skip & ~(S["p_first"] - st > 1) & ~(jnp.abs(_p_t0(ev, S, st) - t) > F32(30))
            phi, err = _p_predict_phi(S, st)
            cx, cy = S["s_v"][i, CX], S["s_v"][i, CY]
            sphi = jnp.where((cx == 0) & (cy == 0), F32(0), _atan2f(cy, cx))
            dphi = _wrapf(phi - sphi)
            consistent = ~(phi > F32(-99)) | (jnp.abs(dphi) < err)
            add = act & consistent
            S = _p_add_seed(ev, S, i, add)
            make = ~skip & ~jnp.any(add)
            n = S["npc"]
            S["err"] = S["err"] | jnp.where(make & (n >= P), ERR_PROTONS, 0)
            S = _p_add_seed(ev, S, i, make & (pv == n))
            S["npc"] = jnp.where(make, n + 1, n)
            return S, jnp.bool_(False)

        return _loop(0, pcount[st], body, S)

    return jax.lax.fori_loop(0, NST, station, S)


def _resolve_overlaps(ev, S, perp, n, K):
    """DeltaFinderAlg::resolveProtonOverlaps on the candidates in time order perp[:n]."""
    H = ev.H
    hv = jnp.arange(H)
    G = NST * NFACE

    def body1(i1, st_):
        S, novr, ohit = st_
        p1 = perp[i1]
        t1 = S["p_v"][p1, PTMID]

        def body2(i2, st2):
            S, novr, ohit = st2
            p2 = perp[i2]
            dt = S["p_v"][p2, PTMID] - t1
            os1 = jnp.maximum(S["p_first"][p1], S["p_first"][p2])
            os2 = jnp.minimum(S["p_last"][p1], S["p_last"][p2])
            ds = jnp.maximum(os2 - os1, 0)
            brk = dt > _f(60 + 5 * ds)
            nov = I32(0)
            stv = jnp.arange(NST)
            both = (stv >= os1) & (stv <= os2) & (S["p_nhst"][p1] > 0) & (S["p_nhst"][p2] > 0) & ~brk
            gst = jnp.repeat(both, NFACE)                                         # [G]: novr reset for these
            hit_g = jnp.where(ev.valid, ev.g, G)
            m = S["p_mem"][p1] & S["p_mem"][p2] & gst[jnp.minimum(hit_g, G - 1)] & (hit_g < G)
            key = jnp.where(m, hit_g.astype(jnp.int64) * (2 ** 31) + S["p_stamp"][p1], 2 ** 62)
            o = jnp.argsort(key, stable=True)
            cnt = jnp.zeros(G + 1, I32).at[jnp.where(m, hit_g, G)].add(1)[:G]
            start = jnp.concatenate([jnp.zeros(1, I32), jnp.cumsum(cnt)[:-1]])
            ck = jnp.minimum(cnt, 100)
            nov = jnp.sum(jnp.where(gst, ck, 0))
            kk = jnp.arange(K)
            newo = o[jnp.clip(start[:, None] + kk[None, :], 0, H - 1)]
            novr = jnp.where(gst, ck, novr)
            ohit = jnp.where(gst[:, None], newo, ohit)
            S["err"] = S["err"] | jnp.where(jnp.any(gst & (cnt > K)), ERR_OHIT, 0)
            dropit = ~brk & (nov >= 2)
            drop = jnp.where(S["p_ntot"][p2] >= S["p_ntot"][p1], p1, p2)
            rng = jnp.repeat((stv >= os1) & (stv <= os2), NFACE)
            S["err"] = S["err"] | jnp.where(dropit & jnp.any(rng & (novr < 0)), ERR_OVERLAP, 0)

            ent = (rng[:, None] & (jnp.arange(K)[None, :] < novr[:, None])).ravel()      # (face, k) in stock order
            order = _i(jnp.argsort(~ent, stable=True))

            def rem(j, S):
                gg, k = order[j] // K, order[j] % K
                return _p_remove_hit(ev, S, drop, gg // NFACE, ohit[gg, k], jnp.bool_(True)), jnp.bool_(False)

            S = _loop(0, jnp.where(dropit, jnp.sum(ent), 0), rem, S)
            S = _p_update_time(ev, S, drop, dropit)
            return (S, novr, ohit), brk | (dropit & (drop == p1))

        S, novr, ohit = _loop(i1 + 1, n, body2, (S, novr, ohit))
        return (S, novr, ohit), jnp.bool_(False)

    st0 = (S, jnp.full(G, -1, I32), jnp.zeros((G, K), I32))
    del hv
    return _loop(0, n - 1, body1, st0)[0]


def _recover_proton_hits(ev, c, S, phmask):
    """DeltaFinderAlg::recoverMissingProtonHits (proton hits: phmask over the face positions)."""
    def body_p(p, S):
        s1 = jnp.maximum(S["p_first"][p] - 1, 0)
        s2 = jnp.minimum(S["p_last"][p] + 1, NST - 1)

        def body_st(st, S):
            pt = _p_t0(ev, S, st)[p]
            phi, err = _p_predict_phi(S, st)
            phi, err = phi[p], err[p]
            for face in range(NFACE):
                g = st * NFACE + face
                run = S["p_cnt"][p, g] == 0

                def body_h(ih, S, g=g):
                    h = ev.hit(g, ih)
                    ok = phmask[jnp.clip(ev.fs[g] + ih, 0, ev.H - 1)]
                    ok &= ~(jnp.abs(ev.ct[h] - pt) > c["maxDriftTime"])
                    dphi = _wrapf(phi - _atan2f(ev.y[h], ev.x[h]))
                    ok &= ~((phi > F32(-99)) & (jnp.abs(dphi) > err))
                    pid = S["p_pid"][p, g]
                    ok &= (pid == -1) | (ev.pid[h] == pid)
                    P = S["p_first"].shape[0]
                    S, add = _p_append(ev, S, None, h, st, ok & (jnp.arange(P) == p), True)
                    ph = _atan2f(S["p_sy"][p, st], S["p_sx"][p, st])
                    S["p_phi"] = jnp.where(add[p], S["p_phi"].at[p, st].set(ph), S["p_phi"])
                    S = _p_update_time(ev, S, p, add[p])
                    return S, jnp.bool_(False)

                S = _loop(0, jnp.where(run, ev.fn[g], 0), body_h, S)
            return S, jnp.bool_(False)

        return _loop_down(s2, s1, body_st, S), jnp.bool_(False)

    return _loop(0, S["npc"], body_p, S)


# --- the event ---------------------------------------------------------------------------------------------------------

def _station_lists(S, mask, ns):
    """Seeds selected by mask (ids < ns), by station then creation order -> (list, start [NST], count [NST])."""
    NS = mask.shape[0]
    m = mask & (jnp.arange(NS) < ns)
    key = jnp.where(m, S["s_st"].astype(jnp.int64) * NS + jnp.arange(NS), 2 ** 40)
    lst = _i(jnp.argsort(key, stable=True))
    cnt = jnp.zeros(NST + 1, I32).at[jnp.where(m, S["s_st"], NST)].add(1)[:NST]
    start = jnp.concatenate([jnp.zeros(1, I32), jnp.cumsum(cnt)[:-1]])
    return lst, start, cnt


def _event(ph, c, geo, caps):
    ev = _Ev(ph, c, geo)
    NS, ND, NP, K = caps
    S = _state(ev, NS, ND, NP)
    # findSeeds: per station, faces 0..2, then pruneSeeds on the station's seeds
    def station(st, S):
        a = S["ns"]
        for face in range(NFACE - 1):
            S = _find_seeds_sf(ev, c, S, st, face)
        return _prune_seeds(c, S, a, S["ns"])

    S = jax.lax.fori_loop(0, NST, station, S)
    ns_find = S["ns"]
    cl, cstart, ccount = _station_lists(S, S["s_comp"], ns_find)
    S = _link_delta_seeds(ev, c, S, cl, cstart, ccount)
    S = _recover_missing_hits(ev, c, S)
    S = _merge_deltas(ev, c, S)
    # mark deltas
    def mark(j, S):
        act = S["d_index"][j] >= 0
        mask = S["d_mask"][j] | jnp.where(S["d_nh"][j] < c["minDeltaNHits"], 2, 0)
        mask = mask | jnp.where(_edep_mean(S["d_nsh"][j], S["d_v"][j, DSUMEDEP]) > c["maxDeltaEDep"], 1, 0)
        S["d_mask"] = jnp.where(act, S["d_mask"].at[j].set(mask), S["d_mask"])
        good = act & (mask == 0)
        sd = S["d_seeds"][j]
        stv = jnp.arange(NST)
        on = good & (stv >= S["d_first"][j]) & (stv <= S["d_last"][j]) & (sd >= 0)
        hd = S["s_hd"][jnp.maximum(sd, 0)]                                   # [NST, 4]
        tgt = jnp.where(on[:, None] & (hd >= 0), hd, ev.H).ravel()
        S["hdelta"] = S["hdelta"].at[tgt].set(S["d_index"][j], mode="drop")
        return S

    S = jax.lax.fori_loop(0, S["nd"], mark, S)
    # segments all of whose hits went into one delta
    did = jnp.full(NS, -1, I32)
    done = jnp.zeros(NS, bool)
    for face in range(NFACE):
        h = S["s_hd"][:, face]
        hdl = S["hdelta"][jnp.maximum(h, 0)]
        act = ~done & (h >= 0)
        done = done | (act & (hdl < 0))
        setd = act & (hdl >= 0) & (did < 0)
        clash = act & (hdl >= 0) & (did >= 0) & (did != hdl)
        did = jnp.where(setd, hdl, jnp.where(clash, -1, did))
        done = done | clash
    seg = (jnp.arange(NS) < S["ns"]) & ~(jax.lax.rem(S["s_delta"], 10000) >= 0) & (did >= 0)
    S["s_delta"] = jnp.where(seg, 10000 + did, S["s_delta"])
    # protons
    if c["flagProtonHits"]:
        fpos = ev.fl
        phmask = (ev.edep[fpos] > c["minProtonHitEDep"]) & (S["hdelta"][fpos] < 0)
        pl, pstart, pcount = _station_lists(S, S["s_prot"], ns_find)
        S = _create_protons(ev, c, S, pl, pstart, pcount)
        npc = S["npc"]
        sel = jnp.arange(NP) < npc
        tmid = S["p_v"][:, PTMID]
        S["err"] = S["err"] | jnp.where(jnp.any(sel & jnp.isnan(tmid)) & (npc > 1), ERR_SORT, 0)
        perp, n, fb = _stock_sort(_sortable(tmid), sel)
        S["err"] = S["err"] | jnp.where(fb, ERR_SORT, 0)
        S = _resolve_overlaps(ev, S, _i(perp), n, K)
        if c["pickupProtonHits"]:
            S = _recover_proton_hits(ev, c, S, phmask)
    # output flags
    flag = jnp.asarray(ph["flag"]).astype(jnp.uint32) & ~jnp.uint32(1 << BKG_BIT)
    if c["flagProtonHits"]:
        flag = flag | jnp.uint32(1 << ESEL_BIT)
    def dflag(j, fl):
        good = (S["d_index"][j] >= 0) & (S["d_mask"][j] == 0)
        sd = S["d_seeds"][j]
        stv = jnp.arange(NST)
        on = good & (stv >= S["d_first"][j]) & (stv <= S["d_last"][j]) & (sd >= 0)
        hd = S["s_hd"][jnp.maximum(sd, 0)]
        tgt = jnp.where(on[:, None] & (hd >= 0), hd, ev.H).ravel()
        return fl.at[tgt].set(fl[jnp.minimum(tgt, ev.H - 1)] | jnp.uint32(1 << BKG_BIT), mode="drop")

    flag = jax.lax.fori_loop(0, S["nd"], dflag, flag)
    np15 = I32(0)
    if c["flagProtonHits"]:
        pv = jnp.arange(NP) < S["npc"]
        np15 = jnp.sum(pv & (S["p_ntot"] >= 15)).astype(I32)
        keep = pv & ~((S["p_nst"] == 1) & (_edep_mean(S["p_nshtot"], S["p_v"][:, PSUMEDEP]) < F32(0.004)))
        clr = jnp.any(keep[:, None] & S["p_mem"], axis=0)
        flag = jnp.where(clr, flag & ~jnp.uint32(1 << ESEL_BIT), flag)
    flag = jnp.where(ph["valid"], flag, jnp.asarray(ph["flag"]).astype(jnp.uint32))
    return flag, np15, S["err"], jnp.stack([S["ns"], S["nd"], S["npc"]])


def capacities(H):
    """Table sizes for H hit slots: seeds, delta candidates, proton candidates, overlap hits per face."""
    return 4 * H + 16, H + 8, H + 8, min(100, H)


def flag_hits(ph, s, geo):
    """Panel hits [E, H] (valid, sid, flag, etime, time, x, y, udx, udy, uvar, edep, ncombo; spill [E]) -> the
    TrigFlagPH flags [E, H] (uint32), the proton candidates with >= 15 hits [E], the error bits [E] (nonzero: the
    event reaches a case stock leaves undefined, or a table overflows) and the table counts [E, 3]."""
    c, gj = _const(s, geo)
    H = ph["valid"].shape[1]
    caps = capacities(H)
    keys = ("valid", "sid", "flag", "etime", "time", "x", "y", "udx", "udy", "uvar", "edep", "ncombo")
    with exact.strict(ph["sid"]):
        d = {k: jnp.asarray(ph[k]) for k in keys}
        d["spill"] = jnp.asarray(ph["spill"])
        return jax.vmap(lambda e: _event(e, c, gj, caps))(d)
