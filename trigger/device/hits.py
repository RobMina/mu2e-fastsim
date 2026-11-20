"""Trigger hit reconstruction on the device: StrawDigis -> straw hits (TrigMakeSH) -> panel hits (TrigMakePH).

Events are dense [E, H] blocks (digis in collection order, padded). The straw-hit kernel follows the tracker
package's StrawHitReco port (`trkfastsim.reco.hitreco_jax`) with the trigger's settings: energy from the
firmware peak-minus-pedestal (FitType 5) with the per-straw scale, and the trigger's energy/radius/time windows.
The panel combine follows the stock CombineStrawHits loop with TestFlag on (only hits with every selection bit and
no mask bit are combined or written), one greedy pass over the event's hits.

Both kernels follow the stock arithmetic, so that the hits equal stock's bit for bit (`exact.py`): they trace in
float64 in either precision of the chain; float where the C++ computes in float (TDCTimes, ComboHit members, the
XYZVectorF centre sum of a combo, the float cuts), double where it promotes; members are summed in collection order;
products are not fused into the adds that follow them.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from trkfastsim.reco.hitreco import CENTRAL_WIRE_POS, EBIN, ERRFAC, FLAG, SLFAC, WBUF, unique_straw
from trkfastsim.reco.hitreco_jax import bucket

from . import exact
from .exact import g as _g, m as _m

F32, F64 = jnp.float32, jnp.float64
MAXNCOMBO = 8                      # ComboHit::MaxNCombo
SPILL_OFF = 0                      # EventWindowMarker::offspill

STRAW_FLAGS = {"TimeSelection": FLAG["timesel"], "EnergySelection": FLAG["energysel"],
               "RadiusSelection": FLAG["radsel"], "Dead": FLAG["dead"], "Noisy": FLAG["noisy"],
               "TimeDivision": FLAG["tdiv"], "PanelCombo": FLAG["panelcombo"]}
BACKGROUND_BIT = 13                # StrawHitFlag::bkg (never set by the device: no DeltaFinder port, see README)


def bits(names):
    m = 0
    for n in names:
        m |= 1 << STRAW_FLAGS[n]
    return m


def sh_settings(m):
    """StrawHitReco module settings -> kernel scalars."""
    return {"minT": m["MinimumTime"], "maxT": m["MaximumTime"], "minTOff": m.get("MinimumTimeOffSpill", m["MinimumTime"]),
            "maxTOff": m.get("MaximumTimeOffSpill", m["MaximumTime"]), "minE": m["MinimumEnergy"],
            "maxE": m["MaximumEnergy"], "minR": m["MinimumRadius"], "maxR": m["MaximumRadius"],
            "fittype": int(m["FitType"])}


def ph_settings(m):
    """CombineStrawHits module settings -> kernel scalars."""
    if not m.get("TestFlag", True) or m.get("Unsorted", False) or m.get("FilterHits", False) or not m.get("UseTOT", True):
        raise ValueError("device panel combine supports TestFlag=true, Unsorted=false, FilterHits=false, UseTOT=true")
    return {"maxdt": m["MaxDt"], "maxds": m["MaxDS"], "maxwdchi": m["MaxWireDistDiffPull"], "uerr": m["UError"],
            "checkWres": bool(m["CheckWres"]), "minN": m["MinimumNHits"], "maxN": m["MaximumNHits"],
            "minT": m["MinimumTime"], "maxT": m["MaximumTime"], "minTOff": m.get("MinimumTimeOffSpill", m["MinimumTime"]),
            "maxTOff": m.get("MaximumTimeOffSpill", m["MaximumTime"]), "minE": m["MinimumEnergy"],
            "maxE": m["MaximumEnergy"], "minR": m["MinimumRadius"], "maxR": m["MaximumRadius"],
            "sel": bits(m["StrawHitSelectionBits"]), "mask": bits(m["StrawHitMask"])}


def device_conditions(cond):
    """Conditions (hitreco.load_conditions layout) -> device arrays, incl. the per-straw PMP scale."""
    sc, st = cond["scalar"], cond["straw"]
    c = {k: np.asarray(st[k], np.float64) for k in
         ("midx", "midy", "midz", "dirx", "diry", "dirz", "hlen", "toff_hv", "toff_cal", "dead", "noisy", "halfvp",
          "pmpscale")}
    for k in ("halfvpscale", "centres", "resslope", "totdtime", "totderror"):
        c[k] = np.asarray(cond[k], np.float64)
    for k in ("tdcLSB", "adcLSB", "totLSB", "strawGain", "eionPerQ", "pmpScaleAvg", "strawOuterRadius"):
        c[k] = np.float64(sc[k])
    return c, tuple(cond["tot_bins"])


def pack_events(ev, nh=None, hmin=64):
    """Per-event digi lists (dict of awkward-like lists of numpy arrays) -> dense [E, H] numpy block (H: nh, else the
    next power of two >= the largest event and >= hmin)."""
    E = len(ev["sid"])
    n = np.array([len(s) for s in ev["sid"]], np.int64)
    H = nh or bucket(int(n.max()) if E else 1, hmin)
    out = {"valid": np.zeros((E, H), bool)}
    for k, dt in (("sid", np.int64), ("tdc0", np.int64), ("tdc1", np.int64), ("tot0", np.int64), ("tot1", np.int64),
                  ("pmp", np.float64)):
        a = np.zeros((E, H), dt)
        for i, v in enumerate(ev[k]):
            a[i, :len(v)] = v
        out[k] = a
    for i in range(E):
        out["valid"][i, :n[i]] = True
    out["pbt"] = np.asarray(ev["pbt"], np.float64)
    out["spill"] = np.asarray(ev["spill"], np.int64)
    return out


def _ambient_int():
    """The int dtype of the chain's precision (int32 with float64 off): outputs leave the float64 trace in it."""
    return jnp.asarray(0).dtype


def _leave(out, it):
    """Kernel outputs for the chain: integer columns in the chain's int dtype (floats are float32 already)."""
    return {k: (v.astype(it) if jnp.issubdtype(v.dtype, jnp.signedinteger) else v) for k, v in out.items()}


def _piece(xv, yv, x):
    """StrawResponse::PieceLine, in its operation order: yvals[i] + (x - xvals[i]) * slope."""
    xbin = (xv[-1] - xv[0]) / (len(xv) - 1)
    ib = jnp.clip(jnp.floor((x - xv[0]) / xbin).astype(jnp.int64), 0, len(xv) - 2)
    slope = (yv[ib + 1] - yv[ib]) / xbin
    return yv[ib] + _m(x - xv[ib], slope)


def straw_hits(p, c, s, tot_bins):
    """[E, H] digis -> straw-hit columns, as StrawHitReco (FilterHits false: one hit per digi, flags carry cuts)."""
    it = _ambient_int()
    with exact.strict(p["sid"]):
        return _leave(_straw_hits(p, c, s, tot_bins), it)


def _straw_hits(p, c, s, tot_bins):
    """StrawHitRecoUtils::createComboHit and the StrawResponse calls it makes, in the stock types and order."""
    if s["fittype"] != 5:
        raise ValueError("device straw hits implement FitType 5 (firmware PMP) only")
    c = {k: jnp.asarray(v) for k, v in c.items()}
    sid, valid = p["sid"].astype(jnp.int64), p["valid"]
    us = unique_straw(sid)
    d = lambda x: jnp.asarray(x).astype(F64)
    f = lambda x: jnp.asarray(x).astype(F32)
    charge = _m(d(p["pmp"]), c["adcLSB"] / c["strawGain"], c["pmpscale"][us])
    energy = f(c["eionPerQ"] * charge)                               # float energy = ionizationEnergy(charge)
    pbt = d(p["pbt"])[:, None]
    # calibrateTimes into float TDCTimes, then times[iend] -= pbtOffset (float -= double)
    t_cal = f(d(f(_m(d(p["tdc0"]), c["tdcLSB"]) + c["toff_cal"][us])) - pbt)
    t_hv = f(d(f(_m(d(p["tdc1"]), c["tdcLSB"]) + c["toff_hv"][us])) - pbt)
    eend = (t_hv < t_cal).astype(jnp.int64)
    tot_cal, tot_hv = f(_m(d(p["tot0"]), c["totLSB"])), f(_m(d(p["tot1"]), c["totLSB"]))
    tot_sel = jnp.where(eend == 1, tot_hv, tot_cal)
    dt = d(t_hv - t_cal)                                              # float - float, then double
    kedep = 1000.0 * d(energy)
    egrid = jnp.arange(c["halfvpscale"].shape[0]) * EBIN
    halfpv = _m(_piece(egrid, c["halfvpscale"], kedep), c["halfvp"][us])
    wdist = _m(halfpv, dt)
    awd = jnp.abs(wdist)
    wres = _piece(egrid, c["centres"], kedep)
    wres = jnp.where(awd > CENTRAL_WIRE_POS, wres + _m(awd - CENTRAL_WIRE_POS, _piece(egrid, c["resslope"], kedep)), wres)
    wres = jnp.maximum(30.0, wres)
    hlen = c["hlen"][us]
    trunc = awd > hlen + _m(WBUF, wres)
    wdist = jnp.where(trunc, jnp.copysign(hlen * SLFAC, wdist), wdist)
    wres = jnp.where(trunc, wres * ERRFAC, wres)
    esign = 2.0 * (eend - 0.5)
    propd = f(hlen + _m(esign, wdist))
    ptime = f(d(propd) / (2 * halfpv))
    x = f(c["midx"][us] + _m(wdist, c["dirx"][us]))
    y = f(c["midy"][us] + _m(wdist, c["diry"][us]))
    z = f(c["midz"][us] + _m(wdist, c["dirz"][us]))
    rho = jnp.sqrt(_m(x, x) + _m(y, y))                               # XYZVectorF::Rho in float
    ntb, wtb, neb, web = tot_bins
    tbin = jnp.minimum(int(ntb) - 1, (d(tot_sel) / wtb).astype(jnp.int64))
    ebin = jnp.minimum(int(neb) - 1, (d(energy) / web).astype(jnp.int64))
    dtime = c["totdtime"][tbin * int(neb) + ebin]
    tres = c["totderror"][tbin * int(neb) + ebin]
    t_early = jnp.where(eend == 1, t_hv, t_cal)
    time_d = d(t_early - ptime) - dtime                               # (float - float) - double
    flag = _sh_flag(valid, p["spill"], c["dead"][us] > 0, c["noisy"][us] > 0, energy, rho, time_d, ~trunc, s)
    rout = c["strawOuterRadius"]
    inv3 = d(np.float32(1.0 / 3.0))                                   # `static const float inv3 = 1.0/3.0`
    return {"valid": valid, "spill": p["spill"].astype(jnp.int64), "sid": sid, "upanel": (sid >> 10) * 6 + ((sid & 0x380) >> 7),
            "straw": sid & 0x7F, "time": f(time_d), "et0": t_cal, "et1": t_hv, "edep": energy, "wdist": f(wdist),
            "uvar": f(_m(wres, wres)), "vvar": jnp.full(sid.shape, f(_m(rout, rout, inv3))),
            "timevar": f(_m(tres, tres)), "x": x, "y": y, "z": z, "udx": f(c["dirx"][us]),
            "udy": f(c["diry"][us]), "flag": flag}


def _f32(v):
    return np.float32(v)


def _sh_flag(valid, spill, dead, noisy, energy, rho, time, tdiv, s):
    """StrawHitReco's flag bits of a hit under settings s (the selections are flags only: FilterHits false). The cut
    values are float (StrawHitRecoUtils); energy and rho are float, the time is the double before it is stored."""
    off = (spill == SPILL_OFF)[:, None]
    minT = jnp.where(off, _f32(s["minTOff"]), _f32(s["minT"]))
    maxT = jnp.where(off, _f32(s["maxTOff"]), _f32(s["maxT"]))
    flag = jnp.zeros(valid.shape, jnp.uint32)
    for bit, ok in ((FLAG["dead"], dead), (FLAG["noisy"], noisy),
                    (FLAG["energysel"], (energy <= _f32(s["maxE"])) & (energy >= _f32(s["minE"]))),
                    (FLAG["radsel"], (rho >= _f32(s["minR"])) & (rho <= _f32(s["maxR"]))),
                    (FLAG["timesel"], (time >= minT) & (time <= maxT)), (FLAG["tdiv"], tdiv)):
        flag = flag | jnp.where(ok, jnp.uint32(1 << bit), jnp.uint32(0))
    return jnp.where(valid, flag, jnp.uint32(0))


def reflag_straw_hits(h, s):
    """The straw hits of `straw_hits` under other StrawHitReco settings s: the same hits and columns (one hit per digi),
    new selection flags (dead, noisy and time-division bits carried over)."""
    bit = lambda b: ((h["flag"] >> FLAG[b]) & 1) == 1
    rho = jnp.hypot(h["x"], h["y"])
    return {**h, "flag": _sh_flag(h["valid"], h["spill"], bit("dead"), bit("noisy"), h["edep"], rho, h["time"],
                                  bit("tdiv"), s)}


def _good(h, s):
    """Hits CombineStrawHits combines and writes (TestFlag on): every selection bit, no mask bit, not dead."""
    return h["valid"] & ((h["flag"] & s["sel"]) == s["sel"]) & ((h["flag"] & s["mask"]) == 0) & \
        ((h["flag"] & (1 << FLAG["dead"])) == 0)


def _ph_flag(ncombo, time, edep, x, y, tdiv, spill, s):
    """CombineStrawHits' flag bits of a combo: float time, energy and r^2 = Perp2() against the float cuts."""
    off = (spill == SPILL_OFF)[:, None]
    minT = jnp.where(off, _f32(s["minTOff"]), _f32(s["minT"]))
    maxT = jnp.where(off, _f32(s["maxTOff"]), _f32(s["maxT"]))
    r2 = _m(x, x) + _m(y, y)
    minR2, maxR2 = _f32(s["minR"]) * _f32(s["minR"]), _f32(s["maxR"]) * _f32(s["maxR"])
    flag = jnp.uint32(1 << FLAG["panelcombo"]) | jnp.where(tdiv, jnp.uint32(1 << FLAG["tdiv"]), jnp.uint32(0))
    for bit, ok in ((FLAG["nhitsel"], (ncombo >= s["minN"]) & (ncombo <= s["maxN"])),
                    (FLAG["timesel"], (time >= minT) & (time <= maxT)),
                    (FLAG["energysel"], (edep >= _f32(s["minE"])) & (edep <= _f32(s["maxE"]))),
                    (FLAG["radsel"], (r2 >= minR2) & (r2 <= maxR2))):
        flag = flag | jnp.where(ok, jnp.uint32(1 << bit), jnp.uint32(0))
    return flag


COMBINE_CUTS = ("maxdt", "maxds", "maxwdchi", "uerr", "checkWres")   # what forms and fits a combo besides `good`


def panel_hits_other(sh, ph, s, sh2, s2):
    """CombineStrawHits under other settings s2 on the re-flagged straw hits sh2 (`reflag_straw_hits`), given the
    combos ph = panel_hits(sh, s). When the combine cuts are the same and no hit of the batch changes whether it is
    combined (`_good`), the combos are those of ph and only their flags change; otherwise the full combine runs on
    sh2. lax.cond picks per batch, so the result is the stock one either way."""
    if any(s[k] != s2[k] for k in COMBINE_CUTS):
        return panel_hits(sh2, s2)

    def reuse():
        tdiv = ((ph["flag"] >> FLAG["tdiv"]) & 1) == 1
        with exact.strict(sh["sid"]):
            return {**ph, "flag": _ph_flag(ph["ncombo"], ph["time"], ph["edep"], ph["x"], ph["y"], tdiv, sh["spill"], s2)}

    return jax.lax.cond(jnp.all(_good(sh, s) == _good(sh2, s2)), reuse, lambda: panel_hits(sh2, s2))


def _group(h, s):
    """Stock CombineStrawHits greedy loop, vectorised over events and over panel runs: owner seed per hit.

    The stock loop takes the event's hits in order; each unused hit seeds a combo and absorbs later unused good hits
    of the same panel until the first hit of another panel (input sorted by panel), so the panel runs of an event are
    independent of each other. Iteration i therefore handles, in every run of every event at once, the hit at
    position i of the run as the seed: the same seeds, candidates, cuts and order as the per-hit loop, in as many
    iterations as the longest run of the batch instead of the hit-axis width."""
    E, H = h["valid"].shape
    valid = h["valid"]
    good = _good(h, s)
    jpos = jnp.broadcast_to(jnp.arange(H)[None, :], (E, H))
    up = h["upanel"]
    cont = jnp.concatenate([jnp.zeros((E, 1), bool), (up[:, 1:] == up[:, :-1]) & valid[:, 1:] & valid[:, :-1]], axis=1)
    newrun = valid & ~cont
    rs = jax.lax.cummax(jnp.where(newrun, jpos, 0), axis=1)          # first hit of the hit's run
    pos = jnp.where(valid, jpos - rs, -1)                             # position of the hit in its run
    nrun = jnp.max(pos) + 1
    rid = jnp.cumsum(newrun, axis=1)                                  # run number in the event (1, 2, ...)

    def cond(state):
        return state[0] < nrun

    def body(state):
        i, used, owner = state
        sidx = rs + i                                                 # the seed of each hit's run in this iteration
        sc = jnp.minimum(sidx, H - 1)
        g = lambda a: jnp.take_along_axis(a, sc, axis=1)
        in_run = (sidx < H) & g(valid) & (g(rid) == rid)
        gseed = in_run & ~g(used) & g(good)                           # the run's seed is unused and good
        seed = valid & (pos == i) & ~used
        ok = valid & (pos > i) & ~used & good & gseed
        ok &= jnp.abs(h["straw"] - g(h["straw"])) <= s["maxds"]
        ok &= jnp.abs(g(h["time"]) - h["time"]) <= _f32(s["maxdt"])                     # float dt, float cut
        ok &= (jnp.abs(g(h["wdist"]) - h["wdist"]) / jnp.sqrt(g(h["uvar"]) + h["uvar"])) <= _f32(s["maxwdchi"])
        cs = jnp.cumsum(ok, axis=1)
        before = jnp.take_along_axis(cs - ok, rs, axis=1)              # candidates of the run before its first hit
        take = ok & (cs - before <= MAXNCOMBO - 1)
        used = used | seed | take
        owner = jnp.where(take, sidx, owner)
        owner = jnp.where(seed & good, jpos, owner)
        return i + 1, used, owner

    _, _, owner = jax.lax.while_loop(cond, body, (jnp.zeros((), nrun.dtype), jnp.zeros((E, H), bool),
                                                  jnp.full((E, H), -1, jnp.int64)))
    return owner


def panel_hits(h, s):
    """Straw hits [E, H] -> panel ComboHits [E, H] compacted to the front in stock output (seed) order."""
    it = _ambient_int()
    with exact.strict(h["sid"]):
        return _leave(_panel_hits(h, s), it)


def _panel_hits(h, s):
    """CombineStrawHits::combineHits in its types and order: double sums over the members in collection order, the
    XYZVectorF centre sum in float (`midpos += centerPos()`, `midpos /= nsh` as a float reciprocal), the position
    midpos + wdist * uDir() and the wire-resolution check in float."""
    E, H = h["valid"].shape
    h = {**h, "sid": h["sid"].astype(jnp.int64), "upanel": h["upanel"].astype(jnp.int64),
         "straw": h["straw"].astype(jnp.int64), "spill": h["spill"].astype(jnp.int64)}
    owner = _group(h, s)
    jpos = jnp.arange(H)[None, :]
    is_seed = h["valid"] & (owner == jpos)
    # members of each seed in collection order, without an [E, seed, hit] matrix: a member's rank in its group is the
    # number of earlier hits of the same owner, all within the owner's panel run (offsets below the longest run)
    own = jnp.where(h["valid"] & (owner >= 0), owner, -1)
    hpos = jnp.broadcast_to(jpos, (E, H))
    span = jnp.max(jnp.where(own >= 0, hpos - own, 0))

    def rank_step(st):
        dd, rank = st
        k = hpos - dd
        same = (own >= 0) & (k >= own) & (jnp.take_along_axis(own, jnp.maximum(k, 0), axis=1) == own)
        return dd + 1, rank + same.astype(rank.dtype)

    _, rank = jax.lax.while_loop(lambda st: st[0] <= span, rank_step, (jnp.ones((), span.dtype), jnp.zeros((E, H), jnp.int32)))
    K = min(MAXNCOMBO, H)
    rows = jnp.broadcast_to(jnp.arange(E)[:, None], (E, H))
    keep = (own >= 0) & (rank < K)
    midx = jnp.zeros((E, H, K), jnp.int32).at[rows, jnp.where(keep, own, H), jnp.where(keep, rank, 0)].set(
        hpos.astype(jnp.int32), mode="drop")                                              # member j of each seed
    ncombo = jnp.zeros((E, H), jnp.int32).at[rows, jnp.where(own >= 0, own, H)].add(1, mode="drop")
    d = lambda x: jnp.asarray(x).astype(F64)
    f = lambda x: jnp.asarray(x).astype(F32)
    col = lambda k, j: jnp.take_along_axis(h[k], midx[:, :, j], axis=1)                      # member j of each seed
    z32, z64 = jnp.zeros((E, H), F32), jnp.zeros((E, H), F64)
    eacc = ctacc = twtsum = wacc = wacc2 = wwtsum = vacc = et0acc = et1acc = z64
    mx = my = mz = z32
    for j in range(K):
        on = j < ncombo
        acc = lambda a, v: jnp.where(on, a + v, a)
        wd = col("wdist", j)
        eacc = acc(eacc, d(col("edep", j)))
        et0acc = acc(et0acc, d(col("et0", j)))
        et1acc = acc(et1acc, d(col("et1", j)))
        vacc = acc(vacc, d(col("vvar", j)))
        twt = 1.0 / d(col("timevar", j))
        twtsum = acc(twtsum, twt)
        ctacc = acc(ctacc, _m(d(col("time", j)), twt))
        wwt = 1.0 / d(col("uvar", j))
        wwtsum = acc(wwtsum, wwt)
        wacc = acc(wacc, _m(d(wd), wwt))
        wacc2 = acc(wacc2, _m(d(_m(wd, wd)), wwt))                                      # float wd*wd, then * double
        mx = acc(mx, col("x", j) - _m(wd, col("udx", j)))                               # centerPos() in float
        my = acc(my, col("y", j) - _m(wd, col("udy", j)))
        mz = acc(mz, col("z", j))
    nsh = d(jnp.maximum(ncombo, 1))
    inv = F32(1) / f(nsh)                                                                # DisplacementVector3D::operator/=
    mx, my, mz = _m(mx, inv), _m(my, inv), _m(mz, inv)
    wdist = f(wacc / wwtsum)
    uerr = _f32(s["uerr"])
    uvar0 = uerr * uerr                                                                  # `_uvar(uerr * uerr)`, float
    uvar = f(1.0 / wwtsum + d(uvar0))
    ndof = jnp.maximum(ncombo - 1, 0)
    nc2 = f(d(ndof) ** 2)
    wvar = jnp.maximum(uvar0, (f(wacc2 / wwtsum) - _m(wdist, wdist)) / nc2)
    uvar = jnp.where(s["checkWres"], jnp.maximum(uvar, wvar), uvar)
    multi = ncombo > 1
    pick = lambda k, v: jnp.where(multi, v, h[k])
    out = {"ncombo": ncombo, "edep": pick("edep", f(eacc / nsh)), "time": pick("time", f(ctacc / twtsum)),
           "timevar": pick("timevar", f(1.0 / twtsum)), "wdist": pick("wdist", wdist),
           "x": pick("x", mx + _m(wdist, h["udx"])), "y": pick("y", my + _m(wdist, h["udy"])),
           "z": pick("z", mz), "uvar": pick("uvar", uvar), "vvar": pick("vvar", f(vacc / nsh)),
           "udx": h["udx"], "udy": h["udy"], "sid": h["sid"]}
    # ComboHit::time(): the end time of the seed hit's early end, the member average of that end for a combo
    hv_early = h["et1"] < h["et0"]
    out["etime"] = jnp.where(hv_early, pick("et1", f(et1acc / nsh)), pick("et0", f(et0acc / nsh)))
    ntdiv = jnp.zeros((E, H), jnp.int32).at[rows, jnp.where(own >= 0, own, H)].add(
        ((h["flag"] >> FLAG["tdiv"]) & 1).astype(jnp.int32), mode="drop")
    tdiv = ntdiv == ncombo
    out["flag"] = _ph_flag(ncombo, out["time"], out["edep"], out["x"], out["y"], tdiv, h["spill"], s)
    # compact seeds to the front, keeping seed order (= stock output order)
    order = jnp.argsort(jnp.where(is_seed, jpos, H + jpos), axis=1)
    out = {k: jnp.take_along_axis(v, order, axis=1) for k, v in out.items()}
    out["valid"] = jnp.take_along_axis(is_seed, order, axis=1)
    out["plane"] = out["sid"] >> 10
    out["face"] = (out["sid"] >> 7) & 1       # StrawId face bit (see _facemsk); checked against stock in validate.py
    return out
