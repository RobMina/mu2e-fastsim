"""Trigger hit reconstruction on the device: StrawDigis -> straw hits (TrigMakeSH) -> panel hits (TrigMakePH).

Events are dense [E, H] blocks (digis in collection order, padded). The straw-hit kernel follows the tracker
package's StrawHitReco port (`trkfastsim.reco.hitreco_jax`) with the trigger's settings: energy from the
firmware peak-minus-pedestal (FitType 5) with the per-straw scale, and the trigger's energy/radius/time windows.
The panel combine follows the stock CombineStrawHits loop with TestFlag on (only hits with every selection bit and
no mask bit are combined or written), one greedy pass over the event's hits.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from trkfastsim.reco.hitreco import CENTRAL_WIRE_POS, EBIN, ERRFAC, FLAG, SLFAC, WBUF, unique_straw
from trkfastsim.reco.hitreco_jax import _pieceline, bucket

jax.config.update("jax_enable_x64", True)
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
    c = {k: jnp.asarray(np.asarray(st[k], np.float64)) for k in
         ("midx", "midy", "midz", "dirx", "diry", "dirz", "hlen", "toff_hv", "toff_cal", "dead", "noisy", "halfvp",
          "pmpscale")}
    for k in ("halfvpscale", "centres", "resslope", "totdtime", "totderror"):
        c[k] = jnp.asarray(np.asarray(cond[k], np.float64))
    for k in ("tdcLSB", "adcLSB", "totLSB", "strawGain", "eionPerQ", "pmpScaleAvg", "strawOuterRadius"):
        c[k] = jnp.float64(sc[k])
    return c, tuple(cond["tot_bins"])


def pack_events(ev, nh=None):
    """Per-event digi lists (dict of awkward-like lists of numpy arrays) -> dense [E, H] numpy block."""
    E = len(ev["sid"])
    n = np.array([len(s) for s in ev["sid"]], np.int64)
    H = nh or bucket(int(n.max()) if E else 1, 64)
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


def straw_hits(p, c, s, tot_bins):
    """[E, H] digis -> straw-hit columns, as StrawHitReco (FilterHits false: one hit per digi, flags carry cuts)."""
    sid, valid = p["sid"], p["valid"]
    us = unique_straw(sid)
    if s["fittype"] != 5:
        raise ValueError("device straw hits implement FitType 5 (firmware PMP) only")
    charge = p["pmp"] * (c["adcLSB"] / c["strawGain"]) * c["pmpscale"][us]
    energy = (c["eionPerQ"] * charge).astype(F32)
    pbt = p["pbt"][:, None].astype(F32)
    t_cal = ((p["tdc0"] * c["tdcLSB"] + c["toff_cal"][us]).astype(F32) - pbt).astype(F32)
    t_hv = ((p["tdc1"] * c["tdcLSB"] + c["toff_hv"][us]).astype(F32) - pbt).astype(F32)
    eend = (t_hv < t_cal).astype(jnp.int64)
    tot_cal, tot_hv = (p["tot0"] * c["totLSB"]).astype(F32), (p["tot1"] * c["totLSB"]).astype(F32)
    tot_sel = jnp.where(eend == 1, tot_hv, tot_cal)
    dt = t_hv.astype(F64) - t_cal.astype(F64)
    kedep = 1000.0 * energy.astype(F64)
    egrid = jnp.arange(c["halfvpscale"].shape[0]) * EBIN
    halfpv = _pieceline(egrid, c["halfvpscale"], kedep) * c["halfvp"][us]
    wdist = halfpv * dt
    wres = _pieceline(egrid, c["centres"], kedep)
    wres = jnp.maximum(30.0, wres + jnp.where(jnp.abs(wdist) > CENTRAL_WIRE_POS,
                                              (jnp.abs(wdist) - CENTRAL_WIRE_POS) * _pieceline(egrid, c["resslope"], kedep), 0.0))
    hlen = c["hlen"][us]
    trunc = jnp.abs(wdist) > hlen + WBUF * wres
    wdist = jnp.where(trunc, jnp.copysign(hlen * SLFAC, wdist), wdist)
    wres = jnp.where(trunc, wres * ERRFAC, wres)
    esign = 2.0 * (eend - 0.5)
    ptime = ((hlen + esign * wdist).astype(F32) / (2 * halfpv)).astype(F32)
    x = (c["midx"][us] + wdist * c["dirx"][us]).astype(F32)
    y = (c["midy"][us] + wdist * c["diry"][us]).astype(F32)
    z = (c["midz"][us] + wdist * c["dirz"][us]).astype(F32)
    rho = jnp.hypot(x, y)
    ntb, wtb, neb, web = tot_bins
    tbin = jnp.minimum(int(ntb) - 1, (tot_sel / F32(wtb)).astype(jnp.int64))
    ebin = jnp.minimum(int(neb) - 1, (energy / F32(web)).astype(jnp.int64))
    dtime = c["totdtime"][tbin * int(neb) + ebin]
    tres = c["totderror"][tbin * int(neb) + ebin]
    t_early = jnp.where(eend == 1, t_hv, t_cal).astype(F64)
    time = (t_early - ptime - dtime).astype(F32)
    off = (p["spill"] == SPILL_OFF)[:, None]
    minT, maxT = jnp.where(off, s["minTOff"], s["minT"]), jnp.where(off, s["maxTOff"], s["maxT"])
    flag = jnp.zeros(sid.shape, jnp.uint32)
    for bit, ok in ((FLAG["dead"], c["dead"][us] > 0), (FLAG["noisy"], c["noisy"][us] > 0),
                    (FLAG["energysel"], (energy <= s["maxE"]) & (energy >= s["minE"])),
                    (FLAG["radsel"], (rho >= s["minR"]) & (rho <= s["maxR"])),
                    (FLAG["timesel"], (time >= minT) & (time <= maxT)), (FLAG["tdiv"], ~trunc)):
        flag = flag | jnp.where(ok, jnp.uint32(1 << bit), jnp.uint32(0))
    flag = jnp.where(valid, flag, jnp.uint32(0))
    rout = c["strawOuterRadius"]
    return {"valid": valid, "spill": p["spill"], "sid": sid, "upanel": (sid >> 10) * 6 + ((sid & 0x380) >> 7), "straw": sid & 0x7F,
            "time": time, "et0": t_cal, "et1": t_hv, "edep": energy, "wdist": wdist.astype(F32),
            "uvar": (wres * wres).astype(F32), "vvar": jnp.full(sid.shape, rout * rout / 3.0).astype(F32),
            "timevar": (tres * tres).astype(F32), "x": x, "y": y, "z": z, "udx": c["dirx"][us].astype(F32),
            "udy": c["diry"][us].astype(F32), "flag": flag}


def _group(h, s):
    """Stock CombineStrawHits greedy loop over the event's hits, vectorised over events: owner seed per hit."""
    E, H = h["valid"].shape
    good = h["valid"] & ((h["flag"] & s["sel"]) == s["sel"]) & ((h["flag"] & s["mask"]) == 0) & \
        ((h["flag"] & (1 << FLAG["dead"])) == 0)
    jpos = jnp.arange(H)[None, :]

    def body(i, state):
        used, owner = state
        col = lambda k: jnp.take(h[k], i, axis=1)[:, None]
        seed = h["valid"][:, i] & ~used[:, i]
        gseed = seed & good[:, i]
        same = (h["upanel"] == col("upanel")) & h["valid"]
        # the stock loop breaks at the first hit of another panel (input sorted by panel)
        run = jnp.cumprod(jnp.where(jpos > i, same, True), axis=1).astype(bool)
        ok = (jpos > i) & run & ~used & good & gseed[:, None]
        ok &= jnp.abs(h["straw"] - col("straw")) <= s["maxds"]
        ok &= jnp.abs(col("time") - h["time"]) <= s["maxdt"]
        ok &= (jnp.abs(col("wdist") - h["wdist"]) / jnp.sqrt(col("uvar") + h["uvar"])) <= s["maxwdchi"]
        take = ok & (jnp.cumsum(ok, axis=1) <= MAXNCOMBO - 1)
        used = used.at[:, i].set(used[:, i] | seed) | take
        owner = jnp.where(take, i, owner)
        owner = owner.at[:, i].set(jnp.where(gseed, i, owner[:, i]))
        return used, owner

    _, owner = jax.lax.fori_loop(0, H, body, (jnp.zeros((E, H), bool), jnp.full((E, H), -1, jnp.int64)))
    return owner


def panel_hits(h, s):
    """Straw hits [E, H] -> panel ComboHits [E, H] compacted to the front in stock output (seed) order.

    Members are summed in collection order (as the tracker port, bitwise equal to stock on CPU)."""
    E, H = h["valid"].shape
    owner = _group(h, s)
    jpos = jnp.arange(H)[None, :]
    is_seed = h["valid"] & (owner == jpos)
    mem = (owner[:, None, :] == jnp.arange(H)[None, :, None]) & h["valid"][:, None, :]     # [E, seed, hit]
    f64 = lambda k: h[k].astype(F64)[:, None, :]
    acc = lambda v: jnp.sum(jnp.where(mem, v, 0.0), axis=2)
    ncombo = jnp.sum(mem, axis=2)
    nsh = jnp.maximum(ncombo, 1).astype(F64)
    wd = f64("wdist")
    twt, wwt = 1.0 / f64("timevar"), 1.0 / f64("uvar")
    cx, cy = acc(f64("x") - wd * f64("udx")) / nsh, acc(f64("y") - wd * f64("udy")) / nsh
    cz = acc(f64("z")) / nsh
    wsum, wacc, wacc2 = acc(wwt), acc(wd * wwt), acc(wd * wd * wwt)
    wdist = wacc / wsum
    multi = ncombo > 1
    uvar = 1.0 / wsum + s["uerr"] ** 2
    ndof2 = (jnp.maximum(ncombo - 1, 1) ** 2).astype(F64)
    uvar = jnp.where(s["checkWres"], jnp.maximum(uvar, jnp.maximum(s["uerr"] ** 2, (wacc2 / wsum - wdist ** 2) / ndof2)), uvar)
    pick = lambda k, v: jnp.where(multi, v, h[k])
    out = {"ncombo": ncombo, "edep": pick("edep", acc(f64("edep")) / nsh).astype(F32),
           "time": pick("time", acc(f64("time") * twt) / acc(twt)).astype(F32),
           "timevar": pick("timevar", 1.0 / acc(twt)).astype(F32),
           "wdist": pick("wdist", wdist).astype(F32),
           "x": pick("x", cx + wdist * h["udx"].astype(F64)).astype(F32),
           "y": pick("y", cy + wdist * h["udy"].astype(F64)).astype(F32),
           "z": pick("z", cz).astype(F32), "uvar": pick("uvar", uvar).astype(F32),
           "vvar": pick("vvar", acc(f64("vvar")) / nsh).astype(F32),
           "udx": h["udx"], "udy": h["udy"], "sid": h["sid"]}
    tdiv = jnp.sum(jnp.where(mem, ((h["flag"] >> FLAG["tdiv"]) & 1)[:, None, :], 0), axis=2) == ncombo
    off = (h["spill"] == SPILL_OFF)[:, None]
    minT, maxT = jnp.where(off, s["minTOff"], s["minT"]), jnp.where(off, s["maxTOff"], s["maxT"])
    r2 = out["x"].astype(F64) ** 2 + out["y"].astype(F64) ** 2
    flag = jnp.uint32(1 << FLAG["panelcombo"]) | jnp.where(tdiv, jnp.uint32(1 << FLAG["tdiv"]), jnp.uint32(0))
    for bit, ok in ((FLAG["nhitsel"], (ncombo >= s["minN"]) & (ncombo <= s["maxN"])),
                    (FLAG["timesel"], (out["time"] >= minT) & (out["time"] <= maxT)),
                    (FLAG["energysel"], (out["edep"] >= s["minE"]) & (out["edep"] <= s["maxE"])),
                    (FLAG["radsel"], (r2 >= s["minR"] ** 2) & (r2 <= s["maxR"] ** 2))):
        flag = flag | jnp.where(ok, jnp.uint32(1 << bit), jnp.uint32(0))
    out["flag"] = flag
    # compact seeds to the front, keeping seed order (= stock output order)
    order = jnp.argsort(jnp.where(is_seed, jpos, H + jpos), axis=1)
    out = {k: jnp.take_along_axis(v, order, axis=1) for k, v in out.items()}
    out["valid"] = jnp.take_along_axis(is_seed, order, axis=1)
    out["plane"] = out["sid"] >> 10
    out["face"] = (out["sid"] >> 7) & 1       # StrawId face bit (see _facemsk); checked against stock in validate.py
    return out
