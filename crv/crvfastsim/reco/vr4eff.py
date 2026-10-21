"""V-R4 EX layer efficiency with extrapolated tracks: fast chain vs stock chain on the crvonly_v05 events.

The stock arm is the crvonly_v05 EventNtuple (timing-fixed CRV-only MC: CRVTime applied with scale -1, one FEB clock
phase per event): its KKLine tracks and its `crvpulses`. The fast arm runs the fast chain (crvfastsim.chain, numpy or
JAX) on the CrvSteps of the same production dts file (CrvStepTableDumper table) with the v05 settings (common FEB
phase). The track lines (one quality track per event, from the tracker, crvextrap.trackline) are identical for both
arms, so the difference is the CRV response and reco only.

Per EX layer L (0-3): denominator = events whose track crosses the EX layer-L plane inside the sector; numerator = an
in-time pulse (PE >= pe_probe, |t - t_track - offset| <= window) on a counter of layer L within `nwin` counters of the
predicted one. Efficiency per arm and the difference, judged against V-R4's 0.3 points (2 sigma allowed: the arms
share the tracks and the steps, so the difference is computed per event and its error from the discordant pairs).
"""
import numpy as np

from .. import chain, geometry, tables

EX = 0                # EX sector index in the extracted counters file
PE_PROBE, NWIN, DT_OFFSET, DT_HALF = 8.0, 1, 47.0, 20.0     # crvextrap exdenom / timewin defaults
THRESH_POINTS = 0.003
# EventNtuple stores detector coordinates (tracker centre): z_mu2e = z_det + 24171, x_mu2e = x_det - 3904 (extracted)
DET_Z0, SOLENOID_OFFSET = 24171.0, 3904.0


def ex_layers(geom):
    """Per EX layer: thickness-axis position, width-axis counter centres (sorted) and their bar ids."""
    out = []
    b = np.flatnonzero(geom.sector == EX)
    dt, dw = geom.local_to_world[b[0], 0], geom.local_to_world[b[0], 1]
    for L in range(4):
        bl = b[geom.layer[b] == L]
        o = np.argsort(geom.center[bl, dw])
        out.append({"y": float(np.median(geom.center[bl, dt])), "w": geom.center[bl[o], dw], "bars": bl[o],
                    "half": float(geom.halflen[bl[0], dw]), "dt": int(dt), "dw": int(dw)})
    return out


def predicted(lines, layers):
    """Per event and layer: predicted counter index (-1 = outside the sector) and track time at that layer."""
    n = len(lines["x"])
    idx = np.full((n, 4), -1, np.int64)
    tl = np.full((n, 4), np.nan)
    p = np.stack([lines["x"] - SOLENOID_OFFSET, lines["y"], lines["z"] + DET_Z0], 1)    # detector -> mu2e frame
    d = np.stack([lines["dx"], lines["dy"], lines["dz"]], 1)
    speed = 299.792458 * np.ones(n)   # mm/ns; muons at c (the reco intersection time is defined the same way)
    for L, lay in enumerate(layers):
        a = lay["dt"]
        s = (lay["y"] - p[:, a]) / np.where(np.abs(d[:, a]) > 1e-12, d[:, a], np.nan)
        q = p + d * s[:, None]
        w = q[:, lay["dw"]]
        k = np.searchsorted(lay["w"], w)
        k = np.clip(k, 1, len(lay["w"]) - 1)
        k = np.where(np.abs(lay["w"][k - 1] - w) < np.abs(lay["w"][k] - w), k - 1, k)
        inside = np.isfinite(w) & (w >= lay["w"][0] - lay["half"]) & (w <= lay["w"][-1] + lay["half"])
        idx[:, L] = np.where(inside, k, -1)
        dist = s * np.linalg.norm(d, axis=1)
        tl[:, L] = lines["time"] + dist / speed
    return idx, tl


def layer_hits(ev_pos, pulses, layers, idx, tl, pe=PE_PROBE, nwin=NWIN, off=DT_OFFSET, half=DT_HALF):
    """[n_events, 4] bool: an in-time pulse >= pe on a counter within nwin of the predicted one.
    pulses: flat ev (row index into the event list), bar, pe, time, seq."""
    n = idx.shape[0]
    hit = np.zeros((n, 4), bool)
    bar_to = {}
    for L, lay in enumerate(layers):
        for j, b in enumerate(lay["bars"]):
            bar_to[int(b)] = (L, j)
    lut_L = np.full(int(max(bar_to) + 1), -1)
    lut_j = np.full(int(max(bar_to) + 1), -1)
    for b, (L, j) in bar_to.items():
        lut_L[b], lut_j[b] = L, j
    b = pulses["bar"]
    ok = (b < len(lut_L)) & (pulses["seq"] == 0) & (pulses["pe"] >= pe) & (ev_pos >= 0)
    b, e = b[ok], ev_pos[ok]
    L = lut_L[np.clip(b, 0, len(lut_L) - 1)]
    m = L >= 0
    b, e, L, t = b[m], e[m], L[m], pulses["time"][ok][m]
    j = lut_j[b]
    near = (idx[e, L] >= 0) & (np.abs(j - idx[e, L]) <= nwin) & (np.abs(t - tl[e, L] - off) <= half)
    hit[e[near], L[near]] = True
    return hit


def compare(hit_f, hit_r, den):
    out = {}
    for L in range(4):
        d = den[:, L]
        nd = int(d.sum())
        ef, er = hit_f[d, L].mean(), hit_r[d, L].mean()
        b = int((hit_f[d, L] & ~hit_r[d, L]).sum())
        c = int((~hit_f[d, L] & hit_r[d, L]).sum())
        err = np.sqrt(b + c) / max(nd, 1)          # paired (McNemar) error of the difference
        out[L] = {"den": nd, "eff_fast": float(ef), "eff_stock": float(er), "diff": float(ef - er),
                  "diff_err": float(err), "only_fast": b, "only_stock": c,
                  "pass": bool(max(abs(ef - er) - 2 * err, 0) <= THRESH_POINTS)}
    out["pass"] = all(out[L]["pass"] for L in range(4))
    return out


def prep(ntuple, out_npz, trk_cuts=None):
    """GPVM step (light, seconds per file; needs crvextrap on PYTHONPATH): one quality track line per event
    (crvextrap.trackline.lines, from the tracker) and the stock EX seq-0 pulses, from a crvonly_v05 EventNtuple."""
    import awkward as ak
    import uproot
    from crvextrap import trackline as TL
    cuts = trk_cuts or {"nactive": 20, "fitcon": 0.01, "chi2": 3.0, "nplanes": 0}
    t = uproot.open(ntuple)["EventNtuple/ntuple"]
    e = t.arrays(filter_name=["evtinfo/subrun", "evtinfo/event"], library="np")
    e = {k.split("/")[-1]: v for k, v in e.items()}
    tk = TL.lines(t, cuts=cuts, from_tracker=True)
    a = t.arrays(["crvpulses/crvpulses.barId", "crvpulses/crvpulses.PEs", "crvpulses/crvpulses.time",
                  "crvpulses/crvpulses.sequenceIndex"], library="ak")
    n = ak.num(a["crvpulses/crvpulses.barId"])
    flat = lambda k: ak.to_numpy(ak.flatten(a[f"crvpulses/crvpulses.{k}"]))
    np.savez_compressed(out_npz, subrun=e["subrun"], event=e["event"],
                        **{f"trk_{k}": tk[k] for k in ("x", "y", "z", "dx", "dy", "dz", "time")},
                        p_ev=np.repeat(np.arange(len(n)), ak.to_numpy(n)), p_bar=flat("barId"), p_pe=flat("PEs"),
                        p_time=flat("time"), p_seq=flat("sequenceIndex"))
    print(f"wrote {out_npz}: {len(n)} events, {int(np.isfinite(tk['x']).sum())} with a quality track")


def run(steps, prep_npz, calib, emit, deviation=None, jax=False, seed=1, batch_events=None, counters=None,
        max_events=None):
    """Grid step: fast chain on the CrvSteps of the same dts file with the v05 settings (one FEB clock phase per event,
    CRVTime net zero), then per-layer hits for both arms on the same tracks; writes counts to `emit` (JSON)."""
    import dataclasses
    import json
    from . import vr4
    from ..response import photons as PH
    from ..response import sipm as SI
    z = np.load(prep_npz)
    cpath = geometry.find_counters(counters or "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt")
    geom = geometry.from_offline_config(cpath)
    ccfg = geometry.parse_simpleconfig(open(cpath).read())
    cond = chain.Conditions.from_csv(calib, deviation)
    # v05: febPhaseMode common (one TDC0 draw per event: every channel on one FEB) and CRVTime net zero
    cond = dataclasses.replace(cond, roc=np.zeros_like(cond.roc), feb=np.zeros_like(cond.feb),
                               timeoffset=np.zeros_like(cond.timeoffset))
    cfg = chain.ChainConfig(jax=bool(jax))
    tabs, key = PH.load_tables(cfg.response), None
    if jax:
        import jax as J
        from ..response import photons_jax
        tabs = {k: photons_jax.DeviceTable(v) for k, v in tabs.items()}
        key = J.random.PRNGKey(int(seed))
    rng = np.random.default_rng(int(seed))
    pulses_f, _, _ = vr4.fast_arm(steps, cond, cfg, rng, tabs, SI.photon_map(), geom, ccfg, max_events, key=key,
                                  batch_events=batch_events)
    layers = ex_layers(geom)
    nkey = z["subrun"].astype(np.int64) << 20 | z["event"].astype(np.int64)
    st = tables.read(steps, "steps", tables.KEY)
    sk = np.unique(chain.event_key(st))
    has_steps = np.isin(nkey, sk[:int(max_events)] if max_events else sk)
    lines = {k: z[f"trk_{k}"] for k in ("x", "y", "z", "dx", "dy", "dz", "time")}
    idx, tl = predicted(lines, layers)
    den = (idx >= 0) & has_steps[:, None] & np.isfinite(lines["time"])[:, None]
    order = np.argsort(nkey)
    pos_f = order[np.clip(np.searchsorted(nkey[order], pulses_f["ev"]), 0, len(nkey) - 1)]
    pos_f = np.where(nkey[pos_f] == pulses_f["ev"], pos_f, -1)
    ps = {"bar": z["p_bar"].astype(np.int64), "pe": z["p_pe"], "time": z["p_time"], "seq": z["p_seq"]}
    # time offset of the probe window from the stock arm itself (median t_pulse - t_track on the predicted counter)
    wide = layer_dt(z["p_ev"], ps, layers, idx, tl)
    off = float(np.median(wide)) if len(wide) else DT_OFFSET
    hit_r = layer_hits(z["p_ev"], ps, layers, idx, tl, off=off)
    hit_f = layer_hits(pos_f, pulses_f, layers, idx, tl, off=off)
    res = {"steps": steps, "prep": prep_npz, "events": int(len(nkey)), "events_with_steps": int(has_steps.sum()),
           "dt_offset": off, "layers": {}}
    for L in range(4):
        d = den[:, L]
        res["layers"][L] = {"den": int(d.sum()), "both": int((hit_f[d, L] & hit_r[d, L]).sum()),
                            "only_fast": int((hit_f[d, L] & ~hit_r[d, L]).sum()),
                            "only_stock": int((~hit_f[d, L] & hit_r[d, L]).sum())}
    json.dump(res, open(emit, "w"), indent=1)
    print(json.dumps(res["layers"]), "dt_offset", off)
    return res


def layer_dt(ev_pos, pulses, layers, idx, tl, pe=PE_PROBE, nwin=NWIN):
    """t_pulse - t_track for seq-0 pulses >= pe on the predicted counter (+-nwin), any time: the window offset."""
    out = []
    for L, lay in enumerate(layers):
        j = np.full(int(lay["bars"].max()) + 1, -1)
        j[lay["bars"]] = np.arange(len(lay["bars"]))
        b = pulses["bar"]
        m = (b <= lay["bars"].max()) & (pulses["seq"] == 0) & (pulses["pe"] >= pe) & (ev_pos >= 0)
        jj = j[np.clip(b, 0, len(j) - 1)]
        m &= jj >= 0
        e = ev_pos[m]
        ok = (idx[e, L] >= 0) & (np.abs(jj[m] - idx[e, L]) <= nwin)
        out.append((pulses["time"][m] - tl[e, L])[ok])
    d = np.concatenate(out) if out else np.zeros(0)
    return d[np.abs(d - np.median(d)) < 100] if len(d) else d


def merge(paths):
    import glob
    import json
    tot = {L: {"den": 0, "both": 0, "only_fast": 0, "only_stock": 0} for L in range(4)}
    files = sorted(sum((glob.glob(p) for p in paths), []))
    for f in files:
        d = json.load(open(f))
        for L in range(4):
            for k in tot[L]:
                tot[L][k] += d["layers"][str(L)][k]
    out = {"files": len(files), "layers": {}}
    for L in range(4):
        t = tot[L]
        nd = max(t["den"], 1)
        ef, er = (t["both"] + t["only_fast"]) / nd, (t["both"] + t["only_stock"]) / nd
        err = np.sqrt(t["only_fast"] + t["only_stock"]) / nd
        out["layers"][L] = {**t, "eff_fast": ef, "eff_stock": er, "diff": ef - er, "diff_err": err,
                            "pass": bool(max(abs(ef - er) - 2 * err, 0) <= THRESH_POINTS)}
    out["pass"] = all(out["layers"][L]["pass"] for L in range(4))
    return out
