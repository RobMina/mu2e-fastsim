"""Run the fast CRV chain on CrvStep tables and write its digis, pulses and clusters as a CrvRecoTableDumper-layout file
(directory crvdump/, trees digis, pulses, clusters, events), the input of the CrvFastSim *FromTable art producers.

Stages (decide which arm the file serves):
  digis : fast response + waveform synthesis -> digis only                         (arm A2: stock reco on fast digis)
  reco  : stock digis (a CrvRecoTableDumper file) -> fast pulses + clusters        (arm A3)
  chain : fast response + fast reco -> digis, pulses, clusters                     (arms A4 on G4 steps, E2E on fast steps)

    python -m valhist.fasttables chain STEPS.root OUT.root --calib C.csv [--deviation D.csv] [--seed S] [--jax]
    python -m valhist.fasttables reco  REF.root   OUT.root --calib C.csv

Layout = CrvRecoTableDumper: keys run/subrun/event (uint32), per row bar, sipm, idx; digis starttdc, nzs, adc[];
pulses pe, peph, height, beta, chi2, ped, time, letime, seq, flags, pedfromdb, wfidx[] (digi rows of the waveform);
clusters sectortype, start, end, pe, slope, layers[], pulses[] (pulse rows of the event), poscalc, avgtime, x, y, z
(Mu2e frame), sidepe0/1, sidetime0/1, sidehits0/1. Rows of one event are contiguous and in the stock collection order.
Pulses use the numpy reference fit (reco.pulsefit, V-R2), which carries chi2, LE time, height and beta; the clusters
use the numpy coincidence finder (reco.coinc, V-R3) on those pulses, so the file is self-consistent.
"""
import argparse
import os

import awkward as ak
import numpy as np
import uproot

from crvfastsim import chain, geometry, tables
from crvfastsim.analyses import v2compare as V2
from crvfastsim.reco import coinc as C
from crvfastsim.reco import pulsefit

FLAG_BITS = {"failed": 0, "zerondf": 3}        # CrvRecoPulseFlagEnums failedFit, zeroNdf


def _split_label(chan_label):
    ev, ch = chan_label // chain.CHAN_MULT, chan_label % chain.CHAN_MULT
    return ev, ch // 4, ch % 4


def _keys(ev, run):
    """Event label (subrun << 20 | event) -> (run, subrun, event)."""
    return np.full(len(ev), run, np.uint32), (ev >> 20).astype(np.uint32), (ev & ((1 << 20) - 1)).astype(np.uint32)


def stock_digis(ref_path):
    """Digis of a CrvRecoTableDumper file, with the event label of the fast chain."""
    with uproot.open(ref_path) as f:
        t = f["crvdump/digis"]
        a = t.arrays(["run", "subrun", "event", "bar", "sipm", "starttdc", "adc"])
    ev = (ak.to_numpy(a.subrun).astype(np.int64) << 20) | ak.to_numpy(a.event).astype(np.int64)
    adc = ak.to_numpy(ak.pad_none(a.adc, int(ak.max(ak.num(a.adc))), clip=True).to_numpy().filled(0)) \
        if len(a) else np.zeros((0, 8), np.int16)
    return {"ev": ev, "bar": ak.to_numpy(a.bar).astype(np.int64), "sipm": ak.to_numpy(a.sipm).astype(np.int64),
            "starttdc": ak.to_numpy(a.starttdc).astype(np.int64), "adc": np.asarray(adc, np.int16),
            "run": int(a.run[0]) if len(a) else 1400}


def fast_reco(D, cond, geom, ccfg, rcfg=None, fcfg=None):
    """Numpy reference reco on digis D (stock order). Returns pulse and cluster tables keyed by event label."""
    rcfg = rcfg or pulsefit.RecoConfig()
    fcfg = fcfg or C.FinderConfig()
    nd = len(D["ev"])
    lens = np.full(nd, D["adc"].shape[1], np.int64)
    gid, mat, n, start, first = pulsefit.concat_waveforms(D["ev"], D["bar"], D["sipm"], D["starttdc"],
                                                          D["adc"].ravel(), lens)
    ch = D["bar"][first] * 4 + D["sipm"][first]
    P = pulsefit.reconstruct_full(mat, n, start, cond.pedestal[ch], cond.pulsearea[ch], cond.pulseheight[ch], rcfg)
    P = pulsefit.filter_channels(P, ch, cond.status, rcfg)
    if rcfg.use_time_offset_db:
        P["time"] = P["time"] + cond.timeoffset[ch][P["wf"]] * rcfg.time_offset_scale
        P["letime"] = P["letime"] + cond.timeoffset[ch][P["wf"]] * rcfg.time_offset_scale
    wf = P["wf"]
    pev = D["ev"][first][wf]
    o = np.lexsort((P["seq"], wf, pev))            # stock order: event, waveform (digi order), sequence
    P = {k: (v[o] if hasattr(v, "__len__") and len(v) == len(wf) else v) for k, v in P.items()}
    wf, pev = wf[o], pev[o]
    pidx = np.arange(len(pev)) - np.searchsorted(pev, pev)      # pulse row within its event
    # digi rows of each waveform, as indices within the event's digi collection
    devstart = np.searchsorted(D["ev"], D["ev"])
    digi_in_ev = np.arange(nd) - devstart
    wf_digis = ak.unflatten(digi_in_ev, np.bincount(gid, minlength=len(first)))
    flags = np.zeros(len(wf), np.int64)
    for k, bit in FLAG_BITS.items():
        flags |= P[k].astype(np.int64) << bit
    pulses = {"ev": pev, "idx": pidx, "bar": D["bar"][first][wf], "sipm": D["sipm"][first][wf],
              "pe": P["pe"], "peph": P["peph"], "height": P["height"], "beta": P["beta"], "chi2": P["chi2"],
              "ped": P["ped"], "time": P["time"], "letime": P["letime"], "seq": P["seq"], "flags": flags,
              "pedfromdb": P["pedfromdb"].astype(np.int64), "wfidx": wf_digis[wf]}
    G = C.sector_geom(geom, ccfg)
    status = cond.status
    cl = {k: [] for k in ("ev", "idx", "sectortype", "start", "end", "pe", "slope", "layers", "pulses", "poscalc",
                          "avgtime", "x", "y", "z", "sidepe0", "sidepe1", "sidetime0", "sidetime1", "sidehits0",
                          "sidehits1")}
    ue, lo = np.unique(pev, return_index=True)
    hi = np.r_[lo[1:], len(pev)]
    for e, a, b in zip(ue, lo, hi):
        p = dict(idx=pidx[a:b], bar=pulses["bar"][a:b], sipm=pulses["sipm"][a:b], time=P["time"][a:b],
                 pe=P["pe"][a:b], seq=P["seq"][a:b])
        for j, r in enumerate(C.find_event(p, G, status, fcfg)):
            pos = np.asarray(r["pos"], float)
            for k, v in (("ev", e), ("idx", j), ("sectortype", r["sectortype"]), ("start", r["start"]),
                         ("end", r["end"]), ("pe", r["pe"]), ("slope", r["slope"]), ("layers", list(r["layers"])),
                         ("pulses", list(r["pulses"])), ("poscalc", int(r["poscalc"])), ("avgtime", r["avgtime"]),
                         ("x", pos[0]), ("y", pos[1]), ("z", pos[2]), ("sidepe0", r["sidepe"][0]),
                         ("sidepe1", r["sidepe"][1]), ("sidetime0", r["sidetime"][0]),
                         ("sidetime1", r["sidetime"][1]), ("sidehits0", r["sidehits"][0]),
                         ("sidehits1", r["sidehits"][1])):
                cl[k].append(v)
    return pulses, cl


def _tree(f, name, cols):
    """Write a TTree (not RNTuple, the uproot >= 5.7 default): flat numpy columns and jagged awkward columns, the latter
    as counted arrays (counter n<name>) that the *FromTable producers read."""
    types = {k: (v.layout.content.dtype if isinstance(v, ak.Array) else v.dtype) for k, v in cols.items()}
    jag = {k for k, v in cols.items() if isinstance(v, ak.Array)}
    f.mktree(name, {k: ("var * " + str(types[k]) if k in jag else types[k]) for k in cols})
    n = len(next(iter(cols.values())))
    if n:
        f[name].extend(cols)


def _jagged(rows, dtype):
    """list of per-row sequences (or an awkward array) -> awkward array of dtype, also for zero rows."""
    if isinstance(rows, ak.Array):
        return ak.values_astype(rows, dtype)
    flat = np.concatenate([np.asarray(r, dtype) for r in rows]) if len(rows) else np.zeros(0, dtype)
    return ak.unflatten(flat, np.array([len(r) for r in rows], np.int64))


def write(path, run, D, pulses=None, cl=None, events=None):
    """CrvRecoTableDumper-layout file (TTrees in crvdump/); events: event labels of the job (also those without rows)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    i32, f32, f64 = np.int32, np.float32, np.float64

    def keyed(ev):
        r, s, e = _keys(np.asarray(ev, np.int64), run)
        return {"run": r, "subrun": s, "event": e}

    with uproot.recreate(path) as f:
        nd = len(D["ev"])
        didx = np.arange(nd) - np.searchsorted(D["ev"], D["ev"])
        _tree(f, "crvdump/digis", {**keyed(D["ev"]), "idx": didx.astype(i32), "bar": D["bar"].astype(i32),
                                   "sipm": D["sipm"].astype(i32), "starttdc": D["starttdc"].astype(i32),
                                   "nzs": np.zeros(nd, i32), "roc": np.full(nd, -1, i32), "feb": np.full(nd, -1, i32),
                                   "febch": np.full(nd, -1, i32), "adc": _jagged(list(D["adc"]), np.int16)})
        if pulses is not None:
            _tree(f, "crvdump/pulses", {**keyed(pulses["ev"]), "idx": pulses["idx"].astype(i32),
                                        "bar": pulses["bar"].astype(i32), "sipm": pulses["sipm"].astype(i32),
                                        **{k: np.asarray(pulses[k], f32) for k in ("pe", "peph", "height", "beta",
                                                                                    "chi2", "ped")},
                                        "time": np.asarray(pulses["time"], f64),
                                        "letime": np.asarray(pulses["letime"], f64),
                                        **{k: np.asarray(pulses[k], i32) for k in ("seq", "flags", "pedfromdb")},
                                        "wfidx": _jagged(pulses["wfidx"], np.int32)})
        if cl is not None:
            n = len(cl["ev"])
            _tree(f, "crvdump/clusters", {**keyed(np.asarray(cl["ev"], np.int64)),
                                          **{k: np.asarray(cl[k], i32).reshape(n) for k in
                                             ("idx", "sectortype", "poscalc", "sidehits0", "sidehits1")},
                                          **{k: np.asarray(cl[k], f64).reshape(n) for k in
                                             ("start", "end", "avgtime", "sidetime0", "sidetime1")},
                                          **{k: np.asarray(cl[k], f32).reshape(n) for k in
                                             ("pe", "slope", "x", "y", "z", "sidepe0", "sidepe1")},
                                          "layers": _jagged(cl["layers"], np.int32),
                                          "pulses": _jagged(cl["pulses"], np.int32)})
        if events is not None:
            ev = np.unique(np.asarray(events, np.int64))
            count = lambda x: np.bincount(np.searchsorted(ev, np.asarray(x, np.int64)), minlength=len(ev)).astype(i32) \
                if len(x) else np.zeros(len(ev), i32)
            _tree(f, "crvdump/events", {**keyed(ev), "ndigi": count(D["ev"]),
                                        "npulse": count(pulses["ev"]) if pulses is not None else np.zeros(len(ev), i32),
                                        "ncluster": count(cl["ev"]) if cl is not None else np.zeros(len(ev), i32)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("digis", "reco", "chain"))
    ap.add_argument("inp", help="CrvStep table (digis, chain) or CrvRecoTableDumper file (reco)")
    ap.add_argument("out")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--deviation", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jax", action="store_true", help="fast response and waveforms on JAX (reco stays numpy)")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--run", type=int, default=1400)
    ap.add_argument("--counters", default=None)
    a = ap.parse_args()
    cpath = geometry.find_counters(a.counters or V2.DEFAULT_COUNTERS)
    geom = geometry.from_offline_config(cpath)
    ccfg = geometry.parse_simpleconfig(open(cpath).read())
    cond = chain.Conditions.from_csv(a.calib, a.deviation)
    rng = np.random.default_rng(a.seed)
    if a.stage == "reco":
        D = stock_digis(a.inp)
        run = D.pop("run")
        pulses, cl = fast_reco(D, cond, geom, ccfg)
        with uproot.open(a.inp) as f:
            e = f["crvdump/events"].arrays(["subrun", "event"], library="np")
        events = (e["subrun"].astype(np.int64) << 20) | e["event"].astype(np.int64)
        write(a.out, run, D, pulses, cl, events)
    else:
        from crvfastsim.response import photons as PH
        from crvfastsim.response import sipm as SI
        cfg = chain.ChainConfig(jax=a.jax, noise=not a.no_noise)
        tabs = PH.load_tables(cfg.response)
        key = None
        if a.jax:
            import jax
            from crvfastsim.response import photons_jax
            tabs = {k: photons_jax.DeviceTable(v) for k, v in tabs.items()}
            key = jax.random.PRNGKey(a.seed)
        pmap = SI.photon_map()
        steps = tables.read(a.inp, "steps")
        parts = tables.read(a.inp, "particles", tables.KEY + ["key", "pdg"])
        run = int(steps.run.iloc[0]) if len(steps) else a.run
        D = fast_digis(steps, parts, geom, cond, cfg, rng, key, tabs, pmap)
        pulses = cl = None
        if a.stage == "chain":
            pulses, cl = fast_reco(D, cond, geom, ccfg)
        write(a.out, run, D, pulses, cl, chain.event_key(steps))
    print(f"wrote {a.out}")


def fast_digis(steps, parts, geom, cond, cfg, rng, key, tabs, pmap, batch_events=None):
    """Fast response + waveforms on one step table; per digi ev, bar, sipm, starttdc, adc [n, L], stock order."""
    evk = chain.event_key(steps).astype(np.int64)
    uev = np.unique(evk)
    batch_events = batch_events or (250 if cfg.jax else max(len(uev), 1))
    out = []
    for b in range(0, len(uev), batch_events):
        st = steps[np.isin(evk, uev[b:b + batch_events])].reset_index(drop=True)
        sub = None
        if key is not None:
            import jax
            key, sub = jax.random.split(key)
        Q = chain.charges(st, parts, geom, cond, cfg, rng, key=sub, tabs=tabs, pmap=pmap)
        if not len(Q["time"]):
            continue
        D = chain.digis(Q, cond, cfg, rng, key=sub)
        if len(D["chan"]):
            out.append({k: np.asarray(D[k]) for k in ("chan", "starttdc", "adc")})
    if not out:
        return {"ev": np.zeros(0, np.int64), "bar": np.zeros(0, np.int64), "sipm": np.zeros(0, np.int64),
                "starttdc": np.zeros(0, np.int64), "adc": np.zeros((0, 8), np.int16)}
    chan = np.concatenate([o["chan"] for o in out]).astype(np.int64)
    st = np.concatenate([o["starttdc"] for o in out]).astype(np.int64)
    L = max(o["adc"].shape[1] for o in out)
    adc = np.concatenate([np.pad(np.asarray(o["adc"]), ((0, 0), (0, L - o["adc"].shape[1]))) for o in out])
    ev, bar, sipm = _split_label(chan)
    o = np.lexsort((st, sipm, bar, ev))       # stock CrvDigi order within an event: by channel, then time
    return {"ev": ev[o], "bar": bar[o], "sipm": sipm[o], "starttdc": st[o], "adc": adc[o].astype(np.int16)}


if __name__ == "__main__":
    main()
