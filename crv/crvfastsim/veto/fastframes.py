"""Deadtime frames from the fast chain (method F with the fast CRV response and reco, plan E4): mixed CrvSteps of an
on-spill digitisation job -> fast photons, SiPM charges, digis, pulses, coincidences, in the on-spill time frame of
the stock job, written as a frames parquet for crvfastsim.veto.deadtime --frames.

On-spill frame (CrvPhotonGenerator, CrvWaveformsGenerator, CrvRecoPulsesFinder): photon times are wrapped into one
microbunch [w, w + 1695 ns) with w = -pbtime (the event's ProtonBunchTimeMC), TDC0 = w + uniform(0, 12.5) per FEB,
digis start inside [w + 200, w + 1500] ns, and pulse times are reported relative to TDC0time = w. Here the steps are
shifted by -w before the response (fold to [0, 1695)), the chain runs with the digitisation window (200, 1500), and
w is added back to the coincidence times, as CrvRecoPulsesFinder adds TDC0time = -pbtime (reco frame: time since the
event-window marker, the frame of the track t0).

    python -m crvfastsim.veto.fastframes mixsteps.root [...] --conditions crvreco_mcs.root --photon-dev CRVPhoton.txt
        --pbtime nts.root [...] [--run-config run1a_v01] [--jax] --out frames.parquet
Inputs: mixsteps.root from CrvFastSim/fcl/dump_crvsteps_mix.fcl on the dig files; crvreco_mcs.root from
CrvFastSim/fcl/dump_crvreco_from_mcs_run1a.fcl (CRVCalib/Status/Ordinal and pbtime per event); the EventNtuple of the
same events for the pbtime of every event (evtinfomc.pbtime), or the mcs dump if it covers all events.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time

import awkward as ak
import numpy as np

from .. import chain, geometry, tables
from ..reco import coinc as C
from ..response import photons as PH
from ..response import sipm as SI

DR_PERIOD = 1695.0      # physicsParams.nominalDRPeriod (globalConstants_01.txt)

# Run 1A (Offline v13_33_00, MDC2025au): CRVResponse/fcl/epilog_run1a_v01.fcl, CRVReco/fcl/epilog_run1a_v01.fcl
RUN_CONFIGS = {
    "run1a_v01": dict(
        counters="Offline/Mu2eG4/geom/crv_counters_run1a_v01.txt",
        sectors={"T1": PH.SectorConfig("6000_0", 0, 39794.0), "T2": PH.SectorConfig("6000_0", 0, 30482.0)},
        photon_yield_scale=0.90,
        cuts={n: C.SectorCuts(8, 20, 20, -6, 6, 3, 3, 0, 250, 0) for n in ("T1", "T2")},
        ignore_secondary_pulses=False,
        window=(200.0, 1500.0)),
}


def run_config(name):
    """(ChainConfig, FinderConfig, counters path) of a named stock run configuration."""
    rc = RUN_CONFIGS[name]
    resp = dataclasses.replace(PH.ResponseConfig(), sectors=dict(rc["sectors"]), photon_yield_scale=rc["photon_yield_scale"])
    fcfg = dataclasses.replace(C.FinderConfig(), cuts=dict(rc["cuts"]), ignore_secondary_pulses=rc["ignore_secondary_pulses"])
    return resp, fcfg, rc["counters"], rc["window"]


def conditions_from_dump(path, photon_dev=None):
    """chain.Conditions from a CrvRecoTableDumper calib tree (+ a dbTool CRVPhoton print-content table)."""
    from ..reco import reftables
    c = reftables.calib([path])
    n = len(c["status"])
    dev = np.zeros(n)
    if photon_dev:
        a = np.loadtxt(photon_dev, delimiter=",", comments=("#", "TABLE"))
        m = a[:, 0].astype(int) < n
        dev[a[m, 0].astype(int)] = a[m, 1]
    return chain.Conditions(c["pedestal"], c["pulsearea"], c["pulseheight"], c["timeoffset"], c["status"], c["roc"],
                            c["feb"], dev)


def read_pbtime(paths):
    """{event key (subrun << 20 | event): -pbtime} from EventNtuples (evtinfomc.pbtime) or CrvRecoTableDumper files."""
    import uproot
    keys, w = [], []
    for p in paths:
        with uproot.open(p) as f:
            if "EventNtuple/ntuple" in f:
                t = f["EventNtuple/ntuple"]
                s, e = t["evtinfo/subrun"].array(library="np"), t["evtinfo/event"].array(library="np")
                pb = t["evtinfomc/pbtime"].array(library="np")
            else:
                a = f["crvdump/events"].arrays(["subrun", "event", "pbtime"], library="np")
                s, e, pb = a["subrun"], a["event"], a["pbtime"]
        keys.append((s.astype(np.int64) << 20) | e.astype(np.int64))
        w.append(-pb.astype(float))
    return np.concatenate(keys), np.concatenate(w)


def fold_steps(st, key_pb, w_pb):
    """Shift every step by -w (its event's window start) and wrap into [0, DR_PERIOD); steps of events without a
    pbtime are dropped (returned count)."""
    k = chain.event_key(st)
    o = np.argsort(key_pb)
    i = np.clip(np.searchsorted(key_pb[o], k), 0, len(o) - 1)
    ok = key_pb[o][i] == k
    w = w_pb[o][i]
    t0 = st.t0.to_numpy() - w
    sh = np.floor(t0 / DR_PERIOD) * DR_PERIOD                 # wrap by the start time; the end time moves with it
    st = st.assign(t0=t0 - sh, t1=st.t1.to_numpy() - w - sh)
    return st[ok].reset_index(drop=True), int((~ok).sum())


def _setup(run, jax=False):
    resp, fcfg, counters, window = run_config(run)
    cpath = geometry.find_counters(counters)
    geom = geometry.from_offline_config(cpath)
    ccfg = geometry.parse_simpleconfig(open(cpath).read())
    cfg = chain.ChainConfig(response=resp, window=window, jax=bool(jax))
    tabs = PH.load_tables(resp)
    if jax:
        from ..response import photons_jax
        tabs = {k: photons_jax.DeviceTable(v) for k, v in tabs.items()}
    return dict(geom=geom, cfg=cfg, fcfg=fcfg, tabs=tabs, pmap=SI.photon_map(), G=C.sector_geom(geom, ccfg))


def frames_from_steps(st, pa, cond, run, allk, key_pb, w_pb, seed=1, chunk=500, timers=None, setup=None, run_no=None,
                      pulse_sink=None):
    """Fast coincidences per event for steps already folded into [0, 1695) ns (fold_steps): awkward record per event of
    allk (every event is a frame, with or without steps); coincidence times back in the reco frame (+ w)."""
    S = setup or _setup(run)
    T = timers if timers is not None else {}
    rng = np.random.default_rng(seed)
    sk = chain.event_key(st) if len(st) else np.zeros(0, np.int64)
    ue = np.unique(sk)
    cls = []
    for a in range(0, len(ue), chunk):                      # memory: ~1 GB per 1000 mixed events on the CPU path
        sub = st[np.isin(sk, ue[a:a + chunk])].reset_index(drop=True)
        pdg = sub.pdg.to_numpy() if "pdg" in sub.columns else None    # compact tables carry pdg per step
        Q = chain.charges(sub, pa, S["geom"], cond, S["cfg"], rng, tabs=S["tabs"], pmap=S["pmap"], timers=T, pdg=pdg)
        if not len(Q["time"]):
            continue
        D = chain.digis(Q, cond, S["cfg"], rng, timers=T)
        if not len(D["chan"]):
            continue
        cls.append(_clusters(chain.reco_waveforms(D, cond), cond, S["G"], S["fcfg"], T, pulse_sink))
    cl = {k: np.concatenate([c[k] for c in cls]) for k in cls[0]} if cls else \
        {k: np.zeros(0) for k in ("ev", "time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType")}
    o = np.argsort(key_pb)
    w = w_pb[o][np.clip(np.searchsorted(key_pb[o], cl["ev"]), 0, len(o) - 1)] if len(cl["ev"]) else np.zeros(0)
    for k in ("time", "timeStart", "timeEnd"):
        cl[k] = cl[k] + w
    if run_no is None:
        run_no = int(st.run.iloc[0]) if len(st) else 0
    return _as_frames(np.union1d(allk, sk), cl, run_no)


def frames(paths, cond, run, jax=False, seed=1, key_pb=None, w_pb=None, max_events=None, timers=None, chunk=1000):
    """Fast coincidences per event of the given mixed-step files, as an awkward record per event (run, subrun, event,
    coincs{time, timeStart, timeEnd, PEs, nLayers, sectorType}). Every event of the file's subruns that has a pbtime
    is a frame, with or without CRV steps (events without steps are frames with no coincidence)."""
    S = _setup(run, jax)
    T = timers if timers is not None else {}
    out = []
    for i, f in enumerate(tables.expand(paths)):
        st = tables.read(f, "steps")
        pa = tables.read(f, "particles")
        run_no = int(st.run.iloc[0]) if len(st) else 0
        allk = key_pb[np.isin(key_pb >> 20, st.subrun.unique())]
        if max_events:
            allk = np.sort(allk)[:int(max_events)]
            st = st[np.isin(chain.event_key(st), allk)].reset_index(drop=True)
        st, lost = fold_steps(st, key_pb, w_pb)
        T["steps_without_pbtime"] = T.get("steps_without_pbtime", 0) + lost
        T["steps"] = T.get("steps", 0) + len(st)
        out.append(frames_from_steps(st, pa, cond, run, allk, key_pb, w_pb, seed=[int(seed), i], chunk=chunk,
                                     timers=T, setup=S, run_no=run_no))
    return ak.concatenate(out) if out else ak.Array([])


def _clusters(W, cond, G, fcfg, T, pulse_sink=None):
    """numpy reference reco (pulsefit + coinc.find_event) with the run configuration's finder settings."""
    from ..reco import pulsefit
    t0 = time.perf_counter()
    rcfg = pulsefit.RecoConfig()
    P = pulsefit.reconstruct(W["adc"], W["n"], W["start"], W["ped"], W["area"], W["height"], rcfg)
    P["time"] = P["time"] + cond.timeoffset[W["chan"]][P["wf"]] * rcfg.time_offset_scale
    wf = P["wf"]
    ev = W["ev"][wf]
    o = np.lexsort((P["seq"], wf, ev))
    ev, wf = ev[o], wf[o]
    idx = np.arange(len(ev)) - np.searchsorted(ev, ev)
    bar, sipm, tm, pe, seq = W["bar"][wf], W["sipm"][wf], P["time"][o], P["pe"][o], P["seq"][o]
    if pulse_sink is not None:
        pulse_sink.append(dict(ev=ev, chan=bar * 4 + sipm, time=tm, pe=pe, seq=seq))
    T["pulses_s"] = T.get("pulses_s", 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    rows = {k: [] for k in ("ev", "time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType")}
    ue, lo = np.unique(ev, return_index=True)
    hi = np.r_[lo[1:], len(ev)]
    for e, a, b in zip(ue, lo, hi):           # events with pulses only (~1 % of frames); the finder is per event
        p = dict(idx=idx[a:b], bar=bar[a:b], sipm=sipm[a:b], time=tm[a:b], pe=pe[a:b], seq=seq[a:b])
        for r in C.find_event(p, G, cond.status, fcfg):
            rows["ev"].append(e)
            rows["time"].append(r["avgtime"]); rows["timeStart"].append(r["start"]); rows["timeEnd"].append(r["end"])
            rows["PEs"].append(r["pe"]); rows["nLayers"].append(len(r["layers"]))
            rows["sectorType"].append(int(r["sectortype"]))
    T["coinc_s"] = T.get("coinc_s", 0.0) + time.perf_counter() - t0
    T["pulses"] = T.get("pulses", 0) + len(ev)
    return {k: np.asarray(v) for k, v in rows.items()}


def _as_frames(evk, cl, run_no):
    allk = np.union1d(evk, cl["ev"]).astype(np.int64)
    o = np.argsort(cl["ev"], kind="stable")
    cnt = np.searchsorted(cl["ev"][o], allk, "right") - np.searchsorted(cl["ev"][o], allk, "left")
    co = {k: ak.unflatten(np.asarray(cl[k])[o], cnt) for k in ("time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType")}
    return ak.zip({"run": np.full(len(allk), run_no, np.int64), "subrun": allk >> 20, "event": allk & ((1 << 20) - 1),
                   "coincs": ak.zip(co)}, depth_limit=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="mixsteps.root files (dump_crvsteps_mix.fcl)")
    ap.add_argument("--conditions", required=True, help="CrvRecoTableDumper file with the calib tree")
    ap.add_argument("--photon-dev", help="dbTool print-content of CRVPhoton (channel,photonYieldDeviation)")
    ap.add_argument("--pbtime", nargs="+", required=True, help="EventNtuple or CrvRecoTableDumper files with pbtime")
    ap.add_argument("--run-config", default="run1a_v01", choices=sorted(RUN_CONFIGS))
    ap.add_argument("--jax", action="store_true", help="JAX photons, SiPM and waveforms (reco stays numpy)")
    ap.add_argument("--max-events", type=int)
    ap.add_argument("--chunk", type=int, default=1000, help="events per response call")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cond = conditions_from_dump(a.conditions, a.photon_dev)
    kp, wp = read_pbtime(a.pbtime)
    T = {}
    t0 = time.perf_counter()
    fr = frames(a.inputs, cond, a.run_config, a.jax, a.seed, kp, wp, a.max_events, T, a.chunk)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ak.to_parquet(fr, a.out)
    n = len(fr)
    nc = int(ak.sum(ak.num(fr.coincs.time))) if n else 0
    print(f"wrote {a.out}: {n} frames, {nc} coincidences ({(ak.sum(ak.num(fr.coincs.time) > 0) / max(n, 1)):.4f} of frames); "
          f"{time.perf_counter() - t0:.1f} s; " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in T.items()))


if __name__ == "__main__":
    main()
