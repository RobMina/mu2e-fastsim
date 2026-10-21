"""R3: one Mu2eBO point end to end (response + waveforms + reco + objective) on cached CrvSteps, timed per stage.

One process per accelerator (the node script sets the visible device). The cached G4 CrvStep tables
(CrvStepTableDumper, e.g. prod_as_tables/crvsteps.*.root) are read once (pdg joined at load), then replayed with fresh
seeds until `target` events with CRV steps have been processed (the R3 unit: an event with at least one CrvStep).
Replays differ only in their random numbers; the cost per event depends on the steps, not on the event being new.

Chain per batch of whole events: response (JAX: fused device photons -> SiPM charges, response/device.py; numpy:
chain.charges) -> waveforms/digis (chain.digis) -> reco waveforms (chain.reco_waveforms) -> pulses + clusters
(chain.clusters) -> objective histograms (crvextrap.analyses.crvresponse on reco.ntview, if crvextrap is importable
and --objective is given). The output JSON has the per-stage times, the per-pass walls (pass 0 includes compilation),
the events, and the response accumulator for a later merge against data.

Timers: setup_s (geometry, conditions, lookup tables), io_s (read tables, pdg join), response_s, digis_s, group_s
(digis -> reco waveforms), reco_s (pulses + coincidences), view_s + objective_s, wall_s. Host glue inside a stage is
in that stage.
"""
import json
import os
import time

import numpy as np

from .. import chain, geometry, tables

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def load_cache(paths, timers):
    """Step tables -> list of (steps sorted by event key, pdg per step, event key per step, events)."""
    t0 = time.perf_counter()
    out = []
    for p in tables.expand(paths):
        st = tables.read(p, "steps")
        pa = tables.read(p, "particles", tables.KEY + ["key", "pdg"])
        pdg = st.merge(pa.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], how="left").pdg.fillna(0).to_numpy()
        ek = chain.event_key(st).astype(np.int64)
        o = np.argsort(ek, kind="stable")
        st, pdg, ek = st.iloc[o].reset_index(drop=True), pdg[o], ek[o]
        out.append(dict(path=p, steps=st, pdg=pdg, ek=ek, events=np.unique(ek)))
    timers["io_s"] = timers.get("io_s", 0.0) + time.perf_counter() - t0
    return out


def _add(T, k, t0):
    T[k] = T.get(k, 0.0) + time.perf_counter() - t0


def run_batch(c, a, b, cond, cfg, geom, ccfg, tabs, pmap, rng, key, T, objective=None, window_ns=1e5):
    """Events [a, b) of the step-sorted table c through the chain. Returns events processed."""
    lo = np.searchsorted(c["ek"], c["events"][a])
    hi = np.searchsorted(c["ek"], c["events"][b - 1], side="right")
    st, pdg, ek = c["steps"].iloc[lo:hi].reset_index(drop=True), c["pdg"][lo:hi], c["ek"][lo:hi]
    t0 = time.perf_counter()
    if cfg.jax:
        import jax
        from ..response import device as DV
        key, k_r, k_w = jax.random.split(key, 3)
        rs = DV.response_batched(st, pdg, geom, cfg.response, cfg.sipm, tabs, cond.deviation, cond.status, ek, rng,
                                 k_r, pmap)
        if any(bool(r["overflow"]) for r in rs):
            T["response_overflow_batches"] = T.get("response_overflow_batches", 0) + 1
        Q = DV.to_host_batches(rs)
        del rs
    else:
        k_w = None
        Q = chain.charges(st, None, geom, cond, cfg, rng, tabs=tabs, pmap=pmap, pdg=pdg)
    _add(T, "response_s", t0)
    T["charges"] = T.get("charges", 0) + len(Q["time"])
    if not len(Q["time"]):
        return b - a, key
    t0 = time.perf_counter()
    D = chain.digis(Q, cond, cfg, rng, key=k_w)
    _add(T, "digis_s", t0)
    if not len(D["chan"]):
        return b - a, key
    t0 = time.perf_counter()
    W = chain.reco_waveforms(D, cond)
    W["toff"] = cond.timeoffset[W["chan"]]
    _add(T, "group_s", t0)
    T["waveforms"] = T.get("waveforms", 0) + len(W["ev"])
    t0 = time.perf_counter()
    outs, _ = chain.clusters(W, cond, geom, ccfg, cfg, timers=T.setdefault("reco_detail", {}),
                             detail=objective is not None)
    _add(T, "reco_s", t0)
    T["clusters"] = T.get("clusters", 0) + int(sum(len(o["cl_event"]) for o in outs))
    if objective is not None:
        from . import ntview
        t0 = time.perf_counter()
        cc, pp, _ = ntview.build(outs, geom, events=c["events"][a:b])
        _add(T, "view_s", t0)
        t0 = time.perf_counter()
        objective["mod"].accumulate_arrays(cc, pp, objective["state"], window_ns)
        _add(T, "objective_s", t0)
    return b - a, key


def run(steps, calib, deviation=None, target=1_000_000, jax=False, seed=1, batch_events=2000, counters=None,
        emit=None, objective=False, noise=True, max_passes=1000, time_limit=None, v05=False):
    """v05: the crvonly_v05 digitisation settings (one FEB clock phase per event, CRVTime net zero), as reco-vr4eff,
    for comparisons with the v05 EventNtuples."""
    T = {}
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    cpath = geometry.find_counters(counters or DEFAULT_COUNTERS)
    geom = geometry.from_offline_config(cpath)
    ccfg = geometry.parse_simpleconfig(open(cpath).read())
    cond = chain.Conditions.from_csv(calib, deviation)
    if v05:
        import dataclasses
        cond = dataclasses.replace(cond, roc=np.zeros_like(cond.roc), feb=np.zeros_like(cond.feb),
                                   timeoffset=np.zeros_like(cond.timeoffset))
    cfg = chain.ChainConfig(jax=bool(jax), noise=bool(noise))
    from ..response import photons as PH
    from ..response import sipm as SI
    tabs = PH.load_tables(cfg.response)
    key = None
    if jax:
        import jax as J
        from ..response import photons_jax
        tabs = {k: photons_jax.DeviceTable(v) for k, v in tabs.items()}
        key = J.random.PRNGKey(int(seed))
    pmap = SI.photon_map()
    obj = None
    if objective:
        from crvextrap.analyses import crvresponse
        obj = {"mod": crvresponse, "state": crvresponse.new_state()}
    _add(T, "setup_s", t0)
    cache = load_cache(steps, T)
    per_pass, done, p = [], 0, 0
    while done < target and p < max_passes:
        tp = time.perf_counter()
        n_pass = 0
        for i, c in enumerate(cache):
            rng = np.random.default_rng([int(seed), p, i])
            ne = len(c["events"])
            for a in range(0, ne, batch_events):
                b = min(ne, a + batch_events, a + (target - done))
                n, key = run_batch(c, a, b, cond, cfg, geom, ccfg, tabs, pmap, rng, key, T, obj,
                                   window_ns=cfg.window[1] - cfg.window[0])
                done += n
                n_pass += n
                if done >= target or (time_limit and time.perf_counter() - t_all > time_limit):
                    break
            if done >= target or (time_limit and time.perf_counter() - t_all > time_limit):
                break
        per_pass.append({"pass": p, "events": n_pass, "wall_s": time.perf_counter() - tp})
        p += 1
        if time_limit and time.perf_counter() - t_all > time_limit:
            break
    wall = time.perf_counter() - t_all
    warm = [x for x in per_pass[1:] if x["events"]]
    res = {"events_with_steps": done, "target": target, "wall_s": wall, "passes": per_pass,
           "events_per_s": done / wall if wall else 0.0,
           "warm_events_per_s": (sum(x["events"] for x in warm) / sum(x["wall_s"] for x in warm)) if warm else None,
           "cache_files": [c["path"] for c in cache], "cache_events": int(sum(len(c["events"]) for c in cache)),
           "timers": T, "jax": bool(jax), "noise": bool(noise), "v05": bool(v05), "batch_events": batch_events, "seed": seed,
           "ms_per_event": {k[:-2]: 1e3 * v / max(done, 1) for k, v in T.items()
                            if isinstance(v, float) and k.endswith("_s")}}
    if jax:
        import jax as J
        res["device"] = str(J.devices()[0])
    if obj is not None:
        res["response_state"] = obj["state"]
    if emit:
        os.makedirs(os.path.dirname(emit) or ".", exist_ok=True)
        json.dump(res, open(emit, "w"), default=float)
    short = {k: res[k] for k in ("events_with_steps", "wall_s", "events_per_s", "warm_events_per_s")}
    print(json.dumps(short), "\nms/event", json.dumps({k: round(v, 3) for k, v in res["ms_per_event"].items()}))
    return res


def merge(paths, wall=None):
    """Node total from the per-process JSONs: events, per-job wall (given, or the slowest process), node rate,
    projected time for 1 M events, merged response accumulator."""
    J = [json.load(open(f)) for f in tables.expand(paths)]
    ev = sum(j["events_with_steps"] for j in J)
    slow = max(j["wall_s"] for j in J)
    w = float(wall) if wall else slow
    warm = [j["warm_events_per_s"] for j in J if j.get("warm_events_per_s")]
    out = {"processes": len(J), "events_with_steps": ev, "wall_s": w, "slowest_process_s": slow,
           "node_events_per_s": ev / w if w else 0.0, "node_warm_events_per_s": float(sum(warm)) if warm else None,
           "minutes_for_1M": (1e6 / (ev / w)) / 60.0 if ev and w else None,
           "devices": sorted({str(j.get("device")) for j in J})}
    states = [j["response_state"] for j in J if "response_state" in j]
    if states:
        from crvextrap.analyses import crvresponse
        out["response_state"] = crvresponse.merge_states(states)
    return out
