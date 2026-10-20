"""V-R4: the fast CRV chain (fast response + fast reco) against the stock chain from the same CrvSteps, per sector group.

Arms on identical inputs: a G4 CrvStep table (crvsteps.<shard>.root, CrvStepTableDumper) goes through the fast chain
(crvfastsim.chain: photons -> SiPM charges -> waveforms/digis -> pulses -> coincidence clusters, numpy or JAX), and the
stock art chain on the same G4 dts file was dumped with CrvRecoTableDumper (ref.<shard>.root, scripts/run_v2_arms.sh).
Both arms are summarised with the v2compare accumulators (PE per seq-0 pulse, bar rates, clusters per event, per
sector group EX/T1/T2/M), and judged with the V-R4 thresholds of docs/fastreco.md: binned PE ratio within +-3 %,
coincidences per event within 1 %, PE median within 2 % (2 sigma allowed each). The EX layer-efficiency row needs
extrapolated tracks and is reported as v2compare's truth-crossing diagnostic only.

Rates are per event of the reference art job; the fast arm counts the same events (the steps of an event the art job
saw), so both share one denominator.
"""
import numpy as np

from .. import chain, geometry, tables
from ..analyses import v2compare as V2

THRESH = dict(V2.THRESH, pe_ratio_band=0.03)   # V-R4: response observables binned ratio within +-3 %


def fast_arm(step_path, cond, cfg, rng, tabs, pmap, geom, ccfg, max_events=None, key=None, timers=None,
             batch_events=None):
    """Run the fast chain on one CrvStep table, in batches of whole events (bounds the memory of the JAX SiPM stage);
    returns flat pulses (ev, bar, seq, pe), cluster sector types and cluster event keys."""
    steps = tables.read(step_path, "steps")
    parts = tables.read(step_path, "particles", tables.KEY + ["key", "pdg"])
    evk = chain.event_key(steps).astype(np.int64)
    uev = np.unique(evk)
    if max_events:
        uev = uev[:int(max_events)]
    batch_events = batch_events or (250 if cfg.jax else len(uev))
    P, T, E = [], [], []
    for b in range(0, len(uev), batch_events):
        sel = np.isin(evk, uev[b:b + batch_events])
        st = steps[sel].reset_index(drop=True)
        if key is not None:
            import jax
            key, sub = jax.random.split(key)
        else:
            sub = None
        Q = chain.charges(st, parts, geom, cond, cfg, rng, key=sub, tabs=tabs, pmap=pmap, timers=timers)
        if not len(Q["time"]):
            continue
        D = chain.digis(Q, cond, cfg, rng, key=sub, timers=timers)
        if not len(D["chan"]):
            continue
        W = chain.reco_waveforms(D, cond)
        W["toff"] = cond.timeoffset[W["chan"]]       # DB time offsets: chain_jax adds them (numpy path does itself)
        P.append(_pulses(W, cond, cfg, timers))
        outs, _ = chain.clusters(W, cond, geom, ccfg, cfg, timers=timers)
        for o in outs:
            T.append(np.asarray(o["cl_type"]).astype(int))
            E.append(np.asarray(o["cl_event"]).astype(np.int64))
    cat = lambda L, dt: np.concatenate(L) if L else np.zeros(0, dt)
    pulses = {k: cat([p[k] for p in P], np.float64 if k == "pe" else np.int64) for k in ("ev", "bar", "seq", "pe")}
    return pulses, cat(T, int), cat(E, np.int64)


def _pulses(W, cond, cfg, timers):
    """Pulses of the fast arm (the numpy reference reco; with cfg.jax the clusters come from chain_jax, whose pulses
    equal these to fit tolerance, V-R2)."""
    from . import pulsefit
    P = pulsefit.reconstruct(W["adc"], W["n"], W["start"], W["ped"], W["area"], W["height"], pulsefit.RecoConfig())
    return {"ev": W["ev"][P["wf"]].astype(np.int64), "bar": W["bar"][P["wf"]].astype(np.int64), "seq": P["seq"],
            "pe": P["pe"].astype(float)}


def fill_fast(A, pulses, cl_type, n_events, geom):
    """v2compare's arm accumulator from the fast arm's flat pulses and clusters."""
    A["events"] += int(n_events)
    A["files"] += 1
    bar, pe, s0 = pulses["bar"], pulses["pe"], pulses["seq"] == 0
    hb = s0 & (pe >= V2.RATE_PE)
    eb = np.unique(np.stack([pulses["ev"][hb], bar[hb]], 1), axis=0) if hb.any() else np.zeros((0, 2), np.int64)
    A["bar_rate"] = V2._add(A["bar_rate"], np.bincount(eb[:, 1], minlength=V2.NBARS))
    sec = geom.sector[bar]
    for g, secs in V2.GROUPS.items():
        G = A["groups"][g]
        m = s0 & np.isin(sec, secs)
        G["npulse"] += int(m.sum())
        G["h_pe"] = V2._add(G["h_pe"], np.histogram(pe[m], bins=V2.PE_EDGES)[0])
        G["nclus"] += int((cl_type == V2.SECTOR_TYPE[g]).sum())


def summarise(S):
    """v2compare.summarise with the V-R4 thresholds."""
    old = V2.THRESH
    V2.THRESH = THRESH
    try:
        return V2.summarise(S)
    finally:
        V2.THRESH = old


def run(steps, refs, calib, outdir, deviation=None, jax=False, max_events=None, seed=1, counters=None, emit=None,
        noise=True):
    """steps / refs: matching lists (same shards, same order) of CrvStep tables and stock CrvRecoTableDumper files."""
    import json
    import os
    import time
    from ..response import photons as PH
    from ..response import sipm as SI
    steps, refs = tables.expand(steps), tables.expand(refs)
    if len(steps) != len(refs):
        raise SystemExit(f"{len(steps)} step tables but {len(refs)} reference files")
    cpath = geometry.find_counters(counters or V2.DEFAULT_COUNTERS)
    geom = geometry.from_offline_config(cpath)
    ccfg = geometry.parse_simpleconfig(open(cpath).read())
    cond = chain.Conditions.from_csv(calib, deviation)
    cfg = chain.ChainConfig(jax=bool(jax), noise=bool(noise))
    tabs = PH.load_tables(cfg.response)
    key = None
    if jax:
        import jax as J
        from ..response import photons_jax
        tabs = {k: photons_jax.DeviceTable(v) for k, v in tabs.items()}
        key = J.random.PRNGKey(int(seed))
    pmap = SI.photon_map()
    S = V2._new_state()
    T, per_shard = {}, []
    for i, (sp, rp) in enumerate(zip(steps, refs)):
        rng = np.random.default_rng([int(seed), i])
        t0 = time.perf_counter()
        ref_ev = V2.reftables.event_key(V2.reftables.read([rp], "events", ["run", "subrun", "event"]))
        pulses, cl_type, cl_event = fast_arm(sp, cond, cfg, rng, tabs, pmap, geom, ccfg, max_events, key, T)
        # the reference art job's events define the universe; the fast chain keys events as (subrun << 20) | event
        ref_short = ref_ev & ((1 << 40) - 1)
        if max_events:
            ref_short = np.unique(ref_short)[:int(max_events)]
        inref = np.isin(pulses["ev"], ref_short)
        pulses = {k: v[inref] for k, v in pulses.items()}
        cm = np.isin(cl_event, ref_short)
        fill_fast(S["fast"], pulses, cl_type[cm], len(ref_short), geom)
        if max_events:
            _fill_ref_subset(S["ref"], rp, ref_short, geom)
        else:
            V2._fill_arm(S["ref"], [rp], None, geom)
        per_shard.append({"steps": sp, "ref": rp, "events": int(len(ref_short)), "wall_s": time.perf_counter() - t0})
    S["meta"] = {"shards": per_shard, "timers": T, "jax": bool(jax), "noise": bool(noise)}
    if emit:
        os.makedirs(os.path.dirname(emit) or ".", exist_ok=True)
        json.dump(S, open(emit, "w"), default=float)
        print(f"wrote {emit}")
        return S
    return report(S, outdir)


def _fill_ref_subset(A, rp, keys, geom):
    """Reference arm restricted to the first N events (--max-events): v2compare._fill_arm on a filtered read."""
    import awkward as ak
    p = V2.reftables.read([rp], "pulses", ["run", "subrun", "event", "bar", "seq", "pe"])
    c = V2.reftables.read([rp], "clusters", ["run", "subrun", "event", "sectortype"])
    pk = V2.reftables.event_key(p) & ((1 << 40) - 1)
    ck = V2.reftables.event_key(c) & ((1 << 40) - 1)
    pm, cm = np.isin(pk, keys), np.isin(ck, keys)
    pulses = {"ev": pk[pm], "bar": ak.to_numpy(p.bar)[pm].astype(np.int64), "seq": ak.to_numpy(p.seq)[pm],
              "pe": ak.to_numpy(p.pe)[pm].astype(float)}
    fill_fast(A, pulses, ak.to_numpy(c.sectortype)[cm].astype(int), len(keys), geom)


def merge(paths):
    import json
    S = V2._new_state()
    meta = []
    for f in tables.expand(paths):
        d = json.load(open(f))
        meta.append(d.pop("meta", {}))
        V2._merge_into(S, d)
    S["meta"] = {"batches": meta}
    return S


def report(S, outdir):
    from mu2efastsim import plots
    meta = S.pop("meta", {})
    summ = summarise(S)
    summ["vr4_thresholds"] = THRESH
    summ["meta"] = meta
    plots.write_json(outdir, "vr4_stats.json", summ)
    br = summ["bar_rate"]
    print(f"bars: chi2/ndf {br['chi2_ndf']:.2f}  max|pull| {br['max_abs_pull']:.1f} | {'ok' if br['pass'] else 'FAIL'}")
    for g, r in summ["groups"].items():
        flags = " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items())
        print(f"{g:3s} pulses/ev {r['pulses_per_event_rel']:+.4f}  PEmed {r['pe_median_ref']:.2f}->"
              f"{r['pe_median_fast']:.2f} ({r['pe_median_rel']:+.4f}+-{r['pe_median_rel_err']:.4f})  "
              f"band {r['pe_ratio']['max_abs_dev_minus_2sigma']}  clus/ev {r['clusters_per_event_rel']:+.4f}"
              f"+-{r['clusters_per_event_rel_err']:.4f} | {flags}")
    print(f"V-R4 {'PASS' if summ['pass'] else 'FAIL'} over {summ['events_ref']:,} events")
    return summ
