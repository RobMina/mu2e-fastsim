"""Beam-only deadtime frames for method F using compact full-step Cat libraries and fast response/reco.

Historical default: legacy_unconditional_v1, a floating, truncated unconditional lognormal.
It is not a CE-conditioned exposure: Offline v13_39_00/v13_41_00 classifies
mu2eCeMinusLeadingLog as beam-based. The opt-in ce_conditioned_v1 API uses
x * lognormal(x), implemented by shifting logmean by sigma^2, then llrint rounding.
The response CLI remains legacy-only; CE replay export has a separate library-validation gate.

Per event with the retained Run 1A parameters:
  * PBI extendedMean 1.58e7, SDF 0.8 -> sigma = sqrt(-ln SDF), strict cutMax 9.48e7
  * per mixer: n_frames ~ Poisson(N * prod(SimEfficiencies of its simStageEfficiencyTags) * prod(meanEventsPerPOTFactors))
    (Sim_best v1_3 SimEfficiencies2, runs 1430-1439, cid 207)
  * frames drawn uniformly from the mixer's pool (stepless frames are counted from the table metadata and contribute
    nothing), their CrvSteps merged; step times as stored (the *Cat stage times), then the response's microbunch wrap
    w.r.t. the event-window start -pbtime, pbtime uniform in {-224.46, ..., -204.46} ns (5 values, as the production).

    python -m crvfastsim.veto.mixframes --tables DIR --events N --seed S --conditions crvreco_mcs.root
        --photon-dev CRVPhoton.txt [--intensity-scale 1.0] --out frames.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from .. import chain
from . import fastframes as FF
from .exposure import CE_CONDITIONED, LEGACY, PBI, ExposureContract, llrint_pbi

MIXERS = {   # name: (product of SimEfficiencies, extra factor) for the CeMLeadingLogMix1BB.MDC2025au mixers
    "EleBeamFlashCat": (0.05532579 * 0.0001061374, 1.0),
    "MuBeamFlashCat": (0.00213816 * 0.00512831825, 1.0),
    "MuStopPileupCat": (0.00213816 * 0.000358773 * 0.0186933071614323, 1000.0),
    "NeutralsFlashCat": (0.53426053 * 0.0000135628358208955, 1.0),
}
PBTIMES = np.array([-224.46, -219.46, -214.46, -209.46, -204.46])


def draw_intensity(rng, n, extended_mean=PBI["extended_mean"], sdf=PBI["sdf"], cut_max=PBI["cut_max"], *, mode=LEGACY):
    """Raw PBI draw; the default preserves the historical RNG sequence exactly."""
    return ExposureContract(mode, extended_mean, sdf, cut_max).draw_raw(rng, n)


class Pool:
    """One mixer's frames: the stepped frames' steps (grouped) and the count of stepless frames."""

    def __init__(self, files):
        from .catsteps import read_meta
        parts, self.frames, self.stepped = [], 0, 0
        for i, f in enumerate(files):
            m = read_meta(f)
            if "schema" in m:
                raise ValueError("legacy Pool refuses versioned Cat tables; use the explicit validated pool path")
            self.frames += m["frames"]
            d = pd.read_parquet(f)
            d["fid"] = (np.int64(i) << 40) | (d.subrun.to_numpy().astype(np.int64) << 20) | d.event.to_numpy().astype(np.int64)
            parts.append(d)
        self.steps = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        fid = self.steps.fid.to_numpy() if len(self.steps) else np.zeros(0, np.int64)
        o = np.argsort(fid, kind="stable")
        self.steps = self.steps.iloc[o].reset_index(drop=True)
        self.ids, self.start, self.count = np.unique(fid[o], return_index=True, return_counts=True)
        self.stepped = len(self.ids)


def mix(pools, n_events, rng, intensity_scale=1.0, *, return_events=False, exposure_mode=LEGACY):
    """Mixed CrvSteps for n_events beam-only events: DataFrame in the steps layout (run, subrun, event, ..., pdg)
    with event = 1..n_events, subrun 0, and the per-event window start w = -pbtime."""
    contract = ExposureContract(mode=exposure_mode, intensity_scale=intensity_scale)
    if isinstance(n_events, (bool, np.bool_)) or not isinstance(n_events, (int, np.integer)) or n_events <= 0:
        raise ValueError("n_events must be a positive integer")
    raw = contract.draw_raw(rng, n_events)
    N = llrint_pbi(raw) if exposure_mode == CE_CONDITIONED else raw
    pb = rng.choice(PBTIMES, n_events)
    rows, mean_frames, species_counts = [], {}, {}
    nsp = 0
    for name, pool in pools.items():
        eff, fac = MIXERS[name]
        nf = rng.poisson(N * eff * fac)                          # frames per event from this mixer
        mean_frames[name] = float(nf.mean())
        if exposure_mode == CE_CONDITIONED:
            species_counts[f"n_frames_{name}"] = nf
        if pool.stepped == 0 or nf.sum() == 0:
            continue
        k = rng.binomial(nf, pool.stepped / pool.frames)        # of those, frames with CRV steps (per event, exact)
        ev = np.repeat(np.arange(n_events), k)
        hit = rng.integers(0, pool.stepped, len(ev))
        if not len(hit):
            continue
        cnt = pool.count[hit]
        idx = np.repeat(pool.start[hit] - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
        d = pool.steps.iloc[idx].copy()
        d["event"] = (np.repeat(ev, cnt) + 1).astype(np.int64)
        d["subrun"] = 0
        d["run"] = 1430
        # unique SimParticle key per drawn frame (frames from different files/mixers share keys)
        d["sp"] = np.where(d.sp.to_numpy() >= 0, np.repeat(nsp + np.arange(len(hit)) * 100000, cnt) + d.sp.to_numpy(), -1)
        nsp += len(hit) * 100000
        rows.append(d)
    st = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    keys = np.arange(1, n_events + 1, dtype=np.int64)          # event key = (subrun 0 << 20) | event
    result = (st, keys, -pb, {"mean_frames_per_event": mean_frames, "mean_intensity": float(N.mean())})
    if exposure_mode == CE_CONDITIONED:
        result[3]["exposure_contract"] = contract.record()
    if return_events:
        events = pd.DataFrame({"run": np.full(n_events, 1430, np.uint32),
                               "subrun": np.zeros(n_events, np.uint32), "event": keys.astype(np.uint32),
                               "pbtime": pb, "intensity": N})
        if exposure_mode == CE_CONDITIONED:
            events["intensity_raw"] = raw
            events["intensity_rounded"] = N
            events["intensity"] = N.astype(np.float64)
            for name, counts in species_counts.items():
                events[name] = counts
        return (*result, events)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", required=True, help="directory with <Mixer>/steps.*.parquet (catsteps outputs)")
    ap.add_argument("--events", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--photon-dev")
    ap.add_argument("--run-config", default="run1a_v01", choices=sorted(FF.RUN_CONFIGS))
    ap.add_argument("--intensity-scale", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    pools = {m: Pool(sorted(glob.glob(os.path.join(a.tables, m, "*.parquet")))) for m in MIXERS}
    info = {m: {"files": len(glob.glob(os.path.join(a.tables, m, "*.parquet"))), "frames": p.frames, "stepped": p.stepped}
            for m, p in pools.items()}
    st, keys, w, minfo = mix(pools, a.events, rng, a.intensity_scale)
    T = {"setup_mix_s": time.perf_counter() - t0, "steps": int(len(st))}
    cond = FF.conditions_from_dump(a.conditions, a.photon_dev)
    pa = st[["run", "subrun", "event", "sp", "pdg"]].rename(columns={"sp": "key"}).drop_duplicates(["run", "subrun", "event", "key"])
    st, _ = FF.fold_steps(st, keys, w)
    fr = FF.frames_from_steps(st, pa, cond, a.run_config, keys, keys, w, seed=a.seed, chunk=a.chunk, timers=T, run_no=1430)
    import awkward as ak
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ak.to_parquet(fr, a.out)
    nc = ak.to_numpy(ak.num(fr.coincs.time))
    rep = {"events": a.events, "seed": a.seed, "pools": info, **minfo, "frames": int(len(fr)),
           "frac_frames_with_coinc": float((nc > 0).mean()), "coincs": int(nc.sum()), "timers": T,
           "wall_s": time.perf_counter() - t0}
    json.dump(rep, open(a.out.replace(".parquet", ".json"), "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
