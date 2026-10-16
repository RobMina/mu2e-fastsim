"""census -- what reaches the CRV, by CrvStep origin (docs/plan.md section 3.3), from CrvStep tables.

Each CrvStep's SimParticle is traced to its primary through the parent chain (vectorised: repeated parent lookups over
all particles at once, bounded by the deepest chain) and classed as the primary muon itself, a descendant of a primary
muon, a non-muon primary, or a non-muon descendant. Reports visible-energy and step shares overall and per sector,
the non-muon share by primary pdg, steps and bars per event, and the fraction of events with muon-origin deposits only.
Batchable: the counts add.

    census crvsteps.*.root --outdir DIR
    census crvsteps.*.root --emit batch.json ;  census "batch*.json" --merge --outdir DIR
"""
import json
import os

import numpy as np
import pandas as pd

from mu2efastsim import plots

from .. import geometry, tables

CLASSES = ("mu_primary", "mu_descendant", "nonmu_primary", "nonmu_descendant")
SECTOR_GROUP = np.array(["EX", "T1", "T2"] + ["M"] * 8)
MAX_DEPTH = 1000
STEP_CAP = 2000  # steps-per-event histogram range; the mean is kept exact from steps_sum
DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def primary_of(parts):
    """For every particle row, the key and pdg of its primary ancestor (same event); -1 where the chain breaks."""
    p = parts[tables.KEY + ["key", "parent", "pdg"]].copy()
    p["uid"] = np.arange(len(p))
    lut = p.set_index(tables.KEY + ["key"]).uid
    par_uid = lut.reindex(pd.MultiIndex.from_frame(p[tables.KEY + ["parent"]])).to_numpy()
    par_uid = np.where(np.isnan(par_uid), -1, par_uid).astype(np.int64)
    cur = np.arange(len(p))
    for _ in range(MAX_DEPTH):
        nxt = par_uid[cur]
        step = nxt >= 0
        if not step.any():
            break
        cur = np.where(step, nxt, cur)
    is_root = par_uid[cur] < 0
    p["prim_key"] = np.where(is_root, p.key.to_numpy()[cur], -1)
    p["prim_pdg"] = np.where(is_root, p.pdg.to_numpy()[cur], 0)
    return p


def classify(steps, parts, sector_of_bar):
    p = primary_of(parts)
    s = steps.merge(p[tables.KEY + ["key", "prim_key", "prim_pdg"]].rename(columns={"key": "sp"}),
                    on=tables.KEY + ["sp"], how="left")
    mu = s.prim_pdg.abs() == 13
    own = s.sp == s.prim_key
    s["cls"] = np.select([mu & own, mu & ~own, ~mu & own, ~mu & ~own], CLASSES, default="unknown")
    s.loc[s.prim_key.isna() | (s.prim_key < 0), "cls"] = "unknown"
    s["group"] = SECTOR_GROUP[np.asarray(sector_of_bar)[s.bar.to_numpy()]]
    return s


def _new_state():
    return {"files": 0, "events_with_steps": 0, "events_mu_only": 0, "edep": {}, "nsteps": {},
            "edep_group": {}, "nonmu_edep_pdg": {}, "steps_per_event": [0] * (STEP_CAP + 1),
            "bars_per_event": [0] * 101, "steps_sum": 0}


def _add(d, ser):
    for k, v in ser.items():
        d[str(k)] = d.get(str(k), 0) + float(v)


def accumulate(S, path, sector_of_bar):
    steps = tables.read(path, "steps", tables.KEY + ["bar", "sp", "edep"])
    parts = tables.read(path, "particles", tables.KEY + ["key", "parent", "pdg"])
    s = classify(steps, parts, sector_of_bar)
    S["files"] += 1
    ev = s.groupby(tables.KEY, sort=False)
    S["events_with_steps"] += int(ev.ngroups)
    S["events_mu_only"] += int(s.cls.isin(CLASSES[:2]).groupby([s[k] for k in tables.KEY], sort=False).all().sum())
    _add(S["edep"], s.groupby("cls").edep.sum())
    _add(S["nsteps"], s.groupby("cls").size())
    for (g, c), v in s.groupby(["group", "cls"]).edep.sum().items():
        S["edep_group"].setdefault(g, {})
        S["edep_group"][g][c] = S["edep_group"][g].get(c, 0.0) + float(v)
    _add(S["nonmu_edep_pdg"], s[s.cls.str.startswith("nonmu")].groupby("prim_pdg").edep.sum())
    n = ev.size().to_numpy()
    S["steps_sum"] += int(n.sum())
    S["steps_per_event"] = (np.asarray(S["steps_per_event"])
                            + np.bincount(np.minimum(n, STEP_CAP), minlength=STEP_CAP + 1)).tolist()
    nb = np.minimum(ev.bar.nunique().to_numpy(), 100)
    S["bars_per_event"] = (np.asarray(S["bars_per_event"]) + np.bincount(nb, minlength=101)).tolist()


def _merge_into(tot, d):
    for k in ("files", "events_with_steps", "events_mu_only", "steps_sum"):
        tot[k] += d[k]
    for k in ("edep", "nsteps", "nonmu_edep_pdg"):
        for kk, v in d[k].items():
            tot[k][kk] = tot[k].get(kk, 0) + v
    for g, dd in d["edep_group"].items():
        for c, v in dd.items():
            tot["edep_group"].setdefault(g, {})
            tot["edep_group"][g][c] = tot["edep_group"][g].get(c, 0.0) + v
    for k in ("steps_per_event", "bars_per_event"):
        tot[k] = (np.asarray(tot[k]) + np.asarray(d[k])).tolist()


def _hist_quantile(h, q):
    c = np.cumsum(h)
    return int(np.searchsorted(c, q * c[-1])) if c[-1] else None


def report(S, outdir):
    tot = sum(S["edep"].values()) or 1.0
    hs, hb = np.asarray(S["steps_per_event"], float), np.asarray(S["bars_per_event"], float)
    kb = np.arange(len(hb))
    out = {
        "files": S["files"],
        "n_events_with_crvsteps": S["events_with_steps"],
        "frac_events_muon_origin_only": S["events_mu_only"] / max(S["events_with_steps"], 1),
        "visible_edep_fraction": {c: S["edep"].get(c, 0.0) / tot for c in CLASSES + ("unknown",)},
        "step_count": S["nsteps"],
        "visible_edep_fraction_by_sector": {g: {c: v / (sum(d.values()) or 1.0) for c, v in d.items()}
                                            for g, d in S["edep_group"].items()},
        "nonmu_edep_by_primary_pdg": {k: v / tot for k, v in
                                      sorted(S["nonmu_edep_pdg"].items(), key=lambda kv: -kv[1])[:10]},
        "crvsteps_per_event": {"mean": S["steps_sum"] / max(hs.sum(), 1), "median": _hist_quantile(hs, 0.5),
                               "p99": _hist_quantile(hs, 0.99)},
        "bars_per_event": {"mean": float((hb * kb).sum() / hb.sum()), "median": _hist_quantile(hb, 0.5)},
    }
    plots.write_json(outdir, "census_stats.json", out)
    f = out["visible_edep_fraction"]
    print(f"events with CRV steps {out['n_events_with_crvsteps']:,}; muon-origin only "
          f"{100 * out['frac_events_muon_origin_only']:.1f}%")
    print("visible energy: " + ", ".join(f"{c} {100 * f[c]:.1f}%" for c in f))
    return out


def run(inputs, outdir, emit=None, merge=False, counters=None, **kw):
    S = _new_state()
    if merge:
        for f in tables.expand(inputs):
            _merge_into(S, json.load(open(f)))
        return report(S, outdir)
    sector_of_bar = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS)).sector
    for f in tables.expand(inputs):
        accumulate(S, f, sector_of_bar)
        print(f"# {f}: cumulative {S['events_with_steps']:,} events", flush=True)
    if emit:
        os.makedirs(os.path.dirname(emit) or ".", exist_ok=True)
        json.dump(S, open(emit, "w"))
        print(f"wrote {emit}")
        return S
    return report(S, outdir)
