"""g4cost -- cost of the stock Geant4 chain from art TimeTracker SQLite (the baseline of docs/plan.md section 3.2).

Inputs: the sim job's timing DB (timing_sim.db, ExtractedCRY with TimeTracker dbOutput) and the digi job's
(timing_digi.db). Events present in the digi DB are the ones the sim filter accepted.

    g4cost timing_sim.db --digi timing_digi.db --outdir DIR      -> g4cost_stats.json
"""
import sqlite3

import numpy as np
import pandas as pd

from .. import plots

CRV_LABELS = ("CrvPhotons", "CrvSiPMCharges", "CrvWaveforms", "CrvDigi")


def module_times(db):
    """Per-module and per-event wall times [s] from a TimeTracker SQLite DB, as pandas objects.

    Returns (wide, event): wide is a DataFrame indexed by Event with one column per module label (summed over paths),
    event a Series of the full-event time indexed by Event.
    """
    with sqlite3.connect(db) as con:
        mod = pd.read_sql("select Event, ModuleLabel, Time from TimeModule", con)
        ev = pd.read_sql("select Event, Time from TimeEvent", con)
    wide = mod.pivot_table(index="Event", columns="ModuleLabel", values="Time", aggfunc="sum")
    return wide, ev.set_index("Event").Time


def stats(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "sum_s": float(a.sum()), "mean_ms": 1e3 * float(a.mean()),
            "median_ms": 1e3 * float(np.median(a)), "p99_ms": 1e3 * float(np.percentile(a, 99)),
            "max_ms": 1e3 * float(a.max())}


def run(inputs, outdir, digi=None, **kw):
    if not digi:
        raise SystemExit("g4cost needs --digi timing_digi.db")
    sim, sim_ev = module_times(inputs[0])
    dig, digi_ev = module_times(digi)
    passed = sim.index.isin(digi_ev.index)
    g4 = sim["g4run"]
    crv = dig.reindex(columns=list(CRV_LABELS)).fillna(0.0).sum(axis=1)
    n_pass = int(passed.sum())
    out = {
        "n_generated": int(len(sim_ev)),
        "n_passed_filter": n_pass,
        "filter_fraction": n_pass / max(len(sim_ev), 1),
        "sim_full_event": stats(sim_ev),
        "sim_modules": {lab: stats(sim[lab].dropna()) for lab in sim.columns},
        "g4run_passed": stats(g4[passed]),
        "g4run_failed": stats(g4[~passed]),
        "digi_full_event": stats(digi_ev),
        "digi_modules": {lab: stats(dig[lab].dropna()) for lab in dig.columns},
        "crv_response_total": stats(crv),
        "cost_per_passed_event_s": {
            "geant4_stage_incl_failed": float(sim_ev.sum()) / max(n_pass, 1),
            "g4run_only_incl_failed": float(g4.sum()) / max(n_pass, 1),
            "crv_response": float(crv.mean()) if len(crv) else None,
            "digi_full": float(digi_ev.mean()) if len(digi_ev) else None,
        },
    }
    plots.write_json(outdir, "g4cost_stats.json", out)
    c = out["cost_per_passed_event_s"]
    print(f"generated {out['n_generated']} passed {out['n_passed_filter']} ({100 * out['filter_fraction']:.2f}%)")
    print(f"per passed event: G4 stage {c['geant4_stage_incl_failed'] * 1e3:.1f} ms, "
          f"CRV response {c['crv_response'] * 1e3:.1f} ms, full digi {c['digi_full'] * 1e3:.1f} ms")
    return out
