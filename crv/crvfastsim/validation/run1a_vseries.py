"""Run 1A V-series: fast arms against Geant4 and stock, paired per shard (the arms share each shard's CRY events).

    python3 -m crvfastsim.validation.run1a_vseries --pairs PAIRS.json --out DIR [--plots]

PAIRS.json: one row per held-out shard {"shard", "g4_steps", "fast_c", "fast_1b", "stock_reco", "fast_reco",
"g4_reco"}: the Geant4 CrvStep table, the fast transport table (transport --stage C), the tier-1b table (tier1b-apply),
the stock reco of the Geant4 steps (CrvRecoTableDumper), the fast response and reco of the tier-1b steps (end to end)
and of the Geant4 steps (response control), both `runchain --steps-input --save-reco` files.

- V1, primary-muon steps (fast_c vs Geant4): crossings (event, bar, primary), their visible energy, per sector.
- V4, steps after the library (fast_1b vs Geant4): visible energy per sector and origin (primary muon, muon
  descendant, non-muon), steps per event with steps.
- End to end (fast_reco vs stock_reco), with the response control (g4_reco vs stock_reco): coincidence clusters, cluster
  PE, layers per cluster, pulses of at least PULSE_PE per sector, and their PE.

Statistics: the ratio of the sums with the paired per-shard error (validation.paired_rates.paired_ratio), and per
histogram bin the summed difference over its paired per-shard standard error (pull); chi2/ndf = mean squared pull over
bins with a nonzero error. Tail bins are merged from the top until each reference bin holds MIN_REF entries. The bin
errors come from the spread of n shards, so each pull follows a t distribution with n - 1 degrees of freedom and chi2/ndf
has the null expectation (n - 1)/(n - 3) (1.33 for nine shards), not 1. Bins with zero error are listed, not used.
"""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .. import runconfig, tables, tier1b
from .paired_rates import paired_ratio as _paired_ratio
from ..analyses.census import classify

KEY = tables.KEY
PULSE_PE = 5.0
PE_EDGES = np.r_[np.arange(0, 3000, 100), 3000, 1e9]
EX_EDGES = np.r_[np.arange(0, 20.5, 1.0), 1e9]
NSTEP_EDGES = np.r_[1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 1e9]
PPE_EDGES = np.r_[np.arange(0, 200, 10), 200, 1e9]
LAYERS = np.arange(1, 6)
ORIGINS = ("mu_primary", "mu_descendant", "nonmu")
MIN_REF = 25


def paired_ratio(f, r):
    """Ratio of sums over shards with its paired per-shard standard error (arms share each shard's events)."""
    q = _paired_ratio(r, f)
    ratio = None if q["relative_difference"] is None else 1 + q["relative_difference"]
    return {"fast": q["total_cand"], "ref": q["total_ref"], "ratio": float("nan") if ratio is None else ratio,
            "err": float("nan") if q["relative_se"] is None else q["relative_se"], "shards": q["events"]}


def merge_tail(F, R, min_ref=MIN_REF):
    """Merge bins from the top down until every merged bin's reference sum is at least min_ref (the last group joins
    its lower neighbour). Returns (F, R, groups: list of original bin indices per merged bin)."""
    tot = R.sum(0)
    groups, cur, acc = [], [], 0.0
    for b in range(len(tot) - 1, -1, -1):
        cur.insert(0, b)
        acc += tot[b]
        if acc >= min_ref:
            groups.insert(0, cur)
            cur, acc = [], 0.0
    if cur:
        if groups:
            groups[0] = cur + groups[0]
        else:
            groups = [cur]
    merge = lambda X: np.stack([X[:, g].sum(1) for g in groups], 1)
    return merge(F), merge(R), groups


def paired_shape(F, R, n_shards=None):
    """Per (tail-merged) bin: summed difference, its paired per-shard error and pull; chi2/ndf over bins with a nonzero
    error, with its null expectation (n - 1)/(n - 3) for t-distributed pulls of n shards."""
    F, R = np.asarray(F, float), np.asarray(R, float)
    F, R, groups = merge_tail(F, R)
    n = len(F)
    d = F - R
    D = d.sum(0)
    se = np.sqrt(n * d.var(0, ddof=1)) if n > 1 else np.zeros(D.shape)
    ok = se > 0
    pull = np.where(ok, D / np.where(ok, se, 1), 0.0)
    return {"fast": F.sum(0).tolist(), "ref": R.sum(0).tolist(), "diff_err": se.tolist(), "pull": pull.tolist(),
            "bins": groups, "zero_error_bins": [groups[i] for i in np.flatnonzero(~ok)],
            "chi2_ndf": float((pull[ok] ** 2).mean()) if ok.any() else np.nan, "ndf": int(ok.sum()),
            "chi2_ndf_null": (n - 1) / (n - 3) if n > 3 else np.nan,
            "max_abs_pull": float(np.abs(pull[ok]).max()) if ok.any() else np.nan}


def _hist(v, edges, w=None):
    return np.histogram(np.clip(v, edges[0], edges[-1] - 1e-9), edges, weights=w)[0]


def _sector_names(geom):
    return [str(n) for n in geom.sector_names]


# ---------------------------------------------------------------------------------------------------------- steps
def _classified(path, geom):
    st = tables.read(path, "steps", KEY + ["bar", "sp", "edep", "t0", "x0", "y0", "z0", "px0", "py0", "pz0", "len"])
    pa = tables.read(path, "particles", KEY + ["key", "parent", "pdg", "isprim"])
    s = classify(st, pa, geom.sector)
    s["origin"] = np.where(s.cls.str.startswith("nonmu"), "nonmu", s.cls)
    s["sector"] = np.asarray(_sector_names(geom), dtype=object)[geom.sector[s.bar.to_numpy()]]
    return s


def step_summary(path, geom):
    """V1 and V4 per-shard quantities of one step table."""
    s = _classified(path, geom)
    out = {}
    mu = s[s.cls == "mu_primary"]
    x = tier1b._crossings(mu, per_primary=True)
    xsec = np.asarray(_sector_names(geom), dtype=object)[geom.sector[x.bar.to_numpy()]]
    for sec in _sector_names(geom):
        m = xsec == sec
        out[f"v1/{sec}/crossings"] = int(m.sum())
        out[f"v1/{sec}/primary_edep"] = float(x.edep.to_numpy()[m].sum())
        out[f"v1/{sec}/edep_per_crossing"] = _hist(x.edep.to_numpy()[m], EX_EDGES)
        for o in ORIGINS:
            out[f"v4/{sec}/{o}_edep"] = float(s.edep[(s.sector == sec) & (s.origin == o)].sum())
        out[f"v4/{sec}/edep"] = float(s.edep[s.sector == sec].sum())
    out["v4/nstep"] = _hist(s.groupby(KEY).size().to_numpy(), NSTEP_EDGES)
    out["v4/events_with_steps"] = int(s[KEY].drop_duplicates().shape[0])
    return out


# ------------------------------------------------------------------------------------------------------------ reco
def _reco_frames_stock(path):
    f = uproot.open(path)
    c = f["crvdump/clusters"].arrays(KEY + ["pe", "layers", "pulses", "sectortype"])
    p = f["crvdump/pulses"].arrays(KEY + ["idx", "bar", "pe"], library="pd")
    ev = (p.run.to_numpy().astype(np.int64) << 40) | (p.subrun.to_numpy().astype(np.int64) << 20) | p.event.to_numpy()
    pulses = pd.DataFrame({"ev": ev, "idx": p.idx.to_numpy(), "bar": p.bar.to_numpy(), "pe": p.pe.to_numpy()})
    cev = (ak.to_numpy(c.run).astype(np.int64) << 40) | (ak.to_numpy(c.subrun).astype(np.int64) << 20) | \
        ak.to_numpy(c.event).astype(np.int64)
    n = ak.to_numpy(ak.num(c.pulses))
    mem = pd.DataFrame({"cl": np.repeat(np.arange(len(c)), n), "ev": np.repeat(cev, n),
                        "idx": ak.to_numpy(ak.flatten(c.pulses)).astype(np.int64)})
    return pd.DataFrame({"cl": np.arange(len(c)), "pe": ak.to_numpy(c.pe)}), mem, pulses


def _reco_frames_fast(path):
    z = np.load(path)
    cl = pd.DataFrame({"cl": np.arange(len(z["cl_pe"])), "pe": z["cl_pe"]})
    ev = z["cl_event"].astype(np.int64)
    mem = pd.DataFrame({"cl": z["mem_cl"], "ev": ev[z["mem_cl"]] if len(ev) else np.zeros(0, np.int64),
                        "idx": z["mem_idx"]})
    pulses = pd.DataFrame({"ev": z["p_ev"].astype(np.int64), "idx": z["p_idx"].astype(np.int64),
                           "bar": z["p_bar"].astype(np.int64), "pe": z["p_pe"]})
    return cl, mem, pulses


def reco_summary(path, geom, stock):
    """End-to-end per-shard quantities of one reco output (stock CrvRecoTableDumper file or runchain --save-reco)."""
    cl, mem, pulses = (_reco_frames_stock if stock else _reco_frames_fast)(path)
    names = np.asarray(_sector_names(geom), dtype=object)
    m = mem.merge(pulses, on=["ev", "idx"], how="left", validate="many_to_one")
    if m.bar.isna().any():
        raise ValueError(f"{path}: cluster member without its pulse")
    m["layer"] = geom.layer[m.bar.to_numpy().astype(np.int64)]
    m["sector"] = names[geom.sector[m.bar.to_numpy().astype(np.int64)]]
    nlay = m.drop_duplicates(["cl", "layer"]).groupby("cl").size().reindex(cl.cl, fill_value=0).to_numpy()
    csec = m.sort_values(["cl", "idx"]).drop_duplicates("cl").set_index("cl").sector.reindex(cl.cl).to_numpy()
    out = {"e2e/clusters": int(len(cl)), "e2e/cluster_pe": _hist(cl.pe.to_numpy(), PE_EDGES),
           "e2e/layers": _hist(nlay, np.r_[LAYERS, 99])}
    psec = names[geom.sector[pulses.bar.to_numpy()]]
    for sec in _sector_names(geom):
        out[f"e2e/{sec}/clusters"] = int((csec == sec).sum())
        out[f"e2e/{sec}/cluster_pe"] = _hist(cl.pe.to_numpy()[csec == sec], PE_EDGES)
        sel = (psec == sec) & (pulses.pe.to_numpy() >= PULSE_PE)
        out[f"e2e/{sec}/pulses"] = int(sel.sum())
        out[f"e2e/{sec}/pulse_pe"] = _hist(pulses.pe.to_numpy()[(psec == sec)], PPE_EDGES)
    return out


# ---------------------------------------------------------------------------------------------------------- driver
def _compare(fast, ref):
    """Paired statistics of every common key of per-shard summaries (lists of dicts in shard order)."""
    res = {}
    for k in fast[0]:
        F = [r[k] for r in fast]
        R = [r[k] for r in ref]
        if np.ndim(F[0]) == 0:
            res[k] = paired_ratio(F, R)
        else:
            res[k] = paired_shape(np.stack(F), np.stack(R))
    return res


def run(pairs, out, run_config="run1a_v01", plots=False, stock_label=None):
    rows = json.loads(Path(pairs).read_text())
    if len(rows) < 2:
        raise ValueError("paired statistics need at least two shards")
    geom = runconfig.get(run_config).geometry()
    S = {k: [] for k in ("g4", "fast_c", "fast_1b", "stock", "e2e", "resp")}
    for r in rows:
        S["g4"].append(step_summary(r["g4_steps"], geom))
        S["fast_c"].append(step_summary(r["fast_c"], geom))
        S["fast_1b"].append(step_summary(r["fast_1b"], geom))
        S["stock"].append(reco_summary(r["stock_reco"], geom, stock=True))
        S["e2e"].append(reco_summary(r["fast_reco"], geom, stock=False))
        S["resp"].append(reco_summary(r["g4_reco"], geom, stock=False))
        print(json.dumps({"shard": r["shard"], "clusters": [S[a][-1]["e2e/clusters"] for a in ("stock", "e2e", "resp")]}),
              flush=True)
    pick = lambda arm, pre: [{k: v for k, v in d.items() if k.startswith(pre)} for d in S[arm]]
    result = {"run_config": run_config, "shards": [r["shard"] for r in rows], "pairs": rows, "pulse_pe_min": PULSE_PE,
              "edges": {"cluster_pe": PE_EDGES.tolist(), "edep_per_crossing": EX_EDGES.tolist(),
                        "nstep": NSTEP_EDGES.tolist(), "pulse_pe": PPE_EDGES.tolist(), "layers": LAYERS.tolist()},
              "v1": _compare(pick("fast_c", "v1/"), pick("g4", "v1/")),
              "v4": _compare(pick("fast_1b", "v4/"), pick("g4", "v4/")),
              "e2e": _compare(S["e2e"], S["stock"]),
              "response_control": _compare(S["resp"], S["stock"]),
              "e2e_vs_response_control": _compare(S["e2e"], S["resp"])}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vseries.json").write_text(json.dumps(result, indent=1, default=float))
    if plots:
        from . import run1a_vseries_plots
        run1a_vseries_plots.make(result, out, {"stock": stock_label} if stock_label else None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-config", default="run1a_v01")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--stock-label", default=None, help="legend of the stock arm (its Offline release)")
    a = p.parse_args()
    run(a.pairs, a.out, a.run_config, a.plots, a.stock_label)


if __name__ == "__main__":
    main()
