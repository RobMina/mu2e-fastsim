"""B1 (FS-CALO-G4): the calorimeter arm of the benchmark from per-cluster and per-hit arrays.

Any host (a Geant4 fast-simulation model in Mu2eG4, the fastsim device chain, stock) describes one arm by flat
arrays: clusters (event index, energy, time, x, y, size, seed_energy, mxx, myy, mxy) and the reconstructed hits
(event index, energy), for a population of N events in a fixed order shared by both arms (the pairing). This module
turns them into the scorer's layout:

    <arm>/hists.json      histograms in event blocks (shards), the yield and the CE peak spectrum
    <arm>/features.npz    the classifier input: "X" (one row per cluster above E_MIN, columns FEATURES), "names",
                          "event" (the population index of each row)

Definitions (all fixed here, the same for every host):
- shards: consecutive blocks of ``block`` events of the population (b00000, b00001, ...), identical in both arms;
- histograms (counts per shard): cluster energy of every cluster; time, x, y, radius and size of the clusters above
  E_MIN; clusters above E_MIN per event; energy of every reconstructed hit;
- yield ``events_cluster_gt10``: events with a cluster above E_MIN (per shard);
- peak ``ce_peak``: the leading-cluster energy of the events whose leading cluster is above E_MIN, 1 MeV bins
  40-115 MeV, fitted by the scorer (CAL1's Gaussian core);
- features: energy (MeV), time (ns), x, y (mm, disk front-face frame, as CaloCluster cog), size (hits), seed_frac
  (largest member-hit energy / cluster energy), mxx, myy, mxy (mm^2: energy-weighted central second moments of the
  member crystal centres about the energy-weighted centre).

    python -m benchmarks.fastsim_score.calo ntuple OUTDIR FILE.root [FILE.root ...] [--events IDS.npy] [--block 100]
    python -m benchmarks.fastsim_score.calo reference OUTDIR [--stock-rate R]

``ntuple`` reads stock calorimeter ntuples (the ``prd/ntuple`` tree with ``caloclusters`` and ``calohits``, as written
by the calorimeter validation ntupler); ``--events`` (an (N, 3) int array of run, subrun, event) fixes the population
and its order, so both arms pair. ``reference`` writes the B1 reference.json. Only this converter needs uproot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WORKFLOW = "B1"
E_MIN = 10.0
TOLERANCE = 0.002                      # CE peak and yield tolerance (user, 2026-11-18)
MAX_SE = TOLERANCE / 3.0               # statistical power: CE peak-ratio standard error (0.067 %)
FEATURES = ("energy", "time", "x", "y", "size", "seed_frac", "mxx", "myy", "mxy")
CLUSTER_KEYS = ("event", "energy", "time", "x", "y", "size", "seed_energy", "mxx", "myy", "mxy")
HIST_EDGES = {
    "cluster_energy": np.arange(0.0, 117.0, 2.0),           # every cluster
    "cluster_time": np.arange(0.0, 1801.0, 20.0),           # clusters above E_MIN (ns)
    "cluster_x": np.arange(-700.0, 701.0, 25.0),
    "cluster_y": np.arange(-700.0, 701.0, 25.0),
    "cluster_radius": np.arange(300.0, 701.0, 20.0),
    "cluster_size": np.arange(-0.5, 25.5, 1.0),
    "clusters_per_event": np.arange(-0.5, 6.5, 1.0),        # clusters above E_MIN per event
    "hit_energy": np.r_[np.arange(0.0, 5.0, 0.5), np.arange(5.0, 81.0, 1.0)],   # every reconstructed hit (MeV)
}
PEAK_EDGES = np.arange(40.0, 116.0, 1.0)
REFERENCE = {
    "task": WORKFLOW,
    "boundary": "e+-/gamma state at the calorimeter disk front -> crystal deposits -> stock calo digi + reco clusters",
    "rate_unit": "events/s",
    "mode": "shape",
    "min_count": 10,
    "yield_tolerance": TOLERANCE,
    "yield_max_se": {"ce_peak": MAX_SE},
    "peak_fit": {"ce_peak": {"start_above": 75.0, "start_width": 3.0}},
    "classifier": {"file": "features.npz", "seed": 20261118},
    "max_memory_mb_per_thread": 250,
}


def shape_from_hits(hit_cluster, energy, x, y, n_clusters: int) -> dict:
    """Per-cluster seed_energy and mxx, myy, mxy from member hits (hit_cluster = cluster index or -1)."""
    hc = np.asarray(hit_cluster, np.int64)
    m = hc >= 0
    hc, e = hc[m], np.asarray(energy, np.float64)[m]
    x, y = np.asarray(x, np.float64)[m], np.asarray(y, np.float64)[m]
    seed = np.zeros(n_clusters)
    np.maximum.at(seed, hc, e)
    w = np.bincount(hc, weights=e, minlength=n_clusters)
    ws = np.where(w > 0, w, 1.0)
    cx = np.bincount(hc, weights=e * x, minlength=n_clusters) / ws
    cy = np.bincount(hc, weights=e * y, minlength=n_clusters) / ws
    dx, dy = x - cx[hc], y - cy[hc]
    out = {"seed_energy": seed}
    for k, v in (("mxx", dx * dx), ("myy", dy * dy), ("mxy", dx * dy)):
        out[k] = np.bincount(hc, weights=e * v, minlength=n_clusters) / ws
    return out


def _shard_hist(shard, values, edges, n_shards):
    return np.histogram2d(shard, values, bins=[np.arange(n_shards + 1) - 0.5, edges])[0]


def _shards(name_rows: np.ndarray) -> dict:
    return {f"b{i:05d}": [float(v) for v in row] for i, row in enumerate(name_rows)}


def arm(clusters: dict, hits: dict | None, n_events: int, block: int = 100) -> tuple[dict, dict]:
    """(hists.json content, features arrays) of one arm; clusters/hits carry the population event index."""
    missing = [k for k in CLUSTER_KEYS if k not in clusters]
    if missing:
        raise ValueError(f"clusters lack {missing}")
    c = {k: np.asarray(clusters[k]) for k in CLUSTER_KEYS}
    ev = c["event"].astype(np.int64)
    if ev.size and (ev.min() < 0 or ev.max() >= n_events):
        raise ValueError("cluster event index outside the population")
    n_shards = (n_events + block - 1) // block
    sh = ev // block
    e = c["energy"].astype(np.float64)
    sel = e > E_MIN
    hists = {"cluster_energy": _shard_hist(sh, e, HIST_EDGES["cluster_energy"], n_shards)}
    radius = np.hypot(c["x"], c["y"])
    for k, v in (("cluster_time", c["time"]), ("cluster_x", c["x"]), ("cluster_y", c["y"]),
                 ("cluster_radius", radius), ("cluster_size", c["size"])):
        hists[k] = _shard_hist(sh[sel], np.asarray(v, np.float64)[sel], HIST_EDGES[k], n_shards)
    n10 = np.bincount(ev[sel], minlength=n_events)
    hists["clusters_per_event"] = _shard_hist(np.arange(n_events) // block, n10, HIST_EDGES["clusters_per_event"],
                                              n_shards)
    if hits is not None:
        hev = np.asarray(hits["event"], np.int64)
        hists["hit_energy"] = _shard_hist(hev // block, np.asarray(hits["energy"], np.float64),
                                          HIST_EDGES["hit_energy"], n_shards)
    lead = np.zeros(n_events)
    np.maximum.at(lead, ev, e)
    has = lead > E_MIN
    yields = {"events_cluster_gt10": {"shards": {f"b{i:05d}": int(v) for i, v in enumerate(
        np.bincount(np.arange(n_events)[has] // block, minlength=n_shards))}}}
    peak = _shard_hist(np.arange(n_events)[has] // block, lead[has], PEAK_EDGES, n_shards)
    out = {"hists": {k: {"edges": HIST_EDGES[k].tolist(), "shards": _shards(v)} for k, v in hists.items()},
           "yields": yields,
           "peaks": {"ce_peak": {"edges": PEAK_EDGES.tolist(), "shards": _shards(peak)}},
           "population": {"events": int(n_events), "block": int(block), "e_min": E_MIN}}
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(e > 0, c["seed_energy"] / e, 0.0)
    cols = {"energy": e, "time": c["time"], "x": c["x"], "y": c["y"], "size": c["size"], "seed_frac": frac,
            "mxx": c["mxx"], "myy": c["myy"], "mxy": c["mxy"]}
    x = np.column_stack([np.asarray(cols[k], np.float64) for k in FEATURES])
    feats = {"X": x[sel], "names": np.array(FEATURES), "event": ev[sel]}
    return out, feats


def write_arm(outdir: Path, hists: dict, feats: dict) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "hists.json").write_text(json.dumps(hists) + "\n")
    np.savez_compressed(outdir / "features.npz", **feats)
    return outdir


def read_ntuple(path, tree: str = "prd/ntuple") -> tuple[np.ndarray, dict, dict]:
    """(ids (n, 3), clusters, hits) of a stock calorimeter ntuple; event index = entry in the file."""
    import awkward as ak
    import uproot
    with uproot.open(path) as f:
        a = f[tree].arrays(["evtinfo", "caloclusters", "calohits"])
    ev = a["evtinfo"]
    ids = np.column_stack([ak.to_numpy(ev[k]).astype(np.int64) for k in ("run", "subrun", "event")])
    cl, ht = a["caloclusters"], a["calohits"]
    get = lambda arr, k: ak.to_numpy(ak.flatten(arr[k])).astype(np.float64)  # noqa: E731
    ncl = ak.to_numpy(ak.num(cl["caloclusters.energyDep_"]))
    nht = ak.to_numpy(ak.num(ht["calohits.eDep_"]))
    clusters = {"event": np.repeat(np.arange(len(ncl)), ncl), "energy": get(cl, "caloclusters.energyDep_"),
                "time": get(cl, "caloclusters.time_"), "x": get(cl, "caloclusters.cog_.fCoordinates.fX"),
                "y": get(cl, "caloclusters.cog_.fCoordinates.fY"), "size": get(cl, "caloclusters.size_")}
    first = np.r_[0, np.cumsum(ncl)[:-1]]
    hev = np.repeat(np.arange(len(nht)), nht)
    cidx = ak.to_numpy(ak.flatten(ht["calohits.clusterIdx_"])).astype(np.int64)
    hc = np.where(cidx >= 0, first[hev] + cidx, -1)
    if np.any(hc >= len(clusters["energy"])) or np.any((cidx >= 0) & (cidx >= ncl[hev])):
        raise ValueError(f"{path}: calohits clusterIdx_ beyond the event's clusters")
    e = get(ht, "calohits.eDep_")
    clusters.update(shape_from_hits(hc, e, get(ht, "calohits.crystalPos_.fCoordinates.fX"),
                                    get(ht, "calohits.crystalPos_.fCoordinates.fY"), len(clusters["energy"])))
    return ids, clusters, {"event": hev, "energy": e}


def population(files, events: np.ndarray | None = None) -> tuple[np.ndarray, dict, dict]:
    """Concatenate ntuples into one population (ids in order); with ``events``, take exactly those, in that order."""
    ids_all, cls, hts, off = [], [], [], 0
    for f in files:
        ids, c, h = read_ntuple(f)
        c = dict(c, event=c["event"] + off)
        h = dict(h, event=h["event"] + off)
        ids_all.append(ids)
        cls.append(c)
        hts.append(h)
        off += len(ids)
    ids = np.concatenate(ids_all)
    cl = {k: np.concatenate([c[k] for c in cls]) for k in cls[0]}
    ht = {k: np.concatenate([h[k] for h in hts]) for k in hts[0]}
    if events is None:
        return ids, cl, ht
    where = {tuple(r): i for i, r in enumerate(ids)}
    try:
        take = np.array([where[tuple(r)] for r in np.asarray(events, np.int64)], np.int64)
    except KeyError as e:
        raise ValueError(f"event {e.args[0]} of the population is not in the ntuples") from None
    new = np.full(len(ids), -1, np.int64)
    new[take] = np.arange(len(take))
    kc, kh = new[cl["event"]] >= 0, new[ht["event"]] >= 0
    cl = {k: v[kc] for k, v in cl.items()}
    cl["event"] = new[cl["event"]]
    ht = {k: v[kh] for k, v in ht.items()}
    ht["event"] = new[ht["event"]]
    o = np.argsort(cl["event"], kind="stable")
    return ids[take], {k: v[o] for k, v in cl.items()}, ht


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("ntuple", help="stock calorimeter ntuples -> hists.json + features.npz")
    a1.add_argument("outdir", type=Path)
    a1.add_argument("files", type=Path, nargs="+")
    a1.add_argument("--events", type=Path, help="(N, 3) int .npy of run, subrun, event: the population, in order")
    a1.add_argument("--block", type=int, default=100)
    a2 = sub.add_parser("reference", help="write the B1 reference.json")
    a2.add_argument("outdir", type=Path)
    a2.add_argument("--stock-rate", type=float, default=None, help="stock events/s per grid slot at the boundary")
    a = ap.parse_args(argv)
    if a.cmd == "ntuple":
        ev = np.load(a.events, allow_pickle=False) if a.events else None
        ids, cl, ht = population(a.files, ev)
        hists, feats = arm(cl, ht, len(ids), a.block)
        write_arm(a.outdir, hists, feats)
        np.save(a.outdir / "events.npy", ids)
        print(f"wrote {a.outdir}: {len(ids)} events, {len(cl['energy'])} clusters", file=sys.stderr)
    else:
        a.outdir.mkdir(parents=True, exist_ok=True)
        (a.outdir / "reference.json").write_text(json.dumps({**REFERENCE, "stock_rate_per_slot": a.stock_rate},
                                                            indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
