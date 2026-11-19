"""Synthetic example of the benchmark layout: toy histograms (Poisson draws from smooth shapes), no Mu2e content.
It is used by the tests and the Codabench dry run, and as a template for real references, which are kept outside
this repository.

    python -m benchmarks.fastsim_score.synthetic OUTDIR

writes OUTDIR/reference/ (two toy workflows, stock + reseeded floor, and a toy B1 calorimeter workflow (task B1) with
per-cluster features, the CE peak and the power rule) and OUTDIR/submissions/{good,shifted,slow}/ .
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import calo

WORKFLOWS = {"WA": {"stock_rate_per_slot": 12.5, "mode": "same_events", "boundary": "toy primaries -> toy hits"},
             "WB": {"stock_rate_per_slot": 3.2, "mode": "shape", "boundary": "toy hits -> toy tracks"}}
HISTS = {"energy": (40, 0.0, 10.0, lambda c: np.exp(-c / 2.5)),
         "time": (50, 0.0, 1000.0, lambda c: 1 + 0.5 * np.sin(c / 160.0)),
         "multiplicity": (20, -0.5, 19.5, lambda c: np.exp(-0.5 * ((c - 6) / 2.5) ** 2))}
N_SHARDS, EVENTS_PER_SHARD = 8, 4000
CALO = {"stock_rate_per_slot": 55.8, "events": 60000, "block": 1500, "feature_events": 5000}


def _calo_arm(seed: int, shift: float = 0.0, scale: float = 1.0) -> tuple[dict, dict]:
    """Toy B1 arm: a CE-like leading cluster (88 MeV core, low tail) in 93 % of events plus soft extra clusters;
    shift moves the shape features, scale the energies. Features are kept for the first feature_events events."""
    rng = np.random.default_rng(seed)
    n = CALO["events"]
    ce = rng.random(n) < 0.93
    n_extra = rng.poisson(0.5, n)
    ev = np.r_[np.flatnonzero(ce), np.repeat(np.arange(n), n_extra)]
    m = ev.size
    is_ce = np.arange(m) < ce.sum()
    energy = np.where(is_ce, 88.0 + 2.2 * rng.standard_normal(m) - (rng.random(m) < 0.2) * rng.exponential(6.0, m),
                      1.0 + rng.exponential(5.0, m)) * scale
    energy = np.clip(energy, 0.5, None)
    r, phi = rng.uniform(380.0, 640.0, m), rng.uniform(0, 2 * np.pi, m)
    size = 1 + rng.poisson(1.0 + energy / (25.0 - 5.0 * shift), m)
    frac = np.clip(rng.beta(6.0 + 4.0 * shift, 4.0, m), 1.0 / size, 1.0)
    mxx, myy = rng.gamma(2.0, 120.0 * (1 + shift), m), rng.gamma(2.0, 120.0, m)
    clusters = {"event": ev, "energy": energy, "time": 600.0 + rng.uniform(0, 1000, n)[ev] + rng.normal(0, 0.3, m),
                "x": r * np.cos(phi), "y": r * np.sin(phi), "size": size, "seed_energy": frac * energy,
                "mxx": mxx, "myy": myy, "mxy": rng.normal(0, 0.2, m) * np.sqrt(mxx * myy)}
    o = np.argsort(ev, kind="stable")
    clusters = {k: v[o] for k, v in clusters.items()}
    hev = np.repeat(clusters["event"], clusters["size"])
    hits = {"event": hev, "energy": rng.exponential(1.0, hev.size) * np.repeat(clusters["energy"] / clusters["size"],
                                                                                 clusters["size"])}
    hists, feats = calo.arm(clusters, hits, n, CALO["block"])
    keep = feats["event"] < CALO["feature_events"]
    return hists, {"X": feats["X"][keep].astype(np.float32), "names": feats["names"], "event": feats["event"][keep]}


def _write_calo(d: Path, arm: tuple[dict, dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "hists.json").write_text(json.dumps(arm[0]))
    np.savez_compressed(d / "features.npz", **arm[1])


def _arm(seed: int, shift: float = 0.0, yield_scale: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    hists = {}
    for key, (nb, lo, hi, f) in HISTS.items():
        edges = np.linspace(lo, hi, nb + 1)
        c = 0.5 * (edges[1:] + edges[:-1])
        pdf = f(c + shift * (hi - lo) / nb)
        pdf = pdf / pdf.sum()
        shards = {f"s{i:02d}": rng.poisson(EVENTS_PER_SHARD * pdf).tolist() for i in range(N_SHARDS)}
        hists[key] = {"edges": edges.tolist(), "shards": shards}
    yields = {"selected": {"shards": {f"s{i:02d}": int(rng.poisson(900 * yield_scale)) for i in range(N_SHARDS)}}}
    return {"hists": hists, "yields": yields}


def _bench(w: str, dtype: str, rate: float, device: str) -> dict:
    return {"workflow": w, "boundary": WORKFLOWS[w]["boundary"], "warm_rate": rate, "dtype": dtype,
            "device": device, "inputs": "fresh"}


def write(out: Path) -> Path:
    ref = out / "reference"
    for i, (w, cfg) in enumerate(WORKFLOWS.items()):
        (ref / w / "stock").mkdir(parents=True, exist_ok=True)
        (ref / w / "floor").mkdir(parents=True, exist_ok=True)
        (ref / w / "reference.json").write_text(json.dumps(
            {**cfg, "yield_tolerance": 0.02, "min_count": 5, "su_per_hour": {"toy-gpu": 3.0, "toy-cpu x8": 1.0}}))
        (ref / w / "stock" / "hists.json").write_text(json.dumps(_arm(100 + i)))
        (ref / w / "floor" / "hists.json").write_text(json.dumps(_arm(200 + i)))
    w = calo.WORKFLOW
    (ref / w).mkdir(parents=True, exist_ok=True)
    (ref / w / "reference.json").write_text(json.dumps(
        {**calo.REFERENCE, "stock_rate_per_slot": CALO["stock_rate_per_slot"], "su_per_hour": {"toy-gpu": 3.0}}))
    _write_calo(ref / w / "stock", _calo_arm(500))
    _write_calo(ref / w / "floor", _calo_arm(600))
    subs = {"good": dict(shift=0.0, ys=1.0, rate=5000.0), "shifted": dict(shift=0.6, ys=0.9, rate=5000.0),
            "slow": dict(shift=0.0, ys=1.0, rate=40.0)}
    for name, s in subs.items():
        for i, w in enumerate(WORKFLOWS):
            for j, dt in enumerate(("float32", "float64")):
                d = out / "submissions" / name / w / dt
                d.mkdir(parents=True, exist_ok=True)
                (d / "hists.json").write_text(json.dumps(_arm(300 + 10 * i + j, s["shift"], s["ys"])))
                (d / "bench.json").write_text(json.dumps(_bench(w, dt, s["rate"] * (1.6 if dt == "float32" else 1),
                                                                "toy-gpu")))
        for j, dt in enumerate(("float32", "float64")):
            d = out / "submissions" / name / calo.WORKFLOW / dt
            _write_calo(d, _calo_arm(700 + j, s["shift"], 0.99 if s["shift"] else 1.0))
            (d / "bench.json").write_text(json.dumps(
                {"workflow": calo.WORKFLOW, "boundary": calo.REFERENCE["boundary"], "dtype": dt, "device": "toy-gpu",
                 "warm_rate": s["rate"] * (1.6 if dt == "float32" else 1), "inputs": "fresh", "host": "device"}))
    return out


if __name__ == "__main__":
    print(write(Path(sys.argv[1])))
