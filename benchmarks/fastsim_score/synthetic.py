"""Synthetic example of the benchmark layout: toy histograms (Poisson draws from smooth shapes), no Mu2e content.
It is used by the tests and the Codabench dry run, and as a template for real references, which are kept outside
this repository.

    python -m benchmarks.fastsim_score.synthetic OUTDIR

writes OUTDIR/reference/ (two toy workflows, stock + reseeded floor) and OUTDIR/submissions/{good,shifted,slow}/ .
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

WORKFLOWS = {"WA": {"stock_rate_per_slot": 12.5, "mode": "same_events", "boundary": "toy primaries -> toy hits"},
             "WB": {"stock_rate_per_slot": 3.2, "mode": "shape", "boundary": "toy hits -> toy tracks"}}
HISTS = {"energy": (40, 0.0, 10.0, lambda c: np.exp(-c / 2.5)),
         "time": (50, 0.0, 1000.0, lambda c: 1 + 0.5 * np.sin(c / 160.0)),
         "multiplicity": (20, -0.5, 19.5, lambda c: np.exp(-0.5 * ((c - 6) / 2.5) ** 2))}
N_SHARDS, EVENTS_PER_SHARD = 8, 4000


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
    return out


if __name__ == "__main__":
    print(write(Path(sys.argv[1])))
