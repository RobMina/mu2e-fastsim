"""Bounded real-input convergence pilot; independent saved seed passes, never warm timing replicas.

Runs nominal, half-step at equal maximum path, doubled path cap, and doubled loss slicing sequentially. No physics
acceptance threshold is implied. Event-cluster standard errors describe the reused input sample, not model error.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from . import g4ref
from .reference_audit import _identity

ARMS = {"nominal": (5.0, 3000, 4), "halfstep": (2.5, 6000, 4),
        "doublecap": (5.0, 6000, 4), "slices8": (5.0, 3000, 8)}


def select_block(g, start, count):
    """Select whole events in a requested row range; reject boundary splits rather than silently overlap."""
    if start < 0 or count <= 0 or start + count > len(g["evt"]):
        raise ValueError("Requested block must be nonempty and wholly within the reference")
    ids = np.column_stack([g[k] for k in ("run", "subrun", "evt")])
    stop = start + count
    chosen = np.zeros(len(ids), bool)
    chosen[start:stop] = True
    _, inv = np.unique(ids, axis=0, return_inverse=True)
    if np.intersect1d(inv[chosen], inv[~chosen]).size:
        raise ValueError("Requested row boundary splits an event; choose a whole-event block")
    selected = {k: v[start:stop] for k, v in g.items()}
    return selected, dict(row_start=start, row_stop_exclusive=stop,
                          events=len(np.unique(inv[chosen])), first_event=ids[start].tolist(),
                          last_event=ids[stop - 1].tolist())


def cluster_mean(values, groups):
    """Muon-weighted mean and event-cluster standard error, conditional on this input sample."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return dict(mean=None, stderr=None, events=0)
    _, inv = np.unique(groups, axis=0, return_inverse=True)
    m = int(inv.max()) + 1
    mean = float(values.mean())
    residual = np.bincount(inv, weights=values - mean)
    err = float(np.sqrt(m / (m - 1) * np.sum(residual ** 2)) / len(values)) if m > 1 else None
    return dict(mean=mean, stderr=err, events=m)


def summarize(g, f, baseline=None):
    """Summarize a saved prefix with event-cluster paired differences and explicit capped lanes."""
    n = len(f["fate"])
    if n == 0 or n > len(g["evt"]):
        raise ValueError("Nonempty fast result must fit within the reference prefix")
    ids = np.column_stack([g[k][:n] for k in ("run", "subrun", "evt")])
    stop = np.asarray(f["fate"]) == 1
    gs = np.asarray(g["foil"][:n]) >= 0
    first = stop & (f["foil"] == 0)
    result = dict(n=n, stop=cluster_mean(stop, ids), g4_stop=cluster_mean(gs, ids),
                  fast_minus_g4=cluster_mean(stop.astype(float) - gs, ids),
                  first_foil=cluster_mean(first, ids),
                  g4_first_foil=cluster_mean(g["foil"][:n] == 0, ids),
                  fates={str(int(k)): int(v) for k, v in zip(*np.unique(f["fate"], return_counts=True))},
                  foil_profile=np.bincount(f["foil"][stop], minlength=int(g["foil"].max()) + 1).tolist(),
                  stop_flight_ns_mean=float(np.mean(f["t"][stop] - g["time"][:n][stop])) if stop.any() else None)
    if baseline is not None:
        if len(baseline["fate"]) != n:
            raise ValueError("Comparison arms must have identical prefix lengths")
        bs = baseline["fate"] == 1
        both = stop & bs
        result["minus_nominal"] = cluster_mean(stop.astype(float) - bs, ids)
        result["first_minus_nominal"] = cluster_mean(first.astype(float) - (bs & (baseline["foil"] == 0)), ids)
        result["both_stop"] = int(both.sum())
        result["flight_time_delta_ns_given_both"] = cluster_mean((f["t"] - baseline["t"])[both], ids[both])
    return result


def main(argv=None):
    from . import __main__ as cli
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g4", required=True)
    ap.add_argument("--out", required=True, help="new directory; existing directories are rejected")
    ap.add_argument("--geom")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0, help="first reference row; whole-event boundaries required")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--budget-seconds", type=float, default=1200, help="stop launching new arms after this wall time")
    a = ap.parse_args(argv)
    if a.n <= 0 or a.budget_seconds <= 0 or len(set(a.seeds)) != len(a.seeds) or len(set(a.arms)) != len(a.arms):
        ap.error("positive n/budget and unique seeds/arms required")
    if not a.arms or a.arms[0] != "nominal":
        ap.error("nominal must be the first arm for paired differences")
    g, block = select_block(g4ref.load(a.g4), a.start, a.n)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    block_path = out / "input_block.npz"
    g4ref.save(g, block_path)
    cli._geom(a.geom, target_contract=True)
    from . import geometry
    sources = []
    geometry.from_offline(a.geom or cli.GEOM_FILE, cli.OFFLINE_ROOT, source_files=sources)
    source_dir = Path(__file__).resolve().parent
    core_dir = Path(__import__("mu2efastsim").__file__).resolve().parent
    manifest = dict(input=_identity(a.g4), selected_input=_identity(block_path), block=block,
                    n=len(g["evt"]), seeds=a.seeds, arms=a.arms,
                    code=[_identity(p) for p in sorted(source_dir.glob("*.py"))],
                    core_code=[_identity(core_dir / p) for p in ("eloss.py", "eloss_jax.py", "materials.py")],
                    geometry=[_identity(p) for p in sources],
                    field=[_identity(Path(cli.MAPS) / f"{cli.MAP}.{ext}") for ext in ("header", "bin")],
                    note="Fixed whole-event block; seeds repeat response only. Same seed is not layout-invariant random coupling.",
                    budget_seconds=a.budget_seconds)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = dict(complete=False, runs={}, unstarted=[], scope="Pilot only; no physics acceptance gate")
    for seed in a.seeds:
        baseline = None
        for arm in a.arms:
            name = f"{arm}_seed{seed}"
            if time.monotonic() - started >= a.budget_seconds:
                summary["unstarted"].append(name)
                continue
            ds, cap, slices = ARMS[arm]
            args = SimpleNamespace(g4=str(block_path), out=str(out / f"{name}.npz"), geom=a.geom, n=a.n,
                                   ds=ds, max_steps=cap, slices=slices, tcut=1.0, no_msc=False, no_decay=False,
                                   seed=seed, batch=a.n, chunk=200, repeat=0)
            cli.cmd_run(args)
            f = g4ref.load(args.out)
            if arm == "nominal":
                baseline = f
            summary["runs"][name] = dict(summarize(g, f, baseline),
                                          meta=json.loads(str(f["meta"])))
            (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
            print(f"finished {name}: {summary['runs'][name]['stop']}", flush=True)
    summary["complete"] = not summary["unstarted"]
    summary["wall_s"] = time.monotonic() - started
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
