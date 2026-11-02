"""Per-stage, per-column content hashes of runchain passes: does the chain repeat run to run, and where not.

Wraps the stage functions without changing them (transport, tier-1b sampling, photons, SiPM charges, charges, digis,
reco waveforms, clusters) and hashes every returned column (dtype, shape, bytes). `--repeat N` runs each shard N times
in one process; `--dump DIR` also saves every stage output as .npz for `diff`.

    python3 scripts/chain_identity.py run --inputs-root ROOT --out A.json --shards 22 24 [--jax ...] [--dump DIR]
    python3 scripts/chain_identity.py compare A.json [B.json]        first differing stage and columns per shard pass
                                                                     (no B: A's in-process repeats against pass 0)
    python3 scripts/chain_identity.py diff DUMP_A DUMP_B             per column: changed elements, max |difference|
"""
import argparse
import functools
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

STAGES = ("transport", "tier1b", "photons", "sipm", "charges", "digis", "reco_waveforms", "clusters")


def columns(obj, prefix=""):
    """Flatten a stage output (DataFrame, dict, list/tuple, arrays, scalars) into {name: numpy array}."""
    out = {}
    if isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            out[f"{prefix}{c}"] = obj[c].to_numpy()
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            out.update(columns(obj[k], f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(columns(v, f"{prefix}{i}."))
    elif obj is None:
        out[prefix.rstrip(".")] = np.zeros(0)
    else:
        try:
            a = np.asarray(obj)
        except Exception:
            a = np.asarray(repr(type(obj)))
        out[prefix.rstrip(".")] = a.astype(str) if a.dtype.hasobject else a
    return out


def digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def run(a):
    from crvfastsim import chain, tier1b, transport_jax
    from crvfastsim.response import photons_jax, sipm_jax
    from crvfastsim.analyses import runchain
    root = Path(a.inputs_root)
    record, current, dumps = {}, {}, {}

    def wrap(obj, name, stage, pick):
        f = getattr(obj, name)

        @functools.wraps(f)
        def call(*args, **kw):
            out = f(*args, **kw)
            cols = columns(pick(out))
            current.setdefault(stage, []).append({k: digest(v) for k, v in cols.items()})
            if a.dump:
                dumps.setdefault(stage, []).append(cols)
            return out
        return patch.object(obj, name, call)

    patches = [wrap(transport_jax, "transport", "transport", lambda o: o),
               wrap(tier1b, "sample", "tier1b", lambda o: o[:2]),
               wrap(photons_jax, "make_photons", "photons", lambda o: o),
               wrap(sipm_jax, "simulate", "sipm", lambda o: o),
               wrap(chain, "charges", "charges", lambda o: o), wrap(chain, "digis", "digis", lambda o: o),
               wrap(chain, "reco_waveforms", "reco_waveforms", lambda o: o),
               wrap(chain, "clusters", "clusters", lambda o: o[0])]
    if a.tier1b_backend == "jax":
        from crvfastsim import tier1b_jax
        patches.append(wrap(tier1b_jax.DeviceTier1b, "sample", "tier1b", lambda o: o[:2]))
    for p in patches:
        p.start()
    try:
        for s in a.shards:
            for rep in range(a.repeat):
                current.clear()
                dumps.clear()
                tag = f"{s}#{rep}"
                res = runchain.run([str(root / f"gen_even/gen.001400_{s:08d}.root")],
                                   str(Path(a.out).parent / f"pass_{Path(a.out).stem}_{s}_{rep}"),
                                   calib=str(root / "data/geometry/crvcalib_sim_best_v1_5_run1400.csv"),
                                   deviation=str(root / "data/geometry/crvphoton_sim_best_v1_5_run1400.csv"),
                                   gdml=str(root / "data/geometry/mu2e_extracted_v04.gdml"), jax=a.jax, seed=a.seed,
                                   max_events=a.max_events, tier1b_set="strict_v4",
                                   tier1b_dir=str(root / "tier1b_strict_v2"),
                                   transport_backend=a.transport_backend, tier1b_backend=a.tier1b_backend)
                record[tag] = {"stages": dict(current), "generated_events": res["generated_events"],
                               "clusters_per_generated": res["clusters_per_generated"]}
                print(json.dumps({"pass": tag, "clusters_per_generated": res["clusters_per_generated"]}), flush=True)
                if a.dump:
                    d = Path(a.dump) / tag.replace("#", "_")
                    d.mkdir(parents=True, exist_ok=True)
                    for st, calls in dumps.items():
                        np.savez(d / f"{st}.npz", **{f"{i}|{k}": v for i, c in enumerate(calls) for k, v in c.items()})
    finally:
        for p in patches:
            p.stop()
    with open(a.out, "x") as f:
        json.dump({"args": vars(a), "passes": record}, f, indent=1)


def _changed(x, y):
    """Columns whose hash differs between two stage records (lists of per-call {column: hash})."""
    if len(x) != len(y):
        return ["<number of calls>"]
    return sorted({k for cx, cy in zip(x, y) for k in set(cx) | set(cy) if cx.get(k) != cy.get(k)})


def compare(a):
    if a.b is None:   # in-process repeats of one run: pass <shard>#k against <shard>#0
        p = json.loads(Path(a.a).read_text())["passes"]
        x = {t: p[t.split("#")[0] + "#0"] for t in p if not t.endswith("#0")}
        y = {t: p[t] for t in x}
    else:
        x, y = (json.loads(Path(p).read_text())["passes"] for p in (a.a, a.b))
    rows, first = {}, {}
    for t in sorted(set(x) | set(y)):
        if t not in x or t not in y:
            first[t] = "missing"
            continue
        rows[t] = {}
        for st in STAGES:
            ch = _changed(x[t]["stages"].get(st, []), y[t]["stages"].get(st, []))
            if ch:
                rows[t][st] = ch
                first.setdefault(t, st)
    print(json.dumps({"passes": len(set(x) & set(y)), "identical": not first, "first_difference": first,
                      "changed_columns": rows}, indent=1))
    return 0 if not first else 1


def diff(a):
    """Per stage and column of two dumps (one pass dir each): elements changed, max |difference|, and whether the
    column is a permutation of the other (same sorted values)."""
    out = {}
    for f in sorted(Path(a.a).glob("*.npz")):
        x, y = np.load(f), np.load(Path(a.b) / f.name)
        st = {}
        for k in x.files:
            u, v = x[k], y[k] if k in y.files else None
            if v is None or u.shape != v.shape:
                st[k] = {"shape": [list(u.shape), None if v is None else list(v.shape)]}
                continue
            if u.tobytes() == v.tobytes():
                continue
            row = {"n": int(u.size)}
            if u.dtype.kind in "fc":
                d = np.abs(u.astype(np.float64) - v.astype(np.float64))
                nz = d[np.isfinite(d)]
                row.update(changed=int((u != v).sum()), max_abs=float(nz.max(initial=0)),
                           max_rel=float(np.max(nz / np.maximum(np.abs(u.astype(np.float64))[np.isfinite(d)], 1e-300),
                                                initial=0)))
            else:
                row.update(changed=int((u != v).sum()))
            row["permutation"] = bool(np.array_equal(np.sort(u, axis=None), np.sort(v, axis=None)))
            st[k] = row
        if st:
            out[f.stem] = st
    print(json.dumps({"identical": not out, "stages": out}, indent=1))
    return 0 if not out else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--inputs-root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shards", nargs="+", type=int, default=[22])
    r.add_argument("--repeat", type=int, default=1, help="passes per shard in this process")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--max-events", type=int)
    r.add_argument("--jax", action="store_true")
    r.add_argument("--transport-backend", choices=("numpy", "jax"), default="numpy")
    r.add_argument("--tier1b-backend", choices=("numpy", "jax"), default="numpy")
    r.add_argument("--dump", help="also save every stage output per pass as DIR/<shard>_<rep>/<stage>.npz")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b", nargs="?", help="second run (omitted: the in-process repeats of A against its first pass)")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = p.parse_args()
    sys.exit({"run": run, "compare": compare, "diff": diff}[a.cmd](a))


if __name__ == "__main__":
    main()
