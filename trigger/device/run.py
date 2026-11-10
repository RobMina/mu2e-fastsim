"""Device trigger CLI.

  python3 -m trigger.device.run decide --dump a/trgdump.root ... --menu replay.fcl --tc-mva TimeCluster.weights.xml \
        [--response response.json] [--logs a/replay.log ... --prepared DIR] --output result.json [--check-hits]
  python3 -m trigger.device.run response --logs dev/*/replay.log --device dev.json --menu replay.fcl --output response.json

`decide` runs the device chain on the digis in StrawHitTableDumper files (one per replay job, in event order), checks
the device straw/panel hits against the stock trigger hits in the same file (`--check-hits`), and, with `--logs`,
pairs the per-path decisions with the stock replay decisions of the same events (trigger/hybrid/pair.py statistics).
`response` builds the response layer (decision D1) from development shards only: per path, the stock KalSeedFilter
pass fraction (passed / visited) over the device's pass fraction of the same cuts on helix quantities, and for the apr paths, whose helix finder (AgnosticHelixFinder)
the device replaces with its own fit, the ratio of the stock to the device HelixFilter pass fraction (capped at 1).
Apply it with `decide --response` to held-out shards.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
import sys
import tempfile
import time

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare  # noqa: E402
from hybrid import pair as pairmod  # noqa: E402

from trkfastsim.reco import hitreco  # noqa: E402

from . import chain as chainmod, hits, menu  # noqa: E402

DIG = ("sid", "tdc0", "tdc1", "tot0", "tot1", "pmp")
STOCK = ("sid", "flag", "time", "edep", "x", "y", "z", "wdist", "uvar", "vvar", "ncombo", "nsh")


class _Shim:
    """hitreco.extract_conditions reads `trkdump/conditions`; point it at another dumper directory."""
    def __init__(self, tree):
        self.tree = tree

    def open(self, path):
        f = uproot.open(path)
        return {"trkdump/conditions": f[f"{self.tree}/conditions"]}


def conditions(path, tree):
    real = hitreco.uproot
    hitreco.uproot = _Shim(tree)
    try:
        with tempfile.TemporaryDirectory() as d:
            hitreco.extract_conditions(path, Path(d) / "cond.json")
            return hitreco.load_conditions(Path(d) / "cond.json")
    finally:
        hitreco.uproot = real


def load_dump(path, tree="trigph"):
    names = ["run", "subrun", "event", "pbt", "spill"] + ["d_" + k for k in DIG] + \
        [p + k for p in ("s_", "p_") for k in STOCK]
    a = uproot.open(path)[f"{tree}/events"].arrays(names)
    ev = {k: [np.asarray(x) for x in ak.to_list(a["d_" + k])] for k in DIG}
    ev["pbt"], ev["spill"] = ak.to_numpy(a.pbt), ak.to_numpy(a.spill)
    keys = list(zip(ak.to_numpy(a.run).tolist(), ak.to_numpy(a.subrun).tolist(), ak.to_numpy(a.event).tolist()))
    stock = {p + k: a[p + k] for p in ("s_", "p_") for k in STOCK}
    return keys, ev, stock


def batches(keys, ev, size):
    for lo in range(0, len(keys), size):
        sl = slice(lo, lo + size)
        sub = {k: (v[sl] if isinstance(v, list) else v[sl]) for k, v in ev.items()}
        n = len(sub["sid"])
        b = hits.pack_events(sub)
        E = size
        if n < E:                                   # pad events to the fixed batch size (one compiled shape)
            for k, v in b.items():
                pad = np.zeros((E - n,) + v.shape[1:], v.dtype)
                b[k] = np.concatenate([v, pad])
        b["valid_event"] = np.arange(E) < n
        b["event"] = np.array([k[2] for k in keys[sl]] + [0] * (E - n), np.int64)
        yield lo, n, b


def check_hits(diag, stock, lo, n, report):
    """Device straw/panel hits vs the stock TrigMakeSH/TrigMakePH of the same events."""
    for tag, d, pre in (("straw", diag["sh"], "s_"), ("panel", diag["ph"], "p_")):
        r = report.setdefault(tag, {"events": 0, "events_same_count": 0, "hits": 0, "exact": {}, "max_abs": {}})
        valid = np.asarray(d["valid"])[:n]
        for i in range(n):
            ns = len(stock[pre + "sid"][lo + i])
            m = valid[i]
            r["events"] += 1
            if int(m.sum()) != ns:
                continue
            r["events_same_count"] += 1
            r["hits"] += ns
            for k in ("sid", "flag", "time", "edep", "x", "y", "z", "wdist", "uvar") + (("ncombo", "vvar") if tag == "panel" else ()):
                dv = np.asarray(d[k])[i][m]
                sv = np.asarray(stock[pre + k][lo + i])
                if k == "flag":
                    dv, sv = dv.astype(np.uint32), sv.astype(np.uint32)
                    eq = dv == sv
                else:
                    eq = dv.astype(np.float32) == sv.astype(np.float32) if k not in ("sid", "ncombo") else dv == sv
                e = r["exact"].setdefault(k, [0, 0])
                e[0] += int(eq.sum())
                e[1] += int(eq.size)
                if k not in ("sid", "flag", "ncombo") and eq.size:
                    r["max_abs"][k] = max(r["max_abs"].get(k, 0.0), float(np.max(np.abs(dv.astype(np.float64) - sv))))


def cmd_response(args):
    cfg = menu.parse_fcl(Path(args.menu).read_text())
    dp = menu.device_paths(cfg)
    stock = {}
    for log in args.logs:
        mods = pairmod.module_counts(Path(log).read_text())
        for name, p in dp.items():
            for role, lbl, _ in p["filters"]:
                if role in ("hs", "ks") and lbl in mods.get(name, {}):
                    t = stock.setdefault(name, {}).setdefault(role, [0, 0])
                    t[0] += mods[name][lbl]["passed"]
                    t[1] += mods[name][lbl]["visited"]
    dev = json.loads(Path(args.device).read_text())["stages"]
    out = {}
    for name, st in sorted(stock.items()):
        row = {"stock": st}
        d = dev.get(name, {})
        if "ks" in st and st["ks"][1] and d.get("hs"):
            e_stock, e_dev = st["ks"][0] / st["ks"][1], d["ks_raw"] / d["hs"]
            row["ks"] = min(1.0, e_stock / e_dev) if e_dev > 0 else 1.0
            row["ks_basis"] = {"stock": e_stock, "device_cuts": e_dev}
        if dp[name]["family"] == "apr" and "hs" in st and st["hs"][1] and d.get("tc"):
            e_stock, e_dev = st["hs"][0] / st["hs"][1], d["hs_raw"] / d["tc"]
            row["hs"] = min(1.0, e_stock / e_dev) if e_dev > 0 else 1.0
            row["hs_basis"] = {"stock": e_stock, "device": e_dev}
        out[name] = row
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def _merge(cnt, edges, min_count):
    """Merge bins with fewer than min_count device passes into their lower neighbour (the first into the next)."""
    e, c = list(edges), np.array(cnt, float)
    i = 1
    while i < len(c):
        if c[i, 1] < min_count:
            c[i - 1] += c[i]; c = np.delete(c, i, 0); del e[i]
        else:
            i += 1
    if len(c) > 1 and c[0, 1] < min_count:
        c[1] += c[0]; c = np.delete(c, 0, 0); del e[1]
    return e, c


def _f(c):
    return [float(min(1.0, s / d)) if d > 0 else 1.0 for s, d in c]


def cmd_response_binned(args):
    """Opt-in binned response: per path one pass probability f = stock passes / device passes of the events in a bin,
    capped at 1 (device run without a response, on the stock digis of the same events). Bins: the straw hits of the
    event's best time cluster of the path's producer, and for paths with a helix stage also the momentum of the event's
    best device helix (--edges-p rows). Sparse hit bins merge with their lower neighbour; a momentum row with fewer than
    --min-count device passes in all takes the path's one-dimensional (hits only) factors. Only paths matching --paths
    get binned rows; with --base (a scalar response from `response`), the binned rows are added to it and every other
    path keeps its scalar hs/ks factors."""
    sel = re.compile(args.paths)
    manifest = json.loads((Path(args.prepared) / "manifest.json").read_text())
    edges, edges_p = [float(e) for e in args.edges], [float(e) for e in args.edges_p]
    nb, npb = len(edges) - 1, len(edges_p) - 1
    acc = {}
    for dev_path, logs in zip(args.device, args.logs_sets):
        dev = json.loads(Path(dev_path).read_text())
        ref, _ = pairmod.read_arm([Path(x) for x in sorted(glob.glob(logs))], manifest)
        for key, passed in dev["decisions"].items():
            k = tuple(int(v) for v in key.split("|"))
            if k not in ref:
                continue
            for name, var in dev["path_vars"].get(key, {}).items():
                if name not in ref[k] or not sel.search(name):
                    continue
                nsh, hp = (var, -1.0) if not isinstance(var, list) else var
                b = min(max(int(np.searchsorted(edges, nsh, side="right")) - 1, 0), nb - 1)
                jp = min(max(int(np.searchsorted(edges_p, hp, side="right")) - 1, 0), npb - 1) if hp >= 0 else -1
                a = acc.setdefault(name, {"1d": np.zeros((nb, 2)), "2d": np.zeros((npb, nb, 2)), "helix": hp >= 0})
                st, dv = bool(ref[k][name]), name in passed
                a["1d"][b] += (st, dv)
                if jp >= 0:
                    a["2d"][jp, b] += (st, dv)
    out = json.loads(Path(args.base).read_text()) if args.base else {}
    for name, a in sorted(acc.items()):
        e1, c1 = _merge(a["1d"], edges, args.min_count)
        row = {"final": {"var": "tc_nsh", "edges": e1, "f": _f(c1)},
               "basis": {"stock_passes": c1[:, 0].tolist(), "device_passes": c1[:, 1].tolist()}}
        if a["helix"] and npb > 1:
            f2 = []
            for jp in range(npb):
                if a["2d"][jp, :, 1].sum() < args.min_count:
                    f2.append(_f(c1))                     # fall back to the hits-only factors
                    continue
                # the row's own hit bins, then mapped onto the common (1D merged) edges
                er, cr = _merge(a["2d"][jp], edges, args.min_count)
                fr = _f(cr)
                f2.append([fr[min(max(int(np.searchsorted(er, lo, side="right")) - 1, 0), len(fr) - 1)] for lo in e1[:-1]])
            row["final"] = {"var": "tc_nsh x helix_p", "edges": e1, "edges_p": edges_p, "f": f2}
            row["basis"]["rows_device_passes"] = a["2d"][:, :, 1].sum(1).tolist()
        out[name] = {**out.get(name, {}), **row}
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps({n: r["final"]["f"] for n, r in out.items() if "final" in r}, indent=1)[:4000])


def cmd_decide(args):
    cfg = menu.parse_fcl(Path(args.menu).read_text())
    cond = conditions(args.dump[0], args.tree)
    resp = json.loads(Path(args.response).read_text()) if args.response else {}
    ch = chainmod.Chain(cfg, cond, args.tc_mva, response=resp, debug=args.check_hits, precision=args.precision)
    report, device, stages, stage_flags, path_vars = {}, {}, {}, {}, {}
    t_dev = 0.0
    for dump in args.dump:
        keys, ev, stock = load_dump(dump, args.tree)
        outs, auxs = [], []
        for lo, n, b in batches(keys, ev, args.batch):
            t0 = time.perf_counter()
            out, aux, diag = ch.decide(b)
            out = {k: np.asarray(v)[:n] for k, v in out.items()}
            t_dev += time.perf_counter() - t0
            outs.append(out)
            auxs.append({k: np.asarray(v)[:n] for k, v in aux.items()})
            if args.check_hits:
                check_hits(diag, stock, lo, n, report)
        dec = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
        aux = {k: np.concatenate([a[k] for a in auxs]) for k in auxs[0]}
        dec = ch.finish(dec, aux, len(keys))
        for i, key in enumerate(keys):            # time-cluster straw hits per path (binned response variable)
            path_vars["|".join(map(str, key))] = {
                n: [int(aux[f"{n}:tcnsh"][i]), round(float(aux[f"{n}:help"][i]), 3) if f"{n}:help" in aux else -1.0]
                for n in ch.plan if f"{n}:tcnsh" in aux}
        for i, key in enumerate(keys):            # deterministic stage results (before response-layer draws)
            stage_flags["|".join(map(str, key))] = {
                st: sorted(n for n in ch.plan if f"{n}:{st}" in aux and aux[f"{n}:{st}"][i])
                for st in ("tc", "hs_raw", "ks_raw")}
        for k, v in aux.items():
            name, _, stage = k.partition(":")
            if stage in ("tc", "hs_raw", "hs", "ks_raw"):
                stages.setdefault(name, {}).setdefault(stage, 0)
                stages[name][stage] += int(np.sum(v))
        for i, key in enumerate(keys):
            if key in device:
                raise ValueError(f"event {key} in two dumps")
            device[key] = {p: bool(v[i]) for p, v in dec.items()}
    result = {"dumps": [str(d) for d in args.dump], "events": len(device), "paths": sorted(ch.plan),
              "precision": ch.precision, "skipped": ch.skipped, "response": resp, "stages": stages, "hits": report,
              "wall_s_including_compile": t_dev}
    if args.logs:
        manifest = json.loads((Path(args.prepared) / "manifest.json").read_text())
        inventory = json.loads((Path(args.prepared) / "inventory.json").read_text())
        ref = pairmod.read_arm([Path(p) for p in args.logs], manifest)
        missing = set(ref[0]) ^ set(device)
        if missing:
            raise ValueError(f"{len(missing)} events differ between the dumps and the replay logs")
        keep = set(ch.plan)
        sub = {"paths": [p for p in manifest["paths"] if p in keep],
               "bits": [b for p, b in zip(manifest["paths"], manifest["bits"]) if p in keep],
               "stateful_paths": [p for p in manifest.get("stateful_paths", []) if p in keep]}
        paired = pairmod.pair(("stock", (ref[0], ref[1])), ("device", (device, {})), sub, inventory,
                              event_mode=args.event_mode)
        result["paired"] = paired
        exact = {p: r["stock_only"] + r["device_only"] for p, r in paired["paths"].items()}
        result["discordant"] = exact
    result["stage_flags"] = stage_flags
    result["path_vars"] = path_vars
    result["decisions"] = {"|".join(map(str, k)): [p for p, v in sorted(d.items()) if v] for k, d in device.items()}
    Path(args.output).write_text(json.dumps(result, indent=1, default=float))
    if "paired" in result:
        for p, r in sorted(result["paired"]["paths"].items()):
            print(f"{p:32s} stock {r['stock_acceptance']:.4f} device {r['device_acceptance']:.4f} "
                  f"diff {100 * r['paired_difference']:+.2f} +- {100 * r['paired_error']:.2f} pp "
                  f"pull {r['pull'] if r['pull'] is None else round(r['pull'], 2)} discordant {result['discordant'][p]}")
    print(json.dumps({"hits": report}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("response")
    k.add_argument("--logs", nargs="+", required=True)
    k.add_argument("--device", required=True)
    k.add_argument("--menu", required=True)
    k.add_argument("--output", required=True)
    rb = sub.add_parser("response-binned", help="opt-in per-path pass probability binned in the time-cluster straw hits")
    rb.add_argument("--device", nargs="+", required=True, help="decide outputs without a response, on stock digis")
    rb.add_argument("--logs-sets", nargs="+", required=True, help="replay-log glob of each --device run, in order")
    rb.add_argument("--prepared", required=True)
    rb.add_argument("--edges", nargs="+", default=[0, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 1000])
    rb.add_argument("--edges-p", nargs="+", default=[0, 90, 98, 101, 1e6], help="helix-momentum rows [MeV/c]")
    rb.add_argument("--min-count", type=int, default=200)
    rb.add_argument("--paths", default=r"^apr_(Trk|Hlx)", help="regex of the paths that get binned rows")
    rb.add_argument("--base", help="scalar response (from `response`) that the binned rows are added to")
    rb.add_argument("--output", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--dump", nargs="+", required=True)
    d.add_argument("--tree", default="trigph")
    d.add_argument("--menu", required=True)
    d.add_argument("--tc-mva", required=True)
    d.add_argument("--response")
    d.add_argument("--logs", nargs="*")
    d.add_argument("--prepared")
    d.add_argument("--event-mode", default="OffSpill")
    d.add_argument("--batch", type=int, default=512)
    d.add_argument("--precision", choices=chainmod.PRECISIONS, default="float64")
    d.add_argument("--check-hits", action="store_true")
    d.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    {"response": cmd_response, "response-binned": cmd_response_binned, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    main()
