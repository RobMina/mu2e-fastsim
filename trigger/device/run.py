"""Device trigger CLI.

  python3 -m trigger.device.run decide --dump a/trgdump.root ... --menu replay.fcl --tc-mva TimeCluster.weights.xml \
        [--response ce|dio|rpc|none|FILE] [--logs a/replay.log ... --prepared DIR] --output result.json [--check-hits]
  python3 -m trigger.device.run response --logs dev/*/replay.log --device dev.json --menu replay.fcl --output response.json

`decide` runs the device chain on the digis in StrawHitTableDumper files (one per replay job, in event order), checks
the device straw/panel hits against the stock trigger hits in the same file (`--check-hits`), and, with `--logs`,
pairs the per-path decisions with the stock replay decisions of the same events (trigger/hybrid/pair.py statistics).
`response` builds the response layer (decision D1) from development shards only: per path, the stock KalSeedFilter
pass fraction (passed / visited) over the device's pass fraction of the same cuts on helix quantities, and for the apr paths, whose helix finder (AgnosticHelixFinder)
the device replaces with its own fit, the ratio of the stock to the device HelixFilter pass fraction (capped at 1).
Build from decide runs with `--response none`. `decide --response` takes a named response (trigger/device/response.py:
ce and dio, the default, or rpc; the shipped binned files) or a file.
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
import jax
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare  # noqa: E402
from hybrid import pair as pairmod  # noqa: E402

from trkfastsim.reco import hitreco  # noqa: E402

from . import chain as chainmod, hits, menu  # noqa: E402
from .response import APR_HELIX_PATHS, DEFAULT as RESPONSE_DEFAULT, NAMES as RESPONSE_NAMES, binned_rows  # noqa: E402
from .response import load as load_response  # noqa: E402

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


def batches(keys, ev, size, hmin=64):
    """Blocks of `size` events in input order: yields (first event, n, block); the hit axis is padded to a power of
    two >= hmin (hits.pack_events)."""
    for lo in range(0, len(keys), size):
        sl = slice(lo, lo + size)
        sub = {k: (v[sl] if isinstance(v, list) else v[sl]) for k, v in ev.items()}
        n = len(sub["sid"])
        b = hits.pack_events(sub, hmin=hmin)
        E = size
        if n < E:                                   # pad events to the fixed batch size (one compiled shape)
            for k, v in b.items():
                pad = np.zeros((E - n,) + v.shape[1:], v.dtype)
                b[k] = np.concatenate([v, pad])
        b["valid_event"] = np.arange(E) < n
        b["event"] = np.array([k[2] for k in keys[sl]] + [0] * (E - n), np.int64)
        yield lo, n, b


def hit_order(ev):
    """Event order by digi count (stable): batches of events of similar size pad the hit axis less."""
    return np.argsort(np.array([len(s) for s in ev["sid"]], np.int64), kind="stable")


def sorted_batches(keys, ev, size, order=None, hmin=64):
    """As `batches`, over the events in `order` (default `hit_order`): yields (rows, n, block), rows = the input
    index of each of the block's n events. Every event's result is the same as in `batches` (each event is computed
    on its own hits; only the padding of the hit axis differs); put results back with `unsort`."""
    order = hit_order(ev) if order is None else np.asarray(order)
    ev_o = {k: ([v[i] for i in order] if isinstance(v, list) else np.asarray(v)[order]) for k, v in ev.items()}
    keys_o = [keys[i] for i in order]
    for lo, n, b in batches(keys_o, ev_o, size, hmin):
        yield order[lo:lo + n], n, b


def unsort(rows, parts):
    """Per-event arrays computed in sorted order (parts, one dict per block, `rows` the blocks' input indices)
    -> the same arrays in input order."""
    rows = np.concatenate(rows)
    out = {}
    for k in parts[0]:
        v = np.concatenate([p[k] for p in parts])
        o = np.empty_like(v)
        o[rows] = v
        out[k] = o
    return out


# per-backend batching defaults of `decide` and the bench (bit-identical decisions at every setting): on a CPU (one
# core per process, as a grid slot) small batches, a low hit-axis floor and the compact cluster path (one CPU core,
# f32 358 -> 587, f64 204 -> 387 events/s against batch 64 and the one-call path); on a GPU the one-call path (the
# compact path's host reads would stall the device) and the tool's own batch size
CPU_BATCHING = {"batch": 16, "hmin": 32, "compact": True}


def on_cpu():
    import jax
    return jax.default_backend() == "cpu"


def check_hits(diag, stock, rows, report):
    """Device straw/panel hits vs the stock TrigMakeSH/TrigMakePH of the same events (rows: each block event's input
    index)."""
    n = len(rows)
    for tag, d, pre in (("straw", diag["sh"], "s_"), ("panel", diag["ph"], "p_")):
        r = report.setdefault(tag, {"events": 0, "events_same_count": 0, "hits": 0, "exact": {}, "max_abs": {}})
        valid = np.asarray(d["valid"])[:n]
        for i in range(n):
            ns = len(stock[pre + "sid"][rows[i]])
            m = valid[i]
            r["events"] += 1
            if int(m.sum()) != ns:
                continue
            r["events_same_count"] += 1
            r["hits"] += ns
            for k in ("sid", "flag", "time", "edep", "x", "y", "z", "wdist", "uvar") + (("ncombo", "vvar") if tag == "panel" else ()):
                dv = np.asarray(d[k])[i][m]
                sv = np.asarray(stock[pre + k][rows[i]])
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


def _dev_run(path):
    """A decide output of a development run; it must have no response layer (decide --response none)."""
    dev = json.loads(Path(path).read_text())
    if dev.get("response"):
        raise SystemExit(f"{path}: decide ran with a response layer ({dev.get('response_spec', 'a file')}); the "
                         "development runs need --response none")
    return dev


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
    dev = _dev_run(args.device)["stages"]
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
        dev = _dev_run(dev_path)
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
    out = binned_rows(acc, edges, edges_p, args.min_count, json.loads(Path(args.base).read_text()) if args.base else None)
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps({n: r["final"]["f"] for n, r in out.items() if "final" in r}, indent=1)[:4000])


def cmd_decide(args):
    cfg = menu.parse_fcl(Path(args.menu).read_text())
    cond = conditions(args.dump[0], args.tree)
    resp = load_response(args.response)
    cpu = CPU_BATCHING if on_cpu() else {}
    batch = args.batch if args.batch is not None else cpu.get("batch", 512)
    hmin = args.hmin if args.hmin is not None else cpu.get("hmin", 64)
    compact = args.compact if args.compact is not None else cpu.get("compact", False)
    ch = chainmod.Chain(cfg, cond, args.tc_mva, response=resp, debug=args.check_hits, precision=args.precision,
                        compact=compact)
    report, device, stages, stage_flags, path_vars = {}, {}, {}, {}, {}
    t_dev = 0.0
    for dump in args.dump:
        keys, ev, stock = load_dump(dump, args.tree)
        outs, auxs, rows_all = [], [], []
        blocks = (sorted_batches(keys, ev, batch, hmin=hmin) if args.order == "hits" else
                  ((np.arange(lo, lo + n), n, b) for lo, n, b in batches(keys, ev, batch, hmin)))
        for rows, n, b in blocks:
            t0 = time.perf_counter()
            out, aux, diag = ch.decide(b)
            out, aux = jax.device_get((out, aux))        # one host transfer per block
            out = {k: np.asarray(v)[:n] for k, v in out.items()}
            t_dev += time.perf_counter() - t0
            outs.append(out)
            auxs.append({k: np.asarray(v)[:n] for k, v in aux.items()})
            rows_all.append(rows)
            if args.check_hits:
                check_hits(diag, stock, rows, report)
        dec, aux = unsort(rows_all, outs), unsort(rows_all, auxs)   # event order (prescale counter)
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
              "precision": ch.precision, "batching": {"order": args.order, "batch": batch, "hmin": hmin,
                                                      "compact": compact}, "skipped": ch.skipped, "response_spec": args.response, "response": resp, "stages": stages, "hits": report,
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
    result["approximations"] = args.approximations_record
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
    rb.add_argument("--device", nargs="+", required=True, help="decide outputs with --response none, on stock digis")
    rb.add_argument("--logs-sets", nargs="+", required=True, help="replay-log glob of each --device run, in order")
    rb.add_argument("--prepared", required=True)
    rb.add_argument("--edges", nargs="+", default=[0, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 1000])
    rb.add_argument("--edges-p", nargs="+", default=[0, 90, 98, 101, 1e6], help="helix-momentum rows [MeV/c]")
    rb.add_argument("--min-count", type=int, default=200)
    rb.add_argument("--paths", default=APR_HELIX_PATHS, help="regex of the paths that get binned rows")
    rb.add_argument("--base", help="scalar response (from `response`) that the binned rows are added to")
    rb.add_argument("--output", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--dump", nargs="+", required=True)
    d.add_argument("--tree", default="trigph")
    d.add_argument("--menu", required=True)
    d.add_argument("--tc-mva", required=True)
    d.add_argument("--response", default=RESPONSE_DEFAULT,
                   help=f"response layer: a named response ({', '.join(RESPONSE_NAMES)}; default {RESPONSE_DEFAULT}: the "
                        "binned CE/DIO response; rpc for RPC events; none for the development runs of response and "
                        "response-binned) or a response JSON file")
    d.add_argument("--logs", nargs="*")
    d.add_argument("--prepared")
    d.add_argument("--event-mode", default="OffSpill")
    d.add_argument("--batch", type=int, help="events per block (default 512; on a CPU 16)")
    d.add_argument("--hmin", type=int, help="hit-axis floor of a block (default 64; on a CPU 32)")
    d.add_argument("--order", choices=("hits", "input"), default="hits",
                   help="hits: blocks of events of similar digi count, results put back in event order (default; the "
                        "same decisions bit for bit); input: the dumps' order")
    d.add_argument("--precision", choices=chainmod.PRECISIONS, default="float64")
    d.add_argument("--compact", action=argparse.BooleanOptionalAction, default=None,
                   help="time-cluster refinement and helix fits on the occupied (event, cluster) pairs only (three "
                        "device calls per batch; the same decisions; default on a CPU, off on a GPU)")
    d.add_argument("--check-hits", action="store_true")
    d.add_argument("--output", required=True)
    from mu2efastsim import approximations
    from ..approximations import WORKFLOW
    approximations.add_arguments(d)
    args = ap.parse_args(argv)
    if args.cmd == "decide":
        try:
            args.approximations_record = approximations.from_args(args, WORKFLOW).record()
        except ValueError as e:
            d.error(str(e))
    {"response": cmd_response, "response-binned": cmd_response_binned, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    main()
