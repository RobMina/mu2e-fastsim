"""Decide the calorimeter trigger paths on the device from stock art files (CaloDigis + ProtonBunchTime), and
compare them with the TriggerResults and CaloClusters stored in the same files.

    python -m trigger.device.calorun --art F.art [...] --menu replay.fcl --templates SIG BKG --nn-weights W.xml
        [--production-config expanded.fcl] [--precision float64|float32] [--batch 128] --output out.json

`--production-config` is the configuration of the job that wrote the files (fhicl-dump, or config_dumper -P output) (its trigger_paths order the
TriggerResults words); with it the output has, per path, the stock and device pass counts and the discordant events,
and the event count whose device clusters differ from the stored stock clusters in any field.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np

from . import calo, caloart

CLUSTER_FIELDS = ("disk", "time", "energy", "cog_x", "cog_y", "size")


def _block(dg, b0, b1):
    sl = slice(int(np.sum(dg["n"][:b0])), int(np.sum(dg["n"][:b1])))
    sub = {k: dg[k][sl] for k in ("sipm", "t0", "peakpos", "wf_len", "wf_start")}
    sub["entry"] = dg["entry"][sl] - b0
    sub["samples"] = dg["samples"]
    sub["n"] = dg["n"][b0:b1]
    pk = calo.pack_digis(sub, b1 - b0)
    D = pk["sipm"].shape[1]
    Dp = max(16, 1 << (D - 1).bit_length())                # bucketed digi axis: few compilations
    if Dp != D:
        for k, v in list(pk.items()):
            if isinstance(v, np.ndarray) and v.ndim == 2:
                pk[k] = np.pad(v, ((0, 0), (0, Dp - D)))
    return pk


def _cluster_diff(cl, stock, b0, n):
    """Events of the block whose device clusters differ from the stored stock clusters (count or any field)."""
    start = np.r_[0, np.cumsum(stock["n"])]
    bad = []
    for i in range(n):
        e = b0 + i
        c0, c1 = start[e], start[e + 1]
        k = int(cl["n"][i])
        if k != c1 - c0:
            bad.append((e, "n"))
            continue
        for f in CLUSTER_FIELDS:
            a = cl[f][i, :k]
            b = stock[f][c0:c1]
            if f in ("time", "energy"):
                a, b = a.astype(np.float32), b.astype(np.float32)
            if not np.array_equal(a, b):
                bad.append((e, f))
                break
    return bad


def run(args):
    import jax
    jax.config.update("jax_enable_x64", args.precision == "float64")
    from trigger import replay
    from . import calopaths, menu
    cfg = menu.parse_fcl(open(args.menu).read())
    stage = calopaths.CaloStage(cfg, tuple(args.templates) if args.templates else None, args.nn_weights)
    names = None
    if args.production_config:
        text = open(args.production_config).read()
        prod = (replay.production_config(text, args.process) if re.search(rf"(?m)^{args.process}\s*:\s*{{", text)
                else replay.parse_expanded(text))
        names = [n for _, n, _ in replay.paths(prod)]
    report = {"precision": args.precision, "paths": sorted(stage.paths), "files": []}
    totals = {p: {"stock": 0, "device": 0, "discordant": 0} for p in stage.paths}
    for path in args.art:
        t0 = time.time()
        data = caloart.read(path, process=args.process if names else None,
                            clusters="CaloClusterFast__" + args.process if names else None)
        n_ev = len(data["digis"]["n"])
        dec = {p: np.zeros(n_ev, bool) for p in stage.paths}
        diag = {}
        bad = []
        for b0 in range(0, n_ev, args.batch):
            b1 = min(n_ev, b0 + args.batch)
            out, dg, cl = stage.decide(_block(data["digis"], b0, b1), data["pbt"][b0:b1], data["ids"][b0:b1, 2],
                                       np.full(b1 - b0, calopaths.spill_code(args.event_mode)))
            for p, v in out.items():
                dec[p][b0:b1] = np.asarray(v)
            for k, v in dg.items():
                if not k.startswith("n_"):
                    diag[k] = diag.get(k, 0) + int(np.sum(np.asarray(v)))
            if names:
                bad += _cluster_diff({k: np.asarray(v) for k, v in cl.items()}, data["clusters"], b0, b1 - b0)
        row = {"file": path, "events": n_ev, "seconds": time.time() - t0, "flags": diag}
        if names:
            acc = caloart.accepted(data["trigger"])
            if acc.shape[1] != len(names):
                raise ValueError("TriggerResults path count differs from the production configuration")
            row["cluster_mismatch_events"] = len({e for e, _ in bad})
            row["cluster_mismatch_examples"] = [(data["ids"][e].tolist(), f) for e, f in bad[:5]]
            row["paths"] = {}
            for p, v in dec.items():
                s = acc[:, names.index(p)]
                d = np.flatnonzero(s != v)
                row["paths"][p] = {"stock": int(s.sum()), "device": int(v.sum()), "discordant": int(d.size),
                                   "discordant_ids": [data["ids"][i].tolist() for i in d[:10]]}
                for k in ("stock", "device", "discordant"):
                    totals[p][k] += row["paths"][p][k]
        else:
            row["decisions"] = {p: np.flatnonzero(v).tolist() for p, v in dec.items()}
        report["files"].append(row)
        print(json.dumps({k: row[k] for k in ("file", "events", "seconds", "flags")}
                         | ({"cluster_mismatch_events": row["cluster_mismatch_events"]} if names else {})), flush=True)
    if names:
        report["totals"] = totals
        for p, t in totals.items():
            print(p, t)
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--art", nargs="+", required=True)
    ap.add_argument("--menu", required=True, help="replay configuration (fhicl-dump) with the calo paths")
    ap.add_argument("--templates", nargs=2, metavar=("SIGNAL", "BACKGROUND"), help="CaloLikelihood templates")
    ap.add_argument("--nn-weights", help="CaloNNEval TMVA weights (XML)")
    ap.add_argument("--production-config", help="expanded configuration of the job that wrote the files")
    ap.add_argument("--process", default="Digitize")
    ap.add_argument("--event-mode", default="OnSpill", choices=("OnSpill", "OffSpill"))
    ap.add_argument("--precision", default="float64", choices=("float64", "float32"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--output", required=True)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
