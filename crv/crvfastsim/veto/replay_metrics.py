"""Paired method-F and strongest-pulse-per-channel diagnostics on verified mixed-event replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np

from .card import VetoCard
from .compareframes import COLS, dump_frames
from .deadtime import frame_coverage, read_coincs
from .frameaudit import track_histogram
from .replay import sha256
from .replay_policy import metadata_identity
from ..reco import reftables


def paired_stats(fast, stock):
    """Conditional paired-frame errors, including empty frames; no finite-library uncertainty."""
    fast, stock = np.asarray(fast, float), np.asarray(stock, float)
    if fast.shape != stock.shape or fast.ndim != 1 or not len(fast):
        raise ValueError("equal nonempty coverage vectors required")
    if not np.isfinite(fast).all() or not np.isfinite(stock).all():
        raise ValueError("nonfinite coverage")
    if np.any((fast < 0) | (fast > 1) | (stock < 0) | (stock > 1)):
        raise ValueError("coverage outside [0,1]")
    n = len(fast)
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else None
    delta = fast - stock
    return {"frames": n, "F_fast": float(fast.mean()), "F_stock": float(stock.mean()),
            "F_fast_err": se(fast), "F_stock_err": se(stock), "delta": float(delta.mean()),
            "delta_err_paired": se(delta), "delta_sum": float(delta.sum()),
            "delta_sq_sum": float(delta @ delta), "fast_sum": float(fast.sum()),
            "stock_sum": float(stock.sum()), "fast_sq_sum": float(fast @ fast),
            "stock_sq_sum": float(stock @ stock)}


def verify_metadata(check, meta, metadata_sha256):
    """Bind the checked payload to its sidecar and validated response identity."""
    required = ("fast_metadata_sha256", "conditions_sha256", "response_identity",
                "response_identity_sha256", "legacy_policy_recovery")
    if any(k not in check for k in required):
        raise ValueError("check lacks sidecar-bound response identity; rerun replay_check on retained outputs")
    if check["fast_metadata_sha256"] != metadata_sha256:
        raise ValueError("fast metadata hash mismatch; rerun replay_check on retained outputs")
    if not check.get("input_sha256") or meta.get("input_sha256") != check["input_sha256"]:
        raise ValueError("fast metadata input identity differs from check")
    if meta.get("conditions_sha256") != check["conditions_sha256"]:
        raise ValueError("fast metadata conditions identity differs from check")
    recorded = check["response_identity"]
    if not isinstance(recorded, dict) or not isinstance(recorded.get("response_contract"), dict):
        raise ValueError("invalid checked response identity")
    legacy_policy = recorded["response_contract"].get("policy") if check["legacy_policy_recovery"] is not None else None
    identity, digest, recovery = metadata_identity(meta, legacy_policy=legacy_policy)
    if identity != recorded or digest != check["response_identity_sha256"]:
        raise ValueError("response identity differs from checked fast metadata")
    if recovery != check["legacy_policy_recovery"]:
        raise ValueError("legacy policy recovery differs from check")
    return identity, digest, recovery


def _count(record, key, actual, label):
    """Require exact nonnegative integer counts, not booleans or rounded values."""
    value = record.get(key)
    if type(value) is not int or value < 0 or value != actual:
        raise ValueError(f"{label} {key} count differs from saved frames/coincidences")


def validate_frames(fast, stock, check, meta):
    """Validate full paired denominators and finite coincidence fields before coverage."""
    n = len(fast)
    if not n or len(stock) != n:
        raise ValueError("equal nonempty paired frame counts required")
    _count(check.get("events_check", {}), "events", n, "checked")
    _count(check.get("entries_highest_cycle", {}), "events", n, "checked entries")
    _count(check.get("comparison", {}), "common_frames", n, "checked comparison")
    _count(meta, "frames", n, "sidecar")
    totals = []
    for name, frames in (("fast", fast), ("stock", stock)):
        ids = np.column_stack([ak.to_numpy(frames[k]) for k in ("run", "subrun", "event")])
        if len(np.unique(ids, axis=0)) != n:
            raise ValueError(f"duplicate {name} frame identity")
        co = frames.coincs
        counts = ak.to_numpy(ak.num(co.time))
        for field in COLS:
            if not np.array_equal(ak.to_numpy(ak.num(co[field])), counts):
                raise ValueError(f"inconsistent {name} coincidence lengths: {field}")
            if not np.isfinite(ak.to_numpy(ak.flatten(co[field]), allow_missing=False)).all():
                raise ValueError(f"nonfinite {name} coincidence field: {field}")
        total = int(counts.sum())
        summary = check.get("comparison", {}).get(name, {})
        _count(summary, "frames", n, f"checked {name}")
        _count(summary, "coincs", total, f"checked {name}")
        totals.append(total)
    for key in ("run", "subrun", "event"):
        if not np.array_equal(ak.to_numpy(fast[key]), ak.to_numpy(stock[key])):
            raise ValueError("paired frame identities differ")
    _count(meta, "coincs", totals[0], "sidecar")
    _count(check.get("events_check", {}), "coincs", totals[1], "checked stock")
    _count(check.get("entries_highest_cycle", {}), "clusters", totals[1], "checked stock entries")
    return totals


def validate_pool(rows):
    """Pool distinct inputs only under one complete conditions/photon/timing/yield identity."""
    if not rows:
        raise ValueError("no replay blocks")
    if len({r["input_sha256"] for r in rows}) != len(rows):
        raise ValueError("replicas of one input cannot be pooled as independent frames")
    if any(r["response_identity"] != rows[0]["response_identity"] or
           r["response_identity_sha256"] != rows[0]["response_identity_sha256"] for r in rows[1:]):
        raise ValueError("cannot pool different response identities (conditions, photon-dev, timing or yields)")


def strongest(p):
    """Vectorised maximum-PE pulse per (event key, channel), independent of pulse order."""
    key = np.asarray(p["ev"], np.int64) * 8192 + np.asarray(p["chan"], np.int64)
    order = np.lexsort((-np.asarray(p["pe"]), key))
    _, first = np.unique(key[order], return_index=True)
    return {k: np.asarray(v)[order][first] for k, v in p.items()} | {"key": key[order][first]}


def pulse_compare(fast, stock):
    """Observed-channel matching, not truth matching; T1/T2 bounds are Run1A-specific."""
    f, s = strongest(fast), strongest(stock)
    common, fi, si = np.intersect1d(f["key"], s["key"], return_indices=True)
    q = lambda v: np.percentile(v, [10, 50, 90]).tolist() if len(v) else []
    out = {"fast_pulses": len(fast["ev"]), "stock_pulses": len(stock["ev"]),
           "common_channels": len(common), "fast_only_channels": len(f["key"]) - len(common),
           "stock_only_channels": len(s["key"]) - len(common), "sectors": {}}
    for name, lo, hi in (("T1", 0, 4352), ("T2", 4352, 5120)):
        fm = (f["chan"] >= lo) & (f["chan"] < hi)
        sm = (s["chan"] >= lo) & (s["chan"] < hi)
        both = sm[si]
        positive = both & (s["pe"][si] > 0)
        out["sectors"][name] = {"fast_channels": int(fm.sum()), "stock_channels": int(sm.sum()),
            "common_channels": int(both.sum()), "fast_pe_sum": float(f["pe"][fm].sum()),
            "stock_pe_sum": float(s["pe"][sm].sum()),
            "matched_dt_q10_50_90": q(f["time"][fi][both] - s["time"][si][both]),
            "matched_pe_ratio_q10_50_90": q(f["pe"][fi][positive] / s["pe"][si][positive])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checks", nargs="+", required=True, help="successful replay_check reports")
    ap.add_argument("--fast", nargs="+", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() or len(a.checks) != len(a.fast) or len(set(a.checks)) != len(a.checks):
        raise ValueError("new report and matching lists required")
    ref = json.loads(Path(a.reference).read_text())
    config = dict(ref["card"])
    expected_sha = config.pop("sha")
    card = VetoCard.from_dict(config)
    if card.sha() != expected_sha:
        raise ValueError("card hash mismatch")
    ev, _ = read_coincs([a.selection])
    times, mom = ak.to_numpy(ak.flatten(ev.t_mid)), ak.to_numpy(ak.flatten(ev.p_mid))
    sel = np.ones(len(times), bool)
    if ref.get("pmin") is not None:
        sel &= mom > ref["pmin"]
    if ref.get("pmax") is not None:
        sel &= mom < ref["pmax"]
    if int(sel.sum()) != ref["selected"]:
        raise ValueError("track selection count mismatch")
    weights, edges = track_histogram(times[sel], ref["F"]["grid_ns"])
    rows, f_all, s_all = [], [], []
    for check_path, fast_path in zip(a.checks, a.fast):
        check = json.loads(Path(check_path).read_text())
        meta_path = Path(fast_path).with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        meta_sha = sha256(meta_path)
        identity, digest, recovery = verify_metadata(check, meta, meta_sha)
        stock_path = check["stock"]
        if check.get("conditions_equal") is not True or sha256(stock_path) != check["stock_sha256"] or sha256(fast_path) != check["fast_sha256"]:
            raise ValueError("verified replay payload changed")
        fast, stock = ak.from_parquet(fast_path), dump_frames([stock_path])
        fast_coincs, stock_coincs = validate_frames(fast, stock, check, meta)
        fc, sc = frame_coverage(fast.coincs, card, weights, edges), frame_coverage(stock.coincs, card, weights, edges)
        row = paired_stats(fc, sc) | {"check": check_path, "check_sha256": sha256(check_path),
            "input_sha256": check["input_sha256"], "fast_sha256": check["fast_sha256"],
            "stock_sha256": check["stock_sha256"], "fast_metadata_sha256": meta_sha,
            "conditions_sha256": check["conditions_sha256"], "response_identity": identity,
            "response_identity_sha256": digest, "legacy_policy_recovery": recovery,
            "stock_shared_table_yields_diagnostic": meta.get("stock_shared_table_yields_diagnostic", False),
            "fast_coincs": fast_coincs, "stock_coincs": stock_coincs}
        if meta.get("pulse_file"):
            if sha256(meta["pulse_file"]) != meta["pulse_sha256"]:
                raise ValueError("pulse file hash mismatch")
            fp = ak.from_parquet(meta["pulse_file"])
            sp = reftables.read([stock_path], "pulses", ["subrun", "event", "bar", "sipm", "time", "pe"])
            s = {"ev": (ak.to_numpy(sp.subrun).astype(np.int64) << 20) | ak.to_numpy(sp.event).astype(np.int64),
                 "chan": ak.to_numpy(sp.bar) * 4 + ak.to_numpy(sp.sipm), "pe": ak.to_numpy(sp.pe), "time": ak.to_numpy(sp.time)}
            row["pulses"] = pulse_compare({k: ak.to_numpy(fp[k]) for k in ("ev", "chan", "pe", "time")}, s)
        rows.append(row)
        f_all.append(fc)
        s_all.append(sc)
    validate_pool(rows)
    pooled = paired_stats(np.concatenate(f_all), np.concatenate(s_all))
    g, total = len(rows), pooled["frames"]
    pooled["delta_err_blocks"] = float(np.sqrt(g / (g - 1) * sum(
        (r["delta_sum"] - r["frames"] * pooled["delta"])**2 for r in rows)) / total) if g > 1 else None
    result = {"pooled": pooled, "blocks": rows, "card": config, "selected": int(sel.sum()),
              "selection": a.selection, "selection_sha256": sha256(a.selection),
              "reference": a.reference, "reference_sha256": sha256(a.reference), "grid_ns": ref["F"]["grid_ns"],
              "response_identity": rows[0]["response_identity"], "response_identity_sha256": rows[0]["response_identity_sha256"],
              "scope": "Errors conditional on selected tracks and sampled Cat pools; disjoint files do not prove independent showers. Pulse summaries use strongest observed pulse per channel, not truth matching."}
    from mu2efastsim.plots import write_json
    write_json(out.parent, out.name, result)
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
