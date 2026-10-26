"""Verify a stock replay's exact event/conditions contract before comparing its coincidences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from ..reco import reftables
from .compareframes import compare, dump_frames
from .replay import sha256
from .replay_policy import POLICIES, metadata_identity


def check_events(expected, actual, count, window=1700.0):
    """Require the requested input prefix, including empty events, and float32 stock bunch times."""
    if count <= 0 or count > len(expected) or len(actual) != count:
        raise ValueError("wrong completed event count")
    ids = np.column_stack([np.asarray(expected[k]) for k in ("run", "subrun", "event")])
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError("duplicate input event ID")
    for key in ("run", "subrun", "event"):
        if not np.array_equal(np.asarray(actual[key]), np.asarray(expected[key])[:count]):
            raise ValueError(f"event ID mismatch: {key}")
    target = np.asarray(expected["pbtime"])[:count].astype(np.float32).astype(float)
    if not np.array_equal(np.asarray(actual["pbtime"]), target):
        raise ValueError("bunch time mismatch")
    if not np.all(np.asarray(actual["spill"]) == 1) or not np.all(np.asarray(actual["evlen"]) == window):
        raise ValueError("event-window mismatch")
    return {"events": count, "spill": 1, "window_ns": window,
            "pbtime_values": np.unique(target).tolist(),
            "digis": int(np.sum(actual["ndigi"])), "pulses": int(np.sum(actual["npulse"])),
            "coincs": int(np.sum(actual["ncluster"]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--events", required=True, type=int)
    ap.add_argument("--conditions", required=True, help="conditions dump used by fast arm")
    ap.add_argument("--fast", help="full fast replay frames (only for complete stock sample)")
    ap.add_argument("--legacy-yield-policy", choices=POLICIES, help="explicitly recover recognized historical sidecars without modifying them")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise ValueError("report already exists")
    manifest = json.loads(Path(a.input).with_suffix(".json").read_text())
    if sha256(a.input) != manifest["output_sha256"]:
        raise ValueError("input hash mismatch")
    with uproot.open(a.input) as f:
        ev = f["dump/events"].arrays(library="np")
    with uproot.open(a.stock) as f:
        stock = f["crvdump/events"].arrays(library="np")
        entries = {k: f[f"crvdump/{k}"].num_entries for k in ("events", "digis", "pulses", "clusters", "calib")}
    expected = {k: ev[k] for k in ev.dtype.names} if hasattr(ev, "dtype") and ev.dtype.names else ev
    n_expected = len(expected["event"])
    summary = check_events(ak.Array(expected), ak.Array(stock), a.events)
    if entries["digis"] != summary["digis"] or entries["pulses"] != summary["pulses"] or entries["clusters"] != summary["coincs"]:
        raise ValueError("product rows disagree with event counters")
    c1, c2 = reftables.calib([a.conditions]), reftables.calib([a.stock])
    differences = {k: not np.array_equal(c1[k], c2[k]) for k in c1}
    if any(differences.values()):
        raise ValueError(f"conditions differ from fast arm: {differences}")
    result = {"input_sha256": manifest["output_sha256"], "stock": a.stock,
              "stock_sha256": sha256(a.stock), "checker_sha256": sha256(__file__),
              "events_check": summary, "entries_highest_cycle": entries, "conditions_equal": True,
              "conditions_sha256": sha256(a.conditions), "input_manifest_sha256": sha256(Path(a.input).with_suffix(".json"))}
    if a.fast:
        if a.events != n_expected:
            raise ValueError("paired comparison requires all input events")
        metadata_path = Path(a.fast).with_suffix(".json")
        fast_meta = json.loads(metadata_path.read_text())
        identity, digest, recovery = metadata_identity(fast_meta, a.legacy_yield_policy)
        result.update(fast_metadata_sha256=sha256(metadata_path), response_identity=identity,
                      response_identity_sha256=digest, legacy_policy_recovery=recovery)
        if fast_meta["input_sha256"] != manifest["output_sha256"]:
            raise ValueError("fast arm used a different replay input")
        if fast_meta["conditions_sha256"] != sha256(a.conditions):
            raise ValueError("fast arm used a different conditions dump")
        result["fast_sha256"] = sha256(a.fast)
        fast = ak.from_parquet(a.fast)
        for k in ("run", "subrun", "event"):
            if not np.array_equal(ak.to_numpy(fast[k]), expected[k]):
                raise ValueError(f"fast event list mismatch: {k}")
        result["comparison"] = compare(fast, dump_frames([a.stock]))
    from mu2efastsim.plots import write_json
    write_json(out.parent, out.name, result)
    print(out.read_text())


if __name__ == "__main__":
    main()
