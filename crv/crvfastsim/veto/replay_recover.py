"""Recover retained fast memberships and export a diagnostic-only pulse adapter table.

No response or pulse fitting is run. The unchanged reference finder is dispatched
once per nonempty event and PE precision. Pulse collection order is never sorted
by channel/time. All analysis outside that dispatch is columnar.

The public recovery/export helpers import neither response nor fitting. The CLI's
required replay_metrics.verify_metadata verification transitively imports response
*configuration* modules; it does not execute response, load LUTs, or fit waveforms.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .. import geometry
from ..reco import coinc as C

IDS = ("run", "subrun", "event")
PULSE_TYPES = {k: "uint32" for k in IDS} | {
    k: "int32" for k in ("idx", "bar", "sipm", "seq", "flags", "pedfromdb")
} | {k: "float32" for k in ("pe", "peph", "height", "beta", "chi2", "ped")} | {
    "time": "float64", "letime": "float64", "wfidx": "var * int32"}
DIGI_TYPES = {k: "uint32" for k in IDS} | {
    k: "int32" for k in ("idx", "bar", "sipm", "starttdc", "nzs", "roc", "feb", "febch")
} | {"adc": "var * int16"}
CLUSTER_TYPES = {k: "uint32" for k in IDS} | {
    k: "int32" for k in ("idx", "sectortype", "nLayers", "sidehits0", "sidehits1")
} | {k: "float64" for k in ("start", "end", "avgtime", "pe", "slope", "x", "y", "z",
                                "sidepe0", "sidepe1", "sidetime0", "sidetime1")} | {"poscalc": "bool"}
SUMMARY_FIELDS = {"time": "avgtime", "timeStart": "start", "timeEnd": "end", "PEs": "pe",
                  "nLayers": "nLayers", "sectorType": "sectortype"}
ADAPTER_INTENT = {
    "scope": "Diagnostic-only CrvRecoPulsesFromTable/CrvDigisFromTable adapter; NOT fitted waveforms or detector response.",
    "retained_fields": [*IDS, "idx", "bar", "sipm", "seq", "time", "pe"],
    "precision": "Only retained PE is rounded to float32. Retained reco-frame time stays float64; no further bunch-time shift.",
    "placeholders": {"peph": 0, "height": 0, "beta": 0, "chi2": 0, "ped": 0,
                     "letime": 0, "flags": 0, "pedfromdb": 0, "wfidx": []},
    "digis": "Empty typed tree; no ADC or waveform references are fabricated.",
    "required_disabled_features": ["pulse-height PE selection", "leading-edge time selection",
                                    "shape/fit-quality/flag selection", "waveform/ADC-dependent analysis"],
    "finder_inputs": "Use integrated PE and fitted peak time only; placeholders are not measurements.",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _columns(a, names):
    return {k: np.asarray(a[k]) if isinstance(a, dict) else ak.to_numpy(a[k], allow_missing=False) for k in names}


def _integer(a, lo, hi, label):
    a = np.asarray(a)
    if a.ndim != 1 or not np.isfinite(a).all() or np.any((a < lo) | (a >= hi) | (a != np.floor(a))):
        raise ValueError(f"invalid {label}")
    return a.astype(np.int64)


def _count(record, key, actual):
    if type(record.get(key)) is not int or record[key] != actual or actual < 0:
        raise ValueError(f"{key} count mismatch")


def _events(events):
    ev = _columns(events, (*IDS, "pbtime"))
    for k in IDS:
        ev[k] = _integer(ev[k], 0, 2**20 if k == "event" else 2**32, k)
    n = len(ev["event"])
    if not n or any(len(v) != n for v in ev.values()) or len(np.unique(ev["run"])) != 1:
        raise ValueError("nonempty, single-run all-event table required")
    ev["ev"] = (ev["subrun"] << 20) | ev["event"]
    if len(np.unique(ev["ev"])) != n:
        raise ValueError("duplicate event identity")
    ev["pbtime"] = ev["pbtime"].astype(np.float32).astype(np.float64)
    if not np.isfinite(ev["pbtime"]).all():
        raise ValueError("nonfinite bunch time")
    return ev


@dataclasses.dataclass
class PulseBatch:
    """Flat, stable event-grouped numpy columns plus offsets (including empty events)."""
    events: dict
    pulses: dict
    offsets: np.ndarray
    finder_time: np.ndarray


def prepare_pulses(pulses, events, nchannels=5120):
    """Convert retained columns ONCE; stable grouping preserves original per-event idx."""
    ev = _events(events)
    p = _columns(pulses, ("ev", "chan", "time", "pe", "seq"))
    n = len(p["ev"])
    if any(v.ndim != 1 or len(v) != n for v in p.values()):
        raise ValueError("inconsistent flat pulse columns")
    for k, hi in (("ev", 2**52), ("chan", nchannels), ("seq", 2**31)):
        p[k] = _integer(p[k], 0, hi, f"pulse {k}")
    for k in ("time", "pe"):
        p[k] = p[k].astype(np.float64)
        if not np.isfinite(p[k]).all():
            raise ValueError(f"nonfinite pulse {k}")
    eo = np.argsort(ev["ev"])
    loc = np.searchsorted(ev["ev"][eo], p["ev"])
    if np.any(loc == len(eo)) or not np.array_equal(ev["ev"][eo[loc]], p["ev"]):
        raise ValueError("pulse outside all-event table")
    event_row = eo[loc]
    order = np.argsort(event_row, kind="stable")
    event_row = event_row[order]
    p = {k: v[order] for k, v in p.items()}
    counts = np.bincount(event_row, minlength=len(eo))
    if np.any(counts >= 2**31):
        raise ValueError("pulse collection index exceeds int32")
    offsets = np.r_[0, np.cumsum(counts)]
    p["idx"] = np.arange(n, dtype=np.int64) - np.repeat(offsets[:-1], counts)
    p["bar"], p["sipm"] = p["chan"] // 4, p["chan"] % 4
    p.update({k: ev[k][event_row] for k in IDS})
    # Retained time = original finder time - rounded pbtime. Undo before finding.
    return PulseBatch(ev, p, offsets, p["time"] + ev["pbtime"][event_row])


def finder_config():
    """Pinned Run1A finder-only configuration, without importing fast response setup."""
    return C.FinderConfig(ignore_secondary_pulses=False,
                          cuts={n: C.SectorCuts(8, 20, 20, -6, 6, 3, 3, 0, 250, 0) for n in ("T1", "T2")})


def _empty_jagged(n, dtype):
    return ak.unflatten(np.empty(0, dtype), np.zeros(n, np.int64))


def recover(batch, G, status, cfg=None, *, pe_float32=False):
    """Run C.find_event unchanged; return stock-layout flat clusters with full IDs.

    All times are returned in the retained reco frame. Missing-side sidetime keeps
    the finder's zero sentinel, rather than shifting a nonexistent measurement.
    """
    cfg = finder_config() if cfg is None else cfg
    status = _integer(status, 0, 2**16, "channel status")
    if len(status) != len(G.sector) * 4:
        raise ValueError("incomplete channel status")
    if np.any(batch.pulses["bar"] >= len(G.sector)):
        raise ValueError("pulse bar outside geometry")
    p = {k: batch.pulses[k] for k in ("idx", "bar", "sipm", "seq", "pe")}
    p["time"] = batch.finder_time
    if pe_float32:
        p["pe"] = p["pe"].astype(np.float32).astype(np.float64)
        if not np.isfinite(p["pe"]).all():
            raise ValueError("PE not representable as float32")
    rows, event_rows = [], []
    counts = np.zeros(len(batch.events["ev"]), np.int64)
    # Dispatch the per-event reference API using numpy views, never per-event awkward conversions.
    for e in np.flatnonzero(np.diff(batch.offsets)):
        a, b = batch.offsets[e:e + 2]
        found = C.find_event({k: v[a:b] for k, v in p.items()}, G, status, cfg)
        counts[e] = len(found)
        rows.extend(found)
        event_rows.extend([e] * len(found))
    if not rows:
        return ak.Array({k: np.empty(0, dt) for k, dt in CLUSTER_TYPES.items()} |
                        {k: _empty_jagged(0, np.int32) for k in ("pulses", "layers")})
    r = ak.Array(rows)
    ei = np.asarray(event_rows, np.int64)
    shift = -batch.events["pbtime"][ei]
    out = {k: batch.events[k][ei] for k in IDS}
    out["idx"] = np.arange(len(rows)) - np.repeat(np.r_[0, counts.cumsum()[:-1]], counts)
    out.update({k: ak.to_numpy(r[k]) for k in ("sectortype", "pe", "slope", "poscalc")})
    out.update({k: ak.to_numpy(r[k]) + shift for k in ("start", "end", "avgtime")})
    out.update({k: ak.to_numpy(r.pos[:, i]) for i, k in enumerate(("x", "y", "z"))})
    for side in (0, 1):
        hits = ak.to_numpy(r.sidehits[:, side])
        out[f"sidehits{side}"] = hits
        out[f"sidepe{side}"] = ak.to_numpy(r.sidepe[:, side])
        out[f"sidetime{side}"] = np.where(hits > 0, ak.to_numpy(r.sidetime[:, side]) + shift, 0.)
    out["nLayers"] = ak.to_numpy(ak.num(r.layers))
    out = {k: np.asarray(out[k], dt) for k, dt in CLUSTER_TYPES.items()}
    if any(not np.isfinite(v).all() for v in out.values()):
        raise ValueError("nonfinite recovered cluster property")
    out.update({k: ak.values_astype(r[k], np.int32) for k in ("pulses", "layers")})
    return ak.Array(out)


def compare_saved(clusters, fast, *, require=True, atol=1e-9):
    """Compare every saved frame/cluster in order; exact counts/IDs/integer fields."""
    counts = ak.to_numpy(ak.num(fast.coincs.time))
    n = int(counts.sum())
    order_equal = len(clusters) == n
    if order_equal:
        order_equal = all(np.array_equal(ak.to_numpy(clusters[k]), np.repeat(ak.to_numpy(fast[k]), counts)) for k in IDS)
        idx = np.arange(n) - np.repeat(np.r_[0, counts.cumsum()[:-1]], counts)
        order_equal = order_equal and np.array_equal(ak.to_numpy(clusters.idx), idx)
    result = {"counts_order_ids_exact": bool(order_equal), "max_abs_differences": {}, "integer_exact": True,
              "passed": bool(order_equal), "absolute_tolerance": atol, "relative_tolerance": 0.0}
    for saved, field in SUMMARY_FIELDS.items():
        if not np.array_equal(ak.to_numpy(ak.num(fast.coincs[saved])), counts):
            raise ValueError(f"inconsistent saved coincidence lengths: {saved}")
        target = ak.to_numpy(ak.flatten(fast.coincs[saved]), allow_missing=False)
        if not np.isfinite(target).all():
            raise ValueError(f"nonfinite saved summary: {saved}")
        if not order_equal:
            result["max_abs_differences"][saved] = None
            result["integer_exact"] = False
            continue
        actual = ak.to_numpy(clusters[field])
        delta = float(np.max(np.abs(actual.astype(float) - target))) if n else 0.0
        result["max_abs_differences"][saved] = delta
        exact = saved in ("nLayers", "sectorType")
        equal = np.array_equal(actual, target) if exact else np.allclose(actual, target, rtol=0, atol=atol)
        if exact:
            result["integer_exact"] &= bool(equal)
        result["passed"] &= bool(equal)
    if require and not result["passed"]:
        raise ValueError(f"recovered fast frame closure failed: {result}")
    return result


def compare_precision(original, rounded):
    """Report float32-PE membership and all scalar-property effects, never hide changes."""
    aligned = len(original) == len(rounded) and all(np.array_equal(ak.to_numpy(original[k]), ak.to_numpy(rounded[k]))
                                                   for k in (*IDS, "idx"))
    exact = {}
    for k in ("pulses", "layers", "sectortype", "nLayers", "sidehits0", "sidehits1", "poscalc"):
        exact[k] = bool(aligned and ak.almost_equal(original[k], rounded[k], rtol=0, atol=0))
    differences = {}
    for k, dt in CLUSTER_TYPES.items():
        if dt == "float64":
            differences[k] = (float(np.max(np.abs(ak.to_numpy(original[k]) - ak.to_numpy(rounded[k]))))
                              if aligned and len(original) else 0.0 if aligned else None)
    return {"original_clusters": len(original), "float32_clusters": len(rounded),
            "counts_order_ids_exact": bool(aligned), "identical_memberships": exact["pulses"],
            "exact_properties": exact, "property_max_abs_differences": differences}


def write_adapter(path, batch):
    """Create exact adapter TTrees, retaining times/order; no fabricated digis."""
    n = len(batch.pulses["pe"])
    cols = {k: np.zeros(n, dt) for k, dt in PULSE_TYPES.items() if k != "wfidx"}
    for k in (*IDS, "idx", "bar", "sipm", "seq", "time", "pe"):
        cols[k] = batch.pulses[k].astype(PULSE_TYPES[k])
    if not np.isfinite(cols["pe"]).all():
        raise ValueError("PE not representable as float32")
    cols["wfidx"] = _empty_jagged(n, np.int32)
    with uproot.create(path) as f:
        f.mktree("crvdump/pulses", PULSE_TYPES)
        f.mktree("crvdump/digis", DIGI_TYPES)
        if n:
            f["crvdump/pulses"].extend(cols)
    return {"pulses": n, "digis": 0}


def run(args):
    """Verify retained inputs, require original closure, then write a NEW directory."""
    out = Path(args.outdir)
    if out.exists():
        raise ValueError("choose a new output directory")
    # Kept lazy: importing the pure recovery API does not import response/fitting.
    from .replay_metrics import verify_metadata, validate_frames
    from .replay_check import check_events
    from .compareframes import dump_frames
    from .fastframes import run_config

    check_path, fast_path, mixed_path = map(Path, (args.check, args.fast, args.input))
    meta_path, manifest_path = fast_path.with_suffix(".json"), mixed_path.with_suffix(".json")
    check_sha, meta_sha, manifest_sha = (sha256(p) for p in (check_path, meta_path, manifest_path))
    check, meta, manifest = (json.loads(p.read_text()) for p in (check_path, meta_path, manifest_path))
    identity, digest, legacy = verify_metadata(check, meta, meta_sha)
    if check.get("conditions_equal") is not True:
        raise ValueError("conditions not verified")
    stock_path = Path(check["stock"])
    for path, expected in ((fast_path, check["fast_sha256"]), (stock_path, check["stock_sha256"]),
                           (mixed_path, check["input_sha256"]), (manifest_path, check["input_manifest_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"bound payload hash mismatch: {path}")
    if manifest.get("output_sha256") != check["input_sha256"]:
        raise ValueError("mixed manifest payload mismatch")
    if not meta.get("pulse_file") or not meta.get("pulse_sha256"):
        raise ValueError("missing retained pulses")
    pulse_path = Path(meta["pulse_file"])
    if sha256(pulse_path) != meta["pulse_sha256"]:
        raise ValueError("pulse hash mismatch")
    with uproot.open(mixed_path) as f:
        events = f["dump/events"].arrays(library="ak")
    with uproot.open(stock_path) as f:
        stock_events = f["crvdump/events"].arrays(library="ak")
        calib = f["crvdump/calib"].arrays(["channel", "status"], library="np")
        for name, expected in check["entries_highest_cycle"].items():
            if f[f"crvdump/{name}"].num_entries != expected:
                raise ValueError(f"highest-cycle {name} count mismatch")
    _count(manifest, "events", len(events))
    ev = _events(events)
    verified_events = check_events(events, stock_events, len(events))
    if verified_events != check["events_check"]:
        raise ValueError("event contract differs from saved check")
    fast, stock = ak.from_parquet(fast_path), dump_frames([str(stock_path)])
    validate_frames(fast, stock, check, meta)
    if any(not np.array_equal(ak.to_numpy(fast[k]), ev[k]) for k in IDS):
        raise ValueError("mixed/fast event identities differ")
    pulse_array = ak.from_parquet(pulse_path)
    _count(meta.get("timers", {}), "pulses", len(pulse_array))
    batch = prepare_pulses(pulse_array, ev)
    del pulse_array
    counters_sha = sha256(args.counters)
    geom = geometry.from_offline_config(args.counters)
    G = C.sector_geom(geom, geometry.parse_simpleconfig(Path(args.counters).read_text()))
    if G.sector_names != ["T1", "T2"] or len(G.sector) != 1280:
        raise ValueError("recovery requires retained Run1A T1/T2 geometry")
    channels = _integer(calib["channel"], 0, 5120, "calibration channel")
    if not np.array_equal(np.sort(channels), np.arange(5120)):
        raise ValueError("calibration channels must cover 0..5119 exactly once")
    status = _integer(calib["status"], 0, 2**16, "channel status")[np.argsort(channels)]
    cfg = finder_config()
    if cfg != run_config("run1a_v01")[1]:
        raise ValueError("pinned finder configuration drift")
    original = recover(batch, G, status, cfg)
    closure = compare_saved(original, fast)  # MUST pass before any export.
    rounded = recover(batch, G, status, cfg, pe_float32=True)
    rounded_closure = compare_saved(rounded, fast, require=False)
    precision = compare_precision(original, rounded)
    sources = {"check": check_path, "fast": fast_path, "fast_metadata": meta_path,
               "input": mixed_path, "input_manifest": manifest_path, "stock": stock_path,
               "pulses": pulse_path, "counters": Path(args.counters)}
    source_hashes = {"check": check_sha, "fast": check["fast_sha256"], "fast_metadata": meta_sha,
                     "input": check["input_sha256"], "input_manifest": manifest_sha,
                     "stock": check["stock_sha256"], "pulses": meta["pulse_sha256"], "counters": counters_sha}
    # A slow finder must not bind a newly changed input to an earlier recovery.
    for name, path in sources.items():
        if sha256(path) != source_hashes[name]:
            raise ValueError(f"input changed during recovery: {name}")
    code = [Path(__file__), Path(C.__file__), Path(geometry.__file__),
            Path(__file__).with_name("fastframes.py"), Path(__file__).with_name("replay_metrics.py"),
            Path(__file__).with_name("replay_policy.py"), Path(__file__).with_name("replay_check.py")]
    adapter_dir = Path(__file__).parents[2] / "art/CrvFastSim/src"
    code += [adapter_dir / n for n in ("CrvRecoTable.hh", "CrvRecoPulsesFromTable_module.cc", "CrvDigisFromTable_module.cc")]
    code_hashes = {str(p): sha256(p) for p in code}
    # Exclusive mkdir/create: nothing retained is overwritten, even on a race.
    out.mkdir(parents=True)
    ak.to_parquet(original, out / "recovered.parquet")
    ak.to_parquet(rounded, out / "recovered_float32.parquet")
    entries = write_adapter(out / "fast_table.root", batch)
    result = {"schema": "crv-retained-fast-recovery-v1", "frames": len(events), "pulses": entries["pulses"],
              "coincs": len(original), "events_check": verified_events,
              "sources": {k: {"path": str(p), "sha256": source_hashes[k]} for k, p in sources.items()},
              "response_identity": identity, "response_identity_sha256": digest, "legacy_policy_recovery": legacy,
              "finder_config": dataclasses.asdict(cfg), "frame_closure": closure,
              "float32_frame_comparison": rounded_closure, "float32_comparison": precision,
              "adapter": ADAPTER_INTENT, "root_schema": {"pulses": PULSE_TYPES, "digis": DIGI_TYPES},
              "event_pulse_counts": np.diff(batch.offsets).tolist(),
              "code_sha256": code_hashes,
              "output_sha256": {p.name: sha256(p) for p in sorted(out.iterdir())},
              "scope": "Recovered fast membership is summary-validated, not persisted original membership or truth. "
                       "Float32 PE effects are reported, not suppressed. No response/fitting, no production change; C4 remains stock."}
    with (out / "recovery.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("check", "fast", "input", "counters", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    result = run(parser.parse_args(argv))
    print(json.dumps({k: result[k] for k in ("frames", "pulses", "coincs", "frame_closure", "float32_comparison")}, indent=2))


if __name__ == "__main__":
    main()
