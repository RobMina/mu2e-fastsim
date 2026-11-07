"""Validate saved single-pass direct benchmarks without rerunning either endpoint."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np
import uproot

from trigger.timing import EVENTS, EXIT, TIME
from .direct_regression import compare_outputs, compare_sampling
from .inspect import summarize_fast, summarize_native

DEFAULT_RUNS = ("fresh_cache", "populated_cache_1", "populated_cache_2")


def number(value, label, positive=False):
    """Reject booleans, nonfinite numbers, negatives and undefined denominators."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'} and finite")
    return value


def json_object(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    result = json.loads(text, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    json.dumps(result, allow_nan=False)
    return result


def validate_time(data):
    required = {"wall_s", "user_s", "system_s", "max_rss_kb", "exit_status"}
    if set(data) != required:
        raise ValueError("time record needs wall_s, user_s, system_s, max_rss_kb and exit_status")
    if type(data["exit_status"]) is not int or data["exit_status"] != 0:
        raise ValueError("timed process exit_status must be zero")
    for key in ("wall_s", "user_s", "system_s", "max_rss_kb"):
        number(data[key], key, positive=key in ("wall_s", "max_rss_kb"))
    if type(data["max_rss_kb"]) is not int:
        raise ValueError("max_rss_kb must be an integer")
    return data


def parse_time_json(text):
    """Parse one GNU time JSON record; rounded user/system times may be zero."""
    if len(text.strip().splitlines()) != 1:
        raise ValueError("need exactly one GNU time JSON line")
    return validate_time(json_object(text))


def parse_elapsed(token):
    """Parse GNU verbose elapsed time in M:SS or H:MM:SS form."""
    if not re.fullmatch(r"\d+:\d{1,2}(?:\.\d+)?|\d+:\d{2}:\d{2}(?:\.\d+)?", token):
        raise ValueError("invalid GNU elapsed time")
    parts = token.split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    if seconds >= 60 or (len(parts) == 3 and minutes >= 60):
        raise ValueError("invalid GNU elapsed time field")
    elapsed = minutes * 60 + seconds + (int(parts[0]) * 3600 if len(parts) == 3 else 0)
    return number(elapsed, "GNU elapsed time", positive=True)


def parse_gnu_time(text):
    """Read one successful original /usr/bin/time -v record."""
    fields = {
        "Elapsed (wall clock) time (h:mm:ss or m:ss)": ("wall_s", parse_elapsed),
        "User time (seconds)": ("user_s", float),
        "System time (seconds)": ("system_s", float),
        "Maximum resident set size (kbytes)": ("max_rss_kb", int),
        "Exit status": ("exit_status", int),
    }
    result = {}
    for raw in text.splitlines():
        line = raw.strip()
        for label, (key, convert) in fields.items():
            if line.startswith(label):
                match = re.fullmatch(re.escape(label) + r":\s*(\S+)", line)
                if not match or key in result:
                    raise ValueError(f"missing, malformed or repeated GNU time field: {label}")
                result[key] = convert(match[1])
    return validate_time(result)


def parse_art_log(text, expected_events):
    """Require one successful TimeReport, art exit and matching event census."""
    found = {"events": [], "time": [], "exit": []}
    for raw in text.splitlines():
        line = " ".join(raw.split())
        for marker, pattern, key in (("TrigReport Events", EVENTS, "events"),
                                     ("TimeReport CPU", TIME, "time"),
                                     ("Art has completed and will exit with status", EXIT, "exit")):
            if marker in line:
                match = pattern.search(line)
                if not match:
                    raise ValueError(f"malformed stock {marker}")
                found[key].append(match.groups())
    if any(len(values) != 1 for values in found.values()):
        raise ValueError("need exactly one stock Events, TimeReport and art success marker")
    events, passed, failed = map(int, found["events"][0])
    if events <= 0 or events != expected_events or passed + failed != events:
        raise ValueError("stock log event count differs from finished ROOT census")
    if int(found["exit"][0][0]) != 0:
        raise ValueError("stock art exit status is not zero")
    cpu, real = map(float, found["time"][0])
    return {"events": events, "accepted_any_path": passed, "rejected_all_paths": failed,
            "cpu_s": number(cpu, "Art CPU", positive=True),
            "real_s": number(real, "Art Real", positive=True)}


def materialize_sampling(directory):
    """Resolve saved NPZ receipts without loading source frames or resampling."""
    directory = Path(directory)
    metadata = json_object((directory / "sampling.json").read_text())
    with np.load(directory / "sampling.npz", allow_pickle=False) as arrays:
        used = set()
        def resolve(value):
            if isinstance(value, dict) and "npz_key" in value:
                if set(value) != {"npz_key", "shape", "dtype"}:
                    raise ValueError("malformed sampling NPZ receipt")
                key = value["npz_key"]
                data = arrays[key]
                if list(data.shape) != value["shape"] or str(data.dtype) != value["dtype"]:
                    raise ValueError(f"sampling NPZ shape/dtype disagrees with metadata: {key}")
                used.add(key)
                return data
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in value.items()}
            if isinstance(value, list):
                return [resolve(item) for item in value]
            return value
        result = resolve(metadata)
        if used != set(arrays.files):
            raise ValueError("sampling NPZ contains unreferenced arrays")
    return result


def unsigned_ids(values):
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("full event identities must be nonnegative integers")
    return values.astype(np.uint64, copy=False)


def read_ids(path, table):
    with uproot.open(path) as root:
        rows = root[table].arrays(["run", "subrun", "event"], library="np")
        ids = np.column_stack([unsigned_ids(rows[key]) for key in ("run", "subrun", "event")])
    if ids.shape != (len(ids), 3) or not len(ids) or np.unique(ids, axis=0).shape[0] != len(ids):
        raise ValueError("finished ROOT census has empty, invalid or duplicate full event identities")
    return ids


def equal_ids(left, right, label):
    right = unsigned_ids(right)
    if right.shape != left.shape or not np.array_equal(left, right):
        raise ValueError(f"{label}: full ordered run/subrun/event identities differ")


def validate_driver(saved, census):
    """Accept exactly one combined preparation/response with paired checks disabled."""
    if saved.get("input_mode") != "direct_frames" or saved.get("paired_comparisons_skipped") is not True:
        raise ValueError("single-pass report requires direct_frames and paired_comparisons_skipped=True")
    if type(saved.get("events")) is not int or saved["events"] != census["events"]:
        raise ValueError("driver event count differs from finished ROOT census")
    preparation, passes = saved["preparation"], saved["passes"]
    if len(preparation["passes"]) != 1 or len(passes) != 1:
        raise ValueError("need exactly one preparation and one response pass")
    if preparation["passes"][0]["pass"] != 0 or passes[0]["pass"] != 0:
        raise ValueError("single preparation/response pass must have index zero")
    if any(saved.get(key) is not None for key in ("crv_paired", "tracker_paired")):
        raise ValueError("paired comparisons must be skipped, not included in process timing")
    if preparation.get("sampling_reference_identical") or preparation.get("repeats_identical") or saved.get("same_seed_repeats_identical"):
        raise ValueError("single-pass timing must not include repeated preparation or regression comparisons")
    if "tracker_digis" not in census or "tracker" not in passes[0]:
        raise ValueError("combined throughput requires finished tracker and CRV output")
    for key in ("setup_s", "input_s", "write_s", "sampling_write_s"):
        number(saved[key], key, positive=True)
    if saved["setup_s"] < saved["input_s"]:
        raise ValueError("setup_s includes input_s and cannot be smaller")
    first = preparation["passes"][0]
    number(first["seconds"], "preparation seconds", positive=True)
    number(preparation["exact_check_s"], "preparation exact_check_s")
    for label, stages in (("frame preparation", saved["frame_preparation_s"]), ("preparation pass", first["stages"])):
        for key, value in stages.items():
            number(value, label + "/" + key)
    for key in ("crv", "tracker"):
        number(passes[0][key]["seconds"], key + " seconds", positive=True)
    number(passes[0]["combined_response_s"], "combined_response_s", positive=True)


def artifact(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def hardware(path):
    """Keep raw provenance private; an optional JSON label is safe for public plots.

    Adjacent hardware.json / stock_hardware.json contain {"label": "..."}.
    A missing receipt never implies a particular accelerator or allocation.
    """
    result = {"path": str(path), "text": path.read_text()} if path.is_file() else {}
    metadata = path.with_suffix(".json")
    if metadata.is_file():
        data = json_object(metadata.read_text())
        label = data.get("label")
        if not isinstance(label, str) or not label.strip() or any(c in label for c in "\r\n"):
            raise ValueError("hardware metadata needs a nonempty single-line public label")
        result.update(label=label.strip(), metadata=artifact(metadata))
    return result or None


def measured(timing, events, scope):
    return {**timing, "events_per_s": number(events / timing["wall_s"], "events_per_s", positive=True),
            "scope": scope}


def report(rootdir, reference, stock, run_names=DEFAULT_RUNS):
    """Validate every saved artifact before constructing a descriptive timing report."""
    rootdir, reference, stock = (Path(path).resolve() for path in (rootdir, reference, stock))
    names = tuple(run_names)
    if len(names) != 3 or len(set(names)) != 3 or any(not name or Path(name).name != name or name in (".", "..") for name in names):
        raise ValueError("supply three distinct ordered run-directory names: fresh, populated 1, populated 2")
    baseline = summarize_fast(reference / "fast.root")
    ids = read_ids(reference / "fast.root", "crvfast/events")
    events = len(ids)
    if baseline["events"] != events or "tracker_digis" not in baseline:
        raise ValueError("saved direct reference must have a complete combined detector census")
    reference_report = json_object((reference / "report.json").read_text())
    if reference_report.get("events") != events or reference_report.get("input_mode") != "direct_frames":
        raise ValueError("saved direct reference report disagrees with finished census or input mode")
    reference_sampling = materialize_sampling(reference)
    equal_ids(ids, reference_sampling["signal_event_ids"], "reference sampling")
    compare_sampling(reference_sampling, reference)
    native = summarize_native(stock / "response.root")
    stock_ids = read_ids(stock / "response.root", "crvdump/events")
    equal_ids(ids, stock_ids, "stock/reference")
    if native["events"] != events:
        raise ValueError("stock finished census differs from direct reference")
    stock_time = parse_gnu_time((stock / "art.time").read_text())
    art = parse_art_log((stock / "art.log").read_text(), events)
    runs = []
    for index, name in enumerate(names):
        directory = rootdir / name
        census = summarize_fast(directory / "fast.root")
        equal_ids(ids, read_ids(directory / "fast.root", "crvfast/events"), name)
        if census["events"] != events:
            raise ValueError("candidate finished census differs from reference")
        saved = json_object((directory / "report.json").read_text())
        validate_driver(saved, census)
        sampling = materialize_sampling(directory)
        equal_ids(ids, sampling["signal_event_ids"], name + " sampling")
        compare_sampling(sampling, reference)
        equality = compare_outputs(reference / "fast.root", directory / "fast.root")
        if equality.get("equal") is not True:
            raise ValueError("saved detector outputs do not match reference")
        outer_path, inner_path = rootdir / (name + ".time.json"), rootdir / (name + ".process.time.json")
        outer = parse_time_json(outer_path.read_text())
        inner = parse_time_json(inner_path.read_text())
        if inner["wall_s"] > outer["wall_s"] + 0.02:
            raise ValueError("inner Python wall time exceeds outer container wall time")
        response = saved["passes"][0]["combined_response_s"]
        response_write = number(response + saved["write_s"], "response plus write", positive=True)
        runs.append({
            "name": name, "directory": str(directory), "census": census,
            "cache_state": "fresh_private_cache" if index == 0 else "populated_cache_fresh_process",
            "outer_container_process": measured(outer, events, "external container launch through process exit"),
            "python_process": measured(inner, events, "external Python startup through output and process exit"),
            "stages": {key: saved[key] for key in ("setup_s", "input_s", "sampling_write_s", "write_s", "frame_preparation_s", "preparation", "passes")},
            "response_plus_write": {"seconds": response_write, "events_per_s": number(events / response_write, "response/write rate", positive=True),
                                    "scope": "derived response + ROOT write only; not measured end-to-end latency"},
            "settings": {key: saved.get(key) for key in ("seed", "chunk", "devices", "crv_device_precision", "tracker_device_precision")},
            "validation": {"paired_comparisons_skipped": True, "sampling_identical": True, "outputs": equality},
            "artifacts": {filename: artifact(directory / filename) for filename in ("fast.root", "report.json", "sampling.json", "sampling.npz")},
            "timing_artifacts": {"outer": artifact(outer_path), "python": artifact(inner_path)},
        })
    fast_hardware = hardware(rootdir / "hardware.txt")
    stock_hardware = hardware(rootdir / "stock_hardware.txt")
    result = {
        "schema_version": 1, "rootdir": str(rootdir), "events": events, "run_order": list(names),
        "denominator": "all primary events in finished highest-cycle ROOT entries, including empty detector events",
        "boundary": "cached detector source steps through direct sampling/preparation, combined GPU-backed tracker/CRV response and reduced ROOT output",
        "excluded": ["source extraction", "input staging", "queue wait", "transport", "calorimeter", "trigger", "track fitting", "full art production chain"],
        "cache_scope": "fresh means only a fresh private JAX/CUDA cache, not a cold OS page cache; populated-cache runs are fresh processes, NOT fully warmed in-process passes; cache labels follow the supplied run order",
        "stage_scope": "setup_s includes input_s; never add them twice. Stage sums are not measured end-to-end time. External Python/container timings are independent measurements; post-run comparisons are excluded.",
        "comparison_scope": "Saved direct outputs and scientific sampling choices must match the supplied saved direct reference exactly. Stock shares ordered primary IDs but has independent pileup/response draws and richer native output, not exact detector equivalence.",
        "endpoint_scope": "Stock and fast rates are separate measurements, not a speedup comparison: devices and timing boundaries differ. Stock art includes native mixing, framework/source I/O and richer native detector diagnostics; fast starts from cached source steps and writes reduced diagnostics. Art TimeReport is a separate timing boundary and is not substituted for process wall.",
        "hardware": fast_hardware,
        "reference": {"directory": str(reference), "census": baseline,
                      "artifacts": {name: artifact(reference / name) for name in ("fast.root", "report.json", "sampling.json", "sampling.npz")}},
        "stock": {"directory": str(stock), "endpoint": (stock_hardware or {}).get("label", "hardware not recorded"), "rerun": False,
                  "census": native, "hardware": stock_hardware,
                  "external_process": measured(stock_time, events, "original GNU time external art process wall, including process startup/shutdown"),
                  "art_time_report": {**art, "real_events_per_s": number(events / art["real_s"], "Art event rate", positive=True),
                                      "scope": "art TimeReport CPU/Real; distinct from external process wall"},
                  "artifacts": {name: artifact(stock / name) for name in ("response.root", "art.time", "art.log")}},
        "runs": runs,
    }
    json.dumps(result, allow_nan=False)
    return result


def publish(path, text):
    """Atomically publish a new file without replacing any existing destination."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".throughput-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rootdir", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", "--run-names", nargs=3, default=DEFAULT_RUNS, metavar=("FRESH", "POPULATED_1", "POPULATED_2"))
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        result = report(args.rootdir, args.reference, args.stock, args.runs)
        publish(args.out, json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, KeyError, TypeError, OverflowError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"throughput report: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
