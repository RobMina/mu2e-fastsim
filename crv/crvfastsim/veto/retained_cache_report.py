"""Verify saved retained-cache audit evidence without reloading ROOT or regenerating products."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import retained_cache_audit as C, saved_qualified_report as Q
from .replay import sha256

I, F = C.I, C.F
SCHEMA = "saved-retained-cache-report-v1"
SCOPE = ("Saved transaction and original full-registry/plan frame and batch census checked exactly. "
         "Raw files are hashed, never opened as ROOT or loaded into a replay cache. Selection hashes, "
         "source-row coverage flags and per-frame particle allocations remain generation-bound measurements: "
         "selections and coverage bitmaps were not persisted, so raw joins/coverage are not independently "
         "rederived. Particle totals are checked against manifests and saved frame/batch aggregations. "
         "RSS high-water marks are cumulative for the generation process, including plan admission, not "
         "per-input peaks or a memory bound. No optics, response, execution admission, F, independent "
         "exposure, runtime projection or finite-library uncertainty certification.")
FRAME_COLUMNS = [*F.COLUMNS, "batch", "event_source_row", "steps", "finite_steps", "particles"]
BATCH_COLUMNS = ["batch", "input_sha256", "frames", "steps", "particles", "typed_payload_bytes",
                 *[f"{n}_sha256" for n in I.TYPES], "select_seconds", "validate_seconds"]
MEMORY_KEYS = ("memory_before", "memory_loaded", "memory_audited", "memory_released")
TIME_KEYS = ("load_seconds", "selection_audit_seconds", "elapsed_seconds")
SUMMARY_KEYS = {"schema", "scope", "plan", "plan_sha256", "registry_sha256", "inputs", "code_sha256", "limits",
                "output_cap", "records", "frames", "batches", "steps", "particles", "all_source_rows_once",
                "whole_frames_exact", "execution_admitted", "response_executed", "optics_executed", "F",
                "runtime_projection", "finite_library_uncertainty", "memory_initial", "memory_after_admission",
                "memory_final", "elapsed_seconds_before_commit", "environment"}
_require, _equal = Q._require, Q._equal


def _integer(value, label):
    _require(type(value) is int and 0 <= value < 2**63, "invalid integer: " + label)
    return value


def _seconds(value, label):
    _require(type(value) in (float, int) and np.isfinite(value) and value >= 0, "invalid timing: " + label)
    return value


def _memory(value):
    _require(type(value) is dict and set(value) == {"peak_rss_bytes_cumulative", "current_rss_bytes"},
             "invalid memory schema")
    peak = _integer(value["peak_rss_bytes_cumulative"], "cumulative RSS")
    if value["current_rss_bytes"] is not None:
        _integer(value["current_rss_bytes"], "current RSS")
    return peak


def _bindings(mapping):
    _require(type(mapping) is dict, "invalid binding mapping")
    for path, digest in mapping.items():
        _require(type(path) is str and bool(path), "invalid binding path")
        I._digest(digest)
        C.M.P.bound(path, digest)


def _table(actual, expected):
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True), check_exact=True)


def _schema(table, columns, integers, times=()):
    _require(list(table) == columns, "saved table schema disagreement")
    for name in integers:
        values = table[name].to_numpy()
        _require(values.dtype == np.dtype("int64") and np.all(values >= 0), "invalid table integer: " + name)
    for name in times:
        values = table[name].to_numpy()
        _require(values.dtype == np.dtype("float64") and np.isfinite(values).all() and np.all(values >= 0),
                 "invalid table timing: " + name)
    for name in columns:
        if name.endswith("sha256"):
            for value in table[name].unique():
                I._digest(value)


def verify(directory, out, *, expected_plan_sha256):
    """Repeat saved-only admission and verify evidence; write one new exclusive JSON report."""
    root, out = Path(directory), Path(out)
    _require(not out.exists(), "choose a new verification file")
    _require(not out.resolve().is_relative_to(root.resolve()), "report must be outside the saved transaction")
    I._digest(expected_plan_sha256)
    _require(Q._regular(root / "summary.json") <= C.OUTPUT_CAP, "summary byte cap exceeded")
    summary = Q._json(root / "summary.json")
    _require(type(summary) is dict and set(summary) == SUMMARY_KEYS, "wrong audit summary schema")
    _equal(summary["schema"], C.SCHEMA, "wrong audit contract")
    _equal(summary["scope"], C.SCOPE, "changed audit scope")
    _equal(summary["plan_sha256"], expected_plan_sha256, "external plan digest disagreement")
    I._digest(summary["registry_sha256"])
    _require(type(summary["plan"]) is str and bool(summary["plan"]), "missing source plan")
    for name, value in {"all_source_rows_once": True, "whole_frames_exact": True, "execution_admitted": False,
                        "optics_executed": False, "response_executed": False, "F": None,
                        "runtime_projection": None, "finite_library_uncertainty": None}.items():
        _require(summary[name] is value, "wrong audit flag: " + name)
    _equal(summary["limits"], asdict(I.Limits()), "changed cache limits")
    _equal(summary["output_cap"], C.OUTPUT_CAP, "changed output cap")
    env = summary["environment"]
    _require(type(env) is dict and set(env) == {"python", "numpy", "pandas", "uproot"}
             and all(type(v) is str and bool(v) for v in env.values()), "invalid environment schema")
    paths = C.code_paths()
    names = ["registry.parquet", "frames.parquet", "batches.parquet", "admission.json", "summary.json"]
    names += [f"code{i}.txt" for i in range(len(paths))]
    context = {"registry_sha256": summary["registry_sha256"],
               "batch_identity": {"plan_sha256": expected_plan_sha256, "contract": C.SCHEMA},
               "arms": ["cache_audit"], "seeds": [0]}
    receipt, total = Q._batch(root, context, names, C.OUTPUT_CAP)
    code = summary["code_sha256"]
    _require(type(code) is dict and set(code) == {str(p) for p in paths}, "mandatory code domain changed")
    _bindings(code)
    I._digest(C.LOADER_SHA)
    _equal(code[str(Path(I.__file__))], C.LOADER_SHA, "unapproved frozen loader")
    C.M.P.bound(I.__file__, C.LOADER_SHA)
    # C.run snapshots in code_paths insertion order; JSON key ordering is not that order.
    for i, path in enumerate(paths):
        _equal(sha256(root / f"code{i}.txt"), code[str(path)], "code snapshot disagreement")
    _bindings(summary["inputs"])
    _require(all(summary["inputs"][p] == code[p] for p in summary["inputs"].keys() & code.keys()),
             "conflicting input/code bindings")
    registry, plan_frames, plan_batches, sources, bindings, admitted = C.admission(summary["plan"], expected_plan_sha256)
    _equal(summary["inputs"], bindings, "source admission bindings changed")
    _equal(Q._json(root / "admission.json"), admitted, "saved admission report changed")
    _equal(F.digest(registry), summary["registry_sha256"], "full registry binding disagreement")
    for name, rows in (("registry", len(registry)), ("frames", len(registry)), ("batches", len(plan_batches))):
        C.parquet_admit(root / f"{name}.parquet", rows)
    _table(pd.read_parquet(root / "registry.parquet"), registry)
    frames, batches = (pd.read_parquet(root / f"{n}.parquet") for n in ("frames", "batches"))
    _schema(frames, FRAME_COLUMNS, [*C.IDS, "frame_id", "batch", "event_source_row", "steps", "finite_steps", "particles"])
    _schema(batches, BATCH_COLUMNS, ["batch", "frames", "steps", "particles", "typed_payload_bytes"],
            ["select_seconds", "validate_seconds"])
    _require(np.all(frames.particles <= I.Limits().max_particles)
             and np.all(batches.particles <= I.Limits().max_particles), "particle count cap exceeded")
    _table(frames[list(F.COLUMNS)], registry)
    ordered = plan_frames.sort_values("frame_id").reset_index(drop=True)
    # Validate replay's uint32-range identities before normalizing to the frozen int64 registry.
    for name in C.IDS:
        values = ordered[name].to_numpy()
        _require(values.dtype.kind in "iu" and np.all(values >= 0)
                 and np.all(values <= np.iinfo(np.uint32).max), "invalid plan frame identity")
    _table(ordered[list(F.COLUMNS)].astype({name: "int64" for name in C.IDS}), registry)
    _table(frames[["frame_id", "batch", "steps", "finite_steps"]], ordered[["frame_id", "batch", "steps", "finite_steps"]])
    _require(np.all(frames.finite_steps <= frames.steps), "invalid finite-step census")
    planned = plan_batches.sort_values("batch").reset_index(drop=True)
    _require(not planned.batch.duplicated().any(), "duplicate planned batch")
    _table(batches[["batch", "input_sha256", "frames", "steps"]], planned[["batch", "input_sha256", "frames", "steps"]])
    _require(not batches.batch.duplicated().any(), "duplicate saved batch")
    grouped = frames.groupby("batch", sort=True)
    _require(grouped.input_sha256.nunique().eq(1).all(), "mixed batch input domain")
    reconstructed = grouped.agg(input_sha256=("input_sha256", "first"), frames=("frame_id", "size"),
                                steps=("steps", "sum"), particles=("particles", "sum")).reset_index()
    _table(batches[list(reconstructed)], reconstructed)
    typed = sum(batches[n].to_numpy() * (I.DTYPES[t].itemsize + 272 + (8 if t == "steps" else 0))
                for n, t in (("frames", "events"), ("steps", "steps"), ("particles", "particles")))
    _require(np.array_equal(batches.typed_payload_bytes, typed)
             and np.all(typed >= 0) and np.all(typed <= I.Limits().max_selection_bytes), "typed selection byte accounting disagreement")
    records = summary["records"]
    _require(type(records) is list and len(records) == len(sources) > 0, "input record domain disagreement")
    digests = [s["input_sha256"] for s in sources]
    _require(len(set(digests)) == len(digests) and set(digests) == set(registry.input_sha256),
             "incomplete or duplicate input domain")
    _require(len({s["path"] for s in sources}) == len(sources), "duplicate input path")
    peaks = [_memory(summary["memory_initial"]), _memory(summary["memory_after_admission"])]
    for record, source in zip(records, sources):
        required = {*source, "counts", "batches", "all_source_rows_once", "coverage_bitmap_bytes", "resident_array_bytes",
                    *MEMORY_KEYS, *TIME_KEYS}
        _require(type(record) is dict and set(record) == required, "input record schema disagreement")
        _equal({k: record[k] for k in source}, source, "input record binding disagreement")
        raw = Path(source["path"])
        _bindings({str(raw): source["input_sha256"], str(raw.with_suffix(".json")): source["manifest_sha256"],
                   source["ledger"]: source["ledger_sha256"]})
        manifest = Q._json(raw.with_suffix(".json"))
        I._validate_manifest(manifest)
        I.require_legacy_response(manifest)
        _require(manifest["raw_times"] is True and manifest["output_sha256"] == source["input_sha256"],
                 "invalid raw manifest binding")
        counts = manifest["entries"]
        I._admit(counts, I.Limits())
        _equal(record["counts"], counts, "manifest/counts disagreement")
        local = frames[frames.input_sha256 == source["input_sha256"]]
        lb = batches[batches.input_sha256 == source["input_sha256"]]
        _require(len(local) == counts["events"] == manifest["events"] and int(local.steps.sum()) == counts["steps"]
                 and int(local.particles.sum()) == counts["particles"], "input census disagreement")
        _require(np.all(local.run == manifest["run"]) and np.all(local.subrun == manifest["subrun"])
                 and np.array_equal(local.event, np.arange(manifest["first_event"], manifest["first_event"] + len(local)))
                 and np.array_equal(local.event_source_row, np.arange(len(local))), "event source-row identity disagreement")
        for name, expected in {"batches": len(lb), "coverage_bitmap_bytes": sum(counts.values()),
                               "resident_array_bytes": I.resident_bytes(counts)}.items():
            _require(_integer(record[name], name) == expected, "input accounting disagreement: " + name)
        _require(record["all_source_rows_once"] is True, "missing generation coverage flag")
        for name in TIME_KEYS:
            _seconds(record[name], name)
        _require(record["load_seconds"] + record["selection_audit_seconds"] <= record["elapsed_seconds"],
                 "inconsistent input elapsed time")
        _require(float(lb.select_seconds.sum() + lb.validate_seconds.sum()) <= record["selection_audit_seconds"],
                 "inconsistent selection audit time")
        peaks.extend(_memory(record[name]) for name in MEMORY_KEYS)
    peaks.append(_memory(summary["memory_final"]))
    _require(all(b >= a for a, b in zip(peaks, peaks[1:])), "cumulative RSS high-water mark decreased")
    elapsed = _seconds(summary["elapsed_seconds_before_commit"], "audit elapsed")
    _require(sum(r["elapsed_seconds"] for r in records) <= elapsed, "inconsistent audit elapsed time")
    totals = {"frames": len(frames), "batches": len(batches), "steps": int(frames.steps.sum()),
              "finite_steps": int(frames.finite_steps.sum()), "particles": int(frames.particles.sum())}
    for name in ("frames", "batches", "steps", "particles"):
        _require(_integer(summary[name], name) == totals[name], "summary census disagreement: " + name)
    # Recheck bindings and the complete saved transaction before publishing a report.
    _bindings(summary["inputs"])
    _bindings(code)
    final_receipt, final_total = Q._batch(root, context, names, C.OUTPUT_CAP)
    _equal(final_receipt, receipt, "saved transaction changed during verification")
    _require(final_total == total, "saved byte accounting changed")
    result = dict(schema=SCHEMA, scope=SCOPE, summary_sha256=receipt["artifacts"]["summary.json"]["sha256"],
                  plan=summary["plan"], plan_sha256=expected_plan_sha256, registry_sha256=F.digest(registry),
                  **totals, records=records, max_selection_bytes=int(batches.typed_payload_bytes.max()),
                  persisted_path_bytes=total, output_cap=C.OUTPUT_CAP,
                  memory_initial=summary["memory_initial"], memory_after_admission=summary["memory_after_admission"],
                  memory_final=summary["memory_final"], elapsed_seconds_before_commit=elapsed,
                  saved_artifact_protocol_exact=True, source_admission_exact=True, saved_frame_batch_census_exact=True,
                  particle_totals_manifest_exact=True, all_source_rows_once=True,
                  coverage_flags_generation_bound=True, selection_hashes_generation_bound=True,
                  raw_joins_independently_rederived=False, source_coverage_independently_rederived=False,
                  raw_payload_reloaded=False, regenerated_optics=False, regenerated_response=False,
                  optics_executed=False, response_executed=False,
                  execution_admitted=False, F=None, runtime_projection=None, finite_library_uncertainty=None,
                  code_sha256=sha256(__file__))
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def compact(result):
    """Small view of verified saved measurements; cumulative RSS is not a per-input peak."""
    keys = ("schema", "summary_sha256", "plan_sha256", "frames", "batches", "steps", "finite_steps", "particles",
            "max_selection_bytes", "persisted_path_bytes", "memory_initial", "memory_after_admission", "memory_final",
            "elapsed_seconds_before_commit", "execution_admitted", "scope")
    return {**{k: result[k] for k in keys}, "inputs": [
        {k: r[k] for k in ("path", "input_sha256", "counts", "batches", "resident_array_bytes", "coverage_bitmap_bytes",
                          *MEMORY_KEYS, *TIME_KEYS)} for r in result["records"]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.directory, args.out, expected_plan_sha256=args.expected_plan_sha256)
    print(json.dumps(compact(result) if args.compact else result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
