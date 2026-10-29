"""Verify saved bounded pilots without physics regeneration; never campaign admission."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import stat

import numpy as np
import pandas as pd

from . import streaming_pilot as S

SR, O, I, F, R, M, V, A = S.SR, S.O, S.I, S.F, S.R, S.M, S.V, S.A
K = O.K
SCHEMA = "retained-streaming-pilot-report-v1"
SCOPE = ("Saved-only bounded new-frame diagnostic. Typed source fingerprints are independently checked against "
         "completed cache evidence; timing fields and saved response structure are checked without regenerating "
         "optics, avalanches, waveforms or fits. Response arithmetic and measured memory/timings remain hash-bound "
         "execution observations, not independent physical verification, isolated memory peaks or a hard RSS cap. "
         "Saved-only prohibits physics generation, not raw/stock ROOT reads in inherited source admission. "
         "No full 9000-frame promotion, stock equivalence, coincidence, F, exposure or uncertainty certification.")
_require, _equal = SR._require, SR._equal
SUMMARY_KEYS = {"schema", "scope", "selection", "plan", "cache_audit", "response", "hashes", "registry_sha256", "registry_frames",
                "frames", "batches", "seeds", "arms", "inputs", "code_sha256", "records", "byte_caps", "root_union",
                "optical_limits", "response_limits", "sipm_config", "waveform_config", "memory_before_execution",
                "memory_after_execution", "execution_seconds_before_commit", "environment", "whole_selected_frames_exact",
                "all_declared_arm_seeds_complete", "optics_executed", "response_executed", "execution_admitted",
                "timing_closure", "F", "finite_library_uncertainty"}
FLOATS = {"origin", "time_raw_absolute", "time_historical_relative", "time_raw_wrapped_absolute",
          "time_raw_wrapped_relative", "time_folded_wrapped_relative", "raw_folded_residual_ns",
          "raw_folded_circular_residual_ns"}
BOOLS = {"live_channel", "stock_step_kept", "historical_kept", "arrival_only_kept", "rejection_only_kept",
         "stock_timing_kept", "folded_arrival_kept", "raw_folded_upper_flip", "raw_folded_late_flip",
         "rescued_by_arrival", "removed_by_arrival", "rescued_survives_step", "accepted_time_changed"}


def _outside(out, roots):
    _require(not out.exists() and not out.is_symlink(), "choose a new report file")
    target = out.resolve()
    for root in roots:
        _require(not target.is_relative_to(Path(root).resolve()), "report must be outside evidence directories")


def _selection(child, admitted, batch, digest):
    tables = {name: pd.read_parquet(child / (name + ".parquet")) for name in S.SOURCE_TABLES}
    size = 0
    for name, table in tables.items():
        types = {**I.TYPES[name], "source_row": np.int64, "frame_id": np.int64}
        if name == "steps":
            types["pdg"] = np.int64
        _require(table.columns.is_unique and set(table) == {*types, "input_sha256"}, "source table schema changed")
        _require(all(table[c].dtype == np.dtype(t) for c, t in types.items()), "source typed schema changed")
        size += len(table) * (I.DTYPES[name].itemsize + 272 + (8 if name == "steps" else 0))
    _require(size <= I.Limits().max_selection_bytes, "saved selection byte cap exceeded")
    selection = I.Selection(**tables, registry_sha256=F.digest(admitted["registry"]),
                            input_sha256=digest, typed_payload_bytes=size)
    # This shared vectorized validator reads only the saved tables, never the cache.
    O._selection(selection, digest, O.Limits())
    selected, counts = S.check_selection(selection, admitted, batch)
    return selection, selected, counts


def _photons(child, selection, calibration, seed, roots):
    q = pd.read_parquet(child / "photons.parquet")
    ints = {*K.KEYS[1:], "photon_id", "event_row"}
    _require(q.columns.is_unique and set(q) == {"input_sha256", *ints, *FLOATS, *BOOLS}, "photon schema changed")
    _require(len(q) <= O.Limits().max_photons, "saved photon cap exceeded")
    for name in ints:
        O._integer(q, name, ceiling=2**64 - 1 if name == "draw_seed" else 2**63 - 1)
    for name in FLOATS:
        _require(q[name].dtype.kind == "f" and np.isfinite(q[name]).all(), "invalid photon floating field: " + name)
    for name in BOOLS:
        _require(q[name].dtype.kind == "b", "invalid photon boolean field: " + name)
    _require(q.input_sha256.eq(selection.input_sha256).all() and q.draw_seed.eq(seed).all(), "photon input/seed changed")
    _require(q.photon_type.between(0, 2).all() and q.event_row.lt(len(selection.events)).all(), "photon key domain changed")
    c = calibration[calibration.input_sha256 == selection.input_sha256].sort_values("chan")
    _require(c.chan.dtype.kind in "iu" and np.array_equal(c.chan, np.arange(len(c)))
             and c.status.dtype.kind in "iu" and c.status.ge(0).all(), "invalid complete photon status domain")
    raw = q[[*K.KEYS, "time_raw_absolute"]].rename(columns={"time_raw_absolute": "time"})
    historical = q[[*K.KEYS, "time_historical_relative"]].rename(columns={"time_historical_relative": "time"})
    rebuilt = M.timing_table(raw, historical, selection.steps, selection.events, c.status.to_numpy(),
                             selection.input_sha256, seed)
    pd.testing.assert_frame_equal(q.reset_index(drop=True), rebuilt, check_exact=True)
    roots.add(q)
    return q


def _measurements(summary):
    """Validate shape, not the truth or isolation of saved generation measurements."""
    measurements = {}
    for name in ("memory_before_execution", "memory_after_execution"):
        memory = summary[name]
        _require(set(memory) == {"peak_rss_bytes_cumulative", "current_rss_bytes"}, "memory measurement schema changed")
        for key, value in memory.items():
            _require((key == "current_rss_bytes" and value is None) or (type(value) is int and value >= 0),
                     "invalid memory measurement")
        measurements[name] = memory
    elapsed = summary["execution_seconds_before_commit"]
    _require(type(elapsed) in (int, float) and np.isfinite(elapsed) and elapsed >= 0, "invalid elapsed measurement")
    _require(measurements["memory_after_execution"]["peak_rss_bytes_cumulative"] >=
             measurements["memory_before_execution"]["peak_rss_bytes_cumulative"], "cumulative memory high-water decreased")
    measurements["execution_seconds_before_commit"] = elapsed
    measurements["maximum_observed_peak_rss_bytes_cumulative"] = max(
        summary[n]["peak_rss_bytes_cumulative"] for n in ("memory_before_execution", "memory_after_execution"))
    measurements["interpretation"] = "Hash-bound generation measurements; cumulative process high-water, not an isolated peak or hard cap."
    return measurements


def verify(directory, out, *, expected_plan_sha256, expected_cache_sha256, expected_response_sha256):
    """Verify saved evidence with three external admission hashes and write an exclusive report."""
    directory, out = Path(directory), Path(out)
    _outside(out, [directory])
    hashes = dict(plan_sha256=expected_plan_sha256, cache_sha256=expected_cache_sha256,
                  response_sha256=expected_response_sha256)
    for value in hashes.values():
        I._digest(value)
    root = directory / "run"
    _require(stat.S_ISDIR(directory.lstat().st_mode) and stat.S_ISDIR(root.lstat().st_mode), "invalid pilot directory")
    _require(SR._regular(root / "summary.json") <= S.TOP_CAP, "top metadata byte cap exceeded")
    summary = SR._json(root / "summary.json")
    _require(type(summary) is dict and set(summary) == SUMMARY_KEYS, "saved summary schema changed")
    environment = summary["environment"]
    _require(type(environment) is dict and set(environment) == {"python", "numpy", "pandas", "scipy", "uproot"}
             and all(type(value) is str and bool(value) for value in environment.values()), "invalid saved environment")
    _equal(summary["hashes"], hashes, "external admission hashes disagree")
    _outside(out, [summary[name] for name in ("plan", "cache_audit", "response")])
    paths = S.code_paths()
    _require(set(summary["code_sha256"]) == {str(p) for p in paths}, "mandatory code bindings changed")
    M.check_bindings(summary["inputs"], summary["code_sha256"])
    d = S.admission(summary["plan"], summary["cache_audit"], summary["response"],
                    expected_plan_sha256=expected_plan_sha256, expected_cache_sha256=expected_cache_sha256,
                    expected_response_sha256=expected_response_sha256)
    _equal(summary["inputs"], d["bindings"], "admitted source bindings changed")
    tags, caps = S.layout(d)
    _require(0 < len(d["chosen"]) <= S.MAX_BATCHES and len(d["selected"]) <= S.MAX_FRAMES,
             "bounded pilot schedule exceeded")
    _require(len(tags) <= 6 and len(set(tags)) == len(tags), "invalid child tag domain")
    _require(set(p.name for p in directory.iterdir()) == {"run", *tags}, "missing or undeclared pilot child")
    names = ["summary.json", "registry.parquet", "selected.parquet", "chosen.parquet", "calibration.parquet",
             "coverage.parquet", "admission.json"] + [tag + ".json" for tag in tags]
    names += [f"code{i}.txt" for i in range(len(paths))]
    top, total = SR._batch(root, S.context(d), names, S.TOP_CAP)
    for name in names:
        if name.endswith(".parquet"):
            S.C.parquet_admit(root / name)
    contract = dict(schema=S.SCHEMA, scope=S.SCOPE, selection=S.SELECTION, registry_sha256=F.digest(d["registry"]),
                    registry_frames=len(d["registry"]), frames=len(d["selected"]), batches=len(d["chosen"]),
                    seeds=list(S.SEEDS), arms=list(S.ARMS), byte_caps=caps, optical_limits=asdict(O.Limits()),
                    response_limits=asdict(R.Limits()), whole_selected_frames_exact=True,
                    all_declared_arm_seeds_complete=True, optics_executed=True, response_executed=True,
                    execution_admitted=False, timing_closure=False, F=None, finite_library_uncertainty=None)
    for name, expected in contract.items():
        _equal(summary[name], expected, "pilot contract changed: " + name)
    for name, keys in (("registry", ["frame_id"]), ("selected", ["frame_id"]), ("chosen", ["batch"]),
                       ("calibration", ["input_sha256", "chan"])):
        SR._table_equal(pd.read_parquet(root / (name + ".parquet")), d[name], keys)
    _equal(SR._json(root / "admission.json"), d["verified"], "saved source admission changed")
    for i, path in enumerate(paths):
        _require(S.C.sha256(root / f"code{i}.txt") == summary["code_sha256"][str(path)], "ordered code snapshot changed")
    cfg = replace(V.SI.SiPMConfig(), thermal_rate=V.THERMAL_RATE)
    template = str(S.one_asset(d["bindings"], R.D.TEMPLATE_SHA))
    wcfg = replace(V.WF.WaveformConfig(), single_pe_file=template)
    _equal(summary["sipm_config"], asdict(cfg), "fixed SiPM configuration changed")
    _equal(summary["waveform_config"], asdict(wcfg), "fixed waveform configuration changed")
    measurements = _measurements(summary)
    roots = O.RootUnion(max_roots=S.MAX_BATCHES * len(S.SEEDS) * R.Limits().photons)
    records, coverage, comparisons, frame_counts = [], [], {}, {}
    _require(len(summary["records"]) == len(tags), "child record count changed")
    schedule = [(row, seed) for _, row in d["chosen"].iterrows() for seed in S.SEEDS]
    for record, tag, (row, seed) in zip(summary["records"], tags, schedule):
        _require(type(record) is dict and set(record) == {"path", "batch", "seed", "input_sha256", "receipt_sha256", "counts", "elapsed_seconds"},
                 "child record schema changed")
        batch, digest = int(row.batch), row.input_sha256
        _equal({k: record[k] for k in ("path", "batch", "seed", "input_sha256")},
               dict(path=tag, batch=batch, seed=seed, input_sha256=digest), "ordered child schedule changed")
        elapsed = record["elapsed_seconds"]
        _require(type(elapsed) in (int, float) and np.isfinite(elapsed) and elapsed >= 0, "invalid child elapsed measurement")
        child = directory / tag
        receipt, size = SR._batch(child, S.context(d, batch, seed), S.CHILD_NAMES, S.CHILD_CAP)
        for name in S.CHILD_NAMES:
            if name.endswith(".parquet"):
                S.C.parquet_admit(child / name)
        total += size
        _require(record["receipt_sha256"] == S.C.sha256(child / A.COMPLETE), "child receipt hash changed")
        _equal(SR._json(root / (tag + ".json")), receipt, "copied child receipt changed")
        selection, selected, counts = _selection(child, d, batch, digest)
        q = _photons(child, selection, d["calibration"], seed, roots)
        receipts, contrasts, reference, stages = {}, {}, None, {}
        for arm in S.ARMS:
            products = {stage: pd.read_parquet(child / f"{arm}_{stage}.parquet") for stage in R.STAGES}
            R.validate_products(products)
            receipts[arm] = SR._products(products, d["registry"], selected, d["calibration"], q, seed, arm, wcfg)
            stages[arm] = {stage: len(table) for stage, table in products.items()}
            coverage.append(products["frames"].assign(batch=batch, optical_seed=np.uint64(seed), response_seed=np.uint64(seed), arm=arm))
            frame_counts[tag + "_" + arm] = products["frames"].to_dict("records")
            if arm == "historical":
                reference = {name: products[name].copy() for name in R.D.STAGES}
            else:
                contrasts[arm] = V.structural_comparisons(reference, products, d["registry"])
            del products
        expected = dict(batch=batch, seed=seed, source_counts=counts, photon_counts=M.counts(q),
                        receipts=receipts, comparisons=contrasts)
        _equal(SR._json(child / "validation.json"), expected, "child saved validation changed")
        _equal(record["counts"], receipts, "child response counts changed")
        records.append(dict(path=tag, batch=batch, seed=seed, input_sha256=digest, source_counts=counts,
                            photon_counts=M.counts(q), counts=receipts, stage_rows=stages, elapsed_seconds=elapsed))
        comparisons[tag] = contrasts
        del q, selection, reference
    ledger = pd.concat(coverage, ignore_index=True).sort_values(["batch", "optical_seed", "arm", "frame_id"]).reset_index(drop=True)
    S.validate_coverage(ledger, d)
    saved_coverage = pd.read_parquet(root / "coverage.parquet")
    S.validate_coverage(saved_coverage, d)
    pd.testing.assert_frame_equal(saved_coverage, ledger, check_exact=True)
    _equal(summary["root_union"], roots.summary(), "global photon root union changed")
    _require(total <= caps["maximum_total"], "total persisted byte cap exceeded")
    totals = {arm: {stage: sum(r["stage_rows"][arm][stage] for r in records) for stage in R.STAGES} for arm in S.ARMS}
    M.check_bindings(summary["inputs"], summary["code_sha256"])
    final_top, final_total = SR._batch(root, S.context(d), names, S.TOP_CAP)
    _equal(final_top, top, "top transaction changed during verification")
    for record in summary["records"]:
        child = directory / record["path"]
        final_child, size = SR._batch(child, S.context(d, record["batch"], record["seed"]), S.CHILD_NAMES, S.CHILD_CAP)
        _require(S.C.sha256(child / A.COMPLETE) == record["receipt_sha256"], "child changed during verification")
        _equal(final_child, SR._json(root / (record["path"] + ".json")), "copied child changed during verification")
        final_total += size
    _require(final_total == total, "persisted byte accounting changed during verification")
    result = dict(schema=SCHEMA, scope=SCOPE, summary_sha256=S.C.sha256(root / "summary.json"), hashes=hashes,
                  registry_sha256=F.digest(d["registry"]), registry_frames=len(d["registry"]), frames=len(d["selected"]),
                  batches=len(d["chosen"]), seeds=list(S.SEEDS), arms=list(S.ARMS), records=records,
                  structural_comparisons=comparisons, frame_counts=frame_counts, stage_totals=totals,
                  root_union=roots.summary(), persisted_path_bytes=total, byte_caps=caps, generation_measurements=measurements,
                  saved_artifact_protocol_exact=True, source_admission_exact=True, typed_source_fingerprints_exact=True,
                  photon_timing_fields_exact=True, authoritative_all_live_conditions_exact=True, whole_selected_frames_exact=True,
                  all_declared_arm_seeds_complete=True, regenerated_optics=False, regenerated_response=False,
                  execution_admitted=False, full_retained_campaign_admitted=False, F=None, finite_library_uncertainty=None,
                  timing_closure=False, code_sha256=S.C.sha256(__file__))
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def compact(result):
    """Numeric view of already verified evidence, with the same non-admission scope."""
    fields = ("photons", "dark_candidates", "attempts", "charges", "digis", "pulses")
    counts = [dict(batch=r["batch"], seed=r["seed"], arm=arm, **{k: c[k] for k in fields})
              for r in result["records"] for arm, c in r["counts"].items()]
    contrast_counts = [dict(path=tag, arm=arm, stage=stage, **{k: value[k] for k in ("common", "only_a", "only_b", "changed_common", "exact")})
                       for tag, arms in result["structural_comparisons"].items() for arm, stages in arms.items() for stage, value in stages.items()]
    return {k: result[k] for k in ("summary_sha256", "frames", "batches", "stage_totals", "root_union",
                                  "persisted_path_bytes", "generation_measurements", "scope")} | {
        "counts": counts, "contrast_counts": contrast_counts, "candidate_photons": [dict(path=r["path"], **r["photon_counts"]) for r in result["records"]],
        "execution_admitted": False, "full_retained_campaign_admitted": False, "F": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    for name in ("plan", "cache", "response"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    parser.add_argument("--compact", action="store_true")
    args = vars(parser.parse_args(argv))
    small = args.pop("compact")
    result = verify(**args)
    print(json.dumps(compact(result) if small else result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
