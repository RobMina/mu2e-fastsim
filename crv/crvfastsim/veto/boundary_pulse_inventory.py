"""Inspect an existing local copy of boundary pulse tables; no finder, refit or raw reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import saved_pulse_inventory as I

M, F, A, SR, P = I.M, I.F, I.A, I.SR, I.P
require = SR._require
IDS = list(M.C.IDS)
SCHEMA = "boundary-saved-pulse-inventory-v1"
SCOPE = ("Local saved pulse/waveform/time inventory only. No finder, refit, response, stock, raw-input reads or "
         "full-registry admission. Refit pulses are not identical stock pulses or new diagnostic-fit outputs.")


def event_census(event_summary, stock_pulses):
    """Restore absent run/subrun only from a bound complete-per-target saved stock collection."""
    require(set(IDS) <= set(stock_pulses) and "event" in event_summary and event_summary.event.is_unique,
            "missing or duplicate boundary event summary")
    identities = stock_pulses[IDS].drop_duplicates()
    for n in IDS:
        P._integer(identities[n], n, 2**32)
    require(identities.event.is_unique and set(identities.event) == set(event_summary.event),
            "stock rows do not supply unique identity for every boundary target")
    if {"run", "subrun"} <= set(event_summary):
        joined = event_summary.merge(identities, on=IDS, how="left", validate="one_to_one", indicator=True)
    else:
        require(not {"run", "subrun"} & set(event_summary), "partial boundary identity")
        joined = event_summary.merge(identities, on="event", how="left", validate="one_to_one", indicator=True, sort=False)
    require(joined._merge.eq("both").all(), "boundary census identity differs")
    return joined.drop(columns="_merge")


def inspect(p, w, conditions, events, input_sha):
    M.S.I._digest(input_sha)
    require(len(p) <= 2048 and len(w) <= 2048 and len(conditions) <= 20480 and 0 < len(events) <= 3,
            "boundary inventory row cap exceeded")
    require("pbtime" in events and events.pbtime.dtype.kind in "fiu" and np.isfinite(events.pbtime).all(),
            "nonfinite boundary census bunch time")
    with np.errstate(over="ignore"):
        rounded_pbtime = events.pbtime.to_numpy().astype(np.float32)
    require(np.isfinite(rounded_pbtime).all(), "boundary census bunch time exceeds float32")
    require(events.run.nunique() == 1 and events.subrun.nunique() == 1 and events.event.is_unique,
            "ambiguous boundary full-event census")
    for n in IDS:
        P._integer(events[n], n, 2**32)
    require(set(["wf", "seq", "chan", "event", "time", "time_relative", "time_reco", "pe", "failed", "zerondf", "channel_ignored"]) <= set(p),
            "boundary pulse fields missing")
    require(set(["wf", "event", "chan", "starttdc"]) <= set(w), "boundary waveform fields missing")
    require("starttdc" not in p, "legacy pulse start TDC unexpectedly present")
    for t in (p, w, conditions):
        require(t.columns.is_unique and t.event.isin(events.event).all(), "boundary table outside census")
        P._integer(t.event, "event", 2**32)
        P._integer(t.chan, "chan", 5120)
    for n in ("wf", "seq"):
        P._integer(p[n], n, 2**31)
    P._integer(w.wf, "waveform", 2**31)
    P._integer(w.starttdc, "starttdc", 2**31)
    require(not w.wf.duplicated().any() and not w.duplicated(["event", "chan", "starttdc"]).any(),
            "boundary waveform mapping not bijective")
    require(not conditions.duplicated(["event", "chan"]).any(), "duplicate boundary conditions")
    q = p.merge(w[["wf", "event", "chan", "starttdc"]], on=["wf", "event", "chan"], how="left", validate="many_to_one", indicator=True)
    require(q._merge.eq("both").all(), "boundary pulse waveform link missing")
    q = q.drop(columns="_merge")
    require(not q.duplicated(["event", "chan", "starttdc", "seq"]).any(), "duplicate boundary structural pulse")
    native, structural = q.sort_values(["event", "wf", "seq"]).index, q.sort_values(["event", "chan", "starttdc", "seq"]).index
    require(np.array_equal(native, structural) and np.array_equal(native, np.arange(len(q))), "boundary native/stored order differs")
    for n in ("time", "time_relative", "time_reco", "pe"):
        require(q[n].dtype == np.dtype("float64") and np.isfinite(q[n]).all(), "invalid boundary floating field: " + n)
    require((q.pe >= 0).all(), "negative boundary PE")
    for n in ("failed", "zerondf", "channel_ignored"):
        require(q[n].dtype == bool, "invalid boundary flag: " + n)
    q = q.merge(conditions[["event", "chan", "timeoffset", "w", "status"]], on=["event", "chan"],
                how="left", validate="many_to_one", indicator=True)
    require(q._merge.eq("both").all(), "boundary pulse conditions missing")
    require(np.isfinite(q[["timeoffset", "w"]]).all().all(), "nonfinite boundary conditions")
    status = P._integer(q.status, "status", 2**16)
    require(np.array_equal(q.channel_ignored, (status & 2) != 0), "boundary ignored flag differs")
    pb = q.event.map(events.set_index("event").pbtime).to_numpy().astype(np.float32).astype(float)
    require(np.array_equal(q.w, -pb), "boundary bunch shift differs")
    for actual, expected, label in ((q.time_relative, q.time + q.timeoffset, "relative"),
                                    (q.time_reco, q.time_relative - pb, "reco")):
        require(actual.to_numpy().tobytes() == expected.to_numpy().tobytes(), "boundary " + label + " time bits differ")
    counts = events[IDS].copy().assign(input_sha256=input_sha)
    counts["saved_pulses"] = counts.event.map(q.groupby("event").size()).fillna(0).astype(np.int64)
    counts["ignored_pulses"] = counts.event.map(q.loc[q.channel_ignored].groupby("event").size()).fillna(0).astype(np.int64)
    counts["nonlive_bit_pulses"] = counts.event.map(q.loc[(status & 5) != 0].groupby("event").size()).fillna(0).astype(np.int64)
    require((counts.saved_pulses <= 512).all(), "boundary per-frame pulse cap exceeded")
    return dict(frames=counts.to_dict("records"), pulse_rows=len(p), waveform_rows=len(w),
                original_relative_and_forward_reco_bits_exact=True, native_structural_order_exact=True,
                waveform_start_recovered_by_saved_link=True, full_registry_admitted=False,
                nonempty_membership_established=False)


def run(downstream, boundary, verified, out, *, expected_downstream_sha256,
        expected_boundary_sha256, expected_verified_sha256):
    downstream, boundary, verified, out = map(Path, (downstream, boundary, verified, out))
    M._outside(out, [downstream, boundary, verified.parent])
    start = time.perf_counter()
    sources = list(dict.fromkeys([Path(__file__), Path(I.__file__), *M.code_paths()]))
    code = {str(p): M.S.C.sha256(p) for p in sources}
    bindings = {}
    def bound(path, sha):
        path = Path(path)
        require(SR._regular(path) <= I.FILE_CAP, "boundary bound-file cap exceeded")
        require(str(path) not in bindings or bindings[str(path)] == sha, "conflicting boundary binding")
        M.bind(path, sha)
        bindings[str(path)] = sha
        return path
    d = M.read_json(bound(downstream / "summary.json", expected_downstream_sha256))
    b = M.read_json(bound(boundary / "summary.json", expected_boundary_sha256))
    v = M.read_json(bound(verified, expected_verified_sha256))
    require(d["schema"] == "thread-d-saved-photon-downstream-v1" and v["schema"] == "thread-d-downstream-report-v1",
            "boundary report schema differs")
    require(d["input_summary_sha256"] == v["boundary_summary_sha256"] == expected_boundary_sha256
            and v["input_summary_sha256"] == expected_downstream_sha256 and v["original_inputs_verified"] is True,
            "boundary report provenance differs")
    require(len(d["blocks"]) == len(b["blocks"]) == 2
            and {r["block"] for r in d["blocks"]} == {r["block"] for r in b["blocks"]} == {0, 1}
            and sum(len(r["targets"]) for r in d["blocks"]) == 3, "boundary block schedule differs")
    records = []
    for block in d["blocks"]:
        number = block["block"]
        require(type(number) is int and number in (0, 1), "invalid boundary block")
        prior = [r for r in b["blocks"] if r["block"] == number]
        require(len(prior) == 1 and prior[0]["targets"] == block["targets"], "boundary target census differs")
        input_shas = [sha for path, sha in prior[0]["input_sha256"].items() if path.endswith("/mixed.root")]
        require(len(input_shas) == 1, "missing boundary input identity")
        def read(root, report, name, cap):
            require(name in report["output_sha256"] and Path(name).name == name, "missing boundary payload binding")
            return I.table(bound(root / name, report["output_sha256"][name]), cap)
        events = read(boundary, b, f"block{number}_events.parquet", 3)
        stock_identity = read(downstream, d, f"block{number}_stock_pulses.parquet", 2048)
        events = event_census(events, stock_identity)
        require(sorted(events.event) == sorted(block["targets"]), "boundary event targets differ")
        cond = read(downstream, d, f"block{number}_conditions.parquet", 20480)
        require(set(block["models"]) == {f"{policy}_thermal{t}" for policy in ("legacy_round", "fixed_integer") for t in (0, 1)},
                "boundary model schedule differs")
        tags = [f"block{number}_stock_adc_refit"]
        for model, arms in block["models"].items():
            require(len(arms) == 3 and {r["arm"] for r in arms} == {"step_only", "arrival_only", "stock_timing"},
                    "boundary arm schedule differs")
            tags.extend(f"block{number}_{model}_{row['arm']}" for row in arms)
        for tag in tags:
            p = read(downstream, d, tag + "_pulses.parquet", 2048)
            w = read(downstream, d, tag + "_waveforms.parquet", 2048)
            result = inspect(p, w, cond, events, input_shas[0])
            records.append(dict(tag=tag, stock_adc_refit=tag.endswith("stock_adc_refit"), **result))
    require(len(records) == 26 and len({r["tag"] for r in records}) == 26, "boundary inventory incomplete")
    result = dict(schema=SCHEMA, scope=SCOPE, inputs=bindings, code_sha256=code, records=records,
                  unique_frames=sum(len(r["targets"]) for r in d["blocks"]), regenerated_finder=False,
                  regenerated_response=False, regenerated_fit=False, inherited_raw_admission_repeated=False,
                  full_registry_admitted=False, execution_admitted=False, timing_closure=False,
                  F=None, finite_library_uncertainty=None, seconds=time.perf_counter() - start, memory=M.S.C.memory())
    snapshots = {path: Path(path).read_text() for path in code}
    M.check_bindings(M.S.C.merge_bindings(bindings, code))
    payload = A._canonical(result | {"code_snapshots": snapshots, "output_cap_bytes": I.CAP,
                                      "publication": "exclusive JSON file; no completion protocol or crash-durability claim"})
    require(len(payload) <= I.CAP, "boundary inventory output byte cap exceeded")
    with out.open("xb") as stream:
        stream.write(payload)
    return result | {"persisted_path_bytes": len(payload)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("downstream", "boundary", "verified", "out"):
        parser.add_argument("--" + name, required=True)
    for name in ("downstream", "boundary", "verified"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("schema", "records", "unique_frames", "seconds", "memory", "persisted_path_bytes", "scope")}, indent=2))


if __name__ == "__main__":
    main()
