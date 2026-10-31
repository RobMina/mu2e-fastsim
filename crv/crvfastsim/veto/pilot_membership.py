"""Bounded coincidence and frozen-grid diagnostics on previously verified saved pilot pulses."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import geometry
from . import pilot_coincidence as P, pilot_coverage as G
from . import streaming_pilot_report as T
from . import replay_recover as C, deadtime, frameaudit, card, apply

S, SR, F, A = T.S, T.SR, T.F, T.A
SCHEMA = "saved-pilot-membership-v1"
OUTPUT_CAP = 32 * 1024**2
SELECTION_CAP = 256 * 1024**2
IGNORE_SOURCE = Path("/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_41_00/Offline/CRVReco/src/CrvRecoPulsesFinder_module.cc")
IGNORE_SOURCE_SHA = "f432feb9823fc76cd6a61e4b6309cb65f5e260726c69ed76141bf5ef300e8b0c"
# Fixed-stock replay configurations, relative to the CRV data root (CRVFS_DATA); IGNORE_CONFIGS overrides when set.
STOCK_BASE = "deadtime_run1a/replay_d_20261025/fixed_stock_v13_41_00_20261025"
STOCK_CONFIGS = {
    "block0/effective_runtime.fcl": "826c274afdd73f611d7a6c99e640532119b85144c1965e5aad63bee48a0fc44b",
    "block1/effective_runtime.fcl": "a8d259be65b936d055a81f57223509944612bbdf22c975f51a053efbf9b52be2",
    "block2_fullinventory/effective_runtime.fcl": "ac076d8215fd3befdb2dc540515719c2ab0432030351b59a84f01e9a138594e3",
}
IGNORE_CONFIGS = None


def ignore_configs():
    if IGNORE_CONFIGS is not None:
        return IGNORE_CONFIGS
    from mu2efastsim import site
    base = Path(site.require("CRVFS_DATA")) / STOCK_BASE
    return {base / name: digest for name, digest in STOCK_CONFIGS.items()}
SCOPE = ("Nine-frame maximum saved-pulse integration diagnostic. Native membership is internal to the "
         "unchanged reference finder, not a same-pulse stock comparison. Frozen track-grid values are "
         "per-frame conditional diagnostics, never a pilot F, new exposure or uncertainty estimate. "
         "Admission rehashes completed pilot artifacts and the externally pinned prior verification; "
         "it does not repeat raw-input admission or regenerate optics, response, fits or stock.")
_require, _equal = SR._require, SR._equal


def code_paths():
    modules = (P, G, C, C.C, geometry, deadtime, frameaudit, card, apply, T, S, SR, F, A, S.C)
    return list(dict.fromkeys([Path(__file__), *(Path(m.__file__) for m in modules), *S.code_paths()]))


def bind(path, digest):
    path = Path(path)
    S.I._digest(digest)
    SR._regular(path)
    _require(S.C.sha256(path) == digest, "changed bound input: " + str(path))
    return path


def read_json(path, cap=8 * 1024**2):
    _require(SR._regular(Path(path)) <= cap, "JSON byte cap exceeded")
    value = SR._json(path)
    _require(type(value) is dict, "JSON object required")
    return value


def transaction_bindings(root, receipt):
    artifacts = {str(root / name): value["sha256"] for name, value in receipt["artifacts"].items()}
    digest = hashlib.sha256(A._canonical(receipt)).hexdigest()
    return artifacts | {str(root / A.COMPLETE): digest, str(root / A.PENDING): digest,
                        str(root / A.INCOMPLETE): receipt["incomplete_sha256"]}


def recheck(d):
    _require(set(p.name for p in d["pilot"].iterdir()) == {"run", *d["domains"]}, "pilot directory changed late")
    for root, context, names, cap, original in d["transactions"]:
        receipt, _ = SR._batch(root, context, names, cap)
        _equal(receipt, original, "pilot transaction changed late")
    check_bindings(d["inputs"])


def _read(path, rows=None):
    S.C.parquet_admit(path, rows)
    return pd.read_parquet(path)


def _outside(out, roots):
    _require(not out.exists() and not out.is_symlink(), "choose a new output directory")
    S.outside_evidence(out, roots)


def ignore_bindings():
    """Bind the reviewed stock ignore-channel policy, not a new pulse-quality selection."""
    bind(IGNORE_SOURCE, IGNORE_SOURCE_SHA)
    result = {str(IGNORE_SOURCE): IGNORE_SOURCE_SHA}
    for path, digest in ignore_configs().items():
        text = bind(path, digest).read_text()
        values = re.findall(r"(?m)^\s*ignoreChannels:\s*(true|false)\b", text)
        _require(values == ["true"], "fixed stock ignoreChannels contract differs")
        result[str(path)] = digest
    return result


def admit(pilot, verified, paired, *, expected_pilot_sha256, expected_verified_sha256, expected_paired_sha256):
    """Reuse one pinned successful verification, with fresh complete pilot transaction hashing."""
    pilot, verified, paired = map(Path, (pilot, verified, paired))
    summary = read_json(bind(pilot / "run/summary.json", expected_pilot_sha256))
    v = read_json(bind(verified, expected_verified_sha256))
    pair = read_json(bind(paired, expected_paired_sha256))
    _require({"selection", "reference", "selection_sha256", "reference_sha256"} <= set(pair), "missing paired bindings")
    _require({"schema", "summary_sha256", "code_sha256", "hashes", "seeds", "arms", "registry_sha256", "registry_frames",
              "frames", "batches", "F", "finite_library_uncertainty", "persisted_path_bytes"} <= set(v), "incomplete prior report")
    _require(set(summary) == T.SUMMARY_KEYS and summary["schema"] == S.SCHEMA, "pilot summary schema changed")
    _require(v["schema"] == T.SCHEMA and v["summary_sha256"] == expected_pilot_sha256,
             "prior verification belongs to another pilot")
    _require(v["code_sha256"] == S.C.sha256(T.__file__), "prior verifier code changed")
    _equal(v["hashes"], summary["hashes"], "verification admission hashes differ")
    for name in ("saved_artifact_protocol_exact", "source_admission_exact", "typed_source_fingerprints_exact",
                 "photon_timing_fields_exact", "authoritative_all_live_conditions_exact", "whole_selected_frames_exact",
                 "all_declared_arm_seeds_complete"):
        _require(v.get(name) is True, "missing prior verification gate: " + name)
    for name in ("execution_admitted", "full_retained_campaign_admitted", "timing_closure", "regenerated_optics", "regenerated_response"):
        _require(v.get(name) is False, "prior verification scope changed")
    _require(v.get("F") is None and v.get("finite_library_uncertainty") is None, "prior verification promotion changed")
    for name in ("seeds", "arms", "registry_sha256", "registry_frames", "frames", "batches"):
        _equal(v[name], summary[name], "verification domain differs: " + name)
    _equal(summary["seeds"], list(S.SEEDS), "seed domain changed")
    _equal(summary["arms"], list(S.ARMS), "arm domain changed")
    _require(type(summary["frames"]) is int and 0 < summary["frames"] <= S.MAX_FRAMES,
             "pilot frame cap exceeded")
    _require(type(summary["batches"]) is int and 0 < summary["batches"] <= S.MAX_BATCHES,
             "pilot batch cap exceeded")
    root = pilot / "run"
    records = summary["records"]
    _require(len(records) == summary["batches"] * len(S.SEEDS), "incomplete pilot children")
    tags = [r["path"] for r in records]
    _require(len(set(tags)) == len(tags) and all(t == f"batch{int(r['batch']):04d}_seed{r['seed']}" for t, r in zip(tags, records)),
             "invalid child schedule")
    _require(set(p.name for p in pilot.iterdir()) == {"run", *tags}, "pilot directory domain changed")
    old_code = {str(p): S.C.sha256(p) for p in S.code_paths()}
    _equal(summary["code_sha256"], old_code, "generation source changed")
    names = ["summary.json", "registry.parquet", "selected.parquet", "chosen.parquet", "calibration.parquet",
             "coverage.parquet", "admission.json", *(t + ".json" for t in tags), *(f"code{i}.txt" for i in range(len(old_code)))]
    context = dict(registry_sha256=summary["registry_sha256"], batch_identity={"contract": S.SCHEMA, **summary["hashes"]},
                   arms=list(S.ARMS), seeds=list(S.SEEDS))
    top, total = SR._batch(root, context, names, S.TOP_CAP)
    _require(top["artifacts"]["summary.json"]["sha256"] == expected_pilot_sha256, "pilot summary changed during admission")
    transactions = [(root, context, names, S.TOP_CAP, top)]
    input_bindings = transaction_bindings(root, top) | {str(verified): expected_verified_sha256, str(paired): expected_paired_sha256}
    for i, digest in enumerate(old_code.values()):
        bind(root / f"code{i}.txt", digest)
    registry, selected, chosen, calibration = (_read(root / (name + ".parquet"))
                                              for name in ("registry", "selected", "chosen", "calibration"))
    F.validate(registry)
    digest = F.digest(registry)
    _require(digest == summary["registry_sha256"] and len(registry) == summary["registry_frames"], "registry changed")
    selected = S.R.selected_frames(registry, selected, digest)
    _require(len(selected) == summary["frames"] and len(chosen) == summary["batches"], "selected domain changed")
    _require(chosen.batch.dtype.kind in "iu" and not chosen.batch.duplicated().any(), "invalid planned batch keys")
    _require(chosen.input_sha256.is_unique and set(chosen.input_sha256) == set(selected.input_sha256), "incomplete input domain")
    schedule = [(int(b), seed) for b in chosen.batch for seed in S.SEEDS]
    _equal([(r["batch"], r["seed"]) for r in records], schedule, "noncanonical child schedule")
    saved_ledger = _read(root / "coverage.parquet")
    assigned = selected.merge(chosen[["batch", "input_sha256"]], on="input_sha256", validate="many_to_one")
    S.validate_coverage(saved_ledger, dict(selected=selected, assigned=assigned))
    domains = {}
    for r in records:
        child = pilot / r["path"]
        subset = selected[selected.input_sha256 == r["input_sha256"]].reset_index(drop=True)
        _require(0 < len(subset) <= 4, "invalid child frame domain")
        child_context = dict(context, batch_identity={**context["batch_identity"], "batch": r["batch"],
                                                     "frame_ids": subset.frame_id.astype(int).tolist()}, seeds=[r["seed"]])
        receipt, size = SR._batch(child, child_context, S.CHILD_NAMES, S.CHILD_CAP)
        transactions.append((child, child_context, S.CHILD_NAMES, S.CHILD_CAP, receipt))
        input_bindings = S.C.merge_bindings(input_bindings, transaction_bindings(child, receipt))
        total += size
        bind(child / A.COMPLETE, r["receipt_sha256"])
        _equal(SR._json(root / (r["path"] + ".json")), receipt, "child receipt copy changed")
        domains[r["path"]] = subset
        for arm in S.ARMS:
            fr = _read(child / f"{arm}_frames.parquet", len(subset))
            SR._table_equal(fr[list(F.COLUMNS)], subset, ["frame_id"])
            wanted = saved_ledger[(saved_ledger.batch == r["batch"]) & (saved_ledger.optical_seed == r["seed"]) & (saved_ledger.arm == arm)]
            SR._table_equal(fr, wanted[fr.columns], ["frame_id"])
            _require(int(fr.pulses.sum()) == r["counts"][arm]["pulses"], "saved pulse count differs")
    _require(total == v["persisted_path_bytes"] and total <= summary["byte_caps"]["maximum_total"], "pilot byte accounting differs")
    counters = S.one_asset(summary["inputs"], S.M.P.COUNTERS_SHA)
    input_bindings[str(counters)] = S.M.P.COUNTERS_SHA
    for name in ("selection", "reference"):
        path = bind(pair[name], pair[name + "_sha256"])
        input_bindings[str(path)] = pair[name + "_sha256"]
    d = dict(pilot=pilot, verified=verified, paired=paired, summary=summary, verification=v, pair=pair,
             registry=registry, selected=selected, chosen=chosen, calibration=calibration, domains=domains,
             inputs=input_bindings, counters=counters, saved_ledger=saved_ledger, transactions=transactions)
    d["inputs"] = S.C.merge_bindings(d["inputs"], ignore_bindings())
    recheck(d)
    return d


def pulse_admission(counts):
    counts = np.asarray(counts)
    limits = P.Limits()
    _require(counts.dtype.kind in "iu" and len(counts) <= limits.max_frames and np.all(counts >= 0)
             and np.all(counts <= limits.max_pulses_per_frame), "per-frame pulse cap exceeded before decode")
    n = counts.astype(np.int64)
    _require(int(n.sum()) <= limits.max_total_pulses, "total pulse cap exceeded before decode")


def saved_pulses(path, expected_rows):
    """Require the frozen fit collection order to equal the canonical physical bridge."""
    _require(type(expected_rows) is int and 0 <= expected_rows <= P.Limits().max_total_pulses, "pulse row cap before decode")
    pulses = _read(path, expected_rows)
    required = {"frame_id", "wf", "seq", "chan", "starttdc", "failed", "zerondf", "channel_ignored"}
    _require(pulses.columns.is_unique and required <= set(pulses), "missing saved fit identity/flags")
    for name in ("frame_id", "wf", "seq", "chan", "starttdc"):
        _require(pulses[name].dtype.kind in "iu", "noninteger saved collection identity")
    for name in ("failed", "zerondf", "channel_ignored"):
        _require(pulses[name].dtype.kind == "b", "nonboolean saved fit flag")
    native = pulses.sort_values(["frame_id", "wf", "seq"]).index.to_numpy()
    structural = pulses.sort_values(["frame_id", "chan", "starttdc", "seq"]).index.to_numpy()
    _require(np.array_equal(native, structural), "native fit order differs from structural pulse bridge")
    _require(np.array_equal(native, np.arange(len(pulses))), "saved pulse collection order changed")
    mapping = pulses[["wf", "frame_id", "chan", "starttdc"]].drop_duplicates()
    _require(not mapping.wf.duplicated().any() and not mapping.duplicated(["frame_id", "chan", "starttdc"]).any(),
             "non-bijective saved waveform identity")
    return pulses


def load_grid(d):
    """Read only frozen track columns after metadata caps, never reread ntuples or rebuild selections."""
    path = Path(d["pair"]["selection"])
    _require(SR._regular(path) <= SELECTION_CAP, "selection file byte cap exceeded")
    meta = pq.ParquetFile(path).metadata
    _require(meta.num_rows <= 3_000_000 and meta.num_row_groups <= 4096 and meta.num_columns <= 128,
             "selection metadata cap exceeded")
    total = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
    _require(total <= 1024**3, "selection uncompressed metadata cap exceeded")
    parquet = pq.ParquetFile(path)
    for name in ("t_mid", "p_mid"):
        indices = [i for i in range(meta.num_columns) if parquet.schema.column(i).path.split(".")[0] == name]
        _require(len(indices) == 1, "selection requires one numeric leaf per track field")
        column = parquet.schema.column(indices[0])
        _require(column.physical_type in ("FLOAT", "DOUBLE", "INT32", "INT64") and column.max_repetition_level == 1,
                 "invalid jagged track schema")
        values = sum(meta.row_group(i).column(indices[0]).num_values for i in range(meta.num_row_groups))
        _require(values <= 3_000_000, "selection jagged value cap exceeded before decode")
    selection = ak.from_parquet(path, columns=["t_mid", "p_mid"])
    return G.frozen_grid(read_json(Path(d["pair"]["reference"])), selection)


def geometry_only(d):
    path = d["counters"]
    geom = geometry.from_offline_config(str(path))
    _require(geom.nbars == 1280 and geom.sector_names == ["T1", "T2"], "not pinned Run1A geometry")
    return C.C.sector_geom(geom, geometry.parse_simpleconfig(path.read_text()))


def check_bindings(bindings):
    for path, digest in bindings.items():
        bind(path, digest)


def validate_ledger(ledger, d):
    keys = [*F.FRAME_KEYS, "frame_id", "batch", "optical_seed", "response_seed", "arm"]
    expected = d["saved_ledger"][keys]
    _require(len(ledger) == len(expected) and not ledger.duplicated(keys).any(), "incomplete coincidence coverage ledger")
    SR._table_equal(ledger[keys], expected, ["batch", "optical_seed", "arm", "frame_id"])


def run(pilot, verified, paired, out, **hashes):
    """Publish new bounded evidence; no execution path for optical/response/stock regeneration."""
    pilot, verified, paired, out = map(Path, (pilot, verified, paired, out))
    _outside(out, [pilot, verified.parent, paired.parent])
    d = admit(pilot, verified, paired, **hashes)
    _outside(out, [Path(d["summary"][n]) for n in ("plan", "cache_audit", "response")])
    _outside(out, [Path(d["pair"][n]).parent for n in ("selection", "reference")])
    cfg, hist, edges, grid_meta = load_grid(d)
    geom = geometry_only(d)
    code = {str(p): S.C.sha256(p) for p in code_paths()}
    all_bindings = S.C.merge_bindings(d["inputs"], code)
    check_bindings(all_bindings)
    tables = {name: [] for name in ("pulses", "clusters", "members", "coverage")}
    comparisons, counts = {}, []
    start = time.perf_counter()
    before = S.C.memory()
    for record in d["summary"]["records"]:
        tag = record["path"]
        reference = None
        for arm in S.ARMS:
            child = pilot / tag
            row = d["saved_ledger"][(d["saved_ledger"].batch == record["batch"]) &
                                    (d["saved_ledger"].optical_seed == record["seed"]) & (d["saved_ledger"].arm == arm)]
            pulse_admission(row.pulses.to_numpy())
            pulses = saved_pulses(child / f"{arm}_pulses.parquet", int(row.pulses.sum()))
            products = P.run(pulses, d["registry"], d["domains"][tag], d["calibration"], geom,
                             registry_digest=d["summary"]["registry_sha256"])
            products["pulses"] = products["pulses"].merge(pulses[[*P.FRAME_KEYS, *P.PULSE_KEYS, "failed", "zerondf"]],
                                                        on=[*P.FRAME_KEYS, *P.PULSE_KEYS], validate="one_to_one")
            products["coverage"] = G.evaluate(products["frames"], products["clusters"], cfg, hist, edges)
            annotation = dict(batch=record["batch"], optical_seed=np.uint64(record["seed"]),
                              response_seed=np.uint64(record["seed"]), arm=arm)
            for name in tables:
                tables[name].append(products[name].assign(**annotation))
            counts.append(dict(batch=record["batch"], seed=record["seed"], arm=arm,
                               frames=len(products["frames"]), pulses=len(products["pulses"]),
                               ignored_channel_pulses=int(products["pulses"].channel_ignored.sum()),
                               finder_pulses=int((~products["pulses"].channel_ignored).sum()),
                               coincidences=len(products["clusters"]), members=len(products["members"]),
                               gate_fail=int((~products["coverage"].gate_pass).sum()),
                               nonzero_coverage=int((products["coverage"].track_grid_coverage != 0).sum())))
            if arm == "historical":
                reference = products
            else:
                comparison = P.compare({k: reference[k] for k in ("frames", "pulses", "clusters", "members")},
                                       {k: products[k] for k in ("frames", "pulses", "clusters", "members")})
                comparison["coverage_exact"] = reference["coverage"].equals(products["coverage"])
                comparisons[tag + "_" + arm] = comparison
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    validate_ledger(tables["coverage"], d)
    result = dict(schema=SCHEMA, scope=SCOPE, hashes=hashes, inputs=d["inputs"], code_sha256=code,
                  registry_sha256=d["summary"]["registry_sha256"], unique_frames=len(d["selected"]),
                  ledger_rows=len(tables["coverage"]), seeds=list(S.SEEDS), arms=list(S.ARMS), counts=counts,
                  comparisons=comparisons, grid=grid_meta, finder_config=asdict(C.finder_config()),
                  ignore_channels=True, ignored_pulse_policy="retain all saved rows; exclude status bit1 only from finder; links use original saved idx", 
                  coincidence_limits=asdict(P.Limits()), output_cap_bytes=OUTPUT_CAP,
                  memory_before=before, memory_after=S.C.memory(), coincidence_coverage_seconds=time.perf_counter() - start,
                  inherited_raw_admission_repeated=False, regenerated_optics=False, regenerated_response=False,
                  stock_rerun=False, same_pulse_stock_validation=False, execution_admitted=False,
                  full_retained_campaign_admitted=False, timing_closure=False, F=None, finite_library_uncertainty=None)
    context = dict(registry_sha256=result["registry_sha256"], batch_identity={"contract": SCHEMA, **hashes},
                   arms=list(S.ARMS), seeds=list(S.SEEDS))
    names = ["summary.json", "registry.parquet", "selected.parquet", "grid.parquet", *(n + ".parquet" for n in tables),
             *(f"code{i}.txt" for i in range(len(code)))]
    with A.BatchArtifacts(out, max_bytes=OUTPUT_CAP, context=context, expected_artifacts=names) as tx:
        tx.write_parquet("registry.parquet", d["registry"])
        tx.write_parquet("selected.parquet", d["selected"])
        tx.write_parquet("grid.parquet", pd.DataFrame(dict(left=edges[:-1], right=edges[1:], weight=hist)))
        for name, table in tables.items():
            tx.write_parquet(name + ".parquet", table)
        for i, path in enumerate(code):
            with tx.open_artifact(f"code{i}.txt") as stream:
                stream.write(Path(path).read_bytes())
        check_bindings(all_bindings)
        recheck(d)
        tx.write_json("summary.json", result)
    receipt, size = SR._batch(out, context, names, OUTPUT_CAP)
    return result | {"persisted_path_bytes": size, "complete_sha256": S.C.sha256(out / A.COMPLETE)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "verified", "paired", "out"):
        parser.add_argument("--" + name, required=True)
    for name in ("pilot", "verified", "paired"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("schema", "unique_frames", "ledger_rows", "counts", "comparisons", "grid",
                                            "persisted_path_bytes", "complete_sha256", "F", "scope")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
