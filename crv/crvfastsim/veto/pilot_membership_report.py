"""Verify pinned saved membership products, without running a finder or upstream raw admission."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import pilot_membership as M

P, G, S, SR, F, A = M.P, M.G, M.S, M.SR, M.F, M.A
_require, _equal = SR._require, SR._equal
SCHEMA = "saved-pilot-membership-report-v1"
CAP = 32 * 1024**2
SCOPE = ("Saved-only, externally pinned membership diagnostic. Source pulse identities, native link structure, "
         "frame census and historical frozen-grid coverage are checked. Scalar coincidence properties remain "
         "hash-bound generation observations, not independently finder-recomputed physics. No finder regeneration, "
         "raw admission repeat, same-pulse stock closure, full-campaign admission, F or uncertainty estimate.")
FALSE_FIELDS = {"inherited_raw_admission_repeated", "regenerated_optics", "regenerated_response", "stock_rerun",
                "same_pulse_stock_validation", "execution_admitted", "full_retained_campaign_admitted", "timing_closure"}
NULL_FIELDS = {"F", "finite_library_uncertainty"}
SUMMARY_KEYS = {"schema", "scope", "hashes", "inputs", "code_sha256", "registry_sha256", "unique_frames",
                "ledger_rows", "seeds", "arms", "counts", "comparisons", "grid", "finder_config", "coincidence_limits",
                "ignore_channels", "ignored_pulse_policy", "output_cap_bytes", "memory_before", "memory_after", "coincidence_coverage_seconds", *FALSE_FIELDS, *NULL_FIELDS}
IGNORED_POLICY = "retain all saved rows; exclude status bit1 only from finder; links use original saved idx"
HASH_KEYS = {"expected_pilot_sha256", "expected_verified_sha256", "expected_paired_sha256"}
ANNOTATION = dict(batch="int64", optical_seed="uint64", response_seed="uint64", arm="object")
IDENTITY = {name: "object" if name == "input_sha256" else "int64" for name in P.FRAME_KEYS}
REGISTRY = {name: "float64" if name in ("pbtime", "intensity") else IDENTITY[name] for name in F.COLUMNS}
PULSES = {**IDENTITY, "idx": "int64", **{name: "int64" for name in (*P.PULSE_KEYS, "wf")},
          **{name: "float64" for name in ("time_relative", "time_reco", "pe")},
          **{name: "bool" for name in ("channel_ignored", "failed", "zerondf")}}
CLUSTERS = {**IDENTITY, "frame_id": "uint32", **{name: P.R.CLUSTER_TYPES[name] for name in P.SCALARS}}
MEMBERS = {**IDENTITY, "frame_id": "uint32", "cluster_idx": "int32", **{name: "int64" for name in
           ("member_order", "pulse_idx", *P.PULSE_KEYS, "layer", "sector")}}
COVERAGE = {**REGISTRY, **{name: "int64" for name in
            ("pulses", "clusters", "members", "n_coincidences", "n_veto_eligible")},
            "gate_pass": "bool", "track_grid_coverage": "float64"}
TABLE_TYPES = {"registry": REGISTRY, "selected": REGISTRY, "grid": dict(left="float64", right="float64", weight="float64"),
               "pulses": PULSES | ANNOTATION, "clusters": CLUSTERS | ANNOTATION,
               "members": MEMBERS | ANNOTATION, "coverage": COVERAGE | ANNOTATION}


def _table(path, types, max_rows):
    """Bound physical metadata and exact flat types before any table decoding."""
    _require(SR._regular(path) <= CAP, "saved Parquet file byte cap exceeded")
    with pq.ParquetFile(path) as parquet:
        meta, schema = parquet.metadata, parquet.schema_arrow
        _require(meta.num_rows <= max_rows and meta.num_row_groups <= 4096 and
                 meta.num_columns == len(types), "saved Parquet metadata cap/schema changed")
        sizes = [meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)]
        _require(all(0 <= size <= 128 * 1024**2 for size in sizes) and sum(sizes) <= 128 * 1024**2,
                 "saved Parquet uncompressed byte cap exceeded")
        _require(schema.names == list(types), "saved Parquet columns changed")
        for field, dtype in types.items():
            actual = schema.field(field).type
            valid = (pa.types.is_string(actual) or pa.types.is_large_string(actual) or
                     (meta.num_rows == 0 and pa.types.is_null(actual))) if dtype == "object" else actual == pa.from_numpy_dtype(dtype)
            _require(valid, "saved Parquet dtype changed: " + field)
    table = pd.read_parquet(path)
    _typed(table, types)
    return table


def _typed(table, types):
    _require(list(table) == list(types), "saved table columns changed")
    for name, dtype in types.items():
        _require(not table[name].isna().any(), "missing saved value: " + name)
        if dtype == "object":
            _require(table[name].dtype == object or isinstance(table[name].dtype, pd.StringDtype), "saved text dtype changed: " + name)
            _require(table[name].astype(object).map(lambda v: type(v) is str).all(), "nonstring saved value")
        else:
            _require(table[name].dtype == np.dtype(dtype), "saved table dtype changed: " + name)
        if dtype != "object" and table[name].dtype.kind in "fiu":
            _require(np.isfinite(table[name]).all(), "nonfinite saved value: " + name)


def _exact(actual, expected, label):
    actual, expected = actual.copy(), expected.copy()
    for name in expected:
        if expected[name].dtype == object or isinstance(expected[name].dtype, pd.StringDtype):
            for table in (actual, expected):
                _require(not table[name].isna().any() and table[name].astype(object).map(lambda v: type(v) is str).all(), "invalid text value")
                table[name] = table[name].astype(object)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True), check_exact=True)
    # Equality alone treats +0 and -0 alike; the pulse/grid bridge preserves bits.
    for name in expected:
        if expected[name].dtype.kind == "f":
            _require(actual[name].to_numpy().tobytes() == expected[name].to_numpy().tobytes(), label + " bits changed")


def _bound_path(bindings, digest):
    found = [Path(path) for path, value in bindings.items() if value == digest]
    _require(len(found) == 1, "missing or ambiguous source hash binding")
    return found[0]


def _census(table, frames, label):
    domain = pd.MultiIndex.from_frame(frames[P.FRAME_KEYS])
    _require(pd.MultiIndex.from_frame(table[P.FRAME_KEYS]).isin(domain).all(), label + " outside qualified frame census")


def _products(source, frames, registry, calibration, geom, saved):
    """Rebuild the canonical pulse bridge and validate saved native links by vector joins."""
    frames = S.R.selected_frames(registry, frames, F.digest(registry))
    sector, layer, sector_type = P._geometry(geom)
    status = P._status(calibration, frames.input_sha256.iloc[0], len(sector) * 4)
    p = source[[*P.FRAME_KEYS, *P.PULSE_FIELDS, "failed", "zerondf"]].copy()
    for name in ("run", "subrun", "event"):
        p[name] = P._integer(p[name], name, 2**32)
    p["frame_id"] = P._integer(p.frame_id, "frame_id", len(registry))
    for name, hi in (("chan", len(sector) * 4), ("starttdc", 2**31), ("seq", 2**31), ("wf", 2**63)):
        p[name] = P._integer(p[name], name, hi)
    for name in ("pe", "time_relative", "time_reco"):
        p[name] = P._real(p[name], name)
    _require((p.pe >= 0).all(), "negative saved pulse PE")
    _census(p, frames, "pulse")
    _require(not p.duplicated([*P.FRAME_KEYS, *P.PULSE_KEYS]).any(), "duplicate pulse structural key")
    pb = p.frame_id.map(frames.set_index("frame_id").pbtime).to_numpy().astype(np.float32).astype(float)
    _require(np.isfinite(pb).all() and np.array_equal(p.time_reco, p.time_relative.to_numpy() - pb),
             "saved pulse reco time changed")
    pulse_status = status[p.chan.to_numpy()]
    _require(p.channel_ignored.dtype == bool and np.array_equal(p.channel_ignored, (pulse_status & 2) != 0) and
             not np.any(pulse_status & 5), "pulse channel status changed")
    p = p.sort_values(["frame_id", *P.PULSE_KEYS]).reset_index(drop=True)
    p["idx"] = p.groupby("frame_id", sort=False).cumcount().astype(np.int64)
    p = p[list(PULSES)]
    _typed(p, PULSES)
    _exact(saved["pulses"], p, "canonical saved pulse")
    c, m = saved["clusters"], saved["members"]
    for table, types, label in ((c, CLUSTERS, "cluster"), (m, MEMBERS, "member")):
        _typed(table, types)
        _census(table, frames, label)
    _require(not c.duplicated(["frame_id", "idx"]).any(), "duplicate cluster identity")
    ordered = c.sort_values(["frame_id", "idx"]).reset_index(drop=True)
    _exact(c, ordered, "cluster order")
    _require(np.array_equal(c.idx, c.groupby("frame_id", sort=False).cumcount()), "noncanonical cluster indices")
    for name in ("idx", "sectortype", "nLayers", "sidehits0", "sidehits1"):
        P._integer(c[name], name, 2**31)
    _require(c.sectortype.isin(sector_type).all() and c.nLayers.between(1, 4).all(), "cluster sector/layer domain changed")
    _require((c.start <= c.end).all() and (c[["pe", "sidepe0", "sidepe1"]] >= 0).all().all(),
             "invalid cluster scalar domain")
    for name in ("cluster_idx", "member_order", "pulse_idx", *P.PULSE_KEYS, "layer", "sector"):
        P._integer(m[name], "member " + name)
    _require(not m.duplicated(["frame_id", "pulse_idx"]).any(), "duplicate native pulse membership")
    _require(not m.duplicated(["frame_id", "cluster_idx", "member_order"]).any(), "duplicate native member order")
    _exact(m, m.sort_values(["frame_id", "cluster_idx", "member_order"]), "member order")
    _require(np.array_equal(m.member_order, m.groupby(["frame_id", "cluster_idx"], sort=False).cumcount()),
             "nondense native member order")
    link = m.merge(p[[*P.FRAME_KEYS, "idx", *P.PULSE_KEYS, "channel_ignored"]].rename(columns={"idx": "pulse_idx"}),
                   on=[*P.FRAME_KEYS, "pulse_idx"], how="left", suffixes=("", "_expected"),
                   validate="many_to_one", indicator=True)
    _require(link._merge.eq("both").all(), "native pulse link outside canonical collection bounds")
    _require(not link.channel_ignored.any(), "native member references ignored pulse")
    for name in P.PULSE_KEYS:
        _require(np.array_equal(link[name], link[name + "_expected"]), "native member structural key changed")
    bars = m.chan.to_numpy() // 4
    _require(np.array_equal(m.layer, layer[bars]) and np.array_equal(m.sector, sector[bars]),
             "native member layer/sector changed")
    linked = m.merge(c[[*P.FRAME_KEYS, "idx", "sectortype"]].rename(columns={"idx": "cluster_idx"}),
                     on=[*P.FRAME_KEYS, "cluster_idx"], how="left", validate="many_to_one", indicator=True)
    _require(linked._merge.eq("both").all(), "native member references absent cluster")
    _require(np.array_equal(linked.sectortype, sector_type[sector[bars]]), "native member sector type changed")
    layer_counts = m.groupby(["frame_id", "cluster_idx"]).layer.nunique()
    actual_layers = pd.MultiIndex.from_frame(c[["frame_id", "idx"]]).map(layer_counts)
    _require(np.array_equal(actual_layers, c.nLayers), "cluster nLayers disagrees with member geometry or empty cluster")
    for name, table in (("pulses", p), ("clusters", c), ("members", m)):
        frames[name] = frames.frame_id.map(table.groupby("frame_id").size()).fillna(0).astype(np.int64)
    M.pulse_admission(frames.pulses.to_numpy())
    _require(P.combination_work(p, geom, M.C.finder_config()) <= P.Limits().max_combinations,
             "conservative finder combination cap exceeded")
    return dict(frames=frames, pulses=p, clusters=c, members=m)


def verify(directory, out, *, expected_summary_sha256):
    """Write an exclusive JSON report outside all input evidence; never regenerate a finder."""
    directory, out = Path(directory), Path(out)
    verifier_sha = S.C.sha256(__file__)
    M._outside(out, [directory])
    summary = M.read_json(M.bind(directory / "summary.json", expected_summary_sha256))
    _require(set(summary) == SUMMARY_KEYS and summary["schema"] == M.SCHEMA and summary["scope"] == M.SCOPE,
             "membership summary schema/scope changed")
    for name in FALSE_FIELDS:
        _require(summary[name] is False, "membership scope changed: " + name)
    for name in NULL_FIELDS:
        _require(summary[name] is None, "membership null scope changed: " + name)
    for name in ("memory_before", "memory_after"):
        value = summary[name]
        _require(type(value) is dict and set(value) == {"peak_rss_bytes_cumulative", "current_rss_bytes"},
                 "generation memory observation schema changed")
        for field, amount in value.items():
            _require((field == "current_rss_bytes" and amount is None) or (type(amount) is int and amount >= 0),
                     "invalid generation memory observation")
    elapsed = summary["coincidence_coverage_seconds"]
    _require(type(elapsed) in (int, float) and np.isfinite(elapsed) and elapsed >= 0, "invalid generation timing observation")
    hashes = summary["hashes"]
    _require(type(hashes) is dict and set(hashes) == HASH_KEYS, "external source pin schema changed")
    for digest in hashes.values():
        S.I._digest(digest)
    _require(type(summary["inputs"]) is dict and bool(summary["inputs"]), "missing source bindings")
    _equal(summary["seeds"], list(S.SEEDS), "seed domain changed")
    _equal(summary["arms"], list(S.ARMS), "arm domain changed")
    _require(summary["ignore_channels"] is True and summary["ignored_pulse_policy"] == IGNORED_POLICY,
             "ignored pulse policy changed")
    _equal(summary["finder_config"], asdict(M.C.finder_config()), "finder configuration changed")
    _equal(summary["coincidence_limits"], asdict(P.Limits()), "coincidence limits changed")
    _require(type(summary["output_cap_bytes"]) is int and summary["output_cap_bytes"] == CAP, "output cap changed")
    code = {str(path): S.C.sha256(path) for path in M.code_paths()}
    _equal(summary["code_sha256"], code, "mandatory generation code bindings changed")
    names = ["summary.json", "registry.parquet", "selected.parquet", "grid.parquet", "pulses.parquet",
             "clusters.parquet", "members.parquet", "coverage.parquet", *(f"code{i}.txt" for i in range(len(code)))]
    context = dict(registry_sha256=summary["registry_sha256"], batch_identity={"contract": M.SCHEMA, **hashes},
                   arms=list(S.ARMS), seeds=list(S.SEEDS))
    receipt, size = SR._batch(directory, context, names, CAP)
    _require(receipt["artifacts"]["summary.json"]["sha256"] == expected_summary_sha256, "summary changed during admission")
    for i, digest in enumerate(code.values()):
        M.bind(directory / f"code{i}.txt", digest)
    bindings = M.transaction_bindings(directory, receipt)
    pilot_summary = _bound_path(summary["inputs"], hashes["expected_pilot_sha256"])
    _require(pilot_summary.name == "summary.json" and pilot_summary.parent.name == "run", "invalid pilot summary path")
    pilot = pilot_summary.parent.parent
    verified = _bound_path(summary["inputs"], hashes["expected_verified_sha256"])
    paired = _bound_path(summary["inputs"], hashes["expected_paired_sha256"])
    M._outside(out, [pilot, verified.parent, paired.parent])
    d = M.admit(pilot, verified, paired, **hashes)
    _equal(summary["inputs"], d["inputs"], "saved input bindings differ from admitted receipts")
    M._outside(out, [Path(d["summary"][name]) for name in ("plan", "cache_audit", "response")])
    M._outside(out, [Path(d["pair"][name]).parent for name in ("selection", "reference")])
    M._outside(out, [Path(path).parent for path in d["inputs"]])
    _equal(summary["registry_sha256"], F.digest(d["registry"]), "registry digest changed")
    _require(type(summary["unique_frames"]) is int and summary["unique_frames"] == len(d["selected"]), "unique frame count changed")
    _require(type(summary["ledger_rows"]) is int and summary["ledger_rows"] == len(d["saved_ledger"]), "ledger row count changed")
    caps = dict(registry=2**20 - 1, selected=9, grid=20000, coverage=72,
                pulses=24 * P.Limits().max_total_pulses, clusters=24 * P.Limits().max_total_pulses,
                members=24 * P.Limits().max_total_pulses)
    tables = {name: _table(directory / (name + ".parquet"), types, caps[name]) for name, types in TABLE_TYPES.items()}
    for name in ("registry", "selected"):
        _exact(tables[name], d[name], name)
    cfg, hist, edges, metadata = M.load_grid(d)
    _equal(summary["grid"], metadata, "frozen grid metadata changed")
    _exact(tables["grid"], pd.DataFrame(dict(left=edges[:-1], right=edges[1:], weight=hist)), "frozen grid")
    geom = M.geometry_only(d)
    counts, comparisons, coverage = [], {}, []
    visited = {name: 0 for name in ("pulses", "clusters", "members", "coverage")}
    for record in d["summary"]["records"]:
        reference = None
        for arm in S.ARMS:
            saved = {}
            for name in visited:
                table = tables[name]
                part = table[(table.batch == record["batch"]) & (table.optical_seed == record["seed"]) & (table.arm == arm)]
                _require(part.response_seed.eq(record["seed"]).all(), "unpaired response seed")
                visited[name] += len(part)
                saved[name] = part.drop(columns=list(ANNOTATION)).reset_index(drop=True)
            row = d["saved_ledger"][(d["saved_ledger"].batch == record["batch"]) &
                  (d["saved_ledger"].optical_seed == record["seed"]) & (d["saved_ledger"].arm == arm)]
            M.pulse_admission(row.pulses.to_numpy())
            source = M.saved_pulses(pilot / record["path"] / f"{arm}_pulses.parquet", int(row.pulses.sum()))
            products = _products(source, d["domains"][record["path"]], d["registry"], d["calibration"], geom, saved)
            _exact(products["frames"][[*F.COLUMNS, "pulses"]], row[[*F.COLUMNS, "pulses"]], "source pulse census")
            products["coverage"] = G.evaluate(products["frames"], products["clusters"], cfg, hist, edges)
            _exact(saved["coverage"], products["coverage"], "per-frame coverage")
            coverage.append(products["coverage"].assign(batch=record["batch"], optical_seed=np.uint64(record["seed"]),
                                                       response_seed=np.uint64(record["seed"]), arm=arm))
            counts.append(dict(batch=record["batch"], seed=record["seed"], arm=arm, frames=len(products["frames"]),
                               pulses=len(products["pulses"]),
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
                comparisons[record["path"] + "_" + arm] = comparison
    _require(all(visited[name] == len(tables[name]) for name in visited), "undeclared record/arm table rows")
    ledger = pd.concat(coverage, ignore_index=True)
    M.validate_ledger(ledger, d)
    _exact(tables["coverage"], ledger, "complete coverage ledger")
    _equal(summary["counts"], counts, "saved product counts changed")
    _equal(summary["comparisons"], comparisons, "saved cross-arm comparisons changed")
    M.bind(__file__, verifier_sha)
    M.recheck(d)
    M.check_bindings(S.C.merge_bindings(bindings, d["inputs"], code))
    final, final_size = SR._batch(directory, context, names, CAP)
    _equal(final, receipt, "membership transaction changed late")
    _require(final_size == size, "membership byte accounting changed late")
    result = dict(schema=SCHEMA, scope=SCOPE, summary_sha256=expected_summary_sha256,
                  code_sha256=verifier_sha, hashes=hashes, registry_sha256=summary["registry_sha256"],
                  registry_frames=len(d["registry"]), unique_frames=len(d["selected"]), ledger_rows=len(ledger),
                  seeds=list(S.SEEDS), arms=list(S.ARMS), counts=counts, comparisons=comparisons, grid=metadata,
                  persisted_path_bytes=size, output_cap_bytes=CAP, saved_artifact_protocol_exact=True,
                  source_transactions_exact=True, qualified_frame_census_exact=True, canonical_pulses_exact=True,
                  native_member_structure_exact=True, frozen_grid_exact=True, per_frame_coverage_exact=True,
                  regenerated_finder=False, cluster_scalars_independently_recomputed=False,
                  ignore_channels=True, ignored_pulse_policy=IGNORED_POLICY,
                  generation_measurements=dict(memory_before=summary["memory_before"], memory_after=summary["memory_after"],
                                               coincidence_coverage_seconds=elapsed,
                                               interpretation="Hash-bound observations, not independent timing, isolated memory peaks or RSS limits."),
                  **{name: False for name in FALSE_FIELDS}, **{name: None for name in NULL_FIELDS})
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    _require(len(payload.encode()) <= CAP, "report byte cap exceeded")
    with out.open("x") as stream:
        stream.write(payload)
    return result


def compact(result):
    """Small numeric view of verified saved evidence, with explicit nonempty-domain limitations."""
    totals = {arm: {name: sum(row[name] for row in result["counts"] if row["arm"] == arm)
                    for name in ("pulses", "ignored_channel_pulses", "finder_pulses", "coincidences", "members", "gate_fail", "nonzero_coverage")}
              for arm in result["arms"]}
    return {k: result[k] for k in ("schema", "summary_sha256", "unique_frames", "ledger_rows", "grid", "persisted_path_bytes",
                                   "generation_measurements", "scope", "F")} | {
        "counts": result["counts"], "totals_per_arm": totals,
        "historical_contrasts": len(result["comparisons"]),
        "exact_contrasts": sum(c["exact"] for c in result["comparisons"].values()),
        "coverage_exact_contrasts": sum(c["coverage_exact"] for c in result["comparisons"].values()),
        "nonempty_real_membership_exercised": any(row["members"] > 0 for row in result["counts"]),
        "nonzero_real_coverage_exercised": any(row["nonzero_coverage"] > 0 for row in result["counts"]),
        "execution_admitted": False, "timing_closure": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--compact", action="store_true")
    args = vars(parser.parse_args(argv))
    small = args.pop("compact")
    result = verify(**args)
    print(json.dumps(compact(result) if small else result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
