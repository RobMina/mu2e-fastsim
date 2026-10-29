"""Compare saved stock/refit member sets by pulse indices, never timing proximity or finder execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import refit_saved_coverage as C

N, R, M, V, A, F, SR = C.N, C.R, C.M, C.V, C.A, C.F, C.SR
require = C.require
SCHEMA = "saved-stock-refit-comparison-v1"
CAP = 2 * 1024**2
IDS = ["run", "subrun", "event"]
KEYS = list(N.P.FRAME_KEYS)
STOCK_MEMBERS = {**dict.fromkeys(IDS, "uint32"), **dict.fromkeys(("idx", "bar", "sipm"), "int32"),
    "time": "float64", "pe": "float32", "seq": "int32", "w": "float64", "chan": "int32",
    **dict.fromkeys(("time_relative", "propagation_shift", "cluster_hit_time", "distance_to_flash_end",
                    "distance_to_digi_end_relative"), "float64")}
STOCK_CLUSTERS = {**dict.fromkeys(IDS, "uint32"), "idx": "int32",
    **dict.fromkeys(("start", "end", "avgtime"), "float64"), "pe": "float32", "sidehits0": "int32", "sidehits1": "int32"}
SCOPE = ("Three deliberately selected retained events: unordered stock-member sets versus saved stock-ADC-refit "
         "finder member sets. Match by full source/event identity and exact saved stock/refit pulse-index bridge. "
         "One cluster per event is required because legacy parent/member ordinals are absent. This measures "
         "robustness under changed pulse values, not identical-input finder equivalence or an F estimate.")
FALSE_FIELDS = {"regenerated_finder", "regenerated_fit", "regenerated_response", "stock_rerun", "raw_inputs_read",
                "selected_tracks_read", "coverage_recomputed", "prior_verifier_rerun", "source_admission_repeated",
                "stock_native_order_available", "stock_native_order_compared", "full_cluster_scalar_schema_available",
                "cluster_scalars_independently_recomputed", "same_pulse_stock_validation", "timing_closure",
                "execution_admitted", "full_retained_campaign_admitted"}


def inspect_schema(boundary, expected_boundary_sha256):
    boundary = Path(boundary)
    evidence = C.Evidence()
    report = evidence.json(boundary / "summary.json", expected_boundary_sha256)
    require(report["schema"] == "thread-d-optical-boundary-v1", "unexpected boundary schema")
    tables = {}
    for block in (0, 1):
        for kind in ("stock_members", "stock_clusters"):
            name = f"block{block}_{kind}.parquet"
            path = evidence.bound(boundary / name, report["output_sha256"][name])
            table = N.I.table(path, 2048)
            tables[name] = dict(rows=len(table), columns={name: str(dtype) for name, dtype in table.dtypes.items()},
                                identities=table[IDS].drop_duplicates().to_dict("records"))
    evidence.recheck()
    return dict(schema="stock-refit-schema-inventory-v1", tables=tables, physics_executed=False)


def locate(left, right, keys, label):
    require(not right.duplicated(keys).any(), "ambiguous " + label)
    indices = pd.MultiIndex.from_frame(right[keys]).get_indexer(pd.MultiIndex.from_frame(left[keys]))
    require(np.all(indices >= 0), "unmatched " + label)
    return right.iloc[indices].reset_index(drop=True)


def qualified(table, frames):
    require(not {"input_sha256", "frame_id"}.intersection(table), "legacy identity unexpectedly qualified")
    lookup = locate(table, frames, IDS, "legacy full event")
    return table.assign(input_sha256=lookup.input_sha256.to_numpy(), frame_id=lookup.frame_id.to_numpy())


def compare(frames, stock_members, stock_clusters, bridge, pulses, clusters, members):
    """Return unordered set contrasts and stored scalar observations without constructing cluster physics."""
    require(0 < len(frames) <= 4 and all(len(t) <= 2048 for t in (stock_members, stock_clusters, bridge, pulses, clusters, members)),
            "stock/refit comparison row cap exceeded")
    V._typed(stock_members, STOCK_MEMBERS)
    V._typed(stock_clusters, STOCK_CLUSTERS)
    V._typed(pulses, R.NATIVE_PULSES)
    V._typed(clusters, V.CLUSTERS)
    V._typed(members, V.MEMBERS)
    require(not frames.duplicated(KEYS).any() and frames.frame_id.is_unique, "duplicate frame census")
    sm, sc = qualified(stock_members, frames), qualified(stock_clusters, frames)
    require(not sm.duplicated(KEYS + ["idx"]).any(), "duplicate legacy member identity")
    require(not sc.duplicated(KEYS).any() and len(sc) == len(frames), "legacy cluster assignment is ambiguous or missing")
    require(not clusters.duplicated(KEYS).any() and len(clusters) == len(frames), "refit cluster pairing is ambiguous or missing")
    require(sc.idx.eq(0).all() and clusters.idx.eq(0).all(), "one-cluster collection index must be zero")
    sc, rc = locate(frames, sc, KEYS, "stock cluster"), locate(frames, clusters, KEYS, "refit cluster")
    require(bridge.columns.is_unique and set([*KEYS, "stock_idx", "refit_idx", "wf", "chan", "starttdc", "bar", "sipm",
        "seq", "time", "pe_stock", "pe_fast", "time_reco", "failed", "zerondf", "flags", "_merge"]) <= set(bridge),
        "missing saved bridge fields")
    for table in (bridge, pulses, members):
        locate(table, frames, KEYS, "native full frame")
    for name, table in (("pulses", pulses), ("clusters", clusters), ("members", members)):
        counts = table.groupby(KEYS).size().reindex(pd.MultiIndex.from_frame(frames[KEYS]), fill_value=0)
        require(np.array_equal(frames[name], counts), "per-frame native census differs: " + name)
    for name in (*KEYS[1:], "stock_idx", "refit_idx", "wf", "chan", "starttdc", "bar", "sipm", "seq", "flags"):
        N.P._integer(bridge[name], "bridge " + name)
    require(bridge._merge.eq("both").all(), "incomplete saved stock/refit bridge")
    require(np.array_equal(bridge.chan, bridge.bar * 4 + bridge.sipm) and (bridge.sipm < 4).all(), "bridge channel differs")
    require(len(bridge) == len(pulses) and not bridge.duplicated(KEYS + ["stock_idx"]).any()
            and not bridge.duplicated(KEYS + ["refit_idx"]).any(), "bridge is not a complete bijection")
    native = locate(bridge.rename(columns={"refit_idx": "idx"}), pulses, KEYS + ["idx"], "bridge native pulse")
    for source, target in (("wf", "wf"), ("chan", "chan"), ("starttdc", "starttdc"), ("seq", "seq")):
        require(np.array_equal(bridge[source], native[target]), "native structural bridge differs: " + source)
    for source, target in (("pe_fast", "pe"), ("time_reco", "time_reco"), ("failed", "failed"), ("zerondf", "zerondf")):
        N.B._exact(bridge[source], native[target], "native bridge " + source)
    require(not native.channel_ignored.any(), "ignored refit pulse changes comparison domain")
    require(np.array_equal((bridge["flags"].to_numpy() & 1) != 0, native.failed)
            and np.array_equal((bridge["flags"].to_numpy() & 8) != 0, native.zerondf), "bridge flags differ")
    original = locate(sm.rename(columns={"idx": "stock_idx"}), bridge, KEYS + ["stock_idx"], "stock member bridge")
    for source, target in (("time", "time"), ("pe", "pe_stock")):
        N.B._exact(sm[source], original[target], "stock member " + source)
    for name in ("bar", "sipm", "seq", "chan"):
        require(np.array_equal(sm[name], original[name]), "stock member channel/sequence differs")
    require((sm.pe >= 0).all() and (sc.pe >= 0).all() and (sc.start <= sc.end).all(), "invalid stock scalar domain")
    require(not members.duplicated(KEYS + ["pulse_idx"]).any(), "duplicate refit member")
    require(members.cluster_idx.eq(0).all(), "native link has no sole cluster")
    linked = locate(members.rename(columns={"pulse_idx": "refit_idx"}), bridge, KEYS + ["refit_idx"], "refit member bridge")
    for name in N.P.PULSE_KEYS:
        require(np.array_equal(members[name], linked[name]), "refit member structural key differs")
    require(np.array_equal(members.member_order, members.groupby(KEYS, sort=False).cumcount()), "native member ordinal differs")
    for side in (0, 1):
        counts = sm.loc[sm.chan.mod(2).eq(side) & sm.pe.gt(0)].groupby(KEYS).size()
        expected = counts.reindex(pd.MultiIndex.from_frame(sc[KEYS]), fill_value=0).to_numpy()
        require(np.array_equal(sc[f"sidehits{side}"], expected), "legacy positive-PE side census differs")
    stock_set = sm[KEYS].assign(stock_idx=sm.idx.to_numpy())
    refit_set = linked[KEYS + ["stock_idx"]]
    contrast = stock_set.merge(refit_set, on=KEYS + ["stock_idx"], how="outer", validate="one_to_one", indicator=True)
    census = contrast.groupby(KEYS + ["_merge"], observed=True).size().unstack("_merge", fill_value=0)
    census = census.reindex(pd.MultiIndex.from_frame(frames[KEYS]), fill_value=0)
    census = census.reindex(columns=["both", "left_only", "right_only"], fill_value=0)
    result = frames.copy()
    result["common_members"] = census["both"].to_numpy(dtype=np.int64)
    result["stock_only_members"] = census["left_only"].to_numpy(dtype=np.int64)
    result["refit_only_members"] = census["right_only"].to_numpy(dtype=np.int64)
    result["member_sets_exact"] = (result.stock_only_members + result.refit_only_members).eq(0)
    require(np.array_equal(result.members, result.common_members + result.refit_only_members), "refit member census differs")
    for name in ("start", "end", "avgtime", "pe", "sidehits0", "sidehits1"):
        result["stock_" + name] = sc[name].to_numpy()
        result["refit_" + name] = rc[name].to_numpy()
        result["refit_minus_stock_" + name] = rc[name].to_numpy().astype(np.float64) - sc[name].to_numpy().astype(np.float64)
    return result, contrast.loc[contrast._merge.ne("both")].to_dict("records")


def run(directory, verified, out, *, expected_summary_sha256, expected_verified_sha256, expected_complete_sha256):
    directory, verified, out = map(Path, (directory, verified, out))
    M._outside(out, [directory, verified.parent])
    begin = time.perf_counter()
    evidence = C.Evidence()
    paths = list(dict.fromkeys([Path(__file__), *C.code_paths()]))
    code = {str(p): C.file_sha(p) for p in paths}
    for path, digest in code.items():
        evidence.bound(path, digest)
    summary, prior = C.refit_admission(evidence, directory, verified, expected_summary_sha256,
                                      expected_verified_sha256, expected_complete_sha256)
    role = evidence.roles["boundary"]
    boundary = Path(role["path"]).parent
    old = evidence.json(role["path"], role["sha256"])
    require(old["schema"] == "thread-d-optical-boundary-v1" and len(old["blocks"]) == 2, "boundary schema/domain differs")
    registry = V._table(directory / "registry.parquet", V.REGISTRY, 9000)
    require(len(registry) == 9000 and F.digest(registry) == N.REGISTRY_SHA, "complete registry differs")
    all_frames, all_clusters = C.saved_products(directory, summary, prior, registry)
    exclusions = [directory, verified.parent, boundary, *[Path(p).parent for p in evidence.bindings],
                  *[Path(p).parent for p in summary["inputs"]]]
    M._outside(out, exclusions)
    rows, differences, schemas = [], [], {}
    for block in (0, 1):
        record = old["blocks"][block]
        require(record["block"] == block and record["targets"] == N.TARGETS[block], "legacy target domain differs")
        shas = [digest for path, digest in record["input_sha256"].items() if path.endswith("/mixed.root")]
        require(shas == [N.INPUTS[block]], "legacy raw identity differs")
        tables = {}
        for kind, types in (("stock_members", STOCK_MEMBERS), ("stock_clusters", STOCK_CLUSTERS)):
            name = f"block{block}_{kind}.parquet"
            path = evidence.bound(boundary / name, old["output_sha256"][name])
            tables[kind] = V._table(path, types, 2048)
            schemas[name] = dict(rows=len(tables[kind]), columns=types)
        SR._equal(tables["stock_clusters"].to_dict("records"), record["stock_clusters"], "legacy stored cluster observations differ")
        frame = all_frames.loc[all_frames.input_sha256.eq(N.INPUTS[block])].reset_index(drop=True)
        cluster = all_clusters.loc[all_clusters.input_sha256.eq(N.INPUTS[block])].reset_index(drop=True)
        bridge = N.I.table(directory / f"block{block}_saved_bridge.parquet", 2048)
        pulses = V._table(directory / f"block{block}_pulses.parquet", R.NATIVE_PULSES, 2048)
        members = V._table(directory / f"block{block}_members.parquet", V.MEMBERS, 2048)
        result, delta = compare(frame, tables["stock_members"], tables["stock_clusters"], bridge, pulses, cluster, members)
        rows.append(result)
        differences.extend(delta)
    result_rows = pd.concat(rows, ignore_index=True)
    snapshots = {}
    for path, digest in code.items():
        payload = C.bounded_bytes(path)
        require(hashlib.sha256(payload).hexdigest() == digest, "comparison source snapshot differs")
        snapshots[path] = payload.decode("utf-8")
    result = dict(schema=SCHEMA, scope=SCOPE, inputs=evidence.bindings, code_sha256=code, code_snapshots=snapshots,
        summary_sha256=expected_summary_sha256, verified_sha256=expected_verified_sha256, complete_sha256=expected_complete_sha256,
        registry_sha256=N.REGISTRY_SHA, registry_frames=9000, unique_frames=3, rows=result_rows.to_dict("records"),
        membership_differences=differences, stock_schema=schemas, all_member_sets_exact=bool(result_rows.member_sets_exact.all()),
        missing_legacy_fields=["cluster_idx on members", "member_ordinal", "nLayers", "sectorType", "sidepe", "sidetime", "position", "slope", "poscalc"],
        parent_cluster_assignment="Unique full-event cluster only; no invented native ordinal or time matching",
        scalar_scope="Stored observations only; stock PE float32 and refit PE float64 are not bit-identical inputs",
        stock_refit_differences=prior["stock_refit_differences"], inherited_native_structure_verified=True,
        saved_artifact_pins_exact=True, unordered_stock_refit_membership_compared=True, F=None, finite_library_uncertainty=None,
        output_cap_bytes=CAP, seconds=time.perf_counter() - begin, memory=M.S.C.memory(),
        publication="exclusive JSON; no atomic-completion or crash-durability claim", **dict.fromkeys(FALSE_FIELDS, False))
    payload = A._canonical(result)
    require(len(payload) <= CAP, "comparison report byte cap exceeded")
    evidence.recheck()
    M._outside(out, exclusions)
    with out.open("xb") as stream:
        stream.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    inventory = modes.add_parser("schema")
    inventory.add_argument("--boundary", required=True)
    inventory.add_argument("--expected-boundary-sha256", required=True)
    compare_parser = modes.add_parser("compare")
    compare_parser.add_argument("directory")
    for name in ("verified", "out", "expected_summary_sha256", "expected_verified_sha256", "expected_complete_sha256"):
        compare_parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = vars(parser.parse_args(argv))
    mode = args.pop("mode")
    if mode == "schema":
        result = inspect_schema(**args)
    else:
        result = run(**args)
        result = {key: result[key] for key in ("schema", "rows", "membership_differences", "all_member_sets_exact", "seconds", "memory", "scope", "F")}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
