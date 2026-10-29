"""Bounded saved native-control and nonzero-grid audit; never execute a finder."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot

from . import pilot_membership_report as V
from . import replay_recover as R
from . import replay_check as RC

M, G, F, A, SR = V.M, V.G, V.F, V.A, V.SR
require = SR._require
SCHEMA = "nonempty-saved-native-control-v1"
CAP = 4 * 1024**2
READ_CAP = 128 * 1024**2
IDS = list(R.IDS)
KEYS = [*F.FRAME_KEYS, "frame_id"]
SCOPE = ("Selected legacy saved-native membership and historical-grid control only. "
         "No finder, response, optical, fit or stock regeneration. Missing waveform/start-TDC/original-relative-time "
         "provenance blocks pilot_coincidence admission. Not timing closure, representative exposure, F or promotion.")


def parquet(path, rows):
    require(SR._regular(Path(path)) <= READ_CAP, "Parquet file cap exceeded")
    with pq.ParquetFile(path) as f:
        meta = f.metadata
        require(meta.num_rows <= rows and meta.num_columns <= 64 and meta.num_row_groups <= 4096,
                "Parquet row/schema cap exceeded before decode")
        sizes = [meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)]
        require(all(0 <= n <= READ_CAP for n in sizes) and sum(sizes) <= READ_CAP,
                "Parquet uncompressed cap exceeded before decode")
    return ak.from_parquet(path)


def tree(file, name, rows, columns=None):
    t = file[name]
    require(0 <= t.num_entries <= rows and 0 <= t.uncompressed_bytes <= READ_CAP,
            "ROOT tree cap exceeded before decode: " + name)
    return t.arrays(columns, library="ak"), {"key": t.object_path, "entries": t.num_entries,
                                            "uncompressed_bytes": t.uncompressed_bytes}


def flat(a, columns):
    out = pd.DataFrame({n: ak.to_numpy(a[n], allow_missing=False) for n in columns})
    for n in columns:
        require(out[n].dtype.kind in "iufb" and np.isfinite(out[n]).all(), "invalid numeric field: " + n)
    return out


def integers(table, columns):
    for n in columns:
        require(table[n].dtype.kind in "iu" and (table[n] >= 0).all(), "invalid integer: " + n)


def qualify(table, frames):
    require(not any(n in table for n in ("frame_id", "input_sha256")), "already qualified table")
    out = table.merge(frames[KEYS], on=IDS, how="left", validate="many_to_one", indicator=True)
    require(out._merge.eq("both").all(), "row outside source-qualified census")
    return out.drop(columns="_merge")


def pulse_bridge(pulse, stock, adapter, events, frames, status):
    """Prove original native order and explicitly serialized PE, without inventing waveform keys."""
    require(set(pulse.fields) == {"ev", "chan", "time", "pe", "seq"}, "legacy pulse schema changed")
    for name in ("time", "pe"):
        require(ak.to_numpy(pulse[name], allow_missing=False).dtype == np.dtype("float64"),
                "legacy pulse floating storage changed: " + name)
    batch = R.prepare_pulses(pulse, events)
    p = pd.DataFrame({n: batch.pulses[n] for n in (*IDS, "idx", "bar", "sipm", "seq", "time", "pe")})
    require(len(p) <= 100_000 and (p.pe >= 0).all(), "pulse bridge domain exceeded")
    rounded = p.pe.to_numpy().astype(np.float32).astype(float)
    require(np.isfinite(rounded).all(), "nonfinite serialized PE")
    for other in (stock, adapter):
        require(len(other) == len(p), "pulse collection count changed")
        for name in p:
            expected = rounded if name == "pe" else p[name].to_numpy()
            actual = ak.to_numpy(other[name], allow_missing=False)
            require(np.array_equal(actual, expected), "ordered pulse bridge changed: " + name)
            if name in ("time", "pe"):
                require(actual.dtype == np.dtype("float32" if name == "pe" else "float64"),
                        "native pulse storage dtype changed: " + name)
                target = expected.astype(actual.dtype)
                require(actual.tobytes() == target.tobytes(), "native pulse floating bits changed: " + name)
            else:
                require(actual.dtype.kind in "iu", "native pulse integer storage changed: " + name)
        require(np.all(ak.to_numpy(ak.num(other.wfidx)) == 0), "legacy waveform-placeholder contract changed")
    p["chan"] = p.bar * 4 + p.sipm
    p["stock_pe"] = rounded
    p["channel_status"] = status[p.chan.to_numpy()]
    p = qualify(p, frames)
    counts = p.groupby("frame_id").size()
    require(np.array_equal(frames.frame_id.map(counts).fillna(0), ak.to_numpy(events.npulse)),
            "native pulse/event count mismatch")
    return p


def cluster_tables(array, frames, pulses, geom):
    fields = [n for n in R.CLUSTER_TYPES if n != "nLayers"]
    c = flat(array, fields)
    integers(c, [*IDS, "idx", "sectortype", "sidehits0", "sidehits1"])
    require((c.poscalc.dtype == bool or c.poscalc.dtype == np.dtype("int32"))
            and c.poscalc.isin([0, 1]).all(), "invalid position-method flag")
    c["poscalc"] = c.poscalc.astype(bool)
    c["nLayers"] = ak.to_numpy(ak.num(array.layers)).astype(np.int64)
    if "nLayers" in array.fields:
        require(np.array_equal(c.nLayers, ak.to_numpy(array.nLayers)), "saved layer count changed")
    c = qualify(c, frames)
    require(not c.duplicated(["frame_id", "idx"]).any(), "duplicate cluster identity")
    ordered = c.sort_values(["frame_id", "idx"])
    require(np.array_equal(ordered.idx, ordered.groupby("frame_id").cumcount()), "noncontiguous cluster indices")
    counts = ak.to_numpy(ak.num(array.pulses))
    require(np.all(counts > 0) and counts.sum() <= 100_000, "empty or oversized native membership")
    target = ak.to_numpy(ak.flatten(array.pulses), allow_missing=False)
    require(target.dtype.kind in "iu" and np.all(target >= 0), "invalid native pulse target")
    m = c[KEYS + ["idx"]].iloc[np.repeat(np.arange(len(c)), counts)].reset_index(drop=True)
    m = m.rename(columns={"idx": "cluster_idx"})
    m["pulse_idx"] = target.astype(np.int64)
    m["member_order"] = np.arange(len(m)) - np.repeat(np.r_[0, counts.cumsum()][:-1], counts)
    require(not m.duplicated(["frame_id", "pulse_idx"]).any(), "duplicate native pulse membership")
    m = m.merge(pulses[["frame_id", "idx", "chan", "channel_status"]].rename(columns={"idx": "pulse_idx"}),
                on=["frame_id", "pulse_idx"], how="left", validate="many_to_one", indicator=True)
    require(m._merge.eq("both").all(), "native member outside pulse collection")
    m = m.drop(columns="_merge")
    integers(m, ["channel_status"])
    require((m.channel_status < 2**16).all(), "invalid native member channel status")
    bar = m.chan.to_numpy(dtype=np.int64) // 4
    m["layer"] = geom.layer[bar]
    m["sector_type"] = geom.sector_type[geom.sector[bar]]
    types = m.merge(c[["frame_id", "idx", "sectortype"]].rename(columns={"idx": "cluster_idx"}),
                    on=["frame_id", "cluster_idx"], validate="many_to_one")
    require(np.array_equal(types.sector_type, types.sectortype), "native member sector mismatch")
    nl = ak.to_numpy(ak.num(array.layers))
    declared = c[["frame_id", "idx"]].iloc[np.repeat(np.arange(len(c)), nl)].rename(columns={"idx": "cluster_idx"})
    declared = declared.reset_index(drop=True)
    declared["layer"] = ak.to_numpy(ak.flatten(array.layers), allow_missing=False)
    integers(declared, ["layer"])
    require((declared.layer < 4).all(), "declared layer outside geometry domain")
    keys = ["frame_id", "cluster_idx", "layer"]
    actual = m[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    declared = declared.sort_values(keys).reset_index(drop=True)
    require(actual.equals(declared.astype(actual.dtypes.to_dict())), "native layer set mismatch")
    require((c.start <= c.end).all() and (c.pe >= 0).all(), "invalid native scalar domain")
    return c, m


def compare(reference, candidate):
    """Pair by exact native member sets, not cluster order or closest time."""
    def signatures(pair):
        c, m = pair
        tokens = m.sort_values(["frame_id", "cluster_idx", "pulse_idx"]).copy()
        tokens["token"] = tokens.pulse_idx.astype(str) + ","
        s = tokens.groupby(["frame_id", "cluster_idx"]).token.sum().rename("signature").reset_index()
        out = c.merge(s, left_on=["frame_id", "idx"], right_on=["frame_id", "cluster_idx"], validate="one_to_one")
        require(not out.duplicated(["frame_id", "sectortype", "signature"]).any(), "ambiguous exact member sets")
        return out
    a, b = map(signatures, (reference, candidate))
    pair = a.merge(b, on=["frame_id", "sectortype", "signature"], how="outer", suffixes=("_stock", "_fast"), indicator=True)
    require(pair._merge.eq("both").all(), "saved native member sets differ")
    properties = {}
    for name in R.CLUSTER_TYPES:
        if name in (*IDS, "idx", "sectortype"):
            continue
        x, y = pair[name + "_stock"].to_numpy(), pair[name + "_fast"].to_numpy()
        delta = y.astype(float) - x.astype(float)
        properties[name] = {"exact_mismatches": int(np.count_nonzero(x != y)),
                            "max_abs_difference": float(np.max(np.abs(delta))) if len(delta) else 0.0}
    return {"exact_member_set_matches": len(pair), "stock_links": len(reference[1]),
            "fast_links": len(candidate[1]), "properties": properties}


def choose(frames, clusters, coverage, control):
    require(len(clusters) > 0, "nonempty saved domain required")
    nonempty = np.sort(clusters.frame_id.unique())
    affected = pd.DataFrame(control.get("affected_clusters", []))
    require(set(IDS + ["added_count"]) <= set(affected), "missing attachment evidence")
    added = qualify(affected.loc[affected.added_count > 0, IDS], frames).frame_id.to_numpy()
    positive = coverage.loc[coverage.gate_pass & (coverage.n_veto_eligible > 0) &
                            (coverage.track_grid_coverage > 0), "frame_id"].to_numpy()
    quiet = frames.loc[~frames.frame_id.isin(nonempty), "frame_id"].to_numpy()
    require(len(added) > 0 and len(positive) > 0 and len(quiet) > 0, "required control role missing")
    roles = dict(nonempty=int(nonempty.min()), opposite_side_attachment=int(added.min()),
                 positive_passing_coverage=int(positive.min()), zero_coincidence=int(quiet.min()))
    ids = sorted(set(roles.values()))
    require(set(ids) <= set(frames.frame_id) and roles["opposite_side_attachment"] in nonempty,
            "invalid attachment census")
    return frames.loc[frames.frame_id.isin(ids)].copy(), roles


def run(recovery, control, membership, verified, out, *, expected_recovery_sha256,
        expected_control_sha256, expected_membership_sha256, expected_verified_sha256):
    recovery, control, membership, verified, out = map(Path, (recovery, control, membership, verified, out))
    M._outside(out, [recovery.parent, control.parent, membership, verified.parent])
    start = time.perf_counter()
    code_paths = list(dict.fromkeys([Path(__file__), Path(V.__file__), Path(RC.__file__), *M.code_paths()]))
    code = {str(path): M.S.C.sha256(path) for path in code_paths}
    bindings = {}
    def bound(path, digest):
        path = Path(path)
        require(str(path) not in bindings or bindings[str(path)] == digest, "conflicting hash binding")
        M.bind(path, digest)
        bindings[str(path)] = digest
        return path
    rec = M.read_json(bound(recovery, expected_recovery_sha256))
    ctl = M.read_json(bound(control, expected_control_sha256))
    summary = M.read_json(bound(membership / "summary.json", expected_membership_sha256))
    prior = M.read_json(bound(verified, expected_verified_sha256))
    require(rec["schema"] == "crv-retained-fast-recovery-v1" and rec["frame_closure"]["passed"] is True,
            "unsuccessful historical recovery")
    require(ctl["schema"] == "crv-retained-pulse-reco-check-v1" and ctl["structural_and_roundtrip_validated"] is True
            and ctl["nested_membership_validated"] is True, "invalid native-control check")
    require(summary["schema"] == M.SCHEMA and set(summary) == V.SUMMARY_KEYS,
            "invalid saved membership summary schema")
    require(prior["schema"] == V.SCHEMA and prior["summary_sha256"] == expected_membership_sha256
            and prior["frozen_grid_exact"] is True and prior["saved_artifact_protocol_exact"] is True
            and prior["grid"] == summary["grid"], "invalid pinned frozen-grid verification")
    require(rec["sources"]["input"] == ctl["bindings"]["input"], "historical input identity differs")
    require(rec["finder_config"] == json.loads(json.dumps(asdict(R.finder_config()))),
            "historical finder configuration differs")
    context = dict(registry_sha256=summary["registry_sha256"], batch_identity={"contract": M.SCHEMA, **summary["hashes"]},
                   arms=list(M.S.ARMS), seeds=list(M.S.SEEDS))
    names = ["summary.json", "registry.parquet", "selected.parquet", "grid.parquet", "pulses.parquet",
             "clusters.parquet", "members.parquet", "coverage.parquet"] + [f"code{i}.txt" for i in range(len(summary["code_sha256"]))]
    receipt, _ = SR._batch(membership, context, names, M.OUTPUT_CAP)
    require(receipt["artifacts"]["summary.json"]["sha256"] == expected_membership_sha256, "late summary mutation")
    bindings.update(M.transaction_bindings(membership, receipt))
    registry = V._table(membership / "registry.parquet", V.REGISTRY, 9000)
    require(F.digest(registry) == summary["registry_sha256"] == prior["registry_sha256"], "registry identity changed")
    grid = V._table(membership / "grid.parquet", V.TABLE_TYPES["grid"], 20000)
    require(np.array_equal(grid.right.iloc[:-1], grid.left.iloc[1:]), "discontinuous saved grid")
    hist, edges = G._grid(grid.weight.to_numpy(), np.r_[grid.left, grid.right.iloc[-1]])
    pair_path = V._bound_path(summary["inputs"], summary["hashes"]["expected_paired_sha256"])
    pair = M.read_json(bound(pair_path, summary["hashes"]["expected_paired_sha256"]))
    reference = M.read_json(bound(pair["reference"], pair["reference_sha256"]))
    card = G._card(reference["card"])
    require(card.sha() == summary["grid"]["card_sha"] and len(hist) == summary["grid"]["bins"], "grid/card mismatch")
    for key in ("pulses", "counters", "stock", "check", "fast_metadata"):
        source = rec["sources"][key]
        bound(source["path"], source["sha256"])
    saved_check = M.read_json(Path(rec["sources"]["check"]["path"]))
    metadata = M.read_json(Path(rec["sources"]["fast_metadata"]["path"]))
    require(saved_check["conditions_equal"] is True, "unverified historical conditions")
    require(saved_check["input_sha256"] == rec["sources"]["input"]["sha256"]
            and saved_check["stock_sha256"] == rec["sources"]["stock"]["sha256"]
            and saved_check["fast_metadata_sha256"] == rec["sources"]["fast_metadata"]["sha256"],
            "historical check input/stock binding differs")
    require(metadata["input_sha256"] == rec["sources"]["input"]["sha256"]
            and metadata["pulse_sha256"] == rec["sources"]["pulses"]["sha256"]
            and metadata["conditions_sha256"] == saved_check["conditions_sha256"], "historical metadata differs")
    require(type(metadata["seed"]) is int and 0 <= metadata["seed"] < 2**64, "invalid retained seed")
    for key in ("reco", "table"):
        source = ctl["bindings"][key]
        bound(source["path"], source["sha256"])
    recovered_path = bound(recovery.parent / "recovered.parquet", rec["output_sha256"]["recovered.parquet"])
    counters = Path(rec["sources"]["counters"]["path"])
    require(SR._regular(counters) <= 2 * 1024**2, "counters text cap exceeded before parse")
    geom = M.geometry_only({"counters": counters})
    input_sha = rec["sources"]["input"]["sha256"]
    frames = registry.loc[registry.input_sha256.eq(input_sha)].copy()
    require(0 < len(frames) <= 3000 and len(frames) == rec["frames"], "historical registry census mismatch")
    roots = {key: Path(ctl["bindings"][key]["path"]) for key in ("reco", "table")}
    roots["fixed_stock"] = Path(rec["sources"]["stock"]["path"])
    require(all(SR._regular(path) <= READ_CAP for path in roots.values()), "ROOT file cap exceeded")
    observations = {}
    with uproot.open(roots["reco"]) as f:
        events, observations["events"] = tree(f, "crvdump/events", 3000)
        stock_p, observations["pulses"] = tree(f, "crvdump/pulses", 100_000)
        stock_c, observations["clusters"] = tree(f, "crvdump/clusters", 1000)
        calib, observations["calib"] = tree(f, "crvdump/calib", 5120)
        require(f["crvdump/digis"].num_entries == 0, "legacy adapter has unexpected digis")
    require(RC.check_events(frames, events, len(frames)) == ctl["events_check"]["on"], "saved event contract changed")
    with uproot.open(roots["table"]) as f:
        adapter, observations["adapter_pulses"] = tree(f, "crvdump/pulses", 100_000)
        require(f["crvdump/digis"].num_entries == 0, "adapter table has unexpected digis")
    with uproot.open(roots["fixed_stock"]) as f:
        fixed_calib, observations["fixed_calib"] = tree(f, "crvdump/calib", 5120)
    require(set(calib.fields) == set(fixed_calib.fields) and len(calib) == len(fixed_calib) == 5120
            and all(np.array_equal(ak.to_numpy(calib[n]), ak.to_numpy(fixed_calib[n])) for n in calib.fields),
            "control/fixed-stock calibration differs")
    calibration = flat(calib, ["channel", "status"]).sort_values("channel")
    integers(calibration, ["channel", "status"])
    require(np.array_equal(calibration.channel, np.arange(5120)) and (calibration.status < 2**16).all(),
            "invalid calibration channel domain")
    p = pulse_bridge(parquet(rec["sources"]["pulses"]["path"], 100_000), stock_p, adapter, events,
                     frames, calibration.status.to_numpy())
    stock = cluster_tables(stock_c, frames, p, geom)
    fast = cluster_tables(parquet(recovered_path, 1000), frames, p, geom)
    require(len(stock[0]) == ctl["clusters"] and len(fast[0]) == rec["coincs"], "saved coincidence counts changed")
    stock_counts = frames.frame_id.map(stock[0].groupby("frame_id").size()).fillna(0).to_numpy()
    require(np.array_equal(stock_counts, ak.to_numpy(events.ncluster)), "native cluster/event count mismatch")
    screen_frames = frames.loc[frames.frame_id.isin(stock[0].frame_id)].copy()
    screen = G.evaluate(screen_frames, stock[0], card, hist, edges)
    historical_member_status = {name: {str(bit): int(np.count_nonzero(products[1].channel_status.to_numpy() & (1 << bit)))
                                       for bit in (0, 1, 2)} for name, products in (("stock", stock), ("fast", fast))}
    selected, roles = choose(frames, stock[0], screen, ctl)
    p = p.loc[p.frame_id.isin(selected.frame_id)].copy()
    counts = selected.frame_id.map(p.groupby("frame_id").size()).fillna(0)
    require(len(selected) <= 4 and len(p) <= 2048 and (counts <= 512).all(), "selected whole-frame pulse cap exceeded")
    stock = tuple(t.loc[t.frame_id.isin(selected.frame_id)].copy() for t in stock)
    fast = tuple(t.loc[t.frame_id.isin(selected.frame_id)].copy() for t in fast)
    comparison = compare(stock, fast)
    coverages = {name: G.evaluate(selected, products[0], card, hist, edges) for name, products in (("stock", stock), ("fast", fast))}
    require(len(stock[1]) > 0 and (coverages["stock"].track_grid_coverage > 0).any(), "vacuous nonempty audit")
    tables = {"selected": selected, "pulses": p}
    for name, products in (("stock", stock), ("fast", fast)):
        tables[name + "_clusters"], tables[name + "_members"] = products
        tables[name + "_coverage"] = coverages[name]
    M._outside(out, [Path(path).parent for path in bindings])
    result = dict(schema=SCHEMA, scope=SCOPE, inputs=bindings, code_sha256=code, roles=roles,
                  registry_sha256=summary["registry_sha256"], input_sha256=input_sha,
                  registry_frames=len(registry), historical_frames=len(frames), historical_pulses=len(stock_p),
                  historical_coincidences=len(stock_c), screening_frames=len(screen),
                  selected_frames=len(selected), selected_pulses=len(p), comparison=comparison,
                  selected_identities=selected[KEYS].to_dict("records"),
                  coverage_rows={name: table[["frame_id", "n_coincidences", "n_veto_eligible", "gate_pass",
                                             "track_grid_coverage"]].to_dict("records") for name, table in coverages.items()},
                  selected_pe_float32_changed=int(np.count_nonzero(p.pe != p.stock_pe)),
                  selected_ignored_pulses=int(np.count_nonzero(p.channel_status.to_numpy() & 2)),
                  selected_status_bit_counts={str(bit): int(np.count_nonzero(p.channel_status.to_numpy() & (1 << bit))) for bit in (0, 1, 2)},
                  historical_member_status_bit_counts=historical_member_status,
                  member_status_bit_counts={name: {str(bit): int(np.count_nonzero(products[1].channel_status.to_numpy() & (1 << bit)))
                                                  for bit in (0, 1, 2)} for name, products in (("stock", stock), ("fast", fast))},
                  legacy_status_policy="Retain every table-import pulse and native link; no production ignoreChannels filtering; incompatible with pilot policy",
                  selection_scope="Deliberately nonrepresentative saved-observation selection; repeated arms are not new exposure",
                  nonempty_real_saved_membership=True, native_order_bridge_exact=True, calibration_exact=True,
                  fast_member_provenance="Previously recovered and summary-validated; not original persisted fast membership or truth",
                  stock_member_provenance="Persisted native stock collection indices",
                  control_adapter_bridge="Independent legacy table hash; ordered fields freshly checked against retained pulses after explicit PE serialization",
                  retained_seed=metadata["seed"],
                  stock_pe_serialization="float32; original float64 retained separately",
                  pilot_adapter_admitted=False, missing_pilot_identity_fields=["wf", "starttdc", "time_relative"],
                  regenerated_finder=False, regenerated_response=False, stock_rerun=False,
                  inherited_raw_admission_repeated=False, full_retained_campaign_admitted=False,
                  timing_closure=False, execution_admitted=False, F=None, finite_library_uncertainty=None,
                  coverage_exact=bool(np.array_equal(coverages["stock"].track_grid_coverage, coverages["fast"].track_grid_coverage)),
                  grid=summary["grid"], root_metadata=observations, seconds=time.perf_counter() - start,
                  memory=M.S.C.memory(), output_cap_bytes=CAP)
    out_context = dict(registry_sha256=result["registry_sha256"], batch_identity={"contract": SCHEMA,
                       "recovery_sha256": expected_recovery_sha256, "control_sha256": expected_control_sha256},
                       arms=["saved_stock", "saved_fast"], seeds=[metadata["seed"]])
    artifacts = ["summary.json", *(n + ".parquet" for n in tables), *(f"code{i}.txt" for i in range(len(code)))]
    with A.BatchArtifacts(out, max_bytes=CAP, context=out_context, expected_artifacts=artifacts) as tx:
        for name, table_value in tables.items():
            tx.write_parquet(name + ".parquet", table_value)
        for i, path in enumerate(code):
            with tx.open_artifact(f"code{i}.txt") as stream:
                stream.write(Path(path).read_bytes())
        M.check_bindings(M.S.C.merge_bindings(bindings, code))
        final, _ = SR._batch(membership, context, names, M.OUTPUT_CAP)
        require(final == receipt, "membership transaction changed late")
        tx.write_json("summary.json", result)
    _, size = SR._batch(out, out_context, artifacts, CAP)
    return result | {"persisted_path_bytes": size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recovery", "control", "membership", "verified", "out"):
        parser.add_argument("--" + name, required=True)
    for name in ("recovery", "control", "membership", "verified"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("schema", "roles", "selected_frames", "selected_pulses", "comparison",
                                          "selected_identities", "coverage_rows", "selected_pe_float32_changed", "selected_ignored_pulses",
                                          "member_status_bit_counts", "historical_member_status_bit_counts",
                                          "coverage_exact", "pilot_adapter_admitted", "seconds", "memory", "persisted_path_bytes", "scope")}, indent=2))


if __name__ == "__main__":
    main()
