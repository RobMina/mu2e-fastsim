"""Inventory retained pulse identity and geometry budgets without a finder or response call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import pilot_membership_report as V
from . import saved_qualified_response as Q
from . import saved_photon_downstream as D

M, P, F, A, SR = V.M, V.P, V.F, V.A, V.SR
require = SR._require
SCHEMA = "saved-pulse-identity-inventory-v1"
CAP = 2 * 1024**2
FILE_CAP = 32 * 1024**2
KEYS = [*F.FRAME_KEYS, "frame_id"]
SCOPE = ("Bounded inventory of two retained response controls, not a coincidence or coverage result. "
         "No finder, optics, response, fit, stock, raw-input admission or prior-verifier rerun. "
         "Identity-compatible saved pulses do not establish nonempty membership or stock equivalence.")


def table(path, rows):
    path = Path(path)
    require(SR._regular(path) <= FILE_CAP, "inventory Parquet file cap exceeded")
    with pq.ParquetFile(path) as f:
        meta = f.metadata
        require(meta.num_rows <= rows and meta.num_columns <= 64 and meta.num_row_groups <= 4096,
                "inventory Parquet row/schema cap exceeded before decode")
        sizes = [meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)]
        require(all(0 <= n <= 128 * 1024**2 for n in sizes) and sum(sizes) <= 128 * 1024**2,
                "inventory uncompressed cap exceeded before decode")
    return pd.read_parquet(path)


def qualify(t, selected):
    require(not any(n in t for n in ("run", "subrun", "input_sha256", "frame_id")), "already qualified legacy table")
    require(selected.event.is_unique and selected.run.nunique() == 1 and selected.subrun.nunique() == 1
            and selected.input_sha256.nunique() == 1, "ambiguous legacy event-only identity")
    out = t.merge(selected[KEYS], on="event", how="left", validate="many_to_one", indicator=True, sort=False)
    require(out._merge.eq("both").all(), "legacy row outside bound event census")
    return out.drop(columns="_merge")


def inspect(pulses, waveforms, selected, registry, calibration, geom):
    """Check stored fields/order without sorting repair or any coincidence decisions."""
    selected = M.S.R.selected_frames(registry, selected, F.digest(registry))
    require(len(selected) <= 4 and len(pulses) <= 2048 and len(waveforms) <= 2048, "inventory table cap exceeded")
    require(set(KEYS + P.PULSE_FIELDS + ["failed", "zerondf", "time"]) <= set(pulses)
            and set(KEYS + ["wf", "chan", "starttdc"]) <= set(waveforms), "missing saved identity/time/flag fields")
    require(pulses.columns.is_unique and waveforms.columns.is_unique, "duplicate saved columns")
    domain = pd.MultiIndex.from_frame(selected[KEYS])
    for t in (pulses, waveforms):
        require(pd.MultiIndex.from_frame(t[KEYS]).isin(domain).all(), "table outside full selected census")
        for n in (*KEYS[1:], "wf", "chan", "starttdc"):
            P._integer(t[n], n)
    P._integer(pulses.seq, "seq", 2**31)
    require(not pulses.duplicated(KEYS + P.PULSE_KEYS).any(), "duplicate structural pulse key")
    for n in ("time", "time_relative", "time_reco", "pe"):
        require(pulses[n].dtype == np.dtype("float64") and np.isfinite(pulses[n]).all(), "invalid saved float64 " + n)
    require((pulses.pe >= 0).all(), "negative pulse PE")
    for n in ("channel_ignored", "failed", "zerondf"):
        require(pulses[n].dtype == bool, "invalid boolean " + n)
    p = pulses.reset_index(drop=True)
    native = p.sort_values(["frame_id", "wf", "seq"]).index.to_numpy()
    structural = p.sort_values(["frame_id", *P.PULSE_KEYS]).index.to_numpy()
    require(np.array_equal(native, structural) and np.array_equal(native, np.arange(len(p))),
            "native/structural/stored order differs")
    mapping = waveforms[[*KEYS, "wf", "chan", "starttdc"]]
    require(not mapping.wf.duplicated().any() and not mapping.duplicated(KEYS + ["chan", "starttdc"]).any(),
            "non-bijective waveform identity")
    linked = p.merge(mapping, on=[*KEYS, "wf", "chan", "starttdc"], how="left", validate="many_to_one", indicator=True)
    require(linked._merge.eq("both").all(), "pulse waveform link differs")
    pb = p.frame_id.map(selected.set_index("frame_id").pbtime).to_numpy().astype(np.float32).astype(float)
    require(np.isfinite(pb).all() and np.array_equal(p.time_reco, p.time_relative.to_numpy() - pb),
            "exact forward bunch-time relation differs")
    require(p.time_reco.to_numpy().tobytes() == (p.time_relative.to_numpy() - pb).tobytes(), "reco time bits differ")
    require(set(["input_sha256", "chan", "status"]) <= set(calibration), "missing calibration fields")
    for digest in selected.input_sha256.unique():
        P._status(calibration, digest, len(geom.sector) * 4)
    cal = calibration[["input_sha256", "chan", "status"]]
    joined = p.merge(cal, on=["input_sha256", "chan"], how="left", validate="many_to_one", indicator=True)
    require(joined._merge.eq("both").all(), "pulse missing channel status")
    status = P._integer(joined.status, "status", 2**16)
    require(np.array_equal(p.channel_ignored, (status & 2) != 0) and not np.any(status & 5), "saved channel policy differs")
    total = selected.frame_id.map(p.groupby("frame_id").size()).fillna(0).astype(np.int64)
    ignored = selected.frame_id.map(p.loc[p.channel_ignored].groupby("frame_id").size()).fillna(0).astype(np.int64)
    require((total <= 512).all(), "per-frame pulse cap exceeded")
    cfg = M.C.finder_config()
    active = p.loc[~p.channel_ignored].copy()
    work = P.combination_work(active, geom, cfg)
    require(work <= P.Limits().max_combinations, "candidate budget exceeded")
    bar = P._integer(active.chan, "chan", len(geom.sector) * 4) // 4
    active["sector_type"] = geom.sector_type[geom.sector[bar]]
    active["layer"] = geom.layer[bar]
    active["side"] = active.chan.to_numpy() % 2
    occupancy = active.groupby(["frame_id", "sector_type", "side"]).agg(
        pulses=("chan", "size"), distinct_layers=("layer", "nunique")).reset_index()
    counts = selected[KEYS].copy()
    counts["saved_pulses"], counts["ignored_pulses"] = total.to_numpy(), ignored.to_numpy()
    counts["finder_input_pulses"] = counts.saved_pulses - counts.ignored_pulses
    return dict(frames=counts.to_dict("records"), occupancy=occupancy.to_dict("records"),
                candidate_tuple_upper_bound=work, saved_waveforms=len(waveforms), pulse_rows=len(p),
                saved_identity_and_order_valid=True, forward_time_bits_exact=True,
                finder_not_run=True, nonempty_membership_established=False)


def run(response, response_verified, extreme, extreme_verified, out, *, expected_response_sha256,
        expected_response_verified_sha256, expected_extreme_sha256, expected_extreme_verified_sha256):
    response, response_verified, extreme, extreme_verified, out = map(Path, (response, response_verified, extreme, extreme_verified, out))
    M._outside(out, [response, response_verified.parent, extreme, extreme_verified.parent])
    begin = time.perf_counter()
    code_paths = list(dict.fromkeys([Path(__file__), Path(V.__file__), Path(Q.__file__), Path(D.__file__), *M.code_paths()]))
    code = {str(path): M.S.C.sha256(path) for path in code_paths}
    bindings, transactions = {}, []
    def bound(path, sha):
        path = Path(path)
        require(SR._regular(path) <= FILE_CAP, "inventory bound file cap exceeded")
        require(str(path) not in bindings or bindings[str(path)] == sha, "conflicting inventory binding")
        M.bind(path, sha)
        bindings[str(path)] = sha
        return path
    def batch(root, context, names, cap):
        receipt, size = SR._batch(root, context, names, cap)
        bindings.update(M.transaction_bindings(root, receipt))
        transactions.append((root, context, names, cap, receipt))
        return receipt
    saved = M.read_json(bound(response / "run/summary.json", expected_response_sha256))
    verified = M.read_json(bound(response_verified, expected_response_verified_sha256))
    old = M.read_json(bound(extreme / "summary.json", expected_extreme_sha256))
    old_verified = M.read_json(bound(extreme_verified, expected_extreme_verified_sha256))
    require(saved["schema"] == Q.SCHEMA and saved["frames"] == 4 and saved["registry_frames"] == 9000,
            "unexpected all-live response domain")
    require(verified["schema"] == SR.SCHEMA and verified["summary_sha256"] == expected_response_sha256
            and verified["authoritative_all_live_conditions_exact"] is True
            and verified["all_tested_physical_partition_fields_exact"] is True, "invalid all-live verification")
    require(old["schema"] == "crv-saved-photon-downstream-v1" and old_verified["schema"] == "crv-saved-downstream-report-v1"
            and old_verified["input_summary_sha256"] == expected_extreme_sha256
            and old_verified["regenerated_response"] is False, "invalid extreme verification")
    require(saved["arms"] == list(Q.M.P.ARMS) and saved["seeds"] == list(Q.SEEDS), "changed saved response schedule")
    require(old["F"] is None and old["timing_closure"] is False and saved["F"] is None
            and saved["timing_closure"] is False, "unexpected generation scope")
    root = response / "run"
    context = M.read_json(root / A.COMPLETE)["context"]
    expected_context = dict(registry_sha256=saved["registry_sha256"], batch_identity={
        "plan_sha256": context["batch_identity"]["plan_sha256"], "smoke_sha256": context["batch_identity"]["smoke_sha256"],
        "contract": Q.R.CONTRACT}, arms=list(Q.M.P.ARMS), seeds=list(Q.SEEDS))
    require(context == expected_context, "changed response context")
    tags = [f"seed{seed}_{arm}" for seed in Q.SEEDS for arm in Q.M.P.ARMS]
    require(set(p.name for p in response.iterdir()) == {"run", *tags}, "changed response inventory")
    names = ["summary.json", "registry.parquet", "selected.parquet", "calibration.parquet", "admission.json"]
    names += [tag + ".json" for tag in tags] + [f"code{i}.txt" for i in range(len(saved["code_sha256"]))]
    top = batch(root, context, names, Q.ROOT_CAP)
    require(top["artifacts"]["summary.json"]["sha256"] == expected_response_sha256, "late response summary change")
    for i, (path, sha) in enumerate(saved["code_sha256"].items()):
        bound(path, sha)
    snapshot_hashes = {top["artifacts"][f"code{i}.txt"]["sha256"] for i in range(len(saved["code_sha256"]))}
    require(snapshot_hashes == set(saved["code_sha256"].values()), "response code snapshot mismatch")
    registry, selected, calibration = (table(root / f"{name}.parquet", cap) for name, cap in
                                        (("registry", 9000), ("selected", 4), ("calibration", 15360)))
    F.validate(registry)
    require(F.digest(registry) == saved["registry_sha256"] == verified["registry_sha256"], "frozen registry differs")
    counters = [path for path, sha in saved["inputs"].items() if Path(path).name == "crv_counters_run1a_v01.txt"]
    require(len(counters) == 1, "missing unique Run1A geometry binding")
    path = bound(counters[0], saved["inputs"][counters[0]])
    require(SR._regular(path) <= 2 * 1024**2, "geometry text cap exceeded")
    geom = M.geometry_only({"counters": path})
    bindings.update(M.ignore_bindings())
    records = []
    require(len(saved["records"]) == len(tags) and {r["path"] for r in saved["records"]} == set(tags), "record schedule differs")
    for record in saved["records"]:
        seed, arm, tag = record["seed"], record["arm"], record["path"]
        require(tag == f"seed{seed}_{arm}", "record identity differs")
        child_context = {**context, "batch_identity": {**context["batch_identity"], "frame_ids": selected.frame_id.tolist(),
                         "variants": list(Q.VARIANTS)}, "arms": [arm], "seeds": [seed]}
        child_names = [f"{v}_{s}.parquet" for v in Q.VARIANTS for s in Q.R.STAGES] + ["validation.json"]
        child = batch(response / tag, child_context, child_names, Q.BATCH_CAP)
        require(child == M.read_json(root / (tag + ".json")) and M.S.C.sha256(response / tag / A.COMPLETE) == record["receipt_sha256"],
                "copied response receipt differs")
        frame_counts = table(response / tag / "full_frames.parquet", 4)
        M.pulse_admission(frame_counts.pulses.to_numpy())
        p = table(response / tag / "full_pulses.parquet", 2048)
        require(len(p) == record["counts"]["pulses"] == int(frame_counts.pulses.sum()), "saved pulse count differs")
        M.S.R.selected_frames(registry, frame_counts[list(F.COLUMNS)], F.digest(registry))
        require(np.array_equal(frame_counts.frame_id, selected.frame_id)
                and np.array_equal(frame_counts.pulses, frame_counts.frame_id.map(p.groupby("frame_id").size()).fillna(0)),
                "per-frame saved pulse census differs")
        w = table(response / tag / "full_waveforms.parquet", 2048)
        result = inspect(p, w, selected, registry, calibration, geom)
        records.append(dict(sample="all_live", tag=tag, seed=seed, arm=arm, thermal="all status-live channels", **result))
    require(0 < len(old["blocks"]) <= 3 and sum(len(b["events"]) for b in old["blocks"]) <= 4,
            "invalid extreme block domain")
    required_old = {"optical_verified.json"}
    for i in range(len(old["blocks"])):
        required_old.add(f"input{i}_conditions.parquet")
        required_old.update(f"input{i}_thermal{t}_{arm}_{stage}.parquet" for t in (0, 1)
                            for arm in Q.M.P.ARMS for stage in ("pulses", "waveforms"))
    require(len(old["output_sha256"]) <= 512, "extreme output inventory cap exceeded")
    require(required_old <= set(old["output_sha256"]), "missing required extreme artifact binding")
    expected_names = {"summary.json", *old["output_sha256"]}
    require(set(p.name for p in extreme.iterdir()) == expected_names, "changed extreme inventory")
    for name, sha in old["output_sha256"].items():
        require(Path(name).name == name, "unsafe extreme artifact name")
        bound(extreme / name, sha)
    for path, sha in old["code_sha256"].items():
        bound(path, sha)
    optical_path = bound(old["input_summary"], old["input_summary_sha256"])
    optical = M.read_json(optical_path)
    require(len(optical["blocks"]) == len(old["blocks"]) == len(old_verified["blocks"]), "extreme optical/report block count differs")
    for i, block in enumerate(old["blocks"]):
        ob = optical["blocks"][i]
        require(ob["input_sha256"] == block["input_sha256"] == old_verified["blocks"][i]["input_sha256"], "extreme input identity differs")
        ev = pd.DataFrame(block["events"])
        require(not ev.duplicated(list(M.C.IDS)).any() and 0 < len(ev) <= 4, "invalid extreme event census")
        wanted = ev.assign(input_sha256=block["input_sha256"])
        loc = pd.MultiIndex.from_frame(registry[list(F.FRAME_KEYS)]).get_indexer(pd.MultiIndex.from_frame(wanted[list(F.FRAME_KEYS)]))
        require(np.all(loc >= 0), "extreme frame outside frozen registry")
        domain = registry.iloc[np.sort(loc)].copy()
        er = ob["artifacts"]["events"]
        require(Path(er["path"]).name == er["path"], "unsafe extreme event artifact")
        event_table = table(bound(optical_path.parent / er["path"], er["sha256"]), 4)
        require(len(event_table) == len(domain) and np.array_equal(event_table[list(M.C.IDS)], domain[list(M.C.IDS)]),
                "extreme optical event census differs")
        require(np.array_equal(event_table.pbtime, domain.pbtime), "extreme bunch time differs")
        cond = table(extreme / f"input{i}_conditions.parquet", 20480)
        c = qualify(cond, domain)
        require(not c.duplicated(["frame_id", "chan"]).any(), "duplicate extreme conditions")
        values = ["pedestal", "pulsearea", "pulseheight", "timeoffset", "status", "roc", "feb"]
        joined = c.merge(calibration[["input_sha256", "chan", *values]], on=["input_sha256", "chan"],
                         how="left", validate="many_to_one", suffixes=("", "_full"), indicator=True)
        require(joined._merge.eq("both").all() and all(np.array_equal(joined[n], joined[n + "_full"]) for n in values),
                "extreme optical-union calibration differs")
        require(set(block["models"]) == {f"input{i}_thermal{n}" for n in (0, 1)}, "extreme thermal schedule differs")
        for prefix, model in block["models"].items():
            require(set(model["arms"]) == set(Q.M.P.ARMS), "extreme arm schedule differs")
            for arm in Q.M.P.ARMS:
                tag = f"{prefix}_{arm}"
                p = table(extreme / (tag + "_pulses.parquet"), 2048)
                w = table(extreme / (tag + "_waveforms.parquet"), 2048)
                require(len(p) == model["arms"][arm]["pulses"], "extreme pulse count differs")
                result = inspect(qualify(p, domain), qualify(w, domain), domain, registry, calibration, geom)
                records.append(dict(sample="extreme_optical_union", tag=tag, seed=block["seed"], arm=arm,
                                    thermal="optical union " + prefix.rsplit("_", 1)[-1], **result))
    M._outside(out, [Path(path).parent for path in bindings])
    result = dict(schema=SCHEMA, scope=SCOPE, inputs=bindings, code_sha256=code, records=records,
                  registry_sha256=F.digest(registry), output_cap_bytes=CAP, runtime_seconds=time.perf_counter() - begin,
                  memory=M.S.C.memory(), regenerated_finder=False, regenerated_response=False, stock_rerun=False,
                  inherited_raw_admission_repeated=False, nonempty_membership_established=False,
                  full_retained_campaign_admitted=False, execution_admitted=False, timing_closure=False,
                  F=None, finite_library_uncertainty=None)
    out_context = dict(registry_sha256=result["registry_sha256"], batch_identity={"contract": SCHEMA,
                       "response_sha256": expected_response_sha256, "extreme_sha256": expected_extreme_sha256},
                       arms=list(Q.M.P.ARMS), seeds=sorted({r["seed"] for r in records}))
    artifacts = ["summary.json", *(f"code{i}.txt" for i in range(len(code)))]
    with A.BatchArtifacts(out, max_bytes=CAP, context=out_context, expected_artifacts=artifacts) as tx:
        for i, path in enumerate(code):
            with tx.open_artifact(f"code{i}.txt") as stream:
                stream.write(Path(path).read_bytes())
        M.check_bindings(M.S.C.merge_bindings(bindings, code))
        require(set(p.name for p in response.iterdir()) == {"run", *tags}, "late response inventory change")
        require(set(p.name for p in extreme.iterdir()) == expected_names, "late extreme inventory change")
        for root, ctx, names, cap, original in transactions:
            current, _ = SR._batch(root, ctx, names, cap)
            require(current == original, "late response transaction change")
        tx.write_json("summary.json", result)
    _, size = SR._batch(out, out_context, artifacts, CAP)
    return result | {"persisted_path_bytes": size}


def compact(result):
    return {k: result[k] for k in ("schema", "registry_sha256", "runtime_seconds", "memory", "persisted_path_bytes", "scope")} | {
        "records": [{k: r[k] for k in ("sample", "tag", "pulse_rows", "candidate_tuple_upper_bound", "frames", "occupancy")}
                    for r in result["records"]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("response", "response_verified", "extreme", "extreme_verified", "out"):
        parser.add_argument("--" + n.replace("_", "-"), required=True)
    for n in ("response", "response_verified", "extreme", "extreme_verified"):
        parser.add_argument("--expected-" + n.replace("_", "-") + "-sha256", required=True)
    print(json.dumps(compact(run(**vars(parser.parse_args(argv)))), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
