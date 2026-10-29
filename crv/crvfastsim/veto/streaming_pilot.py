"""Bounded new-frame cache/optics/response pilot; never a full retained physics campaign."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import pandas as pd
import scipy
import uproot

from . import retained_cache_audit as C, retained_cache_report as CR
from . import saved_qualified_response as V, saved_qualified_report as SR
from . import streaming_optics as O

F, I, A, M, R = C.F, C.I, C.A, C.M, V.R
SCHEMA = "retained-streaming-pilot-v1"
SEEDS, ARMS = V.SEEDS, M.P.ARMS
TOP_CAP = 8 * 1024**2
CHILD_CAP = 64 * 1024**2
MAX_BATCHES, MAX_FRAMES = 3, 9
SELECTION = "lowest complete planned batch per input excluding saved smoke and prior extreme-control frames"
SCOPE = ("Three new whole planned batches at most, not the full retained sample. Paired optical/downstream "
         "seeds on unchanged frames; not independent exposure or a factorial variance design. No stock "
         "rerun, old smoke redraw, coincidence, frozen-track coverage, F or finite-library uncertainty.")
SOURCE_TABLES = ("events", "steps", "particles")
CHILD_NAMES = [n + ".parquet" for n in (*SOURCE_TABLES, "photons")] + ["validation.json"]
CHILD_NAMES += [f"{arm}_{stage}.parquet" for arm in ARMS for stage in R.STAGES]


def code_paths():
    paths = [Path(__file__), Path(O.__file__), Path(C.__file__), Path(CR.__file__), Path(SR.__file__)]
    return list(dict.fromkeys([*paths, *C.code_paths(), *V.code_paths(), *M.code_paths()]))


def schedule(registry, frames, batches, excluded):
    """Choose whole existing batches, never split frames or silently redraw completed controls."""
    F.validate(registry)
    forbidden = set(frames.loc[frames.frame_id.isin(excluded.frame_id) | frames.event.isin(M.EXCLUDE_EVENTS), "batch"])
    allowed = batches[~batches.batch.isin(forbidden)].sort_values("batch")
    chosen = allowed.groupby("input_sha256", sort=False).head(1).sort_values("batch").reset_index(drop=True)
    if (not len(chosen) or len(chosen) > MAX_BATCHES or set(chosen.input_sha256) != set(registry.input_sha256)
            or chosen.input_sha256.duplicated().any()):
        raise ValueError("no complete bounded fresh batch for every input")
    assigned = frames[frames.batch.isin(chosen.batch)].sort_values("frame_id").reset_index(drop=True)
    selected = registry.iloc[assigned.frame_id].reset_index(drop=True)
    if len(selected) > MAX_FRAMES or selected.frame_id.isin(excluded.frame_id).any():
        raise ValueError("pilot frame bound or completed-control exclusion failed")
    if np.any(chosen.frames > R.Limits().frames) or int(chosen.frames.sum()) != len(selected):
        raise ValueError("whole planned batch frame accounting disagreement")
    return selected, assigned, chosen


def _directory_bindings(root):
    """Bind the already verified fixed-layout evidence directory without following symlinks."""
    result = {}
    for path in Path(root).iterdir():
        if path.is_symlink():
            raise ValueError("symlink in verified evidence")
        if path.is_dir():
            for child in path.iterdir():
                SR._regular(child)
                result[str(child)] = C.sha256(child)
        else:
            SR._regular(path)
            result[str(path)] = C.sha256(path)
    return result


def outside_evidence(out, roots):
    target = Path(out).resolve()
    if Path(out).is_symlink() or any(target.is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError("output must be outside prior evidence directories")


def guard_raw_admission(sources):
    """Bound raw headers before the inherited saved-response verifier's legacy payload reads."""
    limits = I.Limits()
    for source in sources:
        path = M.P.bound(source["path"], source["input_sha256"])
        sidecar = M.P.bound(path.with_suffix(".json"), source["manifest_sha256"])
        manifest = SR._json(sidecar)
        I._validate_manifest(manifest)
        I.require_legacy_response(manifest)
        with uproot.open(path, array_cache=None, object_cache=None, minimal_ttree_metadata=True) as root:
            trees = {name: root["dump/" + name] for name in SOURCE_TABLES}
            counts = {name: int(tree.num_entries) for name, tree in trees.items()}
            I._admit(counts, limits)
            if counts != manifest["entries"] or manifest["output_sha256"] != source["input_sha256"] or manifest["raw_times"] is not True:
                raise ValueError("legacy admission source contract mismatch")
            for name, tree in trees.items():
                I._check_tree(tree, name, limits)


def excluded_controls(plan, bindings, registry, smoke_frames):
    """Reuse the original all-frame roundoff census to exclude every completed extreme control."""
    report = SR._json(Path(plan) / "summary.json")
    smoke_path = str(Path(report["smoke"]) / "summary.json")
    smoke = SR._json(M.P.bound(smoke_path, bindings[smoke_path]))
    census_path = str(Path(smoke["paths"]["audit"]) / "summary.json")
    census = SR._json(M.P.bound(census_path, bindings[census_path]))
    domains = [smoke_frames]
    for source in census["inputs"]:
        artifact = source["artifacts"]["frames"]
        path = str(Path(census_path).parent / artifact["path"])
        if bindings.get(path) != artifact["sha256"]:
            raise ValueError("missing original extreme-frame census binding")
        C.parquet_admit(M.P.bound(path, artifact["sha256"]), artifact["rows"])
        table = pd.read_parquet(path, columns=[*C.IDS, "fold_start_roundoff"])
        if table.fold_start_roundoff.dtype.kind not in "iu" or table.fold_start_roundoff.lt(0).any():
            raise ValueError("invalid extreme-frame census")
        keys = table.loc[table.fold_start_roundoff > 0, list(C.IDS)].assign(input_sha256=source["sha256"])
        where = pd.MultiIndex.from_frame(registry[list(F.FRAME_KEYS)]).get_indexer(pd.MultiIndex.from_frame(keys[list(F.FRAME_KEYS)]))
        if np.any(where < 0):
            raise ValueError("extreme frame outside full registry")
        domains.append(registry.iloc[where])
    return pd.concat(domains, ignore_index=True).drop_duplicates("frame_id").sort_values("frame_id").reset_index(drop=True)


def admission(plan, cache_audit, response, *, expected_plan_sha256, expected_cache_sha256, expected_response_sha256):
    """Reuse existing complete cache/downstream evidence; never run the completed controls again."""
    plan, cache_audit, response = map(Path, (plan, cache_audit, response))
    C.M.P.bound(cache_audit / "summary.json", expected_cache_sha256)
    C.M.P.bound(response / "run/summary.json", expected_response_sha256)
    registry, frames, batches, sources, bindings, pv = C.admission(plan, expected_plan_sha256)
    guard_raw_admission(sources)
    with tempfile.TemporaryDirectory(prefix="crv-streaming-admission-") as tmp:
        cv = CR.verify(cache_audit, Path(tmp) / "cache.json", expected_plan_sha256=expected_plan_sha256)
        rv = SR.verify(response, Path(tmp) / "response.json")
    if len(registry) != 9000 or len(sources) != 3 or cv["summary_sha256"] != expected_cache_sha256:
        raise ValueError("requires the complete admitted retained cache domain")
    rs = SR._json(response / "run/summary.json")
    rc = SR._json(response / "run/complete.json")["context"]
    if (rs["registry_sha256"] != F.digest(registry) or rc["batch_identity"]["plan_sha256"] != expected_plan_sha256
            or rs["seeds"] != list(SEEDS) or rs["arms"] != list(ARMS)):
        raise ValueError("cache and saved downstream contracts differ")
    excluded = excluded_controls(plan, bindings, registry, pd.read_parquet(response / "run/selected.parquet"))
    selected, assigned, chosen = schedule(registry, frames, batches, excluded)
    calibration = pd.read_parquet(response / "run/calibration.parquet")
    if set(calibration.input_sha256) != set(registry.input_sha256):
        raise ValueError("incomplete input calibration domain")
    saved_frames, saved_batches = (pd.read_parquet(cache_audit / (name + ".parquet")) for name in ("frames", "batches"))
    bindings = C.merge_bindings(bindings, rs["inputs"], rs["code_sha256"],
                                _directory_bindings(cache_audit), _directory_bindings(response))
    M.check_bindings(bindings)
    return {"registry": registry, "plan_frames": frames, "plan_batches": batches, "sources": sources,
            "selected": selected, "assigned": assigned, "chosen": chosen, "excluded": excluded,
            "calibration": calibration, "cache_frames": saved_frames, "cache_batches": saved_batches,
            "bindings": bindings, "verified": {"cache": cv, "response": rv, "plan": pv},
            "plan": str(plan), "cache_audit": str(cache_audit), "response": str(response),
            "hashes": {"plan_sha256": expected_plan_sha256, "cache_sha256": expected_cache_sha256,
                       "response_sha256": expected_response_sha256}}


def one_asset(bindings, digest):
    found = [p for p, h in bindings.items() if h == digest]
    if len(found) != 1:
        raise ValueError("missing or ambiguous pinned response asset")
    return M.P.bound(found[0], digest)


def resources(admitted):
    """Load only the pinned Run1A assets/configuration already bound by the evidence."""
    bindings = admitted["bindings"]
    geom = M.geometry.from_offline_config(str(one_asset(bindings, M.P.COUNTERS_SHA)))
    if geom.nbars != 1280 or geom.sector_names != ["T1", "T2"]:
        raise ValueError("requires original complete retained Run1A geometry")
    lut = one_asset(bindings, M.P.LUT_SHA)
    cfg, _ = M.P.apply_policy(M.P.fastframes.run_config("run1a_v01")[0], M.P.INTENDED)
    plan = SR._json(Path(admitted["plan"]) / "summary.json")
    smoke = SR._json(Path(plan["smoke"]) / "summary.json")
    _, contract = M.P.apply_policy(M.P.fastframes.run_config("run1a_v01")[0], M.P.INTENDED)
    if contract != smoke["response_contract"]:
        raise ValueError("optical response configuration differs from saved admission")
    tables = M.photons.load_tables(cfg, str(lut.parent))
    dev = np.loadtxt(one_asset(bindings, M.PHOTON_DEV_SHA), delimiter=",", comments=("#", "TABLE"))
    if (dev.shape != (4 * geom.nbars, 2) or not np.isfinite(dev).all()
            or not np.array_equal(np.sort(dev[:, 0]), np.arange(4 * geom.nbars))):
        raise ValueError("invalid pinned photon deviation")
    pmap_path = one_asset(bindings, R.D.MAP_SHA)
    template = one_asset(bindings, R.D.TEMPLATE_SHA)
    with uproot.open(pmap_path) as root:
        pmap, x, y = root["photonMap"].to_numpy()
    if not np.array_equal(x, np.arange(41)) or not np.array_equal(y, np.arange(41)):
        raise ValueError("photon map axes disagree")
    return dict(geom=geom, cfg=cfg, tables=tables, deviation=dev[np.argsort(dev[:, 0]), 1], pmap=pmap,
                sipm_cfg=replace(V.SI.SiPMConfig(), thermal_rate=V.THERMAL_RATE),
                waveform_cfg=replace(V.WF.WaveformConfig(), single_pe_file=str(template)))


def layout(admitted):
    chosen = admitted["chosen"]
    tags = [f"batch{int(b):04d}_seed{seed}" for b in chosen.batch for seed in SEEDS]
    return tags, {"top": TOP_CAP, "each_child": CHILD_CAP, "children": len(tags),
                  "maximum_total": TOP_CAP + len(tags) * CHILD_CAP}


def context(admitted, batch=None, seed=None):
    identity = {"contract": SCHEMA, **admitted["hashes"]}
    if batch is not None:
        identity.update(batch=int(batch), frame_ids=[int(i) for i in admitted["assigned"].loc[admitted["assigned"].batch == batch, "frame_id"]])
    return {"registry_sha256": F.digest(admitted["registry"]), "batch_identity": identity,
            "arms": list(ARMS), "seeds": list(SEEDS) if seed is None else [seed]}


def check_selection(selection, admitted, batch):
    """Require exact original cache-audit frame accounting and every typed source fingerprint."""
    expected = admitted["cache_frames"][admitted["cache_frames"].batch == batch].reset_index(drop=True)
    selected = admitted["registry"].iloc[expected.frame_id].reset_index(drop=True)
    counts = {}
    for name in SOURCE_TABLES:
        table = getattr(selection, name)
        counts[name] = len(table)
        saved = admitted["cache_batches"].loc[admitted["cache_batches"].batch == batch, name + "_sha256"]
        if len(saved) != 1 or C.fingerprint(table, name) != saved.iloc[0]:
            raise ValueError("selection differs from completed cache source fingerprint: " + name)
    ev = selection.events
    pd.testing.assert_frame_equal(ev[list(F.COLUMNS)].astype({n: "int64" for n in C.IDS}).reset_index(drop=True), selected, check_exact=True)
    if not np.array_equal(ev.source_row, expected.event_source_row):
        raise ValueError("selected event source rows disagree")
    for name in ("steps", "particles"):
        table = getattr(selection, name)
        actual = table.groupby("frame_id").size()
        if not np.array_equal(expected[name], expected.frame_id.map(actual).fillna(0).astype(np.int64)):
            raise ValueError("selected whole-frame counts disagree")
    row = admitted["cache_batches"].loc[admitted["cache_batches"].batch == batch].iloc[0]
    if selection.typed_payload_bytes != int(row.typed_payload_bytes):
        raise ValueError("selected typed-payload accounting differs")
    return selected, counts


def preflight(plan, cache_audit, response, out, **hashes):
    """Verify prior evidence and freeze bounded new-frame schedule; no raw cache or generation."""
    out = Path(out)
    if out.exists():
        raise ValueError("choose a new preflight file")
    outside_evidence(out, (plan, cache_audit, response))
    d = admission(plan, cache_audit, response, **hashes)
    tags, caps = layout(d)
    result = {"schema": SCHEMA + "-admission", "scope": SCOPE, "selection": SELECTION, "hashes": d["hashes"],
              "registry_sha256": F.digest(d["registry"]), "selected": d["selected"].to_dict("records"),
              "chosen": d["chosen"].to_dict("records"), "tags": tags, "byte_caps": caps,
              "inputs": d["bindings"], "code_sha256": {str(p): C.sha256(p) for p in code_paths()},
              "execution_admitted": False, "optics_executed": False, "response_executed": False}
    M.check_bindings(result["inputs"], result["code_sha256"])
    M._json(out, result)
    return result


def run(plan, cache_audit, response, out, **hashes):
    out = Path(out)
    if out.exists():
        raise ValueError("choose a new streaming pilot directory")
    outside_evidence(out, (plan, cache_audit, response))
    d = admission(plan, cache_audit, response, **hashes)
    a = resources(d)
    code = {str(p): C.sha256(p) for p in code_paths()}
    M.check_bindings(d["bindings"], code)
    tags, caps = layout(d)
    registry, digest = d["registry"], F.digest(d["registry"])
    names = ["summary.json", "registry.parquet", "selected.parquet", "chosen.parquet", "calibration.parquet", "coverage.parquet", "admission.json"]
    names += [tag + ".json" for tag in tags] + [f"code{i}.txt" for i in range(len(code))]
    if len(names) > 128:
        raise ValueError("top artifact domain cap exceeded")
    roots = O.RootUnion(max_roots=MAX_BATCHES * len(SEEDS) * R.Limits().photons)
    records, coverage, visited = [], [], []
    started, before = time.perf_counter(), C.memory()
    out.mkdir(parents=True)
    with A.BatchArtifacts(out / "run", max_bytes=TOP_CAP, context=context(d), expected_artifacts=names) as top:
        for name in ("registry", "selected", "chosen", "calibration"):
            top.write_parquet(name + ".parquet", d[name])
        top.write_json("admission.json", d["verified"])
        for i, path in enumerate(code):
            with top.open_artifact(f"code{i}.txt") as stream:
                stream.write(Path(path).read_bytes())
        for source in d["sources"]:
            digest_input = source["input_sha256"]
            with I.load_input(source["path"], registry, expected_registry_sha256=digest,
                              expected_input_sha256=digest_input, expected_manifest_sha256=source["manifest_sha256"]) as cache:
                for batch in d["chosen"].loc[d["chosen"].input_sha256 == digest_input, "batch"]:
                    batch = int(batch)
                    selected = registry.iloc[d["assigned"].loc[d["assigned"].batch == batch, "frame_id"]]
                    selection = cache.select(selected, expected_registry_sha256=digest)
                    C.compare_snapshot(cache, selection)
                    selected, counts = check_selection(selection, d, batch)
                    if set(selected.frame_id) & set(visited):
                        raise ValueError("duplicate streamed frame")
                    visited.extend(int(i) for i in selected.frame_id)
                    c = d["calibration"][d["calibration"].input_sha256 == digest_input].sort_values("chan")
                    status = c.status.to_numpy()
                    for seed in SEEDS:
                        tag, tick = f"batch{batch:04d}_seed{seed}", time.perf_counter()
                        with A.BatchArtifacts(out / tag, max_bytes=CHILD_CAP, context=context(d, batch, seed), expected_artifacts=CHILD_NAMES) as tx:
                            for name in SOURCE_TABLES:
                                tx.write_parquet(name + ".parquet", getattr(selection, name))
                            q = O.make_batch(selection, a["geom"], a["cfg"], a["tables"], a["deviation"], status,
                                             input_sha256=digest_input, seed=seed)
                            roots.add(q)
                            tx.write_parquet("photons.parquet", q)
                            receipts, comparisons, reference = {}, {}, None
                            for arm in ARMS:
                                products, receipt = R.run_arm(q, registry, selected, d["calibration"], registry_digest=digest,
                                    optical_seed=seed, response_seed=seed, arm=arm, pmap=a["pmap"], sipm_cfg=a["sipm_cfg"],
                                    waveform_cfg=a["waveform_cfg"], nbars=a["geom"].nbars)
                                R.validate_products(products)
                                actual = SR._products(products, registry, selected, d["calibration"], q, seed, arm, a["waveform_cfg"])
                                SR._equal(actual, receipt, "streamed response receipt disagreement")
                                for stage, table in products.items():
                                    tx.write_parquet(f"{arm}_{stage}.parquet", table)
                                coverage.append(products["frames"].assign(batch=batch, optical_seed=np.uint64(seed), response_seed=np.uint64(seed), arm=arm))
                                receipts[arm] = receipt
                                if arm == "historical":
                                    reference = {name: products[name].copy() for name in R.D.STAGES}
                                else:
                                    comparisons[arm] = V.structural_comparisons(reference, products, registry)
                                del products
                            tx.write_json("validation.json", {"batch": batch, "seed": seed, "source_counts": counts,
                                          "photon_counts": M.counts(q), "receipts": receipts, "comparisons": comparisons})
                            del q, reference
                        top.write_json(tag + ".json", SR._json(out / tag / A.COMPLETE))
                        records.append({"path": tag, "batch": batch, "seed": seed, "input_sha256": digest_input,
                                        "receipt_sha256": C.sha256(out / tag / A.COMPLETE), "counts": receipts,
                                        "elapsed_seconds": time.perf_counter() - tick})
                    del selection
            gc.collect()
        ledger = pd.concat(coverage, ignore_index=True).sort_values(["batch", "optical_seed", "arm", "frame_id"]).reset_index(drop=True)
        validate_coverage(ledger, d)
        if sorted(visited) != sorted(d["selected"].frame_id) or set(r["path"] for r in records) != set(tags):
            raise ValueError("streaming batch/frame domain incomplete")
        records.sort(key=lambda record: (record["batch"], record["seed"]))
        top.write_parquet("coverage.parquet", ledger)
        M.check_bindings(d["bindings"], code)
        summary = {"schema": SCHEMA, "scope": SCOPE, "selection": SELECTION, "plan": d["plan"], "cache_audit": d["cache_audit"],
                   "response": d["response"], "hashes": d["hashes"], "registry_sha256": digest, "registry_frames": len(registry),
                   "frames": len(d["selected"]), "batches": len(d["chosen"]), "seeds": list(SEEDS), "arms": list(ARMS),
                   "inputs": d["bindings"], "code_sha256": code, "records": records, "byte_caps": caps, "root_union": roots.summary(),
                   "optical_limits": asdict(O.Limits()), "response_limits": asdict(R.Limits()),
                   "sipm_config": asdict(a["sipm_cfg"]), "waveform_config": asdict(a["waveform_cfg"]),
                   "memory_before_execution": before, "memory_after_execution": C.memory(), "execution_seconds_before_commit": time.perf_counter() - started,
                   "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "uproot": uproot.__version__},
                   "whole_selected_frames_exact": True, "all_declared_arm_seeds_complete": True, "optics_executed": True,
                   "response_executed": True, "execution_admitted": False, "timing_closure": False, "F": None, "finite_library_uncertainty": None}
        top.write_json("summary.json", summary)
    return summary


def validate_coverage(ledger, admitted):
    keys = ["frame_id", "optical_seed", "response_seed", "arm"]
    pairs = pd.DataFrame({"optical_seed": np.asarray(SEEDS, np.uint64), "response_seed": np.asarray(SEEDS, np.uint64)})
    expected = admitted["selected"].merge(pairs, how="cross").merge(pd.DataFrame({"arm": list(ARMS)}), how="cross")
    if ledger.duplicated(keys).any():
        raise ValueError("duplicate arm/seed/frame coverage")
    cols = [*F.COLUMNS, "optical_seed", "response_seed", "arm"]
    SR._table_equal(ledger[cols], expected[cols], keys)
    assignment = admitted["assigned"].set_index("frame_id").batch
    if not np.array_equal(ledger.batch, ledger.frame_id.map(assignment)):
        raise ValueError("streaming coverage batch changed")
    for stage in R.STAGES[1:]:
        if ledger[stage].dtype.kind not in "iu" or (ledger[stage] < 0).any():
            raise ValueError("invalid response frame stage count")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "cache-audit", "response", "out", "expected-plan-sha256", "expected-cache-sha256", "expected-response-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = vars(parser.parse_args(argv))
    action = preflight if args.pop("preflight") else run
    result = action(args.pop("plan"), args.pop("cache_audit"), args.pop("response"), args.pop("out"), **args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
