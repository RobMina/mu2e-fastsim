"""Bounded hash-bound real-asset keyed-optics admission; no downstream response or F."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
import scipy
import uproot

from .. import geometry
from ..response import lookup, photons
from . import keyed_optics as K, optical_timing_probe as P
from . import replay_sipm_control as S
from .replay import EVENT_TYPES, IDS, sha256
from .replay_timing import DIGI_END, PERIOD, stock_step_mask, wrap_arrivals

SCHEMA = "crv-keyed-optical-smoke-v1"
ROOT_CONTRACT = "optical-full-key-root63-v1"
MAX_STEPS = 10000
MAX_PHOTONS = 300000
EXCLUDE_EVENTS = (2505, 281, 2755)
SELECT = "first nonexceptional non-prior-control frame per input plus every nonexceptional zero-stock-step frame"
SCOPE = "Bounded software/asset admission, two response draws on unchanged frames; no downstream, F, CE exposure or library uncertainty."
PHOTON_DEV_SHA = "67897fe9b6a3bb499434100775ae315bbdeab2bc4538600e2d440681ac8a0417"


def _json(path, content):
    with Path(path).open("x") as stream:
        json.dump(content, stream, indent=2, allow_nan=False)


def check_bindings(*mappings):
    seen = {}
    for mapping in mappings:
        for path, digest in mapping.items():
            if path in seen and seen[path] != digest:
                raise ValueError("conflicting hashes for one path")
            seen[path] = digest
            P.bound(path, digest)


def roots(q):
    """Stable digest bridge; full optical keys remain authoritative and collisions fail."""
    if q.duplicated(list(K.KEYS)).any():
        raise ValueError("duplicate full photon key")
    out = np.empty(len(q), dtype=np.int64)
    tag = np.uint64(int.from_bytes(hashlib.sha256(ROOT_CONTRACT.encode()).digest()[:8], "little"))
    for (digest, seed), indices in q.groupby(["input_sha256", "draw_seed"], sort=False).indices.items():
        draw = K.Draws(digest, int(seed))
        value = S._mix(tag ^ np.uint64(seed))
        for word in draw.words:
            value = S._mix(value ^ S._mix(word))
        for name in K.KEYS[2:]:
            a = q[name].to_numpy()[indices]
            if a.dtype.kind not in "iu" or np.any(a < 0) or (a.dtype.kind == "u" and np.any(a >= np.uint64(2**63))):
                raise ValueError("invalid photon integer key")
            value = S._mix(value ^ S._mix(a.astype(np.uint64)))
        out[indices] = (value & np.uint64(2**63 - 1)).astype(np.int64)
    if len(np.unique(out)) != len(out):
        raise ValueError("root digest collision; full keys must not be merged")
    return out


def canonical(q):
    if q.duplicated(list(K.KEYS)).any():
        raise ValueError("duplicate photon key")
    return q.sort_values(list(K.KEYS)).reset_index(drop=True)


def exact(a, b):
    pd.testing.assert_frame_equal(canonical(a), canonical(b), check_exact=True)


def timing_table(raw, folded, steps, events, status, digest, seed):
    raw, folded = canonical(raw), canonical(folded)
    pd.testing.assert_frame_equal(raw[list(K.KEYS)], folded[list(K.KEYS)], check_exact=True)
    if not len(events) or events.duplicated(list(IDS)).any() or steps.source_row.duplicated().any():
        raise ValueError("empty/duplicate census or source identity")
    if not np.isfinite(events.pbtime).all():
        raise ValueError("invalid bunch phase")
    if not (raw.input_sha256 == digest).all() or not (raw.draw_seed == seed).all():
        raise ValueError("wrong photon input or seed domain")
    if not np.isfinite(raw.time).all() or not np.isfinite(folded.time).all():
        raise ValueError("nonfinite saved arrival")
    si = pd.Index(steps.source_row).get_indexer(raw.source_row)
    er = pd.MultiIndex.from_frame(events[list(IDS)]).get_indexer(pd.MultiIndex.from_frame(raw[list(IDS)]))
    se = pd.MultiIndex.from_frame(events[list(IDS)]).get_indexer(pd.MultiIndex.from_frame(steps[list(IDS)]))
    if np.any(si < 0) or np.any(er < 0) or np.any(se < 0):
        raise ValueError("photon or step outside source census")
    for name in IDS:
        if not np.array_equal(raw[name], steps[name].to_numpy()[si]):
            raise ValueError("photon step full identity disagreement")
    if np.any(raw.chan < 0) or np.any(raw.chan >= len(status)) or not np.array_equal(raw.chan // 4, steps.bar.to_numpy()[si]):
        raise ValueError("invalid photon channel")
    q = raw.drop(columns="time").copy()
    q["photon_id"] = roots(q)
    q["event_row"] = er
    w = -events.pbtime.to_numpy().astype(np.float32).astype(float)
    q["origin"] = w[er]
    q["time_raw_absolute"], q["time_historical_relative"] = raw.time.to_numpy(), folded.time.to_numpy()
    wrapped = wrap_arrivals(raw.time.to_numpy(), w[er])
    rel, fw = wrapped - w[er], wrap_arrivals(folded.time.to_numpy())
    q["time_raw_wrapped_absolute"], q["time_raw_wrapped_relative"] = wrapped, rel
    q["time_folded_wrapped_relative"] = fw
    residual = rel - fw
    q["raw_folded_residual_ns"] = residual
    q["raw_folded_circular_residual_ns"] = np.minimum(abs(residual), abs(abs(residual) - PERIOD))
    live = (np.asarray(status)[raw.chan.to_numpy()] & 0b101) == 0
    accepted = stock_step_mask(steps.t0, steps.t1, w[se])[si]
    h, a, f = live & (folded.time.to_numpy() <= DIGI_END), live & (wrapped <= w[er] + DIGI_END), live & (fw <= DIGI_END)
    for name, value in dict(live_channel=live, stock_step_kept=accepted, historical_kept=h,
                           arrival_only_kept=a, rejection_only_kept=h & accepted, stock_timing_kept=a & accepted,
                           folded_arrival_kept=f, raw_folded_upper_flip=a != f,
                           raw_folded_late_flip=live & ((wrapped >= w[er] + 1400.) != (fw >= 1400.)),
                           rescued_by_arrival=a & ~h, removed_by_arrival=h & ~a,
                           rescued_survives_step=a & ~h & accepted,
                           accepted_time_changed=h & a & accepted & (folded.time.to_numpy() != rel)).items():
        q[name] = value
    return q


def generate(steps, events, geom, cfg, tables, deviation, status, digest, seed):
    """Two timing calls plus exact partition/subset checks with identical immutable domains."""
    if len(steps) > MAX_STEPS or len(events) > 10:
        raise ValueError("bounded smoke source cap exceeded")
    if steps.source_row.duplicated().any() or events.duplicated(list(IDS)).any():
        raise ValueError("duplicate source/census")
    er = pd.MultiIndex.from_frame(events[list(IDS)]).get_indexer(pd.MultiIndex.from_frame(steps[list(IDS)]))
    if np.any(er < 0):
        raise ValueError("step outside census")
    w = -events.pbtime.to_numpy().astype(np.float32).astype(float)[er]
    shift = np.floor((steps.t0.to_numpy() - w) / PERIOD) * PERIOD
    folded = steps.assign(t0=steps.t0 - w - shift, t1=steps.t1 - w - shift)
    def call(frame, batch):
        return K.make_photons(frame, frame.pdg.to_numpy(), geom, cfg, tables, deviation,
                              input_sha256=digest, seed=seed, max_substeps=batch, max_photons=MAX_PHOTONS)
    raw = call(steps, 200000)
    historical = call(folded, 200000)
    parts = [call(steps.iloc[index], 257) for index in np.array_split(np.arange(len(steps))[::-1], 3)]
    exact(raw, pd.concat(parts, ignore_index=True))
    subset = steps.iloc[::2]
    exact(raw[raw.source_row.isin(subset.source_row)], call(subset, 127))
    return timing_table(raw, historical, steps, events, status, digest, seed)


def select_frames(frames):
    if not len(frames) or frames.duplicated(list(IDS)).any():
        raise ValueError("empty/duplicate census frames")
    allowed = frames[(frames.fold_start_roundoff == 0) & ~frames.event.isin(EXCLUDE_EVENTS)].sort_values(list(IDS))
    if not len(allowed):
        raise ValueError("no new smoke frame")
    chosen = pd.concat([allowed.iloc[:1], allowed[allowed.stock_keep == 0]]).drop_duplicates(list(IDS))
    if len(chosen) > 10:
        raise ValueError("bounded smoke frame cap exceeded")
    return chosen.sort_values(list(IDS)).reset_index(drop=True)


def selections(audit, verified):
    audit, verified = Path(audit), Path(verified)
    v = json.loads(verified.read_text())
    if v.get("all_ledgers_reclassified_exactly") is not True or v.get("pinned_geometry_verified") is not True:
        raise ValueError("requires verified complete step census")
    report = json.loads(P.bound(audit / "summary.json", v["summary_sha256"]).read_text())
    if report["schema"] != "crv-step-timing-audit-v1" or report["counters"]["sha256"] != P.COUNTERS_SHA:
        raise ValueError("wrong census contract")
    output, seen, hashes = [], set(), {str(audit / "summary.json"): v["summary_sha256"], str(verified): sha256(verified)}
    for i, record in enumerate(report["inputs"]):
        if record["sha256"] in seen:
            raise ValueError("repeated input")
        seen.add(record["sha256"])
        saved = {}
        for name in ("steps", "frames"):
            artifact = record["artifacts"][name]
            if artifact["path"] != f"input{i}_{name}.parquet":
                raise ValueError("unexpected census path")
            path = P.bound(audit / artifact["path"], artifact["sha256"])
            hashes[str(path)] = artifact["sha256"]
            saved[name] = pd.read_parquet(path)
            if len(saved[name]) != artifact["rows"]:
                raise ValueError("census artifact count mismatch")
        ledger, frames = saved["steps"], saved["frames"]
        if not np.array_equal(ledger.step_row, np.arange(len(ledger))):
            raise ValueError("global original source row coverage mismatch")
        wanted = select_frames(frames)
        chosen = ledger[pd.MultiIndex.from_frame(ledger[list(IDS)]).isin(pd.MultiIndex.from_frame(wanted[list(IDS)]))]
        if len(chosen) > MAX_STEPS:
            raise ValueError("bounded smoke step cap exceeded")
        output.append((record, wanted, chosen))
    if not output or sum(len(item[1]) for item in output) > 10:
        raise ValueError("empty or oversized smoke selection")
    return report, output, hashes


def prepare(item, paired, photon_dev, geom, contract):
    record, wanted, expected = item
    path = P.bound(record["path"], record["sha256"])
    sidecar = P.bound(path.with_suffix(".json"), record["manifest_sha256"])
    check_path = P.bound(paired["check"], paired["check_sha256"])
    check = json.loads(check_path.read_text())
    meta_path = P.bound(path.with_name("fast.json"), check["fast_metadata_sha256"])
    identity, _, _ = P.verify_metadata(check, json.loads(meta_path.read_text()), sha256(meta_path))
    if (identity["response_contract"] != contract or check.get("conditions_equal") is not True
            or check["input_sha256"] != record["sha256"] or check["input_manifest_sha256"] != record["manifest_sha256"]):
        raise ValueError("fixed-stock response/input contract differs")
    P.bound(photon_dev, identity["photon_dev_sha256"])
    stock = P.bound(check["stock"], check["stock_sha256"])
    _, all_events, _ = P.read_input(path)
    with uproot.open(stock) as root:
        P.validate_stock_events(all_events, root["crvdump/events"].arrays(library="np"))
        calib = root["crvdump/calib"].arrays(library="pd")
    if (len(calib) != geom.nbars * 4 or not np.array_equal(np.sort(calib.channel), np.arange(geom.nbars * 4))
            or calib.status.dtype.kind not in "iu" or np.any(calib.status < 0)):
        raise ValueError("incomplete calibration")
    events = all_events[pd.MultiIndex.from_frame(all_events[list(IDS)]).isin(pd.MultiIndex.from_frame(wanted[list(IDS)]))].reset_index(drop=True)
    pd.testing.assert_frame_equal(events, wanted[list(EVENT_TYPES)].reset_index(drop=True), check_exact=True)
    with uproot.open(path) as root:
        steps, particles = (P.select_table(root[f"dump/{name}"], events) for name in ("steps", "particles"))
    np.testing.assert_array_equal(steps.source_row, expected.step_row)
    cols = [*IDS, "bar", "t0", "t1", "edep"]
    pd.testing.assert_frame_equal(steps[cols].reset_index(drop=True), expected[cols].reset_index(drop=True), check_exact=True)
    steps = P.join_particles(steps, particles)
    dev = np.loadtxt(photon_dev, delimiter=",", comments=("#", "TABLE"))
    if dev.shape != (geom.nbars * 4, 2) or not np.isfinite(dev).all() or not np.array_equal(np.sort(dev[:, 0]), np.arange(geom.nbars * 4)):
        raise ValueError("incomplete photon deviation")
    status = calib.sort_values("channel").status.to_numpy(dtype=np.int64)
    deviation = dev[np.argsort(dev[:, 0]), 1]
    hashes = {str(p): sha256(p) for p in (path, sidecar, check_path, meta_path, stock, Path(photon_dev))}
    return steps, events, status, deviation, hashes


def code_paths():
    from . import exposure, fastframes, replay, replay_check, replay_metrics, replay_policy, replay_timing, step_timing_audit
    return [Path(__file__), *[Path(m.__file__) for m in (K, P, S, geometry, photons, lookup, exposure,
            fastframes, replay, replay_check, replay_metrics, replay_policy, replay_timing, step_timing_audit)]]


def save_table(out, name, frame):
    path = out / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return {"path": path.name, "sha256": sha256(path), "rows": len(frame)}


def counts(q):
    return {"photons": len(q), "counts": {name: int(q[name].sum()) for name in P.FLAGS},
            "max_circular_residual_ns": float(q.raw_folded_circular_residual_ns.max()) if len(q) else None}


def run(audit, verified, paired, photon_dev, offline, lut_dir, out, seeds):
    out, offline, lut_dir = Path(out), Path(offline), Path(lut_dir)
    if out.exists():
        raise ValueError("choose a new output directory")
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("requires two distinct response seeds")
    for seed in seeds:
        K.Draws("0" * 64, seed)
    census, selected, hashes = selections(audit, verified)
    paired = Path(paired)
    hashes[str(paired)] = sha256(paired)
    records = json.loads(paired.read_text())["blocks"]
    checks = {row["input_sha256"]: row for row in records}
    if len(checks) != len(records) or any(item[0]["sha256"] not in checks for item in selected):
        raise ValueError("missing/repeated fixed-stock binding")
    assets = {str(P.bound(offline / rel, digest)): digest for rel, digest in P.ASSETS.items()}
    assets[str(P.bound(lut_dir / "LookupTable_6000_0", P.LUT_SHA))] = P.LUT_SHA
    assets[str(P.bound(photon_dev, PHOTON_DEV_SHA))] = PHOTON_DEV_SHA
    old = P.bound(census["stock_source"]["path"], P.STOCK_SHA)
    bridge = P.timing_source_bridge(old, offline / "CRVResponse/src/CrvPhotonGenerator_module.cc")
    assets[str(old)] = P.STOCK_SHA
    cfg, _, _, _ = P.fastframes.run_config("run1a_v01")
    cfg, contract = P.apply_policy(cfg, P.INTENDED)
    geom = geometry.from_offline_config(str(offline / "Mu2eG4/geom/crv_counters_run1a_v01.txt"))
    tables = photons.load_tables(cfg, str(lut_dir))
    code = {str(p): sha256(p) for p in code_paths()}
    out.mkdir(parents=True)
    (out / "code").mkdir()
    for i, path in enumerate(code):
        shutil.copyfile(path, out / "code" / f"{i}_{Path(path).name}")
    blocks, root_union = [], []
    for i, item in enumerate(selected):
        started = time.perf_counter()
        steps, events, status, deviation, inputs = prepare(item, checks[item[0]["sha256"]], photon_dev, geom, contract)
        artifacts = {name: save_table(out, f"input{i}_{name}", frame) for name, frame in
                     (("steps", steps), ("events", events), ("conditions", pd.DataFrame({"chan": np.arange(len(status)), "status": status, "deviation": deviation})))}
        replicas = []
        for seed in seeds:
            start = time.perf_counter()
            q = generate(steps, events, geom, cfg, tables, deviation, status, item[0]["sha256"], seed)
            frames, channels = P.summaries(q, events, geom)
            arts = {name: save_table(out, f"input{i}_seed{seed}_{name}", frame) for name, frame in
                    (("photons", q), ("frames", frames), ("channels", channels))}
            root_union.append(q[list(K.KEYS) + ["photon_id"]])
            replicas.append({"seed": seed, "artifacts": arts, **counts(q), "partition_exact": True, "subset_exact": True,
                             "raw_folded_keys_exact": True, "elapsed_seconds": time.perf_counter() - start})
        blocks.append({"input": item[0]["path"], "input_sha256": item[0]["sha256"], "input_bindings": inputs,
                       "selected_events": events[list(IDS)].to_dict(orient="records"), "artifacts": artifacts,
                       "replicas": replicas, "elapsed_seconds": time.perf_counter() - started})
    union = pd.concat(root_union, ignore_index=True)
    np.testing.assert_array_equal(union.photon_id, roots(union))
    report = {"schema": SCHEMA, "root_contract": ROOT_CONTRACT, "rng_contract": K.CONTRACT, "scope": SCOPE,
              "selection": SELECT, "seeds": list(seeds), "assets": assets, "inputs": hashes, "code_sha256": code,
              "paths": {"audit": str(audit), "verified": str(verified), "paired": str(paired), "photon_dev": str(photon_dev)},
              "code_snapshots": {str(p): f"code/{i}_{Path(p).name}" for i, p in enumerate(code)}, 
              "response_contract": contract, "timing_source_bridge": bridge, "blocks": blocks,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                           "scipy": scipy.__version__, "uproot": uproot.__version__},
              "F": None, "finite_library_uncertainty": None, "timing_closure": False}
    check_bindings(assets, hashes, code, *(b["input_bindings"] for b in blocks))
    _json(out / "summary.json", report)
    return report


def verify(directory, out):
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise ValueError("choose a new verification path")
    summary = directory / "summary.json"
    digest = sha256(summary)
    report = json.loads(summary.read_text())
    if (report["schema"] != SCHEMA or report["root_contract"] != ROOT_CONTRACT or report["rng_contract"] != K.CONTRACT
            or report["F"] is not None or report["finite_library_uncertainty"] is not None or report["timing_closure"] is not False
            or len(report["seeds"]) != 2 or len(set(report["seeds"])) != 2 or not report["blocks"]):
        raise ValueError("invalid smoke contract/domain")
    if set(report["code_sha256"]) != {str(p) for p in code_paths()}:
        raise ValueError("missing required code hash")
    if not set((*P.ASSETS.values(), P.LUT_SHA, P.STOCK_SHA, PHOTON_DEV_SHA)).issubset(set(report["assets"].values())):
        raise ValueError("missing required asset hash")
    check_bindings(report["assets"], report["inputs"], report["code_sha256"],
                   *(b["input_bindings"] for b in report["blocks"]))
    counters = [p for p, h in report["assets"].items() if h == P.COUNTERS_SHA]
    if len(counters) != 1:
        raise ValueError("invalid geometry binding")
    geom = geometry.from_offline_config(counters[0])
    paths = report["paths"]
    for path in (str(Path(paths["audit"]) / "summary.json"), paths["verified"], paths["paired"]):
        if path not in report["inputs"]:
            raise ValueError("missing required evidence hash")
    _, selected, evidence = selections(paths["audit"], paths["verified"])
    if not evidence.items() <= report["inputs"].items() or len(selected) != len(report["blocks"]):
        raise ValueError("missing census hash/domain")
    paired_rows = json.loads(Path(paths["paired"]).read_text())["blocks"]
    paired = {r["input_sha256"]: r for r in paired_rows}
    if len(paired) != len(paired_rows):
        raise ValueError("duplicate paired input")
    _, intended = P.apply_policy(P.fastframes.run_config("run1a_v01")[0], P.INTENDED)
    if report["response_contract"] != intended or report["selection"] != SELECT:
        raise ValueError("changed response/selection contract")
    if set(report["code_snapshots"]) != set(report["code_sha256"]):
        raise ValueError("missing source snapshot")
    for i, (path, h) in enumerate(report["code_sha256"].items()):
        if report["code_snapshots"][path] != f"code/{i}_{Path(path).name}":
            raise ValueError("unexpected source snapshot path")
        P.bound(directory / report["code_snapshots"][path], h)
    union, checked, seen = [], [], set()
    for i, block in enumerate(report["blocks"]):
        if block["input_sha256"] in seen or not block["selected_events"]:
            raise ValueError("repeated input or empty event domain")
        seen.add(block["input_sha256"])
        if block["input_bindings"].get(block["input"]) != block["input_sha256"]:
            raise ValueError("missing input hash")
        item = selected[i]
        if item[0]["sha256"] != block["input_sha256"] or item[0]["path"] != block["input"]:
            raise ValueError("changed selected input")
        pair = paired[block["input_sha256"]]
        check_path = P.bound(pair["check"], pair["check_sha256"])
        check = json.loads(check_path.read_text())
        required = {block["input"]: block["input_sha256"], str(Path(block["input"]).with_suffix(".json")): item[0]["manifest_sha256"],
                    str(check_path): pair["check_sha256"], str(Path(block["input"]).with_name("fast.json")): check["fast_metadata_sha256"],
                    check["stock"]: check["stock_sha256"], paths["photon_dev"]: PHOTON_DEV_SHA}
        if not required.items() <= block["input_bindings"].items():
            raise ValueError("missing required input binding")
        for path, h in block["input_bindings"].items():
            P.bound(path, h)
        def read(arts, name, stem):
            a = arts[name]
            if a["path"] != f"{stem}_{name}.parquet":
                raise ValueError("wrong artifact path")
            frame = pd.read_parquet(P.bound(directory / a["path"], a["sha256"]))
            if len(frame) != a["rows"]:
                raise ValueError("artifact count disagreement")
            return frame
        steps, events, cond = (read(block["artifacts"], name, f"input{i}") for name in ("steps", "events", "conditions"))
        admitted_steps, admitted_events, admitted_status, admitted_dev, admitted_hashes = prepare(
            item, pair, paths["photon_dev"], geom, intended)
        if not admitted_hashes.items() <= block["input_bindings"].items():
            raise ValueError("saved binding differs from repeated admission")
        pd.testing.assert_frame_equal(steps, admitted_steps, check_exact=True)
        pd.testing.assert_frame_equal(events, admitted_events, check_exact=True)
        np.testing.assert_array_equal(cond.status, admitted_status)
        np.testing.assert_array_equal(cond.deviation, admitted_dev)
        if events[list(IDS)].to_dict(orient="records") != block["selected_events"] or len(events) > 10 or len(steps) > MAX_STEPS:
            raise ValueError("changed selected census")
        if (not np.array_equal(cond.chan, np.arange(geom.nbars * 4)) or cond.status.dtype.kind not in "iu"
                or np.any(cond.status < 0) or not np.isfinite(cond.deviation).all()):
            raise ValueError("invalid saved conditions")
        pd.testing.assert_frame_equal(events, item[1][list(EVENT_TYPES)].reset_index(drop=True), check_exact=True)
        cols = [*IDS, "bar", "t0", "t1", "edep"]
        np.testing.assert_array_equal(steps.source_row, item[2].step_row)
        pd.testing.assert_frame_equal(steps[cols].reset_index(drop=True), item[2][cols].reset_index(drop=True), check_exact=True)
        with uproot.open(check["stock"]) as root:
            calib = root["crvdump/calib"].arrays(library="pd").sort_values("channel")
        np.testing.assert_array_equal(cond.chan, calib.channel)
        np.testing.assert_array_equal(cond.status, calib.status)
        dev = np.loadtxt(paths["photon_dev"], delimiter=",", comments=("#", "TABLE"))
        np.testing.assert_array_equal(cond.chan, np.sort(dev[:, 0]))
        np.testing.assert_array_equal(cond.deviation, dev[np.argsort(dev[:, 0]), 1])
        if [r["seed"] for r in block["replicas"]] != report["seeds"]:
            raise ValueError("missing seed domain")
        for replica in block["replicas"]:
            seed = replica["seed"]
            q, frames, channels = (read(replica["artifacts"], name, f"input{i}_seed{seed}") for name in ("photons", "frames", "channels"))
            raw = q[list(K.KEYS)].assign(time=q.time_raw_absolute)
            folded = q[list(K.KEYS)].assign(time=q.time_historical_relative)
            rebuilt = timing_table(raw, folded, steps, events, cond.status.to_numpy(), block["input_sha256"], seed)
            pd.testing.assert_frame_equal(q, rebuilt, check_exact=True)
            ef, ec = P.summaries(q, events, geom)
            pd.testing.assert_frame_equal(frames, ef, check_exact=True)
            pd.testing.assert_frame_equal(channels, ec, check_exact=True)
            if any(replica.get(k) is not True for k in ("partition_exact", "subset_exact", "raw_folded_keys_exact")):
                raise ValueError("missing generation invariance assertion")
            if any(replica[k] != v for k, v in counts(q).items()):
                raise ValueError("summary disagrees with saved photons")
            union.append(q[list(K.KEYS) + ["photon_id"]])
            checked.append({"input": i, "seed": seed, "frames": len(events), "steps": len(steps), **counts(q)})
    allq = pd.concat(union, ignore_index=True)
    np.testing.assert_array_equal(allq.photon_id, roots(allq))
    P.bound(summary, digest)
    result = {"summary_sha256": digest, "saved_ledgers_exact": True, "root_union_collision_free": True,
              "checks": checked, "scope": "Saved-only recomputation; no optical redraw or stock optical equivalence."}
    _json(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify")
    for name in ("audit", "verified", "paired", "photon-dev", "offline", "lut-dir"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=3027000)
    parser.add_argument("--second-seed", type=int, default=3027001)
    a = parser.parse_args(argv)
    if a.verify:
        result = verify(a.verify, a.out)
    else:
        if any(getattr(a, name) is None for name in ("audit", "verified", "paired", "photon_dev", "offline", "lut_dir")):
            parser.error("generation requires all input/asset paths")
        result = run(a.audit, a.verified, a.paired, a.photon_dev, a.offline, a.lut_dir, a.out, [a.seed, a.second_seed])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
