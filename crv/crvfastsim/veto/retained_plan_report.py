"""Saved-only registry/workload batch verification; no raw geometry or response regeneration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import frame_registry as F, retained_plan as P, keyed_optical_smoke as M
from .replay import IDS, EVENT_TYPES, sha256


def verify(directory, out):
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise ValueError("choose a new verification file")
    path = directory / "summary.json"
    digest = sha256(path)
    report = json.loads(path.read_text())
    if (report["schema"] != P.SCHEMA or report["execution_admitted"] is not False or report["runtime_projection"] is not None
            or report["F"] is not None or report["finite_library_uncertainty"] is not None or not report["input_records"]):
        raise ValueError("wrong plan contract/domain")
    expected_code = {str(p) for p in (Path(P.__file__), Path(F.__file__), Path(M.__file__), Path(P.geometry.__file__), Path(P.photons.__file__))}
    if set(report["code_sha256"]) != expected_code:
        raise ValueError("missing plan code binding")
    smoke_summary = str(Path(report["smoke"]) / "summary.json")
    required = (smoke_summary, report["smoke_verified"])
    if any(p not in report["inputs"] for p in required):
        raise ValueError("missing smoke admission hash")
    M.check_bindings(report["inputs"], report["code_sha256"])
    prior = json.loads(Path(smoke_summary).read_text())
    admission = report["admission_bindings"]
    expected = {"assets": prior["assets"], "inputs": prior["inputs"], "code": prior["code_sha256"],
                "blocks": [b["input_bindings"] for b in prior["blocks"]]}
    if admission != expected:
        raise ValueError("missing transitive admission bindings")
    M.check_bindings(report["inputs"], report["code_sha256"], admission["assets"], admission["inputs"],
                     admission["code"], *admission["blocks"])
    verified = json.loads(Path(report["smoke_verified"]).read_text())
    if (verified.get("summary_sha256") != sha256(smoke_summary) or verified.get("saved_ledgers_exact") is not True
            or verified.get("root_union_collision_free") is not True):
        raise ValueError("invalid smoke verification receipt")
    tables = {}
    if set(report["artifacts"]) != {"registry", "frames", "batches"}:
        raise ValueError("missing required plan artifact")
    for name, artifact in report["artifacts"].items():
        if artifact["path"] != f"{name}.parquet":
            raise ValueError("wrong artifact path")
        table = pd.read_parquet(M.P.bound(directory / artifact["path"], artifact["sha256"]))
        if len(table) != artifact["rows"]:
            raise ValueError("wrong artifact count")
        tables[name] = table
    registry, frames, batches = (tables[n] for n in ("registry", "frames", "batches"))
    if report["registry"] != F.manifest(registry, P.SEEDS):
        raise ValueError("changed registry/seed schedule")
    pd.testing.assert_frame_equal(F.freeze(frames), registry, check_exact=True)
    actual_map = frames[[*F.FRAME_KEYS, "frame_id"]].sort_values(list(F.FRAME_KEYS)).reset_index(drop=True)
    for name in (*IDS, "frame_id"):
        if actual_map[name].dtype.kind not in "iu" or np.any(actual_map[name] < 0) or np.any(actual_map[name] >= 2**63):
            raise ValueError("invalid frame mapping integers")
    pd.testing.assert_frame_equal(actual_map.astype({name: np.int64 for name in (*IDS, "frame_id")}),
                                  registry[[*F.FRAME_KEYS, "frame_id"]], check_exact=True)
    assignment, rebuilt = P.batch_plan(frames.drop(columns="batch"), report["limits"])
    pd.testing.assert_frame_equal(frames, assignment, check_exact=True)
    pd.testing.assert_frame_equal(batches, rebuilt, check_exact=True)
    if report["frame_stats"] != P.describe(frames, ["steps", "finite_steps", "simulated_substeps", "optical_support_pairs", "live_pairs"]):
        raise ValueError("changed frame stats")
    if report["batch_stats"] != P.describe(batches, list(P.DEFAULT_LIMITS) + ["sipm_state_bytes", "ranged_rows"]):
        raise ValueError("changed batch stats")
    if (report["noncontiguous_frames"] != int((~frames.contiguous).sum())
            or report["batches_with_extra_range_rows"] != int((~batches.range_has_no_extra_rows).sum())):
        raise ValueError("changed contiguity summary")
    census_path = str(Path(prior["paths"]["audit"]) / "summary.json")
    if census_path not in admission["inputs"]:
        raise ValueError("missing census summary binding")
    census = json.loads(M.P.bound(census_path, admission["inputs"][census_path]).read_text())
    if len(census["inputs"]) != len(report["input_records"]):
        raise ValueError("incomplete source domain")
    for record, source in zip(report["input_records"], census["inputs"]):
        if record["input_sha256"] != source["sha256"] or record["path"] != source["path"]:
            raise ValueError("changed input identity")
        art = source["artifacts"]["steps"]
        ledger_path = str(Path(prior["paths"]["audit"]) / art["path"])
        must = {record["path"]: source["sha256"], str(Path(record["path"]).with_suffix(".json")): source["manifest_sha256"],
                ledger_path: art["sha256"]}
        if not must.items() <= report["inputs"].items():
            raise ValueError("missing original input/census binding")
        ledger = pd.read_parquet(ledger_path)
        sub = frames[frames.input_sha256 == source["sha256"]].reset_index(drop=True)
        frame_art = source["artifacts"]["frames"]
        frame_path = str(Path(prior["paths"]["audit"]) / frame_art["path"])
        if admission["inputs"].get(frame_path) != frame_art["sha256"]:
            raise ValueError("missing original frame census binding")
        original = pd.read_parquet(M.P.bound(frame_path, frame_art["sha256"]))
        pd.testing.assert_frame_equal(sub[list(EVENT_TYPES)].astype(EVENT_TYPES), original[list(EVENT_TYPES)], check_exact=True)
        if (len(sub) != record["frames"] or len(ledger) != record["steps"] or len(sub) != source["frames"]
                or len(ledger) != source["steps"] or sub.steps.sum() != len(ledger)):
            raise ValueError("changed all-frame counts")
        for name, expected in (("steps", original.steps), ("finite_steps", original.finite)):
            np.testing.assert_array_equal(sub[name], expected)
        er = pd.MultiIndex.from_frame(sub[list(IDS)]).get_indexer(pd.MultiIndex.from_frame(ledger[list(IDS)]))
        if np.any(er < 0) or not np.array_equal(ledger.step_row, np.arange(len(ledger))):
            raise ValueError("changed original row identities")
        first, last = np.full(len(sub), len(ledger), np.int64), np.full(len(sub), -1, np.int64)
        np.minimum.at(first, er, ledger.step_row)
        np.maximum.at(last, er, ledger.step_row)
        np.testing.assert_array_equal(sub.source_first, np.where(sub.steps > 0, first, -1))
        np.testing.assert_array_equal(sub.source_stop, np.where(sub.steps > 0, last + 1, -1))
        np.testing.assert_array_equal(sub.contiguous, (sub.steps == 0) | (last - first + 1 == sub.steps))
        if (np.any(sub.simulated_substeps < sub.finite_steps * 2) or np.any(sub.optical_support_pairs > sub.live_pairs)
                or np.any(sub.optical_support_pairs < 0) or np.any(sub.live_pairs < 0)):
            raise ValueError("invalid saved workload bounds")
    M.P.bound(path, digest)
    result = {"summary_sha256": digest, "registry_sha256": F.digest(registry), "frames": len(frames), "batches": len(batches),
              "steps": int(frames.steps.sum()), "simulated_substeps": int(frames.simulated_substeps.sum()),
              "registry_batches_exact": True, "original_census_counts_ranges_exact": True,
              "execution_admitted": False, "scope": "Saved plan and original census recomputed; geometric substeps/support are hash-bound measurements, not rescanned."}
    M._json(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    a = parser.parse_args(argv)
    print(json.dumps(verify(a.directory, a.out), indent=2))


if __name__ == "__main__":
    main()
