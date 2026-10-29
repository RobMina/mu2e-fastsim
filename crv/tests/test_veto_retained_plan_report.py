"""Saved-only full registry and batch verifier with synthetic admitted census."""
import json
from pathlib import Path

import pandas as pd
import pytest

from crvfastsim.veto import retained_plan_report as R, retained_plan as P, frame_registry as F, keyed_optical_smoke as M
from test_veto_retained_plan import inputs, Tree


def saved(tmp_path):
    st, ev, ledger, geom, cfg, tables, dev, status = inputs()
    original = tmp_path / "mixed.root"
    original.write_bytes(b"synthetic input binding")
    original.with_suffix(".json").write_text("synthetic sidecar binding")
    digest = M.sha256(original)
    work = P.workload(Tree(st, 3), ev, ledger, geom, cfg, status)
    frame_census = ev.assign(steps=work.steps, finite=work.finite_steps)
    census = tmp_path / "census"
    census.mkdir()
    arts = {name: M.save_table(census, f"input0_{name}", df) for name, df in (("steps", ledger), ("frames", frame_census))}
    source = {"path": str(original), "sha256": digest, "manifest_sha256": M.sha256(original.with_suffix(".json")),
              "frames": len(ev), "steps": len(st), "artifacts": arts}
    M._json(census / "summary.json", {"inputs": [source]})
    binds = {str(census / "summary.json"): M.sha256(census / "summary.json"),
             **{str(census / a["path"]): a["sha256"] for a in arts.values()}}
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    prior = {"assets": {}, "inputs": binds, "code_sha256": {}, "blocks": [{"input_bindings": {}}],
             "paths": {"audit": str(census)}}
    M._json(smoke / "summary.json", prior)
    verified = tmp_path / "smoke_verified.json"
    M._json(verified, {"summary_sha256": M.sha256(smoke / "summary.json"), "saved_ledgers_exact": True,
                       "root_union_collision_free": True})
    work["input_sha256"] = digest
    registry = F.freeze(work)
    work = work.merge(registry[[*F.FRAME_KEYS, "frame_id"]], on=list(F.FRAME_KEYS))
    assignment, batches = P.batch_plan(work)
    out = tmp_path / "plan"
    out.mkdir()
    art = {name: M.save_table(out, name, table) for name, table in (("registry", registry), ("frames", assignment), ("batches", batches))}
    paths = (smoke / "summary.json", verified, original, original.with_suffix(".json"), census / arts["steps"]["path"])
    code = [Path(m.__file__) for m in (P, F, M, P.geometry, P.photons)]
    report = {"schema": P.SCHEMA, "execution_admitted": False, "runtime_projection": None, "F": None,
              "finite_library_uncertainty": None, "input_records": [{"input_sha256": digest, "path": str(original),
              "frames": len(ev), "steps": len(st)}], "code_sha256": {str(p): M.sha256(p) for p in code},
              "smoke": str(smoke), "smoke_verified": str(verified), "inputs": {str(p): M.sha256(p) for p in paths},
              "admission_bindings": {"assets": {}, "inputs": binds, "code": {}, "blocks": [{}]},
              "artifacts": art, "registry": F.manifest(registry, P.SEEDS), "limits": P.DEFAULT_LIMITS,
              "frame_stats": P.describe(assignment, ["steps", "finite_steps", "simulated_substeps", "optical_support_pairs", "live_pairs"]),
              "batch_stats": P.describe(batches, list(P.DEFAULT_LIMITS) + ["sipm_state_bytes", "ranged_rows"]),
              "noncontiguous_frames": int((~assignment.contiguous).sum()),
              "batches_with_extra_range_rows": int((~batches.range_has_no_extra_rows).sum())}
    M._json(out / "summary.json", report)
    return out, report


def test_saved_only_verify_and_no_overwrite(tmp_path, monkeypatch):
    directory, _ = saved(tmp_path)
    monkeypatch.setattr(P, "workload", lambda *a: pytest.fail("must not rescan geometry"))
    result = R.verify(directory, tmp_path / "verified.json")
    assert result["registry_batches_exact"] and result["frames"] == 4 and result["steps"] == 12
    with pytest.raises(ValueError, match="new verification"):
        R.verify(directory, tmp_path / "verified.json")


@pytest.mark.parametrize("case", ["artifact", "input", "admission", "code", "seed", "frames", "stats", "range", "domain"])
def test_missing_or_corrupt_plan_fails(tmp_path, case):
    directory, report = saved(tmp_path)
    if case == "artifact":
        report["artifacts"].pop("registry")
    elif case == "input":
        report["inputs"].pop(report["input_records"][0]["path"])
    elif case == "admission":
        report["admission_bindings"]["inputs"] = {}
    elif case == "code":
        report["code_sha256"].pop(next(iter(report["code_sha256"])))
    elif case == "seed":
        report["registry"]["helper_seeds"] = {}
    elif case == "frames":
        report["input_records"][0]["frames"] += 1
    elif case == "stats":
        report["frame_stats"]["steps"]["sum"] += 1
    elif case == "range":
        report["noncontiguous_frames"] += 1
    else:
        report["input_records"] = []
    (directory / "summary.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, AssertionError, KeyError)):
        R.verify(directory, tmp_path / "bad.json")


def test_shifted_dense_ids_cannot_rebind_saved_frame_registry(tmp_path):
    directory, report = saved(tmp_path)
    frames = pd.read_parquet(directory / "frames.parquet")
    frames["frame_id"] += 1
    frames, batches = P.batch_plan(frames.drop(columns="batch"), report["limits"])
    for name, table in (("frames", frames), ("batches", batches)):
        report["artifacts"][name] = M.save_table(directory, name, table)
    (directory / "summary.json").write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        R.verify(directory, tmp_path / "shifted.json")


@pytest.mark.parametrize("missing", ["summary.json", "input0_frames.parquet"])
def test_rebound_admission_cannot_omit_mandatory_census_hash(tmp_path, missing):
    directory, report = saved(tmp_path)
    smoke_path = Path(report["smoke"]) / "summary.json"
    prior = json.loads(smoke_path.read_text())
    key = str(tmp_path / "census" / missing)
    prior["inputs"].pop(key)
    report["admission_bindings"]["inputs"].pop(key)
    smoke_path.write_text(json.dumps(prior))
    verified_path = Path(report["smoke_verified"])
    verified = json.loads(verified_path.read_text())
    verified["summary_sha256"] = M.sha256(smoke_path)
    verified_path.write_text(json.dumps(verified))
    report["inputs"][str(smoke_path)] = M.sha256(smoke_path)
    report["inputs"][str(verified_path)] = M.sha256(verified_path)
    (directory / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="census.*binding"):
        R.verify(directory, tmp_path / "unbound.json")
