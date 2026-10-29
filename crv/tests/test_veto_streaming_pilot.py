"""Bounded combined streaming on tiny real indexed inputs and fail-closed orchestration."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import streaming_pilot as S
from test_veto_indexed_replay import sample, artifact
from test_veto_keyed_optics import fixture as optical_fixture
from test_veto_qualified_response import configs


def pilot_fixture(tmp_path, monkeypatch, *, invoke=False):
    tables = sample()
    st, _, geom, cfg, luts, deviation = optical_fixture(5)
    for name in S.I.TYPES["steps"]:
        if name not in (*S.C.IDS, "sp"):
            tables["steps"][name] = st[name].to_numpy().astype(S.I.TYPES["steps"][name])
    tables["steps"]["t0"] = 700.
    tables["steps"]["t1"] = [700.1, 700.1, np.nan, 700.1, 700.1]
    path, full, args, tables = artifact(tmp_path, tables=tables)
    registry = S.F.freeze(full[full.input_sha256 == args["expected_input_sha256"]])
    args["expected_registry_sha256"] = S.F.digest(registry)
    index = pd.MultiIndex.from_frame(registry[list(S.C.IDS)])
    sr = index.get_indexer(pd.MultiIndex.from_frame(tables["steps"][list(S.C.IDS)]))
    counts = np.bincount(sr, minlength=len(registry))
    finite = np.isfinite(tables["steps"].t0) & np.isfinite(tables["steps"].t1)
    frames = registry.copy()
    frames["steps"], frames["finite_steps"] = counts, np.bincount(sr[finite], minlength=len(registry))
    frames["simulated_substeps"], frames["live_pairs"] = frames.finite_steps * 8, 4
    first, last = np.full(len(registry), len(sr), np.int64), np.full(len(registry), -1, np.int64)
    np.minimum.at(first, sr, np.arange(len(sr)))
    np.maximum.at(last, sr, np.arange(len(sr)))
    frames["source_first"], frames["source_stop"] = np.where(counts, first, -1), np.where(counts, last + 1, -1)
    frames, batches = S.C.PR.P.batch_plan(frames, {**S.C.PR.P.DEFAULT_LIMITS, "frames": 2})
    ledger = tables["steps"][S.C.LEDGER_COLUMNS[:-1]].assign(step_row=np.arange(len(sr)))
    lp = tmp_path / "source.parquet"
    ledger.to_parquet(lp, index=False)
    source = dict(path=str(path), input_sha256=args["expected_input_sha256"], manifest_sha256=args["expected_manifest_sha256"],
                  ledger=str(lp), ledger_sha256=S.C.sha256(lp))
    cf, cb, _ = S.C.audit_input(source, registry, frames, batches)
    excluded = registry.iloc[:1]
    selected, assigned, chosen = S.schedule(registry, frames, batches, excluded)
    calib = pd.DataFrame({"chan": np.arange(4), "pedestal": 100., "pulsearea": 1000., "pulseheight": 20.,
                          "timeoffset": 2., "status": 0, "roc": 1, "feb": 1, "input_sha256": source["input_sha256"]})
    _, wcfg = configs(monkeypatch)
    scfg = replace(S.V.SI.SiPMConfig(), thermal_rate=S.V.THERMAL_RATE)
    resource = dict(geom=geom, cfg=cfg, tables=luts, deviation=deviation, pmap=np.ones((40, 40)), sipm_cfg=scfg, waveform_cfg=wcfg)
    bindings = {str(path): source["input_sha256"], str(path.with_suffix(".json")): source["manifest_sha256"], str(lp): source["ledger_sha256"]}
    hashes = dict(expected_plan_sha256="a" * 64, expected_cache_sha256="b" * 64, expected_response_sha256="c" * 64)
    admitted = dict(registry=registry, plan_frames=frames, plan_batches=batches, sources=[source], selected=selected,
                    assigned=assigned, chosen=chosen, excluded=excluded, calibration=calib, cache_frames=cf, cache_batches=cb,
                    bindings=bindings, verified={"fixture": True}, plan="plan", cache_audit="cache", response="response",
                    hashes={k.removeprefix("expected_"): v for k, v in hashes.items()})
    monkeypatch.setattr(S, "admission", lambda *a, **k: admitted)
    monkeypatch.setattr(S, "resources", lambda *a: resource)
    original = S.R.Q.conditions
    monkeypatch.setattr(S.R.Q, "conditions", lambda r, s, c, d, nbars=1: original(r, s, c, d, nbars=1))
    out = tmp_path / "pilot"
    result = S.run("plan", "cache", "response", out, **hashes) if invoke else None
    return out, admitted, resource, hashes, result


def test_real_synthetic_cache_to_optics_to_bounded_four_arm_two_seed(tmp_path, monkeypatch):
    out, d, a, hashes, result = pilot_fixture(tmp_path, monkeypatch, invoke=True)
    assert result["frames"] == 2 and result["batches"] == 1 and len(result["records"]) == 2
    assert result["optics_executed"] and result["response_executed"]
    assert not result["execution_admitted"] and result["F"] is None
    assert result["root_union"]["roots"] > 0 and result["root_union"]["collision_free"]
    assert result["byte_caps"]["maximum_total"] == S.TOP_CAP + 2 * S.CHILD_CAP
    coverage = pd.read_parquet(out / "run/coverage.parquet")
    S.validate_coverage(coverage, d)
    assert len(coverage) == 16 and set(coverage.frame_id) == {2, 3}
    assert set(coverage.event) == {9, 10}
    assert (coverage.loc[coverage.event == 10, "conditions"] == 4).all()
    for record in result["records"]:
        child = out / record["path"]
        assert (child / "complete.json").is_file()
        assert len(json.loads((child / "complete.json").read_text())["artifacts"]) == 53
        st = pd.read_parquet(child / "steps.parquet")
        assert st.source_row.tolist() == [3]
        assert st.event.tolist() == [9]
        q = pd.read_parquet(child / "photons.parquet")
        assert q.source_row.eq(3).all()
        assert set(record["counts"]) == set(S.ARMS)
        assert all(v["frames"] == 2 and v["live_channels"] == 8 for v in record["counts"].values())
    with pytest.raises(ValueError, match="new streaming"):
        S.run("plan", "cache", "response", out, **hashes)


@pytest.mark.parametrize("case", ["source_fingerprint", "typed_bytes", "missing_frame", "optics", "response", "coverage", "child_cap", "top_cap", "late_binding", "global_roots"])
def test_incomplete_run_never_publishes_top_completion(tmp_path, monkeypatch, case):
    out, d, a, hashes, _ = pilot_fixture(tmp_path, monkeypatch)
    def fail(*a, **k):
        raise RuntimeError("injected failure")
    if case == "source_fingerprint":
        d["cache_batches"].loc[d["cache_batches"].batch == 1, "steps_sha256"] = "0" * 64
    elif case == "typed_bytes":
        d["cache_batches"].loc[d["cache_batches"].batch == 1, "typed_payload_bytes"] += 1
    elif case == "missing_frame":
        d["assigned"] = d["assigned"].iloc[:1]
    elif case == "optics":
        monkeypatch.setattr(S.O, "make_batch", fail)
    elif case == "response":
        monkeypatch.setattr(S.R, "run_arm", fail)
    elif case == "coverage":
        monkeypatch.setattr(S, "validate_coverage", fail)
    elif case == "child_cap":
        monkeypatch.setattr(S, "CHILD_CAP", 2000)
    elif case == "top_cap":
        monkeypatch.setattr(S, "TOP_CAP", 2000)
    elif case == "late_binding":
        original = S.validate_coverage
        def mutate(*a):
            original(*a)
            Path(d["sources"][0]["ledger"]).write_bytes(b"changed")
        monkeypatch.setattr(S, "validate_coverage", mutate)
    else:
        monkeypatch.setattr(S.O.RootUnion, "add", fail)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        S.run("plan", "cache", "response", out, **hashes)
    assert not (out / "run/complete.json").exists()
    assert (out / "run/failed.json").exists()


def test_preflight_binds_schedule_no_cache_or_generation(tmp_path, monkeypatch):
    _, d, _, hashes, _ = pilot_fixture(tmp_path, monkeypatch)
    def forbidden(*a, **k):
        pytest.fail("preflight executed physics or cache")
    monkeypatch.setattr(S.I, "load_input", forbidden)
    monkeypatch.setattr(S.O, "make_batch", forbidden)
    monkeypatch.setattr(S.R, "run_arm", forbidden)
    monkeypatch.setattr(S, "resources", forbidden)
    result = S.preflight("plan", "cache", "response", tmp_path / "preflight.json", **hashes)
    assert not result["optics_executed"] and not result["response_executed"]
    assert [x["frame_id"] for x in result["selected"]] == [2, 3]
    assert result["tags"] == [f"batch0001_seed{s}" for s in S.SEEDS]


@pytest.mark.parametrize("case", ["all_excluded", "new_registry_input", "frame_cap", "batch_cap", "oversized_batch"])
def test_schedule_never_splits_or_drops_input(tmp_path, monkeypatch, case):
    _, d, _, _, _ = pilot_fixture(tmp_path, monkeypatch)
    if case == "all_excluded":
        d["excluded"] = d["registry"]
    elif case == "new_registry_input":
        foreign = d["registry"].assign(input_sha256="f" * 64)
        d["registry"] = S.F.freeze(pd.concat([d["registry"], foreign]))
    elif case == "frame_cap":
        monkeypatch.setattr(S, "MAX_FRAMES", 1)
    elif case == "batch_cap":
        monkeypatch.setattr(S, "MAX_BATCHES", 0)
    else:
        d["plan_batches"].loc[d["plan_batches"].batch == 1, "frames"] = 5
    with pytest.raises(ValueError):
        S.schedule(d["registry"], d["plan_frames"], d["plan_batches"], d["excluded"])


@pytest.mark.parametrize("case", ["drop", "duplicate", "seed", "arm", "source", "batch", "negative_count", "fractional_count"])
def test_explicit_arm_seed_frame_coverage(tmp_path, monkeypatch, case):
    _, d, _, _, _ = pilot_fixture(tmp_path, monkeypatch)
    ledger = d["selected"].merge(pd.DataFrame({"optical_seed": np.asarray(S.SEEDS, np.uint64), "response_seed": np.asarray(S.SEEDS, np.uint64)}), how="cross")
    ledger = ledger.merge(pd.DataFrame({"arm": list(S.ARMS)}), how="cross")
    ledger["batch"] = 1
    for stage in S.R.STAGES[1:]:
        ledger[stage] = np.int64(0)
    S.validate_coverage(ledger, d)
    if case == "drop":
        ledger = ledger.iloc[:-1]
    elif case == "duplicate":
        ledger = pd.concat([ledger, ledger.iloc[:1]])
    elif case == "seed":
        ledger.loc[0, "response_seed"] += np.uint64(1)
    elif case == "arm":
        ledger.loc[0, "arm"] = "other"
    elif case == "source":
        ledger.loc[0, "input_sha256"] = "f" * 64
    elif case == "batch":
        ledger.loc[0, "batch"] += 1
    elif case == "negative_count":
        ledger.loc[0, "pulses"] = -1
    else:
        ledger["pulses"] = .5
    with pytest.raises((ValueError, AssertionError)):
        S.validate_coverage(ledger, d)


def test_complete_code_binding_contains_executed_frozen_helpers():
    names = {p.name for p in S.code_paths()}
    assert {"streaming_pilot.py", "streaming_optics.py", "retained_cache_audit.py", "retained_cache_report.py",
            "saved_qualified_response.py", "saved_qualified_report.py", "indexed_replay.py", "bounded_response.py",
            "bounded_sipm.py", "bounded_waveform.py", "diagnostic_fit.py", "diagnostic_lm.py", "keyed_optics.py",
            "replay_sipm_control.py", "replay_waveform_control.py", "replay_timing.py"} <= names
    assert len(S.code_paths()) == len(set(S.code_paths()))


def admitted_inputs(tmp_path, monkeypatch):
    registry = S.F.freeze(pd.concat([pd.DataFrame({"input_sha256": c * 64, "run": 1430, "subrun": 0,
        "event": np.arange(1, 3001, dtype=np.int64), "pbtime": -224.46, "intensity": 1.}) for c in "abc"]))
    frames = registry.assign(steps=0, finite_steps=0, simulated_substeps=0, live_pairs=5120, source_first=-1, source_stop=-1)
    frames, batches = S.C.PR.P.batch_plan(frames)
    sources = [{"path": str(tmp_path / (c + ".root")), "input_sha256": c * 64, "manifest_sha256": "d" * 64,
                "ledger": str(tmp_path / (c + ".parquet")), "ledger_sha256": "e" * 64} for c in "abc"]
    plan, cache, response = (tmp_path / n for n in ("plan", "cache", "response"))
    plan.mkdir()
    cache.mkdir()
    (response / "run").mkdir(parents=True)
    (cache / "summary.json").write_text('{"fixture":true}')
    frames.to_parquet(cache / "frames.parquet", index=False)
    batches.to_parquet(cache / "batches.parquet", index=False)
    excluded = registry.iloc[[0, 3000, 6000, 6030]].reset_index(drop=True)
    excluded.to_parquet(response / "run/selected.parquet", index=False)
    calibration = pd.concat([pd.DataFrame({"input_sha256": c * 64, "chan": np.arange(5120), "status": 0}) for c in "abc"], ignore_index=True)
    calibration.to_parquet(response / "run/calibration.parquet", index=False)
    summary = {"registry_sha256": S.F.digest(registry), "seeds": list(S.SEEDS), "arms": list(S.ARMS), "inputs": {}, "code_sha256": {}}
    (response / "run/summary.json").write_text(json.dumps(summary))
    plan_sha = "f" * 64
    (response / "run/complete.json").write_text(json.dumps({"context": {"batch_identity": {"plan_sha256": plan_sha}}}))
    monkeypatch.setattr(S, "guard_raw_admission", lambda *a: None)
    monkeypatch.setattr(S, "excluded_controls", lambda p, b, r, e: e)
    monkeypatch.setattr(S.CR, "verify", lambda *a, **k: {"summary_sha256": S.C.sha256(cache / "summary.json")})
    monkeypatch.setattr(S.SR, "verify", lambda *a: {"fixture": True})
    monkeypatch.setattr(S.C, "admission", lambda *a: (registry, frames, batches, sources, {}, {"fixture": True}))
    def forbidden(*a, **k):
        pytest.fail("admission executed optics/response/cache")
    monkeypatch.setattr(S.I, "load_input", forbidden)
    monkeypatch.setattr(S.O, "make_batch", forbidden)
    monkeypatch.setattr(S.R, "run_arm", forbidden)
    hashes = dict(expected_plan_sha256=plan_sha, expected_cache_sha256=S.C.sha256(cache / "summary.json"),
                  expected_response_sha256=S.C.sha256(response / "run/summary.json"))
    return plan, cache, response, hashes


def test_unmocked_combined_admission_uses_all_inputs_and_whole_fresh_batches(tmp_path, monkeypatch):
    plan, cache, response, hashes = admitted_inputs(tmp_path, monkeypatch)
    d = S.admission(plan, cache, response, **hashes)
    assert len(d["registry"]) == 9000 and len(d["selected"]) <= 9 and len(d["chosen"]) == 3
    assert not d["selected"].frame_id.isin(d["excluded"].frame_id).any()
    assert set(d["selected"].input_sha256) == {c * 64 for c in "abc"}
    for p in (cache / "summary.json", response / "run/summary.json", response / "run/calibration.parquet"):
        assert d["bindings"][str(p)] == S.C.sha256(p)
    _, caps = S.layout(d)
    assert caps["children"] == 6 and caps["maximum_total"] == 392 * 1024**2


@pytest.mark.parametrize("case", ["cache_hash", "response_hash", "plan_mismatch", "registry", "seeds", "arms", "calibration_domain", "excluded_all", "symlink", "input_conflict"])
def test_combined_admission_rejects_bad_evidence(tmp_path, monkeypatch, case):
    plan, cache, response, hashes = admitted_inputs(tmp_path, monkeypatch)
    p = response / "run/summary.json"
    summary = json.loads(p.read_text())
    if case == "cache_hash":
        hashes["expected_cache_sha256"] = "0" * 64
    elif case == "response_hash":
        hashes["expected_response_sha256"] = "0" * 64
    elif case == "plan_mismatch":
        hashes["expected_plan_sha256"] = "0" * 64
    elif case == "registry":
        summary["registry_sha256"] = "0" * 64
    elif case == "seeds":
        summary["seeds"] = [1, 2]
    elif case == "arms":
        summary["arms"] = ["historical"]
    elif case == "calibration_domain":
        cp = response / "run/calibration.parquet"
        c = pd.read_parquet(cp)
        c[c.input_sha256 != "a" * 64].to_parquet(cp, index=False)
    elif case == "excluded_all":
        pd.read_parquet(cache / "frames.parquet")[list(S.F.COLUMNS)].to_parquet(response / "run/selected.parquet", index=False)
    elif case == "symlink":
        (cache / "other.txt").symlink_to(cache / "summary.json")
    else:
        summary["inputs"] = {str(cache / "summary.json"): "0" * 64}
    if case not in ("cache_hash", "response_hash"):
        p.write_text(json.dumps(summary))
        hashes["expected_response_sha256"] = S.C.sha256(p)
    with pytest.raises((ValueError, AssertionError)):
        S.admission(plan, cache, response, **hashes)


@pytest.mark.parametrize("case", ["missing", "ambiguous", "changed"])
def test_assets_fail_closed(tmp_path, case):
    p = tmp_path / "asset"
    p.write_bytes(b"valid")
    digest = S.C.sha256(p)
    bindings = {str(p): digest}
    if case == "missing":
        bindings = {}
    elif case == "ambiguous":
        bindings["other"] = digest
    else:
        p.write_bytes(b"changed")
    with pytest.raises(ValueError):
        S.one_asset(bindings, digest)


@pytest.mark.parametrize("action", [S.preflight, S.run])
@pytest.mark.parametrize("root", ["plan", "cache", "response"])
def test_output_never_modifies_prior_evidence_directories(tmp_path, monkeypatch, action, root):
    paths = {name: tmp_path / name for name in ("plan", "cache", "response")}
    for p in paths.values():
        p.mkdir()
    monkeypatch.setattr(S, "admission", lambda *a, **k: pytest.fail("admission before output containment check"))
    with pytest.raises(ValueError, match="outside"):
        action(paths["plan"], paths["cache"], paths["response"], paths[root] / "new")
    assert not any(paths[root].iterdir())


def test_header_guard_does_not_decode_raw_baskets(tmp_path, monkeypatch):
    out, d, _, _, _ = pilot_fixture(tmp_path, monkeypatch)
    import uproot
    monkeypatch.setattr(uproot.behaviors.TBranch.TBranch, "basket", lambda *a, **k: pytest.fail("decoded payload during header admission"))
    S.guard_raw_admission(d["sources"])


def test_header_guard_fails_before_legacy_payload_reader(tmp_path, monkeypatch):
    _, d, _, _, _ = pilot_fixture(tmp_path, monkeypatch)
    original = S.I.Limits
    monkeypatch.setattr(S.I, "Limits", lambda: replace(original(), max_steps=1))
    monkeypatch.setattr(S.I, "_check_tree", lambda *a, **k: pytest.fail("header loops before count cap"))
    with pytest.raises(ValueError, match="cap"):
        S.guard_raw_admission(d["sources"])


def test_three_input_out_of_order_sources_canonical_children_one_cache_each(tmp_path, monkeypatch):
    fixtures = []
    for i in range(3):
        directory = tmp_path / str(i)
        directory.mkdir()
        fixtures.append(pilot_fixture(directory, monkeypatch))
    d = fixtures[-1][1]
    allframes = pd.concat([item[1]["plan_frames"] for item in fixtures], ignore_index=True)
    registry = S.F.freeze(allframes)
    index = pd.MultiIndex.from_frame(registry[list(S.F.FRAME_KEYS)])
    allframes["frame_id"] = index.get_indexer(pd.MultiIndex.from_frame(allframes[list(S.F.FRAME_KEYS)]))
    frames, batches = S.C.PR.P.batch_plan(allframes.drop(columns="batch"), {**S.C.PR.P.DEFAULT_LIMITS, "frames": 2})
    sources = sorted([item[1]["sources"][0] for item in fixtures], key=lambda source: source["input_sha256"], reverse=True)
    cf, cb = [], []
    for source in sources:
        f, b, _ = S.C.audit_input(source, registry, frames, batches)
        cf.append(f)
        cb.append(b)
    excluded = registry[registry.event == 7]
    selected, assigned, chosen = S.schedule(registry, frames, batches, excluded)
    d.update(registry=registry, plan_frames=frames, plan_batches=batches, sources=sources,
             excluded=excluded, selected=selected, assigned=assigned, chosen=chosen,
             cache_frames=pd.concat(cf), cache_batches=pd.concat(cb),
             calibration=pd.concat([item[1]["calibration"] for item in fixtures], ignore_index=True),
             bindings=S.C.merge_bindings(*(item[1]["bindings"] for item in fixtures)))
    original_load, original_select = S.I.load_input, S.I.IndexedReplay.select
    loads, selects, active = [], [], []
    def load(*args, **kwargs):
        assert not active
        cache = original_load(*args, **kwargs)
        loads.append(str(args[0]))
        active.append(cache)
        close = cache.close
        def release():
            close()
            active.remove(cache)
        cache.close = release
        return cache
    def select(self, *args, **kwargs):
        selects.append(self.binding["input_sha256"])
        return original_select(self, *args, **kwargs)
    monkeypatch.setattr(S.I, "load_input", load)
    monkeypatch.setattr(S.I.IndexedReplay, "select", select)
    result = S.run("plan", "cache", "response", tmp_path / "multi", **fixtures[-1][3])
    assert len(loads) == len(selects) == 3 and not active
    assert result["frames"] == 6 and result["batches"] == 3
    tags, _ = S.layout(d)
    assert [r["path"] for r in result["records"]] == tags


def test_full_roundoff_census_excludes_extreme_not_in_legacy_list(tmp_path):
    plan, smoke, census = (tmp_path / name for name in ("plan", "smoke", "census"))
    for p in (plan, smoke, census):
        p.mkdir()
    registry = S.F.freeze(pd.DataFrame({"input_sha256": "a" * 64, "run": 1430, "subrun": 0,
        "event": np.array([1, 2453], np.int64), "pbtime": -224.46, "intensity": 1.}))
    frames = registry.assign(fold_start_roundoff=[0, 3])
    frames.to_parquet(census / "frames.parquet", index=False)
    artifact = dict(path="frames.parquet", rows=2, sha256=S.C.sha256(census / "frames.parquet"))
    (census / "summary.json").write_text(json.dumps({"inputs": [{"sha256": "a" * 64, "artifacts": {"frames": artifact}}]}))
    (smoke / "summary.json").write_text(json.dumps({"paths": {"audit": str(census)}}))
    (plan / "summary.json").write_text(json.dumps({"smoke": str(smoke)}))
    bindings = {str(p): S.C.sha256(p) for p in (census / "frames.parquet", census / "summary.json", smoke / "summary.json")}
    excluded = S.excluded_controls(plan, bindings, registry, registry.iloc[:1])
    assert excluded.event.tolist() == [1, 2453]
