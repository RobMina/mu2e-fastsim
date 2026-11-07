"""Bounded synthetic direction-trial lifecycle and authentication guards; no physics."""
from importlib import import_module
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from test_showers_candidate_trial import (
    arrays_equal, ct, forbid_real_physics, lifecycle, reference_case, synthetic_draw,
    write_json, write_npz,
)


@pytest.fixture
def dt():
    return import_module("calofastsim.showers.direction_trial")


@pytest.fixture
def direction(lifecycle, monkeypatch, dt):
    c = lifecycle
    c.real_authenticate = dt.authenticate
    c.real_publish = dt.publish
    c.args.report_only = False
    c.args.report_out = None
    c.args.execution_sha256 = None
    c.legacy_sample = c.sample
    c.legacy_sample.side_effect = AssertionError("frozen sampler forbidden")
    flags = dict(direction_used=np.array([True, True, False]),
                 direction_fallback=np.array([False, False, True]),
                 direction_count=np.array([8, 9, 7]))
    c.state["pools"].update(flags)
    c.legacy_sample.return_value.update({k: v.copy() for k, v in flags.items()})
    c.state.update(production=c.state["baseline"], production_diagnostics=c.state["baseline"],
                   production_draw=c.legacy_sample.return_value)
    rng = np.random.default_rng(dt.SEED)
    entries = c.state["prepared"]["entries"]
    for disk in (0, 1):
        rng.random(np.count_nonzero(c.state["pools"]["supported"] & (entries["disk"] == disk)))
    c.state["production_manifest"] = dict(draw_rng_after=rng.bit_generator.state)

    def sample(packed, queries, draw_rng):
        assert packed is c.state["library"] and queries is entries
        assert (c.args.out / "draw_started.json").is_file()
        assert json.loads((c.args.out / "draw_started.json").read_text())["retry_authorized"] is False
        for disk in (0, 1):
            draw_rng.random(np.count_nonzero(c.state["pools"]["supported"] & (queries["disk"] == disk)))
        return c.legacy_sample.return_value

    c.sample = Mock(side_effect=sample)
    c.report = Mock(return_value=({"synthetic_complete": True}, {"row": np.arange(3)},
                                 {"synthetic": True}, {"row": np.arange(3)}))
    package = import_module("calofastsim.showers")
    for name, module in (("direction_sampler", SimpleNamespace(sample=c.sample, enumerate_pools=Mock())),
                         ("direction_report", SimpleNamespace(report=c.report))):
        monkeypatch.setitem(sys.modules, "calofastsim.showers." + name, module)
        monkeypatch.setattr(package, name, module, raising=False)
    monkeypatch.setattr(package, "reconstruct", sys.modules["calofastsim.showers.reconstruct"], raising=False)
    monkeypatch.setattr(dt, "authenticate", c.authenticate)
    monkeypatch.setattr(dt, "publish", c.publish)
    return c


def snapshot(path):
    return {p.name: p.read_bytes() for p in path.iterdir()}


def test_one_direction_draw_two_reused_calls_and_durable_rng(direction, dt):
    c = direction
    assert dt.run(c.args) == {"synthetic_complete": True}
    c.sample.assert_called_once()
    c.legacy_sample.assert_not_called()
    assert c.sample.call_args.args[:2] == (c.state["library"], c.state["prepared"]["entries"])
    assert c.sample.call_args.kwargs == {}
    assert c.run_chain.call_count == c.steps.call_count == 2
    assert c.geometric.call_args.args[1] is c.crystal.call_args.args[2] is c.legacy_sample.return_value
    assert c.fake_jax.effects_barrier.call_count == 4
    for values in c.seeds:
        np.testing.assert_array_equal(values, np.random.default_rng(dt.RECO_SEED).random(4))
    manifest = json.loads((c.args.out / "preparation.json").read_text())
    assert manifest["draw_rng_before"] == np.random.default_rng(dt.SEED).bit_generator.state
    assert manifest["draw_rng_after"] == c.state["production_manifest"]["draw_rng_after"]
    assert manifest["draw_rng_before"] != manifest["draw_rng_after"]
    assert manifest["policy"] == dt.POLICY
    assert manifest["fingerprints"]["draw"] == ct.fingerprint(c.legacy_sample.return_value)
    execution = json.loads((c.args.out / "execution.json").read_text())
    assert (execution["draws"], execution["calls"]) == (1, 2)
    assert [r["arm"] for r in execution["runs"]] == list(ct.ARMS)
    for name, sha in execution["output_sha256"].items():
        assert ct.digest(c.args.out / name) == sha
    with np.load(c.args.out / "draw.npz", allow_pickle=False) as saved:
        arrays_equal(dict(saved), c.legacy_sample.return_value)
    with np.load(c.args.out / "clusters.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["full_ids"], c.state["prepared"]["full_ids"])
        for index, arm in enumerate(ct.ARMS):
            np.testing.assert_array_equal(saved[arm + "_energy"], [0., 25. + index, 0., 0.])
    c.publish.assert_called_once()


@pytest.mark.parametrize("existing", [False, True])
def test_validate_only_no_rng_sample_place_reco_or_output(direction, dt, monkeypatch, capsys, existing):
    c = direction
    c.args.validate_only = True
    if existing:
        c.args.out.mkdir()
        (c.args.out / "sentinel").write_bytes(b"preserved")
    before = snapshot(c.args.out) if existing else None
    rng = Mock(side_effect=AssertionError("RNG creation forbidden"))
    monkeypatch.setattr(dt.np.random, "default_rng", rng)
    assert dt.run(c.args) is c.state
    c.authenticate.assert_called_once_with(c.args)
    c.runtime.assert_called_once_with(c.state["prior"])
    for mock in (rng, c.sample, c.legacy_sample, c.geometric, c.crystal, c.run_chain, c.steps,
                 c.publish, c.report, *c.loaders.values()):
        mock.assert_not_called()
    assert snapshot(c.args.out) == before if existing else not c.args.out.exists()
    result = json.loads(capsys.readouterr().out)
    assert result["validated"] is True
    assert result["sampled"] is result["reconstructed"] is False
    assert (result["direction_used"], result["direction_fallback"]) == (2, 1)


@pytest.mark.parametrize("kind", ["directory", "partial", "file"])
def test_existing_output_refused_before_authentication(direction, dt, kind):
    c = direction
    if kind == "file":
        c.args.out.write_bytes(b"preserved")
    else:
        c.args.out.mkdir()
        if kind == "partial":
            (c.args.out / "draw_started.json").write_text("{}")
    with pytest.raises(ValueError, match="output exists"):
        dt.run(c.args)
    for mock in (c.authenticate, c.runtime, c.sample, c.run_chain, c.publish, *c.loaders.values()):
        mock.assert_not_called()


@pytest.mark.parametrize("fail_at", ["sample", "draw_validation", "rng_schedule", "first_call", "second_call", "publish"])
def test_failure_persists_evidence_and_never_replays(direction, dt, monkeypatch, fail_at):
    c = direction
    error = RuntimeError("synthetic failure")
    if fail_at == "sample":
        c.sample.side_effect = error
    elif fail_at == "draw_validation":
        monkeypatch.setattr(dt, "validate_draw", Mock(side_effect=error))
    elif fail_at == "rng_schedule":
        c.state["production_manifest"]["draw_rng_after"] = np.random.default_rng(dt.SEED).bit_generator.state
    elif fail_at == "publish":
        c.publish.side_effect = error
    else:
        def fail(*args, **kwargs):
            if fail_at == "first_call" or c.run_chain.call_count == 2:
                raise error
            return c.reconstruct_mock(*args, **kwargs)
        c.run_chain.side_effect = fail
    with pytest.raises(ValueError if fail_at == "rng_schedule" else RuntimeError,
                       match="uniform draw schedule" if fail_at == "rng_schedule" else "synthetic failure"):
        dt.run(c.args)
    calls = {"sample": 0, "draw_validation": 0, "rng_schedule": 0, "first_call": 1,
             "second_call": 2, "publish": 2}[fail_at]
    assert c.sample.call_count == 1 and c.run_chain.call_count == calls
    for name in ("preparation.json", "original.npz", "pools.npz", "draw_started.json"):
        assert (c.args.out / name).is_file()
    assert (c.args.out / "draw.npz").exists() == (fail_at != "sample")
    manifest = json.loads((c.args.out / "preparation.json").read_text())
    assert "draw_rng_before" in manifest and "draw_rng_after" in manifest
    assert manifest["draw_rng_before"]["bit_generator"] == "PCG64"
    if calls:
        assert (c.args.out / "geometric_started.json").is_file()
        assert (c.args.out / "crystal_placement.npz").is_file()
    else:
        c.geometric.assert_not_called()
        c.crystal.assert_not_called()
    if fail_at in ("second_call", "publish"):
        assert (c.args.out / "geometric_completed.json").is_file()
        with np.load(c.args.out / "geometric.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["energy"], [0., 25., 0., 0.])
        assert (c.args.out / "crystal_started.json").is_file()
    assert (c.args.out / "execution.json").exists() == (fail_at == "publish")
    before = snapshot(c.args.out)
    with pytest.raises(ValueError, match="output exists"):
        dt.run(c.args)
    assert c.sample.call_count == 1 and c.run_chain.call_count == calls
    assert snapshot(c.args.out) == before


def test_shared_run_calls_reentry_refused(direction, dt):
    c = direction
    dt.run(c.args)
    before = snapshot(c.args.out)
    resources = tuple(c.loaders[n].return_value for n in
                      ("load_sim_crystal", "load_pulse", "load_noise", "load_propagation", "geometry"))
    with pytest.raises(ValueError, match="already started"):
        ct.run_calls(c.args.out, c.state["prepared"], c.state["baseline"], resources)
    assert c.run_chain.call_count == 2 and c.sample.call_count == 1
    assert snapshot(c.args.out) == before


@pytest.mark.parametrize("guard", ["authenticate", "runtime", "input_mutation"])
def test_preflight_rejection_before_rng_draw_or_output(direction, dt, monkeypatch, guard):
    c = direction
    if guard == "input_mutation":
        def mutate(*args):
            c.watched.write_bytes(b"changed during resource load")
            return object()
        c.loaders["load_noise"].side_effect = mutate
    else:
        getattr(c, guard).side_effect = ValueError("synthetic preflight")
    rng = Mock(side_effect=AssertionError("RNG creation forbidden"))
    monkeypatch.setattr(dt.np.random, "default_rng", rng)
    with pytest.raises(ValueError, match="changed|preflight"):
        dt.run(c.args)
    for mock in (rng, c.sample, c.geometric, c.crystal, c.run_chain, c.publish):
        mock.assert_not_called()
    assert not c.args.out.exists()


@pytest.fixture
def policy_case():
    prepared, production, _ = reference_case(total=2813, nquery=2396)
    original = {key: np.zeros(2396, dtype=np.int64) for key in
                ("query_cell", "chosen_cell", "occupied", "library_cell", "counts", "size", "full_size", "exact_size")}
    original.update({key: np.zeros(2396, bool) for key in ("fallback", "outside", "substituted")})
    original.update(supported=np.ones(2396, bool), eligible=np.tile(np.arange(8), (2396, 1)))
    used = np.arange(2396) < 2284
    census = dict(compatible_within25_count=np.where(used, 8, 7),
                  compatible_eligible=np.tile(np.arange(8, 16), (2396, 1)),
                  event_row=prepared["entries"]["event_row"].copy(), full_ids=production["full_ids"].copy())
    pools = {key: value.copy() for key, value in original.items()}
    pools["eligible"][used] = census["compatible_eligible"][used]
    pools.update(direction_used=used, direction_fallback=~used,
                 direction_count=census["compatible_within25_count"].copy())
    return SimpleNamespace(pools=pools, original=original, census=census,
                           entries=prepared["entries"], production=production)


def test_policy_accepts_only_predeclared_2284_112_population(policy_case, dt):
    c = policy_case
    dt.validate_policy(c.pools, c.original, c.census, c.entries, c.production)
    assert c.pools["direction_used"].sum() == 2284
    assert c.pools["direction_fallback"].sum() == 112
    assert (dt.NEIGHBORS, dt.SEED, dt.RECO_SEED) == (8, 20261105, 20261108)
    assert dt.POLICY == dict(neighbors=8, angle_degrees=10., radius_mm=25., both_direction_signs=True,
                             zero_sign="nonnegative", insufficient="original_K8", orientation="disk")


@pytest.mark.parametrize("fault", ["eligible", "chosen_cell", "supported", "fallback", "outside",
                                   "direction_used", "direction_fallback", "direction_count", "rows", "full_ids", "population"])
def test_policy_rejects_changed_support_identity_or_counts(policy_case, dt, fault):
    c = policy_case
    if fault == "rows":
        c.census["event_row"][-1] += 1
    elif fault == "full_ids":
        c.census["full_ids"]["event"][-1] += 1
    elif fault == "population":
        c.census["compatible_within25_count"][2284] = 8
        c.pools["direction_count"][2284] = 8
        c.pools["direction_used"][2284] = True
        c.pools["direction_fallback"][2284] = False
        c.pools["eligible"][2284] = c.census["compatible_eligible"][2284]
    else:
        values = c.pools[fault]
        values.flat[-1] = not values.flat[-1] if values.dtype.kind == "b" else values.flat[-1] + 1
    with pytest.raises(ValueError, match="mismatch|2284/112"):
        dt.validate_policy(c.pools, c.original, c.census, c.entries, c.production)


@pytest.mark.parametrize("field", [None, "full_ids", "event_row", "query_event_rows", "entry_time",
                                    "geometry_disk_z", "timing_pbtime", "stockref_energy", "control_time"])
def test_production_reference_guard_includes_nonfront_rows(dt, field):
    prepared, baseline, _ = reference_case()
    production = {key: value.copy() for key, value in baseline.items()}
    if field is None:
        production["geometric_energy"][1] += 3.
        dt.validate_production(prepared, baseline, production)
        return
    if field == "full_ids":
        production[field]["event"][-1] += 1
    else:
        production[field][-1] += 1
    with pytest.raises(ValueError, match="mismatch"):
        dt.validate_production(prepared, baseline, production)


@pytest.fixture
def saved_bundle(direction, dt, monkeypatch):
    c = direction
    for name in ("candidate", "support", "baseline", "production", "direction_support"):
        setattr(c.args, name, c.args.out.parent / name)
    baseline_path = c.args.baseline / "clusters.npz"
    write_npz(baseline_path, c.state["baseline"])
    ct.checked_file(c.state["inputs"], baseline_path, ct.digest(baseline_path))
    library_path = c.args.candidate / "library.npz"
    library_path.parent.mkdir()
    ct.library.save(library_path, c.state["library"], {"synthetic": True})
    candidate = {"output_sha256": {"library.npz": ct.digest(library_path)}}
    support = {"output_sha256": {"candidate_pools.npz": write_npz(
        c.args.support / "candidate_pools.npz", c.state["pools"])}}
    saved = {"clusters.npz": c.state["production"], "diagnostics.npz": c.state["production_diagnostics"],
             "draw.npz": c.state["production_draw"]}
    hashes = {name: write_npz(c.args.production / name, values) for name, values in saved.items()}
    hashes["preparation.json"] = write_json(c.args.production / "preparation.json", c.state["production_manifest"])
    production = dict(completed=True, sampler_calls=1, reconstruction_calls=2, output_sha256=hashes)
    census = {"event_row": c.state["prepared"]["entries"]["event_row"]}
    direction_support = {"output_sha256": {"arrays.npz": write_npz(c.args.direction_support / "arrays.npz", census)}}
    pins = dict(dt.PINS)
    for name, value in (("candidate", candidate), ("support", support), ("production", production),
                        ("direction_support", direction_support)):
        pins[name] = write_json(getattr(c.args, name) / "summary.json", value)
    monkeypatch.setattr(dt, "PINS", pins)
    c.policy_validation = Mock()
    monkeypatch.setattr(dt, "validate_policy", c.policy_validation)
    dt.run(c.args)
    c.args.report_only = True
    c.args.execution_sha256 = ct.digest(c.args.out / "execution.json")
    c.args.report_out = c.args.out.parent / "new-report"
    for mock in (c.authenticate, c.runtime, c.sample, c.legacy_sample, c.geometric, c.crystal,
                 c.run_chain, c.steps, c.publish, c.report, *c.loaders.values()):
        mock.reset_mock()
    c.root = Mock(side_effect=AssertionError("ROOT opening forbidden"))
    monkeypatch.setattr(ct.uproot, "open", c.root)
    c.export = Mock(side_effect=AssertionError("query export forbidden"))
    monkeypatch.setattr(ct.export, "read", c.export)
    return c


def assert_no_replay(c):
    for mock in (c.authenticate, c.runtime, c.sample, c.legacy_sample, c.geometric, c.crystal,
                 c.run_chain, c.steps, c.root, c.export, *c.loaders.values()):
        mock.assert_not_called()


def test_saved_report_external_pin_new_destination_without_root_or_physics(saved_bundle, dt, monkeypatch):
    c = saved_bundle
    rng = Mock(side_effect=AssertionError("RNG forbidden"))
    monkeypatch.setattr(dt.np.random, "default_rng", rng)
    before = snapshot(c.args.out)
    assert dt.run(c.args) == {"synthetic_complete": True}
    assert_no_replay(c)
    rng.assert_not_called()
    c.policy_validation.assert_called_once()
    c.publish.assert_called_once()
    assert c.publish.call_args.args[0] == c.args.out
    assert c.publish.call_args.args[3] == c.args.report_out
    with np.load(c.args.out / "clusters.npz", allow_pickle=False) as saved:
        arrays_equal(c.publish.call_args.args[2], dict(saved))
    assert snapshot(c.args.out) == before


@pytest.mark.parametrize("fault", ["missing_pin", "wrong_pin", "draw", "placement", "clusters", "preparation",
                                   "existing_destination", "existing_inline_report", "calls", "arms", "unsafe_artifact"])
def test_saved_report_rejects_untrusted_receipts_payloads_and_destinations(saved_bundle, dt, fault):
    c = saved_bundle
    if fault == "missing_pin":
        c.args.execution_sha256 = None
    elif fault == "wrong_pin":
        c.args.execution_sha256 = "0" * 64
    elif fault == "existing_destination":
        c.args.report_out.mkdir()
        (c.args.report_out / "sentinel").write_bytes(b"preserved")
    elif fault == "existing_inline_report":
        c.args.report_out = None
        (c.args.out / "numbers.json").write_text("{}")
    elif fault in ("calls", "arms", "unsafe_artifact"):
        execution = json.loads((c.args.out / "execution.json").read_text())
        if fault == "calls":
            execution["calls"] = 1
        elif fault == "arms":
            execution["runs"].reverse()
        else:
            execution["output_sha256"]["../escape.npz"] = "0" * 64
        c.args.execution_sha256 = write_json(c.args.out / "execution.json", execution)
    else:
        name = {"draw": "draw.npz", "placement": "crystal_placement.npz", "clusters": "clusters.npz",
                "preparation": "preparation.json"}[fault]
        path = c.args.out / name
        path.write_bytes(path.read_bytes() + b"changed")
    before = snapshot(c.args.out)
    with pytest.raises(ValueError):
        dt.run(c.args)
    assert_no_replay(c)
    c.publish.assert_not_called()
    assert snapshot(c.args.out) == before
    if fault == "existing_destination":
        assert snapshot(c.args.report_out) == {"sentinel": b"preserved"}
    elif c.args.report_out is not None:
        assert not c.args.report_out.exists()


@pytest.mark.parametrize("field", ["completed", "sampler_calls", "reconstruction_calls"])
def test_reference_receipt_requires_completed_one_draw_two_calls(saved_bundle, dt, monkeypatch, field):
    c = saved_bundle
    path = c.args.production / "summary.json"
    receipt = json.loads(path.read_text())
    receipt[field] = False if field == "completed" else 0
    monkeypatch.setitem(dt.PINS, "production", write_json(path, receipt))
    with pytest.raises(ValueError, match="not completed"):
        dt.references(ct.source_overlap_report.Inputs(), c.args)
    assert_no_replay(c)


def test_saved_report_publishes_only_to_new_destination(saved_bundle, dt, monkeypatch):
    c = saved_bundle
    monkeypatch.setattr(dt, "publish", c.real_publish)
    before = snapshot(c.args.out)
    assert dt.run(c.args) == {"synthetic_complete": True}
    assert_no_replay(c)
    c.report.assert_called_once()
    assert snapshot(c.args.out) == before
    assert {p.name for p in c.args.report_out.iterdir()} == {
        "numbers.json", "diagnostics.npz", "conditional.json", "conditional_arrays.npz", "summary.json"}
    summary = json.loads((c.args.report_out / "summary.json").read_text())
    assert summary["completed"] is True
    assert (summary["sampler_calls"], summary["reconstruction_calls"]) == (1, 2)
    assert summary["policy"] == dt.POLICY
    for name, sha in summary["output_sha256"].items():
        assert ct.digest(c.args.report_out / name) == sha
    report_before = snapshot(c.args.report_out)
    with pytest.raises(ValueError, match="new report-out"):
        dt.run(c.args)
    c.report.assert_called_once()
    assert snapshot(c.args.report_out) == report_before
    assert_no_replay(c)


@pytest.mark.parametrize("field", ["full_ids", "control_energy", "entry_time"])
def test_saved_report_rejects_repinned_changed_reference_rows(saved_bundle, dt, field):
    c = saved_bundle
    path = c.args.out / "clusters.npz"
    with np.load(path, allow_pickle=False) as saved:
        output = dict(saved)
    if field == "full_ids":
        output[field]["event"][-1] += 1
    else:
        output[field][-1] += 1
    path.unlink()
    write_npz(path, output)
    execution_path = c.args.out / "execution.json"
    execution = json.loads(execution_path.read_text())
    execution["output_sha256"][path.name] = ct.digest(path)
    c.args.execution_sha256 = write_json(execution_path, execution)
    before = snapshot(c.args.out)
    with pytest.raises(ValueError, match="mismatch|reference|identity"):
        dt.run(c.args)
    c.publish.assert_not_called()
    assert_no_replay(c)
    assert snapshot(c.args.out) == before


@pytest.mark.parametrize("fault", [None, "runtime", "timing", "neighbors", "seed", "reco_seed", "diagnostic_ids"])
def test_authenticate_direction_references_before_pool_enumeration(saved_bundle, dt, monkeypatch, fault):
    c = saved_bundle
    manifest = dict(c.state["production_manifest"], runtime=c.state["prior"]["runtime"],
                    timing=dict(ct.pc.TIMING), neighbors=dt.NEIGHBORS, seed=dt.SEED, reco_seed=dt.RECO_SEED)
    saved = {"clusters.npz": c.state["production"], "diagnostics.npz": {
        key: value.copy() for key, value in c.state["production_diagnostics"].items()},
        "draw.npz": c.state["production_draw"]}
    if fault == "runtime":
        manifest["runtime"] = {"different": True}
    elif fault == "timing":
        manifest["timing"]["initial_phase_shift"] += 1
    elif fault == "diagnostic_ids":
        saved["diagnostics.npz"]["full_ids"]["event"][-1] += 1
    elif fault:
        manifest[fault] += 1
    references = Mock(return_value=(saved, manifest, {"synthetic": np.array([1])}))
    monkeypatch.setattr(dt, "references", references)
    sampler = sys.modules["calofastsim.showers.direction_sampler"]
    sampler.enumerate_pools.return_value = c.state["pools"]
    c.state["pools"]["outcome_weak"] = np.zeros(3, bool)
    monkeypatch.setattr(dt.sa, "_probabilities", Mock(return_value=np.zeros(3)))
    if fault:
        with pytest.raises(ValueError, match="differs|differ|mismatch"):
            c.real_authenticate(c.args)
        sampler.enumerate_pools.assert_not_called()
        c.policy_validation.assert_not_called()
    else:
        result = c.real_authenticate(c.args)
        assert result is c.state
        sampler.enumerate_pools.assert_called_once_with(c.state["library"], c.state["prepared"]["entries"])
        c.policy_validation.assert_called_once()
        assert all("prob_" + name in result["pools"] for name in ("empty", "weak", "energy"))
    c.sample.assert_not_called()
    c.run_chain.assert_not_called()


@pytest.mark.parametrize("fault", [None, "direction_used", "direction_fallback", "direction_count", "fallback_donor"])
def test_draw_direction_flags_and_original_k8_fallback_guard(direction, dt, fault):
    c = direction
    draw = {key: value.copy() for key, value in c.legacy_sample.return_value.items()}
    if fault == "fallback_donor":
        c.state["production_draw"] = dict(c.state["production_draw"], donor=np.array([0, 1, 1]))
    elif fault:
        values = draw[fault]
        values[-1] = not values[-1] if values.dtype.kind == "b" else values[-1] + 1
    if fault:
        with pytest.raises(ValueError, match="mismatch"):
            dt.validate_draw(c.state, draw)
    else:
        dt.validate_draw(c.state, draw)
    c.sample.assert_not_called()
    c.legacy_sample.assert_not_called()
    c.run_chain.assert_not_called()


@pytest.mark.parametrize("field", ["pins", "policy", "seed", "reco_seed"])
def test_saved_report_rejects_repinned_changed_fixed_settings(saved_bundle, dt, field):
    c = saved_bundle
    path = c.args.out / "preparation.json"
    manifest = json.loads(path.read_text())
    if field in ("pins", "policy"):
        manifest[field] = {}
    else:
        manifest[field] += 1
    sha = write_json(path, manifest)
    execution_path = c.args.out / "execution.json"
    execution = json.loads(execution_path.read_text())
    execution["output_sha256"][path.name] = sha
    c.args.execution_sha256 = write_json(execution_path, execution)
    with pytest.raises(ValueError, match="policy|settings"):
        dt.run(c.args)
    assert_no_replay(c)
    c.publish.assert_not_called()
    assert not c.args.report_out.exists()
