"""Synthetic fixed-candidate authentication and lifecycle tests; never run real physics.

The only sampler/chain invocations below are mocks.  Reference/report replay is
not part of this trial; reporting itself is tested separately.
"""
from contextlib import nullcontext
from dataclasses import asdict
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from calofastsim.showers import candidate_trial as ct


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return ct.digest(path)


def write_npz(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    ct.save_npz(path, value)
    return ct.digest(path)


def arrays_equal(actual, expected):
    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def reference_case(total=4, nquery=3):
    """Include nonfront rows, both query disks, and saved zero-energy rows."""
    ids = np.zeros(total, dtype=[(name, "i8") for name in ct.export.EVENT])
    ids["run"] = 1
    ids["event"] = np.arange(total) + 11
    entries = dict(event_row=np.arange(nquery, dtype=np.int64),
                   time=np.arange(nquery, dtype=float) + 800.,
                   disk=np.arange(nquery, dtype=np.int64) % 2)
    geometry = dict(disk_disk=np.array([0, 1]), disk_z=np.array([100., 200.]))
    timing = ct.onspill_timing(ids["event"], **ct.pc.TIMING)
    prepared = dict(entries=entries, full_ids=ids, geometry=geometry, timing=timing,
                    census=dict(events=total), original=dict(crystal=np.array([10, 20])))
    baseline = dict(event_row=np.arange(total), full_ids=ids.copy(),
                    query_event_rows=entries["event_row"].copy(), sampling_donor=np.zeros(nquery, dtype=np.int64))
    for prefix, values in (("entry_", entries), ("geometry_", geometry), ("timing_", timing)):
        baseline.update({prefix + key: value.copy() for key, value in values.items()})
    for arm in ("stockref", "control", *ct.ARMS):
        for field in ct.FIELDS:
            value = np.zeros(total)
            value[1] = 20. if field == "energy" else 810.
            if arm in ("stockref", "control"):
                value[-1] = 7. if field == "energy" else 820.
            baseline[arm + "_" + field] = value
    k8 = {("baseline_" if name.startswith(("geometric_", "crystal_", "sampling_")) else "") + name:
          value.copy() for name, value in baseline.items()}
    for arm in ct.ARMS:
        for field in ct.FIELDS:
            k8[arm + "_" + field] = baseline[arm + "_" + field].copy()
    return prepared, baseline, k8


@pytest.fixture
def synthetic_draw():
    """Empty donor, both-disk donor, and a retained zero-energy nonempty donor."""
    packed = dict(offsets=np.array([0, 0, 2, 3]), step_disk=np.array([0, 1, 1]),
                  step_crystal=np.array([10, 20, 21]), step_energy_g4=np.array([13., 22., 0.]),
                  step_energy_birks=np.array([10., 20., 0.]), step_relative_time=np.array([1., 2., 3.]),
                  step_relative_position=np.zeros((3, 3)))
    entries = reference_case()[0]["entries"]
    draw = dict(donor=np.array([0, 1, 2]), supported=np.ones(3, bool), fallback=np.array([False, False, True]),
                radius_pool_size=np.array([2, 2, 1]), bin_pool_size=np.array([3, 3, 1]),
                offsets=np.array([0, 0, 2, 3]), entry=np.array([1, 1, 2]),
                position=np.array([[1., 2., 100.], [2., 3., 200.], [3., 4., 200.]]))
    draw.update({name.removeprefix("step_"): value.copy() for name, value in packed.items()
                 if name.startswith("step_") and not name.startswith("step_relative_")})
    draw["time"] = entries["time"][draw["entry"]] + packed["step_relative_time"]
    eligible = np.full((3, 8), -1, np.int64)
    eligible[:2, :2] = [0, 1]
    eligible[2, 0] = 2
    pools = dict(eligible=eligible, event_row=entries["event_row"].copy(), supported=draw["supported"].copy(),
                 fallback=draw["fallback"].copy(), size=draw["radius_pool_size"].copy(),
                 full_size=draw["bin_pool_size"].copy(), outside=np.zeros(3, bool),
                 outcome_empty=np.array([True, False, False]), outcome_energy=np.array([0., 30., 0.]))
    return SimpleNamespace(packed=packed, entries=entries, draw=draw, pools=pools)


@pytest.fixture(autouse=True)
def forbid_real_physics(monkeypatch):
    from calofastsim import chain
    monkeypatch.setattr(ct.library, "sample", Mock(side_effect=AssertionError("real sampling forbidden")))
    monkeypatch.setattr(chain, "run_chain", Mock(side_effect=AssertionError("real reconstruction forbidden")))


def test_validate_draw_preserves_empty_weak_zero_and_whole_both_disk_response(synthetic_draw):
    c = synthetic_draw
    before = ct.fingerprint(c.draw)
    ct.validate_draw(c.packed, c.entries, c.draw, c.pools)
    assert ct.fingerprint(c.draw) == before
    np.testing.assert_array_equal(ct.pd.energy_sum(c.draw, 3), [0., 30., 0.])
    np.testing.assert_array_equal(c.draw["disk"], [0, 1, 1])
    np.testing.assert_array_equal(c.draw["offsets"], [0, 0, 2, 3])


@pytest.mark.parametrize("field", ["supported", "fallback", "radius_pool_size", "bin_pool_size", "offsets",
                                   "entry", "disk", "crystal", "energy_g4", "energy_birks", "time"])
def test_validate_draw_rejects_mutated_pool_or_whole_payload(synthetic_draw, field):
    c = synthetic_draw
    value = c.draw[field]
    value[-1] = not value[-1] if value.dtype.kind == "b" else value[-1] + 1
    with pytest.raises(ValueError, match="mismatch"):
        ct.validate_draw(c.packed, c.entries, c.draw, c.pools)


@pytest.mark.parametrize("donor", [np.array([-1, 1, 2]), np.array([0., 1., 2.]), np.array([[0, 1, 2]]),
                                   np.array([2, 1, 2]), np.array([99, 1, 2])])
def test_validate_draw_rejects_invalid_or_out_of_pool_donors(synthetic_draw, donor):
    c = synthetic_draw
    c.draw["donor"] = donor
    with pytest.raises(ValueError, match="donor"):
        ct.validate_draw(c.packed, c.entries, c.draw, c.pools)


@pytest.mark.parametrize("fault", ["position_shape", "position_nan", "empty_outcome", "energy_outcome"])
def test_validate_draw_rejects_position_or_frozen_outcome_change(synthetic_draw, fault):
    c = synthetic_draw
    if fault == "position_shape":
        c.draw["position"] = c.draw["position"][:, :2]
    elif fault == "position_nan":
        c.draw["position"][0, 0] = np.nan
    elif fault == "empty_outcome":
        c.pools["outcome_empty"][0] = False
    else:
        c.pools["outcome_energy"][1] += 1.
    with pytest.raises(ValueError):
        ct.validate_draw(c.packed, c.entries, c.draw, c.pools)


def test_validate_references_reuses_all_saved_rows_and_baseline_arrays():
    prepared, baseline, k8 = reference_case()
    ct.validate_references(prepared, baseline, k8)
    assert len(baseline["full_ids"]) > len(prepared["entries"]["event_row"])
    np.testing.assert_array_equal(baseline["geometric_energy"], [0., 20., 0., 0.])


def test_validate_references_accepts_matching_nan_coordinates_on_zero_clusters():
    prepared, baseline, k8 = reference_case()
    for saved in (baseline, k8):
        for name, value in saved.items():
            if name.endswith(("_time", "_x", "_y", "_z")) and name.rsplit("_", 1)[0] + "_energy" in saved:
                value[saved[name.rsplit("_", 1)[0] + "_energy"] == 0] = np.nan
    ct.validate_references(prepared, baseline, k8)


@pytest.mark.parametrize("which", ["baseline", "k8"])
@pytest.mark.parametrize("field", ["event_row", "full_ids", "query_event_rows", "entry_time", "geometry_disk_z",
                                   "timing_pbtime", "stockref_energy", "control_time", "geometric_energy", "sampling_donor"])
def test_validate_references_rejects_same_length_stale_arrays(which, field):
    prepared, baseline, k8 = reference_case()
    saved = baseline if which == "baseline" else k8
    if which == "k8" and field.startswith(("geometric_", "sampling_")):
        field = "baseline_" + field
    if field == "full_ids":
        saved[field]["event"][-1] += 1  # nonfront identity is part of the denominator
    else:
        saved[field][-1] += 1
    with pytest.raises(ValueError, match="mismatch"):
        ct.validate_references(prepared, baseline, k8)


@pytest.mark.parametrize("arm", ["stockref", "control", *ct.ARMS])
@pytest.mark.parametrize("fault", ["short_energy", "negative", "nan_energy", "inf_energy", "short_time", "nan_time"])
def test_validate_clusters_checks_full_population(arm, fault):
    _, baseline, _ = reference_case()
    field = "time" if fault.endswith("time") else "energy"
    value = baseline[arm + "_" + field]
    if fault.startswith("short"):
        baseline[arm + "_" + field] = value[:-1]
    else:
        baseline[arm + "_energy"][-1] = 25.
        value[-1] = {"negative": -1., "nan_energy": np.nan, "inf_energy": np.inf, "nan_time": np.nan}[fault]
    with pytest.raises(ValueError, match="invalid"):
        ct.validate_clusters(baseline, 4)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch, synthetic_draw):
    from calofastsim import chain, stockio
    from calofastsim.cluster import Geometry

    c = synthetic_draw
    prepared, baseline, k8 = reference_case()
    args = SimpleNamespace(out=tmp_path / "trial", reco_inputs=tmp_path / "conditions",
                           tables=tmp_path / "tables.bin", validate_only=False)
    watched = tmp_path / "authenticated.bin"
    watched.write_bytes(b"synthetic authenticated input")
    inputs = ct.source_overlap_report.Inputs()
    ct.checked_file(inputs, watched, ct.digest(watched))
    state = dict(inputs=inputs, library=c.packed, prepared=prepared, baseline=baseline, k8=k8,
                 pools=c.pools, prior={"runtime": {"mock_cpu": True}}, sources={"synthetic.py": "a" * 64})
    authenticate = Mock(return_value=state)
    runtime = Mock(return_value=state["prior"]["runtime"])
    monkeypatch.setattr(ct, "authenticate", authenticate)
    monkeypatch.setattr(ct, "runtime", runtime)
    fake_jax = SimpleNamespace(effects_barrier=Mock())
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    loaders = {}
    for name in ("load_sim_crystal", "load_pulse", "load_noise", "load_propagation"):
        loaders[name] = Mock(return_value=object())
        monkeypatch.setattr(stockio, name, loaders[name])
    loaders["geometry"] = Mock(return_value=object())
    monkeypatch.setattr(Geometry, "load", loaders["geometry"])
    sample = Mock(return_value=c.draw)
    monkeypatch.setattr(ct.library, "sample", sample)
    placements = {arm: dict(c.draw, mapped=np.ones(len(c.draw["entry"]), bool)) for arm in ct.ARMS}
    geometric = Mock(return_value=placements["geometric"])
    crystal = Mock(return_value=placements["crystal"])
    monkeypatch.setattr(ct, "place", geometric)
    monkeypatch.setattr(ct, "place_crystals", crystal)
    steps = Mock(side_effect=lambda handoff, sim: handoff["steps"])
    def leading_mock(clusters, total):
        result = {name: np.full(total, np.nan) for name in ct.FIELDS}
        result["energy"][:] = 0.
        for name in ct.FIELDS:
            result[name][clusters["event"]] = clusters[name]
        return result
    # Isolate optional transport imports: this suite tests driver orchestration,
    # not the already separately tested leading-cluster or transport adapters.
    monkeypatch.setitem(sys.modules, "calofastsim.showers.reconstruct",
                        SimpleNamespace(steps_for_reco=steps, leading=Mock(side_effect=leading_mock)))
    seeds, before_calls = [], []

    def reconstruct_mock(placed, timing, adc, pulse, noise, geometry, rng, **kwargs):
        index = len(seeds)
        arm = ct.ARMS[index]
        assert placed is placements[arm]
        assert timing is prepared["timing"]
        assert (args.out / (arm + "_started.json")).exists()
        assert json.loads((args.out / (arm + "_started.json")).read_text())["retry_authorized"] is False
        # Both placements and the one whole-event draw must be durable before call 1.
        for name in ("draw", "geometric_placement", "crystal_placement"):
            assert (args.out / (name + ".npz")).is_file()
        if index:
            assert (args.out / "geometric.npz").is_file()
            assert (args.out / "geometric_completed.json").is_file()
        before_calls.append({p.name for p in args.out.iterdir()})
        seeds.append(rng.random(4))
        return dict(clusters=dict(event=np.array([1]), energy=np.array([25. + index]),
                                  time=np.array([805.]), x=np.array([1.]), y=np.array([2.]), z=np.array([3.])),
                    seconds={"mock": 0.}, n_refit_skipped=0, n_nonfinite_reco=0)

    run_chain = Mock(side_effect=reconstruct_mock)
    monkeypatch.setattr(chain, "run_chain", run_chain)
    publish = Mock(return_value={"synthetic_complete": True})
    monkeypatch.setattr(ct, "publish", publish)
    return SimpleNamespace(args=args, state=state, sample=sample, geometric=geometric, crystal=crystal,
                           steps=steps, run_chain=run_chain, fake_jax=fake_jax, authenticate=authenticate,
                           runtime=runtime, publish=publish, loaders=loaders, seeds=seeds, watched=watched,
                           placements=placements, before_calls=before_calls, reconstruct_mock=reconstruct_mock)


def test_run_exactly_one_mock_draw_two_mock_cpu_calls_and_full_row_persistence(lifecycle):
    c = lifecycle
    assert ct.run(c.args) == {"synthetic_complete": True}
    assert (ct.NEIGHBORS, ct.SEED, ct.RECO_SEED) == (8, 20261105, 20261108)
    c.sample.assert_called_once()
    assert c.sample.call_args.args[:2] == (c.state["library"], c.state["prepared"]["entries"])
    assert c.sample.call_args.kwargs == {"orientation": "disk", "radius_neighbors": 8}
    assert c.geometric.call_args.args[1] is c.crystal.call_args.args[2] is c.sample.return_value
    assert c.run_chain.call_count == c.steps.call_count == 2
    assert c.fake_jax.effects_barrier.call_count == 4
    for values in c.seeds:
        np.testing.assert_array_equal(values, np.random.default_rng(20261108).random(4))
    for call in c.run_chain.call_args_list:
        assert call.kwargs["table"] is c.loaders["load_propagation"].return_value
    c.publish.assert_called_once()
    with np.load(c.args.out / "draw.npz", allow_pickle=False) as saved:
        arrays_equal(dict(saved), c.sample.return_value)
        np.testing.assert_array_equal(saved["disk"], [0, 1, 1])
        np.testing.assert_array_equal(saved["offsets"], [0, 0, 2, 3])
    with np.load(c.args.out / "clusters.npz", allow_pickle=False) as saved:
        for name, value in c.state["baseline"].items():
            if not name.startswith(("geometric_", "crystal_", "sampling_")):
                np.testing.assert_array_equal(saved[name], value, err_msg=name)
        for index, arm in enumerate(ct.ARMS):
            np.testing.assert_array_equal(saved[arm + "_energy"], [0., 25. + index, 0., 0.])
            assert np.isnan(saved[arm + "_time"][[0, 2, 3]]).all()
            with np.load(c.args.out / (arm + ".npz"), allow_pickle=False) as per_arm:
                for field in ct.FIELDS:
                    np.testing.assert_array_equal(per_arm[field], saved[arm + "_" + field])
    manifest = json.loads((c.args.out / "preparation.json").read_text())
    assert manifest["draw_rng_before"] == np.random.default_rng(20261105).bit_generator.state
    assert manifest["fingerprints"]["draw"] == ct.fingerprint(c.sample.return_value)
    assert manifest["fingerprints"]["placements"] == {arm: ct.fingerprint(value) for arm, value in c.placements.items()}
    execution = json.loads((c.args.out / "execution.json").read_text())
    assert (execution["draws"], execution["calls"]) == (1, 2)
    assert [run["arm"] for run in execution["runs"]] == list(ct.ARMS)
    for name, sha in execution["output_sha256"].items():
        assert ct.digest(c.args.out / name) == sha


@pytest.mark.parametrize("existing", [False, True])
def test_validate_only_no_sample_reconstruction_loaders_or_output(lifecycle, capsys, existing):
    c = lifecycle
    c.args.validate_only = True
    if existing:
        c.args.out.mkdir()
        (c.args.out / "sentinel").write_bytes(b"never overwrite")
    before = {p.name: p.read_bytes() for p in c.args.out.iterdir()} if existing else None
    assert ct.run(c.args) is c.state
    c.authenticate.assert_called_once_with(c.args)
    c.runtime.assert_called_once_with(c.state["prior"])
    for mock in (c.sample, c.run_chain, c.geometric, c.crystal, c.steps, c.publish, *c.loaders.values()):
        mock.assert_not_called()
    if existing:
        assert {p.name: p.read_bytes() for p in c.args.out.iterdir()} == before
    else:
        assert not c.args.out.exists()
    result = json.loads(capsys.readouterr().out)
    assert result["validated"] is True
    assert result["sampled"] is result["reconstructed"] is False


@pytest.mark.parametrize("kind", ["empty_directory", "partial_run", "file"])
def test_run_existing_output_refused_before_authentication_or_draw(lifecycle, kind):
    c = lifecycle
    if kind == "file":
        c.args.out.write_bytes(b"existing")
    else:
        c.args.out.mkdir()
        if kind == "partial_run":
            (c.args.out / "geometric_started.json").write_text("{}")
    with pytest.raises(ValueError, match="output exists"):
        ct.run(c.args)
    for mock in (c.authenticate, c.runtime, c.sample, c.run_chain, c.publish, *c.loaders.values()):
        mock.assert_not_called()


@pytest.mark.parametrize("fail_at", ["sample", "validate_draw", "first_call", "second_call", "publish"])
def test_failed_trial_durable_evidence_and_no_automatic_or_explicit_retry(lifecycle, monkeypatch, fail_at):
    c = lifecycle
    error = RuntimeError("synthetic failure: " + fail_at)
    if fail_at == "sample":
        c.sample.side_effect = error
    elif fail_at == "validate_draw":
        monkeypatch.setattr(ct, "validate_draw", Mock(side_effect=error))
    elif fail_at == "publish":
        c.publish.side_effect = error
    else:
        def fail(*args, **kwargs):
            if fail_at == "first_call" or c.run_chain.call_count == 2:
                raise error
            return c.reconstruct_mock(*args, **kwargs)
        c.run_chain.side_effect = fail
    with pytest.raises(RuntimeError, match="synthetic failure"):
        ct.run(c.args)
    assert c.sample.call_count == 1
    expected_calls = {"sample": 0, "validate_draw": 0, "first_call": 1, "second_call": 2, "publish": 2}[fail_at]
    assert c.run_chain.call_count == expected_calls
    assert (c.args.out / "preparation.json").exists()
    assert (c.args.out / "draw.npz").exists() == (fail_at != "sample")
    if expected_calls:
        assert (c.args.out / "geometric_started.json").exists()
        assert (c.args.out / "geometric_placement.npz").exists()
        assert (c.args.out / "crystal_placement.npz").exists()
    if fail_at in ("second_call", "publish"):
        with np.load(c.args.out / "geometric.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["energy"], [0., 25., 0., 0.])
        assert (c.args.out / "geometric_completed.json").exists()
        assert (c.args.out / "crystal_started.json").exists()
        assert (c.args.out / "crystal.npz").exists() == (fail_at == "publish")
    before = {p.name: p.read_bytes() for p in c.args.out.iterdir()}
    with pytest.raises(ValueError, match="output exists"):
        ct.run(c.args)
    assert c.sample.call_count == 1 and c.run_chain.call_count == expected_calls
    assert {p.name: p.read_bytes() for p in c.args.out.iterdir()} == before


def test_run_calls_started_marker_forbids_reentry_without_another_chain_call(lifecycle):
    c = lifecycle
    ct.run(c.args)
    before = {p.name: p.read_bytes() for p in c.args.out.iterdir()}
    resources = tuple(c.loaders[name].return_value for name in
                      ("load_sim_crystal", "load_pulse", "load_noise", "load_propagation", "geometry"))
    with pytest.raises(ValueError, match="already started"):
        ct.run_calls(c.args.out, c.state["prepared"], c.state["baseline"], resources)
    assert c.run_chain.call_count == 2
    assert {p.name: p.read_bytes() for p in c.args.out.iterdir()} == before


def test_input_mutation_during_resource_load_prevents_draw_and_output(lifecycle):
    c = lifecycle
    def mutate(*args):
        c.watched.write_bytes(b"changed after authentication")
        return object()
    c.loaders["load_noise"].side_effect = mutate
    with pytest.raises(ValueError, match="changed"):
        ct.run(c.args)
    c.sample.assert_not_called()
    c.run_chain.assert_not_called()
    assert not c.args.out.exists()


def test_save_npz_exclusive_creation_never_overwrites(tmp_path):
    path = tmp_path / "saved.npz"
    ct.save_npz(path, {"energy": np.array([0., 20., 0.])})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        ct.save_npz(path, {"energy": np.array([999.])})
    assert path.read_bytes() == before


@pytest.fixture
def authenticated_bundle(tmp_path, monkeypatch):
    """Generate numeric synthetic arrays; uproot/export never open physics data."""
    prepared, baseline, k8 = reference_case(total=2813, nquery=2396)
    args = SimpleNamespace(**{name: tmp_path / name for name in ("candidate", "support", "baseline", "k8", "reco_inputs")},
                           tables=tmp_path / "synthetic-tables.bin", out=tmp_path / "unused", validate_only=True)
    for name in ("candidate", "support", "baseline", "k8", "reco_inputs"):
        getattr(args, name).mkdir()
    args.tables.write_bytes(b"synthetic query placeholder, not ROOT")
    metadata = {"synthetic": True}
    packed = {"offsets": np.array([0, 0, 2]), "step_energy_birks": np.array([10., 20.])}
    ct.library.save(args.candidate / "library.npz", packed, metadata)
    outputs = {"library.npz": ct.digest(args.candidate / "library.npz"),
               "metadata.json": write_json(args.candidate / "metadata.json", metadata)}
    candidate = dict(output_sha256=outputs)
    pools = dict(event_row=prepared["entries"]["event_row"].copy(), supported=np.ones(2396, bool),
                 outside=np.zeros(2396, bool))
    support = dict(sampling_performed=False, reconstruction_performed=False,
                   candidate_query_overlap=dict(shared_parent_events=0, shared_output_events=0),
                   output_sha256={"candidate_pools.npz": write_npz(args.support / "candidate_pools.npz", pools)},
                   input_sha256={"query_tables": ct.digest(args.tables)}, source_sha256={})
    prior = dict(neighbors=ct.NEIGHBORS, seed=ct.SEED, reco_seed=ct.RECO_SEED,
                 audit=dict(timing=dict(ct.pc.TIMING)), calo_source_sha256={},
                 inputs={"clusters": {"sha256": write_npz(args.baseline / "clusters.npz", baseline)},
                         "baseline_report": {"sha256": write_json(args.baseline / "numbers.json",
                                                                  {"audit": {"timing": dict(ct.pc.TIMING)}})}})
    for name in ct.CONDITIONS:
        path = args.reco_inputs / name
        path.write_bytes(("synthetic condition " + name).encode())
        prior["inputs"][name] = {"sha256": ct.digest(path)}
    pins = dict(candidate=write_json(args.candidate / "summary.json", candidate),
                support=write_json(args.support / "summary.json", support),
                k8_manifest=write_json(args.k8 / "preparation.json", prior),
                k8_numbers=write_json(args.k8 / "numbers.json", {"manifest": prior}),
                k8_clusters=write_npz(args.k8 / "clusters.npz", k8))
    monkeypatch.setattr(ct, "PINS", pins)
    sources = Mock(return_value={"synthetic.py": "a" * 64})
    monkeypatch.setattr(ct, "validate_sources", sources)
    reader = Mock(return_value=(prepared["entries"], prepared["original"], prepared["geometry"], prepared["census"], np.zeros(2813)))
    monkeypatch.setattr(ct.export, "read", reader)
    monkeypatch.setattr(ct.uproot, "open", Mock(side_effect=lambda path: nullcontext("synthetic")))
    monkeypatch.setattr(ct.export, "columns", Mock(return_value={key: prepared["full_ids"][key] for key in ct.export.EVENT}))
    parity = Mock()
    monkeypatch.setattr(ct.auth, "geometry_parity", parity)
    original_place = Mock(return_value=dict(mapped=np.ones(2, bool), crystal=prepared["original"]["crystal"].copy()))
    monkeypatch.setattr(ct, "place", original_place)
    return SimpleNamespace(args=args, pins=pins, candidate=candidate, support=support, prior=prior,
                           prepared=prepared, baseline=baseline, k8=k8, pools=pools, sources=sources,
                           reader=reader, parity=parity, original_place=original_place)


def test_authenticate_pins_all_synthetic_inputs_and_rechecks_mutation(authenticated_bundle):
    c = authenticated_bundle
    result = ct.authenticate(c.args)
    c.reader.assert_called_once_with(c.args.tables, input_mode="primary")
    c.parity.assert_called_once()
    c.sources.assert_called_once()
    ct.library.sample.assert_not_called()
    assert not c.args.out.exists()
    assert len(result["prepared"]["full_ids"]) == 2813
    assert len(result["prepared"]["entries"]["event_row"]) == 2396
    for name in ct.CONDITIONS:
        assert result["inputs"].sha256[str(c.args.reco_inputs / name)] == c.prior["inputs"][name]["sha256"]
    c.args.tables.write_bytes(b"changed after authentication")
    with pytest.raises(ValueError, match="changed"):
        result["inputs"].unchanged()


@pytest.mark.parametrize("name", ["candidate/summary.json", "candidate/library.npz", "candidate/metadata.json",
    "support/summary.json", "support/candidate_pools.npz", "baseline/clusters.npz", "baseline/numbers.json",
    "k8/preparation.json", "k8/numbers.json", "k8/clusters.npz", "tables", *["reco_inputs/" + n for n in ct.CONDITIONS]])
def test_authenticate_rejects_any_changed_pinned_artifact_before_export(authenticated_bundle, name):
    c = authenticated_bundle
    group, _, leaf = name.partition("/")
    path = getattr(c.args, group) / leaf if leaf else getattr(c.args, group)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        ct.authenticate(c.args)
    c.reader.assert_not_called()
    ct.library.sample.assert_not_called()
    assert not c.args.out.exists()


@pytest.mark.parametrize("fault", ["sampled", "reconstructed", "parent_overlap", "output_overlap", "unsafe_artifact",
                                   "neighbors", "seed", "reco_seed", "timing", "manifest"])
def test_authenticate_rejects_semantically_invalid_but_repinned_receipt(authenticated_bundle, fault):
    c = authenticated_bundle
    if fault in ("sampled", "reconstructed", "parent_overlap", "output_overlap"):
        if fault in ("sampled", "reconstructed"):
            c.support[{"sampled": "sampling_performed", "reconstructed": "reconstruction_performed"}[fault]] = True
        else:
            c.support["candidate_query_overlap"]["shared_" + ("parent" if fault == "parent_overlap" else "output") + "_events"] = 1
        c.pins["support"] = write_json(c.args.support / "summary.json", c.support)
    elif fault == "unsafe_artifact":
        c.candidate["output_sha256"]["../escape.npz"] = "a" * 64
        c.pins["candidate"] = write_json(c.args.candidate / "summary.json", c.candidate)
    else:
        if fault == "timing":
            changed = dict(ct.pc.TIMING)
            changed["initial_phase_shift"] += 1
            c.prior["inputs"]["baseline_report"]["sha256"] = write_json(
                c.args.baseline / "numbers.json", {"audit": {"timing": changed}})
        elif fault != "manifest":
            c.prior[fault] += 1
        c.pins["k8_manifest"] = write_json(c.args.k8 / "preparation.json", c.prior)
        c.pins["k8_numbers"] = write_json(c.args.k8 / "numbers.json", {"manifest": {} if fault == "manifest" else c.prior})
    with pytest.raises(ValueError):
        ct.authenticate(c.args)
    c.reader.assert_not_called()
    ct.library.sample.assert_not_called()


@pytest.mark.parametrize("fault", ["total", "queries", "unsupported", "outside", "pool_rows", "unmapped", "crystal"])
def test_authenticate_population_and_original_css_guards(authenticated_bundle, fault):
    c = authenticated_bundle
    if fault == "total":
        c.reader.return_value = (*c.reader.return_value[:-1], np.zeros(2812))
    elif fault == "queries":
        # The complete saved-entry identity check must also reject a lost query.
        for field in c.prepared["entries"]:
            c.prepared["entries"][field] = c.prepared["entries"][field][:-1]
        c.reader.return_value = (c.prepared["entries"], *c.reader.return_value[1:])
    elif fault in ("unsupported", "outside", "pool_rows"):
        key = {"unsupported": "supported", "outside": "outside", "pool_rows": "event_row"}[fault]
        c.pools[key][0] = False if fault == "unsupported" else True if fault == "outside" else 99
        path = c.args.support / "candidate_pools.npz"
        path.unlink()
        c.support["output_sha256"][path.name] = write_npz(path, c.pools)
        c.pins["support"] = write_json(c.args.support / "summary.json", c.support)
    elif fault == "unmapped":
        c.original_place.return_value["mapped"][0] = False
    else:
        c.original_place.return_value["crystal"][0] += 1
    with pytest.raises(ValueError):
        ct.authenticate(c.args)
    ct.library.sample.assert_not_called()


def test_authenticate_detects_mutation_during_validation(authenticated_bundle):
    c = authenticated_bundle
    def change(*args):
        c.args.tables.write_bytes(b"changed while validating geometry")
    c.parity.side_effect = change
    with pytest.raises(ValueError, match="changed"):
        ct.authenticate(c.args)
    ct.library.sample.assert_not_called()
    assert not c.args.out.exists()


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "package"
    fake_driver = root / "showers" / "candidate_trial.py"
    names = ["chain.py", "templatefit.py"] + ["showers/" + name + ".py" for name in
             ("library", "response", "crystal_placement", "reconstruct", "production_compare", "placement_diagnostic",
              "compare", "validity_report", "candidate_support", "source_events")]
    for name in names + ["showers/candidate_trial.py"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source " + name + "\n")
    monkeypatch.setattr(ct, "__file__", str(fake_driver))
    prior = {"calo_source_sha256": {name: ct.digest(root / name) for name in names[:10]}}
    support = {"source_sha256": {"__main__": ct.digest(root / "showers/candidate_support.py"),
                                "calofastsim.showers.source_events": ct.digest(root / "showers/source_events.py")}}
    return SimpleNamespace(root=root, prior=prior, support=support)


def test_validate_sources_authenticates_prior_core_and_support_code(source_tree):
    c = source_tree
    inputs = ct.source_overlap_report.Inputs()
    result = ct.validate_sources(inputs, c.prior, c.support)
    assert set(result) == {str(path.relative_to(c.root)) for path in c.root.rglob("*.py")}
    assert inputs.sha256 == {str(c.root / name): sha for name, sha in result.items()}
    (c.root / "chain.py").write_text("# changed after verification\n")
    with pytest.raises(ValueError, match="changed"):
        inputs.unchanged()


@pytest.mark.parametrize("name", ["chain.py", "templatefit.py", "showers/library.py", "showers/response.py",
    "showers/crystal_placement.py", "showers/reconstruct.py", "showers/production_compare.py",
    "showers/placement_diagnostic.py", "showers/compare.py", "showers/validity_report.py",
    "showers/candidate_support.py", "showers/source_events.py"])
def test_validate_sources_rejects_any_changed_critical_source(source_tree, name):
    c = source_tree
    (c.root / name).write_text("# unexpected source change\n")
    with pytest.raises(ValueError, match="input hash mismatch"):
        ct.validate_sources(ct.source_overlap_report.Inputs(), c.prior, c.support)


def test_checked_file_rejects_conflicting_anchor(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"synthetic")
    inputs = ct.source_overlap_report.Inputs()
    inputs.sha256[str(path)] = "0" * 64
    with pytest.raises(ValueError, match="conflicting input anchors"):
        ct.checked_file(inputs, path, ct.digest(path))


@pytest.mark.parametrize("fault", [None, "backend", "x64", "versions", "fit_config", "iterations"])
def test_runtime_cpu_x64_versions_and_native_15_iteration_guard(monkeypatch, fault):
    from calofastsim import templatefit
    config = asdict(templatefit.FitConfig())
    prior = dict(runtime=dict(backend="cpu", x64=True,
                 versions={name: "synthetic" for name in ("numpy", "jax", "jaxlib", "scipy", "uproot")}, fit_config=config))
    jax = SimpleNamespace(config=SimpleNamespace(update=Mock(), jax_enable_x64=True), default_backend=Mock(return_value="cpu"))
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setattr(ct.importlib.metadata, "version", Mock(return_value="synthetic"))
    if fault == "backend":
        jax.default_backend.return_value = "gpu"
    elif fault == "x64":
        jax.config.jax_enable_x64 = False
    elif fault == "versions":
        prior["runtime"]["versions"]["numpy"] = "other"
    elif fault == "fit_config":
        prior["runtime"]["fit_config"]["iterations"] = 14
    elif fault == "iterations":
        altered = templatefit.FitConfig(iterations=14)
        monkeypatch.setattr(templatefit, "FitConfig", Mock(return_value=altered))
        prior["runtime"]["fit_config"] = asdict(altered)
    if fault:
        with pytest.raises(ValueError, match="runtime differs"):
            ct.runtime(prior)
    else:
        assert ct.runtime(prior) == prior["runtime"]
    assert [call.args for call in jax.config.update.call_args_list] == [("jax_enable_x64", True), ("jax_platform_name", "cpu")]


@pytest.mark.parametrize("guard", ["authenticate", "runtime"])
def test_failed_preflight_never_draws_loads_reconstructs_or_creates_output(lifecycle, guard):
    c = lifecycle
    getattr(c, guard).side_effect = ValueError("synthetic preflight rejection")
    with pytest.raises(ValueError, match="preflight"):
        ct.run(c.args)
    for mock in (c.sample, c.run_chain, c.publish, *c.loaders.values()):
        mock.assert_not_called()
    assert not c.args.out.exists()


@pytest.mark.parametrize("arm_index", [0, 1])
def test_invalid_returned_arm_is_persisted_before_validation_and_never_retried(lifecycle, arm_index):
    c = lifecycle
    def invalid(*args, **kwargs):
        result = c.reconstruct_mock(*args, **kwargs)
        if c.run_chain.call_count == arm_index + 1:
            result["clusters"]["time"][:] = np.nan
        return result
    c.run_chain.side_effect = invalid
    with pytest.raises(ValueError, match="invalid positive-cluster coordinate"):
        ct.run(c.args)
    assert c.sample.call_count == 1
    assert c.run_chain.call_count == arm_index + 1
    for index in range(arm_index + 1):
        arm = ct.ARMS[index]
        assert (c.args.out / (arm + "_started.json")).exists()
        with np.load(c.args.out / (arm + ".npz"), allow_pickle=False) as saved:
            assert np.isnan(saved["time"][1]) == (index == arm_index)
    c.publish.assert_not_called()
    assert not (c.args.out / "clusters.npz").exists()
    with pytest.raises(ValueError, match="output exists"):
        ct.run(c.args)
    assert c.run_chain.call_count == arm_index + 1


@pytest.mark.parametrize("reserved", ["clusters.npz", *[arm + suffix for arm in ct.ARMS
                                                     for suffix in ("_started.json", "_completed.json", ".npz")]])
def test_run_calls_refuses_any_prior_arm_evidence_before_first_call(lifecycle, reserved):
    c = lifecycle
    c.args.out.mkdir()
    path = c.args.out / reserved
    path.write_bytes(b"existing evidence")
    resources = tuple(c.loaders[name].return_value for name in
                      ("load_sim_crystal", "load_pulse", "load_noise", "load_propagation", "geometry"))
    with pytest.raises(ValueError, match="already started"):
        ct.run_calls(c.args.out, c.state["prepared"], c.state["baseline"], resources)
    c.run_chain.assert_not_called()
    c.steps.assert_not_called()
    c.sample.assert_not_called()
    assert path.read_bytes() == b"existing evidence"
    assert list(c.args.out.iterdir()) == [path]


def test_saved_only_recovery_uses_durable_receipt_without_more_draws_or_calls(lifecycle):
    c = lifecycle
    ct.run(c.args)
    c.args.report_only = True
    c.args.execution_sha256 = ct.digest(c.args.out / "execution.json")
    c.args.report_out = None
    for mock in (c.sample, c.run_chain, c.runtime, c.geometric, c.crystal, c.steps, c.publish, *c.loaders.values()):
        mock.reset_mock()
    before = {p.name: p.read_bytes() for p in c.args.out.iterdir()}
    assert ct.run(c.args) == {"synthetic_complete": True}
    c.publish.assert_called_once()
    for mock in (c.sample, c.run_chain, c.runtime, c.geometric, c.crystal, c.steps, *c.loaders.values()):
        mock.assert_not_called()
    assert {p.name: p.read_bytes() for p in c.args.out.iterdir()} == before
    with np.load(c.args.out / "clusters.npz", allow_pickle=False) as saved:
        arrays_equal(c.publish.call_args.args[2], dict(saved))


@pytest.mark.parametrize("fault", ["missing_pin", "wrong_pin", "changed_draw", "changed_placement", "changed_clusters"])
def test_saved_only_recovery_rejects_untrusted_receipt_or_payload_without_replay(lifecycle, fault):
    c = lifecycle
    ct.run(c.args)
    c.args.report_only = True
    c.args.execution_sha256 = ct.digest(c.args.out / "execution.json")
    c.args.report_out = None
    if fault == "missing_pin":
        c.args.execution_sha256 = None
    elif fault == "wrong_pin":
        c.args.execution_sha256 = "0" * 64
    else:
        name = {"changed_draw": "draw.npz", "changed_placement": "crystal_placement.npz",
                "changed_clusters": "clusters.npz"}[fault]
        path = c.args.out / name
        path.write_bytes(path.read_bytes() + b"changed")
    for mock in (c.sample, c.run_chain, c.runtime, c.publish, *c.loaders.values()):
        mock.reset_mock()
    before = {p.name: p.read_bytes() for p in c.args.out.iterdir()}
    with pytest.raises(ValueError):
        ct.run(c.args)
    for mock in (c.sample, c.run_chain, c.runtime, c.publish, *c.loaders.values()):
        mock.assert_not_called()
    assert {p.name: p.read_bytes() for p in c.args.out.iterdir()} == before
