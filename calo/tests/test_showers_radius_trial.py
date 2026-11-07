"""Frozen radius-local candidate checks, with synthetic tables and no real reconstruction."""
import builtins
import copy
import hashlib
import importlib.metadata
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from calofastsim.artsteps import onspill_timing
from calofastsim.showers import placement_diagnostic as pd
from calofastsim.showers import production_compare as pc
from calofastsim.showers import radius_trial as rt
from calofastsim.showers.export import EVENT, keys, read
from calofastsim.showers.library import save
from test_showers_placement_diagnostic import frozen
from test_showers_production_compare import case  # noqa: F401 -- shared ROOT fixture


SEED = 20261105
RECO_SEED = 20261108
NEIGHBORS = 8


def saved_clusters(case):
    saved = frozen(case)
    saved.update({"timing_" + k: v for k, v in onspill_timing(
        saved["full_ids"]["event"], **pc.TIMING).items()})
    # Match leading() while keeping the positive non-query row's fields finite.
    for arm in ("stockref", "control", *rt.ARMS):
        for i, field in enumerate(("time", "x", "y", "z")):
            saved[arm + "_" + field] = np.array([800. + i, np.nan, 900. + i])
    return saved


def prepare(case, saved=None):
    return rt.prepare(case["path"], case["library"], case["metadata"],
                      saved_clusters(case) if saved is None else saved)


@pytest.mark.parametrize("support", ["normal", "one_unsupported", "all_unsupported"])
def test_prepare_exact_baseline_full_population_and_one_candidate_draw(case, support):
    if support == "one_unsupported":
        case["library"]["entry_disk"][1] = 2
    elif support == "all_unsupported":
        case["library"]["entry_disk"][:] = 2
    saved = saved_clusters(case)
    with patch.object(pd, "sample", wraps=pd.sample) as replay, \
         patch.object(rt, "sample", wraps=rt.sample) as sample, \
         patch.object(rt, "place", wraps=rt.place) as geometric, \
         patch.object(rt, "place_crystals", wraps=rt.place_crystals) as crystal:
        result = prepare(case, saved)
    assert (rt.NEIGHBORS, rt.SEED, rt.RECO_SEED) == (NEIGHBORS, SEED, RECO_SEED)
    assert replay.call_count == sample.call_count == geometric.call_count == crystal.call_count == 1
    assert replay.call_args.kwargs == {"orientation": "disk"}
    assert sample.call_args.kwargs == {"orientation": "disk", "radius_neighbors": NEIGHBORS}
    assert geometric.call_args.args[1] is crystal.call_args.args[2] is result["draw"]
    np.testing.assert_array_equal(result["full_ids"], keys(case["data"]["events"], EVENT))
    np.testing.assert_array_equal(result["entries"]["event_row"], [0, 1])
    assert len(result["full_ids"]) == result["audit"]["census"]["events"] == 3
    for name in ("donor", "supported", "fallback", "offsets"):
        np.testing.assert_array_equal(result["baseline_draw"][name], saved["sampling_" + name])
    # All fixture bins have <= K donors: every legacy output, not just donor IDs, replays.
    for name, value in result["baseline_draw"].items():
        np.testing.assert_array_equal(result["draw"][name], value)
    for name, value in result["timing"].items():
        np.testing.assert_array_equal(value, saved["timing_" + name])
    assert not np.array_equal(result["timing"]["pbtime"], onspill_timing(np.arange(3))["pbtime"])
    fingerprints = result["audit"]["fingerprints"]
    assert fingerprints["baseline_draw"] == rt.fingerprint(result["baseline_draw"])
    assert fingerprints["candidate_draw"] == rt.fingerprint(result["draw"])
    assert fingerprints["candidate_placements"] == {k: rt.fingerprint(v) for k, v in result["placed"].items()}
    summary = result["audit"]["sampling"]["candidate"]["all"]
    assert summary["supported"] == {"normal": 2, "one_unsupported": 1, "all_unsupported": 0}[support]
    assert summary["empty"] == int(support == "normal")
    assert json.loads(json.dumps(result["audit"], allow_nan=False)) == result["audit"]


@pytest.mark.parametrize("field", EVENT)
def test_prepare_rejects_changed_full_identity_even_on_nonquery_row(case, field):
    saved = saved_clusters(case)
    saved["full_ids"] = saved["full_ids"].copy()
    saved["full_ids"][field][-1] += 1
    with patch.object(rt, "sample") as sample, pytest.raises(ValueError, match="mismatch: full_ids"):
        prepare(case, saved)
    sample.assert_not_called()


@pytest.mark.parametrize("family", ["event_row", "query_event_rows", "entry_", "geometry_", "sampling_"])
def test_prepare_rejects_same_size_stale_baseline_arrays(case, family):
    saved = saved_clusters(case)
    names = [k for k in saved if k.startswith(family)] if family.endswith("_") else [family]
    assert names
    for name in names:
        changed = saved[name].copy()
        if changed.dtype.kind == "b":
            changed.flat[0] = not changed.flat[0]
        else:
            changed.flat[0] += 1
        with patch.object(rt, "sample") as sample, \
             pytest.raises(ValueError, match="saved comparison mismatch: " + name):
            prepare(case, dict(saved, **{name: changed}))
        sample.assert_not_called()


@pytest.mark.parametrize("field", list(onspill_timing(np.array([11]))))
def test_prepare_rejects_changed_saved_timing(case, field):
    saved = saved_clusters(case)
    value = saved["timing_" + field].copy()
    value[-1] = not value[-1] if value.dtype.kind == "b" else value[-1] + 1
    saved["timing_" + field] = value
    with patch.object(rt, "sample") as sample, \
         pytest.raises(ValueError, match="mismatch: timing_" + field):
        prepare(case, saved)
    sample.assert_not_called()


@pytest.mark.parametrize("field", ["supported", "fallback", "donor"])
def test_prepare_rejects_changed_support_fallback_or_small_pool_draw(case, field):
    actual = rt.sample

    def corrupt(*args, **kwargs):
        draw = actual(*args, **kwargs)
        draw[field] = draw[field].copy()
        draw[field][0] = draw[field][0] + 1 if field == "donor" else not draw[field][0]
        assert draw["bin_pool_size"][0] <= NEIGHBORS
        return draw

    label = "small_pool_donor" if field == "donor" else "candidate_" + field
    with patch.object(rt, "sample", side_effect=corrupt), patch.object(rt, "place") as place, \
         pytest.raises(ValueError, match="mismatch: " + label):
        prepare(case)
    place.assert_not_called()


@pytest.mark.parametrize("arm", ["stockref", "control", *rt.ARMS])
@pytest.mark.parametrize("field", ["time", "x", "y", "z"])
@pytest.mark.parametrize("bad", ["short", "nan", "inf"])
def test_prepare_requires_finite_positive_cluster_fields_including_nonquery(case, arm, field, bad):
    saved = saved_clusters(case)
    name = arm + "_" + field
    if bad == "short":
        saved[name] = saved[name][:-1]
    else:
        saved[name][-1] = np.nan if bad == "nan" else np.inf
    with pytest.raises(ValueError, match="invalid saved cluster field: " + name):
        prepare(case, saved)


@pytest.mark.parametrize("arm", ["stockref", "control", *rt.ARMS])
@pytest.mark.parametrize("bad", ["short", "negative", "nan", "inf"])
def test_prepare_requires_valid_full_population_energy(case, arm, bad):
    saved = saved_clusters(case)
    name = arm + "_energy"
    if bad == "short":
        saved[name] = saved[name][:-1]
    else:
        saved[name][-1] = {"negative": -1., "nan": np.nan, "inf": np.inf}[bad]
    with pytest.raises(ValueError, match="invalid saved leading energy: " + arm):
        prepare(case, saved)


def test_sampling_summary_counts_reuse_empty_unsupported_and_empty_strata(case):
    entries = read(case["path"], input_mode="primary")[0]
    entries = {k: v[[0, 0, 0, 1, 1]].copy() for k, v in entries.items()}
    entries["disk"][-1] = 2
    entries["radius"][:] = [5., 5., 5., 500., 1200.]
    library = case["library"]
    baseline = rt.sample(library, entries, np.random.default_rng(SEED), orientation="disk")
    draw = rt.sample(library, entries, np.random.default_rng(SEED), orientation="disk", radius_neighbors=NEIGHBORS)
    placed = {"geometric": rt.place(library, draw), "crystal": rt.place_crystals(library, entries, draw)}
    result = rt.sampling_summary(library, entries, baseline, placed, draw, placed)
    for label in ("baseline", "candidate"):
        all_rows = result[label]["all"]
        assert (all_rows["entries"], all_rows["supported"], all_rows["empty"]) == (5, 4, 1)
        assert (all_rows["unique_donors"], all_rows["max_donor_reuse"]) == (2, 3)
        assert all_rows["donor_reuse_concentration"] == pytest.approx(10 / 16)
        assert all_rows["absolute_radius_mismatch_mm"]["count"] == 4
        for name, count in (("radius_550_1000", 0), ("radius_1000_inf", 1)):
            empty = result[label][name]
            assert empty["entries"] == count
            assert empty["supported"] == empty["empty"] == empty["unique_donors"] == empty["max_donor_reuse"] == 0
            assert empty["fallback"] == count
            assert empty["donor_reuse_concentration"] is None
            assert empty["absolute_radius_mismatch_mm"] == dict(count=0, mean=None, quantiles=None)
            assert empty["lost_energy_percent"] == dict(geometric=None, crystal=None)
    assert result["changed_donors"] == 0
    assert result["radius_pool_size"] == pd.summarize(draw["radius_pool_size"])
    assert result["bin_pool_size"] == pd.summarize(draw["bin_pool_size"])
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_comparison_uses_identical_six_arm_common_query_rows_only(case):
    prepared = prepare(case)
    rows = np.arange(8) * 101
    prepared["entries"] = {k: np.repeat(v[:1], len(rows), axis=0) for k, v in prepared["entries"].items()}
    prepared["entries"]["event_row"] = rows
    prepared["entries"]["disk"][:] = [0, 1] * 4
    prepared["entries"]["radius"][:] = [5., 500., 600., 1100.] * 2
    n = int(rows[-1] + 2)
    prepared["timing"] = onspill_timing(np.arange(n) + 11, **pc.TIMING)
    baseline = {arm: {field: np.full(n, 999.) for field in rt.FIELDS}
                for arm in ("stockref", "control", *rt.ARMS)}
    for clusters in baseline.values():
        for field, value in clusters.items():
            value[rows] = 20. if field == "energy" else 0.
    candidate = copy.deepcopy(baseline)
    six = [baseline["stockref"], baseline["control"], baseline["geometric"], baseline["crystal"],
           candidate["geometric"], candidate["crystal"]]
    for row, clusters in zip(rows[1:7], six):
        clusters["energy"][row] = 10.  # strict threshold; each of the six arms removes a distinct row
    candidate["stockref"] = baseline["stockref"]
    candidate["control"] = baseline["control"]
    for clusters, arm, xy in ((baseline, "geometric", (3, 4)), (baseline, "crystal", (5, 12)),
                              (candidate, "geometric", (6, 8)), (candidate, "crystal", (8, 15))):
        clusters[arm]["x"][rows], clusters[arm]["y"][rows] = xy
    with patch.object(rt, "paired_mean", wraps=rt.paired_mean) as paired:
        result = rt.comparison(prepared, baseline, candidate)
    assert result["six_arm_common_selected"] == 2
    assert paired.call_count == 4
    for call in paired.call_args_list:
        np.testing.assert_array_equal(call.args[2], rows[[0, 7]] // 100)
        assert len(call.args[0]) == 2
    for label, distance in (("baseline_geometric", 5), ("baseline_crystal", 13),
                             ("candidate_geometric", 10), ("candidate_crystal", 17)):
        assert result["six_arm_distance_mm"][label] == dict(fast=distance, stock=0., difference=distance, error=0.)
    for label in ("baseline", "candidate"):
        for arm in rt.ARMS:
            report = result[label][arm]["paired"]
            assert report["events"] == 8 and report["energy_mean"]["fast"] < 21
    for arm in rt.ARMS:
        assert result["change"][arm]["all"]["events"] == 8
        assert result["change"][arm]["disk0"]["events"] == 4
        for name in ("radius_0_450", "radius_450_550", "radius_550_1000", "radius_1000_inf"):
            assert result["change"][arm][name]["events"] == 2
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    # No common selection is also serializable and has no invented distance.
    candidate["geometric"]["energy"][rows] = 0.
    empty = rt.comparison(prepared, baseline, candidate)
    assert empty["six_arm_common_selected"] == 0
    assert all(v == dict(fast=None, stock=None, difference=None, error=None)
               for v in empty["six_arm_distance_mm"].values())


@pytest.fixture
def cli_case(case, tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    saved = saved_clusters(case)
    np.savez_compressed(baseline / "clusters.npz", **saved)
    library = tmp_path / "library.npz"
    save(library, case["library"], case["metadata"])
    reco = tmp_path / "conditions"
    reco.mkdir()
    for name in rt.CONDITIONS:
        (reco / name).write_bytes(b"synthetic calibration placeholder")
    paths = dict(tables=case["path"], library=library, clusters=baseline / "clusters.npz")
    receipt = {"inputs": {k: {"sha256": hashlib.sha256(v.read_bytes()).hexdigest()} for k, v in paths.items()}}
    previous = dict(provenance=dict(seed=str(SEED), reco_seed=str(RECO_SEED), stock_matched_phase0="True",
                                    reco_inputs=str(reco)), audit=dict(timing=dict(pc.TIMING)))
    (baseline / "numbers.json").write_text(json.dumps(previous))
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(receipt))
    out = tmp_path / "candidate"
    argv = ["radius_trial", "--tables", str(case["path"]), "--library", str(library),
            "--baseline", str(baseline), "--prior-diagnostic", str(prior), "--reco-inputs", str(reco), "--out", str(out)]
    return SimpleNamespace(paths=paths, receipt=receipt, previous=previous, reco=reco, baseline=baseline,
                           prior=prior, out=out, argv=argv, saved=saved)


def test_receipt_digest_and_equivalent_resolved_calibration_path(cli_case):
    c = cli_case
    for path in c.paths.values():
        assert rt.digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    previous = copy.deepcopy(c.previous)
    previous["provenance"]["reco_inputs"] = str(c.reco / ".." / c.reco.name)
    assert rt.verify_receipt(c.paths, c.receipt, previous, c.reco) is None


@pytest.mark.parametrize("name", ["tables", "library", "clusters"])
def test_receipt_rejects_changed_input_bytes(cli_case, name):
    c = cli_case
    with c.paths[name].open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="input hash mismatch: " + name):
        rt.verify_receipt(c.paths, c.receipt, c.previous, c.reco)


@pytest.mark.parametrize("name,value,message", [("seed", "1", "seed or timing"),
    ("reco_seed", "1", "seed or timing"), ("stock_matched_phase0", "False", "seed or timing"),
    ("reco_inputs", "/different/calibration", "calibration input directory"), ("timing", None, "timing settings")])
def test_receipt_rejects_changed_provenance(cli_case, name, value, message):
    c = cli_case
    previous = copy.deepcopy(c.previous)
    if name == "timing":
        previous["audit"]["timing"]["initial_phase_shift"] += 1
    else:
        previous["provenance"][name] = value
    with pytest.raises(ValueError, match=message):
        rt.verify_receipt(c.paths, c.receipt, previous, c.reco)


def test_fingerprint_order_layout_and_complete_payload_mutations():
    arrays = {"a": np.arange(12, dtype=np.float64).reshape(3, 4), "b": np.array([True, False])}
    expected = rt.fingerprint(arrays)
    assert expected == rt.fingerprint({"b": arrays["b"].copy(), "a": np.asfortranarray(arrays["a"])})
    assert len(expected) == 64
    mutations = [dict(arrays, a=arrays["a"].reshape(4, 3)), dict(arrays, a=arrays["a"].astype(np.float32)),
                 {"renamed": arrays["a"], "b": arrays["b"]}, {"a": arrays["a"]}]
    changed = copy.deepcopy(arrays)
    changed["a"][-1, -1] += 1
    mutations.append(changed)
    assert all(rt.fingerprint(value) != expected for value in mutations)
    structured = np.array([(1, 2)], dtype=[("run", "i8"), ("event", "i8")])
    assert rt.fingerprint({"ids": structured}) != rt.fingerprint({"ids": structured.view([("run", "i8"), ("other", "i8")])})


def test_cluster_mapping_reuses_all_five_saved_fields(cli_case):
    mapped = rt.cluster_mapping(cli_case.saved)
    assert set(mapped) == {"stockref", "control", *rt.ARMS}
    for arm, fields in mapped.items():
        assert set(fields) == set(rt.FIELDS)
        for field, value in fields.items():
            assert value is cli_case.saved[arm + "_" + field]


def test_cli_prepare_only_never_imports_or_calls_chain(cli_case, monkeypatch, capsys):
    c = cli_case
    monkeypatch.setattr(sys, "argv", c.argv + ["--prepare-only"])
    real_import = builtins.__import__

    def forbid_chain(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("calofastsim") or (globals or {}).get("__package__", "").startswith("calofastsim"):
            assert "chain" not in name.split(".") and "chain" not in (fromlist or ())
        return real_import(name, globals, locals, fromlist, level)

    before = {k: rt.digest(v) for k, v in c.paths.items()}
    with patch.object(builtins, "__import__", side_effect=forbid_chain), \
         patch.object(rt, "prepare", wraps=rt.prepare) as prepared:
        rt.main()
    prepared.assert_called_once()
    manifest = json.loads((c.out / "preparation.json").read_text())
    assert (manifest["neighbors"], manifest["seed"], manifest["reco_seed"]) == (NEIGHBORS, SEED, RECO_SEED)
    assert json.loads(capsys.readouterr().out) == manifest["audit"]
    assert manifest["calo_source_sha256"]
    for name, path in c.paths.items():
        assert manifest["inputs"][name]["sha256"] == rt.digest(path) == before[name]
    assert set(p.name for p in c.out.iterdir()) == {"preparation.json", "draw.npz"}
    with np.load(c.out / "draw.npz", allow_pickle=False) as draw:
        assert rt.fingerprint(dict(draw)) == manifest["audit"]["fingerprints"]["candidate_draw"]
        assert {"radius_pool_size", "bin_pool_size"} <= set(draw.files)


@pytest.mark.parametrize("name", ["clusters.npz", "numbers.json", "timing.json"])
def test_cli_never_overwrites_existing_run(cli_case, monkeypatch, name):
    c = cli_case
    c.out.mkdir()
    path = c.out / name
    path.write_bytes(b"existing run")
    monkeypatch.setattr(sys, "argv", c.argv + ["--prepare-only"])
    with patch.object(rt, "verify_receipt") as verify, patch.object(rt, "prepare") as prepared, \
         pytest.raises(ValueError, match="output already contains a run"):
        rt.main()
    verify.assert_not_called()
    prepared.assert_not_called()
    assert path.read_bytes() == b"existing run"


def test_cli_reconstructs_only_candidate_arms_and_reuses_saved_baseline(cli_case, monkeypatch, capsys):
    from calofastsim import chain, stockio
    from calofastsim.cluster import Geometry
    from calofastsim.showers import compare, reconstruct

    c = cli_case
    monkeypatch.setattr(sys, "argv", c.argv)
    fake_jax = SimpleNamespace(config=SimpleNamespace(update=Mock(), jax_enable_x64=True), default_backend=Mock(return_value="cpu"),
                               effects_barrier=Mock())
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "mock" if name in ("jax", "jaxlib") else version(name))
    for name in ("load_sim_crystal", "load_pulse", "load_noise", "load_propagation"):
        monkeypatch.setattr(stockio, name, Mock(return_value=object()))
    monkeypatch.setattr(Geometry, "load", Mock(return_value=object()))
    steps = Mock(side_effect=lambda handoff, sim: handoff["steps"])
    monkeypatch.setattr(reconstruct, "steps_for_reco", steps)
    seed_draws = []

    def run(placed, timing, adc, pulse, noise, geometry, rng, **kwargs):
        seed_draws.append(rng.random(4))
        assert len(timing["pbtime"]) == 3
        assert "mapped" in placed
        energy = 25. if len(seed_draws) == 1 else 30.
        return dict(clusters=dict(event=np.array([0]), energy=np.array([energy]), time=np.array([805.]),
                                  x=np.array([801.]), y=np.array([802.]), z=np.array([803.])),
                    seconds={"mock": 0.}, n_refit_skipped=0, n_nonfinite_reco=1,
                    nonfinite_examples={"t_err": [float("nan")]})

    monkeypatch.setattr(chain, "run_chain", Mock(side_effect=run))
    monkeypatch.setattr(compare, "plot", Mock())
    monkeypatch.setattr(pd, "plot_report", Mock())
    inputs_before = {k: rt.digest(v) for k, v in c.paths.items()}
    with patch.object(rt, "comparison", wraps=rt.comparison) as comparison:
        rt.main()
    assert chain.run_chain.call_count == steps.call_count == 2
    assert fake_jax.effects_barrier.call_count == 4
    assert fake_jax.config.update.call_args_list[1].args == ("jax_platform_name", "cpu")
    expected = np.random.default_rng(RECO_SEED).random(4)
    for values in seed_draws:
        np.testing.assert_array_equal(values, expected)
    _, baseline, candidate = comparison.call_args.args
    for arm in ("stockref", "control"):
        assert candidate[arm] is baseline[arm]
    with np.load(c.out / "clusters.npz", allow_pickle=False) as output:
        for name, value in c.saved.items():
            prefix = "baseline_" if name.startswith(("geometric_", "crystal_", "sampling_")) else ""
            np.testing.assert_array_equal(output[prefix + name], value)
        np.testing.assert_array_equal(output["geometric_energy"], [25., 0., 0.])
        np.testing.assert_array_equal(output["crystal_energy"], [30., 0., 0.])
        assert len(output["full_ids"]) == 3
        assert "sampling_radius_pool_size" in output and "sampling_bin_pool_size" in output
    assert {k: rt.digest(v) for k, v in c.paths.items()} == inputs_before
    result = json.loads((c.out / "numbers.json").read_text())
    assert [v["arm"] for v in result["runs"]] == list(rt.ARMS)
    assert result["candidate"]["geometric"]["paired"]["events"] == 2
    assert result["baseline"]["geometric"]["paired"]["energy_mean"]["fast"] == 10.
    assert result["candidate"]["geometric"]["paired"]["energy_mean"]["fast"] == 12.5
    assert json.loads((c.out / "summary.json").read_text()) == rt.compact(result)
    assert compare.plot.call_count == 2 and pd.plot_report.call_count == 1
    assert result["runs"][0]["nonfinite_examples"]["t_err"] == [None]
    text = capsys.readouterr().out
    assert "NaN" not in text and "Infinity" not in text
