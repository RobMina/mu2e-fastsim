"""Synthetic saved-K8 residual diagnostics: no new sampling or reconstruction."""
import builtins
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from calofastsim.readout import ReadoutConfig
from calofastsim.showers import compare, library as shower_library
from calofastsim.showers import placement_diagnostic as pd
from calofastsim.showers import production_compare as pc
from calofastsim.showers import radius_trial as rt
from calofastsim.showers import residual_diagnostic as rd
from calofastsim.showers.export import EVENT
from test_showers_production_compare import case  # noqa: F401 -- shared synthetic ROOT fixture
from test_showers_radius_trial import saved_clusters


@pytest.fixture
def saved_trial(case, request):
    """Only fixture construction may sample; all reconstructed values are synthetic."""
    support = getattr(request, "param", "normal")
    if support == "one_unsupported":
        case["library"]["entry_disk"][1] = 2
    elif support == "all_unsupported":
        case["library"]["entry_disk"][:] = 2
    elif support == "zero_nonempty":
        case["library"]["step_energy_birks"][:] = 0
    baseline = saved_clusters(case)
    trial = rt.prepare(case["path"], case["library"], case["metadata"], baseline)
    saved = {k: v.copy() for k, v in baseline.items()
             if not k.startswith(("geometric_", "crystal_", "sampling_"))}
    saved.update({"baseline_" + k: v.copy() for k, v in baseline.items()
                  if k.startswith(("geometric_", "crystal_", "sampling_"))})
    saved.update({"sampling_" + k: trial["draw"][k].copy() for k in
                  ("donor", "supported", "fallback", "offsets", "radius_pool_size", "bin_pool_size")})
    saved.update({arm + "_" + k: baseline[arm + "_" + k].copy()
                  for arm in rt.ARMS for k in rt.FIELDS})
    manifest = dict(neighbors=rt.NEIGHBORS, seed=rt.SEED, reco_seed=rt.RECO_SEED,
                    audit=copy.deepcopy(trial["audit"]))
    _, css = pd.diagnose(case["library"], trial["entries"], trial["original"],
                         trial["draw"], trial["placed"], saved)
    return dict(case=case, saved=saved, baseline=baseline, draw=trial["draw"],
                manifest=manifest, css=css)


@pytest.fixture
def saved_only(saved_trial, monkeypatch):
    """Forbid both imported sampling aliases and late reconstruction imports."""
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-only diagnostic attempted sampling/reconstruction")

    for module in (shower_library, pd, pc, rt, compare):
        monkeypatch.setattr(module, "sample", forbidden)
    # Also catches a new direct alias added to the diagnostic in a future edit.
    monkeypatch.setattr(rd, "sample", forbidden, raising=False)
    actual_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        forbidden_names = {"chain", "reconstruct", "reconstruction"}
        assert not (forbidden_names & set(name.split("."))), "reconstruction import"
        if name.startswith("calofastsim") or (globals or {}).get("__package__", "").startswith("calofastsim"):
            assert not (forbidden_names & set(fromlist or ())), "reconstruction import"
        return actual_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return saved_trial


def prepare(t):
    c = t["case"]
    return rd.prepare(c["path"], c["library"], c["metadata"], t["saved"],
                      t["draw"], t["manifest"], t["baseline"])


def diagnose(t, prepared=None):
    return rd.diagnose(t["case"]["library"], prepare(t) if prepared is None else prepared,
                       t["saved"], t["css"])


def changed(value):
    result = value.copy()
    if result.dtype.kind == "b":
        result.flat[0] = not result.flat[0]
    else:
        result.flat[0] += 1
    return result


def test_saved_only_preserves_full_population_and_same_shared_draw(saved_only):
    t = saved_only
    before = rt.fingerprint(t["saved"])
    with patch.object(rd, "place", wraps=rd.place) as geometric, \
         patch.object(rd, "place_crystals", wraps=rd.place_crystals) as crystal:
        prepared = prepare(t)
        result, arrays = diagnose(t, prepared)
    assert geometric.call_count == 2 and crystal.call_count == 1
    assert geometric.call_args.args[1] is crystal.call_args.args[2] is t["draw"]
    assert prepared["draw"] is t["draw"]
    assert rt.fingerprint(t["saved"]) == before
    np.testing.assert_array_equal(t["saved"]["event_row"], [0, 1, 2])
    np.testing.assert_array_equal(arrays["event_row"], [0, 1])
    assert result["census"]["events"] == 3
    assert result["census"]["non_front_entry_positive_events"] == 1
    assert result["regions"]["all"]["all"]["entries"] == 2
    assert result["regions"]["disk0"]["all"]["entries"] == 1
    assert result["regions"]["disk1"]["all"]["entries"] == 1
    assert result["closure"] == dict(geometric=0., crystal=0.)
    np.testing.assert_array_equal(arrays["original_crystals"], [2, 0])
    for arm in ("stockref", "control", *rt.ARMS):
        assert t["saved"][arm + "_energy"][-1] == 999
        np.testing.assert_array_equal(arrays[arm + "_selected"], [True, False])
    for arm in ("control", *rt.ARMS):
        report = result["regions"]["all"]["all"]["arms"][arm]
        assert report["energy"]["fast"] == report["energy"]["stock"] == 10
        assert report["selected"] == report["jointly_selected"] == 1
        assert report["ratio"] == dict(ratio=1., error=None)


@pytest.mark.parametrize("field", EVENT)
def test_prepare_rejects_full_identity_change_on_positive_nonquery_row(saved_only, field):
    saved_only["saved"]["full_ids"][field][-1] += 1
    with pytest.raises(ValueError, match="mismatch: full_ids"):
        prepare(saved_only)


@pytest.mark.parametrize("family", ["event_row", "query_event_rows", "entry_", "geometry_", "sampling_"])
def test_prepare_rejects_stale_identity_geometry_entry_and_sampling(saved_only, family):
    t = saved_only
    names = [k for k in t["saved"] if k.startswith(family)] if family.endswith("_") else [family]
    assert names
    for name in names:
        original = t["saved"][name]
        t["saved"][name] = changed(original)
        with pytest.raises(ValueError, match="mismatch: " + name):
            prepare(t)
        t["saved"][name] = original


@pytest.mark.parametrize("field", ["pbtime", "mbtime", "event_length", "onspill", "marker"])
def test_prepare_rejects_changed_full_population_timing(saved_only, field):
    name = "timing_" + field
    value = saved_only["saved"][name]
    value[-1] = not value[-1] if value.dtype.kind == "b" else value[-1] + 1
    with pytest.raises(ValueError, match="mismatch: " + name):
        prepare(saved_only)


@pytest.mark.parametrize("name", ["geometric_energy", "crystal_time", "sampling_donor", "control_energy"])
def test_prepare_rejects_stale_baseline_payload(saved_only, name):
    saved_only["baseline"][name] = changed(saved_only["baseline"][name])
    with pytest.raises(ValueError, match="saved baseline mismatch: " + name):
        prepare(saved_only)


@pytest.mark.parametrize("fault", ["neighbors", "seed", "reco_seed", "draw", "geometric", "crystal"])
def test_prepare_fixed_trial_and_fingerprint_guards(saved_only, fault):
    t = saved_only
    if fault in ("neighbors", "seed", "reco_seed"):
        t["manifest"][fault] += 1
        message = "not the fixed K8 trial"
    elif fault == "draw":
        t["draw"]["time"] = changed(t["draw"]["time"])
        message = "saved draw fingerprint mismatch"
    else:
        t["manifest"]["audit"]["fingerprints"]["candidate_placements"][fault] = "stale"
        message = "saved placement fingerprint mismatch: " + fault
    with pytest.raises(ValueError, match=message):
        prepare(t)


@pytest.mark.parametrize("fault", ["offsets", "entry", "donor", "supported"])
def test_prepare_rejects_internally_inconsistent_saved_draw(saved_only, fault):
    t = saved_only
    if fault == "offsets":
        t["draw"]["offsets"][-1] += 1
    elif fault == "entry":
        t["draw"]["entry"][0] = 1
    elif fault == "donor":
        t["draw"]["donor"][0] = len(t["case"]["library"]["entry_energy"])
    else:
        t["draw"]["supported"][0] = False
    if "sampling_" + fault in t["saved"]:
        t["saved"]["sampling_" + fault] = t["draw"][fault].copy()
    t["manifest"]["audit"]["fingerprints"]["candidate_draw"] = rt.fingerprint(t["draw"])
    with pytest.raises(ValueError, match="invalid (saved draw offsets|donor support)"):
        prepare(t)


@pytest.mark.parametrize("arm", ["stockref", "control", *rt.ARMS])
@pytest.mark.parametrize("field,bad", [("energy", -1.), ("energy", np.nan), ("energy", np.inf),
                                       ("time", np.nan), ("x", np.inf), ("y", np.nan), ("z", np.inf)])
def test_positive_nonquery_rows_are_validated_not_filtered(saved_only, arm, field, bad):
    t = saved_only
    name = arm + "_" + field
    t["saved"][name][-1] = bad
    if arm in ("stockref", "control"):
        t["baseline"][name][-1] = bad  # Reach the full-population field validation.
    with pytest.raises(ValueError, match="invalid (saved energy|positive-cluster field): " + arm):
        prepare(t)


def test_zero_cluster_nan_fields_allowed_but_bad_shapes_rejected(saved_only):
    t = saved_only
    prepare(t)  # The empty query already has NaN time and coordinates.
    for arm in ("stockref", "control", *rt.ARMS):
        for field in rt.FIELDS:
            name = arm + "_" + field
            original = t["saved"][name]
            t["saved"][name] = original[:-1]
            old_baseline = t["baseline"][name]
            if arm in ("stockref", "control"):
                t["baseline"][name] = t["saved"][name].copy()
            with pytest.raises(ValueError, match="invalid (saved energy|positive-cluster field)"):
                prepare(t)
            t["saved"][name], t["baseline"][name] = original, old_baseline


@pytest.mark.parametrize("keep", [None, np.array([True, False, True, True, True, False])])
def test_occupancy_aggregates_positive_birks_crystals_both_disks_and_empty_queries(keep):
    steps = dict(entry=np.array([0, 0, 0, 0, 1, 1]), crystal=np.array([2, 2, 8, 9, 3, 3]),
                 disk=np.array([0, 0, 1, 1, 0, 0]), energy_birks=np.array([2., 3., 5., 0., -1., 7.]))
    result = rd.occupancy(steps, 3, keep)
    np.testing.assert_array_equal(result["crystals"], [2, 1, 0] if keep is None else [2, 0, 0])
    np.testing.assert_allclose(result["max_fraction"], [.5, 1, 0] if keep is None else [5 / 7, 0, 0])
    empty = rd.occupancy({k: v[:0] for k, v in steps.items()}, 3)
    np.testing.assert_array_equal(empty["crystals"], np.zeros(3))
    np.testing.assert_array_equal(empty["max_fraction"], np.zeros(3))


def test_gate_float32_boundary_no_double_pbtime_and_no_upper_cut():
    cfg = ReadoutConfig()
    timing = dict(pbtime=np.array([20., 7., 99.]), marker=np.array([5., 3., 0.]),
                  mbtime=np.array([2000., 2000., 2000.]), event_length=np.array([3000., 3000., 3000.]),
                  onspill=np.array([True, False, True]))
    rows = np.array([1, 0, 1, 0, 0, 1])
    pb, marker = timing["pbtime"][rows].astype(np.float32), timing["marker"][rows].astype(np.float32)
    threshold = ((np.float32(cfg.digitization_start) - pb) - marker) - np.float32(cfg.digitization_buffer)
    times = np.array([threshold[0] - .125, threshold[1], threshold[2] + .125, 1900., 2415., 3430.])
    folded, accepted, margin = rd.gate(times, rows, timing)
    np.testing.assert_allclose(folded, [429.875, 415., 430.125, 1900., 415., 430.], atol=0, rtol=0)
    np.testing.assert_array_equal(accepted, [False, True, True, True, True, True])
    np.testing.assert_allclose(margin, [-.125, 0., .125, 1485., 0., 0.], atol=0, rtol=0)
    assert folded[3] > cfg.digitization_end


def test_gate_uses_source_fmod_not_positive_modulo():
    timing = dict(pbtime=np.array([20.]), marker=np.array([5.]), mbtime=np.array([2000.]),
                  event_length=np.array([5000.]), onspill=np.array([False]))
    folded, accepted, margin = rd.gate(np.array([-500.]), np.array([0]), timing)
    np.testing.assert_array_equal(folded, [-500.])
    np.testing.assert_array_equal(accepted, [False])
    np.testing.assert_array_equal(margin, [-915.])


def test_exact_css_partition_spatial_first_then_time_and_same_source_rows(saved_only):
    t = saved_only
    prepared = prepare(t)
    # Keep the actual fixture donor and geometry; give one query four saved CSS steps.
    # The unmapped, early step is spatial loss ONLY, not also time rejection.
    steps = dict(entry=np.zeros(4, dtype=int), crystal=np.array([0, 1, 6, 7]),
                 disk=np.array([0, 0, 1, 1]), energy_birks=np.array([3., 5., 7., 11.]),
                 time=np.array([100., 100., 800., 1900.]))
    prepared["entries"]["event_row"] = np.array([2, 0])
    prepared["timing"] = dict(pbtime=np.array([1000., 300., 20.]), marker=np.zeros(3),
                              mbtime=np.full(3, 2000.), event_length=np.full(3, 3000.), onspill=np.ones(3, bool))
    prepared["draw"] = dict(prepared["draw"], **steps, offsets=np.array([0, 4, 4]))
    prepared["original"] = dict(steps)
    prepared["placed"] = {arm: dict(steps, mapped=np.array([False, True, True, True])) for arm in rt.ARMS}
    css = dict(t["css"], event_row=np.array([2, 0]), original=np.array([26., 0.]), donor=np.array([26., 0.]))
    for arm in rt.ARMS:
        css.update({arm + "_mapped": np.array([23., 0.]), arm + "_lost": np.array([3., 0.])})
    arrays, closure = rd.features(t["case"]["library"], prepared, css)
    assert closure == dict(geometric=0., crystal=0.)
    for arm in rt.ARMS:
        np.testing.assert_array_equal(arrays[arm + "_time_rejected"], [5., 0.])
        np.testing.assert_array_equal(arrays[arm + "_time_accepted"], [18., 0.])
        np.testing.assert_array_equal(arrays[arm + "_crystals"], [2, 0])
        np.testing.assert_array_equal(arrays[arm + "_end_exposed"], [11., 0.])
        np.testing.assert_allclose(arrays[arm + "_max_fraction"], [11 / 18, 0])
        np.testing.assert_array_equal(arrays["donor"], arrays[arm + "_lost"] +
                                      arrays[arm + "_time_rejected"] + arrays[arm + "_time_accepted"])
    np.testing.assert_array_equal(arrays["donor_time_accepted"], [18., 0.])
    np.testing.assert_array_equal(arrays["original_time_accepted"], [18., 0.])
    prepared["placed"]["geometric"]["energy_birks"] = np.array([3., 5., 7., 12.])
    with pytest.raises(ValueError, match="timing/spatial partition mismatch"):
        rd.features(t["case"]["library"], prepared, css)


@pytest.mark.parametrize("name", ["event_row", "original", "donor", "radius_delta", "energy_delta", "cosine_delta",
                                  "angle_deg", "geometric_mapped", "geometric_lost", "crystal_mapped", "crystal_lost"])
def test_features_reject_stale_saved_css(saved_only, name):
    saved_only["css"][name] = changed(saved_only["css"][name])
    with pytest.raises(ValueError, match="saved CSS mismatch: " + name):
        diagnose(saved_only)


def category_case():
    """Seven mutually exclusive states, once as a miss and once as a gain."""
    n = 14
    tile = lambda values: np.tile(values, 2)
    entries = dict(event_row=np.arange(n) * 101 + 99, radius=np.full(n, 449.), disk=tile([0, 1, 0, 1, 0, 1, 0]),
                   energy=tile([49., 50., 80., 95., 110., 100., 100.]), cosine=tile([0., .4, .7, 1., .3, .6, .9]))
    supported = tile([False, True, True, True, True, True, True])
    lengths = tile([0, 0, 1, 1, 1, 2, 1])
    draw = dict(supported=supported, donor=tile([-1, 0, 1, 2, 3, 4, 4]), fallback=~supported,
                offsets=np.r_[0, np.cumsum(lengths)], bin_pool_size=tile([0, 1, 8, 9, 32, 33, 100]),
                radius_pool_size=tile([0, 1, 8, 8, 8, 8, 8]))
    total, mapped, timed = tile([0., 0., 0., 12., 12., 12., 12.]), tile([0., 0., 0., 0., 12., 12., 12.]), tile([0., 0., 0., 0., 0., 6., 12.])
    arrays = dict(donor=total, original=np.full(n, 12.), original_time_accepted=np.full(n, 10.),
                  donor_time_accepted=timed, donor_steps=lengths, donor_crystals=tile([0, 0, 0, 1, 1, 2, 1]),
                  angle_deg=tile([np.nan, 0., 10., 30., 45., 5., 20.]), energy_delta=np.zeros(n), radius_delta=np.zeros(n),
                  entry_gate_margin=tile([-1., 0., 49., 50., 51., 1., 2.]), donor_reuse=tile([0, 2, 2, 2, 2, 4, 4]))
    arrays.update(original_steps=np.ones(n, dtype=int), original_crystals=np.ones(n, dtype=int))
    arrays.update({name + "_" + field: np.zeros(n) for name in ("original", "donor", *rt.ARMS)
                   for field in ("end_exposed", "delay_ge20", "delay_negative")})
    for arm in rt.ARMS:
        arrays.update({arm + "_mapped": mapped, arm + "_lost": total - mapped, arm + "_time_accepted": timed,
                       arm + "_time_rejected": mapped - timed, arm + "_crystals": arrays["donor_crystals"]})
    obs = {arm: dict(energy=np.r_[np.full(7, 20.), np.full(7, 10.)] if arm == "stockref" else
                    np.r_[np.full(7, 10.), np.full(7, 20.)], time=np.arange(n, dtype=float),
                    radial=np.zeros(n), tangent=np.zeros(n), truth_ep=np.zeros(n))
           for arm in ("stockref", "control", *rt.ARMS)}
    return entries, draw, arrays, obs


def test_masks_boundaries_distinguish_unsupported_empty_and_zero_nonempty():
    entries, draw, arrays, _ = category_case()
    masks = rd.strata(entries, draw, arrays)
    for label, indices in (("unsupported", [0, 7]), ("donor_empty", [1, 8]), ("donor_zero_nonempty", [2, 9]),
                           ("energy_50_80", [1, 8]), ("energy_80_95", [2, 9]), ("energy_110_inf", [4, 11]),
                           ("angle_10_30", [2, 6, 9, 13]), ("margin_-inf_0", [0, 7]),
                           ("margin_50_inf", [3, 4, 10, 11]), ("pool_1_9", [1, 2, 8, 9])):
        np.testing.assert_array_equal(np.flatnonzero(masks[label]), indices)
    assert masks["donor_positive_le10"].sum() == 0
    assert masks["donor_gt10"].sum() == 8
    assert masks["radius_pool_lt8"].sum() == 2


def test_summary_exact_miss_gain_partitions_denominators_and_source_blocks():
    entries, draw, arrays, obs = category_case()
    mask = np.ones(len(entries["event_row"]), bool)
    with patch.object(rd, "paired_mean", wraps=rd.paired_mean) as mean, \
         patch.object(rd, "paired_ratio", wraps=rd.paired_ratio) as ratio:
        result = rd.summarize(mask, entries, draw, arrays, {}, obs, shapes=True)
    assert (result["entries"], result["stock_selected"], result["supported"], result["fallback"], result["empty"]) == (14, 7, 12, 2, 2)
    assert (result["unique_donors"], result["max_donor_reuse"]) == (5, 4)
    expected = dict(unsupported=1, zero_donor=2, all_spatial_loss=1, all_time_rejected=1,
                    partial_time_rejected=1, positive_no_time_rejection=1)
    for arm in rt.ARMS:
        r = result["arms"][arm]
        assert r["selected"] == r["misses"] == r["gains"] == 7
        assert r["miss_partition"] == r["gain_partition"] == expected
        assert r["net_deficit"] == r["jointly_selected"] == 0
        assert r["ratio"]["ratio"] == 1
        assert r["energy"]["fast"] == r["energy"]["stock"] == 15
        assert r["time"] == dict(fast=None, stock=None, difference=None, error=None)
        assert r["css"]["timing_downcross"] == 4 and r["css"]["timing_downcross_misses"] == 2
        assert r["css"]["accepted_gt10"] == 2
        for label, shape in r["shapes"].items():
            assert sum(shape["fast"]) == sum(shape["stock"]) == (14 if label.startswith(("Leading", "E/p")) else 0)
    groups = entries["event_row"] // 100
    for call in ratio.call_args_list:
        np.testing.assert_array_equal(call.args[2], groups)
    for call in mean.call_args_list:
        expected_group = groups[draw['supported']] if len(call.args[0]) == 12 else groups if len(call.args[0]) else []
        np.testing.assert_array_equal(call.args[2], expected_group)
    occupancy = result['css']['occupancy_comparison']
    assert result['css']['occupancy_comparison_queries'] == 12
    assert occupancy['empty']['original'] == 0 and occupancy['empty']['donor'] == 2
    assert occupancy['empty']['donor_only'] == 2 and occupancy['empty']['original_only'] == 0
    assert occupancy['energy_le10']['donor'] == 4
    assert occupancy['energy_le10']['difference']['difference'] == pytest.approx(1 / 3)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_summary_zero_stock_denominator_empty_stratum_and_joint_selection():
    entries, draw, arrays, obs = category_case()
    mask = np.arange(14) >= 7
    r = rd.summarize(mask, entries, draw, arrays, {}, obs)
    assert r["entries"] == 7 and r["stock_selected"] == 0
    assert r["arms"]["geometric"]["ratio"] == dict(ratio=None, error=None)
    assert r["arms"]["geometric"]["gains"] == 7
    obs["geometric"]["energy"][:2] = 20
    obs["geometric"]["time"][:2] += 3
    r = rd.summarize(np.ones(14, bool), entries, draw, arrays, {}, obs)
    assert r["arms"]["geometric"]["jointly_selected"] == 2
    assert r["arms"]["geometric"]["time"] == dict(fast=3.5, stock=.5, difference=3., error=0.)
    empty = rd.summarize(np.zeros(14, bool), entries, draw, arrays, {}, obs, shapes=True)
    assert empty["entries"] == empty["unique_donors"] == empty["max_donor_reuse"] == 0
    assert empty["features"]["angle_deg"] == dict(count=0, mean=None, quantiles=None)
    for arm in ("control", *rt.ARMS):
        assert empty["arms"][arm]["energy"] == dict(fast=None, stock=None, difference=None, error=None)
        assert empty["arms"][arm]["ratio"] == dict(ratio=None, error=None)
    assert json.loads(json.dumps(empty, allow_nan=False)) == empty


@pytest.mark.parametrize("fault", ["origin", "axes"])
def test_diagnose_requires_common_aligned_disk_frames(saved_only, fault):
    prepared = prepare(saved_only)
    key = "disk_origin_x" if fault == "origin" else "disk_axis_x_x"
    prepared["geometry"][key] = prepared["geometry"][key].copy()
    prepared["geometry"][key][1] += 1
    with pytest.raises(ValueError, match="aligned common-origin disks"):
        diagnose(saved_only, prepared)


def test_reports_strict_json_null_and_plot_with_all_inner_strata_empty(saved_only, tmp_path):
    prepared = prepare(saved_only)
    prepared["entries"]["radius"][:] = 500.
    saved_only["css"].update(pd.donor_features(saved_only["case"]["library"], prepared["entries"], prepared["draw"]))
    result, _ = diagnose(saved_only, prepared)
    assert all(s["entries"] == 0 for s in result["regions"]["inner"].values())
    empty = result["regions"]["inner"]["all"]
    assert empty["arms"]["crystal"]["ratio"] == dict(ratio=None, error=None)
    path = tmp_path / "numbers.json"
    pc._json(path, result)
    assert json.loads(path.read_text()) == json.loads(json.dumps(result, allow_nan=False)) == result
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()
    png = tmp_path / "empty.png"
    rd.plot_report(result, png)
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def cli_fixture(t, tmp_path, monkeypatch):
    trial, baseline = tmp_path / "trial", tmp_path / "baseline"
    trial.mkdir()
    baseline.mkdir()
    library = tmp_path / "library.npz"
    shower_library.save(library, t["case"]["library"], t["case"]["metadata"])
    np.savez_compressed(baseline / "clusters.npz", **t["baseline"])
    manifest = copy.deepcopy(t["manifest"])
    manifest["inputs"] = {k: dict(path=str(p), sha256=rt.digest(p)) for k, p in
                          dict(tables=t["case"]["path"], library=library, clusters=baseline / "clusters.npz").items()}
    root = Path(rd.__file__).resolve().parents[1]
    manifest["calo_source_sha256"] = {name: rt.digest(root / name) for name in
                                     ("response.py", "readout.py", "showers/response.py", "showers/crystal_placement.py")}
    result, _ = diagnose(t)
    report = dict(manifest=manifest, candidate={arm: dict(paired=dict(
        fast_selected=result["regions"]["all"]["all"]["arms"][arm]["selected"],
        energy_mean=result["regions"]["all"]["all"]["arms"][arm]["energy"])) for arm in rt.ARMS})
    pc._json(trial / "preparation.json", manifest)
    pc._json(trial / "numbers.json", report)
    for filename, arrays in (("clusters", t["saved"]), ("draw", t["draw"]), ("css_arrays", t["css"])):
        np.savez_compressed(trial / (filename + ".npz"), **arrays)
    out = tmp_path / "diagnostic"
    monkeypatch.setattr(sys, "argv", ["residual_diagnostic", "--tables", str(t["case"]["path"]),
        "--library", str(library), "--trial", str(trial), "--baseline", str(baseline), "--out", str(out)])
    return trial, baseline, out


def test_cli_saved_only_outputs_hashes_archive_and_overwrite_guard(saved_only, tmp_path, monkeypatch, capsys):
    trial, baseline, out = cli_fixture(saved_only, tmp_path, monkeypatch)
    before = {p: rt.digest(p) for p in list(trial.iterdir()) + list(baseline.iterdir())}
    with patch.object(rd, "prepare", wraps=rd.prepare) as replay, patch.object(rd, "diagnose", wraps=rd.diagnose) as diagnostic:
        rd.main()
    assert replay.call_count == diagnostic.call_count == 1
    report = json.loads((out / "numbers.json").read_text())
    assert json.loads((out / "summary.json").read_text()) == {k: v["all"] for k, v in report["regions"].items()}
    assert report["inputs"]["clusters"]["sha256"] == before[baseline / "clusters.npz"]
    assert report["inputs"]["clusters.npz"]["sha256"] == before[trial / "clusters.npz"]
    assert {p: rt.digest(p) for p in before} == before
    assert json.loads(capsys.readouterr().out) == dict(queries=2, closure=dict(geometric=0., crystal=0.), output=str(out))
    arrays = rd.archive(out / "arrays.npz")
    np.testing.assert_array_equal(arrays["event_row"], [0, 1])
    assert (out / "inner_residuals.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="output must be new or empty"):
        rd.main()


@pytest.mark.parametrize("fault", ["input", "source", "manifest", "totals"])
def test_cli_saved_provenance_and_totals_guards(saved_only, tmp_path, monkeypatch, fault):
    trial, baseline, out = cli_fixture(saved_only, tmp_path, monkeypatch)
    if fault == "input":
        with (baseline / "clusters.npz").open("ab") as stream:
            stream.write(b"changed")
        message = "trial input hash mismatch: clusters"
    elif fault == "source":
        manifest = json.loads((trial / "preparation.json").read_text())
        manifest["calo_source_sha256"]["response.py"] = "stale"
        pc._json(trial / "preparation.json", manifest)
        message = "trial deterministic source mismatch: response.py"
    else:
        report = json.loads((trial / "numbers.json").read_text())
        if fault == "manifest":
            report["manifest"]["seed"] += 1
            message = "trial manifests differ"
        else:
            report["candidate"]["geometric"]["paired"]["fast_selected"] += 1
            message = "saved trial reconstructed totals differ: geometric"
        pc._json(trial / "numbers.json", report)
    with pytest.raises(ValueError, match=message):
        rd.main()
    assert not out.exists()


@pytest.mark.parametrize("saved_trial", ["normal", "one_unsupported", "all_unsupported", "zero_nonempty"], indirect=True)
def test_saved_prepare_and_diagnose_empty_unsupported_zero_nonempty(saved_only, request):
    t = saved_only
    support = request.node.callspec.params["saved_trial"]
    result, arrays = diagnose(t)
    expected = {"normal": (2, 0, 1, 0), "one_unsupported": (1, 1, 0, 0),
                "all_unsupported": (0, 2, 0, 0), "zero_nonempty": (2, 0, 1, 1)}[support]
    records = result["regions"]["all"]
    all_rows = records["all"]
    assert (all_rows["supported"], records["unsupported"]["entries"], records["donor_empty"]["entries"],
            records["donor_zero_nonempty"]["entries"]) == expected
    assert all_rows["entries"] == 2 and all_rows["stock_selected"] == 1
    assert all_rows["empty"] == expected[2]
    np.testing.assert_array_equal(arrays["event_row"], [0, 1])
    if support == "all_unsupported":
        assert all_rows["unique_donors"] == all_rows["max_donor_reuse"] == 0
        assert all_rows["donor_concentration"] is None
        np.testing.assert_array_equal(arrays["donor_reuse"], [0, 0])
        assert np.isnan(arrays["angle_deg"]).all()
    else:
        np.testing.assert_array_equal(arrays["donor_reuse"], [1, int(expected[0] == 2)])
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_time_exposure_is_not_an_upper_gate_and_delay_uses_same_query():
    entries = dict(event_row=np.array([2, 0]), energy=np.ones(2), time=np.array([800., 900.]))
    timing = dict(pbtime=np.array([10., 99., 20.]), marker=np.array([0., 1., 5.]),
                  mbtime=np.full(3, 2000.), event_length=np.full(3, 3000.), onspill=np.ones(3, bool))
    steps = dict(entry=np.array([0, 0, 0, 1, 1, 1]), energy_birks=np.array([1., 2., 4., 8., 16., 32.]),
                 time=np.array([1770., 820., 799., 1785., 919., 920.]))
    rows = entries["event_row"][steps["entry"]]
    _, accepted, _ = rd.gate(steps["time"], rows, timing)
    assert accepted.all()
    result = rd.time_exposure(steps, entries, timing, accepted)
    np.testing.assert_array_equal(result["end_exposed"], [1., 8.])
    np.testing.assert_array_equal(result["delay_ge20"], [3., 40.])
    np.testing.assert_array_equal(result["delay_negative"], [4., 0.])
    keep = accepted.copy()
    keep[[0, 3]] = False
    np.testing.assert_array_equal(rd.time_exposure(steps, entries, timing, keep)["end_exposed"], [0., 0.])


def test_inner_high_energy_regions_use_strict_radius_and_inclusive_energy(saved_only):
    prepared = prepare(saved_only)
    prepared["entries"]["energy"][:] = 95.
    prepared["entries"]["radius"][:] = [449., 450.]
    saved_only["css"].update(pd.donor_features(saved_only["case"]["library"], prepared["entries"], prepared["draw"]))
    result, _ = diagnose(saved_only, prepared)
    for name, count in (("all", 2), ("inner", 1), ("inner_highE", 1),
                        ("inner_highE_disk0", 1), ("inner_highE_disk1", 0)):
        assert result["regions"][name]["all"]["entries"] == count
    prepared["entries"]["energy"][0] = np.nextafter(95., 0.)
    saved_only["css"].update(pd.donor_features(saved_only["case"]["library"], prepared["entries"], prepared["draw"]))
    result, _ = diagnose(saved_only, prepared)
    assert result["regions"]["inner_highE"]["all"]["entries"] == 0
