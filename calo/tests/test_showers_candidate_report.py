"""Synthetic saved-only candidate reporting; fixtures never sample, build or reconstruct."""
import builtins
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from calofastsim.showers import candidate_report as cr
from calofastsim.showers import library as shower_library
from calofastsim.showers import placement_diagnostic as pd
from calofastsim.showers import production_compare as pc
from calofastsim.showers import radius_trial as rt
from calofastsim.showers import residual_diagnostic as rd
from calofastsim.showers.compare import paired_mean


def _steps(donors, offsets, energy):
    lengths = np.diff(offsets)
    query = np.repeat(np.arange(len(donors)), np.where(donors >= 0, lengths[np.maximum(donors, 0)], 0))
    selected = np.concatenate([np.arange(offsets[d], offsets[d + 1]) for d in donors if d >= 0]) if np.any(donors >= 0) else np.array([], int)
    disk = np.array([0, 0, 1, 0, 1])[selected]
    position = np.column_stack((np.array([5., 5., 5., 1100., 5.])[selected], np.zeros(len(selected)), 110. + disk * 100))
    return dict(entry=query, energy_birks=energy[selected].copy(), energy_g4=energy[selected].copy(),
                time=np.array([500., 500., 1800., 500., 439.])[selected], disk=disk,
                crystal=10 + disk, position=position)


@pytest.fixture
def saved(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sampling/build/prepare/reconstruction is forbidden")

    previous_reconstruct = sys.modules.get("calofastsim.showers.reconstruct")
    actual_import = builtins.__import__
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert "chain" not in name.split(".") and "chain" not in (fromlist or ()), "chain import forbidden"
        return actual_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(pd.uproot, "open", forbidden)
    monkeypatch.setitem(sys.modules, "calofastsim.showers.handoff", SimpleNamespace(from_cechain=forbidden))
    for module in (shower_library, pd, pc, rt, rd):
        for name in ("sample", "build", "prepare", "reproduce", "place", "place_crystals"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    rows = np.array([1, 2, 101, 102, 201, 202, 301, 401])
    n, full = len(rows), 502
    entries = dict(event_row=rows, energy=np.full(n, 100.), radius=np.array([450., 550., 449., 449., 449., 1000., 100., 550.]),
        disk=np.array([0, 0, 1, 1, 1, 0, 0, 1]), cosine=np.array([.5, .3, .7, .8, .6, .9, .9, .9]),
        time=np.full(n, 500.), direction=np.tile([0., 0., 1.], (n, 1)))
    entries["position"] = np.column_stack((entries["radius"], np.zeros(n), 100. + 100 * entries["disk"]))
    geometry = dict(disk_disk=np.array([0, 1]), disk_size_x=np.zeros(2), disk_size_y=np.full(2, 1000.),
                    crystal_crystal=np.array([10, 11]), crystal_disk=np.array([0, 1]))
    for axis in "xyz":
        geometry["disk_origin_" + axis] = np.array([100., 200.]) if axis == "z" else np.zeros(2)
        geometry["crystal_position_" + axis] = np.array([100., 200.]) if axis == "z" else np.zeros(2)
        geometry["crystal_size_" + axis] = np.full(2, 20.)
        for coord in "xyz":
            geometry["disk_axis_" + axis + "_" + coord] = np.full(2, float(axis == coord))
    offsets, energy = np.array([0, 0, 1, 3, 5]), np.array([10., 12., 18., 7., 23.])
    library = dict(offsets=offsets, step_energy_birks=energy, entry_energy=np.full(4, 100.),
        entry_radius=np.full(4, 5.), entry_cosine=np.ones(4), entry_position=np.tile([5., 0., 100.], (4, 1)),
        entry_direction=np.tile([0., 0., 1.], (4, 1)), **{"geometry_" + k: v for k, v in geometry.items()})
    donors = np.array([0, 1, 2, 3, 2, 3, 2, 3])
    draw = _steps(donors, offsets, energy)
    draw.update(donor=donors, supported=np.ones(n, bool), fallback=np.arange(n) == 1,
                offsets=np.r_[0, np.cumsum(np.diff(offsets)[donors])], radius_pool_size=np.full(n, 4), bin_pool_size=np.full(n, 10))
    placed = {arm: dict(draw, mapped=draw["position"][:, 0] < 1000) for arm in cr.ARMS}
    placed["crystal"]["crystal_loss_reason"] = np.zeros(len(draw["entry"]), int)
    original = _steps(np.full(n, 2), offsets, energy)
    timing = dict(pbtime=np.zeros(full), mbtime=np.full(full, 2000.), event_length=np.full(full, 3000.),
                  onspill=np.ones(full, bool), marker=np.zeros(full))
    ids = np.zeros(full, dtype=[("run", "i8"), ("subrun", "i8"), ("event", "i8")])
    ids["event"] = np.arange(full) + 1000
    prepared = dict(entries=entries, original=original, draw=draw, placed=placed, geometry=geometry, timing=timing, full_ids=ids)
    pools = dict(eligible=np.tile([0, 1, 2, 3, -1, -1, -1, -1], (n, 1)), event_row=rows.copy(),
        supported=draw["supported"].copy(), fallback=draw["fallback"].copy(), size=np.full(n, 4), full_size=np.full(n, 10),
        outcome_energy=np.array([0., 10., 30., 30.]), outcome_empty=np.array([True, False, False, False]),
        outcome_weak=np.array([True, True, False, False]), prob_energy=np.full(n, 17.5), prob_empty=np.full(n, .25), prob_weak=np.full(n, .5))
    common = dict(event_row=np.arange(full), query_event_rows=rows.copy(), full_ids=ids.copy())
    for prefix, value in (("entry_", entries), ("geometry_", geometry), ("timing_", timing)):
        common.update({prefix + k: v.copy() for k, v in value.items()})
    common.update({"sampling_" + k: draw[k].copy() for k in ("donor", "supported", "fallback", "offsets", "radius_pool_size", "bin_pool_size")})
    archives = []
    for index in range(3):
        archive = copy.deepcopy(common)
        for arm in ("stockref", "control", *cr.ARMS):
            e = np.zeros(full)
            e[rows] = [0., 10., 20., 20., 20., 20., 20., 20.]
            if arm in cr.ARMS:
                e[rows] += index * np.array([0., 0., 1., -6., 2., 0., 0., 0.])
                if index == 0:
                    e[rows[4]] = 0
                if index == 1:
                    e[rows[5]] = 0
            else:
                e[-1] = 99.
            archive[arm + "_energy"] = e
            for field in cr.FIELDS[1:]:
                value = np.full(full, np.nan)
                value[e > 0] = 500. if field == "time" else 10. + index * (arm in cr.ARMS)
                archive[arm + "_" + field] = value
        archives.append(archive)
    yield library, prepared, *archives, pools
    if previous_reconstruct is None:
        sys.modules.pop("calofastsim.showers.reconstruct", None)


def test_saved_only_report_full_accounting_common_subset_and_no_mutation(saved, tmp_path):
    before = copy.deepcopy(saved)
    result, arrays = cr.report(*saved)
    library, prepared, baseline, k8, output, pools = saved
    assert result["queries"] == 8 and result["full_stored_rows"] == 502
    assert result["strata_counts"]["inner_highE_disk1"] == 3
    assert result["strata_counts"]["inner_highE_disk1_coshigh"] == 2
    assert result["strata_counts"]["fallback"] == 1
    np.testing.assert_array_equal(arrays["eight_arm_common_event_rows"], [101, 301, 401])
    assert result["eight_arm_common_selected"] == 3
    assert set(result["eight_arm_distance_mm"]) == {"stockref", "control", *(label + "_" + arm for label in ("legacy", "k8", "candidate") for arm in cr.ARMS)}
    for arm, stats in result["eight_arm_distance_mm"].items():
        assert stats["error"] is not None
    for record in result["diagnostics"].values():
        assert all(item["common_selected"] == 3 for item in record.values())
    full = result["full_stored_accounting"]
    assert full["nonfront"] == 494 and full["inclusive_sampled_acceptance_model"] is False
    assert full["arms"]["stockref"]["nonfront"]["selected"] == 1
    assert full["arms"]["candidate_crystal"]["nonfront"]["zero"] == 494
    assert full["original_control_all_rows"]["events"] == 502
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    pc._json(tmp_path / "numbers.json", result)
    assert json.loads((tmp_path / "numbers.json").read_text()) == result
    def unchanged(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                unchanged(a[key], b[key])
        else:
            np.testing.assert_array_equal(a, b)
    for a, b in zip(saved, before):
        unchanged(a, b)


def test_paired_errors_use_original_stored_rows_and_energy_threshold(saved):
    result, _ = cr.report(*saved)
    rows = saved[1]["entries"]["event_row"]
    fast, prior = saved[4]["geometric_energy"][rows], saved[3]["geometric_energy"][rows]
    stats = result["change"]["k8"]["geometric"]["all"]
    assert stats["energy_mean"] == paired_mean(fast, prior, rows // 100)
    assert stats["energy_mean"]["error"] is not None
    assert paired_mean(fast, prior, np.arange(len(rows)) // 100)["error"] is None
    assert stats["fast_selected"] == 5 and stats["stock_selected"] == 5
    assert stats["fast_only"] == stats["stock_only"] == 1
    assert stats["energy_mean"]["fast"] == fast.mean()
    both = (fast > 10) & (prior > 10)
    assert stats["paired_time_ns"] == paired_mean(saved[4]["geometric_time"][rows][both], saved[3]["geometric_time"][rows][both], rows[both] // 100)
    legacy = result["change"]["legacy"]["crystal"]["all"]
    assert legacy["energy_mean"] == paired_mean(saved[4]["crystal_energy"][rows], saved[2]["crystal_energy"][rows], rows // 100)


def test_conditional_pool_sd_is_separate_from_paired_errors(saved):
    result, arrays = cr.report(*saved)
    all_rows = result["draw"]["all"]
    assert all_rows["fallback"] == 1
    empty, weak, energy = (all_rows["outcomes"][key] for key in ("empty", "weak", "energy"))
    assert (empty["actual_total"], empty["expected_total"]) == (1., 2.)
    assert (weak["actual_total"], weak["expected_total"]) == (2., 4.)
    assert empty["conditional_draw_sd_total"] == pytest.approx(np.sqrt(8 * .25 * .75))
    assert weak["conditional_draw_sd_total"] == pytest.approx(np.sqrt(8 * .5 * .5))
    variance = np.var([0., 10., 30., 30.])
    assert energy["conditional_draw_sd_total"] == pytest.approx(np.sqrt(8 * variance))
    assert energy["conditional_draw_sd_mean"] == pytest.approx(np.sqrt(variance / 8))
    np.testing.assert_array_equal(arrays["pool_variance_energy"], np.full(8, variance))
    assert "error" in energy["actual_minus_expected"]
    assert "conditional_draw_sd_total" not in energy["actual_minus_expected"]


def test_css_both_disks_spatial_and_exact_timing_partition(saved):
    result, arrays = cr.report(*saved)
    np.testing.assert_array_equal(arrays["css_donor"], [0., 10., 30., 30., 30., 30., 30., 30.])
    for arm in cr.ARMS:
        np.testing.assert_array_equal(arrays["residual_" + arm + "_lost"], [0., 0., 0., 7., 0., 7., 0., 7.])
        np.testing.assert_array_equal(arrays["residual_" + arm + "_time_rejected"], [0., 0., 0., 23., 0., 23., 0., 23.])
        np.testing.assert_array_equal(arrays["residual_" + arm + "_time_accepted"], [0., 10., 30., 0., 30., 0., 30., 0.])
        assert result["residual"]["closure"][arm] == 0
        all_rows = result["residual"]["regions"]["all"]["all"]["arms"][arm]
        assert sum(all_rows["miss_partition"].values()) == all_rows["misses"]
        assert sum(all_rows["gain_partition"].values()) == all_rows["gains"]
    assert arrays["residual_geometric_end_exposed"].sum() > 0
    assert "output-conditioned" in result["scope"]
    assert "not closure" in result["css"]["caveat"]


@pytest.mark.parametrize("which,key", [(2, "full_ids"), (3, "entry_radius"), (4, "timing_pbtime"),
                                       (4, "geometry_disk_size_x"), (3, "query_event_rows"), (4, "sampling_fallback")])
def test_reject_stale_saved_alignment(saved, which, key):
    value = saved[which][key]
    if value.dtype.names:
        value["event"][-1] += 1
    elif value.dtype.kind == "b":
        value[0] = ~value[0]
    else:
        value[0] += 1
    with pytest.raises(ValueError, match="mismatch"):
        cr.report(*saved)


@pytest.mark.parametrize("fault", ["membership", "fallback", "size", "probability", "outcome", "offsets", "energy", "duplicate", "placement"])
def test_reject_invalid_saved_pool_draw_and_placement(saved, fault):
    library, prepared, baseline, k8, output, pools = saved
    if fault == "membership":
        pools["eligible"][0, 0] = -1
        pools["size"][0] -= 1
        prepared["draw"]["radius_pool_size"][0] -= 1
        output["sampling_radius_pool_size"][0] -= 1
    elif fault == "fallback":
        pools["fallback"][0] = True
    elif fault == "size":
        pools["full_size"][0] += 1
    elif fault == "probability":
        pools["prob_energy"][0] += .01
    elif fault == "outcome":
        pools["outcome_weak"][0] = False
    elif fault == "offsets":
        prepared["draw"]["offsets"][-1] += 1
        output["sampling_offsets"][-1] += 1
    elif fault == "energy":
        prepared["draw"]["energy_birks"][0] += 1
    elif fault == "duplicate":
        pools["eligible"][0, 1] = pools["eligible"][0, 0]
    else:
        prepared["placed"]["crystal"]["time"] = prepared["draw"]["time"] + 1
    with pytest.raises(ValueError):
        cr.report(*saved)


@pytest.mark.parametrize("fault", ["nonfront", "positive_nan", "negative", "stock_drift", "control_drift"])
def test_full_stored_cluster_validation(saved, fault):
    output = saved[4]
    if fault == "nonfront":
        for field in cr.FIELDS:
            output["geometric_" + field][-1] = 20.
    elif fault == "positive_nan":
        output["crystal_time"][101] = np.nan
    elif fault == "negative":
        output["crystal_energy"][-1] = -1
    else:
        output[("stockref" if fault == "stock_drift" else "control") + "_energy"][-1] += 1
    with pytest.raises(ValueError):
        cr.report(*saved)


def test_empty_common_subset_is_explicit_and_json_safe(saved):
    for arm in cr.ARMS:
        saved[4][arm + "_energy"][:] = 0
    result, arrays = cr.report(*saved)
    assert result["eight_arm_common_selected"] == 0
    assert arrays["eight_arm_common_event_rows"].size == 0
    assert all(value["fast"] is None and value["error"] is None for value in result["eight_arm_distance_mm"].values())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("unsupported", [False, True])
def test_empty_and_unsupported_are_distinct_and_keep_all_rows(saved, unsupported):
    library, prepared, baseline, k8, output, pools = saved
    n = len(prepared["entries"]["energy"])
    donor = np.full(n, -1 if unsupported else 0)
    draw = _steps(donor, library["offsets"], library["step_energy_birks"])
    draw.update(donor=donor, supported=np.full(n, not unsupported), fallback=np.full(n, unsupported),
                offsets=np.zeros(n + 1, int), radius_pool_size=np.full(n, 0 if unsupported else 1),
                bin_pool_size=np.full(n, 0 if unsupported else 1))
    prepared["draw"] = draw
    prepared["placed"] = {arm: dict(draw, mapped=np.zeros(0, bool)) for arm in cr.ARMS}
    prepared["placed"]["crystal"]["crystal_loss_reason"] = np.zeros(0, int)
    pools["eligible"][:] = -1
    if not unsupported:
        pools["eligible"][:, 0] = 0
    for key in ("supported", "fallback"):
        pools[key] = draw[key].copy()
    pools["size"] = draw["radius_pool_size"].copy()
    pools["full_size"] = draw["bin_pool_size"].copy()
    for key in ("energy", "empty", "weak"):
        pools["prob_" + key][:] = np.nan if unsupported else (0. if key == "energy" else 1.)
    for key in ("donor", "supported", "fallback", "offsets", "radius_pool_size", "bin_pool_size"):
        output["sampling_" + key] = draw[key].copy()
    result, arrays = cr.report(*saved)
    record = result["draw"]["all"]
    assert record["unsupported"] == (n if unsupported else 0)
    assert record["outcomes"]["empty"]["actual_total"] == (0 if unsupported else n)
    assert record["outcomes"]["empty"]["conditional_draw_sd_total"] == 0
    assert result["residual"]["regions"]["all"]["all"]["empty"] == (0 if unsupported else n)
    assert arrays["query_event_rows"].size == n
    json.dumps(result, allow_nan=False)


def test_timing_partition_keeps_float32_lower_boundary_and_no_upper_cut(saved):
    draw = saved[1]["draw"]
    threshold = float(np.float32(450.) - np.float32(10.))
    draw["time"][:] = threshold
    draw["time"][0] = np.nextafter(threshold, -np.inf)
    result, arrays = cr.report(*saved)
    assert arrays["residual_geometric_time_rejected"][1] == 10.
    assert arrays["residual_geometric_time_accepted"][2] == 30.
    assert result["residual"]["closure"] == dict(geometric=0., crystal=0.)


def test_step_energy_changes_cannot_hide_behind_same_query_total(saved):
    draw = saved[1]["draw"]
    draw["energy_birks"][1:3] += [1., -1.]
    with pytest.raises(ValueError, match="draw_step_energy_birks"):
        cr.report(*saved)


def test_fixed_strata_preserve_the_declared_problem_region_counts():
    n = 2396
    entries = dict(radius=np.full(n, 550.), disk=np.zeros(n, int), energy=np.full(n, 100.), cosine=np.full(n, .5))
    entries["radius"][:190] = 449.
    entries["disk"][:190] = 1
    entries["cosine"][:109] = .7
    masks = cr.fixed_strata(entries, dict(fallback=np.arange(n) == n - 1, supported=np.ones(n, bool)))
    assert masks["all"].sum() == 2396
    assert masks["inner_highE_disk1"].sum() == 190
    assert masks["inner_highE_disk1_coshigh"].sum() == 109
    assert masks["fallback"].sum() == 1
    entries["radius"][0] = 450.
    entries["energy"][1] = np.nextafter(95., 0.)
    entries["cosine"][2] = np.nextafter(.7, 0.)
    masks = cr.fixed_strata(entries, dict(fallback=np.zeros(n, bool), supported=np.ones(n, bool)))
    assert masks["inner_highE_disk1"].sum() == 188
    assert masks["inner_highE_disk1_coshigh"].sum() == 106
