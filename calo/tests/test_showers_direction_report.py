"""Invented saved-array direction reporting; no physics calls or real data reads."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from calofastsim.showers import candidate_report as inclusive
from calofastsim.showers import direction_report as dr
from calofastsim.showers.compare import paired_mean
from test_showers_candidate_report import saved


@pytest.fixture
def direction_saved(saved, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved direction report must not draw, build, place, reconstruct or read data")

    for name in ("default_rng", "RandomState", "Generator", "SeedSequence", "seed", "random", "rand", "randn", "choice", "uniform", "normal", "randint"):
        monkeypatch.setattr(np.random, name, forbidden)
    for name in ("load", "loadtxt", "genfromtxt", "fromfile"):
        monkeypatch.setattr(np, name, forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("calofastsim") and module is not None:
            for name in ("sample", "build", "prepare", "reproduce", "place", "place_crystals", "run_chain", "from_cechain", "read", "load", "run", "run_calls", "pinned_json", "pinned_npz"):
                if callable(getattr(module, name, None)):
                    monkeypatch.setattr(module, name, forbidden)
    library, prepared, baseline, production, output, pools = saved
    entries = prepared["entries"]
    n = len(entries["event_row"])
    entries["direction"] = np.column_stack((np.linspace(-.4, .4, n), np.array([-.3, .2, -.1, .4, -.2, .1, -.4, .3]), np.full(n, .8)))
    for index, archive in enumerate((baseline, production, output)):
        archive["entry_direction"] = entries["direction"].copy()
        for arm in inclusive.ARMS:
            rows = entries["event_row"]
            positive = archive[arm + "_energy"][rows] > 0
            for field, values in (("x", np.arange(n) * (index + 1.) + 10.),
                                  ("y", np.arange(n) ** 2 + index * np.arange(n) + 10.),
                                  ("time", 500. + np.arange(n) * index)):
                archive[arm + "_" + field][rows[positive]] = values[positive]
    _, previous = inclusive.report(library, prepared, baseline, production, production, pools)
    prepared["draw"]["direction_used"] = np.arange(n) < 6
    prepared["draw"]["direction_fallback"] = np.arange(n) >= 6
    donor = prepared["draw"]["donor"]
    pools["eligible"][:] = -1
    pools["eligible"][:, :2] = np.column_stack((donor, (donor + 1) % 4))
    pools["size"][:] = 2
    prepared["draw"]["radius_pool_size"][:] = 2
    output["sampling_radius_pool_size"][:] = 2
    for name in ("energy", "empty", "weak"):
        pools["prob_" + name] = pools["outcome_" + name][pools["eligible"][:, :2]].mean(axis=1)
    return (*saved, previous)


def _scalar(y, x):
    correlation = float(np.corrcoef(y, x)[0, 1]) if np.std(y) > 0 and np.std(x) > 0 else np.nan
    return dict(mean=float(np.mean(y)), std=float(np.std(y)), covariance=float(np.mean((y - np.mean(y)) * (x - np.mean(x)))),
                correlation=correlation)


def _oracle(fast, prior, x, group, fm, pm):
    f, p = _scalar(fast[fm], x[fm]), _scalar(prior[pm], x[pm])
    leave = []
    for block in np.unique(group):
        a, b = _scalar(fast[fm & (group != block)], x[fm & (group != block)]), _scalar(prior[pm & (group != block)], x[pm & (group != block)])
        leave.append([a[k] - b[k] for k in f])
    leave = np.asarray(leave)
    errors = np.sqrt((len(leave) - 1) / len(leave) * np.sum((leave - leave.mean(axis=0)) ** 2, axis=0))
    return {k: f[k] - p[k] for k in f}, dict(zip(f, errors))


def test_saved_report_labels_capped_expectations_zeros_and_no_mutation(direction_saved):
    before = copy.deepcopy(direction_saved)
    result, diagnostic, conditional, arrays = dr.report(*direction_saved)
    prepared, pools = direction_saved[1], direction_saved[5]
    assert "PRODUCTIONK8" in result["archive_labels"]["k8"]
    assert "NOT old505K8" in result["direction_scope"]
    assert result["full_stored_accounting"]["inclusive_sampled_acceptance_model"] is False
    assert result["draw"]["all"]["outcomes"]["energy"]["expected_total"] == pools["prob_energy"].sum()
    assert pools["prob_energy"].sum() != 8 * 17.5
    assert result["direction_draw"]["direction_used"]["entries"] == 6
    assert result["direction_draw"]["direction_fallback"]["entries"] == 2
    regions = conditional["direct_production"]["regions"]
    assert len([k for k in regions if k.startswith("cell_")]) == 96
    assert {"cell_41", "cell_89", "problem109", "direction_used", "direction_fallback", "tangent_direction_negative"} <= regions.keys()
    assert regions["all"]["source_blocks"] == 5
    assert "sampling_direction_used" not in direction_saved[4]
    np.testing.assert_array_equal(diagnostic["css_original"], direction_saved[-1]["css_original"])
    np.testing.assert_array_equal(arrays["event_row"], prepared["entries"]["event_row"])
    assert diagnostic["css_donor"][0] == arrays["geometric_energy"][0] == 0
    for arm in dr.ARMS:
        np.testing.assert_allclose(sum(diagnostic[f"residual_{arm}_{k}"] for k in ("lost", "time_rejected", "time_accepted")), diagnostic["residual_donor"])
        term = sum(arrays[f"{arm}_term_{k}"] for k in ("donor", "spatial", "timing", "downstream", "control"))
        np.testing.assert_allclose(term, arrays[arm + "_energy"] - arrays["stockref_energy"])
        item = regions["all"]["arms"][arm]
        assert sum(v["difference"] for v in item["energy_terms"].values()) == pytest.approx(item["energy"]["difference"])
        assert sum(v["difference"] for v in item["ep_terms"].values()) == pytest.approx(item["truth_ep"]["difference"])
    json.dumps(result, allow_nan=False)
    json.dumps(conditional, allow_nan=False)
    def unchanged(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                unchanged(a[key], b[key])
        else:
            np.testing.assert_array_equal(a, b)
    for a, b in zip(direction_saved, before):
        unchanged(a, b)


def test_direct_pair_original_blocks_own_joint_and_identical_three_way_oracle(direction_saved):
    _, _, conditional, arrays = dr.report(*direction_saved)
    record = conditional["direct_production"]["regions"]["all"]["arms"]["geometric"]
    group = arrays["event_row"] // 100
    fe, pe, se = (arrays[k + "energy"] for k in ("geometric_", "production_geometric_", "stockref_"))
    fm, pm, sm = fe > 10, pe > 10, se > 10
    assert record["energy"] == paired_mean(fe, pe, group)
    assert record["selection"] == paired_mean(fm, pm, group)
    assert record["energy"]["error"] is not None
    assert paired_mean(fe, pe, np.arange(len(fe)) // 100)["error"] is None
    assert record["gains"] == record["misses"] == 1
    for name in ("radial", "tangent"):
        f, p, s = (arrays[k + name] for k in ("geometric_", "production_geometric_", "stockref_"))
        x = arrays["direction_" + name]
        for scope, a, b in (("own", fm, pm), ("joint_two_way", fm & pm, fm & pm)):
            difference, error = _oracle(f, p, x, group, a, b)
            stats = record["spatial"][name][scope]
            for key in difference:
                assert stats["difference"][key] == pytest.approx(difference[key], abs=1e-12)
                assert stats["error"][key] == pytest.approx(error[key], abs=1e-12)
        common = fm & pm & sm
        three = record["common_three_way"][name]
        for comparison, a, b in (("new_vs_stock", f, s), ("production_vs_stock", p, s), ("new_vs_production", f, p)):
            expected, error = _oracle(a, b, x, group, common, common)
            assert three[comparison]["fast"]["n"] == three[comparison]["stock"]["n"] == common.sum()
            for key in expected:
                if np.isfinite(expected[key]):
                    assert three[comparison]["difference"][key] == pytest.approx(expected[key], abs=1e-12)
                else:
                    assert three[comparison]["difference"][key] is None
                if np.isfinite(error[key]):
                    assert three[comparison]["error"][key] == pytest.approx(error[key], abs=1e-12)
                else:
                    assert three[comparison]["error"][key] is None
    np.testing.assert_array_equal(arrays["geometric_common_three_way_selected"], fm & pm & sm)


@pytest.mark.parametrize("fault", ["full_ids", "query_rows", "original_css", "partition", "diagnostic_shape", "diagnostic_pool", "reference", "energy", "positive_coordinate", "entry_coordinate", "geometry", "flags", "optional_flags", "pool_probability"])
def test_reject_stale_or_malformed_saved_inputs(direction_saved, fault):
    library, prepared, baseline, production, output, pools, diagnostic = direction_saved
    if fault == "full_ids":
        diagnostic["full_ids"]["event"][-1] += 1
    elif fault == "query_rows":
        diagnostic["query_event_rows"][0] += 1
    elif fault == "original_css":
        diagnostic["css_original_centroid_radial"][0] += 1
    elif fault == "partition":
        diagnostic["residual_geometric_lost"][0] += 1
    elif fault == "diagnostic_shape":
        diagnostic["css_donor"] = diagnostic["css_donor"][:-1]
    elif fault == "diagnostic_pool":
        diagnostic["pool_donor"][0] += 1
    elif fault == "reference":
        production["stockref_energy"][-1] += 1
    elif fault == "energy":
        output["geometric_energy"][0] = -1
    elif fault == "positive_coordinate":
        output["crystal_x"][101] = np.nan
    elif fault == "entry_coordinate":
        output["entry_direction"][0, 0] = np.nan
    elif fault == "geometry":
        output["geometry_disk_axis_x_y"][0] = .1
    elif fault == "flags":
        prepared["draw"]["direction_used"][7] = True
    elif fault == "optional_flags":
        output["sampling_direction_used"] = ~prepared["draw"]["direction_used"]
    else:
        pools["prob_energy"][0] += 1
    with pytest.raises(ValueError):
        dr.report(*direction_saved)


def test_empty_direction_stratum_and_empty_new_selection_are_explicit(direction_saved):
    prepared, output = direction_saved[1], direction_saved[4]
    prepared["draw"]["direction_used"][:] = False
    prepared["draw"]["direction_fallback"][:] = True
    for arm in dr.ARMS:
        output[arm + "_energy"][:] = 0
    _, _, conditional, arrays = dr.report(*direction_saved)
    regions = conditional["direct_production"]["regions"]
    assert regions["direction_used"]["entries"] == 0
    for arm in dr.ARMS:
        stats = regions["all"]["arms"][arm]
        assert stats["common_three_way_selected"] == 0
        assert stats["spatial"]["radial"]["own"]["fast"]["mean"] is None
        assert stats["common_three_way"]["radial"]["new_vs_production"]["error"]["mean"] is None
        assert not arrays[arm + "_common_three_way_selected"].any()
    json.dumps(conditional, allow_nan=False)


def test_timing_truth_ep_shapes_and_direct_css_partition(direction_saved):
    _, diagnostic, conditional, arrays = dr.report(*direction_saved)
    item = conditional["direct_production"]["regions"]["all"]["arms"]["crystal"]
    group = arrays["event_row"] // 100
    assert item["truth_ep"] == paired_mean(arrays["crystal_truth_ep"], arrays["production_crystal_truth_ep"], group)
    fm, pm = arrays["crystal_energy"] > 10, arrays["production_crystal_energy"] > 10
    joint = fm & pm
    timing = item["spatial"]["time"]["joint_two_way"]
    assert timing["fast"]["n"] == joint.sum()
    assert timing["difference"]["mean"] == pytest.approx(np.mean(arrays["crystal_time"][joint] - arrays["production_crystal_time"][joint]))
    for key, expected in (("energy", len(group)), ("truth_ep", len(group)), ("radial", fm.sum()), ("tangent", fm.sum()), ("time", fm.sum())):
        assert item["shapes"][key]["fast_selected"] == expected
        assert sum(item["shapes"][key]["fast"]) == expected
    for key, value in item["css_energy_partition"].items():
        assert value == paired_mean(diagnostic["residual_" + key], direction_saved[-1]["residual_" + key], group)
    assert item["common_three_way_selected"] < fm.sum()
    assert item["common_three_way_selected"] < pm.sum()


def test_full_id_duplicates_rejected_even_when_all_archives_match(direction_saved):
    for value in (direction_saved[1], *direction_saved[2:5], direction_saved[-1]):
        value["full_ids"][1] = value["full_ids"][0]
    with pytest.raises(ValueError, match="unique"):
        dr.report(*direction_saved)
