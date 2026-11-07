"""Synthetic scalar-oracle checks of the diagnostic witness; no data or sampling."""
from copy import deepcopy
import json
import math
import random
import sys

import numpy as np
import pytest
import uproot

from calofastsim.showers import direction_support as ds
from calofastsim.showers import reconstruct
from calofastsim.showers import support_audit as sa


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("direction support tests must not draw, build, place, reconstruct or read data")
    for module in tuple(sys.modules.values()):
        if module is not None and getattr(module, "__name__", "").startswith("calofastsim.showers"):
            for name in ("sample", "build", "place", "place_crystals", "load_chain", "steps_for_reco"):
                if callable(getattr(module, name, None)):
                    monkeypatch.setattr(module, name, forbidden)
    for module, names in ((np.random, ("default_rng", "RandomState", "Generator", "random", "random_sample", "rand", "randn", "choice", "seed", "uniform")),
                          (random, ("Random", "SystemRandom", "random", "choice", "choices", "seed", "uniform")),
                          (uproot, ("open", "iterate", "concatenate")), (np, ("load",)),
                          (ds.auth, ("pinned_json", "pinned_npz")),
                          (ds.source_overlap_report.Inputs, ("read",))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


def scalar_local(lib, table, prefix=""):
    result = []
    for i, disk in enumerate(table[prefix + "disk"]):
        slot = list(lib["geometry_disk_disk"]).index(disk)
        x, y = (float(table[prefix + "position"][i, j] - lib["geometry_disk_origin_" + axis][slot])
                for j, axis in enumerate("xy"))
        radius = math.hypot(x, y)
        rx, ry = (x / radius, y / radius) if radius else (1., 0.)
        ux, uy, uz = table[prefix + "direction"][i]
        result.append((rx * ux + ry * uy, rx * uy - ry * ux, uz))
    return np.asarray(result, float).reshape(-1, 3)


def scalar_pair(query, donor):
    sign = all((query[j] >= 0) == (donor[j] >= 0) for j in (0, 1))
    dot = sum(float(a) * float(b) for a, b in zip(query, donor))
    angle = math.degrees(math.acos(min(1., max(-1., dot))))
    return sign, sign and angle <= 10., angle


def scalar_assessment(lib, query, pools):
    q, d = scalar_local(lib, query), scalar_local(lib, lib, "entry_")
    n = len(q)
    expected = dict(query_direction=q, donor_direction=d)
    for kind in ("sign", "compatible"):
        for key in ("count", "within25_count", "baseline_count"):
            expected[kind + "_" + key] = np.zeros(n, np.int64)
        for key in ("nearest_radius", "last_radius"):
            expected[kind + "_" + key] = np.full(n, np.nan)
    expected["compatible_eligible"] = np.full((n, 8), -1, np.int64)
    expected["witness_eligible"] = np.full((n, 8), -1, np.int64)
    expected["direction_fallback"] = np.zeros(n, bool)
    for i in range(n):
        if not pools["supported"][i]:
            continue
        rows = [r for r, cell in enumerate(pools["library_cell"]) if cell == pools["chosen_cell"][i]]
        for index, kind in enumerate(("sign", "compatible")):
            accepted = [r for r in rows if scalar_pair(q[i], d[r])[index]]
            distance = lambda r: abs(float(lib["entry_radius"][r]) - float(query["radius"][i]))
            selected = sorted(accepted, key=lambda r: (distance(r), r))[:8]
            expected[kind + "_count"][i] = len(accepted)
            expected[kind + "_within25_count"][i] = sum(distance(r) <= 25. for r in accepted)
            expected[kind + "_baseline_count"][i] = sum(scalar_pair(q[i], d[r])[index] for r in pools["eligible"][i] if r >= 0)
            if selected:
                expected[kind + "_nearest_radius"][i] = min(distance(r) for r in selected)
                expected[kind + "_last_radius"][i] = max(distance(r) for r in selected)
            if kind == "compatible":
                expected["compatible_eligible"][i, :len(selected)] = sorted(selected)
                if not selected:
                    expected["direction_fallback"][i] = True
                    selected = [r for r in pools["eligible"][i] if r >= 0]
                expected["witness_eligible"][i, :len(selected)] = sorted(selected)
    expected["witness_size"] = np.array([sum(r >= 0 for r in row) for row in expected["witness_eligible"]], np.int64)
    expected["baseline_eligible"] = pools["eligible"].copy()
    expected["baseline_size"] = pools["size"].copy()
    for name in ("baseline", "witness"):
        for key in ("radius_delta", "angle_deg", "radial_delta", "tangent_delta", "energy_delta", "cosine_delta"):
            expected[name + "_" + key] = np.full((n, 8), np.nan)
        for i, row in enumerate(expected[name + "_eligible"]):
            for j, donor in enumerate(row):
                if donor < 0:
                    continue
                for key in ("radius", "energy", "cosine"):
                    expected[name + "_" + key + "_delta"][i, j] = lib["entry_" + key][donor] - query[key][i]
                expected[name + "_angle_deg"][i, j] = scalar_pair(q[i], d[donor])[2]
                expected[name + "_radial_delta"][i, j] = d[donor, 0] - q[i, 0]
                expected[name + "_tangent_delta"][i, j] = d[donor, 1] - q[i, 1]
    return expected


def assert_oracle(lib, query):
    from test_showers_support_audit import assert_same, oracle
    pools = sa.enumerate_pools(lib, query)
    assert_same(pools, oracle(lib, query))
    actual = ds.assess(lib, query, pools)
    for key, value in scalar_assessment(lib, query, pools).items():
        if value.dtype.kind == "f":
            np.testing.assert_allclose(actual[key], value, atol=2e-6, rtol=1e-12, err_msg=key)
        else:
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
    for key, source in (("event_row", query["event_row"]), ("original_fallback", pools["fallback"]),
                        ("supported", pools["supported"]), ("query_cell", pools["query_cell"])):
        np.testing.assert_array_equal(actual[key], source)
    return pools, actual


def test_nested_support_radius_ties_sparse_and_original_k8_fallback():
    from test_showers_support_audit import entries, library
    radius = list(range(100, 108)) + [180, 140, 130, 125, 160, 170, 190, 200, 200, 200, 110, 100, 100]
    lib = library(radius, disk=[0] * 19 + [1, 1], energy=[100.] * 19 + [25., 25.])
    lib["entry_direction"][:8] = [-.6, 0., .8]
    lib["entry_direction"][18] = [math.sqrt(.75), 0., .5]
    lib["entry_direction"][20] = [.4, -math.sqrt(.2), .8]
    query = entries([100, 100, 100, 500, 100, 100], disk=[0, 0, 0, 0, 1, 1], energy=[100.] * 4 + [25., 25.])
    query["direction"][1] = [-.6, 0., .8]
    query["direction"][2] = [0., 0., 1.]
    query["direction"][5] = lib["entry_direction"][20]
    pools, arrays = assert_oracle(lib, query)
    assert arrays["sign_count"][0] == 11 and arrays["compatible_count"][0] == 10
    assert arrays["compatible_baseline_count"][0] == 0
    assert arrays["compatible_within25_count"][0] == 1
    np.testing.assert_array_equal(arrays["witness_eligible"][0], np.arange(8, 16))
    assert arrays["compatible_last_radius"][0] == 100.
    assert np.max(np.abs(arrays["baseline_radius_delta"][0])) == 7.
    np.testing.assert_array_equal(arrays["direction_fallback"], [False, False, True, False, False, False])
    np.testing.assert_array_equal(arrays["witness_eligible"][2], pools["eligible"][2])
    assert np.all(arrays["compatible_eligible"][2] == -1)
    assert arrays["original_fallback"][3] and not arrays["direction_fallback"][3]
    np.testing.assert_array_equal(arrays["witness_size"][-2:], [1, 1])
    np.testing.assert_array_equal(arrays["witness_eligible"][-2:, 0], [19, 20])


@pytest.mark.parametrize("theta,accepted", [(0., True), (9.999999, True), (10.000001, False), (180., False)])
def test_fixed_angular_boundary(theta, accepted):
    query = np.array([0., 0., 1.])
    donor = np.array([math.sin(math.radians(theta)), 0., math.cos(math.radians(theta))])
    sign, good, angle = ds.compatible(query, donor)
    assert bool(sign)
    assert bool(good) is accepted
    assert angle == pytest.approx(theta, abs=1e-6)
    assert ds.ANGLE_DEG == 10. and ds.RADIUS_MM == 25. and ds.K == 8


@pytest.mark.parametrize("axis", [0, 1])
def test_zero_and_negative_zero_are_nonnegative_not_a_tolerance(axis):
    query = np.array([0., 0., 1.])
    donor = np.tile(query, (3, 1))
    donor[:, axis] = [-0., 1e-12, -1e-12]
    sign, good, angle = ds.compatible(query, donor)
    np.testing.assert_array_equal(sign, [True, True, False])
    np.testing.assert_array_equal(good, [True, True, False])
    np.testing.assert_allclose(angle, 0.)


@pytest.mark.parametrize("disk", [0, 1])
def test_unsupported_disk_stays_unsupported_not_direction_fallback(disk):
    from test_showers_support_audit import entries, library
    lib, query = library([100.], disk=[disk]), entries([100., 100.], disk=[disk, 1 - disk])
    pools, arrays = assert_oracle(lib, query)
    report = ds.report(lib, query, pools, arrays)
    assert not arrays["supported"][1] and not arrays["direction_fallback"][1]
    assert arrays["witness_size"][1] == 0 and arrays["baseline_size"][1] == 0
    assert np.all(arrays["witness_eligible"][1] == -1)
    for arm in ("baseline", "witness"):
        for outcome in ("empty", "weak", "energy"):
            assert np.isnan(arrays[arm + "_prob_" + outcome][1])
    assert report["regions"]["all"]["queries"] == 2
    assert report["regions"]["disk" + str(1 - disk)]["supported"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("donors", [0, 1])
@pytest.mark.parametrize("queries", [0, 2])
def test_empty_queries_and_empty_library(donors, queries):
    from test_showers_support_audit import entries, library
    lib, query = library([100.] * donors), entries([100.] * queries)
    pools, arrays = assert_oracle(lib, query)
    report = ds.report(lib, query, pools, arrays)
    assert arrays["witness_eligible"].shape == (queries, 8)
    assert report["regions"]["all"]["queries"] == queries
    assert report["regions"]["all"]["supported"] == (queries if donors else 0)
    assert not arrays["direction_fallback"].any()
    assert len([key for key in report["regions"] if key.startswith("cell_")]) == 96
    json.dumps(report, allow_nan=False)


def test_translated_disk_azimuth_rotations_signed_components_and_readonly_inputs():
    from test_showers_support_audit import assert_same, entries, library
    lib = library([100., 110.], disk=[0, 1])
    query = entries([120., 130.], disk=[0, 1], energy=[101., 102.], cosine=[.81, .82])
    lib["geometry_disk_origin_x"][:] = [30., -70.]
    lib["geometry_disk_origin_y"][:] = [-15., 45.]
    for table, prefix, phis, local in ((lib, "entry_", [0., math.pi / 2], [.5, -.3, math.sqrt(.66)]),
                                       (query, "", [math.pi, -math.pi / 2], [.49, -.31, math.sqrt(.6638)])):
        for i, phi in enumerate(phis):
            c, s = math.cos(phi), math.sin(phi)
            table[prefix + "position"][i, :2] = [lib["geometry_disk_origin_x"][i] + table[prefix + "radius"][i] * c,
                                                  lib["geometry_disk_origin_y"][i] + table[prefix + "radius"][i] * s]
            table[prefix + "direction"][i] = [c * local[0] - s * local[1], s * local[0] + c * local[1], local[2]]
    before = deepcopy((lib, query))
    for table in (lib, query):
        for value in table.values():
            value.flags.writeable = False
    pools, arrays = assert_oracle(lib, query)
    np.testing.assert_allclose(arrays["baseline_radial_delta"][:, 0], .01)
    np.testing.assert_allclose(arrays["baseline_tangent_delta"][:, 0], .01)
    np.testing.assert_allclose(arrays["baseline_radius_delta"][:, 0], [-20., -20.])
    assert arrays["compatible_count"].tolist() == [1, 1]
    expected = ds.pool_census.coverage(lib, query, pools)
    np.testing.assert_allclose(arrays["baseline_angle_deg"], expected["angle_deg"], atol=2e-6)
    for actual, original in zip((lib, query), before):
        assert_same(actual, original)
        assert all(not value.flags.writeable for value in actual.values())


def test_uniform_baseline_witness_both_disk_energy_and_no_outcome_veto():
    from test_showers_support_audit import entries, library
    lib = library([100., 101., 102., 103., 140.], lengths=[0, 1, 2, 2, 1], response=[0., 4., 6., 4., 9., 100.])
    lib["step_disk"][:] = [0, 0, 1, 0, 1, 1]
    lib["entry_direction"][-1] = [-.6, 0., .8]
    query = entries([100., 110.])
    pools, arrays = assert_oracle(lib, query)
    baseline_report, baseline_arrays = sa.audit(lib, query, dict(entry=np.array([], np.int64), energy_birks=np.array([])))
    for key in ("empty", "weak", "energy"):
        pools["original_" + key] = baseline_arrays["original_" + key]
    report = ds.report(lib, query, pools, arrays)
    for key, values in (("empty", [.2, .25]), ("weak", [.6, .75]), ("energy", [24.6, 5.75])):
        for arm, mean in zip(("baseline", "witness"), values):
            np.testing.assert_allclose(arrays[arm + "_prob_" + key], [mean, mean])
            outcome = report["regions"]["all"][arm]["outcomes"][key]
            assert outcome["expected_sum"] == pytest.approx(2 * mean)
            assert outcome["original_sum"] == (2 if key != "energy" else 0)
            if key != "energy":
                assert outcome["conditional_count_sd"] == pytest.approx(math.sqrt(2 * mean * (1 - mean)))
        np.testing.assert_array_equal(arrays["baseline_prob_" + key], baseline_arrays["prob_" + key])
        assert report["regions"]["all"]["baseline"]["outcomes"][key]["expected_minus_original"] == baseline_report["regions"]["all"]["outcomes"][key]["expected_minus_original"]
    for arm, size in (("baseline", 5), ("witness", 4)):
        section = report["regions"]["all"][arm]
        assert section["eligible_union"] == size
        assert section["eligible_empty"] == 1 and section["eligible_weak"] == 3
        assert section["max_expected_exposure"] == pytest.approx(2 / size)
    changed = deepcopy(lib)
    changed["step_energy_birks"][:] = 9999.
    changed["step_disk"][:] = 1
    _, changed_arrays = assert_oracle(changed, query)
    np.testing.assert_array_equal(changed_arrays["witness_eligible"], arrays["witness_eligible"])
    np.testing.assert_array_equal(arrays["witness_eligible"][0], [0, 1, 2, 3, -1, -1, -1, -1])
    json.dumps(report, allow_nan=False)


def test_fixed_reporting_strata_all_96_cells_and_sign_partitions():
    from test_showers_support_audit import entries, library
    disk, energy, cosine, radius = np.indices((2, 4, 3, 4)).reshape(4, -1)
    query = entries(np.array([100., 450., 550., 800.])[radius], disk=disk,
                    energy=np.array([25., 65., 85., 100.])[energy], cosine=np.array([.2, .5, .8])[cosine])
    query["direction"][::2] = [-.6, 0., .8]
    query["direction"][1::4] = [.4, -math.sqrt(.2), .8]
    lib = library([100., 100.], disk=[0, 1])
    pools, arrays = assert_oracle(lib, query)
    masks = ds.regions(query, pools, arrays)
    assert all(masks[f"cell_{i:02d}"].sum() == 1 for i in range(96))
    assert masks["disk0"].sum() == masks["disk1"].sum() == 48
    for sign in ("radial", "tangent"):
        np.testing.assert_array_equal(masks[sign + "_negative"] ^ masks[sign + "_nonnegative"], masks["all"])
    assert masks["highEinnerdisk1"].sum() == 3 and masks["problem109"].sum() == 1
    assert masks["radius_0_450"].sum() == 24
    assert masks["radius_450_550"].sum() == 24
    assert masks["radius_550_inf"].sum() == 48
    np.testing.assert_array_equal(masks["fallback"], pools["fallback"])


def test_expectations_use_original_k8_not_full_cell_and_fallback_retains_response():
    from test_showers_support_audit import entries, library
    lib = library(100. + 10 * np.arange(10), lengths=[1] * 8 + [0, 2], response=[20.] * 8 + [4., 9.])
    lib["entry_direction"][:8] = [-.6, 0., .8]
    lib["step_disk"][-2:] = [0, 1]
    query = entries([100., 190., 100.])
    query["direction"][-1] = [0., 0., 1.]
    pools, arrays = assert_oracle(lib, query)
    report = ds.report(lib, query, pools, arrays)
    np.testing.assert_array_equal(arrays["direction_fallback"], [False, False, True])
    np.testing.assert_array_equal(arrays["witness_size"], [2, 2, 8])
    for arm, expectation in (("baseline", dict(empty=[0., .125, 0.], weak=[0., .125, 0.], energy=[20., 16.625, 20.])),
                             ("witness", dict(empty=[.5, .5, 0.], weak=[.5, .5, 0.], energy=[6.5, 6.5, 20.]))):
        for outcome, values in expectation.items():
            np.testing.assert_allclose(arrays[arm + "_prob_" + outcome], values)
            assert report["regions"]["all"][arm]["outcomes"][outcome]["expected_sum"] == pytest.approx(sum(values))
        exposure = [0.] * 10
        for row in arrays[arm + "_eligible"]:
            selected = [int(r) for r in row if r >= 0]
            for r in selected:
                exposure[r] += 1 / len(selected)
        section = report["regions"]["all"][arm]
        assert section["eligible_union"] == sum(value > 0 for value in exposure)
        assert section["max_expected_exposure"] == pytest.approx(max(exposure))
        assert sum(exposure) == pytest.approx(len(query["energy"]))
    assert arrays["baseline_prob_energy"][0] != pytest.approx(sum(lib["step_energy_birks"]) / 10)
    assert report["regions"]["all"]["direction_fallback"] == 1
    for outcome in ("empty", "weak", "energy"):
        assert np.isnan(arrays["compatible_prob_" + outcome][-1])
        assert report["regions"]["all"]["compatible_only_outcomes"][outcome]["unavailable"] == 1
    assert report["regions"]["all"]["compatible"]["eighth_radius"]["unavailable"] == 3


@pytest.mark.parametrize("mutation", ["nonfinite", "nonunit", "negative_row", "bad_axis"])
def test_reject_invalid_local_direction_inputs(mutation):
    from test_showers_support_audit import entries, library
    lib, query = library([100.]), entries([100.])
    if mutation == "nonfinite":
        query["position"][0, 0] = np.nan
    elif mutation == "nonunit":
        query["direction"][0] *= 2
    elif mutation == "negative_row":
        query["event_row"][0] = -1
    else:
        lib["geometry_disk_axis_z_x"][0] = 1
    with pytest.raises(ValueError):
        ds.local_direction(lib, query)


def test_existing_output_refused_before_any_input_read(tmp_path):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="new output directory required"):
        ds.run(SimpleNamespace(out=tmp_path))
