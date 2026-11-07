"""Invented saved arrays and literal-block oracles only; no real report execution."""
import copy
import json
import subprocess

import numpy as np
import pytest

from calofastsim.showers import direction_localization as dl


def invented():
    rows = np.array([1, 2, 3, 101, 102, 103, 201, 202, 203, 301, 302, 303, 401, 402, 501, 601])
    n, full = len(rows), 702
    ids = np.zeros(full, dtype=[("run", "i8"), ("subrun", "i8"), ("event", "i8")])
    ids["run"] = 1
    ids["event"] = np.arange(full) + 100
    stock = np.array([0., 80., 84.9, 85., 89.9, 90., 94.9, 95., 99.9, 100., 120., 121., 0., 10., 0., 0.])
    production = stock + np.array([0., 5., 6., -5., -10., 10., 5., -5., 10., 10., 10., -10., 0., 0., 0., 0.])
    new = stock + np.array([0., 10., -5., 0., 5., -10., 10., 5., -15., 30., 0., -5., 20., -5., 0., 0.])
    control = stock.copy()
    control[8] = 0
    used = np.arange(n) < 10
    radius = np.array([0., 450., 550., 750., 1000., 440., 500., 600.] * 2)
    cosine = np.array([.1, .4, .7, 1.] * 4)
    disk = np.arange(n) % 2
    front = np.zeros(full, bool)
    front[rows] = True
    c = dict(event_row=np.arange(full), full_ids=ids, query_event_rows=rows, entry_event_row=rows,
             entry_disk=disk, entry_energy=np.full(n, np.sqrt(10000. + .51099895 ** 2)), entry_cosine=cosine,
             entry_radius=radius, sampling_supported=np.ones(n, bool), sampling_fallback=np.zeros(n, bool),
             sampling_donor=np.arange(n), sampling_radius_pool_size=np.full(n, 8), sampling_bin_pool_size=np.full(n, 25))
    cell = np.ravel_multi_index((disk, np.full(n, 3), np.searchsorted((.4, .7), cosine, side="right"),
                                np.minimum(np.searchsorted((450., 550., 750., 1000.), radius, side="right"), 3)), (2, 4, 3, 4))
    a = dict(event_row=rows, momentum=np.full(n, 100.), cell_id=cell,
             direction_radial=np.linspace(-.8, .8, n), direction_tangent=np.where(used, .3, -.3),
             direction_used=used, direction_fallback=~used)
    for key in ("event_row", "momentum", "cell_id", "direction_radial", "direction_tangent"):
        a["production_" + key] = a[key].copy()
    energies = dict(stockref=stock, control=control, geometric=new, crystal=new + np.where(new > 10, 2., 0.),
                    production_stockref=stock, production_control=control, production_geometric=production,
                    production_crystal=production + np.where(production > 10, 1., 0.))
    for index, (prefix, energy) in enumerate(energies.items()):
        a[prefix + "_energy"] = energy.copy()
        a[prefix + "_truth_ep"] = energy / 100.
        for coordinate, scale in (("radial", 100.), ("tangent", 50.), ("time", 1.)):
            a[prefix + "_" + coordinate] = (np.arange(n) - 5.) * scale + index * np.arange(n)
        if prefix.startswith("production_") and prefix.split("_")[1] in ("stockref", "control"):
            for field in ("radial", "tangent", "time"):
                a[prefix + "_" + field] = a[prefix.removeprefix("production_") + "_" + field].copy()
        if not prefix.startswith("production_"):
            e = np.zeros(full)
            e[rows] = energy
            c[prefix + "_energy"] = e
            for field in ("x", "y", "z", "time"):
                c[prefix + "_" + field] = np.where(e > 0, 1., np.nan)
    diagnostics = []
    original = stock.copy()
    donor = np.where(new > 0, new + 3., 0.)
    for prefix in ("", "production_"):
        donor_energy = donor.copy()
        if prefix:
            donor_energy[used & (donor_energy > 0)] += 5.
        d = dict(event_row=np.arange(full), full_ids=ids.copy(), query_event_rows=rows.copy(), front=front.copy(),
                 pool_event_row=rows.copy(), css_event_row=rows.copy(), residual_event_row=rows.copy(),
                 pool_supported=np.ones(n, bool), pool_fallback=np.zeros(n, bool), pool_donor=np.arange(n),
                 residual_original=original.copy(), residual_original_time_accepted=original * .8,
                 residual_original_steps=(original > 0).astype(int), residual_donor=donor_energy,
                 residual_donor_steps=(donor_energy > 0).astype(int), css_original=original.copy(), css_donor=donor_energy.copy())
        for outcome, actual, expected in (("energy", donor_energy, 50.), ("empty", donor_energy == 0, .25), ("weak", donor_energy <= 10, .5)):
            d["pool_actual_" + outcome] = actual.astype(float)
            d["pool_expected_" + outcome] = np.full(n, expected)
            d["pool_variance_" + outcome] = np.full(n, 4. if outcome == "energy" else expected * (1 - expected))
        for stage, energy in (("original", original), ("donor", donor_energy), ("geometric", donor_energy * .9), ("crystal", donor_energy * .95)):
            d[f"css_{stage}_centroid_energy"] = energy.copy()
            for coordinate, scale in (("radial", 10.), ("tangent", 5.)):
                d[f"css_{stage}_centroid_{coordinate}"] = np.where(energy > 0, np.arange(n) * scale, np.nan)
        for arm in ("stockref", "control", *dl.ARMS):
            d[f"residual_{arm}_selected"] = a[prefix + arm + "_energy"] > 10
        for arm in dl.ARMS:
            mapped = d[f"css_{arm}_centroid_energy"]
            lost, rejected, kept = donor_energy - mapped, mapped * .1, mapped * .9
            d.update({f"residual_{arm}_mapped": mapped, f"residual_{arm}_lost": lost,
                      f"residual_{arm}_time_rejected": rejected, f"residual_{arm}_time_accepted": kept})
            expected = (donor_energy - original, -lost, -(rejected - original * .2),
                        (a[prefix + arm + "_energy"] - kept) - (control - original * .8), control - stock)
            a.update({prefix + arm + "_term_" + term: value for term, value in zip(dl.TERMS, expected)})
        diagnostics.append(d)
    d, old = diagnostics
    d.update(direction_used=used.copy(), direction_fallback=(~used).copy())
    for arm in dl.ARMS:
        a[arm + "_common_three_way_selected"] = (a[arm + "_energy"] > 10) & (a["production_" + arm + "_energy"] > 10) & (stock > 10)
    return a, d, old, c


@pytest.fixture(scope="module")
def inputs():
    return invented()


@pytest.fixture(scope="module")
def report(inputs):
    return dl.analyze(*inputs)


def literal_slots(values):
    bins = []
    for value in values:
        if value < dl.EP_EDGES[0]:
            bins.append(0)
        elif value > dl.EP_EDGES[-1]:
            bins.append(25)
        elif value == dl.EP_EDGES[-1]:
            bins.append(24)
        else:
            bins.append(next(i + 1 for i in range(24) if dl.EP_EDGES[i] <= value < dl.EP_EDGES[i + 1]))
    return np.array(bins)


def literal_count(fast, stock, group, selected):
    by_block = []
    fslot, sslot = literal_slots(fast), literal_slots(stock)
    for block in np.unique(group):
        delta = np.zeros(26, int)
        for i in range(len(fast)):
            if group[i] == block and selected[i]:
                delta[fslot[i]] += 1
                delta[sslot[i]] -= 1
        by_block.append(delta)
    by_block = np.asarray(by_block)
    return by_block.sum(axis=0), np.sqrt(len(by_block) * np.var(by_block, axis=0, ddof=1))


def literal_pair(fast, stock, group, fm, sm, statistic):
    full = statistic(fast[fm]) - statistic(stock[sm])
    leave = []
    for block in np.unique(group):
        f, s = fast[fm & (group != block)], stock[sm & (group != block)]
        if not len(f) or not len(s):
            return full, None
        leave.append(statistic(f) - statistic(s))
    if len(leave) < 2:
        return full, None
    leave = np.asarray(leave)
    return full, np.sqrt((len(leave) - 1) / len(leave) * np.sum((leave - leave.mean()) ** 2))


def test_public_saved_only_json_no_mutation(inputs, monkeypatch):
    before = copy.deepcopy(inputs)
    def forbidden(*args, **kwargs):
        pytest.fail("saved analysis attempted I/O, RNG or physics execution")
    for name in ("load", "loadtxt", "genfromtxt", "fromfile", "save", "savez"):
        monkeypatch.setattr(np, name, forbidden)
    for name in ("default_rng", "RandomState", "Generator", "SeedSequence", "seed", "random", "choice", "uniform", "normal"):
        monkeypatch.setattr(np.random, name, forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = dl.analyze(*inputs)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["queries"] == 16 and result["source_blocks"] == 7
    for actual, expected in zip(inputs, before):
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_flow_boundaries_and_full_zero_contribution_blocks():
    fast = np.r_[-.1, dl.EP_EDGES, 1.20001]
    stock = fast[::-1].copy()
    group = np.arange(len(fast)) // 3
    labels = np.arange(len(fast)) % 3
    count = dl.count_contributions(fast, stock, group, labels, 4)
    np.testing.assert_array_equal(dl._slots(fast), literal_slots(fast))
    for cell in range(4):
        expected, error = literal_count(fast, stock, group, labels == cell)
        np.testing.assert_array_equal(count["difference"][cell], expected)
        np.testing.assert_allclose(count["error"][cell], error)
    assert count["fast"][3].sum() == 0
    np.testing.assert_array_equal(count["error"][3], np.zeros(26))
    f, s = np.array([.9, .9, .9]), np.array([.8, .9, .9])
    full = dl.count_contributions(f, s, np.array([0, 1, 2]), np.array([0, 1, 1]), 2)
    assert full["error"][0, dl._slots(f[:1])[0]] == pytest.approx(1.)
    one = dl.count_contributions(f[:1], s[:1], np.array([0]), np.array([0]), 2)
    assert np.isnan(one["error"]).all()


def test_all_cells_additive_counts_migration_and_literal_errors(inputs, report):
    a, _, _, c = inputs
    group = a["event_row"] // 100
    for label, item in report["ep_counts"]["arms"].items():
        prefix = label.replace("new_", "") if label.startswith("new_") else label
        fast, stock = a[prefix + "_truth_ep"], a["stockref_truth_ep"]
        expected, error = literal_count(fast, stock, group, np.ones(len(group), bool))
        np.testing.assert_array_equal(item["global_counts"]["difference"], expected)
        np.testing.assert_allclose(item["global_counts"]["error"], error)
        assert len(item["partitions"]["cell"]["fast"]) == 96
        for partition in item["partitions"].values():
            assert partition["source_blocks"] == 7
            for key in ("fast", "stock", "difference"):
                np.testing.assert_array_equal(np.sum(partition[key], axis=0), item["global_counts"][key])
        for cell in range(96):
            expected, error = literal_count(fast, stock, group, a["cell_id"] == cell)
            np.testing.assert_array_equal(item["partitions"]["cell"]["difference"][cell], expected)
            np.testing.assert_allclose(item["partitions"]["cell"]["error"][cell], error)
        migration = np.zeros((26, 26), int)
        for s, f in zip(literal_slots(stock), literal_slots(fast)):
            migration[s, f] += 1
        np.testing.assert_array_equal(item["stock_to_arm"], migration)
    assert sum(np.sum(report["ep_counts"]["arms"]["control"]["partitions"]["radius"]["stock"], axis=1)) == len(group)
    assert np.any(c["entry_radius"] == 1000)


def test_stock_windows_inclusive_energy_same_momentum_five_terms(inputs, report):
    a = inputs[0]
    expected_counts = {"0.80_0.85": 2, "0.85_0.90": 2, "0.90_0.95": 2, "0.95_1.00": 2, "ge1": 3, "gt1p2": 1}
    for key, n in expected_counts.items():
        window = report["stock_ep_windows"][key]
        assert window["entries"] == n
        for label, item in window["arms"].items():
            if label == "control":
                continue
            assert sum(v["difference"] for v in item["energy_terms"].values()) == pytest.approx(item["energy"]["difference"], abs=1e-12)
            assert sum(v["difference"] for v in item["ep_terms"].values()) == pytest.approx(item["truth_ep"]["difference"], abs=1e-12)
    use = (a["stockref_truth_ep"] >= .8) & (a["stockref_truth_ep"] < .85)
    observed = report["stock_ep_windows"]["0.80_0.85"]["arms"]["new_geometric"]["energy"]
    assert observed["fast"] == np.mean(a["geometric_energy"][use])
    assert report["stock_ep_windows"]["gt1p2"]["arms"]["new_crystal"]["energy"]["error"] is None


@pytest.mark.parametrize("coordinate", ["radial", "tangent"])
def test_spatial_own_common_control_mask_tail_and_width_oracles(inputs, report, coordinate):
    a = inputs[0]
    group = a["event_row"] // 100
    values = {key: a[prefix + "_" + coordinate] for key, prefix in (("new", "geometric"), ("production", "production_geometric"), ("stock", "stockref"), ("control", "control"))}
    own = {key: a[prefix + "_energy"] > 10 for key, prefix in (("new", "geometric"), ("production", "production_geometric"), ("stock", "stockref"), ("control", "control"))}
    common = own["new"] & own["production"] & own["stock"]
    assert np.any(common & ~own["control"])
    item = report["spatial"]["all"]["arms"]["geometric"][coordinate]
    modes = (("own", own), ("common_three_way", {k: common for k in values if k != "control"}),
             ("common_four_way", {k: common & own["control"] for k in values}))
    assert item["control_unselected_in_three_way"] == int((common & ~own["control"]).sum())
    for mode, masks in modes:
        for key in masks:
            stats = item[mode]["arms"][key]
            selected = values[key][masks[key]]
            assert stats["moments"]["n"] == len(selected)
            assert stats["moments"]["std"] == pytest.approx(np.std(selected))
            np.testing.assert_allclose(stats["abs_quantiles"], np.quantile(np.abs(selected), [.5, .9, .95, .99]))
            assert stats["abs_max"] == np.abs(selected).max()
            split = dl.TAILS[coordinate][1]
            central = np.abs(selected) <= split
            parts = stats["variance_parts"]
            assert parts["central"] == pytest.approx(np.sum((selected[central] - selected.mean()) ** 2) / len(selected))
            assert parts["tail"] == pytest.approx(np.sum((selected[~central] - selected.mean()) ** 2) / len(selected))
            assert parts["central"] + parts["tail"] == pytest.approx(np.var(selected))
        for fast, stock in (("new", "stock"), ("production", "stock"), ("control", "stock"), ("new", "production")):
            if fast not in masks:
                continue
            contrast = item[mode]["contrasts"][fast + "_minus_" + stock]
            difference, error = literal_pair(values[fast], values[stock], group, masks[fast], masks[stock], np.std)
            assert contrast["moments"]["difference"]["std"] == pytest.approx(difference)
            assert contrast["moments"]["error"]["std"] == pytest.approx(error)
            for index, threshold in enumerate(dl.TAILS[coordinate]):
                statistic = lambda v: np.mean(np.abs(v) > threshold)
                difference, error = literal_pair(values[fast], values[stock], group, masks[fast], masks[stock], statistic)
                assert contrast["tails"]["difference"][index] == pytest.approx(difference)
                assert contrast["tails"]["error"][index] == pytest.approx(error)
    assert {"disk0", "disk1", "direction_used", "direction_fallback", "disk1_radius_0_450", "cell_95"} <= report["spatial"].keys()


def test_tail_boundary_empty_selection_and_unselected_blocks():
    y = np.array([100., 200., 400., 401., np.nan])
    mask = np.array([True, True, True, True, False])
    group = np.array([0, 0, 1, 1, 2])
    stats, deleted = dl._tails(y, mask, group, np.array([100., 200., 400.]))
    np.testing.assert_array_equal(stats["counts"], [3, 2, 1])
    np.testing.assert_array_equal(deleted[2], stats["fraction"])
    singleton = dl._tails(y, np.array([True, False, False, False, False]), group, np.array([100.]))[0]
    assert np.isnan(singleton["error"]).all()
    empty = dl._description(y, np.zeros(5, bool), 200.)
    assert empty["variance_parts"]["total"] is None
    assert empty["abs_quantiles"] is None
    count, leave = dl._tails(np.array([]), np.array([], bool), np.array([], int), np.array([100.]))
    assert count["n"] == 0 and leave.shape == (0, 1)
    assert np.isnan(count["error"]).all()


def test_css_own_positive_and_identical_threeway_original_blocks(inputs, report):
    a, d, old, _ = inputs
    group = a["event_row"] // 100
    for stage in ("donor", *dl.ARMS):
        item = report["css_stages"]["all"]["stages"][stage]
        sources = dict(new=(d, stage), production=(old, stage), original=(d, "original"))
        own = {key: source[f"css_{name}_centroid_energy"] > 0 for key, (source, name) in sources.items()}
        common = own["new"] & own["production"] & own["original"]
        for coordinate in dl.TAILS:
            values = {key: source[f"css_{name}_centroid_{coordinate}"] for key, (source, name) in sources.items()}
            for mode, masks in (("own", own), ("common_three_way", {key: common for key in sources})):
                for key in sources:
                    stats = item[coordinate][mode]["arms"][key]["moments"]
                    assert stats["n"] == masks[key].sum()
                    assert stats["std"] == pytest.approx(np.std(values[key][masks[key]]))
                    assert stats["correlation"] == pytest.approx(np.corrcoef(values[key][masks[key]], a["direction_" + coordinate][masks[key]])[0, 1])
                difference, error = literal_pair(values["new"], values["original"], group, masks["new"], masks["original"], np.std)
                contrast = item[coordinate][mode]["contrasts"]["new_minus_original"]["moments"]
                assert contrast["difference"]["std"] == pytest.approx(difference)
                assert contrast["error"]["std"] == pytest.approx(error)


def test_fallback_census_retains_empty_weak_negative_and_missing_counts(inputs, report):
    a, d, _, _ = inputs
    mask = a["direction_fallback"]
    census = report["fallback"]
    assert census["entries"] == census["negative_tangent_queries"] == 6
    assert census["capped_counts_available"] is False
    assert census["queries"]["direction_count"] is None
    assert len(census["queries"]["event_row"]) == 6
    assert len([key for key in census["regions"] if key.startswith("cell_")]) == 96
    assert sum(census["regions"][f"cell_{i:02d}"]["entries"] for i in range(96)) == 6
    item = census["regions"]["all"]
    assert item["outcomes"]["new_geometric"]["weak_le10"] > 0
    assert item["outcomes"]["new_geometric"]["zero"] > 0
    for outcome in ("energy", "empty", "weak"):
        pool = item["pool"]["new"][outcome]
        assert pool["actual_total"] == d["pool_actual_" + outcome][mask].sum()
        assert pool["expected_total"] == d["pool_expected_" + outcome][mask].sum()
        assert pool["conditional_draw_sd_total"] == pytest.approx(np.sqrt(d["pool_variance_" + outcome][mask].sum()))
        assert pool["conditional_draw_sd_mean"] == pytest.approx(pool["conditional_draw_sd_total"] / 6)
        assert "error" in pool["actual_minus_expected"]


def test_optional_saved_counts_and_empty_regions(inputs):
    a, d, old, c = copy.deepcopy(inputs)
    c["sampling_direction_count"] = np.where(a["direction_used"], 8, np.arange(len(a["event_row"])) % 8)
    dl._validate(a, d, old, c)
    census = dl._fallback(a, d, old, c, dl._regions(a, c))
    assert census["capped_counts_available"] is True
    np.testing.assert_array_equal(census["regions"]["all"]["capped_count_histogram_0_to_7"], [0, 0, 1, 1, 1, 1, 1, 1])
    assert census["regions"]["cell_00"]["entries"] == 0
    assert census["regions"]["cell_00"]["pool"]["new"]["weak"]["actual_minus_expected"]["error"] is None


@pytest.mark.parametrize("fault", ["full_ids", "duplicate_ids", "full_rows", "query_rows", "unsorted_rows", "entry_rows", "production_rows", "production_p", "production_control", "production_stock", "cell", "cell_disk", "flag_overlap", "flag_type", "flag_alignment", "unsupported", "donor", "fallback_donor", "common", "common_type", "ep", "production_ep", "term_cancel", "term_nonfinite", "momentum", "selected_nonfinite", "positive_css", "partition", "original", "selected", "probability", "variance", "actual_empty", "nonfront", "negative_energy", "shape", "capped_count"])
def test_malformed_alignment_flags_invariants_and_identities_fail(inputs, fault):
    a, d, old, c = copy.deepcopy(inputs)
    if fault == "full_ids":
        old["full_ids"]["event"][-1] += 1
    elif fault == "duplicate_ids":
        c["full_ids"][-1] = c["full_ids"][-2]
    elif fault == "full_rows":
        c["event_row"][-1] += 1
    elif fault == "query_rows":
        old["query_event_rows"][0] += 1
    elif fault == "unsorted_rows":
        a["event_row"][1] = a["event_row"][0]
    elif fault == "entry_rows":
        c["entry_event_row"] = c["entry_event_row"] + 1
    elif fault == "production_rows":
        a["production_event_row"][0] += 1
    elif fault == "production_p":
        a["production_momentum"][0] += 1
    elif fault in ("production_control", "production_stock"):
        a["production_" + ("control" if fault.endswith("control") else "stockref") + "_radial"][1] += 1
    elif fault == "cell":
        a["cell_id"][0] = a["production_cell_id"][0] = 96
    elif fault == "cell_disk":
        c["entry_disk"][0] = 1
    elif fault == "flag_overlap":
        a["direction_used"][-1] = True
    elif fault == "flag_type":
        a["direction_used"] = a["direction_used"].astype(int)
    elif fault == "flag_alignment":
        d["direction_used"][0] = False
    elif fault == "unsupported":
        c["sampling_supported"][0] = False
    elif fault == "donor":
        d["pool_donor"][0] += 1
    elif fault == "fallback_donor":
        old["pool_donor"][-1] += 1
    elif fault == "common":
        a["geometric_common_three_way_selected"][1] = False
    elif fault == "common_type":
        a["geometric_common_three_way_selected"] = a["geometric_common_three_way_selected"].astype(int)
    elif fault in ("ep", "production_ep"):
        a[("production_" if fault.startswith("production") else "") + "geometric_truth_ep"][1] += .1
    elif fault == "term_cancel":
        a["geometric_term_donor"][1] += 1
        a["geometric_term_downstream"][1] -= 1
    elif fault == "term_nonfinite":
        a["geometric_term_donor"][0] = np.nan
    elif fault == "momentum":
        a["momentum"][0] = a["production_momentum"][0] = 0
    elif fault == "selected_nonfinite":
        a["geometric_radial"][1] = np.nan
    elif fault == "positive_css":
        d["css_donor_centroid_radial"][1] = np.nan
    elif fault == "partition":
        d["residual_geometric_lost"][1] += 1
    elif fault == "original":
        old["css_original_centroid_radial"][1] += 1
    elif fault == "selected":
        old["residual_geometric_selected"][1] = False
    elif fault == "probability":
        d["pool_expected_empty"][0] = 1.1
    elif fault == "variance":
        d["pool_variance_empty"][0] += 1
    elif fault == "actual_empty":
        d["pool_actual_empty"][0] = 0
    elif fault == "nonfront":
        c["geometric_energy"][-1] = 1
    elif fault == "negative_energy":
        c["control_energy"][-1] = -1
    elif fault == "shape":
        a["production_geometric_energy"] = a["production_geometric_energy"][:-1]
    else:
        c["sampling_direction_count"] = np.full(len(a["event_row"]), 8)
    with pytest.raises(ValueError):
        dl.analyze(a, d, old, c)


def test_variable_momentum_ep_terms_are_mean_of_query_ratios(inputs):
    a, d, old, c = copy.deepcopy(inputs)
    p = np.linspace(96., 108., len(a["event_row"]))
    a["momentum"] = p
    a["production_momentum"] = p.copy()
    c["entry_energy"] = np.sqrt(p ** 2 + .51099895 ** 2)
    for key in list(a):
        if key.endswith("_truth_ep"):
            a[key] = a[key.removesuffix("truth_ep") + "energy"] / p
    dl._validate(a, d, old, c)
    windows = dl._windows(a)
    mask = a["stockref_truth_ep"] >= 1.
    item = windows["ge1"]["arms"]["new_geometric"]
    for term in dl.TERMS:
        expected = np.mean(a["geometric_term_" + term][mask] / p[mask])
        assert item["ep_terms"][term]["difference"] == pytest.approx(expected)
    assert item["truth_ep"]["difference"] == pytest.approx(np.mean((a["geometric_energy"][mask] - a["stockref_energy"][mask]) / p[mask]))
    assert item["truth_ep"]["difference"] != pytest.approx(item["energy"]["difference"] / np.mean(p[mask]))


def test_unselected_nans_allowed_but_selected_control_nans_rejected(inputs):
    a, d, old, c = copy.deepcopy(inputs)
    for coordinate in dl.TAILS:
        a["geometric_" + coordinate][0] = np.nan
    dl._validate(a, d, old, c)
    regions = {"all": np.ones(len(a["event_row"]), bool)}
    dl._spatial(a, regions)
    common = a["geometric_common_three_way_selected"]
    row = np.flatnonzero(common & (a["control_energy"] > 10))[0]
    a["control_radial"][row] = a["production_control_radial"][row] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        dl._spatial(a, regions)


def test_empty_regions_and_sparse_undefined_deletions_are_visible(report):
    region = report["spatial"]["cell_00"]
    assert region["entries"] == region["source_blocks"] == 0
    for mode in ("own", "common_three_way"):
        stats = region["arms"]["geometric"]["radial"][mode]
        assert stats["arms"]["new"]["moments"]["n"] == 0
        assert stats["arms"]["new"]["moments"]["std"] is None
        assert stats["contrasts"]["new_minus_stock"]["tails"]["error"] == [None] * 3
    assert report["fallback"]["regions"]["cell_00"]["full_region_entries"] == 0


def test_all_empty_new_css_positive_masks_are_not_replaced_by_leading_masks(inputs):
    a, d, old, _ = copy.deepcopy(inputs)
    for stage in ("donor", *dl.ARMS):
        d[f"css_{stage}_centroid_energy"][:] = 0
        for coordinate in dl.TAILS:
            d[f"css_{stage}_centroid_{coordinate}"][:] = np.nan
    stages = dl._stages(a, d, old, {"all": np.ones(len(a["event_row"]), bool)})
    for stage in stages["all"]["stages"].values():
        for coordinate in dl.TAILS:
            common = stage[coordinate]["common_three_way"]
            assert all(arm["moments"]["n"] == 0 for arm in common["arms"].values())
            assert common["contrasts"]["new_minus_original"]["moments"]["error"]["std"] is None


def test_missing_control_centroid_keeps_original_threeway_and_separate_fourway(inputs):
    a, d, old, c = copy.deepcopy(inputs)
    absent = a['control_energy'] == 0
    for field in ('radial', 'tangent', 'time'):
        a['control_' + field][absent] = np.nan
        a['production_control_' + field][absent] = np.nan
    report = dl.analyze(a, d, old, c)
    for arm in dl.ARMS:
        common = a[arm + '_common_three_way_selected']
        for coordinate in dl.TAILS:
            item = report['spatial']['all']['arms'][arm][coordinate]
            assert item['common_three_way']['arms']['new']['moments']['n'] == common.sum()
            assert 'control' not in item['common_three_way']['arms']
            assert item['control_unselected_in_three_way'] > 0
            for value in item['common_four_way']['arms'].values():
                assert value['moments']['n'] == (common & (a['control_energy'] > 10)).sum()
