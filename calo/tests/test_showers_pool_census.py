"""Exact frozen-pool diagnostics against synthetic, independently enumerated pools."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from calofastsim import showers
from calofastsim.showers import compare, library as shower_library
from calofastsim.showers import placement_diagnostic as pd
from calofastsim.showers import pool_census as census
from calofastsim.showers import production_compare as pc
from calofastsim.showers import radius_trial as rt
from calofastsim.showers import residual_diagnostic as rd
from test_showers_production_compare import case  # noqa: F401 -- synthetic ROOT fixture
from test_showers_radius import make_entries, make_library, mixed  # noqa: F401
from test_showers_residual_diagnostic import (
    cli_fixture as residual_cli_fixture,
    prepare as prepare_saved,
    saved_only, saved_trial,  # noqa: F401 -- sampling is fixture construction ONLY
)


def prohibit_draws(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pool census attempted a random draw or sampling")

    for module in (showers, shower_library, pd, pc, rt, rd, compare, census):
        monkeypatch.setattr(module, "sample", forbidden, raising=False)
    for name in ("default_rng", "random", "rand", "randn", "choice", "seed"):
        monkeypatch.setattr(np.random, name, forbidden)


@pytest.fixture
def no_draws(monkeypatch):
    prohibit_draws(monkeypatch)


@pytest.fixture
def frozen(saved_only, monkeypatch):
    # saved_only already prohibits reconstruction imports. Install the random guard
    # only AFTER the shared saved_trial fixture has constructed its synthetic draw.
    prohibit_draws(monkeypatch)
    return saved_only


def dense_pools(library, entries):
    """Full row sorting with Python tuple tie-breaks; no production pool/bin helpers."""
    names = ("energy", "cosine", "radius")
    edges = [library["bins_" + name] for name in names]

    def coordinates(values):
        labels, outside = [], []
        for row in zip(*values):
            labels.append(tuple(min(max(sum(x >= edge for edge in e) - 1, 0), len(e) - 2)
                                for x, e in zip(row, edges)))
            outside.append(any(x < e[0] or x > e[-1] for x, e in zip(row, edges)))
        return np.asarray(labels, dtype=np.int64).reshape(-1, 3), np.asarray(outside, bool)

    qcell, outside = coordinates([entries[name] for name in names])
    lcell, _ = coordinates([library["entry_" + name] for name in names])
    labels = [(int(disk), *cell) for disk, cell in zip(library["entry_disk"], lcell)]
    occupied = sorted(set(labels))
    inverse = np.array([occupied.index(label) for label in labels], np.int64)
    n = len(qcell)
    eligible = np.full((n, 8), -1, np.int64)
    chosen = np.full(n, -1, np.int64)
    size, full_size, exact_size = (np.zeros(n, np.int64) for _ in range(3))
    fallback = outside.copy()
    for i, (disk, cell, radius) in enumerate(zip(entries["disk"], qcell, entries["radius"])):
        candidates = [j for j, label in enumerate(occupied) if label[0] == disk]
        if not candidates:
            fallback[i] = True
            continue
        chosen[i] = min(candidates, key=lambda j: (sum((a - b) ** 2 for a, b in
                                                       zip(cell, occupied[j][1:])), occupied[j]))
        rows = [j for j, label in enumerate(labels) if label == occupied[chosen[i]]]
        exact = tuple(cell) == occupied[chosen[i]][1:]
        full_size[i] = len(rows)
        size[i] = min(8, len(rows))
        exact_size[i] = len(rows) if exact else 0
        fallback[i] |= not exact
        nearest = sorted(rows, key=lambda j: (abs(library["entry_radius"][j] - radius), j))[:8]
        eligible[i, :len(nearest)] = sorted(nearest)
    return dict(eligible=eligible, chosen_cell=chosen,
                query_cell=np.column_stack((entries["disk"], qcell)),
                occupied=np.asarray(occupied, np.int64).reshape(-1, 4), library_cell=inverse,
                counts=np.bincount(inverse, minlength=len(occupied)), size=size,
                full_size=full_size, exact_size=exact_size, supported=chosen >= 0, fallback=fallback,
                outside=outside, substituted=(chosen >= 0) & (exact_size == 0))


def synthetic_draw(library, pools, donors=None):
    """Construct a saved payload deterministically, never invoking sample or RNG."""
    donor = pools["eligible"][:, 0].copy() if donors is None else np.asarray(donors, np.int64)
    lengths = np.array([0 if d < 0 else library["offsets"][d + 1] - library["offsets"][d]
                        for d in donor], np.int64)
    source = np.array([step for d in donor if d >= 0
                       for step in range(library["offsets"][d], library["offsets"][d + 1])], np.int64)
    return dict(donor=donor, supported=pools["supported"].copy(), fallback=pools["fallback"].copy(),
                radius_pool_size=pools["size"].copy(), bin_pool_size=pools["full_size"].copy(),
                offsets=np.r_[0, lengths.cumsum()], entry=np.repeat(np.arange(len(donor)), lengths),
                energy_birks=library["step_energy_birks"][source])


def assert_pool_reference(library, entries):
    expected = dense_pools(library, entries)
    draw = synthetic_draw(library, expected)
    actual = census.enumerate_pools(library, entries, draw)
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value, err_msg=name)
    return actual, draw


def test_dense_enumeration_deterministic_and_no_mutation(mixed, no_draws):
    library, entries = mixed
    before = deepcopy((library, entries))
    first, draw = assert_pool_reference(library, entries)
    second = census.enumerate_pools(library, entries, draw)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    for actual, expected in zip((library, entries), before):
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    assert first["supported"].sum() == 9
    np.testing.assert_array_equal(first["eligible"][3], -1)


def test_duplicate_runs_exact_ties_and_original_row_order(no_draws):
    library = make_library([490., 300., 100., 300., 400., 200., 300., 300., 300., 450.], empty=(1,))
    pools, _ = assert_pool_reference(library, make_entries([300., 350.]))
    np.testing.assert_array_equal(pools["eligible"][0], [1, 3, 4, 5, 6, 7, 8, 9])
    # Radius rank is only a selection criterion; the final pool is library-row ordered.
    np.testing.assert_array_equal(pools["eligible"][1], [0, 1, 3, 4, 6, 7, 8, 9])
    duplicate = make_library(np.full(13, 300.), empty=(0,))
    pools, _ = assert_pool_reference(duplicate, make_entries([300., 301.]))
    np.testing.assert_array_equal(pools["eligible"], np.tile(np.arange(8), (2, 1)))
    # Ties at a distinct-radius boundary also resolve in original row order.
    boundary = make_library([200., 400.] * 6)
    pools, _ = assert_pool_reference(boundary, make_entries([300.]))
    np.testing.assert_array_equal(pools["eligible"][0], np.arange(8))


def test_many_deterministic_duplicate_radii_match_full_dense_sort(no_draws):
    radius = 100. + ((np.arange(137) * 37) % 23) * 7.
    library = make_library(radius, disk=np.arange(137) % 2,
                           energy=np.where(np.arange(137) % 5 == 0, 125., 75.))
    entries = make_entries(np.tile(np.arange(90., 280., 2.5), 2),
                           disk=np.repeat([0, 1], 76), energy=np.tile([75., 125.], 76))
    assert_pool_reference(library, entries)


def test_selected_cell_fallback_not_global_radius_and_same_disk(no_draws):
    library = make_library([100., 490., 510., 520., 499.],
                           disk=[0, 0, 0, 0, 1], energy=[75., 125., 75., 75., 20.])
    entries = make_entries([499., 1200., 499., 499.], disk=[0, 0, 9, 1], energy=[20., 75., 20., 20.])
    pools, _ = assert_pool_reference(library, entries)
    np.testing.assert_array_equal(pools["eligible"][:, 0], [0, 2, -1, 4])
    np.testing.assert_array_equal(pools["fallback"], [True, True, True, False])
    np.testing.assert_array_equal(pools["exact_size"], [0, 2, 0, 1])
    np.testing.assert_array_equal(pools["full_size"], [1, 2, 0, 1])
    # Two equally distant occupied cells choose the lexicographically first cell,
    # not whichever donor happens to be earlier in the library.
    tied = make_library([100., 100.], energy=[125., 25.])
    pools, _ = assert_pool_reference(tied, make_entries([100.]))
    assert pools["eligible"][0, 0] == 1


@pytest.mark.parametrize("n", [1, 7, 8])
def test_k8_equals_entire_selected_pool_when_small(n, no_draws):
    library = make_library(np.arange(n) * 30. + 100., empty=(0,))
    pools, _ = assert_pool_reference(library, make_entries([101., 499.]))
    np.testing.assert_array_equal(pools["eligible"][:, :n], np.tile(np.arange(n), (2, 1)))
    np.testing.assert_array_equal(pools["eligible"][:, n:], -1)
    np.testing.assert_array_equal(pools["size"], pools["full_size"])


@pytest.mark.parametrize("mode", ["empty_query", "unsupported", "empty_donors"])
def test_empty_queries_unsupported_and_zero_step_donors(mode, no_draws):
    library = make_library([100., 200.], empty=(0, 1))
    entries = make_entries([] if mode == "empty_query" else [150., 170.])
    if mode == "unsupported":
        entries["disk"][:] = 9
    pools, draw = assert_pool_reference(library, entries)
    original = dict(entry=np.array([], np.int64), energy_birks=np.array([], float))
    saved = {arm + "_energy": np.zeros(len(entries["energy"]))
             for arm in ("stockref", "control", "geometric", "crystal")}
    report, arrays, donors = census.diagnose(library, entries, draw, original, saved)
    all_rows = report["regions"]["all"]
    assert all_rows["queries"] == len(entries["energy"])
    assert report["library_empty"] == report["library_weak"] == 2
    assert all_rows["supported"] == (2 if mode == "empty_donors" else 0)
    if mode == "unsupported":
        assert np.isnan(arrays["prob_empty"]).all()
        assert all_rows["outcomes"]["empty"]["expected_fraction"] is None
        assert all_rows["exposure_concentration"] is None
    if mode == "empty_donors":
        np.testing.assert_array_equal(arrays["prob_empty"], 1.)
        assert all_rows["outcomes"]["empty"]["conditional_draw_sd"] == 0
    np.testing.assert_allclose(donors["exposure_all"].sum(), pools["supported"].sum())
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("field", ["supported", "fallback", "radius_pool_size", "bin_pool_size"])
def test_saved_flags_and_pool_sizes_must_match(mixed, field, no_draws):
    library, entries = mixed
    draw = synthetic_draw(library, dense_pools(library, entries))
    draw[field][0] = not draw[field][0] if draw[field].dtype.kind == "b" else draw[field][0] + 1
    with pytest.raises(ValueError, match="pool_" + field):
        census.enumerate_pools(library, entries, draw)


@pytest.mark.parametrize("fault", ["outside_k8", "wrong_disk", "unsupported_has_donor", "supported_missing", "shape"])
def test_saved_donor_must_be_member_of_its_exact_pool(mixed, fault, no_draws):
    library, entries = mixed
    pools = dense_pools(library, entries)
    draw = synthetic_draw(library, pools)
    if fault == "outside_k8":
        draw["donor"][0] = next(i for i in range(len(library["entry_energy"]))
                                if pools["library_cell"][i] == pools["chosen_cell"][0]
                                and i not in pools["eligible"][0])
    elif fault == "wrong_disk":
        draw["donor"][0] = 1
    elif fault == "unsupported_has_donor":
        draw["donor"][3] = 0
    elif fault == "supported_missing":
        draw["donor"][0] = -1
    else:
        draw["donor"] = draw["donor"][:, None]
    with pytest.raises(ValueError, match="saved donor is outside eligible pool"):
        census.enumerate_pools(library, entries, draw)


@pytest.mark.parametrize("field", ["entry_cell", "entry_outside_bins"])
def test_recomputed_library_cells_and_outside_flags_are_validated(mixed, field, no_draws):
    library, entries = mixed
    draw = synthetic_draw(library, dense_pools(library, entries))
    library[field].flat[0] = (not library[field].flat[0] if library[field].dtype.kind == "b"
                             else library[field].flat[0] + 1)
    with pytest.raises(ValueError, match="library_(cells|outside)"):
        census.enumerate_pools(library, entries, draw)


def test_upper_bin_edge_is_inside_and_nextafter_is_outside(no_draws):
    library = make_library([1000., 1000.], energy=[150., 151.])
    entries = make_entries([1000., np.nextafter(1000., np.inf)], energy=[150., 150.])
    pools, _ = assert_pool_reference(library, entries)
    np.testing.assert_array_equal(pools["fallback"], [False, True])
    np.testing.assert_array_equal(pools["exact_size"], [2, 2])
    np.testing.assert_array_equal(library["entry_outside_bins"], [False, True])


def test_outcomes_sum_both_disks_and_distinguish_empty_zero_weak_threshold(no_draws):
    library = make_library([100., 200., 300., 350., 400.], empty=(0,))
    library["step_energy_birks"][:] = [0., 0., 4., 6., 6., 5., 0., 7.]
    result = census.donor_outcomes(library)
    np.testing.assert_array_equal(result["energy"], [0., 0., 10., 11., 7.])
    np.testing.assert_array_equal(result["empty"], [True, False, False, False, False])
    np.testing.assert_array_equal(result["weak"], [True, True, True, False, True])
    # Changing deposit-disk labels must not discard energy from the other disk.
    library["step_disk"][:] = 1
    for name, value in result.items():
        np.testing.assert_array_equal(census.donor_outcomes(library)[name], value)


@pytest.mark.parametrize("fault", ["offset_shape", "offset_start", "offset_decreasing", "length", "negative", "nan", "inf"])
def test_invalid_library_offsets_and_birks_rejected(fault, no_draws):
    library = make_library([100., 200.])
    if fault == "offset_shape":
        library["offsets"] = library["offsets"][:-1]
    elif fault == "offset_start":
        library["offsets"][0] = 1
    elif fault == "offset_decreasing":
        library["offsets"][:] = [0, 3, 2]
    elif fault == "length":
        library["step_energy_birks"] = library["step_energy_birks"][:-1]
    else:
        library["step_energy_birks"][0] = {"negative": -1., "nan": np.nan, "inf": np.inf}[fault]
    with pytest.raises(ValueError, match="invalid library (offsets|Birks energies)"):
        census.donor_outcomes(library)


def test_probabilities_ignore_padding_and_keep_unsupported_nan(no_draws):
    pools = dict(eligible=np.array([[1, 2, -1, -1, -1, -1, -1, -1], [-1] * 8,
                                  [0, -1, -1, -1, -1, -1, -1, -1]]),
                 size=np.array([2, 0, 1]), supported=np.array([True, False, True]))
    np.testing.assert_allclose(census.probabilities(np.array([100., 0., 10.]), pools), [5., np.nan, 100.])
    np.testing.assert_allclose(census.probabilities(np.array([True, False, True]), pools), [.5, np.nan, 1.])
    # Even NaN in row zero must not leak through the -1 padding slots.
    np.testing.assert_allclose(census.probabilities(np.array([np.nan, 1., 0.]), pools), [.5, np.nan, np.nan])


def independent_paired(fast, original, rows):
    delta = np.asarray(fast, float) - original
    groups = rows // 100
    leave = np.array([delta[groups != group].mean() for group in np.unique(groups)])
    error = np.sqrt((len(leave) - 1) / len(leave) * ((leave - leave.mean()) ** 2).sum())
    return dict(fast=float(np.mean(fast)), stock=float(np.mean(original)),
                difference=float(delta.mean()), error=float(error))


def test_outcome_expected_counts_conditional_sd_and_source_block_errors(no_draws):
    probability = np.array([0., .25, .5, 1.])
    observed = np.array([False, True, False, True])
    original = np.array([False, False, True, True])
    rows = np.array([1, 99, 200, 501])
    result = census.outcome_summary(probability, observed, original, rows)
    assert result["queries"] == 4 and result["deterministic_queries"] == 2
    assert result["expected_count"] == 1.75 and result["expected_fraction"] == .4375
    assert result["original"] == result["saved_draw"] == 2
    assert result["conditional_draw_sd"] == pytest.approx(np.sqrt(.4375))
    assert result["draw_minus_expected"] == .25
    assert result["draw_standardized_difference"] == pytest.approx(.25 / np.sqrt(.4375))
    assert result["expected_minus_original"] == pytest.approx(independent_paired(probability, original, rows))
    empty = census.outcome_summary(np.array([]), np.array([], bool), np.array([], bool), np.array([], int))
    assert empty["expected_count"] == empty["conditional_draw_sd"] == 0
    assert empty["expected_fraction"] is empty["draw_standardized_difference"] is None


def test_overlapping_pools_exposure_diversity_coverage_and_paired_source_rows(no_draws):
    library = make_library([100., 200., 300., 400.], empty=(0,))
    library["step_energy_birks"][:] = [0., 0., 4., 6., 6., 6.]
    entries = make_entries([150., 250., 350., 750.], disk=[0, 0, 0, 9])
    entries["event_row"][:] = [1, 99, 200, 301]
    pools = dense_pools(library, entries)
    draw = synthetic_draw(library, pools, [0, 2, 3, -1])
    outcomes = census.donor_outcomes(library)
    prob = {name: census.probabilities(values, pools) for name, values in outcomes.items()}
    original = dict(energy=np.array([0., 12., 4., 0.]), empty=np.array([True, False, False, True]),
                    weak=np.array([True, False, True, True]))
    cover = census.coverage(library, entries, pools)
    with patch.object(census, "paired_mean", wraps=census.paired_mean) as paired:
        result, exposure, selected = census.region_summary(np.ones(4, bool), library, entries,
            pools, outcomes, prob, original, draw, cover)
    assert result["queries"] == 4 and result["supported"] == 3 and result["unsupported"] == 1
    assert result["fallback"] == result["exact_cell_empty"] == 1
    assert result["unique_eligible_donors"] == 4 and result["unique_drawn_donors"] == 3
    assert result["eligible_empty"] == 1 and result["eligible_weak"] == 3
    assert result["maximum_draw_reuse"] == 1
    np.testing.assert_allclose(exposure, [.75] * 4)
    np.testing.assert_array_equal(selected, [1, 0, 1, 1])
    assert result["maximum_expected_exposure"] == .75
    assert result["exposure_concentration"] == .25
    assert result["exposure_by_outcome"] == dict(empty=.75, weak=2.25)
    # The three queries share IDENTICAL pools. Conditional independent draw
    # variance is still sum p_i(1-p_i), NOT the square of summed Bernoulli SDs.
    for name, expected in (("empty", .75), ("weak", 2.25)):
        outcome = result["outcomes"][name]
        assert outcome["expected_count"] == expected
        assert outcome["conditional_draw_sd"] == .75
        assert outcome["expected_minus_original"] == pytest.approx(
            independent_paired(prob[name][:3], original[name][:3], entries["event_row"][:3]))
    assert paired.call_count == 4
    for call in paired.call_args_list:
        np.testing.assert_array_equal(call.args[2], [0, 0, 2])
    radius = result["coverage"]["radius_delta"]
    assert radius["best"]["mean"] == 50.
    assert radius["worst"]["mean"] == pytest.approx(650 / 3)
    assert radius["mean"]["mean"] == pytest.approx(350 / 3)
    assert result["coverage"]["angle_deg"]["mean"]["mean"] == 0.
    np.testing.assert_array_equal(cover["radius_delta"][0, :4], [-50., 50., 150., 250.])
    assert np.isnan(cover["radius_delta"][:, 4:]).all()
    assert np.isnan(cover["angle_deg"][-1]).all()
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_direction_coverage_rotates_donor_about_disk_origin(no_draws):
    library = make_library([100., 200.])
    library["geometry_disk_origin_x"][:] = 10.
    library["geometry_disk_origin_y"][:] = 20.
    library["entry_position"][:, :2] = [[110., 20.], [210., 20.]]
    library["entry_direction"][:] = [[1., 0., 0.], [0., 1., 0.]]
    entries = make_entries([150.])
    entries["position"][:, :2] = [10., 170.]
    entries["direction"][:] = [0., 1., 0.]
    pools = dense_pools(library, entries)
    cover = census.coverage(library, entries, pools)
    np.testing.assert_allclose(cover["angle_deg"][0, :2], [0., 90.], atol=1e-12)
    np.testing.assert_array_equal(cover["radius_delta"][0, :2], [-50., 50.])


def test_cell_census_targets_include_query_only_and_library_only_cells(mixed, no_draws):
    library, entries = mixed
    pools = dense_pools(library, entries)
    draw = synthetic_draw(library, pools)
    original = dict(entry=np.array([0, 0, 2]), energy_birks=np.array([4., 7., 0.]))
    saved = {arm + "_energy": np.zeros(len(entries["energy"]))
             for arm in ("stockref", "control", "geometric", "crystal")}
    report, arrays, donors = census.diagnose(library, entries, draw, original, saved)
    qlabels = [tuple(row) for row in pools["query_cell"]]
    dlabels = [tuple(pools["occupied"][i]) for i in pools["library_cell"]]
    records = {tuple(record["cell"]): record for record in report["cells"]}
    assert set(records) == set(qlabels) | set(dlabels)
    for label, record in records.items():
        assert record["queries"] == qlabels.count(label)
        assert record["donors"] == dlabels.count(label)
        assert record["query_incident"]["radius"]["count"] == record["queries"]
        assert record["donor_incident"]["radius"]["count"] == record["donors"]
        indices = [i for i, value in enumerate(qlabels) if value == label]
        assert record["original_empty"] == sum(i not in (0, 2) for i in indices)
        assert record["original_weak"] == sum(i != 0 for i in indices)
        assert record["fallback_queries"] == sum(pools["fallback"][i] for i in indices)
    assert any(r["queries"] and not r["donors"] for r in records.values())
    assert any(r["donors"] and not r["queries"] for r in records.values())
    queried = set(qlabels)
    counts = [dlabels.count(label) for label in queried]
    for target in (100, 200, 400):
        plan = report["coverage_targets"][str(target)]
        assert plan["queried_cells"] == len(queried)
        assert plan["existing_in_queried_cells"] == sum(counts)
        assert plan["cells_below_target"] == sum(n < target for n in counts)
        assert plan["additional_donors"] == sum(max(0, target - n) for n in counts)
        assert plan["binomial_worst_case_se"] == pytest.approx(.5 / np.sqrt(target))
    np.testing.assert_array_equal(arrays["event_row"], entries["event_row"])
    assert donors["exposure_all"].sum() == pytest.approx(pools["supported"].sum())


@pytest.mark.parametrize("saved_trial", ["normal", "one_unsupported", "all_unsupported", "zero_nonempty"], indirect=True)
def test_saved_only_diagnose_preserves_full_population_and_supported_denominators(frozen):
    t = frozen
    prepared = prepare_saved(t)
    before = rt.fingerprint(t["draw"])
    report, arrays, donors = census.diagnose(t["case"]["library"], prepared["entries"],
                                            t["draw"], prepared["original"], t["saved"])
    assert rt.fingerprint(t["draw"]) == before
    np.testing.assert_array_equal(arrays["event_row"], [0, 1])
    assert report["regions"]["all"]["queries"] == 2
    for arm in ("stockref", "control", "geometric", "crystal"):
        # The positive non-query event remains saved, but is not in the pool denominator.
        assert t["saved"][arm + "_energy"][-1] == 999
        assert report["regions"]["all"]["saved_selected"][arm] == 1
    supported = t["draw"]["supported"]
    for outcome in ("empty", "weak"):
        stats = report["regions"]["all"]["outcomes"][outcome]
        assert stats["queries"] == int(supported.sum())
        assert np.isnan(arrays["prob_" + outcome][~supported]).all()
    assert donors["exposure_all"].sum() == pytest.approx(supported.sum())
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_regions_have_strict_radius_inclusive_energy_and_cosine_boundaries(no_draws):
    library = make_library([100., 100.], disk=[0, 1], energy=[125., 125.])
    entries = make_entries([449., 450., 449., 449., 449., 449., 449.],
                           disk=[0, 0, 1, 1, 1, 1, 1], energy=[95., 95., 95., 95., 94., 95., 95.])
    entries["cosine"][:] = [.7, .7, np.nextafter(.7, 0.), .7, .8, 0., 1.]
    draw = synthetic_draw(library, dense_pools(library, entries))
    original = dict(entry=np.array([], int), energy_birks=np.array([], float))
    # Deliberately non-contiguous source rows catch accidental query-index selection.
    entries["event_row"][:] = [1, 3, 4, 6, 7, 8, 9]
    saved = {arm + "_energy": np.array([999., 11., 999., 0., 11., 999., 11., 0., 0., 11.])
             for arm in ("stockref", "control", "geometric", "crystal")}
    report, _, _ = census.diagnose(library, entries, draw, original, saved)
    for name, count in (("inner_highE_disk0", 1), ("inner_highE_disk1", 4),
                        ("inner_highE_disk0_coslow", 0), ("inner_highE_disk0_coshigh", 1),
                        ("inner_highE_disk1_coslow", 2), ("inner_highE_disk1_coshigh", 2)):
        assert report["regions"][name]["queries"] == count
    assert report["regions"]["all"]["saved_selected"]["geometric"] == 4
    assert report["regions"]["inner_highE_disk1_coshigh"]["saved_selected"]["geometric"] == 2


def pool_cli_fixture(t, tmp_path, monkeypatch):
    trial, baseline, residual = residual_cli_fixture(t, tmp_path, monkeypatch)
    root = Path(census.__file__).resolve().parents[1]
    manifest = json.loads((trial / "preparation.json").read_text())
    manifest["calo_source_sha256"]["showers/library.py"] = rt.digest(root / "showers/library.py")
    pc._json(trial / "preparation.json", manifest)
    report = json.loads((trial / "numbers.json").read_text())
    report["manifest"] = manifest
    pc._json(trial / "numbers.json", report)
    # Residual plot bytes are irrelevant to this CLI's input receipt.
    with patch.object(rd, "plot_report"):
        rd.main()
    out = tmp_path / "pool_census"
    monkeypatch.setattr(sys, "argv", ["pool_census", "--tables", str(t["case"]["path"]),
        "--library", str(tmp_path / "library.npz"), "--trial", str(trial),
        "--baseline", str(baseline), "--residual", str(residual), "--out", str(out)])
    return trial, baseline, residual, out


def test_cli_saved_only_hashes_archives_plot_and_overwrite_guard(frozen, tmp_path, monkeypatch, capsys):
    trial, baseline, residual, out = pool_cli_fixture(frozen, tmp_path, monkeypatch)
    capsys.readouterr()
    paths = list(trial.iterdir()) + list(baseline.iterdir()) + list(residual.iterdir())
    before = {path: rt.digest(path) for path in paths}
    with patch.object(census, "diagnose", wraps=census.diagnose) as diagnostic:
        census.main()
    assert diagnostic.call_count == 1
    report = json.loads((out / "numbers.json").read_text())
    summary_keys = ("library_donors", "library_empty", "library_weak", "regions", "coverage_targets", "caveat")
    assert json.loads((out / "summary.json").read_text()) == {k: report[k] for k in summary_keys}
    for record in report["inputs"].values():
        assert record["sha256"] == rt.digest(Path(record["path"]))
    assert report["inputs"]["clusters"]["sha256"] == before[baseline / "clusters.npz"]
    assert report["inputs"]["clusters.npz"]["sha256"] == before[trial / "clusters.npz"]
    assert "showers/pool_census.py" in report["source_sha256"]
    assert report["draw_fingerprint"] == rt.fingerprint(frozen["draw"])
    assert report["census"]["events"] == 3
    assert report["census"]["non_front_entry_positive_events"] == 1
    assert {path: rt.digest(path) for path in before} == before
    pools = rd.archive(out / "pools.npz")
    donors = rd.archive(out / "donors.npz")
    np.testing.assert_array_equal(pools["event_row"], [0, 1])
    assert pools["eligible"].shape == (2, 8)
    assert donors["exposure_all"].sum() == 2
    assert (out / "pool_expectations.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "NaN" not in (out / "numbers.json").read_text()
    assert json.loads(capsys.readouterr().out) == dict(queries=2,
        library_donors=len(frozen["case"]["library"]["entry_energy"]), output=str(out))
    with pytest.raises(ValueError, match="output must be new or empty"):
        census.main()


@pytest.mark.parametrize("name", ["tables", "library", "clusters", "preparation.json", "draw.npz", "clusters.npz", "css_arrays.npz", "numbers.json"])
def test_cli_checks_every_saved_input_hash_before_diagnosis(frozen, tmp_path, monkeypatch, name):
    _, _, residual, out = pool_cli_fixture(frozen, tmp_path, monkeypatch)
    receipt = json.loads((residual / "numbers.json").read_text())
    path = Path(receipt["inputs"][name]["path"])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with patch.object(census, "diagnose") as diagnostic, pytest.raises(ValueError, match="residual input hash mismatch: " + name):
        census.main()
    diagnostic.assert_not_called()
    assert not out.exists()


@pytest.mark.parametrize("fault", ["neighbors", "seed", "reco_seed", "source", "residual_arrays"])
def test_cli_fixed_trial_source_and_residual_array_guards(frozen, tmp_path, monkeypatch, fault):
    trial, _, residual, out = pool_cli_fixture(frozen, tmp_path, monkeypatch)
    if fault == "residual_arrays":
        arrays = rd.archive(residual / "arrays.npz")
        arrays["event_row"][0] += 1
        np.savez_compressed(residual / "arrays.npz", **arrays)
        message = "residual array mismatch: event_row"
    else:
        path = trial / "preparation.json"
        manifest = json.loads(path.read_text())
        if fault == "source":
            manifest["calo_source_sha256"]["showers/library.py"] = "stale"
            message = "sampler source changed since K8 trial"
        else:
            manifest[fault] += 1
            message = "not the fixed K8 trial"
        pc._json(path, manifest)
        receipt = json.loads((residual / "numbers.json").read_text())
        receipt["inputs"]["preparation.json"]["sha256"] = rt.digest(path)
        pc._json(residual / "numbers.json", receipt)
    with patch.object(census, "diagnose") as diagnostic, pytest.raises(ValueError, match=message):
        census.main()
    diagnostic.assert_not_called()
    assert not out.exists()


def test_incident_coordinates_use_disk_origins_local_components_and_unknown_nan(no_draws):
    library = make_library([100., 100.], disk=[0, 1])
    library["geometry_disk_origin_x"][:] = [10., -20.]
    library["geometry_disk_origin_y"][:] = [20., 30.]
    entries = make_entries([100., 200., 100., 400., 0., 100.], disk=[0, 0, 1, 1, 0, 9])
    entries["position"][:, :2] = [[110., 20.], [10., 220.], [-120., 30.],
                                  [-20., -370.], [10., 20.], [999., 999.]]
    entries["direction"][:] = [[1., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                               [0., -1., 0.], [.6, .8, 0.], [0., 0., 1.]]
    result = census.incident_coordinates(library, entries)
    np.testing.assert_allclose(result["azimuth_deg"], [0., 90., 180., -90., 0., np.nan])
    np.testing.assert_allclose(result["direction_radial"], [1., 0., 0., 1., .6, np.nan], atol=1e-15)
    np.testing.assert_allclose(result["direction_tangent"], [0., -1., -1., 0., .8, np.nan], atol=1e-15)
    for name in ("energy", "cosine", "radius"):
        np.testing.assert_array_equal(result[name], entries[name])


def test_large_cells_targets_do_not_charge_already_sufficient_cells(no_draws):
    energy = np.repeat([25., 75., 125.], [110, 210, 410])
    library = make_library(np.full(len(energy), 100.), energy=energy)
    entries = make_entries([100.] * 4, disk=[0, 0, 0, 1], energy=[25., 75., 125., 75.])
    draw = synthetic_draw(library, dense_pools(library, entries))
    original = dict(entry=np.array([], int), energy_birks=np.array([], float))
    saved = {arm + "_energy": np.zeros(4) for arm in ("stockref", "control", "geometric", "crystal")}
    report, _, _ = census.diagnose(library, entries, draw, original, saved)
    for target, below, additional in ((100, 1, 100), (200, 2, 290), (400, 3, 880)):
        plan = report["coverage_targets"][str(target)]
        assert plan["queried_cells"] == 4
        assert plan["existing_in_queried_cells"] == 730
        assert plan["cells_below_target"] == below
        assert plan["additional_donors"] == additional


def test_k8_expectation_differs_from_full_cell_not_hidden_by_average(no_draws):
    library = make_library(np.arange(100., 200., 10.), empty=(8, 9))
    entries = make_entries([100., 100., 190.])
    pools, draw = assert_pool_reference(library, entries)
    outcomes = census.donor_outcomes(library)
    prob = {key: census.probabilities(value, pools) for key, value in outcomes.items()}
    original = dict(energy=np.zeros(3), empty=np.ones(3, bool), weak=np.ones(3, bool))
    report, exposure, _ = census.region_summary(np.ones(3, bool), library, entries, pools,
        outcomes, prob, original, draw, census.coverage(library, entries, pools))
    np.testing.assert_array_equal(prob["empty"], [0., 0., .25])
    assert report["outcomes"]["empty"]["expected_count"] == .25
    assert report["outcomes"]["empty"]["full_cell_expected_count"] == pytest.approx(.6)
    # Expected number of unique draws uses per-donor nonselection products,
    # distinct from union size, actual draws, and inverse concentration.
    expected_unique = sum(1 - np.prod([1 - 1 / pools["size"][i]
                                      for i in range(3) if donor in pools["eligible"][i]])
                          for donor in range(10))
    assert report["expected_unique_drawn"] == pytest.approx(expected_unique)
    assert report["unique_eligible_donors"] == 10
    assert exposure.sum() == 3


@pytest.mark.parametrize("fault", ["energy", "empty"])
def test_diagnose_rejects_saved_payload_inconsistent_with_donor(frozen, fault):
    prepared = prepare_saved(frozen)
    draw = deepcopy(frozen["draw"])
    if fault == "energy":
        draw["energy_birks"][0] += 1.
    else:
        # Keep saved energy intact and corrupt only the zero-step classification.
        draw["offsets"][1] = draw["offsets"][0]
    with pytest.raises(ValueError, match="saved_draw_" + fault):
        census.diagnose(frozen["case"]["library"], prepared["entries"], draw,
                         prepared["original"], frozen["saved"])
