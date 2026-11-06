"""Saved-array regression checks; photons sharing an event are never independent."""
import copy
import json

import numpy as np
import pytest

from beamfastsim.flash_response_compare import compare, main


def sample(*, n=4, parents=(0, 0, 1, 1), energy=None, response=(2., 2., 0., 0.),
           status=None, known=None, residual=None):
    parents = np.asarray(parents, int)
    m = len(parents)
    d = dict(event_run=np.ones(n, np.uint64), event_subrun=(np.arange(n) // 2).astype(np.uint64),
             event_event=np.arange(1, n + 1, dtype=np.uint64), event_ordinal=np.arange(n, dtype=np.uint64),
             input_event_count=np.array(n, np.uint64), product=np.ones(m, np.uint64),
             key=np.arange(m, dtype=np.uint64), pre_energy_mev=np.full(m, .5) if energy is None else np.array(energy),
             photon_assigned_ionizing_mev=np.array(response, float),
             frontier_status=np.ones(m, np.int8) if status is None else np.array(status, np.int8),
             photon_response_known=np.ones(m, bool) if known is None else np.array(known, bool))
    for field in ("run", "subrun", "event"):
        d[field] = d["event_" + field][parents]
    d["event_owned_ionizing_mev"] = np.bincount(parents, weights=d["photon_assigned_ionizing_mev"], minlength=n)
    d["event_residual_ionizing_mev"] = np.zeros(n) if residual is None else np.array(residual, float)
    d["event_unknown_ionizing_mev"] = np.zeros(n)
    d["event_tracker_ionizing_mev"] = d["event_owned_ionizing_mev"] + d["event_residual_ionizing_mev"]
    return d


def empty(**kwargs):
    return sample(parents=(), response=(), **kwargs)


def test_paired_totals_and_unbiased_mean_se():
    low = sample(residual=[1, 2, 3, 4])
    high = copy.deepcopy(low)
    for field in ("photon_assigned_ionizing_mev", "event_tracker_ionizing_mev", "event_owned_ionizing_mev",
                  "event_residual_ionizing_mev", "event_unknown_ionizing_mev"):
        high[field] *= 2
    out = compare(low, high)
    for field in ("raw", "owned", "residual"):
        stat = out["totals"][field]
        assert stat["high_over_low"] == 2
        assert stat["paired_delta_ratio_se"] == 0
        assert stat["subrun_cluster_ratio_se"] == 0
    assert out["totals"]["raw"]["low"]["event_mean_se"] == pytest.approx(np.std([5, 2, 3, 4], ddof=1) / 2)


def test_correlated_photons_share_event_cluster():
    out = compare(sample(), sample())["resolved_subset"]["low"][3]
    assert out["mean_edep_mev_per_photon"] == 1
    assert out["event_cluster_ratio_se"] == pytest.approx(np.sqrt(4 / 3 * 8) / 4)
    assert out["event_cluster_ratio_se"] != pytest.approx(np.std([2, 2, 0, 0], ddof=1) / 2)
    assert out["photon_events"] == 2
    assert out["nonzero_response_events"] == 1
    assert out["nonzero_response_photons"] == 2


def test_subrun_cluster_ratio_se_uses_cluster_residuals():
    low, high = empty(residual=[1, 2, 4, 8]), empty(residual=[2, 4, 4, 8])
    out = compare(low, high)["totals"]["residual"]
    r = 18 / 15
    expected = np.sqrt(2 * np.sum((np.array([6, 12]) - r * np.array([3, 12])) ** 2)) / 15
    assert out["subrun_cluster_ratio_se"] == pytest.approx(expected)
    assert out["paired_delta_ratio_se"] != pytest.approx(expected)


def test_all_zero_totals_are_undefined_not_nan():
    out = compare(empty(), empty())
    stat = out["totals"]["raw"]
    assert stat["high_over_low"] is None
    assert stat["low"]["event_mean_se"] == 0
    assert stat["low"]["concentration"]["effective_n"] is None
    assert stat["low"]["concentration"]["largest_event_share"] is None
    json.dumps(out, allow_nan=False)


def test_low_zero_with_positive_high():
    stat = compare(empty(), empty(residual=[1, 0, 0, 0]))["totals"]["residual"]
    assert stat["high_over_low"] is None
    assert stat["paired_delta_ratio_se"] is None
    assert stat["subrun_cluster_ratio_se"] is None


@pytest.mark.parametrize("field", ["event_run", "event_subrun", "event_event", "event_ordinal"])
def test_identity_mismatch(field):
    low, high = empty(), empty()
    high[field][0] += np.uint64(100)
    with pytest.raises(ValueError, match="alignment"):
        compare(low, high)


def test_reordering_or_missing_event_not_imputed():
    low, high = empty(), empty()
    for field in ("event_run", "event_subrun", "event_event", "event_ordinal"):
        high[field] = high[field][::-1]
    with pytest.raises(ValueError, match="alignment"):
        compare(low, high)
    with pytest.raises(ValueError, match="alignment"):
        compare(low, empty(n=3))


def test_full_unsigned_identity_and_exposure_mapping():
    d = sample()
    for field in ("event_run", "event_subrun", "event_event", "event_ordinal", "run", "subrun", "event", "product", "key"):
        d[field] += np.uint64(2**63 + 17)
    d["event_run"][0] = np.iinfo(np.uint64).max
    d["run"][:2] = np.iinfo(np.uint64).max
    out = compare(d, d)
    ids = out["totals"]["owned"]["low"]["concentration"]["top10_events"]
    assert ids["event_run"][0] == 2**64 - 1
    assert out["resolved_subset"]["low"][3]["response_sum_mev"] == 4
    json.dumps(out, allow_nan=False)


def test_mixed_signed_unsigned_keys_remain_exact():
    low, high = sample(), sample()
    for field in ("event_run", "event_subrun", "event_event", "event_ordinal", "run", "subrun", "event"):
        high[field] = high[field].astype(np.int64)
    assert compare(low, high)["events"] == 4


def test_residual_concentration_and_top_ten_identities():
    d = empty(n=12, residual=np.arange(1, 13))
    out = compare(d, d)["totals"]["residual"]["low"]["concentration"]
    assert out["largest_event_share"] == pytest.approx(12 / 78)
    assert out["top5_share"] == pytest.approx(50 / 78)
    assert out["top10_share"] == pytest.approx(75 / 78)
    assert out["effective_n"] == pytest.approx(78**2 / np.sum(np.arange(1, 13)**2))
    assert out["top10_events"]["event_event"] == list(range(12, 2, -1))


def test_fixed_bins_underflow_overflow_and_boundaries():
    energies = [0, .001, .01, .1, 1, 10, 100, 1000, 2000]
    d = sample(parents=[0] * 9, energy=energies, response=np.ones(9))
    bins = compare(d, d)["resolved_subset"]["low"]
    assert [b["photon_count"] for b in bins] == [1, 1, 1, 1, 1, 1, 1, 2]
    assert sum(b["response_sum_mev"] for b in bins) == 9


def test_support_gaps_not_zero_predictions():
    low = sample(parents=[0], energy=[.5], response=[2])
    high = sample(parents=[0, 1, 1], energy=[.5, 5, 5], response=[1, 3, 3])
    out = compare(low, high)["energy_only_prediction"]
    fwd = out["low_to_high"]
    assert fwd["unsupported_target_photons"] == 2
    assert fwd["support_gap_bins"] == [4]
    assert fwd["predicted_mev_by_bin"][4] is None
    assert fwd["full_subset_prediction_sum_mev"] is None
    assert fwd["supported_prediction_sum_mev"] == 2
    assert out["high_to_low"]["full_subset_prediction_sum_mev"] == 1


def test_zero_response_bin_is_unvalidated():
    low = sample(response=[0, 0, 0, 0])
    out = compare(low, sample())
    assert "not evidence of true zero" in out["resolved_subset"]["low"][3]["status"]
    pred = out["energy_only_prediction"]["low_to_high"]
    assert pred["zero_observed_response_unvalidated_bins"] == [3]
    assert pred["zero_observed_response_target_photons"] == 4


def test_unknown_covered_nonexit_excluded_not_inclusive():
    d = sample(parents=[0, 1, 2, 2, 3], response=[2, 7, 0, 0, 0], energy=[.5, .5, .5, .5, np.nan],
               status=[1, 1, 3, 2, 0], known=[True, False, False, False, False])
    out = compare(d, d)
    assert out["totals"]["owned"]["low"]["sum_mev"] == 9
    assert out["resolved_subset"]["low"][3]["photon_count"] == 1
    assert out["resolved_subset"]["low"][3]["response_sum_mev"] == 2
    assert out["exposure_categories"]["low"]["frontier_response_unknown"]["assigned_ionizing_mev"] == 7
    for name in ("covered", "non_exit", "unresolved_frontier"):
        assert out["exposure_categories"]["low"][name]["photons"] == 1
    assert "NOT inclusive" in out["scope"]


@pytest.mark.parametrize("field,value", [("event_tracker_ionizing_mev", np.nan),
                                         ("event_owned_ionizing_mev", -1),
                                         ("photon_assigned_ionizing_mev", np.inf),
                                         ("photon_assigned_ionizing_mev", -1),
                                         ("pre_energy_mev", np.nan), ("pre_energy_mev", -1)])
def test_invalid_numeric_arrays_rejected(field, value):
    d = sample()
    d[field][0] = value
    with pytest.raises(ValueError):
        compare(d, d)


def test_bad_photon_mapping_and_duplicate_identity():
    d = sample()
    d["event"][0] = 999
    with pytest.raises(ValueError, match="missing"):
        compare(d, d)
    d = sample()
    d["key"][1] = d["key"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        compare(d, d)
    d = sample()
    d["event"][:2] = 2
    with pytest.raises(ValueError, match="mapping"):
        compare(d, d)


def test_ordinal_cannot_disambiguate_photons_without_ordinal():
    d = empty()
    d["event_event"][1] = d["event_event"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        compare(d, d)


def test_one_event_and_one_cluster_se_unavailable():
    d = sample(n=1, parents=[0], response=[3])
    out = compare(d, d)
    assert out["totals"]["raw"]["paired_delta_ratio_se"] is None
    assert out["totals"]["raw"]["low"]["event_mean_se"] is None
    assert out["totals"]["raw"]["subrun_cluster_ratio_se"] is None
    assert out["resolved_subset"]["low"][3]["event_cluster_ratio_se"] is None


def test_declared_count_and_accounting_must_agree():
    d = sample()
    d["input_event_count"] = np.array(5)
    with pytest.raises(ValueError, match="ledger"):
        compare(d, d)
    d = sample()
    d["event_residual_ionizing_mev"][0] = 1
    with pytest.raises(ValueError, match="Raw event totals"):
        compare(d, d)


def test_cli_npz_json_and_optional_prediction(tmp_path):
    low, high, out = (tmp_path / name for name in ("foil_response_low.npz", "foil_response_high.npz", "out.json"))
    np.savez(low, **sample())
    np.savez(high, **sample())
    assert main(["--low", str(low), "--high", str(high), "--out", str(out), "--no-cross-predict"]) == 0
    result = json.loads(out.read_text())
    assert result["events"] == 4
    assert "energy_only_prediction" not in result
    assert "fixed G4 inputs" in result["uncertainty"]


def test_response_is_ratio_of_sums_not_mean_of_event_ratios():
    d = sample(parents=[0, 1, 1, 1], response=[4, 0, 0, 0])
    stat = compare(d, d)["resolved_subset"]["low"][3]
    assert stat["mean_edep_mev_per_photon"] == 1
    assert stat["event_cluster_ratio_se"] == pytest.approx(np.sqrt(4 / 3 * 18) / 4)


@pytest.mark.parametrize("field", ["event_run", "run", "product", "key"])
def test_invalid_identity_arrays_rejected(field):
    d = sample()
    d[field] = d[field].astype(float)
    with pytest.raises(ValueError, match="dtype"):
        compare(d, d)
    d[field] = np.full(len(d[field]), -1)
    with pytest.raises(ValueError, match="negative"):
        compare(d, d)


def test_empty_ledger_and_bad_known_flag_rejected():
    with pytest.raises(ValueError, match="Empty event ledger"):
        compare(empty(n=0), empty(n=0))
    d = sample()
    d["photon_response_known"] = np.array([2, 1, 1, 1])
    with pytest.raises(ValueError, match="boolean"):
        compare(d, d)


def test_cross_prediction_paired_uncertainty_and_same_sample_cancellation():
    d = sample()
    out = compare(d, d)["energy_only_prediction"]["low_to_high"]
    assert out["supported_difference_mev"] == 0
    assert out["supported_difference_event_se_mev"] == pytest.approx(0.)
    assert out["supported_prediction_event_se_mev"] == pytest.approx(4.)


def test_cross_prediction_uncertainty_matches_influence_formula():
    l = sample(parents=[0, 1, 1], response=[3., 0., 1.])
    h = sample(parents=[0, 0, 2, 3], response=[1., 0., 5., 2.])
    r = 4 / 3
    cl, el = np.array([1., 2., 0., 0.]), np.array([3., 1., 0., 0.])
    ch, eh = np.array([2., 0., 1., 1.]), np.array([1., 0., 5., 2.])
    influence = (el-r*cl) * 4/3 + ch*r
    out = compare(l, h)["energy_only_prediction"]["low_to_high"]
    assert out["supported_prediction_event_se_mev"] == pytest.approx(np.sqrt(4*np.var(influence, ddof=1)))
    assert out["supported_difference_event_se_mev"] == pytest.approx(np.sqrt(4*np.var(influence-eh, ddof=1)))


def test_photon_energy_excess_is_reported_not_silently_repaired():
    d = sample(parents=[0, 1], energy=[.5, 2.], response=[1., 1.])
    out = compare(d, d)["photon_energy_accounting_diagnostic"]["low"]
    assert out["exceeding_photons"] == 1
    assert out["max_assigned_edep_over_exit_energy"] == 2
    assert out["assigned_mev_on_exceeding"] == 1
