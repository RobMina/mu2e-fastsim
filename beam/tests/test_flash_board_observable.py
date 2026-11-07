"""Normalization contract tests; synthetic records are not detector closure."""
import numpy as np
import pytest

from beamfastsim.flash_board_observable import (
    BUDGET_MEV_PER_GENERATED_EVENT as BUDGET,
    DOCUMENTED_POT_PER_GENERATED_EVENT as POT_FACTOR,
    summarize,
)


def fixture():
    events = dict(run=np.array([1, 1]), subrun=np.array([2, 2]), event=np.array([10, 11]))
    steps = dict(run=np.array([1, 1]), subrun=np.array([2, 2]), event=np.array([10, 10]),
                 product=np.array([3, 3]), index=np.array([0, 1]),
                 ionizing_mev=np.array([2e-6, 5e-7]))
    subruns = dict(run=np.array([1, 1]), subrun=np.array([2, 3]), generated_events=np.array([4, 6]))
    return events, steps, subruns


def test_all_retained_ionization_counts_including_subthreshold_steps():
    result = summarize(*fixture())
    total = 2e-6 + 5e-7
    assert result['tracker_ionizing_mev'] == total
    np.testing.assert_array_equal(result['event_tracker_ionizing_mev'], [total, 0.])
    assert result['generated_events'] == 10
    assert result['retained_events'] == 2
    assert result['generated_subruns'] == 2
    assert result['mev_per_generated_event'] == total / 10
    assert result['mev_per_retained_event'] == total / 2
    assert result['retained_fraction'] == .2
    assert result['central_within_budget']
    assert result['uncertainty'] is result['optimizer_feasibility'] is None


@pytest.mark.parametrize('nretained', [0, 2])
def test_empty_tracker_and_empty_retained_subruns_remain_in_denominator(nretained):
    events, steps, subruns = fixture()
    events = {k: v[:nretained] for k, v in events.items()}
    steps = {k: v[:0] for k, v in steps.items()}
    result = summarize(events, steps, subruns)
    assert result['generated_events'] == 10
    assert result['mev_per_generated_event'] == 0.
    assert result['mev_per_retained_event'] == (0. if nretained else None)
    assert result['event_tracker_ionizing_mev'].shape == (nretained,)


@pytest.mark.parametrize('factor,within', [(1., True), (1.0001, False), (.9999, True)])
def test_central_budget_threshold_has_no_uncertainty_or_pot_rescaling(factor, within):
    events, steps, subruns = fixture()
    subruns['generated_events'] = np.array([4, 4])
    steps['ionizing_mev'] = np.array([8 * BUDGET * factor, 0.])
    result = summarize(events, steps, subruns)
    assert result['central_within_budget'] == within
    assert result['budget_mev_per_generated_event'] == BUDGET


def test_documented_conversion_is_independent_of_anchor_totals():
    assert POT_FACTOR == 25_000_000 / 2_166_994
    assert 6.50684e-7 * POT_FACTOR == pytest.approx(BUDGET, rel=1e-7)


@pytest.mark.parametrize('total,expected,within', [
    (40.74233601982624, 3.703848729075113e-6, True),
    (149.9907398049316, 1.3635521800448327e-5, False),
])
def test_saved_anchor_arithmetic_not_a_raw_output_recount(total, expected, within):
    events, steps, subruns = fixture()
    subruns['generated_events'] = np.array([5_500_000, 5_500_000])
    steps['ionizing_mev'] = np.array([total, 0.])
    result = summarize(events, steps, subruns)
    assert result['mev_per_generated_event'] == expected
    assert result['central_within_budget'] == within


def test_mixed_signed_unsigned_exact_identities_and_ledger_order():
    events, steps, subruns = fixture()
    events['event'] = np.array([2**64 - 1, 2**53 + 1], np.uint64)
    steps['event'] = np.array([2**53 + 1, 2**53 + 1], np.int64)
    steps['product'] = np.full(2, 2**64 - 1, np.uint64)
    result = summarize(events, steps, subruns)
    np.testing.assert_array_equal(result['event_tracker_ionizing_mev'], [0., 2e-6 + 5e-7])


@pytest.mark.parametrize('table', [0, 1, 2])
def test_duplicate_full_identity_rejected(table):
    data = list(fixture())
    data[table] = {k: np.concatenate([v, v[:1]]) for k, v in data[table].items()}
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(*data)


@pytest.mark.parametrize('table', [0, 1, 2])
@pytest.mark.parametrize('bad', [-1, 1.5, True])
def test_invalid_identities_rejected(table, bad):
    data = list(fixture())
    data[table]['run'] = np.full(2, bad)
    with pytest.raises(ValueError, match='identity'):
        summarize(*data)


@pytest.mark.parametrize('bad', [-1., np.nan, np.inf, True, 1j, 'one'])
def test_invalid_energy_rejected(bad):
    events, steps, subruns = fixture()
    steps['ionizing_mev'] = np.full(2, bad)
    with pytest.raises(ValueError, match='Ionizing energy'):
        summarize(events, steps, subruns)


@pytest.mark.parametrize('counts', [[0, 6], [-1, 6], [4., 6.], [True, True],
                                    np.array([2**64 - 1, 2**64 - 1], np.uint64)])
def test_invalid_or_unrepresentable_counters_rejected(counts):
    events, steps, subruns = fixture()
    subruns['generated_events'] = np.asarray(counts)
    with pytest.raises(ValueError, match='Generated'):
        summarize(events, steps, subruns)


def test_missing_event_or_subrun_is_not_a_zero():
    events, steps, subruns = fixture()
    steps['event'][0] = 99
    with pytest.raises(ValueError, match='outside the retained'):
        summarize(events, steps, subruns)
    events, steps, subruns = fixture()
    subruns['subrun'][:] = [8, 9]
    with pytest.raises(ValueError, match='no generated'):
        summarize(events, steps, subruns)


def test_retained_count_cannot_exceed_generated():
    events, steps, subruns = fixture()
    subruns['generated_events'][0] = 1
    with pytest.raises(ValueError, match='exceeds generated'):
        summarize(events, steps, subruns)


def test_missing_or_malformed_columns_rejected():
    events, steps, subruns = fixture()
    del steps['product']
    with pytest.raises(ValueError, match='Missing columns'):
        summarize(events, steps, subruns)
    events, steps, subruns = fixture()
    steps['ionizing_mev'] = np.ones((2, 1))
    with pytest.raises(ValueError, match='one-dimensional'):
        summarize(events, steps, subruns)


@pytest.mark.parametrize('table,column', [(0, 'event'), (1, 'ionizing_mev'), (2, 'generated_events')])
@pytest.mark.parametrize('mask', [False, True])
def test_masked_columns_are_rejected_not_silently_unmasked(table, column, mask):
    data = list(fixture())
    data[table][column] = np.ma.array(data[table][column], mask=mask)
    with pytest.raises(ValueError, match='Masked columns'):
        summarize(*data)


def test_last_bit_order_dependence_is_not_hidden_by_budget_tolerance():
    events = dict(run=np.array([1]), subrun=np.array([2]), event=np.array([3]))
    steps = {k: np.repeat(v, 3) for k, v in events.items()}
    steps.update(product=np.ones(3, np.uint64), index=np.arange(3, dtype=np.uint64))
    subruns = dict(run=np.array([1]), subrun=np.array([2]), generated_events=np.array([1]))
    small = .4 * np.spacing(BUDGET)
    steps['ionizing_mev'] = np.array([BUDGET, small, small])
    first = summarize(events, steps, subruns)
    steps = {k: v[[1, 2, 0]] for k, v in steps.items()}
    second = summarize(events, steps, subruns)
    assert first['mev_per_generated_event'] == BUDGET
    assert second['mev_per_generated_event'] == np.nextafter(BUDGET, np.inf)
    assert first['central_within_budget'] and not second['central_within_budget']


def test_overflow_energy_and_empty_counter_ledger_rejected():
    events, steps, subruns = fixture()
    steps['ionizing_mev'][:] = 1e308
    with pytest.raises(ValueError, match='total is nonfinite'):
        summarize(events, steps, subruns)
    events, steps, subruns = fixture()
    subruns = {k: v[:0] for k, v in subruns.items()}
    with pytest.raises(ValueError, match='positive integers'):
        summarize(events, steps, subruns)
