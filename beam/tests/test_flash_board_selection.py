"""Historical aggregated detector selection; synthetic tests are not G4 closure."""
import numpy as np
import pytest

from beamfastsim import flash_board_selection as S


def events(n=4):
    return dict(run=np.ones(n, np.uint64), subrun=np.full(n, 2, np.uint64),
                event=np.arange(10, 10 + n, dtype=np.uint64))


def steps(role, event=(), energy=(), momentum=None, particle=None, product=None, index=None):
    n = len(event)
    return dict(run=np.ones(n, np.uint64), subrun=np.full(n, 2, np.uint64),
                event=np.asarray(event, np.uint64),
                product=np.full(n, role, np.uint64) if product is None else np.asarray(product, np.uint64),
                index=np.arange(n, dtype=np.uint64) if index is None else np.asarray(index, np.uint64),
                particle_product=np.full(n, 7, np.uint64),
                particle_key=np.ones(n, np.uint64) if particle is None else np.asarray(particle, np.uint64),
                momentum_mev_c=np.ones(n, np.float32) if momentum is None else np.asarray(momentum, np.float32),
                **{{1: 'ionizing_mev', 2: 'birks_mev', 3: 'visible_mev'}[role]: np.asarray(energy, np.float32)})


def products(ledger, tables):
    n = len(ledger['event'])
    result = {k: np.repeat(v, 3) for k, v in ledger.items()}
    result.update(product=np.tile(np.arange(1, 4, dtype=np.uint64), n),
                  detector=np.tile(np.arange(1, 4, dtype=np.uint64), n), size=np.zeros(3*n, np.uint64))
    for role, table in enumerate(tables, 1):
        event, counts = np.unique(table['event'], return_counts=True)
        result['size'][3 * np.searchsorted(ledger['event'], event) + role - 1] = counts
    return result


def fixture():
    e = events()
    tables = (steps(1, [10, 10, 11, 13], [2e-6, 5e-7, 5e-7, 5e-7], index=[0, 1, 0, 0]),
              steps(2, [11], [.02]), steps(3, [12], [2e-6]))
    return (e, products(e, tables), *tables)


def test_or_and_composition_keep_all_tracker_energy_and_complete_exposure():
    data = fixture()
    result = S.select(*data)
    np.testing.assert_array_equal(result['retained'], [True, True, True, False])
    np.testing.assert_array_equal(result['tracker'], [True, False, False, False])
    np.testing.assert_array_equal(result['calo'], [False, True, False, False])
    np.testing.assert_array_equal(result['crv'], [False, False, True, False])
    np.testing.assert_array_equal(result['retained_straw_steps']['index'], [0, 1, 0])
    counters = dict(run=np.array([1, 1]), subrun=np.array([2, 3]), generated_events=np.array([10, 20]))
    summary = S.summarize(*data, counters)
    expected = data[2]['ionizing_mev'][:3].astype(np.float64).sum()
    assert summary['tracker_ionizing_mev'] == expected
    assert summary['mev_per_generated_event'] == expected / 30
    assert summary['generated_events'] == 30
    assert summary['retained_events'] == 3
    assert summary['events_at_detector_filter'] == 4
    assert summary['event_tracker_ionizing_mev'][-1] == 0.
    assert summary['uncertainty'] is summary['optimizer_feasibility'] is None


@pytest.mark.parametrize('role', [1, 3])
def test_strict_step_energy_and_momentum_thresholds_use_promoted_float(role):
    e = events(7)
    threshold = np.float32(1e-6)
    energy = [0., np.nextafter(threshold, np.float32(0)), threshold,
              np.nextafter(threshold, np.float32(np.inf)), 2e-6, 2e-6, 2e-6]
    tables = [steps(1), steps(2), steps(3)]
    tables[role-1] = steps(role, e['event'], energy, momentum=[1, 1, 1, 1, 0, 1e6, 1e6-1], index=np.zeros(7, int))
    result = S.select(e, products(e, tables), *tables)
    np.testing.assert_array_equal(result['retained'], [False, False, False, True, False, False, True])


def test_calo_lower_and_upper_limits_are_double_comparisons_of_float32_sums():
    e = events(6)
    edge = np.float32(.01)
    tables = (steps(1), steps(2, e['event'],
              [edge, np.nextafter(edge, np.float32(np.inf)), 1e6,
               np.nextafter(np.float32(1e6), np.float32(np.inf)), 0., .02],
              momentum=[1, 1, 1, 1, 1, 1e6], index=np.zeros(6, int)), steps(3))
    result = S.select(e, products(e, tables), *tables)
    np.testing.assert_array_equal(result['calo'], [False, True, True, False, False, False])


def test_calo_groups_by_particle_not_event_total():
    e = events(2)
    tables = (steps(1), steps(2, [10, 10, 11, 11], [.006]*4,
              particle=[1, 2, 1, 1], index=[0, 1, 0, 1]), steps(3))
    result = S.select(e, products(e, tables), *tables)
    np.testing.assert_array_equal(result['calo'], [False, True])


def test_calo_different_particle_products_are_distinct():
    e = events(1)
    tables = (steps(1), steps(2, [10, 10], [.006, .006]), steps(3))
    tables[1]['particle_product'] = np.array([2**64-1, 2**64-2], np.uint64)
    assert not S.select(e, products(e, tables), *tables)['retained'][0]


def test_calo_grouping_resets_per_input_collection():
    e = events(1)
    tables = (steps(1), steps(2, [10, 10], [.006, .006], product=[2, 4], index=[0, 0]), steps(3))
    p = products(e, tables)
    p = {k: np.concatenate([v, v[1:2]]) for k, v in p.items()}
    p['product'][-1] = 4
    p['size'][1] = p['size'][-1] = 1
    result = S.select(e, p, *tables)
    assert not result['calo'][0]


def test_float32_accumulation_uses_collection_index_not_export_row_order():
    e = events(1)
    tables = (steps(1), steps(2, [10]*3, [1e6, .02, .02]), steps(3))
    p = products(e, tables)
    assert S.select(e, p, *tables)['calo'][0]
    shuffled = {k: v[[1, 2, 0]] for k, v in tables[1].items()}
    assert S.select(e, p, tables[0], shuffled, tables[2])['calo'][0]
    shuffled['index'] = np.arange(3, dtype=np.uint64)
    assert not S.select(e, p, tables[0], shuffled, tables[2])['calo'][0]


def test_tracker_max_count_is_per_particle_and_collection():
    e = events(1)
    n = 1_000_001
    table = steps(1, np.full(n, 10), np.full(n, 2e-6))
    tables = (table, steps(2), steps(3))
    p = products(e, tables)
    assert not S.select(e, p, *tables)['tracker'][0]
    table['particle_key'][-1] = 2
    assert S.select(e, p, *tables)['tracker'][0]
    table['particle_key'][-1] = 1
    trimmed = {k: v[:-1] for k, v in table.items()}
    p['size'][0] -= 1
    assert S.select(e, p, trimmed, tables[1], tables[2])['tracker'][0]


@pytest.mark.parametrize('role', [1, 3])
def test_unsigned_counter_overflow_fails_instead_of_emulating_wrap(monkeypatch, role):
    table = steps(role, [10], [2e-6])
    monkeypatch.setattr(S.np, 'bincount', lambda *a, **k: np.array([2**32], np.int64))
    with pytest.raises(ValueError, match='unsigned counter'):
        S._accepted_events(table, np.array([2e-6], np.float32), np.array([True]),
                           np.array([0]), 1, role)


def test_exact_identity_above_float64_range_and_mixed_unsigned_columns():
    e, p, t, c, r = fixture()
    for table in (e, p, t, c, r):
        table['event'] += np.uint64(2**63)
    t['particle_key'] = np.array([2**64-1]*4, np.uint64)
    c['product'] = c['product'].astype(np.int64)
    result = S.select(e, p, t, c, r)
    np.testing.assert_array_equal(result['retained_events']['event'], e['event'][:3])


def test_empty_event_ledger_and_explicit_empty_products():
    data = fixture()
    empty = tuple({k: v[:0] for k, v in table.items()} for table in data)
    assert len(S.select(*empty)['retained']) == 0
    e = events(1)
    tables = (steps(1), steps(2), steps(3))
    assert not S.select(e, products(e, tables), *tables)['retained'][0]


@pytest.mark.parametrize('which', range(5))
def test_duplicate_identities_are_rejected(which):
    data = list(fixture())
    data[which] = {k: np.concatenate([v, v[:1]]) for k, v in data[which].items()}
    with pytest.raises(ValueError, match='Duplicate'):
        S.select(*data)


@pytest.mark.parametrize('which,field', [(0, 'event'), (1, 'size'), (2, 'particle_key'),
                                       (3, 'birks_mev'), (4, 'momentum_mev_c')])
def test_masked_columns_rejected_even_if_mask_false(which, field):
    data = list(fixture())
    data[which][field] = np.ma.array(data[which][field], mask=False)
    with pytest.raises(ValueError, match='Masked'):
        S.select(*data)


@pytest.mark.parametrize('bad', [-1, 1.5, True])
def test_invalid_particle_identity(bad):
    data = fixture()
    data[2]['particle_key'] = np.full(4, bad)
    with pytest.raises(ValueError, match='identity'):
        S.select(*data)


@pytest.mark.parametrize('bad', [-1., np.inf, np.nan, True, 1j, 'bad', .0100000000001])
def test_invalid_or_not_stored_energy(bad):
    data = fixture()
    data[3]['birks_mev'] = np.array([bad])
    with pytest.raises(ValueError, match='birks_mev'):
        S.select(*data)


@pytest.mark.parametrize('bad', [-1., np.inf, np.nan, True, 1j, 'bad', 1.000000000001])
def test_invalid_or_not_stored_momentum(bad):
    data = fixture()
    data[3]['momentum_mev_c'] = np.array([bad])
    with pytest.raises(ValueError, match='momentum_mev_c'):
        S.select(*data)


def test_missing_product_is_not_empty():
    data = list(fixture())
    data[1] = {k: v[:-1] for k, v in data[1].items()}
    with pytest.raises(ValueError, match='Missing crv product'):
        S.select(*data)


@pytest.mark.parametrize('fault,match', [('orphan', 'no declared'), ('role', 'wrong product'),
                                       ('count', 'incomplete'), ('index', 'outside.*size'),
                                       ('product_event', 'outside the event')])
def test_incomplete_or_mismatched_detector_products(fault, match):
    data = fixture()
    if fault == 'orphan':
        data[2]['product'][0] = 99
    elif fault == 'role':
        data[2]['product'][0] = 2
    elif fault == 'count':
        data[1]['size'][0] += 1
    elif fault == 'index':
        data[2]['index'][0] = 99
    else:
        data[1]['event'][0] = 99
    with pytest.raises(ValueError, match=match):
        S.select(*data)


@pytest.mark.parametrize('field,bad', [('size', -1), ('size', 1.5), ('size', True),
                                     ('detector', 4), ('detector', 1.5), ('detector', True)])
def test_invalid_product_declarations(field, bad):
    data = fixture()
    data[1][field] = np.full(12, bad)
    with pytest.raises(ValueError):
        S.select(*data)


def test_all_filter_input_events_must_fit_exposure_even_if_rejected():
    data = fixture()
    counters = dict(run=np.array([1]), subrun=np.array([2]), generated_events=np.array([3]))
    with pytest.raises(ValueError, match='exceeds generated'):
        S.summarize(*data, counters)


@pytest.mark.parametrize('dtype', [np.uint64, np.int64])
def test_large_integer_energy_cannot_round_before_lossless_check(dtype):
    data = fixture()
    data[3]['birks_mev'] = np.array([2**53 + 1], dtype)
    with pytest.raises(ValueError, match='exact float64'):
        S.select(*data)


def test_exact_float64_promotions_are_supported_without_rounding():
    data = fixture()
    for table, field in ((data[2], 'ionizing_mev'), (data[3], 'birks_mev'), (data[4], 'visible_mev')):
        table[field] = table[field].astype(np.float64)
        table['momentum_mev_c'] = table['momentum_mev_c'].astype(np.float64)
    np.testing.assert_array_equal(S.select(*data)['retained'], [True, True, True, False])


def test_shapes_missing_fields_and_calo_overflow_fail_closed():
    data = fixture()
    data[3]['birks_mev'] = np.ones((1, 1), np.float32)
    with pytest.raises(ValueError, match='one-dimensional'):
        S.select(*data)
    data = fixture()
    del data[3]['particle_product']
    with pytest.raises(ValueError, match='Missing columns'):
        S.select(*data)
    e = events(1)
    tables = (steps(1), steps(2, [10, 10], [np.finfo(np.float32).max]*2), steps(3))
    with pytest.raises(ValueError, match='particle sum is nonfinite'):
        S.select(e, products(e, tables), *tables)
