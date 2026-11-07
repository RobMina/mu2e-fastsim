"""Typed accessor-export reader tests; no art compilation or transport claim."""
import copy
import json

import numpy as np
import pytest

from beamfastsim import flash_board_export as E


def table(name, n, **columns):
    result = {field: np.zeros(n, dtype=("U100" if dtype == "str" else dtype))
              for field, dtype in E.SCHEMA[name].items()}
    for field, value in columns.items():
        result[field][:] = value
    return result


@pytest.fixture
def data():
    events = table('events', 4, run=1, subrun=[1, 1, 2, 2], event=[1, 2, 3, 4],
                   nproducts=3, nstraw=[2, 1, 0, 1], ncalo=[0, 1, 0, 0], ncrv=0)
    products = table('products', 12, run=1, subrun=np.repeat([1, 1, 2, 2], 3),
                     event=np.repeat([1, 2, 3, 4], 3), product=np.tile([11, 12, 13], 4),
                     detector=np.tile([1, 2, 3], 4), slot=0, size=[2, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                     tag=np.tile(['fixture:tracker', 'fixture:calo', 'fixture:crv'], 4),
                     label='fixture', instance=np.tile(['tracker', 'calo', 'crv'], 4), process='fixtureProcess',
                     branch=np.tile(['tracker_branch', 'calo_branch', 'crv_branch'], 4))
    return dict(
        schema=table('schema', 1, version=1, expected_events=4, expected_subruns=2),
        requirements=table('requirements', 3, detector=[1, 2, 3], slot=0,
                           tag=['fixture:tracker', 'fixture:calo', 'fixture:crv']),
        events=events, products=products,
        straw_steps=table('straw_steps', 4, run=1, subrun=[1, 1, 1, 2], event=[1, 1, 2, 4],
                          product=11, index=[0, 1, 0, 0], particle_product=[21, 21, 22, 21], particle_key=17,
                          momentum_mev_c=1., ionizing_mev=[2e-6, 5e-7, 5e-7, 5e-7]),
        calo_steps=table('calo_steps', 1, run=1, subrun=1, event=2, product=12, index=0,
                         particle_product=22, particle_key=17, momentum_mev_c=1., birks_mev=.02),
        crv_steps=table('crv_steps', 0),
        subruns=table('subruns', 2, run=1, subrun=[1, 2], product=31, generated_events=[10, 20],
                      tag='fixture', label='fixture', instance='', process='fixtureProcess', branch='counts_branch'),
        completion=table('completion', 1, complete=1, events=4, subruns=2, straw_steps=4, calo_steps=1, crv_steps=0),
    )


def summary(data, **kwargs):
    return E.summarize(data, expected_events=4, expected_subruns=2, input_scope='filter-input', **kwargs)


def test_end_to_end_fixture_composition_withheld_normalization_by_default(data):
    result = summary(data)
    assert result['exported_events'] == 4
    assert result['retained_events'] == 2
    assert result['detector_accept_counts'] == dict(tracker=1, calo=1, crv=0)
    expected = data['straw_steps']['ionizing_mev'][:3].astype(np.float64).sum()
    assert result['tracker_ionizing_mev'] == expected
    assert result['generated_events'] == 30
    assert result['generated_subruns'] == 2
    assert result['mev_per_generated_event'] is None
    assert result['central_within_budget'] is None
    assert result['retained_fraction'] is None
    assert not result['normalization_available']
    assert not result['upstream_transport_validated']
    assert result['selected_stream_reselection_agrees'] is None
    json.dumps(result, allow_nan=False)


def test_full_subrun_declaration_enables_q_without_scale(data):
    result = summary(data, complete_subruns=True)
    expected = data['straw_steps']['ionizing_mev'][:3].astype(np.float64).sum()
    assert result['mev_per_generated_event'] == expected / 30
    assert result['retained_fraction'] == 2 / 30
    assert result['central_within_budget']
    assert result['uncertainty'] is result['optimizer_feasibility'] is None


def test_retained_scope_does_not_silently_refilter_disagreeing_saved_events(data):
    result = E.summarize(data, expected_events=4, expected_subruns=2, input_scope='retained', complete_subruns=True)
    assert not result['selected_stream_reselection_agrees']
    assert result['retained_events'] == 4
    assert result['detector_rejected_events'] == 2
    assert result['tracker_ionizing_mev'] == data['straw_steps']['ionizing_mev'].astype(np.float64).sum()


def test_exact_uint64_identity_and_big_endian_storage(data):
    for name in ('products', 'straw_steps', 'calo_steps', 'crv_steps', 'subruns'):
        for field in ('product', 'particle_product', 'particle_key'):
            if field in data[name]:
                data[name][field] += np.uint64(2**63)
    data['straw_steps']['particle_key'][:] = np.uint64(2**64 - 1)
    data['products']['size'] = data['products']['size'].astype('>u8')
    assert summary(data)['retained_events'] == 2


def test_product_slots_enforce_multiple_collections_not_only_roles(data):
    extra = {k: v[1:2].copy() for k, v in data['requirements'].items()}
    extra['slot'][:] = 1
    extra['tag'][:] = 'fixture:othercalo'
    data['requirements'] = {k: np.concatenate([v, extra[k]]) for k, v in data['requirements'].items()}
    with pytest.raises(ValueError, match='Missing configured'):
        summary(data)
    p = data['products']
    extra = {k: v[[1, 4, 7, 10]].copy() for k, v in p.items()}
    extra['slot'][:] = 1
    extra['product'][:] = 14
    extra['size'][:] = 0
    extra['tag'][:] = 'fixture:othercalo'
    extra['instance'][:] = 'othercalo'
    extra['branch'][:] = 'other_branch'
    data['products'] = {k: np.concatenate([v, extra[k]]) for k, v in p.items()}
    data['events']['nproducts'][:] = 4
    assert summary(data)['retained_events'] == 2


@pytest.mark.parametrize('name,field', [('schema', 'expected_events'), ('events', 'event'),
                                     ('products', 'product'), ('subruns', 'generated_events'),
                                     ('straw_steps', 'ionizing_mev')])
def test_wrong_storage_width_is_rejected_not_cast(name, field, data):
    original = data[name][field]
    data[name][field] = original.astype('f8' if original.dtype.kind == 'f' else 'i8')
    with pytest.raises(ValueError, match='expected'):
        summary(data)


@pytest.mark.parametrize('name,field', [('events', 'event'), ('products', 'tag'), ('straw_steps', 'ionizing_mev')])
def test_masked_and_multidimensional_columns_fail(name, field, data):
    original = data[name][field]
    data[name][field] = np.ma.array(original, mask=False)
    with pytest.raises(ValueError, match='masked'):
        summary(data)
    data[name][field] = original[:, None]
    with pytest.raises(ValueError, match='one-dimensional'):
        summary(data)


@pytest.mark.parametrize('bad', [0, -1, True, 4., None])
def test_expected_counts_must_be_positive_integers(data, bad):
    with pytest.raises(ValueError, match='positive integer'):
        E.validate(data, expected_events=bad, expected_subruns=2)


@pytest.mark.parametrize('which', ['schema', 'completion'])
def test_metadata_exactly_one_row(which, data):
    original = data[which]
    for count in (0, 2):
        data[which] = {k: np.repeat(v, count) for k, v in original.items()}
        with pytest.raises(ValueError, match='schema|completion'):
            summary(data)


@pytest.mark.parametrize('name,field', [('schema', 'version'), ('schema', 'expected_events'),
                                     ('schema', 'expected_subruns'), ('completion', 'events'),
                                     ('completion', 'straw_steps'), ('completion', 'subruns'),
                                     ('completion', 'complete')])
def test_metadata_mismatch_rejected(name, field, data):
    data[name][field][0] += 1
    with pytest.raises(ValueError):
        summary(data)


@pytest.mark.parametrize('name', ['schema', 'events', 'requirements', 'products', 'straw_steps', 'subruns'])
def test_missing_trees_and_columns(name, data):
    original = data.pop(name)
    with pytest.raises(ValueError, match='Missing export trees'):
        summary(data)
    data[name] = original
    del data[name][next(iter(original))]
    with pytest.raises(ValueError, match='missing columns'):
        summary(data)


@pytest.mark.parametrize('field', ['tag', 'label', 'process', 'branch'])
def test_empty_provenance_not_accepted(field, data):
    data['products'][field][0] = ''
    with pytest.raises(ValueError, match='empty'):
        summary(data)


@pytest.mark.parametrize('field,value', [('tag', 'other:tracker'), ('instance', 'other'),
                                       ('tag', 'fixture:tracker:wrong'), ('tag', 'fixture:tracker:X:Y')])
def test_provenance_mismatch(field, value, data):
    data['products'][field][0] = value
    with pytest.raises(ValueError, match='provenance|InputTag'):
        summary(data)


def test_single_branch_cannot_describe_three_products_in_one_event(data):
    data['products']['branch'][:3] = 'identical_branch'
    with pytest.raises(ValueError, match='Duplicate resolved branch'):
        summary(data)


def test_configured_counter_tag_cannot_change_between_subruns(data):
    data['subruns']['tag'][1] = 'other'
    data['subruns']['label'][1] = 'other'
    with pytest.raises(ValueError, match='counter tag changes'):
        summary(data)


def test_unspecified_process_may_resolve_differently_across_subruns(data):
    data['subruns']['process'][1] = 'otherProcess'
    data['subruns']['branch'][1] = 'otherBranch'
    data['subruns']['product'][1] = 88
    assert summary(data)['generated_subruns'] == 2


@pytest.mark.parametrize('change', ['duplicate', 'gap', 'unknown', 'missing'])
def test_requirement_contract(change, data):
    if change == 'duplicate':
        data['requirements'] = {k: np.concatenate([v, v[:1]]) for k, v in data['requirements'].items()}
    elif change == 'gap':
        data['requirements']['slot'][0] = 1
    elif change == 'unknown':
        data['requirements']['detector'][0] = 99
    else:
        data['requirements'] = {k: v[1:] for k, v in data['requirements'].items()}
    with pytest.raises(ValueError):
        summary(data)


@pytest.mark.parametrize('field', ['nproducts', 'nstraw', 'ncalo', 'ncrv'])
def test_event_count_contract(field, data):
    data['events'][field][0] += 1
    with pytest.raises(ValueError, match='mismatch'):
        summary(data)


@pytest.mark.parametrize('name,field', [('products', 'product'), ('subruns', 'product'),
                                     ('straw_steps', 'particle_product')])
def test_zero_product_ids_rejected(name, field, data):
    data[name][field][0] = 0
    with pytest.raises(ValueError, match='ProductID'):
        summary(data)


def test_empty_and_duplicate_input_collections_cannot_hide_missing_rows(data):
    data['products']['slot'][1] = 1
    with pytest.raises(ValueError, match='requirement'):
        summary(data)


def test_duplicate_subrun_exposure_not_deduplicated(data):
    data['subruns']['subrun'][1] = 1
    with pytest.raises(ValueError, match='Duplicate'):
        summary(data)


def test_incomplete_steps_and_invalid_energies_still_fail_after_export_validation(data):
    data['products']['size'][0] += 1
    with pytest.raises(ValueError, match='incomplete'):
        summary(data)
    data['products']['size'][0] -= 1
    data['straw_steps']['ionizing_mev'][0] = np.nan
    with pytest.raises(ValueError, match='ionizing_mev'):
        summary(data)


def test_generated_counter_default_sentinel_or_zero_rejected(data):
    data['subruns']['generated_events'][0] = 0
    with pytest.raises(ValueError, match='positive integers'):
        summary(data)
    data['subruns']['generated_events'][0] = np.uint64(2**64-1)
    with pytest.raises(ValueError, match='exact float64'):
        summary(data)


def test_invalid_scope_or_declaration(data):
    with pytest.raises(ValueError, match='input_scope'):
        E.summarize(data, expected_events=4, expected_subruns=2, input_scope='auto')
    with pytest.raises(ValueError, match='boolean'):
        summary(data, complete_subruns=1)


def write_root(path, data, directory='board', rntuple=False):
    uproot = pytest.importorskip('uproot')
    with uproot.recreate(path) as output:
        for name, spec in E.SCHEMA.items():
            payload = data[name]
            if rntuple:
                output[f'{directory}/{name}'] = payload
            else:
                tree = output.mktree(f'{directory}/{name}',
                                     {k: ('string' if d == 'str' else np.dtype(d)) for k, d in spec.items()})
                if len(next(iter(payload.values()))):
                    tree.extend(payload)


def test_uproot_roundtrip_preserves_typed_empty_tables_and_cli(tmp_path, data):
    path, out = tmp_path / 'input.root', tmp_path / 'out.json'
    write_root(path, data)
    loaded = E.read_tables(path)
    for name, columns in data.items():
        for field, expected in columns.items():
            np.testing.assert_array_equal(loaded[name][field], expected)
    args = ['--input', str(path), '--out', str(out), '--events', '4', '--subruns', '2',
            '--input-scope', 'filter-input']
    assert E.main(args) == 0
    report = json.loads(out.read_text())
    assert report['mev_per_generated_event'] is None
    original = out.read_bytes()
    with pytest.raises(FileExistsError):
        E.main(args)
    assert out.read_bytes() == original


def test_reader_uses_highest_tree_cycle_not_stale_complete_ledger(tmp_path, data):
    uproot = pytest.importorskip('uproot')
    path = tmp_path / 'cycles.root'
    write_root(path, data)
    with uproot.update(path) as output:
        tree = output.mktree('board/events', {k: np.dtype(d) for k, d in E.SCHEMA['events'].items()})
        tree.extend({k: v[:-1] for k, v in data['events'].items()})
    loaded = E.read_tables(path)
    assert len(loaded['events']['event']) == 3
    with pytest.raises(ValueError, match='Completion count differs for events'):
        summary(loaded)


def test_rntuple_cannot_silently_replace_exporter_ttree(tmp_path, data):
    path = tmp_path / 'rntuple.root'
    write_root(path, data, rntuple=True)
    with pytest.raises(ValueError, match='expected TTree'):
        E.read_tables(path)


def test_cli_retained_scope_disagreement_is_nonzero_and_keeps_totals(tmp_path, data):
    path, out = tmp_path / 'input.root', tmp_path / 'out.json'
    write_root(path, data)
    assert E.main(['--input', str(path), '--out', str(out), '--events', '4', '--subruns', '2',
                   '--input-scope', 'retained', '--complete-subruns']) == 1
    result = json.loads(out.read_text())
    assert result['retained_events'] == 4 and result['normalization_available']


def actual_fixture(data):
    data = copy.deepcopy(data)
    data['straw_steps']['particle_product'][:] = [21, 22, 21, 21]
    data['crv_steps'] = table('crv_steps', 1, run=1, subrun=1, event=1, product=13, index=0,
                             particle_product=22, particle_key=17, visible_mev=2e-6, momentum_mev_c=5.)
    data['events']['ncrv'][0] = 1
    data['products']['size'][2] = 1
    data['completion']['crv_steps'][:] = 1
    return data


def test_independent_fixture_checker_and_exact_reread(tmp_path, data):
    from beamfastsim import flash_board_fixture as F
    first, second = tmp_path / 'fixture.root', tmp_path / 'reread.root'
    data = actual_fixture(data)
    assert F.check(data)['generated_events'] == 30
    assert F.check(data)['retained_events'] == 2
    F.compare(data, copy.deepcopy(data))
    write_root(first, data)
    write_root(second, data)
    assert F.main(['--input', str(first), '--reread', str(second)]) == 0


@pytest.mark.parametrize('field,change', [('ionizing_mev', 1.01), ('momentum_mev_c', 2.)])
def test_fixture_checker_rejects_valid_but_wrong_accessor_payload(data, field, change):
    from beamfastsim import flash_board_fixture as F
    data = actual_fixture(data)
    data['straw_steps'][field][0] *= np.float32(change)
    with pytest.raises(ValueError, match='Fixture mismatch'):
        F.check(data)


def test_fixture_same_particle_key_must_represent_distinct_products(data):
    from beamfastsim import flash_board_fixture as F
    data = actual_fixture(data)
    data['straw_steps']['particle_product'][:] = 21
    with pytest.raises(ValueError, match='two distinct'):
        F.check(data)


def test_fixture_reread_payload_equality_not_only_final_total(data):
    from beamfastsim import flash_board_fixture as F
    left = actual_fixture(data)
    right = copy.deepcopy(left)
    right['products']['branch'][0] = 'different_actual_branch'
    with pytest.raises(ValueError, match='reread products.branch'):
        F.compare(left, right)


def test_exclusive_output_cannot_overwrite_input(tmp_path, data):
    path = tmp_path / 'input.root'
    write_root(path, data)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        E.main(['--input', str(path), '--out', str(path), '--events', '4', '--subruns', '2',
                '--input-scope', 'filter-input'])
    assert path.read_bytes() == original
