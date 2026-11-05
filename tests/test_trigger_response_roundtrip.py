"""Frozen-dumper numerical SGS round trip; no art runtime or owner modifications."""
import ast
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import response_roundtrip as rt


def payload():
    events = np.array([[4294967295, 0, 4294967295], [1, 2, 2], [1, 2, 3]], dtype='u4')

    def columns(schema, which):
        data = {k: np.zeros(len(which), dtype=dtype) for k, dtype in schema.items()}
        for i, k in enumerate(rt.KEYS):
            data[k] = events[which, i].copy()
        return data

    gen = columns(rt.GEN_SCHEMA, [0, 1, 2])
    gen['pdg'][:] = 11
    gen['e'][:] = 105.
    # Repeated same-particle/same-straw rows must remain in encounter order.
    sgs = columns(rt.SGS_SCHEMA, [0, 0, 0, 1])
    sgs['nsgs'][:] = [3, 3, 3, 1]
    sgs['sp'][:] = [7, 7, 42, 9]
    sgs['sid'][:] = 11653  # plane 11, panel 3, straw 5
    sgs['plane'][:] = 11
    sgs['panel'][:] = 3
    sgs['straw'][:] = 5
    sgs['upanel'][:] = 69
    sgs['shape'][:] = [0, 1, 2, 3]
    sgs['ion'][:] = [0, 1, 2, 0]
    for k in rt.SGS_FLOATS:
        sgs[k][:] = [1.25, 2.5, 4., 8.]
    sgs['t'][:] = [1. + 2. ** -40, 2., 3., 4.]
    particles = columns(rt.PARTICLE_SCHEMA, [0, 0, 1])
    particles['key'][:] = [7, 42, 9]
    particles['parent'][:] = [-1, 7, -1]
    particles['pdg'][:] = [11, 22, -11]
    particles['ismcprim'][:] = [1, 0, 1]
    particles['isprim'][:] = [1, 0, 1]
    return events, {'gen': gen, 'sgsteps': sgs, 'particles': particles}


def write_table(path, data):
    with uproot.recreate(path) as root:
        for name, columns in data.items():
            root.mktree('dump/' + name, {k: v.dtype for k, v in columns.items()})
            root['dump/' + name].extend(columns)
    return path


def reorder(data, **orders):
    return {name: {k: v[orders.get(name, np.arange(len(v)))].copy() for k, v in columns.items()}
            for name, columns in data.items()}


def pair(tmp_path):
    events, original = payload()
    return events, original, write_table(tmp_path / 'original.root', original)


def test_exact_independent_event_traversal_and_repeated_rows(tmp_path):
    events, data, path = pair(tmp_path)
    imported = reorder(data, gen=[2, 0, 1], particles=[2, 1, 0], sgsteps=[3, 0, 1, 2])
    report = rt.compare([path], write_table(tmp_path / 'import.root', imported), events[::-1])
    assert report['equal'] and report['sgs']['equal'] and report['gen']['equal']
    assert report['sgs']['matched_rows'] == 4
    assert report['zero_step_events'] == {'original': 1, 'imported': 1}
    assert set(report['sgs']['fields']) == set(rt.SGS_SCHEMA)
    assert all(f['numeric_mismatches'] == 0 for f in report['sgs']['fields'].values())
    assert report['inputs_before'] == report['inputs_after']
    assert report['inputs_before'][0]['sha256'] and report['inputs_before'][0]['bytes'] > 0
    assert report['trees']['original'][0]['sgsteps']['tree'].endswith('sgsteps;1')
    assert not any(report['claims'].values())
    assert 'present-empty' in ' '.join(report['limitations'])


def test_within_event_reorder_is_difference_not_straw_sort_or_dedup(tmp_path):
    events, data, path = pair(tmp_path)
    imported = reorder(data, sgsteps=[1, 0, 2, 3])
    report = rt.compare([path], write_table(tmp_path / 'import.root', imported), events)
    assert not report['equal']
    assert report['sgs']['fields']['eion']['numeric_mismatches'] == 2
    assert report['sgs']['fields']['sp']['numeric_mismatches'] == 0
    assert report['sgs']['matched_rows'] == 4


def test_two_tables_complete_event_partition_and_multiple_gen_rows(tmp_path):
    events, data = payload()
    # Two generated particles in event 1: coverage must count unique events.
    data['gen'] = {k: np.concatenate([v, v[:1]]) for k, v in data['gen'].items()}
    data['gen']['idx'][-1] = 1
    sources = []
    for i, selected in enumerate(([events[0, 2]], [2, 3])):
        part = {name: {k: v[np.isin(columns['event'], selected)] for k, v in columns.items()}
                for name, columns in data.items()}
        sources.append(write_table(tmp_path / f'source{i}.root', part))
    report = rt.compare(sources[::-1], write_table(tmp_path / 'import.root', data), events)
    assert report['equal'] and report['events'] == 3
    assert report['gen']['original_rows'] == 4
    assert report['zero_step_events']['original'] == 1


@pytest.mark.parametrize('operation', ['extra', 'missing', 'entire-event'])
def test_extra_missing_sgs_rows_are_reported(tmp_path, operation):
    events, data, path = pair(tmp_path)
    if operation == 'extra':
        data['sgsteps'] = {k: np.concatenate([v, v[-1:]]) for k, v in data['sgsteps'].items()}
        data['sgsteps']['nsgs'][-2:] = 2
    elif operation == 'missing':
        data['sgsteps'] = {k: v[[0, 1, 3]] for k, v in data['sgsteps'].items()}
        data['sgsteps']['nsgs'][:2] = 2
    else:
        data['sgsteps'] = {k: v[:3] for k, v in data['sgsteps'].items()}
    report = rt.compare([path], write_table(tmp_path / 'import.root', data), events)
    assert not report['equal']
    assert report['sgs']['extra_rows' if operation == 'extra' else 'missing_rows'] == 1


@pytest.mark.parametrize('tree', ['sgsteps', 'particles'])
def test_zero_rows_in_every_event(tmp_path, tree):
    events, data = payload()
    data['sgsteps'] = {k: v[:0] for k, v in data['sgsteps'].items()}
    if tree == 'particles':
        data['particles'] = {k: v[:0] for k, v in data['particles'].items()}
    report = rt.compare([write_table(tmp_path / 'a.root', data)],
                        write_table(tmp_path / 'b.root', data), events)
    assert report['equal']
    assert report['sgs']['matched_rows'] == 0
    assert report['zero_step_events'] == {'original': 3, 'imported': 3}


@pytest.mark.parametrize('operation', ['missing-zero', 'extra-zero'])
def test_imported_gen_coverage_reports_missing_or_extra_zero_event(tmp_path, operation):
    events, data, path = pair(tmp_path)
    if operation == 'missing-zero':
        data['gen'] = {k: v[:2] for k, v in data['gen'].items()}
    else:
        data['gen'] = {k: np.concatenate([v, v[-1:]]) for k, v in data['gen'].items()}
        data['gen']['event'][-1] = 4
    report = rt.compare([path], write_table(tmp_path / 'b.root', data), events)
    assert not report['equal']
    assert report['imported_coverage']['missing_events' if operation == 'missing-zero' else 'extra_events'] == 1


@pytest.mark.parametrize('operation', ['duplicate-gen', 'duplicate-cross-file', 'missing-zero',
                                      'foreign-denominator', 'duplicate-denominator', 'float-denominator',
                                      'negative-denominator', 'shape-denominator', 'huge-denominator',
                                      'same-table', 'no-tables'])
def test_invalid_source_coverage_and_denominator(tmp_path, operation):
    events, data, path = pair(tmp_path)
    sources = [path]
    imported = write_table(tmp_path / 'b.root', data)
    if operation == 'duplicate-gen':
        data['gen'] = {k: np.concatenate([v, v[:1]]) for k, v in data['gen'].items()}
        sources = [write_table(tmp_path / 'bad.root', data)]
    elif operation == 'duplicate-cross-file':
        sources.append(write_table(tmp_path / 'bad.root', data))
    elif operation == 'missing-zero':
        data['gen'] = {k: v[:2] for k, v in data['gen'].items()}
        sources = [write_table(tmp_path / 'bad.root', data)]
    elif operation == 'foreign-denominator':
        events = events[:2]
    elif operation == 'duplicate-denominator':
        events = np.vstack([events, events[:1]])
    elif operation == 'float-denominator':
        events = events.astype(float)
    elif operation == 'negative-denominator':
        events = events.astype('i8'); events[0, 1] = -1
    elif operation == 'shape-denominator':
        events = events[:, :2]
    elif operation == 'huge-denominator':
        events = events.astype('u8'); events[0, 2] = 2 ** 63
    elif operation == 'same-table':
        imported = path
    elif operation == 'no-tables':
        sources = []
    with pytest.raises(ValueError):
        rt.compare(sources, imported, events)


@pytest.mark.parametrize('field,dtype', [('event', 'i4'), ('run', 'u8'), ('sp', 'u4'),
    ('sid', 'i8'), ('nsgs', 'f4'), ('t', 'f4'), ('eion', 'f8'), ('px', 'i4')])
def test_actual_sgs_dtype_schema_not_just_numeric_kind(tmp_path, field, dtype):
    events, data, path = pair(tmp_path)
    data['sgsteps'][field] = data['sgsteps'][field].astype(dtype)
    with pytest.raises(ValueError, match='dtype'):
        rt.compare([path], write_table(tmp_path / 'bad.root', data), events)


@pytest.mark.parametrize('field', list(rt.SGS_SCHEMA))
def test_no_sgs_column_can_be_omitted(tmp_path, field):
    events, data, path = pair(tmp_path)
    del data['sgsteps'][field]
    with pytest.raises(ValueError, match='schema'):
        rt.compare([path], write_table(tmp_path / 'bad.root', data), events)


@pytest.mark.parametrize('field', list(rt.SGS_FLOATS) + ['t'])
def test_exact_float_precision_no_tolerance(tmp_path, field):
    events, data, path = pair(tmp_path)
    value = data['sgsteps'][field]
    value[0] = np.nextafter(value[0], np.array(np.inf, dtype=value.dtype))
    report = rt.compare([path], write_table(tmp_path / 'b.root', data), events)
    assert not report['equal']
    assert report['sgs']['fields'][field]['numeric_mismatches'] == 1
    assert report['sgs']['fields'][field]['bit_mismatches'] == 1


def test_f64_time_difference_invisible_after_f32_cast_is_detected(tmp_path):
    events, data, path = pair(tmp_path)
    old = data['sgsteps']['t'][0]
    data['sgsteps']['t'][0] += 2. ** -45
    assert np.float32(old) == np.float32(data['sgsteps']['t'][0])
    report = rt.compare([path], write_table(tmp_path / 'b.root', data), events)
    assert report['sgs']['fields']['t']['numeric_mismatches'] == 1


def test_numeric_signed_zero_equal_but_bits_differ(tmp_path):
    events, data = payload()
    data['sgsteps']['x0'][0] = 0.
    path = write_table(tmp_path / 'a.root', data)
    data['sgsteps']['x0'][0] = -0.
    report = rt.compare([path], write_table(tmp_path / 'b.root', data), events)
    assert report['equal']
    assert report['sgs']['fields']['x0']['bit_mismatches'] == 1


@pytest.mark.parametrize('target', ['nan', 'inf', 'negative-sp', 'negative-energy', 'negative-width',
    'shape', 'ion', 'sid', 'plane', 'panel', 'straw', 'upanel', 'nsgs', 'foreign-event',
    'foreign-sp', 'duplicate-particle', 'negative-parent', 'extra-field', 'gen-nan', 'gen-dtype'])
def test_invalid_payloads_rejected(tmp_path, target):
    events, data, path = pair(tmp_path)
    s, p, g = data['sgsteps'], data['particles'], data['gen']
    if target in ('nan', 'inf'):
        s['t'][0] = np.nan if target == 'nan' else np.inf
    elif target.startswith('negative-'):
        if target == 'negative-parent': p['parent'][0] = -2
        else: s[{'negative-sp': 'sp', 'negative-energy': 'eion', 'negative-width': 'width'}[target]][0] = -1
    elif target in ('shape', 'ion', 'sid'):
        s[target][0] = {'shape': 4, 'ion': 3, 'sid': 65536}[target]
    elif target in ('plane', 'panel', 'straw', 'upanel'):
        s[target][0] += 1
    elif target == 'nsgs': s['nsgs'][0] = 2
    elif target == 'foreign-event': s['event'][0] = 99
    elif target == 'foreign-sp': s['sp'][0] = 99
    elif target == 'duplicate-particle': p['key'][1] = p['key'][0]
    elif target == 'extra-field': s['unknown'] = np.zeros(4, 'i4')
    elif target == 'gen-nan': g['e'][0] = np.nan
    elif target == 'gen-dtype': g['idx'] = g['idx'].astype('u4')
    with pytest.raises(ValueError):
        rt.compare([path], write_table(tmp_path / 'bad.root', data), events)


def test_gen_payload_fully_compared(tmp_path):
    events, data, path = pair(tmp_path)
    data['gen']['px'][0] = np.nextafter(0., 1.)
    report = rt.compare([path], write_table(tmp_path / 'b.root', data), events)
    assert report['sgs']['equal'] and not report['equal']
    assert report['gen']['fields']['px']['numeric_mismatches'] == 1


def test_particles_deliberately_lossy_and_key_based_diagnostic_only(tmp_path):
    events, data, path = pair(tmp_path)
    for k in ('ccode', 'scode', 'genidx', 'isprim', 'ismcprim', 'pdg', 'parent', 'sx', 'et', 'se', 'ee'):
        data['particles'][k][0] += 1
    data = reorder(data, particles=[2, 1, 0])
    report = rt.compare([path], write_table(tmp_path / 'b.root', data), events)
    assert report['equal'] and not report['particles']['equal']
    assert not report['particles']['gating']
    assert report['particles']['fields']['se']['numeric_mismatches'] == 1
    assert report['particles']['fields']['ismcprim']['numeric_mismatches'] == 1
    assert report['particles']['fields']['key']['numeric_mismatches'] == 0
    assert 'original PrimaryParticle pointer keys' in ' '.join(report['limitations'])


def test_particle_missing_extra_keys_diagnostic_and_optional_schema(tmp_path):
    events, data, path = pair(tmp_path)
    data['particles']['key'] = np.concatenate([data['particles']['key'], np.array([80], 'i4')])
    data['particles'] = {k: v if k == 'key' else np.concatenate([v, v[:1]])
                         for k, v in data['particles'].items()}
    report = rt.compare([path], write_table(tmp_path / 'b.root', data), events)
    assert report['equal'] and report['particles']['extra_rows'] == 1
    del data['particles']
    report = rt.compare([path], write_table(tmp_path / 'c.root', data), events)
    assert report['equal'] and not report['particles']['available']


def test_highest_cycle_bound_not_stale_first_cycle(tmp_path):
    events, data, path = pair(tmp_path)
    imported = write_table(tmp_path / 'b.root', data)
    data['sgsteps']['px'][0] += 1.
    with uproot.update(imported) as root:
        root.mktree('dump/sgsteps', {k: v.dtype for k, v in data['sgsteps'].items()})
        root['dump/sgsteps'].extend(data['sgsteps'])
    report = rt.compare([path], imported, events)
    assert not report['equal']
    assert report['trees']['imported']['sgsteps']['tree'].endswith('sgsteps;2')


@pytest.mark.parametrize('which', ['gen', 'sgsteps'])
def test_rntuple_not_accepted_as_frozen_ttree(tmp_path, which):
    events, data, path = pair(tmp_path)
    imported = write_table(tmp_path / 'b.root', data)
    with uproot.update(imported) as root:
        root['dump/' + which] = data[which]
    with pytest.raises(ValueError, match='TTree'):
        rt.compare([path], imported, events)


def test_changed_input_bytes_rejected(tmp_path, monkeypatch):
    events, data, path = pair(tmp_path)
    imported = write_table(tmp_path / 'b.root', data)
    original_load = rt._load
    def mutate(p):
        result = original_load(p)
        if Path(p) == imported:
            with path.open('ab') as target:
                target.write(b'changed')
        return result
    monkeypatch.setattr(rt, '_load', mutate)
    with pytest.raises(ValueError, match='changed during'):
        rt.compare([path], imported, events)


def test_vectorized_interleaved_event_ordinals_and_no_event_loops():
    events = np.array([[1, 0, 3], [1, 0, 2], [1, 0, 3], [1, 0, 1], [1, 0, 2]])
    data = {k: events[:, i] for i, k in enumerate(rt.KEYS)}
    np.testing.assert_array_equal(rt._ordinal_keys(data)[:, -1], [0, 0, 1, 0, 1])
    source = inspect.getsource(rt)
    assert 'ak.to_list' not in source and 'import awkward' not in source
    # Loops are over files/columns only, never over individual event/SGS rows.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.For, ast.comprehension)):
            names = {n.id for n in ast.walk(node.iter) if isinstance(n, ast.Name)}
            assert not names & {'denominator', 'births', 'events'}
            assert not any(s in ast.unparse(node.iter) for s in ('range(len(rows))', 'range(len(events))'))


@pytest.mark.parametrize('difference', [False, True])
def test_cli_writes_difference_then_exit1_and_refuses_overwrite(tmp_path, monkeypatch, difference):
    events, data, path = pair(tmp_path)
    if difference:
        data['sgsteps']['pz'][0] += 1
    imported = write_table(tmp_path / 'b.root', data)
    art = tmp_path / 'denominator.root'
    with uproot.recreate(art) as root:
        root.mktree('Events', {'dummy': 'i4'})
        root['Events'].extend({'dummy': np.zeros(3, 'i4')})
    monkeypatch.setattr(rt.event_keys, 'read', lambda tree: events)
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['roundtrip', '--original-tables', str(path),
        '--imported-table', str(imported), '--denominator-art', str(art), '--output', str(output)])
    if difference:
        with pytest.raises(SystemExit) as exc:
            rt.main()
        assert exc.value.code == 1
    else:
        rt.main()
    report = json.loads(output.read_text())
    assert report['equal'] is not difference
    assert len(report['inputs_before']) == 3
    assert report['inputs_before'] == report['inputs_after']
    assert report['denominator_tree'].endswith('Events;1')
    saved = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        rt.main()
    assert output.read_bytes() == saved
    output.unlink()
    output.symlink_to(tmp_path / 'absent')
    with pytest.raises(ValueError, match='overwrite'):
        rt.main()


@pytest.mark.parametrize('field', ['shape', 'ion', 'sp', 'sid'])
def test_valid_integer_payload_differences_are_counted(tmp_path, field):
    events, data, path = pair(tmp_path)
    if field == 'sid':
        data['sgsteps']['sid'][0] += 1
        data['sgsteps']['straw'][0] += 1
    else:
        data['sgsteps'][field][0] = {'shape': 1, 'ion': 1, 'sp': 42}[field]
    report = rt.compare([path], write_table(tmp_path / 'b.root', data), events)
    assert not report['equal']
    assert report['sgs']['fields'][field]['numeric_mismatches'] == 1


def test_high_signed_particle_key_and_negative_pdg_preserved(tmp_path):
    events, data = payload()
    data['sgsteps']['sp'][:2] = np.iinfo('i4').max
    data['particles']['key'][0] = np.iinfo('i4').max
    data['particles']['parent'][1] = np.iinfo('i4').max
    data['particles']['pdg'][0] = -11
    report = rt.compare([write_table(tmp_path / 'a.root', data)],
                        write_table(tmp_path / 'b.root', data), events)
    assert report['equal'] and report['particles']['equal']


@pytest.mark.parametrize('which', ['gen', 'sgsteps'])
def test_missing_required_tree_rejected(tmp_path, which):
    events, data, path = pair(tmp_path)
    del data[which]
    with pytest.raises(ValueError, match='missing'):
        rt.compare([path], write_table(tmp_path / 'bad.root', data), events)


def test_imported_duplicate_gen_index_rejected(tmp_path):
    events, data, path = pair(tmp_path)
    data['gen'] = {k: np.concatenate([v, v[:1]]) for k, v in data['gen'].items()}
    with pytest.raises(ValueError, match='duplicate.*gen'):
        rt.compare([path], write_table(tmp_path / 'bad.root', data), events)
