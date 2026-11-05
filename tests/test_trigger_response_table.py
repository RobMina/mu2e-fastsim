"""Full-stock admission uses real synthetic ROOT TTrees, never a primary-only gate."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import response_table as rt


def payload():
    keys = np.array([[2, 3, 9], [2, 3, 4], [1, 0, 8]], dtype='i8')

    def identities(rows):
        return {k: rows[:, i].copy() for i, k in enumerate(rt.KEYS)}

    g = identities(keys)
    g.update({k: np.ones(3, dtype='f8') for k in rt.GEN_STATE})
    g.update(idx=np.zeros(3, dtype='i4'), pdg=np.array([13, -11, 11]), genid=np.ones(3, dtype='i4'))
    p = identities(keys[[0, 0, 0, 0, 1]])
    p.update({k: np.ones(5, dtype='f8') for k in rt.PART_STATE})
    # Root is not MC primary, MC primary is a child, extra root has no gen link.
    p.update(key=np.array([7, 91, 42, 18, 7]), parent=np.array([-1, 7, 91, -1, -1]),
             pdg=np.array([13, 11, 1000010020, 13, -11]), isprim=np.array([1, 0, 0, 0, 1]),
             ismcprim=np.array([0, 1, 0, 0, 1]), genidx=np.array([0, -1, -1, -1, 0]),
             ccode=np.zeros(5, dtype='i4'), scode=np.zeros(5, dtype='i4'))
    s = identities(keys[[0, 0, 0, 0, 1]])
    s.update({k: np.zeros(5, dtype='i4') for k in rt.INTS + ['plane', 'panel']})
    s.update({k: np.ones(5, dtype='f8') for k in rt.FLOATS})
    s.update(nsgs=np.array([4, 4, 4, 4, 1]), sp=np.array([91, 42, 91, 7, 7]),
             # Not sorted by time, straw, or particle key: preserve these rows.
             t=np.array([9., 3., 1., 7., -5.]), eion=np.array([1., 2., 3., 4., 5.]),
             sid=np.array([0, 128, 0, 1, 0]), panel=np.array([0, 1, 0, 0, 0]),
             straw=np.array([0, 0, 0, 1, 0]), upanel=np.array([0, 1, 0, 0, 0]))
    return keys, {'gen': g, 'particles': p, 'sgsteps': s}


def write_table(path, data):
    with uproot.recreate(path) as root:
        for name, columns in data.items():
            root.mktree('dump/' + name, {k: v.dtype for k, v in columns.items()})
            root['dump/' + name].extend(columns)
    return path


def test_full_stock_all_species_repeated_rows_no_rewriting(tmp_path):
    keys, data = payload()
    table = write_table(tmp_path / 'stock.root', data)
    before = table.read_bytes()
    report = rt.audit([table], keys)
    assert table.read_bytes() == before
    assert report['admitted'] and report['events'] == 3
    assert report['event_keys_in_original_order'] == keys.tolist()
    assert report['per_event']['steps'] == [4, 1, 0]
    assert report['per_event']['eion'] == [10., 5., 0.]
    assert report['generator_rows'] == 3 and report['particles'] == 5
    assert report['steps'] == 5 and report['zero_step_events'] == 1
    assert report['species_pdg']['1000010020']['steps'] == 1
    assert report['species_pdg']['13']['steps'] == 1
    assert report['species_pdg']['11']['eion'] == 4
    assert report['same_particle_same_sid_extra_rows'] == 1
    assert report['table_hashes_before'] == report['table_hashes_after'] == report['inputs']
    assert not report['source_collection_presence_proven']
    assert not report['particle_productid_lossless'] and not report['particle_state_lossless']
    assert report['sgs_rows_filtered'] == 0
    assert report['tables'][0]['ordered_payload_sha256']['sgsteps'] == rt._ordered_hash(data['sgsteps'])
    json.dumps(report, allow_nan=False)


def test_multifile_exact_traversal_and_empty_schemas(tmp_path):
    keys, data = payload()
    first = {name: {k: v[columns['event'] == 9] for k, v in columns.items()} for name, columns in data.items()}
    second = {name: {k: v[columns['event'] != 9] for k, v in columns.items()} for name, columns in data.items()}
    paths = [write_table(tmp_path / 'a.root', first), write_table(tmp_path / 'b.root', second)]
    assert rt.audit(paths, keys)['steps'] == 5
    with pytest.raises(ValueError, match='coverage/order'):
        rt.audit(paths[::-1], keys)
    for name in ('particles', 'sgsteps'):
        data[name] = {k: v[:0] for k, v in data[name].items()}
    report = rt.audit([write_table(tmp_path / 'empty.root', data)], keys)
    assert report['particles'] == report['steps'] == 0
    assert report['zero_step_events'] == 3 and report['species_pdg'] == {}
    assert not report['source_collection_presence_proven']


def test_multiple_generators_and_parented_isprim_are_not_forbidden(tmp_path):
    keys, data = payload()
    data['gen'] = {k: np.insert(v, 1, v[0]) for k, v in data['gen'].items()}
    data['gen']['idx'][1] = 1
    data['particles']['genidx'][3] = 1
    # isPrimary() is generator-pointer + process based, not parent-null based.
    data['particles']['isprim'][3] = 1
    data['particles']['parent'][3] = 7
    assert rt.audit([write_table(tmp_path / 'multi.root', data)], keys)['generator_rows'] == 4


@pytest.mark.parametrize('tree,field', [(tree, field) for tree, fields in [
    ('gen', rt.KEYS + rt.GEN_INTS + rt.GEN_STATE),
    ('particles', rt.KEYS + rt.PART_INTS + rt.PART_STATE),
    ('sgsteps', rt.KEYS + rt.INTS + ['plane', 'panel'] + rt.FLOATS)] for field in fields])
@pytest.mark.parametrize('empty', [False, True])
def test_every_required_schema_field_including_empty_trees(tmp_path, tree, field, empty):
    keys, data = payload()
    if empty:
        data[tree] = {k: v[:0] for k, v in data[tree].items()}
    del data[tree][field]
    with pytest.raises(ValueError, match='schema'):
        rt.audit([write_table(tmp_path / 'missing.root', data)], keys)


@pytest.mark.parametrize('tree,field', [(tree, field) for tree, fields in [
    ('gen', rt.GEN_STATE), ('particles', rt.PART_STATE), ('sgsteps', rt.FLOATS)] for field in fields])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_all_generator_particle_and_sgs_finite_fields(tmp_path, tree, field, bad):
    keys, data = payload()
    data[tree][field][0] = bad
    with pytest.raises(ValueError, match='finite'):
        rt.audit([write_table(tmp_path / 'nonfinite.root', data)], keys)


@pytest.mark.parametrize('tree,field', [(tree, field) for tree, fields in [
    ('gen', rt.KEYS + rt.GEN_INTS), ('particles', rt.KEYS + rt.PART_INTS),
    ('sgsteps', rt.KEYS + rt.INTS + ['plane', 'panel'])] for field in fields])
def test_integer_fields_never_coerced_from_float(tmp_path, tree, field):
    keys, data = payload()
    data[tree][field] = data[tree][field].astype('f8')
    with pytest.raises(ValueError, match='integer'):
        rt.audit([write_table(tmp_path / 'float.root', data)], keys)


@pytest.mark.parametrize('tree,field,bad', [
    ('gen', 'idx', -1), ('gen', 'genid', -1), ('gen', 'genid', 2**31), ('gen', 'e', -1),
    ('particles', 'key', -1), ('particles', 'key', 2**31), ('particles', 'parent', -2),
    ('particles', 'genidx', -2), ('particles', 'isprim', 2), ('particles', 'ismcprim', -1),
    ('particles', 'ccode', -1), ('particles', 'scode', -1), ('particles', 'pdg', 2**31),
    ('particles', 'se', -1), ('particles', 'ee', -1), ('particles', 'spx', 1e40),
    ('sgsteps', 'shape', 4), ('sgsteps', 'ion', 3), ('sgsteps', 'sid', 65536),
    ('sgsteps', 'plane', 36), ('sgsteps', 'panel', 6), ('sgsteps', 'straw', 96),
    ('sgsteps', 'upanel', 216), ('sgsteps', 'nsgs', -1), ('sgsteps', 'sp', -1),
    ('sgsteps', 'eion', -1), ('sgsteps', 'len', -1), ('sgsteps', 'width', -1), ('sgsteps', 'px', 1e40)])
def test_discrete_and_numeric_ranges(tmp_path, tree, field, bad):
    keys, data = payload()
    if abs(bad) > np.iinfo('i4').max and data[tree][field].dtype.kind in 'iu':
        data[tree][field] = data[tree][field].astype('i8')
    data[tree][field][0] = bad
    with pytest.raises(ValueError):
        rt.audit([write_table(tmp_path / 'range.root', data)], keys)


@pytest.mark.parametrize('case', ['particle-duplicate', 'gen-duplicate', 'gen-gap', 'missing-event',
    'foreign-particle', 'foreign-step', 'foreign-sp', 'parent-dangling', 'parent-other-event',
    'parent-self', 'parent-cycle', 'parent-long-cycle', 'gen-link', 'gen-pdg', 'missing-primary-link',
    'counts', 'filter', 'sid', 'panel', 'gen-order', 'particle-event-order', 'step-event-order'])
def test_keys_closure_associations_counts_and_order(tmp_path, case):
    keys, data = payload()
    g, p, s = data['gen'], data['particles'], data['sgsteps']
    if case == 'particle-duplicate': p['key'][1] = 7
    elif case == 'gen-duplicate':
        data['gen'] = {k: np.insert(v, 1, v[0]) for k, v in g.items()}
    elif case == 'gen-gap': g['idx'][0] = 1
    elif case == 'missing-event': data['gen'] = {k: v[:-1] for k, v in g.items()}
    elif case == 'foreign-particle': p['event'][0] = 100
    elif case == 'foreign-step': s['event'][0] = 100
    elif case == 'foreign-sp': s['sp'][0] = 18 + 1
    elif case == 'parent-dangling': p['parent'][1] = 999
    elif case == 'parent-other-event': p['parent'][-1] = 91
    elif case == 'parent-self': p['parent'][1] = 91
    elif case == 'parent-cycle': p['parent'][0] = 91
    elif case == 'parent-long-cycle': p['parent'][0] = 42
    elif case == 'gen-link': p['genidx'][0] = 1
    elif case == 'gen-pdg': p['pdg'][0] = 11
    elif case == 'missing-primary-link': p['genidx'][0] = -1
    elif case == 'counts': s['nsgs'][0] = 5
    elif case == 'filter': data['sgsteps'] = {k: np.delete(v, 1) for k, v in s.items()}
    elif case == 'sid': s['sid'][0] = 1024
    elif case == 'panel': s['panel'][0] = 1
    else:
        name = {'gen-order': 'gen', 'particle-event-order': 'particles', 'step-event-order': 'sgsteps'}[case]
        data[name] = {k: v[::-1] for k, v in data[name].items()}
    with pytest.raises(ValueError):
        rt.audit([write_table(tmp_path / 'bad.root', data)], keys)


@pytest.mark.parametrize('kind', ['duplicate', 'float', 'shape', 'negative', 'zero', 'overflow', 'empty', 'foreign', 'reverse'])
def test_denominator_validation(tmp_path, kind):
    keys, data = payload()
    table = write_table(tmp_path / 'stock.root', data)
    if kind == 'duplicate': keys = np.vstack([keys, keys[:1]])
    elif kind == 'float': keys = keys.astype(float)
    elif kind == 'shape': keys = keys[:, :2]
    elif kind == 'negative': keys[0, 0] = -1
    elif kind == 'zero': keys[0, 2] = 0
    elif kind == 'overflow': keys[0, 0] = 2**32
    elif kind == 'empty': keys = keys[:0]
    elif kind == 'foreign': keys[0, 2] = 123
    elif kind == 'reverse': keys = keys[::-1]
    with pytest.raises(ValueError): rt.audit([table], keys)


def test_duplicate_files_symlinks_and_tree_schemas(tmp_path):
    keys, data = payload()
    table = write_table(tmp_path / 'stock.root', data)
    alias = tmp_path / 'alias.root'
    alias.symlink_to(table)
    with pytest.raises(ValueError, match='unique'): rt.audit([table, alias], keys)
    with pytest.raises(ValueError, match='unique'): rt.audit([], keys)
    del data['particles']
    with pytest.raises(ValueError, match='TTree'): rt.audit([write_table(tmp_path / 'no-particles.root', data)], keys)


def test_long_acyclic_genealogy_vectorized_pointer_doubling(tmp_path):
    keys, data = payload()
    n = 4097
    p = {k: np.repeat(v[:1], n) for k, v in data['particles'].items()}
    p['key'] = np.arange(n)
    p['parent'] = np.arange(n) - 1
    p['isprim'][:] = 0
    p['genidx'][:] = -1
    p['pdg'][:] = 13
    data['particles'] = p
    for k in data['sgsteps']: data['sgsteps'][k] = data['sgsteps'][k][:0]
    assert rt.audit([write_table(tmp_path / 'long.root', data)], keys)['particles'] == n


def test_observed_step_order_hash_changes_not_step_sorting(tmp_path):
    keys, data = payload()
    a = rt.audit([write_table(tmp_path / 'a.root', data)], keys)
    swap = [2, 1, 0, 3, 4]
    data['sgsteps'] = {k: v[swap] for k, v in data['sgsteps'].items()}
    b = rt.audit([write_table(tmp_path / 'b.root', data)], keys)
    assert a['per_event'] == b['per_event']
    assert a['tables'][0]['ordered_payload_sha256']['sgsteps'] != b['tables'][0]['ordered_payload_sha256']['sgsteps']


def test_mutation_during_admission_is_rejected(tmp_path, monkeypatch):
    keys, data = payload()
    table = write_table(tmp_path / 'stock.root', data)
    original = rt._ordered_hash
    def mutate(data):
        result = original(data)
        with table.open('ab') as stream: stream.write(b'changed')
        return result
    monkeypatch.setattr(rt, '_ordered_hash', mutate)
    with pytest.raises(ValueError, match='changed during'): rt.audit([table], keys)


@pytest.mark.parametrize('name', ['surfsteps', 'trajpts'])
def test_optional_imported_tree_foreign_keys_and_finite_payload(tmp_path, name):
    keys, data = payload()
    extra = {k: v[:1].copy() for k, v in data['gen'].items() if k in rt.KEYS}
    if name == 'surfsteps':
        extra.update({k: np.ones(1) for k in ['edep', *rt.FLOATS[3:]]})
        extra.update(sp=np.array([7]), surf=np.array([1]), sindex=np.array([0]))
    else:
        extra.update({k: np.ones(1) for k in ['x', 'y', 'z', 't', 'ke']})
        extra.update(sp=np.array([7]), ipt=np.array([0]))
    data[name] = extra
    assert rt.audit([write_table(tmp_path / 'valid.root', data)], keys)['steps'] == 5
    extra['sp'][0] = 999
    with pytest.raises(ValueError): rt.audit([write_table(tmp_path / 'foreign.root', data)], keys)
    extra['sp'][0] = 7
    extra['t'][0] = np.inf
    with pytest.raises(ValueError, match='finite'): rt.audit([write_table(tmp_path / 'nan.root', data)], keys)


def test_cli_eventauxiliary_only_fresh_and_symlink_refusal(tmp_path, monkeypatch):
    keys, data = payload()
    table = write_table(tmp_path / 'stock.root', data)
    art = tmp_path / 'original.art'
    with uproot.recreate(art) as root:
        root.mktree('Events', {'metadata_seam': 'uint32'})
    reads = []
    def read(tree):
        reads.append(tree.name)
        assert tree.name == 'Events'
        return keys
    monkeypatch.setattr(rt.event_keys, 'read', read)
    output = tmp_path / 'receipt.json'
    monkeypatch.setattr(sys, 'argv', ['response_table', '--tables', str(table), '--denominator-art', str(art), '--output', str(output)])
    rt.main()
    assert reads == ['Events']
    report = json.loads(output.read_text())
    assert report['events'] == 3 and report['denominator_inputs'][0]['path'] == str(art)
    before = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite'): rt.main()
    assert output.read_bytes() == before
    output.unlink()
    output.symlink_to(tmp_path / 'absent')
    with pytest.raises(ValueError, match='overwrite'): rt.main()


def test_cli_denominator_mutation_and_output_race(tmp_path, monkeypatch):
    keys, data = payload()
    table = write_table(tmp_path / 'stock.root', data)
    art = tmp_path / 'original.art'
    with uproot.recreate(art) as root: root.mktree('Events', {'metadata_seam': 'uint32'})
    monkeypatch.setattr(rt.event_keys, 'read', lambda tree: keys)
    output = tmp_path / 'receipt.json'
    monkeypatch.setattr(sys, 'argv', ['response_table', '--tables', str(table), '--denominator-art', str(art), '--output', str(output)])
    original = rt.audit
    def mutate(*args):
        report = original(*args)
        with art.open('ab') as stream: stream.write(b'changed')
        return report
    monkeypatch.setattr(rt, 'audit', mutate)
    with pytest.raises(ValueError, match='denominator art bytes changed'): rt.main()
    assert not output.exists()
    def race(*args):
        report = original(*args)
        output.write_text('concurrent owner')
        return report
    monkeypatch.setattr(rt, 'audit', race)
    with pytest.raises(FileExistsError): rt.main()
    assert output.read_text() == 'concurrent owner'


def test_generator_link_on_nonprimary_and_unsigned_event_bounds(tmp_path):
    keys, data = payload()
    data['particles']['genidx'][3] = 0  # fromGenerator() need not be isPrimary().
    for name in data:
        for field in rt.KEYS:
            data[name][field] = data[name][field].astype('u8')
        data[name]['run'][:] = 2**32 - 1
    keys[:, 0] = 2**32 - 1
    assert rt.audit([write_table(tmp_path / 'valid.root', data)], keys)['events'] == 3


def test_duplicate_event_coverage_across_distinct_files(tmp_path):
    keys, data = payload()
    a = write_table(tmp_path / 'a.root', data)
    b = write_table(tmp_path / 'b.root', data)
    with pytest.raises(ValueError, match='coverage/order'):
        rt.audit([a, b], keys)


@pytest.mark.parametrize('tree,field', [('gen', 'x'), ('particles', 'sx'), ('sgsteps', 't')])
def test_nonscalar_and_string_payloads_rejected(tmp_path, tree, field):
    keys, data = payload()
    data[tree][field] = np.ones((len(data[tree][field]), 2), dtype='f8')
    table = tmp_path / 'matrix.root'
    with uproot.recreate(table) as root:
        for name, columns in data.items():
            schema = {k: np.dtype((v.dtype, v.shape[1:])) if v.ndim > 1 else v.dtype for k, v in columns.items()}
            root.mktree('dump/' + name, schema)
            root['dump/' + name].extend(columns)
    with pytest.raises(ValueError, match='non-scalar'):
        rt.audit([table], keys)
    # Actual ROOT string branch, not a validator-only mock.
    keys, data = payload()
    with uproot.recreate(tmp_path / 'string.root') as root:
        for name, columns in data.items():
            schema = {k: ('string' if name == tree and k == field else v.dtype) for k, v in columns.items()}
            root.mktree('dump/' + name, schema)
            if name == tree:
                columns[field] = ['bad'] * len(columns[field])
            root['dump/' + name].extend(columns)
    with pytest.raises(ValueError, match='finite'):
        rt.audit([tmp_path / 'string.root'], keys)
