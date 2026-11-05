"""Exact saved-only source SGS joins, validation and completed all52 admission."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import integrity
import stock_step_census as census
from test_trigger_stage_flow import saved  # Existing comprehensive saved-art mock seam.

REAL_OPEN = uproot.open

def payload():
    births = np.array([[1, 2, 1], [1, 2, 2], [1, 10, 3]], dtype='i8')
    def keys(rows):
        return {k: rows[:, i].copy() for i, k in enumerate(census.KEYS)}
    gen = keys(births)
    # Two electron associations: key 7 is primary by isprim, key 90 by ismcprim.
    particles = keys(births[[0, 0, 0, 1]])
    particles.update(key=np.array([7, 42, 90, 8]), pdg=np.array([11, 22, 11, -11]),
        parent=np.array([-1, 7, -1, 999]), isprim=np.array([1, 0, 0, 0]), ismcprim=np.array([0, 0, 1, 0]))
    steps = keys(births[[0, 0, 0, 1]])
    steps.update({k: np.zeros(4, dtype='i4') for k in census.INTS})
    steps.update({k: np.ones(4, dtype='f8') for k in census.FLOATS})
    steps.update(nsgs=np.array([3, 3, 3, 1]), sp=np.array([90, 42, 42, 8]),
                 eion=np.array([2., 3., 4., 5.]), len=np.array([10., 20., 30., 40.]))
    return births, {'gen': gen, 'particles': particles, 'sgsteps': steps}

def write_table(path, data):
    with uproot.recreate(path) as root:
        for name, columns in data.items():
            root.mktree('dump/' + name, {k: v.dtype for k, v in columns.items()})
            root['dump/' + name].extend(columns)
    return path

def test_exact_association_species_zeros_primary_and_multiplicity(tmp_path):
    keys, data = payload()
    table = write_table(tmp_path / 'source.root', data)
    values, report = census.census([table], keys[::-1])
    assert values['steps'].tolist() == [0, 1, 3]
    assert values['primary_steps'].tolist() == [0, 0, 1]
    assert values['other_steps'].tolist() == [0, 1, 2]
    assert values['primary_eion'].sum() == 2
    assert values['isprim_electron_steps'].sum() == 0
    assert values['ismcprim_electron_steps'].sum() == 1
    assert values['ismcprim_electron_eion'].sum() == 2
    assert values['ismcprim_electron_length'].sum() == 10
    assert report['species_pdg']['22'] == {'steps': 2, 'eion': 7., 'length': 50.}
    assert report['dangling_particle_parents'] == 1
    assert report['same_particle_same_sid_extra_rows'] == 1
    assert report['same_particle_same_sid_multirow_groups'] == 1
    assert report['primary_definition'] == 'pdg==11 and ismcprim==1'
    total = census.aggregate(values, np.ones(3, bool))
    assert total['events'] == 3 and total['steps'] == 4
    assert total['events_with_primary_steps'] == 1 and total['zero_step_events'] == 1
    # All three independent tree row orderings may change.
    for columns in data.values():
        for k, v in columns.items():
            columns[k] = v[::-1]
    other, _ = census.census([write_table(tmp_path / 'sorted.root', data)], keys[::-1])
    for name in values:
        np.testing.assert_array_equal(values[name], other[name])

def test_empty_schemas_and_isprim_fallback(tmp_path):
    keys, data = payload()
    for name in ('particles', 'sgsteps'):
        data[name] = {k: v[:0] for k, v in data[name].items()}
    del data['particles']['ismcprim']
    values, report = census.census([write_table(tmp_path / 'empty.root', data)], keys)
    assert values['steps'].sum() == 0 and not report['species_pdg']
    assert census.aggregate(values, np.ones(3, bool))['zero_step_events'] == 3
    assert report['primary_definition'] == 'pdg==11 and isprim==1'

@pytest.mark.parametrize('target', ['float-key', 'float-sp', 'float-pdg', 'foreign-step', 'foreign-particle',
    'foreign-sp', 'duplicate-particle', 'duplicate-gen', 'missing-gen', 'nsgs', 'nan', 'string',
    'shape', 'ion', 'sid', 'upanel', 'straw', 'negative-energy', 'negative-length', 'flag', 'parent', 'sid-mapping', 'panel-dtype'])
def test_reject_invalid_identity_dtype_payload_and_association(tmp_path, target):
    keys, data = payload()
    s, p, g = data['sgsteps'], data['particles'], data['gen']
    if target == 'float-key':
        g['event'] = g['event'].astype(float)
    elif target == 'float-sp':
        s['sp'] = s['sp'].astype(float)
    elif target == 'float-pdg':
        p['pdg'] = p['pdg'].astype(float)
    elif target == 'foreign-step':
        s['event'][0] = 999
    elif target == 'foreign-particle':
        p['event'][0] = 999
    elif target == 'foreign-sp':
        s['sp'][0] = 91  # No key arithmetic or key+1 fallback.
    elif target == 'duplicate-particle':
        p['key'][1] = p['key'][0]
    elif target == 'duplicate-gen':
        for k in census.KEYS:
            g[k][1] = g[k][0]
    elif target == 'missing-gen':
        data['gen'] = {k: v[:-1] for k, v in g.items()}
    elif target == 'nsgs':
        s['nsgs'][0] = 2
    elif target == 'nan':
        s['t'][0] = np.nan
    elif target == 'string':
        s['t'] = s['t'].astype('S4')
        # Uproot can't write fixed bytes as scalar branches: exercise validator via mock below.
        class Tree:
            def __init__(self, name): self.name = name
            def arrays(self, **kwargs): return data[self.name.split('/')[-1]]
        class Root:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __getitem__(self, name): return Tree(name)
        from unittest.mock import patch
        with patch.object(census.uproot, 'open', return_value=Root()), pytest.raises(ValueError):
            census.census(['mock'], keys)
        return
    elif target in ('shape', 'ion', 'sid', 'upanel', 'straw'):
        s[target][0] = {'shape': 4, 'ion': 3, 'sid': 65536, 'upanel': 216, 'straw': 96}[target]
    elif target.startswith('negative-'):
        s['eion' if target == 'negative-energy' else 'len'][0] = -1
    elif target == 'sid-mapping':
        s['sid'][0] = 128
    elif target == 'panel-dtype':
        s['panel'] = np.zeros(4, dtype=float)
    elif target == 'flag':
        p['ismcprim'][0] = 2
    elif target == 'parent':
        p['parent'][0] = -2
    with pytest.raises(ValueError):
        census.census([write_table(tmp_path / 'invalid.root', data)], keys)

def test_duplicate_tables_births_and_foreign_denominator(tmp_path):
    keys, data = payload()
    path = write_table(tmp_path / 'a.root', data)
    with pytest.raises(ValueError, match='duplicate'):
        census.census([path, path], keys)
    with pytest.raises(ValueError, match='duplicate'):
        census.census([path], np.vstack([keys, keys[:1]]))
    with pytest.raises(ValueError):
        census.census([path], keys[:2])

def test_fixed_truth_bins_and_cpr_migrations(tmp_path):
    keys, data = payload()
    values, _ = census.census([write_table(tmp_path / 'a.root', data)], keys)
    # Each source event appears in exactly one group, even with no SGS.
    report = census.migrations(values, np.array([[5], [5], [6]], 'u2'), np.array([[5], [6], [5]], 'u2'), ['cprCE'])
    groups = report['cprCE']
    assert groups['both_pass']['totals']['steps'] == 3
    assert groups['stock_only']['totals']['steps'] == 1
    assert groups['fast_only']['totals']['zero_step_events'] == 1
    assert groups['neither_pass']['totals']['events'] == 0
    assert groups['stock_only']['omitted_nonprimary_step_count']['1-9']['events'] == 1
    assert groups['fast_only']['omitted_nonprimary_step_count']['0']['events'] == 1
    assert groups['both_pass']['omitted_nonprimary_step_count']['10+']['events'] == 0
    values['other_steps'][0] = 10
    assert census.migrations(values, np.array([[5], [5], [6]], 'u2'), np.array([[5], [6], [5]], 'u2'), ['cprCE'])['cprCE']['both_pass']['omitted_nonprimary_step_count']['10+']['events'] == 1

@pytest.fixture
def admitted(saved, tmp_path, monkeypatch):
    _, data = payload()
    table = write_table(tmp_path / 'original.root', data)
    receipt = tmp_path / 'source-table-audit.json'
    receipt.write_text(json.dumps({'events': 3, 'source_birth_identities_verified': True,
        'inputs': integrity.files([table])}))
    saved_open = census.uproot.open
    monkeypatch.setattr(census.uproot, 'open', lambda path, **kw:
        REAL_OPEN(path, **kw) if Path(path) == table else saved_open(path, **kw))
    return saved, table, receipt

def analyze(admitted, receipt=True):
    s, table, audit = admitted
    return census.analyze([table], s.art, s.comparison_path, s.config, s.prepared, audit if receipt else None)

def test_saved_all52_exact_denominator_and_receipt(admitted):
    report = analyze(admitted)
    assert len(report['paths']) == len(report['flowreport']['paths']) == 52
    assert report['event_keys_in_saved_order'] == admitted[0].tree.keys.tolist()
    assert report['per_event']['steps'] == [0, 3, 1]
    assert report['paths']['p0']['stock_only']['totals']['zero_step_events'] == 1
    assert report['paths']['p0']['fast_only']['totals']['steps'] == 3
    assert all(sum(g['totals']['events'] for g in groups.values()) == 3 for groups in report['paths'].values())
    assert 'DIAGNOSTIC' in report['scope'] and 'SEM' in ' '.join(report['limitations'])
    assert report['inputs'][-1] == integrity.digest(admitted[2])

@pytest.mark.parametrize('target', ['no-receipt', 'bad-hash', 'bad-size', 'bad-events', 'missing52', 'art', 'config'])
def test_saved_admission_failures(admitted, target):
    saved, table, receipt = admitted
    if target.startswith('bad-'):
        obj = json.loads(receipt.read_text())
        if target == 'bad-events': obj['events'] = 4
        else: obj['inputs'][0]['sha256' if target == 'bad-hash' else 'bytes'] = 'changed'
        receipt.write_text(json.dumps(obj))
    elif target == 'missing52':
        del saved.comparison['paths']['p1']
        saved.comparison_path.write_text(json.dumps(saved.comparison))
    elif target in ('art', 'config'):
        path = getattr(saved, target)
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        analyze(admitted, receipt=target != 'no-receipt')

def test_manifest_table_hash_binding(admitted):
    saved, table, _ = admitted
    saved.manifest['injection_campaign_evidence'] = integrity.files([table])
    (saved.prepared / 'manifest.json').write_text(json.dumps(saved.manifest))
    saved.comparison['validation_evidence'] = integrity.files([saved.prepared / 'manifest.json', saved.prepared / 'replay.fcl'])
    saved.comparison_path.write_text(json.dumps(saved.comparison))
    assert analyze(admitted, receipt=False)['events'] == 3

def test_input_change_during_census_rejected(admitted, monkeypatch):
    original = census.census
    def mutate(*args):
        result = original(*args)
        admitted[2].write_text(admitted[2].read_text() + ' ')
        return result
    monkeypatch.setattr(census, 'census', mutate)
    with pytest.raises(ValueError, match='changed during'):
        analyze(admitted)

def test_cli_fresh_only(admitted, tmp_path, monkeypatch):
    saved, table, receipt = admitted
    output = tmp_path / 'out.json'
    monkeypatch.setattr(sys, 'argv', ['census', '--tables', str(table), '--decision-art', str(saved.art),
        '--comparison', str(saved.comparison_path), '--config', str(saved.config),
        '--prepared', str(saved.prepared), '--table-receipt', str(receipt), '--output', str(output)])
    census.main()
    assert json.loads(output.read_text())['events'] == 3
    before = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        census.main()
    assert output.read_bytes() == before
    output.unlink()
    output.symlink_to(tmp_path / 'absent')
    with pytest.raises(ValueError, match='overwrite'):
        census.main()


def test_zero_rows_are_not_source_collection_presence_certificate(tmp_path):
    keys, data = payload()
    for name in ('particles', 'sgsteps'):
        data[name] = {k: v[:0] for k, v in data[name].items()}
    _, report = census.census([write_table(tmp_path / 'zero.root', data)], keys)
    assert 'does not persist source collection presence' in report['zero_policy']
