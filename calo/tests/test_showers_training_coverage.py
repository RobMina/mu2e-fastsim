"""Synthetic merge/authentication/orchestration tests; no real art or physics reads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calofastsim.showers import training_coverage as training


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return sha(path.read_bytes())


def part(event_rows, total, deposits, *, energy=100.):
    """Tiny independent synthetic event oracle (event loops only in tests)."""
    n, m = len(event_rows), len(deposits)
    e = dict(event_row=np.array(event_rows, dtype=np.int64), disk=np.arange(n) % 2,
             energy=np.full(n, energy), cosine=np.full(n, .8), radius=np.full(n, 300.),
             time=np.zeros(n), position=np.zeros((n, 3)), direction=np.tile([0., 0., 1.], (n, 1)))
    s = dict(entry=np.array([d[0] for d in deposits], dtype=np.int64),
             disk=np.array([d[1] for d in deposits], dtype=np.int64),
             energy_birks=np.array([d[2] for d in deposits], dtype=float),
             energy_g4=np.array([d[2] for d in deposits], dtype=float),
             crystal=np.arange(m, dtype=np.int64), time=np.zeros(m), position=np.zeros((m, 3)))
    nonfront = [i for i in range(len(total)) if i not in event_rows]
    empty_front = sum(not any(d[0] == i for d in deposits) for i in range(n))
    positive_nonfront = sum(total[i] > 0 for i in nonfront)
    c = dict(events=len(total), front_entry_events=n, non_front_entry_events=len(nonfront),
             empty_css_events=empty_front + sum(total[i] == 0 for i in nonfront),
             non_front_entry_positive_events=positive_nonfront,
             css_rows=m + positive_nonfront, retained_css_rows=m)
    return e, s, c, np.array(total, dtype=float)


def parts():
    return [part([1, 3], [9, 0, 0, 11], [(0, 1, 0), (1, 0, 4), (1, 1, 7)]),
            part([2], [0, 6, 0], [], energy=120.)]


def packed(piece):
    e, s, _, _ = piece
    lib = {'entry_' + k: v.copy() for k, v in e.items()}
    lib.update({'step_' + k: v.copy() for k, v in s.items() if k != 'entry'})
    lib['step_relative_position'] = lib.pop('step_position')
    lib['step_relative_time'] = lib.pop('step_time')
    lib['offsets'] = np.r_[0, np.cumsum(np.bincount(s['entry'], minlength=len(e['energy'])))]
    for key, edges in zip(('energy', 'cosine', 'radius'), training.library.Binning().arrays()):
        lib['bins_' + key] = edges
    lib['entry_cell'], lib['entry_outside_bins'] = training.library.cells(
        e['energy'], e['cosine'], e['radius'], training.library.Binning())
    return lib


def test_merge_uses_stored_offsets_not_front_offsets_and_preserves_inputs():
    pieces = parts()
    before = deepcopy(pieces)
    e, s, c, total = training.merge(pieces)
    np.testing.assert_array_equal(e['event_row'], [1, 3, 6])
    np.testing.assert_array_equal(s['entry'], [0, 1, 1])
    np.testing.assert_array_equal(total, [9, 0, 0, 11, 0, 6, 0])
    arrays, report = training.coverage.census(e, s, c, total)
    assert report['stored_events'] == 7 and report['non_front_entry_events'] == 4
    assert report['front']['counts'] == 3
    assert report['front']['empty'] == 1 and report['front']['zero_energy'] == 2
    assert report['front']['weak'] == 2
    assert report['front']['other_disk']['birks_sum'] == 4
    assert report['stored_census']['non_front_entry_positive_events'] == 2
    assert arrays['counts'].shape == (96,) and arrays['counts'].sum() == 2
    assert report['out_of_bins']['counts'] == 1
    for original, saved in zip(pieces, before):
        for actual, expected in zip(original[:2], saved[:2]):
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
        assert original[2] == saved[2]
        np.testing.assert_array_equal(original[3], saved[3])
    e['energy'][0] = 1
    total[0] = 1
    assert pieces[0][0]['energy'][0] == 100 and pieces[0][3][0] == 9


def test_merge_step_offsets_and_frontless_file():
    pieces = [part([], [5, 0], []), part([1], [0, 3], [(0, 0, 3)]),
              part([0], [8], [(0, 1, 8)])]
    e, s, c, total = training.merge(pieces)
    np.testing.assert_array_equal(e['event_row'], [3, 4])
    np.testing.assert_array_equal(s['entry'], [0, 1])
    assert training.coverage.census(e, s, c, total)[1]['stored_events'] == 5


@pytest.mark.parametrize('bad', [-1, True, 1., '1', np.uint64(1), {'nested': 1}])
def test_merge_rejects_noninteger_or_negative_census(bad):
    pieces = parts()
    pieces[0][2]['events'] = bad
    with pytest.raises(ValueError, match='nonnegative integer'):
        training.merge(pieces)


@pytest.mark.parametrize('table', [0, 1])
def test_merge_rejects_array_schema_mismatch(table):
    pieces = parts()
    pieces[1][table]['extra'] = np.array([])
    with pytest.raises(ValueError, match='schema mismatch'):
        training.merge(pieces)


def test_merge_rejects_no_parts():
    with pytest.raises(ValueError, match='no training parts'):
        training.merge([])


def test_extraction_manifest_and_required_artifacts(tmp_path):
    directory = tmp_path / 'file/export'
    directory.mkdir(parents=True)
    table = directory / 'calo_library_tables.root'
    table.write_bytes(b'synthetic numeric table')
    receipt = directory.parent / 'status.json'
    receipt.write_bytes(b'{}')
    mapping = {'file/export/calo_library_tables.root': sha(table.read_bytes()),
               'file/status.json': sha(receipt.read_bytes())}
    manifest = tmp_path / 'manifest.sha256'
    manifest.write_text(''.join(f'{digest}  {name}\n' for name, digest in mapping.items()))
    digest = sha(manifest.read_bytes())
    assert training.extraction_manifest(manifest, tmp_path, digest) == mapping
    validation = dict(tables_sha256=mapping['file/export/calo_library_tables.root'],
                      evidence_sha256={'status.json': mapping['file/status.json']})
    training._require_artifacts(mapping, 'file', validation)
    table.write_bytes(b'altered')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        training.extraction_manifest(manifest, tmp_path, digest)


@pytest.mark.parametrize('defect', ['wrong-pin', 'bad-pin', 'empty', 'absolute', 'traversal',
                                  'duplicate', 'alias-duplicate', 'bad-digest', 'malformed'])
def test_extraction_manifest_rejects_bad_authentication_or_paths(tmp_path, defect):
    artifact = tmp_path / 'artifact'
    artifact.write_bytes(b'x')
    line = f'{sha(b"x")}  artifact\n'
    text = line
    if defect == 'empty': text = ''
    elif defect == 'absolute': text = f'{sha(b"x")}  {artifact}\n'
    elif defect == 'traversal': text = f'{sha(b"x")}  ../artifact\n'
    elif defect == 'duplicate': text = line + line
    elif defect == 'alias-duplicate': text = line + line.replace('artifact', './artifact')
    elif defect == 'bad-digest': text = 'not-a-sha  artifact\n'
    elif defect == 'malformed': text = 'no separator'
    manifest = tmp_path / 'manifest'
    manifest.write_text(text)
    digest = sha(manifest.read_bytes())
    if defect == 'wrong-pin': digest = '0' * 64
    elif defect == 'bad-pin': digest = 'INVALID'
    with pytest.raises(ValueError):
        training.extraction_manifest(manifest, tmp_path, digest)


@pytest.mark.parametrize('key', ['file/export/calo_library_tables.root', 'file/status.json'])
@pytest.mark.parametrize('defect', ['missing', 'changed', 'wrong-file'])
def test_required_artifacts_fail_closed(key, defect):
    mapping = {'file/export/calo_library_tables.root': 'a' * 64, 'file/status.json': 'b' * 64}
    validation = dict(tables_sha256='a' * 64, evidence_sha256={'status.json': 'b' * 64})
    if defect == 'missing': del mapping[key]
    elif defect == 'changed': mapping[key] = 'c' * 64
    else: mapping['other/' + key] = mapping.pop(key)
    with pytest.raises(ValueError, match='missing from authenticated'):
        training._require_artifacts(mapping, 'file', validation)


def test_checked_json_is_byte_pinned_and_duplicate_safe(tmp_path):
    path = tmp_path / 'summary.json'
    digest = save_json(path, {'input_sha256': {'proof': 'a' * 64}})
    assert training._checked_json(path, digest)['input_sha256']['proof'] == 'a' * 64
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        training._checked_json(path, digest)


@pytest.mark.parametrize('blob', [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{'])
def test_checked_json_rejects_invalid_even_when_hash_matches(tmp_path, blob):
    path = tmp_path / 'summary.json'
    path.write_bytes(blob)
    with pytest.raises(ValueError):
        training._checked_json(path, sha(blob))


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    import uproot
    args = SimpleNamespace(**{key: tmp_path / key for key in (
        'inventory', 'tables', 'original_tables', 'extraction_manifest', 'evidence', 'bound',
        'parent_summary', 'reference_config', 'library', 'out')})
    args.inventory_sha256 = 'a' * 64
    args.reference_config.write_text('synthetic geometry service')
    args.library.write_bytes(b'synthetic frozen bytes')
    args.library_sha256 = sha(args.library.read_bytes())
    rows = [dict(name='training-one', fid='1', sha256='b' * 64),
            dict(name='training-two', fid='2', sha256='c' * 64)]
    cases = parts()
    validations, manifest, bindings = {}, {}, {}
    for row, piece in zip(rows, cases):
        directory = args.tables / row['name']
        table = directory / 'export/calo_library_tables.root'
        table.parent.mkdir(parents=True)
        table.write_bytes(row['name'].encode())
        config = directory / 'source_config.txt'
        config.write_text(args.reference_config.read_text())
        proof = {'input_sha256': {'printer_log': 'd' * 64}}
        validation = dict(tables_sha256=sha(table.read_bytes()), source_proof=proof,
                          evidence_sha256={'source_config.txt': sha(config.read_bytes())},
                          raw_reconciliation={'census': piece[2], 'geometry': {'disks': 2}})
        validations[row['name']] = validation
        manifest[row['name'] + '/export/calo_library_tables.root'] = validation['tables_sha256']
        manifest[row['name'] + '/source_config.txt'] = sha(config.read_bytes())
        bound = args.bound / row['name'] / 'bound/summary.json'
        bindings['/remote/' + row['name'] + '/bound/summary.json'] = save_json(
            bound, dict(input_sha256=proof['input_sha256'], art_sha256=row['sha256']))
    args.parent_sha256 = save_json(args.parent_summary, {'input_sha256': bindings})
    args.extraction_manifest.write_text(''.join(f'{v}  {k}\n' for k, v in manifest.items()))
    args.extraction_sha256 = sha(args.extraction_manifest.read_bytes())
    calls = []
    def inventory(*unused):
        calls.append('inventory')
        return rows
    def validate(table, inventory, row, evidence, directory, *, original_tables):
        assert original_tables == args.original_tables / row['name'] / 'export/calo_library_tables.root'
        calls.append(('validate', row['name']))
        return validations[row['name']]
    def read(table, *, input_mode):
        assert input_mode == 'primary'
        name = table.parent.parent.name
        calls.append(('read', name))
        e, s, c, total = cases[[r['name'] for r in rows].index(name)]
        return e, s, {'disk': np.array([0, 1])}, c, total
    class Root:
        def __enter__(self): return self
        def __exit__(self, *unused): pass
    def forbidden(*unused, **kwargs):
        pytest.fail('orchestration must not build, sample, run art, or launch commands')
    monkeypatch.setattr(training.training_audit, 'training_rows', inventory)
    monkeypatch.setattr(training.training_audit, 'validate_saved', validate)
    monkeypatch.setattr(training.training_audit, 'check_geometry_config', lambda a, b: a == b or forbidden())
    monkeypatch.setattr(training.export, 'read', read)
    monkeypatch.setattr(uproot, 'open', lambda *unused: Root())
    monkeypatch.setattr(training.production_check, 'geometry_check', lambda *unused: {'disks': 2})
    legacy = packed(part([0], [2], [(0, 0, 2)], energy=20.))
    monkeypatch.setattr(training.library, 'load', lambda path: (legacy, {}))
    monkeypatch.setattr(training.library, 'build', forbidden)
    monkeypatch.setattr(training.library, 'sample', forbidden)
    monkeypatch.setattr(training.source_audit.subprocess, 'run', forbidden)
    return SimpleNamespace(args=args, rows=rows, validations=validations, calls=calls, cases=cases)


def test_run_separate_frozen_population_complete_census_and_output_pins(orchestration):
    case = orchestration
    report = training.run(case.args)
    assert report['training']['stored_events'] == 7
    assert report['training']['front_entry_events'] == 3
    assert report['legacy_frozen']['front_entry_events'] == 1
    assert report['legacy_frozen']['stored_events'] is None
    assert report['comparison']['cell_count'] == 96
    assert report['comparison']['training_entries_in_bins'] == 2
    assert report['comparison']['legacy_entries_in_bins'] == 1
    assert report['candidate_library_constructed'] is False
    assert report['source_independence_certified'] is False
    assert case.calls.count('inventory') == 2
    assert [c[1] for c in case.calls if isinstance(c, tuple) and c[0] == 'read'] == [r['name'] for r in case.rows]
    for name, digest in report['output_sha256'].items():
        assert sha((case.args.out / name).read_bytes()) == digest
    assert json.loads((case.args.out / 'summary.json').read_text()) == report
    with pytest.raises(FileExistsError):
        training.run(case.args)


@pytest.mark.parametrize('defect, message', [
    ('manifest', 'manifest hash mismatch'), ('parent', 'JSON hash mismatch'),
    ('library', 'frozen library hash mismatch'), ('bound', 'JSON hash mismatch'),
    ('proof', 'source proof differs'), ('census', 'census differs'),
    ('geometry', 'geometry check differs'), ('receipt', 'missing from authenticated'),
    ('missing-binding', 'unique authenticated parent'), ('duplicate-binding', 'unique authenticated parent'),
])
def test_run_stops_before_outputs_on_authentication_or_reconciliation_failure(orchestration, defect, message):
    case, args = orchestration, orchestration.args
    name = case.rows[0]['name']
    if defect == 'manifest': args.extraction_manifest.write_text('changed')
    elif defect == 'parent': args.parent_summary.write_text('{}')
    elif defect == 'library': args.library.write_bytes(b'changed')
    elif defect == 'bound': (args.bound / name / 'bound/summary.json').write_text('{}')
    elif defect == 'proof': case.validations[name]['source_proof']['input_sha256']['printer_log'] = 'e' * 64
    elif defect == 'census': case.validations[name]['raw_reconciliation']['census'] = {'events': 999}
    elif defect == 'geometry': case.validations[name]['raw_reconciliation']['geometry'] = {'disks': 3}
    elif defect == 'receipt': case.validations[name]['evidence_sha256']['source_config.txt'] = 'f' * 64
    else:
        parent = json.loads(args.parent_summary.read_text())
        key = '/remote/' + name + '/bound/summary.json'
        if defect == 'missing-binding': del parent['input_sha256'][key]
        else: parent['input_sha256']['/duplicate' + key] = parent['input_sha256'][key]
        args.parent_sha256 = save_json(args.parent_summary, parent)
    with pytest.raises(ValueError, match=message):
        training.run(args)
    assert not args.out.exists()


@pytest.mark.parametrize('digest', ['0' * 64, 'ABC', 'A' * 64, None])
def test_checked_json_rejects_wrong_or_noncanonical_pin(tmp_path, digest):
    path = tmp_path / 'summary.json'
    path.write_text('{}')
    with pytest.raises(ValueError):
        training._checked_json(path, digest)


@pytest.mark.parametrize('target, message', [
    ('table', 'artifact hash mismatch'), ('receipt', 'artifact hash mismatch'),
    ('manifest', 'manifest hash mismatch'), ('parent', 'JSON hash mismatch'),
    ('reference', 'configuration changed'), ('frozen', 'frozen library changed'),
])
def test_run_rechecks_inputs_after_computation(orchestration, monkeypatch, target, message):
    args = orchestration.args
    directory = args.tables / orchestration.rows[0]['name']
    paths = dict(table=directory / 'export/calo_library_tables.root',
                 receipt=directory / 'source_config.txt', manifest=args.extraction_manifest,
                 parent=args.parent_summary, reference=args.reference_config, frozen=args.library)
    original = training.library.load
    def load(path):
        result = original(path)
        paths[target].write_bytes(paths[target].read_bytes() + b' changed')
        return result
    monkeypatch.setattr(training.library, 'load', load)
    with pytest.raises(ValueError, match=message):
        training.run(args)
    assert not args.out.exists()


def test_run_detects_table_change_during_file_census(orchestration, monkeypatch):
    original = training.export.read
    def read(path, **kwargs):
        result = original(path, **kwargs)
        path.write_bytes(path.read_bytes() + b' changed')
        return result
    monkeypatch.setattr(training.export, 'read', read)
    with pytest.raises(ValueError, match='tables changed during coverage'):
        training.run(orchestration.args)
    assert not orchestration.args.out.exists()


def test_run_rejects_different_geometry_tables_between_files(orchestration, monkeypatch):
    original = training.export.read
    def read(path, **kwargs):
        e, s, geometry, census, total = original(path, **kwargs)
        if path.parent.parent.name == orchestration.rows[1]['name']:
            geometry['disk'] = np.array([0, 2])
        return e, s, geometry, census, total
    monkeypatch.setattr(training.export, 'read', read)
    with pytest.raises(ValueError, match='geometry tables differ'):
        training.run(orchestration.args)
    assert not orchestration.args.out.exists()
