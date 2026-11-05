"""Original production log binding uses real completion, seed, order and count checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

TRIGGER = Path(__file__).resolve().parents[1] / 'trigger'
sys.path.insert(0, str(TRIGGER))
import production_history_log


CONFIG = 'synthetic embedded configuration\n'
PARENTS = [
    {'configured_ordinal': 0, 'did': 'mu2e:dts.test.00012_00001.art',
     'url': 'root://door//pnfs/test/dts.test.00012_00001.art'},
    {'configured_ordinal': 1, 'did': 'mu2e:dts.test.00012_00007.art',
     'url': 'root://door//pnfs/test/dts.test.00012_00007.art'},
]
LOG = ('Opened input file "' + PARENTS[0]['url'] + '"\n'
       'services.SeedService.baseSeed: 9\n'
       'Begin processing event 1 run 12 subRun 4\n'
       'Begin processing event 2 run 12 subRun 9\n'
       'Opened input file "' + PARENTS[1]['url'] + '"\n'
       'TrigReport Events total = 20 passed = 19 failed = 1\n'
       'Art has completed and will exit with status 0.\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    audit = {'config_sha256': sha(CONFIG.encode()), 'parents': deepcopy(PARENTS),
             'seed_service': {'baseSeed': '9'}}
    calls = []

    def audited(text):
        calls.append(text)
        return deepcopy(audit)

    monkeypatch.setattr(production_history_log.source_audit, 'audit', audited)
    log_path = tmp_path / 'original-production.log'
    log_path.write_bytes(LOG.encode())
    manifest = {'config_sha256': audit['config_sha256'], 'parents': deepcopy(PARENTS),
                'local_parent_paths': ['/not-staged/' + p['did'].split(':', 1)[1] for p in PARENTS],
                'production_log_sha256': sha(LOG.encode()), 'original_processed_events': 20,
                'opened_parent_order_verified': True}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest_path, log_path, manifest, calls


def run(inputs):
    return production_history_log.verify(CONFIG, inputs[0], inputs[1])


def test_real_history_and_exact_evidence_bytes(inputs):
    manifest_path, log_path, manifest, calls = inputs
    result = run(inputs)
    assert calls == [CONFIG]
    assert result['production_history_log_verified'] is True
    assert result['opened_parent_order_verified'] is True
    assert result['original_processed_events'] == 20
    assert result['production_log_sha256'] == sha(log_path.read_bytes())
    assert result['config_sha256'] == sha(CONFIG.encode())
    assert result['parents'] == manifest['parents']
    assert result['local_parent_paths'] == manifest['local_parent_paths']
    assert result['regeneration_manifest_path'] == str(manifest_path.resolve())
    assert result['production_log_path'] == str(log_path.resolve())
    assert result['evidence_files'] == [
        {'path': str(path.resolve()), 'bytes': len(path.read_bytes()), 'sha256': sha(path.read_bytes())}
        for path in (manifest_path, log_path)]
    assert not Path(manifest['local_parent_paths'][0]).exists()


def test_mixed_subruns_do_not_infer_event_ids_from_parent_names(inputs):
    assert 'subRun 4' in LOG and '00012_00001' in LOG
    assert run(inputs)['production_history_log_verified']


@pytest.mark.parametrize('field,value', [
    ('config_sha256', '0' * 64), ('production_log_sha256', '0' * 64),
    ('original_processed_events', 19), ('original_processed_events', 20.0),
    ('original_processed_events', True), ('opened_parent_order_verified', False),
    ('opened_parent_order_verified', 1), ('parents', list(reversed(PARENTS))),
    ('parents', PARENTS[:1]), ('parents', []),
    ('local_parent_paths', []), ('local_parent_paths', ['bad.art', 'bad.art']),
    ('local_parent_paths', [None, '/parents/dts.test.00012_00007.art']),
    ('local_parent_paths', ['root://door/dts.test.00012_00001.art', '/parents/dts.test.00012_00007.art']),
])
def test_manifest_mutations_fail(inputs, field, value):
    inputs[2][field] = value
    inputs[0].write_text(json.dumps(inputs[2]))
    with pytest.raises(ValueError):
        run(inputs)


@pytest.mark.parametrize('field', ['config_sha256', 'production_log_sha256', 'original_processed_events',
                                   'opened_parent_order_verified', 'parents', 'local_parent_paths'])
def test_missing_required_manifest_fields_fail(inputs, field):
    del inputs[2][field]
    inputs[0].write_text(json.dumps(inputs[2]))
    with pytest.raises(ValueError):
        run(inputs)


@pytest.mark.parametrize('field,value', [('did', 'mu2e:dts.other.art'), ('url', 'root://other/file.art'),
                                        ('configured_ordinal', 99), ('extra', True)])
def test_exact_audited_parent_records_required(inputs, field, value):
    inputs[2]['parents'][0][field] = value
    inputs[0].write_text(json.dumps(inputs[2]))
    with pytest.raises(ValueError, match='parents differ'):
        run(inputs)


def test_local_parent_paths_must_follow_did_order(inputs):
    inputs[2]['local_parent_paths'].reverse()
    inputs[0].write_text(json.dumps(inputs[2]))
    with pytest.raises(ValueError, match='basename'):
        run(inputs)


@pytest.mark.parametrize('old,new', [
    ('status 0.', 'status 1.'), ('status 0.', 'status 0.\nArt has completed and will exit with status 0.'),
    ('services.SeedService.baseSeed: 9\n', ''), ('baseSeed: 9', 'baseSeed: 10'),
    ('baseSeed: 9', 'baseSeed: 9\nservices.SeedService.baseSeed: 9'),
    ('passed = 19', 'passed = 20'), ('total = 20 passed = 19 failed = 1', 'total = 0 passed = 0 failed = 0'),
    ('TrigReport Events total = 20 passed = 19 failed = 1\n', ''),
    ('TrigReport Events total = 20 passed = 19 failed = 1\n',
     'TrigReport Events total = 20 passed = 19 failed = 1\n' * 2),
    (PARENTS[0]['url'], PARENTS[1]['url']),
])
def test_actual_history_checks_fail_even_with_rebound_log_hash(inputs, old, new):
    changed = LOG.replace(old, new).encode()
    inputs[1].write_bytes(changed)
    inputs[2]['production_log_sha256'] = sha(changed)
    inputs[0].write_text(json.dumps(inputs[2]))
    with pytest.raises(ValueError):
        run(inputs)


def test_open_parent_order_is_checked_not_just_membership(inputs):
    changed = LOG.replace(PARENTS[0]['url'], 'PLACEHOLDER').replace(
        PARENTS[1]['url'], PARENTS[0]['url']).replace('PLACEHOLDER', PARENTS[1]['url']).encode()
    inputs[1].write_bytes(changed)
    inputs[2]['production_log_sha256'] = sha(changed)
    inputs[0].write_text(json.dumps(inputs[2]))
    with pytest.raises(ValueError, match='opened-parent order'):
        run(inputs)


def test_otherwise_valid_log_byte_mutation_fails_binding(inputs):
    inputs[1].write_bytes(LOG.encode() + b'unrelated log note\n')
    with pytest.raises(ValueError, match='log hash'):
        run(inputs)


def test_crlf_log_is_hashed_without_newline_translation(inputs):
    raw = LOG.replace('\n', '\r\n').encode()
    inputs[1].write_bytes(raw)
    inputs[2]['production_log_sha256'] = sha(raw)
    inputs[0].write_text(json.dumps(inputs[2]))
    assert run(inputs)['production_log_sha256'] == sha(raw)


def test_manifest_whitespace_is_bound_exactly(inputs):
    old = run(inputs)['evidence_files'][0]['sha256']
    inputs[0].write_bytes(inputs[0].read_bytes() + b'  \n')
    assert run(inputs)['evidence_files'][0]['sha256'] != old


@pytest.mark.parametrize('text', ['[]', 'null', 'true', '{'])
def test_nonobject_or_invalid_manifest_fails(inputs, text):
    inputs[0].write_text(text)
    with pytest.raises(ValueError):
        run(inputs)
