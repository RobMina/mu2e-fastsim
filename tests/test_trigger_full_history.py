"""Complete-history census and preparation do not waive missing-event/production checks."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import full_replay
import regeneration_census as census
import replay


class Root:
    def __init__(self, keys):
        self.tree = SimpleNamespace(classname='TTree', num_entries=len(keys), keys_for_test=np.asarray(keys, dtype=np.uint64))
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __getitem__(self, name):
        assert name == 'Events'
        return self.tree


def test_census_retains_parent_order_not_numeric_subrun_sort(tmp_path, monkeypatch):
    (tmp_path / 'unfiltered_1.art').touch()
    (tmp_path / 'unfiltered_2.art').touch()
    roots = {'unfiltered_1.art': [[1430, 133, 1], [1430, 133, 3]], 'unfiltered_2.art': [[1430, 55, 1]],
             'parent133': [[1430, 133, 3], [1430, 133, 1]], 'parent55': [[1430, 55, 1]]}
    monkeypatch.setattr(census.uproot, 'open', lambda path: Root(roots[Path(path).name]))
    monkeypatch.setattr(census.event_keys, 'read', lambda tree: tree.keys_for_test)
    manifest = {'split_input_files': True, 'parents': [{'did': 'mu2e:dts.test.001430_00000133.art'},
                                                     {'did': 'mu2e:dts.test.001430_00000055.art'}],
                'local_parent_paths': ['parent133', 'parent55'], 'original_processed_events': 3}
    result = census.census(manifest, tmp_path, 3)
    assert result['equal'] and result['complete_parent_event_ids_verified']
    roots['unfiltered_2.art'] = [[1430, 55, 2]]
    with pytest.raises(ValueError, match='original parent'):
        census.census(manifest, tmp_path, 3)
    roots['unfiltered_1.art'] = [[1430, 55, 1]]
    with pytest.raises(ValueError, match='run/subrun order'):
        census.census(manifest, tmp_path, 2)


def config():
    return '''Digitize: {
      physics: { DigitizePath: [ Digi ] p: [ Digi, Cut ] trigger_paths: [ DigitizePath, "10:p" ]
        producers: { Digi: { module_type: TestDigi } } filters: { Cut: { module_type: TestFilter } } }
      outputs: { Output: { SelectEvents: [ p ] } }
      services: { GeometryService: {} DbService: {} }
    }'''


def test_full_replay_requires_complete_validated_production_overlap(tmp_path, monkeypatch):
    counts = {'events': 3, 'complete_parent_event_ids_verified': True, 'files': [{'filename': 'a.art'}]}
    monkeypatch.setattr(full_replay.regeneration_census, 'census', lambda *args: counts)
    manifest = {'split_input_files': True, 'original_processed_events': 3,
                'config_sha256': hashlib.sha256(config().encode()).hexdigest()}
    (tmp_path / 'a.art').write_bytes(b'candidate')
    (tmp_path / 'reference.art').write_bytes(b'reference')
    production = {'equal': True, 'all_reference_events_present': True, 'regenerated_events': 3,
        'original_output_selection_event_set_equal': True,
        'config_sha256': manifest['config_sha256'], 'candidate_files': full_replay.integrity.files(tmp_path / 'a.art'),
        'reference_files': full_replay.integrity.files(tmp_path / 'reference.art')}
    result, receipt, _ = full_replay.prepare(config(), manifest, tmp_path, 3, production)
    assert 'stock_output_selection' not in receipt
    assert receipt['original_selected_output_patterns'] == ['p']
    assert result['source']['fileNames'] == [str(tmp_path / 'a.art')]
    production['all_reference_events_present'] = False
    with pytest.raises(ValueError, match='overlap'):
        full_replay.prepare(config(), manifest, tmp_path, 3, production)
    production['all_reference_events_present'] = True
    manifest['original_processed_events'] = 4
    with pytest.raises(ValueError, match='complete original'):
        full_replay.prepare(config(), manifest, tmp_path, 3, production)


def test_preflight_binds_exact_inputs_and_count(tmp_path):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    path = inputs / 'input.art'
    path.write_bytes(b'payload')
    manifest = {'expected_events': 3, 'config_sha256': hashlib.sha256(b'config').hexdigest(),
                'input_integrity': full_replay.integrity.files(path)}
    fcl_text = f'source: {{ fileNames: [ "{path}" ] }}'
    manifest['replay_fcl_sha256'] = hashlib.sha256(fcl_text.encode()).hexdigest()
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    (prepared / 'replay.fcl').write_text(fcl_text)
    assert full_replay.preflight(prepared, inputs, 'config', 3) == manifest
    for text, count in [('other', 3), ('config', 2)]:
        with pytest.raises(ValueError, match='count or configuration'):
            full_replay.preflight(prepared, inputs, text, count)
    path.write_bytes(b'different')
    with pytest.raises(ValueError, match='input bytes'):
        full_replay.preflight(prepared, inputs, 'config', 3)


def test_production_overlap_report_cannot_authorize_changed_candidate_bytes(tmp_path, monkeypatch):
    counts = {'events': 3, 'complete_parent_event_ids_verified': True, 'files': [{'filename': 'a.art'}]}
    monkeypatch.setattr(full_replay.regeneration_census, 'census', lambda *args: counts)
    path = tmp_path / 'a.art'
    path.write_bytes(b'old')
    reference = tmp_path / 'reference.art'
    reference.write_bytes(b'original')
    config_hash = hashlib.sha256(config().encode()).hexdigest()
    production = {'equal': True, 'all_reference_events_present': True, 'regenerated_events': 3,
        'config_sha256': config_hash, 'candidate_files': full_replay.integrity.files(path),
        'reference_files': full_replay.integrity.files(reference)}
    regeneration = {'split_input_files': True, 'original_processed_events': 3, 'config_sha256': config_hash}
    path.write_bytes(b'new')
    with pytest.raises(ValueError, match='different candidate'):
        full_replay.prepare(config(), regeneration, tmp_path, 3, production)
    production['candidate_files'] = full_replay.integrity.files(path)
    production['config_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='different configuration'):
        full_replay.prepare(config(), regeneration, tmp_path, 3, production)
