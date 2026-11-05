"""Exercise the complete-validator/log/ROOT boundary with event-distinct synthetic states."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import full_replay
import full_validate
import path_state
import replay


class Root:
    def __init__(self, tree):
        self.tree = tree
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __getitem__(self, name):
        assert name == 'Events'
        return self.tree
    def keys(self, **kwargs):
        return ['Events;1']


@pytest.fixture
def case(tmp_path, monkeypatch):
    config = '''Digitize: {
      physics: { DigitizePath: [ Digi ] p: [ Digi, Cut ] trigger_paths: [ DigitizePath, "10:p" ]
        producers: { Digi: { module_type: TestDigi } } filters: { Cut: { module_type: TestFilter } } }
      outputs: { Output: { SelectEvents: [ p ] } }
      services: { GeometryService: {} DbService: {} }
    }'''
    result, manifest = replay.prepare(replay.production_config(config, 'Digitize'))
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    paths = [tmp_path / 'unfiltered_1.art', tmp_path / 'unfiltered_2.art']
    for path in paths:
        path.write_bytes(path.name.encode())
    result['source']['fileNames'] = [str(path) for path in paths]
    fcl = full_replay.render_config(result)
    manifest.pop('stock_output_selection')
    manifest.update(expected_events=3, config_sha256=hashlib.sha256(config.encode()).hexdigest(),
                    replay_fcl_sha256=hashlib.sha256(fcl.encode()).hexdigest(),
                    input_integrity=full_replay.integrity.files(paths))
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    (prepared / 'replay.fcl').write_text(fcl)
    census = {'files': [{'filename': path.name} for path in paths]}
    (prepared / 'regeneration-census.json').write_text(json.dumps(census))
    keys = np.array([[1430, 133, 1], [1430, 133, 3], [1430, 55, 2]], dtype=np.uint64)
    words = np.array([[5, 9], [5, 10], [5, 9]], dtype=np.uint16)
    candidate = np.array([[5], [6], [5]], dtype=np.uint16)
    (tmp_path / 'decisions.art').write_bytes(b'synthetic output metadata')
    trees = {str(paths[0]): SimpleNamespace(keys=keys[:2][::-1], stock=words[:2][::-1]),
             str(paths[1]): SimpleNamespace(keys=keys[2:], stock=words[2:]),
             str(tmp_path / 'decisions.art'): SimpleNamespace(keys=keys.copy(), stock=words.copy(), fast=candidate,
                                                           num_entries=3, classname='TTree')}
    monkeypatch.setattr(full_validate.uproot, 'open', lambda path, **kwargs: Root(trees[str(path)]))
    monkeypatch.setattr(path_state, 'check_schema', lambda root: None)
    monkeypatch.setattr(path_state.schedule_registry, 'verify', lambda *args: {'registry_schedule_verified': True})
    monkeypatch.setattr(full_validate.event_keys, 'read', lambda tree: tree.keys)
    monkeypatch.setattr(path_state, 'read_status', lambda tree, process, n: (tree.stock, 'stock-id')
                        if process == 'Digitize' else (tree.fast, 'replay-id'))
    lines = []
    for key, passed in zip(keys, [True, False, True]):
        value = 'pass' if passed else 'fail'
        lines.extend([f'PrintModule Run/Subrun/Event {key[0]} {key[1]} {key[2]}',
                      'ProductPrint art::TriggerResults_TriggerResults__Digitize', f'  {value} 10 p', '',
                      'ProductPrint art::TriggerResults_TriggerResults__TriggerReplay', f'  {value} 10 p', ''])
    log = '\n'.join(lines) + '\nArt has completed and will exit with status 0.\n'
    return SimpleNamespace(path=tmp_path / 'decisions.art', prepared=prepared, directory=tmp_path,
                           config=config, log=log, trees=trees, paths=paths)


def run(case):
    return full_validate.validate(case.path, case.prepared, case.directory, case.config, case.log)


def test_full_validator_keeps_parent_order_and_normalizes_list_identities(case):
    report = run(case)
    assert report['events'] == 3
    assert report['complete_first_pass_words_preserved'] and report['complete_parent_order_verified']
    assert report['all_menu_accepted_equal'] and report['all_menu_states_and_endpoints_equal']
    assert not report['raw_packed_equal'] and report['raw_index_mismatches'] == 3
    assert report['root_event_ids_independently_decoded'] and report['wasrun_mismatches'] == 0


def test_full_validator_rejects_changed_log_order(case):
    case.log = case.log.replace('1430 133 1', 'SWAP').replace('1430 55 2', '1430 133 1').replace('SWAP', '1430 55 2')
    with pytest.raises(ValueError, match='IDs/order'):
        run(case)


@pytest.mark.parametrize('change', ['fcl', 'input', 'census', 'root-order', 'retained-word'])
def test_full_validator_rechecks_bindings_and_retained_history(case, change):
    if change == 'fcl':
        path = case.prepared / 'replay.fcl'
        path.write_text(path.read_text().replace('TestFilter', 'OtherFilter'))
        error = 'FCL bytes'
    elif change == 'input':
        case.paths[0].write_bytes(b'changed')
        error = 'input bytes'
    elif change == 'census':
        path = case.prepared / 'regeneration-census.json'
        value = json.loads(path.read_text())
        value['files'].reverse()
        path.write_text(json.dumps(value))
        error = 'census file order'
    elif change == 'root-order':
        case.trees[str(case.path)].keys = case.trees[str(case.path)].keys[::-1]
        error = 'event identity order'
    else:
        case.trees[str(case.path)].stock[0, 1] += 1
        error = 'first-pass packed statuses'
    with pytest.raises(ValueError, match=error):
        run(case)
