"""Fast step admission keeps every original birth and rejects missing or foreign table events."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import step_table_audit as audit


def write(path, trees):
    with uproot.recreate(path) as root:
        for name, events in trees.items():
            columns = {'run': np.ones(len(events), dtype=np.uint32),
                'subrun': np.ones(len(events), dtype=np.uint32), 'event': np.asarray(events, dtype=np.uint32)}
            if name in ('gen', 'entries'):
                columns['pdg'] = np.full(len(events), 11, dtype=np.int32)
            if name == 'particles':
                columns.update({key: np.full(len(events), value, dtype=np.int32)
                    for key, value in {'key': 1, 'pdg': 11, 'parent': -1, 'isprim': 1}.items()})
            if name == 'sgsteps':
                columns.update({key: np.ones(len(events)) for key in ['sp', 'eion', 'len', 'width', 't',
                    'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'px', 'py', 'pz']})
            root.mktree('dump/' + name, {key: value.dtype for key, value in columns.items()})
            if events:
                root['dump/' + name].extend(columns)


@pytest.fixture
def data(tmp_path, monkeypatch):
    source, states, table, receipt = [tmp_path / name for name in ('source.art', 'states.root', 'table.root', 'handoff.json')]
    with uproot.recreate(source) as root:
        root.mktree('Events', {'event': 'uint32'})
        root['Events'].extend({'event': np.arange(1, 4, dtype=np.uint32)})
    monkeypatch.setattr(audit.event_keys, 'read', lambda tree: np.array([[1, 1, 1], [1, 1, 2], [1, 1, 3]], dtype=np.uint64))
    write(states, {'gen': [1, 2, 3], 'entries': [1, 2]})
    write(table, {'particles': [1, 2], 'sgsteps': [1, 1]})
    receipt.write_text(json.dumps({'generated_starts': 3, 'entries': 2, 'steps': 2, 'seed': 42}))
    return source, states, table, receipt


def test_explicit_zero_handoff_and_zero_step_events(data):
    report = audit.audit(*data, 3)
    assert report['no_handoff_events'] == 1 and report['zero_step_events'] == 2
    assert report['source_birth_keys_complete'] and report['transport_seed'] == 42


@pytest.mark.parametrize('failure', ['missing-birth', 'foreign-step', 'duplicate-particle', 'count'])
def test_bad_population_fails_closed(data, failure):
    source, states, table, receipt = data
    if failure == 'missing-birth':
        write(states, {'gen': [1, 2], 'entries': [1, 2]})
    elif failure == 'foreign-step':
        write(table, {'particles': [1, 2], 'sgsteps': [4]})
    elif failure == 'duplicate-particle':
        write(table, {'particles': [1, 1], 'sgsteps': [1]})
    else:
        receipt.write_text(json.dumps({'generated_starts': 4, 'entries': 2, 'steps': 2}))
    with pytest.raises(ValueError):
        audit.audit(*data, 3)
