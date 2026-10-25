"""Explicit held-out input identities and immutable manifest checks."""
import json

import numpy as np
import pytest

from crvfastsim.objective import validation as V


def event_keys(events, run=1400):
    return np.array([(run, 1, e) for e in events], dtype=[('run', 'i8'), ('subrun', 'i8'), ('event', 'i8')])


def test_manifest_counts_and_bytes(tmp_path, monkeypatch):
    p = tmp_path / 'input.root'
    p.write_bytes(b'synthetic input')
    asset = tmp_path / 'ref.json'
    asset.write_text('{}')
    monkeypatch.setattr(V, 'keys', lambda path, tree: event_keys([1, 2, 3] if tree == 'gen' else [1, 3]))
    out = tmp_path / 'manifest.json'
    m = V.inventory([str(p)], [], {'reference': str(asset)}, out)
    assert (m['events'], m['events_with_steps']) == (3, 2)
    V.verify(json.loads(out.read_text()))
    asset.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='changed'):
        V.verify(m)
    with pytest.raises(FileExistsError):
        V.inventory([str(p)], [], {}, out)


@pytest.mark.parametrize('mode', ['duplicate', 'excluded', 'overlap', 'missinggen', 'run', 'encoding'])
def test_invalid_input_sets(tmp_path, monkeypatch, mode):
    p, q = tmp_path / 'a.root', tmp_path / 'b.root'
    p.write_bytes(b'a')
    q.write_bytes(b'b')
    def keys(path, tree):
        if mode == 'missinggen' and tree == 'steps':
            return event_keys([9])
        if mode == 'run':
            return event_keys([1], run=1400 if str(path) == str(p) else 1401)
        if mode == 'encoding':
            return event_keys([2 ** 20])
        return event_keys([1, 2])
    monkeypatch.setattr(V, 'keys', keys)
    paths = [str(p), str(p if mode == 'duplicate' else q)]
    with pytest.raises(ValueError):
        V.inspect_inputs(paths, [str(q)] if mode == 'excluded' else [])
