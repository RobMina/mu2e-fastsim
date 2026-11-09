"""Passive R3 capture and full source-event identity wiring."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.reco import r3


class Recorder:
    def __init__(self):
        self.rows = []

    def add(self, c, p, o, event_keys, source_index=0):
        self.rows.append((event_keys.copy(), source_index, o))


def test_empty_capture_preserves_full_keys():
    recorder = Recorder()
    keys = np.array([[1400, 1155, 42], [1400, 1155, 43]], dtype=np.int64)
    r3._capture_empty(recorder, keys, 3)
    actual, source, measured = recorder.rows[0]
    np.testing.assert_array_equal(actual, keys)
    assert source == 3
    assert measured['nev'] == 2
    assert not measured['has_ex'].any()
    assert not measured['hit_evt'].size


def test_cache_retains_run_subrun_event_and_generated_empty(monkeypatch):
    steps = pd.DataFrame({'run': [1400, 1400], 'subrun': [1155, 1155], 'event': [43, 42], 'sp': [0, 0]})
    particles = steps.rename(columns={'sp': 'key'}).assign(pdg=13)
    gen = pd.DataFrame({'run': [1400]*3, 'subrun': [1155]*3, 'event': [42, 43, 44]})
    monkeypatch.setattr(r3.tables, 'expand', lambda paths: paths)
    monkeypatch.setattr(r3.tables, 'read', lambda path, tree, *a: {'steps': steps, 'particles': particles, 'gen': gen}[tree])
    c = r3.load_cache(['test.root'], {})[0]
    np.testing.assert_array_equal(c['event_keys'], [[1400, 1155, 42], [1400, 1155, 43]])
    np.testing.assert_array_equal(c['gen_keys'], gen.to_numpy())
    assert c['ngen'] == 3


@pytest.mark.parametrize('objective,max_passes', [(False, 1), (True, 2)])
def test_capture_requires_one_pass_objective(objective, max_passes):
    with pytest.raises(ValueError, match='one pass'):
        r3.run([], joint_capture=Recorder(), objective=objective, max_passes=max_passes)


def test_no_charge_capture_does_not_change_response_or_rng(monkeypatch):
    from crvfastsim.objective import crvresponse as R
    cfg = SimpleNamespace(sipm=SimpleNamespace(thermal_rate=0, trap0_prob=0, trap1_prob=0),
                          window=(0, 1e5), jax=False)
    keys = np.array([[1400, 1155, 42]], dtype=np.int64)
    c = dict(ek=np.array([42]), events=np.array([42]), steps=pd.DataFrame({'x': [0]}),
             pdg=np.array([13]), event_keys=keys)
    monkeypatch.setattr(r3.chain, 'charges', lambda *a, **kw: {'time': np.array([])})
    monkeypatch.setattr(r3.chain, 'background_sampler', lambda *a: None)
    states = []
    recorder = Recorder()
    for capture in (None, recorder):
        state = R.new_state()
        obj = dict(mod=R, state=state, joint=capture, source_index=0)
        rng = np.random.default_rng(123)
        r3.run_batch(c, 0, 1, None, cfg, None, None, None, None, rng, None, {}, obj)
        states.append((state, rng.bit_generator.state))
    assert states[0] == states[1]
    np.testing.assert_array_equal(recorder.rows[0][0], keys)
