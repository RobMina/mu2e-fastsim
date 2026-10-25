"""Fail-closed general-chain and fused response boundary tests."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from crvfastsim import chain
from crvfastsim.response import contract


@pytest.mark.parametrize('jax,fused', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('field', ['thermal_rate', 'trap0_prob', 'trap1_prob'])
def test_chain_rejects_unsupported_before_inputs(jax, fused, field):
    cfg = chain.ChainConfig(jax=jax, fused_response=fused)
    cfg = replace(cfg, sipm=replace(cfg.sipm, **{field: .01}))
    with pytest.raises(ValueError, match=field):
        chain.charges(None, None, None, None, cfg, None)


@pytest.mark.parametrize('window', [(0, 100001), (0, np.nan), (10, 5), (0,), (0, np.inf)])
def test_chain_rejects_unsupported_window(window):
    with pytest.raises(ValueError, match='window'):
        chain.charges(None, None, None, None, chain.ChainConfig(jax=True, fused_response=True, window=window), None)


def test_named_windows_remain_supported():
    for window in [(0, 1e5), (200, 1500), (-100, 1500)]:
        contract.validate_window(window, fused=True)
    contract.validate_window((0, 200000), fused=False)


@pytest.mark.parametrize('flag', ['overflow', 'overflow_photons', 'overflow_sipm'])
def test_chain_overflow_before_host_conversion(monkeypatch, flag):
    pytest.importorskip('jax')
    from crvfastsim.response import device
    monkeypatch.setattr(device, 'response_batched', lambda *a, **kw: [{'overflow': False, flag: True}])
    monkeypatch.setattr(device, 'to_host_batches', lambda *a: pytest.fail('consumed truncated result'))
    cond = chain.Conditions(*(np.zeros(4) for _ in range(8)))
    steps = pd.DataFrame({'subrun': [1], 'event': [1]})
    timers = {}
    with pytest.raises(RuntimeError, match='overflow'):
        chain.charges(steps, None, None, cond, chain.ChainConfig(jax=True, fused_response=True),
                      np.random.default_rng(1), pdg=np.array([13]), timers=timers)
    assert timers['response_overflow_batches'] == 1


@pytest.mark.parametrize('entry', ['response', 'response_batched'])
def test_direct_device_rejects_unsupported(entry):
    pytest.importorskip('jax')
    from crvfastsim.response import device
    cfg = replace(chain.ChainConfig().sipm, trap0_prob=.1)
    with pytest.raises(ValueError, match='trap0_prob'):
        getattr(device, entry)(None, None, None, None, cfg, None, None, None, None, None, None, None)


@pytest.mark.parametrize('keep', [False, True])
def test_batch_refuses_overflow_even_in_timing_mode(monkeypatch, keep):
    jax = pytest.importorskip('jax')
    from crvfastsim.response import device
    monkeypatch.setattr(device, 'photon_estimate', lambda *a: np.ones(1))
    monkeypatch.setattr(device, 'response', lambda *a: {'overflow': True})
    with pytest.raises(RuntimeError, match='overflow'):
        device.response_batched(pd.DataFrame({'x': [0]}), np.array([13]), None, None, chain.ChainConfig().sipm,
                                None, None, None, np.array([1]), None, jax.random.PRNGKey(1), None,
                                keep=keep, budget=None)


def test_conversion_refuses_overflow():
    pytest.importorskip('jax')
    from crvfastsim.response import device
    with pytest.raises(RuntimeError, match='overflow'):
        device.to_host({'overflow_sipm': True})
    assert all(len(v) == 0 for v in device.to_host_batches([]).values())
