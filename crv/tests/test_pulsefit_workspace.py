"""Workspace reuse must not alter a fit, pulse flag, residual threshold or pulse order."""
import numpy as np
import pytest

from crvfastsim.reco import pulsefit as PF
from crvfastsim.reco import synth


def _edge_batch():
    adc, n, start, ped, area, height = synth.waveforms(8, seed=24, L=36)
    adc[0] = ped[0]
    adc[1, :11] = 4095
    adc[2, 0] = ped[2] - 10
    n[3] = 12
    return adc, n, start, ped, area, height


@pytest.mark.parametrize("inputs", [_edge_batch, lambda: synth.waveforms(20, seed=15)])
def test_workspace_identical(monkeypatch, inputs):
    args = inputs()
    monkeypatch.setenv("CRVFS_CPU_FIT_WORKSPACE", "0")
    reference = PF.reconstruct_full(*args)
    monkeypatch.setenv("CRVFS_CPU_FIT_WORKSPACE", "1")
    candidate = PF.reconstruct_full(*args)
    assert reference.keys() == candidate.keys()
    for key, value in reference.items():
        assert value.dtype == candidate[key].dtype
        assert value.shape == candidate[key].shape
        assert value.tobytes() == candidate[key].tobytes(), key


def test_workspace_switch_is_explicit_and_fail_closed(monkeypatch):
    args = synth.waveforms(1, seed=21)
    monkeypatch.setenv('CRVFS_CPU_FIT_WORKSPACE', 'typo')
    with pytest.raises(ValueError, match='CRVFS_CPU_FIT_WORKSPACE'):
        PF.reconstruct_full(*args)


@pytest.mark.parametrize('inputs', [_edge_batch, lambda: synth.waveforms(20, seed=15)])
def test_normal_equation_cache_identical(monkeypatch, inputs):
    args = inputs()
    monkeypatch.setenv('CRVFS_CPU_FIT_NORMAL_CACHE', '0')
    reference = PF.reconstruct_full(*args)
    monkeypatch.setenv('CRVFS_CPU_FIT_NORMAL_CACHE', '1')
    candidate = PF.reconstruct_full(*args)
    for key, value in reference.items():
        assert value.dtype == candidate[key].dtype
        assert value.shape == candidate[key].shape
        assert value.tobytes() == candidate[key].tobytes(), key
