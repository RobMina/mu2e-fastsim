"""Fixed populations must be identical across adjacent-slice comparisons."""
import numpy as np
import pytest
from beamfastsim.slicestudy import fixed_population, verify_hashes


def test_fixed_populations_and_decomposition():
    g = dict(run=np.ones(4), subrun=np.zeros(4), evt=np.arange(4), time=np.zeros(4))
    arms = {}
    for s in (4, 8, 16, 32):
        arms[s] = dict(fate=np.array([1, 1, 1, 6]), foil=np.array([0, 1 + (s > 4), 2, -1]),
                       t=np.array([10., 20., 30., 40.]) + s / 4, x=np.zeros((4, 3)))
    arms[32]['fate'][2] = 6
    result = fixed_population(g, arms)
    assert result['common_stoppers']['rows'] == [0, 1]
    assert result['same_foil_all_arms']['rows'] == [0]
    assert result['changed_foil_any_arm']['rows'] == [1]
    assert result['common_stoppers']['differences']['32_minus_4']['time_ns']['mean'] == 7
    assert result['common_stoppers']['differences']['8_minus_4']['foil_index']['mean'] == 0.5
    arms[32]['t'][0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        fixed_population(g, arms)


def test_hash_gate_compares_content_and_order():
    verify_hashes([{'sha256': 'a', 'path': 'x'}], [{'sha256': 'a', 'path': 'y'}], 'test')
    with pytest.raises(ValueError, match='changed'):
        verify_hashes([{'sha256': 'a'}], [{'sha256': 'b'}], 'test')


def test_empty_intersection_is_explicit():
    g = dict(run=np.ones(1), subrun=np.zeros(1), evt=np.ones(1), time=np.zeros(1))
    arms = {s: dict(fate=np.array([6]), foil=np.array([-1]), t=np.array([2.]), x=np.zeros((1, 3)))
            for s in (4, 8, 16, 32)}
    result = fixed_population(g, arms)
    assert result['common_stoppers']['n'] == 0
    assert result['common_stoppers']['differences']['8_minus_4']['time_ns']['mean'] is None
