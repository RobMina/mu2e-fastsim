"""Synthetic event census and packed-collision regressions."""
import pandas as pd
import pytest
from crvfastsim.veto.census_check import census_stats


def test_empty_gen_and_packed_collisions():
    ev = pd.DataFrame({'run': [1, 1, 1], 'subrun': [1, 1, 3],
                       'event': [7, 2**20 + 7, 9], 'ngen': [1, 1, 0], 'nsteps': [0, 0, 0]})
    gen = ev.iloc[:2][['run', 'subrun', 'event']]
    steps = gen.iloc[:0]
    result = census_stats(ev, gen, steps)
    assert result['events'] == 3
    assert result['empty_generator_events'] == 1
    assert result['packed_collisions_lost'] == 1
    assert result['legacy_packed_generator_events'] == 1
    with pytest.raises(ValueError, match='duplicate native'):
        census_stats(pd.concat([ev, ev]), gen, steps)
    bad = ev.copy()
    bad.loc[0, 'ngen'] = 2
    with pytest.raises(ValueError, match='ngen'):
        census_stats(bad, gen, steps)


def test_gen_only_sentinel():
    ev = pd.DataFrame({'run': [1], 'subrun': [0], 'event': [1], 'ngen': [0], 'nsteps': [-1]})
    gen = ev.iloc[:0][['run', 'subrun', 'event']]
    assert census_stats(ev, gen)['step_rows'] is None
    ev['nsteps'] = 0
    with pytest.raises(ValueError, match='sentinel'):
        census_stats(ev, gen)
