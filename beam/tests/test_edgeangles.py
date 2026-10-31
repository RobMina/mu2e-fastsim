"""Identity-bound angular reductions at first exit, including chunk boundaries."""
import numpy as np
import pytest

from beamfastsim import edgeangles, foilreport
from test_foilclosure import fixture


class Chunks:
    def __init__(self, arrays):
        self.arrays = arrays

    def iterate(self, fields, *, step_size, library):
        for start in range(0, len(self.arrays['evt']), step_size):
            yield {k: self.arrays[k][start:start + step_size] for k in fields}


@pytest.mark.parametrize('chunk', [1, 2, 3, 100])
def test_first_exit_matches_existing_census_and_stops_are_nan(chunk):
    p, s, case = fixture()
    _, ref = foilreport.census(p, s, case)
    angles = edgeangles.first_exit(Chunks(s), ref, step_size=chunk)
    np.testing.assert_array_equal(angles['resolved_exit'], [True, False])
    np.testing.assert_array_equal(angles['direction'][0], [0., 0., 1.])
    assert np.all(np.isnan(angles['direction'][1]))
    result = edgeangles.summary(ref['exit_MeV'], ref['exit'], angles['direction'], 1.)
    assert result['all_exits']['mean_deflection_cosine'] == 1.
    assert result['above_threshold_exits']['n'] == 1


def test_direction_census_and_position_must_match():
    p, s, case = fixture()
    _, ref = foilreport.census(p, s, case)
    bad = dict(ref, exit=np.array([False, False]))
    with pytest.raises(ValueError, match='census'):
        edgeangles.first_exit(Chunks(s), bad, step_size=2)
    bad = dict(ref, exit_z_mm=ref['exit_z_mm'] + .01)
    with pytest.raises(ValueError, match='position'):
        edgeangles.first_exit(Chunks(s), bad, step_size=2)


def test_conditioned_angular_summary_does_not_select_stoppers_as_zero_angle():
    direction = np.array([[0., 0., 1.], [1., 0., 0.], [np.nan, np.nan, np.nan]])
    result = edgeangles.summary([.2, .05, 0.], [True, True, False], direction, 1.)
    assert result['all_exits']['n'] == 2
    assert result['all_exits']['mean_deflection_cosine'] == .5
    assert result['above_threshold_exits']['n'] == 1
    assert result['above_threshold_exits']['mean_deflection_cosine'] == 1.
