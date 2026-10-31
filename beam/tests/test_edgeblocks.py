"""Fixed chord-block aggregation preserves primary identities and observed budgets."""
import numpy as np
import pytest

from beamfastsim import edgeblocks as eb, foilreport as fr
from test_edgeangles import Chunks
from test_foilclosure import fixture


@pytest.mark.parametrize('chunk', [1, 2, 3, 100])
def test_primary_projection_streams_without_losing_rows(chunk):
    p, s, case = fixture()
    _, ref = fr.census(p, s, case)
    projected = eb.primary_rows(Chunks(s), ref, step_size=chunk)
    np.testing.assert_array_equal(projected['evt'], s['evt'])
    out = eb.blocks(projected, ref, np.array([.1, 3.]), np.ones(2), .15)
    assert len(out['event']) == 1
    assert out['event'][0] == 0
    assert out['budget'][0] == pytest.approx(.2)
    assert out['fine_pairs'][0] == 2
    assert out['theta'][0] == 0


def test_coarse_angular_increment_uses_endpoints_not_sum_of_fine_angles():
    p, s, case = fixture()
    s['px'][2], s['pz'][2] = s['pz'][2], 0.
    _, ref = fr.census(p, s, case)
    coarse = eb.blocks(s, ref, np.array([.1, 3.]), np.ones(2), .15)
    fine = eb.blocks(s, ref, np.array([.1, 3.]), np.ones(2), .05)
    assert coarse['theta'][0] == 0.
    np.testing.assert_allclose(fine['theta'], [np.pi/2, np.pi/2])
    assert coarse['budget'].sum() == pytest.approx(fine['budget'].sum())


def test_census_and_domain_refuse_extrapolation():
    p, s, case = fixture()
    _, ref = fr.census(p, s, case)
    with pytest.raises(ValueError, match='domain'):
        eb.blocks(s, ref, np.array([.1, 1.]), np.ones(2), .1)
    ref['exit'][0] = False
    with pytest.raises(ValueError, match='census'):
        eb.blocks(s, ref, np.array([.1, 3.]), np.ones(2), .1)
