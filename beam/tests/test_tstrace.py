import json

import numpy as np
import pytest

from beamfastsim import tstrace, tstransport as ts
from test_tstransport import Field, Geometry, source
from mu2efastsim import materials


def test_collector_replay_and_material_accounting():
    s = source(n=4, p=20)
    trace = tstrace.Trace(np.array([3, 1]))
    f = ts.transport(s, Field(), Geometry(materials.ALUMINIUM), seed=91, trace=trace)
    plain = ts.transport(s, Field(), Geometry(materials.ALUMINIUM), seed=91)
    checked = tstrace.exact_replay(f, plain)
    assert 'seconds' not in checked
    a = trace.arrays()
    assert set(a['step_row']) == {1, 3}
    assert set(a['material_row']) == {1, 3}
    for row in (1, 3):
        step = a['step_row'] == row
        mat = a['material_row'] == row
        np.testing.assert_array_equal(a['step_step'][step], np.arange(f['steps'][row]))
        np.testing.assert_array_equal(a['step_x1'][step][-1], f['x'][row])
        np.testing.assert_array_equal(a['step_p1'][step][-1], f['p'][row])
        assert a['step_fate'][step][-1] == ts.RANGEOUT
        assert np.all(np.diff(a['step_path0'][step]) >= 0)
        assert np.all(np.diff(a['step_t0'][step]) >= 0)
        np.testing.assert_allclose(np.sum(a['material_requested_mm'][mat] * a['material_fraction'][mat]), f['material_path'][row])
        np.testing.assert_array_equal(a['material_step'][mat], a['step_step'][step])
        assert a['material_stopped'][mat][-1]
        assert np.all(a['material_order'][mat] == 0)


def test_empty_and_cold_trace_have_typed_arrays():
    for rows, momentum in ((np.array([], dtype=int), 1000), (np.array([0]), 1)):
        trace = tstrace.Trace(rows)
        ts.transport(source(n=1, p=momentum), Field(), Geometry(), trace=trace)
        a = trace.arrays()
        assert a['step_x0'].shape == (0, 3)
        assert a['material_p_after'].shape == (0, 3)
        assert a['step_row'].dtype.kind == 'i'
        assert a['material_name'].dtype.kind == 'U'


def test_collector_records_applied_material_order():
    class Two(Geometry):
        def __init__(self):
            self.materials = {'first': materials.ALUMINIUM, 'second': materials.ALUMINIUM}
    trace = tstrace.Trace(np.array([0]))
    ts.transport(source(n=1, p=1000), Field(), Two(), max_path_mm=.01, trace=trace)
    a = trace.arrays()
    np.testing.assert_array_equal(a['material_name'], ['first', 'second'])
    np.testing.assert_array_equal(a['material_order'], [0, 1])
    np.testing.assert_array_equal(a['material_p_after'][0], a['material_p_before'][1])


def test_collector_rejects_incomplete_callback_sequence():
    trace = tstrace.Trace(np.array([0]))
    with pytest.raises(ValueError, match='without begin'):
        trace.end()
    with pytest.raises(ValueError, match='outside accepted step'):
        trace.material(name='x', row=np.array([0]))
    trace.begin(row=np.array([0]), step=np.array([0]))
    with pytest.raises(ValueError, match='without end'):
        trace.begin()
    with pytest.raises(ValueError, match='unfinished'):
        trace.arrays()
    with pytest.raises(ValueError, match='absent'):
        trace.material(name='x', row=np.array([1]))


def test_exact_replay_checks_keys_dtype_and_values():
    a = {'x': np.array([1.]), 'seconds': np.array(3.)}
    assert tstrace.exact_replay(a, {**a, 'seconds': np.array(5.)}) == ['x']
    for bad in ({'x': a['x']}, {**a, 'x': np.array([2.])}, {**a, 'x': np.array([1], dtype=int)}):
        with pytest.raises(ValueError, match='differ|mismatch'):
            tstrace.exact_replay(a, bad)


@pytest.mark.parametrize('changes', [{'seed': 20261114}, {'variant': 'half_step'}, {'settings': [5, .25, .05]}])
def test_driver_rejects_unselected_settings_before_loading(tmp_path, changes):
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(dict(seed=20261111, variant='baseline', settings=[2.5, .25, .05]) | changes))
    with pytest.raises(ValueError, match='authorized'):
        tstrace.replay([path], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
