import json

import numpy as np

from beamfastsim.tshistory import material_episodes, x_plane_samples, _plain, summarize
from beamfastsim.tscrossings import TARGETS
from beamfastsim import tstrace, tstransport as ts
from test_tstransport import Field, Geometry, source


def history():
    return dict(material_row=np.zeros(4, dtype=int), material_step=np.array([0, 1, 3, 3]),
                material_order=np.array([0, 0, 1, 0]), material_name=np.array(['a', 'a', 'b', 'a']),
                material_requested_mm=np.array([1., 2., 3., 4.]),
                material_fraction=np.array([1., .5, .25, 1.]), material_stopped=np.array([False, False, True, False]),
                material_p_before=np.array([[10., 0, 0], [9., 0, 0], [5., 0, 0], [7., 0, 0]]),
                material_p_after=np.array([[9., 0, 0], [8., 0, 0], [0., 0, 0], [5., 0, 0]]),
                step_row=np.zeros(4, dtype=int), step_step=np.arange(4),
                step_x0=np.column_stack([np.arange(4), np.zeros((4, 2))]),
                step_x1=np.column_stack([np.arange(4) + 1, np.zeros((4, 2))]),
                step_t0=np.arange(4, dtype=float), step_t1=np.arange(4, dtype=float) + 1,
                step_path0=np.arange(4, dtype=float), step_path1=np.arange(4, dtype=float) + 1)


def test_episode_order_gap_and_partial_material_accounting():
    e = material_episodes(history(), 0)
    assert e['count'] == 3
    assert e['material'] == ['a', 'a', 'b']
    assert e['first_step'] == [0, 3, 3]
    assert e['last_step'] == [1, 3, 3]
    assert e['calls'] == [2, 1, 1]
    assert e['used_mm'] == [2., 4., .75]
    assert e['p_before'] == [10., 7., 5.]
    assert e['p_after'] == [8., 5., 0.]
    assert e['stopped'] == [False, False, True]
    assert material_episodes(history(), 1) == {'count': 0}
    json.dumps(e, allow_nan=False)


def test_x_planes_keep_repeats_directions_and_endpoint_convention():
    a = dict(step_row=np.zeros(3, dtype=int), step_step=np.arange(3),
             step_x0=np.array([[2., 0, 0], [0., 0, 0], [2., 0, 0]]),
             step_x1=np.array([[0., 0, 0], [2., 0, 0], [0., 0, 0]]),
             step_t0=np.arange(3, dtype=float), step_t1=np.arange(3, dtype=float) + 1,
             step_p0=np.tile([3., 0, 0], (3, 1)), step_p1=np.tile([2., 0, 0], (3, 1)))
    c = x_plane_samples(a, 0, [1., 0.])
    assert c['plane'] == [0, 1, 0, 0, 1]
    assert c['direction'] == [-1, -1, 1, -1, -1]
    assert c['step'] == [0, 0, 1, 2, 2]
    assert c['t'] == [.5, 1., 1.5, 2.5, 3.]
    assert c['p_before'] == [3.] * 5
    assert c['p_after'] == [2.] * 5
    assert x_plane_samples(a, 2, [1.])['step'] == []
    reversed_planes = x_plane_samples(a, 0, [0., 1.])
    assert reversed_planes['plane'] == [1, 0, 1, 1, 0]
    assert np.all(np.diff(reversed_planes['t']) >= 0)


def test_saved_trace_and_stock_report_roundtrip(tmp_path):
    trace = tstrace.Trace(np.array([0, 1]))
    s = source(n=2)
    f = ts.transport(s, Field(), Geometry(), max_path_mm=1, trace=trace)
    a = trace.arrays()
    a.update(target_id=np.array(TARGETS), target_row=np.array([0, 1]))
    a.update({'terminal_' + k: f[k] for k in ('x', 'p', 't', 'path', 'material_path', 'steps', 'fate')})
    fast = tmp_path / 'trace.npz'
    np.savez(fast, **a)
    fast.with_suffix('.json').write_text(json.dumps(dict(seed=20261111, exact_replay=True)))
    stock = dict(target_id=np.array(TARGETS), reference_p=s['p'],
                 vd_target=np.repeat([0, 1], 4), vd_sid=np.tile([3, 4, 5, 6], 2),
                 ds_target=np.array([0, 1]))
    for prefix, size in (('vd_', 8), ('ds_', 2)):
        stock.update({prefix + k: np.zeros(size) for k in ('x', 'y', 'z', 'time', 'momentum_mev_c')})
    stock_path = tmp_path / 'stock.npz'
    np.savez(stock_path, **stock)
    report = summarize(stock_path, [fast])
    assert len(report['runs'][0]['targets']) == 2
    assert report['runs'][0]['targets'][0]['episodes']['count'] == 0
    assert report['runs'][0]['targets'][0]['plane_crossings']['plane'] == []
    json.dumps(report, allow_nan=False)


def test_numpy_scalars_are_json_serializable():
    assert json.loads(json.dumps(_plain(dict(i=np.int32(2), x=np.float64(1.))))) == dict(i=2, x=1.)
