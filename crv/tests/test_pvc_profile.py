"""Exact comparison diagnostics must distinguish byte equality from tolerances."""
import numpy as np
import pytest

from crvfastsim.reco.pvc_profile import _differences


def test_exact_bytes_detect_signed_zero_and_small_float_changes():
    a = {'x': np.array([0., 1., np.nan])}
    b = {'x': np.array([-0., np.nextafter(1., 2.), np.nan])}
    result = _differences(a, b)['x']
    assert result['shape_equal'] and not result['bit_identical']
    assert result['max_abs'] == np.nextafter(1., 2.) - 1.


def test_identical_nan_payload_passes_and_shape_mismatch_fails():
    x = np.array([1., np.nan])
    assert _differences({'x': x}, {'x': x.copy()})['x']['bit_identical']
    assert not _differences({'x': x}, {'x': x[:1]})['x']['shape_equal']
    assert not _differences({'x': x}, {'x': x.astype(np.float32)})['x']['bit_identical']


def test_boolean_and_empty_arrays():
    a = dict(found=np.array([True, False]), empty=np.zeros(0))
    assert all(x['bit_identical'] for x in _differences(a, a).values())


def test_selected_comparison_ignores_only_unreported_slots():
    from crvfastsim.reco.pvc_profile import _selected_differences
    a = {'pulse__found': np.array([[True, False]]), 'pulse__pe': np.array([[1., 2.]])}
    b = {**a, 'pulse__pe': np.array([[1., 99.]])}
    result = _selected_differences(a, b)['pulse__']
    assert result['selection_equal'] and result['selected'] == 1
    assert all(v['bit_identical'] for v in result['columns'].values())
    b['pulse__pe'][0, 0] = 1.01
    assert not _selected_differences(a, b)['pulse__']['columns']['pulse__pe']['bit_identical']
    b['pulse__found'] = np.array([[True, True]])
    assert not _selected_differences(a, b)['pulse__']['selection_equal']


def test_node_summary_reports_units_and_missing_workers(tmp_path):
    import json
    from crvfastsim.reco.pvc_profile import _node_summary
    tile = tmp_path / 'node_cpu' / 'tile0'
    tile.mkdir(parents=True)
    record = dict(metadata=dict(rows=4096), measurements=[dict(resident_s=[2., 4., 3.], compile_s=1.)])
    (tile / 'profile.json').write_text(json.dumps(record))
    result = _node_summary(tmp_path)
    assert result['cpu']['workers'] == 1 and not result['cpu']['complete']
    assert result['cpu']['bottleneck_rate'] == dict(waveforms_per_s=4096 / 3)
    assert result['oneapi'] == dict(workers=0, complete=False)


def test_trace_summary_counts_duration_events_only(tmp_path):
    import gzip
    import json
    from crvfastsim.reco.pvc_profile import _trace_summary
    events = [dict(ph='X', name='eigh', dur=12), dict(ph='X', name='eigh', dur=8),
              dict(ph='M', name='metadata', dur=99)]
    with gzip.open(tmp_path / 'test.trace.json.gz', 'wt') as out:
        json.dump(dict(traceEvents=events), out)
    summary = _trace_summary(tmp_path)
    assert len(summary['files']) == 1
    assert summary['duration_events'] == [dict(name='eigh', calls=2, total_us=20)]


def test_timed_waits_for_all_jax_leaves():
    jax = pytest.importorskip('jax')
    from crvfastsim.reco.pvc_profile import _timed
    value, dt = _timed(lambda: {'x': jax.numpy.arange(4)})
    np.testing.assert_array_equal(value['x'], np.arange(4))
    assert dt >= 0
