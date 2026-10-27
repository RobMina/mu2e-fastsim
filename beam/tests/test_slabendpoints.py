"""Endpoint identification contracts; synthetic arrays only, no transport jobs."""
from copy import deepcopy
import json

import numpy as np
import pytest

from beamfastsim import slabdiffusion as sd, slabendpoints as se


def response(status, energy, face=None, z=None):
    s, e = np.asarray(status, np.int32), np.asarray(energy, float)
    p = np.zeros(e.shape + (3,))
    p[..., 2] = .25 if z is None else z
    if face is None:
        face = np.isin(s, [sd.TRANSMITTED, sd.REFLECTED])
    return sd.Response(e, s, np.zeros(e.shape), p, np.zeros_like(p), np.zeros(e.shape, int),
                       dict(face_reached=np.asarray(face), rawzero=s == sd.RAWZERO))


def test_singletons_reduce_to_empirical_cdf_and_mean():
    r = response([[sd.TRANSMITTED, sd.REFLECTED, sd.RAWZERO]], [[1., .05, 0.]])
    intervals = se.endpoint_intervals(r, [2., 1., 3.])
    for side in ('lower', 'upper'):
        np.testing.assert_array_equal(intervals['raw_' + side], r.raw_energy)
        np.testing.assert_array_equal(intervals['threshold_' + side], [[1., 0., 0.]])
    grid = [-np.inf, 0., .05, .1, 1., np.inf]
    cdf = se.cdf_envelope(intervals['raw_lower'], intervals['raw_upper'], grid)
    np.testing.assert_allclose(cdf['lower'], [0., 1/3, 2/3, 2/3, 1., 1.])
    np.testing.assert_array_equal(cdf['lower'], cdf['upper'])
    summary = se.summarize_endpoints(r, 3., 1., None)
    assert summary['raw']['fully_identified']
    assert summary['raw']['mean_bounds_MeV'] == pytest.approx([.35, .35])
    assert summary['thresholded']['mean_bounds_MeV'] == pytest.approx([1/3, 1/3])
    assert summary['raw']['interval_width_MeV']['max'] == 0.
    assert summary['floor']['n'] == 0
    assert summary['floor']['residual_energy_MeV']['mean'] is None
    json.dumps(summary, allow_nan=False)


def test_floor_cap_numerical_rawzero_and_broadcast_entry():
    r = response([[sd.UNRESOLVED_FLOOR, sd.UNRESOLVED_CAP, sd.UNRESOLVED_NUMERICAL],
                  [sd.TRANSMITTED, sd.REFLECTED, sd.RAWZERO]],
                 [[.02, .4, np.nan], [1., .05, 0.]])
    bounds = se.endpoint_intervals(r, [2., 3., 4.])
    np.testing.assert_array_equal(bounds['raw_lower'], [[0., 0., 0.], [1., .05, 0.]])
    np.testing.assert_array_equal(bounds['raw_upper'], [[.02, .4, 4.], [1., .05, 0.]])
    np.testing.assert_array_equal(bounds['threshold_upper'], [[0., .4, 4.], [1., 0., 0.]])
    assert all(x.shape == r.raw_energy.shape for x in bounds.values())
    report = se.summarize_endpoints(r, [2., 3., 4.], [1., 2., 3.], None)
    assert report['n'] == report['denominator'] == 6
    assert report['raw']['mean_bounds_MeV'] == pytest.approx([1.05/6, 5.47/6])
    assert report['thresholded']['mean_bounds_MeV'] == pytest.approx([1/6, 5.4/6])
    assert report['raw']['unidentified_count'] == 3
    assert report['thresholded']['unidentified_count'] == 2
    assert not report['raw']['fully_identified']
    assert not report['thresholded']['fully_identified']
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('residual', [np.nan, np.inf, -np.inf, -1., 0., .01, 7.])
def test_numerical_never_uses_even_plausible_residual(residual):
    r = response([sd.UNRESOLVED_NUMERICAL], [residual])
    bounds = se.endpoint_intervals(r, 2.)
    assert bounds['raw_lower'][0] == 0.
    assert bounds['raw_upper'][0] == bounds['threshold_upper'][0] == 2.


def test_exact_threshold_and_nextafter_edges():
    below, above = np.nextafter(.1, 0.), np.nextafter(.1, np.inf)
    r = response([sd.UNRESOLVED_FLOOR] * 3 + [sd.TRANSMITTED] * 3,
                 [below, .1, above, below, .1, above])
    b = se.endpoint_intervals(r, 1.)
    np.testing.assert_array_equal(b['threshold_upper'], [0., .1, above, 0., .1, above])
    np.testing.assert_array_equal(b['threshold_lower'], [0., 0., 0., 0., .1, above])
    unresolved = response([sd.UNRESOLVED_FLOOR, sd.UNRESOLVED_CAP], [.02, .09])
    report = se.summarize_endpoints(unresolved, 1., 1., None)
    assert not report['raw']['fully_identified']
    assert report['thresholded']['fully_identified']
    assert report['thresholded']['mean_bounds_MeV'] == [0., 0.]
    b0 = se.endpoint_intervals(r, 1., threshold=0.)
    np.testing.assert_array_equal(b0['raw_lower'], b0['threshold_lower'])
    np.testing.assert_array_equal(b0['raw_upper'], b0['threshold_upper'])


def test_cdf_envelope_order_denominator_inclusive_edges_and_grid_shape():
    lower, upper = np.array([[0., 1.], [0., 2.]]), np.array([[.5, 1.], [3., 2.]])
    grid = np.array([[-np.inf, 0., .5, 1.], [1.5, 2., 3., np.inf]])
    cdf = se.cdf_envelope(lower, upper, grid)
    np.testing.assert_array_equal(cdf['lower'], [[0., 0., .25, .5], [.5, .75, 1., 1.]])
    np.testing.assert_array_equal(cdf['upper'], [[0., .5, .5, .75], [.75, 1., 1., 1.]])
    assert np.all(cdf['lower'] <= cdf['upper'])
    assert np.all(np.diff(cdf['lower'].ravel()) >= 0)
    assert np.all(np.diff(cdf['upper'].ravel()) >= 0)
    unsorted = se.cdf_envelope(lower, upper, [3., 0., 1.])
    np.testing.assert_array_equal(unsorted['lower'], [1., 0., .5])
    assert se.cdf_envelope(lower, upper, 1.)['lower'] == .5


def test_face_rawzero_census_floor_geometry_and_no_mutation():
    r = response([sd.RAWZERO, sd.RAWZERO, sd.UNRESOLVED_FLOOR, sd.UNRESOLVED_FLOOR,
                  sd.TRANSMITTED, sd.UNRESOLVED_NUMERICAL], [0., 0., .03, .04, .5, np.nan],
                 face=[True, False, False, False, True, False], z=[1., .3, .2, np.nan, 1., np.nan])
    before = deepcopy(r)
    entry, thickness = np.ones(6), np.ones(6)
    b = se.endpoint_intervals(r, entry)
    se.cdf_envelope(b['raw_lower'], b['raw_upper'], [0., .1, 1.])
    s = se.summarize_endpoints(r, entry, thickness, object())
    census = s['status_face_reached_counts']
    assert sum(v['false'] + v['true'] for v in census.values()) == 6
    assert census[sd.STATUS_NAMES[sd.RAWZERO]] == {'false': 1, 'true': 1}
    assert census[sd.STATUS_NAMES[sd.UNRESOLVED_FLOOR]] == {'false': 2, 'true': 0}
    assert s['rawzero']['count'] == 2 and s['rawzero']['face_reached_count'] == 1
    assert s['floor']['residual_energy_MeV']['mean'] == pytest.approx(.035)
    assert s['floor']['normal_distance_to_front_mm']['mean'] == pytest.approx(.2)
    assert s['floor']['normal_distance_to_back_mm']['mean'] == pytest.approx(.8)
    assert s['floor']['normal_distance_to_nearest_face_mm']['mean'] == pytest.approx(.2)
    assert s['floor']['position_unavailable_count'] == 1
    assert s['csda_comparison'] is None
    assert 'conditional' in s['scope'] and 'physical-stop' in s['scope']
    json.dumps(s, allow_nan=False)
    for name in sd.Response._fields[:-1]:
        np.testing.assert_array_equal(getattr(r, name), getattr(before, name))
    for name in r.ledger:
        np.testing.assert_array_equal(r.ledger[name], before.ledger[name])
    np.testing.assert_array_equal(entry, np.ones(6))
    np.testing.assert_array_equal(thickness, np.ones(6))
    b['raw_upper'][0] = 99.
    assert r.raw_energy[0] == 0.


@pytest.mark.parametrize('status,energy', [([sd.ACTIVE], [1.]), ([999], [1.]),
    ([sd.TRANSMITTED], [0.]), ([sd.REFLECTED], [np.nan]), ([sd.RAWZERO], [.01]),
    ([sd.UNRESOLVED_FLOOR], [-.01]), ([sd.UNRESOLVED_CAP], [np.inf]),
    ([sd.UNRESOLVED_CAP], [2.]), ([], [])])
def test_invalid_endpoint_status_or_residual(status, energy):
    with pytest.raises(ValueError):
        se.endpoint_intervals(response(status, energy), 1.)


@pytest.mark.parametrize('entry', [0., -1., np.nan, np.inf, [1., 2.]])
def test_invalid_entry(entry):
    with pytest.raises(ValueError):
        se.endpoint_intervals(response([sd.TRANSMITTED], [.5]), entry)


@pytest.mark.parametrize('threshold', [-1., np.nan, np.inf, [.1], True])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        se.endpoint_intervals(response([sd.TRANSMITTED], [.5]), 1., threshold)


def test_invalid_shapes_and_noninteger_status():
    r = response([sd.TRANSMITTED], [.5])
    for status in (np.array([[sd.TRANSMITTED]]), np.array([1.]), np.array([True])):
        with pytest.raises(ValueError, match='status'):
            se.endpoint_intervals(r._replace(status=status), 1.)
    with pytest.raises(ValueError, match='position'):
        se.summarize_endpoints(r._replace(position=np.zeros(3)), 1., 1., None)
    for face in ([2], [np.nan], [[True]]):
        with pytest.raises(ValueError, match='face_reached'):
            se.summarize_endpoints(r._replace(ledger={'face_reached': face}), 1., 1., None)
    with pytest.raises(ValueError, match='face_reached'):
        se.summarize_endpoints(r._replace(ledger={}), 1., 1., None)
    with pytest.raises(ValueError, match='thickness'):
        se.summarize_endpoints(r, 1., 0., None)
    with pytest.raises(ValueError, match='position'):
        se.summarize_endpoints(response([sd.UNRESOLVED_FLOOR], [.02], z=2.), 1., 1., None)


@pytest.mark.parametrize('lower,upper,grid', [([], [], [0.]), ([0.], [[1.]], [0.]),
    ([np.nan], [1.], [0.]), ([0.], [np.inf], [0.]), ([1.], [0.], [0.]),
    ([-1.], [0.], [0.]), ([0.], [1.], [np.nan])])
def test_invalid_cdf_inputs(lower, upper, grid):
    with pytest.raises(ValueError):
        se.cdf_envelope(lower, upper, grid)
