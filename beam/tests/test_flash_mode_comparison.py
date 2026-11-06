"""Cross-JIT comparison allowances must not hide discrete, payload or chunking changes."""
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip('jax')

from test_flash_vacuum_fast import _exact, _bucket_roundoff, _roundoff_only, _float_ulps


def shifted(value, steps):
    result = np.float64(value)
    for _ in range(steps):
        result = np.nextafter(result, np.inf)
    return result


def fixture():
    return (np.int32(2), dict(u=np.array([[.25, -.125, 0.]]), t=np.array([.03]),
                            T=np.array([100.]), fate=np.array([1], np.int32),
                            slab=dict(it=np.int32(2)),
                            records=dict(valid=np.array([[False]]), direction=np.zeros((1, 1, 3)))))


@pytest.mark.parametrize('value', [.25, -.25])
def test_direction_budget_is_two_ulps_only(value):
    a = fixture()
    a[1]['u'][0, 0] = value
    b = deepcopy(a)
    b[1]['u'][0, 0] = shifted(value, 2)
    _bucket_roundoff(a, b)
    with pytest.raises(AssertionError):
        _exact(a, b)
    b[1]['u'][0, 0] = shifted(value, 3)
    with pytest.raises(AssertionError, match='u'):
        _bucket_roundoff(a, b)


@pytest.mark.parametrize('kind', ['energy', 'time', 'fate', 'clock', 'outer_clock', 'valid', 'payload',
                                  'schema', 'shape', 'dtype', 'outer_schema'])
def test_bucket_rejects_every_unapproved_change(kind):
    a, b = fixture(), fixture()
    if kind == 'energy':
        b[1]['T'][0] = shifted(b[1]['T'][0], 1)
    elif kind == 'time':
        b[1]['t'][0] = shifted(b[1]['t'][0], 1)
    elif kind == 'fate':
        b[1]['fate'][0] += 1
    elif kind == 'clock':
        b[1]['slab']['it'] += 1
    elif kind == 'outer_clock':
        b = (np.int32(3), b[1])
    elif kind == 'valid':
        b[1]['records']['valid'][0, 0] = True
    elif kind == 'payload':
        b[1]['records']['direction'][0, 0, 0] = 1e-15
    elif kind == 'schema':
        b[1]['extra'] = np.zeros(1)
    elif kind == 'shape':
        b[1]['u'] = b[1]['u'].reshape(3)
    elif kind == 'outer_schema':
        b = (*b, np.int32(0))
    else:
        b[1]['u'] = b[1]['u'].astype(np.float32)
    with pytest.raises(AssertionError):
        _bucket_roundoff(a, b)


@pytest.mark.parametrize('value', [-0., np.nextafter(0., 1.), np.nan, np.inf, -np.inf])
def test_direction_rejects_zero_or_nonfinite_changes(value):
    a, b = fixture(), fixture()
    b[1]['u'][0, 2] = value
    with pytest.raises(AssertionError):
        _bucket_roundoff(a, b)


def test_tilted_comparison_keeps_invalid_payload_exact():
    a, b = fixture(), fixture()
    b[1]['records']['direction'][0, 0, 0] = 1e-15
    with pytest.raises(AssertionError):
        _roundoff_only(a, b)
    b = deepcopy(a)
    b[1]['records']['valid'][0, 0] = True
    with pytest.raises(AssertionError):
        _roundoff_only(a, b)


def test_ulp_comparator_rejects_shape_dtype_and_sign_of_zero():
    for b in (np.array([.25], np.float32), np.array([[.25]]), np.array([np.nan])):
        with pytest.raises(AssertionError):
            _float_ulps(np.array([.25]), b, 2, 'named')
    with pytest.raises(AssertionError, match='named'):
        _float_ulps(np.array([0.]), np.array([-0.]), 2, 'named')


@pytest.mark.parametrize('value', [.03, -.03])
def test_tilted_time_budget_is_one_ulp_only(value):
    a, b = fixture(), fixture()
    a[1]['t'][0] = b[1]['t'][0] = value
    b[1]['t'][0] = shifted(value, 1)
    _roundoff_only(a, b)
    with pytest.raises(AssertionError):
        _exact(a, b)
    b[1]['t'][0] = shifted(value, 2)
    with pytest.raises(AssertionError, match='t'):
        _roundoff_only(a, b)


@pytest.mark.parametrize('value', [-0., np.nextafter(0., 1.), np.nan, np.inf, -np.inf])
def test_tilted_time_rejects_zero_or_nonfinite_changes(value):
    a, b = fixture(), fixture()
    a[1]['t'][0] = 0.
    b[1]['t'][0] = value
    with pytest.raises(AssertionError):
        _roundoff_only(a, b)


@pytest.mark.parametrize('kind', ['energy', 'fate', 'clock', 'outer_clock', 'schema',
                                  'shape', 'dtype', 'outer_schema'])
def test_tilted_time_budget_preserves_exact_contracts(kind):
    a, b = fixture(), fixture()
    if kind == 'energy':
        b[1]['T'][0] = shifted(b[1]['T'][0], 1)
    elif kind == 'fate':
        b[1]['fate'][0] += 1
    elif kind == 'clock':
        b[1]['slab']['it'] += 1
    elif kind == 'outer_clock':
        b = (np.int32(3), b[1])
    elif kind == 'schema':
        b[1]['extra'] = np.zeros(1)
    elif kind == 'shape':
        b[1]['t'] = b[1]['t'].reshape(1, 1)
    elif kind == 'outer_schema':
        b = (*b, np.int32(0))
    else:
        b[1]['t'] = b[1]['t'].astype(np.float32)
    with pytest.raises(AssertionError):
        _roundoff_only(a, b)
