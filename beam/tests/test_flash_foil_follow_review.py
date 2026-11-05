"""Independent host continuation controls, synthetic geometry only."""
from dataclasses import replace

import numpy as np
import pytest

from beamfastsim import flash_foil_helix as H
from beamfastsim import flash_foil_receipt as R
from beamfastsim import flash_foil_follow as F


def _circle(dtype=np.float64, **changes):
    args = dict(x=np.array([0., 0., 0.], dtype), u=np.array([1., 0., 0.], dtype),
                p=H.K_FIELD, q=1, B=[0., 0., 1.], rin=1., rout=3.,
                zlo=-2., zhi=2., s_max=7., foil_id=17, frame_binding=np.eye(4))
    args.update(changes)
    return args


def _query(args, cap):
    binding = R.bind_inputs(**args)
    _, receipts = R.first_transition_with_receipts(**args)
    return F.continue_same_trajectory(receipts, expected=binding, absolute_s_max=cap), receipts, binding


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_broadcast_hole_return_independent_circle_oracle(dtype):
    radii = np.array([[.8, 1., 1.5]], dtype)
    charge = np.array([[-1], [1]], np.int32)
    p = (radii*H.K_FIELD).astype(dtype)
    actual_radii = p.astype(float)/H.K_FIELD
    first = 2*actual_radii*np.arcsin(.5/actual_radii)
    second = 2*np.pi*actual_radii-first
    args = _circle(dtype, x=np.zeros((2, 1, 3), dtype), p=p, q=charge, s_max=11., rout=4.)
    result, receipts, binding = _query(args, dtype(10.5))
    assert all(r is not None for r in receipts)
    assert result.next_hit.all(), result.status
    np.testing.assert_array_equal(result.kind, H.RIN)
    np.testing.assert_allclose(result.distance, np.broadcast_to(second, (2, 3)),
                               atol=8.e-5 if dtype == np.float32 else 1.e-11, rtol=0)
    old = np.stack([r.evidence['isolation'] for r in receipts]).reshape((2, 3, 2))
    for axis in (0, 1):
        bound = result.next_isolation[..., axis]
        assert np.all(bound <= second) if axis == 0 else np.all(bound >= second)
    expected_relative = np.stack((result.next_isolation[..., 0]-old[..., 1],
                                  result.next_isolation[..., 1]-old[..., 0]), axis=-1)
    np.testing.assert_array_equal(result.relative_isolation, expected_relative)
    assert result.transport_restart_valid is False
    again = F.continue_same_trajectory(receipts, expected=binding, absolute_s_max=dtype(10.5))
    assert result.distance.tobytes() == again.distance.tobytes()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('charge', [-1, 1])
def test_tilted_field_plane_return_uses_original_absolute_phase(dtype, charge):
    radius = 1.25
    p = dtype(H.K_FIELD*radius)
    radius = float(p)/H.K_FIELD
    args = _circle(dtype, x=np.array([2., 0., 0.], dtype), u=np.array([0., 0., 1.], dtype),
                   p=p, q=charge, B=[0., 1., 0.], rin=0., rout=20., zlo=-.7, zhi=.5,
                   exiting=True, s_max=9.)
    first = radius*np.arcsin(.5/radius)
    second = radius*np.pi-first
    result, receipts, _ = _query(args, dtype(8.5))
    assert receipts[0] is not None
    assert result.next_hit and result.kind == H.ZHI, result.status
    np.testing.assert_allclose(result.distance, second, atol=1.e-4 if dtype == np.float32 else 1.e-11, rtol=0)
    assert result.distance > second-first
    assert result.next_isolation[0] <= second <= result.next_isolation[1]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_requested_cap_cannot_hide_original_conditioning(dtype):
    args = _circle(dtype)
    _, receipts, binding = _query(args, dtype(6.5))
    short = F.continue_same_trajectory(receipts, expected=binding, absolute_s_max=dtype(3.))
    assert short.clear_to_cap and not short.next_hit
    full = F.continue_same_trajectory(receipts, expected=binding, absolute_s_max=dtype(6.5))
    assert full.next_hit
    assert full.next_evidence[0]['guard'].tobytes() == receipts[0].evidence['guard'].tobytes()
    assert full.next_evidence[0]['resolution'].tobytes() == receipts[0].evidence['resolution'].tobytes()
    assert full.next_evidence[0]['isolation'][0] >= receipts[0].evidence['isolation'][1]


@pytest.mark.parametrize('cap', [np.uint64(2**53+1), np.int64(2**53+1), True, '6.5', complex(6.5), np.nan, np.inf])
def test_nonrepresentable_or_invalid_cap_is_never_clear(cap):
    args = _circle()
    _, rs = R.first_transition_with_receipts(**args)
    out = F.continue_same_trajectory(rs, expected=R.bind_inputs(**args), absolute_s_max=cap)
    assert out.status == F.INVALID_CAP and not out.accepted and out.unresolved
    assert not out.valid and np.isinf(out.distance)


def test_float32_silent_cap_rounding_refuses():
    args = _circle(np.float32)
    _, rs = R.first_transition_with_receipts(**args)
    assert rs[0] is not None
    for cap in (1.5+2.**-30, np.float64(np.nextafter(np.float64(6.5), np.inf))):
        out = F.continue_same_trajectory(rs, expected=R.bind_inputs(**args), absolute_s_max=cap)
        assert out.status == F.INVALID_CAP and not out.accepted and out.unresolved
    represented = F.continue_same_trajectory(rs, expected=R.bind_inputs(**args), absolute_s_max=np.float32(6.5))
    assert represented.next_hit


def test_next_evidence_cannot_be_reused_as_initial_receipt():
    first, rs, binding = _query(_circle(), 6.5)
    assert first.next_hit
    rejected = F.continue_same_trajectory(first.next_evidence, expected=binding, absolute_s_max=7.)
    assert rejected.status == F.INVALID_RECEIPT
    forged = replace(rs[0], evidence=first.next_evidence[0])
    rejected = F.continue_same_trajectory((forged,), expected=binding, absolute_s_max=7.)
    assert rejected.status == F.INVALID_RECEIPT and not rejected.accepted


@pytest.mark.parametrize('field', ['p', 'q', 'B', 'x', 'u', 'foil_id', 'frame_binding', 's_max', 'max_intervals'])
def test_changed_external_binding_never_authorizes_ownership_skip(field):
    args = _circle()
    _, rs = R.first_transition_with_receipts(**args)
    changes = dict(p=.31, q=-1, B=[0., .01, 1.], x=[.01, 0., 0.], u=[0., 1., 0.],
                   foil_id=18, frame_binding=np.diag([-1., -1., 1., 1.]), s_max=8., max_intervals=513)
    expected = R.bind_inputs(**dict(args, **{field: changes[field]}))
    out = F.continue_same_trajectory(rs, expected=expected, absolute_s_max=6.5)
    assert out.status == F.INVALID_RECEIPT and not out.accepted and out.unresolved
    assert not out.valid and np.isinf(out.distance)


def test_immutable_return_and_inputs_no_particle_projection():
    args = _circle()
    before = {k: np.asarray(v).tobytes() for k, v in args.items()}
    result, rs, binding = _query(args, 6.5)
    for value in (*result.result.values(), result.status, result.accepted, result.next_hit,
                  result.clear_to_cap, result.unresolved, result.next_isolation, result.relative_isolation,
                  *result.next_evidence[0].values()):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert {k: np.asarray(v).tobytes() for k, v in args.items()} == before
    assert rs[0].binding.inputs['x'].tobytes() == binding.inputs['x'].tobytes()
    assert np.all(rs[0].binding.inputs['x'] == 0)
    assert not any(k in result.result for k in ('x', 'u', 'position', 'direction'))


def test_heterogeneous_nested_caps_keep_value_and_type_per_lane():
    args = _circle(x=np.zeros((2, 2, 3)), rin=.1)
    _, receipts = R.first_transition_with_receipts(**args)
    binding = R.bind_inputs(**args)
    caps = [[True, 1.], [2**10000, 6.5]]
    out = F.continue_same_trajectory(receipts, expected=binding, absolute_s_max=caps)
    np.testing.assert_array_equal(out.status, [[F.INVALID_CAP, F.CLEAR_TO_CAP],
                                              [F.INVALID_CAP, F.NEXT_HIT]])
    assert np.all(out.unresolved[~out.accepted])


def test_explicit_object_array_not_a_numeric_cap_contract():
    args = _circle(x=np.zeros((2, 3)), rin=.1)
    _, receipts = R.first_transition_with_receipts(**args)
    out = F.continue_same_trajectory(receipts, expected=R.bind_inputs(**args),
                                     absolute_s_max=np.array([1., 6.5], dtype=object))
    assert np.all(out.status == F.INVALID_CAP)
    assert not out.accepted.any()
