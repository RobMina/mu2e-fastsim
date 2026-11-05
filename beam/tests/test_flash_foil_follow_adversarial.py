"""Independent adversarial host review; no source or reference-test mutations."""
from dataclasses import replace

import numpy as np
import pytest

from beamfastsim import flash_foil_follow as F
from beamfastsim import flash_foil_helix as H
from beamfastsim import flash_foil_receipt as R


def _binding(dtype=np.float64, scale=1., rin=1., tolerance=None):
    return R.bind_inputs(
        x=np.zeros((2, 3), dtype),
        u=np.tile(np.array([1., 0., 0.], dtype), (2, 1)),
        p=H.K_FIELD*scale, q=1, B=[0., 0., 1.],
        rin=rin*scale, rout=3.*scale, zlo=-2.*scale, zhi=2.*scale,
        s_max=7.*scale, foil_id=131, tolerance=tolerance)


def _receipts(binding):
    # Use public emitter; expected binding remains independently retained.
    _, rs = R.first_transition_with_receipts(
        **binding.inputs, foil_id=binding.foil_id,
        max_intervals=binding.max_intervals)
    assert all(r is not None for r in rs)
    return rs


def test_heterogeneous_cap_list_must_not_erase_unrepresentable_integer():
    b = _binding(scale=2.**54)
    rs = _receipts(b)
    # Python int is not representable in float64. Its neighbor is representable.
    cap = [2**56+1, float(2**56)]
    out = F.continue_same_trajectory(rs, expected=b, absolute_s_max=cap)
    assert out.status[0] == F.INVALID_CAP, out.status
    assert not out.accepted[0] and out.unresolved[0]
    assert out.status[1] == F.CLEAR_TO_CAP


def test_heterogeneous_cap_list_must_not_erase_boolean_type():
    b = _binding(rin=.1)
    rs = _receipts(b)
    out = F.continue_same_trajectory(rs, expected=b, absolute_s_max=[True, 6.5])
    assert out.status[0] == F.INVALID_CAP, out.status
    assert not out.accepted[0] and out.unresolved[0]
    assert out.status[1] == F.NEXT_HIT


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('tolerance', [None, .01])
def test_frontier_next_float_and_cap_root_do_not_hide_return(dtype, tolerance):
    b = _binding(dtype=dtype, tolerance=tolerance)
    rs = _receipts(b)
    frontier = np.array([r.evidence['isolation'][1] for r in rs], dtype=dtype)
    for cap in (frontier, np.nextafter(frontier, dtype(np.inf))):
        out = F.continue_same_trajectory(rs, expected=b, absolute_s_max=cap)
        assert out.clear_to_cap.all()
    # Actual orbit radius includes represented p and evaluator K_FIELD.
    p = dtype(b.inputs['p'])
    k = dtype(H.K_FIELD)/p
    radius = 1./float(k)
    second = 2.*np.pi*radius-2.*radius*np.arcsin(.5/radius)
    at_root = F.continue_same_trajectory(
        rs, expected=b, absolute_s_max=dtype(second))
    assert not at_root.accepted.any(), at_root.status
    next_hit = F.continue_same_trajectory(
        rs, expected=b, absolute_s_max=dtype(second+.1))
    assert next_hit.next_hit.all(), next_hit.status
    for evidence in next_hit.next_evidence:
        assert not evidence['forward_material']
    np.testing.assert_array_equal(
        next_hit.relative_isolation[:, 0],
        next_hit.next_isolation[:, 0]-frontier)


def test_mixed_tampered_receipt_lane_is_not_clear_and_does_not_poison_neighbor():
    b = _binding()
    rs = _receipts(b)
    bad = replace(rs[0], transport_restart_valid=True)
    out = F.continue_same_trajectory((bad, rs[1]), expected=b, absolute_s_max=6.5)
    assert out.status[0] == F.INVALID_RECEIPT and not out.accepted[0]
    assert out.unresolved[0] and not out.valid[0] and np.isinf(out.distance[0])
    assert out.status[1] == F.NEXT_HIT and out.next_hit[1]
