"""Synthetic first-foil competition preserves unique identity and refusal status."""
import numpy as np
import pytest

from beamfastsim import flash_foil_navigation as N


def _trace(x, u, **kwargs):
    args = dict(p=1., q=-1, B=[0., 0., 0.], foil_z=[0., 2.], foil_ht=[.1, .1],
                foil_rin=[1., 2.], foil_rout=[3., 4.], s_max=10.)
    args.update(kwargs)
    return N.first_foil(np.asarray(x, float), np.asarray(u, float), **args)


def test_equal_thickness_is_not_foil_identity():
    r = _trace([[3.5, 0, -1.], [1.5, 0, -1.]], [0., 0, 1.])
    np.testing.assert_array_equal(r["foil_id"], [1, 0])
    np.testing.assert_allclose(r["distance"], [2.9, .9])
    assert np.all(r["valid"]) and not np.any(r["unresolved"])


@pytest.mark.parametrize("x,u,current,foil_id,kind,expected", [
    ([5., 0., 0.], [-1., 0., 0.], -1, 0, 4, 2.),
    ([0., 0., 2.], [1., 0., 0.], -1, 1, 3, 2.),
    ([2., 0., 0.], [1., 0., 0.], 0, 0, 4, 1.),
    ([3., 0., 2.], [-1., 0., 0.], 1, 1, 3, 1.),
    ([2., 0., 0.], [0., 0., -1.], 0, 0, 1, .1),
])
def test_faces_rims_and_holes_compete(x, u, current, foil_id, kind, expected):
    r = _trace(x, u, current_foil=current)
    assert r["valid"] and not r["unresolved"]
    assert r["foil_id"] == foil_id and r["kind"] == kind
    np.testing.assert_allclose(r["distance"], expected)


def test_reordered_lanes_and_axis_translation_preserve_foil_ids():
    x = np.array([[3.5, 0., -1.], [1.5, 0., -1.], [20., 0., -1.]])
    x[:, 0] -= 3904
    r = _trace(x, [0., 0., 1.], axis_x=-3904.)
    order = np.array([2, 0, 1, 0])
    shuffled = _trace(x[order], [0., 0., 1.], axis_x=-3904.)
    for name in r:
        np.testing.assert_array_equal(shuffled[name], r[name][order])
    np.testing.assert_array_equal(r["foil_id"], [1, 0, -1])


def test_unknown_competitor_is_not_hidden_by_known_hit():
    r = N.select_first([[1., np.inf], [1., 2.]], [[1, 0], [1, 2]],
                       [[True, False], [True, True]], [[False, True], [False, False]])
    np.testing.assert_array_equal(r["unresolved"], [True, False])
    np.testing.assert_array_equal(r["valid"], [False, True])
    np.testing.assert_array_equal(r["foil_id"], [-1, 0])
    assert np.isinf(r["distance"][0])


@pytest.mark.parametrize("distances", [[1., 1.], [1., 1.+1.e-15]])
def test_ties_refuse_instead_of_picking_first_array_index(distances):
    r = N.select_first(np.array(distances), [1, 2], [True, True], [False, False])
    assert r["unresolved"] and not r["valid"] and r["foil_id"] == -1


@pytest.mark.parametrize("distance,kind", [(np.nan, 0), (-np.inf, 0), (1., 0), (np.inf, 2)])
def test_malformed_unknown_candidate_refuses_instead_of_clear(distance, kind):
    r = N.select_first([distance], [kind], [False], [False])
    assert r["unresolved"] and not r["valid"] and r["foil_id"] == -1


def test_per_lane_axis_and_tolerance_do_not_broadcast_as_foil_axis():
    x = np.array([[11.5, 0., -1.], [21.5, 0., -1.], [31.5, 0., -1.]])
    r = _trace(x, [0., 0., 1.], axis_x=np.array([10., 20., 30.]),
               tolerance=np.array([1.e-10, 2.e-10, 3.e-10]))
    np.testing.assert_array_equal(r["foil_id"], [0, 0, 0])
    np.testing.assert_allclose(r["distance"], .9)
    assert np.all(r["valid"])


def test_cap_and_nohit_keep_negative_sentinel_without_alias():
    r = _trace([3.5, 0., -1.], [0., 0., 1.], s_max=2.8)
    assert not r["valid"] and not r["unresolved"] and r["foil_id"] == -1
    hit = _trace([3.5, 0., -1.], [0., 0., 1.], s_max=2.9)
    assert hit["valid"] and hit["foil_id"] == 1


@pytest.mark.parametrize("kwargs", [dict(current_foil=-2), dict(current_foil=2),
    dict(current_foil=.5), dict(consumed_foil=0), dict(after_surface=1),
    dict(consumed_foil=0, after_surface=5), dict(consumed_foil=0, after_surface=.5),
    dict(foil_z=[2., 0.]), dict(foil_z=[0., .1]), dict(foil_ht=[.1]),
    dict(foil_ht=[-.1, .1]), dict(foil_rout=[3., np.nan])])
def test_invalid_host_identity_or_geometry_fails(kwargs):
    with pytest.raises(ValueError):
        _trace([2., 0., -1.], [0., 0., 1.], **kwargs)


def test_mismatched_current_region_is_unresolved_not_reclassified():
    r = _trace([2., 0., 0.], [1., 0., 0.])
    assert r["unresolved"] and not r["valid"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_consumed_root_entry_exit_and_later_foil(dtype):
    kw = dict(p=1., q=-1, B=[0., 0., 0.], foil_z=[0., 2.], foil_ht=[.5, .5],
              foil_rin=[0., 0.], foil_rout=[4., 4.], s_max=10.)
    entry = N.first_foil(np.array([2., 0., -.5], dtype), np.array([0., 0., 1.], dtype),
                         current_foil=0, consumed_foil=0, after_surface=1, **kw)
    assert entry["valid"] and entry["foil_id"] == 0 and entry["kind"] == 2
    np.testing.assert_allclose(entry["distance"], 1.)
    exited = N.first_foil(np.array([2., 0., .5], dtype), np.array([0., 0., 1.], dtype),
                          current_foil=-1, consumed_foil=0, after_surface=2, **kw)
    assert exited["valid"] and exited["foil_id"] == 1 and exited["kind"] == 1
    np.testing.assert_allclose(exited["distance"], 1.)


def test_jax_optional_mixed_forward_and_backward_lanes():
    jax = pytest.importorskip("jax")
    x = np.array([[2., 0., -.5], [2., 0., -1.], [2., 0., .5]], np.float64)
    u = np.array([0., 0., 1.])
    kw = dict(p=1., q=-1, B=[0., 0., 0.], foil_z=np.array([0., 2.]), foil_ht=np.array([.5, .5]),
              foil_rin=np.array([0., 0.]), foil_rout=np.array([4., 4.]), s_max=10.,
              current_foil=np.array([0, -1, -1]), consumed_foil=np.array([0, -1, 0]),
              after_surface=np.array([1, 0, 2]))
    reference = N.first_foil(x, u, **kw)
    fn = jax.jit(N.first_foil_jax, static_argnames=("max_intervals",))
    result = fn(x, u, **kw)
    for name in reference:
        np.testing.assert_array_equal(result[name], reference[name])
    invalid = fn(x, u, **dict(kw, current_foil=np.array([0, 9, -1])))
    assert invalid["unresolved"][1] and invalid["foil_id"][1] == -1
