"""Read-only adapter review: synthetic inputs and independent pre-receipt snapshots.

No geometry data, transport integration, projection, or numerical closure claim.
The snapshot directory is local and optional outside this review checkout.
"""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import beamfastsim
from beamfastsim import flash_foil_boundary as R
from beamfastsim import flash_foil_helix as H
from beamfastsim import flash_foil_navigation as N
from beamfastsim import flash_foil_frames as F

@pytest.fixture(scope='module')
def snapshots():
    source = os.environ.get('FLASH_FOIL_BASELINE_DIR')
    if not source:
        pytest.skip('Provide independent pre-after_surface snapshots for byte regression')
    snapshot = Path(source)
    if not all((snapshot / name).is_file() for name in ('flash_foil_boundary.py', 'flash_foil_helix.py')):
        pytest.fail('Requested independent foil baseline is incomplete')

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, snapshot / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    old_ray = load('beamfastsim._adapters_review_old_ray', 'flash_foil_boundary.py')
    # The old helix must bind the OLD ray, not the currently installed one.
    name = 'beamfastsim.flash_foil_boundary'
    saved = sys.modules[name]
    saved_attr = beamfastsim.flash_foil_boundary
    try:
        sys.modules[name] = old_ray
        beamfastsim.flash_foil_boundary = old_ray
        old_helix = load('beamfastsim._adapters_review_old_helix', 'flash_foil_helix.py')
    finally:
        sys.modules[name] = saved
        beamfastsim.flash_foil_boundary = saved_attr
    assert old_helix.ray is old_ray
    return old_ray, old_helix


def same(a, b, *, atol=0):
    for key in a:
        if key == 'distance' and atol:
            np.testing.assert_allclose(a[key], b[key], atol=atol, rtol=0)
        else:
            np.testing.assert_array_equal(a[key], b[key])


def refused(r):
    assert np.all(r['unresolved']) and not np.any(r['valid'])
    assert np.all(np.isinf(r['distance'])) and np.all(r['kind'] == 0)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('curved', [False, True])
def test_snapshot_default_none_and_mixed_none(snapshots, dtype, curved):
    module = H if curved else R
    old = snapshots[int(curved)]
    # Exact starts, interior, hole, certified misses, corner, tangent, and near
    # surface. Random lanes add non-start geometries/ownerships independently.
    rng = np.random.default_rng(63014)
    fixed = [[3,0,0], [2,0,0], [0,0,0], [10,0,0], [3,0,-2],
             [3,0,0], [np.nextafter(dtype(3), dtype(0)),0,0]]
    x = np.asarray(fixed + rng.uniform(-4,4,(24,3)).tolist(), dtype)
    u = rng.normal(size=x.shape).astype(dtype)
    u /= np.linalg.norm(u, axis=-1)[:,None]
    u[:7] = np.array([[-1,0,0], [1,0,0], [1,0,0], [1,0,0],
                       [-1,0,0], [0,1,0], [-1,0,0]], dtype)
    radius = np.hypot(x[:,0],x[:,1])
    ex = (radius > 1) & (radius < 3) & (abs(x[:,2]) < 2)
    kw = dict(rin=1., rout=3., zlo=-2., zhi=2., s_max=7., exiting=ex)
    if curved:
        q = np.resize(np.array([-1,0,1]),len(x))
        kw.update(p=H.K_FIELD, q=q, B=np.array([.2,.1,1.],dtype))
    reference = old.first_transition(x,u,**kw)
    default = module.first_transition(x,u,**kw)
    same(default, reference)
    same(module.first_transition(x,u,after_surface=np.zeros(len(x),np.int32),**kw),reference)
    tokens = np.zeros(len(x),np.int32)
    tokens[0] = R.ROUT
    mixed_ex = ex.copy(); mixed_ex[0] = True
    mixed = module.first_transition(x,u,**dict(kw,exiting=mixed_ex),after_surface=tokens)
    same({k:v[1:] for k,v in mixed.items()}, {k:v[1:] for k,v in reference.items()})


@pytest.mark.parametrize('module', [R,H])
@pytest.mark.parametrize('gap', [2.**-20, 2.**-30])
def test_no_finite_departure_skip_of_very_short_material(module,gap):
    # Exact rim receipt, followed almost immediately by the hole. Any hidden
    # fixed advance larger than gap could silently erase the material interval.
    kw = dict(rin=3-gap,rout=3.,zlo=-2.,zhi=2.,s_max=.01,
              exiting=True,after_surface=R.ROUT)
    if module is H:
        kw.update(p=H.K_FIELD,q=1,B=[0.,0.,1.])
    r = module.first_transition(np.array([3.,0,0]),np.array([-1.,0,0]),**kw)
    assert r['valid'] and not r['unresolved'] and r['kind'] == R.RIN
    assert 0 < r['distance'] < 2*gap
    np.testing.assert_allclose(r['distance'],gap,rtol=0,atol=2e-12)
    # Unresolved other clearance cannot be waived by a consumed root.
    refused(module.first_transition(np.array([3.,0,0]),np.array([-1.,0,0]),
                                    **dict(kw,tolerance=4*gap)))


@pytest.mark.parametrize('cap', [.1, 2., 4.])
def test_consumed_outward_rim_later_return_is_not_clear(cap):
    # Circle centre (3,-1), radius 1; start (3,0), outward tangent
    # direction +x. It returns to r=3 at s=2*pi-2*atan(3).
    root = 2*np.pi-2*np.arctan(3)
    r = H.first_transition(np.array([3.,0,0]),np.array([1.,0,0]),
                           H.K_FIELD,1,[0.,0.,1.],1.,3.,-2.,2.,
                           s_max=cap,exiting=False,after_surface=R.ROUT)
    if cap < root:
        assert not r['valid'] and not r['unresolved']
    else:
        assert r['valid'] and not r['unresolved'] and r['kind'] == R.ROUT
        np.testing.assert_allclose(r['distance'],root,rtol=0,atol=2e-11)


def test_unknown_candidate_prevents_false_clear_even_without_known_hit():
    r = N.select_first([[np.inf,np.inf],[.1,np.inf],[.1,1e6]],
                       [[0,0],[1,0],[1,2]],
                       [[False,False],[True,False],[True,True]],
                       [[False,True],[False,True],[False,True]])
    refused(r)
    np.testing.assert_array_equal(r['foil_id'],[-1,-1,-1])


def table():
    return F.build_table(SimpleNamespace(axis_x=10.,foil_z=[2.,-2.],
                       foil_ht=[.5,.5],foil_rin=[1.,1.],foil_rout=[3.,3.]))


def test_frame_side_entry_depth_proper_rotation_and_saved_orientation():
    t = table()
    x = np.array([[13.,0,2.125],[11.,0,2.125]])
    u = np.array([[-.6,0,.8],[.6,0,-.8]])
    r = F.entry_frame(t,[0,0],x,u,[.2,.3,1.])
    assert np.all(r['valid']) and np.all(r['boundary'])
    np.testing.assert_array_equal(r['depth'],[.625,.375])
    np.testing.assert_array_equal(r['local_x'],[[0,0,.625],[0,0,.375]])
    np.testing.assert_array_equal(np.linalg.det(r['basis']),[1.,1.])
    F.validate_basis(r['basis'])
    np.testing.assert_array_equal(r['global_x'],x)
    np.testing.assert_allclose(F.to_global(r,r['local_x'],point=True),x,atol=0,rtol=0)
    # Transform a backwards scattered vector in the SAVED frame.
    backwards = -r['local_u']
    np.testing.assert_allclose(F.to_local(r,F.to_global(r,backwards)),backwards)
    with pytest.raises(ValueError):
        F.validate_basis(np.diag([1.,1.,-1.]))


def test_frame_unsigned_out_of_range_and_sentinel_never_alias():
    jax = pytest.importorskip('jax')
    t = table()
    for ids in [np.array([0,1,2,np.iinfo(np.uint64).max],np.uint64),
                np.array([0,1,-1,-2,np.iinfo(np.int64).max],np.int64)]:
        expected = F.gather(t,ids)
        same(jax.jit(lambda i:F.gather_jax(t,i))(ids),expected)
        n = len(ids)
        host = F.entry_frame(t,ids,np.tile([12.,0.,2.],(n,1)),[0.,0.,1.],[0.,0.,1.])
        actual = jax.jit(lambda i:F.entry_frame_jax(t,i,np.tile([12.,0.,2.],(n,1)),
                                                   np.array([0.,0.,1.]),np.array([0.,0.,1.])))(ids)
        for key in host:
            np.testing.assert_allclose(actual[key],host[key],equal_nan=True)
        invalid = ~expected['valid']
        assert np.all(expected['foil_id'][invalid] == -1)
        assert np.all(np.isnan(expected['foil_z'][invalid]))
        assert np.all(expected['material_type'][invalid] == -1)
        assert not np.any(host['valid'][invalid])


@pytest.mark.parametrize('id_dtype',[np.int64,np.uint32,np.uint64])
def test_navigation_valid_unsigned_identity_matches_numpy_under_jit(id_dtype):
    jax = pytest.importorskip('jax')
    kw = dict(p=1.,q=0,B=np.array([0.,0.,0.]),foil_z=np.array([0.,2.]),
              foil_ht=np.array([.5,.5]),foil_rin=np.array([0.,0.]),
              foil_rout=np.array([4.,4.]),s_max=4.,current_foil=id_dtype(0),
              consumed_foil=id_dtype(0),after_surface=np.uint64(R.ZLO))
    x=np.array([2.,0.,-.5]); u=np.array([0.,0.,1.])
    reference=N.first_foil(x,u,**kw)
    assert reference['valid'] and reference['foil_id'] == 0
    actual=jax.jit(N.first_foil_jax)(x,u,**kw)
    same(actual,reference)


def test_navigation_large_unsigned_ids_invalid_not_wrapped():
    jax=pytest.importorskip('jax')
    kw=dict(p=1.,q=0,B=np.array([0.,0.,0.]),foil_z=np.array([0.,2.]),
            foil_ht=np.array([.5,.5]),foil_rin=np.array([0.,0.]),
            foil_rout=np.array([4.,4.]),s_max=4.,current_foil=np.uint64(2**32),
            consumed_foil=np.uint64(0),after_surface=np.uint64(R.ZLO))
    x=np.array([2.,0.,-.5]); u=np.array([0.,0.,1.])
    with pytest.raises(ValueError):
        N.first_foil(x,u,**kw)
    refused(jax.jit(N.first_foil_jax)(x,u,**kw))
