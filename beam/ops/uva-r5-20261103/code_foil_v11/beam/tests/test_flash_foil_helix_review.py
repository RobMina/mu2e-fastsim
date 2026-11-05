"""Independent review coverage. Synthetic geometry only; no transport integration.

The realistic plane oracle has an analytic derivative bound over the entire
cap, and radial clearance follows from the unit-speed displacement bound. This
is stronger than a sampled oracle, but not outward-rounded interval arithmetic.
"""
import itertools
import numpy as np
import pytest
from scipy.optimize import brentq
from beamfastsim import flash_foil_helix as nav
from beamfastsim import flash_foil_boundary as ray


def orbit(x, u, p, q, B, s):
    B = np.asarray(B, float)
    bh = B / np.linalg.norm(B)
    par = np.dot(u, bh)*bh
    perp = u-par
    k = nav.K_FIELD*q*np.linalg.norm(B)/p
    t = k*s
    # Separate integrated Rodrigues representation, stable in the small angle.
    return (x+s*par + s*np.sinc(t/np.pi)*perp
            + s*(t/2)*np.sinc(t/(2*np.pi))**2*np.cross(perp,bh))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_realistic_thin_annulus_monotonic_oracle(dtype):
    cases = list(itertools.product([5000.,5500.,6000.], [.05,.1,.2],
                                  [.05,.15,.5], [-1,1], [False,True]))
    B = np.array([.13,.07,1.], dtype=dtype)
    u = np.array([.2,-.1,np.sqrt(.95)], dtype=dtype)
    x = np.array([[-3904.+200., 0., z+(th/2 if ex else -.02)]
                  for z,th,p,q,ex in cases], dtype=dtype)
    zl = np.array([c[0] for c in cases], dtype=dtype)
    zh = np.array([c[0]+c[1] for c in cases], dtype=dtype)
    p = np.array([c[2] for c in cases], dtype=dtype)
    q = np.array([c[3] for c in cases], dtype=dtype)
    ex = np.array([c[4] for c in cases])
    cap = .3
    # Work with cast inputs, not unrepresentable intended foil coordinates.
    uf, bf = u.astype(float), B.astype(float)
    bh = bf/np.linalg.norm(bf)
    A = nav.K_FIELD*np.linalg.norm(bf)/p*np.linalg.norm(uf-np.dot(uf,bh)*bh)
    assert np.all(uf[2]-A*cap > .5)  # z' strictly positive on whole cap
    # r(0)=200 and speed ~1 => no rim, hole, or corner can be encountered.
    assert np.linalg.norm(uf)*cap < 1
    roots = np.array([brentq(lambda s: orbit(xx,uf,pp,qq,bf,s)[2]-target,
                            0,cap,xtol=1.e-13)
                      for xx,pp,qq,target in zip(x.astype(float),p,q,np.where(ex,zh,zl))])
    r = nav.first_transition(x,u,p,q,B,150.,250.,zl,zh,axis_x=-3904.,
                             s_max=cap,exiting=ex)
    # Every lane has exactly one *first* transition at the oracle root. Refusal
    # is allowed; a certified miss or wrong accepted surface is not.
    assert np.all(r['valid'] | r['unresolved'])
    chosen = r['valid']
    np.testing.assert_array_equal(r['kind'][chosen],np.where(ex,nav.ZHI,nav.ZLO)[chosen])
    eps = np.finfo(dtype).eps
    guard = 128*eps*(6001+cap)*(1+np.max(nav.K_FIELD*np.linalg.norm(bf)/p)*cap)
    assert np.all(np.abs(r['distance'][chosen]-roots[chosen]) <= guard)
    if dtype == np.float64:
        assert np.all(chosen)
    print(f'realistic {np.dtype(dtype)}: {np.sum(chosen)}/{len(cases)} hits, '
          f'{np.sum(r["unresolved"])} refusals; guard <= {guard:.6g} mm')


def test_surface_adjacency_requires_backward_not_post_transition_ownership():
    # An exact entry at s=0 is owned by vacuum even if a transport caller has
    # already switched its region state to material. No hidden nudge is valid.
    x = np.array([-3704.,0.,5500.])
    u = np.array([.2,-.1,np.sqrt(.95)])
    kw = dict(p=.05,q=-1,B=[.13,.07,1.],rin=150.,rout=250.,
              zlo=5500.,zhi=5500.1,axis_x=-3904.,s_max=.2)
    a = nav.first_transition(x,u,exiting=False,**kw)
    assert a['valid'] and a['distance']==0 and a['kind']==nav.ZLO
    b = nav.first_transition(x,u,exiting=True,**kw)
    assert b['unresolved'] and not b['valid']
    # Test post-transition material ownership at a deliberately interior point,
    # not by relabeling the same surface. Propagate both position and direction.
    delta=.01
    xx=orbit(x,u,kw['p'],kw['q'],kw['B'],delta)
    bh=np.array(kw['B'])/np.linalg.norm(kw['B'])
    k=nav.K_FIELD*kw['q']*np.linalg.norm(kw['B'])/kw['p']
    par=np.dot(u,bh)*bh; perp=u-par
    uu=par+np.cos(k*delta)*perp+np.sin(k*delta)*np.cross(perp,bh)
    c=nav.first_transition(xx,uu,exiting=True,**kw)
    assert c['valid'] and c['kind']==nav.ZHI and c['distance']>0


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
def test_mixed_zero_force_jit_guard_and_core_convention(dtype):
    jax=pytest.importorskip('jax')
    from mu2efastsim.eslab_jax import _helix
    x=np.array([[0.,0.,0.]]*4,dtype=dtype)
    u=np.array([1.,0.,0.],dtype=dtype)
    p=np.array([.05,.1,.3,.5],dtype=dtype)
    q=np.array([0,1,-1,1],dtype=dtype)
    B=np.array([[.2,.1,.9],[0,0,0],[.13,.07,1],[.13,.07,1]],dtype=dtype)
    kw=dict(p=p,q=q,B=B,rin=.02,rout=3.,zlo=-2.,zhi=2.,s_max=.2)
    a=nav.first_transition(x,u,**kw)
    b=jax.jit(nav.first_transition_jax,static_argnames=('max_intervals',))(x,u,**kw)
    for key in ('valid','unresolved','kind'):
        np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_allclose(a['distance'],b['distance'],atol=1.e-5 if dtype==np.float32 else 1.e-12,rtol=0)
    r=ray.first_transition(x[:2],u,.02,3.,-2.,2.,s_max=.2)
    for key in r:
        np.testing.assert_array_equal(a[key][:2],r[key])
    s=np.array([.03]*4,dtype=dtype)
    xx,_=_helix(x,np.broadcast_to(u,x.shape),p,q,B,s)
    expected=np.stack([x[i]+s[i]*u if i<2 else orbit(x[i],u,p[i],q[i],B[i],s[i]) for i in range(4)])
    np.testing.assert_allclose(xx,expected,atol=2.e-7 if dtype==np.float32 else 1.e-13,rtol=0)
    badkw=dict(kw,p=np.array([0.,.1,.3,.5],dtype=dtype))
    bad=jax.jit(nav.first_transition_jax,static_argnames=('max_intervals',))(x,u,**badkw)
    assert bad['unresolved'][0] and not bad['valid'][0]


def test_axis_crossing_rmin_fallback_cannot_miss_thin_shell():
    # This interval includes r=0, invalidating a 1/r curvature bound. The
    # Lipschitz fallback must split rather than certify the whole orbit clear.
    r=nav.first_transition(np.array([0.,0.,0.]),np.array([1.,0.,0.]),
                           nav.K_FIELD,1,[0.,0.,1.],.05,.1,-1.,1.,s_max=2*np.pi)
    assert r['valid'] and r['kind']==nav.RIN
    np.testing.assert_allclose(r['distance'],2*np.arcsin(.025),atol=1.e-11,rtol=0)


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
def test_realistic_clear_lanes_never_false_hit(dtype):
    # A distance > cap from one violated constraint proves vacuum over the
    # entire path, independent of curvature/sign. Likewise margin > cap from
    # every boundary proves a material path cannot exit.
    x=np.array([[-3704.,0.,5499.],[-3904.,0.,5500.1],
                [-3604.,0.,5500.1],[-3704.,0.,5500.1]],dtype=dtype)
    u=np.array([.2,-.1,np.sqrt(.95)],dtype=dtype)
    kw=dict(p=np.array([.05,.1,.3,.5],dtype=dtype),q=[-1,1,-1,1],
            B=np.array([.13,.07,1.],dtype=dtype),rin=150.,rout=250.,
            zlo=5500.,zhi=5500.2,axis_x=-3904.,s_max=.02,
            exiting=np.array([False,False,False,True]))
    a=nav.first_transition(x,u,**kw)
    assert not np.any(a['valid'])
    if dtype==np.float64:
        assert not np.any(a['unresolved'])
    jax=pytest.importorskip('jax')
    b=jax.jit(nav.first_transition_jax,static_argnames=('max_intervals',))(x,u,**kw)
    assert not np.any(b['valid'])
    for key in a:
        np.testing.assert_array_equal(a[key],b[key])
