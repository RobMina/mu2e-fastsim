"""Unchanged anchored-orbit continuation; no transport or receipt chaining."""
from dataclasses import FrozenInstanceError, replace
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from beamfastsim import flash_foil_follow as follow
from beamfastsim import flash_foil_helix as helix
from beamfastsim import flash_foil_receipt as receipt


def arguments(dtype=np.float64, **changes):
    kw = dict(x=np.array([0., 0., 0.], dtype), u=np.array([1., 0., 0.], dtype),
              p=helix.K_FIELD, q=1, B=[0., 0., 1.], rin=1., rout=3.,
              zlo=-2., zhi=2., s_max=2*np.pi, foil_id=19)
    kw.update(changes)
    return kw


def emit(kw):
    expected = receipt.bind_inputs(**kw)
    ordinary, rs = receipt.first_transition_with_receipts(**kw)
    return expected, ordinary, rs


def same_bytes(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].shape == b[k].shape
        assert a[k].dtype == b[k].dtype
        assert a[k].tobytes() == b[k].tobytes()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('charge', [-1, 1])
@pytest.mark.parametrize('surface', ['hole', 'rim', 'face'])
def test_independent_same_surface_return_absolute_and_relative(dtype, charge, surface):
    kw = arguments(dtype, q=charge)
    first, second, kind = np.pi/3, 5*np.pi/3, helix.RIN
    if surface == 'rim':
        kw.update(x=np.array([2., 0., 0.], dtype), u=np.array([charge, 0., 0.], dtype),
                  exiting=True)
        first = np.arcsin(3/np.sqrt(20))+np.arctan(.5)
        second = np.pi-np.arcsin(3/np.sqrt(20))+np.arctan(.5)
        if charge == -1:
            first, second = 2*np.pi-second, 2*np.pi-first
        kind = helix.ROUT
    elif surface == 'face':
        kw.update(x=np.array([2., 0., 0.], dtype), B=[0., 1., 0.], rin=0.,
                  rout=20., exiting=True)
        kw.update(zhi=.5) if charge == 1 else kw.update(zlo=-.5)
        kind = helix.ZHI if charge == 1 else helix.ZLO
    expected, ordinary, rs = emit(kw)
    assert rs[0] is not None
    old = rs[0].evidence
    assert old['isolation'][0] < first < old['isolation'][1]
    cap = np.asarray(kw['s_max'], dtype)
    out = follow.continue_same_trajectory(rs, expected=expected, absolute_s_max=cap)
    assert out.status == follow.NEXT_HIT
    assert out.accepted and out.next_hit and not out.clear_to_cap and not out.unresolved
    assert out.kind == kind and out.valid
    assert out.next_isolation[0] < second < out.next_isolation[1]
    assert out.next_isolation[0] >= old['isolation'][1]
    assert out.next_isolation[0] <= out.distance <= out.next_isolation[1]
    e = out.next_evidence[0]
    assert e['kind'] == kind and e['derivative_margin'] > 0
    assert e['endpoint_clearance'][0]*e['endpoint_clearance'][1] < 0
    assert np.all(e['other_clearance_lower'][e['other_clearance_enabled']] > 0)
    assert e['forward_material'] == (surface != 'hole')
    assert e['guard'].tobytes() == old['guard'].tobytes()
    assert e['resolution'].tobytes() == old['resolution'].tobytes()
    assert out.relative_isolation[0] <= second-first <= out.relative_isolation[1]
    np.testing.assert_array_equal(out.relative_isolation,
        [out.next_isolation[0]-old['isolation'][1],
         out.next_isolation[1]-old['isolation'][0]])
    assert not out.transport_restart_valid
    assert not isinstance(e, receipt.GeometricReceipt)
    same_bytes(ordinary, helix.first_transition(**{k: v for k, v in kw.items()
                                                  if k != 'foil_id'}))
    refused = follow.continue_same_trajectory(out.next_evidence, expected=expected,
                                              absolute_s_max=cap)
    assert refused.status == follow.INVALID_RECEIPT
    assert not refused.accepted and refused.unresolved


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_partial_cap_and_verified_empty_window(dtype):
    expected, _, rs = emit(arguments(dtype))
    for cap in (np.asarray(2., dtype), rs[0].evidence['isolation'][1]):
        out = follow.continue_same_trajectory(rs, expected=expected, absolute_s_max=cap)
        assert out.status == follow.CLEAR_TO_CAP and out.accepted and out.clear_to_cap
        assert not out.valid and not out.next_hit and not out.unresolved
        assert np.isinf(out.distance) and out.kind == helix.NONE
        assert np.isnan(out.next_isolation).all() and out.next_evidence == (None,)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_invalid_caps_no_rounding_no_skipped_isolation(dtype):
    expected, _, rs = emit(arguments(dtype))
    e = rs[0].evidence
    a, b = e['isolation']
    rounded_bound = np.asarray(2*np.pi, dtype)
    invalid = [a+(b-a)/2, np.nextafter(b, dtype(-np.inf)), -1., np.inf, np.nan,
               np.nextafter(rounded_bound, dtype(np.inf)), True, '2', 2+0j, [2., 3.]]
    if dtype == np.float32:
        invalid += [2*np.pi, np.nextafter(float(b), np.inf)]
    else:
        invalid += [2**53+1]
    for cap in invalid:
        out = follow.continue_same_trajectory(rs, expected=expected, absolute_s_max=cap)
        assert out.status == follow.INVALID_CAP, repr(cap)
        assert not out.accepted and not out.clear_to_cap and out.unresolved
        assert not out.valid and np.isinf(out.distance)


def test_tolerance_dominates_no_departure_resolution_relaxation():
    expected, _, rs = emit(arguments(tolerance=.2))
    assert rs[0] is not None
    e = rs[0].evidence
    assert e['guard'] < abs(e['endpoint_clearance'][1]) <= e['resolution']
    for cap in (e['isolation'][1], 2*np.pi):
        out = follow.continue_same_trajectory(rs, expected=expected, absolute_s_max=cap)
        assert out.status == follow.UNRESOLVED
        assert not out.accepted and not out.clear_to_cap


def test_next_exact_endpoint_is_explicit_unsupported_hit_not_clear():
    cap = 5.
    height = cap*(2*np.sin(cap/2)**2/cap)
    kw = arguments(x=np.array([2., 0., 0.]), B=[0., 1., 0.], rin=0., rout=20.,
                   zhi=height, exiting=True, s_max=cap)
    expected, _, rs = emit(kw)
    assert rs[0] is not None
    out = follow.continue_same_trajectory(rs, expected=expected, absolute_s_max=cap)
    assert out.status == follow.UNSUPPORTED_HIT
    assert out.valid and out.distance == cap
    assert not out.accepted and not out.next_hit and not out.clear_to_cap
    assert out.next_evidence == (None,)
    assert np.isnan(out.next_isolation).all()


@pytest.mark.parametrize('budget', [8, 9, 10, 11])
def test_budget_bound_reused_not_widened(budget):
    expected, _, rs = emit(arguments(max_intervals=budget))
    assert rs[0] is not None
    out = follow.continue_same_trajectory(rs, expected=expected, absolute_s_max=2*np.pi)
    assert out.status == follow.UNRESOLVED and out.unresolved and not out.accepted
    changed = receipt.bind_inputs(**arguments(max_intervals=512))
    out = follow.continue_same_trajectory(rs, expected=changed, absolute_s_max=2*np.pi)
    assert out.status == follow.INVALID_RECEIPT


@pytest.mark.parametrize('field', ['isolation', 'refined', 'distance', 'kind', 'guard',
                                  'resolution', 'derivative_mid', 'derivative_margin',
                                  'derivative_variation', 'endpoint_clearance',
                                  'other_clearance_lower', 'other_clearance_enabled',
                                  'forward_material'])
def test_all_evidence_tampering_refused(field):
    expected, _, (r,) = emit(arguments())
    changed = r.evidence[field].copy()
    if changed.dtype.kind == 'b':
        changed.flat[0] = not changed.flat[0]
    elif changed.dtype.kind == 'i':
        changed.flat[0] += 1
    else:
        changed.flat[0] = np.nextafter(changed.flat[0], np.inf)
    bad = replace(r, evidence=dict(r.evidence, **{field: changed}))
    out = follow.continue_same_trajectory((bad,), expected=expected, absolute_s_max=2*np.pi)
    assert out.status == follow.INVALID_RECEIPT and not out.accepted and out.unresolved


@pytest.mark.parametrize('change', [dict(foil_id=20), dict(frame_binding=np.eye(4)),
    dict(x=np.array([.001, 0., 0.])), dict(p=helix.K_FIELD+.01), dict(q=-1),
    dict(B=[.01, 0., 1.]), dict(rin=.99), dict(rout=3.01), dict(zlo=-2.01),
    dict(zhi=2.01), dict(axis_x=.001), dict(s_max=7.), dict(tolerance=1.e-8),
    dict(max_intervals=513), dict(after_surface=helix.RIN)])
def test_external_binding_mutation_refuses(change):
    _, _, rs = emit(arguments())
    expected = receipt.bind_inputs(**arguments(**change))
    out = follow.continue_same_trajectory(rs, expected=expected, absolute_s_max=2*np.pi)
    assert out.status == follow.INVALID_RECEIPT and not out.accepted


def test_malformed_receipts_binding_and_mandatory_external_expected():
    expected, _, (r,) = emit(arguments())
    for bad in (None, {}, (None,), [r], (), (r, r),
                (replace(r, schema='other'),), (replace(r, transport_restart_valid=True),),
                (replace(r, lane_index=(0,)),), (replace(r, evidence={}),),
                (replace(r, binding=None),)):
        out = follow.continue_same_trajectory(bad, expected=expected, absolute_s_max=2*np.pi)
        assert out.status == follow.INVALID_RECEIPT
        assert not out.accepted and out.unresolved and not out.valid
    with pytest.raises(TypeError):
        follow.continue_same_trajectory((r,), absolute_s_max=2*np.pi)
    with pytest.raises(ValueError):
        follow.continue_same_trajectory((r,), expected=None, absolute_s_max=2*np.pi)
    with pytest.raises(ValueError):
        follow.continue_same_trajectory((r,), expected=replace(expected, inputs={}),
                                        absolute_s_max=2*np.pi)
    with pytest.raises(TypeError):
        follow.continue_same_trajectory((r,), expected=expected, absolute_s_max=2*np.pi,
                                        start=2.)


def test_mixed_flat_c_order_lanes_one_ordinary_rerun(monkeypatch):
    kw = arguments(x=np.zeros((2, 3, 3)), q=np.array([[1, 0, -1], [1, 1, 1]]),
                   s_max=np.array([[2*np.pi, 2*np.pi, 2*np.pi], [2*np.pi, .1, 2*np.pi]]))
    expected, _, rs = emit(kw)
    assert len(rs) == 6 and rs[1] is None and rs[4] is None
    supplied = list(rs)
    supplied[3] = replace(rs[3], lane_index=(0, 0))
    cap = np.full((2, 3), 2*np.pi)
    cap[0, 2] = 2.
    cap[1, 1] = .1
    cap[1, 2] = 7.
    calls = []
    original = receipt._emit
    def counted(binding):
        calls.append(binding)
        return original(binding)
    monkeypatch.setattr(receipt, '_emit', counted)
    out = follow.continue_same_trajectory(tuple(supplied), expected=expected,
                                          absolute_s_max=cap)
    assert len(calls) == 1
    np.testing.assert_array_equal(out.status, [[follow.NEXT_HIT, follow.INVALID_RECEIPT,
        follow.CLEAR_TO_CAP], [follow.INVALID_RECEIPT, follow.INVALID_RECEIPT,
        follow.INVALID_CAP]])
    np.testing.assert_array_equal(out.accepted, [[True, False, True], [False, False, False]])
    assert np.all(out.unresolved[~out.accepted])
    assert np.all(~out.valid[~out.accepted])
    swapped = follow.continue_same_trajectory(tuple(reversed(rs)), expected=expected,
                                              absolute_s_max=cap)
    assert np.all(swapped.status == follow.INVALID_RECEIPT)


def test_output_immutable_and_no_backend_import():
    expected, _, rs = emit(arguments())
    out = follow.continue_same_trajectory(rs, expected=expected, absolute_s_max=2*np.pi)
    arrays = [out.status, out.accepted, out.next_hit, out.clear_to_cap, out.unresolved,
              out.next_isolation, out.relative_isolation, *out.result.values(),
              *out.next_evidence[0].values()]
    for a in arrays:
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        out.transport_restart_valid = True
    with pytest.raises(TypeError):
        out.result['distance'] = 1.
    with pytest.raises(TypeError):
        out.next_evidence[0]['distance'] = 1.
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c', "import sys; from beamfastsim import flash_foil_follow; "
                    "assert not any(k == 'jax' or k.startswith('jax.') for k in sys.modules)"],
                   check=True, env=os.environ.copy())


def test_private_window_numpy_only_and_invalid_lanes_refuse():
    kw = {k: v for k, v in arguments().items() if k != 'foil_id'}
    args = [kw[k] for k in ('x', 'u', 'p', 'q', 'B', 'rin', 'rout', 'zlo', 'zhi')]
    args += [0., kw['s_max'], False, None, 512]
    with pytest.raises(ValueError, match='NumPy-only'):
        helix._transition(object(), *args, _search_window=(1., 2., True))
    for window in [(1., 2., False), (3., 2., True), (1., 7., True),
                   (np.nan, 2., True), (1., np.inf, True), (1., 1., True)]:
        out = helix._transition(np, *args, _search_window=window)
        assert out['unresolved'] and not out['valid'] and np.isinf(out['distance'])
    args[3] = 0
    out = helix._transition(np, *args, _search_window=(1., 2., True))
    assert out['unresolved'] and not out['valid']


def _baseline():
    source = os.environ.get('FLASH_FOIL_FOLLOW_BASELINE_SOURCE')
    if not source:
        pytest.skip('provide independent pre-follow helix source for bytes/graph checks')
    path = Path(source)
    if not path.is_file():
        pytest.fail('requested independent pre-follow helix source is absent')
    spec = importlib.util.spec_from_file_location('beamfastsim._pre_follow', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_default_bytes_match_independent_preedit(dtype):
    old = _baseline()
    rng = np.random.default_rng(20261104)
    x = rng.uniform(-4., 4., (64, 3)).astype(dtype)
    u = rng.normal(size=(64, 3)).astype(dtype)
    u /= np.linalg.norm(u, axis=-1)[:, None]
    material = (np.hypot(x[:, 0], x[:, 1]) > 1) & (np.hypot(x[:, 0], x[:, 1]) < 3)
    material &= abs(x[:, 2]) < 2
    kw = arguments(dtype, x=x, u=u, q=rng.choice([-1, 0, 1], 64),
                   p=rng.uniform(.1, 1., 64), B=rng.normal(size=(64, 3)), exiting=material)
    legacy_kw = {k: v for k, v in kw.items() if k != 'foil_id'}
    baseline = old.first_transition(**legacy_kw)
    same_bytes(helix.first_transition(**legacy_kw), baseline)
    expected, ordinary, rs = emit(kw)
    same_bytes(ordinary, baseline)
    follow.continue_same_trajectory(rs, expected=expected,
                                    absolute_s_max=np.asarray(2*np.pi, dtype))
    same_bytes(helix.first_transition(**legacy_kw), baseline)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_default_jax_graph_identical_to_independent_preedit(dtype):
    old = _baseline()
    jax = pytest.importorskip('jax')
    kw = {k: v for k, v in arguments(dtype).items() if k != 'foil_id'}
    assert str(jax.make_jaxpr(old.first_transition_jax)(**kw)) == str(
        jax.make_jaxpr(helix.first_transition_jax)(**kw))
