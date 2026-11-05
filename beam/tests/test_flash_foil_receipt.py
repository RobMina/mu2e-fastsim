"""Standalone guarded geometric receipts, never transport/restart validation."""
from dataclasses import FrozenInstanceError, replace
import importlib.util
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from beamfastsim import flash_foil_helix as helix
from beamfastsim import flash_foil_receipt as receipt


def arguments(dtype=np.float64, **changes):
    kw = dict(x=np.array([0., 0., 0.], dtype), u=np.array([1., 0., 0.], dtype),
              p=helix.K_FIELD, q=1, B=[0., 0., 1.], rin=1., rout=3.,
              zlo=-2., zhi=2., s_max=2*np.pi, foil_id=19)
    kw.update(changes)
    return kw


def legacy(kw):
    return helix.first_transition(**{k: v for k, v in kw.items()
                                   if k not in ('foil_id', 'frame_binding')})


def same_bytes(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype
        assert a[k].shape == b[k].shape
        assert a[k].tobytes() == b[k].tobytes()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('kind', [helix.RIN, helix.ROUT, helix.ZLO, helix.ZHI])
def test_independent_analytic_crossing_enclosure(dtype, kind):
    kw = arguments(dtype)
    if kind == helix.RIN:
        root = np.pi/3  # r=2*sin(s/2), monotone before pi
    elif kind == helix.ROUT:
        # r^2=6+4*sin(s)-2*cos(s); first exit of outer cylinder.
        kw.update(x=np.array([2., 0., 0.], dtype), exiting=True)
        root = np.arcsin(3/np.sqrt(20))+np.arctan(.5)
    else:
        # Independent tilted-field closed form z=q*(1-cos(s)).
        kw.update(x=np.array([2., 0., 0.], dtype), B=[0., 1., 0.], rin=0.,
                  rout=20., exiting=True)
        kw.update(zhi=.5) if kind == helix.ZHI else kw.update(zlo=-.5, q=-1)
        root = np.pi/3
    expected = receipt.bind_inputs(**kw)
    ordinary, rs = receipt.first_transition_with_receipts(**kw)
    same_bytes(ordinary, legacy(kw))
    assert len(rs) == 1 and rs[0] is not None
    r = rs[0]
    e = r.evidence
    assert ordinary['valid'] and e['kind'] == kind
    assert e['isolation'][0] < root < e['isolation'][1]
    assert e['isolation'][0] <= e['refined'][0] <= e['distance'] <= e['refined'][1] <= e['isolation'][1]
    assert e['endpoint_clearance'][0]*e['endpoint_clearance'][1] < 0
    assert e['derivative_margin'] > 0
    assert np.all(e['other_clearance_lower'][e['other_clearance_enabled']] > 0)
    assert e['resolution'] >= e['guard'] > 0
    assert e['forward_material'] == (not kw.get('exiting', False))
    assert r.transport_restart_valid is False
    # Check isolation only, never claim refined width is an FP error bound.
    assert receipt.verify_receipt(r, expected=expected)


def test_copied_immutable_exact_binding_and_evidence():
    kw = arguments(frame_binding=np.eye(4), foil_id=np.array(81, dtype=np.int32))
    _, (r,) = receipt.first_transition_with_receipts(**kw)
    expected = receipt.bind_inputs(**kw)
    kw['x'][0] = 5
    kw['frame_binding'][0, 3] = 7
    assert r.binding.inputs['x'][0] == 0
    assert r.binding.frame_binding[0, 3] == 0
    assert r.binding.foil_id.dtype == np.int32
    arrays = [v for v in r.binding.inputs.values() if v is not None]
    arrays += [r.binding.foil_id, r.binding.frame_binding, *r.evidence.values()]
    for a in arrays:
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a.setflags(write=True)
        with pytest.raises(ValueError):
            a.flat[0] = 0
    with pytest.raises(TypeError):
        r.evidence['distance'] = 0
    with pytest.raises(FrozenInstanceError):
        r.transport_restart_valid = True
    assert receipt.verify_receipt(r, expected=expected)


@pytest.mark.parametrize('field', ['isolation', 'refined', 'distance', 'kind', 'guard',
                                  'resolution', 'derivative_margin', 'derivative_mid',
                                  'derivative_variation', 'endpoint_clearance',
                                  'other_clearance_lower', 'other_clearance_enabled',
                                  'forward_material'])
def test_every_evidence_field_recomputed(field):
    kw = arguments()
    expected = receipt.bind_inputs(**kw)
    _, (r,) = receipt.first_transition_with_receipts(**kw)
    e = dict(r.evidence)
    a = e[field].copy()
    if a.dtype.kind == 'b':
        a.flat[0] = not a.flat[0]
    elif a.dtype.kind == 'i':
        a.flat[0] += 1
    else:
        a.flat[0] = np.nextafter(a.flat[0], np.inf)
    e[field] = a
    assert not receipt.verify_receipt(replace(r, evidence=MappingProxyType(e)), expected=expected)


@pytest.mark.parametrize('field', ['x', 'u', 'p', 'q', 'B', 'rin', 'rout', 'zlo', 'zhi',
                                  'axis_x', 's_max', 'exiting', 'tolerance', 'foil_id',
                                  'frame_binding', 'max_intervals', 'after_surface'])
def test_external_expected_rebinding_rejected(field):
    kw = arguments(frame_binding=np.eye(4))
    _, (r,) = receipt.first_transition_with_receipts(**kw)
    changes = dict(x=[.001, 0, 0], u=[0, 1., 0], p=helix.K_FIELD+.001, q=-1,
                   B=[.01, 0, 1], rin=.99, rout=3.01, zlo=-2.01, zhi=2.01,
                   axis_x=.001, s_max=7., exiting=True, tolerance=1.e-8,
                   foil_id=20, frame_binding=np.diag([-1., -1., 1., 1.]),
                   max_intervals=513, after_surface=helix.RIN)
    altered = receipt.bind_inputs(**dict(kw, **{field: changes[field]}))
    assert not receipt.verify_receipt(r, expected=altered)
    # Receipt-supplied identity changes do not count as independent evidence.
    forged = replace(r, binding=altered)
    assert not receipt.verify_receipt(forged, expected=receipt.bind_inputs(**kw))


def test_schema_shape_dtype_nonfinite_and_restart_tampering():
    kw = arguments()
    expected = receipt.bind_inputs(**kw)
    _, (r,) = receipt.first_transition_with_receipts(**kw)
    for bad in (replace(r, schema='other'), replace(r, transport_restart_valid=True),
                replace(r, lane_index=(0,)), replace(r, evidence={}),
                replace(r, binding=None), replace(r, transport_restart_valid=0),
                None, {'distance': 1.}):
        assert not receipt.verify_receipt(bad, expected=expected)
    for bad in (np.array(np.nan), np.array([float(r.evidence['distance'])]),
                r.evidence['distance'].astype(np.float32)):
        assert not receipt.verify_receipt(replace(r, evidence=dict(r.evidence, distance=bad)),
                                          expected=expected)
    with pytest.raises(ValueError):
        receipt.verify_receipt(r, expected=None)
    with pytest.raises(ValueError):
        receipt.verify_receipt(r, expected=replace(expected, inputs=None))


def test_expected_binding_preserves_precast_bytes_and_optional_frame_presence():
    kw = arguments(np.float32, frame_binding=np.eye(4))
    _, (r,) = receipt.first_transition_with_receipts(**kw)
    # This B perturbation is lost by the float32 solver cast but must still
    # reject an external input rebinding. Identity is not just solver outputs.
    changed_B = np.array(kw['B'])
    changed_B[2] = np.nextafter(changed_B[2], np.inf)
    for changes in (dict(B=changed_B), dict(frame_binding=None),
                    dict(x=kw['x'].astype(np.float64))):
        assert not receipt.verify_receipt(r, expected=receipt.bind_inputs(**dict(kw, **changes)))


def test_emission_import_has_no_jax_dependency():
    import os
    import subprocess
    import sys
    code = """
import sys
from beamfastsim import flash_foil_receipt
assert not any(k == 'jax' or k.startswith('jax.') for k in sys.modules)
"""
    subprocess.run([sys.executable, '-c', code], check=True, env=os.environ.copy())


def test_same_thickness_distinct_foil_identity_and_no_guessed_frame():
    kw = arguments(x=np.array([[0., 0., 0.], [0., 0., 10.]]), zlo=np.array([-2., 8.]),
                   zhi=np.array([2., 12.]), foil_id=np.array([71, 72]))
    _, rs = receipt.first_transition_with_receipts(**kw)
    expected = receipt.bind_inputs(**kw)
    assert all(r is not None and receipt.verify_receipt(r, expected=expected) for r in rs)
    assert rs[0].binding.frame_binding is None
    assert rs[0].lane_index == (0,) and rs[1].lane_index == (1,)
    swapped = receipt.bind_inputs(**dict(kw, foil_id=np.array([72, 71])))
    assert not receipt.verify_receipt(rs[0], expected=swapped)
    with pytest.raises(ValueError, match='same foil_id'):
        receipt.bind_inputs(**dict(kw, foil_id=[71, 71]))


@pytest.mark.parametrize('changes', [dict(q=0), dict(B=[0., 0., 0.]),
    dict(x=np.array([10., 0., 0.])), dict(max_intervals=1),
    dict(x=np.array([1., 0., 0.])),
    dict(x=np.array([2., 0., -4.]), u=np.array([0., 0., 1.]), s_max=2.),
    dict(x=np.array([1., 0., 0.]), exiting=True, after_surface=helix.RIN),
    dict(x=np.array([3., 0., 0.]), u=np.array([0., 1., 0.])),
    dict(rout=1.0001, tolerance=.01)])
def test_explicit_refusal_preserves_ordinary_result(changes):
    kw = arguments(**changes)
    ordinary, rs = receipt.first_transition_with_receipts(**kw)
    same_bytes(ordinary, legacy(kw))
    assert rs == (None,)


def test_supported_unsupported_mix_and_multidimensional_lane_order():
    kw = arguments(x=np.zeros((2, 2, 3)), q=np.array([[1, 0], [-1, 1]]),
                   s_max=np.array([[2*np.pi, 2*np.pi], [2*np.pi, .1]]))
    ordinary, rs = receipt.first_transition_with_receipts(**kw)
    same_bytes(ordinary, legacy(kw))
    assert len(rs) == 4 and rs[1] is None and rs[3] is None
    assert rs[0].lane_index == (0, 0) and rs[2].lane_index == (1, 0)
    expected = receipt.bind_inputs(**kw)
    assert receipt.verify_receipt(rs[2], expected=expected)
    assert not receipt.verify_receipt(replace(rs[0], lane_index=(False, False)), expected=expected)


def test_approximate_endpoint_does_not_authorize_restart_or_snap():
    kw = arguments(x=np.array([2., 0., 0.]), B=[0., 1., 0.], rin=0., rout=20.,
                   zhi=.371, exiting=True)
    ordinary, (r,) = receipt.first_transition_with_receipts(**kw)
    assert r is not None and not r.transport_restart_valid
    s = float(ordinary['distance'])
    # Independent orbit evaluation: result is left untouched, not projected.
    x = np.array([2.+np.sin(s), 0., 2*np.sin(s/2)**2])
    u = np.array([np.cos(s), 0., np.sin(s)])
    resumed = helix.first_transition(**{k: v for k, v in dict(kw, x=x, u=u, exiting=False,
                                        after_surface=helix.ZHI).items() if k != 'foil_id'})
    if x[2] != kw['zhi']:
        assert resumed['unresolved'] and not resumed['valid']
    else:
        assert not resumed['unresolved']
    perturbed = x.copy()
    perturbed[2] = np.nextafter(kw['zhi'], np.inf)
    refused = helix.first_transition(**{k: v for k, v in dict(kw, x=perturbed, u=u, exiting=False,
                                        after_surface=helix.ZHI).items() if k != 'foil_id'})
    assert refused['unresolved'] and not refused['valid']
    assert receipt.verify_receipt(r, expected=receipt.bind_inputs(**kw))


@pytest.mark.parametrize('changes', [dict(max_intervals=True), dict(max_intervals=0),
    dict(max_intervals=1.5), dict(max_intervals=4097), dict(foil_id=True),
    dict(foil_id=1.2), dict(foil_id=-1), dict(foil_id=[1, 2]),
    dict(foil_id=np.uint64(2**63)), dict(frame_binding='lab'),
    dict(frame_binding=np.eye(3)), dict(frame_binding=np.full((4, 4), np.nan)),
    dict(x=[0, 0]), dict(u=[2., 0., 0.]), dict(B=[0., np.nan, 1.]),
    dict(p=np.inf), dict(q=2), dict(s_max=-1.), dict(tolerance=-1.),
    dict(exiting=0), dict(after_surface=1.), dict(rin=-1.), dict(zhi=-3.),
    dict(x=np.array([0, 0, 0], dtype=complex)), dict(B=[0., 1.]),
    dict(B=[False, False, True]), dict(x=np.array([0, 0, 0], dtype=object)),
    dict(frame_binding=np.full((4, 4), 1j)), dict(frame_binding=np.ones((4, 4), bool))])
def test_invalid_inputs_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        receipt.first_transition_with_receipts(**arguments(**changes))


def test_unknown_options_rejected():
    with pytest.raises(TypeError):
        receipt.first_transition_with_receipts(**arguments(restart=True))


def _preedit():
    source = os.environ.get('FLASH_FOIL_RECEIPT_BASELINE_SOURCE')
    if not source:
        pytest.skip('provide independent pre-receipt helix source for byte and graph checks')
    path = Path(source)
    if not path.is_file():
        pytest.fail('requested independent pre-receipt helix source is absent')
    spec = importlib.util.spec_from_file_location('beamfastsim._pre_receipt', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_default_and_observed_bytes_against_preedit_snapshot(dtype):
    old = _preedit()
    rng = np.random.default_rng(20261104)
    x = rng.uniform(-4., 4., (120, 3)).astype(dtype)
    u = rng.normal(size=(120, 3)).astype(dtype)
    u /= np.linalg.norm(u, axis=-1)[:, None]
    kw = arguments(dtype, x=x, u=u, q=rng.choice([-1, 0, 1], 120),
                   p=rng.uniform(.1, 1., 120), B=rng.normal(size=(120, 3)),
                   exiting=np.hypot(x[:, 0], x[:, 1]) > 1)
    # Determine initial material ownership exactly from all constraints.
    kw['exiting'] = ((np.hypot(x[:, 0], x[:, 1]) > 1) &
                     (np.hypot(x[:, 0], x[:, 1]) < 3) & (abs(x[:, 2]) < 2))
    old_result = old.first_transition(**{k: v for k, v in kw.items() if k != 'foil_id'})
    same_bytes(legacy(kw), old_result)
    observed, _ = receipt.first_transition_with_receipts(**kw)
    same_bytes(observed, old_result)


def test_default_jax_graph_matches_preedit_snapshot():
    old = _preedit()
    jax = pytest.importorskip('jax')
    kw = {k: v for k, v in arguments().items() if k != 'foil_id'}
    # No observer operations/constants enter the traced graph.
    assert str(jax.make_jaxpr(old.first_transition_jax)(**kw)) == str(
        jax.make_jaxpr(helix.first_transition_jax)(**kw))


def test_singleton_axis_cross_broadcast_preserves_lane_and_foil_bindings():
    kw = arguments(x=np.array([[[0., 0., 0.]], [[0., 0., 10.]]]),
                   p=np.array([[.8, 1., 1.2]])*helix.K_FIELD,
                   zlo=np.array([[-2.], [8.]]), zhi=np.array([[2.], [12.]]),
                   foil_id=np.array([[71], [72]], dtype=np.uint64))
    ordinary, rs = receipt.first_transition_with_receipts(**kw)
    same_bytes(ordinary, legacy(kw))
    assert ordinary['valid'].shape == (2, 3)
    assert ordinary['valid'].all() and len(rs) == 6
    expected = receipt.bind_inputs(**kw)
    assert tuple(r.lane_index for r in rs) == tuple(np.ndindex((2, 3)))
    assert all(receipt.verify_receipt(r, expected=expected) for r in rs)
    with pytest.raises(ValueError, match='same foil_id'):
        receipt.bind_inputs(**dict(kw, foil_id=np.array([[71], [71]], np.uint64)))


@pytest.mark.parametrize('dtype', [np.int64, np.uint64])
def test_identity_array_integer_limit_never_wraps(dtype):
    kw = arguments(x=np.zeros((2, 3)), foil_id=np.array([0, 2**63-1], dtype))
    _, rs = receipt.first_transition_with_receipts(**kw)
    expected = receipt.bind_inputs(**kw)
    assert expected.foil_id.dtype == dtype
    assert all(receipt.verify_receipt(r, expected=expected) for r in rs)
    for invalid in (np.array([0, 2**63], np.uint64), np.array([0, -1], np.int64)):
        with pytest.raises(ValueError, match='foil_id'):
            receipt.bind_inputs(**dict(kw, foil_id=invalid))


def test_both_sides_of_numerical_representative_never_authorize_restart():
    kw = arguments(x=np.array([2., 0., 0.]), B=[0., 1., 0.], rin=0., rout=20.,
                   zhi=.371, exiting=True)
    _, (r,) = receipt.first_transition_with_receipts(**kw)
    s = float(r.evidence['distance'])
    direction = np.array([np.cos(s), 0., np.sin(s)])
    for side in (-np.inf, np.inf):
        point = np.array([2.+np.sin(s), 0., np.nextafter(kw['zhi'], side)])
        assert (point[2] > kw['zhi']) == (side > 0)
        result = helix.first_transition(**{k: v for k, v in dict(kw, x=point, u=direction,
                 exiting=False, after_surface=helix.ZHI).items() if k != 'foil_id'})
        assert result['unresolved'] and not result['valid']
        assert r.transport_restart_valid is False
    assert receipt.verify_receipt(r, expected=receipt.bind_inputs(**kw))
