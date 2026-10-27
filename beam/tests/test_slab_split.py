"""Split debit and endpoint contracts; no physics-closure claim from synthetic tests."""
import numpy as np
import pytest
import jax.numpy as jnp

from mu2efastsim import materials, eloss_split_jax as split
from beamfastsim import slabdiffusion as sd, slabstudy as ss


def tables():
    e = np.geomspace(.001, 100., 80)
    return sd.make_tables(e, e, e, np.ones_like(e))


def run(**kwargs):
    args = dict(material=materials.ALUMINIUM, scatter=False, loss='split', range_kind='restricted',
                tcut=.005, max_step=.01, replicas=5, seed=109)
    args.update(kwargs)
    return sd.respond(np.array([5., 2.]), np.tile([0., 0., 1.], (2, 1)), .02, tables(), **args)


def test_split_requires_restricted_range_and_material():
    with pytest.raises(ValueError, match='restricted'):
        run(range_kind='inclusive')
    with pytest.raises(ValueError, match='material'):
        run(material=None)


def test_split_energy_ledger_and_batch_replay():
    a, b = run(batch=10), run(batch=1)
    for name, values in ss.flatten(a).items():
        np.testing.assert_array_equal(values, ss.flatten(b)[name])
    l = a.ledger
    np.testing.assert_allclose(l['requested_loss'], l['soft_requested'] + l['hard_requested'], atol=1e-14)
    np.testing.assert_allclose(l['applied_loss'], l['soft_applied'] + l['hard_applied'], atol=1e-14)
    np.testing.assert_allclose(l['mean_loss'], l['soft_mean'] + l['hard_mean'], atol=1e-14)
    np.testing.assert_allclose(a.raw_energy + l['applied_loss'], np.tile([5., 2.], (5, 1)), atol=1e-14)
    assert np.all(l['hard_count'] == np.floor(l['hard_count']))
    assert np.any(l['hard_count'] > 0)
    assert np.all(a.status == sd.TRANSMITTED)


def test_no_hard_above_tmax():
    a = run(tcut=100.)
    assert not np.any(a.ledger['hard_count'])
    assert not np.any(a.ledger['hard_requested'])
    assert not np.any(a.ledger['hard_mean'])


def test_floor_never_called_stop():
    a = sd.respond(np.array([.001]), np.array([[0., 0., 1.]]), 1., tables(),
                   material=materials.ALUMINIUM, loss='split', range_kind='restricted')
    assert a.status[0, 0] == sd.UNRESOLVED_FLOOR
    assert a.raw_energy[0, 0] > 0
    assert not a.ledger['hard_count'][0, 0]
    assert ss.summarize(a)['certified_physical_stop_count'] is None


def test_clipping_keeps_soft_and_hard_requests(monkeypatch):
    def controlled(*args, **kw):
        return split.SplitLoss(jnp.array(3.), jnp.array(4.), jnp.array(2))
    monkeypatch.setattr(split, 'sample_loss_split', controlled)
    a = run(max_steps=17, replicas=1)
    assert np.all(a.status == sd.RAWZERO)
    np.testing.assert_allclose(a.ledger['soft_applied'], [[3., 2.]])
    np.testing.assert_allclose(a.ledger['hard_applied'], [[2., 0.]])
    np.testing.assert_allclose(a.ledger['clipped_loss'], [[2., 5.]])
    np.testing.assert_array_equal(a.ledger['hard_count'], [[2., 2.]])


def test_nonfinite_split_request_is_unresolved(monkeypatch):
    monkeypatch.setattr(split, 'sample_loss_split', lambda *a, **kw:
                        split.SplitLoss(jnp.array(1.), jnp.array(jnp.nan), jnp.array(1)))
    a = run(max_steps=19, replicas=1)
    assert np.all(a.status == sd.UNRESOLVED_NUMERICAL)
    for name in ('soft_requested', 'hard_requested', 'hard_count', 'soft_applied', 'hard_applied'):
        np.testing.assert_array_equal(a.ledger[name], [[0., 0.]])
    np.testing.assert_array_equal(a.raw_energy, [[5., 2.]])


def test_split_scattered_permuted_and_sharded_replay():
    n = 260
    energy = np.linspace(2., 8., n)
    direction = np.tile([.6, 0., .8], (n, 1))
    rows = np.arange(n, dtype=np.uint32) + 500
    kw = dict(material=materials.ALUMINIUM, loss='split', range_kind='restricted',
              scatter=True, tcut=.005, max_steps=3, seed=193, replicas=2, replica_offset=7)
    a = sd.respond(energy, direction, .1, tables(), row_ids=rows, batch=256, **kw)
    b = sd.respond(energy[::-1], direction, .1, tables(), row_ids=rows[::-1], batch=73, **kw)
    for name, values in ss.flatten(a).items():
        np.testing.assert_array_equal(values, ss.flatten(b)[name][:, ::-1])
    kw.update(replicas=1, replica_offset=8)
    c = sd.respond(energy[17:29], direction[17:29], .1, tables(), row_ids=rows[17:29], batch=5, **kw)
    for name, values in ss.flatten(c).items():
        np.testing.assert_array_equal(values, ss.flatten(a)[name][1:2, 17:29])


@pytest.mark.parametrize('soft,hard', [(np.nan, 1.), (1., np.inf), (-1., 2.), (2., -1.)])
def test_malformed_component_never_partially_debits(monkeypatch, soft, hard):
    monkeypatch.setattr(split, 'sample_loss_split', lambda *a, **kw:
                        split.SplitLoss(jnp.array(soft), jnp.array(hard), jnp.array(1)))
    sd._run.clear_cache()
    a = run(max_steps=23, replicas=1)
    assert np.all(a.status == sd.UNRESOLVED_NUMERICAL)
    assert not np.any(a.ledger['applied_loss'])
    assert not np.any(a.path)


def test_split_face_clipping_and_positive_subfloor_exit(monkeypatch):
    monkeypatch.setattr(split, 'sample_loss_split', lambda *a, **kw:
                        split.SplitLoss(jnp.array(.1995), jnp.array(0.), jnp.array(0)))
    sd._run.clear_cache()
    a = sd.respond(np.array([.2, .1]), np.tile([0., 0., -1.], (2, 1)), .00001, tables(),
                   material=materials.ALUMINIUM, loss='split', range_kind='restricted', scatter=False)
    assert a.status[0, 0] == sd.TRANSMITTED
    assert 0 < a.raw_energy[0, 0] < .001
    assert a.status[0, 1] == sd.RAWZERO
    assert np.all(a.ledger['face_reached'])
    assert np.all(a.position[..., 2] == 0)
    assert a.ledger['clipped_loss'][0, 1] > 0
