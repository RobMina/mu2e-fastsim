"""Synthetic, site-independent checks of the opt-in target photon birth source."""
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from mu2efastsim import eslab_jax as SJ
from beamfastsim import cechain as C
from beamfastsim import cechain_jax as CJ
from beamfastsim import flash_target_jax as F
from beamfastsim import fieldmap as FM
from beamfastsim import geometry as G

pytestmark = pytest.mark.skipif(not hasattr(SJ, "lane_step_emissions"), reason="full core emission interface required")


def _geom(foils=(5.0,), half=0.05):
    n = len(foils)
    b = G.BeamGeometry(0.0, np.asarray(foils), np.zeros(n), np.full(n, 75.0), np.full(n, half),
                       -50.0, 1.0, 100.0, 200.0, 300.0, 0.5, 100.0, 200.0, 900.0, 100.0, 200.0)
    return C.CEGeometry(b, 900.0, 900.0, 1.0, z_plane=20.0, z_up=-20.0, z_ds=-30.0,
                        coll5_z0=-29.0, coll5_z1=-28.0)


def _source(g=None, **changes):
    args = dict(key=jax.random.PRNGKey(7), geom=_geom() if g is None else g, field=[0.0, 0.0, 0.0],
                x=[[1.0, 0.0, 10.0]], u=[[0.0, 0.0, 1.0]], p=100.0, t=0.0, pdg=11,
                h=1.0, max_steps=1000, capacity=32, msc=False)
    args.update(changes)
    return F.target_source(**args)


@pytest.mark.parametrize("dtype,atol", [(np.float64, 1e-9), (np.float32, 1e-4)])
def test_vacuum_charges_mirror_analytic_helix_and_preserve_rows(dtype, atol):
    g = _geom()
    x = np.tile([1.0, 0.0, 10.0], (3, 1))
    u = np.tile([0.6, 0.0, 0.8], (3, 1))
    r = _source(g, x=x, u=u, p=[100.0] * 3, t=[1e12 + 0.125, 2e12, 3e12], pdg=[-11, 11, -11],
                field=[0.0, 0.0, 1.0], dtype=dtype)
    F.validate_complete(r)
    assert np.array_equal(r["parent_index"], np.arange(3))
    assert np.array_equal(r["pdg"], [-11, 11, -11])
    assert r["t0"].dtype == np.float64 and r["t0"][0] == 1e12 + 0.125
    st = r["state"]
    assert np.all(np.asarray(st["t"]) < 1.0)
    assert np.all(np.asarray(st["fate"]) == C.PLANE)
    for row, q in enumerate((1.0, -1.0, 1.0)):
        xe, ue = SJ.helix(jnp.asarray(x[row:row+1], dtype), jnp.asarray(u[row:row+1], dtype),
                          jnp.asarray([100.0], dtype), q, jnp.asarray([[0.0, 0.0, 1.0]], dtype),
                          jnp.asarray([12.5], dtype))
        np.testing.assert_allclose(st["x"][row], xe[0], atol=atol, rtol=atol)
        np.testing.assert_allclose(st["u"][row], ue[0], atol=atol, rtol=atol)
    np.testing.assert_allclose(st["x"][0, 1], -st["x"][1, 1], atol=atol)
    assert not np.asarray(r["count"]).any()


def test_multiple_foils_accumulate_global_births_and_energy_counters():
    g = _geom((5.0, 10.0), half=0.3)
    n = 64
    x = np.tile([10.0, 0.0, 0.0], (n, 1))
    r = _source(g, x=x, u=np.tile([0.0, 0.0, 1.0], (n, 1)), max_steps=3000, capacity=128)
    F.validate_complete(r)
    rec, st = r["records"], r["state"]
    valid = np.asarray(rec["valid"])
    assert valid.sum() > 5
    assert np.array_equal(valid.sum(axis=1), r["count"])
    assert np.array_equal(r["count"], st["nph"])
    np.testing.assert_allclose(np.asarray(rec["energy"]).sum(axis=1), st["eph"], atol=1e-12)
    assert np.all(np.asarray(rec["mechanism"])[valid] == 1)
    z = np.asarray(rec["position"])[valid, 2]
    assert np.all((np.abs(z - 5) <= 0.30001) | (np.abs(z - 10) <= 0.30001))
    assert np.any(z < 6.0) and np.any(z > 9.0)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(rec["direction"])[valid], axis=1), 1, atol=1e-12)
    assert np.all(np.asarray(rec["path"])[valid] >= 4.7)
    assert np.all(np.asarray(rec["time"])[valid] > 0)
    assert np.any(np.asarray(st["cnt"])[:, 0] == 2)
    assert np.array_equal(r["ndelta"], st["nde"])
    assert np.asarray(r["ndelta"]).sum() > 0
    initial = np.hypot(100.0, CJ.ME) - CJ.ME
    np.testing.assert_allclose(st["T"] + st["cont"] + st["eph"] + st["ede"], initial, atol=2e-11)
    for name, values in rec.items():
        assert not np.asarray(values)[~valid].any(), name


@pytest.mark.parametrize("z", [5.0, 10.0])
def test_initial_rest_annihilation_once_foil_or_vacuum(z):
    r = _source(x=[[2.0, 3.0, z]], u=[[0.0, 0.0, -1.0]], p=0.0001, pdg=-11, capacity=2)
    F.validate_complete(r)
    assert int(r["count"][0]) == 2 and int(r["nann"][0]) == 2
    assert int(r["state"]["fate"][0]) == C.STOPPED
    assert int(r["state"]["nslab"][0]) == 0
    np.testing.assert_array_equal(r["records"]["mechanism"], [[3, 3]])
    np.testing.assert_allclose(r["records"]["energy"], CJ.ME)
    np.testing.assert_allclose(r["records"]["position"][0], [[2, 3, z], [2, 3, z]])
    np.testing.assert_allclose(r["records"]["direction"][0, 0], -r["records"]["direction"][0, 1])
    assert not np.asarray(r["records"]["time"]).any()


def test_overflow_iteration_cap_and_looper_guard():
    r = _source(p=0.0001, pdg=-11, capacity=1)
    assert int(r["overflow"][0]) == 1 and int(r["count"][0]) == 2
    with pytest.raises(ValueError, match="dropped births=1"):
        F.validate_complete(r)
    r = _source(max_steps=1)
    assert bool(r["capped"][0])
    with pytest.raises(ValueError, match="capped parents=1"):
        F.validate_complete(r)
    r = _source(s_max=0.5)
    assert bool(r["loopers"][0])
    with pytest.raises(ValueError, match="loopers=1"):
        F.validate_complete(r)


def test_low_momentum_turns_are_shortened_and_capped_not_exit_flux():
    r = _source(p=0.1, field=[0.0, 0.0, 1.0], u=[[1.0, 0.0, 0.0]], max_steps=4)
    assert bool(r["capped"][0])
    assert int(r["state"]["nadapt"][0]) == 4
    assert float(r["state"]["s"][0]) < 0.1
    assert not np.asarray(r["count"]).any()


def test_device_map_matches_constant_field():
    g = _geom()
    fm = FM.FieldMap(np.array([-100, -100, -100.0]), np.full(3, 200.0), np.full(3, 2),
                      np.broadcast_to([0.0, 0.0, 1.0], (2, 2, 2, 3)))
    a = _source(g, field=[0.0, 0.0, 1.0], u=[[0.6, 0, 0.8]])
    b = _source(g, field=(fm,), u=[[0.6, 0, 0.8]])
    np.testing.assert_allclose(a["state"]["x"], b["state"]["x"], atol=1e-10)


@pytest.mark.parametrize("positron", [False, True])
def test_build_default_and_explicit_state_only_are_identical(positron):
    g = _geom()
    args = (g, np.dtype(np.float64), 1.0, 100.0, False, True, True, None, False, np.zeros(3))
    default = CJ._build(*args) if not positron else CJ._build(*args, positron=True)
    explicit = CJ._build(*args, positron=positron, photon_capacity=None)
    assert default is explicit
    init, run = default
    st = init((), jnp.array([[1.0, 0.0, 5.0]]), jnp.array([[0.0, 0.0, 1.0]]),
              jnp.array([100.0]), jnp.zeros(1), jnp.array([0]), jnp.array([0.05]))
    key = jax.random.PRNGKey(1)
    _, a = run((), key, st, jnp.int32(300), 1, 1)
    ip, rp = CJ._build(*args, positron=positron, photon_capacity=16)
    sp = ip((), jnp.array([[1.0, 0.0, 5.0]]), jnp.array([[0.0, 0.0, 1.0]]),
            jnp.array([100.0]), jnp.zeros(1), jnp.array([0]), jnp.array([0.05]), key)
    _, b = rp((), key, sp, jnp.int32(300), 1, 1)
    for name in CJ.OUT_FIELDS:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


@pytest.mark.parametrize("change", [dict(p=0), dict(p=np.nan), dict(p=2000.0), dict(pdg=13), dict(capacity=0),
                                   dict(capacity=1.5), dict(max_steps=True), dict(x=[[0, 0]]),
                                   dict(u=[[0, 0, 0]]), dict(u=[[np.inf, 0, 1]]), dict(t=[0, 1]),
                                   dict(field=[0, np.nan, 0]), dict(h=0), dict(dtype=np.int32)])
def test_reject_malformed_inputs(change):
    with pytest.raises(ValueError):
        _source(**change)


def test_reject_unresolved_geometry_and_initial_absorber_states():
    with pytest.raises(ValueError, match="gap"):
        _source(_geom((5.0, 6.0)), h=1.0)
    g = _geom()
    with pytest.raises(ValueError, match="vacuum or foils"):
        _source(g, x=[[300.1, 0, 150.0]])
    b = replace(g.beam, foil_ht=np.array([-0.1]))
    with pytest.raises(ValueError):
        _source(replace(g, beam=b))


def test_positron_step_annihilation_records_and_global_energy_accounting():
    g = _geom(half=2.0)
    n = 256
    r = _source(g, x=np.tile([1.0, 0.0, 2.0], (n, 1)), u=np.tile([0.0, 0.0, 1.0], (n, 1)),
                p=1.0, pdg=-11, capacity=128, max_steps=3000, key=jax.random.PRNGKey(21))
    F.validate_complete(r)
    st, rec = r["state"], r["records"]
    valid = np.asarray(rec["valid"])
    mechanism = np.asarray(rec["mechanism"])
    assert set(np.unique(mechanism[valid])) == {1, 2, 3}
    assert np.array_equal(r["count"], st["nph"] + st["nan"])
    assert np.array_equal((mechanism >= 2).sum(axis=1), st["nan"])
    np.testing.assert_allclose(np.asarray(rec["energy"]).sum(axis=1), st["eph"] + st["ean"], atol=1e-12)
    expected = np.hypot(1.0, CJ.ME) - CJ.ME + np.where(np.asarray(st["nan"]) > 0, 2 * CJ.ME, 0)
    np.testing.assert_allclose(st["T"] + st["cont"] + st["eph"] + st["ede"] + st["ean"], expected, atol=1e-12)
    rest = mechanism == 3
    np.testing.assert_allclose(np.asarray(rec["energy"])[rest], CJ.ME)
    for row in np.flatnonzero(rest.any(axis=1)):
        directions = np.asarray(rec["direction"])[row, rest[row]]
        np.testing.assert_allclose(directions[0], -directions[1], atol=1e-12)
    z = np.asarray(rec["position"])[valid, 2]
    assert np.all((z >= 3.0) & (z <= 7.0))


def test_optional_surface_endpoint_is_not_skipped():
    g = _geom(half=2.0)
    r = _source(g, x=[[1.0, 0.0, 2.0]], p=0.1, pdg=-11, max_steps=100)
    F.validate_complete(r)
    assert int(r["count"][0]) == 2
    assert int(r["state"]["fate"][0]) == C.STOPPED


@pytest.mark.parametrize("code", [11, -11])
def test_scattering_path_counts_match_full_records(code):
    n = 16
    r = _source(x=np.tile([10.0, 0.0, 5.0], (n, 1)), u=np.tile([0.0, 0.0, 1.0], (n, 1)),
                p=1.0, pdg=code, msc=True, max_steps=3000, capacity=128)
    F.validate_complete(r)
    assert np.array_equal(r["count"], r["state"]["nph"] + r["state"]["nan"])
    np.testing.assert_allclose(np.asarray(r["records"]["energy"]).sum(axis=1),
                               r["state"]["eph"] + r["state"]["ean"], atol=1e-12)


def test_capped_slab_state_retains_partial_primary_progress():
    g = _geom(half=2.0)
    r = _source(g, x=[[1.0, 0.0, 5.0]], p=100.0, max_steps=1)
    assert bool(r["capped"][0])
    assert float(r["state"]["s"][0]) > 0
    assert float(r["state"]["t"][0]) > 0
    assert float(r["state"]["x"][0, 2]) > 5.0
    assert float(r["state"]["T"][0]) < np.hypot(100.0, CJ.ME) - CJ.ME


def test_reject_initial_opa_ring_and_boundary():
    g = replace(_geom(), ring_t=5.0, ring_hl=2.0)
    for point in ([900.1, 0.0, 150.0], [298.0, 0.0, 101.0], [1.0, 0.0, 20.0]):
        with pytest.raises(ValueError, match="vacuum or foils"):
            _source(g, x=[point])


def test_shell_equivalent_curvature_field_reverses_with_charge():
    g = _geom()
    g = replace(g, beam=replace(g.beam, ipa_r=3.0, ipa_t=0.2, ipa_z0=10.0, ipa_z1=15.0))
    states = []
    for positron in (False, True):
        init, run = CJ._build(g, np.dtype(np.float64), 1.0, 100.0, False, True, True, None,
                              False, np.zeros(3), positron=positron, photon_capacity=4)
        st = init((), jnp.array([[2.5, 0.0, 12.0]]), jnp.array([[0.8, 0.6, 0.0]]),
                  jnp.array([100.0]), jnp.zeros(1), jnp.array([-1]), jnp.zeros(1))
        _, st = run((), jax.random.PRNGKey(0), st, jnp.int32(1), 1, 1)
        assert bool(st["inl"][0])
        states.append(st)
    assert np.linalg.norm(states[0]["Bl"]) > 0
    np.testing.assert_allclose(states[0]["Bl"], -states[1]["Bl"], atol=1e-12)
    np.testing.assert_allclose(states[0]["xent"], states[1]["xent"], atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("positron", [False, True])
def test_transverse_low_momentum_caps_but_keeps_advancing(dtype, positron):
    geom = _geom()
    init, run = CJ._build(geom, np.dtype(dtype), 1.0, 100.0, False, True, True, None,
                          False, np.array([0., 0., 1.]), positron=positron, photon_capacity=2)
    momentum = .1
    kinetic = momentum * momentum / (np.hypot(momentum, CJ.ME) + CJ.ME)
    x = jnp.array([[1., 0., 10.]], dtype)
    u = jnp.array([[1., 0., 0.]], dtype)
    key = jax.random.PRNGKey(31)
    state = init((), x, u, jnp.array([kinetic], dtype), jnp.zeros(1, dtype),
                 jnp.array([-1]), jnp.zeros(1, dtype), key)
    _, first = run((), key, state, jnp.int32(100), 1, 1)
    _, second = run((), key, first, jnp.int32(100), 1, 1)
    step = .05 * momentum / SJ.K_FIELD
    np.testing.assert_allclose(first["s"], [100 * step], rtol=2e-5)
    np.testing.assert_allclose(second["s"], [200 * step], rtol=2e-5)
    expected_x, expected_u = SJ.helix(x, u, jnp.array([momentum], dtype), 1. if positron else -1.,
                                     jnp.array([[0., 0., 1.]], dtype), jnp.array([200 * step], dtype))
    np.testing.assert_allclose(second["x"], expected_x, atol=2e-5)
    np.testing.assert_allclose(second["u"], expected_u, atol=2e-5)
    assert int(second["nvac"][0]) == int(second["nadapt"][0]) == 200
    assert int(second["nstall"][0]) == int(second["nslab"][0]) == 0
    assert int(second["fate"][0]) == C.LIVE
    assert not np.asarray(second["count"]).any()
