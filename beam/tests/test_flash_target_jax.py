"""Synthetic tests of the opt-in brems-only charged source, not full S1 validation."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from beamfastsim import cechain as C
from beamfastsim import geometry as G
from beamfastsim import flash_navigator_jax as FN
from beamfastsim.flash_target_jax import target_source, validate_complete
from mu2efastsim import eslab_jax as SJ

AXIS = -3904.0
KEY = jax.random.PRNGKey(204)


def _geom(foil_z=(5800.0, 5822.0), ht=2.5, plane=5840.0):
    n = len(foil_z)
    beam = G.BeamGeometry(AXIS, np.array(foil_z), np.full(n, 21.5), np.full(n, 75.0), np.full(n, ht),
                          4204.6, 25.4, 235.0, 525.0, 300.0, 0.5, 6900.0, 7900.0,
                          900.0, 4280.0, 8530.0)
    return C.CEGeometry(beam, 900.0, 900.0, 1.0, z_plane=plane)


def _starts(n, z=5796.5):
    return np.tile([AXIS + 40.0, 0.0, z], (n, 1)), np.tile([0.0, 0.0, 1.0], (n, 1))


@pytest.fixture(scope="module")
def vacuum_geom():
    return _geom(foil_z=(5000.0,), ht=0.0528, plane=5710.0)


@pytest.fixture(scope="module")
def slab_case():
    g = _geom()
    x, u = _starts(24)
    pdg = np.tile([11, -11], 12)
    r = target_source(KEY, g, [0., 0., 0.], x, u, 105., 0., pdg, msc=False, max_steps=5000)
    return g, x, u, pdg, r


@pytest.mark.slow
def test_vacuum_charge_conjugation_and_exact_endpoint(vacuum_geom):
    x, u = _starts(2, z=5700.)
    u[:] = [0.6, 0., 0.8]
    r = target_source(KEY, vacuum_geom, [0., 0., 1.], x, u, 10., 0., [11, -11], msc=False)
    st = r["state"]
    assert np.all(np.asarray(st["fate"]) == C.PLANE)
    assert not np.any(r["count"]) and not np.any(r["records"]["valid"])
    assert np.all(np.asarray(st["nadapt"]) == 0)
    np.testing.assert_allclose(st["x"][:, 2], vacuum_geom.z_plane, atol=1e-9)
    np.testing.assert_allclose(st["x"][0, 0], st["x"][1, 0], atol=1e-9)
    np.testing.assert_allclose(st["x"][0, 1], -st["x"][1, 1], atol=1e-9)
    np.testing.assert_allclose(st["u"][0], np.asarray(st["u"][1]) * [1., -1., 1.], atol=1e-10)
    np.testing.assert_allclose(st["s"], 12.5, atol=1e-9)
    validate_complete(r)
    # Exactly one 1-mm vacuum step to the plane must set PLANE, not skip the face.
    x, u = _starts(1, z=5709.)
    r = target_source(KEY, vacuum_geom, [0., 0., 0.], x, u, 105., 0., 11, msc=False, max_steps=1)
    assert int(r["state"]["fate"][0]) == C.PLANE
    assert not bool(r["capped"][0])


def test_slab_birth_totals_across_crossings_and_overflow(slab_case):
    g, x, u, pdg, r = slab_case
    validate_complete(r)
    st, rec = r["state"], r["records"]
    valid = np.asarray(rec["valid"])
    assert set(rec) == {"energy", "position", "direction", "path", "time", "valid"}
    assert int(np.sum(r["count"])) > 10
    assert np.sum(np.asarray(st["cnt"])[:, 0] >= 2) >= 12
    np.testing.assert_array_equal(r["count"], st["nph"])
    np.testing.assert_array_equal(valid.sum(axis=1), r["count"])
    np.testing.assert_allclose(np.sum(rec["energy"], axis=1), st["eph"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(r["parent_index"], np.arange(len(x)))
    np.testing.assert_array_equal(r["pdg"], pdg)
    for k in ("energy", "position", "direction", "path", "time"):
        assert np.all(np.asarray(rec[k])[~valid] == 0), k
    np.testing.assert_allclose(np.linalg.norm(np.asarray(rec["direction"])[valid], axis=1), 1., atol=1e-10)
    assert np.all(np.asarray(rec["time"])[valid] >= 0.)
    assert np.all(np.asarray(rec["path"]) <= np.asarray(st["s"])[:, None] + 1e-9)
    T0 = np.hypot(105., FN.ME) - FN.ME
    np.testing.assert_allclose(st["T"] + st["eph"] + st["ede"] + st["cont"] + st["ean"],
                               T0 + st["nan"] * FN.ME, atol=1e-9)
    np.testing.assert_array_equal(r["diagnostics"]["annihilation_photons_omitted"], st["nan"])
    np.testing.assert_array_equal(r["diagnostics"]["delta_cascades_omitted"], st["nde"])


@pytest.mark.slow   # a second build (capacity is static); the overflow refusal itself is checked host-only below
def test_slab_birth_capacity_one_counts_overflow_and_fails_closed(slab_case):
    g, x, u, pdg, r = slab_case
    st, rec = r["state"], r["records"]
    small = target_source(KEY, g, [0., 0., 0.], x, u, 105., 0., pdg, capacity=1, msc=False, max_steps=5000)
    np.testing.assert_array_equal(small["count"], r["count"])
    np.testing.assert_array_equal(small["overflow"], np.maximum(np.asarray(r["count"]) - 1, 0))
    for name in st:
        np.testing.assert_allclose(small["state"][name], st[name], atol=1e-10)
    for name in rec:
        np.testing.assert_allclose(small["records"][name], rec[name][:, :1], atol=1e-10)
    with pytest.raises(ValueError, match="dropped births"):
        validate_complete(small)


def test_validate_complete_rejects_overflow_capped_and_loopers():
    ok = dict(overflow=np.zeros(3, np.int32), capped=np.zeros(3, bool), loopers=np.zeros(3, bool))
    assert validate_complete(ok) is ok
    for name, value in (("overflow", [0, 2, 0]), ("capped", [False, True, False]), ("loopers", [True, False, False])):
        with pytest.raises(ValueError, match="Incomplete birth source"):
            validate_complete(dict(ok, **{name: np.asarray(value)}))


def test_late_source_time_unchanged_no_time_selection(slab_case):
    g, x, u, pdg, reference = slab_case
    late = 1e12 + np.arange(len(x), dtype=np.float64) * 1695. + 250.125
    original = late.copy()
    r = target_source(KEY, g, [0., 0., 0.], x, u, 105., late, pdg, msc=False, max_steps=5000)
    np.testing.assert_array_equal(late, original)
    np.testing.assert_array_equal(r["t0"], original)
    assert r["t0"].dtype == np.float64
    for section in ("state", "records"):
        for name in reference[section]:
            np.testing.assert_array_equal(r[section][name], reference[section][name])


def test_float32_keeps_float64_t0(vacuum_geom):
    x, u = _starts(1, z=5709.)
    r = target_source(KEY, vacuum_geom, [0., 0., 0.], x, u, 105., 1e12 + 0.125, 11,
                      dtype=np.float32, msc=False, max_steps=1)
    assert r["t0"][0] == 1e12 + 0.125 and r["t0"].dtype == np.float64
    assert r["state"]["t"].dtype == np.float32
    assert 0. < float(r["state"]["t"][0]) < 0.01


@pytest.mark.slow
def test_caps_loopers_turn_limit_and_partial_slab_state(vacuum_geom):
    x, u = _starts(1, z=5700.)
    u[:] = [1., 0., 0.]
    capped = target_source(KEY, vacuum_geom, [0., 0., 1.], x, u, 0.1, 0., 11, max_steps=1, msc=False)
    assert bool(capped["capped"][0]) and not bool(capped["loopers"][0])
    assert int(capped["state"]["nadapt"][0]) == 1
    np.testing.assert_allclose(capped["state"]["s"], 0.05 * 0.1 / SJ.K_FIELD)
    looper = target_source(KEY, vacuum_geom, [0., 0., 0.], x, u, 105., 0., 11,
                           max_steps=10, s_max=1.5, msc=False)
    assert bool(looper["loopers"][0]) and not bool(looper["capped"][0])
    for r in (capped, looper):
        with pytest.raises(ValueError, match="Incomplete"):
            validate_complete(r)
    g = _geom()
    x, u = _starts(1, z=5800.)
    r = target_source(KEY, g, [0., 0., 0.], x, u, 105., 0., 11, max_steps=1, msc=False)
    st = r["state"]
    assert bool(r["capped"][0]) and int(st["nslab"][0]) == 1
    assert float(st["s"][0]) > 0. and float(st["t"][0]) > 0.
    assert float(st["x"][0, 2]) > x[0, 2]
    np.testing.assert_allclose(st["x"][0, 2] - x[0, 2], st["s"][0], atol=1e-9)
    assert int(st["cnt"][0, 0]) == 0


@pytest.mark.slow
def test_finished_lanes_and_no_initial_annihilation_births(vacuum_geom):
    x, u = _starts(4, z=5700.)
    r = target_source(KEY, vacuum_geom, [0., 0., 0.], x, u, [0.01, 105., 0.01, 105.],
                      1e12, [11, 11, -11, -11], msc=False)
    np.testing.assert_array_equal(r["state"]["fate"], [C.STOPPED, C.PLANE, C.STOPPED, C.PLANE])
    np.testing.assert_array_equal(r["state"]["nan"], [0, 0, 2, 0])
    assert not np.any(r["count"])
    for values in r["records"].values():
        assert not np.any(values)
    np.testing.assert_array_equal(np.asarray(r["state"]["x"])[[0, 2]], x[[0, 2]])
    assert np.all(np.asarray(r["state"]["t"])[[0, 2]] == 0.)


@pytest.mark.slow
def test_unroll_equivalence(slab_case):
    g, x, u, pdg, reference = slab_case
    r = target_source(KEY, g, [0., 0., 0.], x, u, 105., 0., pdg, msc=False, max_steps=5000, unroll=True)
    for name in ("count", "overflow", "capped", "loopers"):
        np.testing.assert_array_equal(r[name], reference[name])
    for section in ("state", "records"):
        for name in reference[section]:
            np.testing.assert_allclose(r[section][name], reference[section][name], rtol=1e-9, atol=1e-8)


def test_empty_input_explicitly_rejected(vacuum_geom):
    with pytest.raises(ValueError, match="nonempty"):
        target_source(KEY, vacuum_geom, [0., 0., 0.], np.empty((0, 3)), np.empty((0, 3)), [], [], [])


@pytest.mark.slow
@pytest.mark.parametrize("unroll", [False, True])
def test_partial_magnetic_msc_step_matches_current_core(unroll):
    # A literal current-core one-step reference, for both charges, also exercises
    # the partial-slab projection when max_steps ends before a crossing completes.
    import jax.numpy as jnp
    g = _geom()
    n = 32
    x, u = _starts(n, z=5800.)
    u[:] = [0.2, 0.1, np.sqrt(0.95)]
    B = np.array([0.05, 0.02, 1.0])
    pdg = np.tile([11, -11], n // 2)
    r = target_source(KEY, g, B, x, u, 105., 0., pdg, max_steps=1, unroll=unroll)
    specs = FN.layer_types(g)[0]
    T0 = 105. * (105. / (np.hypot(105., FN.ME) + FN.ME))
    for species, code in enumerate((11, -11)):
        ids = np.flatnonzero(pdg == code)
        m = len(ids)
        lt = SJ.lane_tables(specs, positron=code == -11, msc=True)
        before = SJ.lane_state(jnp.full(m, T0), jnp.asarray(u[ids]), jnp.full(m, 2.5),
                               jnp.full(m, 5.), positron=code == -11)
        step = jax.jit(lambda st: SJ.lane_step_photons(
            jax.random.fold_in(KEY, species), lt, jnp.zeros(m, jnp.int32), st,
            jnp.broadcast_to(jnp.asarray(B), (m, 3)), positron=code == -11, msc=True, unroll=unroll))
        after, emitted = step(before)
        assert np.all(np.asarray(r["capped"])[ids])
        for name in FN.TOTAL_FIELDS + ("T",):
            np.testing.assert_allclose(r["state"][name][ids], after[name], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(r["state"]["x"][ids], np.asarray(after["x"]) + x[ids] - [0., 0., 2.5],
                                   atol=1e-9)
        np.testing.assert_allclose(r["state"]["u"][ids], after["d"], atol=1e-11)
        np.testing.assert_allclose(r["state"]["s"][ids], after["path"], atol=1e-11)
        np.testing.assert_array_equal(r["count"][ids], emitted.valid)
        np.testing.assert_allclose(r["records"]["energy"][ids, 0], emitted.energy, atol=1e-11)
        valid = np.asarray(emitted.valid)
        np.testing.assert_allclose(r["records"]["position"][ids, 0][valid],
                                   np.asarray(emitted.position)[valid] + x[ids][valid] - [0., 0., 2.5], atol=1e-9)


@pytest.mark.slow
def test_step_time_uses_continuous_loss_not_end_step_recoil():
    from beamfastsim.flash_navigator_jax import _slab_elapsed, _beta
    before = dict(T=np.array([100., 1.]), cont=np.zeros(2), path=np.array([2., 3.]))
    after = dict(T=np.array([.001, .001]), cont=np.array([.1, .2]), path=np.array([3., 5.]))
    got = np.asarray(_slab_elapsed(before, after))
    expected = np.array([1., 2.]) / (C.C_MM_NS * .5 *
               (np.asarray(_beta(before["T"])) + np.asarray(_beta(np.array([99.9, .8])))))
    np.testing.assert_allclose(got, expected)
    after["T"][:] = 0.
    np.testing.assert_array_equal(_slab_elapsed(before, after), got)


@pytest.mark.slow
def test_initial_containing_foil_unequal_thickness_and_radial_exclusion():
    # The nearest centre is foil 0, but these axial coordinates lie in foil 1.
    g = _geom(foil_z=(5800., 5804.), ht=np.array([.05, 3.]), plane=5810.)
    x, u = _starts(4, z=5801.5)
    x[2, 0] = AXIS + 20.  # hole, not material
    x[3, 0] = AXIS + 76.  # outside the rim
    u[1, 2] = -1.
    r = target_source(KEY, g, [0., 0., 0.], x, u, 105., 0., [11, -11, 11, -11],
                      h=.5, max_steps=1, msc=False, capacity=1)
    np.testing.assert_array_equal(r["state"]["nslab"], [1, 1, 0, 0])
    # The backward lane may already exit its nearby face and take a vacuum step.
    np.testing.assert_array_equal(np.asarray(r["state"]["nvac"])[[0, 2, 3]], [0, 1, 1])
    assert r["state"]["x"][0, 2] > x[0, 2]
    assert r["state"]["x"][1, 2] < x[1, 2]


def test_absent_tsda_nominal_plane_does_not_reject_start():
    from dataclasses import replace
    from beamfastsim.flash_target_jax import _geometry
    g = _geom()
    x = np.array([[AXIS + 240., 0., g.beam.tsda_z]])
    _geometry(replace(g, beam=replace(g.beam, tsda_hl=0.)), x, .5, np.dtype(np.float64))
    with pytest.raises(ValueError, match="absorbers"):
        _geometry(g, x, .5, np.dtype(np.float64))
