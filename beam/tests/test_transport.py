"""beamfastsim unit tests on synthetic inputs (no CVMFS): helix closure in a uniform field, range-out in a thick foil,
trilinear lookup, SimpleConfig parsing.
"""
import numpy as np

import jax
import jax.numpy as jnp

from beamfastsim import fieldmap, geometry, simpleconfig, transport


def _uniform_map(bz=1.0):
    n = np.array([20, 20, 200])
    lo = np.array([-4400.0, -500.0, 3500.0])
    B = np.zeros((*n, 3))
    B[..., 2] = bz
    return fieldmap.FieldMap(jnp.asarray(lo), jnp.asarray(np.full(3, 50.0)), jnp.asarray(n), jnp.asarray(B))


def _geom(nfoil=0, ht=0.05, z0=6000.0):
    z = z0 + np.arange(nfoil) * 20.0
    return geometry.BeamGeometry(-3904.0, z, np.zeros(nfoil), np.full(nfoil, 75.0), np.full(nfoil, ht),
                                 1e9, 1.0, 0.0, 0.0, 300.0, 0.5, 2e9, 2e9 + 1, 1e9, 3e9, 3e9 + 1)


def test_trilinear_exact_on_linear_field():
    fm = _uniform_map()
    B = np.asarray(fm.B).copy()
    gx = np.asarray(fm.lo)[0] + np.arange(B.shape[0]) * 50.0
    B[..., 0] = (gx / 1000.0)[:, None, None]
    fm = fm._replace(B=jnp.asarray(B))
    pos = jnp.asarray([[-4100.3, 10.0, 5000.0], [-3904.0, -20.0, 6000.0]])
    got = np.asarray(fieldmap.field(fm, pos))
    assert np.allclose(got[:, 0], np.asarray(pos)[:, 0] / 1000.0, atol=1e-12)
    assert np.allclose(got[:, 2], 1.0)


def test_helix_radius_and_z_advance():
    fm = _uniform_map(1.0)
    prm = transport.Params(ds=2.0, max_steps=400, msc=False, decay=False, z_up=-1e9, r_wall=1e9)
    run = transport.make_stepper(fm, _geom(nfoil=1, z0=1e6), prm)
    pt, pz = 30.0, 40.0
    x = np.array([[-3904.0 + 100.0, 0.0, 4000.0]])
    p3 = np.array([[0.0, pt, pz]])
    st = transport.initial_state(jax.random.PRNGKey(0), x, p3, np.zeros(1), np.array([-1.0]), prm)
    res, n = run(jax.random.PRNGKey(1), st)
    # after 400 steps of 2 mm the path is 800 mm; z advance = 800 pz/p, radius about the guiding centre pt/(0.2998 B)
    assert abs(float(res.x[0, 2]) - (4000.0 + 800.0 * pz / 50.0)) < 1e-3
    R = pt / (transport.K_LORENTZ * 1.0)
    # mu- (q=-1) in +Bz: guiding centre at x0 - R for initial momentum +y
    xc = x[0, 0] - R * np.sign(-1.0) * -1
    r_now = np.hypot(float(res.x[0, 0]) - xc, float(res.x[0, 1]))
    assert abs(r_now - R) < 1e-3, (r_now, R)
    assert abs(float(res.p[0]) - 50.0) < 1e-12


def test_range_out_in_thick_foil():
    fm = _uniform_map(0.0)
    prm = transport.Params(ds=1.0, max_steps=5000, msc=False, decay=False, z_up=-1e9, r_wall=1e9, n_slices=8)
    g = _geom(nfoil=1, ht=50.0)            # 10 cm of Al stops a 40 MeV/c muon (range ~ 1.5 mm Al)
    run = transport.make_stepper(fm, g, prm)
    x = np.array([[-3904.0, 0.0, 5900.0]] * 64)
    p3 = np.array([[0.0, 0.0, 40.0]] * 64)
    st = transport.initial_state(jax.random.PRNGKey(0), x, p3, np.zeros(64), -np.ones(64), prm)
    res, _ = run(jax.random.PRNGKey(2), st)
    assert np.all(np.asarray(res.fate) == 1)
    assert np.all(np.asarray(res.foil) == 0)


def test_simpleconfig(tmp_path):
    (tmp_path / "Offline").mkdir()
    (tmp_path / "Offline" / "inc.txt").write_text("double a.b = 1.5; // c\nvector<double> v = { 1, 2,\n 3 };\n")
    (tmp_path / "top.txt").write_text('#include "Offline/inc.txt"\ndouble a.b = 2.5;\nstring s = "X";\nbool t = true;\n')
    c = simpleconfig.read(str(tmp_path / "top.txt"), str(tmp_path))
    assert c["a.b"] == 2.5 and c["v"] == [1.0, 2.0, 3.0] and c["s"] == "X" and c["t"] is True


def test_transport_all_matches_single_pass_fates():
    fm = _uniform_map(1.0)
    prm = transport.Params(ds=2.0, max_steps=3000, msc=False, decay=False, z_up=-1e9, r_wall=1e9)
    g = _geom(nfoil=3, ht=0.5, z0=4200.0)
    rng = np.random.default_rng(3)
    n = 300
    x = np.stack([-3904.0 + rng.normal(0, 20, n), rng.normal(0, 20, n), np.full(n, 4000.0)], 1)
    p3 = np.stack([rng.normal(0, 10, n), rng.normal(0, 10, n), rng.uniform(5, 30, n)], 1)
    st = transport.initial_state(jax.random.PRNGKey(0), x, p3, np.zeros(n), -np.ones(n), prm)
    one, _ = transport.make_stepper(fm, g, prm)(jax.random.PRNGKey(1), st)
    many, ls = transport.transport_all(fm, g, prm, jax.random.PRNGKey(1), st, chunk=50, min_bucket=8)
    # deterministic physics (no msc/decay) except the loss sampling keys, so compare range-out-free observables
    assert ls < n * 3000
    assert np.all((np.asarray(one.fate) == 1) == (many.fate == 1)) or abs(np.mean(np.asarray(one.fate) == 1) - np.mean(many.fate == 1)) < 0.05


def test_kit_task_renders():
    from dataclasses import dataclass
    from beamfastsim import kit_task

    @dataclass(frozen=True)
    class R:
        cores: int; mem_gb: int; gpus: int; gpu_type: str; minutes: int

    @dataclass(frozen=True)
    class T:
        params: tuple; metrics: tuple; resources: object; body: str

    tasks, fnal = {}, set()
    kit_task.register(tasks, T, R, fnal)
    assert "beam_point" in tasks and "beam_point" in fnal
    b = tasks["beam_point"].body
    for p in kit_task.PARAMS:
        assert f"@p_{p}@" in b
    assert "@fnal_data@" in b and "@runtime_a@" in b
