import numpy as np
import pytest

from beamfastsim import tstransport as ts, tsphysics as phys
from mu2efastsim import materials


class Field:
    def __init__(self, value=(0, 0, 0), valid=True):
        self.value, self.valid = np.asarray(value, float), valid

    def evaluate(self, x):
        return np.broadcast_to(self.value, x.shape), np.full(len(x), self.valid)

    def __call__(self, x):
        return self.evaluate(x)[0]


class Geometry:
    z_start, z_end = -1.0, 100.0

    def __init__(self, material=None):
        self.materials = {} if material is None else {material.name: material}

    def inside(self, x):
        return (x[:, 2] >= self.z_start - 1e-8) & (x[:, 2] <= self.z_end + 1e-8) & (np.abs(x[:, 1]) < 1000)

    def columns(self, a, b):
        return {k: np.linalg.norm(b - a, axis=1) for k in self.materials}


def source(pdg=13, n=100, p=1000):
    return dict(x=np.tile([-100, 0, 0.0], (n, 1)), p=np.tile([0, 0, float(p)], (n, 1)),
                t=np.zeros(n), pdg=np.full(n, pdg))


def test_straight_flight_time_and_surface():
    out = ts.transport(source(), Field(), Geometry(), seed=42, step_mm=7)
    reached = out["fate"] == ts.DS
    assert reached.sum() >= 99
    np.testing.assert_allclose(out["x"][reached], np.tile([-100, 0, 100], (reached.sum(), 1)))
    np.testing.assert_allclose(out["path"][reached], 100)
    np.testing.assert_allclose(out["t"][reached], 100 * np.hypot(1000, phys.MMU) / (1000 * phys.C_MM_NS))


def test_charge_and_pion_decay():
    np.testing.assert_array_equal(ts.charge(np.array([13, -13, 211, -211])), [-1, 1, 1, -1])
    geom = Geometry()
    geom.z_end = 20000
    out = ts.transport(source(211, n=1000, p=100), Field(), geom, step_mm=100, max_path_mm=22000)
    probability = 1 - np.exp(-geom.z_end * phys.MPI / (100 * phys.C_MM_NS * phys.TAU_PI_NS))
    assert abs(out["pion_decay"].sum() - 1000 * probability) < 4 * np.sqrt(1000 * probability * (1 - probability))
    np.testing.assert_array_equal(out["pdg"][out["pion_decay"]], -13)
    assert np.all(out["fate"] != ts.LIVE)


def test_missing_field_and_path_cap():
    out = ts.transport(source(n=2), Field(valid=False), Geometry())
    np.testing.assert_array_equal(out["fate"], ts.FIELD_MISSING)
    out = ts.transport(source(n=2), Field(), Geometry(), max_path_mm=20)
    np.testing.assert_array_equal(out["fate"], ts.PATH_LIMIT)
    np.testing.assert_allclose(out["path"], 20)


def test_rangeout_and_inputs_unchanged():
    s = source(n=10, p=20)
    out = ts.transport(s, Field(), Geometry(materials.ALUMINIUM), material_step_mm=0.1)
    np.testing.assert_array_equal(out["fate"], ts.RANGEOUT)
    assert np.all(out["material_path"] < 3)
    np.testing.assert_array_equal(s["p"][:, 2], 20)
    np.testing.assert_array_equal(out["p"], 0)


def test_seed_replay_and_validation():
    s = source(n=10)
    a = ts.transport(s, Field(), Geometry())
    b = ts.transport(s, Field(), Geometry())
    for key in a:
        if key != "seconds":
            np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(ValueError):
        ts.transport(s, Field(), Geometry(), step_mm=0)
    with pytest.raises(ValueError):
        ts.transport(s, Field(), Geometry(), method="wrong")


def test_intermediate_field_gap_is_not_vacuum():
    class Gap(Field):
        def evaluate(self, x):
            ok = (x[:, 2] <= 2) | (x[:, 2] >= 3)
            return np.where(ok[:, None], np.array([0, 4, 0.0]), 0.0), ok
    out = ts.transport(source(n=1), Gap(), Geometry(), step_mm=5)
    assert out['fate'][0] == ts.FIELD_MISSING


def test_wall_exit_reentry_terminates():
    class Cylinder(Geometry):
        z_start, z_end = -10., 100.
        def inside(self, x):
            return np.hypot(x[:, 0]-3904, x[:, 1]) <= 250 + 1e-10
    s = source(n=1, p=29.9792458)
    s['x'][0] = [4153.95, 0, 0]
    s['p'][0] = 29.9792458 * np.array([.1, 0, np.sqrt(.99)])
    out = ts.transport(s, Field([0, -4, 0]), Cylinder(), step_mm=5, max_path_mm=5)
    assert out['fate'][0] == ts.WALL
    assert out['path'][0] < 2.5
    assert np.hypot(out['x'][0, 0]-3904, out['x'][0, 1]) <= 250 + 1e-8


def test_wall_before_downstream_cannot_be_ds():
    class Gap(Geometry):
        z_end = 5.
        def inside(self, x):
            return (x[:, 2] <= 1) | (x[:, 2] >= 3)
    out = ts.transport(source(n=1), Field(), Gap(), step_mm=5)
    assert out['fate'][0] == ts.WALL
    assert out['x'][0, 2] == pytest.approx(1, abs=1e-5)


def test_discarded_field_gap_after_wall_does_not_change_wall_fate():
    class Wall(Geometry):
        def inside(self, x):
            return x[:, 2] <= 1
    class GapField(Field):
        def evaluate(self, x):
            return np.zeros_like(x), (x[:, 2] <= 2) | (x[:, 2] >= 3)
    out = ts.transport(source(n=1), GapField(), Wall(), step_mm=5)
    assert out['fate'][0] == ts.WALL
    assert out['x'][0, 2] == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize('method', ['rk4', 'helix'])
def test_boundary_bisections_only_advance_crossing_rows(method):
    class CountingField(Field):
        def __init__(self):
            super().__init__()
            self.sizes = []
        def evaluate(self, x):
            self.sizes.append(len(x))
            return super().evaluate(x)
    class Wall(Geometry):
        def inside(self, x):
            return super().inside(x) & (x[:, 1] <= 1)
    s = source(n=20)
    s['x'][0, 2] = 99
    s['p'][1] = [0, 1000, 0]
    field = CountingField()
    out = ts.transport(s, field, Wall(), step_mm=5, max_path_mm=5, method=method)
    np.testing.assert_array_equal(out['fate'][:2], [ts.DS, ts.WALL])
    assert out['x'][0, 2] == Wall.z_end
    np.testing.assert_array_equal(out['fate'][2:], ts.PATH_LIMIT)
    assert field.sizes.count(20) < 25
    assert field.sizes.count(1) >= (3 if method == 'rk4' else 1) * (25 + 20)


@pytest.mark.parametrize('method', ['rk4', 'helix'])
def test_mixed_boundary_rows_preserve_independent_field_validity(method):
    class MixedField(Field):
        def evaluate(self, x):
            discarded = (x[:, 0] == -200) & (x[:, 1] > 2) & (x[:, 1] < 3)
            gap = (x[:, 0] == -300) & (x[:, 2] > 2) & (x[:, 2] < 3)
            return np.zeros_like(x), ~(discarded | gap | (x[:, 0] == -500))
    class Wall(Geometry):
        def inside(self, x):
            return super().inside(x) & (x[:, 1] <= 1)
    s = source(n=5)
    s['x'][:, 0] = -100 * np.arange(1, 6)
    s['x'][0, 2] = 99
    s['p'][1] = [0, 1000, 0]
    out = ts.transport(s, MixedField(), Wall(), step_mm=5, max_path_mm=5, method=method)
    np.testing.assert_array_equal(out['fate'], [ts.DS, ts.WALL, ts.FIELD_MISSING,
                                              ts.PATH_LIMIT, ts.FIELD_MISSING])
    assert out['x'][0, 2] == Wall.z_end
    assert out['x'][1, 1] == pytest.approx(1, abs=1e-5)


def test_downstream_closed_geometry_endpoint_does_not_round_into_wall():
    geom = Geometry()
    geom.z_end = 0.003562410123123
    s = source(n=1)
    out = ts.transport(s, Field(), geom, step_mm=5)
    assert out['fate'][0] == ts.DS
    assert out['x'][0, 2] == geom.z_end


@pytest.mark.parametrize('method', ['rk4', 'helix'])
def test_safety_skips_only_redundant_probes_exactly(method):
    class SafeField(Field):
        def __init__(self):
            super().__init__((0, .2, .5))
            self.max_field_t = np.linalg.norm(self.value) * (1 + 1e-12)
            self.queries = 0
        def evaluate(self, x):
            self.queries += len(x)
            return super().evaluate(x)
        def safety(self, x):
            return np.full(len(x), 1000.)
    class SafeGeometry(Geometry):
        def safety(self, x):
            return np.maximum(0, np.minimum(np.minimum(x[:, 2]-self.z_start,
                                self.z_end-x[:, 2]), 1000-np.abs(x[:, 1]))-1e-8)
    s = source(n=20)
    s['x'][:, 2] = np.linspace(0, 99, 20)
    old, new = SafeField(), SafeField()
    a = ts.transport(s, old, SafeGeometry(materials.ALUMINIUM), method=method, boundary_safety=False)
    b = ts.transport(s, new, SafeGeometry(materials.ALUMINIUM), method=method)
    for key in a:
        if key != 'seconds':
            np.testing.assert_array_equal(a[key], b[key])
    assert new.queries < old.queries * (0.5 if method == 'rk4' else 0.6)


def test_field_hole_disables_geometry_only_safety():
    class Gap(Field):
        max_field_t = 0.
        def evaluate(self, x):
            return np.zeros_like(x), (x[:, 2] <= 2) | (x[:, 2] >= 3)
        def safety(self, x):
            return np.maximum(0, np.maximum(2-x[:, 2], x[:, 2]-3))
    class SafeGeometry(Geometry):
        def safety(self, x):
            return np.full(len(x), 100.)
    out = ts.transport(source(n=1), Gap(), SafeGeometry(), step_mm=5)
    assert out['fate'][0] == ts.FIELD_MISSING


@pytest.mark.parametrize('method', ['rk4', 'helix'])
def test_safety_with_priority_transition_wall_and_field_gap(tmp_path, method):
    from beamfastsim.tsfield import FieldBank
    specs = [((-150, -10, 2), (100, 20, 1), (0, 20, 2)),
             ((-250, -10, -5), (200, 20, 35), (0, .1, 1)),
             ((-350, -10, -5), (100, 20, 7), (0, 0, 0)),
             ((-350, -10, 3), (100, 20, 27), (0, 0, 0))]
    paths = []
    for index, (lo, spacing, value) in enumerate(specs):
        path = tmp_path / f'map{index}.header'
        path.write_text(f'grid X0={lo[0]} Y0={lo[1]} Z0={lo[2]} nX=2 nY=2 nZ=2 '
                        f'dX={spacing[0]} dY={spacing[1]} dZ={spacing[2]}\ndata\n')
        with path.with_suffix('.bin').open('wb') as handle:
            handle.write(np.array([0xDEADBEEF], np.uint32).tobytes())
            handle.write(np.broadcast_to(np.array(value, float), (2, 2, 2, 3)).tobytes())
        paths.append(path)
    class LocalWall(Geometry):
        z_start, z_end = -5., 100.
        def inside(self, x):
            slab = np.abs(x[:, 0]+200) < 10
            return super().inside(x) & (~slab | (x[:, 1] <= 1))
        def safety(self, x):
            wall = np.maximum(np.abs(x[:, 0]+200)-10, 1-x[:, 1])
            return np.maximum(0, np.minimum(wall, x[:, 2]+5)-1e-8)
    s = source(n=3)
    s['x'][:, 0] = [-100, -200, -300]
    s['p'][1] = [0, 1000, 0]
    field = FieldBank(paths)
    assert field.max_field_t > 20
    a = ts.transport(s, field, LocalWall(), method=method, step_mm=5, max_path_mm=5, boundary_safety=False)
    b = ts.transport(s, field, LocalWall(), method=method, step_mm=5, max_path_mm=5)
    for key in a:
        if key != 'seconds':
            np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(b['fate'], [ts.PATH_LIMIT, ts.WALL, ts.FIELD_MISSING])


def test_material_shortening_only_recalculates_changed_columns():
    class OneMaterial(Geometry):
        def __init__(self):
            super().__init__(materials.ALUMINIUM)
            self.sizes = []
        def columns(self, a, b):
            self.sizes.append(len(a))
            return {k: np.where(a[:, 0] == -100, np.linalg.norm(b-a, axis=1), 0.) for k in self.materials}
    s = source(n=20)
    s['x'][:, 0] -= np.arange(20)
    geom = OneMaterial()
    out = ts.transport(s, Field(), geom, step_mm=5, max_path_mm=1)
    np.testing.assert_array_equal(out['fate'], ts.PATH_LIMIT)
    assert geom.sizes.count(20) == 1
    assert 1 in geom.sizes


@pytest.mark.parametrize('method', ['rk4', 'helix'])
@pytest.mark.parametrize('step', [2.5, 5., 20.])
@pytest.mark.parametrize('radial', [False, True])
def test_upstream_bracket_distinguishes_radial_exit(method, step, radial):
    class Entrance(Geometry):
        z_start, z_end = -4004., 100.
        def inside(self, x):
            return (x[:, 2] >= self.z_start) & (np.abs(x[:, 1]) <= 1.)
    s = source(n=1)
    s['x'][0] = [3904., 0., Entrance.z_start + (5e-7 if radial else 1.1)]
    s['p'][0] = [0., 1000., 0.] if radial else [0., 0., -1000.]
    out = ts.transport(s, Field(), Entrance(), step_mm=step, max_path_mm=25., method=method)
    assert out['fate'][0] == (ts.WALL if radial else ts.UPSTREAM)
    assert Entrance().inside(out['x']).all()


def assert_sampler_caps(monkeypatch, material_cap=.25, loss_cap=.05):
    from mu2efastsim import eloss
    calls = []
    original = phys.material_step
    def checked(rng, mat, pvec, pdg, length, history):
        pmag, mass = np.linalg.norm(pvec, axis=1), phys.mass_mev(pdg)
        kinetic = pmag**2 / (np.hypot(pmag, mass) + mass)
        mean = eloss.dedx_restricted(mat, pmag, 1.e9, mass=mass) * length
        assert np.all(length <= material_cap * (1 + 1e-10))
        assert np.all(mean <= loss_cap * kinetic * (1 + 1e-10))
        calls.append(len(length))
        return original(rng, mat, pvec, pdg, length, history)
    monkeypatch.setattr(phys, 'material_step', checked)
    return calls


@pytest.mark.parametrize('method', ['rk4', 'helix'])
@pytest.mark.parametrize('pdg', [13, -13, 211, -211])
@pytest.mark.parametrize('case', ['thin', 'near_exit', 'vacuum_gap', 'length'])
def test_sampler_receives_bounded_thin_absorber_columns(monkeypatch, method, pdg, case):
    from mu2efastsim import eloss
    momentum = 10. if case == 'near_exit' else (1000. if case == 'length' else 40.)
    mass = phys.mass_mev(pdg)
    kinetic = momentum**2 / (np.hypot(momentum, mass) + mass)
    permitted = .05 * kinetic / eloss.dedx_restricted(materials.ALUMINIUM, momentum, 1.e9, mass=mass)
    thickness = .4 if case == 'length' else 1.3 * permitted
    class Slab(Geometry):
        z_start = -10.
        def columns(self, a, b):
            dz = b[:, 2] - a[:, 2]
            lo, hi = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
            overlap = np.maximum(0., np.minimum(hi, thickness) - np.maximum(lo, 0.))
            length = np.linalg.norm(b-a, axis=1)
            return {materials.ALUMINIUM.name: np.divide(overlap * length, np.abs(dz),
                    out=np.zeros(len(a)), where=dz != 0)}
    s = source(pdg=pdg, n=1, p=momentum)
    s['x'][0, 2] = thickness - 3 * permitted if case == 'near_exit' else (-1. if case == 'vacuum_gap' else 0.)
    calls = assert_sampler_caps(monkeypatch)
    out = ts.transport(s, Field(), Slab(materials.ALUMINIUM), step_mm=2.5,
                       max_path_mm=2.5, method=method)
    assert calls and out['fate'][0] != ts.LIVE


@pytest.mark.parametrize('momentum,near_exit', [(40., False), (10., True)])
def test_saved_run1a_absorber_sampler_bound(monkeypatch, momentum, near_exit):
    import os
    from beamfastsim.tsgeometry import Geometry as SavedGeometry
    from mu2efastsim import eloss
    path = os.environ.get('TS1_RUN1A_GDML')
    if not path:
        pytest.skip('opt-in saved Run1A GDML')
    geom = SavedGeometry(path)
    kinetic = momentum**2 / (np.hypot(momentum, phys.MMU) + phys.MMU)
    permitted = .05 * kinetic / eloss.dedx_restricted(geom.materials['Ti6Al4V'], momentum, 1.e9, mass=phys.MMU)
    s = source(n=1, p=momentum)
    s['x'][0] = [-.075 + 3 * permitted if near_exit else .075, 0., 0.]
    s['p'][0] = [-momentum, 0., 0.]
    calls = assert_sampler_caps(monkeypatch)
    ts.transport(s, Field(), geom, step_mm=2.5, max_path_mm=2.5)
    assert calls


@pytest.mark.parametrize('method', ['rk4', 'helix'])
@pytest.mark.parametrize('boundary', ['ds', 'wall'])
@pytest.mark.parametrize('cap', ['loss', 'length'])
def test_curved_boundary_rechecks_final_sampler_columns(monkeypatch, method, boundary, cap):
    from mu2efastsim import eloss, ehelix
    momentum = 40. if cap == 'loss' else 50.
    field = Field([0., 4., 0.])
    s = source(n=1, p=momentum)
    end, _ = ehelix.helix(s['x'], s['p']/momentum, np.array([momentum]),
                          np.array([-1]), field(s['x']), np.array([2.5]))
    kinetic = momentum**2 / (np.hypot(momentum, phys.MMU) + phys.MMU)
    column = .04 * kinetic / eloss.dedx_restricted(materials.ALUMINIUM, momentum, 1.e9, mass=phys.MMU)
    if cap == 'length':
        column = .21
    left = -99.98
    right = left + column * (end[0, 0]+100) / np.linalg.norm(end[0]-s['x'][0])
    class CurvedSlab(Geometry):
        z_end = 1.5 if boundary == 'ds' else 100.
        def inside(self, x):
            return (x[:, 2] >= -1.) & (x[:, 2] <= 1.5+1e-10)
        def columns(self, a, b):
            dx = b[:, 0]-a[:, 0]
            lo, hi = np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 0], b[:, 0])
            overlap = np.maximum(0., np.minimum(hi, right)-np.maximum(lo, left))
            return {materials.ALUMINIUM.name: np.divide(overlap*np.linalg.norm(b-a, axis=1),
                    np.abs(dx), out=np.zeros(len(a)), where=dx != 0)}
    loss_cap = .2 if cap == 'length' else .05
    s = source(n=2, p=momentum)
    s['x'][1] = [-200., 0., 1.4]
    calls = assert_sampler_caps(monkeypatch, loss_cap=loss_cap)
    out = ts.transport(s, field, CurvedSlab(materials.ALUMINIUM), step_mm=2.5,
                       max_path_mm=2.5, loss_fraction=loss_cap, method=method)
    assert calls
    np.testing.assert_array_equal(out['fate'], ts.DS if boundary == 'ds' else ts.WALL)
    np.testing.assert_allclose(out['x'][:, 2], 1.5, atol=2e-6, rtol=0)
    single = {key: value[1:] for key, value in s.items()}
    control = ts.transport(single, field, CurvedSlab(materials.ALUMINIUM), step_mm=2.5,
                           max_path_mm=2.5, loss_fraction=loss_cap, method=method)
    for key in ('x', 'p', 't', 'fate', 'path', 'steps'):
        np.testing.assert_array_equal(out[key][1:], control[key])


@pytest.mark.parametrize('method', ['rk4', 'helix'])
@pytest.mark.parametrize('first', ['upstream', 'radial'])
def test_oblique_exit_order_within_wall_bracket(method, first):
    class Rim(Geometry):
        z_start, z_end = -4004., 100.
        def inside(self, x):
            return (x[:, 2] >= self.z_start) & (np.abs(x[:, 1]) <= 250.)
    s = source(n=1)
    offset = -1e-7 if first == 'upstream' else 1e-7
    s['x'][0] = [3904., 248.9 + offset, -4002.9]
    s['p'][0] = [0., 1000., -1000.]
    out = ts.transport(s, Field(), Rim(), step_mm=5., max_path_mm=10., method=method)
    assert out['fate'][0] == (ts.UPSTREAM if first == 'upstream' else ts.WALL)
