"""Host reference of the production-target model: bank format, target geometry, frames, transport (synthetic)."""
import numpy as np
import pytest

from beamfastsim import ptgt_device as P
from mu2efastsim import library as L

MATS = ("W", "Inconel718")
EDGES = np.linspace(0.0, 4.0, 9)          # 1 MeV .. 10 GeV


def _fields():
    return dict(name="beam.thin_target.synthetic", source=dict(samples=[], description="synthetic test bank"),
                generator=dict(name="test", release="none", physics_list="none", configuration="none"),
                release=dict(repository="test", commit="none", entry_point="test"), seeds=dict(build=1),
                config_hash=None, split=dict(training="all", held_out="none", rule="synthetic"),
                unrecorded=dict(config_hash="synthetic test bank"))


def _bank(tmp_path, sigma_inel=0.0, sigma_el=0.0, dedx=0.0, x0=1e12, secondaries=None, el_cos=1.0):
    """Every cell gets the same constants; each cell holds one interaction with `secondaries` [(pdg, px, py, pz)]."""
    shape = (len(P.SPECIES), len(MATS), EDGES.size - 1)
    cells = {k: np.full(shape, v) for k, v in
             dict(sigma_inel=sigma_inel, sigma_el=sigma_el, dedx=dedx, x0=x0).items()}
    cells["el_cos_icdf"] = np.full(shape + (P.N_EL,), el_cos)
    ncell = int(np.prod(shape))
    sec = secondaries or []
    trials = dict(trial_cell=np.arange(ncell), sec_offsets=np.arange(ncell + 1) * len(sec),
                  edep_local=np.zeros(ncell))
    secs = dict(sec_pdg=np.tile([s[0] for s in sec], ncell).astype(np.int32),
                sec_p=np.tile(np.array([s[1:] for s in sec], float).reshape(-1, 3), (ncell, 1)))
    path = tmp_path / "bank.npz"
    P.save_bank(path, MATS, EDGES, cells, trials, secs, _fields())
    return P.Bank.load(path)


def _p(pdg, ke):
    m = P.MASS[pdg]
    return np.sqrt((ke + m) ** 2 - m ** 2)


def test_bank_round_trip_through_library_core(tmp_path):
    b = _bank(tmp_path, sigma_inel=0.01, secondaries=[(-211, 0.0, 0.0, 100.0)])
    m = b.lib.manifest
    assert m["subsystem"] == "beam" and m["kind"] == "pool" and m["name"] == "beam.thin_target.synthetic"
    assert b.materials == MATS
    assert [a["name"] for a in m["axes"]] == ["species", "material", "log10_ke"]
    assert m["geometry"]["hash"] is None                       # no geometry in the bank
    reloaded = L.load(tmp_path / "bank.npz")
    assert reloaded.content_hash == m["content_hash"]


def test_bank_refuses_bad_shapes(tmp_path):
    shape = (len(P.SPECIES), len(MATS), EDGES.size - 1)
    cells = {k: np.zeros(shape) for k in P.CELL_ARRAYS}
    cells["el_cos_icdf"] = np.ones(shape + (P.N_EL,))
    cells["dedx"] = np.zeros(shape[:-1])
    trials = dict(trial_cell=np.zeros(0, int), sec_offsets=np.zeros(1, int), edep_local=np.zeros(0))
    secs = dict(sec_pdg=np.zeros(0, int), sec_p=np.zeros((0, 3)))
    with pytest.raises(ValueError, match="dedx"):
        P.save_bank(tmp_path / "b.npz", MATS, EDGES, cells, trials, secs, _fields())


def test_bank_refuses_energy_outside_domain(tmp_path):
    b = _bank(tmp_path)
    with pytest.raises(L.OutOfDomain):
        b.cell(np.array([0]), np.array([0]), np.array([2e4]))


def test_rod_locate_and_distances():
    t = P.Target.rod(160.0, 3.0, 3.0, 3.0)
    assert list(t.locate(np.array([[0, 0, 0], [0, 2.9, 79], [0, 3.1, 0], [0, 0, 81]]))) == [1, 1, -1, -1]
    x = np.array([[0.0, 0.0, -10.0], [0.0, 0.0, 10.0]])
    d = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    seg = t.locate(x)
    assert np.allclose(t.distance_out(x, d, seg), [10.0, 3.0])       # segment boundary at z=0; radial wall
    out = np.array([[0.0, 0.0, -100.0], [10.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
    dout = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(t.distance_in(out, dout), [20.0, 7.0, np.inf])


def test_clad_rod_locate_and_distances():
    t = P.Target.clad_rod(100.0, 3.0, 0.25, mat_core=0, mat_clad=1)
    pts = np.array([[1.0, 0, 0], [3.1, 0, 0], [0, 3.2, -49.0], [3.3, 0, 0], [0, 0, 51.0], [2.9, 0, 50.0]])
    assert list(t.locate(pts)) == [0, 1, 1, -1, -1, -1] and list(t.mat) == [0, 1]
    assert (t.z_lo, t.z_hi) == (-50.0, 50.0)
    x = np.array([[0.0, 0, 0], [3.1, 0, 0], [3.1, 0, 0], [3.1, 0, 0]])
    d = np.array([[1.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0], [0, 0, 1.0]])
    assert np.allclose(t.distance_out(x, d, t.locate(x)), [3.0, 0.15, 0.1, 50.0])   # core wall; sleeve out / in; end
    out = np.array([[5.0, 0, 0], [0.0, 0, 80.0]])
    dout = np.array([[-1.0, 0, 0], [0, 0, -1.0]])
    assert np.allclose(t.distance_in(out, dout), [1.75, 30.0])
    with pytest.raises(ValueError):
        P.Target(np.array([-1.0, 1.0]), np.zeros((1, 2)), np.ones((1, 2)), np.array([0]), z_ranges=np.array([[1.0, -1.0]]))


def _finned():
    """A rod (r 3) with two fins: +y (x -0.5..0.5, y 3..8) and +x (x 3..8, y -0.5..0.5), z -20..20, material 1."""
    return P.Target(np.array([-20.0, 20.0]), np.zeros((1, 2)), np.full((1, 2), 3.0), np.array([0, 1, 1]),
                    boxes=np.array([[-0.5, 0.5, 3.0, 8.0, -20.0, 20.0], [3.0, 8.0, -0.5, 0.5, -20.0, 20.0]]))


def test_boxes_locate_and_distances():
    t = _finned()
    assert t.n_cone == 1 and t.r_max == pytest.approx(np.hypot(8.0, 0.5)) and (t.z_lo, t.z_hi) == (-20.0, 20.0)
    pts = np.array([[0, 0, 0], [0, 5.0, 0], [5.0, 0, 0], [5.0, 5.0, 0], [0, 5.0, 25.0], [0.6, 5.0, 0]])
    assert list(t.locate(pts)) == [0, 1, 2, -1, -1, -1]
    x = np.array([[0, 5.0, 0], [0, 5.0, 0], [5.0, 0, 10.0], [0.0, 0, 0]])
    d = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
    assert np.allclose(t.distance_out(x, d, t.locate(x)), [3.0, 0.5, 10.0, 3.0])     # box faces; the core wall
    out = np.array([[0, 20.0, 0], [5.0, 5.0, 0], [20.0, 0, 0], [0, 5.0, 40.0], [5.0, 5.0, 0]])
    dout = np.array([[0, -1.0, 0], [-1.0, 0, 0], [-1.0, 0, 0], [0, 0, -1.0], [0, 0, 1.0]])
    assert np.allclose(t.distance_in(out, dout), [12.0, 4.5, 12.0, 20.0, np.inf])
    with pytest.raises(ValueError):
        P.Target(np.array([-1.0, 1.0]), np.zeros((1, 2)), np.ones((1, 2)), np.array([0, 0]),
                 boxes=np.array([[1.0, 0.0, 0, 1, 0, 1]]))
    with pytest.raises(ValueError):
        P.Target(np.array([-1.0, 1.0]), np.zeros((1, 2)), np.ones((1, 2)), np.array([0]),
                 boxes=np.array([[0.0, 1.0, 0, 1, 0, 1]]))       # mat must cover the box


def _hubbed():
    """A rod with a sphere on its side and a tilted spoke (cylinder) from the sphere outwards."""
    return P.Target(np.array([-20.0, 20.0]), np.zeros((1, 2)), np.full((1, 2), 3.0), np.array([0, 0, 1]),
                    spheres=np.array([[0.0, 4.0, 10.0, 2.0]]), cylinders=np.array([[0.0, 5.0, 10.0, 0.0, 25.0, 0.0, 0.5]]),
                    beam_z=15.0)


def test_spheres_and_cylinders_locate_and_distances():
    t = _hubbed()
    assert (t.n_cone, t.n_box, t.n_sphere) == (1, 0, 1) and t.z_beam == 15.0 and t.z_hi == 20.0
    assert t.r_max == pytest.approx(6.0)                           # the sphere; the spoke has its own near test
    pts = np.array([[0, 0, 0], [0, 4.0, 10.0], [0, 15.0, 5.0], [0.6, 15.0, 5.0], [0, 2.5, 10.0], [0, 30.0, -2.5]])
    assert list(t.locate(pts)) == [0, 1, 2, -1, 0, -1]             # the rod wins where it overlaps the sphere
    u = np.array([0.0, 20.0, -10.0]) / np.hypot(20.0, 10.0)        # the spoke axis
    x = np.array([[0, 4.0, 10.0], [0, 15.0, 5.0], [0, 15.0, 5.0], [0, 15.0, 5.0]])
    d = np.array([[1.0, 0, 0], [1.0, 0, 0], u, -u])
    assert np.allclose(t.distance_out(x, d, t.locate(x)), [2.0, 0.5, np.linalg.norm([0, 10.0, -5.0]),
                                                            np.linalg.norm([0, -10.0, 5.0])])
    out = np.array([[-10.0, 4.0, 10.0], [-10.0, 15.0, 5.0], [0, 15.0 + 20 * u[1], 5.0 + 20 * u[2]], [5.0, 15.0, 5.0]])
    dout = np.array([[1.0, 0, 0], [1.0, 0, 0], -u, [0, 0, 1.0]])
    assert np.allclose(t.distance_in(out, dout), [8.0, 9.5, 20.0 - np.linalg.norm([0, 10.0, -5.0]), np.inf])
    rng = np.random.default_rng(3)                                 # distance_in agrees with locate along random rays
    x0 = np.stack([rng.uniform(-8, 8, 400), rng.uniform(-5, 30, 400), rng.uniform(-25, 25, 400)], 1)
    x0 = x0[t.locate(x0) < 0]
    d0 = rng.normal(size=x0.shape); d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
    ts = np.linspace(0, 60, 24001)
    first = np.array([ts[np.argmax(t.locate(p + ts[:, None] * q) >= 0)] if np.any(t.locate(p + ts[:, None] * q) >= 0)
                      else np.inf for p, q in zip(x0, d0)])
    di = t.distance_in(x0, d0)
    hit = np.isfinite(first)
    agree = (np.isfinite(di) & (di < 60)) == hit                   # a grazing chord can slip between grid points
    assert hit.sum() > 20 and agree.mean() > 0.99
    assert np.allclose(di[hit & agree], first[hit & agree], atol=3e-3)
    with pytest.raises(ValueError):
        P.Target(np.array([-1.0, 1.0]), np.zeros((1, 2)), np.ones((1, 2)), np.array([0, 0]),
                 cylinders=np.array([[0.0, 0, 0, 0, 0, 0, 1.0]]))  # a == b


def test_contiguous_segments_locate_as_before():
    t = P.Target(np.array([-50.0, -10.0, 20.0, 50.0]), np.zeros((3, 2)), np.array([[2, 3], [3, 3], [3, 1.5]]),
                 np.array([0, 1, 0]))
    rng = np.random.default_rng(1)
    xl = np.stack([rng.uniform(-4, 4, 5000), rng.uniform(-4, 4, 5000), rng.uniform(-60, 60, 5000)], 1)
    xl[:50, 2] = np.repeat([-50.0, -10.0, 20.0, 50.0, 0.0], 10)          # on the planes
    z, r = xl[:, 2], np.hypot(xl[:, 0], xl[:, 1])                          # the reference: one segment per z slice
    i = np.clip(np.searchsorted(t.z_edges, z, side="right") - 1, 0, 2)
    rin, rout = t._radii(i, z)
    ref = np.where((z >= -50) & (z < 50) & (r >= rin) & (r < rout), i, -1)
    assert np.array_equal(t.locate(xl), ref)


def test_tapered_radius_distance():
    t = P.Target.rod(100.0, r_up=2.0, r_mid=4.0, r_dn=6.0)            # r_up at local z=-50, r_dn at +50
    x = np.array([[0.0, 0.0, -25.0], [0.0, 0.0, 25.0]])
    assert np.allclose(t.distance_out(x, np.array([[1.0, 0, 0]] * 2), t.locate(x)), [3.0, 5.0])


def test_frames_and_rotation():
    rng = np.random.default_rng(3)
    d = rng.normal(size=(50, 3))
    R = P.frame_to_lab(d)
    assert np.allclose(R[:, :, 2], d / np.linalg.norm(d, axis=1, keepdims=True))
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3))
    p = rng.normal(size=(50, 3)) * 100
    q = P.rotate_secondaries(p, d, rng.random(50) * 2 * np.pi)
    assert np.allclose(np.linalg.norm(q, axis=1), np.linalg.norm(p, axis=1))


def test_placement_round_trip():
    a = np.deg2rad(14.0)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    t = P.Target.rod(160.0, 3.1, 3.1, 3.1, origin=np.array([3904.0, 0.0, -6164.5]), rot=rot)
    x = np.array([[3900.0, 1.0, -6170.0]])
    assert np.allclose(t.to_global(t.to_local(x)), x)


def test_survival_through_rod_matches_attenuation(tmp_path):
    sigma, length, n = 0.01, 160.0, 20000
    b = _bank(tmp_path, sigma_inel=sigma)                                 # interactions with no secondaries
    t = P.Target.rod(length, 3.0, 3.0, 3.0)
    x = np.tile([[0.0, 0.0, -length / 2 + 1e-6]], (n, 1))
    p = np.tile([[0.0, 0.0, _p(2212, 8000.0)]], (n, 1))
    r = P.transport(b, t, np.full(n, 2212), x, p, np.random.default_rng(1))
    expect = n * np.exp(-sigma * length)
    assert abs(len(r["exit_pdg"]) - expect) < 5 * np.sqrt(expect)
    assert r["counts"]["inelastic"] == n - len(r["exit_pdg"])
    assert np.allclose(r["exit_x"][:, 2], length / 2, atol=1e-5)


def test_secondaries_are_pushed_and_leave(tmp_path):
    b = _bank(tmp_path, sigma_inel=0.05, secondaries=[(-211, 0.0, 0.0, 100.0)])
    t = P.Target.rod(10.0, 3.0, 3.0, 3.0)
    n = 2000
    x = np.tile([[0.0, 0.0, -5.0 + 1e-6]], (n, 1))
    p = np.tile([[0.0, 0.0, _p(2212, 8000.0)]], (n, 1))
    r = P.transport(b, t, np.full(n, 2212), x, p, np.random.default_rng(2))
    pions = int(np.sum(r["exit_pdg"] == -211))
    assert r["counts"]["inelastic"] >= pions > 0                           # a pion interaction yields a pion again
    assert np.all(np.isin(r["exit_pdg"], (2212, -211)))


def test_muon_loses_energy_without_interacting(tmp_path):
    shape = (len(P.SPECIES), len(MATS), EDGES.size - 1)
    cells = {k: np.full(shape, v) for k, v in dict(sigma_inel=1.0, sigma_el=0.0, dedx=0.1, x0=1e12).items()}
    cells["sigma_inel"][P.SPECIES.index(13)] = 0.0
    cells["el_cos_icdf"] = np.ones(shape + (P.N_EL,))
    ncell = int(np.prod(shape))
    P.save_bank(tmp_path / "m.npz", MATS, EDGES, cells,
                dict(trial_cell=np.arange(ncell), sec_offsets=np.zeros(ncell + 1, int), edep_local=np.zeros(ncell)),
                dict(sec_pdg=np.zeros(0, np.int32), sec_p=np.zeros((0, 3))), _fields())
    b = P.Bank.load(tmp_path / "m.npz")
    t = P.Target.rod(20.0, 3.0, 3.0, 3.0)
    x = np.array([[0.0, 0.0, -10.0 + 1e-6]])
    r = P.transport(b, t, np.array([13]), x, np.array([[0.0, 0.0, _p(13, 500.0)]]), np.random.default_rng(4))
    assert list(r["exit_pdg"]) == [13] and r["counts"]["inelastic"] == 0
    ke_out = np.hypot(np.linalg.norm(r["exit_p"][0]), P.MASS[13]) - P.MASS[13]
    assert np.isclose(500.0 - ke_out, 2.0, rtol=1e-6) and np.isclose(r["edep"].sum(), 2.0, rtol=1e-6)


def test_rotated_boxes_match_their_axis_aligned_twins():
    """A box turned by 90 degrees is the swapped axis-aligned box; a fin at 45 degrees holds points on its diagonal,
    and its distances are the axis-aligned ones measured in its own frame."""
    fin = [0.0, 8.0, -0.5, 0.5, -20.0, 20.0]
    rot = P.Target(np.array([-20.0, 20.0]), np.zeros((1, 2)), np.full((1, 2), 1.0), np.array([0, 1, 1]),
                   boxes=np.array([fin, fin]), box_phi=np.array([np.pi / 2, np.pi / 4]))
    twin = P.Target(np.array([-20.0, 20.0]), np.zeros((1, 2)), np.full((1, 2), 1.0), np.array([0, 1]),
                    boxes=np.array([[-0.5, 0.5, 0.0, 8.0, -20.0, 20.0]]))
    rng = np.random.default_rng(1)
    x = np.stack([rng.uniform(-3, 3, 4000), rng.uniform(-1, 9, 4000), rng.uniform(-25, 25, 4000)], 1)
    near45 = np.abs(x[:, 0] - x[:, 1]) / np.sqrt(2) < 0.6                      # points the 45-degree fin may hold
    assert np.array_equal(rot.locate(x[~near45]) >= 0, twin.locate(x[~near45]) >= 0)
    s = np.sqrt(0.5)
    pts = np.array([[5 * s, 5 * s, 0.0], [5 * s + 0.4 * s, 5 * s - 0.4 * s, 0.0], [5 * s + 0.6 * s, 5 * s - 0.6 * s, 0.0]])
    assert list(rot.locate(pts)) == [2, 2, -1]
    d = np.array([[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(rot.distance_out(pts[:1].repeat(3, 0), d, np.full(3, 2)), [3.0, 0.5, 20.0])
    assert np.allclose(rot.distance_in(np.array([[10 * s - 2 * s, 2 * s, 0.0]]), np.array([[-s, s, 0.0]])), [2.5])
    with pytest.raises(ValueError):
        P.Target(np.array([-1.0, 1.0]), np.zeros((1, 2)), np.ones((1, 2)), np.array([0, 0]),
                 boxes=np.array([fin]), box_phi=np.array([0.0, 1.0]))


def test_heat_edges_set_the_slices():
    t = P.Target(np.array([-10.0, 10.0]), np.zeros((1, 2)), np.ones((1, 2)), np.array([0]),
                 heat_edges=np.array([-10.0, -4.0, 3.0, 10.0]))
    e, n = P.slice_edges(t)
    assert n == 3 and e.size == P.EDEP_SLICE_MAX + 1 and np.allclose(e[:4], [-10, -4, 3, 10]) and np.all(np.diff(e) > 0)
    with pytest.raises(ValueError):
        P.slice_edges(P.Target(np.array([-1.0, 1.0]), np.zeros((1, 2)), np.ones((1, 2)), np.array([0]),
                               heat_edges=np.array([0.0, 0.0])))
