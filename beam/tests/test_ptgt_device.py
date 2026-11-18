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
    assert np.allclose(t.to_mu2e(t.to_local(x)), x)


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
