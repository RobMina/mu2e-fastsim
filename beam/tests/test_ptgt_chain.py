"""Production-target chain (host reference) and thin-target bank builder, on synthetic inputs."""
import json

import numpy as np
import pytest

from beamfastsim import ptgt_bank as PB
from beamfastsim import ptgt_chain as C
from beamfastsim import ptgt_device as D
from mu2efastsim import library as L

MATS = ("G4_W",)
EDGES = np.linspace(0.0, 4.0, 9)


def _uniform(bz):
    return lambda x: np.tile([0.0, 0.0, bz], (len(x), 1))


def _empty_bank(tmp_path, **const):
    shape = (len(D.SPECIES), len(MATS), EDGES.size - 1)
    vals = dict(sigma_inel=0.0, sigma_el=0.0, dedx=0.0, x0=1e12)
    vals.update(const)
    cells = {k: np.full(shape, v) for k, v in vals.items()}
    cells["el_cos_icdf"] = np.ones(shape + (D.N_EL,))
    ncell = int(np.prod(shape))
    trials = dict(trial_cell=np.arange(ncell), sec_offsets=np.zeros(ncell + 1, np.int64), edep_local=np.zeros(ncell))
    secs = dict(sec_pdg=np.zeros(0, np.int32), sec_p=np.zeros((0, 3)))
    fields = dict(name="beam.thin_target.synthetic", source=dict(samples=[], description="synthetic"),
                  generator=dict(name="test", release="none", physics_list="none", configuration="none"),
                  release=dict(repository="test", commit="none", entry_point="test"), seeds=dict(build=1),
                  config_hash=None, split=dict(training="all", held_out="none", rule="synthetic"),
                  unrecorded=dict(config_hash="synthetic"))
    path = tmp_path / "bank.npz"
    D.save_bank(path, MATS, EDGES, cells, trials, secs, fields)
    return D.Bank.load(path)


def _far_target():
    """A target far from every test lane (no interaction possible)."""
    return D.Target.rod(10.0, 1.0, 1.0, 1.0, origin=np.array([0.0, 0.0, 1e6]))


def test_pion_decay_momentum_and_energy():
    rng = np.random.default_rng(1)
    n = 2000
    p = np.full(n, 300.0)
    u = np.tile([0.0, 0.0, 1.0], (n, 1))
    d = C.decay(np.full(n, -211), np.zeros((n, 3)), u, p, np.zeros(n), rng)
    assert np.all(d["pdg"] == 13) and np.array_equal(d["parent"], np.arange(n))
    e_pi = np.hypot(300.0, C.MASS[211])
    e_mu = np.hypot(d["p"], C.MASS[13])
    # muon energy range for pi -> mu nu in flight
    pst = (C.MASS[211] ** 2 - C.MASS[13] ** 2) / (2 * C.MASS[211])
    g, b = e_pi / C.MASS[211], 300.0 / e_pi
    est = np.hypot(pst, C.MASS[13])
    assert e_mu.min() >= g * (est - b * pst) - 1e-6 and e_mu.max() <= g * (est + b * pst) + 1e-6
    # mean muon energy equals gamma * E* (isotropic decay)
    assert abs(e_mu.mean() - g * est) < 0.5


def test_kaon_three_body_conserves_momentum_at_rest():
    rng = np.random.default_rng(2)
    m = C._three_body(C.MASS[321], [C.MASS[211], C.MASS[211], C.MASS[211]], 500, rng)
    assert np.allclose(m.sum(axis=1), 0.0, atol=1e-6)
    e = np.sqrt((m ** 2).sum(-1) + C.MASS[211] ** 2).sum(1)
    assert np.allclose(e, C.MASS[321], rtol=1e-9)


def test_rk4_helix_radius_in_uniform_field():
    p, bz = 100.0, 4.0
    r = p / (C.K_FIELD * bz)
    x = np.zeros((1, 3)); u = np.array([[1.0, 0.0, 0.0]])
    for _ in range(200):
        x, u = C._rk4(x, u, np.array([p]), np.array([1.0]), np.array([1.0]), _uniform(bz))
    # the circle passes through the origin with its centre on the y axis at distance r
    cy = np.sign(np.mean(x[:, 1])) * r
    assert abs(np.hypot(x[0, 0], x[0, 1] - cy) - r) < 1e-6 * r


def test_plane_recording_and_envelope(tmp_path):
    bank = _empty_bank(tmp_path)
    env = C.Envelope(z_nodes=np.array([-1e4, 1e4]), r_nodes=np.array([100.0, 100.0]), z_min=-1000.0,
                     z_plane=500.0, r_plane=80.0, axis_x=0.0)
    rng = np.random.default_rng(3)
    # mu- straight along z (no field): recorded at z = 500 with its time; one at r = 90 crosses outside r_plane; one
    # at r = 120 is absorbed by the envelope; one going upstream is dropped.
    x = np.array([[0.0, 0.0, 0.0], [90.0, 0.0, 0.0], [120.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    u = np.array([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0], [0, 0, -1.0]])
    prim = dict(pdg=np.full(4, 13), x=x, u=u, p=np.full(4, 200.0), t=np.zeros(4), event=np.arange(4))
    out = C.run(bank, _far_target(), _uniform(0.0), env, prim, rng)
    r = out["record"]
    assert list(r["ev"]) == [0] and np.allclose(r["x"][0], [0, 0, 500.0])
    beta = 200.0 / np.hypot(200.0, C.MASS[13])
    assert abs(r["t"][0] - 500.0 / (beta * C.C_MM_NS)) < 1e-6
    assert out["counts"]["outside_plane"] == 1 and out["counts"]["absorbed"] == 1 and out["counts"]["upstream"] == 1


def test_reference_format_matches_adapter(tmp_path):
    rec = dict(ev=np.array([3, 1]), pdg=np.array([13, -211]), x=np.array([[3904.0, 1.0, -4044.45]] * 2),
               p=np.array([[1.0, 2.0, 50.0]] * 2), t=np.array([5.0, 6.0]))
    ref, meta = C.reference(rec, pot=1000, point=7, shard=2)
    assert np.all(ref["x"][:, 2] == -4004.0) and list(ref["id"][:, 2]) == [1, 3]
    assert meta["yield_count"] == 2 and meta["yield_per_pot"] == 0.002
    assert ref["parent_shard_event_counts"].tolist() == [1000]


def _write_inel(path, events):
    with open(path, "wb") as f:
        for b, ke, edep, esc, secs in events:
            np.array([b, len(secs)], "<i4").tofile(f)
            np.array([ke, edep, esc], "<f4").tofile(f)
            for pdg, px, py, pz in secs:
                np.array([pdg], "<i4").tofile(f)
                np.array([px, py, pz], "<f4").tofile(f)


def test_bank_builder_round_trip(tmp_path):
    d = tmp_path / "camp"
    d.mkdir()
    edges = [0.0, 1.0, 2.0]
    json.dump(dict(log10_ke_edges=edges, materials=list(MATS), physics_list="TEST", geant4="none"),
              open(d / "grid.json", "w"))
    for sp in PB.G4NAME.values():
        with open(d / f"xs_{sp}_G4_W_b0.txt", "w") as f:
            dedx = (0.0, 0.0) if sp == "neutron" else (1.5, 2.5)
            f.write(f"# test\n0 0.5 0.01 0.002 {dedx[0]} 3.5\n1 1.5 0.02 0.003 {dedx[1]} 3.5\n")
    _write_inel(d / "inel_proton_G4_W_b0_s1.bin",
                [(1, 30.0, 4.0, 0.5, [(211, 1.0, 2.0, 3.0), (2112, 0.0, 0.0, 300.0)]), (0, 5.0, 1.0, 0.0, [])])
    _write_inel(d / "inel_pi-_G4_W_b1_s1.bin", [(0, 40.0, 2.0, 0.0, [(-211, 0.0, 1.0, 9.0)])])
    np.array([[0, np.float32(5.0).view("<i4"), np.float32(0.9).view("<i4")]] * 4, "<i4").tofile(
        d / "el_proton_G4_W_b0_s1.bin")
    out = tmp_path / "bank.npz"
    PB.main([str(d), "--out", str(out), "--commit", "0123abcd"])
    b = D.Bank.load(out)
    m = b.lib.manifest                                 # a library-core file with a complete manifest
    assert L.manifest.complete(m) and not L.manifest.validate(m)
    assert m["name"] == "beam.thin_target.TEST" and m["release"]["commit"] == "0123abcd"
    assert m["seeds"]["source"] == [1] and m["source"]["samples"][0].startswith("el_proton")
    grid = json.load(open(d / "grid.json"))
    assert m["config_hash"] == L.hashing.digest(dict(grid=grid, xs_only=False))
    assert D.Bank.load(out, expect=dict(content_hash=m["content_hash"])).lib.content_hash == m["content_hash"]
    assert b["sigma_inel"].shape == (len(D.SPECIES), 1, 2)
    assert b["sigma_inel"][0, 0].tolist() == [0.01, 0.02]
    co = b["cell_offsets"]
    p_cell = (0 * 1 + 0) * 2 + 1                       # proton, W, bin 1
    pim_cell = (3 * 1 + 0) * 2 + 1                     # pi-, W, bin 0 + 1
    assert co[p_cell + 1] - co[p_cell] == 1 and co[pim_cell + 1] - co[pim_cell] == 1
    j = co[p_cell]
    off = b["sec_offsets"]
    assert b["sec_pdg"][off[j]:off[j + 1]].tolist() == [211, 2112]
    assert np.allclose(b["el_cos_icdf"][0, 0, 0], 0.9) and np.allclose(b["el_cos_icdf"][0, 0, 1], 1.0)


def test_chain_cascade_conserves_bookkeeping(tmp_path):
    """Every proton interacts once at the target centre into one forward pi- that decays or reaches the plane."""
    shape = (len(D.SPECIES), len(MATS), EDGES.size - 1)
    cells = {k: np.zeros(shape) for k in D.CELL_ARRAYS}
    cells["x0"][:] = 1e12
    cells["sigma_inel"][0] = 1e3                         # protons interact immediately
    cells["el_cos_icdf"] = np.ones(shape + (D.N_EL,))
    ncell = int(np.prod(shape))
    tc = np.arange(ncell)
    secs_per = np.array([1 if (c // (len(MATS) * (EDGES.size - 1))) == 0 else 0 for c in tc])
    trials = dict(trial_cell=tc, sec_offsets=np.concatenate([[0], np.cumsum(secs_per)]), edep_local=np.full(ncell, 2.0))
    n_sec = int(secs_per.sum())
    secs = dict(sec_pdg=np.full(n_sec, -211, np.int32), sec_p=np.tile([0.0, 0.0, 200.0], (n_sec, 1)))
    fields = dict(name="beam.thin_target.synthetic", source=dict(samples=[], description="synthetic"),
                  generator=dict(name="test", release="none", physics_list="none", configuration="none"),
                  release=dict(repository="test", commit="none", entry_point="test"), seeds=dict(build=1),
                  config_hash=None, split=dict(training="all", held_out="none", rule="synthetic"),
                  unrecorded=dict(config_hash="synthetic"))
    D.save_bank(tmp_path / "b.npz", MATS, EDGES, cells, trials, secs, fields)
    bank = D.Bank.load(tmp_path / "b.npz")
    tgt = D.Target.rod(100.0, 10.0, 10.0, 10.0)               # wide: the 1 mm beam never misses it
    env = C.Envelope(z_nodes=np.array([-1e4, 1e4]), r_nodes=np.array([1e4, 1e4]), z_min=-500.0, z_plane=-2000.0,
                     r_plane=1e4, axis_x=0.0)
    rng = np.random.default_rng(5)
    prim = C.primaries(200, tgt, rng)
    out = C.run(bank, tgt, _uniform(0.0), env, prim, rng)
    c = out["counts"]
    assert c["inelastic"] == 200
    # the pi- go along the proton (-z): they reach the upstream plane at -500 (dropped) or decay before it
    assert c["upstream"] + c["decays"] >= 200
    assert np.isclose(out["edep"].sum(), 2.0 * 200, rtol=0.05)


def test_decay_without_charged_daughters_returns_integer_parents():
    rng = np.random.default_rng(9)
    d = C.decay(np.array([2212]), np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), np.array([100.0]), np.zeros(1), rng)
    assert d["parent"].dtype.kind == "i" and d["pdg"].dtype.kind == "i" and d["x"].shape == (0, 3)


def test_bank_builder_refuses_zero_stopping_power(tmp_path):
    d = tmp_path / "camp"
    d.mkdir()
    json.dump(dict(log10_ke_edges=[0.0, 1.0], materials=list(MATS)), open(d / "grid.json", "w"))
    for sp in PB.G4NAME.values():
        with open(d / f"xs_{sp}_G4_W_b0.txt", "w") as f:
            f.write("0 0.5 0.01 0.002 0.0 3.5\n")
    _write_inel(d / "inel_proton_G4_W_b0_s1.bin", [(0, 5.0, 1.0, 0.0, [])])
    with pytest.raises(ValueError, match="stopping power"):
        PB.build(str(d))


def test_slab_cross_absorption_loss_and_scattering(tmp_path):
    """Constant tables: absorbed fraction 1 - exp(-sigma path), exact loss dedx * path, projected rms angle = Highland
    with the geometric-mean p beta; the path grows as 1 / |u_z|."""
    sb = _empty_bank(tmp_path, sigma_inel=0.01, dedx=1.0, x0=20.0)
    rng = np.random.default_rng(21)
    n, t, p0 = 20000, 10.0, 300.0
    m = C.MASS[211]
    for uz in (1.0, 0.5):
        u = np.tile([np.sqrt(1 - uz * uz), 0.0, uz], (n, 1))
        p, un, ab = C.slab_cross(sb, 0, np.full(n, -211), np.full(n, p0), u, np.full(n, t), rng)
        path = t / uz
        frac = -np.expm1(-0.01 * path)
        assert abs(ab.mean() - frac) < 4 * np.sqrt(frac * (1 - frac) / n), (uz, ab.mean(), frac)
        ke1 = np.hypot(p0, m) - m - 1.0 * path
        assert np.allclose(p, np.sqrt((ke1 + m) ** 2 - m ** 2))
        if uz == 1.0:
            b0, b1 = p0 / np.hypot(p0, m), p[0] / np.hypot(p[0], m)
            pb, bet = np.sqrt(p0 * b0 * p[0] * b1), np.sqrt(b0 * b1)
            th0 = 13.6 / pb * np.sqrt(path / 20.0) * (1 + 0.038 * np.log(path / (20.0 * bet ** 2)))
            assert abs(np.std(un[:, 0]) / th0 - 1) < 0.05, (np.std(un[:, 0]), th0)


def test_chain_slab_ring_acts_inside_its_radii_only(tmp_path):
    """mu- along z: the lane at r = 50 crosses the ring (40-60 mm) and is absorbed (huge sigma) or slowed (no sigma);
    the lanes at r = 0 and r = 90 pass untouched."""
    bank = _empty_bank(tmp_path)
    env = C.Envelope(z_nodes=np.array([-1e4, 1e4]), r_nodes=np.array([200.0, 200.0]), z_min=-1000.0, z_plane=500.0,
                     r_plane=150.0, axis_x=0.0, slabs=((250.0, 40.0, 60.0, 10.0),))
    x = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [90.0, 0.0, 0.0]])
    prim = dict(pdg=np.full(3, 13), x=x, u=np.tile([0, 0, 1.0], (3, 1)), p=np.full(3, 200.0), t=np.zeros(3),
                event=np.arange(3))
    d1 = tmp_path / "a"; d1.mkdir()
    absorber = _empty_bank(d1, sigma_inel=1e3, dedx=1.0, x0=1e12)
    out = C.run(bank, _far_target(), _uniform(0.0), env, prim, np.random.default_rng(4), slab=(absorber, 0))
    assert sorted(out["record"]["ev"].tolist()) == [0, 2]
    assert out["counts"]["slab_crossed"] == 1 and out["counts"]["slab_absorbed"] == 1
    d2 = tmp_path / "b"; d2.mkdir()
    lossy = _empty_bank(d2, sigma_inel=0.0, dedx=2.0, x0=1e12)
    out = C.run(bank, _far_target(), _uniform(0.0), env, prim, np.random.default_rng(4), slab=(lossy, 0))
    r = out["record"]
    pz = dict(zip(r["ev"].tolist(), np.linalg.norm(r["p"], axis=1)))
    m = C.MASS[13]
    ke1 = np.hypot(200.0, m) - m - 20.0
    assert np.isclose(pz[1], np.sqrt((ke1 + m) ** 2 - m ** 2)) and np.isclose(pz[0], 200.0) and np.isclose(pz[2], 200.0)


def test_bank_builder_xs_only(tmp_path):
    d = tmp_path / "camp"
    d.mkdir()
    json.dump(dict(log10_ke_edges=[0.0, 1.0, 2.0], materials=["G4_STAINLESS-STEEL"]), open(d / "grid.json", "w"))
    for sp in PB.G4NAME.values():
        with open(d / f"xs_{sp}_G4_STAINLESS-STEEL_b0.txt", "w") as f:
            dedx = (0.0, 0.0) if sp == "neutron" else (1.5, 2.5)
            f.write(f"0 0.5 0.01 0.002 {dedx[0]} 17.6\n1 1.5 0.02 0.003 {dedx[1]} 17.6\n")
    with pytest.raises(ValueError, match="no inel files"):
        PB.build(str(d))
    out = tmp_path / "slab.npz"
    PB.main([str(d), "--out", str(out), "--xs-only", "--commit", "0123abcd"])
    b = D.Bank.load(out)
    m = b.lib.manifest
    assert L.manifest.complete(m) and m["name"].endswith(".xs") and m["seeds"]["source"] == []
    assert m["config_hash"] != L.hashing.digest(dict(grid=json.load(open(d / "grid.json")), xs_only=False))
    assert b.materials == ("G4_STAINLESS-STEEL",) and b["trial_cell"].size == 0
    assert b["sigma_inel"][0, 0].tolist() == [0.01, 0.02] and np.allclose(b["x0"], 17.6)
