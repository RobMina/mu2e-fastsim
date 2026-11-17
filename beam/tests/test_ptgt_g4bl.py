"""G4beamline -> TS-entrance adapter on synthetic virtual-detector rows and synthetic tsdevice outputs."""
import json

import numpy as np
import pytest

from beamfastsim import ptgt_g4bl as pg

Z_G4BL = 3884.7   # deck Coll_01_DetIn centre (Mu2e -4044.3)


def rows(**over):
    base = dict(EventID=[1, 1, 1, 1, 2, 2, 3], TrackID=[5, 5, 6, 7, 5, 8, 2], ParentID=[1] * 7,
                PDGid=[-211, -211, 13, 2212, 211, -13, 22],
                x=[10.0, 11, -20, 0, 5, 1, 0], y=[0.0, 1, 2, 3, 4, 5, 6], z=[Z_G4BL] * 7,
                Px=[1.0, 2, 3, 4, 5, 6, 7], Py=[0.0] * 7, Pz=[80.0, 90, 50, 900, -40, 60, 10],
                t=[12.0, 9.0, 20, 5, 30, 40, 1])
    base.update(over)
    return {k: np.asarray(v, dtype=np.int64 if k in ("EventID", "TrackID", "ParentID", "PDGid") else float)
            for k, v in base.items()}


def test_first_inward_selects_earliest_forward_mu_pi_per_track():
    r = rows()
    i = pg.first_inward(r)
    # track (1,5): two crossings, earliest t=9 wins; proton, backward pi+ and photon dropped
    assert list(i) == [1, 2, 5]
    assert pg.first_inward(rows(PDGid=[2212] * 7)).size == 0


def test_charge_and_mass_conventions():
    assert list(pg.charge([13, -13, 211, -211])) == [-1, 1, 1, -1]
    assert pg.mass([211])[0] == pytest.approx(139.57039)


def test_frame_puts_deck_collimator_face_on_the_ts_axis():
    x = pg.to_mu2e([[0.0, 0.0, Z_G4BL]])
    assert x[0] == pytest.approx([3904.0, 0.0, -4044.3])
    assert pg.to_mu2e([[0.0, 0.0, 1764.5]])[0, 2] == pytest.approx(-6164.5)   # production target centre


def test_log_events_and_sequencer(tmp_path):
    log = tmp_path / "log.o.ptg5k01R0000g4blf01c22.17bc967.00000011-1791004409.log"
    log.write_text("Event 1 ...\nRun complete  5000 Events  2757.3 seconds\n")
    assert pg.log_events(log) == 5000
    assert pg.sequencer(log) == 11
    assert pg.sequencer("nts.o.d.c.00000011.root") == 11
    bad = tmp_path / "log.o.d.c.00000012-1.log"
    bad.write_text("G4Exception\n")
    assert pg.log_events(bad) is None
    with pytest.raises(ValueError):
        pg.sequencer("nts.o.d.c.root")


def fake_reader(tables):
    return lambda path: tables[pg.sequencer(path)]


def test_build_translate_keeps_state_and_counts_board_objective():
    tables = {0: rows(), 1: rows(EventID=[9] * 7, TrackID=list(range(1, 8)))}
    files = ["nts.o.d.c.00000000.root", "nts.o.d.c.00000001.root"]
    ref, meta = pg.build(files, -4004.0, "translate", point=3, events=[100, 50], reader=fake_reader(tables))
    assert ref["x"][:, 2] == pytest.approx(np.full(len(ref["t"]), -4004.0))
    assert ref["x"][0, :2] == pytest.approx([3904.0 + 11, 1.0])
    assert ref["t"][0] == 9.0
    assert ref["id"][0].tolist() == [3, 0, 1, 5]
    assert ref["shard"].tolist() == [0, 0, 0] + [1] * 4
    assert ref["parent_shard_event_counts"].tolist() == [100, 50]
    assert meta["source_plane_z"] == pytest.approx(-4044.3)
    # mu- and pi- only: file 0 has pi-(1,5) and mu-(1,6); file 1 keeps pi-, mu-, pi+... per track
    sel1 = sum(1 for p in tables[1]["PDGid"][pg.first_inward(tables[1])] if p in (13, -211))
    assert meta["board_selected"] == 2 + sel1
    assert meta["board_objective"] == pytest.approx((2 + sel1) / 150)


def test_build_rejects_two_planes_and_missing_pot():
    z = [Z_G4BL] * 6 + [Z_G4BL + 50]
    bad = rows(z=z, PDGid=[13] * 7, Pz=[10.0] * 7)
    with pytest.raises(ValueError, match="one plane"):
        pg.build(["nts.o.d.c.00000000.root"], -4004.0, events=[1], reader=fake_reader({0: bad}))
    with pytest.raises(ValueError, match="POT"):
        pg.build(["nts.o.d.c.00000000.root"], -4004.0, events=None, reader=fake_reader({0: rows()}))


def uniform(bz):
    return lambda x: np.tile([0.0, 0.0, bz], (len(x), 1))


@pytest.mark.parametrize("pdg", [13, -211])
def test_helix_matches_analytic_uniform_field(pdg):
    b, p_t, p_z, dz = 2.5, 30.0, 90.0, 40.3
    x0 = np.array([[3904.0 + 50, 0.0, -4044.3]])
    p0 = np.array([[p_t, 0.0, p_z]])
    x, p, t, ret = pg.helix_to_plane(x0, p0, [10.0], [pdg], -4004.0, uniform(b), step_mm=0.5)
    assert not ret.any()
    q = pg.charge([pdg])[0]
    omega = -q * pg.K_FIELD * b / p_z           # turning angle per mm of z
    phi = omega * dz
    assert p[0, 0] == pytest.approx(p_t * np.cos(phi), abs=1e-6)
    assert p[0, 1] == pytest.approx(p_t * np.sin(phi), abs=1e-6)
    r = p_t / (pg.K_FIELD * b)
    assert x[0, 0] - x0[0, 0] == pytest.approx(r * np.sin(phi) / (-q), abs=1e-4)
    assert x[0, 2] == -4004.0
    ptot = np.hypot(p_t, p_z)
    e = np.hypot(ptot, pg.mass([pdg])[0])
    assert t[0] - 10.0 == pytest.approx(dz * ptot / p_z * e / (ptot * pg.C_MM_NS), rel=1e-9)
    assert np.linalg.norm(p[0]) == pytest.approx(ptot, rel=1e-9)


def test_helix_zero_field_is_a_straight_line_and_flags_returns():
    x, p, t, ret = pg.helix_to_plane([[0.0, 0.0, -10.0], [0, 0, -10]], [[3.0, 4.0, 12.0], [1.0, 0, 0.5]], [0.0, 0.0],
                                     [13, 13], 0.0, uniform(0.0))
    assert x[0] == pytest.approx([2.5, 10 / 3, 0.0])
    assert ret.tolist() == [False, True]
    with pytest.raises(ValueError):
        pg.helix_to_plane([[0.0, 0.0, 5.0]], [[0, 0, 10.0]], [0.0], [13], 0.0, uniform(0.0))


def test_ratio_error_matches_bootstrap_scale():
    rng = np.random.default_rng(1)
    pot = np.full(20, 5000.0)
    num = rng.poisson(8.0, 20).astype(float)
    r, e = pg.ratio_error(num, pot)
    assert r == pytest.approx(num.sum() / pot.sum())
    assert e == pytest.approx(np.std(num, ddof=1) * np.sqrt(20) / pot.sum(), rel=1e-12)


def test_score_averages_seeds_per_shard(tmp_path):
    ref_rows = dict(id=np.array([[0, 0, 1, 1], [0, 0, 1, 2], [0, 4, 2, 1]]), shard=np.array([0, 0, 4]),
                    parent_shard_ids=np.array([[0, 0], [0, 4]]), parent_shard_event_counts=np.array([100, 300]))
    ref = tmp_path / "ref.npz"
    pg.save(ref, {**ref_rows, "pdg": np.array([13, -211, 13])},
            dict(point=0, mode="translate", entrances=3, board_objective=0.01))
    outs = []
    for seed, fate in ((1, [1, 1, 0]), (2, [1, 0, 1])):
        path = tmp_path / f"o{seed}.npz"
        np.savez(path, id=ref_rows["id"], shard=ref_rows["shard"], ds_fate=np.array(fate), pdg_ds=np.array([13, 13, -13]),
                 meta=json.dumps(dict(seed=seed, dtype="float64")))
        outs.append(path)
    res = pg.score(ref, outs)
    assert res["stops_per_pot"] == pytest.approx(2.0 / 400)
    assert res["mu_minus_stops_per_pot"] == pytest.approx(1.5 / 400)
    assert res["stops_per_seed"] == [2.0, 2.0]
    with pytest.raises(ValueError, match="distinct seeds"):
        pg.score(ref, [outs[0], outs[0]])


def test_root_reader_roundtrip(tmp_path):
    uproot = pytest.importorskip("uproot")
    r = rows()
    path = tmp_path / "nts.o.d.c.00000007.root"
    with uproot.recreate(path) as f:
        f.mktree(pg.TREE, {k: (v.astype(np.float32)) for k, v in r.items()})
    back = pg.read_plane(path)
    assert back["PDGid"].tolist() == r["PDGid"].tolist()
    assert back["Pz"] == pytest.approx(r["Pz"])
    ref, meta = pg.build([str(path)], -4004.0, events=[5000])
    assert meta["entrances"] == 3 and ref["shard"].tolist() == [7, 7, 7]


def test_score_selects_one_reference_of_a_multi_reference_run(tmp_path):
    refs, ids = [], []
    for k in range(2):
        i = np.array([[k, 0, 1, 1], [k, 0, 1, 2]])
        path = tmp_path / f"r{k}.npz"
        pg.save(path, dict(id=i, shard=np.array([0, 0]), pdg=np.array([13, 13]), parent_shard_ids=np.array([[k, 0]]),
                           parent_shard_event_counts=np.array([10 * (k + 1)])),
                dict(point=k, mode="translate", entrances=2, board_objective=None))
        refs.append(path)
        ids.append(i)
    out = tmp_path / "o.npz"
    np.savez(out, id=np.concatenate(ids), shard=np.zeros(4, int), batch=np.array([0, 0, 1, 1]),
             ds_fate=np.array([1, 0, 1, 1]), pdg_ds=np.full(4, 13), meta=json.dumps(dict(seed=5, dtype="float32")))
    assert pg.score(refs[0], [out], batch=0)["stops_per_pot"] == pytest.approx(0.1)
    assert pg.score(refs[1], [out], batch=1)["stops_per_pot"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="rows differ"):
        pg.score(refs[1], [out], batch=0)
    pg.main(["score", "--references", str(refs[0]), str(refs[1]), "--outputs", str(out), "--out",
             str(tmp_path / "s.jsonl")])
    lines = (tmp_path / "s.jsonl").read_text().splitlines()
    assert [json.loads(x)["point"] for x in lines] == [0, 1]


def test_covered_field_raises_outside_the_maps():
    class Bank:
        def evaluate(self, x):
            return np.zeros_like(x), x[:, 2] < 0
    f = pg.covered(Bank())
    assert f(np.array([[0.0, 0, -1]])).shape == (1, 3)
    with pytest.raises(ValueError, match="outside the field maps"):
        f(np.array([[0.0, 0, -1], [0, 0, 1]]))
