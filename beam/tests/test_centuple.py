"""Fast CE EventNtuple chain (beamfastsim.centuple): the NumPy parts (helix to a plane, reco surfaces, LoopHelix
parameters, donor matching, plan bookkeeping, contract). The GPU stages and the PyROOT writer run in their own jobs
(record/trigger: the image on a GPU; write: the Mu2e environment of the stock release)."""
import json

import numpy as np
import pytest

from beamfastsim.centuple import assemble as A
from beamfastsim.centuple import contract as K
from beamfastsim.centuple import donor as D
from beamfastsim.centuple import record as R

# stock EventNtuple (AnalysisMDC2025 v02_02_02, CeEndpoint OnSpill): e- fit segments at TT_Mid, TT_Front, TT_Back of
# three tracks: pos (3), mom (3), time, then LoopHelix rad, lam, cx, cy, phi0, t0, d0, maxr, tanDip
STOCK = np.array([
    [1.8057, 54.5068, 10.1, -40.4937, 56.1736, 78.2265, 1052.4857, 228.128, 258.1393, -183.4468, -78.6, 2.1547,
     1052.4408, 28.5517, 427.7042, 1.1298],
    [23.0322, 23.3189, -1631.11, -31.1319, 62.9116, 77.4969, 1045.1453, 225.8047, 249.3351, -180.8771, -77.1978,
     2.288, 1052.4858, 29.1425, 422.4669, 1.1035],
    [3.4091, 50.3908, 1639.11, -38.0724, 56.8599, 78.7752, 1059.7101, 229.8356, 264.7785, -185.8744, -76.7065,
     2.2523, 1052.4701, 28.7556, 430.9156, 1.151],
    [-113.3379, 131.4435, 10.1, 25.0232, -68.9348, 66.5297, 1249.8831, 242.0042, 219.1359, 114.0166, 214.4238,
     -1.2681, 1249.8315, 0.8484, 484.8568, 0.907],
    [-30.0982, 398.6758, -1631.11, -57.68, -45.6483, 66.6301, 1241.7472, 237.0894, 214.2501, 115.8412, 218.4717,
     -1.1455, 1249.8556, 10.194, 484.3728, 0.9098],
    [66.8339, -22.1918, 1639.11, 70.5913, -14.0675, 67.6493, 1257.8992, 241.9589, 227.1071, 115.2458, 216.2197,
     -1.1299, 1249.9103, 3.0565, 486.9743, 0.9381],
    [-49.8505, 115.8671, 10.1, 65.0208, -54.8482, 59.1352, 738.049, 280.7815, 194.6188, 131.0382, 330.6322, -0.7523,
     737.9894, 74.871, 636.434, 0.6951],
    [78.1459, 593.5854, -1631.11, -84.1576, -17.83, 58.779, 728.3957, 277.4417, 189.3405, 136.0123, 327.9724,
     -0.6098, 738.0356, 77.615, 632.4983, 0.6856],
    [399.8509, 248.9255, 1639.11, 24.7942, 80.3776, 59.8384, 747.5117, 282.7032, 201.2404, 128.966, 331.891, -0.5911,
     738.0823, 73.364, 638.7704, 0.7114]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_loophelix_matches_stock_segments(dtype):
    """The LoopHelix parameters of a reco state in its local field reproduce the stock per-segment values."""
    s = STOCK.astype(dtype).astype(np.float64)
    x, p, t = s[:, :3], s[:, 3:6], s[:, 6]
    bz = np.hypot(p[:, 0], p[:, 1]) / (A.KB * s[:, 7])          # the local field that gives the stock radius
    lh = A.loophelix(x, p, t, bz)
    tol = 1e-3 if dtype == np.float64 else 2e-2
    np.testing.assert_allclose(lh["rad"], s[:, 7], rtol=tol)
    np.testing.assert_allclose(lh["lam"], s[:, 8], rtol=0.03)   # stock lam is the fit's (local field, energy loss)
    np.testing.assert_allclose(lh["cx"], s[:, 9], atol=8.0)     # fit centre vs the local-field geometry (gradient at TT_Front); a sign error is ~2R off
    np.testing.assert_allclose(lh["cy"], s[:, 10], atol=8.0)
    np.testing.assert_allclose(A._wrap(lh["phi0"] - s[:, 11]), 0, atol=0.05)
    np.testing.assert_allclose(lh["t0"], s[:, 12], atol=0.15)
    np.testing.assert_allclose(lh["d0"], s[:, 13], atol=8.0)
    np.testing.assert_allclose(lh["maxr"], s[:, 14], atol=8.0)
    np.testing.assert_allclose(lh["tanDip"], s[:, 15], atol=5e-3)   # stock: from the fit parameters


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_helix_to_z_round_trip(dtype):
    rng = np.random.default_rng(1)
    n = 50
    x = np.c_[rng.uniform(-300, 300, (n, 2)), np.full(n, -1631.1)].astype(dtype)
    pt = rng.uniform(50, 90, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    p = np.c_[pt * np.cos(phi), pt * np.sin(phi), rng.uniform(40, 80, n)].astype(dtype)
    t = rng.uniform(500, 1500, n).astype(dtype)
    xm, pm, tm = R.helix_to_z(x, p, t, 10.1)
    xb, pb, tb = R.helix_to_z(xm, pm, tm, -1631.1)
    tol = 1e-6 if dtype == np.float64 else 1e-2
    np.testing.assert_allclose(xb, x.astype(float), atol=tol * 100)
    np.testing.assert_allclose(pb, p.astype(float), atol=tol * 10)
    np.testing.assert_allclose(tb, t.astype(float), atol=tol * 10)
    # the circle centre and |p| are invariant, z advances by the path times pz / p
    np.testing.assert_allclose(np.linalg.norm(pm, axis=1), np.linalg.norm(p.astype(float), axis=1), rtol=1e-6)
    assert np.all(tm > t.astype(float))


def test_helix_e_minus_bends_like_stock():
    """A stock TT_Front state carried to TT_Mid lands near the stock TT_Mid position (same track, local field ~1 T)."""
    front, mid = STOCK[1], STOCK[0]
    xm, pm, tm = R.helix_to_z(front[None, :3], front[None, 3:6], front[None, 6:7].ravel(), mid[2])
    assert np.hypot(*(xm[0, :2] - mid[:2])) < 40.0         # 1 T vs the real field over 1.6 m
    assert abs(tm[0] - mid[6]) < 0.5


def test_plane_truth_picks_last_forward_step():
    steps = {k: np.zeros((2, 4)) for k in ("x1", "y1", "z1", "px", "py", "pz", "t")}
    steps["valid"] = np.array([[True, True, True, False], [False, False, False, False]])
    steps["z1"][0] = [-500.0, -20.0, 300.0, 0.0]
    steps["px"][0], steps["pz"][0] = 60.0, 70.0
    steps["t"][0] = [900.0, 902.0, 905.0, 0.0]
    found, x, p, t = R.plane_truth(steps, 10.1)
    assert found.tolist() == [True, False]
    assert x[0, 2] == pytest.approx(10.1)
    assert t[0] > 902.0 and np.isnan(t[1])


def test_reco_state_applies_front_residuals():
    x = np.array([[100.0, 0.0, 10.1]])
    p = np.array([[60.0, 0.0, 60.0]])
    xs, ps, ts = A.reco_state(x, p, np.array([1000.0]), np.array([1.01]), np.array([0.02]), np.array([-0.5]))
    assert np.linalg.norm(ps) == pytest.approx(np.linalg.norm(p) * 1.01)
    assert ps[0, 2] / np.hypot(ps[0, 0], ps[0, 1]) == pytest.approx(1.02)
    assert ts[0] == pytest.approx(999.5)
    assert np.arctan2(ps[0, 1], ps[0, 0]) == pytest.approx(0.0)


def _lib(n=300, seed=3):
    rng = np.random.default_rng(seed)
    return dict(cls=rng.integers(0, 3, n).astype(np.int8), cov=rng.random(n) < 0.8,
                tandip_t=rng.uniform(0.5, 1.1, n), nactive=rng.integers(15, 60, n), fitcon=rng.random(n),
                calo=rng.random(n) < 0.9, eop=rng.uniform(0.6, 1.0, n), ecl=rng.uniform(0, 100, n),
                file_index=np.zeros(n, np.int64), entry=np.arange(n), pid=rng.random(n), prim=np.zeros(n, np.int64),
                phi_t=rng.uniform(-3, 3, n), t_t=rng.uniform(500, 1600, n))


def test_donor_match_is_deterministic_and_respects_class():
    lib = _lib()
    n = 40
    rng = np.random.default_rng(5)
    cls = rng.integers(0, 3, n).astype(np.int8)
    cov = rng.random(n) < 0.7
    keys = dict(tandip_t=rng.uniform(0.5, 1.1, n), nactive=rng.integers(15, 60, n), fitcon=rng.random(n),
                calo=rng.random(n) < 0.9, eop=rng.uniform(0.6, 1.0, n), ecl=rng.uniform(0, 100, n))
    run, sub, ev = np.full(n, 1430), np.zeros(n, int), np.arange(n)
    f1, r1 = D.match(lib, cls, cov, keys, 7, run, sub, ev)
    f2, r2 = D.match(lib, cls, cov, keys, 7, run, sub, ev)
    assert np.array_equal(r1, r2)
    assert np.all(lib["cls"][r1] == cls)
    # per-event draws do not depend on the batch: the same event alone gets the same donor
    f3, r3 = D.match(lib, cls[5:6], cov[5:6], {k: v[5:6] for k, v in keys.items()}, 7, run[5:6], sub[5:6], ev[5:6])
    assert r3[0] == r1[5]
    q = np.flatnonzero((cls == D.CLASS_Q))
    if q.size:      # class Q donors are near in active hits
        assert np.median(np.abs(lib["nactive"][r1[q]] - keys["nactive"][q])) <= 3


def test_classes_and_calo_match():
    rec = dict(trk_qual=np.array([True, False, False]), trk_fit=np.array([True, True, False]))
    assert A.classes(rec).tolist() == [D.CLASS_Q, D.CLASS_F, D.CLASS_N]
    rec = dict(cl_valid=np.array([[True, True], [True, False]]), cl_energy=np.array([[5.0, 80.0], [90.0, 0.0]]),
               cl_disk=np.array([[0, 0], [1, 0]]), calo_disk=np.array([0, 0]), calo_kind=np.array([1, 1]))
    has, j = A.calo_match(rec, rule="contact")
    assert has.tolist() == [True, False] and j[0] == 1
    # stock rule: the other disk counts when the contact disk has none (event 1); kept with the inclusion efficiency
    has, j = A.calo_match(rec, np.full(2, 1430), np.zeros(2, int), np.arange(2), 7)
    assert j.tolist() == [1, 0]
    assert has.tolist() == [True, True]                       # 80 and 90 MeV: efficiency >= 0.989 (these draws pass)


def test_calo_match_stock_rule_follows_the_inclusion_efficiency():
    n = 40000
    e = np.where(np.arange(n) % 2 == 0, 15.0, 85.0)
    rec = dict(cl_valid=np.ones((n, 1), bool), cl_energy=e[:, None], cl_disk=np.zeros((n, 1), int),
               calo_disk=np.zeros(n, int), calo_kind=np.ones(n, int))
    has, _ = A.calo_match(rec, np.full(n, 1430), np.zeros(n, int), np.arange(n), 3)
    assert abs(has[e == 15.0].mean() - A.CALO_EFF[0]) < 0.01 and abs(has[e == 85.0].mean() - A.CALO_EFF[7]) < 0.01
    again, _ = A.calo_match(rec, np.full(n, 1430), np.zeros(n, int), np.arange(n), 3)
    np.testing.assert_array_equal(again, has)
    with pytest.raises(ValueError):
        A.calo_match(rec, rule="nearest")


def test_contract_covers_every_reader_and_has_classes():
    assert {"exact", "fast", "donor", "fast+donor", "not available"} >= {c for _, _, c, _ in K.COVERAGE}
    md = K.table_markdown()
    for leaf in ("trkqual.result", "trkpid.result", "trksegpars_lh", "caloclusters", "hitcount.nsd"):
        assert leaf in md
    for reader, leaves in K.READERS.items():
        assert leaves, reader
    json.dumps(K.READERS)


def test_writer_stream_patterns():
    import fnmatch
    from beamfastsim.centuple import writer as W
    names = ["apr_TrkDe_80m70p", "cpr_TrkDe_80m70p_D0200", "tpr_TrkDe_80m70p", "TriggerableLHPath", "calo_photon",
             "apr_TrkUe_80m70p", "apr_TwoTrkDe_80m70p_D0200"]
    sel = [n for n in names if any(fnmatch.fnmatch(n, p) for p in W.STREAM_PATTERNS)]
    assert sel == ["apr_TrkDe_80m70p", "cpr_TrkDe_80m70p_D0200", "TriggerableLHPath", "apr_TrkUe_80m70p"]


def test_trigcal_factors_reach_the_target():
    from beamfastsim.centuple import trigcal
    T = np.array([0.2, 0.9, 0.5, 1.0])
    Dv = np.array([0.5, 0.6, 0.5, 0.0])
    f, g = trigcal.factors(T, Dv)
    np.testing.assert_allclose(Dv * f + (1 - Dv) * g, T)
    assert np.all((f >= 0) & (f <= 1) & (g >= 0) & (g <= 1))


def test_trigcal_apply_shares_one_uniform_and_keeps_unlisted_paths():
    from beamfastsim.centuple import trigcal
    from beamfastsim.centuple import donor as D
    n = 20000
    rng = np.random.default_rng(1)
    edges = np.array(trigcal.EDGES)
    nb = len(edges) - 1
    keep = np.full((2, 3, nb), 0.5)
    rescue = np.full((2, 3, nb), 0.0)
    rescue[1] = 0.25
    lib = dict(paths=np.array(["a", "b"]), classes=np.array(trigcal.CLASSES, np.int8), keep=keep, rescue=rescue,
               tbirth_edges=edges)
    raw = np.stack([np.ones(n, bool), np.zeros(n, bool), rng.random(n) < 0.5], 1)
    cls = rng.choice([D.CLASS_N, D.CLASS_F, D.CLASS_Q], n)
    run, sub, ev = np.full(n, 1430), np.zeros(n, int), np.arange(n)
    out = trigcal.apply(lib, ["a", "b", "c"], raw, cls, rng.uniform(400, 1700, n), run, sub, ev, 7)
    assert abs(out[:, 0].mean() - 0.5) < 0.02 and abs(out[:, 1].mean() - 0.25) < 0.02
    np.testing.assert_array_equal(out[:, 2], raw[:, 2])
    # one uniform per event: every rescued "b" is a kept "a" (u < 0.25 implies u < 0.5)
    assert np.all(out[out[:, 1], 0])
    again = trigcal.apply(lib, ["a", "b", "c"], raw, cls, rng.uniform(400, 1700, n), run, sub, ev, 7)
    np.testing.assert_array_equal(again, out)


def test_shipped_trigcal_table_loads_and_is_bounded():
    from beamfastsim.centuple import trigcal
    lib = trigcal.load(trigcal.SHIPPED)
    paths = [str(p) for p in lib["paths"]]
    assert "apr_TrkDe_80m70p" in paths and "tpr_TrkDe_80m70p" in paths
    assert lib["keep"].shape == lib["rescue"].shape == (len(paths), 3, len(lib["tbirth_edges"]) - 1)
    for k in ("keep", "rescue"):
        assert np.all((lib[k] >= 0) & (lib[k] <= 1))


def test_cluster_truth_time_folds_like_the_reco_time():
    n, m = 3, R.MAX_CLUSTERS
    valid = np.zeros((n, m), bool)
    valid[:, 0] = True
    reco = np.array([862.0, 819.6, 1131.2])
    truth = np.array([860.4, 818.6 + A.MBTIME, np.inf])          # no true deposit: the contact time
    tp = A.cluster_truth_time(np.where(valid, truth[:, None], np.inf), np.where(valid, reco[:, None], 0.0),
                              np.array([0.0, 0.0, 1129.4 + A.MBTIME]), valid)
    np.testing.assert_allclose(reco - tp[:, 0], [1.6, 1.0, 1.8], atol=1e-9)
    assert np.all(tp[:, 1:] == 0.0)
