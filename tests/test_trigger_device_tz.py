"""Device TZClusterFinder on synthetic events against the literal transcription of the stock module
(tests/tz_cluster_reference.py), in both precisions, and the stock TimeCluster art decoder on memberwise blobs."""
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# the TrigTZClusterFinder table of the trigger menu
TZ = {"chunkSep": 5, "chunkWindow": 20, "chunkThresh": 3, "combineWindow": 30, "maxCombineSep": 2500,
      "chunkFitThresh": 8, "recoverWindow": 30, "clusterThresh": 15, "minCaloSize": 2, "minCaloEnergy": 50,
      "caloDtMax": 30, "caloTimeOffset": -3.1, "useCCs": 1, "recoverCCs": 1, "protonTCs": True, "radSelect": 0,
      "doRefine": 0, "hitBkgBits": ["Noisy", "Dead", "Background"]}
ESEL, BKG, NOISY = 1 << 1, 1 << 6, 1 << 24
E, H, C = 24, 160, 6


def _zplane(p):
    return -1500.0 + 85.7 * p


def _event(rng, kind):
    """Hits (stock collection order is the list order) and calo clusters of one synthetic event."""
    hits, cls = [], []

    def hit(p, t, esel, nsh=None):
        hits.append({"plane": int(p), "time": np.float32(t), "z": np.float32(_zplane(p) + rng.choice([-10.0, 10.0])),
                     "timevar": np.float32(rng.uniform(4, 30)), "nsh": int(nsh or rng.integers(1, 4)),
                     "flag": (ESEL if esel else 0) | (BKG if rng.random() < 0.08 else 0)})

    def track(esel, t0, slope, planes, per=(1, 3)):
        for p in planes:
            for _ in range(int(rng.integers(*per))):
                hit(p, t0 + slope * _zplane(p) + rng.normal(0, 3), esel)

    def calo(t, e=None, size=None):
        cls.append({"time": np.float32(t + 3.1 + rng.normal(0, 4)), "energy": np.float32(e or rng.uniform(40, 120)),
                    "size": int(size or rng.integers(1, 5))})

    if kind == "sparse":            # calo recovery only: CE hits 6 planes apart (no chunk), a cluster at their time
        t = rng.uniform(700, 1400)
        for p in range(35, -1, -6):
            hit(p, t + rng.normal(0, 4), True, nsh=3)
        calo(t, e=90, size=3)
    elif kind == "protons":         # only proton chunks (the last chunk saved is a proton one)
        for _ in range(int(rng.integers(1, 3))):
            t0, sl = rng.uniform(600, 1500), rng.uniform(-0.004, 0.004)
            track(False, t0, sl, range(int(rng.integers(8, 20)), -1, -1), (2, 4))
    elif kind == "ties":            # a crowded plane with equal times, and many equal-time chunks of both kinds
        t = np.float32(rng.uniform(600, 1200))
        for i in range(22):
            hit(30, t if i % 3 == 0 else t + np.float32(rng.integers(-3, 4)), i % 2 == 0)
        for j in range(18):
            p = 33 - (j % 4) * 8
            tj = np.float32(800 + 40 * (j // 4))
            for _ in range(3):
                hits.append({"plane": p, "time": tj, "z": np.float32(_zplane(p)), "timevar": np.float32(9),
                             "nsh": 1, "flag": ESEL if j % 3 else 0})
    else:                           # tracks with gaps (combines, recovery, line combines), protons, noise, calo
        for _ in range(int(rng.integers(1, 4))):
            t0, sl = rng.uniform(600, 1500), rng.uniform(-0.03, 0.03)
            top = int(rng.integers(20, 36))
            gap = int(rng.integers(6, 12))
            track(True, t0, sl, range(top, top - 8, -1))
            track(True, t0, sl, range(top - 8 - gap, max(top - 16 - gap, -1), -1))
            r = rng.random()
            if r < 0.5:
                calo(t0)
            if r < 0.25:
                calo(t0)                                 # a second match: a copy of the chunk
        if rng.random() < 0.5:
            track(False, rng.uniform(600, 1500), 0.0, range(int(rng.integers(5, 15)), -1, -1), (2, 4))
        for _ in range(int(rng.integers(5, 30))):
            hit(int(rng.integers(0, 36)), rng.uniform(450, 1650), rng.random() < 0.7)
        if rng.random() < 0.4:
            t = rng.uniform(450, 1650)
            for p in range(int(rng.integers(30, 36)), -1, -6):
                hit(p, t + rng.normal(0, 4), True, nsh=3)
            calo(t)
        if rng.random() < 0.3:
            calo(rng.uniform(450, 1650))
    if len(hits) > H:
        hits = hits[:H]
    order = rng.permutation(len(hits))                   # the collection order is not the plane order
    hits = [hits[i] for i in order]
    for h in hits:
        if rng.random() < 0.03:
            h["flag"] |= NOISY
    return hits, cls[:C]


def _arrays(evs, ft):
    ph = {"valid": np.zeros((E, H), bool), "time": np.zeros((E, H), ft), "timevar": np.ones((E, H), ft),
          "z": np.zeros((E, H), ft), "ncombo": np.zeros((E, H), np.int32), "plane": np.zeros((E, H), np.int32),
          "flag": np.zeros((E, H), np.int32)}
    cl = {"valid": np.zeros((E, C), bool), "energy": np.zeros((E, C), ft), "size": np.zeros((E, C), np.int32),
          "time": np.zeros((E, C), ft)}
    for e, (hs, cs) in enumerate(evs):
        for i, h in enumerate(hs):
            ph["valid"][e, i] = True
            for k in ("time", "timevar", "z", "plane", "flag"):
                ph[k][e, i] = h[k]
            ph["ncombo"][e, i] = h["nsh"]
        for j, c in enumerate(cs):
            cl["valid"][e, j] = True
            for k in ("time", "energy", "size"):
                cl[k][e, j] = c[k]
    return ph, cl


def _ref_inputs(hs, cs):
    col = lambda k, dt: np.array([h[k] for h in hs], dt)
    return ({"time": col("time", np.float32), "timevar": col("timevar", np.float32), "z": col("z", np.float32),
             "nsh": col("nsh", np.int64), "plane": col("plane", np.int64), "flag": col("flag", np.int64)},
            {k: np.array([c[k] for c in cs], np.float32 if k != "size" else np.int64)
             for k in ("time", "energy", "size")})


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_tz_clusters_match_the_transcription(precision):
    """24 synthetic events in two calls (the stale energy selection carried across): the device electron clusters,
    their hits in stock order, t0, nsh, calo index, and the electron / proton counts equal the transcription's; the
    sample exercises every branch the counters name."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", precision == "float64")
    import tz_cluster_reference as R
    from trigger.device import tzcluster
    rng = np.random.default_rng(11)
    kinds = ["normal"] * 6 + ["protons", "sparse", "ties", "normal", "sparse", "normal", "normal", "protons",
                              "normal", "sparse"] + ["normal"] * 8
    evs = [_event(rng, k) for k in kinds]
    ft = np.float64 if precision == "float64" else np.float32
    ph, cl = _arrays(evs, ft)
    s, rs = tzcluster.settings(TZ), R.settings(TZ)
    fn = jax.jit(lambda p, c, n: tzcluster.clusters(p, c, s, n, K=8, KC=48, A=8))
    half = E // 2
    outs, nrg = [], np.int32(0)
    for lo in (0, half):
        o = jax.device_get(fn({k: v[lo:lo + half] for k, v in ph.items()}, {k: v[lo:lo + half] for k, v in cl.items()},
                              nrg))
        nrg = np.int32(o["nrg_out"])
        outs.append(o)
    dev = {k: np.concatenate([o[k] for o in outs]) for k in outs[0] if k != "nrg_out"}
    st, diag = {"nrg": 0}, {}
    for e, (hs, cs) in enumerate(evs):
        el, _, ne, npr, st = R.tz_cluster(*_ref_inputs(hs, cs), rs, st, diag)
        assert dev["overflow"][e] == 0 and dev["indeterminate"][e] == 0
        assert (int(dev["n_e"][e]), int(dev["n_p"][e]), int(dev["valid"][e].sum())) == (ne, npr, len(el)), e
        for k, a in enumerate(el):
            assert [int(x) for x in dev["order"][e, k] if x >= 0] == a["hits"], (e, k)
            assert np.flatnonzero(dev["members"][e, k]).tolist() == sorted(a["hits"])
            assert (int(dev["nsh"][e, k]), int(dev["calo"][e, k])) == (a["nsh"], a["calo"]), (e, k)
            assert dev["t0"][e, k] == ft(a["t0"]), (e, k)
    assert int(nrg) == st["nrg"]
    for key in ("combine", "line", "recover_hit", "dup", "recovered_nrg0", "recovered_nrg1", "recovered_previous",
                "plane_ties", "chunk_ties"):
        assert diag.get(key, 0) > 0, (key, diag)


def _bc(body):
    return struct.pack(">I", 0x40000000 | len(body)) + body


def _tc_blob(tcs):
    body = struct.pack(">HHIi", 0x400A, 0, 0x1234ABCD, len(tcs))
    if tcs:
        hb = struct.pack(">H", 0) + b"".join(struct.pack(">i", len(t["hits"])) +
                                             b"".join(struct.pack(">H", i) for i in t["hits"]) for t in tcs)
        body += _bc(hb)
        body += b"".join(_bc(struct.pack(">HI", 0, 0xBEEF) + struct.pack(">dd", t["t0"], t["t0err"])) for t in tcs)
        body += b"".join(_bc(struct.pack(">HI", 0, 0xCAFE) + _bc(struct.pack(">HI", 0, 0xF00D) +
                                                                  struct.pack(">fff", *t["pos"]))) for t in tcs)
        body += b"".join(struct.pack(">H", t["nsh"]) for t in tcs)
        body += b"".join(_bc(struct.pack(">HI", 0, 0xD00D) + bytes(18) + struct.pack(">Q", t["key"])) for t in tcs)
    return _bc(body)


def test_time_cluster_decoder_on_memberwise_blobs():
    """strawart.decode_time_clusters reads the memberwise TimeCluster layout for n = 0, 1 and 3 clusters (hit lists
    of several lengths, a null calo Ptr), and rejects a truncated blob."""
    from trigger.device import strawart
    evs = [[], [{"hits": [5, 3, 9], "t0": 1032.125, "t0err": 0.0, "pos": (1.5, -2.0, 3.25), "nsh": 17, "key": 2}],
           [{"hits": [1, 2], "t0": 950.5, "t0err": 0.25, "pos": (0.0, 0.0, 0.0), "nsh": 15, "key": 2 ** 64 - 1},
            {"hits": [], "t0": -1.0, "t0err": 0.0, "pos": (0.0, 1.0, 0.0), "nsh": 0, "key": 0},
            {"hits": list(range(40, 60)), "t0": 1700.0625, "t0err": 1.0, "pos": (9.0, 8.0, 7.0), "nsh": 41, "key": 7}]]
    blobs = [_tc_blob(t) for t in evs]
    assert len(blobs[0]) == 16
    data = np.frombuffer(b"".join(blobs), np.uint8)
    off = np.r_[0, np.cumsum([len(b) for b in blobs])]
    out = strawart.decode_time_clusters(data, off)
    assert [len(o) for o in out] == [0, 1, 3]
    for got, want in zip(out, evs):
        for g, w in zip(got, want):
            assert g["hits"].tolist() == w["hits"] and g["t0"] == w["t0"] and g["t0err"] == w["t0err"]
            assert g["pos"].tolist() == list(w["pos"]) and g["nsh"] == w["nsh"]
            assert g["calo"] == (-1 if w["key"] == 2 ** 64 - 1 else w["key"])
    with pytest.raises(ValueError):
        strawart.decode_time_clusters(data[:-2], off - np.r_[0, 0, 0, 2])


def test_time_cluster_filter_require_calo_cluster():
    """TimeClusterFilter: requireCaloCluster passes only clusters with a calo cluster index (the TZ port's `calo`);
    clusters that carry none (the other device producers) fail it."""
    import jax.numpy as jnp
    from trigger.device import paths
    tc = {"valid": jnp.array([[True, True], [True, False], [True, True]]),
          "nsh": jnp.array([[20, 15], [30, 40], [14, 16]]), "calo": jnp.array([[-1, 2], [-1, 0], [0, -1]])}
    cut = {"minNStrawHits": 15}
    assert paths.tc_pass(tc, cut).tolist() == [True, True, True]
    assert paths.tc_pass(tc, {**cut, "requireCaloCluster": True}).tolist() == [True, False, False]
    no_calo = {k: v for k, v in tc.items() if k != "calo"}
    assert paths.tc_pass(no_calo, {**cut, "requireCaloCluster": True}).tolist() == [False, False, False]


def test_chain_tz_needs_deltafinder():
    from trigger.device import chain
    with pytest.raises(ValueError, match="deltafinder"):
        chain.Chain({"physics": {"producers": {}, "filters": {}, "trigger_paths": []}}, None, None, tz=True)
