"""Device TimeClusterFinder (trigger/device/timecluster.py) against the literal transcription of the stock module
(tests/time_cluster_reference.py), in both precisions of the chain: synthetic events with calo seeds and synthetic
MLPs; the pieces (weighted P^2 median, glibc expf, evalMVA, strtof); and, when TRIGGER_TC_DUMP (a pickled list of
events: hits, cc, tc) and TRIGGER_TC_MVA / TRIGGER_TC_CALO_MVA (the TimeCluster and TimeClusterCalo weight files) are
set, the stored stock TimeClusters of a replay."""
from pathlib import Path
import math
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

# the TrigTimeClusterFinder table of the trigger menu (TrigTimeClusterFinderUe: AveragePitch -0.63)
TCF = {"AveragePitch": 0.6, "CaloClusterMinE": 50, "CaloClusterWeight": 10, "DtMax": 25, "MaxdPhi": 1.5,
       "MinAddHitMVA": 0.2, "MinKeepHitMVA": 0.2, "MinNHits": 10, "PeakWidth": 1, "PrefilterCluster": True,
       "RecoverHits": True, "RefineClusters": True, "T0Calculator": {"CaloTimeErr": 0.5, "TrkToCaloTimeOffset": 0},
       "Tbin": 15, "TestFlag": False, "Tmax": 100000, "Tmin": 0, "UseCaloCluster": True,
       "UseCaloClusterPosition": True, "Ymin": 5}
PITCHES = (0.6, -0.63)
FK = ("time", "x", "y", "z", "uvar", "wdist")
IK = ("ncombo", "plane")
CK = ("disk", "time", "energy", "cog_x", "cog_y", "cog_z")
VARS = [(8.7e-04, 6.0e+01), (9.9e-05, 3.1e+00), (1.56e+05, 4.22e+05), (1.0, 4.0), (0.0, 35.0), (18.6, 95.0),
        (0.0, 522.9)]


def _xml(rng, scale=1.5):
    """A TMVA MLP weight file with the TimeCluster layout (7 inputs, sigmoid 7-6 hidden, N normalisation)."""
    def num(v):
        return f"{v:.16e}"
    var = "".join(f'<Variable VarIndex="{i}" Min="{num(lo)}" Max="{num(hi)}"/>' for i, (lo, hi) in enumerate(VARS))
    sizes = [8, 8, 7, 1]
    layers = []
    for li in range(3):
        nsyn = sizes[li + 1] - (1 if li < 2 else 0)
        neur = "".join(f'<Neuron NSynapses="{nsyn}">' + " ".join(num(v) for v in rng.normal(0, scale, nsyn))
                       + "</Neuron>" for _ in range(sizes[li]))
        layers.append(f'<Layer Index="{li}" NNeurons="{sizes[li]}">{neur}</Layer>')
    layers.append('<Layer Index="3" NNeurons="1"><Neuron NSynapses="0"/></Layer>')
    return ('<?xml version="1.0"?><MethodSetup Method="MLP::MLP"><GeneralInfo>'
            '<Info name="TMVA Release" value="4.2.1 [262657]"/></GeneralInfo><Options>'
            '<Option name="NeuronType" modified="No">sigmoid</Option>'
            '<Option name="VarTransform" modified="Yes">N</Option></Options>'
            f'<Variables NVar="7">{var}</Variables><Weights><Layout NLayers="4">{"".join(layers)}</Layout>'
            '</Weights></MethodSetup>')


def _nets(tmp_path, seed=5):
    import time_cluster_reference as R
    from trigger.device import mva
    rng = np.random.default_rng(seed)
    out = []
    for name in ("tc.xml", "tccalo.xml"):
        p = tmp_path / name
        p.write_text(_xml(rng))
        out.append((R.load_mva(p), mva.load(p)))
    return (out[0][0], out[1][0]), (out[0][1], out[1][1])


def _event(rng, kind):
    """Hits (collection order) and calo clusters of one synthetic event."""
    hits = []

    def hit(r, phi, z, t, nsh=None):
        hits.append({"time": t, "x": r * math.cos(phi), "y": r * math.sin(phi), "z": z,
                     "uvar": rng.uniform(18.7, 94.0) ** 2, "wdist": rng.uniform(-500, 500),
                     "ncombo": int(nsh or rng.integers(1, 4)), "plane": int(rng.integers(0, 36))})

    def track(t0, phi0, n, lam=None, spread=3.0):
        lam = lam if lam is not None else rng.choice([-1, 1]) * rng.uniform(170, 260)
        for z in np.sort(rng.uniform(-1500, 1500, n)):
            hit(rng.uniform(380, 640), phi0 + z / lam / 4 + rng.normal(0, 0.15), z,
                t0 + z / (0.6 * 299.792458) + rng.normal(0, spread))

    def noise(n, lo=500, hi=1700):
        for _ in range(n):
            hit(rng.uniform(380, 680), rng.uniform(-math.pi, math.pi), rng.uniform(-1500, 1500), rng.uniform(lo, hi))

    cc = []
    if kind == "calo":
        track(1000, 1.0, 22)
        noise(10)
        cc.append({"disk": 0, "time": 1006.0, "energy": 80.0, "cog_x": -420.123456789, "cog_y": 310.5})
        cc.append({"disk": 1, "time": 1500.0, "energy": 30.0, "cog_x": 400.0, "cog_y": 5.0})       # below MinE
    elif kind == "two":
        track(900, -2.0, 18)
        track(1300, 2.5, 20)
        noise(12)
    elif kind == "calo2":
        track(1100, 0.3, 25)
        track(1400, -1.2, 15)
        noise(8)
        cc.append({"disk": 1, "time": 1408.3, "energy": 70.0, "cog_x": -100.25, "cog_y": -520.75})
        cc.append({"disk": 0, "time": 1110.0, "energy": 95.0, "cog_x": 330.5, "cog_y": 60.125})
    elif kind == "edge":                       # calo seed outside the histogram (underflow), a far one (overflow)
        track(700, 0.8, 16)
        noise(6)
        cc.append({"disk": 0, "time": -50.0, "energy": 60.0, "cog_x": 0.0, "cog_y": 450.0})
        cc.append({"disk": 1, "time": 100090.0, "energy": 60.0, "cog_x": 300.0, "cog_y": -300.0})
    elif kind == "dense":                      # many hits: refinement and recovery do work
        track(1200, -0.4, 40, spread=6.0)
        noise(40, 1150, 1250)
    else:
        noise(30)
    h = {k: np.array([x[k] for x in hits], np.float32) for k in FK}
    h.update({k: np.array([x[k] for x in hits], np.int64) for k in IK})
    c = {k: np.array([x[k] for x in cc], np.float64 if k.startswith("cog") else np.float32) for k in CK if k != "cog_z"}
    c["disk"] = np.array([x["disk"] for x in cc], np.int32)
    c["cog_z"] = np.zeros(len(cc))
    return h, c


def _events():
    rng = np.random.default_rng(17)
    return [_event(rng, k) for k in ("calo", "two", "calo2", "edge", "dense", "noise", "calo", "two")]


def _arrays(evs, ft, H=None, C=4):
    E = len(evs)
    H = H or max(64, max(len(h["x"]) for h, _ in evs))
    it = np.int64 if ft == np.float64 else np.int32
    ph = {k: np.zeros((E, H), ft) for k in FK}
    ph.update({k: np.zeros((E, H), it) for k in IK})
    ph["valid"] = np.zeros((E, H), bool)
    cl = {k: np.zeros((E, C), np.float64 if k.startswith("cog") else np.float32) for k in CK}
    cl["disk"] = np.zeros((E, C), np.int32)
    cl["valid"] = np.zeros((E, C), bool)
    for e, (h, c) in enumerate(evs):
        n = len(h["x"])
        for k in FK + IK:
            ph[k][e, :n] = h[k]
        ph["valid"][e, :n] = True
        m = len(c["time"])
        for k in CK:
            cl[k][e, :m] = c[k]
        cl["valid"][e, :m] = True
    return ph, cl


def _cl_pack(cl):
    """The calo clusters for a jit argument: each double cog as three floats (hi, mid, lo: exact), since a float64
    argument is rounded to float32 when x64 is off (in the chain the cog is made on the device in double)."""
    out = dict(cl)
    for k in ("cog_x", "cog_y", "cog_z"):
        a = np.asarray(cl[k], np.float64)
        hi = a.astype(np.float32)
        r = a - hi
        mid = r.astype(np.float32)
        out[k] = np.stack([hi, mid, (r - mid).astype(np.float32)], -1)
    return out


def _cl_unpack(cl):
    import jax
    import jax.numpy as jnp
    out = dict(cl)
    with jax.enable_x64(True):
        for k in ("cog_x", "cog_y", "cog_z"):
            b = jnp.asarray(cl[k]).astype(jnp.float64)
            out[k] = (b[..., 0] + b[..., 1]) + b[..., 2]
    return out


def _unpack(r, e, order_none):
    out = []
    for c in np.flatnonzero(r["valid"][e]):
        key = r["order"][e, c]
        mem = np.flatnonzero(key < order_none)
        out.append({"hits": [int(i) for i in mem[np.argsort(key[mem], kind="stable")]], "t0": float(r["t0"][e, c]),
                    "t0err": float(r["t0err"][e, c]), "pos": np.asarray(r["pos"][e, c], np.float32),
                    "nsh": int(r["nsh"][e, c]), "calo": int(r["calo"][e, c])})
    return out


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert list(x["hits"]) == [int(i) for i in y["hits"]]
        assert x["nsh"] == y["nsh"] and x["calo"] == y["calo"]
        assert np.float64(x["t0"]).tobytes() == np.float64(y["t0"]).tobytes(), (x["t0"], y["t0"])
        assert np.float64(x["t0err"]).tobytes() == np.float64(y["t0err"]).tobytes(), (x["t0err"], y["t0err"])
        assert np.asarray(x["pos"], np.float32).tobytes() == np.asarray(y["pos"], np.float32).tobytes()


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_time_clusters_match_the_transcription(precision, tmp_path):
    """8 synthetic events, both pitches: the device clusters (collection order, hits in _strawHitIdxs order, t0 and
    t0err as doubles, pos, nsh, calo index) equal the transcription's, in the full, pairs and compact paths; the
    sample has calo seeds (one below CaloClusterMinE, seeds in the under- and overflow bins), several peaks,
    prefilter removals, MVA refinement and recovery."""
    jax = pytest.importorskip("jax")
    import time_cluster_reference as R
    from trigger.device import timecluster as TC
    rnets, dnets = _nets(tmp_path)
    m = {**TCF, "MinKeepHitMVA": 0.45, "MinAddHitMVA": 0.55}
    ft = np.float64 if precision == "float64" else np.float32
    evs = _events()
    ph, cl = _arrays(evs, ft)
    ss = [{**TC.settings(m, pitch=p), "order": True} for p in PITCHES]
    jax.config.update("jax_enable_x64", precision == "float64")
    E = len(evs)
    cl = _cl_pack(cl)
    full = jax.device_get(jax.jit(lambda p, c: TC.time_clusters_multi(p, ss, dnets[0], _cl_unpack(c), dnets[1]))(
        ph, cl))
    pairs = jax.device_get(jax.jit(lambda p, c: TC.time_clusters_multi_pairs(p, ss, dnets[0], 2 * E, _cl_unpack(c),
                                                                             dnets[1]))(ph, cl))
    comp = []
    for s in ss:
        cand = jax.jit(lambda p, c: TC.candidates(p, s, _cl_unpack(c)))(ph, cl)
        comp.append(jax.device_get(jax.jit(lambda c_: TC.refine_pairs(*c_, s, dnets[0], E * 4, dnets[1]))(cand)))
    diag, ncl, ncalo = {}, 0, 0
    for j, p in enumerate(PITCHES):
        rs = R.settings(m, pitch=p)
        assert full[j]["t0"].dtype == np.float64 and full[j]["pos"].dtype == np.float32
        for e, (h, c) in enumerate(evs):
            ref = R.time_clusters(h, c, rs, rnets, diag=diag)
            ncl += len(ref)
            ncalo += sum(r["calo"] >= 0 for r in ref)
            for res in (full[j], pairs[j], comp[j]):
                assert res["overflow"][e] == 0
                _same(_unpack(res, e, TC.ORDER_NONE), ref)
    assert ncl >= 12 and ncalo >= 4, (ncl, ncalo)
    for key in ("prefiltered", "refined", "recovered", "p2_linear"):
        assert diag.get(key, 0) > 0, (key, diag)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_weighted_p_square_median_matches_the_transcription(precision):
    """boost's weighted P^2 median on the device equals the transcription's for 1..60 float samples with integer
    weights, with and without a final double sample of weight 10 (the calo cog)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import time_cluster_reference as R
    from trigger.device import timecluster as TC
    from trigger.device.exact import strict
    jax.config.update("jax_enable_x64", precision == "float64")
    rng = np.random.default_rng(4)
    v = rng.normal(0, 300, 64).astype(np.float32)
    w = rng.integers(1, 4, 64).astype(np.float32)

    def med(v, w, on, extra, eon):
        with strict(on.astype(jnp.int32)):
            st = TC._seq_scan(lambda st, x, ww, o: TC._wp2_step(st, x[None], ww[None], o[None]), TC._wp2_init(1),
                              (v.astype(jnp.float64), w, on))
            mi, n = TC._members(on)
            st2 = TC._wp2_run(v.astype(jnp.float64)[:, None], w, mi, n)          # the cluster's form
            out = []
            for s_ in (st, st2):
                s_ = TC._wp2_step(s_, extra[None], jnp.full(1, np.float32(10)), eon[None])
                out.append(s_[2][0, 2])
            return out

    fn = jax.jit(med)
    for n in (1, 2, 3, 4, 5, 6, 7, 12, 40, 60):
        on = np.arange(64) < n
        for extra in (None, -123.456789012345):
            a = R.WeightedP2()
            for x, ww in zip(v[:n], w[:n]):
                a.add(x, ww)
            if extra is not None:
                a.add(extra, np.float32(10))
            got = jax.device_get(fn(v, w, on, np.float64(extra or 0.0), np.bool_(extra is not None)))
            assert np.float32(got[0]) == a.result() and np.float32(got[1]) == a.result(), (n, extra, got)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_expf_and_mva_match_the_transcription(precision, tmp_path):
    """glibc's expf (generic build) and MVATools::evalMVA on the device equal the transcription's, bit for bit."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import time_cluster_reference as R
    from trigger.device import mva
    from trigger.device.exact import strict
    jax.config.update("jax_enable_x64", precision == "float64")
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.uniform(-110, 95, 20000), rng.normal(0, 3, 20000), [-np.inf, 0.0, -0.0, 88.8, -104.0]])
    x = x.astype(np.float32)

    def ex(x):
        # the guards' zero from an input (a constant one lets XLA fold them; small ints: |INT_MIN| is not >= 0)
        with strict(jnp.isnan(x).astype(jnp.int32)):
            return mva.expf(x)
    got = np.asarray(jax.device_get(jax.jit(ex)(x)))
    ref = R.expf(x)
    # XLA's CPU code flushes float32 subnormals to zero (glibc returns them for x in about [-104, -87.3]); the MVA's
    # sigmoid 1 / (1 + expf(-a)) is 1.0 either way
    sub = (ref != 0) & (np.abs(ref) < np.finfo(np.float32).tiny)
    assert np.array_equal(got[~sub].view(np.uint32), ref[~sub].view(np.uint32))
    assert sub.any() and np.all((got[sub] == 0) | (got[sub] == ref[sub]))
    rnets, dnets = _nets(tmp_path, seed=3)
    X = np.stack([rng.uniform(lo, hi, 4000) for lo, hi in VARS], 1).astype(np.float32)

    def ev(X):
        with strict(jnp.isnan(X).astype(jnp.int32)):
            return mva.evaluate_stock(dnets[0], X)
    got = np.asarray(jax.device_get(jax.jit(ev)(X)))
    ref = R.eval_mva(rnets[0], X)
    assert np.array_equal(got, ref)
    assert np.ptp(ref) > 0.05                       # the inputs span the net's response
    for a, b in zip(rnets[0]["layers"], dnets[0]["layers"]):
        assert np.array_equal(a, b)
    assert np.array_equal(rnets[0]["scale"], dnets[0]["scale"])


def test_strtof_rounds_once():
    """strtof is the correctly rounded decimal-to-float conversion (the double-then-float path rounds twice on this
    input)."""
    pytest.importorskip("jax")
    from trigger.device import mva
    s = "1.00000005960464477550"           # just above the midpoint between 1 and the next float
    assert mva.strtof(s) == np.nextafter(np.float32(1), np.float32(2))
    assert np.float32(float("1.0000000596046447")) == np.float32(1.0)    # the midpoint itself: ties to even
    assert mva.strtof("4.2340006549771914e-01") == np.float32(0.42340006549771914)


def test_settings_and_calo_inputs_are_checked():
    pytest.importorskip("jax")
    from trigger.device import timecluster as TC
    with pytest.raises(ValueError, match="TestFlag"):
        TC.settings({**TCF, "TestFlag": True})
    with pytest.raises(ValueError, match="UseTOTDrift"):
        TC.settings({**TCF, "T0Calculator": {"UseTOTDrift": False}})
    s = TC.settings(TCF)
    ph, cl = _arrays(_events()[:1], np.float32)
    with pytest.raises(ValueError, match="ClusterCaloMVA"):
        TC.time_clusters(ph, s, {"act": "sigmoid"}, cl)


DUMP = os.environ.get("TRIGGER_TC_DUMP")
MVA = os.environ.get("TRIGGER_TC_MVA")
CMVA = os.environ.get("TRIGGER_TC_CALO_MVA")


@pytest.mark.skipif(not (DUMP and MVA and CMVA), reason="TRIGGER_TC_DUMP / TRIGGER_TC_MVA / TRIGGER_TC_CALO_MVA")
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_dump_matches_stock(precision):
    """The first 96 events of a replay dump: the transcription and the device give the stored stock TimeClusters of
    TrigTimeClusterFinder and TrigTimeClusterFinderUe."""
    jax = pytest.importorskip("jax")
    import time_cluster_reference as R
    from trigger.device import mva, timecluster as TC
    evs = pickle.load(open(DUMP, "rb"))[:96]
    rnets = (R.load_mva(MVA), R.load_mva(CMVA))
    dnets = (mva.load(MVA), mva.load(CMVA))
    ft = np.float64 if precision == "float64" else np.float32
    jax.config.update("jax_enable_x64", precision == "float64")
    ss = [{**TC.settings(TCF, pitch=p), "order": True} for p in PITCHES]
    fn = jax.jit(lambda p, c: TC.time_clusters_multi_pairs(p, ss, dnets[0], 64, _cl_unpack(c), dnets[1]))
    for lo in range(0, len(evs), 32):
        part = [(e["hits"], e["cc"]) for e in evs[lo:lo + 32]]
        H = max(64, 1 << (max(len(h["x"]) for h, _ in part) - 1).bit_length())
        ph, cl = _arrays(part, ft, H=H, C=8)
        res = jax.device_get(fn(ph, _cl_pack(cl)))
        for j, (inst, p) in enumerate(zip(("De", "Ue"), PITCHES)):
            rs = R.settings(TCF, pitch=p)
            for e, ev in enumerate(evs[lo:lo + 32]):
                ref = R.time_clusters(ev["hits"], ev["cc"], rs, rnets)
                _same(ref, ev["tc"][inst])
                _same(_unpack(res[j], e, TC.ORDER_NONE), ref)
