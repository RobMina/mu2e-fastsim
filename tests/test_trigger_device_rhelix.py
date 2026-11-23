"""Device RobustHelixFinder and MergeHelices on synthetic events against the literal transcriptions of the stock
modules (tests/robust_helix_reference.py, tests/merge_helices_reference.py), in both precisions."""
from pathlib import Path
import math
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# the TrigHelixFinder and TrigHelixMergerDe tables of the trigger menu
RHF = {"AveragePitch": 0.6, "CentralPerpResolution": 12, "CentralRadialResolution": 20, "Helicities": [-1, 1],
       "HelixFitter": {"CaloClusterWeight": 10, "CircleFitType": 0, "CircleInitType": 0, "DontUseFlag": ["Outlier"],
                       "HitErrorWeight": False, "MinNPhi": 2, "NPhiHistBins": 25, "PhiHistRangeFactor": 1.2,
                       "RadiusWindow": 10, "TripleRadius": False, "UseCaloCluster": False, "fitFZMaxLambda": 1000,
                       "fitFZMinLambda": 30, "fitFZStepLambda": 10, "initFZMaxLambda": 1000, "initFZMinLambda": 30,
                       "initFZStepLambda": 20, "maxAbsLambda": 700, "maxCenterR": 700, "maxR": 600,
                       "maxdist": 2000, "maxniter": 100, "maxzsep": 1000, "minAbsLambda": 10, "minArea": 500,
                       "minCenterR": 100, "minNHit": 5, "minR": 100, "mindist": 50, "mindphi": 0.5, "minzsep": 100,
                       "nLoopsdfdz": 2, "ntripleMax": 500, "ntripleMin": 5, "targetradius": 600,
                       "trackerradius": 700, "use_initFZ_from_dzFrequency": False},
       "HitBackgroundBits": ["Background", "Noisy", "Dead"], "HitSelectionBits": ["TimeDivision"],
       "MaxChisquared": 25, "MaxIterations": 10, "MaxPhiHitSeparation": 1, "MaxRPull": 5, "MaxRWDot": 1,
       "MaxRadiusDiff": 100, "MaxTransDistance": 80, "MaxWireDistance": 200, "MinNStrawHits": 10,
       "PrefilterHits": True, "RPullScaleF": 1, "SaveHelixFlag": ["HelixOK"],
       "T0Calculator": {"CaloTimeErr": 0.5, "TrkToCaloTimeOffset": -3.1}, "UpdateStereo": False,
       "UseHitMVA": False, "UseTripletArea": False, "doSingleOutput": True, "maxEDepAvg": 2.5e-3,
       "targetconsistent": False, "targetconsistent_init": False}
MERGER = {"MinHitOverlapFraction": 0.5, "MinNHitOverlap": 10, "deltanh": 5, "scaleXY": 1.1, "scaleZPhi": 0.75}
TDIV, BKG = 1 << 10, 1 << 6
E, H, KT, C = 8, 128, 3, 4


def _zplane(p):
    return -1530.0 + 87.4 * p


def _event(rng, kind):
    """Hits (collection order), time clusters (hit lists in stored order, t0, t0err, calo) and calo clusters."""
    hits, tcs, cls = [], [], []

    def calo():
        cls.append({"disk": int(rng.integers(0, 2)), "time": float(np.float32(1030.0 + rng.normal(0, 2)))})
        return len(cls) - 1

    def hit(x, y, z, plane, t, nsh=None, flag=TDIV, edep=None):
        a = rng.uniform(0, 2 * math.pi)
        su, sv = 15.0, 3.0
        u, v = rng.normal(0, su), rng.normal(0, sv)
        phi = math.atan2(y, x) % (2 * math.pi)
        panel = int(phi / (math.pi / 3)) % 6
        hits.append({"x": x + u * math.cos(a) - v * math.sin(a), "y": y + u * math.sin(a) + v * math.cos(a),
                     "z": z, "udx": math.cos(a), "udy": math.sin(a), "uvar": su * su, "vvar": sv * sv,
                     "edep": edep if edep is not None else rng.uniform(1.0e-3, 2.4e-3), "time": t + rng.normal(0, 3),
                     "ncombo": int(nsh or rng.integers(1, 3)), "flag": flag,
                     "sid": (plane << 10) | (panel << 7) | int(rng.integers(0, 96))})
        return len(hits) - 1

    def track(planes=None, lam=None, edep=None):
        R, f0 = rng.uniform(240, 300), rng.uniform(0, 2 * math.pi)
        rc = R + rng.uniform(-60, 60)
        lam = lam if lam is not None else rng.choice([-1, 1]) * rng.uniform(170, 260)
        phi0 = rng.uniform(0, 2 * math.pi)
        cx, cy = rc * math.cos(f0), rc * math.sin(f0)
        idx = []
        for p in planes if planes is not None else range(36):
            if rng.random() < 0.15:
                continue
            for _ in range(1 + int(rng.random() < 0.25)):
                z = _zplane(p) + rng.uniform(-12, 12)
                ph = phi0 + z / lam
                idx.append(hit(cx + R * math.cos(ph), cy + R * math.sin(ph), z, p, 1000 + 0.005 * z, edep=edep))
        return idx

    def noise(k, flag=TDIV):
        out = []
        for _ in range(k):
            r, a = rng.uniform(380, 680), rng.uniform(0, 2 * math.pi)
            p = int(rng.integers(0, 36))
            out.append(hit(r * math.cos(a), r * math.sin(a), _zplane(p), p, 1000, flag=flag))
        return out

    if kind == "two":                       # two time clusters, two seeds
        tcs.append({"hits": track() + noise(4), "calo": calo()})
        tcs.append({"hits": track(planes=range(6, 36)) + noise(2)})
    elif kind == "bkg":                     # background and unselected hits are dropped by fillFaceOrderedHits
        tcs.append({"hits": track() + noise(5, BKG | TDIV) + noise(3, 0) + noise(3)})
    elif kind == "noise":                   # a cluster without a helix
        tcs.append({"hits": noise(18)})
    elif kind == "edep":                    # a helix dropped by maxEDepAvg
        tcs.append({"hits": track(edep=2.9e-3) + noise(2)})
    elif kind == "dup":                     # the same hits in two clusters, the second with a calo cluster (as the
        a = track() + noise(3)              # calo-seeded time clusters): MergeHelices (UseCalo) keeps the second
        tcs.append({"hits": list(a)})
        tcs.append({"hits": list(a), "calo": calo()})
    else:                                   # one helix with noise
        tcs.append({"hits": track() + noise(int(rng.integers(0, 10))), "calo": calo() if rng.random() < 0.5 else -1})
    for t in tcs:
        t.setdefault("calo", -1)
        t["hits"] = [int(i) for i in rng.permutation(t["hits"])]
        t["t0"], t["t0err"] = 1000.0 + rng.normal(0, 2), 1.5
    return hits, tcs, cls


FK = ("x", "y", "z", "udx", "udy", "uvar", "vvar", "edep", "time")


def _ref_hits(hs):
    out = {k: np.array([h[k] for h in hs], np.float32) for k in FK}
    for k in ("ncombo", "flag", "sid"):
        out[k] = np.array([h[k] for h in hs], np.int64)
    return out


def _arrays(evs, ft, A):
    ph = {k: np.zeros((E, H), ft) for k in FK}
    for k in ("ncombo", "flag", "sid"):
        ph[k] = np.zeros((E, H), np.int32)
    tc = {"valid": np.zeros((E, KT), bool), "t0": np.zeros((E, KT), ft), "t0err": np.zeros((E, KT), ft),
          "calo": np.full((E, KT), -1, np.int32)}
    cl = {"disk": np.zeros((E, C), np.int32), "time": np.zeros((E, C), ft)}
    for e, (hs, tcs, cls) in enumerate(evs):
        r = _ref_hits(hs)
        for k in ph:
            ph[k][e, :len(hs)] = r[k]
        for k, t in enumerate(tcs):
            tc["valid"][e, k] = True
            tc["t0"][e, k], tc["t0err"][e, k] = t["t0"], t["t0err"]
            tc["calo"][e, k] = t["calo"]
        for j, c in enumerate(cls):
            cl["disk"][e, j], cl["time"][e, j] = c["disk"], c["time"]
    tc["order"] = A.order_keys([[t["hits"] for t in tcs] for _, tcs, _ in evs], E, KT, H)
    return ph, tc, cl


def _events():
    rng = np.random.default_rng(11)
    kinds = ["two", "bkg", "one", "noise", "edep", "dup", "one", "two"]
    return [_event(rng, k) for k in kinds]


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_tpr_helices_match_the_transcription(precision):
    """8 synthetic events: the device seeds (count, collection order, every helix field, t0, t0err, the hit lists in
    hhits order with their hphi and flags) equal the transcription's, and MergeHelices (UseCalo) keeps the seeds the
    merger transcription keeps, with its chi2; the sample exercises prefilter removals, unique-panel sort ties, hit
    selection, calo clusters in updateT0, a seed dropped by maxEDepAvg, a cluster without a helix and a merge of
    duplicate seeds decided by the calo cluster."""
    jax = pytest.importorskip("jax")
    import robust_helix_reference as R
    import merge_helices_reference as MR
    from trigger.device import rhelix
    ft = np.float64 if precision == "float64" else np.float32
    evs = _events()
    ph, tc, cl = _arrays(evs, ft, rhelix)
    s, rs = rhelix.settings(RHF), R.settings(RHF)
    ms = rhelix.merge_settings(MERGER)
    jax.config.update("jax_enable_x64", precision == "float64")     # after the imports: some modules enable x64
    fz = np.linspace(-1500, 1500, 72).astype(np.float32)
    sd, _ = jax.jit(lambda p, t, c: rhelix.helices(p, t, c, s, K=4, NH=96, B=2))(ph, tc, cl)
    mg = jax.device_get(jax.jit(lambda x: rhelix.merge(x, H, ms, fz))(sd))
    sd = jax.device_get(sd)
    assert sd["t0"].dtype == ft and sd["rcent"].dtype == ft
    diag, nseeds, nmerged = {}, 0, 0
    ncalo = 0
    for e, (hs, tcs, cls) in enumerate(evs):
        rh = _ref_hits(hs)
        cc = {"disk": np.array([c["disk"] for c in cls], np.int64), "time": np.array([c["time"] for c in cls])}
        ref = R.robust_helices(rh, tcs, cc, rs, diag)
        ncalo += sum(r["calo"] >= 0 for r in ref)
        assert sd["overflow"][e] == 0
        assert int(sd["nseeds"][e]) == len(ref), e
        nseeds += len(ref)
        for k, r in enumerate(ref):
            m = sd["mask"][e, k]
            assert [int(x) for x in sd["ci"][e, k][m]] == r["hits"], (e, k)
            assert np.array_equal(sd["hphi"][e, k][m], np.asarray(r["hphi"], np.float32).astype(ft)), (e, k)
            assert [int(x) for x in sd["flag"][e, k][m]] == r["flags"], (e, k)
            for f in ("rcent", "fcent", "radius", "lambda", "fz0", "chi2xy", "chi2zphi", "edep_avg"):
                assert sd[f][e, k] == ft(r[f]), (e, k, f)
            assert sd["t0"][e, k] == ft(r["t0"]) and sd["t0err"][e, k] == ft(r["t0err"]), (e, k)
            assert int(sd["helicity"][e, k]) == r["helicity"] and int(sd["tc"][e, k]) == r["tc"], (e, k)
            assert int(sd["calo"][e, k]) == r["calo"], (e, k)
        hd = [{k: rh[k][i] for k in ("x", "y", "z", "udx", "udy", "uvar", "vvar")} | {"nsh": int(rh["ncombo"][i])}
              for i in range(len(hs))]
        seeds = [{**r, "i": i, "has_calo": r["calo"] >= 0,
                  "hit_copies": [{"hphi": p, "flag": fl} for p, fl in zip(r["hphi"], r["flags"])]}
                 for i, r in enumerate(ref)]
        kept = MR.merge_helices(seeds, hd, MERGER, 1 << 8)
        alive = np.flatnonzero(mg["ok"][e])
        nmerged += len(ref) - len(kept)
        assert [k["i"] for k in kept] == alive.tolist(), e
        for k, a in zip(kept, alive):
            assert mg["chi2xy"][e, a] == ft(k["chi2xy"]) and mg["chi2zphi"][e, a] == ft(k["chi2zphi"]), (e, a)
    assert nseeds >= 6 and nmerged > 0 and ncalo >= 2, (nseeds, nmerged, ncalo)
    for key in ("prefiltered", "sort_ties", "dropped_edep"):
        assert diag.get(key, 0) > 0, (key, diag)


def test_settings_reject_unsupported():
    """The port refuses the settings it does not reproduce."""
    pytest.importorskip("jax")
    from trigger.device import rhelix
    with pytest.raises(ValueError, match="UseHitMVA"):
        rhelix.settings({**RHF, "UseHitMVA": True})
    with pytest.raises(ValueError, match="TripleRadius"):
        rhelix.settings({**RHF, "HelixFitter": {**RHF["HelixFitter"], "TripleRadius": True}})


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_p_square_median_matches_the_transcription(precision):
    """prefilterHits' boost P^2 median on the device equals the transcription's, sample for sample counts 1..40."""
    jax = pytest.importorskip("jax")
    import robust_helix_reference as R
    from trigger.device import rhelix
    jax.config.update("jax_enable_x64", precision == "float64")
    rng = np.random.default_rng(4)
    v = rng.normal(0, 300, 48).astype(np.float32)
    on = rng.random(48) < 0.8
    def med(x, m):
        with rhelix.strict(m.astype(np.int32)):
            return rhelix._psq_median(x, m)

    fn = jax.jit(med)
    for n in (1, 3, 5, 6, 12, 40):
        m = on & (np.cumsum(on) <= n)
        ps = R.PSquare()
        for x in v[m]:
            ps.add(x)
        assert np.float32(jax.device_get(fn(v, m))) == ps.result(), n


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_scalar_division_is_ieee(precision):
    """rhelix._div divides by a broadcast scalar and by a constant as IEEE double division does (XLA compiles the
    plain division as a reciprocal product)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from trigger.device import rhelix
    jax.config.update("jax_enable_x64", precision == "float64")
    rng = np.random.default_rng(2)
    z = rng.uniform(-1600, 1600, 4096).astype(np.float32)
    lam = np.float32(160.0)

    def fn(z, lam):
        with rhelix.strict(jnp.zeros(2, jnp.int32)):
            zd = z.astype(jnp.float64)
            return rhelix._div(zd, lam.astype(jnp.float64)), rhelix._div(zd, 2. * np.pi)

    q, r = jax.device_get(jax.jit(fn)(z, lam))
    assert np.array_equal(q, z.astype(np.float64) / float(lam))
    assert np.array_equal(r, z.astype(np.float64) / (2. * np.pi))


def test_chain_tpr_helix_needs_deltafinder():
    pytest.importorskip("jax")
    from trigger.device import chain
    with pytest.raises(ValueError, match="deltafinder=True"):
        chain.Chain({"physics": {"producers": {}, "filters": {}, "trigger_paths": []}}, None, None, tpr_helix=True)


def test_hit_capacity_above_the_batch_hit_axis():
    """A batch whose hit axis H is narrower than the per-cluster capacity NH (the chain's hit-sorted blocks): the
    seeds equal those of a capacity that fits the axis."""
    jax = pytest.importorskip("jax")
    from trigger.device import rhelix
    jax.config.update("jax_enable_x64", True)
    ph, tc, cl = _arrays(_events(), np.float64, rhelix)
    s = rhelix.settings(RHF)
    run = lambda nh: jax.device_get(jax.jit(lambda p, t, c: rhelix.helices(p, t, c, s, K=4, NH=nh, B=2)[0])(ph, tc, cl))
    a, b = run(H), run(H + 32)
    assert np.array_equal(a["ok"], b["ok"]) and a["ok"].any()
    for k in ("t0", "rcent", "fcent", "radius", "lambda", "fz0", "chi2xy", "chi2zphi", "helicity"):
        assert np.array_equal(np.where(a["ok"], a[k], 0), np.where(b["ok"], b[k], 0)), k
