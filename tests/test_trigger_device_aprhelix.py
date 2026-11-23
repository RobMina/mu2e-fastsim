"""Device AgnosticHelixFinder and MergeHelices (UseCalo) on synthetic events against the literal transcriptions of
the stock modules (tests/agnostic_helix_reference.py, tests/merge_helices_reference.py), in both precisions."""
from pathlib import Path
import math
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# the TrigAprHelixFinder and TrigAprHelixMerger tables of the trigger menu
AHF = {"caloClusterSigma": 30, "chi2LineSaveThresh": 5, "doAverageFlag": True, "doEDepFlag": True,
       "doIsolationFlag": True, "eDepFlagThresh": 3e-3, "findMultipleHelices": True, "intenseClusterThresh": 60,
       "intenseEventThresh": 30, "isoMinHitsNear": 2, "isoRad": 150, "max2PiAmbigResidual": 2,
       "maxCircleRecoverSigma": 3, "maxDphiDz": 2.5e-2, "maxEDepAvg": 2.5e-3, "maxHelixMomentum": 400,
       "maxHelixPerpMomentum": 140, "maxLineRecoverSigma": 3, "maxNHitsRatio": 1.5, "maxPhiZResidual": 2,
       "maxSeedCircleResidual": 2, "maxSeedLineGap": 250, "maxSegmentChi2": 2, "maxTripletDz": 500,
       "maxZWindow": 700, "minDistCut": 20, "minFinalSeedHits": 7, "minHelixMomentum": 30,
       "minHelixPerpMomentum": 20, "minLineSegmentHits": 3, "minNHelixComboHits": 10, "minNHelixStrawHits": 12,
       "minSeedCircleHits": 7, "minTripletArea": 10000, "minTripletDist": 50, "minTripletDz": 50,
       "minTripletSeedZ": 0, "segMultiplier": 3, "tzSlopeSigThresh": 5, "useStoppingTarget": True,
       "validHelixDirections": ["downstream", "upstream", "unknown"]}
MERGER = {"MinHitOverlapFraction": 0.5, "MinNHitOverlap": 10, "deltanh": 5, "scaleXY": 1.1, "scaleZPhi": 0.75}
E, H, KT, C = 10, 128, 4, 4


def _zplane(p):
    return -1500.0 + 85.7 * p


def _event(rng, kind):
    """Hits (collection order), time clusters (hit lists in stored order, calo index) and calo clusters."""
    hits, tcs, cls = [], [], []

    def hit(x, y, z, t, edep=None, nsh=None):
        a = rng.uniform(0, 2 * math.pi)
        su, sv = 18.0, 3.0
        u, v = rng.normal(0, su), rng.normal(0, sv)
        hits.append({"x": x + u * math.cos(a) - v * math.sin(a), "y": y + u * math.sin(a) + v * math.cos(a),
                     "z": z, "udx": math.cos(a), "udy": math.sin(a), "uvar": su * su, "vvar": sv * sv,
                     "edep": edep if edep is not None else rng.uniform(1.2e-3, 2.6e-3), "time": t + rng.normal(0, 3),
                     "timevar": 9.0, "ncombo": int(nsh or rng.integers(1, 3)), "flag": 0})
        return len(hits) - 1

    def track(edep=None, planes=None, lam=None):
        R, f0 = rng.uniform(240, 300), rng.uniform(0, 2 * math.pi)
        rc = R + rng.uniform(-50, 50)                  # passes near the target, as a conversion electron
        lam = lam if lam is not None else rng.choice([-1, 1]) * rng.uniform(170, 260)
        phi0 = rng.uniform(0, 2 * math.pi)
        cx, cy = rc * math.cos(f0), rc * math.sin(f0)
        idx = []
        for p in planes if planes is not None else range(35, -1, -1):
            if rng.random() < 0.15:
                continue
            z = _zplane(p) + rng.choice([-10.0, 10.0])
            ph = phi0 + z / lam
            idx.append(hit(cx + R * math.cos(ph), cy + R * math.sin(ph), z, 1000 + 0.004 * z, edep))
        return idx, (cx, cy, R, phi0, lam)

    def noise(k):
        out = []
        for _ in range(k):
            r, a = rng.uniform(380, 700), rng.uniform(0, 2 * math.pi)
            out.append(hit(r * math.cos(a), r * math.sin(a), _zplane(int(rng.integers(0, 36))), 1000))
        return out

    def calo(par):
        cx, cy, R, phi0, lam = par
        ph = phi0 + 1550.0 / lam
        cls.append({"disk": 0, "cog_x": float(np.float32(cx + R * math.cos(ph))),
                    "cog_y": float(np.float32(cy + R * math.sin(ph)))})
        return len(cls) - 1

    if kind == "two":                 # two helices in one (not intense) time cluster: a second findHelix
        a, pa = track(planes=range(35, 11, -1))
        b, _ = track(planes=range(30, 6, -1))
        tcs.append({"hits": a + b + noise(6), "calo": calo(pa)})
    elif kind == "edep":              # a helix dropped by maxEDepAvg (its hits stay inHelix), and another
        a, _ = track(edep=2.9e-3, planes=range(35, 11, -1))
        b, pb = track(planes=range(33, 9, -1))
        tcs.append({"hits": b + a + noise(4), "calo": -1})
    elif kind == "intense":           # more than intenseClusterThresh hits: triplets start at the calo / target
        a, pa = track()
        b, _ = track()
        tcs.append({"hits": a + b + noise(30), "calo": calo(pa)})
    elif kind == "ties":              # many hits at equal z (std::sort ties above 16 elements), two clusters
        a, pa = track()
        for i in a[::2]:
            hits.append(dict(hits[i], x=hits[i]["x"] + 25.0, y=hits[i]["y"] - 25.0))
            a.append(len(hits) - 1)
        tcs.append({"hits": a + noise(3), "calo": -1})
        b, pb = track(planes=range(30, 5, -1))
        tcs.append({"hits": b + noise(8), "calo": calo(pb)})
    elif kind == "dup":               # the same hits in two clusters, the second with a calo cluster (as a TZ
        a, pa = track()                 # calo copy): MergeHelices keeps the seed with the calo cluster
        a = a + noise(3)
        tcs.append({"hits": list(a), "calo": -1})
        tcs.append({"hits": list(a), "calo": calo(pa)})
    else:                             # one helix with noise, sometimes a calo cluster
        a, pa = track()
        tcs.append({"hits": a + noise(int(rng.integers(0, 12))), "calo": calo(pa) if rng.random() < 0.5 else -1})
    for t in tcs:
        t["hits"] = [int(i) for i in rng.permutation(t["hits"])]
    return hits, tcs, cls


def _ref_hits(hs):
    out = {k: np.array([h[k] for h in hs], np.float32) for k in ("x", "y", "z", "udx", "udy", "uvar", "vvar",
                                                                    "edep", "time", "timevar")}
    out["ncombo"] = np.array([h["ncombo"] for h in hs], np.int64)
    out["flag"] = np.array([h["flag"] for h in hs], np.int64)
    return out


def _arrays(evs, ft):
    ph = {k: np.zeros((E, H), ft) for k in ("x", "y", "z", "udx", "udy", "uvar", "vvar", "edep", "time", "timevar")}
    ph["ncombo"] = np.zeros((E, H), np.int32)
    ph["flag"] = np.zeros((E, H), np.int32)
    tc = {"valid": np.zeros((E, KT), bool), "order": np.full((E, KT, H), -1, np.int32),
          "calo": np.full((E, KT), -1, np.int32)}
    cl = {"disk": np.zeros((E, C), np.int32), "cog_x": np.zeros((E, C), ft), "cog_y": np.zeros((E, C), ft)}
    for e, (hs, tcs, cls) in enumerate(evs):
        r = _ref_hits(hs)
        for k in ph:
            ph[k][e, :len(hs)] = r[k]
        for k, t in enumerate(tcs):
            tc["valid"][e, k] = True
            tc["order"][e, k, :len(t["hits"])] = t["hits"]
            tc["calo"][e, k] = t["calo"]
        for j, c in enumerate(cls):
            for k in cl:
                cl[k][e, j] = c[k]
    return ph, tc, cl


def _ref_cc(cls):
    return {"disk": np.array([c["disk"] for c in cls], np.int64),
            "cog_x": np.array([c["cog_x"] for c in cls], np.float64),
            "cog_y": np.array([c["cog_y"] for c in cls], np.float64)}


def _events():
    rng = np.random.default_rng(5)
    kinds = ["two", "edep", "intense", "ties", "one", "dup", "two", "one", "ties", "one"]
    return [_event(rng, k) for k in kinds]


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_apr_helices_match_the_transcription(precision):
    """10 synthetic events: the device seeds (count, collection order, every helix field, t0, the hit lists in hhits
    order and their hphi) equal the transcription's, and MergeHelices (UseCalo) keeps the seeds the merger
    transcription keeps, with its chi2; the sample exercises two helices per cluster, a seed dropped by maxEDepAvg,
    an intense cluster, z-sort ties, hit recovery, phi-line refinement and a merge decided by the calo cluster."""
    jax = pytest.importorskip("jax")
    import agnostic_helix_reference as R
    import merge_helices_reference as MR
    from trigger.device import aprhelix
    ft = np.float64 if precision == "float64" else np.float32
    evs = _events()
    ph, tc, cl = _arrays(evs, ft)
    s, rs = aprhelix.settings(AHF), R.settings(AHF)
    ms = aprhelix.merge_settings(MERGER)
    jax.config.update("jax_enable_x64", precision == "float64")     # after the imports: some modules enable x64
    fz = np.linspace(-1500, 1500, 72).astype(np.float32)
    sd, _ = jax.jit(lambda p, t, c: aprhelix.helices(p, t, c, s, K=6, NH=96, M=3, LC=12, B=4))(ph, tc, cl)
    mg = jax.device_get(jax.jit(lambda x: aprhelix.merge(x, H, ms, fz))(sd))
    sd = jax.device_get(sd)
    assert sd["t0"].dtype == ft and sd["rcent"].dtype == ft
    diag, nseeds, nmerged = {}, 0, 0
    mcfg = {**MERGER}
    for e, (hs, tcs, cls) in enumerate(evs):
        rh = _ref_hits(hs)
        ref = R.agnostic_helices(rh, tcs, _ref_cc(cls), rs, diag)
        assert sd["overflow"][e] == 0
        assert int(sd["nseeds"][e]) == len(ref), e
        nseeds += len(ref)
        for k, r in enumerate(ref):
            m = sd["mask"][e, k]
            assert [int(x) for x in sd["ci"][e, k][m]] == r["hits"], (e, k)
            assert np.array_equal(sd["hphi"][e, k][m], np.asarray(r["hphi"], np.float32).astype(ft)), (e, k)
            for f in ("rcent", "fcent", "radius", "lambda", "fz0", "chi2xy", "chi2zphi", "edep_avg"):
                assert sd[f][e, k] == ft(r[f]), (e, k, f)
            assert sd["t0"][e, k] == ft(r["t0"]) and sd["t0err"][e, k] == ft(r["t0err"]), (e, k)
            assert int(sd["helicity"][e, k]) == r["helicity"] and int(sd["tc"][e, k]) == r["tc"], (e, k)
        # MergeHelices: the transcription on the transcription's seeds
        hd = [{k: rh[k][i] for k in ("x", "y", "z", "udx", "udy", "uvar", "vvar")} | {"nsh": int(rh["ncombo"][i])}
              for i in range(len(hs))]
        seeds = [{**r, "i": i, "has_calo": r["calo"] >= 0, "hit_copies": [{"hphi": p, "flag": 0} for p in r["hphi"]]}
                 for i, r in enumerate(ref)]
        kept = MR.merge_helices(seeds, hd, mcfg, 1 << 8)
        alive = np.flatnonzero(mg["ok"][e])
        nmerged += len(ref) - len(kept)
        assert [k["i"] for k in kept] == alive.tolist(), e
        for k, a in zip(kept, alive):
            assert mg["chi2xy"][e, a] == ft(k["chi2xy"]) and mg["chi2zphi"][e, a] == ft(k["chi2zphi"]), (e, a)
    assert nseeds >= 8 and nmerged > 0, (nseeds, nmerged)
    for key in ("recovered", "refine_removed", "dropped_edep", "sort_ties"):
        assert diag.get(key, 0) > 0, (key, diag)
    assert diag["max_helices_tc"] >= 2 and diag["max_tchits"] > 62, diag


def test_settings_reject_unsupported():
    """The port refuses the settings it does not reproduce."""
    pytest.importorskip("jax")
    from trigger.device import aprhelix
    with pytest.raises(ValueError, match="validHelixDirections"):
        aprhelix.settings({**AHF, "validHelixDirections": ["downstream"]})
    with pytest.raises(ValueError, match="maxSeedCircleHits"):
        aprhelix.settings({**AHF, "maxSeedCircleHits": 20})


def _antiqsort(n):
    """McIlroy's adversary ("A killer adversary for quicksort") against the reference std::sort: keys that drive
    introsort to its depth limit, so that libstdc++ heap sorts the range."""
    import agnostic_helix_reference as R
    gas = n
    val = [gas] * n
    st = {"solid": 0, "cand": None}

    def freeze(i):
        val[i] = st["solid"]
        st["solid"] += 1

    def less(x, y):
        if val[x] == gas and val[y] == gas:
            freeze(x if x == st["cand"] else y)
        if val[x] == gas:
            st["cand"] = x
        elif val[y] == gas:
            st["cand"] = y
        return val[x] < val[y]

    R.std_sort(list(range(n)), less)
    return val


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_z_sort_heap_fallback_matches_the_transcription(precision):
    """The device z sort equals libstdc++'s std::sort (the transcription) slot for slot: adversarial keys that reach
    the introsort depth limit (heap-sort fallback, ties among the unfrozen keys), and tie-heavy random keys."""
    jax = pytest.importorskip("jax")
    import agnostic_helix_reference as R
    from trigger.device import aprhelix
    jax.config.update("jax_enable_x64", precision == "float64")
    rng = np.random.default_rng(3)
    fn = jax.jit(aprhelix._zsort)
    cases = [(np.asarray(_antiqsort(n), np.float32), np.ones(n, bool)) for n in (40, 64, 100)]
    for n in (20, 57, 90):
        key = rng.integers(0, 6, 128).astype(np.float32) * np.float32(85.7)
        sel = np.zeros(128, bool)
        sel[rng.choice(128, n, replace=False)] = True
        cases.append((key, sel))
    heaps = 0
    for key, sel in cases:
        d = {}
        ref = R.std_sort([int(i) for i in np.flatnonzero(sel)], lambda a, b: key[a] < key[b], d)
        perm, n, heap = jax.device_get(fn(key, sel))
        assert int(n) == len(ref) and [int(i) for i in perm[:len(ref)]] == ref
        assert bool(heap) == (d.get("heap_sort", 0) > 0)
        heaps += int(bool(heap))
    assert heaps >= 2, heaps


def test_chain_apr_helix_needs_tz():
    pytest.importorskip("jax")
    from trigger.device import chain
    with pytest.raises(ValueError, match="tz=True"):
        chain.Chain({"physics": {"producers": {}, "filters": {}, "trigger_paths": []}}, None, None,
                    deltafinder=True, apr_helix=True)
