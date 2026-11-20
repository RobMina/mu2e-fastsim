"""DeltaFinder (TrigFlagPH) port: settings, the nominal geometry, and the device flags against the literal stock
transcription (tests/deltafinder_reference.py) on synthetic events with a delta electron, a proton and noise hits."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

jax = pytest.importorskip("jax")                  # deltafinder.py is a JAX module

from trigger.device import deltafinder as dfm  # noqa: E402

# TrigFlagPH of the Run 1A trigger menu (Offline v13_41_01)
FINDER = {"debugLevel": 0, "diagLevel": 0, "flagProtonHits": 1, "goodHitMask": ["TimeDivision"], "maxChi2Par": 12,
          "maxChi2Perp": 12, "maxChi2Seed": 12, "maxChi2SeedDelta": 20, "maxDeltaEDep": 4e-3, "maxDriftTime": 40,
          "maxDtDc": 10, "maxDtDs": 10, "maxEleHitEnergy": 2e-2, "maxGap": 3, "maxHitDt": 30, "maxHitSeedDt": 10,
          "maxSeedDt": 30, "maxSeedEDep": 5e-3, "maximumTime": 1750, "mergePC": 0, "minDeltaNHits": 5, "minNSeeds": 2,
          "minProtonHitEDep": 3e-3, "minProtonSeedEDep": 3e-3, "pickupProtonHits": 1, "printErrors": 0, "rCore": 2,
          "sigmaR": 1.5, "testHitMask": False, "testOrder": 0, "timeBin": 40, "bkgHitMask": []}
MODULE = {"module_type": "DeltaFinder", "finderParameters": FINDER, "writeFilteredComboHits": 0, "writeStrawHits": 0,
          "bkgHitMask": [], "testHitMask": False}


def test_settings():
    s = dfm.settings(MODULE)
    assert s["minNSeeds"] == 2 and s["timeBin"] == 40.0 and s["flagProtonHits"] == 1
    with pytest.raises(ValueError):
        dfm.settings({**MODULE, "writeFilteredComboHits": 1})
    with pytest.raises(ValueError):
        dfm.settings({**MODULE, "finderParameters": {**FINDER, "testHitMask": True}})


def test_geometry():
    g = dfm.geometry()
    assert g["nx"].shape == (18, 4, 3)
    assert np.allclose(g["nx"] ** 2 + g["ny"] ** 2, 1.0, atol=1e-6)
    assert np.allclose(g["wx"] * g["nx"] + g["wy"] * g["ny"], 0.0, atol=1e-6)      # normals perpendicular to the wires
    seen = {dfm.order_id(s, p, n) for s in range(18) for p in range(2) for n in range(6)}
    assert len(seen) == 18 * 12                                                     # one slot per panel
    z = g["stationZ"]
    assert np.all(np.diff(z) > 0) and abs(z[0] + 1479.0) < 1e-3                     # station 0 between its planes
    assert np.all(np.diff(g["fz"], axis=1) != 0)                                    # faces at distinct z


def _panel(st, zf, pf):
    for plane in (2 * st, 2 * st + 1):
        for panel in range(6):
            if dfm.order_id(st, plane % 2, panel) == (st, zf, pf):
                return plane, panel
    raise AssertionError


def _event(rng, g, delta=True, proton=True, nnoise=40):
    hits = []

    def add(st, zf, pf, x, y, t, ct, edep):
        plane, panel = _panel(st, zf, pf)
        hits.append(dict(sid=(plane << 10) | (panel << 7) | int(rng.integers(0, 96)), x=x, y=y, t=t, ct=ct, edep=edep,
                         wx=g["wx"][st, zf, pf], wy=g["wy"][st, zf, pf]))

    def track(cx, cy, stations, t0, edep):
        u = np.array([cx, cy]) / np.hypot(cx, cy)
        for st in stations:
            for zf in range(4):
                pf = int(np.argmax(g["nx"][st, zf] * u[0] + g["ny"][st, zf] * u[1]))
                w = np.array([g["wx"][st, zf, pf], g["wy"][st, zf, pf]])
                n = np.array([w[1], -w[0]])
                p = np.array([cx, cy]) + rng.uniform(-6, 6) * w + rng.uniform(-1, 1) * n
                ct = t0 + 1.5 * (st - stations[0]) + rng.uniform(0, 3)
                add(st, zf, pf, p[0], p[1], ct + rng.uniform(0, 15), ct, edep * rng.uniform(0.8, 1.2))

    if delta:
        a = rng.uniform(0, 2 * np.pi)
        track(520 * np.cos(a), 520 * np.sin(a), list(range(3, 9)), rng.uniform(550, 900), 0.002)
    if proton:
        a = rng.uniform(0, 2 * np.pi)
        track(420 * np.cos(a), 420 * np.sin(a), list(range(11, 14)), rng.uniform(600, 1200), 0.008)
    for _ in range(nnoise):
        st, zf, pf = int(rng.integers(0, 18)), int(rng.integers(0, 4)), int(rng.integers(0, 3))
        w = np.array([g["wx"][st, zf, pf], g["wy"][st, zf, pf]])
        n = np.array([g["nx"][st, zf, pf], g["ny"][st, zf, pf]])
        p = rng.uniform(380, 680) * n + rng.uniform(-300, 300) * w
        ct = rng.uniform(450, 1700)
        add(st, zf, pf, p[0], p[1], ct + rng.uniform(0, 20), ct, rng.uniform(0.0005, 0.006))
    order = rng.permutation(len(hits))
    return [hits[i] for i in order]


def _arrays(evs, H):
    E = len(evs)
    ph = {k: np.zeros((E, H), dt) for k, dt in (("valid", bool), ("sid", np.int32), ("flag", np.uint32),
                                                ("etime", np.float32), ("time", np.float32), ("x", np.float32),
                                                ("y", np.float32), ("udx", np.float32), ("udy", np.float32),
                                                ("uvar", np.float32), ("edep", np.float32), ("ncombo", np.int32))}
    ph["spill"] = np.ones(E, np.int32)
    for i, ev in enumerate(evs):
        for j, h in enumerate(ev):
            ph["valid"][i, j] = True
            ph["sid"][i, j] = h["sid"]
            ph["flag"][i, j] = (1 << 10) | (1 << 3) | (1 << 21)
            ph["etime"][i, j], ph["time"][i, j] = h["t"], h["ct"]
            ph["x"][i, j], ph["y"][i, j] = h["x"], h["y"]
            ph["udx"][i, j], ph["udy"][i, j] = h["wx"], h["wy"]
            ph["uvar"][i, j], ph["edep"][i, j], ph["ncombo"][i, j] = 64.0, h["edep"], 1
    return ph


@pytest.mark.slow
def test_port_matches_the_stock_transcription():
    import deltafinder_reference as ref
    g = dfm.geometry()
    s = dfm.settings(MODULE)
    rng = np.random.default_rng(20261120)
    evs = [_event(rng, g) for _ in range(5)] + [_event(rng, g, delta=False), _event(rng, g, proton=False, nnoise=10),
                                                _event(rng, g, delta=False, proton=False, nnoise=0)]
    ph = _arrays(evs, 128)
    flag, np15, err, cnt = (np.asarray(x) for x in dfm.flag_hits(ph, s, g))
    geo = ref.geo_from(g)
    nbkg = 0
    for i, ev in enumerate(evs):
        n = len(ev)
        hits = {"sid": ph["sid"][i, :n], "flag": ph["flag"][i, :n], "time": ph["etime"][i, :n], "ctime": ph["time"][i, :n],
                "x": ph["x"][i, :n], "y": ph["y"][i, :n], "udx": ph["udx"][i, :n], "udy": ph["udy"][i, :n],
                "uvar": ph["uvar"][i, :n], "edep": ph["edep"][i, :n], "nsh": ph["ncombo"][i, :n]}
        want, want15 = ref.run(hits, geo, s)
        assert err[i] == 0
        np.testing.assert_array_equal(flag[i, :n], want, err_msg=f"event {i}")
        assert np15[i] == want15
        nbkg += int(((want >> dfm.BKG_BIT) & 1).sum())
    assert nbkg >= 5 * 20                       # the delta electrons are flagged
    assert ((flag[:5, :] >> dfm.ESEL_BIT) & 1).sum() < ph["valid"][:5].sum()   # some proton hits lose EnergySelection
