"""Light-propagation knobs: per-module fibre speed and attenuation length ratios (Offline CRVLightPropagation)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import chain                                          # noqa: E402
from crvfastsim.response import lookup, photons as P                  # noqa: E402

HAVE_TABLES = os.path.exists(os.path.join(lookup.lookup_dir(), "LookupTable_2370_1"))


def _table(inv_v=1 / 160.0, att=6000.0, half=1000.0):
    """Scintillation table with a known speed and attenuation: prob ~ exp(-d / att), delay index = round(d inv_v)."""
    edges = {"x": np.linspace(0, 10, 3), "y": np.linspace(-26, 26, 3), "z": np.linspace(-half, half, 41),
             "beta": np.linspace(0.4, 1, 3), "theta": np.linspace(0, np.pi, 3), "phi": np.linspace(0, np.pi, 3),
             "r": np.linspace(0, 0.7, 3)}
    d = 0.5 * (edges["z"][1:] + edges["z"][:-1]) + half
    nz, D = len(d), 40
    prob = np.tile(0.01 * np.exp(-d / att), 4).astype(np.float32)
    delays = np.zeros((4 * nz, D), np.uint8)
    delays[np.arange(4 * nz), np.tile(np.rint(d * inv_v).astype(int) + 2, 4)] = 200
    emis = np.full((4 * nz, 1), 1, np.uint8)
    T = lookup.Table(prob, delays, delays.sum(1, dtype=np.int32), emis, emis.sum(1, dtype=np.int32))
    def empty(nb):          # Cerenkov tables: right size, never used (no Cerenkov photons in these tests)
        return lookup.Table(np.zeros(nb, np.float32), np.ones((nb, 1), np.uint8), np.ones(nb, np.int32),
                            np.ones((nb, 1), np.uint8), np.ones(nb, np.int32))
    nck = 2 * 2 * nz * 2
    return lookup.LookupTable("LookupTable_synthprop", (6, 0), 0, 9.64, 25.42, half, 26.0, 2.0, 1.0, 0.7, 1.75, 0.126,
                              0.0, np.array([[0.5, 0], [1, 0.0]]), np.array([[0.5, 0], [1, 0.0]]), edges,
                              {"scint": T, "ckscint": empty(nck), "ckfiber": empty(2 * 2 * 2 * 2 * nz)})


def test_table_propagation_recovers_known_values():
    inv_v, att = P.table_propagation(_table())
    assert abs(1 / inv_v - 160.0) < 2.0 and abs(att - 6000.0) < 30.0


@pytest.mark.skipif(not HAVE_TABLES, reason="CRV lookup tables not available (Mu2e-internal)")
def test_table_propagation_matches_offline():
    for name, v, att in (("6000_0", 166.464, 6618.02), ("2370_1", 164.23, 7864.6), ("3200_0", 163.686, 5168.96)):
        inv_v, a = P.table_propagation(lookup.load(name))        # Offline FindTablePropagation (debug 1)
        assert abs(1 / inv_v - v) < 0.01 and abs(a - att) < 0.1


class _G:
    sector_names = ["EX", "T1", "T2"] + [f"M{i}" for i in range(1, 9)]
    sector = np.r_[np.zeros(256), np.ones(128), np.full(128, 2), np.repeat(np.arange(3, 11), 8)].astype(int)
    module = np.r_[np.arange(256) // 64, np.arange(128) // 64, np.arange(128) // 64, np.zeros(64)].astype(int)
    nbars = 576


def test_bar_keys_and_ratios():
    sk, mk = P.bar_keys(_G)
    assert mk[0] == "EX0" and mk[255] == "EX3" and mk[256] == "T10" and mk[511] == "T21" and mk[512] == "M1"
    r = P.bar_ratios({"EX": 1.08, "EX3": 1.1, "TAG": 0.9, "M2": 0.8}, _G)
    assert r[0] == 1.08 and r[200] == 1.1 and r[300] == 1.0 and r[512] == 0.9 and r[520] == 0.8
    with pytest.raises(KeyError):
        P.bar_ratios({"EX7": 1.0}, _G)


def test_knobs_set_ratios():
    cfg = chain.apply_knobs(chain.ChainConfig(), {"speed_EX": 1.08, "atten_T10": 0.55, "atten_T1": 0.6})
    assert cfg.response.speed_ratio == {"EX": 1.08} and cfg.response.atten_ratio == {"T10": 0.55, "T1": 0.6}
    assert chain.KNOB_TABLE["atten_T10"] == ("attenuationLengthRatio", "T10")
    with pytest.raises(ValueError):
        chain.apply_knobs(chain.ChainConfig(), {"speed_EX": 0})


def _rows(L, R, rng, mu=0.0, k=0.0):
    z = rng.uniform(-0.95, 0.95, R) * L.half_length
    a = np.c_[rng.uniform(-9, 9, R), np.full(R, 12.0), z]
    d = np.c_[np.full(R, 0.05), np.zeros(R), np.zeros(R)]
    t0 = k * (z + L.half_length)
    return {"s": a, "d": d, "t0": t0, "t1": t0, "nsub": np.ones(R, np.int64), "beta": np.ones(R),
            "theta": np.full(R, np.pi / 2), "dev": np.ones(R), "mu": np.full(R, mu), "step": np.arange(R),
            "chan": np.arange(R) * 4, "n_scint": np.full(R, 200), "n_ck_s": np.zeros(R, np.int64),
            "n_ck_f": np.zeros(R, np.int64)}


def _slope(L, rows, s):
    """d ln(arrivals per row) / dz."""
    z = rows["s"][:, 2]
    zb = np.linspace(-L.half_length, L.half_length, 11)
    n = np.histogram(z[s], zb)[0] / np.histogram(z, zb)[0]
    return np.polyfit(0.5 * (zb[1:] + zb[:-1]), np.log(n), 1)[0]


def test_attenuation_shift_numpy():
    L = _table()
    rng = np.random.default_rng(101)                       # row and kernel seeds differ (same seed = correlated draws)
    base = _rows(L, 100000, rng)
    s0, _, _ = P._substep_photons(L, base, np.random.default_rng(1))
    mu = (1 / 0.6 - 1) / 6000.0
    att = dict(base, mu=np.full(len(base["nsub"]), mu))
    s1, _, _ = P._substep_photons(L, att, np.random.default_rng(1))
    d_slope = _slope(L, att, s1) - _slope(L, base, s0)
    assert abs(d_slope + mu) < 0.25 * mu                    # extra log-slope -mu (prob ~ exp(-mu z)); noise ~0.05 mu
    zc = np.abs(base["s"][:, 2]) < 0.05 * L.half_length
    n0c, n1c = np.sum(zc[s0]), np.sum(zc[s1])
    assert abs(n1c / n0c - 1) < 5 / np.sqrt(n0c)            # yield at the bar centre unchanged


def test_attenuation_jax_matches_numpy():
    jax = pytest.importorskip("jax")
    from crvfastsim.response import photons_jax as PJ
    L = _table()
    rng = np.random.default_rng(202)
    mu = (1 / 0.6 - 1) / 6000.0
    rows = _rows(L, 100000, rng, mu=mu)
    s1, _, _ = P._substep_photons(L, rows, np.random.default_rng(1))
    s2, _, _ = PJ.substep_photons(PJ.DeviceTable(L), rows, jax.random.PRNGKey(1))
    assert abs(len(s2) / len(s1) - 1) < 4 / np.sqrt(len(s1))
    assert abs(_slope(L, rows, s2) - _slope(L, rows, s1)) < 0.25 * mu   # pair noise ~0.05 mu at 1e5 rows
    no_mu = {k: v for k, v in rows.items() if k != "mu"}
    zero = dict(rows, mu=np.zeros(len(rows["nsub"])))
    a = PJ.substep_photons(PJ.DeviceTable(L), no_mu, jax.random.PRNGKey(1))
    b = PJ.substep_photons(PJ.DeviceTable(L), zero, jax.random.PRNGKey(1))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))   # rows without mu = mu 0, bitwise
    c = P._substep_photons(L, no_mu, np.random.default_rng(1))
    d = P._substep_photons(L, zero, np.random.default_rng(1))
    assert all(np.array_equal(x, y) for x, y in zip(c, d))


class _G1:
    """One EX-like bar along Mu2e z, centred at the origin."""
    sector_names = ["EX"]
    sector = np.zeros(1, int)
    module = np.zeros(1, int)
    nbars = 1
    center = np.zeros((1, 3))
    local_to_world = np.array([[1, 0, 2]])


def test_speed_ratio_through_iter_rows():
    import pandas as pd
    L = _table()
    rng = np.random.default_rng(7)
    n = 400
    z = rng.uniform(-900, 900, n)
    st = pd.DataFrame({"bar": np.zeros(n, int), "edep": np.full(n, 1.0), "len": np.full(n, 0.05),
                       "t0": np.zeros(n), "t1": np.zeros(n), "x0": np.zeros(n), "y0": np.full(n, 12.0), "z0": z,
                       "x1": np.full(n, 0.05), "y1": np.full(n, 12.0), "z1": z, "px0": np.zeros(n),
                       "py0": np.zeros(n), "pz0": np.full(n, 1e4), "p1": np.full(n, 1e4)})
    pdg = np.full(n, 13)
    dev = np.zeros(4)
    cfg = P.ResponseConfig(sectors={"EX": P.SectorConfig("synthprop", 0, 1000.0)}, photon_yield_scale=1.0)
    inv_v, _ = P.table_propagation(L)
    out = {}
    for r in (1.0, 1.25):
        c = P.ResponseConfig(sectors=cfg.sectors, photon_yield_scale=1.0, speed_ratio={"EX": r} if r != 1 else {})
        rows = list(P.iter_rows(st, pdg, _G1, c, {"synthprop": L}, dev, np.random.default_rng(1)))
        sipm0 = rows[0][1]                                    # SiPM 0: readout at local z = -half_length
        out[r] = np.polyfit(sipm0["s"][:, 2], sipm0["t0"], 1)[0]
    assert abs(out[1.0]) < 1e-12
    assert abs(out[1.25] - inv_v * (1 / 1.25 - 1)) < 1e-12   # extra delay per mm to the SiPM end
