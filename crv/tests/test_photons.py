"""Unit tests for response/photons.py helpers; the lookup-table tests run only where the (Mu2e-internal) tables are."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim.response import lookup, photons as P                   # noqa: E402


def test_find_bin_matches_stock_inclusive_edges():
    e = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.array([-0.1, 0.0, 0.5, 1.0, 2.999, 3.0, 3.1])
    assert P.find_bin(e, v).tolist() == [-1, 0, 0, 0, 2, 2, -1]   # stock: first i with e[i] <= v <= e[i+1]


def test_cerenkov_interp_and_charge():
    cmap = np.array([[0.5, 0.0], [0.6, 10.0], [1.0, 50.0]])
    assert np.allclose(P.cerenkov_per_mm(np.array([0.25, 0.55, 0.8, 1.2]), np.array([1, 1, -2, 1]), cmap),
                       [0.0, 5.0, 60.0, 50.0])


def test_beta_and_particles():
    m, q, known = P.particle_props(np.array([13, -11, 22, 2212, 1000020040, 999]))
    assert q.tolist() == [-1, 1, 0, 1, 2, 0] and known.tolist() == [True] * 5 + [False]
    b = P.beta_of(np.array([105.66, 0.0]), np.array([105.66, 0.0]), np.array([105.66, 0.0]))
    assert np.isclose(b[0], 1 / np.sqrt(2)) and b[1] == 1.0


@pytest.mark.skipif(not os.path.exists(os.path.join(lookup.lookup_dir(), "LookupTable_3200_0")),
                    reason="CRV lookup tables not available (Mu2e-internal)")
def test_lookup_table_reads_completely():
    L = lookup.load("3200_0")
    assert L.version[0] == 6 and L.reflector == 0 and np.isclose(L.half_length, 1600.0)
    n = (len(L.edges["x"]) - 1) * (len(L.edges["y"]) - 1) * (len(L.edges["z"]) - 1)
    assert L.tables["scint"].prob.shape == (n,)


def _fake_table(rng):
    edges = {"x": np.linspace(0, 10, 5), "y": np.linspace(-26, 26, 9), "z": np.linspace(-1000, 1000, 11),
             "beta": np.linspace(0.4, 1, 4), "theta": np.linspace(0, np.pi, 5), "phi": np.linspace(0, np.pi, 4),
             "r": np.linspace(0, 0.7, 4)}

    def tab(n, D=20, E=3):
        d = rng.integers(0, 40, (n, D)).astype(np.uint8)
        e = rng.integers(0, 40, (n, E)).astype(np.uint8)
        return lookup.Table(rng.uniform(0, 0.01, n).astype(np.float32), d, d.sum(1, dtype=np.int32), e,
                            e.sum(1, dtype=np.int32))

    T = {"scint": tab(320), "ckscint": tab(960), "ckfiber": tab(1080)}
    return lookup.LookupTable("LookupTable_fake", (6, 0), 0, 9.64, 25.42, 1000.0, 26.0, 2.0, 1.0, 0.7, 1.75, 0.126,
                              0.0, np.array([[0.5, 0], [1, 200.0]]), np.array([[0.5, 0], [1, 200.0]]), edges, T)


def _rows(rng, R=3000):
    a = np.c_[rng.uniform(-9, 9, R), rng.uniform(-25, 25, R), rng.uniform(-900, 900, R)]
    d = np.c_[rng.normal(0, 5, R), rng.normal(0, 5, R), rng.normal(0, 30, R)]
    t0 = rng.uniform(0, 1e3, R)
    return {"s": a, "d": d, "t0": t0, "t1": t0 + 0.1, "nsub": np.maximum((np.linalg.norm(d, axis=1) / 0.1).astype(np.int64), 1),
            "beta": rng.uniform(0.6, 1, R), "theta": rng.uniform(0, np.pi, R), "dev": np.ones(R), "step": np.arange(R),
            "chan": np.arange(R) * 4, "n_scint": rng.integers(0, 30, R), "n_ck_s": rng.integers(0, 3, R),
            "n_ck_f": rng.integers(0, 2, R)}


def test_jax_substeps_match_numpy_statistically():
    jax = pytest.importorskip("jax")
    from crvfastsim.response import photons_jax as PJ
    rng = np.random.default_rng(0)
    L, rows = _fake_table(rng), _rows(rng)
    s1, _, t1 = P._substep_photons(L, rows, np.random.default_rng(1))
    s2, _, t2 = PJ.substep_photons(PJ.DeviceTable(L), rows, jax.random.PRNGKey(1))
    assert abs(len(t2) / len(t1) - 1) < 4 / np.sqrt(len(t1))
    d1, d2 = t1 - rows["t0"][s1], t2 - rows["t0"][s2]
    assert abs(d1.mean() - d2.mean()) < 0.1
    s3, _, t3 = PJ.substep_photons(PJ.DeviceTable(L), rows, jax.random.PRNGKey(1))
    assert np.array_equal(t2, t3)                                     # same key -> bit-identical


def test_sipm_single_photons_full_pe_and_crosstalk_rate():
    from crvfastsim.response import sipm
    rng = np.random.default_rng(3)
    cfg = sipm.SiPMConfig()
    n = 200_000                                    # one photon per SiPM: every firing has v = Vov -> 1 PE
    out = sipm.simulate(np.arange(n), np.zeros(n), np.arange(n), np.ones((40, 40)), cfg, rng)
    assert np.allclose(out["pe"], 1.0)
    p = cfg.p_fire(cfg.overvoltage)                # 0.407: p1 (1 - exp(-3 / 2.7))
    live = 1 - 0.1 * 0.1                           # dead 4 x 4 centre: 4/40 per axis
    ct = 4 * (1 - (1 - cfg.cross_talk) ** 0.25)    # expected cross-talk firings per firing (edges ignored)
    assert abs(len(out["pe"]) / n - p * live * (1 + ct)) < 0.004


def test_sipm_recovery_reduces_second_firing():
    from crvfastsim.response import sipm
    cfg = sipm.SiPMConfig(cross_talk=0.0)
    pmap = np.zeros((40, 40)); pmap[5, 5] = 1.0    # lands on pixels 5-6 x 5-6
    n = 50_000
    out = sipm.simulate(np.repeat(np.arange(n), 2), np.tile([0.0, 1.0], n), np.arange(2 * n), pmap, cfg,
                        np.random.default_rng(4))
    late = out["pe"][out["time"] == 1.0]
    v1 = 1 - np.exp(-1 / cfg.time_constant)        # a pixel that fired 1 ns earlier is at 7 % of Vov
    assert np.isclose(late.min(), v1) and np.isclose(late.max(), 1.0)


def test_sipm_jax_matches_numpy_statistically():
    jax = pytest.importorskip("jax")
    from crvfastsim.response import sipm, sipm_jax
    rng = np.random.default_rng(5)
    cfg, pmap = sipm.SiPMConfig(), rng.random((40, 40)) ** 4
    n = rng.poisson(60, 3000)
    chan = np.repeat(np.arange(3000), n)
    t = np.repeat(rng.uniform(0, 1e4, 3000), n) + rng.normal(0, 3, len(chan))
    F = sipm.simulate(chan, t, np.arange(len(chan)), pmap, cfg, np.random.default_rng(1))
    J = sipm_jax.simulate(chan, t, np.arange(len(chan)), pmap, cfg, np.random.default_rng(2), jax.random.PRNGKey(2))
    assert abs(J["pe"].sum() / F["pe"].sum() - 1) < 4 / np.sqrt(len(F["pe"]))
    assert abs((J["pe"] < 0.99).mean() - (F["pe"] < 0.99).mean()) < 0.003
