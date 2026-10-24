"""Named run configurations (crvfastsim.runconfig): sector settings cover every sector of the counters file, the
extracted default is unchanged, labels fit the full CRV, the two-level crossings equal the flat ones, the finder cuts
reach the chain, and the transport runs on the Run 1A and Run 2 geometries (synthetic slab world, no GDML)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import chain, crossings, runconfig                     # noqa: E402
from crvfastsim import transport as TR                                  # noqa: E402
from crvfastsim.reco import coinc as C                                  # noqa: E402
from crvfastsim.response import photons as PH                          # noqa: E402
from mu2efastsim import crossings as CX                                 # noqa: E402


def _geom(name):
    try:
        return runconfig.get(name).geometry()
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("name", runconfig.names())
def test_every_sector_configured(name):
    rc, g = runconfig.get(name), _geom(name)
    assert set(g.sector_names) <= set(rc.sectors) and set(g.sector_names) <= set(rc.cuts)
    assert 4 * g.nbars < chain.CHAN_MULT
    if rc.lib_group:
        assert set(rc.group_of_sector(g)) <= {"EX", "T1", "T2", "M"}


def test_extracted_default_unchanged():
    rc = runconfig.get()
    assert rc.sectors == PH.EXTRACTED_V05 and rc.cuts == C.EXTRACTED_V04 and rc.window == (0.0, 1e5)
    assert rc.photon_yield_scale == PH.ResponseConfig().photon_yield_scale
    g = _geom("extracted_v04")
    assert not hasattr(g, "lib_group")
    assert (rc.tagger_bar(g) == (np.arange(g.nbars) >= TR.TAGGER_FIRST_BAR)).all()
    assert TR.msc_plane(g) == TR.Y_MSC_PLANE
    assert np.array_equal(TR.band_edges(TR.Y_MSC_PLANE), TR.BAND_EDGES_Y)


def test_run2_counts_and_planes():
    g = _geom("run2_v10")
    assert g.nbars == 5344 and len(g.sector_names) == 23
    assert TR.msc_plane(g) == np.ceil((g.center[:, 1] + g.halflen[:, 1]).max() + 0.8)
    e = TR.band_edges(TR.msc_plane(g))
    assert e[-1] == TR.msc_plane(g) and np.all(np.diff(e) < 0)


def test_window_override():
    assert runconfig.get("run2_v10").window == runconfig.ONSPILL_WINDOW
    assert runconfig.get("run2_v10", "offspill").window == runconfig.OFFSPILL_WINDOW


@pytest.mark.parametrize("name", ["run1a_v01", "run2_v10"])
def test_two_level_crossings_equal_flat(name):
    g = _geom(name)
    rng = np.random.default_rng(2)
    n = 3000
    lo, hi = (g.center - g.halflen).min(0), (g.center + g.halflen).max(0)
    o = np.c_[rng.uniform(lo[0] - 1000, hi[0] + 1000, n), np.full(n, 8000.0), rng.uniform(lo[2] - 1000, hi[2] + 1000, n)]
    d = np.c_[rng.normal(0, .5, n), -np.ones(n), rng.normal(0, .5, n)]
    d /= np.linalg.norm(d, axis=1)[:, None]
    a = crossings.crossings(o, d, g)
    blo, bhi = crossings.bar_bounds(g)
    b = CX.crossings(o, d, blo, bhi, chunk=500, backend="numpy")
    assert len(a) > 100 and np.array_equal(a, b)


def test_finder_reaches_chain():
    cfg = runconfig.get("run1a_v01").chain_config()
    assert cfg.finder.cuts["T1"].max_dt == 20 and cfg.finder.ignore_secondary_pulses is False
    assert cfg.response.photon_yield_scale == 0.90 and cfg.window == runconfig.ONSPILL_WINDOW


def test_nominal_conditions_cover_channels():
    g = _geom("run2_v10")
    c = runconfig.nominal_conditions(g)
    assert len(c.status) == 4 * g.nbars and (c.status == 0).all() and c.feb.max() < 24


@pytest.mark.parametrize("name", ["run1a_v01", "run2_v10"])
def test_transport_runs(name):
    g = _geom(name)
    yp = TR.msc_plane(g)
    rng = np.random.default_rng(4)
    n = 3000
    lo, hi = (g.center - g.halflen).min(0), (g.center + g.halflen).max(0)
    x, z = rng.uniform(lo[0], hi[0], n), rng.uniform(lo[2], hi[2], n)
    d = np.c_[rng.normal(0, .05, n), -np.ones(n), rng.normal(0, .05, n)]
    d /= np.linalg.norm(d, axis=1)[:, None]
    L = (15000.0 - yp) / -d[:, 1]
    o = np.c_[x - d[:, 0] * L, np.full(n, 15000.0), z - d[:, 2] * L]
    p = 3000.0
    gen = pd.DataFrame({"run": 1400, "subrun": 1, "event": np.arange(n, dtype=np.uint32), "idx": 0, "pdg": 13,
                        "genid": 0, "x": o[:, 0], "y": o[:, 1], "z": o[:, 2], "px": d[:, 0] * p, "py": d[:, 1] * p,
                        "pz": d[:, 2] * p, "e": np.sqrt(p ** 2 + 105.658 ** 2), "t": 0.0})
    P = np.array([[1, 0, 0, 5e4], [-1, 0, 0, 5e4], [0, 1, 0, yp + 2000], [0, -1, 0, -(yp + 1000)], [0, 0, 1, 1e6],
                  [0, 0, -1, 1e6]], float)
    world = TR.banded_world([("CONCRETE_MARS", P, np.array([-5e4, yp + 1000, -1e6]), np.array([5e4, yp + 2000, 1e6]))],
                            TR.band_edges(yp))
    st, pa = TR.transport(gen, g, np.random.default_rng(1), stage="C", world=world,
                          tagger_bar=runconfig.get(name).tagger_bar(g))
    assert len(st) > n and st.bar.max() < g.nbars and (st.edep > 0).all()
