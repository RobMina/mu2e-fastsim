"""Unit tests for tier 1b: rotation, and sampling bookkeeping on a synthetic library (needs the counters file)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import tier1b as T                                     # noqa: E402


def test_rotate_maps_direction_and_keeps_length():
    rng = np.random.default_rng(1)
    a, b = T._unit(rng.normal(size=(100, 3))), T._unit(rng.normal(size=(100, 3)))
    assert np.allclose(T.rotate(a, a, b), b, atol=1e-9)
    v = rng.normal(size=(100, 3))
    assert np.allclose(np.linalg.norm(T.rotate(v, a, b), axis=1), np.linalg.norm(v, axis=1))


def test_fit_inside_keeps_midpoint_off_faces():
    lo, hi = np.zeros((3, 3)), np.array([[10.0, 20.0, 30.0]] * 3)
    r0 = np.array([[-5.0, 1.0, 1.0], [9.0, 19.0, 29.0], [2.0, 2.0, 2.0]])
    r1 = np.array([[-1.0, 3.0, 3.0], [12.0, 25.0, 29.5], [4.0, 4.0, 4.0]])
    a, b = T.fit_inside(r0, r1, lo, hi)
    mid = 0.5 * (a + b)
    assert np.all(mid >= T.INSET_MM - 1e-9) and np.all(mid <= hi - T.INSET_MM + 1e-9)
    assert np.all((a >= lo) & (a <= hi) & (b >= lo) & (b <= hi))
    assert np.allclose(a[2], r0[2]) and np.allclose(b[2], r1[2])   # a step already inside is untouched


@pytest.fixture(scope="module")
def geom():
    from crvfastsim import geometry
    try:
        return geometry.from_offline_config(geometry.find_counters("Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no counters file: {e}")


def _lib(geom, bar):
    """One library crossing per bin, each with one same-bar descendant at +1 mm along the muon; one gamma shower."""
    nb = len(T.GROUPS) * T.N_PBINS * T.N_LBINS
    c = geom.center[bar]
    return {
        "libbins": pd.DataFrame([(0, g, e) for g in range(len(T.GROUPS)) for e in (1e3, 2e3, 3e3)] +
                                [(1, g, e) for g in range(len(T.GROUPS)) for e in (20.0, 22.0, 25.0, 30.0)],
                                columns=["kind", "group", "edge"]),
        "libxing": pd.DataFrame({"xid": np.arange(nb), "bin": np.arange(nb), "ux": 0.0, "uy": -1.0, "uz": 0.0}),
        "libdesc": pd.DataFrame({"xid": np.arange(nb), "sp": 7, "samebar": 1, "pdg": 11, "dt0": 0.1, "dt1": 0.2,
                                 "dx0": 0.0, "dy0": -1.0, "dz0": 0.0, "dx1": 0.0, "dy1": -2.0, "dz1": 0.0,
                                 "px0": 0.0, "py0": -1.0, "pz0": 0.0, "p1": 0.0, "len": 1.0, "edep": 0.5}),
        "libshower": pd.DataFrame({"shid": [0], "apdg": [22], "kebin": [T._ke_bin(np.array([100.0]))[0]], "ke": [100.0]}),
        "libshowersteps": pd.DataFrame({"shid": [0, 0], "sp": [1, 5], "isroot": [0, 0], "pdg": [11, -11], "bar": [bar, bar],
                                        "dt0": [5.0, 6.0], "dt1": [5.1, 6.1],
                                        **{k: [c[i % 3], c[i % 3]] for i, k in enumerate(T.STEP_COLS)},
                                        "px0": 0.0, "py0": -1.0, "pz0": 0.0, "p1": 0.0, "len": 2.0, "edep": [3.0, 4.0]}),
        "libgen": pd.DataFrame({"apdg": [22], "kebin": [T._ke_bin(np.array([100.0]))[0]], "ngen": [1], "nshower": [1]}),
    }


def test_sample_bookkeeping(geom):
    bar = 10
    c, h = geom.center[bar], geom.halflen[bar]
    top = c.copy()
    top[1] += h[1] - 0.1                                     # entry just below the bar's top face, going down
    fs = pd.DataFrame({"run": [1400], "subrun": [1], "event": [5], "nstep": [1], "bar": [bar], "sp": [1],
                       "t0": [100.0], "t1": [100.1], "x0": [top[0]], "y0": [top[1]], "z0": [top[2]],
                       "x1": [top[0]], "y1": [top[1] - 20], "z1": [top[2]], "px0": [0.0], "py0": [-2500.0],
                       "pz0": [0.0], "p1": [2490.0], "len": [20.0], "edep": [3.0]})
    fp = pd.DataFrame({"run": [1400], "subrun": [1], "event": [5], "key": [1], "parent": [-1], "pdg": [13], "isprim": [1]})
    gen = pd.DataFrame({"run": [1400, 1400], "subrun": [1, 1], "event": [5, 5], "idx": [0, 1], "pdg": [13, 22],
                        "genid": 0, "x": 0.0, "y": 0.0, "z": 0.0, "px": 0.0, "py": [-2500.0, -100.0], "pz": 0.0,
                        "e": [2502.0, 100.0], "t": [50.0, 70.0]})
    steps, parts, info = T.sample(fs, fp, gen, _lib(geom, bar), geom, np.random.default_rng(0))
    assert info["desc_dropped"] == 0 and info["showers"] == 1
    assert set(steps.sp) <= set(parts.key)                  # every step points at a particle
    assert set(parts.parent[parts.parent >= 0]) <= set(parts.key)
    d = steps[(steps.sp > 1) & (steps.sp < T.SHOWER_KEY_BASE)]
    assert len(d) == 1 and d.bar.iloc[0] == bar and np.isclose(d.t0.iloc[0], 100.1)
    assert np.isclose(d.y0.iloc[0], top[1] - 1.0)          # offset applied along the (same) muon direction
    s = steps[steps.sp >= T.SHOWER_KEY_BASE]
    assert len(s) == 2 and np.allclose(sorted(s.t0), [75.0, 76.0])
    root = parts[parts.key == T.SHOWER_KEY_BASE]
    assert len(root) == 1 and root.pdg.iloc[0] == 22 and root.isprim.iloc[0] == 1
