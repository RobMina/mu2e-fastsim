"""Stage-C transport options: defaults unchanged; --msc-tail / --sliced-msc run and widen the large-angle deflection
(synthetic concrete slab above the CRV; needs only the Offline counters file)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import geometry as geo                                  # noqa: E402
from crvfastsim import transport as TR                                  # noqa: E402


@pytest.fixture(scope="module")
def g():
    try:
        return geo.from_offline_config(geo.find_counters("Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"))
    except FileNotFoundError as e:
        pytest.skip(str(e))


def _slab(y_lo, y_hi, half=50000.0):
    P = np.array([[1, 0, 0, half], [-1, 0, 0, half], [0, 1, 0, y_hi], [0, -1, 0, -y_lo], [0, 0, 1, 1e6],
                  [0, 0, -1, 1e6]], float)
    return [("CONCRETE_MARS", P, np.array([-half, y_lo, -1e6]), np.array([half, y_hi, 1e6]))]


def _gen(g, n=4000, p=800.0, seed=3):
    """Vertical-ish soft muons aimed at the EX footprint, from y = 15 m."""
    rng = np.random.default_rng(seed)
    lo, hi = (g.center - g.halflen).min(0), (g.center + g.halflen).max(0)
    x = rng.uniform(lo[0] + 500, hi[0] - 500, n)
    z = rng.uniform(lo[2] + 500, hi[2] - 500, n)
    d = np.c_[rng.normal(0, .05, n), -np.ones(n), rng.normal(0, .05, n)]
    d /= np.linalg.norm(d, axis=1)[:, None]
    y0 = 15000.0
    L = (y0 - TR.Y_MSC_PLANE) / -d[:, 1]
    o = np.c_[x - d[:, 0] * L, np.full(n, y0), z - d[:, 2] * L]
    return pd.DataFrame({"run": 1400, "subrun": 1, "event": np.arange(n, dtype=np.uint32), "idx": 0, "pdg": 13,
                         "genid": 0, "x": o[:, 0], "y": o[:, 1], "z": o[:, 2], "px": d[:, 0] * p, "py": d[:, 1] * p,
                         "pz": d[:, 2] * p, "e": np.sqrt(p ** 2 + 105.658 ** 2), "t": 0.0})


def _entry_angle(gen, steps):
    first = steps.sort_values(["event", "y0"], ascending=[True, False]).drop_duplicates("event").set_index("event")
    g0 = gen.set_index("event").loc[first.index]
    d0 = g0[["px", "py", "pz"]].to_numpy()
    d1 = first[["px0", "py0", "pz0"]].to_numpy()
    c = (d0 * d1).sum(1) / np.linalg.norm(d0, axis=1) / np.linalg.norm(d1, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def test_defaults_unchanged(g):
    gen, world = _gen(g), _slab(6000.0, 7000.0)
    a, _ = TR.transport(gen, g, np.random.default_rng(1), stage="C", world=world)
    b, _ = TR.transport(gen, g, np.random.default_rng(1), stage="C", world=world, msc_tail=False, sliced_msc=False,
                        all_muons=False)                                 # one muon per event: all_muons is a no-op
    pd.testing.assert_frame_equal(a, b)


def test_tail_widens_large_angles(g):
    gen, world = _gen(g, n=20000), _slab(6000.0, 6050.0)                     # 5 cm of concrete: tail-dominated q99.9
    base, _ = TR.transport(gen, g, np.random.default_rng(2), stage="C", world=world)
    tail, _ = TR.transport(gen, g, np.random.default_rng(2), stage="C", world=world, msc_tail=True, tail_k=2.0)
    ab, at = _entry_angle(gen, base), _entry_angle(gen, tail)
    assert np.percentile(at, 99.9) > 1.5 * np.percentile(ab, 99.9)        # single scatters beyond the Gaussian
    assert abs(np.median(at) / np.median(ab) - 1) < 0.05                  # the core is unchanged
    assert abs(len(tail.event.unique()) / len(base.event.unique()) - 1) < 0.02


def test_sliced_only_runs(g):
    gen, world = _gen(g, n=500), _slab(6000.0, 7000.0)
    s, p = TR.transport(gen, g, np.random.default_rng(5), stage="C", world=world, sliced_msc=True)
    assert len(s) > 0 and (s.edep > 0).all()


def test_all_muons(g):
    gen = _gen(g, n=3000)
    second = gen.iloc[::10].copy()                                        # every 10th event gets a second muon
    second["idx"] = 1
    second["x"] += 300.0
    gen2 = pd.concat([gen, second], ignore_index=True)
    one, _ = TR.transport(gen, g, np.random.default_rng(4), stage="B")
    one_f, _ = TR.transport(gen, g, np.random.default_rng(4), stage="B", all_muons=False)
    pd.testing.assert_frame_equal(one, one_f)                             # single-muon events: identical either way
    a, pa = TR.transport(gen2, g, np.random.default_rng(4), stage="B", all_muons=False)
    assert set(pa.key) == {1}                                             # --first-muon-only: the earlier behaviour
    c, pc = TR.transport(gen2, g, np.random.default_rng(4), stage="B")    # default: every muon
    assert set(pc.key) == {1, 2} and (pc.key == 2).sum() == len(second)
    assert len(c) > len(a) and set(c.sp) <= {1, 2}
    ev2 = set(second.event)
    assert set(c[c.sp == 2].event) <= ev2                                 # second-muon steps only where it exists


def test_absorbers_default_off_and_effect(g):
    gen = _gen(g, n=4000, p=400.0)                                         # soft muons, aimed at EX (T1 is under it)
    a, _ = TR.transport(gen, g, np.random.default_rng(8), stage="B")
    b, _ = TR.transport(gen, g, np.random.default_rng(8), stage="B", absorbers=None)
    pd.testing.assert_frame_equal(a, b)                                    # default: no absorbers, unchanged
    lo, hi = (g.center - g.halflen).min(0), (g.center + g.halflen).max(0)
    ex_bot = (g.center - g.halflen)[np.isin(g.sector, [0]), 1].min()
    t1_top = (g.center + g.halflen)[np.isin(g.sector, [1]), 1].max()
    sheet = (np.array([[lo[0], t1_top + 5.0, lo[2]]]), np.array([[hi[0], ex_bot - 5.0, hi[2]]]), "G4_Al")  # gap EX/T1
    c, _ = TR.transport(gen, g, np.random.default_rng(8), stage="B", absorbers=sheet)
    pmag = lambda s: np.linalg.norm(s[["px0", "py0", "pz0"]].to_numpy(), axis=1)
    above = lambda s: s[s.y0 > ex_bot - 1.0].reset_index(drop=True)
    below = lambda s: s[s.y0 < t1_top + 1.0].reset_index(drop=True)
    d, _ = TR.transport(gen, g, np.random.default_rng(8), stage="B", absorbers=sheet, absorber_straggling=False)
    np.testing.assert_allclose(pmag(above(a)), pmag(above(d)))            # mean loss: nothing changes before the sheet
    assert abs(pmag(above(c)).mean() / pmag(above(a)).mean() - 1) < 0.01   # straggling: other draws, same physics above
    assert len(below(c)) and pmag(below(c)).mean() < pmag(below(a)).mean() - 5.0   # ~14 MeV lost in 32 mm of Al
