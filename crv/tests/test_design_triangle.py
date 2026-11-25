"""crvfastsim.design with triangular counters (Design.profile "triangle"): interleaved prisms, chord clipping, native
optics, cost and deadtime terms, and the rectangular default left unchanged. Synthetic counters and tables only."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crvfastsim.design import fom as F                                   # noqa: E402
from crvfastsim.design import module as M                                # noqa: E402
from crvfastsim.design import optics as O                                # noqa: E402
from crvfastsim.design import veto as V                                  # noqa: E402
from crvfastsim.response import photons as PH                            # noqa: E402
from test_design import COUNTERS, _ctx, _muons, _table                   # noqa: E402

TRI = M.Design(n_layers=4, thickness=30.0, width=42.38, profile="triangle")


@pytest.fixture
def counters(tmp_path):
    p = tmp_path / "crv_counters_test.txt"
    p.write_text(COUNTERS)
    return str(p)


def _inside(u, v, t, w, o):
    return (np.abs(u) <= t / 2) & (np.abs(v) <= (w / 2) * (t / 2 - o * u) / t)


def _tri_table(name="3000_0", reflector=0, length=3000.0, t=30.0, w=42.38, coat=0.25, apex_only=False):
    """A table in the triangle's frame: x from the centroid (signed), |y|; uniform, or light only at x > 0."""
    ex = np.array([-t / 3, -5.0, 0.0, 5.0, 2 * t / 3])
    ey = np.linspace(0.0, w / 2, 5)
    ez = np.linspace(-length / 2, length / 2, 31)
    zc = 0.5 * (ez[1:] + ez[:-1]) + length / 2
    prob = np.ones((4, 4, 30)) * 0.0005 * np.exp(-zc / 4000.0)[None, None, :]
    if apex_only:
        prob[:2] = 0.0
    return O.TableModel(name, prob, ex, ey, ez, (t / 2 - coat, w / 2 - coat, length / 2), 4000.0, reflector,
                        fold_x=False, fold_y=True)


def _rect_table(name="3000_0", reflector=0, length=3000.0, t=30.0, w=42.38, coat=0.25):
    ex = np.linspace(0, t / 2, 5)
    ey = np.linspace(-w / 2, w / 2, 9)
    ez = np.linspace(-length / 2, length / 2, 31)
    zc = 0.5 * (ez[1:] + ez[:-1]) + length / 2
    prob = np.ones((4, 8, 30)) * 0.0005 * np.exp(-zc / 4000.0)[None, None, :]
    return O.TableModel(name, prob, ex, ey, ez, (t / 2 - coat, w / 2 - coat, length / 2), 4000.0, reflector)


def _native_ctx(counters, tri=True):
    ctx = _ctx(counters)
    mk = _tri_table if tri else _rect_table
    ctx.native_lookup_dir = "synthetic"
    ctx.native_tables = {"3000_0": mk(), "3000_1": mk("3000_1", 1)}
    return ctx


NATIVE = F.FomConfig(optics=O.OpticsConfig(table_mode="native"))


def test_prisms_interleave_and_fill_the_layer(counters):
    dg = M.build(TRI, counters)
    rect = M.build(M.Design(n_layers=4, thickness=30.0, width=42.38), counters)
    assert rect.orient is None and dg.orient is not None
    b = (dg.sector == 0) & (dg.module == 0) & (dg.layer == 0) & (dg.segment == 0)
    assert b.sum() == 32                                      # 2 x the stock 16 counters per module
    g = dg.center[b] @ dg.axes[0][1]
    assert np.allclose(np.diff(g), 42.38 / 2)                 # every half width, no gap
    assert np.array_equal(dg.orient[b], np.tile([1, -1], 16))
    assert dg.module_pitch[0] == pytest.approx(33 * 42.38 / 2 + 3.66)
    assert dg.triangle.all() and not rect.triangle.any()
    with pytest.raises(ValueError):
        M.to_crv_geometry(dg)
    with pytest.raises(ValueError):
        M.Design(profile="hexagon")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_clip_prisms_against_point_sampling(dtype):
    rng = np.random.default_rng(4)
    n = 3000
    t, w = 30.0, 42.38
    half = np.tile([t / 2, w / 2, 500.0], (n, 1)).astype(dtype)
    a = rng.uniform(-1, 1, (n, 3)) * half
    b = rng.uniform(-1, 1, (n, 3)) * half
    o = rng.choice([1, -1], n)
    a2, b2, keep = V.clip_prisms(a.astype(dtype), b.astype(dtype), half, o)
    s = np.linspace(0, 1, 2001)
    pts = a[:, None, :] + (b - a)[:, None, :] * s[None, :, None]
    ins = _inside(pts[..., 0], pts[..., 1], t, w, o[:, None])
    frac = ins.mean(1)
    clipped = np.where(keep, np.linalg.norm(b2 - a2, axis=1) / np.linalg.norm(b - a, axis=1), 0.0)
    tol = 2e-3 if dtype == np.float64 else 3e-3
    assert np.all(np.abs(clipped - frac) < tol)
    mid = 0.5 * (a2 + b2)[keep]
    assert _inside(mid[:, 0], mid[:, 1], t, w, o[keep]).all()
    z, zk = V.clip_prisms(a, b, half, np.zeros(n, int))       # orient 0 (rectangles): unchanged
    assert zk.all() and np.array_equal(z, a)


def test_a_layer_is_a_solid_slab(counters):
    """Vertical lines through the middle of sector A: about two prisms per layer, path per layer = thickness."""
    dg = M.build(TRI, counters)
    n = 400
    x = np.full(n, -1000.0)
    z = np.linspace(300.0, 500.0, n)                           # module 0 in every layer (stagger 42 mm per layer)
    o = np.stack([x, np.full(n, 6000.0), z], 1)
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    idx, pos = V.sector_crossings(dg, o, d)
    a = idx[dg.sector[idx[:, 1]] == 0]
    p = pos[dg.sector[idx[:, 1]] == 0]
    path = np.linalg.norm(p[:, 3:] - p[:, :3], axis=1)
    lay = dg.layer[a[:, 1]]
    tot = np.zeros((n, 4))
    cnt = np.zeros((n, 4), int)
    np.add.at(tot, (a[:, 0], lay), path)
    np.add.at(cnt, (a[:, 0], lay), 1)
    assert np.allclose(tot, 30.0, atol=1e-6) and set(np.unique(cnt)) <= {1, 2}


def test_native_pe_and_frame():
    t, w = 30.0, 42.38
    h = np.array([[t / 2, w / 2, 1500.0]])
    p = np.array([1000.0])
    tab = _tri_table()
    up_a, up_b = np.array([[-t / 2, 0.0, 0.0]]), np.array([[t / 2, 0.0, 0.0]])
    pe = O.expected_pe_native(up_a, up_b, h, p, tab, 30482.0, 1.4, (0, 1), np.array([1]))
    half = O.expected_pe_native(up_a * [0.5, 1, 1], up_b * [0.5, 1, 1], h, p, tab, 30482.0, 1.4, (0, 1),
                                np.array([1]))
    np.testing.assert_allclose(half, pe / 2, rtol=1e-12)          # uniform table: PE ~ path
    f18 = O.expected_pe_native(up_a, up_b, h, p, tab, 39794.0, 1.8, (0, 1), np.array([1]))
    np.testing.assert_allclose(f18, pe, rtol=1e-12)              # the stock-fiber factor is taken out of the yield
    # light only at table x > 0 (towards the apex): apex-up prism -> upper half; apex-down -> lower half
    ap = _tri_table(apex_only=True)
    top, bot = np.array([[0.0, 0.0, 0.0]]), np.array([[t / 2, 0.0, 0.0]])
    lo_a, lo_b = np.array([[-t / 2, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]])
    up_top = O.expected_pe_native(top, bot, h, p, ap, 30482.0, 1.4, (0,), np.array([1]))[0, 0]
    up_low = O.expected_pe_native(lo_a, lo_b, h, p, ap, 30482.0, 1.4, (0,), np.array([1]))[0, 0]
    dn_top = O.expected_pe_native(top, bot, h, p, ap, 30482.0, 1.4, (0,), np.array([-1]))[0, 0]
    dn_low = O.expected_pe_native(lo_a, lo_b, h, p, ap, 30482.0, 1.4, (0,), np.array([-1]))[0, 0]
    assert up_top > 0 and up_top == pytest.approx(dn_low, rel=1e-12)
    assert up_low == pytest.approx(3 / 8 * up_top) and dn_top == pytest.approx(3 / 8 * dn_low)   # 3 of 8 points
    with pytest.raises(ValueError, match="frame"):
        O.expected_pe_native(up_a, up_b, h, p, _rect_table(), 30482.0, 1.4, (0,), np.array([1]))
    with pytest.raises(ValueError, match="cross-section"):
        O.expected_pe_native(up_a, up_b, np.array([[12.0, w / 2, 1500.0]]), p, tab, 30482.0, 1.4, (0,),
                             np.array([1]))


def test_scaled_mode_refuses_triangles(counters):
    with pytest.raises(ValueError, match="native"):
        F.evaluate(TRI, _muons(500), _ctx(counters))
    with pytest.raises(ValueError, match="table_mode"):
        O.OpticsConfig(table_mode="other")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_evaluate_triangle_native(counters, dtype):
    mu = _muons()
    tri = F.evaluate(TRI, mu, _native_ctx(counters), NATIVE, dtype=dtype)
    rect = F.evaluate(M.Design(n_layers=4, thickness=30.0, width=42.38), mu, _native_ctx(counters, tri=False),
                      NATIVE, dtype=dtype)
    for r in (tri, rect):
        assert 0 <= r["inefficiency"] <= 1 and r["n_crossings"] > 0 and r["ly_margin"] > 0
        assert r["native_table_substitutions"] == {}
    assert tri["profile"] == "triangle" and tri["n_bars"] == pytest.approx(2 * rect["n_bars"], rel=0.1)
    # same scintillator slab and channel count per module: cost terms within a few % of the rectangles
    assert tri["cost_rel"]["scint_m3"] == pytest.approx(rect["cost_rel"]["scint_m3"], rel=0.05)
    assert tri["cost_rel"]["channels"] == pytest.approx(rect["cost_rel"]["channels"], rel=0.1)
    assert tri["cost_rel"]["fiber_m_mm2"] == pytest.approx(rect["cost_rel"]["fiber_m_mm2"], rel=0.1)
    assert tri["n_crossings"] > 1.6 * rect["n_crossings"]       # about two prisms per layer


def test_native_rect_and_nominal_use_their_tables(counters):
    """Native mode reads the design's own tables; the nominal design keeps the stock (scaled) tables, so a native
    evaluation of the nominal design with tables equal to the stock ones gives the scaled result."""
    mu = _muons(2000)

    def ctx14():                       # both sectors with 1.4 mm fibers (a native table holds its own fiber)
        c = _ctx(counters)
        c.sectors = dict(c.sectors, B=PH.SectorConfig("3000_1", 1, 30482.0))
        return c
    ctx = ctx14()
    ctx.native_lookup_dir = "synthetic"
    ctx.native_tables = {"3000_0": _table(), "3000_1": _table("3000_1", 1)}
    nat = F.evaluate(M.Design(), mu, ctx, F.FomConfig(optics=O.OpticsConfig(table_mode="native",
                                                                            native_model_scale=1.0)))
    sc = F.evaluate(M.Design(), mu, ctx14())
    assert nat["ly_margin"] == pytest.approx(sc["ly_margin"], rel=0.02)
    assert nat["inefficiency"] == pytest.approx(sc["inefficiency"], rel=0.05, abs=1e-4)
    with pytest.raises(ValueError, match="native_lookup_dir"):
        F.evaluate(M.Design(), mu, _ctx(counters), NATIVE)


def test_default_rect_is_unchanged(counters):
    """The rectangular default has no orient and the cost terms of the plain box formula."""
    dg = M.build(M.Design(), counters)
    assert dg.orient is None
    ct = M.cost_terms(dg, {"A": 0, "B": 1}, 2.7)
    assert ct["scint_m3"] == float(np.prod(2 * dg.halflen, axis=1).sum()) * 1e-9
    assert ct["channels"] == int(2 * (2 * (dg.sector == 0).sum() + (dg.sector == 1).sum()))
    assert math.isclose(F.evaluate(M.Design(), _muons(1000), _ctx(counters))["cost"], 1.0)
