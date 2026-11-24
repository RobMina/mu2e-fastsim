"""Design muon sample on the device: overburden transport against the host reference, the device-CRY builder's
bookkeeping and the entry-sample comparison; synthetic world, no experiment data."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
from crvfastsim import transport as TR                       # noqa: E402
from crvfastsim.design import device_sample as DS            # noqa: E402
from crvfastsim.design import fom as F                       # noqa: E402
from crvfastsim.design import sample as S                    # noqa: E402
from mu2efastsim import eloss, materials                     # noqa: E402
from mu2efastsim import transport as core_tr                 # noqa: E402

Y_PLANE = 3252.0


def _geom():
    return SimpleNamespace(nbars=1, center=np.array([[-3904.0, Y_PLANE - 10.8, 8000.0]]),
                           halflen=np.array([[3000.0, 10.0, 6000.0]]))


def _box(mat, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    planes = np.c_[np.r_[np.eye(3), -np.eye(3)], np.r_[hi, -lo]]
    return (mat, planes, lo, hi)


def _world():
    """1 m of concrete 1-2 m above the plane and a 0.5 m steel plate higher up, both wider than the proxy region."""
    pieces = [_box("CONCRETE_MARS", (-20000, Y_PLANE + 1000, -20000), (20000, Y_PLANE + 2000, 30000)),
              _box("MildSteel", (-20000, Y_PLANE + 6000, -20000), (20000, Y_PLANE + 6500, 30000))]
    return TR.banded_world(pieces, TR.band_edges(Y_PLANE))


def _prepared():
    w = _world()
    ew = S.EntryWorld.__new__(S.EntryWorld)
    ew.geom, ew.y_plane, ew.world, ew.edges, ew._device = _geom(), Y_PLANE, w, w.edges, None
    return ew


def _gen(n, seed=3, p=(300.0, 20000.0)):
    rng = np.random.default_rng(seed)
    d = np.stack([rng.normal(0, 0.2, n), -np.ones(n), rng.normal(0, 0.2, n)], 1)
    d /= np.linalg.norm(d, axis=1)[:, None]
    end = np.stack([rng.uniform(-4800, -3000, n), np.zeros(n), rng.uniform(5000, 12000, n)], 1)
    o = end - d * (15000.0 / -d[:, 1])[:, None]
    mom = np.exp(rng.uniform(*np.log(p), n))
    pv = mom[:, None] * d
    return pd.DataFrame(dict(run=1400, subrun=1, event=np.arange(n), idx=0, pdg=np.where(rng.random(n) < .5, 13, -13),
                             x=o[:, 0], y=o[:, 1], z=o[:, 2], px=pv[:, 0], py=pv[:, 1], pz=pv[:, 2], t=0.0))


def test_device_world_matches_host_reference():
    ew = _prepared()
    g = _gen(20000)
    o = g[["x", "y", "z"]].to_numpy()
    p0 = np.linalg.norm(g[["px", "py", "pz"]].to_numpy(), axis=1)
    d = g[["px", "py", "pz"]].to_numpy() / p0[:, None]
    rng = np.random.default_rng(5)
    pp, xx0, alive, bxx0, bh = TR.world_loss(rng, o, d, p0, ew.world, ew.edges)
    pm = np.maximum(np.sqrt(p0 * np.maximum(pp, 1.0)), 1.0)
    oh, dh, _ = core_tr.msc_to_plane(rng, o, d, pm, Y_PLANE, eloss.radiation_length_mm(materials.AIR), xx0, bxx0, bh)
    od, dd, pd_, ad = ew.device(batch_size=4096).run(o, d, p0, seed=11)
    np.testing.assert_allclose(od[:, 1], Y_PLANE, atol=1e-6)
    assert abs(ad.mean() - alive.mean()) < 5 * np.sqrt(alive.mean() * (1 - alive.mean()) * 2 / len(alive))
    both = alive & ad
    loss_h, loss_d = (p0 - pp)[both], (p0 - pd_)[both]
    se = np.sqrt(loss_h.var() / both.sum() + loss_d.var() / both.sum())
    assert abs(loss_h.mean() - loss_d.mean()) < 5 * se
    line = o + d * ((o[:, 1] - Y_PLANE) / -d[:, 1])[:, None]
    for k in (0, 2):
        sh, sd = np.std((oh - line)[both, k]), np.std((od - line)[both, k])
        assert abs(sd / sh - 1) < 0.05
        assert abs(np.std((dh - d)[both, k]) / np.std((dd - d)[both, k]) - 1) < 0.05
    again = ew.device().run(o, d, p0, seed=11)
    for a, b in zip(again, (od, dd, pd_, ad)):
        np.testing.assert_array_equal(a, b)


def test_crv_entry_states_backends_agree_and_count():
    ew = _prepared()
    g = _gen(30000, seed=7)
    host = S.crv_entry_states(g, None, None, np.random.default_rng(1), prepared=ew)
    dev = S.crv_entry_states(g, None, None, np.random.default_rng(1), backend="jax", prepared=ew)
    for m in (host, dev):
        assert m["n_generated_events"] == m["n_generated_muons"] == 30000
        assert len(m["p"]) == len(m["p_generated"]) == len(m["event_key"])
        assert F.proxy_hits(m["origin"], m["direction"]).all()
        assert (m["p"] <= m["p_generated"]).all()
    n_h, n_d = len(host["p"]), len(dev["p"])
    assert abs(n_h - n_d) < 5 * np.sqrt(n_h + n_d)
    assert dev["n_preselected"] == host["n_preselected"]
    with pytest.raises(ValueError, match="backend"):
        S.crv_entry_states(g, None, None, np.random.default_rng(1), backend="other", prepared=ew)


def test_entry_transport_rejects_upward_lines():
    ew = _prepared()
    with pytest.raises(ValueError, match="downward"):
        ew.device().run(np.zeros((1, 3)), np.array([[0.0, 1.0, 0.0]]), np.ones(1), seed=1)


def test_entry_compare_identical_and_scaled(tmp_path):
    ew = _prepared()
    m = S.crv_entry_states(_gen(20000), None, None, np.random.default_rng(2), prepared=ew)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    DS.save(a, m, F.PROXY)
    m2 = dict(m, n_generated_events=2 * m["n_generated_events"])
    DS.save(b, m2, F.PROXY)
    out = DS.entry_compare([str(a), str(a), str(b)])
    same, half = out["samples"][1], out["samples"][2]
    assert same["rate_ratio"] == 1.0 and half["rate_ratio"] == 0.5
    for k, s in same["spectra"].items():
        assert s["chi2"] == 0.0 and s["ks_p"] == 1.0, k
    assert half["spectra"]["p"]["chi2"] > 0
    assert json.loads(str(np.load(a)["provenance"])) == {}


def test_device_build_bookkeeping(monkeypatch):
    calls = []

    class Source:
        def __init__(self, config, seed, require_gpu, selected_capacity):
            assert config == "cfg" and not require_gpu and selected_capacity == 99
            self.subrun = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def generate(self, n, subrun):
            self.subrun.append(subrun)
            g = _gen(n, seed=subrun)
            g["subrun"] = subrun
            return g, pd.DataFrame({"livetime": [0.5 * n]}), {"generator": "jax-cry-1.7", "device": "cpu"}

    import mu2efastsim.cosmic_cry as CC
    import mu2efastsim.cry_device as CD
    monkeypatch.setitem(CC.PRESETS, "run2", lambda gdml: "cfg")
    monkeypatch.setattr(CD, "DeviceCosmicSource", Source)
    ew = _prepared()
    monkeypatch.setattr(S, "EntryWorld", lambda gdml, counters: ew)
    real = S.crv_entry_states

    def spy(gen, *a, **kw):
        calls.append(len(gen))
        return real(gen, *a, **kw)

    monkeypatch.setattr(S, "crv_entry_states", spy)
    out = DS.build("run2", "x.gdml", "c.txt", 25000, seed=4, chunk=10000, require_gpu=False, selected_capacity=99)
    assert calls == [10000, 10000, 5000]
    assert out["n_generated_events"] == 25000 and out["live_seconds"] == 12500.0
    assert len(out["p"]) == len(out["event_key"]) and set(out["event_key"][:, 1]) <= {1, 2, 3}
    assert out["provenance"]["cry"]["generator"] == "jax-cry-1.7" and out["provenance"]["transport"] == "jax"


def test_materials_without_constants_follow_the_host_rule():
    from crvfastsim.design import entry_jax as EJ
    pieces = [_box("CONCRETE_MARS", (-20000, Y_PLANE + 1000, -20000), (20000, Y_PLANE + 2000, 30000)),
              _box("NoSuchMaterial", (-20000, Y_PLANE + 3000, -20000), (20000, Y_PLANE + 4000, 30000))]
    w = TR.banded_world(pieces, TR.band_edges(Y_PLANE))
    sub, names = EJ.host_material_rule(w)
    assert names == ["NoSuchMaterial"]
    assert {q[0] for q in sub._pieces} == {"CONCRETE_MARS"} and np.array_equal(sub.edges, w.edges)
    assert EJ.host_material_rule(_world())[1] == []
    ew = _prepared()
    ew.world, ew._device = w, None
    g = _gen(4000, seed=9)
    host = S.crv_entry_states(g, None, None, np.random.default_rng(1), prepared=ew)
    dev = S.crv_entry_states(g, None, None, np.random.default_rng(1), backend="jax", prepared=ew)
    assert dev["concrete_substituted"] == ["NoSuchMaterial"]
    assert abs(np.mean(host["p_generated"] - host["p"]) / np.mean(dev["p_generated"] - dev["p"]) - 1) < 0.03
