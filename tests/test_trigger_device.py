"""Device trigger pieces on synthetic inputs: menu parsing, the TMVA MLP, panel combining, the helix fit."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trigger.device import menu  # noqa: E402

MENU = """
process_name: "R"
physics: {
  producers: {
    TC: { module_type: "TimeClusterFinder" AveragePitch: 6e-1 }
    HF: { module_type: "RobustHelixFinder" }
  }
  filters: {
    PS: { module_type: "PrescaleEvent" eventModeConfig: [ { eventMode: "OnSpill" prescale: -1 }, { eventMode: "OffSpill" prescale: 1 } ] }
    TCF: { module_type: "TimeClusterFilter" timeClusterCollection: "TC" minNStrawHits: 15 }
    HSF: { module_type: "HelixFilter" posHelicitySelection: { configured: true minMomentum: 70 } negHelicitySelection: { configured: false } }
    CAL: { module_type: "CaloClusterFilter" }
  }
  tpr_X: [ "PS", "TC", "TCF", "HF", "HSF" ]
  calo_Y: [ "PS", "CAL" ]
  trigger_paths: [ "100:tpr_X", "400:calo_Y" ]
}
# a comment
physics.filters.TCF.minNStrawHits: 12
"""


def test_parse_and_paths():
    cfg = menu.parse_fcl(MENU)
    assert cfg["physics"]["filters"]["TCF"]["minNStrawHits"] == 12          # dotted override wins
    assert cfg["physics"]["filters"]["HSF"]["posHelicitySelection"]["minMomentum"] == 70
    dp = menu.device_paths(cfg)
    assert list(dp) == ["tpr_X"]
    assert dp["tpr_X"]["family"] == "tpr" and [r for r, _, _ in dp["tpr_X"]["filters"]] == ["tc", "hs"]
    assert menu.prescales(cfg, "tpr_X", "OnSpill") == -1 and menu.prescales(cfg, "tpr_X", "OffSpill") == 1


XML = """<MethodSetup Method="MLP::MLP"><GeneralInfo><Info name="TMVA Release" value="4.2.1 [262657]"/></GeneralInfo>
<Options><Option name="NeuronType" modified="No">sigmoid</Option><Option name="VarTransform" modified="Yes">N</Option></Options>
<Variables NVar="2"><Variable VarIndex="0" Min="0" Max="2"/><Variable VarIndex="1" Min="-1" Max="1"/></Variables>
<Weights><Layout NLayers="3">
<Layer Index="0" NNeurons="3"><Neuron NSynapses="2">0.5 -1.0</Neuron><Neuron NSynapses="2">2.0 0.25</Neuron><Neuron NSynapses="2">0.1 0.2</Neuron></Layer>
<Layer Index="1" NNeurons="3"><Neuron NSynapses="1">1.5</Neuron><Neuron NSynapses="1">-0.7</Neuron><Neuron NSynapses="1">0.3</Neuron></Layer>
<Layer Index="2" NNeurons="1"><Neuron NSynapses="0"/></Layer></Layout></Weights></MethodSetup>"""


def test_mlp_matches_numpy(tmp_path):
    pytest.importorskip("jax")
    from trigger.device import mva
    p = tmp_path / "w.xml"
    p.write_text(XML)
    net = mva.load(p)
    x = np.array([[0.3, 0.1], [1.9, -0.8]], np.float32)
    xn = (x - [0, -1]) * [1.0, 1.0] - 1.0
    w0 = np.array([[0.5, -1.0], [2.0, 0.25], [0.1, 0.2]])
    w1 = np.array([[1.5], [-0.7], [0.3]])
    h = 1 / (1 + np.exp(-(np.c_[xn, np.ones(2)] @ w0)))
    ref = 1 / (1 + np.exp(-(np.c_[h, np.ones(2)] @ w1)[:, 0]))
    np.testing.assert_allclose(np.asarray(mva.evaluate(net, x)), ref, rtol=1e-6)
    import jax
    assert "HIGHEST" in jax.jit(lambda v: mva.evaluate(net, v)).lower(x).as_text()   # float32 matmuls, no TF32


def test_panel_combine_greedy():
    pytest.importorskip("jax")
    pytest.importorskip("trkfastsim")
    import jax.numpy as jnp
    from trigger.device import hits
    sid = np.array([[10, 11, 200, 0]])                 # straws 10, 11 of panel 0; straw 72 of panel 1; padding
    flag = np.full((1, 4), (1 << 3), np.uint32)        # TimeSelection
    h = {"valid": np.array([[True, True, True, False]]), "spill": np.array([0]), "sid": sid,
         "upanel": (sid >> 10) * 6 + ((sid & 0x380) >> 7), "straw": sid & 0x7F, "flag": flag,
         "time": np.array([[100.0, 110.0, 105.0, 0]], np.float32), "et0": np.zeros((1, 4), np.float32),
         "et1": np.zeros((1, 4), np.float32), "edep": np.full((1, 4), 1e-3, np.float32),
         "wdist": np.array([[5.0, 9.0, 0.0, 0]], np.float32), "uvar": np.full((1, 4), 900.0, np.float32),
         "vvar": np.full((1, 4), 2.0, np.float32), "timevar": np.full((1, 4), 4.0, np.float32),
         "x": np.full((1, 4), 400.0, np.float32), "y": np.zeros((1, 4), np.float32), "z": np.zeros((1, 4), np.float32),
         "udx": np.ones((1, 4), np.float32), "udy": np.zeros((1, 4), np.float32)}
    s = {"maxdt": 45.0, "maxds": 3, "maxwdchi": 5.0, "uerr": 10.0, "checkWres": True, "minN": 1, "maxN": 8,
         "minT": 0.0, "maxT": 1e5, "minTOff": 0.0, "maxTOff": 1e5, "minE": 1e-4, "maxE": 5e-3, "minR": 380.0,
         "maxR": 700.0, "sel": 1 << 3, "mask": 1 << 23}
    out = hits.panel_hits({k: jnp.asarray(v) for k, v in h.items()}, s)
    assert np.asarray(out["valid"]).tolist() == [[True, True, False, False]]
    assert np.asarray(out["ncombo"])[0, :2].tolist() == [2, 1]
    assert np.asarray(out["time"])[0, 0] == pytest.approx(105.0)



def test_helix_pass_rejects_the_unconfigured_helicity():
    """HelixFilter: a helix of a helicity whose selection is not configured fails (it matches no cut set)."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from trigger.device import paths
    h = {"ok": jnp.array([[True, True, True]]), "helicity": jnp.array([[1.0, -1.0, 0.0]]),
         "nsh": jnp.array([[20.0, 20.0, 20.0]]), "pt": jnp.full((1, 3), 60.0), "chi2xy": jnp.ones((1, 3)),
         "chi2zphi": jnp.ones((1, 3)), "d0": jnp.zeros((1, 3)), "lambda": jnp.array([[200.0, -200.0, 0.0]]),
         "nloops": jnp.full((1, 3), 2.0), "p": jnp.full((1, 3), 90.0), "hitratio": jnp.ones((1, 3))}
    pos_only = {"posHelicitySelection": {"configured": True, "minMomentum": 70}, "negHelicitySelection": {"configured": False}}
    assert np.asarray(paths.helix_pass(h, pos_only)).tolist() == [[True, False, False]]
    both = {"posHelicitySelection": {"configured": True, "minMomentum": 95},
            "negHelicitySelection": {"configured": True, "minMomentum": 70}}
    assert np.asarray(paths.helix_pass(h, both)).tolist() == [[False, True, False]]
    assert np.asarray(paths.helix_pass(h, {"posHelicitySelection": {"configured": False},
                                           "negHelicitySelection": {"configured": False}})).tolist() == [[False] * 3]

def test_helix_fit_recovers_parameters():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from trigger.device import helix
    R, lam, cx, cy, fz0 = 250.0, 180.0, 120.0, -60.0, 0.4
    z = np.linspace(-1400, 1400, 40)
    ph = fz0 + z / lam
    x, y = cx + R * np.cos(ph), cy + R * np.sin(ph)
    n = len(z)
    hit = {"x": jnp.asarray(x), "y": jnp.asarray(y), "z": jnp.asarray(z), "udx": jnp.ones(n), "udy": jnp.zeros(n),
           "uvar": jnp.full(n, 900.0), "vvar": jnp.full(n, 2.0), "uface": jnp.arange(n), "nsh": jnp.ones(n, int)}
    s = {"maxrpull": 5.0, "maxdr": 100.0, "maxchisq": 25.0, "cradres": 20.0, "cperpres": 12.0, "minnsh": 10,
         "lmin": 30.0, "lmax": 1000.0, "lstep": 10.0, "minR": 100.0, "maxR": 600.0, "minCR": 100.0, "maxCR": 700.0,
         "minabsl": 10.0, "maxabsl": 700.0, "helicities": (-1, 1)}
    f = helix._fit(hit, jnp.ones(n, bool), s, 1.0, jnp.asarray(z))
    assert bool(f["ok"]) and float(f["helicity"]) == 1.0
    assert float(f["R"]) == pytest.approx(R, abs=1e-3) and float(f["lambda"]) == pytest.approx(lam, rel=1e-4)
    assert float(f["d0"]) == pytest.approx(np.hypot(cx, cy) - R, abs=1e-3)
    assert float(f["nsh"]) == n and float(f["chi2xy"]) < 1e-6


F32_SCRIPT = r'''
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import jax
from trigger.device import helix
jax.config.update("jax_enable_x64", False)          # what Chain(precision="float32") does
import jax.numpy as jnp
R, lam, cx, cy, fz0 = 250.0, 180.0, 120.0, -60.0, 0.4
z = np.linspace(-1400, 1400, 40); ph = fz0 + z / lam
x, y = cx + R * np.cos(ph), cy + R * np.sin(ph); n = len(z)
hit = {"x": jnp.asarray(x), "y": jnp.asarray(y), "z": jnp.asarray(z), "udx": jnp.ones(n), "udy": jnp.zeros(n),
       "uvar": jnp.full(n, 900.0), "vvar": jnp.full(n, 2.0), "uface": jnp.arange(n), "nsh": jnp.ones(n, int)}
s = {"maxrpull": 5.0, "maxdr": 100.0, "maxchisq": 25.0, "cradres": 20.0, "cperpres": 12.0, "minnsh": 10,
     "lmin": 30.0, "lmax": 1000.0, "lstep": 10.0, "minR": 100.0, "maxR": 600.0, "minCR": 100.0, "maxCR": 700.0,
     "minabsl": 10.0, "maxabsl": 700.0, "helicities": (-1, 1)}
f = helix._fit(hit, jnp.ones(n, bool), s, 1.0, jnp.asarray(z))
assert f["R"].dtype == jnp.float32
assert bool(f["ok"]) and abs(float(f["R"]) - R) < 0.05 and abs(float(f["lambda"]) / lam - 1) < 1e-3, f
assert float(f["nsh"]) == n
print("ok")
'''


def test_helix_fit_float32_subprocess():
    """float32 mode flips JAX's process-wide x64 flag, so it runs in its own interpreter."""
    pytest.importorskip("jax")
    import subprocess
    r = subprocess.run([sys.executable, "-c", F32_SCRIPT, str(ROOT)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]


def test_digi_blocks_pack_flat_digis():
    pytest.importorskip("jax")
    pytest.importorskip("trkfastsim")
    from trigger.device import digis
    dg = {"ev": np.array([0, 0, 2, 2, 2]), "sid": np.array([5, 9, 1, 1, 300]),
          "tdc": np.arange(10).reshape(5, 2), "tot": np.arange(10, 20).reshape(5, 2), "pmp": np.arange(5) * 1.0}
    out = list(digis.blocks(dg, nev=4, size=3, events=[10, 11, 12, 13], nh=4))
    assert [(lo, n) for lo, n, _ in out] == [(0, 3), (3, 1)]
    b = out[0][2]
    assert b["valid"].sum(axis=1).tolist() == [2, 0, 3]
    assert b["sid"][2, :3].tolist() == [1, 1, 300] and b["tdc1"][0, :2].tolist() == [1, 3]
    assert b["event"].tolist() == [10, 11, 12] and out[1][2]["valid_event"].tolist() == [True, False, False]
    with pytest.raises(ValueError):
        list(digis.blocks({**dg, "sid": np.array([9, 5, 1, 1, 300])}, nev=4, size=3))


def test_binned_rows_added_to_scalar_base():
    import re
    from trigger.device import response
    edges, edges_p = [0.0, 20.0, 40.0, 1000.0], [0.0, 98.0, 1e6]
    one = np.array([[50, 100], [10, 5], [180, 200]], float)          # middle bin sparse: merges into the first
    two = np.stack([one, np.zeros((3, 2))])                          # second momentum row empty: hits-only fallback
    acc = {"apr_TrkDe_80m70p": {"1d": one, "2d": two, "helix": True}}
    base = {"apr_TrkDe_80m70p": {"hs": 0.9, "ks": 0.95}, "tpr_TrkDe_80m70p": {"ks": 0.97}}
    out = response.binned_rows(acc, edges, edges_p, 50, base)
    assert out["tpr_TrkDe_80m70p"] == {"ks": 0.97}                   # other paths keep their scalar factors
    assert base["apr_TrkDe_80m70p"] == {"hs": 0.9, "ks": 0.95}      # base not modified
    row = out["apr_TrkDe_80m70p"]
    assert row["hs"] == 0.9 and row["final"]["edges"] == [0.0, 40.0, 1000.0]
    assert row["final"]["edges_p"] == edges_p
    np.testing.assert_allclose(row["final"]["f"], [[60 / 105, 0.9], [60 / 105, 0.9]])
    sel = re.compile(response.APR_HELIX_PATHS)
    assert [p for p in ("apr_TrkDe_70", "apr_Hlx_30", "apr_TrkUe_80m70p", "apr_TC", "tpr_TrkDe_80m70p")
            if sel.search(p)] == ["apr_TrkDe_70", "apr_Hlx_30", "apr_TrkUe_80m70p"]


def test_named_responses_resolve_and_the_shipped_files_validate(tmp_path):
    import re
    from trigger.device import response
    assert response.path("ce") == response.path("dio") == response.DATA / "response_ce_dio.json"
    assert response.path("rpc") == response.DATA / "response_rpc.json"
    assert response.path("none") is None and response.load("none") == {} and response.load(None) == {}
    assert response.DEFAULT == "ce" and set(response.NAMES) == {"ce", "dio", "rpc", "none"}
    ce, rpc = response.load("ce"), response.load("rpc")
    binned = {n: sorted(p for p, r in d.items() if "final" in r) for n, d in (("ce", ce), ("rpc", rpc))}
    assert binned["ce"] == sorted(p for p in ce if re.search(response.APR_HELIX_PATHS, p)) and len(binned["ce"]) == 12
    assert binned["rpc"] == sorted(p for p in rpc if re.search(r"^(tpr|apr)_(Trk|Hlx)", p)) and len(binned["rpc"]) == 19
    # both files share the scalar rows (from CE development shards) of the paths they do not bin
    assert all(ce[p] == rpc[p] for p in ce if p not in binned["rpc"]) and set(ce) == set(rpc)
    f = tmp_path / "r.json"
    f.write_text('{"tpr_X": {"ks": 0.9}}')
    assert response.load(str(f)) == {"tpr_X": {"ks": 0.9}}
    with pytest.raises(FileNotFoundError, match="named responses"):
        response.load(str(tmp_path / "missing.json"))


def test_response_validate_rejects_malformed_rows():
    from trigger.device import response
    good = {"final": {"edges": [0, 20, 1000], "edges_p": [0, 90, 1e6], "f": [[0.5, 1.0], [0.9, 0.8]]}}
    assert response.validate({"a": good}) == {"a": good}
    bad = [{"ks": 1.2},
           {"final": {"edges": [0, 20, 1000], "f": [0.5]}},                                   # wrong length
           {"final": {"edges": [0, 20, 1000], "edges_p": [0, 90, 1e6], "f": [0.5, 1.0]}},     # rows missing
           {"final": {"edges": [0, 1000, 20], "f": [0.5, 1.0]}},                              # edges not increasing
           {"final": {"edges": [0, 20, 1000], "f": [0.5, 1.5]}}]                              # above 1
    for row in bad:
        with pytest.raises(ValueError):
            response.validate({"a": row})


def test_response_defaults_per_process(monkeypatch):
    """The CE chain (centuple trigger) and the generic tools default to the CE/DIO response; rpc and none by name."""
    monkeypatch.syspath_prepend(str(ROOT / "beam"))
    from beamfastsim.centuple import __main__ as cm
    seen = []
    monkeypatch.setattr(cm, "cmd_trigger", lambda a: seen.append(a.response))
    base = ["trigger", "--digis", "d.npz", "--out", "o", "--menu", "m.fcl", "--conditions", "c.root", "--tc-mva", "t.xml"]
    cm.main(base)
    cm.main(base + ["--response", "none"])
    assert seen == ["ce", "none"]
    pytest.importorskip("jax")
    pytest.importorskip("trkfastsim")
    from trigger.device import bench, run
    monkeypatch.setattr(run, "cmd_decide", lambda a: seen.append(a.response))
    run.main(["decide", "--dump", "a.root", "--menu", "m.fcl", "--tc-mva", "t.xml", "--output", "o.json"])
    run.main(["decide", "--dump", "a.root", "--menu", "m.fcl", "--tc-mva", "t.xml", "--output", "o.json",
              "--response", "rpc"])
    assert seen[2:] == ["ce", "rpc"]
    monkeypatch.delenv("TRIGGER_DEVICE_RESPONSE", raising=False)
    assert bench._response() == "ce"
    monkeypatch.setenv("TRIGGER_DEVICE_RESPONSE", "rpc")
    assert bench._response() == "rpc"


def test_development_runs_refuse_a_response_layer(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("trkfastsim")
    from trigger.device import run
    f = tmp_path / "dev.json"
    f.write_text('{"response": {"tpr_X": {"ks": 0.9}}, "response_spec": "ce", "stages": {}}')
    with pytest.raises(SystemExit, match="--response none"):
        run._dev_run(f)
    f.write_text('{"response": {}, "response_spec": "none", "stages": {}}')
    assert run._dev_run(f)["stages"] == {}


def _synthetic_panel_hits(E=6, H=48, seed=5):
    """Events with 0-3 helix-like hit groups at different times, padded to H; per-event panel-hit columns."""
    rng = np.random.default_rng(seed)
    cols = {k: np.zeros((E, H), np.float32) for k in ("time", "x", "y", "z", "uvar", "vvar", "wdist", "udx", "udy")}
    valid = np.zeros((E, H), bool)
    plane, face = np.zeros((E, H), np.int64), np.zeros((E, H), np.int64)
    for e in range(E):
        i = 0
        for g in range(e % 4):
            n = int(rng.integers(12, 20))
            R, lam, cx, cy, t0 = rng.uniform(200, 300), rng.uniform(150, 250), *rng.uniform(-100, 100, 2), 600 + 300 * g
            z = np.sort(rng.uniform(-1500, 1500, n))
            ph_ = rng.uniform(0, 6) + z / lam
            sl = slice(i, i + n)
            cols["x"][e, sl], cols["y"][e, sl], cols["z"][e, sl] = cx + R * np.cos(ph_), cy + R * np.sin(ph_), z
            cols["time"][e, sl] = t0 + z / (0.6 * 299.792458) + rng.normal(0, 5, n)
            plane[e, sl] = np.arange(n) % 36
            face[e, sl] = np.arange(n) % 2
            valid[e, sl] = True
            i += n
        noise = slice(i, min(H, i + 5))
        cols["time"][e, noise] = rng.uniform(500, 1700, noise.stop - noise.start)
        cols["x"][e, noise], cols["y"][e, noise] = rng.uniform(-600, 600, (2, noise.stop - noise.start))
        valid[e, noise] = True
    cols["uvar"][:] = 900.0
    cols["vvar"][:] = 2.0
    cols["udx"][:] = 1.0
    cols["wdist"][:] = 3.0
    return dict(cols, valid=valid, plane=plane, face=face, ncombo=valid.astype(np.int64),
                flag=np.where(valid, 1 << 10, 0).astype(np.uint32))


def test_compact_clusters_and_helices_match_the_one_call_path():
    """chain compact path: time-cluster refinement and helices on the occupied (event, cluster) pairs only give the
    same clusters and helices (bit for bit where a consumer reads them) as the vmapped one-call path."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from trigger.device import chain, helix, timecluster
    rng = np.random.default_rng(2)
    net = {"act": "sigmoid", "norm": False, "offset": np.zeros(7, np.float32), "scale": np.ones(7, np.float32),
           "layers": [rng.normal(0, 0.3, (8, 4)).astype(np.float32), rng.normal(0, 0.3, (5, 1)).astype(np.float32)]}
    s = {"tmin": 400.0, "tmax": 1800.0, "tbin": 10.0, "ymin": 3.0, "npeak": 1, "maxdt": 30.0, "minnhits": 5,
         "maxdphi": 2.0, "minkeep": 0.1, "minadd": 0.9, "pitch": 0.6, "prefilter": True, "refine": True,
         "recover": True}
    hs = {"maxrpull": 5.0, "maxdr": 100.0, "maxchisq": 25.0, "cradres": 20.0, "cperpres": 12.0, "minnsh": 5,
          "lmin": 30.0, "lmax": 1000.0, "lstep": 10.0, "minR": 100.0, "maxR": 600.0, "minCR": 0.0, "maxCR": 700.0,
          "minabsl": 10.0, "maxabsl": 700.0, "helicities": (-1, 1)}
    fz = np.linspace(-1500, 1500, 72)
    ph = {k: jnp.asarray(v) for k, v in _synthetic_panel_hits().items()}
    full = jax.jit(lambda p: timecluster.time_clusters(p, s, net))(ph)
    cand = jax.jit(lambda p: timecluster.candidates(p, s))(ph)
    k = chain.pair_bucket(int(jnp.sum(cand[2])), cand[2].size)
    comp = jax.jit(lambda c: timecluster.refine_pairs(*c, s, net, k))(cand)
    v = np.asarray(full["valid"])
    assert v.any() and not v.all()
    np.testing.assert_array_equal(np.asarray(comp["valid"]), v)
    for key in ("nsh", "t0", "t0err"):
        np.testing.assert_array_equal(np.asarray(comp[key])[v], np.asarray(full[key])[v])
    np.testing.assert_array_equal(np.asarray(comp["members"]), np.asarray(full["members"]))
    hf = jax.jit(lambda p, tc: helix.helices(p, tc, hs, fz))(ph, full)
    kv = chain.pair_bucket(int(v.sum()), v.size)
    hc = jax.jit(lambda p, tc: helix.helices_pairs(p, tc, hs, fz, kv))(ph, comp)
    ok = np.asarray(hf["ok"])
    np.testing.assert_array_equal(np.asarray(hc["ok"]), ok)
    for key in hf:
        np.testing.assert_array_equal(np.asarray(hc[key])[ok], np.asarray(hf[key])[ok])
    assert [chain.pair_bucket(n, 64) for n in (0, 16, 17, 32, 33, 64)] == [16, 16, 32, 32, 64, 64]
