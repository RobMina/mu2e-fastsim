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
    """The generic tools default to the CE/DIO response, rpc and none by name; centuple trigger defaults to none,
    because assemble applies the on-spill trigger table by default, which replaces the response layer."""
    monkeypatch.syspath_prepend(str(ROOT / "beam"))
    from beamfastsim.centuple import __main__ as cm
    seen = []
    monkeypatch.setattr(cm, "cmd_trigger", lambda a: seen.append(a.response))
    base = ["trigger", "--digis", "d.npz", "--out", "o", "--menu", "m.fcl", "--conditions", "c.root", "--tc-mva", "t.xml"]
    cm.main(base)
    cm.main(base + ["--response", "ce"])
    assert seen == ["none", "ce"]
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


def test_time_cluster_order_key_leaves_the_clusters_unchanged():
    """timecluster with s["order"] (the TimeCluster hit order the mpr finder reads) gives the same clusters as without
    it, in the full, compact and pairs paths; the key is the hit index for the peak-assigned hits, above every index
    for the recovered ones (in the order they were added), and ORDER_NONE for non-members."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from trigger.device import chain, timecluster
    rng = np.random.default_rng(2)
    net = {"act": "sigmoid", "norm": False, "offset": np.zeros(7, np.float32), "scale": np.ones(7, np.float32),
           "layers": [rng.normal(0, 0.3, (8, 4)).astype(np.float32), rng.normal(0, 0.3, (5, 1)).astype(np.float32)]}
    s = {"tmin": 400.0, "tmax": 1800.0, "tbin": 10.0, "ymin": 3.0, "npeak": 1, "maxdt": 30.0, "minnhits": 5,
         "maxdphi": 2.0, "minkeep": 0.1, "minadd": 0.9, "pitch": 0.6, "prefilter": True, "refine": True,
         "recover": True}
    so = {**s, "order": True}
    ph = {k: jnp.asarray(v) for k, v in _synthetic_panel_hits().items()}
    H = ph["x"].shape[1]
    ref = jax.jit(lambda p: timecluster.time_clusters(p, s, net))(ph)
    full = jax.jit(lambda p: timecluster.time_clusters(p, so, net))(ph)
    cand = jax.jit(lambda p: timecluster.candidates(p, so))(ph)
    k = chain.pair_bucket(int(jnp.sum(cand[2])), cand[2].size)
    comp = jax.jit(lambda c: timecluster.refine_pairs(*c, so, net, k))(cand)
    pairs = jax.jit(lambda p: timecluster.time_clusters_multi_pairs(p, [so], net, K=ph["x"].shape[0] * 8))(ph)[0]
    assert "order" not in ref
    for got in (full, comp, pairs):
        for key in ("valid", "nsh", "t0", "t0err", "members"):
            np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(ref[key]))
        np.testing.assert_array_equal(np.asarray(got["order"]), np.asarray(full["order"]))
    mem, key = np.asarray(full["members"]), np.asarray(full["order"])
    assert ((key < timecluster.ORDER_NONE) == mem).all()
    idx = np.broadcast_to(np.arange(H), key.shape)
    assert ((key == idx) | (key >= H) | ~mem).all() and (key[mem & (key >= H)] >= H).all()
    assert mem.any() and (key[mem] >= H).any()                  # the synthetic events recover hits


def test_pair_paths_match_the_batched_multi_path():
    """chain pairs path (the default): cluster refinement and helix fits on the gathered (producer, event, slot) pairs
    give the same clusters as the batched multi path and the same helices as the per-set fit, bit for bit where a
    consumer reads them, both when the pairs fit in K and when they do not (then the full batched path runs in the
    same call). The batched helices_multi (pairs off) fuses differently: its helix quantities agree with the per-set
    fit to rounding (float64: 1 ulp in the momentum)."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from trigger.device import helix, timecluster
    rng = np.random.default_rng(3)
    net = {"act": "sigmoid", "norm": False, "offset": np.zeros(7, np.float32), "scale": np.ones(7, np.float32),
           "layers": [rng.normal(0, 0.3, (8, 4)).astype(np.float32), rng.normal(0, 0.3, (5, 1)).astype(np.float32)]}
    s = {"tmin": 400.0, "tmax": 1800.0, "tbin": 10.0, "ymin": 3.0, "npeak": 1, "maxdt": 30.0, "minnhits": 5,
         "maxdphi": 2.0, "minkeep": 0.1, "minadd": 0.9, "pitch": 0.6, "prefilter": True, "refine": True,
         "recover": True}
    ss = [s, {**s, "pitch": -0.63}]
    hs = {"maxrpull": 5.0, "maxdr": 100.0, "maxchisq": 25.0, "cradres": 20.0, "cperpres": 12.0, "minnsh": 5,
          "lmin": 30.0, "lmax": 1000.0, "lstep": 10.0, "minR": 100.0, "maxR": 600.0, "minCR": 0.0, "maxCR": 700.0,
          "minabsl": 10.0, "maxabsl": 700.0, "helicities": (-1, 1)}
    fz = np.linspace(-1500, 1500, 72)
    ph = {k: jnp.asarray(v) for k, v in _synthetic_panel_hits().items()}
    ref = jax.jit(lambda p: timecluster.time_clusters_multi(p, ss, net))(ph)
    E, C = ref[0]["valid"].shape
    assert sum(int(np.asarray(r["valid"]).sum()) for r in ref) > 1
    href = [jax.jit(lambda p, t: helix.helices(p, t, hs, fz))(ph, r) for r in ref]      # per-set fit
    hmulti = jax.jit(lambda p, t: helix.helices_multi(p, t, hs, fz))(ph, ref)
    for r, m in zip(href, hmulti):
        ok = np.asarray(r["ok"])
        np.testing.assert_array_equal(np.asarray(m["ok"]), ok)
        for key in r:
            np.testing.assert_allclose(np.asarray(m[key])[ok], np.asarray(r[key])[ok], rtol=1e-12, atol=0)
    for K in (len(ss) * E * C, 1):               # pairs branch; more candidates than K: the full branch
        got = jax.jit(lambda p: timecluster.time_clusters_multi_pairs(p, ss, net, K))(ph)
        for r, g in zip(ref, got):
            v = np.asarray(r["valid"])
            assert v.any()
            np.testing.assert_array_equal(np.asarray(g["valid"]), v)
            for key in ("nsh", "t0", "t0err", "members"):
                np.testing.assert_array_equal(np.asarray(g[key])[v], np.asarray(r[key])[v])
        hgot = jax.jit(lambda p, t: helix.helices_multi_pairs(p, t, hs, fz, K))(ph, got)
        for r, g in zip(href, hgot):
            ok = np.asarray(r["ok"])
            np.testing.assert_array_equal(np.asarray(g["ok"]), ok)
            for key in r:
                np.testing.assert_array_equal(np.asarray(g[key])[ok], np.asarray(r[key])[ok])


def _group_reference(h, s, maxn=8):
    """The stock CombineStrawHits loop (TestFlag on), one event at a time, hit by hit: owner seed per hit."""
    E, H = h["valid"].shape
    owner = np.full((E, H), -1)
    dead = 1 << 23
    for e in range(E):
        n = int(h["valid"][e].sum())
        good = [((h["flag"][e, i] & s["sel"]) == s["sel"]) and not (h["flag"][e, i] & (s["mask"] | dead)) for i in range(n)]
        used = [False] * n
        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            if not good[i]:
                continue
            owner[e, i], k = i, 1
            for j in range(i + 1, n):
                if used[j]:
                    continue
                if h["upanel"][e, j] != h["upanel"][e, i]:
                    break
                if abs(int(h["straw"][e, j]) - int(h["straw"][e, i])) > s["maxds"] or not good[j]:
                    continue
                if abs(h["time"][e, i] - h["time"][e, j]) > s["maxdt"]:
                    continue
                if abs(h["wdist"][e, i] - h["wdist"][e, j]) / np.sqrt(h["uvar"][e, i] + h["uvar"][e, j]) > s["maxwdchi"]:
                    continue
                if k >= maxn:
                    continue
                used[j], owner[e, j], k = True, i, k + 1
    return owner


def test_panel_group_matches_per_hit_loop():
    pytest.importorskip("jax")
    pytest.importorskip("trkfastsim")
    import jax.numpy as jnp
    from trigger.device import hits
    rng = np.random.default_rng(7)
    E, H = 12, 32
    n = rng.integers(0, H + 1, E)
    valid = np.arange(H)[None, :] < n[:, None]
    panel = np.sort(rng.integers(0, 4, (E, H)), axis=1)                     # few panels: long runs
    straw = rng.integers(0, 12, (E, H))
    sid = np.where(valid, (panel << 7) | straw, 0)
    order = np.argsort(np.where(valid, sid, 1 << 20), axis=1, kind="stable")
    sid = np.take_along_axis(sid, order, axis=1)
    flag = np.where(rng.random((E, H)) < 0.9, 1 << 3, 0).astype(np.uint32)
    h = {"valid": valid, "upanel": (sid >> 10) * 6 + ((sid & 0x380) >> 7), "straw": sid & 0x7F, "flag": flag,
         "time": rng.normal(100, 30, (E, H)).astype(np.float32), "wdist": rng.normal(0, 80, (E, H)).astype(np.float32),
         "uvar": np.full((E, H), 900.0, np.float32)}
    s = {"sel": 1 << 3, "mask": 0, "maxds": 3, "maxdt": 45.0, "maxwdchi": 5.0}
    got = np.asarray(hits._group({k: jnp.asarray(v) for k, v in h.items()}, s))
    np.testing.assert_array_equal(np.where(valid, got, -1), _group_reference(h, s))


def test_packed_results_unpack_to_the_same_arrays():
    """chain.pack / unpack: decisions and stage arrays of mixed dtypes and shapes, stacked on the device under jit,
    come back on the host as the same dicts, bit for bit; and the KalSeedFilter draw at pass probability 1 is
    skipped because a uniform draw in [0, 1) is always below 1."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from trigger.device import chain, paths
    rng = np.random.default_rng(4)
    E, C = 16, 4
    out = {f"p{i}": jnp.asarray(rng.random(E) < 0.5) for i in range(3)}
    aux = {"p0:tc": jnp.asarray(rng.random(E) < 0.5), "p0:tcnsh": jnp.asarray(rng.integers(0, 50, E)),
           "p0:help": jnp.asarray(rng.normal(100, 5, E)), "p0:hsmask": jnp.asarray(rng.random((E, C)) < 0.5),
           "p0:fcent": jnp.asarray(rng.normal(0, 1, (E, C))), "p1:fcent": jnp.asarray(rng.normal(0, 1, (E, C)))}
    layout = chain.pack_layout(out, aux)
    assert len(layout) == 5                      # bool, int, float [E]; bool, float [E, C]
    arrs = jax.device_get(jax.jit(lambda o, a: chain.pack(o, a, layout))(out, aux))
    o2, a2 = chain.unpack(arrs, layout)
    assert sorted(o2) == sorted(out) and sorted(a2) == sorted(aux)
    for src, got in ((out, o2), (aux, a2)):
        for k, v in src.items():
            assert got[k].dtype == v.dtype
            np.testing.assert_array_equal(got[k], np.asarray(v))
    ev = jnp.arange(200_000)
    assert float(jnp.max(paths.draw(ev, 7))) < 1.0


def _stc_reference(times, nsh, tstep, twin, minnsh):
    """The stock SimpleTimeCluster::findClusters walk at the trigger settings (useTimeStep, no time window in the
    walk, TestFlag false, no non-noise bits), one event: the nStrawHits of every cluster it writes."""
    order = sorted(range(len(times)), key=lambda i: (times[i], i))
    t = [float(times[i]) for i in order]
    n = [int(nsh[i]) for i in order]
    out, end = [], 0
    for start in range(len(t)):
        count = n[start]
        if end >= len(t):
            break
        if start < end:
            continue
        if start > end:
            end = start
        st = t[start]
        while True:
            end += 1
            if end >= len(t):
                break
            et = t[end]
            count += n[end]
            if et - t[end - 1] > tstep:
                break
        if end < len(t):
            count -= n[end]
        end -= 1
        if t[end] - st > twin:
            continue
        if count >= minnsh:
            out.append(count)
    return sorted(out)


def test_simple_time_clusters_match_the_stock_walk():
    """simpletc.clusters: the same clusters (by straw-hit count) as the stock walk, including the second cluster at
    the last hit of a run and the runs dropped for their time span."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from trigger.device import simpletc
    rng = np.random.default_rng(11)
    E, H = 40, 48
    nv = rng.integers(0, H + 1, E)
    valid = np.arange(H)[None, :] < nv[:, None]
    # clumps of hits 0-15 ns apart, clumps 0-60 ns apart: runs of many sizes and spans around the 100 ns window
    t = np.cumsum(np.where(rng.random((E, H)) < 0.25, rng.uniform(15, 60, (E, H)), rng.uniform(0, 15, (E, H))), axis=1)
    perm = np.argsort(rng.random((E, H)), axis=1)                  # hits in arbitrary (not time) order
    t = np.take_along_axis(np.where(valid, t + 500, 0.0), perm, axis=1).astype(np.float32)
    valid = np.take_along_axis(valid, perm, axis=1)
    ncombo = rng.integers(1, 9, (E, H))
    s = simpletc.settings({"TestFlag": False, "useTimeStep": True, "useTimeWindow": False, "TimeStep": 20,
                           "TimeWindow": 100, "minNStrawHits": 2, "minNPanels": 1})
    got = simpletc.clusters({"valid": jnp.asarray(valid), "time": jnp.asarray(t), "ncombo": jnp.asarray(ncombo)}, s)
    gv, gn = np.asarray(got["valid"]), np.asarray(got["nsh"])
    tested = 0
    for e in range(E):
        m = valid[e]
        ref = _stc_reference(t[e][m], ncombo[e][m], 20.0, 100.0, 2)
        assert sorted(gn[e][gv[e]].tolist()) == ref, e
        tested += len(ref)
    assert tested > 100
    with pytest.raises(ValueError):
        simpletc.settings({"TestFlag": True, "useTimeStep": True, "TimeStep": 20, "TimeWindow": 100, "minNStrawHits": 2})


def _lumi_reference(reached, run, subrun, freq, first):
    out, count, first_flag, prev = [], 0, True, None
    for r, ru, su in zip(reached, run, subrun):
        if prev is not None and (ru, su) != prev:      # endSubRun: the counter and the first-event flag restart
            count, first_flag = 0, True
        prev = (ru, su)
        if not r:
            out.append(False)
            continue
        count += 1
        ret = freq > 0 and count % freq == 0
        if ret:
            count = 0
        ret = ret or (first and first_flag)
        first_flag = False
        out.append(ret)
    return out


def test_prescale_digi_and_lumi_filters():
    """PrescaleEvent (event number modulo the factor of the event's mode), DigiFilter on the straw digi count, and the
    host LumiStreamFilter counter against a per-event reference of the stock filter."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from trigger.device import paths
    ev = np.arange(1, 4001)
    spill = np.where(ev % 7 == 0, 1, 0)
    got = np.asarray(paths.prescale_pass(jnp.asarray(ev), jnp.asarray(spill), {"OffSpill": 10, "OnSpill": -1}))
    np.testing.assert_array_equal(got, (spill == 0) & (ev % 10 == 0))
    assert not np.asarray(paths.prescale_pass(jnp.asarray(ev), jnp.asarray(spill), {"OnSpill": 1}))[spill == 0].any()
    nsd = np.arange(0, 30)
    cfg = {"useStrawDigi": True, "useCaloDigi": False, "minNStrawDigi": 8, "maxNStrawDigi": 20}
    np.testing.assert_array_equal(np.asarray(paths.digi_pass(jnp.asarray(nsd), cfg)), (nsd >= 8) & (nsd <= 20))
    rng = np.random.default_rng(5)
    reached = rng.random(3000) < 0.8
    subrun = np.repeat([3, 4, 9, 4], [700, 1300, 2, 998])
    run = np.full(3000, 1430)
    for freq, first in ((7, True), (10000, True), (5, False), (0, True)):
        np.testing.assert_array_equal(paths.lumi_stream(reached, run, subrun, {"eventFreq": freq, "passFirst": first}),
                                      _lumi_reference(reached, run, subrun, freq, first))


def test_offspill_panel_hits_reuse_or_recombine():
    """hits.panel_hits_other: the CombineStrawHits result under other settings, whether the combos are reused (no hit
    changes whether it is combined) or recombined (the on-spill time window differs), bit for bit against a full
    combine of the re-flagged straw hits."""
    pytest.importorskip("jax")
    pytest.importorskip("trkfastsim")
    import jax
    import jax.numpy as jnp
    from trigger.device import hits
    rng = np.random.default_rng(13)
    E, H = 16, 32
    n = rng.integers(0, H + 1, E)
    valid = np.arange(H)[None, :] < n[:, None]
    panel = np.sort(rng.integers(0, 4, (E, H)), axis=1)
    sid = np.where(valid, (panel << 7) | rng.integers(0, 12, (E, H)), 0)
    sid = np.take_along_axis(sid, np.argsort(np.where(valid, sid, 1 << 20), axis=1, kind="stable"), axis=1)
    f32 = lambda a: np.asarray(a, np.float32)
    tdiv = np.where(rng.random((E, H)) < 0.8, 1 << hits.FLAG["tdiv"], 0)
    base = {"valid": valid, "spill": np.zeros(E, np.int64), "sid": sid, "upanel": (sid >> 10) * 6 + ((sid & 0x380) >> 7),
            "straw": sid & 0x7F, "time": f32(rng.uniform(0, 2000, (E, H))), "edep": f32(rng.uniform(0, 0.008, (E, H))),
            "wdist": f32(rng.normal(0, 80, (E, H))), "uvar": f32(np.full((E, H), 900.0)), "vvar": f32(np.full((E, H), 2.0)),
            "timevar": f32(rng.uniform(4, 30, (E, H))), "x": f32(rng.uniform(-800, 800, (E, H))),
            "y": f32(rng.uniform(-800, 800, (E, H))), "z": f32(rng.uniform(-1500, 1500, (E, H))),
            "udx": f32(np.full((E, H), 0.6)), "udy": f32(np.full((E, H), 0.8)), "flag": np.where(valid, tdiv, 0).astype(np.uint32)}
    win = {"minTOff": 0.0, "maxTOff": 100000.0, "fittype": 5}
    s_sh = {"minT": 400.0, "maxT": 1710.0, "minE": 0.0001, "maxE": 0.005, "minR": 350.0, "maxR": 750.0, **win}
    s_sh2 = {"minT": 0.0, "maxT": 100000.0, "minE": 0.0, "maxE": 1.0, "minR": 0.0, "maxR": 800.0, **win}
    cuts = {"maxdt": 45.0, "maxds": 3, "maxwdchi": 5.0, "uerr": 10.0, "checkWres": True, "minN": 1, "maxN": 8,
            "sel": 1 << hits.FLAG["timesel"], "mask": 1 << hits.FLAG["dead"], "minTOff": 0.0, "maxTOff": 100000.0}
    s_ph = {**cuts, "minT": 410.0, "maxT": 1700.0, "minE": 0.0001, "maxE": 0.005, "minR": 380.0, "maxR": 700.0}
    s_ph2 = {**cuts, "minT": 0.0, "maxT": 100000.0, "minE": 0.0, "maxE": 1.0, "minR": 0.0, "maxR": 800.0}
    for spill in (0, 1):                         # off-spill: the combos are reused; on-spill: recombined
        b = {k: jnp.asarray(v) for k, v in {**base, "spill": np.full(E, spill, np.int64)}.items()}
        sh = hits.reflag_straw_hits(b, s_sh)
        sh2 = hits.reflag_straw_hits(b, s_sh2)
        ph = hits.panel_hits(sh, s_ph)
        got = jax.jit(lambda a, p, c: hits.panel_hits_other(a, p, s_ph, c, s_ph2))(sh, ph, sh2)
        ref = hits.panel_hits(sh2, s_ph2)
        same = bool(jnp.all(hits._good(sh, s_ph) == hits._good(sh2, s_ph2)))
        assert same == (spill == 0)
        v = np.asarray(ref["valid"])
        np.testing.assert_array_equal(np.asarray(got["valid"]), v)
        for k in ref:
            np.testing.assert_array_equal(np.asarray(got[k])[v], np.asarray(ref[k])[v], err_msg=f"spill {spill} {k}")



MPR_CFG = {"CaloClusterMinE": 30, "CaloClusterWeight": 5, "ClusteringMinBin": 30, "ClusteringPhiBin": 0.1,
           "DPDZStep": 25, "FitCircleStrategy": "ChiSquared", "Helicities": [-1, 1], "MaxChi2Hit": 50,
           "MaxDPDZSlope": 350, "MaxDPhiHelFit": 0.7, "MaxDPhiHelInit": 1, "MaxDTHelFit": 25, "MaxRadCircle": 350,
           "MinDPDZSlope": 150, "MinDRCircle": 40, "MinDXY2Circle": 1600, "MinDZTrk": 1500, "MinRadCircle": 150,
           "MinStrawHits": 10, "NMaxTrkIter": 3, "TargetCon": False, "TargetRadius": 150, "maxEDepAvg": 2.5e-3}


def _mpr_cluster(rng, L, helices, order="tc"):
    """Synthetic time cluster: hits on `helices` (cx, cy, R, lambda, fz0) at the plane faces, plus noise, in a
    TimeCluster-like order: by panel with some hits moved to the end (recovered hits), or random."""
    zf = np.linspace(-1500, 1500, 72)
    rows = []
    for cx, cy, R, lam, fz0 in helices:
        for z in zf:
            if rng.random() < 0.35:
                continue
            ph = fz0 + z / lam
            rows.append((cx + R * np.cos(ph) + rng.normal(0, 2), cy + R * np.sin(ph) + rng.normal(0, 2), z,
                         600 + z / 250 + rng.normal(0, 3)))
    for _ in range(rng.integers(0, 6)):
        rows.append((rng.uniform(-600, 600), rng.uniform(-600, 600), rng.choice(zf), rng.uniform(500, 1600)))
    rows = rows[:L]
    n = len(rows)
    x, y, z, t = (np.array([r[k] for r in rows], np.float32) for k in range(4))
    plane = np.searchsorted(zf, z) // 2
    panel = ((np.arctan2(y, x) + np.pi) / (np.pi / 3)).astype(int) % 6
    upanel = plane * 6 + panel
    if order == "random":
        o = rng.permutation(n)
    else:
        o = np.lexsort((np.arange(n), upanel))
        tail = rng.random(n) < 0.2
        o = np.concatenate([o[~tail[o]], o[tail[o]]])
    pad = lambda a, v: np.concatenate([a[o], np.full(L - n, v, a.dtype)])
    ang = rng.uniform(0, np.pi, n)
    h = {"x": pad(x, 0), "y": pad(y, 0), "z": pad(z, 0), "time": pad(t, 0),
         "nsh": pad(rng.integers(1, 4, n).astype(np.int32), 0), "uvar": pad(rng.uniform(100, 900, n).astype(np.float32), 1),
         "vvar": pad(rng.uniform(2, 30, n).astype(np.float32), 1), "udx": pad(np.cos(ang).astype(np.float32), 1),
         "udy": pad(np.sin(ang).astype(np.float32), 0), "edep": pad(rng.uniform(5e-4, 2e-3, n).astype(np.float32), 0),
         "upanel": pad(upanel.astype(np.int32), 1 << 20)}
    return h, np.arange(L) < n


def test_mpr_stock_sort_matches_libstdcxx():
    """mpr._stock_sort (the device std::sort) gives the selected slots in the order of the reference's libstdc++
    introsort transcription (itself checked against GCC 14), below and above the 16-element threshold."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    sys_path = str(Path(__file__).parent)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import mpr_reference as ref
    from trigger.device import mpr
    rng = np.random.default_rng(5)
    L, N = 64, 400
    keys = rng.integers(0, rng.integers(2, 40, N)[:, None], (N, L)).astype(np.int32)
    sel = rng.random((N, L)) < rng.uniform(0.1, 1.0, N)[:, None]
    perm, n, fb = jax.jit(jax.vmap(mpr._stock_sort))(jnp.asarray(keys), jnp.asarray(sel))
    perm, n, fb = map(np.asarray, (perm, n, fb))
    assert not fb.any()
    for k in range(N):
        want = ref.std_sort([int(i) for i in np.flatnonzero(sel[k])], lambda i, j: keys[k, i] < keys[k, j])
        assert int(n[k]) == len(want) and perm[k, :n[k]].tolist() == want, k
    assert (np.sum(sel, axis=1) > 16).sum() > 100


def test_mpr_cluster_matches_the_stock_walk():
    """mpr._cluster (the RobustMultiHelixFinder port on one time cluster) against the literal transcription of the
    stock findHelicesInTC: per track iteration, kept or not, helicity, straw hits and hit set equal, and the circle
    and slope parameters bit for bit (the port keeps the stock float / double steps and fuses no multiply-add)."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    sys_path = str(Path(__file__).parent)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import mpr_reference as ref
    from trigger.device import mpr
    s = mpr.settings(MPR_CFG)
    rng = np.random.default_rng(21)
    L = 48
    cases = []
    for c in range(12):
        hel = 1 if c % 2 else -1
        hx = [(rng.uniform(-150, 150), rng.uniform(-150, 150), rng.uniform(200, 320), hel * rng.uniform(170, 300),
               rng.uniform(0, 6))]
        if c % 4 == 3:                                                  # a second track in the cluster
            hx.append((rng.uniform(-150, 150), rng.uniform(-150, 150), rng.uniform(200, 320), -hel * 220.0, 1.0))
        cases.append(_mpr_cluster(rng, L, hx, order="random" if c % 3 == 0 else "tc"))
    hs = {k: jnp.asarray(np.stack([c[0][k] for c in cases])) for k in cases[0][0]}
    ms = jnp.asarray(np.stack([c[1] for c in cases]))

    def one(h, m):
        with mpr.strict(h["upanel"]):
            return mpr._cluster(h, m, s)
    got = jax.tree.map(np.asarray, jax.jit(jax.vmap(one))(hs, ms))
    assert not got["fallback"].any()
    kept = 0
    for c, (h, m) in enumerate(cases):
        want = ref.find_helices_in_tc(h, list(np.flatnonzero(m)), s)
        want += [None] * (s["niter"] - len(want))
        for it, w in enumerate(want):
            assert bool(got["ok"][c, it]) == (w is not None), (c, it)
            if w is None:
                continue
            kept += 1
            assert int(got["hel"][c, it]) == w["hel"] and int(got["nsh"][c, it]) == w["nsh"], (c, it)
            assert sorted(np.flatnonzero(got["mask"][c, it]).tolist()) == w["hits"], (c, it)
            for k in ("x", "y", "r", "fa", "fb"):
                g, r = np.float32(got[{"x": "cx", "y": "cy"}.get(k, k)][c, it]), np.float32(w[k])
                assert g == r, f"{c} {it} {k}: {g!r} != {r!r}"
    assert kept >= 6
