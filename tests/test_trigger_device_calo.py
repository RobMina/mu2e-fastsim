"""Device calo trigger stage on synthetic digis: the geometry as Offline builds it, and the hit, cluster and filter
ports against the literal stock transcription (`calo_reference`), bit for bit."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

HITS = {"deltaTPulses": 11.0, "nPEperMeV": 30.0, "noiseLevelMeV": 0.55, "nSigmaNoise": 4.0,
        "caphriEDepMax": 5.0, "caphriEDepMin": 1.0}
CLUSTERS = {"EminSeed": 10.0, "EnoiseCut": 1.0, "ExpandCut": 1.0, "deltaTime": 10.0, "timeOffset": -61.0,
            "minSiPMPerHit": 2, "extendSearch": True}
CALIB = {"ADC2MeVCsI": 0.0625, "ADC2MeVlyso": 0.00375, "timeoffset": 0.0, "useDb": False}
COUNTER = {"DiskID": [0, 1], "MinClusterEnergy": 50.0, "MaxClusterEnergy": 200.0, "MinNCel": 1, "MaxNCel": 20,
           "MinClusterRadius": 0.0, "MinNCl": 1}
RMC = {"DiskID": [0, 1], "MinClusterEnergy": 20.0, "MaxClusterEnergy": 200.0, "MinNCel": 2, "MaxNCel": 10,
       "MinClusterRadius": 490.0, "MinNCl": 1}
COSMIC = {"MinNCrystalHits": 0, "MinEnergy": 20.0, "MaxEnergy": 600.0, "MinNumCelinout": 1, "MinNumCeldiagver": 2,
          "MinCelEneCut": 3.0, "OutRadius": 580.0, "InnerRadius": 460.0, "diagminval": 1.5, "diagmaxval": 2.5,
          "dyminval": 50.0}
NN_EVAL = {"minEtoTest": 20.0, "maxEtoTest": 2000.0, "minRtoTest": 0.0, "maxRtoTest": 1000.0, "minTtoTest": 0.0,
           "maxTtoTest": 1600.0}
NN_FILTER = {**NN_EVAL, "minMVAScore": 0.5}
LH = {"MinClusterEnergy": 20.0, "ClusterEnergyStep": 10.0, "MinClusterRadialDist": 350.0,
      "ClusterRadialDistStep": 50.0, "MinLikelihoodCut": [-5.0, -5.0]}


def _geo():
    from trigger.device import calogeom
    return calogeom.build()


def test_calo_geometry_as_offline_builds_it():
    """674 crystals per disk from Offline's crystal file; first ring of at most 6 and second of at most 12
    neighbours, symmetric, on the same disk; the FF frame is the disk-local x, y."""
    g = _geo()
    assert g["disk"].tolist() == [0] * 674 + [1] * 674
    nb, nn = g["neighbors"], g["next_neighbors"]
    assert (nb >= 0).sum(1).max() == 6 and (nn >= 0).sum(1).max() == 12
    for c in range(1348):
        for x in nb[c][nb[c] >= 0]:
            assert c in nb[x] and g["disk"][x] == g["disk"][c]
        for x in nn[c][nn[c] >= 0]:
            assert c in nn[x] and x not in nb[c]
    assert np.array_equal(g["ff"][:, 0], g["local_x"]) and np.array_equal(g["ff"][:, 1], g["local_y"])
    assert np.allclose(g["ff"][:, 2], 0.0, atol=1e-9)
    assert g["caphri"].sum() == 4 and abs(g["crystal_size"] - 34.3) < 1e-12


def _events(rng, n_events, geo):
    """Synthetic digis: a few showers of neighbouring crystals per event, two SiPMs each (with readout noise), some
    lone SiPMs, a pin diode, a short waveform; and pulses at other times."""
    nb = geo["neighbors"]
    rows = []
    for e in range(n_events):
        ev = []
        for _ in range(rng.integers(0, 4)):
            seed = int(rng.integers(0, 1348))
            ring = [seed] + [int(x) for x in nb[seed] if x >= 0 and rng.random() < 0.7]
            for x in list(ring[1:]):
                ring += [int(y) for y in nb[x] if y >= 0 and rng.random() < 0.2]
            t0 = int(rng.integers(40, 300))
            e0 = rng.uniform(5, 120)
            for k, c in enumerate(dict.fromkeys(ring)):
                ec = e0 * (1.0 if k == 0 else rng.uniform(0.01, 0.4))
                dt = int(rng.integers(0, 2))
                pk0 = int(rng.integers(2, 20))
                for sp in (0, 1):
                    if rng.random() < 0.05:
                        continue
                    amp = max(1, int(ec / 0.0625 * rng.normal(1, 0.04)))
                    pk = pk0 + int(rng.integers(0, 2))
                    length = int(rng.integers(3, 40)) if rng.random() < 0.03 else int(rng.integers(pk + 2, 60))
                    wf = (rng.integers(0, 6, length) + 50).tolist()
                    if pk < length:
                        wf[pk] = 50 + amp
                    ev.append((2 * c + sp, t0 + dt, min(pk, length - 1), wf))
        for _ in range(rng.integers(0, 3)):
            c = int(rng.integers(0, 1348))
            ev.append((2 * c, int(rng.integers(0, 400)), 5, (rng.integers(0, 6, 30) + 50 + 300 * (np.arange(30) == 5)).tolist()))
        if rng.random() < 0.2:
            ev.append((2696 + int(rng.integers(0, 4)), 100, 5, [50] * 30))
        order = rng.permutation(len(ev))
        rows.append([ev[i] for i in order])
    return rows


def _flat(rows):
    out = {k: [] for k in ("entry", "sipm", "t0", "peakpos", "wf_start", "wf_len")}
    samples = []
    for e, ev in enumerate(rows):
        for sipm, t0, pk, wf in ev:
            out["entry"].append(e); out["sipm"].append(sipm); out["t0"].append(t0); out["peakpos"].append(pk)
            out["wf_start"].append(len(samples)); out["wf_len"].append(len(wf)); samples += wf
    out = {k: np.asarray(v, np.int64) for k, v in out.items()}
    out["samples"] = np.asarray(samples, np.int64)
    out["n"] = np.asarray([len(ev) for ev in rows], np.int64)
    return out


@pytest.fixture(scope="module")
def stage():
    pytest.importorskip("jax")
    import jax
    from trigger.device import calo
    import calo_reference as ref
    geo = _geo()
    s = calo.settings(HITS, CLUSTERS, CALIB, geo)
    rng = np.random.default_rng(11)
    rows = _events(rng, 96, geo)
    pbt = rng.uniform(-230, -190, len(rows)).astype(np.float32)
    dg = calo.pack_digis(_flat(rows), len(rows))
    keys = ("valid", "sipm", "t0", "peakpos", "wf_len", "s0", "s1", "s2", "s3", "s_peak")
    hits, cl = jax.jit(lambda d, p: calo.reconstruct(d, p, s))({k: dg[k] for k in keys}, pbt)
    hits = {k: np.asarray(v) for k, v in hits.items()}
    cl = {k: np.asarray(v) for k, v in cl.items()}
    adc = lambda sipm: np.float32(0.00375) if geo["caphri"][sipm // 2] else np.float32(0.0625)
    want = []
    for e, ev in enumerate(rows):
        h, _ = ref.make_hits([r[0] for r in ev], [r[1] for r in ev], [r[2] for r in ev], [r[3] for r in ev],
                             pbt[e], HITS, adc)
        want.append((h, ref.make_clusters(h, geo, CLUSTERS)))
    return {"s": s, "geo": geo, "hits": hits, "cl": cl, "want": want, "dg": dg, "calo": calo, "ref": ref}


def test_calo_hits_and_clusters_match_the_stock_transcription(stage):
    hits, cl, want = stage["hits"], stage["cl"], stage["want"]
    n_cl = 0
    assert not hits["tie"].any() and not cl["tie"].any() and not cl["fallback"].any()
    for e, (h, c) in enumerate(want):
        n = int(hits["n"][e])
        assert n == len(h), e
        got = [(int(hits["crystal"][e, i]), int(hits["nsipm"][e, i]), hits["time"][e, i], hits["edep"][e, i])
               for i in range(n)]
        assert got == h, e
        assert int(cl["n"][e]) == len(c), e
        for j, ref_c in enumerate(c):
            for k in ("disk", "time", "energy", "cog_x", "cog_y", "size"):
                assert cl[k][e, j] == ref_c[k], (e, j, k)
            assert cl["members"][e, j, :ref_c["size"]].tolist() == ref_c["members"], (e, j)
            n_cl += 1
    assert n_cl > 100 and max(len(c) for _, c in want) >= 3
    assert any(m["size"] >= 4 for _, c in want for m in c)


def test_calo_filters_match_the_stock_transcription(stage, tmp_path):
    import jax
    import jax.numpy as jnp
    calo, ref = stage["calo"], stage["ref"]
    s, geo = stage["s"], stage["geo"]
    hits = {k: jnp.asarray(v) for k, v in stage["hits"].items()}
    cl = {k: jnp.asarray(v) for k, v in stage["cl"].items()}
    rng = np.random.default_rng(3)
    net = {"act": "relu", "norm": True,
           "offset": np.array([0, 1, 0, 0, 0, 0, 0], np.float32), "scale": np.full(7, 0.25, np.float32),
           "layers": [rng.normal(0, 1, (8, 7)).astype(np.float32), rng.normal(0, 1, (8, 1)).astype(np.float32)]}
    tpl_paths = _write_templates(tmp_path, rng)
    tpl = calo.load_templates(*tpl_paths, LH["MinClusterRadialDist"], LH["ClusterRadialDistStep"])
    rtpl = ref.Templates(*tpl_paths, LH["MinClusterRadialDist"], LH["ClusterRadialDistStep"])
    with calo.strict(jnp.asarray(stage["dg"]["sipm"])):
        got = {"counter": calo.counter_pass(cl, COUNTER), "rmc": calo.counter_pass(cl, RMC),
               "cosmic": calo.cosmic_pass(cl, hits, s, COSMIC), "cosmic_int": calo.cosmic_pass(cl, hits, s, COSMIC, True),
               "nn": calo.nn_pass(cl, hits, s, net, NN_EVAL, NN_FILTER),
               "lh": calo.likelihood_pass(cl, hits, s, tpl, LH)}
    got = {k: np.asarray(v) for k, v in got.items()}
    for e, (h, c) in enumerate(stage["want"]):
        assert got["counter"][e] == ref.counter_pass(c, COUNTER), e
        assert got["rmc"][e] == ref.counter_pass(c, RMC), e
        assert got["cosmic"][e] == ref.cosmic_pass(c, h, geo, COSMIC), e
        assert got["cosmic_int"][e] == ref.cosmic_pass(c, h, geo, COSMIC, int_abs=True), e
        assert got["nn"][e] == ref.nn_pass(c, ref.nn_scores(c, h, geo, net, NN_EVAL), NN_FILTER), e
        assert got["lh"][e] == ref.likelihood_pass(c, h, geo, rtpl, LH), e
    for k in ("counter", "nn", "lh"):
        assert 0 < got[k].sum() < got[k].size, (k, got[k].sum())


def _write_templates(tmp_path, rng):
    """Small CaloLikelihood templates in the stock layout (TH1F rDist, TH2F x = radius, y = variable)."""
    uproot = pytest.importorskip("uproot")
    names2 = {"eRDist": (0, 800), "e1RDist": (0, 1.2), "e2RDist": (0, 1.2), "e2e1RDist": (0, 1.2), "e3RDist": (0, 1.2),
              "e4RDist": (0, 1.2), "nCrRDist": (0, 20)}
    paths = []
    for kind, sub, suffix in (("sig", "cluster_electron", "2"), ("bkg", "cluster_all", "0")):
        p = tmp_path / f"lh_{kind}.root"
        with uproot.recreate(p) as f:
            for disk in (0, 1):
                f[f"ceAna/{sub}/clDisk{disk}_rDist{suffix}"] = (rng.random(40).astype(np.float32) + 0.05,
                                                               np.linspace(0, 800, 41))
                for nm, (lo, hi) in names2.items():
                    f[f"ceAna/{sub}/clDisk{disk}_{nm}{suffix}"] = (
                        (rng.random((80, 24)) * (rng.random((80, 24)) > 0.2)).astype(np.float32),
                        np.linspace(0, 800, 81), np.linspace(lo, hi, 25))
        paths.append(str(p))
    return paths


def test_calo_digi_count_and_paths_plan():
    pytest.importorskip("jax")
    from trigger.device import calo, calopaths, menu
    n = np.array([0, 199, 200, 5000, 5001])
    ok = np.asarray(calo.digi_count_pass(n, {"useCaloDigi": True, "useStrawDigi": False, "minNCaloDigi": 200,
                                             "maxNCaloDigi": 5000, "maxCaloEnergy": -1}))
    assert ok.tolist() == [False, False, True, True, False]
    cfg = menu.parse_fcl(PLAN_MENU)
    assert calopaths.plan(cfg) == {"calo_cluster_50": ("counter", ["F50"]), "calo_NNCE": ("nn", ["NE", "NF"]),
                                   "minBias_CDCount": ("digicount", ["DC"])}


PLAN_MENU = """
physics: {
  producers: {
    CaloHitMakerFast: { module_type: "CaloHitMakerFast" }
    CaloClusterFast: { module_type: "CaloClusterFast" }
    NE: { module_type: "CaloNNEval" }
    TZ: { module_type: "TZClusterFinder" }
  }
  filters: {
    PS: { module_type: "PrescaleEvent" }
    F50: { module_type: "CaloClusterCounterFilter" }
    NF: { module_type: "FilterEcalNNTrigger" }
    DC: { module_type: "DigiFilter" useCaloDigi: true }
    TCF: { module_type: "TimeClusterFilter" }
    TI: { module_type: "MergeTriggerInfo" }
  }
  calo_cluster_50: [ PS, CaloHitMakerFast, CaloClusterFast, F50, TI ]
  calo_NNCE: [ PS, CaloHitMakerFast, CaloClusterFast, NE, NF, TI ]
  minBias_CDCount: [ PS, DC ]
  apr_TC_calo: [ PS, CaloHitMakerFast, CaloClusterFast, TZ, TCF, TI ]
  trigger_paths: [ "425:calo_cluster_50", "401:calo_NNCE", "610:minBias_CDCount", "511:apr_TC_calo" ]
}
"""


def test_calo_seed_sort_is_libstdcxx_on_the_tolerant_comparator():
    """calo._sort_lt gives the order of the reference's libstdc++ std::sort with CaloClusterFast's comparator
    (|dt| < 1e-3 -> crystal order, else time order; not a strict weak order), below and above 16 elements."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    import calo_reference as ref
    from trigger.device import calo
    rng = np.random.default_rng(7)
    N, L = 300, 40
    base = rng.choice([500.0, 502.5, 505.0, 507.5], (N, L)).astype(np.float32)
    jitter = (rng.integers(-3, 4, (N, L)) * np.float32(6.1e-5)).astype(np.float32)   # a few ulps at 500 ns
    t = (base + jitter).astype(np.float32)
    cry = rng.integers(0, 1348, (N, L)).astype(np.int32)
    sel = rng.random((N, L)) < rng.uniform(0.2, 1.0, N)[:, None]

    def lt_of(tt, cc):
        dtm = tt[:, None] - tt[None, :]
        return jnp.where(jnp.abs(dtm).astype(jnp.float64) < 1e-3, cc[:, None] < cc[None, :], dtm < 0)

    with jax.enable_x64(True):
        perm, n, fb = jax.jit(jax.vmap(lambda tt, cc, ss: calo._sort_lt(lt_of(tt, cc), ss)))(
            jnp.asarray(t), jnp.asarray(cry), jnp.asarray(sel))
    perm, n, fb = map(np.asarray, (perm, n, fb))
    big = 0
    for k in range(N):
        def less(a, b):
            dt = np.float32(t[k, a] - t[k, b])
            return bool(cry[k, a] < cry[k, b]) if float(abs(dt)) < 1e-3 else bool(dt < 0)
        want = ref.std_sort([int(i) for i in np.flatnonzero(sel[k])], less)
        assert not fb[k] and perm[k, :n[k]].tolist() == want, k
        big += len(want) > 16
    assert big > 100


def _bc(payload):
    import struct
    return struct.pack(">I", 0x40000000 | len(payload)) + payload


def test_calo_art_decoders_on_memberwise_blobs():
    """caloart decodes the memberwise CaloDigi (v10), CaloCluster (v1, with checksum), ProtonBunchTime and
    HLTPathStatus layouts, including empty collections, and rejects a wrong length."""
    import struct
    from trigger.device import caloart
    digis = [[(110, 282, 16, [13, 14, 15, 700, 12]), (111, 281, 2, [9, 8, 300])], []]
    blobs = []
    for ev in digis:
        n = len(ev)
        body = struct.pack(">HHi", 0x400A, 10, n)
        body += b"".join(struct.pack(">i", d[0]) for d in ev) + b"".join(struct.pack(">i", d[1]) for d in ev)
        if n:
            wf = b"".join(struct.pack(">i", len(d[3])) + b"".join(struct.pack(">i", x) for x in d[3]) for d in ev)
            body += _bc(struct.pack(">H", 10) + wf)
        body += b"".join(struct.pack(">i", d[2]) for d in ev)
        blobs.append(_bc(body))
    data = np.frombuffer(b"".join(blobs), np.uint8)
    off = np.r_[0, np.cumsum([len(b) for b in blobs])]
    dg = caloart.decode_digis(data, off)
    assert dg["n"].tolist() == [2, 0] and dg["sipm"].tolist() == [110, 111] and dg["t0"].tolist() == [282, 281]
    assert dg["peakpos"].tolist() == [16, 2] and dg["wf_len"].tolist() == [5, 3]
    assert dg["samples"][dg["wf_start"][1]:dg["wf_start"][1] + 3].tolist() == [9, 8, 300]
    with pytest.raises(ValueError):
        caloart.decode_digis(data[:-4], off - np.r_[0, 0, 4])

    cl = [(1, 660.5, 14.9765625, -154.35, -445.9, 1, False), (0, 1646.75, 80.328125, -422.15, 12.9, 4, True)]
    body = struct.pack(">HHIi", 0x400A, 0, 0xDC2D5AD4, len(cl))
    body += b"".join(struct.pack(">i", c[0]) for c in cl)
    for k in (1, None, 2, None):
        body += b"".join(struct.pack(">f", c[k] if k is not None else 0.0) for c in cl)
    body += b"".join(_bc(struct.pack(">H", 11) + struct.pack(">ddd", c[3], c[4], 0.0)) for c in cl)
    body += _bc(struct.pack(">HH", 0x400A, 10) + b"\x00" * 20)                      # Ptr block, skipped
    body += b"".join(struct.pack(">I", c[5]) for c in cl) + bytes(int(c[6]) for c in cl)
    blob = _bc(body)
    empty = _bc(struct.pack(">HHIi", 0x400A, 0, 0xDC2D5AD4, 0))
    data = np.frombuffer(blob + empty, np.uint8)
    c = caloart.decode_clusters(data, np.array([0, len(blob), len(blob) + len(empty)]))
    assert c["n"].tolist() == [2, 0] and c["disk"].tolist() == [1, 0] and c["size"].tolist() == [1, 4]
    assert c["energy"].tolist() == [np.float32(14.9765625), np.float32(80.328125)] and c["cog_x"].tolist() == [-154.35, -422.15]
    assert c["is_split"].tolist() == [False, True]

    pbt = b"".join(_bc(struct.pack(">HIff", 0, 0xD7B521F8, v, 0.5)) for v in (-217.73679, -221.5181))
    assert caloart.decode_pbt(np.frombuffer(pbt, np.uint8), np.array([0, 18, 36])).tolist() == \
        [np.float32(-217.73679), np.float32(-221.5181)]

    words = [0x29, 0x31, 0x2E]
    vec = _bc(struct.pack(">HHI", 0x400A, 10, len(words)) + b"".join(struct.pack(">H", w) for w in words))
    tr = _bc(struct.pack(">H", 12) + _bc(struct.pack(">H", 10) + vec) + b"\x00" * 24)
    got = caloart.decode_trigger(np.frombuffer(tr, np.uint8), np.array([0, len(tr)]))
    assert got.tolist() == [words] and caloart.accepted(got).tolist() == [[True, True, False]]
