"""Device cpr stage on synthetic inputs: CalTimePeakFinder against a literal transcription of the stock module, the
TimeClusterFilter on its peaks, the StrawDigi / fixed-size art decoders and the path-status stage helper."""
from pathlib import Path
import math
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TPF = {"Beta": 1, "BkgSelBits": ["Background", "Noisy", "Dead"], "DtMax": 20, "DtMin": -20, "DtOffset": 0,
       "HitSelBits": ["TimeSelection", "EnergySelection"], "MinClusterEnergy": 50, "MinClusterSize": 2,
       "MinNHits": 10, "PitchAngle": 0.67}


def _poly_atan2(y, x):
    """Mu2eUtilities polyAtan2, literally (float arguments, the M_PI branch in double)."""
    f = np.float32
    y, x = f(y), f(x)
    at = lambda z: (f(0.97239411) + f(-0.19194795) * z * z) * z
    if x != 0:
        if abs(x) > abs(y):
            z = f(y / x)
            if x > 0:
                return at(z)
            return f(float(at(z)) + math.pi) if y >= 0 else f(float(at(z)) - math.pi)
        z = f(x / y)
        return -at(z) + f(math.pi / 2.) if y > 0 else -at(z) - f(math.pi / 2.)
    return f(math.pi / 2.) if y > 0 else (-f(math.pi / 2.) if y < 0 else f(0.))


def _reference(hits, clusters, cfg, origin):
    """CalTimePeakFinder::findTimePeaks, transcribed: per cluster, (selected hit indices, nsh) of the peaks made."""
    from trigger.device.cpr import C_LIGHT, DETECTOR_SYSTEM_Z0
    from trigger.device.hits import STRAW_FLAGS
    hsel = sum(1 << STRAW_FLAGS[b] for b in cfg["HitSelBits"])
    bkg = sum(1 << STRAW_FLAGS[b] for b in cfg["BkgSelBits"] if b in STRAW_FLAGS)
    sinp = math.sin(cfg["PitchAngle"])
    out = []
    for c in clusters:
        if not (float(np.float32(c["energy"])) >= cfg["MinClusterEnergy"] and c["size"] >= cfg["MinClusterSize"]):
            out.append(None)
            continue
        cl_time = float(np.float32(c["time"])) + cfg["DtOffset"]
        g = origin[c["disk"]]
        xcl, ycl, zcl = (g[0] + c["cog_x"]) - (-3904.0), (g[1] + c["cog_y"]) - 0.0, (g[2] + 0.0) - DETECTOR_SYSTEM_Z0
        mphi = float(_poly_atan2(ycl, xcl))
        sel, nsh = [], 0
        for i, h in enumerate(hits):
            if not (h["flag"] & hsel) or (h["flag"] & bkg):
                continue
            tof = (zcl - float(np.float32(h["z"]))) / sinp / (C_LIGHT * cfg["Beta"])
            dt = cl_time - (float(np.float32(h["time"])) + tof)
            if cfg["DtMin"] <= dt < cfg["DtMax"]:
                dphi = float(_poly_atan2(h["y"], h["x"])) - mphi
                if dphi > math.pi:
                    dphi -= 2 * math.pi
                if dphi < -math.pi:
                    dphi += 2 * math.pi
                if abs(dphi) <= math.pi / 2.:
                    sel.append(i)
                    nsh += h["nsh"]
        out.append((sel, nsh) if nsh >= cfg["MinNHits"] else None)
    return out


def _event(rng, n_hits, n_cl, geo):
    from trigger.device.hits import STRAW_FLAGS
    good = (1 << STRAW_FLAGS["TimeSelection"]) | (1 << STRAW_FLAGS["EnergySelection"])
    hits = []
    for _ in range(n_hits):
        r, phi = rng.uniform(380, 700), rng.uniform(-math.pi, math.pi)
        flag = good if rng.random() > 0.1 else (good | (1 << STRAW_FLAGS["Noisy"]) if rng.random() > 0.5 else 0)
        hits.append({"x": np.float32(r * math.cos(phi)), "y": np.float32(r * math.sin(phi)),
                     "z": np.float32(rng.uniform(-1500, 1500)), "time": np.float32(rng.uniform(500, 600)),
                     "flag": flag, "nsh": int(rng.integers(1, 3))})
    cls = []
    for _ in range(n_cl):
        r, phi = rng.uniform(400, 600), rng.uniform(-math.pi, math.pi)
        cls.append({"disk": int(rng.integers(0, 2)), "time": np.float32(rng.uniform(560, 640)),
                    "energy": np.float32(rng.uniform(30, 110)), "size": int(rng.integers(1, 6)),
                    "cog_x": r * math.cos(phi), "cog_y": r * math.sin(phi)})
    return hits, cls


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_time_peaks_match_the_stock_transcription(precision):
    """CalTimePeakFinder on 40 synthetic events (De and Ue settings): the device peaks, their hits and nsh equal the
    transcription's, and the TimeClusterFilter passes as stock's would."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", precision == "float64")
    from trigger.device import calogeom, cpr
    geo = calogeom.build()
    rng = np.random.default_rng(7)
    E, H, C = 40, 48, 4
    evs = [_event(rng, int(rng.integers(5, H)), int(rng.integers(0, C + 1)), geo) for _ in range(E)]
    ph = {k: np.zeros((E, H), d) for k, d in (("x", np.float32), ("y", np.float32), ("z", np.float32),
                                               ("time", np.float32), ("flag", np.int64), ("ncombo", np.int64))}
    ph["valid"] = np.zeros((E, H), bool)
    cl = {k: np.zeros((E, C), d) for k, d in (("disk", np.int32), ("time", np.float32), ("energy", np.float32),
                                               ("size", np.int32), ("cog_x", np.float64), ("cog_y", np.float64))}
    cl["valid"] = np.zeros((E, C), bool)
    for e, (hs, cs) in enumerate(evs):
        for i, h in enumerate(hs):
            for k in ("x", "y", "z", "time", "flag"):
                ph[k][e, i] = h[k]
            ph["ncombo"][e, i] = h["nsh"]
            ph["valid"][e, i] = True
        for j, c in enumerate(cs):
            for k in ("disk", "time", "energy", "size", "cog_x", "cog_y"):
                cl[k][e, j] = c[k]
            cl["valid"][e, j] = True
    n_peaks = 0
    for cfg in (TPF, {**TPF, "PitchAngle": -0.67, "DtOffset": -3.1}):
        s = cpr.settings(cfg, geo)
        pk = jax.device_get(cpr.jit_time_peaks(s)(ph, cl))
        for e, (hs, cs) in enumerate(evs):
            ref = _reference(hs, cs, cfg, np.asarray(geo["origin"]))
            for j in range(C):
                r = ref[j] if j < len(cs) else None
                assert bool(pk["tc"][e, j]) == (r is not None), (e, j)
                if r is not None:
                    n_peaks += 1
                    assert np.flatnonzero(pk["hits"][e, j]).tolist() == r[0]
                    assert int(pk["nsh"][e, j]) == r[1]
        tcf = jax.device_get(cpr.tc_filter_pass(pk, {"minNStrawHits": 15}))
        for e, (hs, cs) in enumerate(evs):
            ref = _reference(hs, cs, cfg, np.asarray(geo["origin"]))
            assert bool(tcf[e]) == any(r is not None and r[1] >= 15 for r in ref)
    assert n_peaks > 10                     # the sample makes peaks with both settings


def _bc(body):
    return struct.pack(">I", 0x40000000 | len(body)) + body


def test_straw_art_decoders_on_memberwise_blobs():
    """strawart decodes the memberwise StrawDigi layout (16 + 47 n bytes) and the fixed-size EventWindowMarker and
    IntensityInfoTimeCluster blobs, and rejects a wrong length."""
    from trigger.device import strawart
    digis = [[(0x4104, (13112, 13228), (4, 4), 86), (0x48F7, (20000, 20010), (6, 5), 131)], []]
    blobs = []
    for ev in digis:
        body = struct.pack(">HHIi", 0x400A, 0, 0x0D8B3C88, len(ev))
        body += b"".join(_bc(struct.pack(">HIH", 0, 0x48F70F97, d[0])) for d in ev)
        body += b"".join(struct.pack(">II", *d[1]) for d in ev) + b"".join(struct.pack(">HH", *d[2]) for d in ev)
        body += b"".join(_bc(struct.pack(">HI", 0, 0xE0AECEFF) + _bc(struct.pack(">HI", 0, 0xEDE1B0C8)) + b"\x00")
                         for _ in ev)
        body += b"".join(struct.pack(">H", d[3]) for d in ev)
        blobs.append(_bc(body))
    assert [len(b) for b in blobs] == [16 + 47 * 2, 16]
    data = np.frombuffer(b"".join(blobs), np.uint8)
    off = np.r_[0, np.cumsum([len(b) for b in blobs])]
    sd = strawart.decode_straw_digis(data, off)
    assert sd["sid"][0].tolist() == [0x4104, 0x48F7] and sd["tdc0"][0].tolist() == [13112, 20000]
    assert sd["tdc1"][0].tolist() == [13228, 20010] and sd["tot1"][0].tolist() == [4, 5]
    assert sd["pmp"][0].tolist() == [86, 131] and sd["sid"][1].size == 0
    with pytest.raises(ValueError):
        strawart.decode_straw_digis(data[:-2], off - np.r_[0, 0, 2])


def test_path_stage_from_the_status_index():
    """stage_passed: art's index runs one ahead of the listed modules; passed and ready count as past every stage."""
    from trigger.device import strawart
    n = 5                                     # listed modules 0..4, TimeClusterFilter at 2, HelixFilter at 3
    words = np.array([(5 << 2) | 1, (3 << 2) | 2, (4 << 2) | 2, (0 << 2) | 0], np.uint16)
    assert strawart.stage_passed(words, 2, n).tolist() == [True, False, True, True]
    assert strawart.stage_passed(words, 3, n).tolist() == [True, False, False, True]
    assert strawart.stage_passed(words, n, n).tolist() == [True, False, False, True]
