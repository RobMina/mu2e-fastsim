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


# --- CalHelixFinder + MergeHelices (trigger/device/calhelix.py) against the literal transcriptions ----------------

CHF = {"Helicities": [-1, 1], "maxEDepAvg": 2.5e-3, "minNHitsTimeCluster": 10,
       "HelixFinderAlg": {"BackgroundSelectionBits": ["Background", "Noisy", "Dead"], "HelixFitSelectionBits": [],
                          "chi2hel3DMax": 9, "chi2xyMax": 5, "chi2zphiMax": 5, "dfdzErr": 1e-1, "distPatRec": 100,
                          "filter": True, "hitChi2Max": 25, "initDfDz": 0, "maxAbsTanDip": 2, "maxDfDz": 2.5e-2,
                          "maxElectronHitEnergy": 5e-3, "maxNHitsRatio": 1.5, "maxP": 150,
                          "maxPanelToHelixDPhi": 1.309, "maxXDPhi": 5, "minAbsTanDip": 3e-1, "minArea": 5000,
                          "minDfDz": 2e-3, "minNHit": 10, "minP": 50, "mindist": 500, "mostProbableDfDz": 4.75e-3,
                          "procAllTCs": False, "sigmaPhi": 1.636e-1, "slopeRatioLimit": 8e-1, "targetconsistent": 0,
                          "weight3D": 1.2e-1, "weightXY": 1.11, "weightZPhi": 7.5e-1}}
MERGE = {"MinHitOverlapFraction": 5e-1, "MinNHitOverlap": 10, "deltanh": 5, "scaleXY": 1.1, "scaleZPhi": 7.5e-1}
HELIX_FIELDS = ("rcent", "fcent", "radius", "lambda", "fz0", "chi2xy", "chi2zphi", "eDepAvg")


def _chf_geometry():
    """zFace from the nominal straws, a synthetic panel layout (three panels 120 degrees apart per face), the stock
    calorimeter disks and offset; and the HelixTool face z."""
    import cal_helix_reference as CH
    from trigger.device import calhelix, calogeom, helix
    g = helix.TRACKER_GEOM
    mz = [[[helix.nominal_straw_z(g, p, q, s) for s in (0, 1)] for q in range(6)] for p in range(36)]
    po, inv = [], {}
    for p in range(36):
        row = []
        for q in range(6):
            st, of, op = CH.order_id(p // 2, p % 2, q)
            fid = of + st * 4
            a = op * 2 * math.pi / 3 + 0.37 * fid
            row.append((600 * math.cos(a), 600 * math.sin(a)))
            inv[(fid, op)] = (p, q)
        po.append(row)
    zf, pp = CH.module_geometry(mz, po)
    zf2, pp2 = calhelix.module_geometry(mz, po)
    assert np.array_equal(np.asarray(zf, np.float32), zf2) and np.array_equal(np.asarray(pp, np.float32), pp2)
    geo = calogeom.build()
    cg = calogeom.CALO_GEOM
    off = CH.calo_offset(cg["crystalZLength"] + cg["crystalCapZLength"], cg["BPPipeZOffset"], cg["BPHoleZLength"],
                         cg["FEEZLength"], cg["FPCarbonZLength"], cg["FPFoamZLength"])
    G = {"zFace": zf2, "phiPanel": pp2, "bz": 1.0, "fCaloOffset": np.float32(off),
         "disk_origin": np.asarray(geo["origin"], np.float64), "tracker_center": np.array([-3904.0, 0.0, 10171.0])}
    fz = np.array([[np.float32((helix.nominal_straw_z(g, p, f, 0) + helix.nominal_straw_z(g, p, f, 1)) / 2.)
                    for f in (0, 1)] for p in range(36)]).ravel().astype(np.float64)
    return G, inv, fz


def _chf_hit(rng, G, inv, fid, x, y, flag=0):
    a = math.atan2(y, x)
    pp = G["phiPanel"][fid * 3:fid * 3 + 3].astype(float)
    op = int(np.argmin([abs((a - p + math.pi) % (2 * math.pi) - math.pi) for p in pp]))
    plane, panel = inv[(fid, op)]
    w = pp[op]
    ux, uy = -math.sin(w), math.cos(w)
    along, across = rng.normal(0, 20), rng.normal(0, 2)
    return {"x": np.float32(x + along * ux - across * uy), "y": np.float32(y + along * uy + across * ux),
            "z": np.float32(G["zFace"][fid] + rng.uniform(-6, 6)), "time": np.float32(rng.uniform(500, 600)),
            "edep": np.float32(rng.uniform(0.0008, 0.0024)), "uvar": np.float32(rng.uniform(400, 2500)),
            "vvar": np.float32(rng.uniform(2, 8)), "udx": np.float32(ux), "udy": np.float32(uy),
            "sid": (plane << 10) | (panel << 7) | int(rng.integers(0, 96)), "flag": flag,
            "nsh": int(rng.integers(1, 3))}


def _chf_track(rng, G, inv, Q, zc, hel, eff=0.92):
    """A helix through the target axis and the calo point Q at z = zc: the circle of radius R through (0, 0) and Q
    on the side that `hel` picks, dphi/dz of the helicity's sign."""
    R = rng.uniform(260, 320)
    q = np.asarray(Q, float)
    d = np.linalg.norm(q)
    mid, perp = q / 2, np.array([-q[1], q[0]]) / d
    h = math.sqrt(max(R * R - d * d / 4, 1.0))
    c = mid + (h if hel > 0 else -h) * perp
    dfdz = hel / rng.uniform(170, 250)
    phic = math.atan2(q[1] - c[1], q[0] - c[0])
    hits = []
    for fid in range(72):
        z = float(G["zFace"][fid])
        phi = phic + dfdz * (z - zc)
        x, y = c[0] + R * math.cos(phi), c[1] + R * math.sin(phi)
        if 390 < math.hypot(x, y) < 690 and rng.random() < eff:
            hits.append(_chf_hit(rng, G, inv, fid, x, y))
    return hits


def _chf_events(rng, G, inv, n):
    """Synthetic events of panel hits, calo clusters and time peaks (members): noise only, one track, two tracks of
    opposite helicity on one cluster, the same track in two peaks (MergeHelices removes one), and noise peaks."""
    evs = []
    for i in range(n):
        kind = i % 5
        disk = int(rng.integers(0, 2))
        zc = float(G["disk_origin"][disk][2] - G["tracker_center"][2] - float(G["fCaloOffset"]))
        a = rng.uniform(-math.pi, math.pi)
        Q = (rng.uniform(420, 600) * math.cos(a), rng.uniform(420, 600) * math.sin(a))
        noise = [_chf_hit(rng, G, inv, int(rng.integers(0, 72)), *(r * np.array([math.cos(b), math.sin(b)])),
                          flag=(1 << 24) if rng.random() < 0.1 else 0)
                 for r, b in zip(rng.uniform(390, 690, 30), rng.uniform(-math.pi, math.pi, 30))]
        if kind == 0:
            hits, peaks = noise, [list(range(len(noise)))]
        elif kind in (1, 4):
            t = _chf_track(rng, G, inv, Q, zc, 1 if rng.random() < 0.5 else -1)
            hits = t + noise[:int(rng.integers(0, 12))]
            peaks = [list(range(len(hits)))]
            if kind == 4:
                peaks.append(list(range(len(t), len(hits))) + [0, 1])
        elif kind == 2:
            t1, t2 = _chf_track(rng, G, inv, Q, zc, 1), _chf_track(rng, G, inv, Q, zc, -1)
            hits = t1 + t2 + noise[:4]
            peaks = [list(range(len(hits)))]
        else:
            t = _chf_track(rng, G, inv, Q, zc, 1 if rng.random() < 0.5 else -1)
            hits = t + noise[:6]
            peaks = [list(range(len(hits))), list(range(len(t)))]
        cls = [{"disk": disk, "time": np.float32(rng.uniform(560, 640)), "cog_x": Q[0] + rng.normal(0, 3) * k,
                "cog_y": Q[1] + rng.normal(0, 3) * k} for k in range(len(peaks))]
        evs.append((hits, peaks, cls))
    return evs


def _chf_reference(evs, G):
    import cal_helix_reference as CH
    import merge_helices_reference as MH
    geom = {**G, "zFace": list(G["zFace"]), "phiPanel": list(G["phiPanel"])}
    bad = 1 << CH.SHF_BITS["Outlier"]
    out, per_peak, merged_away = [], [], 0
    for hits, peaks, cls in evs:
        seeds = []
        for mem, c in zip(peaks, cls):
            tc = {"hits": sorted(mem), "has_calo": True, **c}
            s = CH.cal_helix_finder(hits, tc, CHF, geom)
            per_peak.append(len(s))
            seeds += [{**x, "hit_copies": x["hhits"], "has_calo": True} for x in s]
        m = MH.merge_helices(seeds, hits, MERGE, bad)
        merged_away += len(seeds) - len(m)
        out.append(m)
    return out, per_peak, merged_away


def _chf_device(evs, G, fz, precision):
    from trigger.device import calhelix
    import jax
    E, H, C = len(evs), max(len(h) for h, _, _ in evs), max(len(p) for _, p, _ in evs)
    ph = {k: np.zeros((E, H), np.float32) for k in ("x", "y", "z", "time", "edep", "uvar", "vvar", "udx", "udy")}
    ph.update(sid=np.zeros((E, H), np.int64), flag=np.zeros((E, H), np.uint32), ncombo=np.zeros((E, H), np.int64),
              valid=np.zeros((E, H), bool))
    tc = {"valid": np.zeros((E, C), bool), "members": np.zeros((E, C, H), bool)}
    cl = {"disk": np.zeros((E, C), np.int32), "time": np.zeros((E, C), np.float32),
          "cog_x": np.zeros((E, C), np.float64), "cog_y": np.zeros((E, C), np.float64)}
    for e, (hits, peaks, cls) in enumerate(evs):
        for i, h in enumerate(hits):
            for k in ("x", "y", "z", "time", "edep", "uvar", "vvar", "udx", "udy", "sid", "flag"):
                ph[k][e, i] = h[k]
            ph["ncombo"][e, i], ph["valid"][e, i] = h["nsh"], True
        for j, (mem, c) in enumerate(zip(peaks, cls)):
            tc["valid"][e, j] = True
            tc["members"][e, j, mem] = True
            for k in cl:
                cl[k][e, j] = c[k]
    ft = np.float64 if precision == "float64" else np.float32
    ph = {k: (v.astype(ft) if v.dtype == np.float32 else v) for k, v in ph.items()}
    S = calhelix.settings(CHF, MERGE)
    fn = jax.jit(lambda ph, tc, cl: calhelix.helices(ph, [tc], cl, S, G, fz, K=8, L=128)[0])
    return jax.device_get(fn(ph, tc, cl))


def _chf_compare(ref, dev, precision):
    """Seed-by-seed bit equality of the merged seeds; returns (seeds compared, first mismatches)."""
    bad, n = [], 0
    f32 = lambda v: np.asarray(v, np.float32).view(np.uint32)
    for e, m in enumerate(ref):
        slots = np.flatnonzero(dev["ok"][e])
        if len(slots) != len(m):
            bad.append((e, "count", len(m), len(slots)))
            continue
        for s, j in zip(m, slots):
            n += 1
            hits = [int(i) for i in dev["hits"][e, j] if i >= 0]
            if hits != [int(i) for i in s["hits"]]:
                bad.append((e, "hits", s["hits"], hits))
                continue
            for k in HELIX_FIELDS:
                dk = {"radius": "R", "eDepAvg": "edep"}.get(k, k)
                if f32(dev[dk][e, j]) != f32(s[k]):
                    bad.append((e, k, float(s[k]), float(dev[dk][e, j])))
            if int(dev["helicity"][e, j]) != s["helicity"]:
                bad.append((e, "helicity"))
            if precision == "float64" and np.float64(dev["t0"][e, j]) != np.float64(s["t0"]):
                bad.append((e, "t0", float(s["t0"]), float(dev["t0"][e, j])))
            k = len(hits)
            for name, dname in (("hphi", "hphi"), ("xyWeight", "xyw"), ("zphiWeight", "zw")):
                r_ = np.array([hh[name] for hh in s["hhits"]], np.float32)
                d_ = np.asarray(dev[dname][e, j][dev["hits"][e, j] >= 0][:k], np.float32)
                if not np.array_equal(r_.view(np.uint32), d_.view(np.uint32)):
                    bad.append((e, name, r_.tolist()[:6], d_.tolist()[:6]))
            fl = [int(hh["flag"]) for hh in s["hhits"]]
            if fl != [int(v) & 0xFFFFFFFF for v in dev["flag"][e, j][dev["hits"][e, j] >= 0][:k]]:
                bad.append((e, "flag"))
    return n, bad


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_cal_helix_finder_matches_the_transcriptions(precision):
    """CalHelixFinder + MergeHelices on synthetic events (no seed, one, two per peak, and duplicate peaks that
    MergeHelices reduces): the device seeds equal the literal transcriptions' bit for bit."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", precision == "float64")
    sys.path.insert(0, str(ROOT / "tests"))
    G, inv, fz = _chf_geometry()
    evs = _chf_events(np.random.default_rng(11), G, inv, 30)
    ref, per_peak, merged_away = _chf_reference(evs, G)
    assert {0, 1, 2} <= set(per_peak) and merged_away > 0, (per_peak, merged_away)
    dev = _chf_device(evs, G, fz, precision)
    n, bad = _chf_compare(ref, dev, precision)
    assert n > 10 and not bad, (n, bad[:10])
    assert not np.any(dev["err"]) and not np.any(dev["overflow"])
