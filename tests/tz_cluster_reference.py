"""A literal transcription of TZClusterFinder (CalPatRec, Offline v13_41_01), hit by hit, for the device port's tests
(radSelect 0, doRefine 0, timeouts off). numpy float32 scalars round every float operation as the stock float code
does (cHit and chunkInfo fields, the chunk averages, the time windows); LsqSums2 accumulates in double (Python
floats, IEEE double without fused multiply-add) and its results are rounded where stock assigns them to a float. The
per-plane hit sort and the chunk sorts are libstdc++'s std::sort (`mpr_reference.std_sort`), ties included.

Stock state that crosses events is kept: the module member `_f._chunkInfo` is reused by every seed and by the
calorimeter steps, and `clear_chunkInfo()` clears only its hit list and fitter. The calorimeter recovery pushes a
copy of it, so a recovered chunk's `nrgSelection` (whether it becomes an electron or a proton cluster) is whatever
the last assignment left: the last chunk saved by chunkHits, or the last chunk copied by an earlier calorimeter
cluster's second match, in this event or a previous one. `tz_cluster` takes and returns that value (`state["nrg"]`,
None before any assignment: stock's value is then indeterminate and the call raises if it is needed).
"""
import math

import numpy as np

from mpr_reference import std_sort

f = np.float32
NPLANES = 36
FLAG_BITS = {"EnergySelection": 1, "RadiusSelection": 2, "Background": 6, "Dead": 23, "Noisy": 24}
ENERGYSEL = 1 << FLAG_BITS["EnergySelection"]


class Indeterminate(RuntimeError):
    """The recovery read `_chunkInfo.nrgSelection` before anything in the job assigned it."""


def settings(m):
    """TZClusterFinder module table -> scalars (floats as the stock float members)."""
    if int(m.get("radSelect", 0)) != 0 or int(m.get("doRefine", 0)) != 0:
        raise ValueError("TZClusterFinder reference supports radSelect 0, doRefine 0")
    if float(m.get("timeoutMs", 0) or 0) > 0:
        raise ValueError("TZClusterFinder reference does not model timeouts")
    bkg = 0
    for b in m["hitBkgBits"]:
        bkg |= 1 << FLAG_BITS[b]
    fl = lambda k: f(m[k])
    return {"chunkSep": int(m["chunkSep"]), "chunkWindow": fl("chunkWindow"), "chunkThresh": int(m["chunkThresh"]),
            "combineWindow": fl("combineWindow"), "maxCombineSep": fl("maxCombineSep"),
            "chunkFitThresh": int(m["chunkFitThresh"]), "recoverWindow": fl("recoverWindow"),
            "clusterThresh": int(m["clusterThresh"]), "minCaloSize": int(m["minCaloSize"]),
            "minCaloEnergy": fl("minCaloEnergy"), "caloDtMax": fl("caloDtMax"), "caloTimeOffset": fl("caloTimeOffset"),
            "useCCs": int(m["useCCs"]), "recoverCCs": int(m["recoverCCs"]),
            "requireCCs": bool(m.get("requireCCs", False)),
            "protonTCs": bool(m.get("protonTCs", False)), "bkg": bkg}


def _div(a, b):
    """IEEE double division (Python raises on a zero divisor)."""
    if b != 0:
        return a / b
    if a != a or a == 0:
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


class LsqSums2:
    """Mu2eUtilities LsqSums2 with the default zero offsets, in double."""
    __slots__ = ("qn", "sw", "sx", "sy", "sx2", "sxy", "sy2")

    def __init__(self):
        self.clear()

    def clear(self):
        self.qn = self.sw = self.sx = self.sy = self.sx2 = self.sxy = self.sy2 = 0.0

    def copy(self):
        c = LsqSums2()
        for k in self.__slots__:
            setattr(c, k, getattr(self, k))
        return c

    def addPoint(self, XX, YY, W):
        X = float(XX) - 0.0
        Y = float(YY) - 0.0
        W = float(W)
        self.qn += 1
        self.sw += W
        self.sx += X * W
        self.sy += Y * W
        self.sx2 += X * X * W
        self.sxy += X * Y * W
        self.sy2 += Y * Y * W

    def addSum(self, o):
        for k in self.__slots__:
            setattr(self, k, getattr(self, k) + getattr(o, k))

    def dydx(self):
        if self.qn > 1:
            return _div(self.sw * self.sxy - self.sy * self.sx, self.sw * self.sx2 - self.sx * self.sx)
        return 0.0

    def y0(self):
        dx = 0.0 - 0.0
        return (0.0 + _div(self.sy, self.sw)) + self.dydx() * (dx - _div(self.sx, self.sw))

    def chi2Dof(self):
        if self.qn > 2:
            xm = lambda: _div(self.sx, self.sw)
            ym = lambda: _div(self.sy, self.sw)
            sxx = lambda: _div(self.sx2, self.sw) - xm() * xm()
            sxy = lambda: _div(self.sxy, self.sw) - xm() * ym()
            syy = lambda: _div(self.sy2, self.sw) - ym() * ym()
            chi2 = syy() * sxx() - sxy() * sxy()
            chi2 = _div(chi2, sxx())
            chi2 *= _div(self.sw, self.qn - 2)
            return chi2
        return 0.0


class Chunk:
    """chunkInfo."""
    __slots__ = ("hIndices", "fitter", "avgTime", "avgZpos", "nHits", "nStrawHits", "zMin", "zMax", "nrgSelection",
                 "nCombines", "caloIndex", "goodCluster")

    def __init__(self):
        self.hIndices = []
        self.fitter = LsqSums2()
        self.avgTime = self.avgZpos = self.zMin = self.zMax = f(0)
        self.nHits = self.nStrawHits = self.nCombines = 0
        self.nrgSelection = None
        self.caloIndex = -1
        self.goodCluster = True

    def copy(self):
        c = Chunk()
        for k in self.__slots__:
            setattr(c, k, getattr(self, k))
        c.hIndices = list(self.hIndices)
        c.fitter = self.fitter.copy()
        return c


def _merge(a, b):
    """The chunk merge of combineChunks / combineLines (the fitter is handled by the caller)."""
    a.hIndices.extend(b.hIndices)
    a.avgTime = f(a.avgTime * f(a.nHits))
    a.avgTime = f(a.avgTime + f(b.avgTime * f(b.nHits)))
    a.avgTime = f(a.avgTime / f(a.nHits + b.nHits))
    a.avgZpos = f(a.avgZpos * f(a.nHits))
    a.avgZpos = f(a.avgZpos + f(b.avgZpos * f(b.nHits)))
    a.avgZpos = f(a.avgZpos / f(a.nHits + b.nHits))
    a.nHits = a.nHits + b.nHits
    a.nStrawHits = a.nStrawHits + b.nStrawHits
    a.zMin = min(a.zMin, b.zMin)            # std::min(a, b): a unless b < a
    a.zMax = max(a.zMax, b.zMax)            # std::max(a, b): a unless a < b
    a.nCombines += 1


def _fmin(a, b):
    return b if b < a else a


def _fmax(a, b):
    return b if a < b else a


def tz_cluster(hits, cc, s, state=None, diag=None):
    """One event: hits {time, timevar, z (float32), nsh, plane, flag} in ComboHit collection order, calo clusters cc
    {energy, size, time} (None: no CaloCluster collection) -> (electron TCs, proton TCs, n_e, n_p, state). A TC is
    {hits (ComboHit indices, stock order), t0 (double), nsh, calo (index or -1)}. `diag` (a dict) counts what
    happened: merges, recovered hits, calo copies and recoveries (with the source of their stale nrgSelection), sort
    ties above the insertion-sort threshold, and the largest chunk counts."""
    state = dict(state or {"nrg": None})
    diag = {} if diag is None else diag
    count = lambda key, k=1: diag.__setitem__(key, diag.get(key, 0) + k)
    here = False                               # nrgSelection assigned in this event
    time = np.asarray(hits["time"], np.float32)
    tvar = np.asarray(hits["timevar"], np.float32)
    zz = np.asarray(hits["z"], np.float32)
    nsh = np.asarray(hits["nsh"]).astype(int)
    plane = np.asarray(hits["plane"]).astype(int)
    flag = np.asarray(hits["flag"]).astype(np.int64)
    nh = len(time)
    esel = [(int(x) & ENERGYSEL) != 0 for x in flag]
    bkg = [(int(x) & s["bkg"]) != 0 for x in flag]
    weight = [f(f(1) / tvar[i]) if tvar[i] > 0 else f(0) for i in range(nh)]

    # cHitsFill
    planes = [[] for _ in range(NPLANES)]
    for i in range(nh):
        if esel[i] and bkg[i]:
            continue
        planes[plane[i]].append({"i": i, "t": time[i], "w": weight[i], "z": zz[i], "nsh": int(nsh[i]), "used": 0})
    for p in planes:
        if len(p) > 16 and len(set(float(x["t"]) for x in p)) < len(p):
            count("plane_ties")
        std_sort(p, lambda a, b: a["t"] > b["t"])

    # chunkHits
    chunks = []
    nprot = nce = 0
    ci = None                                  # the reused _f._chunkInfo fields that matter: hits, fitter
    for i in range(NPLANES - 1, -1, -1):
        for j in range(len(planes[i])):
            sh = planes[i][j]
            if sh["used"] != 0:
                continue
            hold = [(i, j)]
            idx = [sh["i"]]
            fit = LsqSums2()
            fit.addPoint(sh["z"], sh["t"], sh["w"])
            seed_t = sh["t"]
            seed_nrg = 1 if esel[sh["i"]] else 0
            nhit, nstraw = 1, sh["nsh"]
            tot_t, tot_z, zmin, zmax = sh["t"], sh["z"], sh["z"], sh["z"]
            for k in range(i, -1, -1):
                if abs(k - i) > s["chunkSep"]:
                    break
                if k == i:
                    start = -1 if j == len(planes[i]) - 1 else j + 1
                else:
                    start = 0
                if start == -1:
                    continue
                for n in range(start, len(planes[k])):
                    th = planes[k][n]
                    if th["used"] != 0:
                        continue
                    if seed_nrg != (1 if esel[th["i"]] else 0):
                        continue
                    if abs(f(th["t"] - seed_t)) < s["chunkWindow"]:
                        hold.append((k, n))
                        idx.append(th["i"])
                        fit.addPoint(th["z"], th["t"], th["w"])
                        nhit += 1
                        nstraw += th["nsh"]
                        tot_t = f(tot_t + th["t"])
                        tot_z = f(tot_z + th["z"])
                        if th["z"] < zmin:
                            zmin = th["z"]
                        if th["z"] > zmax:
                            zmax = th["z"]
            if len(idx) >= s["chunkThresh"]:
                c = Chunk()
                c.hIndices, c.fitter = idx, fit
                c.avgTime = f(tot_t / f(nhit))
                c.avgZpos = f(tot_z / f(nhit))
                c.nHits, c.nStrawHits, c.zMin, c.zMax = nhit, nstraw, zmin, zmax
                c.nrgSelection, c.nCombines, c.caloIndex, c.goodCluster = seed_nrg, 0, -1, True
                chunks.append(c)
                state["nrg"] = seed_nrg
                here = True
                if seed_nrg == 0:
                    nprot += 1
                else:
                    nce += 1
                for (a, b) in hold:
                    planes[a][b]["used"] = 1
    diag["max_chunks"] = max(diag.get("max_chunks", 0), len(chunks))
    if len(chunks) > 16 and len(set(float(c.avgTime) for c in chunks)) < len(chunks):
        count("chunk_ties")
    if nprot != 0 and nce != 0:
        std_sort(chunks, lambda a, b: a.nrgSelection > b.nrgSelection)
        head = std_sort(chunks[:nce], lambda a, b: a.avgTime < b.avgTime)
        tail = std_sort(chunks[nce:], lambda a, b: a.avgTime < b.avgTime)
        chunks[:] = head + tail
    else:
        std_sort(chunks, lambda a, b: a.avgTime < b.avgTime)

    # combineChunks
    if chunks:
        more = True
        while more:
            found, mindt, one, two = 0, f(0), 0, 0
            for i in range(len(chunks) - 1):
                a, b = chunks[i], chunks[i + 1]
                if a.nrgSelection != b.nrgSelection:
                    continue
                if a.nCombines == 0 and b.nCombines == 0 and abs(f(a.avgZpos - b.avgZpos)) > s["maxCombineSep"]:
                    continue
                dt = abs(f(a.avgTime - b.avgTime))
                if dt > s["combineWindow"]:
                    if found != 0:
                        break
                    continue
                found += 1
                if found == 1 or dt < mindt:
                    mindt, one, two = dt, i, i + 1
            if found:
                a, b = chunks[one], chunks[two]
                a.fitter.addSum(b.fitter)
                _merge(a, b)
                del chunks[two]
                count("combine")
            else:
                more = False

    # recoverHits
    for i in range(NPLANES - 1, -1, -1):
        for j in range(len(planes[i])):
            th = planes[i][j]
            if th["used"] == 1:
                continue
            tnrg = 1 if esel[th["i"]] else 0
            found, mindt, best = 0, f(0), 0
            for k, c in enumerate(chunks):
                if len(c.hIndices) < s["chunkFitThresh"]:
                    continue
                if tnrg != c.nrgSelection:
                    continue
                dt = f(abs(float(th["t"]) - (c.fitter.dydx() * float(th["z"]) + c.fitter.y0())))
                if dt > s["recoverWindow"]:
                    continue
                found += 1
                if found == 1 or dt < mindt:
                    mindt, best = dt, k
            if found:
                c = chunks[best]
                c.hIndices.append(th["i"])
                c.nStrawHits += th["nsh"]
                c.zMin = _fmin(c.zMin, th["z"])
                c.zMax = _fmax(c.zMax, th["z"])
                c.nHits += 1
                c.fitter.addPoint(th["z"], th["t"], th["w"])
                count("recover_hit")

    # combineLines
    if chunks:
        more = True
        while more:
            more = False
            done = False
            for i in range(len(chunks) - 1):
                if chunks[i].nStrawHits < s["clusterThresh"]:
                    continue
                for j in range(i + 1, len(chunks)):
                    a, b = chunks[i], chunks[j]
                    if b.nStrawHits < s["clusterThresh"]:
                        continue
                    if a.nrgSelection != b.nrgSelection:
                        continue
                    if a.zMax < b.zMin or a.zMin > b.zMax:
                        seed_chi2 = f(a.fitter.chi2Dof())
                        test_chi2 = f(b.fitter.chi2Dof())
                        fit = a.fitter.copy()
                        fit.addSum(b.fitter)
                        comb = f(fit.chi2Dof())
                        if comb < test_chi2 or comb < seed_chi2:
                            a.fitter = fit
                            _merge(a, b)
                            del chunks[j]
                            count("line")
                            more = done = True
                            break
                if done:
                    break

    # countProtons
    protons = []
    for c in chunks:
        if c.nrgSelection == 1 or c.nStrawHits < s["clusterThresh"]:
            continue
        if s["protonTCs"]:
            protons.append({"hits": list(c.hIndices), "t0": c.fitter.y0(), "nsh": c.nStrawHits & 0xFFFF,
                            "calo": c.caloIndex if s["useCCs"] == 1 else -1})
    n_p = sum(1 for c in chunks if c.nrgSelection != 1 and c.nStrawHits >= s["clusterThresh"]) & 0xFFFF

    # checkCaloClusters
    if s["useCCs"] == 1 and cc is not None:
        ce, csz, ctm = np.asarray(cc["energy"], np.float32), np.asarray(cc["size"]), np.asarray(cc["time"], np.float32)
        nchunks = len(chunks)
        for i in range(len(ce)):
            if ce[i] < s["minCaloEnergy"] or int(csz[i]) < s["minCaloSize"]:
                continue
            cct = f(ctm[i] + s["caloTimeOffset"])
            added = 0
            for j in range(nchunks):
                c = chunks[j]
                if len(c.hIndices) < s["chunkFitThresh"]:
                    continue
                dT = abs(f(f(c.fitter.y0()) - cct))
                if dT < s["caloDtMax"]:
                    if c.caloIndex != -1:
                        d = c.copy()
                        d.caloIndex = i
                        chunks.append(d)
                        state["nrg"] = c.nrgSelection
                        here = True
                        count("dup")
                    else:
                        c.caloIndex = i
                    added = 1
            if s["recoverCCs"] == 1 and added == 0:
                r = Chunk()
                r.nHits, r.nStrawHits, r.caloIndex = 0, 0, i
                for k in range(nh):
                    if bkg[k] or not esel[k]:
                        continue
                    if abs(f(time[k] - cct)) < s["caloDtMax"]:
                        r.hIndices.append(k)
                        r.fitter.addPoint(zz[k], time[k], weight[k])
                        r.nHits += 1
                        r.nStrawHits += int(nsh[k])
                if r.nStrawHits >= s["clusterThresh"]:
                    if state["nrg"] is None:
                        raise Indeterminate("recovered chunk reads _chunkInfo.nrgSelection before any assignment")
                    r.nrgSelection = state["nrg"]
                    chunks.append(r)
                    for key in ("recovered", "recovered_nrg%d" % r.nrgSelection,
                                "recovered_" + ("here" if here else "previous")):
                        count(key)

    appended = len(chunks) - nchunks if s["useCCs"] == 1 and cc is not None else 0
    diag["max_appended"] = max(diag.get("max_appended", 0), appended)

    # findClusters output
    electrons = []
    for c in chunks:
        if c.nrgSelection == 0 or c.nStrawHits < s["clusterThresh"]:
            continue
        electrons.append({"hits": list(c.hIndices), "t0": c.fitter.y0(), "nsh": c.nStrawHits & 0xFFFF,
                          "calo": c.caloIndex})
    n_e = len(electrons)
    if s["requireCCs"]:
        electrons = [t for t in electrons if t["calo"] != -1]
    return electrons, protons, n_e, n_p, state
