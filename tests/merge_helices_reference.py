"""A literal transcription of MergeHelices::produce (TrkReco, Offline v13_41_01) for the cpr device port's tests:
the list-erase loop over the seeds of the HelixFinders in order, compareHelices (UseCalo: a seed with a calo
cluster beats one without), countHits / countOverlaps over the straw hits of the seeds' panel hits that carry none
of BadHitFlags, and findchisq, which overwrites chi2dXY / chi2dZPhi of the surviving seeds. numpy float32 scalars
round every float operation; LsqSums4 runs in double, as stock."""
import math

import numpy as np

f = np.float32


class LsqSums4:
    """Mu2eUtilities LsqSums4, every sum and result in double."""

    def __init__(self):
        self.qn = self.sw = self.sx = self.sy = self.sx2 = self.sxy = self.sy2 = 0.0
        self.sx3 = self.sx2y = self.sxy2 = self.sy3 = self.sx4 = self.sx3y = self.sx2y2 = self.sxy3 = self.sy4 = 0.0

    def add(self, X, Y, W):
        X, Y, W = float(X), float(Y), float(W)
        X2, Y2, XY = X * X, Y * Y, X * Y
        W_X, W_Y, W_X2, W_Y2 = W * X, W * Y, W * X2, W * Y2
        self.qn += 1; self.sw += W; self.sx += W_X; self.sy += W_Y; self.sx2 += W_X2; self.sxy += W * XY
        self.sy2 += W_Y2; self.sx3 += W_X2 * X; self.sx2y += W_X2 * Y; self.sxy2 += W_Y2 * X; self.sy3 += W_Y2 * Y
        self.sx4 += W_X2 * X2; self.sx3y += W_X2 * XY; self.sx2y2 += W_X2 * Y2; self.sxy3 += W_Y2 * XY
        self.sy4 += W_Y2 * Y2

    def _m(self, s):
        return s / self.sw

    def sig(self):
        m = self._m
        xm, ym, x2m, y2m = m(self.sx), m(self.sy), m(self.sx2), m(self.sy2)
        return {"XX": x2m - xm * xm, "XY": m(self.sxy) - xm * ym, "YY": y2m - ym * ym,
                "X2X": m(self.sx3) - xm * x2m, "X2Y": m(self.sx2y) - ym * x2m, "XY2": m(self.sxy2) - xm * y2m,
                "YY2": m(self.sy3) - ym * y2m, "X2X2": m(self.sx4) - x2m * x2m, "X2Y2": m(self.sx2y2) - x2m * y2m,
                "Y2Y2": m(self.sy4) - y2m * y2m, "xm": xm, "ym": ym}

    def det(self, s):
        return s["XX"] * s["YY"] - s["XY"] * s["XY"]

    def x0(self):
        s = self.sig()
        return (s["YY"] * (s["X2X"] + s["XY2"]) - s["XY"] * (s["X2Y"] + s["YY2"])) / 2 / self.det(s) + 0.0

    def y0(self):
        s = self.sig()
        return (s["XX"] * (s["X2Y"] + s["YY2"]) - s["XY"] * (s["X2X"] + s["XY2"])) / 2 / self.det(s) + 0.0

    def radius(self):
        s = self.sig()
        x_0, y_0 = self.x0() - 0.0, self.y0() - 0.0
        dx, dy = s["xm"] - x_0, s["ym"] - y_0
        return math.sqrt(s["XX"] + s["YY"] + dx * dx + dy * dy)

    def chi2_circle(self):
        s = self.sig()
        r = self.radius()
        sx2 = s["X2X"] + s["XY2"]
        sy2 = s["X2Y"] + s["YY2"]
        chi2 = s["X2X2"] + 2. * s["X2Y2"] + s["Y2Y2"] - (s["YY"] * sx2 * sx2 + s["XX"] * sy2 * sy2 - 2 * s["XY"] * sx2 * sy2) / self.det(s)
        chi2 = chi2 / (4 * r * r)
        return chi2 * (self.sw / self.qn)

    def chi2_line(self):
        s = self.sig()
        chi2 = s["YY"] * s["XX"] - s["XY"] * s["XY"]
        chi2 /= s["XX"]
        return chi2 * (self.sw / self.qn)


def _center(seed):
    """RobustHelix::centerx/centery: float _rcent * cos(float _fcent) (cos in double, as the mpr port found)."""
    rc, fc = f(seed["rcent"]), f(seed["fcent"])
    return rc * f(math.cos(float(fc))), rc * f(math.sin(float(fc)))


def findchisq(seed, hits, cfg, bad):
    """chi2dXY, chi2dZPhi of a seed over its hit copies (seed["hits"]: collection indices; seed["hit_copies"]: the
    copies' fields hphi, flag), skipping hits with a BadHitFlags bit."""
    sxy, szphi = LsqSums4(), LsqSums4()
    cx, cy = _center(seed)
    r = f(seed["radius"])
    for k, idx in enumerate(seed["hits"]):
        hc = seed["hit_copies"][k]
        if int(hc["flag"]) & bad:
            continue
        h = hits[idx]
        dx = f(h["x"]) - cx
        dy = f(h["y"]) - cy
        vx, vy = -f(h["udy"]), f(h["udx"])
        dxn = dx * vx + dy * vy
        costh2 = dxn * dxn / (dx * dx + dy * dy)
        sinth2 = f(1) - costh2
        e2xy = f(h["uvar"]) * sinth2 + f(h["vvar"]) * costh2
        wtxy = f(1. / float(e2xy))
        e2zphi = f(h["uvar"]) * costh2 + f(h["vvar"]) * sinth2
        wtzphi = r * r / e2zphi
        wtxy = wtxy * f(cfg["scaleXY"])
        wtzphi = wtzphi * f(cfg["scaleZPhi"])
        sxy.add(h["x"], h["y"], wtxy)
        szphi.add(h["z"], hc["hphi"], wtzphi)
    return f(sxy.chi2_circle()), f(szphi.chi2_line())


def _straw_hits(seed, hits, bad):
    """countHits: the straw-hit indices of the seed's panel hits without a BadHitFlags bit (a panel hit's straw hits
    are its own; two panel hits share none), as (panel-hit index, count)."""
    return [(idx, int(hits[idx]["nsh"])) for k, idx in enumerate(seed["hits"])
            if not (int(seed["hit_copies"][k]["flag"]) & bad)]


def compare(h1, h2, hits, cfg, bad):
    """compareHelices: 'unique', 'first' or 'second'."""
    s1, s2 = _straw_hits(h1, hits, bad), _straw_hits(h2, hits, bad)
    nh1, nh2 = sum(n for _, n in s1), sum(n for _, n in s2)
    common = {i for i, _ in s1} & {i for i, _ in s2}
    nover = sum(n for i, n in s1 if i in common)
    minh = min(nh1, nh2)
    c1 = findchisq(h1, hits, cfg, bad)
    c2 = findchisq(h2, hits, cfg, bad)
    if nover >= int(cfg["MinNHitOverlap"]) and f(nover / f(minh)) > f(cfg["MinHitOverlapFraction"]):
        if h1["has_calo"] and not h2["has_calo"]:
            return "first"
        if h2["has_calo"] and not h1["has_calo"]:
            return "second"
        if nh1 > nh2 and (nh1 - nh2) > int(cfg["deltanh"]):
            return "first"
        if nh2 > nh1 and (nh2 - nh1) > int(cfg["deltanh"]):
            return "second"
        return "first" if (c1[0] + c1[1]) < (c2[0] + c2[1]) else "second"
    return "unique"


def merge_helices(seeds, hits, cfg, bad):
    """MergeHelices::produce on the concatenated seeds (HelixFinders order): the surviving seeds, in order, with
    chi2xy / chi2zphi replaced by findchisq's."""
    lst = list(seeds)
    i = 0
    while i < len(lst):
        j = i + 1
        restarted = False
        while j < len(lst):
            c = compare(lst[i], lst[j], hits, cfg, bad)
            if c == "unique":
                j += 1
            elif c == "first":
                del lst[j]
            else:
                del lst[i]
                restarted = True
                break
        if not restarted:
            i += 1
    out = []
    for s in lst:
        chixy, chizphi = findchisq(s, hits, cfg, bad)
        out.append({**s, "chi2xy": chixy, "chi2zphi": chizphi})
    return out
