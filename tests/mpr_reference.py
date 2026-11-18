"""A literal transcription of RobustMultiHelixFinder::findHelicesInTC (Offline v13_39), hit by hit, for the device
port's tests (no calorimeter cluster, TargetCon false, ChiSquared circle fit). numpy float32 scalars round every
operation, as the stock float code does; the steps a C++ double literal promotes run in numpy float64 and are
rounded where stock stores them in a float. The time cluster's hits come in the stock TimeCluster order and each
track iteration sorts them with libstdc++'s std::sort (`std_sort`). On the stock replay's ComboHits this reproduces
the stock HelixFilter's helix parameters bit for bit."""
import math

import numpy as np

f = np.float32


def std_round(x):
    t = np.trunc(x)
    return t + np.sign(x) if abs(x - t) >= 0.5 else t


def approx_atan(z):
    return (f(0.97239411) + f(-0.19194795) * z * z) * z


def poly_atan2(y, x):
    y, x = f(y), f(x)
    pi_2 = f(math.pi / 2.)
    if x != 0:
        if abs(x) > abs(y):
            z = y / x
            if x > 0:
                return approx_atan(z)
            if y >= 0:
                return f(np.float64(approx_atan(z)) + math.pi)
            return f(np.float64(approx_atan(z)) - math.pi)
        z = x / y
        return -approx_atan(z) + pi_2 if y > 0 else -approx_atan(z) - pi_2
    if y > 0:
        return pi_2
    if y < 0:
        return -pi_2
    return f(0.)


class LS:
    def __init__(self):
        self.sn = self.sx = self.sx2 = self.sy = self.sxy = f(0)

    def add(self, x, y, w=f(1.0)):
        x, y, w = f(x), f(y), f(w)
        self.sx += x * w; self.sx2 += x * x * w; self.sy += y * w; self.sxy += x * y * w; self.sn += w

    def fa(self):
        d = self.sn * self.sx2 - self.sx * self.sx
        return (self.sn * self.sxy - self.sx * self.sy) / d if abs(d) > 1e-6 else f(0)

    def fb(self):
        d = self.sn * self.sx2 - self.sx * self.sx
        return (self.sy * self.sx2 - self.sx * self.sxy) / d if abs(d) > 1e-6 else f(0)


def std_sort(a, less):
    """libstdc++ (GCC 13) std::sort on the list a, in place: introsort (median of three, unguarded Hoare partition,
    threshold 16) then the final insertion sort. Not stable for more than 16 elements."""
    def lg(n):
        return n.bit_length() - 1

    def move_median_to_first(r, x, y, z):
        if less(a[x], a[y]):
            if less(a[y], a[z]): a[r], a[y] = a[y], a[r]
            elif less(a[x], a[z]): a[r], a[z] = a[z], a[r]
            else: a[r], a[x] = a[x], a[r]
        elif less(a[x], a[z]): a[r], a[x] = a[x], a[r]
        elif less(a[y], a[z]): a[r], a[z] = a[z], a[r]
        else: a[r], a[y] = a[y], a[r]

    def partition(first, last, pivot):
        while True:
            while less(a[first], a[pivot]): first += 1
            last -= 1
            while less(a[pivot], a[last]): last -= 1
            if not first < last:
                return first
            a[first], a[last] = a[last], a[first]
            first += 1

    def introsort(first, last, depth):
        while last - first > 16:
            if depth == 0:
                raise NotImplementedError("heap sort fallback")
            depth -= 1
            mid = first + (last - first) // 2
            move_median_to_first(first, first + 1, mid, last - 1)
            cut = partition(first + 1, last, first)
            introsort(cut, last, depth)
            last = cut

    def linear_insert(i):
        v = a[i]; j = i - 1
        while less(v, a[j]):
            a[j + 1] = a[j]; j -= 1
        a[j + 1] = v

    def insertion(first, last):
        for i in range(first + 1, last):
            if less(a[i], a[first]):
                v = a[i]; a[first + 1:i + 1] = a[first:i]; a[first] = v
            else:
                linear_insert(i)

    n = len(a)
    if n:
        introsort(0, n, lg(n) * 2)
        if n > 16:
            insertion(0, 16)
            for i in range(16, n):
                linear_insert(i)
        else:
            insertion(0, n)
    return a




def flag_compton(h, hits, s):
    phi = {}
    for i in hits:
        p = poly_atan2(h["y"][i], h["x"][i])
        phi[i] = f(np.float64(p) + 6.28318) if p < 0 else p
    nb = int(f(6.28318) / s["phibin"]) + 1
    cl = [0] * nb
    for i in hits:
        ib = max(int(phi[i] / s["phibin"]), 0)
        cl[ib] += int(h["nsh"][i])
    smax, imax = 0, 0
    for i in range(nb + 3):
        sm = cl[(i + nb - 1) % nb] + cl[i % nb] + cl[(i + 1) % nb]
        if sm > smax:
            smax, imax = sm, i
    flags = set()
    if smax < s["minbin"]:
        return flags
    pm = f((imax + 0.5) * np.float64(s["phibin"]))
    for i in hits:
        d = phi[i] - pm
        if np.float64(d) > 6.2831:
            d = f(np.float64(d) - 6.2831)
        if np.float64(d) < -6.2831:
            d = f(np.float64(d) + 6.2831)
        if np.float64(abs(d)) < 1.5 * np.float64(s["phibin"]):
            flags.add(i)
    return flags


def find_circle(h, hits, s):
    best = dict(nsh=0, x=f(0), y=f(0), r=f(0), hits=[])
    best_c = f(1e6)
    comp = flag_compton(h, hits, s)
    X, Y = h["x"], h["y"]
    n = len(hits)
    for a in range(n - 2):
        if hits[a] in comp:
            continue
        x1, y1 = X[hits[a]], Y[hits[a]]
        r1 = x1 * x1 + y1 * y1
        for b in range(a + 1, n - 1):
            if hits[b] in comp:
                continue
            x2, y2 = X[hits[b]] - x1, Y[hits[b]] - y1
            d12 = x2 * x2 + y2 * y2
            if d12 < s["mindxy2"]:
                continue
            r2 = X[hits[b]] * X[hits[b]] + Y[hits[b]] * Y[hits[b]]
            for c in range(b + 1, n):
                if hits[c] in comp:
                    continue
                x3, y3 = X[hits[c]] - x1, Y[hits[c]] - y1
                d13 = x3 * x3 + y3 * y3
                if d13 < s["mindxy2"]:
                    continue
                x4, y4 = X[hits[c]] - x2, Y[hits[c]] - y2
                if x4 * x4 + y4 * y4 < s["mindxy2"]:
                    continue
                r3 = X[hits[c]] * X[hits[c]] + Y[hits[c]] * Y[hits[c]]
                den = f(2) * (x2 * y3 - x3 * y2)
                nx, ny = d12 * y3 - d13 * y2, d13 * x2 - d12 * x3
                if abs(den) < 1e-9:
                    continue
                cx, cy = nx / den + x1, ny / den + y1
                r = np.sqrt((x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy))
                rc2 = cx * cx + cy * cy
                if r < s["minrad"] or r > s["maxrad"]:
                    continue
                if rc2 > r1 or rc2 > r2 or rc2 > r3:
                    continue
                stubby, idp, ilast, nsh, val, sc = [], int(h["upanel"][hits[0]]), None, 0, f(1e6), f(0)
                for k in range(n):
                    i = hits[k]
                    dx, dy = X[i] - cx, Y[i] - cy
                    dr = np.sqrt(dx * dx + dy * dy) - r
                    chi = dr * dr
                    if chi > s["mindr2"]:
                        continue
                    if int(h["upanel"][i]) != idp and ilast is not None:
                        stubby.append(hits[ilast]); nsh += int(h["nsh"][hits[ilast]])
                        val = chi; sc += val; idp = int(h["upanel"][i]); ilast = k
                    elif chi < val:
                        val, ilast = chi, k
                if ilast is not None:
                    stubby.append(hits[ilast]); nsh += int(h["nsh"][hits[ilast]]); sc += val
                if nsh > 0:
                    sc /= f(nsh)
                if nsh < best["nsh"] or (nsh == best["nsh"] and sc > best_c):
                    continue
                best = dict(nsh=nsh, x=cx, y=cy, r=r, hits=list(stubby))
                best_c = sc
    best["nsh"] = sum(int(h["nsh"][i]) for i in best["hits"])
    return best


def init_dzdp(c, h, hel, s):
    if not c["hits"]:
        return
    p = {i: poly_atan2(h["y"][i] - c["y"], h["x"][i] - c["x"]) for i in c["hits"]}
    nmax, smax = 0, f(1e6)
    fi = s["mindpdz"] if hel > 0 else -s["maxdpdz"]
    fe = s["maxdpdz"] if hel > 0 else -s["mindpdz"]
    ns = int(std_round(f((fe - fi) / s["dpdzstep"])))
    ch = c["hits"]
    for i in range(ns + 1):
        fa = f(fi) + f(i) * s["dpdzstep"]
        for j in range(0, len(ch), 2):
            fb = h["z"][ch[j]] - fa * p[ch[j]]
            nh, sd = 0, f(0)
            prev = int(std_round(np.float64(p[ch[0]] - (h["z"][ch[0]] - fb) / fa) / 6.283185))
            for k in ch:
                nl = int(std_round(np.float64(p[k] - (h["z"][k] - fb) / fa) / 6.283185))
                p2 = f(np.float64(p[k]) - nl * 6.283185)
                d = abs((h["z"][k] - fb) / fa - p2)
                if nl > prev + 1:
                    nh = 0
                    break
                if d > s["dphiinit"]:
                    continue
                prev = nl; nh += int(h["nsh"][k]); sd += abs(d)
            if nh < nmax or (nh == nmax and sd < smax):
                continue
            c["fa"], c["fb"], nmax, smax = fa, fb, nh, sd


def fit_dzdp(c, h, hel, s):
    ls = LS()
    with np.errstate(all="ignore"):
        for i in c["hits"]:
            ph = poly_atan2(h["y"][i] - c["y"], h["x"][i] - c["x"])
            q = (h["z"][i] - c["fb"]) / c["fa"]
            if not np.isfinite(q):
                continue
            nl = std_round(np.float64(ph - q) / 6.283185)
            pl = f(np.float64(ph) - nl * 6.283185)
            if abs(q - pl) < s["dphiinit"]:
                ls.add(pl, h["z"][i], f(h["nsh"][i]))
    fa, fb = ls.fa(), ls.fb()
    if (hel > 0 and fa < 1e-3) or (hel < 0 and fa > -1e-3):
        return
    c["fa"] = fa
    c["fb"] = f(np.float64(fb) - int(np.float64(fb / fa) / 6.283185) * 6.283185 * np.float64(fa))


def fit_dzdt(c, h):
    ls = LS()
    for i in c["hits"]:
        ls.add(h["z"][i], h["time"][i])
    c["fat"], c["fbt"] = ls.fa(), ls.fb()


def chi2_circle(h, i, cx, cy, r):
    dx, dy = h["x"][i] - cx, h["y"][i] - cy
    rr = np.sqrt(dx * dx + dy * dy)
    dr = rr - r
    rw = (h["udx"][i] * dx + h["udy"][i] * dy) / rr
    we, te = np.sqrt(h["uvar"][i]), np.sqrt(h["vvar"][i])
    rw2 = rw * rw
    return dr * dr / f(np.float64(we * we * rw2) + np.float64(te * te) * (1.0 - np.float64(rw2)))


def fit_circle(c, h):
    S = dict(w=f(0), x=f(0), x2=f(0), x3=f(0), y=f(0), y2=f(0), y3=f(0), xy=f(0), x2y=f(0), xy2=f(0))
    for i in c["hits"]:
        dx, dy = h["x"][i] - c["x"], h["y"][i] - c["y"]
        rw = h["udx"][i] * dx + h["udy"][i] * dy
        c2 = rw * rw / (dx * dx + dy * dy)
        we, te = np.sqrt(h["uvar"][i]), np.sqrt(h["vvar"][i])
        w = f(1.0 / np.float64(f(np.float64(we * we * c2) + np.float64(te * te) * (1.0 - np.float64(c2)))))
        x, y = h["x"][i], h["y"][i]
        S["x"] += x * w; S["x2"] += x * x * w; S["x3"] += x * x * x * w; S["y"] += y * w; S["y2"] += y * y * w
        S["y3"] += y * y * y * w; S["xy"] += x * y * w; S["x2y"] += x * x * y * w; S["xy2"] += x * y * y * w; S["w"] += w
    w_, x_, x2_, x3_, y_, y2_, y3_ = S["w"], S["x"], S["x2"], S["x3"], S["y"], S["y2"], S["y3"]
    xy_, x2y_, xy2_ = S["xy"], S["x2y"], S["xy2"]
    d11, d20, d30, d21 = w_ * xy_ - x_ * y_, w_ * x2_ - x_ * x_, w_ * x3_ - x2_ * x_, w_ * x2y_ - x2_ * y_
    d02, d03, d12 = w_ * y2_ - y_ * y_, w_ * y3_ - y2_ * y_, w_ * xy2_ - x_ * y2_
    den = f(2) * (d20 * d02 - d11 * d11)
    cx = ((d30 + d12) * d02 - (d03 + d21) * d11) / den
    cy = ((d03 + d21) * d20 - (d30 + d12) * d11) / den
    cc = (x2_ + y2_ - f(2) * cx * x_ - f(2) * cy * y_) / w_
    return cx, cy, np.sqrt(cc + cx * cx + cy * cy)


def find_helix(c, h, hel, s):
    init_dzdp(c, h, hel, s)
    fit_dzdp(c, h, hel, s)
    if abs(c["fa"]) < 1e-3:
        c["hits"], c["nsh"] = [], 0
        return
    keep = []
    for i in c["hits"]:
        ph = poly_atan2(h["y"][i] - c["y"], h["x"][i] - c["x"])
        q = (h["z"][i] - c["fb"]) / c["fa"]
        n = std_round(np.float64(ph - q) / 6.283185)
        if f(abs(np.float64(q - ph) + n * 6.283185)) < s["dphifit"]:
            keep.append(i)
        else:
            c["nsh"] -= int(h["nsh"][i])
    c["hits"] = keep
    fit_dzdt(c, h)
    keep = []
    for i in c["hits"]:
        if abs(c["fat"] * h["z"][i] + c["fbt"] - h["time"][i]) < s["dtfit"]:
            keep.append(i)
        else:
            c["nsh"] -= int(h["nsh"][i])
    c["hits"] = keep
    while c["nsh"] > s["minnsh"]:
        cx, cy, r = fit_circle(c, h)
        if r < s["minrad"] or r > s["maxrad"]:
            break
        c["x"], c["y"], c["r"] = cx, cy, r
        iw, cw = None, f(0)
        for k, i in enumerate(c["hits"]):
            ch = chi2_circle(h, i, cx, cy, r)
            if ch > cw:
                cw, iw = ch, k
        if cw < s["maxchi2"]:
            break
        c["nsh"] -= int(h["nsh"][c["hits"][iw]])
        del c["hits"][iw]
    fit_dzdp(c, h, hel, s)
    fit_dzdt(c, h)
    if abs(c["fa"]) < 1e-3 or abs(c["fat"]) < 1e-3 or not (np.isfinite(c["fa"]) and np.isfinite(c["fat"])) or \
            c["fa"] == 0 or c["fat"] == 0:
        c["hits"], c["nsh"] = [], 0
        return
    c["nsh"] = sum(int(h["nsh"][i]) for i in c["hits"])


def find_helices_in_tc(h, members, s):
    """members: the time cluster's hits in TimeCluster order -> list per track iteration of None or dict(hel, nsh,
    hits, x, y, r, fa, fb) of the kept helix."""
    used, out = set(), []
    for _ in range(s["niter"]):
        proc = [i for i in members if i not in used]
        if sum(int(h["nsh"][i]) for i in proc) < s["minnsh"]:
            break
        proc = std_sort(list(proc), lambda i, j: int(h["upanel"][i]) < int(h["upanel"][j]))
        circ = find_circle(h, proc, s)
        cand = {}
        for hel in (1, -1):
            c = dict(circ, hits=list(circ["hits"]), fa=f(0), fb=f(0), fat=f(0), fbt=f(0))
            if circ["nsh"] < s["minnsh"]:
                c["hits"], c["nsh"] = [], 0
            else:
                find_helix(c, h, hel, s)
            cand[hel] = c
        hel = 1 if cand[1]["nsh"] >= cand[-1]["nsh"] else -1
        b = cand[hel]
        used |= set(b["hits"])
        if b["nsh"] < s["minnsh"]:
            out.append(None)
            continue
        if h["z"][b["hits"][-1]] - h["z"][b["hits"][0]] < s["mindz"]:
            out.append(None)
            continue
        es = f(0)
        for i in b["hits"]:
            es += h["edep"][i] * f(h["nsh"][i])
        if f(np.float64(es) / (sum(int(h["nsh"][i]) for i in b["hits"]) + 1e-10)) > s["maxedep"]:
            out.append(None)
            continue
        out.append(dict(hel=hel, nsh=b["nsh"], hits=sorted(b["hits"]), x=b["x"], y=b["y"], r=b["r"], fa=b["fa"], fb=b["fb"]))
    return out
