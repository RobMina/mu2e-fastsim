"""A literal transcription of the stock calorimeter trigger modules (Offline v13_41_01), event by event, for the device
port's tests and the stock comparison: CaloHitMakerFast, CaloClusterFast, CaloClusterCounterFilter, CaloLikelihood,
CaloNNEval + FilterEcalNNTrigger, CaloCosmicCalib and DigiFilter (calo digi count).

numpy float32 scalars round every float operation as the stock code does; C++ double steps run in float64 and are
rounded where stock stores a float. std::sort is libstdc++'s (`mpr_reference.std_sort`), including for the time
comparator of CaloClusterFast, which is not a strict weak order (1e-3 ns tolerance).
"""
import math

import numpy as np

from mpr_reference import std_sort

f = np.float32
d = np.float64


# ---------------------------------------------------------------- CaloHitMakerFast
def make_hits(sipm, t0, peakpos, waveforms, pbtime, cfg, adc2mev, caphri=(582, 609, 610, 637)):
    """One event's digis -> (calo hits, caphri hits), each a list of (crystal, nSiPM, time f32, eDep f32)."""
    delta_t, npe = f(cfg["deltaTPulses"]), f(cfg["nPEperMeV"])
    noise = f(cfg["noiseLevelMeV"])
    noise2 = f(noise * noise)
    nsigma, min_edep = f(cfg["nSigmaNoise"]), f(cfg.get("minEDep", 0.0))
    min_wfm = int(cfg.get("minWfmLength", 5))
    order = std_sort(list(range(len(sipm))),
                     lambda a, b: (t0[a] < t0[b]) if sipm[a] == sipm[b] else (sipm[a] < sipm[b]))
    pulses = {}
    pbt = d(f(pbtime))
    for i in order:
        crystal = int(sipm[i]) // 2
        if crystal >= 1348:                     # pin diodes
            continue
        wf = waveforms[i]
        if len(wf) < min_wfm:
            continue
        pk = int(peakpos[i])
        nped = 4 if pk > 3 else max(pk - 1, 1)
        base = d(0.0)
        for s in range(nped):
            base += d(wf[s])
        base /= nped
        a2m = d(adc2mev(int(sipm[i])))
        edep = (d(wf[pk]) - base) * a2m
        time = d(int(t0[i]) + pk) * 5.0 - pbt - d(f(0.0))
        if edep < d(min_edep):
            continue
        t, e = f(time), f(edep)
        lst = pulses.setdefault(crystal, [])
        for p in lst:
            if abs(f(p[1] - t)) > delta_t:
                continue
            ratio = f(f(e - p[2]) / f(e + p[2]))
            emean = f(d(f(e + p[2])) / 2.0)
            sigma = f(0.707 * math.sqrt(1.0 / d(emean) / d(npe) + d(f(f(noise2 / emean) / emean))))
            if abs(ratio) < f(nsigma * sigma):
                n = f(p[0])
                p[1] = f(f(f(n * p[1]) + t) / f(p[0] + 1))
                p[2] = f(f(f(n * p[2]) + e) / f(p[0] + 1))
                p[0] += 1
                break
        else:
            lst.append([1, t, e])
    calo, cap = [], []
    for crystal in sorted(pulses):
        for n, t, e in pulses[crystal]:
            (cap if crystal in caphri else calo).append((crystal, n, t, e))
    return calo, cap


# ---------------------------------------------------------------- CaloClusterFast
def make_clusters(hits, geo, cfg):
    """Hits (one event) -> clusters: dict(disk, time f32, energy f32, cog_x, cog_y, size, members [hit indices])."""
    emin, enoise, expand = d(cfg["EminSeed"]), d(cfg["EnoiseCut"]), d(cfg["ExpandCut"])
    dt_max, toff = d(cfg["deltaTime"]), d(cfg["timeOffset"])
    min_sipm, extend = int(cfg["minSiPMPerHit"]), bool(cfg["extendSearch"])
    if not hits:
        return []
    cry = [h[0] for h in hits]
    nsi = [h[1] for h in hits]
    tim = [h[2] for h in hits]
    ene = [h[3] for h in hits]
    sel = [i for i in range(len(hits)) if cry[i] < 1348 and d(ene[i]) > enoise and nsi[i] >= min_sipm]

    def by_time(a, b):
        dt = f(tim[a] - tim[b])
        if d(abs(dt)) < 1e-3:
            return cry[a] < cry[b]
        return dt < 0
    std_sort(sel, by_time)
    # stock marks a used hit by overwriting its index with hits.size() (the number of selected hits); a selected hit
    # whose calo-hit index equals that number is therefore taken as used from the start (kept, as stock does)
    used = len(sel)
    out = []
    s = 0
    while s != len(sel):
        if sel[s] == used or d(ene[sel[s]]) < emin:
            s += 1
            continue
        t_start = d(tim[sel[s]])
        stop = s
        while stop != len(sel) and (sel[stop] == used or d(tim[sel[stop]]) - t_start < dt_max):
            stop += 1
        start = s
        while start != 0 and (sel[start] == used or t_start - d(tim[sel[start]]) < dt_max):
            start -= 1
        start += 1
        members = [sel[s]]
        queue = [cry[sel[s]]]
        visited = np.zeros(1348, bool)
        sel[s] = used
        while queue:
            vid = queue[0]
            visited[vid] = True
            nbrs = [int(x) for x in geo["neighbors"][vid] if x >= 0]
            if extend:
                nbrs += [int(x) for x in geo["next_neighbors"][vid] if x >= 0]
            for nid in nbrs:
                if visited[nid]:
                    continue
                visited[nid] = True
                for it in range(start, stop):
                    if sel[it] == used or cry[sel[it]] != nid:
                        continue
                    if d(ene[sel[it]]) > expand:
                        queue.append(nid)
                    members.append(sel[it])
                    sel[it] = used
            queue.pop(0)
        std_sort(members, lambda a, b: ene[a] > ene[b])
        tot, x, y = d(0.0), d(0.0), d(0.0)
        for m in members:
            tot += d(ene[m])
            x += geo["local_x"][cry[m]] * d(ene[m])
            y += geo["local_y"][cry[m]] * d(ene[m])
        x /= tot
        y /= tot
        seed = members[0]
        out.append(dict(disk=int(geo["disk"][cry[seed]]), time=f(d(tim[seed]) + toff), energy=f(tot),
                        cog_x=x, cog_y=y, size=len(members), members=members))
        s += 1
    return out


# ---------------------------------------------------------------- filters
def counter_pass(clusters, cfg):
    n = 0
    for c in clusters:
        r = f(math.sqrt(c["cog_x"] ** 2 + c["cog_y"] ** 2))
        if (c["disk"] in cfg["DiskID"] and d(c["energy"]) >= d(cfg["MinClusterEnergy"])
                and d(c["energy"]) <= d(cfg["MaxClusterEnergy"]) and c["size"] >= int(cfg["MinNCel"])
                and c["size"] <= int(cfg["MaxNCel"]) and d(r) >= d(cfg["MinClusterRadius"])):
            n += 1
    return n >= int(cfg["MinNCl"])


def _cog_ff(c, hits, geo):
    """ClusterUtils::cog3Vector (Linear): Mu2e-frame energy-weighted position on the first hit's disk, in its FF."""
    first = hits[c["members"][0]][0]
    disk = geo["disk"][first]
    sw = sx = sy = sz = d(0.0)
    for m in c["members"]:
        cr = hits[m][0]
        if geo["disk"][cr] != disk:
            continue
        w = d(hits[m][3])
        p = geo["position"][cr]
        sw += w
        sx += p[0] * w
        sy += p[1] * w
        sz += p[2] * w
    o = geo["origin"][disk]
    return np.array([(sx / sw - o[0]) - 0.0, (sy / sw - o[1]) - 0.0, (sz / sw - o[2]) - geo["disp"]])


def _second_moment(c, hits, geo):
    first = hits[c["members"][0]][0]
    disk = geo["disk"][first]
    sw = sx = sy = sx2 = sy2 = d(0.0)
    for m in c["members"]:
        cr = hits[m][0]
        if geo["disk"][cr] != disk:
            continue
        w = d(hits[m][3])
        x, y = geo["position"][cr][0], geo["position"][cr][1]
        sw += w; sx += x * w; sy += y * w; sx2 += x * x * w; sy2 += y * y * w
    return (sx2 - sx * sx / sw + sy2 - sy * sy / sw) / sw


class Templates:
    """CaloLikelihood's templates: the 1D TH1F contents and the per-slice TH1D projections of the 2D TH2F."""

    def __init__(self, signal_path, bkg_path, min_rdist, rdist_step, n_cor=10):
        import uproot
        self.h1, self.corr = {}, {}
        names1 = {3: "rDist"}
        names2 = {0: "eRDist", 1: "e1RDist", 2: "e2RDist", 3: "e2e1RDist", 4: "e3RDist", 5: "e4RDist", 6: "nCrRDist"}
        for kind, path, sub, suffix in (("sig", signal_path, "cluster_electron", "2"), ("bkg", bkg_path, "cluster_all", "0")):
            with uproot.open(path) as fh:
                for disk in (0, 1):
                    for k, nm in names1.items():
                        h = fh[f"ceAna/{sub}/clDisk{disk}_{nm}{suffix}"]
                        self.h1[kind, disk, k] = (h.values(flow=True).astype(np.float64), h.axis().edges())
                    for k, nm in names2.items():
                        h = fh[f"ceAna/{sub}/clDisk{disk}_{nm}{suffix}"]
                        v = h.values(flow=True).astype(np.float64)
                        ex, ey = h.axis(0).edges(), h.axis(1).edges()
                        binsize, binlow = ex[1] - ex[0], ex[0]
                        nx = len(ex) - 1
                        for i in range(n_cor):
                            start = min_rdist + i * rdist_step
                            b0 = int((start - binlow) / binsize)
                            b1 = int(b0 + rdist_step / binsize)
                            b0 = max(b0, 0)
                            b1 = nx + 1 if (b1 < 0 or b1 > nx + 1) else b1
                            proj = np.zeros(v.shape[1])
                            for bx in range(b0, b1 + 1):
                                proj = proj + v[bx]
                            area = d(0.0)
                            for by in range(1, len(ey)):
                                area += proj[by]
                            self.corr[kind, disk, k, i] = (proj * (1 / area), ey)

    @staticmethod
    def prob(var, hist):
        vals, edges = hist
        nb = len(edges) - 1
        size = edges[1] - edges[0]
        idx = int((var - edges[0]) / size)
        if idx > nb:
            p = 1e-17
        else:
            p = vals[min(max(idx, 0), nb + 1)]
        return 1e-17 if p < 1e-10 else p

    def prob2(self, kind, disk, k, ref, var, min_x, step, n_cor=10):
        for j in range(n_cor):
            if min_x + d(j) * step <= ref < min_x + d(j + 1) * step:
                return self.prob(var, self.corr[kind, disk, k, j])
        return 1e-17


def likelihood_pass(clusters, hits, geo, tpl, cfg):
    min_e, e_step = d(cfg["MinClusterEnergy"]), d(cfg["ClusterEnergyStep"])
    min_r, r_step = d(cfg["MinClusterRadialDist"]), d(cfg["ClusterRadialDistStep"])
    cut = [d(v) for v in cfg["MinLikelihoodCut"]]
    ok = False
    for c in clusters:
        disk = c["disk"]
        if cfg.get("DropSecondDisk", False) and disk == 1:
            continue
        e = d(c["energy"])
        if e < min_e:
            continue
        cog = _cog_ff(c, hits, geo)
        r = math.sqrt(cog[0] * cog[0] + cog[1] * cog[1])
        es = [d(hits[m][3]) for m in c["members"]]
        n = len(es)
        i1 = 0; m1 = d(0.0)
        for j in range(n):
            if es[j] > m1: m1, i1 = es[j], j
        m2, i2 = d(-1.0), -1
        for j in range(n):
            if j != i1 and es[j] > m2: m2, i2 = es[j], j
        m3, i3 = d(-1.0), -1
        for j in range(n):
            if j not in (i1, i2) and es[j] > m3: m3, i3 = es[j], j
        m4 = d(-1.0)
        for j in range(n):
            if j not in (i1, i2, i3) and es[j] > m4: m4 = es[j]
        e1, e2, e3, e4, e21 = m1 / e, m2 / e, m3 / e, m4 / e, m2 / m1
        size = d(n)
        lr = lambda s, b: math.log(s / b)
        sig_r, bkg_r = tpl.prob(r, tpl.h1["sig", disk, 3]), tpl.prob(r, tpl.h1["bkg", disk, 3])
        two = lambda k, var, mn, st: lr(tpl.prob2("sig", disk, k, r, var, mn, st), tpl.prob2("bkg", disk, k, r, var, mn, st))
        l_r = lr(sig_r, bkg_r)
        l_e = two(0, r, min_e, e_step)
        l_e1, l_e2, l_e21 = two(1, e1, min_r, r_step), two(2, e2, min_r, r_step), two(3, e21, min_r, r_step)
        l_e3, l_e4, l_n = two(4, e3, min_r, r_step), two(5, e4, min_r, r_step), two(6, size, min_r, r_step)
        lh = l_r + l_e + l_e1 + l_r + l_n
        if n >= 2: lh += l_e2 + l_e21
        if n >= 3: lh += l_e3
        if n >= 4: lh += l_e4
        if lh > cut[disk]:
            ok = True
    return ok


def nn_scores(clusters, hits, geo, net, cfg):
    """CaloNNEval: one score per cluster (-1 outside the test window). net = mva-style dict (layers [src+1, dst])."""
    lo = [f(cfg["minEtoTest"]), f(cfg["minRtoTest"]), f(cfg["minTtoTest"])]
    hi = [f(cfg["maxEtoTest"]), f(cfg["maxRtoTest"]), f(cfg["maxTtoTest"])]
    out = []
    for c in clusters:
        e, r, t = f(c["energy"]), f(math.sqrt(c["cog_x"] ** 2 + c["cog_y"] ** 2)), f(c["time"])
        if e < lo[0] or e > hi[0] or r < lo[1] or r > hi[1] or t < lo[2] or t > hi[2]:
            out.append(f(-1.0))
            continue
        mem = c["members"]
        h0 = hits[mem[0]]
        e1v = d(h0[3])
        e2v = d(h0[3]) + d(hits[mem[1]][3]) if len(mem) > 1 else d(h0[3])
        nb = set(int(x) for x in geo["neighbors"][h0[0]] if x >= 0)
        nn = set(int(x) for x in geo["next_neighbors"][h0[0]] if x >= 0)
        e9 = e1v
        for m in mem:
            if hits[m][0] in nb: e9 += d(hits[m][3])
        e25 = e9
        for m in mem:
            if hits[m][0] in nn: e25 += d(hits[m][3])
        ed = d(c["energy"])
        v = [f(r), f(c["size"]), f(e1v / ed), f(e2v / ed), f(e9 / ed), f(e25 / ed), f(_second_moment(c, hits, geo))]
        out.append(mva_eval(net, v))
    return out


def expf(x):
    """expf: correctly rounded float exp, inf on overflow (as glibc)."""
    x = d(x)
    return f(math.exp(x)) if x < 709.0 else f(np.inf)


def mva_eval(net, v):
    x = [f(f(f(v[i] - net["offset"][i]) * net["scale"][i]) - f(1.0)) if net["norm"] else f(v[i]) for i in range(len(v))]
    x.append(f(1.0))
    layers = net["layers"]
    for k, w in enumerate(layers):
        y = []
        for j in range(w.shape[1]):
            s = f(0.0)
            for i in range(w.shape[0]):
                s = f(s + f(w[i, j] * x[i]))
            y.append(s)
        if k < len(layers) - 1:
            if net["act"] == "relu":
                y = [max(f(0.0), s) for s in y]
            elif net["act"] == "sigmoid":
                y = [f(1.0 / (1.0 + d(expf(-d(s))))) for s in y]
            else:
                raise NotImplementedError(net["act"])
            x = y + [f(1.0)]
        else:
            return f(1.0 / (1.0 + d(expf(-d(y[0])))))


def nn_pass(clusters, scores, cfg):
    lo = [f(cfg["minEtoTest"]), f(cfg["minRtoTest"]), f(cfg["minTtoTest"])]
    hi = [f(cfg["maxEtoTest"]), f(cfg["maxRtoTest"]), f(cfg["maxTtoTest"])]
    for c, s in zip(clusters, scores):
        e, r, t = f(c["energy"]), f(math.sqrt(c["cog_x"] ** 2 + c["cog_y"] ** 2)), f(c["time"])
        if e < lo[0] or e > hi[0] or r < lo[1] or r > hi[1] or t < lo[2] or t > hi[2]:
            continue
        if s < f(cfg["minMVAScore"]):
            continue
        return True
    return False


def cosmic_pass(clusters, hits, geo, cfg):
    mincel, outr, innr = f(cfg["MinCelEneCut"]), f(cfg["OutRadius"]), f(cfg["InnerRadius"])
    dmin, dmax, dymin = f(cfg["diagminval"]), f(cfg["diagmaxval"]), f(cfg["dyminval"])
    csize = f(geo["crystal_size"])
    ok = False
    for c in clusters:
        e = f(c["energy"])
        if not (d(e) >= d(cfg["MinEnergy"]) and d(e) <= d(cfg["MaxEnergy"]) and c["size"] >= int(cfg["MinNCrystalHits"])):
            continue
        flag, nmip = 0, 0
        eo = ei = f(0.0)
        rmax, rmin, xmin, xmax, ymin, ymax = f(-999), f(-999), f(999), f(-999), f(999), f(-999)
        for m in c["members"]:
            cr, ene = hits[m][0], hits[m][3]
            if ene > mincel:
                p = geo["ff"][cr]
                x, y = f(p[0]), f(p[1])
                r = f(math.sqrt(d(x) ** 2 + d(y) ** 2))
                if y > ymax: ymax, xmax, rmax = y, x, r
                if y < ymin: ymin, xmin, rmin = y, x, r
                if r > outr: eo = f(eo + ene)
                if r < innr: ei = f(ei + ene)
                nmip += 1
        if eo > mincel and ei > mincel and nmip > int(cfg["MinNumCelinout"]):
            flag += 1
        if nmip > int(cfg["MinNumCeldiagver"]):
            dx = f(xmax - xmin)
            dy = f(ymax - ymin)
            dx = abs(dx)
            dy = abs(dy)
            diag = f(dy / dx) if dx > 0 else f(0.0)
            if dx < csize: flag += 100
            if dmin < diag < dmax: flag += 200
            if dy > dymin and rmax > outr and rmin > outr: flag += 10
        if flag > 0:
            ok = True
    return ok


def digi_count_pass(n_digis, cfg):
    lo, hi = int(cfg["minNCaloDigi"]), int(cfg["maxNCaloDigi"])
    return n_digis >= lo and n_digis <= hi
