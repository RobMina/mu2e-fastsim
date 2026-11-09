"""Score DQM arms against a reference from per-shard valhist files that share the generated events.

For every common CRV TH1, summed over the shards: chi2/ndf with Poisson variance n1+n2 (the time histograms with the
clustering factor k of their collection, as the DQM comparison used for the deck) and the paired chi2/ndf with the per-shard
variance of the difference, var_b = N/(N-1) sum_s (d_sb - mean d_b)^2, on bins where a shard differs, times (N-3)/(N-1) so
that a null has mean 1. Overflow of the cluster PE and paired yields are printed. Nothing here is fitted.

    python -m valhist.arm_score --ref 'DIR/a0g/valhist.{s}.root' --arm a='DIR/a/valhist.{s}.root' --shards 22-120:2 --out J
"""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from crvfastsim.validation import compare as C
from . import steplayers as S

PE = "Validation/CrvCoincidenceClusters_CrvCoincidenceClusterFinder_noName/PE"
SHOW = {"cluster PE": PE, "dqm PEc": "dqmCrv/PEc", "digi SiPM": "crvDigiDQM/SiPM", "ADC": "crvDigiDQM/ADC",
        "peakAdc": "crvDigiDQM/h1_peakAdc", "pulse SiPM": "Validation/CrvRecoPulses_CrvRecoPulses_noName/SiPM",
        "pulse PE": "Validation/CrvRecoPulses_CrvRecoPulses_noName/PE", "PEHeight": "crvRecoDQM/PEHeight",
        "pulse chi2": "crvRecoDQM/chi2", "NDigis2": "dqmCrv/NDigis2", "NPulse2": "dqmCrv/NPulse2",
        "NDigis": "dqmCrv/NDigis", "digisPerEvt": "crvDigiDQM/h1_digisPerEvt", "NPulses": "dqmCrv/NPulses",
        "NClus": "dqmCrv/NClus", "NPc": "dqmCrv/NPc",
        "BarIdr": "crvRecoDQM/BarIdr"}
ST_T = {"Validation/CrvDigis_CrvDigi_noName/t", "Validation/CrvDigis_CrvDigi_noName/t2", "crvDigiDQM/h1_tdc",
        "crvDigiDQM/h1_tdc2", "dqmCrv/td", "dqmCrv/t2d"}
PU_T = {"LeadingTime", "LeadingTime2", "PulseTime", "PulseTime2"}
MULT = {"digis": "crvDigiDQM/h1_digisPerEvt", "pulses": "crvRecoDQM/NPulses"}


def read(path, keys=None):
    with uproot.open(path) as f:
        keys = keys or [k for k, c in C._hists(f).items() if c.startswith("TH1") and C.re.search(C.CRV_ONLY, k)]
        out = {}
        for k in keys:
            h = f[k]
            out[k] = (h.values(flow=True).astype(float), h.axis(0).edges())
    return out


def clustering_of(k):
    if k in ST_T:
        return "digis"
    if k.rsplit("/", 1)[-1] in PU_T and ("Pulses" in k or k.startswith(("crvRecoDQM/", "dqmCrv/"))):
        return "pulses"
    return None


def factor(h):
    v, e = h[0][1:-1], h[1]
    c = 0.5 * (e[:-1] + e[1:])
    m = c > 0.5
    return float((v[m] * c[m] ** 2).sum() / max((v[m] * c[m]).sum(), 1e-300))


def score(ref, cand, shards):
    keys = sorted(set(ref[0]) & set(cand[0]))
    res = {}
    n = len(shards)
    summed = lambda arm, key: (np.sum([r[key][0] for r in arm], 0), arm[0][key][1])
    kf = {c: (factor(summed(ref, MULT[c])), factor(summed(cand, MULT[c]))) for c in MULT}
    for k in keys:
        a = np.stack([r[k][0] for r in ref])
        b = np.stack([r[k][0] for r in cand])
        if a.shape != b.shape:
            continue
        s1, s2 = a[:, 1:-1].sum(0), b[:, 1:-1].sum(0)
        if s1.sum() == 0 or s2.sum() == 0:
            continue
        m = (s1 + s2) > 0
        coll = clustering_of(k)
        var = (kf[coll][0] * s1 + kf[coll][1] * s2) if coll else (s1 + s2)
        chi = float(((s1 - s2)[m] ** 2 / var[m]).sum() / m.sum())
        d = (b - a)[:, 1:-1]
        vp = n / (n - 1) * ((d - d.mean(0)) ** 2).sum(0)
        q = vp > 0
        pair = float(((s2 - s1)[q] ** 2 / vp[q]).sum() / max(q.sum(), 1)) * (n - 3) / (n - 1)
        t = (b.sum(1) - a.sum(1))
        res[k] = {"chi2": chi, "paired": pair, "bins": int(m.sum()), "n_ref": float(a.sum()), "n_cand": float(b.sum()),
                  "rel": float(t.sum() / a.sum()), "rel_se": float(np.sqrt(n / (n - 1) * ((t - t.mean()) ** 2).sum()) / a.sum())}
    p, c = np.stack([r[PE][0] for r in ref]), np.stack([r[PE][0] for r in cand])
    d = c[:, -1] - p[:, -1]
    res["_overflow"] = {"ref": float(p[:, -1].sum()), "cand": float(c[:, -1].sum()), "ratio": float(c[:, -1].sum() / p[:, -1].sum()),
                        "ratio_se": float(np.sqrt(n / (n - 1) * ((d - d.mean()) ** 2).sum()) / p[:, -1].sum())}
    return res


def detail(ref, cand, key, n):
    """Per bin of one histogram: reference count, candidate/reference and its paired error."""
    a = np.stack([r[key][0] for r in ref])[:, 1:-1]
    b = np.stack([r[key][0] for r in cand])[:, 1:-1]
    d = b - a
    se = np.sqrt(n / (n - 1) * ((d - d.mean(0)) ** 2).sum(0))
    e = ref[0][key][1]
    s1 = a.sum(0)
    return [(float(e[i]), float(e[i + 1]), float(s1[i]), float(b.sum(0)[i] / s1[i]), float(se[i] / s1[i]))
            for i in range(len(s1)) if s1[i] > 0]


def run(ref_pat, arms, shards, out, details=()):
    ref = [read(ref_pat.format(s=S.shard(k), k=k)) for k in shards]
    res = {}
    for a, pat in arms.items():
        cand = [read(pat.format(s=S.shard(k), k=k)) for k in shards]
        res[a] = score(ref, cand, shards)
        for key in details:
            res[a].setdefault("_detail", {})[key] = detail(ref, cand, key, len(shards))
            print(f"{a} {key}: " + "; ".join(f"[{lo:g},{hi:g}) {r:.4f}+-{e:.4f}" for lo, hi, _, r, e in res[a]["_detail"][key]))
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    names = list(res)
    print(f"{'histogram':>12} " + " ".join(f"{n:>17}" for n in names) + "   (Poisson or k chi2/ndf [paired])")
    for lab, k in SHOW.items():
        print(f"{lab:>12} " + " ".join((f"{res[n][k]['chi2']:7.2f} [{res[n][k]['paired']:6.2f}]  " if k in res[n] else " " * 17) for n in names))
    print(f"{'PE overflow':>12} " + " ".join(f"{res[n]['_overflow']['ratio']:7.3f} +- {res[n]['_overflow']['ratio_se']:.3f} " for n in names))
    print(f"{'flags>2':>12} " + " ".join(f"{sum(1 for k, v in res[n].items() if k[0] != '_' and v['chi2'] > 2):17d}" for n in names)
          + f"   of {sum(1 for k in res[names[0]] if k[0] != '_')}")
    print(f"{'paired>2':>12} " + " ".join(f"{sum(1 for k, v in res[n].items() if k[0] != '_' and v['paired'] > 2):17d}" for n in names))
    for lab, k in (("digis", "crvDigiDQM/BarId"), ("pulses", "crvRecoDQM/BarIdr"),
                   ("clusters", "Validation/CrvCoincidenceClusters_CrvCoincidenceClusterFinder_noName/SecType")):
        print(f"{lab:>12} " + " ".join(f"{100 * res[n][k]['rel']:+7.2f}+-{100 * res[n][k]['rel_se']:.2f} %   " for n in names))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=PATTERN")
    p.add_argument("--shards", required=True, help="first-last:step")
    p.add_argument("--out")
    p.add_argument("--detail", action="append", default=[], help="histogram key: print per-bin paired ratios")
    a = p.parse_args()
    lo, _, st = a.shards.partition(":")
    x, _, y = lo.partition("-")
    run(a.ref, dict(s.split("=", 1) for s in a.arm), list(range(int(x), int(y or x) + 1, int(st or 1))), a.out, a.detail)


if __name__ == "__main__":
    main()
