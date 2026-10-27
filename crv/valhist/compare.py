"""Merge and compare stock-equivalent validation/DQM histogram files.

usage:
  python -m valhist.compare merge OUT.root 'DIR/valhist.*.root' [more globs ...] [--alias A=B ...]
  python -m valhist.compare compare REF.root CAND.root OUTDIR [--floor FLOOR_REF.root FLOOR_CAND.root] [--drop-zero]
         [--label-ref A0 --label-cand A1] [--pdf]

merge: sums every TH1/TH2 over the shard files (same booking in every file), under/overflow, ROOT statistics, titles,
class and Sumw2 (when every source has it) included, and writes one file with the directory names canonicalised (--alias renames a top-level Validation subdirectory prefix, e.g. CrvSteps_fastCrvSteps=CrvSteps_g4,
so arms whose products come from different modules line up).
compare: per 1D histogram present in both files, with both arms on the same generated events (no scaling): chi2/ndf of
the two-sample test sum (n1-n2)^2/(n1+n2) over bins with n1+n2 > 0, the binned KS distance of the normalised CDFs, the
relative difference of the totals and of the means, and max |pull|. --drop-zero leaves out the zero bin of per-event
multiplicity histograms (arms on different event sets: G4 dts vs gen stream). --floor gives a second pair (e.g. A0'
vs A0) whose statistics set the per-histogram reference: a histogram is flagged when its chi2/ndf exceeds
max(2, 3 x floor). --select (regex, default the CRV sets: the three DQM sets and Validation/Crv*) limits the
histograms. The statistics use the booked bins only; the under/overflow contents and their fraction of all entries are
reported per arm, and a histogram whose entries are all outside the booked range gets status all_flow (listed in
summary.json, not flagged). Writes summary.json, report.tsv and, with --pdf, overlays with a ratio panel.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import uproot

CRV_ONLY = r"^(crvDigiDQM|crvRecoDQM|dqmCrv)/|^Validation/Crv"
MULT = re.compile(r"(^|/)(N[A-Z]\w*|NDigis\w*|NPulse\w*|NClus|NStep|\w*PerEvt\w*|nEvents\w*)(;\d+)?$")


def _hists(f):
    return {k.rsplit(";", 1)[0]: c for k, c in f.classnames().items() if c.startswith(("TH1", "TH2", "TProfile"))}


_STATS = {1: ("fEntries", "fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2"),
          2: ("fEntries", "fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2", "fTsumwy", "fTsumwy2", "fTsumwxy")}
_DTYPE = {"C": np.int8, "S": np.int16, "I": np.int32, "F": np.float32, "D": np.float64}


def _sumw2(h, v):
    w = np.asarray(h.member("fSumw2"), dtype=float)
    return w.reshape(v.shape, order="F") if w.size == v.size else None


def _writable(c, edges, title, axes, v, stats, w2):
    """TH1x/TH2x with the summed flow-inclusive contents, ROOT statistics and (when every source had it) Sumw2."""
    from uproot.writing import identify
    ax = []
    for i, (e, (name, atitle)) in enumerate(zip(edges, axes)):
        uniform = np.allclose(e, np.linspace(e[0], e[-1], len(e)))
        ax.append(identify.to_TAxis(name, atitle, len(e) - 1, float(e[0]), float(e[-1]),
                                    np.array([], ">f8") if uniform else np.asarray(e, ">f8")))
    data = v.ravel(order="F").astype(_DTYPE.get(c[-1], np.float64))
    kw = dict(fName=None, fTitle=title, data=data, fSumw2=None if w2 is None else w2.ravel(order="F"),
              fXaxis=ax[0], **dict(zip(_STATS[len(edges)], stats)))
    if len(edges) == 2:
        return identify.to_TH2x(fYaxis=ax[1], **kw)
    return identify.to_TH1x(**kw)


def merge(out, patterns, aliases=()):
    """Sum every TH1/TH2 over the files, keeping under/overflow, ROOT statistics, titles, class and Sumw2."""
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not files:
        raise SystemExit(f"no files match {patterns}")
    acc, meta = {}, {}
    for p in files:
        with uproot.open(p) as f:
            for k, c in _hists(f).items():
                if c.startswith(("TProfile", "TH3")):
                    continue
                h = f[k]
                v = h.values(flow=True).astype(float)
                nd = len(h.axes)
                st = np.array([float(h.member(m)) for m in _STATS[nd]])
                w2 = _sumw2(h, v)
                if k in acc:
                    a = acc[k]
                    if a[0].shape != v.shape:
                        raise SystemExit(f"{k}: booking differs in {p}")
                    a[0] += v
                    a[1] += st
                    a[2] = None if (a[2] is None or w2 is None) else a[2] + w2
                else:
                    acc[k] = [v.copy(), st, w2]
                    meta[k] = (c, [a.edges() for a in h.axes], h.title,
                               [(a.member("fName"), a.member("fTitle")) for a in h.axes])
    amap = [a.split("=", 1) for a in aliases]
    nflow = 0
    with uproot.recreate(out) as f:
        for k, (v, st, w2) in acc.items():
            name = k
            for a, b in amap:
                name = re.sub(rf"(^|/){re.escape(a)}(/|$)", rf"\g<1>{b}\g<2>", name)
            c, edges, title, axes = meta[k]
            inner = v[tuple(slice(1, -1) for _ in edges)]
            nflow += bool(v.sum() - inner.sum())
            f[name] = _writable(c, edges, title, axes, v, st, w2)
    print(f"merged {len(files)} files, {len(acc)} histograms ({nflow} with under/overflow entries) -> {out}")


def _read1d(f, k):
    h = f[k]
    return h.values(flow=False).astype(float), h.axis(0).edges()


def flow(f, k):
    """Under- and overflow contents of a 1D histogram and their fraction of all its entries (flow included)."""
    v = f[k].values(flow=True).astype(float)
    tot = float(v.sum())
    return {"under": float(v[0]), "over": float(v[-1]), "frac": float((v[0] + v[-1]) / tot) if tot > 0 else 0.0}


def stats(n1, n2, edges, drop_zero=False):
    if drop_zero:
        keep = ~((edges[:-1] <= 0) & (edges[1:] > 0))
        n1, n2, edges_c = n1[keep], n2[keep], (edges[:-1][keep] + edges[1:][keep]) / 2
    else:
        edges_c = (edges[:-1] + edges[1:]) / 2
    s = n1 + n2
    m = s > 0
    t1, t2 = n1.sum(), n2.sum()
    out = {"n_ref": float(t1), "n_cand": float(t2), "bins": int(m.sum())}
    if t1 == 0 and t2 == 0:
        return {**out, "status": "both_empty"}
    if t1 == 0 or t2 == 0:
        return {**out, "status": "one_empty"}
    chi2 = float(((n1[m] - n2[m]) ** 2 / s[m]).sum())
    pull = (n1[m] - n2[m]) / np.sqrt(s[m])
    c1, c2 = np.cumsum(n1) / t1, np.cumsum(n2) / t2
    mu1, mu2 = (edges_c * n1).sum() / t1, (edges_c * n2).sum() / t2
    w = edges_c.max() - edges_c.min() or 1.0
    return {**out, "status": "ok", "chi2ndf": chi2 / max(int(m.sum()), 1), "ks": float(np.abs(c1 - c2).max()),
            "total_rel": float((t2 - t1) / t1), "mean_shift_rel_width": float((mu2 - mu1) / w),
            "max_abs_pull": float(np.abs(pull).max())}


def compare(ref, cand, outdir, floor=None, drop_zero=False, labels=("ref", "cand"), pdf=False, select=CRV_ONLY):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fr, fc = uproot.open(ref), uproot.open(cand)
    hr, hc = _hists(fr), _hists(fc)
    sel = re.compile(select) if select else None
    hr = {k: v for k, v in hr.items() if not sel or sel.search(k)}
    hc = {k: v for k, v in hc.items() if not sel or sel.search(k)}
    common = sorted(k for k in set(hr) & set(hc) if hr[k].startswith("TH1") and hc[k].startswith("TH1"))
    res = {}
    for k in common:
        (n1, e1), (n2, e2) = _read1d(fr, k), _read1d(fc, k)
        if len(e1) != len(e2) or not np.allclose(e1, e2):
            res[k] = {"status": "binning_differs"}
            continue
        r = res[k] = stats(n1, n2, e1, drop_zero and bool(MULT.search(k)))
        for side, ff in (("ref", fr), ("cand", fc)):
            w = flow(ff, k)
            r.update({f"under_{side}": w["under"], f"over_{side}": w["over"], f"flow_frac_{side}": w["frac"]})
        if r["status"] == "both_empty" and r["flow_frac_ref"] + r["flow_frac_cand"] > 0:
            r["status"] = "all_flow"
    fl = {}
    if floor:
        ff0, ff1 = uproot.open(floor[0]), uproot.open(floor[1])
        h0, h1 = _hists(ff0), _hists(ff1)
        for k in common:
            if k in h0 and k in h1:
                (a, e), (b, _) = _read1d(ff0, k), _read1d(ff1, k)
                fl[k] = stats(a, b, e, drop_zero and bool(MULT.search(k)))
    for k, r in res.items():
        if r.get("status") != "ok":
            r["flag"] = r.get("status") == "one_empty"
            continue
        f0 = fl.get(k, {}).get("chi2ndf")
        r["floor_chi2ndf"] = f0
        r["flag"] = r["chi2ndf"] > max(2.0, 3.0 * f0) if f0 is not None else r["chi2ndf"] > 2.0
    sets = {}
    for k, r in res.items():
        top = k.split("/")[0]
        s = sets.setdefault(top, {"compared": 0, "ok": 0, "flagged": 0, "both_empty": 0, "one_empty": 0, "all_flow": 0,
                                  "binning_differs": 0, "with_flow": 0})
        s["compared"] += 1
        s["with_flow"] += bool(r.get("flow_frac_ref") or r.get("flow_frac_cand"))
        st = r.get("status")
        if st == "ok":
            s["ok"] += 1
        elif st in s:
            s[st] += 1
        s["flagged"] += bool(r.get("flag"))
    summary = {"ref": str(ref), "cand": str(cand), "labels": labels, "floor": floor, "drop_zero": drop_zero,
               "only_ref": sorted(set(hr) - set(hc)), "only_cand": sorted(set(hc) - set(hr)), "sets": sets,
               "flagged": sorted(k for k, r in res.items() if r.get("flag")),
               "all_flow": sorted(k for k, r in res.items() if r.get("status") == "all_flow"),
               "flow_fractions": {k: [r["flow_frac_ref"], r["flow_frac_cand"]] for k, r in sorted(res.items())
                                  if r.get("flow_frac_ref") or r.get("flow_frac_cand")}}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    cols = ["status", "n_ref", "n_cand", "chi2ndf", "floor_chi2ndf", "ks", "total_rel", "mean_shift_rel_width",
            "max_abs_pull", "flag", "under_ref", "over_ref", "flow_frac_ref", "under_cand", "over_cand", "flow_frac_cand"]
    with open(outdir / "report.tsv", "w") as fh:
        fh.write("histogram\t" + "\t".join(cols) + "\n")
        for k, r in res.items():
            fh.write(k + "\t" + "\t".join("" if r.get(c) is None else (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]))
                                          for c in cols) + "\n")
    if pdf:
        _plots(fr, fc, res, outdir / "overlays.pdf", labels)
    print(json.dumps(sets, indent=1))
    print(f"flagged {len(summary['flagged'])}: {summary['flagged'][:20]}")
    if summary["all_flow"]:
        print(f"all entries outside the booked range {len(summary['all_flow'])}: {summary['all_flow'][:20]}")
    return summary


def _plots(fr, fc, res, path, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    keys = sorted(res, key=lambda k: (not res[k].get("flag"), k))
    with PdfPages(path) as pdf:
        for i in range(0, len(keys), 4):
            fig, axs = plt.subplots(4, 2, figsize=(11, 14), gridspec_kw={"height_ratios": [3, 1, 3, 1]})
            for j, k in enumerate(keys[i:i + 4]):
                if res[k].get("status") == "binning_differs":
                    continue
                (n1, e), (n2, _) = _read1d(fr, k), _read1d(fc, k)
                a, r = axs[2 * (j // 2), j % 2], axs[2 * (j // 2) + 1, j % 2]
                a.stairs(n1, e, label=labels[0])
                a.stairs(n2, e, label=labels[1])
                a.set_title(("[FLAG] " if res[k].get("flag") else "") + k, fontsize=7)
                a.legend(fontsize=6)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(n1 > 0, n2 / n1, np.nan)
                    err = np.where(n1 > 0, np.sqrt(n2 + n2 ** 2 / np.maximum(n1, 1)) / n1, np.nan)
                c = (e[:-1] + e[1:]) / 2
                r.errorbar(c, ratio, yerr=err, fmt=".", ms=2)
                r.axhline(1, lw=0.5)
                r.set_ylim(0.5, 1.5)
                chi = res[k].get("chi2ndf")
                r.set_xlabel(f"chi2/ndf {chi:.2f}" if chi is not None else res[k].get("status", ""), fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("out")
    m.add_argument("patterns", nargs="+")
    m.add_argument("--alias", action="append", default=[])
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("cand")
    c.add_argument("outdir")
    c.add_argument("--floor", nargs=2, default=None)
    c.add_argument("--drop-zero", action="store_true")
    c.add_argument("--label-ref", default="ref")
    c.add_argument("--label-cand", default="cand")
    c.add_argument("--pdf", action="store_true")
    c.add_argument("--select", default=CRV_ONLY, help="regex on histogram paths (default: CRV sets only; '' = all)")
    a = ap.parse_args()
    if a.cmd == "merge":
        merge(a.out, a.patterns, a.alias)
    else:
        compare(a.ref, a.cand, a.outdir, a.floor, a.drop_zero, (a.label_ref, a.label_cand), a.pdf, a.select)


if __name__ == "__main__":
    main()
