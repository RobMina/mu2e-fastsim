"""Compare unbiased released RPCGun histograms to the opt-in pair-source marginals."""
import argparse
import json

import numpy as np

from .target_sources import ME


def histogram_compare(stock, fast, edges, min_expected=25.):
    """Equal-normalized independent shape test, with adjacent sparse-bin pooling and flows."""
    stock, fast, edges = np.asarray(stock, float), np.asarray(fast, float), np.asarray(edges, float)
    if (stock.shape != fast.shape or stock.ndim != 1 or stock.size != edges.size + 1
            or not np.isfinite(stock).all() or not np.isfinite(fast).all()
            or np.any(stock < 0) or np.any(fast < 0) or stock.sum() <= 0 or fast.sum() <= 0):
        raise ValueError("invalid histogram counts or flow-bin shape")
    scale = stock.sum() / fast.sum()
    expected = stock + fast * scale
    cumulative = np.cumsum(expected)
    cuts = np.searchsorted(cumulative, np.arange(2 * min_expected, cumulative[-1], 2 * min_expected)) + 1
    boundaries = np.unique(np.r_[0, cuts, stock.size]).astype(int)
    s = np.add.reduceat(stock, boundaries[:-1])
    f = np.add.reduceat(fast, boundaries[:-1])
    if s.size > 1 and s[-1] + f[-1] * scale < 2 * min_expected:
        s[-2] += s[-1]
        f[-2] += f[-1]
        s, f = s[:-1], f[:-1]
    variance = s + scale ** 2 * f
    good = variance > 0
    chi = float(np.sum((s[good] - scale * f[good]) ** 2 / variance[good]))
    nd = max(int(good.sum()) - 1, 0)
    from scipy.stats import chi2
    return dict(stock_entries=int(stock.sum()), fast_entries=int(fast.sum()),
                chi2=chi, ndf=nd, pvalue=float(chi2.sf(chi, nd)) if nd else None,
                max_cdf_difference=float(np.max(np.abs(np.cumsum(stock) / stock.sum() - np.cumsum(fast) / fast.sum()))),
                flow_stock=[float(stock[0]), float(stock[-1])], flow_fast=[float(fast[0]), float(fast[-1])],
                variance_method="independent source draws, normalized shape, adjacent sparse-bin pooling; flows included")


def compare(root_path, source_path):
    import uproot
    with np.load(source_path) as source:
        p, pp = source["p"], source["positron_p"]
        ue, up = source["u"], source["positron_u"]
        ee, ep = np.hypot(p, ME), np.hypot(pp, ME)
        pair_p = p[:, None] * ue + pp[:, None] * up
        mass = np.sqrt(np.maximum(0, (ee + ep) ** 2 - np.sum(pair_p ** 2, axis=1)))
        q = dict(hmomentum=ee + ep, hElecMom=p, hPosiMom=pp, hMee=mass,
                 hMeeOverE=mass / (ee + ep), hy=(ee - ep) / np.linalg.norm(pair_p, axis=1))
        for axis, k in enumerate(("x", "y", "z")):
            q["hElecP" + k] = p * ue[:, axis]
            q["hPosiP" + k] = pp * up[:, axis]
        stop_q = dict(hTime=source["time"], hStopZ=source["z"])
    reports, stop_reports = {}, {}
    with uproot.open(root_path) as root:
        for name, values in {**q, **stop_q}.items():
            hist = root["generate/RPCGun/" + name]
            edges = hist.axis().edges()
            counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=edges.size + 1)
            result = histogram_compare(hist.values(flow=True), counts, edges)
            if name in stop_q:
                result["pvalue"] = None
                result["variance_method"] = "descriptive only: repeated fixed parent pool, not independent stop draws"
                stop_reports[name] = result
            else:
                reports[name] = result
    pvalues = sorted((r["pvalue"], k) for k, r in reports.items() if r["pvalue"] is not None)
    rejected = []
    for rank, (pvalue, name) in enumerate(pvalues):
        if pvalue > 0.05 / (len(pvalues) - rank):
            break
        rejected.append(name)
    return dict(mode="unbiased-internal-RPC-source-marginals", transport_validated=False,
                kinematics=reports, repeated_parent_stop_diagnostics=stop_reports,
                multiple_test_rule="Holm familywise alpha 0.05 over the predeclared 12 kinematic 1D marginals",
                holm_rejected=rejected, source_marginal_test_passed=not rejected,
                caveat="binned source pilot only; no joint distribution, transport, yield or pair-response certification")


def compare_dio(root_path, source_path):
    """Compare pass-all GenFilter plots with independent DIO energy/direction draws."""
    import uproot
    with np.load(source_path) as source:
        p = source["p"]
        fast = dict(energy=np.hypot(p, ME), mom=p, momT=p * np.hypot(source["u"][:, 0], source["u"][:, 1]),
                    cosTheta=source["u"][:, 2])
    with uproot.open(root_path) as root:
        stock = root["GenFilter/GenAna"].arrays(list(fast), library="np")
    edges = dict(energy=np.linspace(95., 105., 101), mom=np.linspace(94., 105., 111),
                 momT=np.linspace(0., 105., 106), cosTheta=np.linspace(-1., 1., 81))
    reports = {}
    for k, e in edges.items():
        s = np.bincount(np.searchsorted(e, stock[k], side="right"), minlength=e.size + 1)
        f = np.bincount(np.searchsorted(e, fast[k], side="right"), minlength=e.size + 1)
        reports[k] = histogram_compare(s, f, e)
    pvalues = sorted((r["pvalue"], k) for k, r in reports.items() if r["pvalue"] is not None)
    rejected = []
    for rank, (pvalue, name) in enumerate(pvalues):
        if pvalue > 0.05 / (len(pvalues) - rank):
            break
        rejected.append(name)
    return dict(mode="unbiased-DIO-source-kinematic-marginals", transport_validated=False, kinematics=reports,
                stock_electrons=int(stock["energy"].size), fast_electrons=int(fast["energy"].size),
                holm_rejected=rejected, source_marginal_test_passed=not rejected,
                multiple_test_rule="Holm familywise alpha 0.05 over four predeclared 1D kinematic marginals",
                caveat="source-only energy/direction pilot; parent position/time sampling and transport not certified")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--process", choices=("dio", "rpc-internal"), default="rpc-internal")
    a = ap.parse_args(argv)
    r = (compare_dio if a.process == "dio" else compare)(a.stock, a.source)
    with open(a.out, "w") as stream:
        json.dump(r, stream, indent=2, allow_nan=False)
    print(json.dumps(r, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
