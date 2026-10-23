"""Effective fibre signal speed of EX and T2 from data or MC EventNtuples, without timing in the reference.

Per event the x line is fitted to the T1 and tagger first-pulse bar positions only (crvline hits, stacking x), then
the chosen EX / T2 cluster's timing position (crvcoincs.pos x, hitPosAndTimeCalculated) is compared with the line at
the cluster height. Reco builds x_timing = v_reco (t0 - t1) / 2 + centre with v_reco = fiberSignalSpeed (140 mm/ns),
so x_timing - centre = (v_reco / v_true) (x_line - centre): v_true = v_reco / slope. Robust slope (clipped LSQ on
|x_line - centre| < 0.75 half length).

    python3 fiber_speed.py NTUPLE.root [...] [--entry-stop N] [--out OUT.json]

--truth (MC diagnostic, closure test of a simulated speed): every timing-positioned cluster of the sector against its
own truth position (crvcoincsmc.pos x), no line fit and no cluster choice, so ~15x more clusters per file.
"""
import argparse
import json

import awkward as ak
import numpy as np
import uproot

from crvfastsim.objective import crvline as CL, crvresponse as R

V_RECO = 140.0


def per_file(path, entry_stop):
    t = uproot.open(path)[R.TREE]
    c = t.arrays(CL.COINC_BR, entry_stop=entry_stop)
    p = t.arrays(CL.PULSE_BR, entry_stop=entry_stop)
    o = CL.fit_arrays(c, p)
    e, s, y, x, w = o["hit_evt"], o["hit_slot"], o["hit_y"], o["hit_u"], o["hit_w"]
    xs = (s == 1) | (s == 3)
    n = o["nev"]
    a, b, _, nh, ok = CL._wls(e[xs], y[xs], x[xs], w[xs], n)
    has_t1 = np.bincount(e[xs & (s == 1)], minlength=n) > 0
    has_tag = np.bincount(e[xs & (s == 3)], minlength=n) > 0
    ok &= has_t1 & has_tag                                 # two heights 412 mm apart
    chosen, _ = CL._choose_clusters(c)
    st, idx = c["crvcoincs.sectorType"], ak.local_index(c["crvcoincs.sectorType"])
    out = {}
    for name, slot in (("EX", 0), ("T2", 2)):
        pick = idx == chosen[slot][:, None]
        poscalc = ak.to_numpy(ak.fill_none(ak.firsts(c["crvcoincs.hitPosAndTimeCalculated"][pick]), False))
        cx = ak.to_numpy(ak.fill_none(ak.firsts(c["crvcoincs.pos.fCoordinates.fX"][pick]), np.nan))
        cy = ak.to_numpy(ak.fill_none(ak.firsts(c["crvcoincs.pos.fCoordinates.fY"][pick]), np.nan))
        m = ok & poscalc & np.isfinite(cx) & (chosen[slot] >= 0)
        xl = a[m] + b[m] * (cy[m] - CL.Y_REF)
        out[name] = (xl, cx[m])
    return out


TRUTH_BR = ["crvcoincs.sectorType", "crvcoincs.hitPosAndTimeCalculated", "crvcoincs.pos.fCoordinates.fX",
            "crvcoincsmc.valid", "crvcoincsmc.pos.fCoordinates.fX"]
SECTOR_CODE = {"EX": 1, "T2": 3}


def per_file_truth(path, entry_stop):
    a = uproot.open(path)[R.TREE].arrays(TRUTH_BR, entry_stop=entry_stop)
    st = ak.to_numpy(ak.flatten(a["crvcoincs.sectorType"]))
    ok = ak.to_numpy(ak.flatten(a["crvcoincs.hitPosAndTimeCalculated"] & a["crvcoincsmc.valid"]))
    xr = ak.to_numpy(ak.flatten(a["crvcoincs.pos.fCoordinates.fX"]))
    xm = ak.to_numpy(ak.flatten(a["crvcoincsmc.pos.fCoordinates.fX"]))
    return {k: (xm[ok & (st == c)], xr[ok & (st == c)]) for k, c in SECTOR_CODE.items()}


def fit(xl, xt, half):
    keep = np.abs(xl) < 0.75 * half                         # bar centre at x = 0 for EX and T2 (detector frame)
    xl, xt = xl[keep], xt[keep]
    sl, ic = np.polyfit(xl, xt, 1)
    for _ in range(4):
        r = xt - (sl * xl + ic)
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        k = np.abs(r - np.median(r)) < 2.5 * mad
        sl, ic = np.polyfit(xl[k], xt[k], 1)
    r = xt[k] - (sl * xl[k] + ic)
    esl = np.std(r) / (np.std(xl[k]) * np.sqrt(k.sum()))   # LSQ slope error on the kept clusters
    return sl, ic, int(k.sum()), esl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--entry-stop", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--truth", action="store_true", help="MC diagnostic: reco vs truth cluster position")
    a = ap.parse_args()
    acc = {"EX": ([], []), "T2": ([], [])}
    one = per_file_truth if a.truth else per_file
    for f in a.inputs:
        for k, (xl, xt) in one(f, a.entry_stop).items():
            acc[k][0].append(xl)
            acc[k][1].append(xt)
    res = {}
    for k, (xl, xt) in acc.items():
        xl, xt = np.concatenate(xl), np.concatenate(xt)
        sl, ic, n, esl = fit(xl, xt, R.HALF[k])
        v, ev = V_RECO / sl, V_RECO * esl / sl ** 2
        res[k] = {"slope": float(sl), "slope_err": float(esl), "intercept_mm": float(ic), "speed_mm_ns": v,
                  "speed_err": ev, "n": n}
        print(f"{k}: slope {sl:.4f} +- {esl:.4f} -> effective speed {v:.1f} +- {ev:.1f} mm/ns "
              f"(offset {ic:+.0f} mm, {n} clusters)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
