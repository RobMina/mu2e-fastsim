"""V-R1 / V-R2 comparisons and the direct-path fidelity, plus their threshold summaries (docs/fastreco.md).

Pulses are matched per event and channel by sequence index (identical pulse finding is part of V-R2), digis by event,
channel and start TDC. Everything is array joins on int64 keys; no per-event loops.
"""
import numpy as np

V_R2 = {"finding_identical": True, "failed_flag_agree": 0.995, "pe_rel": 0.005, "time_ns": 0.1, "good_frac": 0.99}
V_R1_KS_P = 0.01


def _join(ka, kb):
    """Indices (ia, ib) of equal int64 keys present in both (keys unique within each side)."""
    sa, sb = np.argsort(ka), np.argsort(kb)
    common, ia, ib = np.intersect1d(ka[sa], kb[sb], assume_unique=False, return_indices=True)
    return sa[ia], sb[ib]


def pulse_key(evkey, chan, seq):
    return (np.asarray(evkey, np.int64) << 20) ^ (np.asarray(chan, np.int64) << 4) ^ np.asarray(seq, np.int64)


def vr2(fast, ref):
    """fast, ref: dicts with 'key' (pulse_key), 'pe', 'time', 'failed'. Returns the V-R2 summary."""
    ia, ib = _join(fast["key"], ref["key"])
    nf, nr, nm = len(fast["key"]), len(ref["key"]), len(ia)
    ff, fr = fast["failed"][ia].astype(bool), ref["failed"][ib].astype(bool)
    flag_agree = float((ff == fr).mean()) if nm else float("nan")
    good = ~ff & ~fr
    rel = np.abs(fast["pe"][ia] - ref["pe"][ib]) / np.maximum(np.abs(ref["pe"][ib]), 1e-9)
    dt = np.abs(fast["time"][ia] - ref["time"][ib])
    within = (rel <= V_R2["pe_rel"]) & (dt <= V_R2["time_ns"])
    gfrac = float(within[good].mean()) if good.any() else float("nan")
    res = {"pulses_fast": nf, "pulses_ref": nr, "matched": nm, "finding_identical": bool(nf == nr == nm),
           "failed_flag_agree": flag_agree, "good_fits_matched": int(good.sum()),
           "good_within_pe_time": gfrac,
           "pe_rel_quantiles": np.quantile(rel[good], [0.5, 0.9, 0.99]).tolist() if good.any() else None,
           "time_abs_quantiles_ns": np.quantile(dt[good], [0.5, 0.9, 0.99]).tolist() if good.any() else None,
           "failed_ref_frac": float(fr.mean()) if nm else float("nan")}
    res["pass"] = bool(res["finding_identical"] and flag_agree >= V_R2["failed_flag_agree"]
                       and gfrac >= V_R2["good_frac"])
    return res


def vr1(fast, ref):
    """Digis: fast, ref dicts with 'key' (event/chan/starttdc), 'adc' [D,12]. Noise-free: sample-identical."""
    ia, ib = _join(fast["key"], ref["key"])
    same = (fast["adc"][ia] == ref["adc"][ib]).all(1) if len(ia) else np.zeros(0, bool)
    d = (fast["adc"][ia].astype(int) - ref["adc"][ib].astype(int)) if len(ia) else np.zeros((0, 1))
    return {"digis_fast": len(fast["key"]), "digis_ref": len(ref["key"]), "matched": len(ia),
            "sample_identical_frac": float(same.mean()) if len(ia) else float("nan"),
            "max_abs_adc_diff": int(np.abs(d).max()) if d.size else 0,
            "pass_noise_free": bool(len(ia) == len(fast["key"]) == len(ref["key"]) and same.all())}


def ks_2samp(a, b):
    """Two-sample KS statistic and asymptotic p-value (no scipy dependency on the GPVM env)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return float("nan"), float("nan")
    x = np.concatenate([a, b])
    d = np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b)))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * (k * lam) ** 2)), 0, 1))
    return float(d), p


def distribution_summary(fast, ref, columns=("pe", "time", "beta")):
    """Per-column medians, relative median shift and KS for statistical (unpaired) comparisons."""
    out = {}
    for c in columns:
        a, b = np.asarray(fast[c], float), np.asarray(ref[c], float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        d, p = ks_2samp(a, b)
        out[c] = {"median_fast": float(np.median(a)) if len(a) else None,
                  "median_ref": float(np.median(b)) if len(b) else None,
                  "median_rel": float(np.median(a) / np.median(b) - 1) if len(a) and len(b) and np.median(b) else None,
                  "ks_d": d, "ks_p": p, "n_fast": int(len(a)), "n_ref": int(len(b))}
    return out
