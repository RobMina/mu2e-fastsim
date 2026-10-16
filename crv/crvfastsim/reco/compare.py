"""V-R1 / V-R2 comparisons and the direct-path fidelity, plus their threshold summaries (docs/fastreco.md).

Pulses are matched per event and channel by sequence index (identical pulse finding is part of V-R2), digis by event,
channel and start TDC. Everything is array joins on int64 keys; no per-event loops.
"""
import numpy as np

V_R2 = {"finding_identical": True, "failed_flag_agree": 0.995, "pe_rel": 0.005, "time_ns": 0.1, "good_frac": 0.99}
V_R1_KS_P = 0.01


def _as_void(k):
    k = np.ascontiguousarray(np.asarray(k, np.int64).reshape(len(k), -1))
    return k.view(np.dtype((np.void, k.dtype.itemsize * k.shape[1]))).ravel()


def _join(ka, kb):
    """Indices (ia, ib) of rows with equal keys in both; keys are int64 arrays [N] or [N, ncol], unique per side."""
    va, vb = _as_void(ka), _as_void(kb)
    _, ia, ib = np.intersect1d(va, vb, assume_unique=False, return_indices=True)
    return ia, ib


def lookup(keys, table_keys, table_vals, missing=np.nan):
    """Many-to-one lookup: table_vals of the row of table_keys equal to each row of keys (table keys unique)."""
    vk, vt = _as_void(keys), _as_void(table_keys)
    order = np.argsort(vt)
    pos = np.clip(np.searchsorted(vt[order], vk), 0, max(len(vt) - 1, 0))
    hit = (len(vt) > 0) & (vt[order][pos] == vk) if len(vt) else np.zeros(len(vk), bool)
    out = np.full(len(vk), missing, dtype=float)
    out[hit] = np.asarray(table_vals, float)[order][pos[hit]]
    return out


def pulse_key(evkey, wf_first_digi, seq):
    """(event, collection index of the waveform's first digi, sequence index): unique per pulse."""
    return np.stack([np.asarray(evkey, np.int64), np.asarray(wf_first_digi, np.int64), np.asarray(seq, np.int64)], 1)


def vr2(fast, ref, excluded_wf=None):
    """fast, ref: dicts with 'key' (pulse_key), 'pe', 'time', 'failed'. Returns the V-R2 summary.

    excluded_wf: [M,2] (event, first digi) of waveforms outside the finding-identity rule (agreed 2026-10-16): those
    with a failed fit (in either arm) or a residual within 1 ADC of minADCdifference; they are covered by the flag
    agreement instead.
    """
    ia, ib = _join(fast["key"], ref["key"])
    nf, nr, nm = len(fast["key"]), len(ref["key"]), len(ia)
    assert len(np.unique(_as_void(fast["key"]))) == nf and len(np.unique(_as_void(ref["key"]))) == nr, "keys not unique"
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
    wf_f, wf_r = fast["key"][:, :2], ref["key"][:, :2]
    excl = np.zeros((0, 2), np.int64) if excluded_wf is None else np.asarray(excluded_wf, np.int64).reshape(-1, 2)
    excl = np.concatenate([excl, wf_f[fast["failed"].astype(bool)], wf_r[ref["failed"].astype(bool)]])
    xv = np.unique(_as_void(excl))
    in_f = ~np.isin(_as_void(wf_f), xv)
    in_r = ~np.isin(_as_void(wf_r), xv)
    ja, jb = _join(fast["key"][in_f], ref["key"][in_r])
    clean_ident = bool(len(ja) == in_f.sum() == in_r.sum())
    res.update({"clean_waveform_pulses_fast": int(in_f.sum()), "clean_waveform_pulses_ref": int(in_r.sum()),
                "clean_matched": int(len(ja)), "finding_identical_clean": clean_ident,
                "excluded_waveforms": int(len(xv))})
    res["pass"] = bool(clean_ident and flag_agree >= V_R2["failed_flag_agree"] and gfrac >= V_R2["good_frac"])
    ufa = np.setdiff1d(np.arange(nf), ia)
    ufr = np.setdiff1d(np.arange(nr), ib)
    res["unmatched_fast_keys"] = fast["key"][ufa][:50].tolist()
    res["unmatched_ref_keys"] = ref["key"][ufr][:50].tolist()
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
