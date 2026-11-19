"""Bin-by-bin comparison of generator output with a reference lookup table (e.g. a released one).

Per bin: arrival probability (binomial errors on both sides; the reference's generated count is taken as the
generator's for the same bin and photon number, both are the photons of the bin box inside the scintillator), mean
time delay and mean number of fiber emissions (from the uint8-encoded distributions, errors from their spread and the
arriving counts). Summary statistics only: pull mean, pull RMS, chi2/ndf and the probability-weighted ratio.
"""
import numpy as np

from . import writer as W

TABLE_KEYS = {0: "scint", 1: "ckscint", 2: "ckfiber"}


def _decode(rec):
    """(prob, delays uint8, emissions uint8) of an encoded bin record."""
    import struct
    _, p = struct.unpack_from("<If", rec, 0)
    nd = struct.unpack_from("<Q", rec, 8)[0]
    d = np.frombuffer(rec, np.uint8, nd, 16)
    ne = struct.unpack_from("<Q", rec, 16 + nd)[0]
    e = np.frombuffer(rec, np.uint8, ne, 24 + nd)
    return float(p), d, e


def _moments(w):
    w = np.asarray(w, np.float64)
    s = w.sum(axis=-1)
    i = np.arange(w.shape[-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (w * i).sum(-1) / s
        v = (w * i * i).sum(-1) / s - m * m
    return m, np.maximum(v, 0)


def compare(ref_table, simtype, out, nphot_ref, convention="v6"):
    """ref_table: crvfastsim.response.lookup.LookupTable; out: generator output dict (bins, gen, ht, he)."""
    T = ref_table.tables[TABLE_KEYS[simtype]]
    bins = np.asarray(out["bins"])
    gen = np.asarray(out["gen"], np.float64)
    ht, he = np.asarray(out["ht"]), np.asarray(out["he"])
    arr = (np.asarray(out["nall"]) if convention == "v6" else ht.sum(1)).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_new = np.where(arr > 0, (arr.astype(np.float32) / gen.astype(np.float32)).astype(np.float64), 0.0)
    p_ref = T.prob[bins].astype(np.float64)
    gen_ref = gen * float(nphot_ref) / float(out["nphot"])
    with np.errstate(invalid="ignore", divide="ignore"):
        var = p_new * (1 - p_new) / gen + p_ref * (1 - p_ref) / gen_ref
        pull = (p_new - p_ref) / np.sqrt(var)
    use = (gen > 0) & (var > 0) & ((p_new > 0) | (p_ref > 0))
    # time and emission moments from the encoded (uint8) distributions on both sides
    enc_d = np.zeros((len(bins), ht.shape[1]))
    enc_e = np.zeros((len(bins), he.shape[1]))
    for j in range(len(bins)):
        _, d, e = _decode(W.bin_record(0, int(gen[j]), {k: out[k][j] for k in W.KEYS[2:]}, convention))
        enc_d[j, :len(d)], enc_e[j, :len(e)] = d, e
    rd = np.zeros_like(enc_d)
    rd[:, :min(T.delays.shape[1], rd.shape[1])] = T.delays[bins, :rd.shape[1]]
    re_ = np.zeros_like(enc_e)
    re_[:, :min(T.emis.shape[1], re_.shape[1])] = T.emis[bins, :re_.shape[1]]
    mt_n, vt_n = _moments(enc_d)
    mt_r, vt_r = _moments(rd)
    me_n, ve_n = _moments(enc_e)
    me_r, ve_r = _moments(re_)
    arr_ref = p_ref * gen_ref
    with np.errstate(invalid="ignore", divide="ignore"):
        pt = (mt_n - mt_r) / np.sqrt(vt_n / arr + vt_r / arr_ref)
        pe = (me_n - me_r) / np.sqrt(ve_n / arr + ve_r / arr_ref)
    ut = use & (arr >= 20) & (arr_ref >= 20) & np.isfinite(pt)
    ue = use & (arr >= 20) & (arr_ref >= 20) & np.isfinite(pe) & ((ve_n > 0) | (ve_r > 0))

    def summ(x, m):
        x = x[m]
        if len(x) == 0:
            return {"n": 0}
        return {"n": int(len(x)), "mean": float(x.mean()), "mean_err": float(x.std() / np.sqrt(len(x))),
                "rms": float(x.std()), "chi2_ndf": float((x * x).mean()), "frac_gt3": float((np.abs(x) > 3).mean())}

    w = arr[use]
    ratio = float((p_new[use] * gen[use]).sum() / (p_ref[use] * gen[use]).sum()) if use.any() else float("nan")
    return {"simtype": int(simtype), "convention": convention, "bins": int(len(bins)), "nphot": int(out["nphot"]), "nphot_ref": int(nphot_ref),
            "prob_pull": summ(pull, use), "prob_ratio_weighted": ratio,
            "prob_ratio_err": float(1 / np.sqrt(max(w.sum(), 1)) * np.sqrt(2)),
            "time_mean_pull": summ(pt, ut), "time_mean_new": float(np.nanmean(mt_n[ut])) if ut.any() else None,
            "time_mean_ref": float(np.nanmean(mt_r[ut])) if ut.any() else None,
            "emis_mean_pull": summ(pe, ue), "emis_mean_new": float(np.nanmean(me_n[ue])) if ue.any() else None,
            "emis_mean_ref": float(np.nanmean(me_r[ue])) if ue.any() else None}
