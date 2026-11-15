"""RobustMultiHelixFinder and MergeHelices (TrkPatRec / TrkReco, Offline v13_39) on the device, for the mpr paths.

A port of the stock finder per time cluster, at the trigger's settings: no calorimeter cluster on the time cluster
(our inputs have none; a cluster with one would need the calo terms), TargetCon false, FitCircleStrategy ChiSquared.
The stock module computes in float, and so does this port in both precisions. The quirks of the stock code are kept:

- polyAtan2 (Mu2eUtilities), with the `+- M_PI` of its x < 0 branch added in double and rounded to float;
- the triplet distance cut on (x3 - x2, y3 - y2), x2 and y2 being already relative to the first hit;
- the per-panel stub choice of the circle scan and its chi2 sum (see `_stubs`);
- the circle kept is the last triplet with the most straw hits and the smallest chi2 sum;
- init_dzdp keeps, among the slopes with the most hits, the LAST one with the LARGEST sum of residuals;
- MergeHelices recomputes chi2dXY / chi2dZPhi (LsqSums4) with the RobustHelix centre rebuilt from rcent and the
  polyAtan2 azimuth; the helix azimuth of the hits is the ComboHit default 0, so chi2dZPhi is 0.

Hits of one panel keep their collection order (stock's std::sort by panel is not stable). Float sums run in hit order.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32
TWO_PI = np.float32(6.283185)
BIG = np.float32(3.0e38)
CLUSTER_HITS = 64                    # hit slots per time cluster for the finder (more: counted in `overflow`)


def settings(m):
    """RobustMultiHelixFinder module settings -> kernel scalars; raises for settings the port does not cover."""
    bad = []
    if m.get("TargetCon", False):
        bad.append("TargetCon true")
    if m.get("FitCircleStrategy", "ChiSquared") != "ChiSquared":
        bad.append("FitCircleStrategy " + str(m.get("FitCircleStrategy")))
    if bad:
        raise ValueError("device RobustMultiHelixFinder does not support " + ", ".join(bad))
    f = lambda k: np.float32(m[k])
    return {"minnsh": int(m["MinStrawHits"]), "niter": int(m["NMaxTrkIter"]), "mindz": f("MinDZTrk"),
            "phibin": f("ClusteringPhiBin"), "minbin": int(m["ClusteringMinBin"]), "mindxy2": f("MinDXY2Circle"),
            "minrad": f("MinRadCircle"), "maxrad": f("MaxRadCircle"),
            "mindr2": np.float32(f("MinDRCircle") * f("MinDRCircle")),
            "dpdzstep": f("DPDZStep"), "mindpdz": f("MinDPDZSlope"), "maxdpdz": f("MaxDPDZSlope"),
            "dphiinit": f("MaxDPhiHelInit"), "dphifit": f("MaxDPhiHelFit"), "dtfit": f("MaxDTHelFit"),
            "maxchi2": f("MaxChi2Hit"), "maxedep": f("maxEDepAvg"), "helicities": tuple(int(h) for h in m["Helicities"])}


def merge_settings(m):
    """MergeHelices module settings (SelectBest true, the stock default, is the only mode)."""
    if not m.get("SelectBest", True):
        raise ValueError("device MergeHelices supports SelectBest true")
    return {"minnover": int(m.get("MinNHitOverlap", 10)), "minfrac": float(m.get("MinHitOverlapFraction", 0.5)),
            "deltanh": int(m.get("deltanh", 5)), "scalexy": float(m.get("scaleXY", 1.1))}


# --- float helpers --------------------------------------------------------------------------------------------------

def _f(x):
    return jnp.asarray(x, F32)


def _round(x):
    """std::round on float: halves away from zero."""
    t = jnp.trunc(x)
    return jnp.where(jnp.abs(x - t) >= 0.5, t + jnp.sign(x), t)


_PI_HI = np.float32(math.pi)
_PI_LO = np.float32(math.pi - float(_PI_HI))
_PI_2 = np.float32(math.pi / 2.0)


def _add_pi(a, sign):
    """(float)((double)a + sign * M_PI) in float arithmetic: two-sum of a and the float pair (pi_hi, pi_lo)."""
    hi = np.float32(sign) * _PI_HI
    s = a + hi
    bp = s - a
    err = (a - (s - bp)) + (hi - bp)
    return s + (err + np.float32(sign) * _PI_LO)


def _approx_atan(z):
    return (np.float32(0.97239411) + np.float32(-0.19194795) * z * z) * z


def poly_atan2(y, x):
    """Mu2eUtilities polyAtan2 (float)."""
    y, x = _f(y), _f(x)
    ax = _approx_atan(y / jnp.where(x != 0, x, np.float32(1.0)))
    ay = _approx_atan(x / jnp.where(y != 0, y, np.float32(1.0)))
    r_x = jnp.where(x > 0, ax, jnp.where(y >= 0, _add_pi(ax, 1.0), _add_pi(ax, -1.0)))
    r_y = jnp.where(y > 0, -ay + _PI_2, -ay - _PI_2)
    r0 = jnp.where(y > 0, _PI_2, jnp.where(y < 0, -_PI_2, np.float32(0.0)))
    return jnp.where(x != 0, jnp.where(jnp.abs(x) > jnp.abs(y), r_x, r_y), r0)


def _seq_sum(v, mask, axis=-1):
    """Float sum in index order along `axis` (the stock accumulation order)."""
    v = jnp.moveaxis(jnp.where(mask, v, np.float32(0.0)), axis, 0)
    s, _ = jax.lax.scan(lambda c, a: (c + a, None), jnp.zeros(v.shape[1:], F32), v)
    return s


def _ls_fit(x, y, w, mask):
    """LSFitter: y = fa x + fb, weights w, float sums in hit order; 0 when |det| <= 1e-6."""
    sn, sx, sx2 = _seq_sum(w, mask), _seq_sum(x * w, mask), _seq_sum(x * x * w, mask)
    sy, sxy = _seq_sum(y * w, mask), _seq_sum(x * y * w, mask)
    den = sn * sx2 - sx * sx
    ok = jnp.abs(den) > np.float32(1e-6)
    d = jnp.where(ok, den, np.float32(1.0))
    return jnp.where(ok, (sn * sxy - sx * sy) / d, 0.0), jnp.where(ok, (sy * sx2 - sx * sxy) / d, 0.0)


def _first_strict_max(v, mask, start):
    """Index kept by `if (v > best) {best = v; i = k;}` from best = start; -1 if no element beats start."""
    vv = jnp.where(mask, v, -BIG)
    m = jnp.max(vv)
    hit = mask & (vv == m) & (m > start)
    return jnp.where(jnp.any(hit), jnp.argmax(hit), -1)


def _cummax(x):
    return jax.lax.cummax(x, axis=x.ndim - 1)


def _seg_scan(vals, flags, fn, reverse=False):
    """Segmented inclusive scan along the last axis: `flags` start a new segment (end one, when reverse)."""
    def comb(a, b):
        av, af = a
        bv, bf = b
        return jax.tree.map(lambda x, y: jnp.where(bf, y, x), fn(av, bv), bv), af | bf
    out, _ = jax.lax.associative_scan(comb, (vals, flags), axis=flags.ndim - 1, reverse=reverse)
    return out


# --- circle candidate -----------------------------------------------------------------------------------------------

def _compton(h, proc, s):
    phi = poly_atan2(h["y"], h["x"])
    phi = jnp.where(phi < 0, phi + np.float32(6.28318), phi)
    nb = int(np.float32(6.28318) / s["phibin"]) + 1
    ib = jnp.clip((phi / s["phibin"]).astype(jnp.int32), 0, nb - 1)
    hist = jnp.zeros(nb, jnp.int32).at[ib].add(jnp.where(proc, h["nsh"], 0))
    i = jnp.arange(nb + 3)
    tri = hist[(i + nb - 1) % nb] + hist[i % nb] + hist[(i + 1) % nb]
    imax = _first_strict_max(tri, jnp.ones(tri.shape, bool), 0)
    smax = jnp.where(imax >= 0, tri[jnp.maximum(imax, 0)], 0)
    phimax = (jnp.maximum(imax, 0).astype(F32) + np.float32(0.5)) * s["phibin"]
    dphi = phi - phimax
    dphi = jnp.where(dphi > np.float32(6.2831), dphi - np.float32(6.2831), dphi)
    dphi = jnp.where(dphi < np.float32(-6.2831), dphi + np.float32(6.2831), dphi)
    return proc & (smax >= s["minbin"]) & (jnp.abs(dphi) < np.float32(1.5) * s["phibin"])


def _stubs(h, proc, first_panel, cx, cy, r, s):
    """The circle's stub hits, their straw hits and the chi2 sum per straw hit (circles broadcast over leading axes).

    Stock walks the processed hits in order with state (panel id_prev, best hit ilast, val_prev): a passing hit
    (dR^2 <= MinDRCircle^2) of another panel than id_prev pushes ilast when ilast is set, and starts a new group;
    otherwise it replaces ilast when strictly closer. id_prev starts as the first processed hit's panel and changes
    only at a push. So the groups are runs of passing hits split where the panel changes (the second passing hit
    compares with the first processed hit's panel); each group contributes its first closest hit; the chi2 sum is the
    first hit's dR^2 of every group after the first, plus the closest dR^2 of the last group."""
    dx, dy = h["x"] - cx[..., None], h["y"] - cy[..., None]
    dr = jnp.sqrt(dx * dx + dy * dy) - r[..., None]
    chisq = dr * dr
    ok = proc & ~(chisq > s["mindr2"])
    H = ok.shape[-1]
    pos = jnp.broadcast_to(jnp.arange(H), ok.shape)
    pan = jnp.broadcast_to(h["upanel"], ok.shape)
    nth = jnp.cumsum(ok, axis=-1)
    prev = jnp.concatenate([jnp.full(ok.shape[:-1] + (1,), -1), _cummax(jnp.where(ok, pos, -1))[..., :-1]],
                           axis=-1)
    prev_pan = jnp.where(nth == 2, first_panel, jnp.take_along_axis(pan, jnp.maximum(prev, 0), axis=-1))
    start = ok & ((nth == 1) | (pan != prev_pan))
    gid = jnp.cumsum(start, axis=-1)
    val = jnp.where(ok, chisq, BIG)
    # running (closest dR^2, its first position) within each group, strict improvement
    run = _seg_scan((val, jnp.where(ok, pos, H)), start,
                    lambda a, b: jax.tree.map(lambda x, y: jnp.where(b[0] < a[0], y, x), a, b))
    # the group's final value, propagated back over the group
    nxt_start = jnp.concatenate([start[..., 1:], jnp.ones(ok.shape[:-1] + (1,), bool)], axis=-1)
    fin = _seg_scan(run, nxt_start, lambda a, b: a, reverse=True)
    stub = ok & (fin[1] == pos)
    nsh = jnp.sum(jnp.where(stub, h["nsh"], 0), axis=-1).astype(jnp.int32)
    later_first = start & (gid >= 2)
    s1 = _seq_sum(chisq, later_first)
    ng = gid[..., -1]
    last_min = jnp.min(jnp.where(ok & (gid == ng[..., None]), chisq, BIG), axis=-1)
    tot = jnp.where(ng > 0, s1 + last_min, np.float32(0.0))
    return stub, nsh, jnp.where(nsh > 0, tot / jnp.maximum(nsh, 1).astype(F32), tot)


def _circle(h, proc, s):
    """findCircleCandidate: (cx, cy, r, stub mask) of the best circle through processed, non-Compton hit triplets."""
    H = proc.shape[0]
    seed = proc & ~_compton(h, proc, s)
    first_panel = h["upanel"][jnp.argmax(proc)]
    x, y = h["x"], h["y"]
    r2 = x * x + y * y
    jj, kk = jnp.meshgrid(jnp.arange(H), jnp.arange(H), indexing="ij")
    flat = jj * H + kk

    def per_i(best, i):
        bn, bc = best[0], best[1]
        x1, y1 = x[i], y[i]
        x2, y2 = (x - x1)[:, None], (y - y1)[:, None]                  # second hit j
        x3, y3 = (x - x1)[None, :], (y - y1)[None, :]                  # third hit k
        d12, d13 = x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
        x4, y4 = x[None, :] - x2, y[None, :] - y2
        d4 = x4 * x4 + y4 * y4
        den = np.float32(2) * (x2 * y3 - x3 * y2)
        dd = jnp.where(jnp.abs(den) < np.float32(1e-9), np.float32(1.0), den)
        cx = (d12 * y3 - d13 * y2) / dd + x1
        cy = (d13 * x2 - d12 * x3) / dd + y1
        r = jnp.sqrt((x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy))
        rc2 = cx * cx + cy * cy
        okt = seed[i] & seed[:, None] & seed[None, :] & (jj > i) & (kk > jj)
        okt &= ~(d12 < s["mindxy2"]) & ~(d13 < s["mindxy2"]) & ~(d4 < s["mindxy2"]) & ~(jnp.abs(den) < np.float32(1e-9))
        okt &= ~((r < s["minrad"]) | (r > s["maxrad"]))
        okt &= ~((rc2 > r2[i]) | (rc2 > r2[:, None]) | (rc2 > r2[None, :]))
        stub, nsh, tot = _stubs(h, proc, first_panel, cx, cy, r, s)
        # the stock loop replaces the best when nsh > best or (nsh == best and chi2 sum <= best): the result is the
        # last triplet (in loop order) with the most straw hits and, among them, the smallest sum
        n_i = jnp.max(jnp.where(okt, nsh, -1)).astype(jnp.int32)
        c_i = jnp.min(jnp.where(okt & (nsh == n_i), tot, BIG))
        last = jnp.max(jnp.where(okt & (nsh == n_i) & (tot == c_i), flat, -1))
        take = (last >= 0) & ((n_i > bn) | ((n_i == bn) & ~(c_i > bc)))
        j_, k_ = jnp.maximum(last, 0) // H, jnp.maximum(last, 0) % H
        new = (n_i, c_i, cx[j_, k_], cy[j_, k_], r[j_, k_], stub[j_, k_])
        return jax.tree.map(lambda a, b: jnp.where(take, a, b), new, best), None

    init = (jnp.int32(0), np.float32(1e6), np.float32(0), np.float32(0), np.float32(0), jnp.zeros(H, bool))
    (_, _, cx, cy, r, stub), _ = jax.lax.scan(per_i, init, jnp.arange(H))
    return cx, cy, r, stub


# --- helix candidate ------------------------------------------------------------------------------------------------

def _init_dzdp(h, cx, cy, mem, hel, s):
    """init_dzdp: (fa, fb, found) over the slope grid and every other chit as the intercept anchor."""
    H = mem.shape[0]
    p = poly_atan2(h["y"] - cy, h["x"] - cx)
    fa_init = s["mindpdz"] if hel > 0 else -s["maxdpdz"]
    fa_end = s["maxdpdz"] if hel > 0 else -s["mindpdz"]
    nsteps = int(np.round((fa_end - fa_init) / s["dpdzstep"]))
    fas = jnp.asarray(np.float32(fa_init) + np.arange(nsteps + 1).astype(np.float32) * s["dpdzstep"])
    jsel = mem & ((jnp.cumsum(mem) - 1) % 2 == 0)
    i0 = jnp.argmax(mem)
    pos = jnp.arange(H)

    def per_fa(fa):
        fb = h["z"] - fa * p                                           # candidate intercept per anchor j
        q = (h["z"][None, :] - fb[:, None]) / fa                       # [j, hit]
        nl = _round((p[None, :] - q) / TWO_PI)
        delta = jnp.abs(q - (p[None, :] - nl * TWO_PI))
        acc = mem[None, :] & ~(delta > s["dphiinit"])
        last = jax.lax.cummax(jnp.where(acc, pos[None, :], -1), axis=1)
        prev = jnp.concatenate([jnp.full((H, 1), -1), last[:, :-1]], axis=1)
        prevnl = jnp.where(prev >= 0, jnp.take_along_axis(nl, jnp.maximum(prev, 0), axis=1), nl[:, i0][:, None])
        broken = jnp.any(mem[None, :] & (nl > prevnl + 1), axis=1)
        nh = jnp.where(broken, 0, jnp.sum(jnp.where(acc, h["nsh"][None, :], 0), axis=1))
        return fb, nh, _seq_sum(delta, acc)

    fbs, nhs, sds = jax.vmap(per_fa)(fas)                              # [n_fa, j]
    ok = jnp.broadcast_to(jsel[None, :], nhs.shape) & (nhs > 0)
    n_m = jnp.max(jnp.where(ok, nhs, 0))
    s_m = jnp.max(jnp.where(ok & (nhs == n_m), sds, -BIG))
    idx = jnp.max(jnp.where(ok & (nhs == n_m) & (sds == s_m), jnp.arange(nhs.size).reshape(nhs.shape), -1))
    found = idx >= 0
    ia, ij = jnp.maximum(idx, 0) // H, jnp.maximum(idx, 0) % H
    return jnp.where(found, fas[ia], np.float32(0)), jnp.where(found, fbs[ia, ij], np.float32(0)), found


def _fit_dzdp(h, cx, cy, fa0, fb0, mem, hel, s):
    """fit_dzdp from a valid slope fa0 != 0: (fa, fb), unchanged when the fitted slope has the wrong sign."""
    phic = poly_atan2(h["y"] - cy, h["x"] - cx)
    q = (h["z"] - fb0) / fa0
    nl = _round((phic - q) / TWO_PI)
    ploop = phic - nl * TWO_PI
    use = mem & (jnp.abs(q - ploop) < s["dphiinit"])
    fa, fb = _ls_fit(ploop, h["z"], h["nsh"].astype(F32), use)
    keep = (fa < np.float32(1e-3)) if hel > 0 else (fa > np.float32(-1e-3))
    fs = jnp.where(keep, np.float32(1.0), fa)
    return jnp.where(keep, fa0, fa), jnp.where(keep, fb0, fb - jnp.trunc(fb / fs / TWO_PI) * TWO_PI * fs)


def _chi2_circle(h, cx, cy, r):
    dx, dy = h["x"] - cx, h["y"] - cy
    rr = jnp.sqrt(dx * dx + dy * dy)
    rw = (h["udx"] * dx + h["udy"] * dy) / rr
    werr, terr = jnp.sqrt(h["uvar"]), jnp.sqrt(h["vvar"])
    res2 = werr * werr * (rw * rw) + terr * terr * (np.float32(1.0) - rw * rw)
    dr = rr - r
    return dr * dr / res2


def _fit_circle_chi2(h, cx0, cy0, mem):
    dx, dy = h["x"] - cx0, h["y"] - cy0
    rw = h["udx"] * dx + h["udy"] * dy
    c2 = rw * rw / (dx * dx + dy * dy)
    werr, terr = jnp.sqrt(h["uvar"]), jnp.sqrt(h["vvar"])
    w = np.float32(1.0) / (werr * werr * c2 + terr * terr * (np.float32(1.0) - c2))
    x, y = h["x"], h["y"]
    S = lambda v: _seq_sum(v, mem)
    x_, x2_, x3_, y_, y2_, y3_ = S(x * w), S(x * x * w), S(x * x * x * w), S(y * w), S(y * y * w), S(y * y * y * w)
    xy_, x2y_, xy2_, w_ = S(x * y * w), S(x * x * y * w), S(x * y * y * w), S(w)
    d11, d20, d30, d21 = w_ * xy_ - x_ * y_, w_ * x2_ - x_ * x_, w_ * x3_ - x2_ * x_, w_ * x2y_ - x2_ * y_
    d02, d03, d12 = w_ * y2_ - y_ * y_, w_ * y3_ - y2_ * y_, w_ * xy2_ - x_ * y2_
    den = np.float32(2) * (d20 * d02 - d11 * d11)
    cx = ((d30 + d12) * d02 - (d03 + d21) * d11) / den
    cy = ((d03 + d21) * d20 - (d30 + d12) * d11) / den
    c = (x2_ + y2_ - np.float32(2) * cx * x_ - np.float32(2) * cy * y_) / w_
    return cx, cy, jnp.sqrt(c + cx * cx + cy * cy)


def _isnormal(x):
    return jnp.isfinite(x) & (jnp.abs(x) >= np.float32(np.finfo(np.float32).tiny))


def _helix(h, cx, cy, r, chits, hel, s):
    """findHelixCandidate for one helicity, from the circle and its stub hits."""
    nsh_c = jnp.sum(jnp.where(chits, h["nsh"], 0))
    fa, fb, found = _init_dzdp(h, cx, cy, chits, hel, s)
    fa, fb = _fit_dzdp(h, cx, cy, jnp.where(found, fa, np.float32(1)), fb, chits, hel, s)
    alive = found & ~(jnp.abs(fa) < np.float32(1e-3))                  # no slope: stock divides by 0, no hit, cleared
    fs = jnp.where(alive, fa, np.float32(1))
    phi = poly_atan2(h["y"] - cy, h["x"] - cx)
    q = (h["z"] - fb) / fs
    m = chits & (jnp.abs(q - phi + _round((phi - q) / TWO_PI) * TWO_PI) < s["dphifit"])     # filterZPhi
    fat, fbt = _ls_fit(h["z"], h["time"], jnp.ones_like(h["z"]), m)                         # fit_dzdt
    m = m & (jnp.abs(fat * h["z"] + fbt - h["time"]) < s["dtfit"])                           # filterZT
    nsh = nsh_c - jnp.sum(jnp.where(chits & ~m, h["nsh"], 0))
    H = m.shape[0]

    def cond(st):
        return st[5] & (st[4] > s["minnsh"])

    def body(st):
        cx_, cy_, r_, m_, nsh_, _ = st
        nx, ny, nr = _fit_circle_chi2(h, cx_, cy_, m_)
        inr = ~((nr < s["minrad"]) | (nr > s["maxrad"]))
        cx2, cy2, r2 = jnp.where(inr, nx, cx_), jnp.where(inr, ny, cy_), jnp.where(inr, nr, r_)
        chi2 = _chi2_circle(h, cx2, cy2, r2)
        iw = _first_strict_max(chi2, m_, np.float32(0))
        worst = jnp.where(iw >= 0, chi2[jnp.maximum(iw, 0)], np.float32(0))
        go = inr & ~(worst < s["maxchi2"])
        rm = go & (jnp.arange(H) == iw)
        return cx2, cy2, r2, m_ & ~rm, nsh_ - jnp.sum(jnp.where(rm, h["nsh"], 0)), go

    cx, cy, r, m, nsh, _ = jax.lax.while_loop(cond, body, (cx, cy, r, m, nsh, alive))
    fa, fb = _fit_dzdp(h, cx, cy, fs, fb, m, hel, s)
    fat, fbt = _ls_fit(h["z"], h["time"], jnp.ones_like(h["z"]), m)
    ok = alive & ~(jnp.abs(fa) < np.float32(1e-3)) & ~(jnp.abs(fat) < np.float32(1e-3)) & _isnormal(fa) & _isnormal(fat)
    m = m & ok
    return {"nsh": jnp.sum(jnp.where(m, h["nsh"], 0)), "mask": m, "cx": cx, "cy": cy, "r": r, "fa": fa, "fb": fb}


# --- one time cluster -------------------------------------------------------------------------------------------------

def _cluster(h, members, s):
    """findHelicesInTC on one time cluster (hits already in panel order): per track iteration, the kept helix (or
    none), as arrays [niter] (ok, helicity, nsh, cx, cy, r, fa, fb) and hit masks [niter, H]."""
    H = members.shape[0]
    outs = []
    used = jnp.zeros(H, bool)
    stopped = jnp.bool_(False)
    for _ in range(s["niter"]):
        proc = members & ~used
        togo = jnp.sum(jnp.where(proc, h["nsh"], 0))
        stopped = stopped | (togo < s["minnsh"])
        cx, cy, r, stub = _circle(h, proc, s)
        ncirc = jnp.sum(jnp.where(stub, h["nsh"], 0))
        has = ~stopped & (ncirc >= s["minnsh"])
        hx = {hel: _helix(h, cx, cy, r, stub, hel, s) for hel in s["helicities"]}
        pos, neg = hx[1], hx[-1]
        pos_n = jnp.where(has, pos["nsh"], 0)
        neg_n = jnp.where(has, neg["nsh"], 0)
        use_pos = pos_n >= neg_n
        best = jax.tree.map(lambda a, b: jnp.where(use_pos, a, b), pos, neg)
        bm = best["mask"] & has
        used = used | bm
        bn = jnp.where(use_pos, pos_n, neg_n)
        z = h["z"]
        zfirst = z[jnp.argmax(bm)]
        zlast = z[H - 1 - jnp.argmax(bm[::-1])]
        edep = jnp.sum(jnp.where(bm, h["edep"] * h["nsh"].astype(F32), 0.0)) / (jnp.sum(jnp.where(bm, h["nsh"], 0)) +
                                                                               np.float32(1e-10))
        ok = ~stopped & (bn >= s["minnsh"]) & ~((zlast - zfirst) < s["mindz"]) & ~(edep > s["maxedep"])
        outs.append({"ok": ok, "hel": jnp.where(use_pos, 1, -1), "nsh": bn, "mask": bm & ok, "cx": best["cx"],
                     "cy": best["cy"], "r": best["r"], "fa": best["fa"], "fb": best["fb"]})
    return jax.tree.map(lambda *a: jnp.stack(a), *outs)


def _dedupe(ok, hel, mask, which):
    """filterDuplicateHelices on the helices of one helicity, in collection order (cluster, iteration): a helix whose
    hits are a subset of a later one's, or a superset, loses the smaller (the first on equal size)."""
    n = ok.shape[0]
    alive = ok & (hel == which)
    size = jnp.sum(mask, axis=1)
    inter = jnp.sum(mask[:, None, :] & mask[None, :, :], axis=2)

    def body(i, al):
        def inner(j, al):
            si = jnp.where(al[i], size[i], 0)
            sj = jnp.where(al[j], size[j], 0)
            res = jnp.where(al[i] & al[j], inter[i, j], 0)
            dup = (j > i) & ((res == si) | (res == sj))
            clear_j = dup & (si > sj)
            clear_i = dup & ~(si > sj)
            return al.at[j].set(al[j] & ~clear_j).at[i].set(al[i] & ~clear_i)
        return jax.lax.fori_loop(0, n, inner, al)
    return jax.lax.fori_loop(0, n, body, alive) & (hel == which)


# --- MergeHelices and the helix quantities --------------------------------------------------------------------------

def _lsq_chi2_circle(x, y, w, mask, ft):
    """LsqSums4::chi2DofCircle of the points (fX0 = fY0 = 0), in the float type ft (double in stock)."""
    x, y, w = x.astype(ft), y.astype(ft), w.astype(ft)
    S = lambda v: jnp.sum(jnp.where(mask, v, 0), axis=-1)
    sw, qn = S(w), S(jnp.ones_like(w))
    m = lambda v: S(w * v) / sw
    xm, ym, x2m, xym, y2m = m(x), m(y), m(x * x), m(x * y), m(y * y)
    x3m, x2ym, xy2m, y3m = m(x ** 3), m(x * x * y), m(x * y * y), m(y ** 3)
    x4m, x2y2m, y4m = m(x ** 4), m(x * x * y * y), m(y ** 4)
    sxx, sxy, syy = x2m - xm * xm, xym - xm * ym, y2m - ym * ym
    sx2x, sx2y, sxy2, syy2 = x3m - xm * x2m, x2ym - ym * x2m, xy2m - xm * y2m, y3m - ym * y2m
    sx2x2, sx2y2, sy2y2 = x4m - x2m * x2m, x2y2m - x2m * y2m, y4m - y2m * y2m
    det = sxx * syy - sxy * sxy
    x0 = (syy * (sx2x + sxy2) - sxy * (sx2y + syy2)) / 2 / det
    y0 = (sxx * (sx2y + syy2) - sxy * (sx2x + sxy2)) / 2 / det
    dx, dy = xm - x0, ym - y0
    r = jnp.sqrt(sxx + syy + dx * dx + dy * dy)
    a, b = sx2x + sxy2, sx2y + syy2
    chi2 = sx2x2 + 2. * sx2y2 + sy2y2 - (syy * a * a + sxx * b * b - 2 * sxy * a * b) / det
    return chi2 / (4 * r * r) * sw / qn


def _robust(hx):
    """RobustHelix of a finder helix: rcent, fcent (polyAtan2), radius, lambda, fz0, and its centre as stock rebuilds it."""
    rcent = jnp.sqrt(hx["cx"] * hx["cx"] + hx["cy"] * hx["cy"])
    fcent = poly_atan2(hx["cy"], hx["cx"])
    fa = jnp.where(hx["fa"] != 0, hx["fa"], np.float32(1))
    return {"rcent": rcent, "fcent": fcent, "R": hx["r"], "lambda": hx["fa"], "fz0": -hx["fb"] / fa,
            "ccx": rcent * jnp.cos(fcent), "ccy": rcent * jnp.sin(fcent)}


def _merge_chi2(h, rh, mask, ms, ft):
    """MergeHelices::findchisq: (chi2dXY, chi2dZPhi) of each helix [n] over its hits [n, H]."""
    dx = h["x"][None, :] - rh["ccx"][:, None]
    dy = h["y"][None, :] - rh["ccy"][:, None]
    vx, vy = -h["udy"], h["udx"]
    dxn = dx * vx[None, :] + dy * vy[None, :]
    costh2 = dxn * dxn / (dx * dx + dy * dy)
    sinth2 = np.float32(1) - costh2
    e2xy = h["uvar"][None, :] * sinth2 + h["vvar"][None, :] * costh2
    wxy = np.float32(1.) / e2xy * np.float32(ms["scalexy"])
    chixy = _lsq_chi2_circle(jnp.broadcast_to(h["x"], mask.shape), jnp.broadcast_to(h["y"], mask.shape), wxy, mask, ft)
    # z-phi line of (z, helixPhi = 0): sigYY = sigXY = 0, so chi2DofLine = 0 (sigXX > 0: MinDZTrk keeps the hits
    # of every helix at least that far apart in z)
    chizphi = jnp.zeros_like(chixy)
    return chixy, chizphi


def _merge(h, ok, mask, chi, ms):
    """MergeHelices over the seed list (Positive helices first, then Negative), the stock list-erase loop: returns
    the surviving helices' mask [n]."""
    n = ok.shape[0]
    nh = jnp.sum(jnp.where(mask, h["nsh"][None, :], 0), axis=1)
    nover = jnp.sum(jnp.where(mask[:, None, :] & mask[None, :, :], h["nsh"][None, None, :], 0), axis=2)

    def comp(i, j):                                      # 0 unique, 1 first better, 2 second better
        minh = jnp.minimum(nh[i], nh[j])
        over = (nover[i, j] >= ms["minnover"]) & (nover[i, j].astype(F32) / minh.astype(F32) > np.float32(ms["minfrac"]))
        first = ((nh[i] > nh[j]) & (nh[i] - nh[j] > ms["deltanh"])) | \
            (~((nh[j] > nh[i]) & (nh[j] - nh[i] > ms["deltanh"])) & ~((nh[i] > nh[j]) & (nh[i] - nh[j] > ms["deltanh"])) &
             (chi[i] < chi[j]))
        return jnp.where(over, jnp.where(first, 1, 2), 0)

    def nxt(alive, k):                                   # first alive index > k (n when none)
        cand = alive & (jnp.arange(n) > k)
        return jnp.where(jnp.any(cand), jnp.argmax(cand), n)

    def cond(st):
        return st[0] < n

    def body(st):
        i, j, alive = st
        c = jnp.where(j < n, comp(i, jnp.minimum(j, n - 1)), -1)
        # j at the end: advance i; unique: advance j; first better: erase j; second better: erase i, restart from next
        alive2 = jnp.where(c == 1, alive.at[jnp.minimum(j, n - 1)].set(False), alive)
        alive2 = jnp.where(c == 2, alive2.at[i].set(False), alive2)
        i2 = jnp.where((c == -1) | (c == 2), nxt(alive2, i), i)
        j2 = jnp.where(c == -1, nxt(alive2, i2), jnp.where(c == 2, nxt(alive2, i2), nxt(alive2, j)))
        return i2, j2, alive2

    i0 = nxt(ok, -1)
    _, _, alive = jax.lax.while_loop(cond, body, (i0, nxt(ok, i0), ok))
    return alive


def event_helices(h, members, tc_valid, s, ms, fz, ft):
    """One event: panel hits h (each [H], panel order), time clusters [C, H] -> merged helix seeds [M] with the
    HelixFilter / KalSeedFilter quantities (M = C x NMaxTrkIter x helicities)."""
    mem = members & tc_valid[:, None]
    H = mem.shape[1]
    L = min(H, CLUSTER_HITS)
    idx = jnp.argsort(~mem, axis=1, stable=True)[:, :L]                           # each cluster's hits, panel order
    hc = jax.tree.map(lambda a: a[idx], h)                                         # [C, L]
    vc = jnp.take_along_axis(mem, idx, axis=1)
    per = jax.vmap(lambda hh, vv: _cluster(hh, vv, s))(hc, vc)
    gmask = jax.vmap(lambda m, ix: jnp.zeros(m.shape[:-1] + (H,), bool).at[:, ix].set(m))(per["mask"], idx)
    per = {**per, "mask": gmask}
    flat = jax.tree.map(lambda a: a.reshape((-1,) + a.shape[2:]), per)            # (cluster, iteration) order
    overflow = jnp.sum(jnp.sum(mem, axis=1) > L)
    keep = {hel: _dedupe(flat["ok"], flat["hel"], flat["mask"], hel) for hel in (1, -1)}
    # MergeHelices list: the Positive collection, then the Negative one
    order = [(1, keep[1]), (-1, keep[-1])]
    sel = jnp.concatenate([k for _, k in order])
    cat = jax.tree.map(lambda a: jnp.concatenate([a, a]), flat)
    rh = _robust(cat)
    chixy, chizphi = _merge_chi2(h, rh, cat["mask"] & sel[:, None], ms, ft)
    alive = _merge(h, sel, cat["mask"] & sel[:, None], chixy.astype(F32) + chizphi.astype(F32), ms)
    mask = cat["mask"] & alive[:, None]
    nsh = jnp.sum(jnp.where(mask, h["nsh"][None, :], 0), axis=1)
    H = mask.shape[1]
    zf = h["z"][jnp.argmax(mask, axis=1)]
    zl = h["z"][H - 1 - jnp.argmax(mask[:, ::-1], axis=1)]
    lam, R = rh["lambda"], rh["R"]
    lam_s = jnp.where(lam != 0, lam, np.float32(1))
    nloops = jnp.abs((zl - zf) / (lam_s * np.float32(2. * math.pi)))
    phf = rh["fz0"][:, None] + jnp.asarray(fz, F32)[None, :] / lam_s[:, None]
    px = rh["ccx"][:, None] + R[:, None] * jnp.cos(phf)
    py = rh["ccy"][:, None] + R[:, None] * jnp.sin(phf)
    rf = jnp.sqrt(px * px + py * py)
    from .helix import TRACKER_RIN, TRACKER_ROUT, MM2MEV
    expected = np.float32(1e-6) + jnp.sum(~(rf < TRACKER_RIN) & ~(rf > TRACKER_ROUT), axis=1).astype(F32)
    return {"overflow": overflow, "ok": alive, "helicity": jnp.concatenate([jnp.ones(flat["ok"].shape[0]), -jnp.ones(flat["ok"].shape[0])]),
            "R": R, "rcent": rh["rcent"], "fcent": rh["fcent"], "lambda": lam,
            "p": MM2MEV * jnp.sqrt(R * R + lam * lam), "pt": MM2MEV * R, "d0": rh["rcent"] - R,
            "chi2xy": chixy.astype(F32), "chi2zphi": chizphi.astype(F32), "nsh": nsh.astype(F32), "nloops": nloops,
            "hitratio": nsh.astype(F32) / expected, "tandip": jnp.abs(lam) / R}


def helices(ph, tc, s, ms, fz):
    """Panel hits [E, H] + time clusters [E, C] -> merged mpr helix seeds [E, M] (process float type for the merge
    chi2 sums, double in stock)."""
    ft = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    upanel = (ph["sid"] >> 10) * 6 + ((ph["sid"] & 0x380) >> 7)
    order = jnp.argsort(jnp.where(ph["valid"], upanel, 1 << 30), axis=1, stable=True)
    take = lambda a: jnp.take_along_axis(a, order, axis=1)
    h = {"x": take(ph["x"]), "y": take(ph["y"]), "z": take(ph["z"]), "time": take(ph["time"]),
         "nsh": take(ph["ncombo"]).astype(jnp.int32), "uvar": take(ph["uvar"]), "vvar": take(ph["vvar"]),
         "udx": take(ph["udx"]), "udy": take(ph["udy"]), "edep": take(ph["edep"]), "upanel": take(upanel)}
    h = {k: (v.astype(F32) if k not in ("nsh", "upanel") else v) for k, v in h.items()}
    members = jnp.take_along_axis(tc["members"], order[:, None, :], axis=2) & take(ph["valid"])[:, None, :]
    return jax.vmap(lambda hh, mm, vv: event_helices(hh, mm, vv, s, ms, fz, ft))(h, members, tc["valid"])
