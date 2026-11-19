"""RobustMultiHelixFinder and MergeHelices (TrkPatRec / TrkReco, Offline v13_39) on the device, for the mpr paths.

A port of the stock finder per time cluster, at the trigger's settings: no calorimeter cluster on the time cluster
(our inputs have none; a cluster with one would need the calo terms), TargetCon false, FitCircleStrategy ChiSquared.
Each stock expression is evaluated in its own precision and operation order, in both precisions of the chain:

- float where the stock code computes in float, double where a C++ double literal or a double function promotes it
  (`1.0 - x`, `1.0 / res2`, `/ 6.283185`, `n * 6.283185`, `+ M_PI`, `lambda * 2. * M_PI`, circleAzimuth, LsqSums4):
  the module traces under `jax.enable_x64`, so float64 is there in both precisions; RobustHelix's cos and sin of a
  float are the float overloads (the correctly rounded value);
- every product is rounded on its own before it is added (`_m`): XLA contracts a * b + c into one fused multiply-add,
  which the stock build does not;
- the hits of a time cluster are in the stock TimeCluster order (index order, recovered hits appended in the order
  they were added) and each track iteration orders them with libstdc++'s std::sort by panel, which is not stable for
  more than 16 hits (`_stock_sort`); float sums run in that order, and MergeHelices and HelixTool read each seed's hits
  in it.

The quirks of the stock code are kept:

- the triplet distance cut on (x3 - x2, y3 - y2), x2 and y2 being already relative to the first hit;
- the per-panel stub choice of the circle scan and its chi2 sum (see `_stubs`);
- the circle kept is the last triplet with the most straw hits and the smallest chi2 sum;
- init_dzdp keeps, among the slopes with the most hits, the LAST one with the LARGEST sum of residuals;
- MergeHelices recomputes chi2dXY / chi2dZPhi (LsqSums4) with the RobustHelix centre rebuilt from rcent and the
  polyAtan2 azimuth; the helix azimuth of the hits is the ComboHit default 0, so chi2dZPhi is 0.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .exact import g as _g, m as _m, strict  # noqa: F401  (strict: the context mpr traces under)

F32 = jnp.float32
F64 = jnp.float64
BIG = np.float32(3.0e38)
CLUSTER_HITS = 64                    # hit slots per time cluster for the finder (more: counted in `overflow`)
SORT_THRESHOLD = 16                  # libstdc++ _S_threshold
_PI_2 = np.float32(math.pi / 2.0)    # polyAtan2's `float PI_2 = M_PI/2.`
MM2MEV = np.float32(3. / 10.)        # HelixFilter's `float mm2MeV = 3./10.`


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
    return {"minnover": int(m.get("MinNHitOverlap", 10)), "minfrac": np.float32(m.get("MinHitOverlapFraction", 0.5)),
            "deltanh": int(m.get("deltanh", 5)), "scalexy": np.float32(m.get("scaleXY", 1.1))}


# --- stock arithmetic -----------------------------------------------------------------------------------------------

def _argmax(x, axis=None):
    """jnp.argmax with int32 indices: under `strict` jnp.argmax asks for int64 ones, and the chain lowers its jit
    after the x64 context has closed, which lowers the reduction's index type as int32 (a verifier error)."""
    x = jnp.asarray(x)
    if axis is None:
        x, axis = jnp.ravel(x), 0
    if x.dtype == jnp.bool_:
        x = x.astype(jnp.int32)
    return jax.lax.argmax(x, axis % x.ndim, jnp.int32)


def _d(x):
    return jnp.asarray(x).astype(F64)


def _f(x):
    return jnp.asarray(x).astype(F32)


def _round(x):
    """std::round: halves away from zero."""
    t = jnp.trunc(x)
    return jnp.where(jnp.abs(x - t) >= 0.5, t + jnp.sign(x), t)


def _loops(a):
    """std::round(a / 6.283185) of a float numerator with the double literal (the dz/dphi loop count)."""
    return _round(_d(a) / 6.283185)


def _approx_atan(z):
    return _m(np.float32(0.97239411) + _m(np.float32(-0.19194795), z, z), z)


def poly_atan2(y, x):
    """Mu2eUtilities polyAtan2 (float; the x < 0 branch adds M_PI in double)."""
    y, x = _f(y), _f(x)
    ax = _approx_atan(y / jnp.where(x != 0, x, np.float32(1.0)))
    ay = _approx_atan(x / jnp.where(y != 0, y, np.float32(1.0)))
    r_x = jnp.where(x > 0, ax, jnp.where(y >= 0, _f(_d(ax) + math.pi), _f(_d(ax) - math.pi)))
    r_y = jnp.where(y > 0, -ay + _PI_2, -ay - _PI_2)
    r0 = jnp.where(y > 0, _PI_2, jnp.where(y < 0, -_PI_2, np.float32(0.0)))
    return jnp.where(x != 0, jnp.where(jnp.abs(x) > jnp.abs(y), r_x, r_y), r0)


def _seq_sum(v, mask, axis=-1):
    """Sum in index order along `axis` (the stock accumulation order), in v's precision."""
    v = jnp.moveaxis(jnp.where(mask, v, jnp.zeros((), v.dtype)), axis, 0)
    s, _ = jax.lax.scan(lambda c, a: (c + a, None), jnp.zeros(v.shape[1:], v.dtype), v)
    return s


def _ls_fit(x, y, w, mask):
    """LSFitter (float sums in hit order): y = fa x + fb; 0 when |det| <= 1e-6."""
    sn, sx, sx2 = _seq_sum(w, mask), _seq_sum(_m(x, w), mask), _seq_sum(_m(x, x, w), mask)
    sy, sxy = _seq_sum(_m(y, w), mask), _seq_sum(_m(x, y, w), mask)
    den = _m(sn, sx2) - _m(sx, sx)
    ok = _d(jnp.abs(den)) > 1e-6
    d = jnp.where(ok, den, np.float32(1.0))
    return (jnp.where(ok, (_m(sn, sxy) - _m(sx, sy)) / d, np.float32(0.0)),
            jnp.where(ok, (_m(sy, sx2) - _m(sx, sxy)) / d, np.float32(0.0)))


def _first_strict_max(v, mask, start):
    """Index kept by `if (v > best) {best = v; i = k;}` from best = start; -1 if no element beats start."""
    vv = jnp.where(mask, v, -BIG)
    m = jnp.max(vv)
    hit = mask & (vv == m) & (m > start)
    return jnp.where(jnp.any(hit), _argmax(hit), -1)


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


# --- std::sort ------------------------------------------------------------------------------------------------------

def _stock_sort(key, sel):
    """The order libstdc++'s std::sort (GCC 13 / 14) gives the selected slots, taken in slot order, by `key` with
    operator<: introsort (median of three to the first element, unguarded Hoare partition, recursion above 16
    elements), then the final insertion sort, which is stable and so equals a stable sort of the arrangement the
    partitions leave. Returns (perm, n, fallback): perm [L] lists the selected slots in that order, then the others;
    fallback is True where introsort would have switched to heap sort (depth limit), which this does not reproduce."""
    L = sel.shape[0]
    n = jnp.sum(sel).astype(jnp.int32)
    a = jnp.argsort(~sel, stable=True).astype(jnp.int32)               # selected slots first, in slot order
    S = 2 * max(L, 2).bit_length() + 4                                  # stack bound: one pending range per level
    depth0 = jnp.where(n > 0, 2 * (31 - jax.lax.clz(jnp.maximum(n, 1))), 0)
    pos = jnp.arange(L, dtype=jnp.int32)

    def lt(i, j, a):
        return key[a[i]] < key[a[j]]

    def swap(a, i, j):
        ai, aj = a[i], a[j]
        return a.at[i].set(aj).at[j].set(ai)

    def partition(a, f, l, go):
        mid = f + (l - f) // 2
        x, y, z = f + 1, mid, l - 1
        sw = jnp.where(lt(x, y, a), jnp.where(lt(y, z, a), y, jnp.where(lt(x, z, a), z, x)),
                       jnp.where(lt(x, z, a), x, jnp.where(lt(y, z, a), z, y)))
        a = swap(a, f, sw)
        pk = key[a[f]]

        def cond(st):
            return go & ~st[3]

        def body(st):
            a, first, last, _ = st
            ka = key[a]
            cand = (pos >= first) & ~(ka < pk)
            first = jnp.where(jnp.any(cand), _argmax(cand), L).astype(jnp.int32)
            last = (L - 1 - _argmax(((pos <= last - 1) & ~(pk < ka))[::-1])).astype(jnp.int32)
            done = ~(first < last)
            a = jnp.where(done, a, swap(a, first, last))
            return a, jnp.where(done, first, first + 1), last, done

        a, cut, _, _ = jax.lax.while_loop(cond, body, (a, f + 1, l, jnp.bool_(False)))
        return a, cut

    def cond(st):
        return st[4] > 0

    def body(st):
        a, sf, sl, sd, top, fb = st
        top = top - 1
        f, l, d = sf[top], sl[top], sd[top]
        big = (l - f) > SORT_THRESHOLD
        fb = fb | (big & (d == 0))
        go = big & (d > 0)
        a2, cut = partition(a, f, l, go)
        a = jnp.where(go, a2, a)
        sf = jnp.where(go, sf.at[top].set(cut).at[top + 1].set(f, mode="drop"), sf)
        sl = jnp.where(go, sl.at[top].set(l).at[top + 1].set(cut, mode="drop"), sl)
        sd = jnp.where(go, sd.at[top].set(d - 1).at[top + 1].set(d - 1, mode="drop"), sd)
        return a, sf, sl, sd, jnp.where(go, top + 2, top), fb

    z = jnp.zeros(S, jnp.int32)
    st = (a, z, z.at[0].set(n), z.at[0].set(depth0), jnp.int32(1), jnp.bool_(False))
    a, _, _, _, _, fb = jax.lax.while_loop(cond, body, st)
    fin = jnp.argsort(jnp.where(pos < n, key[a], jnp.iinfo(jnp.int32).max), stable=True)   # final insertion sort
    return a[fin], n, fb


# --- circle candidate -----------------------------------------------------------------------------------------------

def _compton(h, proc, s):
    phi = poly_atan2(h["y"], h["x"])
    phi = jnp.where(phi < 0, _f(_d(phi) + 6.28318), phi)
    nb = int(6.28318 / float(s["phibin"])) + 1
    ib = jnp.clip((phi / s["phibin"]).astype(jnp.int32), 0, nb - 1)
    hist = jnp.zeros(nb, jnp.int32).at[ib].add(jnp.where(proc, h["nsh"], 0))
    i = jnp.arange(nb + 3)
    tri = hist[(i + nb - 1) % nb] + hist[i % nb] + hist[(i + 1) % nb]
    imax = _first_strict_max(tri, jnp.ones(tri.shape, bool), 0)
    smax = jnp.where(imax >= 0, tri[jnp.maximum(imax, 0)], 0)
    phimax = _f((_d(jnp.maximum(imax, 0)) + 0.5) * _d(s["phibin"]))
    dphi = phi - phimax
    dphi = jnp.where(_d(dphi) > 6.2831, _f(_d(dphi) - 6.2831), dphi)
    dphi = jnp.where(_d(dphi) < -6.2831, _f(_d(dphi) + 6.2831), dphi)
    return proc & (smax >= s["minbin"]) & (_d(jnp.abs(dphi)) < 1.5 * _d(s["phibin"]))


def _stubs(h, proc, first_panel, cx, cy, r, s):
    """The circle's stub hits, their straw hits and the chi2 sum per straw hit (circles broadcast over leading axes).

    Stock walks the processed hits in order with state (panel id_prev, best hit ilast, val_prev): a passing hit
    (dR^2 <= MinDRCircle^2) of another panel than id_prev pushes ilast when ilast is set, and starts a new group;
    otherwise it replaces ilast when strictly closer. id_prev starts as the first processed hit's panel and changes
    only at a push. So the groups are runs of passing hits split where the panel changes (the second passing hit
    compares with the first processed hit's panel); each group contributes its first closest hit; the chi2 sum is the
    first hit's dR^2 of every group after the first, plus the closest dR^2 of the last group."""
    dx, dy = h["x"] - cx[..., None], h["y"] - cy[..., None]
    dr = jnp.sqrt(_m(dx, dx) + _m(dy, dy)) - r[..., None]
    chisq = _m(dr, dr)
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
    first_panel = h["upanel"][_argmax(proc)]
    x, y = h["x"], h["y"]
    r2 = _m(x, x) + _m(y, y)
    jj, kk = jnp.meshgrid(jnp.arange(H), jnp.arange(H), indexing="ij")
    flat = jj * H + kk

    def per_i(best, i):
        bn, bc = best[0], best[1]
        x1, y1 = x[i], y[i]
        x2, y2 = (x - x1)[:, None], (y - y1)[:, None]                  # second hit j
        x3, y3 = (x - x1)[None, :], (y - y1)[None, :]                  # third hit k
        d12, d13 = _m(x2, x2) + _m(y2, y2), _m(x3, x3) + _m(y3, y3)
        x4, y4 = x[None, :] - x2, y[None, :] - y2
        d4 = _m(x4, x4) + _m(y4, y4)
        den = np.float32(2) * (_m(x2, y3) - _m(x3, y2))
        bad = _d(jnp.abs(den)) < 1e-9
        dd = jnp.where(bad, np.float32(1.0), den)
        cx = (_m(d12, y3) - _m(d13, y2)) / dd + x1
        cy = (_m(d13, x2) - _m(d12, x3)) / dd + y1
        r = jnp.sqrt(_m(x1 - cx, x1 - cx) + _m(y1 - cy, y1 - cy))
        rc2 = _m(cx, cx) + _m(cy, cy)
        okt = seed[i] & seed[:, None] & seed[None, :] & (jj > i) & (kk > jj)
        okt &= ~(d12 < s["mindxy2"]) & ~(d13 < s["mindxy2"]) & ~(d4 < s["mindxy2"]) & ~bad
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
    nsteps = int(np.round(np.float32(np.float32(fa_end - fa_init) / s["dpdzstep"])))
    fas = jnp.asarray(np.float32(fa_init) + np.arange(nsteps + 1).astype(np.float32) * s["dpdzstep"], F32)
    jsel = mem & ((jnp.cumsum(mem) - 1) % 2 == 0)
    i0 = _argmax(mem)
    pos = jnp.arange(H)

    def per_fa(fa):
        fb = h["z"] - _m(fa, p)                                        # candidate intercept per anchor j
        q = (h["z"][None, :] - fb[:, None]) / fa                       # [j, hit]
        nl = _loops(p[None, :] - q)
        delta = jnp.abs(q - _f(_d(p[None, :]) - _m(nl, 6.283185)))
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
    nl = _loops(phic - q)
    ploop = _f(_d(phic) - _m(nl, 6.283185))
    use = mem & (jnp.abs(q - ploop) < s["dphiinit"])
    fa, fb = _ls_fit(ploop, h["z"], h["nsh"].astype(F32), use)
    keep = (_d(fa) < 1e-3) if hel > 0 else (_d(fa) > -1e-3)
    fs = jnp.where(keep, np.float32(1.0), fa)
    fbn = _f(_d(fb) - _m(jnp.trunc(_d(fb / fs) / 6.283185), 6.283185, _d(fs)))
    return jnp.where(keep, fa0, fa), jnp.where(keep, fb0, fbn)


def _res2(werr, terr, c2):
    """werr*werr*c2 + terr*terr*(1.0-c2): float first term, double second term and sum, stored as float."""
    return _f(_d(_m(werr, werr, c2)) + _m(_d(_m(terr, terr)), 1.0 - _d(c2)))


def _chi2_circle(h, cx, cy, r):
    dx, dy = h["x"] - cx, h["y"] - cy
    rr = jnp.sqrt(_m(dx, dx) + _m(dy, dy))
    dr = rr - r
    rw = (_m(h["udx"], dx) + _m(h["udy"], dy)) / rr
    werr, terr = jnp.sqrt(h["uvar"]), jnp.sqrt(h["vvar"])
    return _m(dr, dr) / _res2(werr, terr, _m(rw, rw))


def _fit_circle_chi2(h, cx0, cy0, mem):
    dx, dy = h["x"] - cx0, h["y"] - cy0
    rw = _m(h["udx"], dx) + _m(h["udy"], dy)
    c2 = _m(rw, rw) / (_m(dx, dx) + _m(dy, dy))
    werr, terr = jnp.sqrt(h["uvar"]), jnp.sqrt(h["vvar"])
    w = _f(1.0 / _d(_res2(werr, terr, c2)))
    x, y = h["x"], h["y"]
    S = lambda *v: _seq_sum(_m(*v), mem)
    x_, x2_, x3_, y_, y2_, y3_ = S(x, w), S(x, x, w), S(x, x, x, w), S(y, w), S(y, y, w), S(y, y, y, w)
    xy_, x2y_, xy2_, w_ = S(x, y, w), S(x, x, y, w), S(x, y, y, w), _seq_sum(w, mem)
    d11, d20, d30 = _m(w_, xy_) - _m(x_, y_), _m(w_, x2_) - _m(x_, x_), _m(w_, x3_) - _m(x2_, x_)
    d21, d02, d03, d12 = _m(w_, x2y_) - _m(x2_, y_), _m(w_, y2_) - _m(y_, y_), _m(w_, y3_) - _m(y2_, y_), \
        _m(w_, xy2_) - _m(x_, y2_)
    den = np.float32(2) * (_m(d20, d02) - _m(d11, d11))
    cx = (_m(d30 + d12, d02) - _m(d03 + d21, d11)) / den
    cy = (_m(d03 + d21, d20) - _m(d30 + d12, d11)) / den
    c = (x2_ + y2_ - _m(np.float32(2), cx, x_) - _m(np.float32(2), cy, y_)) / w_
    return cx, cy, jnp.sqrt(c + _m(cx, cx) + _m(cy, cy))


def _isnormal(x):
    return jnp.isfinite(x) & (jnp.abs(x) >= np.float32(np.finfo(np.float32).tiny))


def _helix(h, cx, cy, r, chits, hel, s):
    """findHelixCandidate for one helicity, from the circle and its stub hits."""
    nsh_c = jnp.sum(jnp.where(chits, h["nsh"], 0))
    fa, fb, found = _init_dzdp(h, cx, cy, chits, hel, s)
    fa, fb = _fit_dzdp(h, cx, cy, jnp.where(found, fa, np.float32(1)), fb, chits, hel, s)
    alive = found & ~(_d(jnp.abs(fa)) < 1e-3)                          # no slope: stock divides by 0, no hit, cleared
    fs = jnp.where(alive, fa, np.float32(1))
    phi = poly_atan2(h["y"] - cy, h["x"] - cx)
    q = (h["z"] - fb) / fs
    m = chits & (_f(jnp.abs(_d(q - phi) + _m(_loops(phi - q), 6.283185))) < s["dphifit"])  # filterZPhi
    fat, fbt = _ls_fit(h["z"], h["time"], jnp.ones_like(h["z"]), m)                         # fit_dzdt
    m = m & (jnp.abs(_m(fat, h["z"]) + fbt - h["time"]) < s["dtfit"])                       # filterZT
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
    ok = alive & ~(_d(jnp.abs(fa)) < 1e-3) & ~(_d(jnp.abs(fat)) < 1e-3) & _isnormal(fa) & _isnormal(fat)
    m = m & ok
    return {"nsh": jnp.sum(jnp.where(m, h["nsh"], 0)), "mask": m, "cx": cx, "cy": cy, "r": r, "fa": fa, "fb": fb}


# --- one time cluster -------------------------------------------------------------------------------------------------

def _cluster(h, members, s):
    """findHelicesInTC on one time cluster, its hits in the stock TimeCluster order: per track iteration, the kept
    helix (or none), as arrays [niter] (ok, helicity, nsh, cx, cy, r, fa, fb), hit masks [niter, H] and each hit's
    position in that iteration's hit order (rank [niter, H]), plus `fallback` (a sort this port does not cover).
    Call it inside `strict`."""
    H = members.shape[0]
    outs = []
    used = jnp.zeros(H, bool)
    stopped = jnp.bool_(False)
    fallback = jnp.bool_(False)
    for _ in range(s["niter"]):
        proc = members & ~used
        togo = jnp.sum(jnp.where(proc, h["nsh"], 0))
        stopped = stopped | (togo < s["minnsh"])
        perm, n, fb = _stock_sort(h["upanel"], proc)                   # hitsToProcess, sorted by panel
        fallback = fallback | (fb & ~stopped)
        hp = {k: v[perm] for k, v in h.items()}
        pp = jnp.arange(H) < n
        rank = jnp.zeros(H, jnp.int32).at[perm].set(jnp.arange(H, dtype=jnp.int32))
        cx, cy, r, stub = _circle(hp, pp, s)
        ncirc = jnp.sum(jnp.where(stub, hp["nsh"], 0))
        has = ~stopped & (ncirc >= s["minnsh"])
        hx = {hel: _helix(hp, cx, cy, r, stub, hel, s) for hel in s["helicities"]}
        pos, neg = hx[1], hx[-1]
        pos_n = jnp.where(has, pos["nsh"], 0)
        neg_n = jnp.where(has, neg["nsh"], 0)
        use_pos = pos_n >= neg_n
        best = jax.tree.map(lambda a, b: jnp.where(use_pos, a, b), pos, neg)
        bm = best["mask"] & has                                        # in the iteration's hit order
        used = used | bm[rank]
        bn = jnp.where(use_pos, pos_n, neg_n)
        z = hp["z"]
        zfirst = z[_argmax(bm)]
        zlast = z[H - 1 - _argmax(bm[::-1])]
        nshf = hp["nsh"].astype(F32)
        es = _seq_sum(_m(hp["edep"], nshf), bm)
        edep = _f(_d(es) / (_d(jnp.sum(jnp.where(bm, hp["nsh"], 0))) + 1e-10))
        ok = ~stopped & (bn >= s["minnsh"]) & ~((zlast - zfirst) < s["mindz"]) & ~(edep > s["maxedep"])
        outs.append({"ok": ok, "hel": jnp.where(use_pos, 1, -1), "nsh": bn, "mask": (bm & ok)[rank], "rank": rank,
                     "cx": best["cx"], "cy": best["cy"], "r": best["r"], "fa": best["fa"], "fb": best["fb"]})
    out = jax.tree.map(lambda *a: jnp.stack(a), *outs)
    return {**out, "fallback": fallback}


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

def _lsq_chi2_circle(x, y, w):
    """LsqSums4::chi2DofCircle (fX0 = fY0 = 0) of the points [n, L] in seed hit order (w = 0 past the end), double
    sums in that order, every term as addPoint and the sig* members compute it."""
    X, Y, W = _d(x), _d(y), _d(w)
    used = W != 0
    X2, Y2, XY = _m(X, X), _m(Y, Y), _m(X, Y)
    W_X, W_Y, W_X2, W_Y2 = _m(W, X), _m(W, Y), _m(W, X2), _m(W, Y2)
    S = lambda v: _seq_sum(v, used)
    qn = jnp.sum(used, axis=-1).astype(F64)
    sw, sx, sy, sx2, sxy, sy2 = S(W), S(W_X), S(W_Y), S(W_X2), S(_m(W, XY)), S(W_Y2)
    sx3, sx2y, sxy2, sy3 = S(_m(W_X2, X)), S(_m(W_X2, Y)), S(_m(W_Y2, X)), S(_m(W_Y2, Y))
    sx4, sx2y2, sy4 = S(_m(W_X2, X2)), S(_m(W_X2, Y2)), S(_m(W_Y2, Y2))
    xm, ym, x2m, xym, y2m = sx / sw, sy / sw, sx2 / sw, sxy / sw, sy2 / sw
    x3m, x2ym, xy2m, y3m = sx3 / sw, sx2y / sw, sxy2 / sw, sy3 / sw
    x4m, x2y2m, y4m = sx4 / sw, sx2y2 / sw, sy4 / sw
    sxx, sxy_, syy = x2m - _m(xm, xm), xym - _m(xm, ym), y2m - _m(ym, ym)
    sx2x, sx2y_, sxy2_, syy2 = x3m - _m(xm, x2m), x2ym - _m(ym, x2m), xy2m - _m(xm, y2m), y3m - _m(ym, y2m)
    sx2x2, sx2y2_, sy2y2 = x4m - _m(x2m, x2m), x2y2m - _m(x2m, y2m), y4m - _m(y2m, y2m)
    det = _m(sxx, syy) - _m(sxy_, sxy_)
    x0 = (_m(syy, sx2x + sxy2_) - _m(sxy_, sx2y_ + syy2)) / 2 / det
    y0 = (_m(sxx, sx2y_ + syy2) - _m(sxy_, sx2x + sxy2_)) / 2 / det
    dx, dy = xm - x0, ym - y0
    r = jnp.sqrt(sxx + syy + _m(dx, dx) + _m(dy, dy))
    a, b = sx2x + sxy2_, sx2y_ + syy2
    chi2 = sx2x2 + _m(2., sx2y2_) + sy2y2 - (_m(syy, a, a) + _m(sxx, b, b) - _m(2., sxy_, a, b)) / det
    chi2 = chi2 / _m(4., r, r)
    return _f(_m(chi2, sw / qn))


def _robust(hx):
    """RobustHelix of a finder helix: rcent, fcent (polyAtan2), radius, lambda, fz0, and its centre as stock rebuilds
    it (float _rcent * cos(float _fcent), all float)."""
    rcent = jnp.sqrt(_m(hx["cx"], hx["cx"]) + _m(hx["cy"], hx["cy"]))
    fcent = poly_atan2(hx["cy"], hx["cx"])
    fa = jnp.where(hx["fa"] != 0, hx["fa"], np.float32(1))
    return {"rcent": rcent, "fcent": fcent, "R": hx["r"], "lambda": hx["fa"], "fz0": -hx["fb"] / fa,
            "ccx": _m(rcent, _f(jnp.cos(_d(fcent)))), "ccy": _m(rcent, _f(jnp.sin(_d(fcent))))}


def _merge_chi2(hs, rh, ms):
    """MergeHelices::findchisq chi2dXY of each helix [n] over its hits hs [n, L] (seed hit order, w = 0 past the
    end); chi2dZPhi of the (z, helixPhi = 0) line is 0 (sigXX > 0: MinDZTrk keeps every helix's hits that far apart
    in z)."""
    dx = hs["x"] - rh["ccx"][:, None]
    dy = hs["y"] - rh["ccy"][:, None]
    vx, vy = -hs["udy"], hs["udx"]
    dxn = _m(dx, vx) + _m(dy, vy)
    costh2 = _m(dxn, dxn) / (_m(dx, dx) + _m(dy, dy))
    sinth2 = np.float32(1) - costh2
    e2xy = _m(hs["uvar"], sinth2) + _m(hs["vvar"], costh2)
    wxy = _m(_f(1. / _d(e2xy)), ms["scalexy"])
    chixy = _lsq_chi2_circle(hs["x"], hs["y"], jnp.where(hs["valid"], wxy, np.float32(0)))
    return chixy, jnp.zeros_like(chixy)


def _merge(h, ok, mask, chi, ms):
    """MergeHelices over the seed list (Positive helices first, then Negative), the stock list-erase loop: returns
    the surviving helices' mask [n]."""
    n = ok.shape[0]
    nh = jnp.sum(jnp.where(mask, h["nsh"][None, :], 0), axis=1)
    nover = jnp.sum(jnp.where(mask[:, None, :] & mask[None, :, :], h["nsh"][None, None, :], 0), axis=2)

    def comp(i, j):                                      # 0 unique, 1 first better, 2 second better
        minh = jnp.minimum(nh[i], nh[j])
        over = (nover[i, j] >= ms["minnover"]) & (nover[i, j].astype(F32) / minh.astype(F32) > ms["minfrac"])
        first = ((nh[i] > nh[j]) & (nh[i] - nh[j] > ms["deltanh"])) | \
            (~((nh[j] > nh[i]) & (nh[j] - nh[i] > ms["deltanh"])) & ~((nh[i] > nh[j]) & (nh[i] - nh[j] > ms["deltanh"])) &
             (chi[i] < chi[j]))
        return jnp.where(over, jnp.where(first, 1, 2), 0)

    def nxt(alive, k):                                   # first alive index > k (n when none)
        cand = alive & (jnp.arange(n) > k)
        return jnp.where(jnp.any(cand), _argmax(cand), n)

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


def _event(h, per, idx, s, ms, fz):
    """One event: panel hits h (each [H], collection order) and the finder results of its clusters `per`
    [C, niter, ...] on the cluster slots idx [C, L] -> merged helix seeds [M] with the HelixFilter / KalSeedFilter
    quantities."""
    H = h["x"].shape[0]
    L = idx.shape[-1]
    seeds = jax.tree.map(lambda a: a.reshape((-1,) + a.shape[2:]), per)                    # (cluster, iteration)
    sidx = jnp.repeat(idx, per["ok"].shape[1], axis=0)                                     # [n, L] hit of each slot
    gmask = jax.vmap(lambda m, ix: jnp.zeros(H, bool).at[ix].set(m))(seeds["mask"], sidx)
    keep = {hel: _dedupe(seeds["ok"], seeds["hel"], gmask, hel) for hel in (1, -1)}
    # MergeHelices list: the Positive collection, then the Negative one
    sel = jnp.concatenate([keep[1], keep[-1]])
    cat = jax.tree.map(lambda a: jnp.concatenate([a, a]), {**seeds, "gmask": gmask, "sidx": sidx})
    # each seed's hits in its hit order (HelixSeed::hits(): the iteration's sorted order)
    order = jnp.argsort(jnp.where(cat["mask"], cat["rank"], H + L), axis=1, stable=True)  # [n, L] slot positions
    hit = jnp.take_along_axis(cat["sidx"], order, axis=1)
    valid = jnp.take_along_axis(cat["mask"], order, axis=1)
    hs = {k: h[k][hit] for k in ("x", "y", "z", "uvar", "vvar", "udx", "udy")}
    hs["valid"] = valid
    rh = _robust(cat)
    chixy, chizphi = _merge_chi2(hs, rh, ms)
    gm = cat["gmask"] & sel[:, None]
    alive = _merge(h, sel, gm, chixy + chizphi, ms)
    mask = gm & alive[:, None]
    nsh = jnp.sum(jnp.where(mask, h["nsh"][None, :], 0), axis=1)
    # HelixTool: first / last hit z in hit order, loops in double, the faces the helix crosses inside the tracker
    nv = jnp.sum(valid, axis=1)
    zf = hs["z"][:, 0]
    zl = jnp.take_along_axis(hs["z"], jnp.maximum(nv - 1, 0)[:, None], axis=1)[:, 0]
    lam, R = rh["lambda"], rh["R"]
    nloops = _f(jnp.abs(_d(zl - zf) / _m(_d(lam), 2., math.pi)))
    lam_d = jnp.where(lam != 0, _d(lam), 1.0)
    az = jnp.where(lam[:, None] != 0, _f(_d(rh["fz0"])[:, None] + _d(jnp.asarray(fz, F32))[None, :] / lam_d[:, None]),
                   np.float32(0))
    px = rh["ccx"][:, None] + _m(R[:, None], _f(jnp.cos(_d(az))))
    py = rh["ccy"][:, None] + _m(R[:, None], _f(jnp.sin(_d(az))))
    rf = jnp.sqrt(_m(px, px) + _m(py, py))
    from .helix import TRACKER_RIN, TRACKER_ROUT, expected_faces
    nface = jnp.sum(~(rf < TRACKER_RIN) & ~(rf > TRACKER_ROUT), axis=1)
    expected = jnp.asarray(expected_faces(len(fz)))[nface]
    n = seeds["ok"].shape[0]
    return {"ok": alive, "helicity": jnp.concatenate([jnp.ones(n, F32), -jnp.ones(n, F32)]),
            "R": R, "rcent": rh["rcent"], "fcent": rh["fcent"], "lambda": lam,
            "p": _m(jnp.sqrt(_m(R, R) + _m(lam, lam)), MM2MEV), "pt": _m(R, MM2MEV), "d0": rh["rcent"] - R,
            "chi2xy": chixy, "chi2zphi": chizphi, "nsh": nsh.astype(F32), "nloops": nloops,
            "hitratio": nsh.astype(F32) / expected, "tandip": jnp.abs(lam) / R}


def helices(ph, tc, s, ms, fz):
    """Panel hits [E, H] + time clusters [E, C] (with the TimeCluster hit order `tc["order"]`) -> merged mpr helix
    seeds [E, M], `overflow` [E] (time clusters with more hits than CLUSTER_HITS: their finder result would be
    truncated, so they are counted) and `fallback` [E] (a hit sort this port does not reproduce). The finder runs on
    the valid clusters only, gathered up to one per event, else on every cluster in the same call."""
    ft = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    with strict(ph["sid"]):
        out = _helices(ph, tc, s, ms, fz)
    return {k: (v.astype(ft) if jnp.issubdtype(v.dtype, jnp.floating) else v) for k, v in out.items()}


def _helices(ph, tc, s, ms, fz):
    upanel = ((ph["sid"] >> 10) * 6 + ((ph["sid"] & 0x380) >> 7)).astype(jnp.int32)
    h = {"x": ph["x"], "y": ph["y"], "z": ph["z"], "time": ph["time"], "nsh": ph["ncombo"], "uvar": ph["uvar"],
         "vvar": ph["vvar"], "udx": ph["udx"], "udy": ph["udy"], "edep": ph["edep"], "upanel": upanel}
    h = {k: (v.astype(F32) if k not in ("nsh", "upanel") else v.astype(jnp.int32)) for k, v in h.items()}
    mem = tc["members"] & ph["valid"][:, None, :] & tc["valid"][:, :, None]
    E, C, H = mem.shape
    L = min(H, CLUSTER_HITS)
    key = jnp.where(mem, tc["order"].astype(jnp.int64), jnp.int64(1) << 40)
    idx = jnp.argsort(key, axis=2, stable=True)[:, :, :L]                            # each cluster's hits, TC order
    hc = jax.tree.map(lambda a: jax.vmap(lambda v, ix: v[ix])(a, idx), h)            # [E, C, L]
    vc = jnp.take_along_axis(mem, idx, axis=2)
    overflow = jnp.sum(jnp.sum(mem, axis=2) > L, axis=1)
    run = jax.vmap(lambda hh, vv: _cluster(hh, vv, s))
    flat_h = jax.tree.map(lambda a: a.reshape((E * C, L)), hc)
    flat_v = vc.reshape(E * C, L)
    use = jnp.any(flat_v, axis=1)
    K = E

    def gathered():
        sel = jnp.nonzero(use, size=K, fill_value=E * C)[0]
        ok = sel < E * C
        si = jnp.minimum(sel, E * C - 1)
        r = run(jax.tree.map(lambda a: a[si], flat_h), flat_v[si] & ok[:, None])
        empty = jax.tree.map(lambda a: jnp.zeros((E * C,) + a.shape[1:], a.dtype), r)
        return jax.tree.map(lambda e, a: e.at[jnp.where(ok, si, E * C)].set(a, mode="drop"), empty, r)

    def full():
        return run(flat_h, flat_v)

    per = jax.lax.cond(jnp.sum(use) <= K, gathered, full)
    fallback = jnp.any(per.pop("fallback").reshape(E, C), axis=1)
    per = jax.tree.map(lambda a: a.reshape((E, C) + a.shape[1:]), per)
    out = jax.vmap(lambda hh, pp, ix: _event(hh, pp, ix, s, ms, fz))(h, per, idx)
    return {**out, "overflow": overflow, "fallback": fallback}
