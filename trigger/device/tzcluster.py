"""TZClusterFinder (CalPatRec), Offline v13_41_01, on the device: the time clusters of the apr paths.

A port of the stock module per event, with the stock arithmetic, as the literal transcription
tests/tz_cluster_reference.py writes it (radSelect 0, doRefine 0, no timeout):

- float32 where stock computes in float (the hit fields, the chunk averages and z range, the time windows, the
  float-assigned fit results), float64 for LsqSums2 (its sums, dydx, y0, chi2Dof), traced under `exact.strict` in
  both precisions of the chain; products that an add follows are guarded (`exact.m`) against fused multiply-add,
  and the intermediates of chi2Dof against XLA's division rewrites (`exact.g`);
- cHitsFill's per-plane sort by descending time and the chunk sorts (by energy selection, then by average time
  within the two groups) are libstdc++'s std::sort (`mpr._stock_sort`), ties included;
- the hits are laid out in the order chunkHits and recoverHits walk them: planes descending, the sorted order within
  a plane. A seed's test hits are then the slots that follow it up to the end of the plane chunkSep below, and the
  membership of a chunk is a mask; its LsqSums2 and float sums are accumulated afterwards in member order, for all
  chunks at once (the walk itself never reads them);
- combineChunks, recoverHits (hit by hit: each recovered hit moves a fit) and combineLines are while / fori loops on
  the chunk slots, chunk order kept as a list of slots (erase shifts it);
- checkCaloClusters walks the calo clusters in order: a second match appends a copy of the chunk (the copies are not
  revisited, `nchunks` is fixed), an unmatched cluster may append a recovered chunk.

The stale `_chunkInfo` state: the recovery pushes the module's reused `_chunkInfo`, whose `nrgSelection` nobody
resets. It holds the energy selection of the last chunk saved by chunkHits or copied by a second calo match, in this
event or an earlier one in file order; a recovered chunk is an electron cluster only if that is 1. Each event's own
last assignment does not depend on the incoming value, so the port resolves it per batch: the events of a call are
taken as consecutive in file order, `nrg_in` is the value entering the first one (-1: unknown), the value entering
each later event is the last assignment before it, and `nrg_out` is the value after the last event (pass it to the
next call). A recovered chunk that needs an unknown value is flagged in `indeterminate` (and is not an electron).

Fixed capacities (all counted in `overflow`, bits: 1 chunk slots, 2 appended-chunk slots, 4 output slots, 8 a sort
that would fall back to heap sort): K_CHUNKS chunks made by chunkHits, A_APPEND chunks appended by the calo step,
the output slots K.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .exact import g as _g, m as _m, strict
from .mpr import _argmax, _d, _f, _stock_sort

F32, F64, I32, I64 = jnp.float32, jnp.float64, jnp.int32, jnp.int64
NPLANES = 36
K_CHUNKS = 64                         # chunks made by chunkHits per event (dumps: at most 29)
A_APPEND = 16                         # chunks appended by checkCaloClusters per event (copies and recoveries; dumps: 3)
K_SLOTS = 16                          # electron time-cluster output slots per event (dumps: at most 8)
OVF_CHUNKS, OVF_APPEND, OVF_SLOTS, OVF_SORT = 1, 2, 4, 8
FLAG_BITS = {"EnergySelection": 1, "Background": 6, "Dead": 23, "Noisy": 24}
_BIG = np.float32(np.inf)


def settings(m):
    """TZClusterFinder module table -> kernel constants; raises for settings the port does not cover."""
    bad = []
    if int(m.get("radSelect", 0)) != 0:
        bad.append("radSelect " + str(m.get("radSelect")))
    if int(m.get("doRefine", 0)) != 0:
        bad.append("doRefine " + str(m.get("doRefine")))
    if float(m.get("timeoutMs", 0) or 0) > 0:
        bad.append("timeoutMs")
    if bad:
        raise ValueError("device TZClusterFinder does not support " + ", ".join(bad))
    bkg = 0
    for b in m["hitBkgBits"]:
        bkg |= 1 << FLAG_BITS[b]
    f = lambda k: np.float32(m[k])
    return {"sep": int(m["chunkSep"]), "win": f("chunkWindow"), "thresh": int(m["chunkThresh"]),
            "cwin": f("combineWindow"), "maxsep": f("maxCombineSep"), "fitthr": int(m["chunkFitThresh"]),
            "rwin": f("recoverWindow"), "cthr": int(m["clusterThresh"]), "minsize": int(m["minCaloSize"]),
            "mine": f("minCaloEnergy"), "dtmax": f("caloDtMax"), "toff": f("caloTimeOffset"),
            "usecc": int(m["useCCs"]) == 1, "reccc": int(m["recoverCCs"]) == 1,
            "requirecc": bool(m.get("requireCCs", False)), "bkg": bkg}


# --- LsqSums2 (double), fields (qn, sw, sx, sy, sx2, sxy, sy2) on the last axis -------------------------------------

def _terms(X, Y, W):
    one = jnp.ones_like(X)
    return jnp.stack([one, W, _m(X, W), _m(Y, W), _m(X, X, W), _m(X, Y, W), _m(Y, Y, W)], -1)


def _add_point(F, X, Y, W, on):
    """addPoint (zero offsets); every field is one double add, so the product is rounded first."""
    return jnp.where(on[..., None], F + _terms(_d(X), _d(Y), _d(W)), F)


def _dydx(F):
    qn, sw, sx, sy, sx2, sxy = (F[..., i] for i in range(6))
    return jnp.where(qn > 1, (_m(sw, sxy) - _m(sy, sx)) / (_m(sw, sx2) - _m(sx, sx)), 0.0)


def _y0(F):
    sw, sx, sy = F[..., 1], F[..., 2], F[..., 3]
    return (0.0 + _g(sy / sw)) + _m(_dydx(F), 0.0 - _g(sx / sw))


def _chi2dof(F):
    qn, sw, sx, sy, sx2, sxy, sy2 = (F[..., i] for i in range(7))
    xm, ym = _g(sx / sw), _g(sy / sw)
    sxx = _g(sx2 / sw) - _m(xm, xm)
    sxy_ = _g(sxy / sw) - _m(xm, ym)
    syy = _g(sy2 / sw) - _m(ym, ym)
    c = _m(syy, sxx) - _m(sxy_, sxy_)
    c = _g(c / sxx)
    c = _m(c, _g(sw / (qn - 2)))
    return jnp.where(qn > 2, c, 0.0)


def _bit(cond, b):
    return jnp.where(cond, I32(b), I32(0))


def _first_min(v, mask):
    """Index of the first strict minimum of v over mask (the stock `found == 1 || dt < min` scan), -1 if none."""
    mn = jnp.min(jnp.where(mask, v, _BIG))
    hit = mask & (v == mn)
    return jnp.where(jnp.any(mask), _argmax(hit), -1)


# --- one event --------------------------------------------------------------------------------------------------------

def _event(h, c, s, K, A):
    H = h["time"].shape[0]
    C = c["time"].shape[0]
    g = jnp.arange(H, dtype=I32)
    P = jnp.arange(K, dtype=I32)
    t, tv, z = _f(h["time"]), _f(h["timevar"]), _f(h["z"])
    nsh = h["ncombo"].astype(I32)
    pl = h["plane"].astype(I32)
    flag = h["flag"].astype(I64)
    esel = ((flag >> FLAG_BITS["EnergySelection"]) & 1).astype(I32)
    bkg = (flag & s["bkg"]) != 0
    w = jnp.where(tv > 0, F32(1) / jnp.where(tv > 0, tv, F32(1)), F32(0))
    hv = h["valid"]
    sel = hv & ~((esel == 1) & bkg)

    # cHitsFill: per-plane std::sort by descending time, planes walked from the last
    perm, cnt, fb = jax.vmap(lambda p: _stock_sort(-t, sel & (pl == p)))(jnp.arange(NPLANES, dtype=I32))
    cnt = cnt.astype(I32)
    nsel = jnp.sum(cnt)
    start = nsel - jnp.cumsum(cnt)
    r = jnp.arange(H, dtype=I32)
    dst = jnp.where(r[None, :] < cnt[:, None], start[:, None] + r[None, :], H)
    walk = jnp.zeros(H + 1, I32).at[dst.reshape(-1)].set(perm.reshape(-1).astype(I32), mode="drop")[:H]
    wv = g < nsel
    wt, wz, ww, wn, we, wpl = t[walk], z[walk], w[walk], nsh[walk], esel[walk], pl[walk]
    blk_end = start + cnt
    ovf = _bit(jnp.any(fb), OVF_SORT)

    # chunkHits: the walk (memberships, counts, z range); the sums follow in member order
    def seed(q, st):
        used, own, rnk, nh, ns, nrg, zmin, zmax, nch, last, nsave = st
        go = ~used[q]
        end = blk_end[jnp.maximum(wpl[q] - s["sep"], 0)]
        cand = (g > q) & (g < end) & ~used & (we == we[q]) & (jnp.abs(wt - wt[q]) < s["win"])
        mem = cand | (g == q)
        n_ = jnp.sum(mem).astype(I32)
        save = go & (n_ >= s["thresh"])
        ok = save & (nch < K)
        k = jnp.where(ok, nch, K)
        rank = jnp.cumsum(mem).astype(I32) - 1
        put = mem & ok
        own = jnp.where(put, nch, own)
        rnk = jnp.where(put, rank, rnk)
        nh = nh.at[k].set(n_, mode="drop")
        ns = ns.at[k].set(jnp.sum(jnp.where(mem, wn, 0)).astype(I32), mode="drop")
        nrg = nrg.at[k].set(we[q], mode="drop")
        zmin = zmin.at[k].set(jnp.min(jnp.where(mem, wz, _BIG)), mode="drop")
        zmax = zmax.at[k].set(jnp.max(jnp.where(mem, wz, -_BIG)), mode="drop")
        return (used | (mem & save), own, rnk, nh, ns, nrg, zmin, zmax, nch + ok.astype(I32),
                jnp.where(save, we[q], last), nsave + save.astype(I32))

    zK = jnp.zeros(K, I32)
    st = (jnp.zeros(H, bool), jnp.full(H, -1, I32), jnp.zeros(H, I32), zK, zK, zK, jnp.zeros(K, F32),
          jnp.zeros(K, F32), I32(0), I32(-1), I32(0))
    used, own, rnk, nh, ns, nrg, zmin, zmax, nch, last, nsave = jax.lax.fori_loop(0, nsel, seed, st)
    ovf = ovf | _bit(nsave > K, OVF_CHUNKS)

    tab = jnp.zeros(K * H + 1, I32).at[jnp.where(own >= 0, own * H + rnk, K * H)].set(g, mode="drop")[:K * H]
    tab = tab.reshape(K, H)

    def sums(i, acc):
        F, tt, tz = acc
        on = (i < nh) & (P < nch)
        q = tab[:, i]
        F = _add_point(F, wz[q], wt[q], ww[q], on)
        return F, jnp.where(on, tt + wt[q], tt), jnp.where(on, tz + wz[q], tz)

    F, tt, tz = jax.lax.fori_loop(0, jnp.max(jnp.where(P < nch, nh, 0)), sums,
                                  (jnp.zeros((K, 7), F64), jnp.zeros(K, F32), jnp.zeros(K, F32)))
    fnh = _f(jnp.maximum(nh, 1))
    at, az = tt / fnh, tz / fnh
    ncb = zK

    # chunk order: CE first, then protons, each by average time (std::sort, three calls), or by time alone
    live = P < nch
    nce = jnp.sum(live & (nrg == 1)).astype(I32)
    both = (nce > 0) & (nce < nch)
    a1, _, fb1 = _stock_sort(-nrg, live)
    hp, _, fb2 = _stock_sort(at[a1], P < nce)
    tp, _, fb3 = _stock_sort(at[a1], (P >= nce) & live)
    ord_both = jnp.where(P < nce, a1[hp], a1[tp[jnp.clip(P - nce, 0, K - 1)]])
    a0, _, fb0 = _stock_sort(at, live)
    order = jnp.where(both, ord_both, a0)
    ovf = ovf | _bit(jnp.where(both, fb1 | fb2 | fb3, fb0), OVF_SORT)
    n = nch

    ch = {"F": F, "at": at, "az": az, "nh": nh, "ns": ns, "nrg": nrg, "zmin": zmin, "zmax": zmax, "ncb": ncb}

    def merge(ch, own, rnk, order, n, pa, pb, Fnew):
        """Chunk at position pb into the one at pa (fitter Fnew), then erase position pb."""
        a, b = order[pa], order[pb]
        na, nb = ch["nh"][a], ch["nh"][b]
        fa, fb_, fs = _f(na), _f(nb), _f(na + nb)
        avg = lambda k: (_m(ch[k][a], fa) + _m(ch[k][b], fb_)) / fs      # (x * na + y * nb) / (na + nb), float
        zmn = jnp.where(ch["zmin"][b] < ch["zmin"][a], ch["zmin"][b], ch["zmin"][a])
        zmx = jnp.where(ch["zmax"][a] < ch["zmax"][b], ch["zmax"][b], ch["zmax"][a])
        ch = {**ch, "F": ch["F"].at[a].set(Fnew), "at": ch["at"].at[a].set(avg("at")),
              "az": ch["az"].at[a].set(avg("az")), "nh": ch["nh"].at[a].set(na + nb),
              "ns": ch["ns"].at[a].set(ch["ns"][a] + ch["ns"][b]), "zmin": ch["zmin"].at[a].set(zmn),
              "zmax": ch["zmax"].at[a].set(zmx), "ncb": ch["ncb"].at[a].add(1)}
        mv = own == b
        rnk = jnp.where(mv, rnk + na, rnk)
        own = jnp.where(mv, a, own)
        order = jnp.where(P >= pb, order[jnp.minimum(P + 1, K - 1)], order)
        return ch, own, rnk, order, n - 1

    def pick(sel, upd, st):
        return jax.tree.map(lambda x, y: jnp.where(sel, x, y), upd, st)

    # combineChunks
    def cc_cond(st):
        return st[-1]

    def cc_body(st):
        ch, own, rnk, order, n, _ = st
        sa, sb = order, order[jnp.minimum(P + 1, K - 1)]
        pair = P < n - 1
        elig = pair & (ch["nrg"][sa] == ch["nrg"][sb])
        elig &= ~((ch["ncb"][sa] == 0) & (ch["ncb"][sb] == 0) & (jnp.abs(ch["az"][sa] - ch["az"][sb]) > s["maxsep"]))
        dt = jnp.abs(ch["at"][sa] - ch["at"][sb])
        big = dt > s["cwin"]
        valid = elig & ~big
        fv = jnp.where(jnp.any(valid), _argmax(valid), K)
        brk = elig & big & (P > fv)
        fbk = jnp.where(jnp.any(brk), _argmax(brk), K)
        cons = valid & (P < fbk)
        pa = _first_min(dt, cons)
        found = pa >= 0
        pa = jnp.maximum(pa, 0)
        Fnew = ch["F"][order[pa]] + ch["F"][order[pa + 1]]
        upd = merge(ch, own, rnk, order, n, pa, pa + 1, Fnew)
        return (*pick(found, upd, (ch, own, rnk, order, n)), found)

    ch, own, rnk, order, n, _ = jax.lax.while_loop(cc_cond, cc_body, (ch, own, rnk, order, n, n > 0))

    # recoverHits: the unused hits in walk order, each into the nearest fit line of its kind
    def recover(q, st):
        ch, own, rnk = st
        sa = order
        elig = (P < n) & (ch["nh"][sa] >= s["fitthr"]) & (ch["nrg"][sa] == we[q])
        Fs = ch["F"][sa]
        dt = _f(jnp.abs(_d(wt[q]) - (_m(_dydx(Fs), _d(wz[q])) + _y0(Fs))))
        p = _first_min(dt, elig & ~(dt > s["rwin"]))
        go = ~used[q] & (p >= 0)
        b = jnp.where(go, sa[jnp.maximum(p, 0)], K)
        bb = jnp.minimum(b, K - 1)
        zq = wz[q]
        ch = {**ch, "F": ch["F"].at[b].set(_add_point(ch["F"][bb], zq, wt[q], ww[q], jnp.bool_(True)), mode="drop"),
              "ns": ch["ns"].at[b].add(wn[q], mode="drop"),
              "zmin": ch["zmin"].at[b].set(jnp.where(zq < ch["zmin"][bb], zq, ch["zmin"][bb]), mode="drop"),
              "zmax": ch["zmax"].at[b].set(jnp.where(ch["zmax"][bb] < zq, zq, ch["zmax"][bb]), mode="drop"),
              "nh": ch["nh"].at[b].add(1, mode="drop")}
        own = own.at[q].set(jnp.where(go, b, own[q]))
        rnk = rnk.at[q].set(jnp.where(go, ch["nh"][bb] - 1, rnk[q]))
        return ch, own, rnk

    ch, own, rnk = jax.lax.fori_loop(0, nsel, recover, (ch, own, rnk))

    # combineLines: the first (i, j) pair of disjoint clusters whose joint fit is better, until none
    iu = P[:, None] < P[None, :]

    def cl_body(st):
        ch, own, rnk, order, n, _ = st
        sa = order
        okp = (P < n) & (ch["ns"][sa] >= s["cthr"])
        Fs = ch["F"][sa]
        nr = ch["nrg"][sa]
        zmn, zmx = ch["zmin"][sa], ch["zmax"][sa]
        c1 = _f(_chi2dof(Fs))
        Fc = Fs[:, None, :] + Fs[None, :, :]
        cc = _f(_chi2dof(Fc))
        m = iu & okp[:, None] & okp[None, :] & (nr[:, None] == nr[None, :])
        m &= (zmx[:, None] < zmn[None, :]) | (zmn[:, None] > zmx[None, :])
        m &= (cc < c1[None, :]) | (cc < c1[:, None])
        found = jnp.any(m)
        ij = _argmax(m.reshape(-1))
        pa, pb = ij // K, ij % K
        upd = merge(ch, own, rnk, order, n, pa, pb, Fc[pa, pb])
        return (*pick(found, upd, (ch, own, rnk, order, n)), found)

    ch, own, rnk, order, n, _ = jax.lax.while_loop(cc_cond, cl_body, (ch, own, rnk, order, n, n > 0))

    sa = order
    live = P < n
    n_p = jnp.sum(live & (ch["nrg"][sa] != 1) & (ch["ns"][sa] >= s["cthr"])).astype(I32) & 0xFFFF
    y0 = _y0(ch["F"])                                    # per chunk slot
    calo = jnp.full(K, -1, I32)

    # checkCaloClusters
    akind, asrc, acalo, anrg, ans = (jnp.zeros(A, I32) for _ in range(5))
    arec = jnp.zeros((A, H), bool)
    assigned = nsave > 0
    stale = last

    def calo_step(i, st):
        calo, akind, asrc, acalo, anrg, ans, arec, nA, assigned, stale, ovf = st
        okc = c["valid"][i] & ~(c["energy"][i] < s["mine"]) & ~(c["size"][i] < s["minsize"])
        cct = _f(c["time"][i]) + s["toff"]
        y0f = _f(y0[sa])
        match = okc & live & (ch["nh"][sa] >= s["fitthr"]) & (jnp.abs(y0f - cct) < s["dtmax"])
        dup = match & (calo[sa] != -1)
        di = nA + jnp.cumsum(dup).astype(I32) - 1
        dd = jnp.where(dup & (di < A), di, A)
        akind = akind.at[dd].set(1, mode="drop")
        asrc = asrc.at[dd].set(sa, mode="drop")
        acalo = acalo.at[dd].set(i, mode="drop")
        anrg = anrg.at[dd].set(ch["nrg"][sa], mode="drop")
        ans = ans.at[dd].set(ch["ns"][sa], mode="drop")
        nd = jnp.sum(dup).astype(I32)
        lastd = K - 1 - _argmax(dup[::-1])
        stale = jnp.where(nd > 0, ch["nrg"][sa[lastd]], stale)
        assigned = assigned | (nd > 0)
        calo = calo.at[jnp.where(match & (calo[sa] == -1), sa, K)].set(i, mode="drop")
        nA = nA + nd
        rm = hv & ~bkg & (esel == 1) & (jnp.abs(t - cct) < s["dtmax"])
        nsr = jnp.sum(jnp.where(rm, nsh, 0)).astype(I32)
        push = okc & s["reccc"] & ~jnp.any(match) & (nsr >= s["cthr"])
        k = jnp.where(push & (nA < A), nA, A)
        akind = akind.at[k].set(2, mode="drop")
        acalo = acalo.at[k].set(i, mode="drop")
        anrg = anrg.at[k].set(jnp.where(assigned, stale, -2), mode="drop")
        ans = ans.at[k].set(nsr, mode="drop")
        arec = arec.at[k].set(rm, mode="drop")
        nA = nA + push.astype(I32)
        return calo, akind, asrc, acalo, anrg, ans, arec, nA, assigned, stale, ovf

    st = (calo, akind, asrc, acalo, anrg, ans, arec, I32(0), assigned, stale, ovf)
    if s["usecc"]:
        st = jax.lax.fori_loop(0, C, calo_step, st)
    calo, akind, asrc, acalo, anrg, ans, arec, nA, assigned, stale, ovf = st
    ovf = ovf | _bit(nA > A, OVF_APPEND)

    # the recovered chunks' fits, in collection order
    arnk = jnp.cumsum(arec, axis=1).astype(I32) - 1
    acnt = jnp.sum(arec, axis=1).astype(I32)
    atab = jnp.zeros(A * H + 1, I32).at[jnp.where(arec, jnp.arange(A)[:, None] * H + arnk, A * H).reshape(-1)].set(
        jnp.broadcast_to(g, (A, H)).reshape(-1), mode="drop")[:A * H].reshape(A, H)

    def rsums(i, Fa):
        q = atab[:, i]
        return _add_point(Fa, z[q], t[q], w[q], i < acnt)

    Fa = jax.lax.fori_loop(0, jnp.max(acnt), rsums, jnp.zeros((A, 7), F64))
    at0 = jnp.where(akind == 2, _y0(Fa), y0[asrc])

    # hit-space membership of the chunk slots
    hown = jnp.full(H + 1, -1, I32).at[jnp.where(wv, walk, H)].set(own, mode="drop")[:H]
    hrnk = jnp.zeros(H + 1, I32).at[jnp.where(wv, walk, H)].set(rnk, mode="drop")[:H]
    return {"order": order, "n": n, "nrg": ch["nrg"], "ns": ch["ns"], "y0": y0, "calo": calo, "hown": hown,
            "hrnk": hrnk, "akind": akind, "asrc": asrc, "acalo": acalo, "anrg": anrg, "ans": ans, "arec": arec,
            "arnk": arnk, "at0": at0, "nA": jnp.minimum(nA, A), "assigned": assigned, "stale": stale, "n_p": n_p,
            "ovf": ovf}


def _outputs(r, nrg_in, s, KO, carry=True):
    E, K = r["order"].shape
    A, H = r["arec"].shape[1:]
    idx = jnp.arange(E, dtype=I32)
    li = jax.lax.cummax(jnp.where(r["assigned"], idx, -1), axis=0)
    prev = jnp.concatenate([jnp.full(1, -1, I32), li[:-1]])
    if not carry:                    # every event starts from nrg_in (events not consecutive in file order)
        prev = jnp.full(E, -1, I32)
    inc = jnp.where(prev >= 0, r["stale"][jnp.maximum(prev, 0)], jnp.asarray(nrg_in, I32))
    nrg_out = jnp.where(li[-1] >= 0, r["stale"][jnp.maximum(li[-1], 0)], jnp.asarray(nrg_in, I32))
    pa_ = jnp.arange(A, dtype=I32)[None, :]
    live_a = pa_ < r["nA"][:, None]
    anrg = jnp.where(r["anrg"] == -2, inc[:, None], r["anrg"])
    indet = jnp.sum(live_a & (r["akind"] == 2) & (anrg == -1) & (r["ans"] >= s["cthr"]), axis=1).astype(I32)

    # entries: the chunks in order, then the appended ones; electrons are nrg 1 with clusterThresh straw hits
    P = jnp.arange(K, dtype=I32)[None, :]
    so = r["order"]
    take = lambda a, i: jnp.take_along_axis(a, i, axis=1)
    e_nrg = jnp.concatenate([take(r["nrg"], so), anrg], 1)
    e_ns = jnp.concatenate([take(r["ns"], so), r["ans"]], 1)
    e_live = jnp.concatenate([P < r["n"][:, None], live_a], 1)
    e_t0 = jnp.concatenate([take(r["y0"], so), r["at0"]], 1)
    e_calo = jnp.concatenate([take(r["calo"], so), r["acalo"]], 1)
    e_src = jnp.concatenate([so, jnp.where(r["akind"] == 1, r["asrc"], -1)], 1)
    e_rec = jnp.concatenate([jnp.full((E, K), -1, I32), jnp.where(r["akind"] == 2, pa_, -1)], 1)
    el = e_live & (e_nrg == 1) & (e_ns >= s["cthr"])
    if s["requirecc"]:
        el_out = el & (e_calo != -1)
    else:
        el_out = el
    n_e = jnp.sum(el, axis=1).astype(I32)
    pos = jnp.cumsum(el_out, axis=1).astype(I32) - 1
    nout = jnp.sum(el_out, axis=1).astype(I32)
    rows = jnp.arange(E)[:, None]
    dst = jnp.where(el_out & (pos < KO), pos, KO)
    ent = jnp.full((E, KO + 1), 0, I32).at[rows, dst].set(jnp.arange(K + A, dtype=I32)[None, :], mode="drop")[:, :KO]
    valid = jnp.arange(KO)[None, :] < jnp.minimum(nout, KO)[:, None]
    src, rec = take(e_src, ent), take(e_rec, ent)
    mem_c = (r["hown"][:, None, :] == src[:, :, None]) & (src[:, :, None] >= 0)
    arec_k = jnp.take_along_axis(r["arec"], jnp.maximum(rec, 0)[:, :, None], axis=1)
    arnk_k = jnp.take_along_axis(r["arnk"], jnp.maximum(rec, 0)[:, :, None], axis=1)
    isrec = (rec >= 0)[:, :, None]
    members = jnp.where(isrec, arec_k, mem_c) & valid[:, :, None]
    rank = jnp.where(members, jnp.where(isrec, arnk_k, r["hrnk"][:, None, :]), -1)
    hh = jnp.broadcast_to(jnp.arange(H, dtype=I32), members.shape)
    order = jnp.full(members.shape[:2] + (H + 1,), -1, I32)
    order = order.at[jnp.arange(E)[:, None, None], jnp.arange(KO)[None, :, None], jnp.where(members, rank, H)].set(
        hh, mode="drop")[..., :H]
    ovf = r["ovf"] | _bit(nout > KO, OVF_SLOTS)
    return {"valid": valid, "nsh": jnp.where(valid, take(e_ns, ent) & 0xFFFF, 0),
            "t0": jnp.where(valid, take(e_t0, ent), 0.0), "calo": jnp.where(valid, take(e_calo, ent), -1),
            "members": members, "rank": rank, "order": order, "n_e": n_e, "n_p": r["n_p"], "overflow": ovf,
            "indeterminate": indet, "nrg_out": nrg_out}


def clusters(ph, cl, s, nrg_in=-1, K=K_SLOTS, KC=K_CHUNKS, A=A_APPEND, carry=True):
    """Panel hits ph [E, H] (valid, time, timevar, z, ncombo, plane, flag; stock collection order) and calo clusters
    cl [E, C] (valid, energy, size, time; stock order) of consecutive events in file order -> the electron time
    clusters in K slots: valid [E, K], nsh, t0 (y0 of the cluster fit), calo (cluster index or -1), members
    [E, K, H], rank [E, K, H] (position of the hit in the cluster's stock hit list, -1 if not a member), order
    [E, K, H] (the member hit indices in stock order, then -1); n_e [E] (IntensityInfoTimeCluster electron, the
    number of electron clusters), n_p [E] (proton), overflow [E] (bits in the module docstring), indeterminate [E]
    (recovered chunks whose stale energy selection is unknown) and nrg_out (the stale value after the last event).
    `nrg_in` is the stale value entering the first event (-1: unknown). With carry=False every event starts from
    `nrg_in` (for blocks whose events are not consecutive in file order, e.g. hit-sorted): a recovered chunk that
    would need an earlier event's value is then flagged in `indeterminate` instead."""
    ft = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    with strict(ph["plane"]):
        h = {k: ph[k] for k in ("valid", "time", "timevar", "z", "ncombo", "plane", "flag")}
        c = {"valid": cl["valid"], "energy": _f(cl["energy"]), "size": jnp.asarray(cl["size"]).astype(I32),
             "time": _f(cl["time"])}
        r = jax.vmap(lambda hh, cc: _event(hh, cc, s, KC, A))(h, c)
        out = _outputs(r, nrg_in, s, K, carry)
    conv = lambda v: v.astype(ft) if jnp.issubdtype(v.dtype, jnp.floating) else \
        (v.astype(I32) if v.dtype == I64 else v)
    return {k: conv(v) for k, v in out.items()}
