"""Calorimeter-acceptance proxy of cesob on the device (JAX): cesob.calo_accept as one jitted loop with lane compaction.

Same geometry, physics and step logic as beamfastsim.cesob.calo_accept (the NumPy reference):

- vacuum: helix steps in the field maps (field at the step midpoint), h_far until `margin` before the first disk,
  then h_near; the path inside crystals (Offline crystal map, crystal z range of each disk) is sampled at `sub`
  points per step; a lane ends when the path reaches path_min, or it leaves through z_back of the last disk, z_lo,
  r_wall, or s_max;
- wall (cesob.CaloWall): a step that crosses the Al ring face (outward, inside an Al ring's z), the carbon-fibre inner
  face (outward, outside the Al rings) or the carbon-fibre outer face (inward) inside a case stops at the crossing
  (radius bisection, for a gathered sub-batch of the candidate lanes; the others wait one iteration, unchanged); the
  lane then crosses the ring(s) radially with the option-4 slab physics (mu2efastsim.eslab_jax.lane_step, Al and
  carbon fibre as two layer types of one table; entry-point field plus the shell-curvature term): outward Al then
  carbon fibre, outward carbon fibre, inward carbon fibre then Al where the exit lies in an Al ring's z. A lane that
  comes out on the far side has crossed, else it is reflected back (or stopped).

Statistically equivalent to the NumPy proxy, not identical per seed (the wall draws differ); without the wall it is
deterministic and equal to it up to rounding. float64 or float32. Mu2e frame, mm, MeV.
"""
import time
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from mu2efastsim import eioni
from mu2efastsim import eslab_jax as SJ
from mu2efastsim import materials as Mt

from . import cechain as C
from . import cechain_jax as CJ

ME = eioni.ME
AL, CF = 0, 1                     # layer types of the wall table
_BUILT = {}


def _mom(T):
    return jnp.sqrt(T * (T + 2 * ME))


def _build(calo, wall, dt, h_far, h_near, sub, margin, r_wall, s_max, goal, msc, unroll, const_field):
    key = (id(calo), id(wall), np.dtype(dt).name, h_far, h_near, sub, margin, r_wall, s_max, goal, msc, unroll,
           None if const_field is None else tuple(np.asarray(const_field, float)))
    if key in _BUILT:
        return _BUILT[key]
    q = -1.0
    ax = float(calo.axis_x)
    zf = jnp.asarray(calo.z_front, dt)
    zb = jnp.asarray(calo.z_back, dt)
    zf_min, zb_max = float(min(calo.z_front)), float(max(calo.z_back))
    grid = jnp.asarray(calo.grid)
    nr, nc = (calo.grid.shape[0] - 1) // 2, (calo.grid.shape[1] - 1) // 2
    cell, half = float(calo.cell), float(calo.half)
    if wall is not None:
        lt = SJ.lane_tables([(Mt.ALUMINIUM, float(wall.cf_rin - wall.al_rin), *C.CUTS[Mt.ALUMINIUM.name]),
                             (Mt.CARBON_FIBER, float(wall.cf_rout - wall.cf_rin), *C.CUTS[Mt.CARBON_FIBER.name])],
                            msc=msc)
        Ltype = jnp.asarray(lt.length, dt)
        case_lo, case_hi = [float(z) for z in wall.case_lo], [float(z) for z in wall.case_hi]
        al_c = [c for a, b in zip(case_lo, case_hi) for c in (a + wall.al_hl, 0.5 * (a + b), b - wall.al_hl)]

    def radial(x):
        return jnp.hypot(x[:, 0] - ax, x[:, 1])

    def field(maps, x):
        return CJ._field(maps, const_field, x)

    def in_case(z):
        out = jnp.zeros(z.shape, bool)
        for a, b in zip(case_lo, case_hi):
            out = out | ((z >= a) & (z <= b))
        return out

    def in_al(z):
        out = jnp.zeros(z.shape, bool)
        for c in al_c:
            out = out | (jnp.abs(z - c) <= wall.al_hl)
        return out

    def hit(xk):
        dx, dy = xk[:, 0] - ax, xk[:, 1]
        zin = jnp.any((xk[:, 2, None] >= zf) & (xk[:, 2, None] <= zb), axis=1)
        if goal < 1e-6:                                   # annulus form
            r = jnp.hypot(dx, dy)
            return zin & (r >= calo.r_in) & (r <= calo.r_out)
        n = jnp.rint(dy / cell).astype(jnp.int32)
        j = n + 2 * jnp.rint((2 * dx / cell - n) / 2).astype(jnp.int32)
        ok = (jnp.abs(n) <= nr) & (jnp.abs(j) <= nc)
        v = grid[jnp.clip(n + nr, 0, 2 * nr), jnp.clip(j + nc, 0, 2 * nc)] & ok
        return zin & v & (jnp.abs(dx - j * cell / 2) <= half) & (jnp.abs(dy - n * cell) <= half)

    def bisect(x, u, p, B, s_hi, target):
        def f(s):
            return radial(SJ.helix(x, u, p, q, B, s)[0]) - target

        def it(_, c):
            lo, flo, hi = c
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            same = jnp.sign(fm) == jnp.sign(flo)
            return jnp.where(same, mid, lo), jnp.where(same, fm, flo), jnp.where(same, hi, mid)
        lo = jnp.zeros_like(s_hi)
        return lax.fori_loop(0, 40, it, (lo, f(lo), s_hi), **SJ._unroll(unroll))[2]

    def gather(mask, m):
        n = mask.shape[0]
        idx = jnp.nonzero(mask, size=m, fill_value=n)[0]
        ok = idx < n
        chosen = jnp.zeros(n, bool).at[idx].set(True, mode="drop")
        return idx, jnp.minimum(idx, n - 1), ok, chosen

    def enter(maps, x, u, T, li, outward):
        """Radial slab frame at x (cechain._frame_radial), entry radius, local direction and field (entry-point field
        plus the shell-curvature term) of lanes entering wall layer li."""
        n = x.shape[0]
        zero, one = jnp.zeros(n, dt), jnp.ones(n, dt)
        rv = jnp.stack([x[:, 0] - ax, x[:, 1], zero], axis=1)
        rn = rv / jnp.maximum(jnp.linalg.norm(rv, axis=1, keepdims=True), 1e-30)
        nn = jnp.where(outward[:, None], rn, -rn)
        e1 = jnp.stack([zero, zero, one], axis=1)
        e2 = jnp.cross(nn, e1)
        r = radial(x)
        B = field(maps, x)
        B = B.at[:, 2].add(jnp.where(outward, 1.0, -1.0) * (u * e2).sum(1) * _mom(T) /
                           (SJ.K_FIELD * jnp.maximum(r, 1e-30)))

        def loc(v):
            return jnp.stack([(v * e1).sum(1), (v * e2).sum(1), (v * nn).sum(1)], axis=1)
        return e1, e2, nn, r, loc(u), loc(B)

    def init(x, u, T):
        n = x.shape[0]
        st = dict(x=x, u=u, T=T, path=jnp.zeros(n, dt), s=jnp.zeros(n, dt), live=jnp.ones(n, bool),
                  nvac=jnp.zeros(n, jnp.int32))
        if wall is not None:
            z3 = jnp.zeros((n, 3), dt)
            st.update(inl=jnp.zeros(n, bool), li=jnp.zeros(n, jnp.int32), kind=jnp.zeros(n, jnp.int32),
                      stage=jnp.zeros(n, jnp.int32), outw=jnp.zeros(n, bool), rent=jnp.zeros(n, dt), e1=z3, e2=z3,
                      en=z3, xent=z3, Bl=z3, Tin=T, slab=SJ.lane_state(T, z3.at[:, 2].set(1.0), jnp.zeros(n, dt),
                                                                       Ltype[jnp.zeros(n, jnp.int32)]),
                      nal=jnp.zeros(n, jnp.int32), ncf=jnp.zeros(n, jnp.int32), nrefl=jnp.zeros(n, jnp.int32),
                      nstop=jnp.zeros(n, jnp.int32), nslab=jnp.zeros(n, jnp.int32), nstall=jnp.zeros(n, jnp.int32))
            st["slab"]["exited"] = jnp.ones(n, bool)
        return st

    def set_layer(st, mask, idx, ic, maps, x, u, T, li, outward, kind, stage):
        """Lanes idx (gathered, valid where mask) enter wall layer li at x."""
        n = st["x"].shape[0]
        e1, e2, nn, r, dloc, Bl = enter(maps, x, u, T, li, outward)
        tgt = jnp.where(mask, idx, n)
        put = lambda old, v: old.at[tgt].set(v, mode="drop")
        new = SJ.lane_state(T, dloc, jnp.zeros_like(T), Ltype[li])
        sl = st["slab"]
        sl = {k: (v if k == "it" else v.at[tgt].set(new[k], mode="drop")) for k, v in sl.items()}
        cnt_al = put(st["nal"], st["nal"][ic] + (li == AL))
        cnt_cf = put(st["ncf"], st["ncf"][ic] + (li == CF))
        return dict(st, slab=sl, inl=put(st["inl"], jnp.ones_like(mask)), li=put(st["li"], li),
                    kind=put(st["kind"], kind), stage=put(st["stage"], stage), outw=put(st["outw"], outward),
                    rent=put(st["rent"], r), e1=put(st["e1"], e1), e2=put(st["e2"], e2), en=put(st["en"], nn),
                    xent=put(st["xent"], x), Bl=put(st["Bl"], Bl), Tin=put(st["Tin"], T), nal=cnt_al, ncf=cnt_cf)

    def slab(maps, key, st, m_slab):
        """One slab step for up to m_slab lanes in a wall layer; finished crossings go on to the next layer or back
        to vacuum."""
        n = st["x"].shape[0]
        act = st["live"] & st["inl"]
        idx, ic, ok, chosen = gather(act, m_slab)
        sl = st["slab"]
        sub_ = {k: (v if k == "it" else v[ic]) for k, v in sl.items()}
        sub_["exited"] = sub_["exited"] | ~ok
        sub_ = lax.cond(jnp.any(act), lambda s: SJ.lane_step(key, lt, st["li"][ic], s, st["Bl"][ic], msc=msc,
                                                             unroll=unroll),
                        lambda s: {**s, "it": s["it"] + 1}, sub_)
        done = ok & (sub_["stopped"] | sub_["exited"])
        e1, e2, en, xe = st["e1"][ic], st["e2"][ic], st["en"][ic], st["xent"][ic]
        lp, dn = sub_["x"], sub_["d"]
        xg = xe + lp[:, :1] * e1 + lp[:, 1:2] * e2 + lp[:, 2:3] * en
        ug = dn[:, :1] * e1 + dn[:, 1:2] * e2 + dn[:, 2:3] * en
        ug = ug / jnp.linalg.norm(ug, axis=1, keepdims=True)
        T1 = sub_["T"]
        stop = sub_["stopped"] | (T1 <= 0)
        li, kind, stage, outw, rent = st["li"][ic], st["kind"][ic], st["stage"][ic], st["outw"][ic], st["rent"][ic]
        r1 = radial(xg)
        th = Ltype[li]
        through = jnp.where(outw, r1 > rent + 0.5 * th, r1 < rent - 0.5 * th) & ~stop
        nxt = done & through & (stage == 0) & ((kind == 1) | ((kind == 3) & in_al(xg[:, 2])))
        tgt = jnp.where(ok, idx, n)
        put = lambda old, v: old.at[tgt].set(v, mode="drop")
        sl = {k: (sub_[k] if k == "it" else sl[k].at[tgt].set(sub_[k], mode="drop")) for k in sl}
        x = put(st["x"], jnp.where(done[:, None], xg, st["x"][ic]))
        u = put(st["u"], jnp.where(done[:, None], ug, st["u"][ic]))
        T = put(st["T"], jnp.where(done, T1, st["T"][ic]))
        st = dict(st, slab=sl, x=x, u=u, T=T, live=put(st["live"], st["live"][ic] & ~(done & stop)),
                  inl=put(st["inl"], st["inl"][ic] & ~done),
                  nrefl=put(st["nrefl"], st["nrefl"][ic] + (done & ~through & ~stop)),
                  nstop=put(st["nstop"], st["nstop"][ic] + (done & stop)),
                  nslab=put(st["nslab"], st["nslab"][ic] + ok))
        li2 = jnp.where(kind == 1, CF, AL).astype(jnp.int32)
        return set_layer(st, nxt, idx, ic, maps, xg, ug, T1, li2, outw, kind, jnp.ones_like(stage))

    def vacuum(maps, st, m_cand, z_lo):
        x, u, T = st["x"], st["u"], st["T"]
        n = x.shape[0]
        act = st["live"] & (~st["inl"] if wall is not None else True)
        p = _mom(T)
        h = jnp.where(x[:, 2] > zf_min - margin, h_near, h_far).astype(dt)
        B0 = field(maps, x)
        Bm = field(maps, SJ.helix(x, u, p, q, B0, 0.5 * h)[0])
        s_ev = h
        stall = jnp.zeros(n, bool)
        if wall is not None:
            xe = SJ.helix(x, u, p, q, Bm, h)[0]
            r0, r1 = radial(x), radial(xe)
            s1 = (r0 < wall.al_rin) & (r1 >= wall.al_rin)
            s2 = (r0 < wall.cf_rin) & (r1 >= wall.cf_rin)
            s3 = (r0 > wall.cf_rout) & (r1 <= wall.cf_rout)
            cand = act & (s1 | s2 | s3)
            idx, ic, ok, chosen = gather(cand, m_cand)
            stall = cand & ~chosen
            g = lambda a: a[ic]
            xc, uc, pc, Bc, hc = g(x), g(u), g(p), g(Bm), g(h)
            sc, kc = hc, jnp.zeros(idx.shape[0], jnp.int32)
            for kd, rt, sel in ((1, wall.al_rin, g(s1) & ok), (2, wall.cf_rin, g(s2) & ok), (3, wall.cf_rout, g(s3) & ok)):
                sv = bisect(xc, uc, pc, Bc, jnp.where(sel, hc, 0.0), rt)
                zh = SJ.helix(xc, uc, pc, q, Bc, sv)[0][:, 2]
                okz = in_case(zh) & (in_al(zh) if kd == 1 else ~in_al(zh) if kd == 2 else True)
                better = sel & okz & (sv < sc)
                sc, kc = jnp.where(better, sv, sc), jnp.where(better, kd, kc)
            tgt = jnp.where(ok, idx, n)
            s_ev = s_ev.at[tgt].set(sc, mode="drop")
            kind = jnp.zeros(n, jnp.int32).at[tgt].set(kc, mode="drop")
        moved = act & ~stall
        s_ev = jnp.where(moved, s_ev, 0.0)
        path = st["path"]
        for k in range(1, sub + 1):
            xk = SJ.helix(x, u, p, q, Bm, s_ev * (k / sub))[0]
            path = path + jnp.where(moved & hit(xk), s_ev / sub, 0.0)
        xn, un = SJ.helix(x, u, p, q, Bm, s_ev)
        xn = jnp.where(moved[:, None], xn, x)
        un = jnp.where(moved[:, None], un, u)
        s = st["s"] + s_ev
        rn = radial(xn)
        done = moved & ((path >= goal) | (xn[:, 2] > zb_max) | (xn[:, 2] < z_lo) | (rn > r_wall) | (s > s_max))
        st = dict(st, x=xn, u=un, path=path, s=s, live=st["live"] & ~done, nvac=st["nvac"] + moved)
        if wall is not None:
            st["nstall"] = st["nstall"] + stall
            ent = g(moved & ~done & (kind > 0)) & ok
            kg = kc
            st = set_layer(st, ent, idx, ic, maps, xn[ic], un[ic], g(T), jnp.where(kg == 1, AL, CF).astype(jnp.int32),
                           kg != 3, kg, jnp.zeros_like(kg))
        return st

    def step(maps, key, st, m_cand, m_slab, z_lo):
        if wall is not None:
            st = slab(maps, key, st, m_slab)
        return vacuum(maps, st, m_cand, z_lo)

    @partial(jax.jit, static_argnames=("m_cand", "m_slab"))
    def run(maps, key, st, n_it, m_cand, m_slab, z_lo):
        def cond(c):
            i, s = c
            return (i < n_it) & jnp.any(s["live"])

        def body(c):
            i, s = c
            return i + 1, step(maps, key, s, m_cand, m_slab, z_lo)
        return lax.while_loop(cond, body, (jnp.int32(0), st))

    out = (jax.jit(init), run)
    _BUILT[key] = out
    return out


OUT_FIELDS = ("path", "s", "nvac")
WALL_FIELDS = ("nal", "ncf", "nrefl", "nstop", "nslab", "nstall")


def calo_accept(key, field, calo, x, u, T, h_far=50.0, h_near=10.0, sub=5, margin=60.0, r_wall=950.0, s_max=30000.0,
                path_min=None, annulus=False, wall=None, msc=True, dtype=np.float32, chunk=256, min_bucket=1024,
                cand_frac=0.125, slab_frac=0.25, shrink=32, unroll=False, info=None, stats=None):
    """cesob.calo_accept on the device: True where the crystal path reaches path_min (annulus: the helix enters
    the crystal annulus). key: jax.random key (wall draws); field: a constant 3-vector [T] or device FieldMaps
    (cechain_jax.device_maps). chunk: iterations per launch; the live lanes are then compacted into a power-of-two
    bucket once they fit 1/shrink of it. info (dict): wall crossing counts as cesob.calo_accept; stats (dict):
    iterations, rounds and a per-launch timeline."""
    from . import cesob
    dt = np.dtype(dtype)
    path_min = cesob.CSI_PATH_MIN if path_min is None else path_min
    goal = 1e-9 if annulus else float(path_min)
    maps, const = (tuple(field), None) if isinstance(field, (tuple, list)) and field and \
        isinstance(field[0], CJ.FM.FieldMap) else ((), np.asarray(field, float))
    init, run = _build(calo, wall, dt, float(h_far), float(h_near), int(sub), float(margin), float(r_wall),
                       float(s_max), goal, bool(msc), bool(unroll), const)
    x, u, T = np.asarray(x, float), np.asarray(u, float), np.asarray(T, float)
    n = x.shape[0]
    fields = OUT_FIELDS + (WALL_FIELDS if wall is not None else ())
    final = {f: np.zeros(n, np.int64 if f.startswith("n") else np.float64) for f in fields}
    if n == 0:
        if info is not None:
            info.update(al=0, cf=0, reflected=0, stopped=0)
        return np.zeros(0, bool)
    z_lo = jnp.asarray(x[:, 2].min() - 100.0, dt)
    pow2 = lambda k: max(min_bucket, 1 << int(np.ceil(np.log2(max(k, 1)))))
    m = pow2(n)
    j = np.concatenate([np.arange(n), np.zeros(m - n, int)])
    st = init(jnp.asarray(x[j], dt), jnp.asarray(u[j], dt), jnp.asarray(T[j], dt))
    st["live"] = st["live"].at[n:].set(False)                      # padding lanes are inert
    idx = np.concatenate([np.arange(n), np.full(m - n, -1)])
    rounds = iters = 0
    timeline = []
    while True:
        nb = idx.size
        tr = time.time()
        i, st = run(maps, jax.random.fold_in(key, rounds), st, jnp.int32(chunk),
                    max(1, min(nb, int(np.ceil(cand_frac * nb)))), max(1, min(nb, int(np.ceil(slab_frac * nb)))),
                    z_lo)
        i = int(i)
        timeline.append((nb, i, round(time.time() - tr, 3)))
        rounds += 1
        iters += i
        live = np.asarray(st["live"])
        fin = np.flatnonzero((idx >= 0) & ~live)
        if fin.size:
            fd = jnp.asarray(fin)
            for f in fields:
                final[f][idx[fin]] = np.asarray(st[f][fd])
            idx[fin] = -1
        keep = np.flatnonzero(live & (idx >= 0))
        if keep.size == 0:
            break
        mb = pow2(keep.size)
        if mb * shrink <= idx.size:
            sel = np.concatenate([keep, np.full(mb - keep.size, keep[0])])
            sd = jnp.asarray(sel)
            st = jax.tree_util.tree_map(lambda a: a[sd] if getattr(a, "ndim", 0) > 0 else a, st)
            st["live"] = st["live"].at[keep.size:].set(False)
            idx = np.concatenate([idx[keep], np.full(mb - keep.size, -1)])
    if info is not None and wall is not None:
        info.update(al=int(final["nal"].sum()), cf=int(final["ncf"].sum()), reflected=int(final["nrefl"].sum()),
                    stopped=int(final["nstop"].sum()))
    if stats is not None:
        stats.update(iterations=iters, rounds=rounds, lanes=int(pow2(n)), timeline=timeline,
                     nvac_mean=float(final["nvac"].mean()), nvac_max=int(final["nvac"].max()))
    return final["path"] >= goal
