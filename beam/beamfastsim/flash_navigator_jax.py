"""Standalone opt-in charged navigator for brems births; no CE transport/refill.

The charged navigator uses target_geometry_jax helpers shared with CE while
retaining its own scheduling and records. Charge-aware tables use core lane_step_photons.
Planar charged foils have no lateral escape; radial shells are locally flat,
layer fields frozen and vacuum fields midpoint sampled. This is a source
approximation, not a full S1 implementation or photon exit-flux calculation.
"""
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from mu2efastsim import eslab_jax as SJ

from . import cechain as C
from . import target_geometry_jax as TG

ME = TG.ME
NCOUNT = TG.NCOUNT


layer_types = TG.layer_types
device_maps = TG.device_maps
_field = TG.field
_mom = TG.momentum
_beta = TG.beta
_dot = TG.dot
gather = TG.gather


def _slab_elapsed(before, after):
    """Step time from continuous-loss endpoints, before discrete end-step recoil."""
    pre_hard = jnp.maximum(before["T"] - (after["cont"] - before["cont"]), 0.)
    beta = jnp.maximum(.5 * (_beta(before["T"]) + _beta(pre_hard)), 1e-6)
    return (after["path"] - before["path"]) / (C.C_MM_NS * beta)


_BUILT = {}


def _build(geom, dt, h, s_max, msc, layer_field, shell_curvature, unroll, const_field,
           positron=False, photon_capacity=128, skip_empty=False, vacuum_fast=False, vacuum_bucket=0):
    key = (id(geom), np.dtype(dt).name, h, s_max, msc, layer_field, shell_curvature, unroll,
           None if const_field is None else tuple(np.asarray(const_field, float)), bool(positron), photon_capacity,
           bool(skip_empty), bool(vacuum_fast), int(vacuum_bucket))
    if key in _BUILT:
        return _BUILT[key]
    b = geom.beam
    specs, Lf, t_ipa, t_opa, t_ring, counter = layer_types(geom)
    lt = SJ.lane_tables(specs, positron=positron, msc=msc)
    Ltype = jnp.asarray(lt.length, dt)
    counter_d = jnp.asarray(counter)
    geometry = TG.build(geom, dt, Lf, charge=1.0 if positron else -1.0, const_field=const_field,
                        layer_field=layer_field, shell_curvature=shell_curvature, unroll=unroll)
    ftype, face_up, face_dn = geometry.ftype, geometry.face_up, geometry.face_dn
    frin, frout, nf = geometry.frin, geometry.frout, geometry.nf
    eps = 1e-7 if np.dtype(dt) == np.float64 else 2e-3       # foil-face search offset (a few ulp of z in float32)
    ztsda = float(b.tsda_z + b.tsda_hl)
    zr = [(float(z), float(z + 2 * geom.ring_hl)) for z in (b.ipa_z0, b.ipa_z1)]
    q = 1.0 if positron else -1.0
    cms = C.C_MM_NS

    def W(c, a, b_):
        return jnp.where(c, a, b_).astype(dt)

    radial, opa_rin, bisect, frames = geometry.radial, geometry.opa_rin, geometry.bisect, geometry.frames

    def init(maps, x, u, T, t, k0, zl):
        n = x.shape[0]
        born = k0 >= 0
        ty = jnp.where(born, ftype[jnp.clip(k0, 0, nf - 1)], 0)
        e1, e2, nn, entry, dloc, Bl = frames(maps, x, u, T, ty, k0)
        sl = SJ.lane_state(T, dloc, jnp.where(born, zl, 0.0), Ltype[ty], positron=positron)
        sl["exited"] = sl["exited"] | ~born
        st = dict(x=x, u=u, T=T, t=t, s=jnp.zeros(n, dt), fate=jnp.zeros(n, jnp.int32), inl=born, lty=ty,
                  e1=e1, e2=e2, en=nn, xent=entry, Tin=T, Bl=Bl, cnt=jnp.zeros((n, NCOUNT), jnp.int32), slab=sl,
                  nvac=jnp.zeros(n, jnp.int32), nslab=jnp.zeros(n, jnp.int32), nstall=jnp.zeros(n, jnp.int32))
        st.update(records=dict(energy=jnp.zeros((n, photon_capacity), dt),
                               position=jnp.zeros((n, photon_capacity, 3), dt),
                               direction=jnp.zeros((n, photon_capacity, 3), dt),
                               path=jnp.zeros((n, photon_capacity), dt),
                               time=jnp.zeros((n, photon_capacity), dt),
                               valid=jnp.zeros((n, photon_capacity), bool)),
                  count=jnp.zeros(n, jnp.int32), nadapt=jnp.zeros(n, jnp.int32),
                  slab_time=jnp.zeros(n, dt))
        st.update({name: sl[name] for name in ("nph", "eph", "nde", "ede", "nan", "ean", "cont")})
        st["fate"] = jnp.where(sl["stopped"], C.STOPPED, st["fate"])
        st["T"] = jnp.where(sl["stopped"], 0.0, T)
        st["xent"] = jnp.where(born[:, None], entry, x)
        return st

    def append_births(st, emitted, idx, ic, ok):
        """One brems slot per emitting step; invalid/overflow rows never overwrite."""
        n = st["T"].shape[0]
        valid = emitted.valid & ok
        slots = st["count"][ic]
        rows = jnp.where(valid & (slots < photon_capacity), idx, n)
        lp, dn = emitted.position, emitted.direction
        e1, e2, en = (st[k][ic] for k in ("e1", "e2", "en"))
        position = st["xent"][ic] + lp[:, :1] * e1 + lp[:, 1:2] * e2 + lp[:, 2:3] * en
        direction = dn[:, :1] * e1 + dn[:, 1:2] * e2 + dn[:, 2:3] * en
        values = dict(energy=emitted.energy, position=position, direction=direction,
                      path=st["s"][ic] + emitted.path,
                      time=st["t"][ic] + st["slab_time"][ic], valid=valid)
        records = {name: old.at[rows, jnp.minimum(slots, photon_capacity - 1)].set(values[name], mode="drop")
                   for name, old in st["records"].items()}
        count = st["count"].at[idx].add(valid.astype(jnp.int32), mode="drop")
        return dict(st, records=records, count=count)

    def crossings(x, u, p, Bm, h_, z0, z1, r0, r1, sel_p, sel_f, sel_t, face, kc, sel_i, sel_o, out_i, out_o,
                  sel_r, out_r):
        """First surface crossing in the step for candidate lanes: (path to it, kind) with kind 0 for none."""
        m = x.shape[0]
        zp = jnp.where(sel_p, geom.z_plane, jnp.where(sel_f, face, ztsda)).astype(dt)
        anyz = sel_p | sel_f | sel_t
        sz = SJ.to_plane(x, u, p, q, Bm, zp, jnp.where(anyz, h_, 0.0), unroll)
        sz = jnp.where(anyz, sz, jnp.inf)
        rz = radial(SJ.helix(x, u, p, q, Bm, jnp.where(jnp.isfinite(sz), sz, 0.0))[0])
        s_ev, kind = h_, jnp.zeros(m, jnp.int32)

        def take(sel, sv, kd, s_ev, kind):
            better = sel & jnp.isfinite(sv) & (sv <= s_ev)
            return jnp.where(better, sv, s_ev), jnp.where(better, kd, kind)
        s_ev, kind = take(sel_p, sz, 1, s_ev, kind)
        s_ev, kind = take(sel_f & ~sel_p & (rz >= frin[kc]) & (rz <= frout[kc]), sz, 2, s_ev, kind)
        ri, ro = b.ipa_r, b.ipa_r + b.ipa_t
        anys = sel_i | sel_o
        r_ipa = W(out_i, ri, ro)
        off_o = W(out_o, 0.0, geom.opa_t)
        ss = bisect(x, u, p, Bm, jnp.where(anys, h_, 0.0), lambda z: W(sel_i, r_ipa, opa_rin(z) + off_o))
        zs = SJ.helix(x, u, p, q, Bm, ss)[0][:, 2]
        s_ev, kind = take(sel_i & (zs >= b.ipa_z0) & (zs <= b.ipa_z1), ss, 3, s_ev, kind)
        if t_ring >= 0:
            rri, rro = b.ipa_r - geom.ring_t, b.ipa_r
            sr = bisect(x, u, p, Bm, jnp.where(sel_r, h_, 0.0), lambda z: W(out_r, rri, rro))
            zrr = SJ.helix(x, u, p, q, Bm, sr)[0][:, 2]
            okr = jnp.zeros(m, bool)
            for za, zb in zr:
                okr = okr | ((zrr >= za) & (zrr <= zb))
            s_ev, kind = take(sel_r & okr, sr, 8, s_ev, kind)
        s_ev, kind = take(sel_o & (zs >= b.opa_z0) & (zs <= b.opa_z1), ss, 4, s_ev, kind)
        s_ev, kind = take(sel_t & ~sel_p & ~sel_f & (rz >= b.tsda_rin) & (rz <= b.tsda_rout), sz, 5, s_ev, kind)
        return s_ev, kind

    def move_vacuum(st, Bm, s_ev, kind, stall, z1, r1):
        x, u, T = st["x"], st["u"], st["T"]
        vac = (st["fate"] == C.LIVE) & ~st["inl"]
        free = (kind == 0) & ~stall
        kind = jnp.where(free & (z1 < geom.z_up), 6, kind)
        kind = jnp.where(free & (kind == 0) & ((r1 > geom.r_wall) | ((z1 < geom.z_ds) & (r1 > geom.r_ts))), 7, kind)
        kind = jnp.where(free & (kind == 0) & (z1 >= geom.coll5_z0) & (z1 <= geom.coll5_z1) &
                         (r1 > geom.coll5_rin), 5, kind)
        moved = vac & ~stall
        s_ev = jnp.where(moved, s_ev, 0.0)
        kind = jnp.where(vac, kind, 0)
        xn, un = SJ.helix(x, u, _mom(T), q, Bm, s_ev)
        xn = jnp.where(moved[:, None], xn, x)
        un = jnp.where(moved[:, None], un, u)
        new = dict(st)
        new["x"], new["u"] = xn, un
        new["s"] = st["s"] + s_ev
        new["t"] = st["t"] + s_ev / (cms * jnp.maximum(_beta(T), 1e-6))
        new["nvac"] = st["nvac"] + moved
        new["nstall"] = st["nstall"] + stall
        fate = st["fate"]
        for kd, fv in ((1, C.PLANE), (5, C.ABSORBED), (6, C.UPSTREAM), (7, C.WALL)):
            fate = jnp.where(kind == kd, fv, fate)
        enter = (kind == 2) | (kind == 3) | (kind == 4) | (kind == 8)
        new["fate"] = jnp.where(moved & ~enter & (fate == C.LIVE) & (new["s"] > s_max), C.LOOPER, fate)
        return new, enter, kind

    def vacuum(maps, st, m_cand, free_only=False):
        """One helix step for live lanes in vacuum. Lanes whose step may cross a surface are solved in a gathered
        sub-batch of m_cand lanes (the others of them wait one iteration, unchanged); lanes that reach a layer get
        their slab frame. Returns the new state and the entering mask."""
        x, u, T = st["x"], st["u"], st["T"]
        vac = (st["fate"] == C.LIVE) & ~st["inl"]
        n = x.shape[0]
        p = _mom(T)
        B0 = _field(maps, const_field, x)
        # Limit each optional-source vacuum turn to 0.05 rad; field variation remains a midpoint approximation.
        hh = jnp.minimum(h, 0.05 * p / jnp.maximum(SJ.K_FIELD * jnp.linalg.norm(B0, axis=1), 1e-30))
        Bm = _field(maps, const_field, SJ.helix(x, u, p, q, B0, 0.5 * hh)[0])
        hh = jnp.minimum(hh, 0.05 * p / jnp.maximum(SJ.K_FIELD * jnp.linalg.norm(Bm, axis=1), 1e-30))
        xe = SJ.helix(x, u, p, q, Bm, hh)[0]
        z0, z1 = x[:, 2], xe[:, 2]
        zlo, zhi = jnp.minimum(z0, z1), jnp.maximum(z0, z1)
        r0, r1 = radial(x), radial(xe)
        sel_p = (z0 < geom.z_plane) & (z1 >= geom.z_plane)
        up = z1 > z0
        k = jnp.where(up, jnp.searchsorted(face_up, z0 + eps, side="left"),
                      jnp.searchsorted(face_dn, z0 - eps, side="right") - 1)
        kc = jnp.clip(k, 0, nf - 1).astype(jnp.int32)
        face = jnp.where(up, face_up[kc], face_dn[kc])
        sel_f = (k >= 0) & (k < nf) & jnp.where(up, face <= z1, face >= z1)
        sel_t = (b.tsda_hl > 0) & (z0 > ztsda) & (z1 <= ztsda)
        ri, ro = b.ipa_r, b.ipa_r + b.ipa_t
        out_i, inn_i = (r0 < ri) & (r1 >= ri), (r0 > ro) & (r1 <= ro)
        sel_i = (out_i | inn_i) & (zhi >= b.ipa_z0) & (zlo <= b.ipa_z1)
        rc0, rc1 = opa_rin(z0), opa_rin(z1)
        out_o = (r0 < rc0) & (r1 >= rc1)
        sel_o = (out_o | ((r0 > rc0 + geom.opa_t) & (r1 <= rc1 + geom.opa_t))) & (zhi >= b.opa_z0) & \
            (zlo <= b.opa_z1) & ~sel_i
        if t_ring >= 0:
            rri, rro = b.ipa_r - geom.ring_t, b.ipa_r
            out_r = (r0 < rri) & (r1 >= rri)
            near = jnp.zeros(n, bool)
            for za, zb in zr:
                near = near | ((zhi >= za) & (zlo <= zb))
            sel_r = (out_r | ((r0 > rro) & (r1 <= rro))) & near
        else:
            out_r = sel_r = jnp.zeros(n, bool)
        cand = vac & (sel_p | sel_f | sel_t | sel_i | sel_o | sel_r)
        if free_only:
            new, _, _ = move_vacuum(st, Bm, hh, jnp.zeros(n, jnp.int32), jnp.zeros(n, bool), z1, r1)
            new["nadapt"] = st["nadapt"] + (vac & (hh < h))
            return new, jnp.any(cand)
        idx, ic, ok, chosen = gather(cand, m_cand)
        stall = cand & ~chosen
        g = lambda a: a[ic]
        def solve_crossings(_):
            return crossings(g(x), g(u), g(p), g(Bm), g(hh), g(z0), g(z1), g(r0), g(r1), g(sel_p) & ok,
                             g(sel_f) & ok, g(sel_t) & ok, g(face), g(kc), g(sel_i) & ok, g(sel_o) & ok, g(out_i),
                             g(out_o), g(sel_r) & ok, g(out_r))

        # Empty candidates have idx == n, so the same-size placeholders are dropped by scatter.
        if skip_empty:
            s_c, k_c = lax.cond(jnp.any(cand), solve_crossings,
                                lambda _: (g(hh), jnp.zeros(m_cand, jnp.int32)), operand=None)
        else:
            s_c, k_c = solve_crossings(None)
        s_ev = hh.at[idx].set(s_c, mode="drop")
        kind = jnp.zeros(n, jnp.int32).at[idx].set(k_c, mode="drop")
        free = (kind == 0) & ~stall
        kind = jnp.where(free & (z1 < geom.z_up), 6, kind)
        kind = jnp.where(free & (kind == 0) & ((r1 > geom.r_wall) | ((z1 < geom.z_ds) & (r1 > geom.r_ts))), 7, kind)
        kind = jnp.where(free & (kind == 0) & (z1 >= geom.coll5_z0) & (z1 <= geom.coll5_z1) &
                         (r1 > geom.coll5_rin), 5, kind)
        moved = vac & ~stall
        s_ev = jnp.where(moved, s_ev, 0.0)
        kind = jnp.where(vac, kind, 0)
        xn, un = SJ.helix(x, u, p, q, Bm, s_ev)
        xn = jnp.where(moved[:, None], xn, x)
        un = jnp.where(moved[:, None], un, u)
        new = dict(st)
        new["x"], new["u"] = xn, un
        new["s"] = st["s"] + s_ev
        new["t"] = st["t"] + s_ev / (cms * jnp.maximum(_beta(T), 1e-6))
        new["nadapt"] = st["nadapt"] + (moved & (hh < h))
        new["nvac"] = st["nvac"] + moved
        new["nstall"] = st["nstall"] + stall
        fate = st["fate"]
        for kd, fv in ((1, C.PLANE), (5, C.ABSORBED), (6, C.UPSTREAM), (7, C.WALL)):
            fate = jnp.where(kind == kd, fv, fate)
        enter = (kind == 2) | (kind == 3) | (kind == 4) | (kind == 8)
        new["fate"] = jnp.where(moved & ~enter & (fate == C.LIVE) & (new["s"] > s_max), C.LOOPER, fate)
        # slab frames, entry points and fields of the entering lanes (all among the gathered candidates)
        ty = jnp.where(kind == 2, ftype[kc], jnp.where(kind == 3, t_ipa, jnp.where(kind == 4, t_opa, t_ring)))
        ty = ty.astype(jnp.int32)
        ent_c = g(enter) & ok
        def entry_frames(_):
            e1, e2, nn, entry, dloc, Bl = frames(maps, xn[ic], un[ic], g(T), g(ty), g(kc))
            put = lambda old, v: old.at[jnp.where(ent_c, idx, n)].set(v, mode="drop")
            return (put(st["e1"], e1), put(st["e2"], e2), put(st["en"], nn), put(st["xent"], entry),
                    put(st["Bl"], Bl), put(jnp.zeros((n, 3), dt), dloc))

        # No entrants leave stored frames unchanged and need no local entry directions.
        if skip_empty:
            e1, e2, nn, entry, Bl, dfull = lax.cond(jnp.any(ent_c), entry_frames,
                lambda _: (st["e1"], st["e2"], st["en"], st["xent"], st["Bl"], jnp.zeros((n, 3), dt)), operand=None)
        else:
            e1, e2, nn, entry, Bl, dfull = entry_frames(None)
        new.update(e1=e1, e2=e2, en=nn, xent=entry, Bl=Bl,
                   Tin=jnp.where(enter, T, st["Tin"]), lty=jnp.where(enter, ty, st["lty"]))
        return new, enter, dfull

    def step(maps, key, st, m_cand, m_slab):
        live = st["fate"] == C.LIVE
        # slab step for up to m_slab lanes inside a layer (gathered), then the exits back to the global frame
        sl = st["slab"]
        act = live & st["inl"]
        idx, ic, ok, chosen = gather(act, m_slab)
        sub = {k: (v if k == "it" else v[ic]) for k, v in sl.items()}
        sub["exited"] = sub["exited"] | ~ok
        def brems(carry):
            sub, st = carry
            before = sub
            sub, emitted = SJ.lane_step_photons(key, lt, st["lty"][ic], sub, st["Bl"][ic],
                                                   positron=positron, msc=msc, unroll=unroll)
            dt_step = jnp.where(ok & act[ic], _slab_elapsed(before, sub), 0.)
            st = dict(st, slab_time=st["slab_time"].at[idx].add(dt_step, mode="drop"))
            st = append_births(st, emitted, idx, ic, ok & act[ic])
            for name in ("nph", "eph", "nde", "ede", "nan", "ean", "cont"):
                delta = jnp.where(ok & act[ic], sub[name] - before[name], 0)
                st[name] = st[name].at[idx].add(delta, mode="drop")
            return sub, st

        def idle(carry):
            sub, st = carry
            # Finished core lanes only advance the shared RNG iteration; no births or counter deltas.
            return {**sub, "it": sub["it"] + 1}, st

        sub, st = lax.cond(jnp.any(act), brems, idle, (sub, st))
        sl = {k: (sub[k] if k == "it" else sl[k].at[idx].set(sub[k], mode="drop")) for k in sl}
        stepped = act & chosen
        done = stepped & (sl["stopped"] | sl["exited"])
        lp, dn = sl["x"], sl["d"]
        xg = st["xent"] + lp[:, :1] * st["e1"] + lp[:, 1:2] * st["e2"] + lp[:, 2:3] * st["en"]
        ug = dn[:, :1] * st["e1"] + dn[:, 1:2] * st["e2"] + dn[:, 2:3] * st["en"]
        ug = ug / jnp.linalg.norm(ug, axis=1, keepdims=True)
        T1 = sl["T"]
        stop = sl["stopped"] | (T1 <= 0)
        path = jnp.where(done, sl["path"], 0.0)
        st = dict(st, slab=sl, x=jnp.where(done[:, None], xg, st["x"]), u=jnp.where(done[:, None], ug, st["u"]),
                  T=jnp.where(done, T1, st["T"]), s=st["s"] + path,
                  t=st["t"] + jnp.where(done, st["slab_time"], 0.),
                  cnt=st["cnt"] + (done[:, None] & (jnp.arange(NCOUNT) == counter_d[st["lty"]][:, None])),
                  fate=jnp.where(done & stop, C.STOPPED, st["fate"]), inl=st["inl"] & ~done,
                  nslab=st["nslab"] + stepped)
        st, enter, dloc = vacuum(maps, st, m_cand)
        new = SJ.lane_state(st["T"], dloc, jnp.zeros_like(st["T"]), Ltype[st["lty"]], positron=positron)
        sl = st["slab"]
        sl = {k: (v if k == "it" else jnp.where(enter.reshape((-1,) + (1,) * (v.ndim - 1)), new[k], v))
              for k, v in sl.items()}
        sl["exited"] = jnp.where(enter, new["exited"], sl["exited"] | ~st["inl"])
        return dict(st, slab=sl, inl=st["inl"] | enter, slab_time=jnp.where(enter, 0., st["slab_time"]))

    @jax.jit
    def vacuum_burst(maps, hot, n_it):
        def keep_going(c):
            j, state, blocked = c
            return (j < n_it) & ~blocked & jnp.any(state["fate"] == C.LIVE)

        def advance(c):
            j, state, _ = c
            trial, blocked = vacuum(maps, state, state["T"].shape[0], free_only=True)
            state = jax.tree_util.tree_map(lambda a, b: jnp.where(blocked, a, b), state, trial)
            return j + (~blocked).astype(jnp.int32), state, blocked
        return lax.while_loop(keep_going, advance, (jnp.int32(0), hot, jnp.bool_(False)))

    @partial(jax.jit, static_argnames=("m_cand", "m_slab", "stop_vacuum"))
    def run(maps, key, st, n_it, m_cand, m_slab, stop_vacuum=0):
        def cond(c):
            i, s = c
            live = s["fate"] == C.LIVE
            keep = (i < n_it) & jnp.any(live)
            if stop_vacuum:
                eligible = (jnp.sum(live) <= stop_vacuum) & ~jnp.any(live & s["inl"])
                keep = keep & ((i == 0) | ~eligible)
            return keep

        def compact_free(c):
            i, s = c
            n = s["T"].shape[0]
            idx, ic, ok, _ = gather(s["fate"] == C.LIVE, vacuum_bucket)
            names = ("x", "u", "T", "t", "s", "fate", "inl", "nvac", "nstall", "nadapt")
            hot = {name: s[name][ic] for name in names}
            hot["fate"] = jnp.where(ok, hot["fate"], C.PLANE)

            def keep_going(c):
                j, state, blocked = c
                return (i + j < n_it) & ~blocked & jnp.any(state["fate"] == C.LIVE)

            def advance(c):
                j, state, _ = c
                trial, blocked = vacuum(maps, state, vacuum_bucket, free_only=True)
                state = jax.tree_util.tree_map(lambda a, b: jnp.where(blocked, a, b), state, trial)
                return j + (~blocked).astype(jnp.int32), state, blocked

            used, hot, blocked = lax.while_loop(keep_going, advance, (jnp.int32(0), hot, jnp.bool_(False)))
            updated = {name: s[name].at[idx].set(hot[name], mode="drop") for name in names}
            sl = dict(s["slab"], it=s["slab"]["it"] + used,
                      exited=s["slab"]["exited"] | ((used > 0) & ~s["inl"]))
            s = dict(s, **updated, slab=sl)
            s = lax.cond(blocked, lambda s: step(maps, key, s, m_cand, m_slab), lambda s: s, s)
            return i + used + blocked.astype(jnp.int32), s

        def regular_body(c):
            i, s = c
            if vacuum_fast:
                advanced, candidate = vacuum(maps, s, m_cand, free_only=True)
                sl = dict(s["slab"], it=s["slab"]["it"] + 1,
                          exited=s["slab"]["exited"] | ~s["inl"])
                advanced = dict(advanced, slab=sl)
                fallback = candidate | jnp.any((s["fate"] == C.LIVE) & s["inl"])
                s = lax.cond(fallback, lambda _: step(maps, key, s, m_cand, m_slab),
                             lambda _: advanced, operand=None)
                return i + 1, s
            return i + 1, step(maps, key, s, m_cand, m_slab)

        def body(c):
            if vacuum_bucket:
                _, s = c
                live = s["fate"] == C.LIVE
                eligible = (jnp.sum(live) <= vacuum_bucket) & ~jnp.any(live & s["inl"])
                return lax.cond(eligible, compact_free, regular_body, c)
            return regular_body(c)
        return lax.while_loop(cond, body, (jnp.int32(0), st))

    run.vacuum_burst = vacuum_burst
    out = (jax.jit(init), run)
    _BUILT[key] = out
    return out


OUT_FIELDS = ("fate", "x", "u", "T", "t", "s", "cnt", "nvac", "nslab", "nstall")
TOTAL_FIELDS = ("nph", "eph", "nde", "ede", "nan", "ean", "cont")
