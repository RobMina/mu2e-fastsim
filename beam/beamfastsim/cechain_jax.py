"""Conversion-electron chain on the device (JAX): one loop of vacuum helix steps and slab steps, lane compaction.

Same geometry, physics and step logic as beamfastsim.cechain.transport (the NumPy reference, `--backend numpy|jax`),
with the whole chain from the start state to the tracker entry plane in one jitted lax.while_loop:

- each iteration runs one slab step for the lanes inside a layer (mu2efastsim.eslab_jax.lane_step: foils, IPA, OPA
  and IPA end rings are layer types of one stacked table, so lanes in different materials share the kernel; at most
  slab_frac of the lanes per iteration, gathered), then one helix step of length h (field at the step midpoint, DS
  and TSd maps on the device) for the lanes in vacuum. The lanes whose step ends beyond a surface (cheap tests on
  the step end points) are gathered, at most cand_frac of the lanes, and their first crossing is solved there in
  vectorised form: one plane solve for the tracker plane, the next foil face or the TSdA face (a 10 mm step cannot
  reach two of them), one radius bisection for the IPA or OPA shell and one for the end rings. Lanes over either cap
  wait one iteration unchanged (the vacuum step is deterministic, so this changes nothing but the time);
- a lane that reaches a layer enters it in the next iteration (no waiting for the others, unlike the NumPy driver:
  the per-lane physics is the same, the random streams differ);
- every `chunk` iterations the live lanes are gathered into a power-of-two bucket (finished lanes are copied out).

Statistically equivalent to the NumPy chain, not identical per seed. Computes in float64 or float32 (dtype); the
field maps and tables are cast to it. Mu2e frame, mm, ns, MeV.
"""
import time
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from mu2efastsim import eslab_jax as SJ

from . import cechain as C
from . import target_geometry_jax as TG
from . import fieldmap as FM
from . import geometry as G

ME = TG.ME
NCOUNT = TG.NCOUNT


layer_types = TG.layer_types
device_maps = TG.device_maps
_field = TG.field
_mom = TG.momentum
_beta = TG.beta
_dot = TG.dot
gather = TG.gather


_BUILT = {}


def _build(geom, dt, h, s_max, msc, layer_field, shell_curvature, z_record, unroll, const_field):
    key = (id(geom), np.dtype(dt).name, h, s_max, msc, layer_field, shell_curvature, z_record, unroll,
           None if const_field is None else tuple(np.asarray(const_field, float)))
    if key in _BUILT:
        return _BUILT[key]
    b = geom.beam
    specs, Lf, t_ipa, t_opa, t_ring, counter = layer_types(geom)
    lt = SJ.lane_tables(specs, msc=msc)
    Ltype = jnp.asarray(lt.length, dt)
    counter_d = jnp.asarray(counter)
    geometry = TG.build(geom, dt, Lf, charge=-1.0, const_field=const_field,
                        layer_field=layer_field, shell_curvature=shell_curvature, unroll=unroll)
    ftype, face_up, face_dn = geometry.ftype, geometry.face_up, geometry.face_dn
    frin, frout, nf = geometry.frin, geometry.frout, geometry.nf
    eps = 1e-7 if np.dtype(dt) == np.float64 else 2e-3       # foil-face search offset (a few ulp of z in float32)
    ztsda = float(b.tsda_z + b.tsda_hl)
    zr = [(float(z), float(z + 2 * geom.ring_hl)) for z in (b.ipa_z0, b.ipa_z1)]
    q = -1.0
    cms = C.C_MM_NS

    def W(c, a, b_):
        return jnp.where(c, a, b_).astype(dt)

    radial, opa_rin, bisect, frames = geometry.radial, geometry.opa_rin, geometry.bisect, geometry.frames

    def reset_slab(sl, enter, keep, T, d, ty):
        """Slab state: a fresh one (depth 0) for entering lanes, unchanged for lanes still inside, exited otherwise."""
        new = SJ.lane_state(T, d, jnp.zeros_like(T), Ltype[ty])
        out = {k: (v if k == "it" else jnp.where(enter.reshape((-1,) + (1,) * (v.ndim - 1)), new[k], v))
               for k, v in sl.items()}
        out["exited"] = jnp.where(enter, new["exited"], keep & sl["exited"] | ~keep)
        return out

    def init(maps, x, u, T, t, k0, zl):
        n = x.shape[0]
        born = k0 >= 0
        ty = jnp.where(born, ftype[jnp.clip(k0, 0, nf - 1)], 0)
        e1, e2, nn, entry, dloc, Bl = frames(maps, x, u, T, ty, k0)
        sl = SJ.lane_state(T, dloc, jnp.where(born, zl, 0.0), Ltype[ty])
        sl["exited"] = sl["exited"] | ~born
        st = dict(x=x, u=u, T=T, t=t, s=jnp.zeros(n, dt), fate=jnp.zeros(n, jnp.int32), inl=born, lty=ty,
                  e1=e1, e2=e2, en=nn, xent=entry, Tin=T, Bl=Bl, cnt=jnp.zeros((n, NCOUNT), jnp.int32), slab=sl,
                  nvac=jnp.zeros(n, jnp.int32), nslab=jnp.zeros(n, jnp.int32), nstall=jnp.zeros(n, jnp.int32))
        if z_record is not None:
            st.update(rec_ok=jnp.zeros(n, bool), rec_x=jnp.zeros((n, 3), dt), rec_u=jnp.zeros((n, 3), dt),
                      rec_T=jnp.zeros(n, dt), rec_t=jnp.zeros(n, dt))
        return st

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
            better = sel & jnp.isfinite(sv) & (sv < s_ev)
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

    def vacuum(maps, st, m_cand):
        """One helix step for live lanes in vacuum. Lanes whose step may cross a surface are solved in a gathered
        sub-batch of m_cand lanes (the others of them wait one iteration, unchanged); lanes that reach a layer get
        their slab frame. Returns the new state and the entering mask."""
        x, u, T = st["x"], st["u"], st["T"]
        vac = (st["fate"] == C.LIVE) & ~st["inl"]
        n = x.shape[0]
        p = _mom(T)
        B0 = _field(maps, const_field, x)
        Bm = _field(maps, const_field, SJ.helix(x, u, p, q, B0, jnp.full(n, 0.5 * h, dt))[0])
        hh = jnp.full(n, h, dt)
        xe = SJ.helix(x, u, p, q, Bm, hh)[0]
        z0, z1 = x[:, 2], xe[:, 2]
        zlo, zhi = jnp.minimum(z0, z1), jnp.maximum(z0, z1)
        r0, r1 = radial(x), radial(xe)
        sel_p = (z0 < geom.z_plane) & (z1 >= geom.z_plane)
        up = z1 > z0
        k = jnp.where(up, jnp.searchsorted(face_up, z0 + eps, side="left", method="compare_all"),
                      jnp.searchsorted(face_dn, z0 - eps, side="right", method="compare_all") - 1)
        kc = jnp.clip(k, 0, nf - 1).astype(jnp.int32)
        face = jnp.where(up, face_up[kc], face_dn[kc])
        sel_f = (k >= 0) & (k < nf) & jnp.where(up, face <= z1, face >= z1)
        sel_t = (z0 > ztsda) & (z1 <= ztsda)
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
        idx, ic, ok, chosen = gather(cand, m_cand)
        stall = cand & ~chosen
        g = lambda a: a[ic]
        s_c, k_c = crossings(g(x), g(u), g(p), g(Bm), g(hh), g(z0), g(z1), g(r0), g(r1), g(sel_p) & ok,
                             g(sel_f) & ok, g(sel_t) & ok, g(face), g(kc), g(sel_i) & ok, g(sel_o) & ok, g(out_i),
                             g(out_o), g(sel_r) & ok, g(out_r))
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
        if z_record is not None:
            crec = moved & ~st["rec_ok"] & ((z0 - z_record) * (xn[:, 2] - z_record) <= 0) & (z0 != z_record)
            srec = SJ.to_plane(x, u, p, q, Bm, jnp.full(n, z_record, dt), jnp.where(crec, s_ev, 0.0), unroll)
            fin = crec & jnp.isfinite(srec)
            xr, ur = SJ.helix(x, u, p, q, Bm, jnp.where(fin, srec, 0.0))
            new["rec_ok"] = st["rec_ok"] | fin
            new["rec_x"] = jnp.where(fin[:, None], xr, st["rec_x"])
            new["rec_u"] = jnp.where(fin[:, None], ur, st["rec_u"])
            new["rec_T"] = jnp.where(fin, T, st["rec_T"])
            new["rec_t"] = jnp.where(fin, st["t"] + srec / (cms * jnp.maximum(_beta(T), 1e-6)), st["rec_t"])
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
        # slab frames, entry points and fields of the entering lanes (all among the gathered candidates)
        ty = jnp.where(kind == 2, ftype[kc], jnp.where(kind == 3, t_ipa, jnp.where(kind == 4, t_opa, t_ring)))
        ty = ty.astype(jnp.int32)
        ent_c = g(enter) & ok
        e1, e2, nn, entry, dloc, Bl = frames(maps, xn[ic], un[ic], g(T), g(ty), g(kc))
        put = lambda old, v: old.at[jnp.where(ent_c, idx, n)].set(v, mode="drop")
        new.update(e1=put(st["e1"], e1), e2=put(st["e2"], e2), en=put(st["en"], nn), xent=put(st["xent"], entry),
                   Bl=put(st["Bl"], Bl), Tin=jnp.where(enter, T, st["Tin"]), lty=jnp.where(enter, ty, st["lty"]))
        dfull = put(jnp.zeros((n, 3), dt), dloc)
        return new, enter, dfull

    def step(maps, key, st, m_cand, m_slab):
        live = st["fate"] == C.LIVE
        # slab step for up to m_slab lanes inside a layer (gathered), then the exits back to the global frame
        sl = st["slab"]
        act = live & st["inl"]
        idx, ic, ok, chosen = gather(act, m_slab)
        sub = {k: (v if k == "it" else v[ic]) for k, v in sl.items()}
        sub["exited"] = sub["exited"] | ~ok
        sub = lax.cond(jnp.any(act), lambda s: SJ.lane_step(key, lt, st["lty"][ic], s, st["Bl"][ic], msc=msc,
                                                            unroll=unroll),
                       lambda s: {**s, "it": s["it"] + 1}, sub)
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
        bm = jnp.maximum(0.5 * (_beta(st["Tin"]) + _beta(T1)), 1e-6)
        st = dict(st, slab=sl, x=jnp.where(done[:, None], xg, st["x"]), u=jnp.where(done[:, None], ug, st["u"]),
                  T=jnp.where(done, T1, st["T"]), s=st["s"] + path, t=st["t"] + path / (cms * bm),
                  cnt=st["cnt"] + (done[:, None] & (jnp.arange(NCOUNT) == counter_d[st["lty"]][:, None])),
                  fate=jnp.where(done & stop, C.STOPPED, st["fate"]), inl=st["inl"] & ~done,
                  nslab=st["nslab"] + stepped)
        st, enter, dloc = vacuum(maps, st, m_cand)
        new = SJ.lane_state(st["T"], dloc, jnp.zeros_like(st["T"]), Ltype[st["lty"]])
        sl = st["slab"]
        sl = {k: (v if k == "it" else jnp.where(enter.reshape((-1,) + (1,) * (v.ndim - 1)), new[k], v))
              for k, v in sl.items()}
        sl["exited"] = jnp.where(enter, new["exited"], sl["exited"] | ~st["inl"])
        return dict(st, slab=sl, inl=st["inl"] | enter)

    @partial(jax.jit, static_argnames=("m_cand", "m_slab"))
    def run(maps, key, st, n_it, m_cand, m_slab):
        def cond(c):
            i, s = c
            return (i < n_it) & jnp.any(s["fate"] == C.LIVE)

        def body(c):
            i, s = c
            return i + 1, step(maps, key, s, m_cand, m_slab)
        return lax.while_loop(cond, body, (jnp.int32(0), st))

    out = (jax.jit(init), run)
    _BUILT[key] = out
    _STEPS[key] = (jax.jit(init), step)
    return out


_STEPS = {}


def lane_kernels(geom, field, h=10.0, s_max=30000.0, msc=True, layer_field=True, shell_curvature=True, dtype=np.float64,
                 unroll=False):
    """(init, step) of the chain for callers that drive it from their own device loop (e.g. a fused chain):
    init(maps, x, u, T, t, k0, zl) -> lane state, step(maps, key, state, m_cand, m_slab) -> state after one
    iteration (traceable). field: the device FieldMaps tuple or a constant 3-vector, as transport. Same kernels as
    transport."""
    dt = np.dtype(dtype)
    const = None if (isinstance(field, (tuple, list)) and field and isinstance(field[0], FM.FieldMap)) else \
        np.asarray(field, float)
    _build(geom, dt, float(h), float(s_max), bool(msc), bool(layer_field), bool(shell_curvature), None, unroll, const)
    key = (id(geom), dt.name, float(h), float(s_max), bool(msc), bool(layer_field), bool(shell_curvature), None, unroll,
           None if const is None else tuple(np.asarray(const, float)))
    return _STEPS[key]


def start_tables(geom, x, u, p):
    """Host start quantities of transport for births (x, u, p): kinetic energy T, the foil index k0 and the depth zl
    in the foil, as transport computes them."""
    b = geom.beam
    x = np.asarray(x, float)
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    n = x.shape[0]
    T = np.sqrt(np.broadcast_to(np.asarray(p, float), (n,)) ** 2 + ME ** 2) - ME
    k0 = G.foil_index(b, x[:, 0], x[:, 1], x[:, 2], tol=1e-3).astype(np.int32)
    kc = np.clip(k0, 0, b.foil_z.size - 1)
    L = 2 * b.foil_ht[kc]
    zl = np.clip(np.where(u[:, 2] >= 0, x[:, 2] - (b.foil_z[kc] - b.foil_ht[kc]),
                          (b.foil_z[kc] + b.foil_ht[kc]) - x[:, 2]), 0.0, L)
    return x, u, T, k0, zl



OUT_FIELDS = ("fate", "x", "u", "T", "t", "s", "cnt", "nvac", "nslab", "nstall")
REC_FIELDS = ("rec_ok", "rec_x", "rec_u", "rec_T", "rec_t")


@jax.jit
def _merge(st, new, mask):
    """Lanes in mask take the state `new` (refill); scalars (the slab iteration count) stay."""
    def m(a, b):
        return a if a.ndim == 0 else jnp.where(mask.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)
    return jax.tree_util.tree_map(m, st, new)


def transport(key, geom, field, x0, u0, p0, t0, h=10.0, s_max=30000.0, msc=True, layer_field=True, z_record=None,
              shell_curvature=True, dtype=np.float64, chunk=256, min_bucket=64, cand_frac=0.125, slab_frac=0.5,
              unroll=False, lanes=None, shrink=2, stats=None, device_out=False):
    """cechain.transport on the device. key: jax.random key; field: a constant 3-vector [T] or a tuple of device
    FieldMaps (device_maps). chunk: iterations per launch; between launches finished lanes are copied out and, while
    starts remain, refilled with new starts (lanes: lane count, a power of two, default all starts at once); then
    the live lanes are compacted into power-of-two buckets of at least min_bucket lanes, once they fit in 1/shrink of
    the current bucket (each bucket size compiles once: on a GPU, where an iteration costs about the same at any
    lane count, a large shrink saves compiles). cand_frac, slab_frac: sizes
    of the gathered crossing and slab sub-batches as fractions of the bucket; unroll (opt-in): fixed-trip loops
    unrolled (fewer device launches per iteration on a GPU). Returns cechain.Result (NumPy, float64) with per-lane
    vacuum steps, slab steps and waits (nvac, nslab, nstall). stats (dict): filled with iterations, rounds and
    lane-iterations. device_out (opt-in): the per-lane outputs are gathered on the device and returned as device
    arrays (same values; the host reads only the lane fates between launches)."""
    dt = np.dtype(dtype)
    maps, const = (tuple(field), None) if isinstance(field, (tuple, list)) and field and \
        isinstance(field[0], FM.FieldMap) else ((), np.asarray(field, float))
    init, run = _build(geom, dt, float(h), float(s_max), bool(msc), bool(layer_field), bool(shell_curvature),
                       None if z_record is None else float(z_record), unroll, const)
    b = geom.beam
    x = np.array(x0, float)
    u = np.array(u0, float)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    n = x.shape[0]
    T = np.sqrt(np.broadcast_to(np.asarray(p0, float), (n,)) ** 2 + ME ** 2) - ME
    t = np.array(np.broadcast_to(np.asarray(t0, float), (n,)), float)
    k0 = G.foil_index(b, x[:, 0], x[:, 1], x[:, 2], tol=1e-3).astype(np.int32)
    kc = np.clip(k0, 0, b.foil_z.size - 1)
    L = 2 * b.foil_ht[kc]
    zl = np.clip(np.where(u[:, 2] >= 0, x[:, 2] - (b.foil_z[kc] - b.foil_ht[kc]),
                          (b.foil_z[kc] + b.foil_ht[kc]) - x[:, 2]), 0.0, L)
    pow2 = lambda k: max(min_bucket, 1 << int(np.ceil(np.log2(max(k, 1)))))
    m = pow2(n if lanes is None else min(n, lanes))

    def block(slots, ids):
        """init state of m lanes: start ids in slots, copies of the first start elsewhere (inert after merging)."""
        j = np.full(m, ids[0])
        j[slots] = ids
        return init(maps, jnp.asarray(x[j], dt), jnp.asarray(u[j], dt), jnp.asarray(T[j], dt), jnp.zeros(m, dt),
                    jnp.asarray(k0[j]), jnp.asarray(zl[j], dt))
    first = np.arange(min(n, m))
    st = block(first, first)
    st["fate"] = st["fate"].at[first.size:].set(C.LOOPER)               # padding lanes are inert
    idx = np.concatenate([first, np.full(m - first.size, -1)])
    nxt = first.size
    fields = OUT_FIELDS + (REC_FIELDS if z_record is not None else ())
    ints = ("fate", "cnt", "rec_ok", "nvac", "nslab", "nstall")
    if device_out:
        final = {f: jnp.zeros((n,) + tuple(st[f].shape[1:]), st[f].dtype) for f in fields}
    else:
        final = {f: np.zeros((n,) + tuple(st[f].shape[1:]), np.asarray(st[f]).dtype if f in ints else np.float64)
                 for f in fields}
    rounds = iters = lane_it = refills = 0
    timeline = []
    while True:
        nb = idx.size
        tr = time.time()
        i, st = run(maps, jax.random.fold_in(key, rounds), st, jnp.int32(chunk),
                    max(1, min(nb, int(np.ceil(cand_frac * nb)))), max(1, min(nb, int(np.ceil(slab_frac * nb)))))
        i = int(i)
        timeline.append((nb, i, round(time.time() - tr, 3)))
        rounds += 1
        iters += i
        lane_it += i * nb
        live = np.asarray(st["fate"]) == C.LIVE
        fin = np.flatnonzero((idx >= 0) & ~live)
        if fin.size:
            fd = jnp.asarray(fin)
            if device_out:
                dst = jnp.asarray(idx[fin])
                for f in fields:
                    final[f] = final[f].at[dst].set(st[f][fd])
            else:
                for f in fields:
                    final[f][idx[fin]] = np.asarray(st[f][fd])
            idx[fin] = -1
        if nxt < n:                                                    # refill free lanes with new starts
            free = np.flatnonzero(idx < 0)
            k = min(free.size, n - nxt)
            if k:
                ids = np.arange(nxt, nxt + k)
                mask = np.zeros(m, bool)
                mask[free[:k]] = True
                st = _merge(st, block(free[:k], ids), jnp.asarray(mask))
                idx[free[:k]] = ids
                nxt += k
                refills += 1
            continue
        keep = np.flatnonzero(live & (idx >= 0))
        if keep.size == 0:
            break
        mb = pow2(keep.size)
        if mb * shrink <= idx.size:
            sel = np.concatenate([keep, np.full(mb - keep.size, keep[0])])
            sd = jnp.asarray(sel)
            st = jax.tree_util.tree_map(lambda a: a[sd] if getattr(a, "ndim", 0) > 0 else a, st)
            st["fate"] = st["fate"].at[keep.size:].set(C.LOOPER)
            idx = np.concatenate([idx[keep], np.full(mb - keep.size, -1)])
            m = mb
    if stats is not None:
        stats.update(iterations=iters, rounds=rounds, refills=refills, lane_iterations=lane_it,
                     lanes=int(pow2(n if lanes is None else min(n, lanes))), timeline=timeline)
    c = final["cnt"]
    if device_out:
        t = jnp.asarray(t, final["t"].dtype)
    # the device accumulates the time since the start (small, so float32 rounding does not bias it); t0 added here
    res = dict(fate=final["fate"].astype(jnp.int32 if device_out else int), x=final["x"], u=final["u"], T=final["T"],
               t=t + final["t"],
               path=final["s"], nfoil=c[:, 0].astype(int), nipa=c[:, 1].astype(int), nopa=c[:, 2].astype(int),
               nring=c[:, 3].astype(int), nvac=final["nvac"].astype(int), nslab=final["nslab"].astype(int),
               nstall=final["nstall"].astype(int))
    if z_record is not None:
        res.update(rec_ok=final["rec_ok"], rec_x=final["rec_x"], rec_u=final["rec_u"], rec_T=final["rec_T"],
                   rec_t=(jnp if device_out else np).where(final["rec_ok"], t + final["rec_t"], 0.0))
    return C.Result(**res)
