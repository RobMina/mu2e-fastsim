"""Opt-in production-target + PS chain on the device (JAX, CPU or GPU): the physics of `ptgt_chain.run` (host
reference) on a fixed pool of lanes.

A jitted chunk runs `steps` steps over a pool of N lanes. In a step every live lane takes either the material step
(bank cross sections, continuous loss with a fractional cap, Highland scattering, first-order field kick, inelastic /
elastic interactions, decay-length limit) or the vacuum step (RK4 in the field map, target re-entry along the chord,
record plane, envelope and thin absorbing annuli, upstream kill). Secondaries of inelastic interactions and decay
daughters are appended to an emission buffer, plane crossings to a record buffer (both with counters; a chunk stops
early before either can overflow). Between chunks the host refills dead lanes from the emission buffer, then with new
primaries, until every primary is done. Both float32 and float64 are supported (`dtype`); the bank stays float32.

Differences to the host reference (statistical, not bitwise): random streams; muons and pions are the only decaying
species with daughters (K decays use the same channel table); the chord re-entry, plane and annulus tests are the
same geometry evaluated per step.
"""
import dataclasses
from functools import partial
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from beamfastsim import fieldmap
from beamfastsim import ptgt_chain as C
from beamfastsim import ptgt_device as D

jax.config.update("jax_enable_x64", True)

SPECIES = np.asarray(D.SPECIES)
MASS_SP = np.array([D.MASS[s] for s in D.SPECIES])
CHARGE_SP = np.array([C.CHARGE[s] for s in D.SPECIES], float)
TAU_SP = np.array([C.TAU_NS.get(s, np.inf) for s in D.SPECIES])
HADRON_SP = np.isin(SPECIES, (2212, 2112))
I_PIP, I_PIM, I_KP, I_KM, I_MU, I_MUP = (D.SPECIES.index(s) for s in (211, -211, 321, -321, 13, -13))
K_BR = np.cumsum([br for br, _ in C.K_CHANNELS]) / sum(br for br, _ in C.K_CHANNELS)


class Static(NamedTuple):
    """Device constants: bank, target, envelope, field."""
    edges: object; sig_in: object; sig_el: object; dedx: object; x0: object; el_icdf: object
    cell_off: object; sec_off: object; sec_sp: object; sec_p: object
    z0: object; z1: object; rin: object; rout: object; mat: object; origin: object; rot: object   # per segment
    env_z: object; env_r: object; annuli: object
    fm: object
    rb: object; zc: object; hl: object          # target bounding cylinder (device scalars: no recompile per rod)
    slabs: object                               # passive thin rings (n, 4): z, r_in, r_out, thickness
    sl_edges: object; sl_sig_in: object; sl_dedx: object; sl_x0: object   # slab material tables (species x ke)


class Cfg(NamedTuple):
    nk: int; nmat: int; nseg: int
    z_plane: float; r_plane: float; z_min: float; axis_x: float
    h_vac: float; h_near: float; near: float; h_mat: float; neutral_step: float
    ke_min_hadron: float; ke_min_pion: float; max_loss_fraction: float
    eps: float; nudge: float; max_lane_steps: int


class Lanes(NamedTuple):
    alive: object; sp: object; x: object; u: object; p: object; t: object; tau: object; seg: object; ev: object
    n: object       # steps taken (a lane past Cfg.max_lane_steps is stopped and counted)
    # pending emission of a deferred lane (exact deferral: the sampled outcome is kept, never redrawn):
    # pk 0 none, 1 inelastic with bank row pj, 2 hadron decay with channel pch
    pk: object; pj: object; pch: object


def species_of(pdg):
    out = np.full(np.shape(pdg), -1, np.int32)
    for i, s in enumerate(D.SPECIES):
        out[np.asarray(pdg) == s] = i
    return out


def make_static(bank, target, env, fm, dtype, slab=None):
    f = lambda a: jnp.asarray(np.asarray(a), dtype)
    slabs = np.asarray(env.slabs, float).reshape(-1, 4) if len(env.slabs) else np.zeros((0, 4))
    if len(slabs) and slab is None:
        raise ValueError("envelope slabs need slab=(bank, material index)")
    if slab is not None:
        sb, sm = slab
        sl = {k: np.asarray(sb[k])[:, sm, :].reshape(-1) for k in ("sigma_inel", "dedx", "x0")}
        sl_edges = np.asarray(sb["log10_ke_edges"])
    else:
        sl, sl_edges = {k: np.ones(len(D.SPECIES)) for k in ("sigma_inel", "dedx", "x0")}, np.array([0.0, 4.0])
    nmat = len(bank.materials)
    nk = bank["log10_ke_edges"].size - 1
    sec_sp = species_of(bank["sec_pdg"])
    annuli = np.asarray(env.annuli, float).reshape(-1, 3) if len(env.annuli) else np.zeros((0, 3))
    st = Static(edges=f(bank["log10_ke_edges"]), sig_in=f(bank["sigma_inel"].reshape(-1)),
                sig_el=f(bank["sigma_el"].reshape(-1)), dedx=f(bank["dedx"].reshape(-1)), x0=f(bank["x0"].reshape(-1)),
                el_icdf=f(bank["el_cos_icdf"].reshape(-1, D.N_EL)), cell_off=jnp.asarray(bank["cell_offsets"], jnp.int64),
                sec_off=jnp.asarray(bank["sec_offsets"], jnp.int64), sec_sp=jnp.asarray(sec_sp, jnp.int32),
                sec_p=jnp.asarray(bank["sec_p"], jnp.float32),
                z0=f(target.zr[:, 0]), z1=f(target.zr[:, 1]), rin=f(target.r_in), rout=f(target.r_out), mat=jnp.asarray(target.mat, jnp.int32),
                origin=f(target.origin), rot=f(target.rot), env_z=f(env.z_nodes), env_r=f(env.r_nodes), annuli=f(annuli),
                fm=fieldmap.FieldMap(f(fm.lo), f(fm.d), jnp.asarray(fm.n), f(fm.B)),
                rb=f(np.max(target.r_out)), zc=f(0.5 * (target.z_lo + target.z_hi)),
                hl=f(0.5 * (target.z_hi - target.z_lo)),
                slabs=f(slabs), sl_edges=f(sl_edges), sl_sig_in=f(sl["sigma_inel"]), sl_dedx=f(sl["dedx"]),
                sl_x0=f(sl["x0"]))
    return st, nk, nmat


def make_cfg(target, env, stp, nk, nmat, dtype=jnp.float64, max_lane_steps=200_000):
    f32 = jnp.dtype(dtype) == jnp.float32
    return Cfg(eps=1e-4 if f32 else 1e-7, nudge=1e-3 if f32 else 1e-5, max_lane_steps=int(max_lane_steps),nk=nk, nmat=nmat, nseg=len(target.mat), z_plane=float(env.z_plane), r_plane=float(env.r_plane),
               z_min=float(env.z_min), axis_x=float(env.axis_x), h_vac=stp.h_vac, h_near=stp.h_near, near=stp.near,
               h_mat=stp.h_mat, neutral_step=stp.neutral_step, ke_min_hadron=stp.cuts.ke_min_hadron,
               ke_min_pion=stp.cuts.ke_min_pion, max_loss_fraction=stp.cuts.max_loss_fraction)


# ------------------------------------------------------------------------------------------------ geometry (jnp)

def to_local(S, x):
    return (x - S.origin) @ S.rot


def dir_local(S, u):
    return u @ S.rot


def _radii(S, i, z):
    z0, z1 = S.z0[i], S.z1[i]
    f = (z - z0) / (z1 - z0)
    return (S.rin[i, 0] + f * (S.rin[i, 1] - S.rin[i, 0]), S.rout[i, 0] + f * (S.rout[i, 1] - S.rout[i, 0]))


def locate(S, cfg, xl):
    z, r = xl[:, 2], jnp.hypot(xl[:, 0], xl[:, 1])
    seg = jnp.full(z.shape, -1, jnp.int32)
    for i in range(cfg.nseg - 1, -1, -1):              # the first segment wins where several hold the point
        rin, rout = _radii(S, i, z)
        inside = (z >= S.z0[i]) & (z < S.z1[i]) & (r >= rin) & (r < rout)
        seg = jnp.where(inside, i, seg)
    return seg


def _cone_hit(xl, dl, z0, z1, ra, rb, eps=1e-7):
    s = (rb - ra) / (z1 - z0)
    c0 = ra - s * z0
    x, y, z = xl[:, 0], xl[:, 1], xl[:, 2]
    dx, dy, dz = dl[:, 0], dl[:, 1], dl[:, 2]
    a = dx * dx + dy * dy - s * s * dz * dz
    b = 2 * (x * dx + y * dy - s * dz * (c0 + s * z))
    c = x * x + y * y - (c0 + s * z) ** 2
    disc = b * b - 4 * a * c
    ok = disc >= 0
    sq = jnp.sqrt(jnp.where(ok, disc, 0.0))
    lin = jnp.abs(a) < 1e-12
    big = jnp.asarray(jnp.inf, xl.dtype)
    safe_a = jnp.where(lin, 1.0, 2 * a)
    r1 = jnp.where(lin, -c / jnp.where(b == 0, 1.0, b), (-b - sq) / safe_a)
    r1 = jnp.where(lin & (b == 0), big, r1)
    r2 = jnp.where(lin, big, (-b + sq) / safe_a)
    best = jnp.full(x.shape, big)
    for t in (r1, r2):
        zt = z + t * dz
        good = ok & (t > eps) & (zt >= z0) & (zt <= z1) & (c0 + s * zt >= 0) & jnp.isfinite(t)
        best = jnp.where(good & (t < best), t, best)
    return best


def _plane_hit(xl, dl, zp, eps=1e-7):
    safe = jnp.where(dl[:, 2] == 0, 1.0, dl[:, 2])
    t = (zp - xl[:, 2]) / safe
    return jnp.where((dl[:, 2] != 0) & (t > eps), t, jnp.inf)


def distance_out(S, cfg, xl, dl, seg):
    big = jnp.full(xl.shape[0], jnp.inf, xl.dtype)
    out = big
    for i in range(cfg.nseg):
        z0, z1 = S.z0[i], S.z1[i]
        e = cfg.eps
        ti = jnp.minimum(_plane_hit(xl, dl, z0, e), _plane_hit(xl, dl, z1, e))
        ti = jnp.minimum(ti, _cone_hit(xl, dl, z0, z1, S.rout[i, 0], S.rout[i, 1], e))
        has_in = (S.rin[i, 0] > 0) | (S.rin[i, 1] > 0)
        ti = jnp.where(has_in, jnp.minimum(ti, _cone_hit(xl, dl, z0, z1, S.rin[i, 0], S.rin[i, 1], e)), ti)
        out = jnp.where(seg == i, ti, out)
    return out


def distance_in(S, cfg, xl, dl):
    t = jnp.full(xl.shape[0], jnp.inf, xl.dtype)
    for i in range(cfg.nseg):
        z0, z1 = S.z0[i], S.z1[i]
        t = jnp.minimum(t, _cone_hit(xl, dl, z0, z1, S.rout[i, 0], S.rout[i, 1], cfg.eps))
        has_in = (S.rin[i, 0] > 0) | (S.rin[i, 1] > 0)
        t = jnp.where(has_in, jnp.minimum(t, _cone_hit(xl, dl, z0, z1, S.rin[i, 0], S.rin[i, 1], cfg.eps)), t)
        for zp, k in ((z0, 0), (z1, 1)):
            tp = _plane_hit(xl, dl, zp, cfg.eps)
            pp = xl + jnp.where(jnp.isfinite(tp), tp, 0.0)[:, None] * dl
            r = jnp.hypot(pp[:, 0], pp[:, 1])
            t = jnp.where((r >= S.rin[i, k]) & (r < S.rout[i, k]) & (tp < t), tp, t)
    return t


def frame_rotate(pl, d, phi):
    """Vectors pl [N,3] from the frame whose z is d (unit) to the lab, after an azimuth phi about that z."""
    c, s = jnp.cos(phi), jnp.sin(phi)
    pr = jnp.stack([c * pl[:, 0] - s * pl[:, 1], s * pl[:, 0] + c * pl[:, 1], pl[:, 2]], 1)
    helper = jnp.where(jnp.abs(d[:, 2:3]) < 0.9, jnp.array([[0.0, 0.0, 1.0]], d.dtype), jnp.array([[1.0, 0.0, 0.0]], d.dtype))
    ua = jnp.cross(helper, d)
    ua = ua / jnp.linalg.norm(ua, axis=1, keepdims=True)
    va = jnp.cross(d, ua)
    return ua * pr[:, 0:1] + va * pr[:, 1:2] + d * pr[:, 2:3]


def _rk4(fm, x, u, k, h):
    def f(xp, up):
        return up, k[:, None] * jnp.cross(up, fieldmap.field(fm, xp))
    x1, u1 = f(x, u)
    x2, u2 = f(x + 0.5 * h[:, None] * x1, u + 0.5 * h[:, None] * u1)
    x3, u3 = f(x + 0.5 * h[:, None] * x2, u + 0.5 * h[:, None] * u2)
    x4, u4 = f(x + h[:, None] * x3, u + h[:, None] * u3)
    xn = x + h[:, None] / 6.0 * (x1 + 2 * x2 + 2 * x3 + x4)
    un = u + h[:, None] / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4)
    return xn, un / jnp.linalg.norm(un, axis=1, keepdims=True)


def _iso(key, n, dtype):
    k1, k2 = jax.random.split(key)
    c = 2 * jax.random.uniform(k1, (n,), dtype) - 1
    s = jnp.sqrt(jnp.maximum(1 - c * c, 0.0))
    ph = 2 * jnp.pi * jax.random.uniform(k2, (n,), dtype)
    return jnp.stack([s * jnp.cos(ph), s * jnp.sin(ph), c], 1)


def _boost(e, pv, P, E, M):
    b = P / E[:, None]
    g = E / M
    bp = jnp.sum(b * pv, axis=1)
    b2 = jnp.sum(b * b, axis=1)
    coef = jnp.where(b2 > 0, (g - 1) * bp / jnp.where(b2 > 0, b2, 1.0), 0.0) + g * e
    return pv + coef[:, None] * b


# ------------------------------------------------------------------------------------------------ one step

SLAB_SUBSTEPS = C.SLAB_SUBSTEPS


def _slab_cross(S, sp, m, q, p, u, thick, k1, k2, k3):
    """Device version of ptgt_chain.slab_cross (same model); lanes with thick == 0 come back unchanged."""
    dt = p.dtype
    N = p.shape[0]
    nk = S.sl_edges.shape[0] - 1
    lo, hi = 10.0 ** S.sl_edges[0], 10.0 ** S.sl_edges[-1]
    path = thick / jnp.maximum(jnp.abs(u[:, 2]), C.SLAB_MIN_COS)

    def cell(ke):
        lk = jnp.log10(jnp.clip(ke, lo * (1 + 1e-6), hi * (1 - 1e-6)))
        return sp * nk + jnp.clip(jnp.searchsorted(S.sl_edges, lk, side="right") - 1, 0, nk - 1)

    ke0 = jnp.hypot(p, m) - m
    c0 = cell(ke0)
    absorbed = jax.random.uniform(k1, (N,), dt) < -jnp.expm1(-S.sl_sig_in[c0] * path)
    x0 = S.sl_x0[c0]
    ke = ke0
    for _ in range(SLAB_SUBSTEPS):
        ke = ke - S.sl_dedx[cell(jnp.maximum(ke, lo))] * path / SLAB_SUBSTEPS
    stopped = ke <= 0
    ke = jnp.maximum(ke, 0.0)
    pn = jnp.sqrt(jnp.maximum((ke + m) ** 2 - m ** 2, 0.0))
    b0 = p / jnp.hypot(p, m)
    b1 = pn / jnp.maximum(jnp.hypot(pn, m), 1e-30)
    pb, bet = jnp.sqrt(jnp.maximum(p * b0 * pn * b1, 1e-30)), jnp.sqrt(b0 * b1)
    xr = path / x0
    th0 = 13.6 / pb * jnp.sqrt(xr) * (1 + 0.038 * jnp.log(jnp.maximum(xr / jnp.maximum(bet, 1e-6) ** 2, 1e-12)))
    th0 = jnp.where((q != 0) & (pn > 0) & (thick > 0), jnp.abs(th0), 0.0)
    th = jnp.minimum(th0 * jnp.sqrt(-2.0 * jnp.log(jax.random.uniform(k2, (N,), dt, 1e-12, 1.0))), jnp.pi)
    un = frame_rotate(jnp.stack([jnp.sin(th), jnp.zeros_like(th), jnp.cos(th)], 1), u,
                      2 * jnp.pi * jax.random.uniform(k3, (N,), dt))
    un = un / jnp.linalg.norm(un, axis=1, keepdims=True)
    on = thick > 0
    return jnp.where(on, pn, p), jnp.where(on[:, None], un, u), on & (absorbed | stopped)


def _step(S, cfg, carry, key):
    L, emit, ecount, rec, rcount, cnt = carry
    dt = L.x.dtype
    N = L.alive.shape[0]
    ks = jax.random.split(key, 15)
    sp = jnp.clip(L.sp, 0, len(D.SPECIES) - 1)
    m = jnp.asarray(MASS_SP, dt)[sp]
    q = jnp.asarray(CHARGE_SP, dt)[sp]
    ke = jnp.hypot(L.p, m) - m
    low = jnp.where(jnp.asarray(HADRON_SP)[sp], ke < cfg.ke_min_hadron, ke < cfg.ke_min_pion)
    pend = L.alive & (L.pk > 0)                    # deferred lanes: emit their kept outcome, do not move
    capped = L.alive & ~pend & (L.n >= cfg.max_lane_steps)
    stop = L.alive & ~pend & (low | (L.sp < 0) | capped)
    alive = L.alive & ~stop & ~pend
    beta = L.p / jnp.hypot(L.p, m)
    finite_tau = jnp.isfinite(L.tau)
    d_dec = jnp.where(finite_tau, L.tau * C.C_MM_NS * L.p / m, jnp.inf)
    inm = alive & (L.seg >= 0)
    vac = alive & (L.seg < 0)
    # ---------------------------------------------------------------- material step (computed for every lane)
    xl, dl = to_local(S, L.x), dir_local(S, L.u)
    seg = jnp.clip(L.seg, 0, cfg.nseg - 1)
    lk = jnp.log10(jnp.maximum(ke, 1e-30))
    kbin = jnp.clip(jnp.searchsorted(S.edges, lk, side="right") - 1, 0, cfg.nk - 1)
    cell = (sp * cfg.nmat + S.mat[seg]) * cfg.nk + kbin
    s_in, s_el, dedx, x0 = S.sig_in[cell], S.sig_el[cell], S.dedx[cell], S.x0[cell]
    s_tot = s_in + s_el
    u1 = jax.random.uniform(ks[0], (N,), dt, 1e-12, 1.0)
    t_int = jnp.where(s_tot > 0, -jnp.log(u1) / jnp.where(s_tot > 0, s_tot, 1.0), jnp.inf)
    t_loss = jnp.where(dedx > 0, cfg.max_loss_fraction * ke / jnp.where(dedx > 0, dedx, 1.0), jnp.inf)
    t_bnd = distance_out(S, cfg, xl, dl, seg)
    stp = jnp.minimum(jnp.minimum(jnp.minimum(t_int, t_loss), jnp.minimum(t_bnd, d_dec)), cfg.h_mat)
    B = fieldmap.field(S.fm, L.x)
    um = L.u + (C.K_FIELD * q / L.p * stp)[:, None] * jnp.cross(L.u, B)
    um = um / jnp.linalg.norm(um, axis=1, keepdims=True)
    xm = L.x + stp[:, None] * L.u
    de = jnp.minimum(dedx * stp, ke)
    e_new = ke - de + m
    pm = jnp.sqrt(jnp.maximum(e_new ** 2 - m ** 2, 0.0))
    bet = pm / jnp.maximum(e_new, 1e-30)
    xr = stp / x0
    th0 = 13.6 / jnp.maximum(bet * pm, 1e-30) * jnp.sqrt(xr) * (1 + 0.038 * jnp.log(jnp.maximum(xr / jnp.maximum(bet, 1e-6) ** 2, 1e-12)))
    th0 = jnp.where((stp > 0) & (q != 0), jnp.abs(th0), 0.0)
    th = jnp.minimum(th0 * jnp.sqrt(-2.0 * jnp.log(jax.random.uniform(ks[1], (N,), dt, 1e-12, 1.0))), jnp.pi)
    um = frame_rotate(jnp.stack([jnp.sin(th), jnp.zeros_like(th), jnp.cos(th)], 1), um,
                      2 * jnp.pi * jax.random.uniform(ks[2], (N,), dt))
    um = um / jnp.linalg.norm(um, axis=1, keepdims=True)
    hit = inm & (t_int <= stp) & (t_int <= t_bnd)
    crossed = inm & (t_bnd <= stp) & ~hit
    xm = jnp.where(crossed[:, None], xm + cfg.nudge * um, xm)
    inel = hit & (jax.random.uniform(ks[3], (N,), dt) * s_tot < s_in)
    el = hit & ~inel
    # elastic: angle from the bank's inverse CDF
    f = jax.random.uniform(ks[4], (N,), dt) * (D.N_EL - 1)
    kk = jnp.minimum(f.astype(jnp.int32), D.N_EL - 2)
    w = f - kk
    ct = (1 - w) * S.el_icdf[cell, kk] + w * S.el_icdf[cell, kk + 1]
    st_ = jnp.sqrt(jnp.maximum(1 - ct * ct, 0.0))
    ue = frame_rotate(jnp.stack([st_, jnp.zeros_like(st_), ct], 1), um, 2 * jnp.pi * jax.random.uniform(ks[5], (N,), dt))
    um = jnp.where(el[:, None], ue, um)
    # ---------------------------------------------------------------- vacuum step
    rloc = jnp.hypot(xl[:, 0], xl[:, 1])
    near = (rloc < S.rb + cfg.near) & (jnp.abs(xl[:, 2] - S.zc) < S.hl + cfg.near)
    charged = q != 0
    hv = jnp.where(near, cfg.h_near, jnp.where(charged, cfg.h_vac, cfg.neutral_step))
    tn = distance_in(S, cfg, xl, dl)                              # neutral: straight to the next entry, or out
    hv = jnp.where(charged, hv, jnp.where(jnp.isfinite(tn), tn + cfg.nudge, 0.0))
    hv = jnp.minimum(hv, d_dec)
    xv, uv = _rk4(S.fm, L.x, L.u, C.K_FIELD * q / L.p, jnp.where(charged, hv, 0.0))
    xv = jnp.where(charged[:, None], xv, L.x + hv[:, None] * L.u)
    uv = jnp.where(charged[:, None], uv, L.u)
    chord = xv - L.x
    clen = jnp.linalg.norm(chord, axis=1)
    cd = chord / jnp.maximum(clen, 1e-30)[:, None]
    ti = distance_in(S, cfg, xl, dir_local(S, cd))
    enter = near & charged & (ti < clen)
    fr = jnp.where(enter, (ti + cfg.nudge) / jnp.maximum(clen, 1e-30), 1.0)
    xv = jnp.where(enter[:, None], L.x + (ti + cfg.nudge)[:, None] * cd, xv)
    hv = hv * fr
    # passive slabs: the step ends just past the first slab plane crossed inside the ring (loss, scattering,
    # absorption in _slab_cross)
    fs = jnp.full(N, jnp.inf, dt)
    sid = jnp.full(N, -1, jnp.int32)
    for a in range(S.slabs.shape[0]):
        za, ra, rbb = S.slabs[a, 0], S.slabs[a, 1], S.slabs[a, 2]
        hz = vac & ((L.x[:, 2] - za) * (xv[:, 2] - za) <= 0) & (xv[:, 2] != L.x[:, 2])
        fa = jnp.where(hz, (za - L.x[:, 2]) / jnp.where(hz, xv[:, 2] - L.x[:, 2], 1.0), jnp.inf)
        xa = L.x + jnp.where(hz, fa, 0.0)[:, None] * (xv - L.x)
        ra_ = jnp.hypot(xa[:, 0] - cfg.axis_x, xa[:, 1])
        hit = hz & (ra_ >= ra) & (ra_ < rbb) & (fa < fs)
        fs, sid = jnp.where(hit, fa, fs), jnp.where(hit, a, sid)
    slab = sid >= 0
    fs0 = jnp.where(slab, fs, 0.0)
    us = L.u + fs0[:, None] * (uv - L.u)
    us = us / jnp.linalg.norm(us, axis=1, keepdims=True)
    xs = L.x + fs0[:, None] * (xv - L.x)
    xs = xs.at[:, 2].add(jnp.where(us[:, 2] >= 0, cfg.nudge, -cfg.nudge))
    thick = jnp.where(slab, S.slabs[jnp.clip(sid, 0, max(S.slabs.shape[0] - 1, 0)), 3] if S.slabs.shape[0] else 0.0, 0.0)
    p_sl, u_sl, abs_sl = _slab_cross(S, sp, m, q, L.p, us, thick, ks[12], ks[13], ks[14])
    xv = jnp.where(slab[:, None], xs, xv)
    uv = jnp.where(slab[:, None], u_sl, uv)
    hv = jnp.where(slab, hv * fs0, hv)
    abs_sl = slab & abs_sl
    zp = cfg.z_plane
    cross = vac & (L.x[:, 2] < zp) & (xv[:, 2] >= zp)
    fz = jnp.where(cross, (zp - L.x[:, 2]) / jnp.where(cross, xv[:, 2] - L.x[:, 2], 1.0), 0.0)
    xp = L.x + fz[:, None] * (xv - L.x)
    upl = L.u + fz[:, None] * (uv - L.u)
    upl = upl / jnp.linalg.norm(upl, axis=1, keepdims=True)
    rad = lambda xx: jnp.hypot(xx[:, 0] - cfg.axis_x, xx[:, 1])
    rec_ok = cross & (rad(xp) < cfg.r_plane) & ((sp == I_PIP) | (sp == I_PIM) | (sp == I_MU) | (sp == I_MUP))
    tp = L.t + fz * hv / (beta * C.C_MM_NS)
    gone_neutral = vac & ~charged & (hv == 0)
    ab = vac & ~cross & ((rad(xv) > jnp.interp(xv[:, 2], S.env_z, S.env_r)) | abs_sl)
    for a in range(S.annuli.shape[0]):
        za, ra, rbb = S.annuli[a, 0], S.annuli[a, 1], S.annuli[a, 2]
        hz = vac & ((L.x[:, 2] - za) * (xv[:, 2] - za) <= 0) & (xv[:, 2] != L.x[:, 2])
        fa = jnp.where(hz, (za - L.x[:, 2]) / jnp.where(hz, xv[:, 2] - L.x[:, 2], 1.0), 0.0)
        ra_ = rad(L.x + fa[:, None] * (xv - L.x))
        ab = ab | (hz & ~cross & (ra_ >= ra) & (ra_ < rbb))
    upstream = vac & ~cross & ~ab & (xv[:, 2] < cfg.z_min)
    # ---------------------------------------------------------------- combine
    h = jnp.where(inm, stp, hv)
    x_new = jnp.where(inm[:, None], xm, xv)
    u_new = jnp.where(inm[:, None], um, uv)
    p_new = jnp.where(inm, pm, jnp.where(slab, p_sl, L.p))
    t_new = L.t + h / (beta * C.C_MM_NS)
    tau_new = jnp.where(finite_tau, L.tau - h * m / (L.p * C.C_MM_NS), L.tau)
    seg_new = locate(S, cfg, to_local(S, x_new))
    keep = lambda new, old: jnp.where(pend.reshape((-1,) + (1,) * (new.ndim - 1)), old, new)
    x_new, u_new, p_new, t_new = keep(x_new, L.x), keep(u_new, L.u), keep(p_new, L.p), keep(t_new, L.t)
    tau_new, seg_new = keep(tau_new, L.tau), keep(seg_new, L.seg)
    decays = alive & finite_tau & (tau_new <= 1e-9) & ~cross & ~ab & ~upstream & ~gone_neutral & ~inel
    dec_had = decays & ((sp == I_PIP) | (sp == I_PIM) | (sp == I_KP) | (sp == I_KM))
    p_inel, p_dec = pend & (L.pk == 1), pend & (L.pk == 2)
    inel = inel | p_inel
    dec_had = dec_had | p_dec
    end = stop | inel | cross | ab | upstream | gone_neutral | decays | p_dec
    alive_new = L.alive & ~end
    # ---------------------------------------------------------------- emissions: inelastic secondaries
    cellc = jnp.clip(cell, 0, S.cell_off.shape[0] - 2)
    nco = S.cell_off[cellc + 1] - S.cell_off[cellc]
    j = S.cell_off[cellc] + jnp.floor(jax.random.uniform(ks[6], (N,), jnp.float64) * nco).astype(jnp.int64)
    j = jnp.where(p_inel, L.pj, jnp.clip(j, 0, S.sec_off.shape[0] - 2))
    n_sec = jnp.where(inel & ((nco > 0) | p_inel), S.sec_off[j + 1] - S.sec_off[j], 0)
    # decays: pi -> mu (1 daughter); K: channel daughters (charged mu / pi only)
    ch = jnp.searchsorted(jnp.asarray(K_BR, dt), jax.random.uniform(ks[7], (N,), dt), side="right")
    ch = jnp.where(p_dec, L.pch, ch)
    n_dec = jnp.where(dec_had & ((sp == I_PIP) | (sp == I_PIM)), 1,
                      jnp.where(dec_had, jnp.asarray([1, 1, 3, 1, 1, 0])[jnp.clip(ch, 0, 5)], 0))
    n_emit = (n_sec + n_dec).astype(jnp.int64)
    E_CAP = emit[0].shape[0]
    M = cfg_emit_per_step(N)
    # lanes whose emissions do not fit this step's budget are deferred: they keep their post-step state and the
    # sampled outcome (pk, pj, pch) and emit it on a later step. Redrawing the step instead would bias the physics:
    # deferral is selected on the outcome (an interaction, a row with many secondaries)
    budget = jnp.minimum(M, E_CAP - ecount)
    defer = (n_emit > 0) & (jnp.cumsum(n_emit) > budget)
    pk_new = jnp.where(defer, jnp.where(inel, 1, 2), 0).astype(jnp.int32)
    pj_new = jnp.where(defer, j, 0).astype(jnp.int64)
    pch_new = jnp.where(defer, ch, 0).astype(jnp.int32)
    n_emit = jnp.where(defer, 0, n_emit)
    n_sec = jnp.where(defer, 0, n_sec)
    inel = inel & ~defer
    dec_had = dec_had & ~defer
    alive_new = jnp.where(defer, L.alive, alive_new)
    tot = jnp.sum(n_emit)
    lane = jnp.repeat(jnp.arange(N), n_emit, total_repeat_length=M)
    start = jnp.cumsum(n_emit) - n_emit
    k_in = jnp.arange(M) - start[lane]
    valid = jnp.arange(M) < tot
    is_dec = k_in >= n_sec[lane]
    # inelastic rows
    row = jnp.clip(S.sec_off[j[lane]] + k_in, 0, S.sec_p.shape[0] - 1)
    pl = S.sec_p[row].astype(dt)
    d_par = u_new[lane]
    phi = 2 * jnp.pi * jax.random.uniform(ks[8], (M,), dt)
    pvec = frame_rotate(pl, d_par, phi)
    sp_e = S.sec_sp[row]
    # decay daughters (isotropic two/three body approximated as two-body per daughter for K 3pi: see note below)
    kd = k_in - n_sec[lane]
    par_sp = sp[lane]
    par_m = m[lane]
    P = u_new[lane] * p_new[lane][:, None]
    Epar = jnp.hypot(p_new[lane], par_m)
    chl = jnp.clip(ch[lane], 0, 5)
    is_pi_par = (par_sp == I_PIP) | (par_sp == I_PIM)
    sign = jnp.where((par_sp == I_PIP) | (par_sp == I_KP), 1, -1)
    # daughter species: pi -> mu (mu+ for pi+); K channels: 0 mu nu, 1 pi pi0, 2 pi pi pi (charges +,+,- for K+),
    # 3 pi pi0 pi0, 4 pi0 mu nu
    d_mu = jnp.where(sign > 0, I_MUP, I_MU)
    d_pi_same = jnp.where(sign > 0, I_PIP, I_PIM)
    d_pi_opp = jnp.where(sign > 0, I_PIM, I_PIP)
    k_sp = jnp.where(chl == 0, d_mu, jnp.where(chl == 4, d_mu, jnp.where(chl == 2,
                     jnp.where(kd < 2, d_pi_same, d_pi_opp), d_pi_same)))
    dsp = jnp.where(is_pi_par, d_mu, k_sp)
    m_d = jnp.asarray(MASS_SP, dt)[jnp.clip(dsp, 0, len(D.SPECIES) - 1)]
    # rest-frame momentum: exact two-body for pi -> mu nu, K -> mu nu, K -> pi pi0; three-body channels use the
    # mean daughter momentum of uniform phase space with an isotropic direction (approximation; < 6 % of K decays)
    def p2(Mp, m1, m2):
        return jnp.sqrt(jnp.maximum((Mp ** 2 - (m1 + m2) ** 2) * (Mp ** 2 - (m1 - m2) ** 2), 0.0)) / (2 * Mp)
    MK, MPI, MMU, MPI0 = C.MASS[321], C.MASS[211], C.MASS[13], C.MASS[111]
    pst = jnp.where(is_pi_par, p2(MPI, MMU, 0.0), jnp.where(chl == 0, p2(MK, MMU, 0.0), jnp.where(chl == 1, p2(MK, MPI, MPI0),
                    jnp.where(chl == 2, 0.13, jnp.where(chl == 3, 0.13, 0.15)) * MK)))
    dirr = _iso(ks[9], M, dt)
    pd = _boost(jnp.hypot(pst, m_d), dirr * pst[:, None], P, Epar, par_m)
    pvec = jnp.where(is_dec[:, None], pd, pvec)
    sp_e = jnp.where(is_dec, dsp, sp_e)
    pa = jnp.linalg.norm(pvec, axis=1)
    slot = ecount + jnp.arange(M)
    ok = valid & (slot < E_CAP)
    slot = jnp.where(ok, slot, E_CAP)
    e_alive, e_sp, e_x, e_u, e_p, e_t, e_seg, e_ev = emit
    e_sp = e_sp.at[slot].set(sp_e.astype(jnp.int32), mode="drop")
    e_x = e_x.at[slot].set(x_new[lane], mode="drop")
    e_u = e_u.at[slot].set(pvec / jnp.maximum(pa, 1e-30)[:, None], mode="drop")
    e_p = e_p.at[slot].set(pa, mode="drop")
    e_t = e_t.at[slot].set(t_new[lane], mode="drop")
    e_seg = e_seg.at[slot].set(seg_new[lane], mode="drop")
    e_ev = e_ev.at[slot].set(L.ev[lane], mode="drop")
    e_alive = e_alive.at[slot].set(True, mode="drop")
    emit = (e_alive, e_sp, e_x, e_u, e_p, e_t, e_seg, e_ev)
    over_e = jnp.maximum(tot - M, 0) + jnp.maximum(ecount + jnp.minimum(tot, M) - E_CAP, 0)
    ecount = jnp.minimum(ecount + jnp.minimum(tot, M), E_CAP)
    # ---------------------------------------------------------------- records
    R_CAP = rec[0].shape[0]
    rk = jnp.cumsum(rec_ok.astype(jnp.int64)) - 1 + rcount
    rslot = jnp.where(rec_ok & (rk < R_CAP), rk, R_CAP)
    r_ev, r_sp, r_x, r_p, r_t = rec
    rec = (r_ev.at[rslot].set(L.ev, mode="drop"), r_sp.at[rslot].set(sp, mode="drop"),
           r_x.at[rslot].set(xp, mode="drop"), r_p.at[rslot].set(upl * L.p[:, None], mode="drop"),
           r_t.at[rslot].set(tp, mode="drop"))
    nrec = jnp.sum(rec_ok)
    over_r = jnp.maximum(rcount + nrec - R_CAP, 0)
    rcount = jnp.minimum(rcount + nrec, R_CAP)
    cnt = cnt + jnp.stack([jnp.sum(inel), jnp.sum(el & ~defer), jnp.sum(stop), jnp.sum(dec_had), jnp.sum(ab), jnp.sum(upstream),
                           nrec, over_e, over_r, jnp.sum(capped),
                           jnp.sum(defer), jnp.sum(slab & ~defer), jnp.sum(abs_sl)]).astype(jnp.int64)
    L = Lanes(alive=alive_new, sp=L.sp, x=x_new, u=u_new, p=p_new, t=t_new, tau=tau_new, seg=seg_new, ev=L.ev,
              n=L.n + L.alive.astype(jnp.int32), pk=pk_new, pj=pj_new, pch=pch_new)
    return (L, emit, ecount, rec, rcount, cnt)


def cfg_emit_per_step(N):
    """Largest number of lanes emitted in one step (static): the pool size (a step that would emit more counts the
    excess in emit_overflow, which makes run() raise)."""
    return N


COUNTERS = ("inelastic", "elastic", "stopped", "decays", "absorbed", "upstream", "recorded", "emit_overflow",
            "record_overflow", "step_cap", "deferred", "slab_crossed",
            "slab_absorbed")


@partial(jax.jit, static_argnums=(1, 3))
def chunk(S, cfg, carry, steps, key):
    """Up to `steps` steps (fewer when every lane is done)."""
    N = carry[0].alive.shape[0]
    M = cfg_emit_per_step(N)
    E_CAP = carry[1][0].shape[0]

    def cond(c):
        i, cr = c
        return (i < steps) & jnp.any(cr[0].alive)        # emissions beyond the buffer are deferred, never lost

    def body(c):
        i, cr = c
        return i + 1, _step(S, cfg, cr, jax.random.fold_in(key, i))

    i, carry = lax.while_loop(cond, body, (jnp.int32(0), carry))
    return carry, i


def _empty_emit(E, dt):
    return (jnp.zeros(E, bool), jnp.zeros(E, jnp.int32), jnp.zeros((E, 3), dt), jnp.zeros((E, 3), dt), jnp.zeros(E, dt),
            jnp.zeros(E, dt), jnp.full(E, -1, jnp.int32), jnp.zeros(E, jnp.int64))


def _empty_rec(R, dt):
    return (jnp.zeros(R, jnp.int64), jnp.zeros(R, jnp.int32), jnp.zeros((R, 3), dt), jnp.zeros((R, 3), dt), jnp.zeros(R, dt))


@jax.jit
def _refill(L, emit, ecount, key, tau_scale):
    """Move up to ecount emitted lanes into dead slots of the pool (in order). Returns the pool, the remaining
    emission buffer (shifted) and its count."""
    N = L.alive.shape[0]
    E = emit[0].shape[0]
    dead = jnp.nonzero(~L.alive, size=N, fill_value=N)[0]
    ndead = jnp.sum(~L.alive)
    k = jnp.minimum(ndead, ecount)
    src = jnp.arange(N)
    take = src < k
    dst = jnp.where(take, dead, N)
    srcc = jnp.where(take, src, 0)
    e_alive, e_sp, e_x, e_u, e_p, e_t, e_seg, e_ev = emit
    life = jnp.asarray(TAU_SP, L.p.dtype)[jnp.clip(e_sp[srcc], 0, len(D.SPECIES) - 1)]
    tau = jnp.where(jnp.isfinite(life), -jnp.log(jax.random.uniform(key, (N,), L.p.dtype, 1e-12, 1.0)) * life, jnp.inf)
    L = Lanes(alive=L.alive.at[dst].set(True, mode="drop"), sp=L.sp.at[dst].set(e_sp[srcc], mode="drop"),
              x=L.x.at[dst].set(e_x[srcc], mode="drop"), u=L.u.at[dst].set(e_u[srcc], mode="drop"),
              p=L.p.at[dst].set(e_p[srcc], mode="drop"), t=L.t.at[dst].set(e_t[srcc], mode="drop"),
              tau=L.tau.at[dst].set(tau, mode="drop"), seg=L.seg.at[dst].set(e_seg[srcc], mode="drop"),
              ev=L.ev.at[dst].set(e_ev[srcc], mode="drop"), n=L.n.at[dst].set(0, mode="drop"),
              pk=L.pk.at[dst].set(0, mode="drop"), pj=L.pj, pch=L.pch)
    # shift the rest of the buffer to the front
    idx = jnp.arange(E) + k
    keep = idx < ecount
    idxc = jnp.where(keep, idx, 0)
    emit = tuple(jnp.where(keep.reshape((-1,) + (1,) * (a.ndim - 1)), a[idxc], jnp.zeros_like(a)) for a in emit)
    return L, emit, ecount - k


@jax.jit
def _inject(L, prim, first, n_prim):
    """Fill dead slots with primaries first..first+k (k = min(dead, remaining)); returns the pool and k."""
    N = L.alive.shape[0]
    dead = jnp.nonzero(~L.alive, size=N, fill_value=N)[0]
    k = jnp.minimum(jnp.sum(~L.alive), n_prim - first)
    src = jnp.arange(N)
    take = src < k
    dst = jnp.where(take, dead, N)
    s = jnp.clip(first + src, 0, prim["p"].shape[0] - 1)
    L = Lanes(alive=L.alive.at[dst].set(True, mode="drop"), sp=L.sp.at[dst].set(prim["sp"][s], mode="drop"),
              x=L.x.at[dst].set(prim["x"][s], mode="drop"), u=L.u.at[dst].set(prim["u"][s], mode="drop"),
              p=L.p.at[dst].set(prim["p"][s], mode="drop"), t=L.t.at[dst].set(prim["t"][s], mode="drop"),
              tau=L.tau.at[dst].set(jnp.inf, mode="drop"), seg=L.seg.at[dst].set(prim["seg"][s], mode="drop"),
              ev=L.ev.at[dst].set(prim["ev"][s], mode="drop"), n=L.n.at[dst].set(0, mode="drop"),
              pk=L.pk.at[dst].set(0, mode="drop"), pj=L.pj, pch=L.pch)
    return L, k


def run(bank, target, fm, env, prim, seed=1, st=C.Stepping(), pool=1 << 16, emit_cap=None, rec_cap=None, slab=None,
        steps=64, dtype=jnp.float64, max_chunks=10_000_000):
    """Device chain for the primaries `prim` (host dict as ptgt_chain.primaries). Returns (record dict, counters)."""
    # transport in a frame centred on the target (float32: millimetre coordinates near the target keep sub-micron
    # resolution for boundary steps); records are shifted back to the Mu2e frame
    shift = np.array([target.origin[0], 0.0, target.origin[2]])
    target = dataclasses.replace(target, origin=np.asarray(target.origin) - shift)
    env = C.Envelope(z_nodes=np.asarray(env.z_nodes) - shift[2], r_nodes=np.asarray(env.r_nodes), z_min=env.z_min - shift[2],
                     z_plane=env.z_plane - shift[2], r_plane=env.r_plane, axis_x=env.axis_x - shift[0],
                     annuli=tuple((a[0] - shift[2], a[1], a[2]) for a in env.annuli),
                     slabs=tuple((a[0] - shift[2], a[1], a[2], a[3]) for a in env.slabs))
    fm = fieldmap.FieldMap(np.asarray(fm.lo) - shift, fm.d, fm.n, fm.B)
    prim = dict(prim, x=np.asarray(prim["x"], float) - shift)
    S, nk, nmat = make_static(bank, target, env, fm, dtype, slab=slab)
    cfg = make_cfg(target, env, st, nk, nmat, dtype)
    N = int(pool)
    E = int(emit_cap or 8 * N)
    R = int(rec_cap or max(1 << 16, len(prim["p"]) // 4))
    if E < cfg_emit_per_step(N) * 2:
        raise ValueError("emission buffer must hold at least two steps of emissions")
    L = Lanes(alive=jnp.zeros(N, bool), sp=jnp.zeros(N, jnp.int32), x=jnp.zeros((N, 3), dtype), u=jnp.zeros((N, 3), dtype),
              p=jnp.ones(N, dtype), t=jnp.zeros(N, dtype), tau=jnp.full(N, jnp.inf, dtype), seg=jnp.full(N, -1, jnp.int32),
              ev=jnp.zeros(N, jnp.int64), n=jnp.zeros(N, jnp.int32), pk=jnp.zeros(N, jnp.int32), pj=jnp.zeros(N, jnp.int64),
              pch=jnp.zeros(N, jnp.int32))
    seg0 = target.locate(target.to_local(prim["x"]))
    P = dict(sp=jnp.asarray(species_of(prim["pdg"])), x=jnp.asarray(prim["x"], dtype), u=jnp.asarray(prim["u"], dtype),
             p=jnp.asarray(prim["p"], dtype), t=jnp.asarray(prim["t"], dtype), seg=jnp.asarray(seg0, jnp.int32),
             ev=jnp.asarray(prim["event"], jnp.int64))
    n_prim = len(prim["p"])
    emit, ecount = _empty_emit(E, dtype), jnp.int64(0)
    rec, rcount = _empty_rec(R, dtype), jnp.int64(0)
    cnt = jnp.zeros(len(COUNTERS), jnp.int64)
    key = jax.random.PRNGKey(seed)
    first, steps_run, stuck_prev = 0, 0, None
    for it in range(max_chunks):
        kr = jax.random.fold_in(key, 2 * it)
        L, emit, ecount = _refill(L, emit, ecount, kr, 1.0)
        if int(ecount) == 0 and first < n_prim:
            L, k = _inject(L, P, jnp.int64(first), jnp.int64(n_prim))
            first += int(k)
        if not bool(jnp.any(L.alive)) and int(ecount) == 0 and first >= n_prim:
            break
        (L, emit, ecount, rec, rcount, cnt), i = chunk(S, cfg, (L, emit, ecount, rec, rcount, cnt), steps,
                                                       jax.random.fold_in(key, 2 * it + 1))
        steps_run += int(i)
        if int(i) == 0 and bool(jnp.all(L.alive)):
            raise RuntimeError("emission buffer full with no free lane: raise emit_cap or pool")
        # every lane holding a deferred emission while the buffer has no room for any of them cannot progress
        stuck = int(ecount) if bool(jnp.all(L.alive & (L.pk > 0))) else None
        if stuck is not None and stuck == stuck_prev:
            raise RuntimeError("every lane holds a deferred emission and the buffer is full: raise emit_cap or pool")
        stuck_prev = stuck
    else:
        raise RuntimeError("device chain did not finish")
    c = {k: int(v) for k, v in zip(COUNTERS, np.asarray(cnt))}
    if c["emit_overflow"] or c["record_overflow"]:
        raise RuntimeError(f"buffer overflow: {c}")
    n = int(rcount)
    out = dict(ev=np.asarray(rec[0][:n]), pdg=SPECIES[np.asarray(rec[1][:n])], x=np.asarray(rec[2][:n], float) + shift,
               p=np.asarray(rec[3][:n], float), t=np.asarray(rec[4][:n], float))
    c["steps"] = steps_run
    return out, c


def main(argv=None):
    import argparse
    import json
    import time

    from beamfastsim import ptgt_g4bl

    ap = argparse.ArgumentParser(description="device target + PS chain for one rod design point")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--rod", nargs=4, type=float, metavar=("TLENGTH", "R_UP", "R_MID", "R_DN"))
    ap.add_argument("--rods-tsv", help="point table: index, Tlength, R_up, R_mid, R_dn per line (with --points)")
    ap.add_argument("--points", nargs="+", type=int, help="rows of --rods-tsv to run in this process (one compile)")
    ap.add_argument("--material", default="G4_W")
    C.add_design_arguments(ap)
    ap.add_argument("--map", required=True, help="field map .header (PS region)")
    ap.add_argument("--env-z", nargs="+", type=float, required=True)
    ap.add_argument("--env-r", nargs="+", type=float, required=True)
    ap.add_argument("--z-min", type=float, required=True)
    ap.add_argument("--r-plane", type=float, required=True)
    ap.add_argument("--z-plane", type=float, default=-4044.45)
    ap.add_argument("--annulus", nargs=3, type=float, action="append", default=[])
    ap.add_argument("--slab", nargs=4, type=float, action="append", default=[], metavar=("Z", "R_IN", "R_OUT", "THICK"),
                    help="passive thin ring (repeatable): loss, scattering, absorption from --slab-bank")
    ap.add_argument("--slab-bank")
    ap.add_argument("--slab-material")
    ap.add_argument("--pot", type=int, required=True)
    ap.add_argument("--warm-pot", type=int, default=2000, help="primaries of an untimed first run (compile)")
    ap.add_argument("--pool", type=int, default=1 << 18)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--ke-min-hadron", type=float, default=D.Cuts.ke_min_hadron,
                    help="nucleon transport cut, MeV (below: local deposit)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    bank = D.Bank.load(a.bank)
    st = C.Stepping(cuts=D.Cuts(ke_min_hadron=a.ke_min_hadron))
    slab = C.load_slab(a.slab, a.slab_bank, a.slab_material)
    if a.rods_tsv:
        table = {int(r[0]): r[1:5] for r in np.loadtxt(a.rods_tsv, ndmin=2)}
        jobs = [(pt, table[pt]) for pt in a.points]
    else:
        jobs = [(a.point, a.rod)]
    env = C.Envelope(z_nodes=np.asarray(a.env_z), r_nodes=np.asarray(a.env_r), z_min=a.z_min, z_plane=a.z_plane,
                     r_plane=a.r_plane, annuli=tuple(tuple(x) for x in a.annulus), slabs=tuple(tuple(x) for x in a.slab))
    fm = fieldmap.load(a.map)
    dt = jnp.float64 if a.dtype == "float64" else jnp.float32
    for n_job, (point, rod) in enumerate(jobs):
        rod = [float(v) for v in rod]
        tgt, beam_tgt = C.design_targets(a, bank, rod)
        rng = np.random.default_rng(a.seed + 7919 * point)
        if a.warm_pot and n_job == 0:
            run(bank, tgt, fm, env, C.primaries(a.warm_pot, beam_tgt, rng), st=st, seed=a.seed + 1000, pool=a.pool,
                steps=a.steps, dtype=dt, slab=slab)
        prim = C.primaries(a.pot, beam_tgt, rng)
        t0 = time.perf_counter()
        rec, cnt = run(bank, tgt, fm, env, prim, st=st, seed=a.seed + point, pool=a.pool, steps=a.steps, dtype=dt,
                       slab=slab)
        wall = time.perf_counter() - t0
        out = a.out.replace("{point}", f"{point:02d}") if "{point}" in a.out else a.out
        _save_point(a, bank, env, rec, cnt, point, rod, wall, out, ptgt_g4bl, slab)


def _save_point(a, bank, env, rec, cnt, point, rod, wall, out, ptgt_g4bl, slab=None):
    import json
    ref, meta = C.reference(rec, a.pot, point, 0)
    meta.update(rod=rod, material=a.material, seed=a.seed + point, counts=cnt, wall_s=wall, s_per_pot=wall / a.pot,
                dtype=a.dtype, pool=a.pool, steps=a.steps, ke_min_hadron=a.ke_min_hadron, device=str(jax.devices()[0]),
                generator="ptgt_chain_jax",
                envelope=dict(z=a.env_z, r=a.env_r, z_min=a.z_min, r_plane=a.r_plane, z_plane=a.z_plane, annuli=a.annulus,
                              slabs=a.slab, slab_bank=C.slab_meta(slab, a.slab_bank)),
                bank=dict(path=str(a.bank), content_hash=bank.lib.content_hash))
    ptgt_g4bl.save(out, ref, meta)
    print(json.dumps({k: meta[k] for k in ("point", "pot", "entrances", "yield_per_pot", "species", "s_per_pot",
                                           "wall_s", "counts", "device", "dtype")}), flush=True)

if __name__ == "__main__":
    main()
