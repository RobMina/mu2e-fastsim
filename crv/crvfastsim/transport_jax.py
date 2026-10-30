"""Opt-in JAX twin of the extracted straight-line CRV transport.

Physics follows transport.py, including its single entry-momentum bar-loss draw,
range-out rule, unsliced MSC momentum and generated-beta clock. RNG streams differ.
World preparation and table assembly are host operations; numeric batches remain
on the selected JAX device until the host tier-1b boundary.
"""
from functools import partial
import time

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import lax

from mu2efastsim import eloss, eloss_jax as ej, materials, transport as ct, world as wm
from . import tables
from . import transport as nt
from .crossings_jax import crossings, slab
from . import eloss_transport_jax as sampling

jax.config.update("jax_enable_x64", True)


def p_after(p, loss):
    tk = jnp.sqrt(p * p + eloss.MMU ** 2) - eloss.MMU - loss
    return jnp.sqrt(jnp.maximum((jnp.maximum(tk, 0.) + eloss.MMU) ** 2 - eloss.MMU ** 2, 0.))


@jax.jit
def slab_loss(key, c, p, length):
    """Core slab_loss slicing, stopping and mean-tail rules with fixed-shape lanes."""
    tk0 = jnp.sqrt(p * p + eloss.MMU ** 2) - eloss.MMU
    def cond(state):
        i, left, lost = state
        return (i < ct.N_SLICES_MAX) & jnp.any((left > 0) & (tk0 - lost > ct.TK_STOP))
    def body(state):
        i, left, lost = state
        tk = tk0 - lost
        go = (left > 0) & (tk > ct.TK_STOP)
        pg = jnp.sqrt(jnp.maximum((jnp.maximum(tk, 0.) + eloss.MMU) ** 2 - eloss.MMU ** 2, 1e-6))
        dd = ej.dedx_restricted(c, pg, 1e9)
        fine = ct.SLICE_FRAC * tk / jnp.maximum(dd, 1e-30)
        step = jnp.where(tk > 3. * dd * left, left / ct.MAX_SLICES_PER_SLAB, fine)
        step = jnp.minimum(left, jnp.maximum(step, fine))
        step = jnp.where(left <= step * 1.0001, left, step)
        step = jnp.where(go, step, 0.)
        dl = sampling.sample_loss(jax.random.fold_in(key, i), c, pg, step, 1e9)
        dl = jnp.where(dl >= tk, dd * step, dl)
        return i + 1, left - step, lost + jnp.where(go, dl, 0.)
    _, left, lost = lax.while_loop(cond, body, (0, length, jnp.zeros_like(p)))
    rest = (left > 0) & (tk0 - lost > ct.TK_STOP)
    pg = jnp.sqrt(jnp.maximum((jnp.maximum(tk0 - lost, 0.) + eloss.MMU) ** 2 - eloss.MMU ** 2, 1e-6))
    lost += jnp.where(rest, ej.dedx_restricted(c, pg, 1e9) * left, 0.)
    return jnp.where(tk0 - lost <= ct.TK_STOP, tk0, jnp.minimum(lost, tk0))


@partial(jax.jit, static_argnames=("nmats",))
def column(o, d, length, planes, mat, lo, hi, nmats):
    """Convex-piece clipping, reduced per material without host sparse compaction."""
    enter = jnp.zeros((len(o), len(mat)))
    leave = jnp.broadcast_to(length[:, None], enter.shape)
    outside = jnp.zeros(enter.shape, bool)
    def clip(k, state):
        a, b, bad = state
        n, dd = planes[:, k, :3], planes[:, k, 3]
        den = d @ n.T
        num = dd[None] - o @ n.T
        s = num / jnp.where(den == 0., 1., den)
        return (jnp.maximum(a, jnp.where(den < 0, s, -jnp.inf)),
                jnp.minimum(b, jnp.where(den > 0, s, jnp.inf)), bad | ((den == 0.) & (num < 0.)))
    enter, leave, outside = lax.fori_loop(0, planes.shape[1], clip, (enter, leave, outside))
    box_in, box_out = slab(o, d, lo, hi)
    box_hit = jnp.minimum(box_out, length[:, None]) > jnp.maximum(box_in, 0.)
    lengths = jnp.where(outside | ~box_hit, 0., jnp.maximum(leave - enter, 0.))
    return (jnp.stack([jnp.where(mat[None] == k, lengths, 0.).sum(1) for k in range(nmats)], axis=1)
            if nmats else jnp.zeros((len(o), 0), jnp.float64))


def pack(pieces):
    ps = pieces if isinstance(pieces, wm.PieceSet) else wm.PieceSet(pieces)
    unknown = sorted(set(ps.mats) - materials.BY_NAME.keys())
    if unknown:
        raise ValueError(f"JAX transport requires explicit material constants: {unknown}")
    return (jnp.asarray(ps.P), jnp.asarray(ps.mat), jnp.asarray(ps.lo), jnp.asarray(ps.hi)), ps.mats


@jax.jit
def scatter(key, o, d, p, y_plane, xx0, bh):
    """CRV default band-weighted Highland covariance and projection onto the plane."""
    length = (o[:, 1] - y_plane) / -d[:, 1]
    total = xx0.sum(1)
    beta = p / jnp.sqrt(p * p + eloss.MMU ** 2)
    x = jnp.maximum(total, 1e-12)
    th = 13.6 / (beta * p) * jnp.sqrt(x) * (1. + .038 * jnp.log(x / beta ** 2))
    w = xx0 / jnp.maximum(total[:, None], 1e-30)
    syt = th ** 2 * (w * bh).sum(1)
    syy = th ** 2 * (w * bh ** 2).sum(1)
    kt, ky = jax.random.split(key)
    zt = jax.random.normal(kt, (len(o), 2), jnp.float64)
    zy = jax.random.normal(ky, (len(o), 2), jnp.float64)
    dth = zt * th[:, None]
    dy = zt * (syt / jnp.maximum(th, 1e-30))[:, None]
    dy += zy * jnp.sqrt(jnp.maximum(syy - syt ** 2 / jnp.maximum(th ** 2, 1e-30), 0.))[:, None]
    ref = jnp.where(jnp.abs(d[:, :1]) < .9, jnp.array([1., 0., 0.]), jnp.array([0., 0., 1.]))
    u = jnp.cross(d, ref)
    u /= jnp.linalg.norm(u, axis=1)[:, None]
    v = jnp.cross(d, u)
    end = o + d * length[:, None] + u * dy[:, :1] + v * dy[:, 1:]
    dn = d + u * jnp.tan(dth[:, :1]) + v * jnp.tan(dth[:, 1:])
    dn /= jnp.linalg.norm(dn, axis=1)[:, None]
    back = (end[:, 1] - y_plane) / dn[:, 1]
    return end - dn * back[:, None], dn, length - back


class DeviceTransport:
    """Prepared immutable geometry/world constants; reuse for warm passes."""
    def __init__(self, geom, world=None, absorbers=None, tagger_bar=None, bar_cut=None,
                 stage="C", air_loss=True, absorber_straggling=True, batch_size=4096, capacity=64):
        if stage not in ("A", "C"):
            raise ValueError("JAX transport supports stages A and C only")
        if batch_size < 1 or capacity < 1:
            raise ValueError("batch_size and capacity must be positive")
        self.geom, self.stage = geom, stage
        self.batch_size, self.capacity = batch_size, capacity
        self.y = nt.msc_plane(geom)
        self.lo, self.hi = (jnp.asarray(a) for a in (geom.center - geom.halflen, geom.center + geom.halflen))
        tag = np.arange(geom.nbars) >= nt.TAGGER_FIRST_BAR if tagger_bar is None else np.asarray(tagger_bar, bool)
        cut = np.where(tag, materials.POLYSTYRENE.cuts_e_MeV["tagger"], materials.POLYSTYRENE.cuts_e_MeV["crv"])
        self.cut = jnp.asarray(cut if bar_cut is None else bar_cut)
        self.bands = []
        self.edges = world.edges if isinstance(world, ct.BandedWorld) else nt.band_edges(self.y)
        if stage == "C":
            if world is None:
                raise ValueError("stage C requires a world (an empty list is allowed)")
            for k in range(len(self.edges) - 1):
                ps = world.band(k) if isinstance(world, ct.BandedWorld) else [
                    q for q in world if q[3][1] >= self.edges[k+1] and q[2][1] <= self.edges[k]]
                data, names = pack(ps)
                order = (world.mat_order[k] if hasattr(world, "mat_order") else
                         names if isinstance(ps, wm.PieceSet) else list(dict.fromkeys(q[0] for q in ps)))
                self.bands.append((data, names, order))
        self.absorbers = None
        if absorbers is not None:
            alo, ahi, mat = absorbers
            mat = materials.BY_NAME[mat] if isinstance(mat, str) else mat
            self.absorbers = (jnp.asarray(alo), jnp.asarray(ahi), ej.consts(mat))
        self.bar_kernel = self._make_bar_kernel(air_loss, absorber_straggling)
        self.batch_kernel = jax.jit(self.batch)

    def world(self, key, o, d, p, active):
        loss = jnp.zeros_like(p)
        length = jnp.maximum((o[:, 1] - self.y) / -jnp.where(d[:, 1] < 0, d[:, 1], -1.), 0.)
        bxx, bh = [], []
        for k, (data, names, order) in enumerate(self.bands):
            top, bot = self.edges[k:k+2]
            st = jnp.clip((o[:, 1] - top) / -jnp.where(d[:, 1] < 0, d[:, 1], -1.), 0., length)
            sb = jnp.clip((o[:, 1] - bot) / -jnp.where(d[:, 1] < 0, d[:, 1], -1.), 0., length)
            seg = jnp.where(active, sb - st, 0.)
            col = column(o + d * st[:, None], d, seg, *data, len(names))
            solid, xx = jnp.zeros_like(p), jnp.zeros_like(p)
            for j, name in enumerate(order):
                mat = materials.BY_NAME[name]
                c = col[:, names.index(name)]
                dl = slab_loss(jax.random.fold_in(key, k * 100 + j), ej.consts(mat), p_after(p, loss), c)
                loss += jnp.where(c > 0., dl, 0.)
                solid += c
                xx += c / eloss.radiation_length_mm(mat)
            air = jnp.maximum(seg - solid, 0.)
            loss += ej.dedx_restricted(materials.AIR, jnp.maximum(p_after(p, loss), 1.), 1e9) * air
            bxx.append(xx + air / eloss.radiation_length_mm(materials.AIR))
            bh.append(length - .5 * (st + sb))
        alive = jnp.sqrt(p*p + eloss.MMU**2) - eloss.MMU - loss > .5
        return p_after(p, loss), alive, jnp.stack(bxx, 1), jnp.stack(bh, 1)

    def _make_bar_kernel(self, air_loss, absorber_straggling):
        lo, hi, cut, capacity, ab, stage = self.lo, self.hi, self.cut, self.capacity, self.absorbers, self.stage
        @jax.jit
        def kernel(key, o, d, p, offset, active):
            xs = crossings(o, d, lo, hi, capacity, active)
            bar, ti, to, valid, overflow = xs
            valid &= active[:, None]
            path = to - ti
            air = ej.dedx_restricted(materials.AIR, p, 1e9)[:, None] * (ti + (0. if stage == "C" else offset[:, None]))
            if stage != "C" and not air_loss:
                air = jnp.zeros_like(ti)
            gap = jnp.zeros_like(ti)
            if ab is not None:
                ai, ao = slab(o, d, ab[0], ab[1])
                prev = jnp.concatenate([jnp.full((len(o), 1), -jnp.inf), to[:, :-1]], 1)
                def gap_one(a, b):
                    return jnp.maximum(jnp.minimum(ao, b[:, None]) - jnp.maximum(ai, a[:, None]), 0.).sum(1)
                gap = jax.vmap(gap_one, in_axes=(1, 1), out_axes=1)(prev, ti)
            def step(state, inputs):
                carried, live = state
                k, b, a, ln, gl, ok = inputs
                pa = p_after(p, a + carried)
                if ab is not None:
                    hit = live & ok & (pa > 1.) & (gl > 0.)
                    dl = (lax.cond(jnp.any(hit),
                                   lambda: sampling.sample_loss(jax.random.fold_in(key, 2*k), ab[2], pa, gl, 1e9),
                                   lambda: jnp.zeros_like(pa))
                          if absorber_straggling else ej.dedx_restricted(ab[2], pa, 1e9) * gl)
                    carried += jnp.where(hit, dl, 0.)
                ph = p_after(p, a + carried)
                go = live & ok & (ph > 1.)
                loss = lax.cond(jnp.any(go),
                                lambda: sampling.sample_loss(jax.random.fold_in(key, 2*k+1), materials.POLYSTYRENE,
                                                       ph, ln, cut[b]), lambda: jnp.zeros_like(ph))
                tk = jnp.sqrt(ph*ph + eloss.MMU**2) - eloss.MMU
                stops = loss >= tk
                ln = jnp.where(stops, jnp.minimum(ln, tk / ej.dedx_restricted(materials.POLYSTYRENE, ph, 1e9)), ln)
                loss = jnp.where(go, jnp.minimum(loss, tk), 0.)
                return (carried + loss, live & (~ok | ((ph > 1.) & ~stops))), (ph, ln, loss, go & (loss > 0))
            inputs = (jnp.arange(bar.shape[1]), bar.T, air.T, path.T, gap.T, valid.T)
            _, out = lax.scan(step, (jnp.zeros_like(p), jnp.ones(len(p), bool)), inputs)
            pin, path, loss, keep = (a.T for a in out)
            vis = loss / (1. + materials.POLYSTYRENE.birks_mm_per_MeV * loss / jnp.where(path > 0, path, 1.))
            return bar, ti, to, pin, path, loss, vis, keep, overflow
        return kernel

    def batch(self, key, o, d, p, active):
        offset = jnp.zeros_like(p)
        if self.stage == "C":
            length = (o[:, 1] - self.y) / -jnp.where(d[:, 1] < 0, d[:, 1], -1.)
            plane = o + d * length[:, None]
            near = (active & (d[:, 1] < 0) & (plane[:, 0] > self.lo[:, 0].min() - nt.CRV_FOOTPRINT_MARGIN)
                    & (plane[:, 0] < self.hi[:, 0].max() + nt.CRV_FOOTPRINT_MARGIN)
                    & (plane[:, 2] > self.lo[:, 2].min() - nt.CRV_FOOTPRINT_MARGIN)
                    & (plane[:, 2] < self.hi[:, 2].max() + nt.CRV_FOOTPRINT_MARGIN))
            pp, alive, xx, bh = self.world(jax.random.fold_in(key, 0), o, d, p, near)
            pm = jnp.maximum(jnp.sqrt(p * jnp.maximum(pp, 1.)), 1.)
            os, ds, ls = scatter(jax.random.fold_in(key, 1), o, d, pm, self.y, xx, bh)
            o, d, offset = jnp.where(near[:, None], os, o), jnp.where(near[:, None], ds, d), jnp.where(near, ls, 0.)
            p = jnp.where(near, jnp.where(alive, pp, 0.), p)
        result = self.bar_kernel(jax.random.fold_in(key, 2), o, d, p, offset, active)
        return o, d, offset, result


def transport(gen, geom, rng, *, stage="C", world=None, absorbers=None, tagger_bar=None, bar_cut=None,
              air_loss=True, all_muons=True, absorber_straggling=True, prepared=None, timers=None,
              batch_size=4096, capacity=64, **unsupported):
    """Return ordinary CrvStep/particle frames at the host tier-1b boundary.

    Optional NumPy physics variants are rejected, never silently approximated.
    An explicit prepared DeviceTransport avoids geometry transfers between passes.
    """
    bad = [k for k, v in unsupported.items() if v is not None and v is not False]
    if bad:
        raise ValueError(f"JAX transport does not support: {bad}")
    a = time.perf_counter()
    dev = prepared or DeviceTransport(geom, world, absorbers, tagger_bar, bar_cut, stage, air_loss,
                                      absorber_straggling, batch_size, capacity)
    if dev.geom is not geom or dev.stage != stage:
        raise ValueError("prepared transport does not match the geometry/stage")
    mu = nt.all_primary_muons(gen) if all_muons else tables.primary_muons(gen)
    mukey = mu.mukey.to_numpy() if all_muons else np.ones(len(mu), np.int64)
    mom = mu[["px", "py", "pz"]].to_numpy()
    p0 = np.linalg.norm(mom, axis=1)
    if not np.isfinite(p0).all() or (p0 <= 0).any():
        raise ValueError("JAX transport requires finite nonzero generated momentum")
    origins = mu[["x", "y", "z"]].to_numpy()
    if not np.isfinite(origins).all():
        raise ValueError("JAX transport requires finite origins")
    direction = mom / p0[:, None]
    key = jax.random.key(int(rng.integers(0, 2**32, dtype=np.uint64)))
    if timers is not None:
        timers["prepare_s"] = time.perf_counter() - a
    a = time.perf_counter()
    rows = []
    indices = np.arange(len(mu))
    if stage == "C":
        lo, hi = (geom.center-geom.halflen).min(0), (geom.center+geom.halflen).max(0)
        ti, to = nt.crossings.slab_crossings(origins, direction, lo[None], hi[None], backend="numpy")
        indices = np.flatnonzero(nt.near_crv(origins, direction, geom, y_plane=dev.y)
                                 | (to[:, 0] > np.maximum(ti[:, 0], 0.)))
    for start in range(0, len(indices), dev.batch_size):
        selected = indices[start:start+dev.batch_size]
        n = len(selected)
        o = np.zeros((dev.batch_size, 3)); o[:n] = origins[selected]
        d = np.zeros_like(o); d[:, 1] = -1.; d[:n] = direction[selected]
        p = np.ones(dev.batch_size); p[:n] = p0[selected]
        out = dev.batch_kernel(jax.random.fold_in(key, start), jnp.asarray(o), jnp.asarray(d), jnp.asarray(p),
                        jnp.arange(dev.batch_size) < n)
        o, d, offset, result = jax.device_get(out)
        bar, ti, to, pin, path, loss, vis, keep, overflow = result
        if overflow:
            raise RuntimeError(f"JAX crossing capacity {dev.capacity} exceeded; increase capacity, no table published")
        it, rank = np.nonzero(keep)
        ix = (it, rank)
        rows.append((selected[it], bar[ix], ti[ix], to[ix], pin[ix], path[ix], loss[ix], vis[ix],
                     o[it], d[it], offset[it]))
    if rows:
        it, bar, ti, to, pin, path, loss, vis, o, d, offset = (np.concatenate([r[k] for r in rows]) for k in range(11))
    else:
        it = bar = np.zeros(0, np.int64)
        ti = to = pin = path = loss = vis = offset = np.zeros(0)
        o = d = np.zeros((0, 3))
    if timers is not None:
        timers["device_and_transfer_s"] = time.perf_counter() - a
    a = time.perf_counter()
    beta = p0[it] / np.sqrt(p0[it]**2 + eloss.MMU**2)
    t0 = mu.t.to_numpy()[it] + (ti + offset) / (beta * nt.C_MM_PER_NS)
    steps = pd.DataFrame({k: mu[k].to_numpy()[it] for k in tables.KEY})
    steps["nstep"] = np.bincount(it, minlength=len(mu))[it]
    steps["bar"], steps["sp"] = bar.astype(np.int64), mukey[it]
    steps["t0"], steps["t1"] = t0, t0 + path / (beta * nt.C_MM_PER_NS)
    pos0, pos1 = o + d * ti[:, None], o + d * to[:, None]
    for j, c in enumerate("xyz"):
        steps[f"{c}0"], steps[f"{c}1"], steps[f"p{c}0"] = pos0[:, j], pos1[:, j], d[:, j] * pin
    steps["p1"] = ct._p_after(pin, loss)
    steps["len"], steps["edep"] = path, vis
    parts = pd.DataFrame({k: mu[k].to_numpy() for k in tables.KEY})
    parts["key"], parts["parent"], parts["pdg"], parts["isprim"] = mukey, -1, mu.pdg.to_numpy(), 1
    if timers is not None:
        timers["table_assembly_s"] = time.perf_counter() - a
    return steps, parts
