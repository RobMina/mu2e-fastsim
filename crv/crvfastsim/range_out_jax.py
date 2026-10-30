"""Device twin of the default kinetic-loss, substep and absorber-range rules."""
import jax
import jax.numpy as j
from jax import lax

from mu2efastsim import eloss as E, eloss_jax as ej, materials as M
from . import eloss_transport_jax as sampling, transport as T

jax.config.update("jax_enable_x64", True)


def table(mat):
    """Prepare the exact NumPy reference interpolation knots once, not per event."""
    import numpy as np
    T._range_loss(mat, np.zeros(0), np.zeros(0))
    return tuple(j.asarray(a) for a in T._RANGE_TABLES[getattr(mat, "name", id(mat))])


def range_loss(p, length, knots, normal=None):
    tin = j.sqrt(p * p + E.MMU ** 2) - E.MMU
    tg, rg = knots
    r0 = j.interp(tin, tg, rg)
    if normal is not None:
        r0 *= j.maximum(1 + T.RANGE_STRAGGLE * normal, 0.)
    rout = r0 - length
    return tin - j.where(rout > 0, j.interp(rout, rg, tg), 0.)


def delta_rate(p, cut):
    _, beta2, _, _, tmax = ej.kinematics(p, E.MMU)
    tot = j.sqrt(p * p + E.MMU ** 2)
    t2 = j.where(tmax > cut, tmax, 2 * cut)
    integral = (1 / cut - 1 / t2) - beta2 / tmax * j.log(t2 / cut) + (t2 - cut) / (2 * tot * tot)
    return j.where(tmax > cut, E.TWOPI_MC2_RCL2 * M.POLYSTYRENE.electron_density_per_mm3 / beta2 * integral, 0.)


def delta_sum(key, p, length, cut):
    """Poisson count plus unbounded spin-weight rejection, with no tail/count cap."""
    _, beta2, _, _, tmax = ej.kinematics(p, E.MMU)
    tot = j.sqrt(p * p + E.MMU ** 2)
    kn, ke = jax.random.split(key)
    count = jax.random.poisson(kn, j.maximum(delta_rate(p, cut) * length, 0.), dtype=j.int64)
    def cond(state):
        return j.any(state[0] < count)
    def body(state):
        done, energy, stream = state
        stream, kr, ka = jax.random.split(stream, 3)
        shape = (32,) + p.shape
        r = jax.random.uniform(kr, shape, j.float64)
        value = cut * tmax / (tmax * (1 - r) + cut * r)
        weight = 1 - beta2 * value / tmax + value * value / (2 * tot * tot)
        accept = jax.random.uniform(ka, shape, j.float64) * (1 + tmax * tmax / (2 * tot * tot)) <= weight
        take = accept & (j.cumsum(accept, axis=0) <= (count - done)[None])
        return done + j.sum(take, axis=0), energy + j.sum(j.where(take, value, 0.), axis=0), stream
    return lax.while_loop(cond, body, (j.zeros_like(count), j.zeros_like(p), ke))[1]


def absorber_loss(key, mat, p, length, knots, straggling=True):
    ka, kr = jax.random.split(key)
    naive = ej.dedx_restricted(mat, p, 1e9) * length
    dl = sampling.sample_loss(ka, mat, p, length, 1e9) if straggling else naive
    rl = range_loss(p, length, knots)
    tin = j.sqrt(p * p + E.MMU ** 2) - E.MMU
    dl *= rl / j.maximum(naive, 1e-300)
    normal = jax.random.normal(kr, p.shape, j.float64)
    return j.where(rl > T.RANGE_NEAR * tin, range_loss(p, length, knots, normal), dl)


def bar_loss(keys, p, length, cut):
    """Entry draw plus ten running-momentum substeps when mean loss exceeds 10% of T."""
    from .transport_jax import p_after
    main, sub, delta = keys
    tin = j.sqrt(p * p + E.MMU ** 2) - E.MMU
    dep = sampling.sample_loss(main, M.POLYSTYRENE, p, length, cut)
    dk = delta_sum(jax.random.fold_in(delta, 0), p, length, cut)
    stops = dep + dk >= tin
    near = ej.dedx_restricted(M.POLYSTYRENE, p, 1e9) * length > T.SUBSTEP_FRACTION * tin
    ds = length / T.SUBSTEP_N
    def substeps():
        def body(k, state):
            total, deposited, plen, done = state
            go = near & ~done
            pc = j.maximum(p_after(p, total), 1.)
            step = j.where(go, ds, 0.)
            dl = sampling.sample_loss(jax.random.fold_in(sub, k), M.POLYSTYRENE, pc, step, cut)
            kd = dl + delta_sum(jax.random.fold_in(delta, k + 1), pc, step, cut)
            rest = tin - total
            out = go & (kd >= rest)
            frac = j.where(out, rest / j.maximum(kd, 1e-300), 1.)
            return (total + j.where(go, j.minimum(kd, rest), 0.), deposited + j.where(go, dl * frac, 0.),
                    j.where(out, (k + frac) * ds, plen), done | out)
        return lax.fori_loop(0, T.SUBSTEP_N, body, (j.zeros_like(p), j.zeros_like(p), length, j.zeros(p.shape, bool)))
    total, subdep, plen, done = lax.cond(j.any(near), substeps,
                                       lambda: (j.zeros_like(p), j.zeros_like(p), length, j.zeros(p.shape, bool)))
    dep, dk = j.where(near, subdep, dep), j.where(near, total - subdep, dk)
    stops = j.where(near, done, stops)
    length = j.where(near & done, plen, length)
    ordinary_stop = stops & ~(near & done)
    share = dep / j.maximum(dep + dk, 1e-300)
    dep = j.where(ordinary_stop, tin * share, dep)
    dk = j.where(ordinary_stop, tin * (1 - share), dk)
    stop_path = tin / ej.dedx_restricted(M.POLYSTYRENE, p, 1e9)
    length = j.where(ordinary_stop, j.minimum(length, stop_path), length)
    return dep, dk, length, stops


def make_kernel(dev, air_loss, absorber_straggling):
    from .crossings_jax import crossings, slab
    from .transport_jax import p_after
    ab, knots = dev.absorbers, dev.range_knots
    @jax.jit
    def kernel(key, o, d, p, offset, active):
        bar, ti, to, valid, overflow = crossings(o, d, dev.lo, dev.hi, dev.capacity, active)
        valid &= active[:, None]
        path = to - ti
        air = ej.dedx_restricted(M.AIR, p, 1e9)[:, None] * (ti + (0. if dev.stage == "C" else offset[:, None]))
        if dev.stage != "C" and not air_loss:
            air = j.zeros_like(ti)
        gap = j.zeros_like(ti)
        if ab is not None:
            ai, ao = slab(o, d, ab[0], ab[1])
            prev = j.concatenate((j.full((len(o), 1), -j.inf), to[:, :-1]), axis=1)
            def gap_one(a, b):
                return j.maximum(j.minimum(ao, b[:, None]) - j.maximum(ai, a[:, None]), 0.).sum(1)
            gap = jax.vmap(gap_one, in_axes=(1, 1), out_axes=1)(prev, ti)
        def step(state, inputs):
            carried, live = state
            k, b, a, ln, gl, ok = inputs
            pa = p_after(p, a + carried)
            if ab is not None:
                hit = live & ok & (pa > 1.) & (gl > 0.)
                dl = lax.cond(j.any(hit),
                              lambda: absorber_loss(jax.random.fold_in(key, 4*k), ab[2], j.maximum(pa, 1.),
                                                    j.where(hit, gl, 0.), knots, absorber_straggling),
                              lambda: j.zeros_like(pa))
                carried += j.where(hit, dl, 0.)
            ph = p_after(p, a + carried)
            go = live & ok & (ph > 1.)
            keys = tuple(jax.random.fold_in(key, 4*k+i) for i in (1, 2, 3))
            dep, dk, ln, stop = lax.cond(j.any(go),
                        lambda: bar_loss(keys, j.maximum(ph, 1.), j.where(go, ln, 0.), dev.cut[b]),
                        lambda: (j.zeros_like(ph), j.zeros_like(ph), ln, j.zeros(ph.shape, bool)))
            dep, dk = j.where(go, dep, 0.), j.where(go, dk, 0.)
            return (carried + dep + dk, live & (~ok | ((ph > 1.) & ~stop))), (ph, ln, dep, dk, go & (dep > 0))
        inputs = (j.arange(bar.shape[1]), bar.T, air.T, path.T, gap.T, valid.T)
        _, out = lax.scan(step, (j.zeros_like(p), j.ones(len(p), bool)), inputs)
        pin, path, dep, dk, keep = (a.T for a in out)
        vis = dep / (1 + M.POLYSTYRENE.birks_mm_per_MeV * dep / j.where(path > 0, path, 1.))
        return bar, ti, to, pin, path, dep + dk, vis, keep, overflow
    return kernel
