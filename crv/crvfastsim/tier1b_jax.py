"""Experimental strict-v3 device sampler; NumPy reference and defaults stay unchanged."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import tier1b as T


def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jax, jnp


def _bucket(n):
    return max(1, 1 << (max(int(n), 1) - 1).bit_length())


def _pad(a, size, value=0):
    a = np.asarray(a)
    return np.pad(a, [(0, size - len(a))] + [(0, 0)] * (a.ndim - 1), constant_values=value)


def _array_frame(frame):
    _, j = _jax()
    return {c: j.asarray(frame[c].to_numpy()) for c in frame.columns}


def _vectors(d, cols):
    _, j = _jax()
    return j.stack([d[c] for c in cols], axis=1)


def _unit(a):
    _, j = _jax()
    return a / j.maximum(j.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def _rotate(v, a, b):
    _, j = _jax()
    cross = j.cross(a, b)
    c = j.sum(a * b, axis=1, keepdims=True)
    return v * c + j.cross(cross, v) + cross * j.sum(cross * v, axis=1, keepdims=True) / j.maximum(1 + c, 1e-9)


def _fit(a, b, c, h):
    _, j = _jax()
    mid = (a + b) * 0.5
    shift = j.clip(mid, c - h + T.INSET_MM, c + h - T.INSET_MM) - mid
    return j.clip(a + shift, c - h, c + h), j.clip(b + shift, c - h, c + h)


def _bar(q, g):
    _, j = _jax()
    hit = j.all(j.abs(q[:, None, :] - g["center"][None]) <= g["half"][None], axis=2)
    return j.where(j.any(hit, axis=1), j.argmax(hit, axis=1), -1).astype(j.int64)


def _slot(lid, width, g):
    _, j = _jax()
    code = lid * 10000 + width
    i = j.minimum(j.searchsorted(g["slotcode"], code), len(g["slotcode"]) - 1)
    return j.where((g["slotcode"][i] == code) & (width >= 0) & (width < 10000), g["slotorder"][i], -1)


def _lean(u, ax):
    _, j = _jax()
    s = j.sign(j.take_along_axis(u, ax[:, None], axis=1)[:, 0])
    return j.where(s == 0, 1., s)


def _at(a, ax):
    _, j = _jax()
    return j.take_along_axis(a, ax[:, None], axis=1)[:, 0]


def _set_axis(a, ax, v):
    _, j = _jax()
    return a.at[j.arange(len(a)), ax].set(v)


def _select(code, want, order, u):
    _, j = _jax()
    lo = j.searchsorted(code, want, side="left")
    hi = j.searchsorted(code, want, side="right")
    idx = j.minimum(lo + (u * (hi - lo)).astype(j.int64), len(order) - 1)
    return order[idx], lo, hi


def _main_select(lib, bins, ctx, u):
    _, j = _jax()
    pick, lo, hi = _select(lib["code"], bins * T.N_CTX + ctx, lib["order"], u)
    fb, a, b = _select(lib["bin"], bins, lib["order"], u)
    few = hi - lo < T.MIN_CTX_CROSSINGS
    return j.where(few, fb, pick), j.where(few, a, lo), j.where(few, b, hi)


def _uv_select(lib, cells, u):
    _, j = _jax()
    pick, lev = j.full(len(cells), -1, j.int64), j.full(len(cells), -1, j.int64)
    for k in reversed(range(T.N_UV_LEVELS)):
        p, lo, hi = _select(lib[f"code{k}"], cells[:, k], lib[f"order{k}"], u)
        use = (hi - lo >= T.MIN_UV_CROSSINGS) | ((lev < 0) & (hi > lo))
        pick, lev = j.where(use, p, pick), j.where(use, k, lev)
    return pick, lev


def _main_place(ld, lx, xgeo, fx, visits, g, rows, ix, picks):
    _, j = _jax()
    d = {c: v[rows] for c, v in ld.items()}
    xid = picks[ix]
    lu = _vectors(lx, ["ux", "uy", "uz"])[xid]
    fu = _unit(fx["mom"])[ix]
    entry, xb = fx["entry"][ix], fx["bar"][ix]
    d0, d1 = _vectors(d, ["dx0", "dy0", "dz0"]), _vectors(d, ["dx1", "dy1", "dz1"])
    mo = _vectors(d, ["px0", "py0", "pz0"])
    r0, r1, mom = entry + _rotate(d0, lu, fu), entry + _rotate(d1, lu, fu), _rotate(mo, lu, fu)
    same = d["samebar"] == 1
    cl, cf = xgeo["exit"][xid] - xgeo["entry"][xid], fx["exit"][ix] - entry
    ll, lf = d["chord_ll"], j.linalg.norm(cf, axis=1)
    ul, uf = _vectors(d, ["chord_ux", "chord_uy", "chord_uz"]), cf / j.maximum(lf, 1e-12)[:, None]
    a = d["chord_a"]
    an = j.where(a < 0, a, j.where(a > ll, lf + a - ll, a * lf / j.maximum(ll, 1e-12)))
    s0 = an[:, None] * uf + _rotate(_vectors(d, ["chord_perpx", "chord_perpy", "chord_perpz"]), ul, uf)
    s1 = s0 + _rotate(_vectors(d, ["chord_deltax", "chord_deltay", "chord_deltaz"]), ul, uf)
    chord = same & (ll > 1e-6) & (lf > 1e-6)
    r0 = j.where(chord[:, None], entry + s0, r0)
    r1 = j.where(chord[:, None], entry + s1, r1)
    mom = j.where(chord[:, None], _rotate(mo, ul, uf), mom)
    bar = _bar((r0 + r1) * .5, g)
    target = g["center"][xb] + _vectors(d, ["dcx", "dcy", "dcz"])
    l0, l1 = _vectors(d, ["lx0", "ly0", "lz0"]), _vectors(d, ["lx1", "ly1", "lz1"])
    tb = _bar(target, g)
    xa = g["axis"][xb, 2]
    no_bar = tb < 0
    bad = no_bar | (g["axis"][j.maximum(tb, 0), 2] != xa)
    sx = _at(entry - g["center"][xb], xa) - d["entrylen"]
    q0, q1 = target + l0, target + l1
    q0, q1 = _set_axis(q0, xa, _at(q0, xa) + sx), _set_axis(q1, xa, _at(q1, xa) + sx)
    pb = _bar((q0 + q1) * .5, g)
    point = bad & (pb >= 0)
    tb, bad = j.where(point, pb, tb), bad & ~point
    want = fx["mu"][ix] * 100000 + fx["vis"][ix] + d["tvis"].astype(j.int64)
    vi = j.minimum(j.searchsorted(visits["key"], want), len(visits["key"]) - 1)
    found = (visits["key"][vi] == want) & (fx["vis"][ix] + d["tvis"] >= 0) & (d["tvis"] != T.TOPO_NONE)
    lid, wmin, wmax = visits["lid"][vi], visits["wmin"][vi], visits["wmax"][vi]
    ref = _slot(lid, wmin, g)
    ax = g["axis"][j.maximum(ref, 0)]
    sw, sl = _lean(fu, ax[:, 1]), _lean(fu, ax[:, 2])
    k = j.rint(d["tside"] * sw).astype(j.int64)
    tq = _slot(lid, j.where(k > 0, wmax + k, wmin + k), g)
    topology = found & (ref >= 0) & (tq >= 0)
    def topo_end(end):
        q = g["center"][j.maximum(tq, 0)]
        q = _set_axis(q, ax[:, 0], _at(q, ax[:, 0]) + d["tt" + end])
        q = _set_axis(q, ax[:, 1], _at(q, ax[:, 1]) + d["tw" + end] * sw)
        return _set_axis(q, ax[:, 2], _at(visits["entry"][vi], ax[:, 2]) + d["tl" + end] * sl)
    q0, q1 = j.where(topology[:, None], topo_end("0"), q0), j.where(topology[:, None], topo_end("1"), q1)
    tb, bad = j.where(topology, tq, tb), bad & ~topology
    delta = target[:, None, :] - g["center"][None]
    dist = j.sum(j.where(j.arange(3)[None, None, :] != xa[:, None, None], delta * delta, 0.), axis=2)
    allowed = g["axis"][None, :, 2] == xa[:, None]
    nb = j.argmin(j.where(allowed, dist, j.inf), axis=1)
    nb = j.where(j.any(allowed, axis=1), nb, -1)
    tb = j.where(bad, nb, tb)
    tbs = j.maximum(tb, 0)
    la = g["axis"][tbs, 2]
    shift = _at(entry - g["center"][xb], la) - d["entrylen"]
    o0, o1 = g["center"][tbs] + l0, g["center"][tbs] + l1
    o0, o1 = _set_axis(o0, la, _at(o0, la) + shift), _set_axis(o1, la, _at(o1, la) + shift)
    pt = point | topology
    o0, o1 = j.where(pt[:, None], q0, o0), j.where(pt[:, None], q1, o1)
    midpoint = (o0 + o1) * .5
    inside = j.all(midpoint >= g["center"][tbs] - g["half"][tbs], axis=1) & j.all(midpoint <= g["center"][tbs] + g["half"][tbs], axis=1)
    off_end = (tb >= 0) & ~inside
    o0, o1 = _fit(o0, o1, g["center"][tbs], g["half"][tbs])
    r0, r1 = j.where(same[:, None], r0, o0), j.where(same[:, None], r1, o1)
    bar, mom = j.where(same, bar, tb), j.where(same[:, None], mom, mo)
    clamp = same & (bar != xb)
    for axidx in (0, 1):
        aa = g["axis"][xb, axidx]
        rr0 = _set_axis(r0, aa, _at(g["center"][xb] + l0, aa))
        rr1 = _set_axis(r1, aa, _at(g["center"][xb] + l1, aa))
        r0, r1 = j.where(clamp[:, None], rr0, r0), j.where(clamp[:, None], rr1, r1)
    a0, a1 = _fit(r0, r1, g["center"][xb], g["half"][xb])
    r0, r1 = j.where(clamp[:, None], a0, r0), j.where(clamp[:, None], a1, r1)
    bar = j.where(clamp, xb, bar)
    escaped = same & (bar == xb) & j.any((j.abs(r0 - g["center"][xb]) > g["half"][xb]) |
                                                       (j.abs(r1 - g["center"][xb]) > g["half"][xb]), axis=1)
    a0, a1 = _fit(r0, r1, g["center"][xb], g["half"][xb])
    r0, r1 = j.where(escaped[:, None], a0, r0), j.where(escaped[:, None], a1, r1)
    return dict(r0=r0, r1=r1, mom=mom, bar=bar, clamp=clamp, chord=chord, other=~same,
                no_bar=no_bar & ~same, axis=(~no_bar) & (g["axis"][j.maximum(_bar(target, g), 0), 2] != xa) & ~same,
                point=point & ~same, topology=topology & ~same, off_end=off_end & ~same)


def _uv_place(uvd, uvx, fx, visits, g, rows, ix, picks):
    _, j = _jax()
    d = {c: v[rows] for c, v in uvd.items()}
    a0, a1 = _vectors(d, ["x0", "y0", "z0"]), _vectors(d, ["x1", "y1", "z1"])
    shift = fx["entry"][ix] - uvx[picks[ix]]
    q0, q1 = a0 + shift, a1 + shift
    tb = _bar((q0 + q1) * .5, g)
    tbs, lb = j.maximum(tb, 0), d["bar"].astype(j.int64)
    want = fx["mu"][ix] * 1000000 + g["lid"][tbs]
    vi = j.minimum(j.searchsorted(visits["vkey"], want), len(visits["vkey"]) - 1)
    ok = (tb >= 0) & (g["lid"][tbs] == g["lid"][lb]) & (visits["vkey"][vi] != want)
    bar = j.where(ok, tb, lb)
    r0, r1 = _fit(j.where(ok[:, None], q0, a0), j.where(ok[:, None], q1, a1), g["center"][bar], g["half"][bar])
    return dict(r0=r0, r1=r1, mom=_vectors(d, ["px0", "py0", "pz0"]), bar=bar, translated=ok)


def _closest(pos, u, lo, hi):
    jax, j = _jax()
    u = _unit(u)
    def one(low, high):
        par = u == 0
        inside = (pos >= low) & (pos <= high)
        t1, t2 = (low - pos) / u, (high - pos) / u
        tn = j.max(j.where(par, j.where(inside, -j.inf, j.inf), j.minimum(t1, t2)), axis=1)
        tf = j.min(j.where(par, j.where(inside, j.inf, -j.inf), j.maximum(t1, t2)), axis=1)
        enters = (tn <= tf) & (tf >= 0)
        corners = j.stack(j.meshgrid(*[j.stack([low[k], high[k]]) for k in range(3)], indexing="ij"), axis=-1).reshape(8, 3)
        proj = j.sum((corners[None] - pos[:, None, :]) * u[:, None, :], axis=2)
        a, b = j.maximum(j.min(proj, axis=1), 0.), j.maximum(j.max(proj, axis=1), 0.)
        def f(t):
            q = pos + t[:, None] * u
            return j.linalg.norm(j.maximum(j.maximum(low - q, 0.), q - high), axis=1)
        r = (j.sqrt(5.) - 1) / 2
        c, d = b - r * (b - a), a + r * (b - a)
        def body(_, state):
            a, b, c, d, fc, fd = state
            left = fc < fd
            a, b = j.where(left, a, c), j.where(left, d, b)
            cn, dn = b - r * (b - a), a + r * (b - a)
            return a, b, j.where(left, cn, d), j.where(left, c, dn), j.where(left, f(cn), fd), j.where(left, fc, f(dn))
        a, b, *_ = jax.lax.fori_loop(0, T.NM_GOLDEN_ITER, body, (a, b, c, d, f(c), f(d)))
        tm = (a + b) * .5
        return j.where(enters, 0., f(tm)), j.where(enters, j.maximum(tn, 0.), tm)
    dist, tb = jax.vmap(one)(lo, hi)
    dist, tb = dist.T, tb.T
    g = j.where(j.any(dist == 0, axis=1), j.argmin(j.where(dist == 0, tb, j.inf), axis=1), j.argmin(dist, axis=1))
    return g, dist[j.arange(len(pos)), g], tb[j.arange(len(pos)), g]


class DeviceRandom:
    """Independent JAX uniform stream with optional recording for reference replay."""
    def __init__(self, seed, record=False):
        jax, _ = _jax()
        self.key = jax.random.key(int(seed))
        self.draws = [] if record else None

    def random(self, n):
        jax, j = _jax()
        self.key, key = jax.random.split(self.key)
        out = jax.random.uniform(key, (int(n),), dtype=j.float64)
        if self.draws is not None:
            self.draws.append(np.asarray(out))
        return out


class ReplayRandom:
    """Shape-checked fixed uniforms, usable by both the reference and device sampler."""
    def __init__(self, draws):
        self.draws, self.index = draws, 0

    def random(self, n):
        if self.index >= len(self.draws):
            raise ValueError("replay exhausted")
        out = self.draws[self.index]
        self.index += 1
        if np.shape(out) != (n,):
            raise ValueError(f"replay draw shape {np.shape(out)} != {(n,)}")
        return out


def _expand(frame, key, picks):
    vals = frame[key].to_numpy()
    a, z = np.searchsorted(vals, picks, "left"), np.searchsorted(vals, picks, "right")
    count = np.where(picks >= 0, z - a, 0)
    ix = np.repeat(np.arange(len(picks)), count)
    rows = np.repeat(a - np.r_[0, np.cumsum(count)[:-1]], count) + np.arange(count.sum()) if len(count) else np.zeros(0, np.int64)
    return ix, rows


def _out(fx, d, ix, values, parent_col=None):
    out = pd.DataFrame({k: fx[k].to_numpy()[ix] for k in T.KEY})
    out["bar"], out["lsp"], out["xi"], out["pdg"] = values["bar"], d.sp.to_numpy(), ix, d.pdg.to_numpy()
    t0 = fx["t0" if "t0" in fx.columns else "t"].to_numpy()[ix]
    out["t0"], out["t1"] = t0 + d.dt0.to_numpy(), t0 + d.dt1.to_numpy()
    for k, c in enumerate("xyz"):
        out[c + "0"], out[c + "1"], out["p" + c + "0"] = values["r0"][:, k], values["r1"][:, k], values["mom"][:, k]
    for c in ("p1", "len", "edep"):
        out[c] = d[c].to_numpy()
    if parent_col:
        out["mukey"] = fx[parent_col].to_numpy()[ix]
    return out


def _block_fields(values, d, fx, rows, ix, valid, source):
    _, j = _jax()
    out = dict(values, ix=ix, rows=rows, valid=valid & (values["bar"] >= 0))
    out.update({k: fx[k][ix].astype(j.uint32) for k in T.KEY})
    out.update(lsp=d["sp"][rows].astype(j.int64), parent=fx["sp"][ix], src=j.full(len(ix), source, j.int64),
               pdg=d["pdg"][rows].astype(j.int32), t0=fx["t0"][ix] + d["dt0"][rows],
               t1=fx["t0"][ix] + d["dt1"][rows])
    out.update({c: d[c][rows].astype(j.float64) for c in ("p1", "len", "edep")})
    return out


def _block_frame(values):
    out = pd.DataFrame({c: values[c] for c in T.KEY + ["bar", "lsp", "pdg", "t0", "t1", "p1", "len", "edep"]})
    out["xi"] = values["ix"]
    for k, c in enumerate("xyz"):
        out[c + "0"], out[c + "1"], out["p" + c + "0"] = values["r0"][:, k], values["r1"][:, k], values["mom"][:, k]
    return out


def _main_block(csr, plan, offset, lanes, ld, lx, xgeo, fx, visits, g, picks):
    from .tier1b_csr_jax import block
    ix, rows, valid = block(csr, plan, offset, lanes)
    values = _main_place(ld, lx, xgeo, fx, visits, g, rows, ix, picks)
    return _block_fields(values, ld, fx, rows, ix, valid, 0)


def _uv_block(csr, plan, offset, lanes, uvd, uvx, fx, visits, g, picks):
    from .tier1b_csr_jax import block
    ix, rows, valid = block(csr, plan, offset, lanes)
    values = _uv_place(uvd, uvx, fx, visits, g, rows, ix, picks)
    return _block_fields(values, uvd, fx, rows, ix, valid, 1)


class DeviceTier1b:
    """Prepared strict-v3 device metadata/CSR/placement; final ancestry publication stays on the host."""
    def __init__(self, lib, geom, batch_size=2048):
        tick = time.perf_counter()
        jax, j = _jax()
        required = {"xgeo", "uvdesc", "uvxing", "nmtmpl", "nmsteps", "nmreach"}
        if not required <= set(lib) or not set(T.TOPO_COLS) <= set(lib["libdesc"].columns):
            raise ValueError("device strict_v3 needs checked topology and v3 template sidecars")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.lib, self.geom, self.batch_size = lib, geom, int(batch_size)
        self.ld = lib["libdesc"].sort_values("xid", kind="stable").reset_index(drop=True)
        self.uvd = lib["uvdesc"].sort_values("xid", kind="stable").reset_index(drop=True)
        self.lx = lib["libxing"].sort_values("xid", kind="stable").reset_index(drop=True)
        if not np.array_equal(self.lx.xid, np.arange(len(self.lx))):
            raise ValueError("device library needs contiguous xid")
        if self.lx.duplicated("xid").any():
            raise ValueError("duplicate library xid")
        lx = lib["libxing"].sort_values(["bin", "ctx"], kind="stable")
        self.support = {"code": j.asarray(lx.bin.to_numpy() * T.N_CTX + lx.ctx.to_numpy()),
                        "bin": j.asarray(lx.bin.to_numpy()), "order": j.asarray(lx.xid.to_numpy())}
        self.uvsupport = {f"{kind}{k}": j.asarray(a) for k, pair in enumerate(lib["uvxing"]["levels"])
                          for kind, a in zip(("code", "order"), pair)}
        xid = self.ld.xid.to_numpy().astype(np.int64)
        cl = (lib["xgeo"]["exit"] - lib["xgeo"]["entry"])[xid]
        ll = np.linalg.norm(cl, axis=1)
        ul = cl / np.maximum(ll, 1e-12)[:, None]
        d0 = self.ld[["dx0", "dy0", "dz0"]].to_numpy()
        self.ld["chord_ll"] = ll
        self.ld["chord_a"] = (d0 * ul).sum(1)
        perp = d0 - self.ld.chord_a.to_numpy()[:, None] * ul
        delta = self.ld[["dx1", "dy1", "dz1"]].to_numpy() - d0
        for k, c in enumerate("xyz"):
            self.ld["chord_u" + c] = ul[:, k]
            self.ld["chord_perp" + c] = perp[:, k]
            self.ld["chord_delta" + c] = delta[:, k]
        self.dld, self.dlx, self.duvd = _array_frame(self.ld), _array_frame(self.lx), _array_frame(self.uvd)
        self.xgeo = {k: j.asarray(v) for k, v in lib["xgeo"].items() if k in ("entry", "exit")}
        self.uvx = j.asarray(lib["uvxing"]["entry"])
        lid, width = T.layer_slots(geom)
        code, order = T._slot_lookup(geom)
        self.g = {"center": j.asarray(geom.center), "half": j.asarray(geom.halflen),
                  "axis": j.asarray(geom.local_to_world, dtype=j.int64), "lid": j.asarray(lid),
                  "width": j.asarray(width), "slotcode": j.asarray(code), "slotorder": j.asarray(order)}
        self.boxes = tuple(j.asarray(a) for a in T.group_boxes(geom))
        from . import tier1b_metadata_jax as metadata
        self.metadata_constants = metadata.prepare(geom, lib["libbins"])
        self.crossing_metadata = jax.jit(metadata.metadata)
        from . import tier1b_csr_jax as csr
        self.main_csr = csr.prepare(self.ld.xid.to_numpy(), len(self.lx), ~self.ld.uv.to_numpy())
        self.uv_csr = csr.prepare(self.uvd.xid.to_numpy(), len(lib["uvxing"]["entry"]))
        from . import tier1b_hard_jax as hard
        self.hard = hard.prepare(lib["hard"]) if "hard" in lib else None
        self.hard_patterns, self.hard_choose, self.hard_redraw = jax.jit(hard.patterns), jax.jit(hard.choose), jax.jit(hard.redraw)
        self.csr_layout = jax.jit(csr.layout)
        self.main_block, self.uv_block = jax.jit(_main_block), jax.jit(_uv_block)
        self.main_select = jax.jit(_main_select)
        self.uv_select = jax.jit(_uv_select)
        self.main_place, self.uv_place, self.closest = jax.jit(_main_place), jax.jit(_uv_place), jax.jit(_closest)
        jax.block_until_ready((self.support, self.g, self.dld, self.dlx, self.duvd, self.xgeo, self.uvsupport,
                               self.metadata_constants, self.main_csr, self.uv_csr, self.hard))
        self.preparation_s = time.perf_counter() - tick
        self.device_bytes = sum(a.nbytes for a in jax.tree.leaves((self.support, self.g, self.dld, self.dlx,
                                                                  self.duvd, self.xgeo, self.uvx, self.uvsupport, self.metadata_constants,
                                                                  self.main_csr, self.uv_csr, self.hard)))
        self.last = {}

    def device_metadata(self, arrays):
        """Accept padded crossing buffers without pandas or device-to-host grouping."""
        required = set(T.KEY) | {"sp", "bar", "valid", "entry", "exit", "mom", "t0", "length"}
        if set(arrays) != required or arrays["bar"].ndim != 1 or not len(arrays["bar"]):
            raise ValueError("device crossing buffers require a nonempty padded lane axis and exact fields")
        n = len(arrays["bar"])
        for c, a in arrays.items():
            if a.shape != ((n, 3) if c in ("entry", "exit", "mom") else (n,)):
                raise ValueError(f"device crossing buffer shape mismatch: {c}")
        return self.crossing_metadata(arrays, self.metadata_constants)

    def _metadata(self, fx):
        from .tier1b_metadata_jax import crossing_arrays
        return self.device_metadata(crossing_arrays(fx))

    def plan_descendants(self, arrays, uniforms, metadata=None):
        """Select donors and prepare complete CSR histories, retaining device buffers."""
        xx, vv, meta = self.device_metadata(arrays) if metadata is None else metadata
        if uniforms.shape != meta["valid"].shape:
            raise ValueError("device donor uniforms must match padded crossing lanes")
        picks, lo, hi = self.main_select(self.support, meta["bins"], meta["ctx"], uniforms)
        plan = self.csr_layout(self.main_csr, picks, meta["valid"])
        return dict(picks=picks, lo=lo, hi=hi, plan=plan, metadata=(xx, vv, meta),
                    missing=((hi <= lo) & meta["valid"]).any())

    def plan_unvisited(self, metadata, uniforms):
        """Select complete unvisited histories, with missing cells represented by -1."""
        xx, vv, meta = metadata
        if uniforms.shape != meta["valid"].shape:
            raise ValueError("device donor uniforms must match padded crossing lanes")
        picks, level = self.uv_select(self.uvsupport, meta["cells"], uniforms)
        plan = self.csr_layout(self.uv_csr, picks, meta["valid"])
        return dict(picks=picks, level=level, plan=plan, metadata=metadata)

    def device_blocks(self, selection, component="main"):
        """Yield device placement blocks; only a scalar row count controls host dispatch."""
        _, j = _jax()
        xx, vv, _ = selection["metadata"]
        if component == "main":
            if bool(selection["missing"]):
                raise ValueError("empty library bin")
            kernel, csr = self.main_block, self.main_csr
            args = (self.dld, self.dlx, self.xgeo, xx, vv, self.g)
        elif component == "unvisited":
            kernel, csr = self.uv_block, self.uv_csr
            args = (self.duvd, self.uvx, xx, vv, self.g)
        else:
            raise ValueError(f"unknown device component: {component}")
        total = int(selection["plan"]["total"])
        lanes = j.arange(self.batch_size, dtype=j.int64)
        for start in range(0, total, self.batch_size):
            yield kernel(csr, selection["plan"], j.asarray(start), lanes, *args, selection["picks"])

    def _collect_blocks(self, selection, component):
        jax, _ = _jax()
        chunks = {}
        for out in self.device_blocks(selection, component):
            for c, a in jax.device_get(out).items():
                chunks.setdefault(c, []).append(a)
        total = int(selection["plan"]["total"])
        if not total:
            return dict(bar=np.zeros(0, np.int64), r0=np.zeros((0, 3)), r1=np.zeros((0, 3)),
                        mom=np.zeros((0, 3)), clamp=np.zeros(0, bool), translated=np.zeros(0, bool),
                        ix=np.zeros(0, np.int64), rows=np.zeros(0, np.int64),
                        **{c: np.zeros(0, np.int64 if c in T.KEY + ["lsp", "pdg", "parent", "src"] else np.float64)
                           for c in T.KEY + ["lsp", "pdg", "parent", "src", "t0", "t1", "p1", "len", "edep"]})
        return {c: np.concatenate(a)[:total] for c, a in chunks.items()}

    def _hard_picks(self, selection, rng):
        if self.hard is None:
            raise ValueError("device hard draw needs attach_hard")
        _, j = _jax()
        x = selection["metadata"][0]
        pat = self.hard_patterns(x, self.metadata_constants, self.hard)
        nmu, size = int(pat["nmu"]), len(x["bar"])
        if nmu == 0:
            return selection, dict(fast_muons=0, hard=0, expected=0., no_key=0., redrawn=0, left_hard=0)
        u = j.pad(j.asarray(rng.random(nmu)), (0, size - nmu))
        v = j.pad(j.asarray(rng.random(nmu)), (0, size - nmu))
        picks, redo, info = self.hard_choose(self.hard, x, pat, selection["picks"], u, v)
        redrawn = int(redo.sum())
        for _ in range(50):
            active = int(redo.sum())
            if not active:
                break
            u = j.pad(j.asarray(rng.random(active)), (0, _bucket(active) - active))
            picks, redo = self.hard_redraw(self.hard, self.support["order"], picks,
                                            selection["lo"], selection["hi"], redo, u, active)
        selection = dict(selection, picks=picks, plan=self.csr_layout(self.main_csr, picks, x["valid"]))
        info = {k: float(v) if k in ("expected", "no_key") else int(v) for k, v in info.items()}
        info.update(redrawn=redrawn, left_hard=int(redo.sum()))
        return selection, info

    def descendants(self, fx, rng, trace=None, hard=False):
        _, j = _jax()
        from .tier1b_metadata_jax import crossing_arrays
        arrays = crossing_arrays(fx)
        n = len(fx)
        uniforms = j.pad(j.asarray(rng.random(n)), (0, _bucket(n) - n))
        selection = self.plan_descendants(arrays, uniforms)
        if hard:
            selection, hinfo = self._hard_picks(selection, rng)
        val = self._collect_blocks(selection, "main")
        ix, rows = val["ix"], val["rows"]
        out = _block_frame(val)
        kept = val["bar"] >= 0
        why = {c: int(val.get(c, np.zeros(0)).sum()) for c in ("other", "no_bar", "axis", "off_end", "chord", "point", "topology")}
        if hard:
            why.update({f"hard_{k}": v for k, v in hinfo.items()})
        if trace is not None:
            trace.update(picks=np.asarray(selection["picks"])[:n], lo=np.asarray(selection["lo"])[:n],
                         hi=np.asarray(selection["hi"])[:n], rows=rows, ix=ix, values=val, kept=kept)
        return out[kept].reset_index(drop=True), int((~kept).sum()), int(val["clamp"].sum()), why, selection["metadata"]

    def unvisited(self, fx, rng, metadata):
        _, j = _jax()
        n = len(fx)
        uniforms = j.pad(j.asarray(rng.random(n)), (0, _bucket(n) - n))
        selection = self.plan_unvisited(metadata, uniforms)
        val = self._collect_blocks(selection, "unvisited")
        lev = np.asarray(selection["level"])[:n]
        out = _block_frame(val)
        return out, dict(steps=len(out), translated=int(val["translated"].sum()),
                         level=np.bincount(lev[lev >= 0], minlength=T.N_UV_LEVELS).tolist(), no_cell=int((lev < 0).sum()))

    def showers(self, gen, rng):
        _, j = _jax()
        g = gen[gen.pdg.abs() != T.tables.PRIMARY_MUON_PDG].reset_index(drop=True)
        g["apdg"] = g.pdg.abs()
        g["kebin"] = T._ke_bin((g.e - g.apdg.map(T.MASS).fillna(0.)).to_numpy())
        g = g.merge(self.lib["libgen"], on=["apdg", "kebin"], how="left").fillna({"ngen": 0, "nshower": 0})
        prob = j.where(j.asarray(g.ngen.to_numpy()) > 0, j.asarray(g.nshower.to_numpy()) / j.maximum(j.asarray(g.ngen.to_numpy()), 1), 0.)
        reach = np.asarray(j.asarray(rng.random(len(g))) < prob)
        g = g[reach].reset_index(drop=True)
        ls = self.lib["libshower"].sort_values(["apdg", "kebin"], kind="stable")
        u = j.asarray(rng.random(len(g)))
        if len(g):
            pick, lo, hi = _select(j.asarray(ls.apdg.to_numpy() * 100 + ls.kebin.to_numpy()),
                                    j.asarray(g.apdg.to_numpy() * 100 + g.kebin.to_numpy()), j.asarray(ls.shid.to_numpy()), u)
            if np.any(np.asarray(hi) <= np.asarray(lo)):
                raise ValueError("nonmuon cell reaches without donor shower")
            g["shid"] = np.asarray(pick)
        else:
            g["shid"] = np.zeros(0, np.int64)
        g["k"] = g.groupby(T.KEY).cumcount()
        st = self.lib["libshowersteps"].merge(g[T.KEY + ["shid", "k", "t"]], on="shid")
        st["t0"], st["t1"] = st.t + st.dt0, st.t + st.dt1
        return st

    def nearmiss(self, fast_steps, gen, rng):
        _, j = _jax()
        mu = T.muon_rays(gen)
        crossed = pd.MultiIndex.from_frame(fast_steps[T.KEY].drop_duplicates().astype(np.int64))
        cand = mu[~pd.MultiIndex.from_frame(mu[T.KEY].astype(np.int64)).isin(crossed)].reset_index(drop=True)
        size = _bucket(len(cand))
        group, dist, _ = self.closest(j.asarray(_pad(cand[["x", "y", "z"]].to_numpy(), size)),
                                       j.asarray(_pad(cand[["px", "py", "pz"]].to_numpy(), size)), *self.boxes)
        group, dist = group[:len(cand)], dist[:len(cand)]
        db = j.where(dist <= 0, 0, 1 + j.searchsorted(j.asarray(T.NM_D_EDGES), dist, side="right"))
        kb = j.where(group == 0, j.searchsorted(j.asarray(T.NM_KE_EDGES), j.asarray(cand.ke.to_numpy()), side="right"), 0)
        cell = (group * 10 + db) * 10 + kb
        r = self.lib["nmreach"]
        prob = j.zeros(len(cand), j.float64)
        if len(r):
            rc = j.asarray(r.cell.to_numpy())
            ii = j.minimum(j.searchsorted(rc, cell), len(r) - 1)
            prob = j.where(rc[ii] == cell, j.asarray(r.prob.to_numpy())[ii], 0.)
        sel = np.flatnonzero(np.asarray(j.asarray(rng.random(len(cand))) < prob))
        tt = self.lib["nmtmpl"]
        u = j.asarray(rng.random(len(sel)))
        if len(sel):
            tid, lo, hi = _select(j.asarray(tt.cell.to_numpy()), cell[j.asarray(sel)], j.asarray(tt.tid.to_numpy()), u)
            if np.any(np.asarray(hi) <= np.asarray(lo)):
                raise ValueError("near-miss cell reaches without template")
            tid = np.asarray(tid)
        else:
            tid = np.zeros(0, np.int64)
        ns = self.lib["nmsteps"]
        im, rows = _expand(ns, "tid", tid)
        im = sel[im]
        d = ns.iloc[rows]
        val = dict(bar=d.bar.to_numpy().astype(np.int64), r0=d[["x0", "y0", "z0"]].to_numpy(),
                   r1=d[["x1", "y1", "z1"]].to_numpy(), mom=d[["px0", "py0", "pz0"]].to_numpy())
        out = _out(cand, d, im, val, "mukey")
        return out, dict(candidates=len(cand), reached=len(sel), steps=len(out), expected_reached=float(j.sum(prob)))

    def sample(self, fast_steps, fast_parts, gen, rng, **options):
        required = dict(context=True, contain=True, point=True, topology=True, unvisited=True, nearmiss=True, chord=True)
        if any(options.get(k, False) != v for k, v in required.items()) or any(options.get(k, False) for k in ("mirror", "translate", "follow")):
            raise ValueError("device sampler currently supports only strict_v3 options")
        if options.get("showers", True) is not True:
            raise ValueError("device sampler currently requires strict_v3 showers")
        unknown = set(options) - set(required) - {"showers", "mirror", "translate", "follow", "hard"}
        if unknown:
            raise ValueError(f"unknown device options: {sorted(unknown)}")
        tick = time.perf_counter()
        fx = fast_steps.copy()
        desc, n_drop, n_clamp, why, metadata = self.descendants(fx, rng, hard=options.get("hard", False))
        desc_done = time.perf_counter()
        sh = self.showers(gen, rng)
        shower_done = time.perf_counter()
        sp_of = fx.sp.to_numpy() if "sp" in fx.columns else np.ones(len(fx), np.int64)
        desc["src"], desc["parent"] = 0, sp_of[desc.xi.to_numpy()]
        uv, ui = self.unvisited(fx, rng, metadata)
        uv_done = time.perf_counter()
        nm, ni = self.nearmiss(fx, gen, rng)
        nm_done = time.perf_counter()
        desc = pd.concat([desc, uv.assign(src=1, parent=sp_of[uv.xi.to_numpy()]),
                          nm.drop(columns="mukey").assign(src=2, parent=nm.mukey.to_numpy())], ignore_index=True)
        component_s = time.perf_counter() - tick
        result = _publish(fx, fast_parts, gen, self.lib, desc, sh, n_drop, n_clamp, why,
                          {"unvisited": ui, "nearmiss": ni})
        self.last = dict(component_s=component_s, publication_s=time.perf_counter() - tick - component_s,
                         main_s=desc_done-tick, showers_s=shower_done-desc_done, unvisited_s=uv_done-shower_done,
                         nearmiss_s=nm_done-uv_done, total_s=time.perf_counter() - tick)
        return result


def _publish(fx, fast_parts, gen, lib, desc, sh, n_drop, n_clamp, why, tinfo):
    nmu = fast_parts.groupby(T.KEY, sort=False).key.max().rename("kmax").reset_index() if len(fast_parts) else None
    pk = T.KEY + ["xi", "src", "lsp"]
    u = desc[pk + ["pdg", "parent"]].drop_duplicates(pk).reset_index(drop=True)
    base = (u[T.KEY].merge(nmu, on=T.KEY, how="left").kmax.fillna(1).to_numpy().astype(np.int64)
            if nmu is not None else np.ones(len(u), np.int64))
    u["key"] = u.groupby(T.KEY).cumcount().to_numpy() + 1 + base
    desc = desc.merge(u[pk + ["key"]], on=pk)
    desc["sp"] = desc.key
    dparts = pd.DataFrame({k: u[k].to_numpy() for k in T.KEY})
    dparts["key"], dparts["parent"], dparts["pdg"], dparts["isprim"] = u.key, u.parent, u.pdg, 0
    sh["sp"] = (sh.k + 1) * T.SHOWER_KEY_BASE + sh.sp
    su = sh[T.KEY + ["sp", "pdg", "isroot", "k", "shid"]].drop_duplicates(T.KEY + ["sp"])
    roots = sh[T.KEY + ["k", "shid"]].drop_duplicates()
    rootpdg = lib["libshower"].set_index("shid").apdg
    rparts = pd.DataFrame({k: roots[k].to_numpy() for k in T.KEY})
    rparts["key"] = (roots.k.to_numpy() + 1) * T.SHOWER_KEY_BASE
    rparts["parent"], rparts["isprim"] = -1, 1
    rparts["pdg"] = rootpdg.loc[roots.shid.to_numpy()].to_numpy()
    sh.loc[sh.isroot == 1, "sp"] = (sh.k[sh.isroot == 1] + 1) * T.SHOWER_KEY_BASE
    su = su[su.isroot == 0]
    sparts = pd.DataFrame({k: su[k].to_numpy() for k in T.KEY})
    sparts["key"], sparts["parent"] = su.sp.to_numpy(), (su.k.to_numpy() + 1) * T.SHOWER_KEY_BASE
    sparts["pdg"], sparts["isprim"] = su.pdg.to_numpy(), 0
    cols = list(fx.columns)
    steps = pd.concat([fx[cols], desc.reindex(columns=cols), sh.reindex(columns=cols)], ignore_index=True)
    steps["nstep"] = steps.groupby(T.KEY).bar.transform("size")
    parts = pd.concat([fast_parts, dparts, rparts, sparts], ignore_index=True)
    for c in T.KEY:
        steps[c], parts[c] = steps[c].astype(np.uint32), parts[c].astype(np.uint32)
    for c in ("nstep", "bar", "sp"):
        steps[c] = steps[c].astype(np.int64)
    for c in steps.columns:
        if c not in T.KEY and c not in ("nstep", "bar", "sp"):
            steps[c] = steps[c].astype(np.float64)
    parts = parts.astype({"key": np.int64, "parent": np.int64, "isprim": np.int64, "pdg": np.int32})
    steps = steps.sort_values(T.KEY + ["t0"], kind="stable").reset_index(drop=True)
    return steps, parts, dict(desc_steps=len(desc), desc_dropped=n_drop, desc_clamped=n_clamp, other_bar=why,
                             showers=int(len(roots)), shower_steps=len(sh),
                             nonmu_primaries=int((gen.pdg.abs() != T.tables.PRIMARY_MUON_PDG).sum()), templates=tinfo)
