"""Shared Mu2e target-region geometry for CE and flash device navigators.

These helpers contain no transport loop, random draws, recording or scheduling.
The current frame and bracketed-shell conventions retain the slab approximation;
using this module does not enable finite charged-foil exits or lateral re-entry.
"""
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax
import numpy as np

from mu2efastsim import eioni
from mu2efastsim import eslab_jax as SJ
from . import cechain as C
from . import fieldmap as FM
from . import geometry as G

ME = eioni.ME
NCOUNT = 4


class TargetKernels(NamedTuple):
    """Static geometry arrays and traceable callbacks; not a mutable track state."""
    ftype: object
    face_up: object
    face_dn: object
    frin: object
    frout: object
    nf: int
    radial: object
    opa_rin: object
    bisect: object
    frames: object


def layer_types(geom):
    """Material/thickness types, foil lengths, shell types and crossing-counter indices."""
    b = geom.beam
    Lf = np.unique(2 * b.foil_ht)
    specs = [(G.STOPPING_TARGET_AL, float(L), *C.CUTS[G.STOPPING_TARGET_AL.name]) for L in Lf]
    counter = [0] * len(Lf)
    t_ipa = len(specs)
    specs.append((G.IPA_PE, float(b.ipa_t), *C.CUTS[G.IPA_PE.name]))
    counter.append(1)
    t_opa = len(specs)
    specs.append((C.POLYETHYLENE096, float(geom.opa_t), *C.CUTS[C.POLYETHYLENE096.name]))
    counter.append(2)
    t_ring = -1
    if geom.ring_t > 0:
        t_ring = len(specs)
        specs.append((C.CORRUGATED_PP, float(geom.ring_t), *C.CUTS[C.CORRUGATED_PP.name]))
        counter.append(3)
    return specs, Lf, t_ipa, t_opa, t_ring, np.array(counter, np.int32)


def device_maps(headers, dtype=np.float64):
    """G4BL headers as device FieldMaps in dtype; the first containing map wins."""
    out = []
    for hdr in headers:
        fm = FM.load(hdr)
        out.append(FM.FieldMap(jnp.asarray(fm.lo, dtype), jnp.asarray(fm.d, dtype), jnp.asarray(fm.n, jnp.int32),
                               jnp.asarray(fm.B, dtype)))
    return tuple(out)


def field(maps, const, x):
    """Sample the first containing field map, or broadcast a constant field."""
    if not maps:
        return jnp.broadcast_to(jnp.asarray(const, x.dtype), x.shape)
    out = jnp.zeros(x.shape, x.dtype)
    todo = jnp.ones(x.shape[0], bool)
    for fm in maps:
        hi = fm.lo + fm.d * (fm.n - 1).astype(x.dtype)
        inside = todo & jnp.all((x >= fm.lo) & (x < hi), axis=1)
        out = jnp.where(inside[:, None], FM.field(fm, x), out)
        todo = todo & ~inside
    return out


def momentum(T):
    """Electron/positron momentum in MeV/c from kinetic energy in MeV."""
    return jnp.sqrt(T * (T + 2 * ME))


def beta(T):
    """Electron/positron speed divided by c."""
    g = 1.0 + T / ME
    return jnp.sqrt(jnp.maximum(1.0 - 1.0 / (g * g), 0.0))


def dot(a, b):
    """Per-lane vector dot product."""
    return (a * b).sum(axis=1)


def gather(mask, size):
    """First size selected lanes, clipped read indices, validity and selected mask."""
    n = mask.shape[0]
    idx = jnp.nonzero(mask, size=size, fill_value=n)[0]
    ok = idx < n
    chosen = jnp.zeros(n, bool).at[idx].set(True, mode="drop")
    return idx, jnp.minimum(idx, n - 1), ok, chosen


def geometry_params(geom, dtype, foil_lengths, nf_pad=None):
    """The numbers the kernels read, from a geometry: foil arrays in dtype (NumPy) and Python scalars.

    nf_pad: foil count of the arrays (default: the geometry's). Padding foils have faces at +inf (the sorted face
    searches never select them) and zero radii; the kernels only read foils below ``nf``. Differences of geometry
    numbers are taken in float64 here, as the kernels took them before, so a traced copy of these values (cast to
    dtype) gives the same arithmetic as the constants.
    """
    b = geom.beam
    nf = int(b.foil_z.size)
    m = nf if nf_pad is None else int(nf_pad)
    if m < nf:
        raise ValueError(f"geometry_params: nf_pad {m} below the foil count {nf}")

    def pad(a, fill, kind=dtype):
        a = np.asarray(a)
        return np.concatenate([a, np.full(m - nf, fill, a.dtype)]).astype(kind)
    return dict(ftype=pad(np.searchsorted(foil_lengths, 2 * b.foil_ht), 0, np.int32),
                face_up=pad(b.foil_z - b.foil_ht, np.inf), face_dn=pad(b.foil_z + b.foil_ht, np.inf),
                frin=pad(b.foil_rin, 0.0), frout=pad(b.foil_rout, 0.0), nf=nf, ax=float(b.axis_x),
                opa_z0=float(b.opa_z0), opa_dz=float(b.opa_z1 - b.opa_z0), opa_rin0=float(geom.opa_rin0),
                opa_drin=float(geom.opa_rin1 - geom.opa_rin0))


def build(geom, dtype, foil_lengths, *, charge, const_field, layer_field=True, shell_curvature=True, unroll=False):
    """Close over geometry and static policy without adding device dispatch or RNG.

    Frame origins for foils lie on an axial face. The caller supplies the actual
    local start depth. Shell intersection uses an already selected bracket; it
    is not a general earliest-root solver. Candidate selection and unsupported
    geometry handling remain responsibilities of the calling navigator.
    """
    g = geometry_params(geom, dtype, foil_lengths)
    g.update({k: jnp.asarray(g[k]) for k in ("ftype", "face_up", "face_dn", "frin", "frout")})
    return kernels(g, dtype, len(foil_lengths), charge=charge, const_field=const_field, layer_field=layer_field,
                   shell_curvature=shell_curvature, unroll=unroll)


def kernels(g, dtype, n_foil_types, *, charge, const_field, layer_field=True, shell_curvature=True, unroll=False):
    """The kernels of build() over the numbers g of geometry_params: device constants and Python scalars (build),
    or traced arrays of the same values, so one compiled navigator serves every geometry of the same array sizes.
    n_foil_types: layer types below it are foils."""
    dt, q = dtype, charge
    ftype, face_up, face_dn, frin, frout = (g[k] for k in ("ftype", "face_up", "face_dn", "frin", "frout"))
    nf, ax = g["nf"], g["ax"]

    def W(c, a, b_):
        return jnp.where(c, a, b_).astype(dt)

    def radial(x):
        return jnp.hypot(x[:, 0] - ax, x[:, 1])

    def opa_rin(z):
        f = jnp.clip((z - g["opa_z0"]) / g["opa_dz"], 0.0, 1.0)
        return (g["opa_rin0"] + f * g["opa_drin"]).astype(dt)

    def bisect(x, u, p, B, s_hi, target):
        def f(s):
            xs = SJ.helix(x, u, p, q, B, s)[0]
            return radial(xs) - target(xs[:, 2])

        def it(_, c):
            lo, flo, hi = c
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            same = jnp.sign(fm) == jnp.sign(flo)
            return jnp.where(same, mid, lo), jnp.where(same, fm, flo), jnp.where(same, hi, mid)
        lo = jnp.zeros_like(s_hi)
        return lax.fori_loop(0, 40, it, (lo, f(lo), s_hi), **SJ._unroll(unroll))[2]

    def frames(maps, x, u, T, ty, kf):
        """Local basis, face origin, direction and frozen field for a layer entry."""
        n = x.shape[0]
        foil = ty < n_foil_types
        sg = W(u[:, 2] >= 0, 1.0, -1.0)
        zero, one = jnp.zeros(n, dt), jnp.ones(n, dt)
        fe1 = jnp.stack([one, zero, zero], axis=1)
        fe2 = jnp.stack([zero, sg, zero], axis=1)
        fn = jnp.stack([zero, zero, sg], axis=1)
        kc = jnp.clip(kf, 0, nf - 1)
        fentry = x.at[:, 2].set(jnp.where(sg > 0, face_up[kc], face_dn[kc]))
        rv = jnp.stack([x[:, 0] - ax, x[:, 1], zero], axis=1)
        rn = rv / jnp.maximum(jnp.linalg.norm(rv, axis=1, keepdims=True), 1e-30)
        outward = ((x[:, 0] - ax) * u[:, 0] + x[:, 1] * u[:, 1]) > 0
        sn = jnp.where(outward[:, None], rn, -rn)
        se1 = jnp.stack([zero, zero, one], axis=1)
        se2 = jnp.cross(sn, se1)
        e1 = jnp.where(foil[:, None], fe1, se1)
        e2 = jnp.where(foil[:, None], fe2, se2)
        nn = jnp.where(foil[:, None], fn, sn)
        entry = jnp.where(foil[:, None], fentry, x)
        B = field(maps, const_field, x) if layer_field else jnp.zeros((n, 3), dt)
        if shell_curvature:
            sgn = W(outward, 1.0, -1.0)
            dBz = sgn * dot(u, se2) * momentum(T) / (SJ.K_FIELD * jnp.maximum(radial(x), 1e-30))
            B = B.at[:, 2].add(W(foil, 0.0, dBz if q == -1.0 else -q * dBz))

        def loc(v):
            return jnp.stack([dot(v, e1), dot(v, e2), dot(v, nn)], axis=1)
        return e1, e2, nn, entry, loc(u), loc(B)

    return TargetKernels(ftype, face_up, face_dn, frin, frout, nf, radial, opa_rin, bisect, frames)
