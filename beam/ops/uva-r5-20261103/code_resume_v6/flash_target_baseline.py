"""Opt-in target-region charged-primary photon BIRTH source, using the CE device navigator.

This is not photon exit flux or a shower simulation. It retains the navigator's
absorbing boundaries and approximations: planar foils without lateral escape,
radial flat-shell IPA/OPA/rings, frozen layer field and midpoint vacuum field.
Delta rays are counted but not transported; photon attenuation, conversions and
secondary cascades are absent. No full detector-physics validity is implied.
"""
import numpy as np
import jax
import jax.numpy as jnp

from mu2efastsim import ebrem
from mu2efastsim import eslab_jax as SJ
from . import cechain as C
from . import cechain_jax as CJ
from . import fieldmap as FM
from . import geometry as G


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _column(value, n, name):
    a = np.asarray(value)
    if a.shape not in ((), (n,)) or a.dtype.kind not in "iuf" or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite, scalar or shape (parents,)")
    return np.broadcast_to(a, (n,))


def _geometry(geom, x, h, dt):
    b = geom.beam
    G.validate_foils(b, h)
    gap = (b.foil_z - b.foil_ht)[1:] - (b.foil_z + b.foil_ht)[:-1]
    if np.any(gap < h):
        raise ValueError("h must not exceed the minimum gap between foil faces")
    for obj in (geom, b):
        for value in vars(obj).values():
            if isinstance(value, (float, int, np.number)) and not np.isfinite(value):
                raise ValueError("geometry scalars must be finite")
    if not (b.ipa_r > 0 and b.ipa_t > 0 and b.ipa_z1 > b.ipa_z0 and
            geom.opa_rin0 > 0 and geom.opa_rin1 > 0 and geom.opa_t > 0 and b.opa_z1 > b.opa_z0 and
            geom.ring_t >= 0 and geom.ring_hl >= 0 and geom.ring_t < b.ipa_r and
            geom.z_up < geom.z_plane and geom.r_wall > 0 and b.tsda_hl >= 0):
        raise ValueError("invalid represented geometry dimensions")
    if geom.ring_t > 0 and geom.ring_hl <= 0:
        raise ValueError("rings require positive half length")
    r, z = C.radial(geom, x), x[:, 2]
    ri = C.opa_rin(geom, z)
    unsupported = ((r >= b.ipa_r) & (r <= b.ipa_r + b.ipa_t) & (z >= b.ipa_z0) & (z <= b.ipa_z1))
    unsupported |= (r >= ri) & (r <= ri + geom.opa_t) & (z >= b.opa_z0) & (z <= b.opa_z1)
    if geom.ring_t > 0:
        for za in (b.ipa_z0, b.ipa_z1):
            unsupported |= (r >= b.ipa_r - geom.ring_t) & (r <= b.ipa_r) & (z >= za) & (z <= za + 2 * geom.ring_hl)
    unsupported |= ((np.abs(z - b.tsda_z) <= b.tsda_hl) & (r >= b.tsda_rin) & (r <= b.tsda_rout))
    unsupported |= ((z <= geom.z_up) | (z >= geom.z_plane) | (r >= geom.r_wall) |
                    ((z < geom.z_ds) & (r >= geom.r_ts)) |
                    ((z >= geom.coll5_z0) & (z <= geom.coll5_z1) & (r >= geom.coll5_rin)))
    if np.any(unsupported):
        raise ValueError("starts must lie in represented vacuum or foils, not absorbers or navigator boundaries")
    geometry_values = [np.asarray(v, dt) for obj in (geom, b) for v in vars(obj).values()
                       if isinstance(v, (float, int, np.number, np.ndarray))]
    if not all(np.all(np.isfinite(v)) for v in geometry_values):
        raise ValueError("geometry exceeds the requested dtype's finite range")
    eps = 1e-7 if dt == np.dtype(np.float64) else 2e-3
    if np.any(2 * b.foil_ht <= 2 * eps):
        raise ValueError("foil thickness is unresolved at the requested navigator dtype")


def _field(field, dt):
    if isinstance(field, (tuple, list)) and field and isinstance(field[0], FM.FieldMap):
        maps = []
        for fm in field:
            if not isinstance(fm, FM.FieldMap):
                raise ValueError("field maps must all be FieldMap objects")
            lo, d, n, values = map(np.asarray, (fm.lo, fm.d, fm.n, fm.B))
            if (lo.shape != (3,) or d.shape != (3,) or n.shape != (3,) or n.dtype.kind not in "iu" or
                    np.any(n < 2) or np.any(d <= 0) or values.shape != tuple(n) + (3,) or
                    not all(np.all(np.isfinite(a)) for a in (lo, d, values))):
                raise ValueError("malformed field map")
            if not all(np.all(np.isfinite(np.asarray(a, dt))) for a in (lo, d, values)):
                raise ValueError("field map exceeds the requested dtype's finite range")
            maps.append(FM.FieldMap(jnp.asarray(lo, dt), jnp.asarray(d, dt), jnp.asarray(n, jnp.int32),
                                    jnp.asarray(values, dt)))
        return tuple(maps), None
    const = np.asarray(field, float)
    if const.shape != (3,) or not np.all(np.isfinite(const)) or not np.all(np.isfinite(np.asarray(const, dt))):
        raise ValueError("field must be a finite constant 3-vector or nonempty FieldMap sequence")
    return (), const


def target_source(key, geom, field, x, u, p, t, pdg, *, capacity=64, max_steps=10000,
                  h=1.0, s_max=30000.0, dtype=np.float64, msc=True, layer_field=True,
                  shell_curvature=True):
    """Collect bounded global-frame births for e-/e+ starts in mm, MeV, ns.

    Each species is one fixed device batch, with no refill or per-parent host loop.
    Output rows retain input indices. Records have shape (parents, capacity) with
    trailing xyz for position/direction, validity masks and mechanism codes
    1=brems, 2=in-flight annihilation, 3=at-rest annihilation. Count includes dropped
    births; overflow counts dropped records. nann counts annihilation photons (two
    per annihilation), ndelta counts omitted delta rays, globally across crossings.
    `state['t']` and `records['time']` are relative times on device; `t0` is kept
    separately as host float64, never cast to float32.
    Within a slab the birth time is cumulative slab path divided by c times the
    mean of entry and current parent beta (floor 1e-6), as for CE layer-exit time;
    this is an approximate slowing-time convention, not a time integral.

    Optional vacuum steps are limited to 0.05 rad using entry and midpoint field
    magnitudes. This reduces missed low-momentum turns, but does not certify all
    grazing/re-crossings or rapidly varying maps. `nadapt` counts shortened steps.
    max_steps is a strict device iteration cap; s_max retains the CE path/looper
    boundary. Already stopped (core 1 keV kinetic cutoff) positrons emit once at
    start, including vacuum starts, using the core at-rest convention. Inputs above
    the core 1 GeV kinetic table limit are rejected. State at an iteration cap is
    the current partial-slab state; crossing counters count completed layers only.
    """
    capacity = _positive_integer(capacity, "capacity")
    max_steps = _positive_integer(max_steps, "max_steps")
    if max_steps > np.iinfo(np.int32).max // 3 or capacity > np.iinfo(np.int32).max // 3:
        raise ValueError("capacity/max_steps exceeds device integer range")
    if not all(np.isscalar(a) and np.isfinite(a) and a > 0 for a in (h, s_max)):
        raise ValueError("h and s_max must be finite and positive")
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    if dt == np.dtype(np.float64) and not jax.config.x64_enabled:
        raise ValueError("float64 source requires JAX_ENABLE_X64=1")
    x, u = np.asarray(x, float), np.asarray(u, float)
    if (x.ndim != 2 or x.shape[1:] != (3,) or not x.shape[0] or u.shape != x.shape or
            not np.all(np.isfinite(x)) or not np.all(np.isfinite(u))):
        raise ValueError("x and u must be nonempty finite (parents, 3) arrays")
    n = x.shape[0]
    norm = np.linalg.norm(u, axis=1, keepdims=True)
    if not np.all(np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError("directions must have finite nonzero norms")
    u = u / norm
    p, t, pdg = (_column(a, n, name) for a, name in ((p, "p"), (t, "t"), (pdg, "pdg")))
    if np.any(p <= 0) or np.any(~np.isin(pdg, (11, -11))):
        raise ValueError("p must be positive and pdg must be e- (11) or e+ (-11)")
    _geometry(geom, x, float(h), dt)
    maps, const = _field(field, dt)
    if not all(hasattr(SJ, name) for name in ("lane_step_emissions", "initial_annihilation_photons")):
        raise RuntimeError("Core full photon-birth interface is required")
    k0 = G.foil_index(geom.beam, x[:, 0], x[:, 1], x[:, 2], tol=0.0).astype(np.int32)
    kc = np.clip(k0, 0, geom.beam.foil_z.size - 1)
    zl = np.where(u[:, 2] >= 0, x[:, 2] - (geom.beam.foil_z[kc] - geom.beam.foil_ht[kc]),
                  geom.beam.foil_z[kc] + geom.beam.foil_ht[kc] - x[:, 2])
    zl = np.clip(zl, 0.0, 2 * geom.beam.foil_ht[kc])
    T = p * (p / (np.hypot(p, CJ.ME) + CJ.ME))
    if not all(np.all(np.isfinite(np.asarray(a, dt))) for a in (x, T)) or np.any(np.asarray(T, dt) <= 0):
        raise ValueError("starts exceed the requested dtype's finite positive range")
    if np.any(T > ebrem.T_MAX):
        raise ValueError("kinetic energy exceeds the core photon model table limit")
    merged, records = {}, {}
    iterations = jnp.zeros(2, jnp.int32)
    fields = CJ.OUT_FIELDS + ("nph", "eph", "nde", "ede", "nan", "ean", "cont", "nadapt")
    for species, code in enumerate((11, -11)):
        ids = np.flatnonzero(pdg == code)
        if not ids.size:
            continue
        init, run = CJ._build(geom, dt, float(h), float(s_max), bool(msc), bool(layer_field),
                              bool(shell_curvature), None, False, const,
                              positron=code == -11, photon_capacity=capacity)
        species_key = jax.random.fold_in(key, species)
        st = init(maps, jnp.asarray(x[ids], dt), jnp.asarray(u[ids], dt), jnp.asarray(T[ids], dt),
                  jnp.zeros(ids.size, dt), jnp.asarray(k0[ids]), jnp.asarray(zl[ids], dt), species_key)
        steps, st = run(maps, species_key, st, jnp.int32(max_steps), ids.size, ids.size)
        iterations = iterations.at[species].set(steps)
        partial = st["inl"] & (st["fate"] == C.LIVE)
        lp, dn = st["slab"]["x"], st["slab"]["d"]
        xg = st["xent"] + lp[:, :1] * st["e1"] + lp[:, 1:2] * st["e2"] + lp[:, 2:3] * st["en"]
        ug = dn[:, :1] * st["e1"] + dn[:, 1:2] * st["e2"] + dn[:, 2:3] * st["en"]
        path = st["slab"]["path"]
        bm = jnp.maximum(0.5 * (CJ._beta(st["Tin"]) + CJ._beta(st["slab"]["T"])), 1e-6)
        st = dict(st, x=jnp.where(partial[:, None], xg, st["x"]),
                  u=jnp.where(partial[:, None], ug, st["u"]),
                  T=jnp.where(partial, st["slab"]["T"], st["T"]),
                  s=st["s"] + jnp.where(partial, path, 0.0),
                  t=st["t"] + jnp.where(partial, path / (C.C_MM_NS * bm), 0.0))
        for name, value in {**{k: st[k] for k in fields}, "count": st["count"]}.items():
            if name not in merged:
                merged[name] = jnp.zeros((n,) + value.shape[1:], value.dtype)
            merged[name] = merged[name].at[ids].set(value)
        for name, value in st["records"].items():
            if name not in records:
                records[name] = jnp.zeros((n,) + value.shape[1:], value.dtype)
            records[name] = records[name].at[ids].set(value)
    count = merged.pop("count")
    return dict(state=merged, records=records, count=count, overflow=jnp.maximum(count - capacity, 0),
                capped=merged["fate"] == C.LIVE, loopers=merged["fate"] == C.LOOPER,
                parent_index=jnp.arange(n), pdg=jnp.asarray(pdg, jnp.int32), t0=np.array(t, np.float64),
                nann=merged["nan"], ndelta=merged["nde"],
                diagnostics=dict(iterations=iterations, adaptive_steps=merged["nadapt"],
                                 delta_cascades_omitted=merged["nde"],
                                 annihilation_photons=merged["nan"]))


def validate_complete(result):
    """Reject lost records or unfinished/looper parents; not a physics-validity certificate."""
    overflow, capped, loopers = (np.asarray(result[k]) for k in ("overflow", "capped", "loopers"))
    if np.any(overflow) or np.any(capped) or np.any(loopers):
        raise ValueError(f"Incomplete birth source: dropped births={int(overflow.sum())}, "
                         f"capped parents={int(capped.sum())}, loopers={int(loopers.sum())}")
    return result
