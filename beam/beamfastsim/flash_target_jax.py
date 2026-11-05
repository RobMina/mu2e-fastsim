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
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
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


def initialize_source(key, geom, field, x, u, p, t, pdg, *, capacity=64,
                      h=1.0, s_max=30000.0, dtype=np.float64, msc=True,
                      layer_field=True, shell_curvature=True):
    """Initialize fixed species batches, including initial at-rest births exactly once.

    Sessions are Python dictionaries containing runtime closures/device maps. Treat
    them as immutable: use advance_source and import_state to obtain new sessions.
    Only export_state's flat numeric arrays are suitable for allow_pickle=False
    persistence. External input/configuration/geometry/map/core identity must be
    bound by the caller before importing with an exactly matching fresh template.
    """
    capacity = _positive_integer(capacity, "capacity")
    if capacity > np.iinfo(np.int32).max // 3:
        raise ValueError("capacity exceeds device integer range")
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
    batches = {}
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
        batches[code] = dict(ids=np.array(ids, np.int64), key=species_key, state=st, run=run)
    return dict(schema_version=1, capacity=capacity, maps=maps, species=batches,
                layer_count=len(CJ.layer_types(geom)[0]),
                parent_index=jnp.arange(n), pdg=jnp.asarray(pdg, jnp.int32),
                t0=np.array(t, np.float64), iterations=jnp.zeros(2, jnp.int32),
                total_iterations=jnp.zeros(2, jnp.int32))


# Three possible births per step, plus the two initial at-rest photons.
_ITERATION_LIMIT = (np.iinfo(np.int32).max - 2) // 3


def advance_source(session, max_steps):
    """Return a new session after at most max_steps per fixed species batch.

    No lane refill, compaction, RNG reinitialization or result projection occurs.
    Cumulative work is bounded to keep device RNG iterations and counters int32.
    """
    max_steps = _positive_integer(max_steps, "max_steps")
    totals = np.asarray(session["total_iterations"], np.int64)
    if max_steps > _ITERATION_LIMIT or np.any(totals + max_steps > _ITERATION_LIMIT):
        raise ValueError("max_steps/total_iterations exceeds device integer range")
    batches = {}
    iterations = jnp.zeros(2, jnp.int32)
    for species, code in enumerate((11, -11)):
        if code not in session["species"]:
            continue
        batch = session["species"][code]
        n = batch["ids"].size
        steps, st = batch["run"](session["maps"], batch["key"], batch["state"],
                                  jnp.int32(max_steps), n, n)
        batches[code] = dict(batch, state=st)
        iterations = iterations.at[species].set(steps)
    return dict(session, species=batches, iterations=iterations,
                total_iterations=session["total_iterations"] + iterations)


def source_result(session):
    """Read current births and partial-slab primary state, without changing raw state.

    Schema and partial-slab arithmetic match target_source. Diagnostic iterations
    are cumulative, so chunking does not alter result bytes (including diagnostics).
    Missing cascades, attenuation and foil lateral escape remain explicit limitations.
    """
    n, capacity = session["pdg"].size, session["capacity"]
    merged, records = {}, {}
    fields = CJ.OUT_FIELDS + ("nph", "eph", "nde", "ede", "nan", "ean", "cont", "nadapt")
    for code in (11, -11):
        if code not in session["species"]:
            continue
        batch = session["species"][code]
        ids, st = batch["ids"], batch["state"]
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
                parent_index=session["parent_index"], pdg=session["pdg"], t0=session["t0"],
                nann=merged["nan"], ndelta=merged["nde"],
                diagnostics=dict(iterations=session["total_iterations"], adaptive_steps=merged["nadapt"],
                                 delta_cascades_omitted=merged["nde"],
                                 annihilation_photons=merged["nan"]))


def _tree_arrays(tree, prefix):
    # Tree fields, never parents. Full CJ state (including otherwise hidden fields).
    out = {}
    for name, value in tree.items():
        path = prefix + "/" + name
        if isinstance(value, dict):
            out.update(_tree_arrays(value, path))
        else:
            out[path] = np.array(value, copy=True)
    return out


def export_state(session):
    """Copy a complete resumable state into flat numeric, pickle-free arrays.

    Names are schema_version, capacity, parent_index, pdg, t0, iterations,
    total_iterations, species_codes, and species/{PDG}/{ids,key,state/...}.
    iterations is the last advance; total_iterations is cumulative. Both vectors
    use the fixed order (11, -11); absent slots are zero.
    The entire raw CJ tree is stored, never the projected source_result state.
    """
    arrays = {name: np.array(session[name], copy=True) for name in
              ("parent_index", "pdg", "t0", "iterations", "total_iterations")}
    arrays.update(schema_version=np.array(session["schema_version"], np.int32),
                  capacity=np.array(session["capacity"], np.int64),
                  species_codes=np.array([c for c in (11, -11) if c in session["species"]], np.int32))
    for code, batch in session["species"].items():
        prefix = f"species/{code}"
        arrays[prefix + "/ids"] = np.array(batch["ids"], copy=True)
        arrays[prefix + "/key"] = np.array(jax.random.key_data(batch["key"]), copy=True)
        arrays.update(_tree_arrays(batch["state"], prefix + "/state"))
    return arrays


def _restore_tree(tree, arrays, prefix):
    return {name: (_restore_tree(value, arrays, prefix + "/" + name)
                   if isinstance(value, dict) else jnp.asarray(arrays[prefix + "/" + name]))
            for name, value in tree.items()}


def import_state(template_session, arrays):
    """Fail-closed restore using a fresh, exactly matching initialized template.

    Caller must first verify external config/source/geometry/maps/core hashes.
    This checks the complete template-derived schema, array dtypes/shapes, lane
    identity and RNG keys; it never replays maps or reconstructs partial state.
    """
    if np.any(np.asarray(template_session["total_iterations"])):
        raise ValueError("checkpoint template must be freshly initialized")
    expected = export_state(template_session)
    if set(arrays) != set(expected):
        raise ValueError("checkpoint schema has missing or extra arrays")
    data = {}
    for name, reference in expected.items():
        value = arrays[name]
        if (not isinstance(value, np.ndarray) or value.dtype != reference.dtype or
                value.shape != reference.shape or value.dtype.kind not in "biuf" or
                not np.all(np.isfinite(value))):
            raise ValueError(f"checkpoint shape/dtype/finite mismatch: {name}")
        # Own all imported storage; caller mutation must not affect the session.
        data[name] = np.array(value, copy=True)
        if ("/state/" not in name and name not in ("iterations", "total_iterations") and
                data[name].tobytes() != reference.tobytes()):
            raise ValueError(f"checkpoint identity/key/schema mismatch: {name}")
    totals, last = data["total_iterations"], data["iterations"]
    if (np.any(totals < 0) or np.any(totals > _ITERATION_LIMIT) or
            np.any(last < 0) or np.any(last > totals)):
        raise ValueError("checkpoint iteration range mismatch")
    batches = {}
    for species, code in enumerate((11, -11)):
        if code not in template_session["species"]:
            if totals[species] or last[species]:
                raise ValueError("checkpoint absent species iterations")
            continue
        batch = template_session["species"][code]
        prefix = f"species/{code}/state"
        total = int(totals[species])
        raw = {name[len(prefix) + 1:]: value for name, value in data.items() if name.startswith(prefix + "/")}
        if total == 0 and any(data[name].tobytes() != reference.tobytes()
                              for name, reference in expected.items() if name.startswith(prefix + "/")):
            raise ValueError("checkpoint zero-iteration state differs from initialization")
        if np.any(raw["T"] > ebrem.T_MAX) or np.any(raw["Tin"] > ebrem.T_MAX) or np.any(raw["slab/T"] > ebrem.T_MAX):
            raise ValueError("checkpoint energy outside core tables")
        if np.any(np.linalg.norm(raw["u"], axis=1) == 0):
            raise ValueError("checkpoint zero parent direction")
        if np.any(raw["lty"] >= template_session["layer_count"]):
            raise ValueError("checkpoint layer type outside represented tables")
        if int(raw["slab/it"]) != total:
            raise ValueError("checkpoint slab RNG iteration mismatch")
        for name, value in raw.items():
            if value.dtype.kind in "iu" and np.any(value < 0):
                raise ValueError(f"checkpoint negative counter: {name}")
        for name in ("nvac", "nslab", "nstall", "nadapt", "cnt"):
            if np.any(raw[name] > total):
                raise ValueError(f"checkpoint counter exceeds iterations: {name}")
        for name in ("count", "nph", "nan", "nde", "slab/nph", "slab/nan", "slab/nde"):
            if np.any(raw[name].astype(np.int64) > 3 * total + 2):
                raise ValueError(f"checkpoint emission counter overflow: {name}")
        for name in ("T", "t", "s", "Tin", "eph", "ede", "ean", "cont",
                     "slab/T", "slab/path", "slab/pend", "slab/cont", "slab/eph", "slab/ede", "slab/ean"):
            if np.any(raw[name] < 0):
                raise ValueError(f"checkpoint negative state: {name}")
        count = raw["count"].astype(np.int64)
        valid, mechanism = raw["records/valid"], raw["records/mechanism"]
        wanted = np.arange(template_session["capacity"])[None, :] < count[:, None]
        if (not np.array_equal(valid, wanted) or
                np.any(mechanism[valid] < 1) or np.any(mechanism[valid] > 3) or
                np.any(mechanism[~valid] != 0) or
                np.any(count != raw["nph"].astype(np.int64) + raw["nan"]) or
                np.any(raw["nan"] % 2) or np.any(raw["fate"] > C.LOOPER) or
                np.any(raw["nadapt"] > raw["nvac"]) or
                np.any(raw["nvac"].astype(np.int64) + raw["nstall"] > total) or
                np.any(raw["cnt"].astype(np.int64).sum(axis=1) > raw["nslab"])):
            raise ValueError("checkpoint record/counter/fate invariant mismatch")
        for name in ("energy", "position", "direction", "path", "time"):
            if np.any(raw["records/" + name][~valid] != 0):
                raise ValueError("checkpoint nonzero unused record")
        if np.any(raw["records/energy"] < 0) or np.any(raw["records/path"] < 0) or np.any(raw["records/time"] < 0):
            raise ValueError("checkpoint negative record")
        batches[code] = dict(batch, state=_restore_tree(batch["state"], data, prefix))
    return dict(template_session, species=batches, iterations=jnp.asarray(last),
                total_iterations=jnp.asarray(totals))


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
    max_steps = _positive_integer(max_steps, "max_steps")
    if max_steps > _ITERATION_LIMIT:
        raise ValueError("capacity/max_steps exceeds device integer range")
    session = initialize_source(key, geom, field, x, u, p, t, pdg, capacity=capacity,
                                h=h, s_max=s_max, dtype=dtype, msc=msc,
                                layer_field=layer_field, shell_curvature=shell_curvature)
    return source_result(advance_source(session, max_steps))


def validate_complete(result):
    """Reject lost records or unfinished/looper parents; not a physics-validity certificate."""
    overflow, capped, loopers = (np.asarray(result[k]) for k in ("overflow", "capped", "loopers"))
    if np.any(overflow) or np.any(capped) or np.any(loopers):
        raise ValueError(f"Incomplete birth source: dropped births={int(overflow.sum())}, "
                         f"capped parents={int(capped.sum())}, loopers={int(loopers.sum())}")
    return result
