"""Opt-in local interaction replay and vacuum boundary handoff; docs/component-response.md.

A bank is a library of the shared core (mu2efastsim.library, docs/libraries.md, kind "replay"): save_library writes
it with its manifest (one writer for every bank, including future converters from capture products), load_library
checks the manifest and the content hash and rebuilds the Library, whose constructor re-runs every structural check.
An unsupported entrance stays UnsupportedEntrance, raised before any draw; domain_counts reports the refused rows of
a batch as a DomainCounter without drawing. A complete manifest is necessary, not sufficient, for a bank to be
admissible (docs/component-training.md): the manifest's validation records bank_admissible false.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

C_MM_NS = 299.792458
POSITION_ATOL, DIRECTION_ATOL = 1e-7, 1e-12
STATE_RTOL, STATE_ATOL = 1e-12, 1e-9
ENTRANCE = dict(pdg=(), position=(3,), momentum=(3,), energy=())
INCOMING = dict(event=(3,), track=(), parent=(), **ENTRANCE, time=())
PARTICLES = dict(pdg=(), parent=(), exited=())
STEPS = dict(particle=(), volume=(), x0=(3,), x1=(3,), p0=(3,), p1=(3,), t0=(), t1=(), length=(), edep=())
EXITS = dict(particle=(), position=(3,), momentum=(3,), energy=(), time=())
INTEGER = {"pdg", "event", "track", "parent", "particle", "volume"}


def _integer(value, name, shape=None):
    a = np.asarray(value)
    if a.dtype.kind not in "iu" or (a.dtype.kind == "u" and np.any(a > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} requires signed-int64-compatible integers")
    a = np.array(a, dtype=np.int64, copy=True)
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return a


def _real(value, name, shape=None):
    a = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError(f"invalid finite values or shape for {name}")
    return a


def _record(table, schema):
    if set(table) != set(schema):
        raise ValueError(f"expected columns {tuple(schema)}")
    n = len(np.asarray(table[next(iter(schema))]))
    out = {}
    for key, tail in schema.items():
        if key == "exited":
            a = np.asarray(table[key])
            if a.dtype.kind != "b" or a.shape != (n,):
                raise ValueError("exited must be a boolean vector")
            a = a.copy()
        else:
            a = (_integer if key in INTEGER else _real)(table[key], key, (n, *tail))
        a.flags.writeable = False
        out[key] = a
    return MappingProxyType(out)


def _kinematics(table):
    p = np.linalg.norm(table["momentum"], axis=1)
    if np.any(p <= 0) or np.any(table["energy"] < p - STATE_ATOL - STATE_RTOL * p):
        raise ValueError("propagating particles require nonzero momentum and energy >= |p|")
    return p


@dataclass(frozen=True)
class ComponentType:
    """Identical local geometry/material and physics/cuts; field-free box envelope."""
    geometry: str
    physics: str
    half_size: tuple[float, float, float]
    volumes: int

    def __post_init__(self):
        h = _real(self.half_size, "half_size", (3,))
        v = _integer(self.volumes, "volumes", ())
        if not self.geometry or not self.physics or np.any(h <= POSITION_ATOL) or v <= 0:
            raise ValueError("component requires construction/physics labels, positive sizes and volumes")
        object.__setattr__(self, "half_size", tuple(h))
        object.__setattr__(self, "volumes", int(v))

    def inside(self, points):
        return np.all(np.abs(points) <= np.asarray(self.half_size) + POSITION_ATOL, axis=-1)

    def on_face(self, points):
        """Inside the envelope and on at least one face (within POSITION_ATOL), whatever the direction."""
        on = np.abs(np.abs(points) - np.asarray(self.half_size)) <= POSITION_ATOL
        return self.inside(points) & np.any(on, axis=-1)

    def on_boundary(self, points, momentum, *, entering):
        h = np.asarray(self.half_size)
        on = np.abs(np.abs(points) - h) <= POSITION_ATOL
        direction = momentum / np.linalg.norm(momentum, axis=1)[:, None]
        normal_dot = np.sign(points) * direction
        inward = np.all(~on | (normal_dot < -DIRECTION_ATOL), axis=1) & np.any(on, axis=1)
        outward = np.any(on & (normal_dot > DIRECTION_ATOL), axis=1)
        return self.inside(points) & (inward if entering else outward)


@dataclass(frozen=True)
class Placement:
    """A rigidly placed component with receiving global sensitive/material volume IDs."""
    component: ComponentType
    instance: int
    rotation: np.ndarray
    translation: np.ndarray
    volume_ids: np.ndarray

    def __post_init__(self):
        r = _real(self.rotation, "rotation", (3, 3))
        a = _real(self.translation, "translation", (3,))
        ids = _integer(self.volume_ids, "volume_ids", (self.component.volumes,))
        instance = _integer(self.instance, "instance", ())
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(np.linalg.det(r), 1., atol=1e-12, rtol=0):
            raise ValueError("rotation must be proper orthonormal, not a reflection or scale")
        if instance < 0 or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("instance/volume IDs must be nonnegative and volume IDs unique")
        for key, value in (("rotation", r), ("translation", a), ("volume_ids", ids)):
            value.flags.writeable = False
            object.__setattr__(self, key, value)
        object.__setattr__(self, "instance", int(instance))

    def local_points(self, value):
        return (np.asarray(value) - self.translation) @ self.rotation

    def global_points(self, value):
        return np.asarray(value) @ self.rotation.T + self.translation

    def local_vectors(self, value):
        return np.asarray(value) @ self.rotation

    def global_vectors(self, value):
        return np.asarray(value) @ self.rotation.T


def _offsets(offsets, trials, rows, name):
    a = _integer(offsets, name, (trials + 1,))
    if a[0] != 0 or a[-1] != rows or np.any(np.diff(a) < 0):
        raise ValueError(f"invalid offsets for {name}")
    a.flags.writeable = False
    return a


def _trial_rows(offsets):
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def _gather(offsets, trials):
    counts = np.diff(offsets)[trials]
    owner = np.repeat(np.arange(len(trials)), counts)
    begin = np.repeat(offsets[trials] - np.r_[0, np.cumsum(counts)[:-1]], counts)
    return owner, begin + np.arange(counts.sum()), counts


class Library:
    """Complete equal-weight trials at recorded local entrance states, including null outcomes."""

    def __init__(self, component, entrances, particles, particle_offsets, steps, step_offsets, exits, exit_offsets):
        self.component = component
        self.entrances = _record(entrances, ENTRANCE)
        self.particles, self.steps, self.exits = (_record(t, s) for t, s in
                                                ((particles, PARTICLES), (steps, STEPS), (exits, EXITS)))
        n = len(self.entrances["pdg"])
        if n == 0:
            raise ValueError("library needs at least one incident trial")
        for name, table, offsets in (("particle", self.particles, particle_offsets),
                                     ("step", self.steps, step_offsets), ("exit", self.exits, exit_offsets)):
            setattr(self, name + "_offsets", _offsets(offsets, n, len(table[next(iter(table))]), name))
        _kinematics(self.entrances)
        _kinematics(self.exits)
        if not component.on_boundary(self.entrances["position"], self.entrances["momentum"], entering=True).all():
            raise ValueError("entrances must be inward on the component boundary")
        # an exit is the first boundary crossing as Geant4 records it: on a face, with the post-step direction as
        # recorded. Multiple scattering over the last, boundary-limited step can turn that direction back inward
        # (the particle re-enters at once); next_in_vacuum then hands it back to the same placement at distance 0
        if not component.on_face(self.exits["position"]).all():
            raise ValueError("exits must lie on the component boundary")
        npart = np.diff(self.particle_offsets)
        if np.any(npart == 0):
            raise ValueError("every trial needs its incident root particle")
        pt = _trial_rows(self.particle_offsets)
        local = np.arange(len(pt)) - self.particle_offsets[pt]
        parents = self.particles["parent"]
        if np.any(np.where(local == 0, parents != -1, (parents < 0) | (parents >= local))):
            raise ValueError("local ancestry must have root0 and parents preceding children")
        if np.any(self.particles["pdg"][self.particle_offsets[:-1]] != self.entrances["pdg"]):
            raise ValueError("root species differs from incident species")
        for table, offsets in ((self.steps, self.step_offsets), (self.exits, self.exit_offsets)):
            owner = _trial_rows(offsets)
            if np.any(table["particle"] < 0) or np.any(table["particle"] >= npart[owner]):
                raise ValueError("particle index outside trial")
        sx = self.steps
        if np.any(sx["volume"] < 0) or np.any(sx["volume"] >= component.volumes):
            raise ValueError("local step volume outside component")
        if not component.inside(sx["x0"]).all() or not component.inside(sx["x1"]).all():
            raise ValueError("step endpoint outside component; no clamping is allowed")
        if np.any(sx["edep"] < 0) or np.any(sx["length"] < 0) or np.any(sx["t0"] < 0) or np.any(sx["t1"] < sx["t0"]) or np.any(self.exits["time"] < 0):
            raise ValueError("negative deposits/lengths or invalid relative times")
        # no chord rule: a Geant4 step length is the true path length, while the post-step point carries the
        # multiple-scattering lateral displacement, so |x1 - x0| can exceed the length; both are kept as recorded
        ep = self.particle_offsets[_trial_rows(self.exit_offsets)] + self.exits["particle"]
        exited = np.bincount(ep, minlength=len(pt))
        if np.any(exited != self.particles["exited"].astype(int)):
            raise ValueError("exited particles need exactly one first-exit record")
        exit_time = np.full(len(pt), np.inf)
        exit_time[ep] = self.exits["time"]
        sp = self.particle_offsets[_trial_rows(self.step_offsets)] + sx["particle"]
        if np.any(sx["t1"] > exit_time[sp] + STATE_ATOL):
            raise ValueError("step after first exit is not component-local")


class UnsupportedEntrance(ValueError):
    """Entrance rows outside the bank's explicit state support, never zero-response draws."""

    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.int64)
        super().__init__(f"unsupported local entrance rows: {self.rows.tolist()}")


def _matches(library, local):
    """(query row, donor trial) pairs whose recorded entrance state equals the query's within the state tolerances."""
    n, donors = len(local["pdg"]), library.entrances
    left, right = [], []
    chunk = max(1, 2_000_000 // len(donors["pdg"]))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        good = local["pdg"][start:stop, None] == donors["pdg"][None, :]
        for key in ("position", "momentum", "energy"):
            a, b = local[key][start:stop], donors[key]
            atol, rtol = (POSITION_ATOL, 0.) if key == "position" else (STATE_ATOL, STATE_RTOL)
            close = np.isclose(a[:, None], b[None, :], atol=atol, rtol=rtol)
            good &= np.all(close, axis=-1) if a.ndim == 2 else close
        ii, jj = np.nonzero(good)
        left.append(ii + start)
        right.append(jj)
    left = np.concatenate(left) if left else np.empty(0, dtype=np.int64)
    right = np.concatenate(right) if right else np.empty(0, dtype=np.int64)
    return left, right


def _choose(library, local, rng):
    n = len(local["pdg"])
    left, right = _matches(library, local)
    counts = np.bincount(left, minlength=n)
    if np.any(counts == 0):
        raise UnsupportedEntrance(np.flatnonzero(counts == 0))
    pick = np.cumsum(counts) - counts + (rng.random(n) * counts).astype(np.int64)
    return right[pick]


def _incoming(table):
    incoming = _record(table, INCOMING)
    _kinematics(incoming)
    if np.any(incoming["track"] <= 0) or np.any((incoming["parent"] != -1) & (incoming["parent"] <= 0)):
        raise ValueError("track/parent IDs must be positive or parent=-1")
    if np.any(incoming["parent"] == incoming["track"]):
        raise ValueError("particle cannot parent itself")
    return incoming


@dataclass
class Replay:
    """Correlated transformed particle tree, deposit rows, outgoing states and allocator cursor."""
    particles: dict
    steps: dict
    exits: dict
    trials: np.ndarray
    next_track: int


def replay(library, placement, incoming, interaction, rng, *, next_track):
    """Replay exact supported local entrance states in a rigid placement; no energy/shape interpolation."""
    incoming = _incoming(incoming)
    n = len(incoming["track"])
    interaction = _integer(interaction, "interaction", (n,))
    cursor = int(_integer(next_track, "next_track", ()))
    if placement.component != library.component:
        raise ValueError("incompatible component geometry/material/physics")
    if cursor <= max(incoming["track"].max(initial=0), incoming["parent"].max(initial=0)):
        raise ValueError("next_track must exceed input track/parent IDs and reserve unused IDs")
    key = np.column_stack((incoming["event"], interaction))
    if np.any(interaction < 0) or len(np.unique(key, axis=0)) != n:
        raise ValueError("interaction IDs must be nonnegative and unique within each event")
    local = {**incoming, "position": placement.local_points(incoming["position"]),
             "momentum": placement.local_vectors(incoming["momentum"])}
    valid = library.component.on_boundary(local["position"], local["momentum"], entering=True)
    if not valid.all():
        raise UnsupportedEntrance(np.flatnonzero(~valid))
    trial = _choose(library, local, rng)
    owner, pidx, counts = _gather(library.particle_offsets, trial)
    starts = np.r_[0, np.cumsum(counts)[:-1]] if n else np.empty(0, dtype=np.int64)
    local_id = np.arange(len(pidx)) - starts[owner]
    isroot = local_id == 0
    children = int((~isroot).sum())
    if cursor + children > np.iinfo(np.int64).max:
        raise ValueError("track ID allocator overflow")
    track = np.empty(len(pidx), dtype=np.int64)
    track[isroot] = incoming["track"][owner[isroot]]
    track[~isroot] = cursor + np.arange(children)
    lp = library.particles["parent"][pidx]
    parent = track[starts[owner] + np.maximum(lp, 0)]
    parent[isroot] = incoming["parent"][owner[isroot]]

    def common(ii):
        return dict(event=incoming["event"][ii].copy(), interaction=interaction[ii].copy(),
                    instance=np.full(len(ii), placement.instance, dtype=np.int64))

    particles = dict(**common(owner), track=track, parent=parent,
                     pdg=library.particles["pdg"][pidx].copy(), exited=library.particles["exited"][pidx].copy())
    outputs = []
    for table, offsets, isstep in ((library.steps, library.step_offsets, True), (library.exits, library.exit_offsets, False)):
        ii, idx, _ = _gather(offsets, trial)
        pi = starts[ii] + table["particle"][idx]
        out = dict(**common(ii), track=track[pi], parent=parent[pi], pdg=particles["pdg"][pi])
        if isstep:
            out["volume"] = placement.volume_ids[table["volume"][idx]]
            for name in ("x0", "x1"):
                out[name] = placement.global_points(table[name][idx])
            for name in ("p0", "p1"):
                out[name] = placement.global_vectors(table[name][idx])
            for name in ("t0", "t1"):
                out[name] = table[name][idx] + incoming["time"][ii]
            for name in ("length", "edep"):
                out[name] = table[name][idx].copy()
        else:
            out["position"] = placement.global_points(table["position"][idx])
            out["momentum"] = placement.global_vectors(table["momentum"][idx])
            out["time"] = table["time"][idx] + incoming["time"][ii]
            out["energy"] = table["energy"][idx].copy()
        outputs.append(out)
    return Replay(particles, *outputs, trial, cursor + children)


def next_in_vacuum(particles, placements):
    """Nearest outside-to-box entry in field-free vacuum; return source rows, placement indices and states."""
    particles = _incoming(particles)
    n = len(particles["track"])
    instances = [p.instance for p in placements]
    if len(set(instances)) != len(instances):
        raise ValueError("placement instance IDs must be unique")
    best, which = np.full(n, np.inf), np.full(n, -1, dtype=np.int64)
    ambiguous = np.zeros(n, dtype=bool)
    pmag = np.linalg.norm(particles["momentum"], axis=1)
    for index, placement in enumerate(placements):
        pos = placement.local_points(particles["position"])
        direction = placement.local_vectors(particles["momentum"]) / pmag[:, None]
        h = np.asarray(placement.component.half_size)
        if np.any(np.all(np.abs(pos) < h - POSITION_ATOL, axis=1)):
            raise ValueError("vacuum handoff starts inside a component")
        on = np.abs(np.abs(pos) - h) <= POSITION_ATOL
        pos = np.where(on, np.sign(pos) * h, pos)
        parallel = np.abs(direction) <= DIRECTION_ATOL
        inv = np.divide(1., direction, out=np.zeros_like(direction), where=~parallel)
        a, b = (-h - pos) * inv, (h - pos) * inv
        lo = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=1)
        hi = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=1)
        miss = np.any(parallel & (np.abs(pos) >= h), axis=1)
        distance = np.maximum(lo, 0.)
        valid = ~miss & (hi > distance + POSITION_ATOL) & (lo >= -POSITION_ATOL)
        distance = np.where(valid, distance, np.inf)
        tie = valid & np.isfinite(best) & (np.abs(distance - np.where(np.isfinite(best), best, 0.)) <= POSITION_ATOL)
        nearer = valid & (distance < best - POSITION_ATOL)
        ambiguous = np.where(nearer, False, ambiguous | tie)
        best, which = np.where(nearer, distance, best), np.where(nearer, index, which)
    if ambiguous.any():
        raise ValueError("ambiguous nearest component entry")
    rows = np.flatnonzero(which >= 0)
    out = {key: a[rows].copy() for key, a in particles.items()}
    out["position"] += particles["momentum"][rows] / pmag[rows, None] * best[rows, None]
    out["time"] += best[rows] * particles["energy"][rows] / (pmag[rows] * C_MM_NS)
    return rows, which[rows], out


# ------------------------------------------------------------------ the bank as a library (mu2efastsim.library)
BANK_TABLES = (("entrances", ENTRANCE), ("particles", PARTICLES), ("steps", STEPS), ("exits", EXITS))
OFFSETS = ("particle_offsets", "step_offsets", "exit_offsets")
BANK_PREFIX = "secondaries.component."


def bank_arrays(library):
    """The stored arrays of a bank: the tables under their prefixes, the offsets and the component description."""
    c = library.component
    out = {f"{t}_{k}": np.asarray(getattr(library, t)[k]) for t, schema in BANK_TABLES for k in schema}
    out.update({k: np.asarray(getattr(library, k)) for k in OFFSETS})
    out.update(component_geometry=np.str_(c.geometry), component_physics=np.str_(c.physics),
               component_half_size=np.asarray(c.half_size, dtype=np.float64), component_volumes=np.int64(c.volumes))
    return out


def bank_name(component):
    return BANK_PREFIX + re.sub(r"[^A-Za-z0-9_-]", "_", component.geometry)


def bank_manifest_fields(library, **provenance):
    """Manifest fields of a bank (mu2efastsim.library.manifest); provenance fields not given are unrecorded."""
    c = library.component
    fields = dict(
        name=bank_name(c), subsystem="secondaries", version=1, kind="replay",
        description="Complete equal-weight local interaction trials of one component type at recorded local entrance "
                    "states (null outcomes included): the particle tree, deposit steps and first-boundary exits, "
                    "replayed whole in a rigid placement (replay).",
        axes=[],
        outputs=[dict(name="particles", unit="1", dtype="int64/bool", description="placed particle tree"),
                 dict(name="steps", unit="mm, MeV/c, ns, MeV", dtype="float64",
                      description="deposit rows in global coordinates and receiving volume IDs"),
                 dict(name="exits", unit="mm, MeV/c, MeV, ns", dtype="float64",
                      description="first-boundary exits in global coordinates")],
        dtype=dict(storage={"states": "float64", "ids": "int64"}, compute=["float64"],
                   notes="host replay in float64; there is no device or float32 path"),
        rng=dict(host="numpy.random.Generator: one uniform per incoming row (rng.random(n)) to pick among the "
                      "matching trials; unsupported rows are refused before the draw", device="none",
                 draws_per_row=1),
        domain=dict(coverage="exact recorded local entrance states (species, position, momentum, energy within the "
                             "state tolerances) on the component boundary; anything else is refused "
                             "(UnsupportedEntrance), never interpolated, rescaled or answered with zero",
                    fallbacks=[]),
        geometry=dict(tag=c.geometry, hash=None, physics=c.physics, half_size=list(c.half_size), volumes=c.volumes),
        config_hash=None,
        validation=dict(status="not validated", record=None, bank_admissible=False,
                        reason="a complete manifest is necessary, not sufficient (docs/component-training.md); the "
                               "energy accounting stays unrecorded until the bank carries a capture_accounting report "
                               "hash"),
    )
    fields.update(provenance)
    unrecorded = dict(provenance.get("unrecorded", {}))
    reasons = dict(source="the capture products the bank was built from are not given",
                   generator="physics-list, material and cut hashes are not recorded yet (capture P2b)",
                   release="not given to save_library()", seeds="not given to save_library()",
                   split="not given to save_library()", config_hash="no build configuration manifest")
    for k, why in reasons.items():
        if fields.get(k) is None:
            fields[k] = None
            unrecorded.setdefault(k, why)
    fields["unrecorded"] = unrecorded
    return fields


def save_library(path, library, **provenance):
    """Write a bank with its manifest (mu2efastsim.library.store); returns the manifest."""
    from .library import store
    return store.save(path, bank_arrays(library), bank_manifest_fields(library, **provenance))


def load_library(path):
    """The checked bank: manifest, arrays and content hash (mu2efastsim.library.store.load), then the Library
    constructor's structural checks."""
    from .library import store
    lib = store.load(path)
    if lib.manifest["kind"] != "replay" or not lib.name.startswith(BANK_PREFIX):
        raise store.LibraryError(path, [f"{lib.name}: not a component bank (kind {lib.manifest['kind']!r})"])
    a = lib.arrays
    component = ComponentType(str(a["component_geometry"]), str(a["component_physics"]),
                              tuple(float(x) for x in a["component_half_size"]), int(a["component_volumes"]))
    tables = {t: {k: a[f"{t}_{k}"] for k in schema} for t, schema in BANK_TABLES}
    return Library(component, tables["entrances"], tables["particles"], a["particle_offsets"], tables["steps"],
                   a["step_offsets"], tables["exits"], a["exit_offsets"])


def domain_counts(library, placement, incoming, counter=None):
    """DomainCounter of a batch of incoming states (no draw, nothing raised): rows off the component boundary
    (axis "boundary") and on it without a recorded entrance state (axis "state") are counted as refused."""
    from .library.axes import ABOVE
    from .library.domain import DomainCounter
    incoming = _incoming(incoming)
    n = len(incoming["track"])
    counter = counter or DomainCounter(bank_name(library.component))
    counter.add_lookups(n)
    local = {**incoming, "position": placement.local_points(incoming["position"]),
             "momentum": placement.local_vectors(incoming["momentum"])}
    on = library.component.on_boundary(local["position"], local["momentum"], entering=True)
    matched = np.zeros(n, bool)
    if on.any():
        sub = {k: v[on] for k, v in local.items()}
        left, _ = _matches(library, sub)
        matched[np.flatnonzero(on)[np.unique(left)]] = True
    counter.add_codes("boundary", np.where(on, 0, ABOVE).astype(np.int8))
    counter.add_codes("state", np.where(on & ~matched, ABOVE, 0).astype(np.int8))
    counter.add_refused(int((~(on & matched)).sum()))
    return counter
