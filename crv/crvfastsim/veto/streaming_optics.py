"""Bounded whole-frame optical batches, without rerunning the frozen smoke campaign.

Diagnostic only: roots bridge full optical keys to legacy response helpers; their
uniqueness is not proof of independent random streams. The caller retains the
complete Selection.events census (including empty frames), binds the underlying
files/registry, and supplies one RootUnion across every batch AND response seed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from . import indexed_replay as I, keyed_optical_smoke as M, keyed_optics as K
from .replay import EVENT_TYPES, IDS, PART_TYPES, STEP_TYPES
from .replay_timing import PERIOD


@dataclass(frozen=True)
class Limits:
    """Admission ceilings, never truncation thresholds or a peak-RSS guarantee."""

    max_frames: int = 4
    max_steps: int = 20_000
    max_total_substeps: int = 2_000_000
    max_substeps: int = 200_000
    max_photons: int = 300_000

    def __post_init__(self):
        for name, ceiling in (("max_frames", 4), ("max_steps", 20_000),
                              ("max_total_substeps", 2_000_000),
                              ("max_substeps", 200_000), ("max_photons", 300_000)):
            _limit(getattr(self, name), name, ceiling)


def _limit(value, name, ceiling):
    if type(value) is not int or not 0 < value <= ceiling:
        raise ValueError(f"{name} must be a positive native int <= {ceiling}")


def _integer(table, name, *, signed=False, ceiling=2**63 - 1):
    values = table[name].to_numpy()
    if (values.dtype.kind not in "iu" or np.any(values > ceiling)
            or np.any(values < (-ceiling if signed else 0))):
        raise ValueError(f"invalid integer identity: {name}")
    return values


def _index(table, names):
    return pd.MultiIndex.from_frame(table[list(names)])


def _selection(selection, digest, limits):
    if not isinstance(selection, I.Selection):
        raise ValueError("requires indexed_replay.Selection")
    if selection.input_sha256 != digest:
        raise ValueError("selection input SHA256 disagreement")
    K.Draws(selection.registry_sha256, 0)
    events, steps, particles = selection.events, selection.steps, selection.particles
    if not 0 < len(events) <= limits.max_frames or len(steps) > limits.max_steps:
        raise ValueError("frame or raw step cap exceeded (or empty census)")
    for name, table, types in (("events", events, EVENT_TYPES), ("steps", steps, STEP_TYPES),
                               ("particles", particles, PART_TYPES)):
        required = {*types, "source_row", "input_sha256", "frame_id"}
        if name == "steps":
            required.add("pdg")
        if (not isinstance(table, pd.DataFrame) or not table.columns.is_unique
                or not required <= set(table)):
            raise ValueError(f"missing or duplicate {name} columns")
        if table.input_sha256.isna().any() or not table.input_sha256.eq(digest).all():
            raise ValueError(f"missing or foreign {name} source input")
        for column in (*IDS, "source_row", "frame_id"):
            _integer(table, column)
        rows = table.source_row.to_numpy()
        if np.any(rows[1:] <= rows[:-1]):
            raise ValueError(f"{name} original source rows must be unique and source ordered")
    event_index = _index(events, IDS)
    if event_index.has_duplicates or events.frame_id.duplicated().any():
        raise ValueError("duplicate all-event identity or frame_id")
    for name in ("pbtime", "intensity"):
        values = events[name].to_numpy()
        if (values.dtype.kind not in "fiu" or not np.isfinite(values).all()
                or (name == "intensity" and np.any(values < 0))):
            raise ValueError("invalid all-event exposure")
    locations = {}
    for name, table in (("steps", steps), ("particles", particles)):
        er = event_index.get_indexer(_index(table, IDS))
        if np.any(er < 0):
            raise ValueError(f"{name} outside complete all-event census")
        if not np.array_equal(table.frame_id, events.frame_id.to_numpy()[er]):
            raise ValueError(f"{name} full frame identity disagreement")
        locations[name] = er
        _integer(table, "pdg", signed=True, ceiling=2**31 - 1)
        if np.any(table.pdg == 0):
            raise ValueError("invalid zero PDG")
    for name in STEP_TYPES:
        values = steps[name].to_numpy()
        if (values.dtype.kind not in "fiu" or
                (np.isinf(values).any() if name in ("t0", "t1") else not np.isfinite(values).all())):
            raise ValueError(f"invalid raw step value: {name}")
    _integer(steps, "bar")
    _integer(steps, "sp")
    _integer(particles, "key")
    _integer(particles, "parent", signed=True)
    _integer(particles, "isprim")
    if (np.any(steps.sp <= 0) or np.any(particles.key <= 0)
            or np.any(particles.parent != -1) or np.any(particles.isprim != 1)):
        raise ValueError("invalid synthetic replay particle convention")
    particle_index = _index(particles, (*IDS, "key"))
    if particle_index.has_duplicates:
        raise ValueError("duplicate full particle identity")
    pr = particle_index.get_indexer(_index(steps, (*IDS, "sp")))
    if np.any(pr < 0) or not np.array_equal(steps.pdg, particles.pdg.to_numpy()[pr]):
        raise ValueError("missing or inconsistent full particle PDG join")
    return locations["steps"]


def make_batch(selection, geom, cfg, tables, deviation, status, *, input_sha256,
               seed, limits=Limits()):
    """Make exactly two keyed optical draws and the unchanged frozen timing table.

    Original source_row and full tuples are authoritative. No timing/status mask
    deletes a step, photon, or event. event_row is a batch-local lookup only;
    source-qualified keys, never that lookup, identify photons across batches.
    """
    if not isinstance(limits, Limits):
        raise ValueError("requires streaming optics Limits")
    K.Draws(input_sha256, seed)
    er = _selection(selection, input_sha256, limits)
    status = np.asarray(status)
    if (status.shape != (geom.nbars * 4,) or status.dtype.kind not in "iu"
            or np.any(status < 0)):
        raise ValueError("requires complete nonnegative integer channel status")
    steps, events = selection.steps, selection.events
    with np.errstate(over="ignore", invalid="ignore"):
        w = -events.pbtime.to_numpy().astype(np.float32).astype(float)[er]
        shift = np.floor((steps.t0.to_numpy() - w) / PERIOD) * PERIOD
        folded = steps.assign(t0=steps.t0 - w - shift, t1=steps.t1 - w - shift)
    # K rejects infinite endpoints and skips NaNs, identically in both calls.
    # Check origins for quiet frames too, not just the frames producing photons.
    with np.errstate(over="ignore"):
        if not np.isfinite(events.pbtime.to_numpy().astype(np.float32)).all():
            raise ValueError("nonfinite float32 bunch origin")

    def call(frame):
        q = K.make_photons(frame, frame.pdg.to_numpy(), geom, cfg, tables, deviation,
                           input_sha256=input_sha256, seed=seed,
                           max_substeps=limits.max_substeps,
                           max_total_substeps=limits.max_total_substeps,
                           max_photons=limits.max_photons)
        if len(q) > limits.max_photons:
            raise ValueError("photon cap exceeded before timing-table allocation")
        return q

    raw, historical = call(steps), call(folded)
    # Fail before allocating the wider timing table, even for changed key types.
    try:
        pd.testing.assert_frame_equal(M.canonical(raw[list(K.KEYS)]),
                                      M.canonical(historical[list(K.KEYS)]), check_exact=True)
    except AssertionError as exc:
        raise ValueError("raw/historical full photon keys disagree") from exc
    return M.timing_table(raw, historical, steps, events, status, input_sha256, seed)


class RootUnion:
    """Bounded sorted root ledger shared across inputs, batches, and seeds.

    Only roots are retained, not per-photon Python objects or a photon-table
    concatenation. M.roots validates each batch's full keys; a repeated root is
    rejected globally whether it represents duplicate keys or a hash collision.
    """

    def __init__(self, *, max_roots=1_800_000):
        _limit(max_roots, "max_roots", 1_800_000)
        self._max_roots = max_roots
        self._roots = np.empty(max_roots, dtype=np.int64)
        self._n = 0

    def add(self, q):
        """Validate then atomically append; failure never changes the ledger."""
        if (not isinstance(q, pd.DataFrame) or not q.columns.is_unique
                or not {*K.KEYS, "photon_id"} <= set(q)):
            raise ValueError("missing or duplicate full photon key/root columns")
        if len(q) > self._max_roots - self._n:
            raise ValueError("global root cap exceeded before append")
        if not q.input_sha256.astype("string").str.fullmatch(r"[0-9a-f]{64}", na=False).all():
            raise ValueError("invalid photon input SHA256")
        _integer(q, "draw_seed", ceiling=2**64 - 1)
        for name in (*K.KEYS[2:], "photon_id"):
            _integer(q, name)
        expected = np.asarray(M.roots(q))
        if (expected.shape != (len(q),) or expected.dtype != np.dtype(np.int64)
                or np.any(expected < 0) or not np.array_equal(q.photon_id, expected)):
            raise ValueError("saved optical root/full key disagreement")
        fresh = np.unique(expected)
        if len(fresh) != len(q):
            raise ValueError("duplicate root or root digest collision within batch")
        if self._n:
            positions = np.searchsorted(self._roots[:self._n], fresh)
            if np.any((positions < self._n) &
                      (self._roots[np.minimum(positions, self._n - 1)] == fresh)):
                raise ValueError("duplicate root or root digest collision across batches/seeds")
        end = self._n + len(fresh)
        self._roots[self._n:end] = fresh
        self._roots[:end].sort()
        self._n = end

    def summary(self):
        """Canonical sorted little-endian signed-int64 bytes, with no extra tag."""
        digest = hashlib.sha256(self._roots[:self._n].astype("<i8", copy=False).tobytes()).hexdigest()
        return {"roots": self._n, "sha256": digest, "collision_free": True,
                "contract": M.ROOT_CONTRACT}
