"""Bounded single-input raw replay cache; diagnostic infrastructure, not execution admission."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import uproot

from . import frame_registry as F
from .exposure import require_legacy_response
from .replay import EVENT_TYPES, IDS, PART_TYPES, STEP_TYPES, sha256

TYPES = {"events": EVENT_TYPES, "steps": STEP_TYPES, "particles": PART_TYPES}
DTYPES = {name: np.dtype(list(types.items())) for name, types in TYPES.items()}


@dataclass(frozen=True)
class Limits:
    """Fixed typed payload limits, not a pandas/uproot/process peak-RSS guarantee."""

    max_events: int = 1_000_000
    max_steps: int = 2_000_000
    max_particles: int = 2_000_000
    max_resident_bytes: int = 512 * 1024**2
    chunk_rows: int = 32_768
    max_basket_bytes: int = 16 * 1024**2
    max_selection_bytes: int = 512 * 1024**2

    def __post_init__(self):
        if any(type(value) is not int or not 0 < value <= 2**52 for value in vars(self).values()):
            raise ValueError("limits must be positive integers no larger than 2**52")


@dataclass(frozen=True)
class Selection:
    """Whole frames, raw source-ordered rows, and full input-qualified identities."""

    events: pd.DataFrame
    steps: pd.DataFrame
    particles: pd.DataFrame
    registry_sha256: str
    input_sha256: str
    typed_payload_bytes: int


def resident_bytes(counts):
    """Exact persistent NumPy payload, including orders, offsets, frame IDs and joins."""
    e, s, p = (int(counts[name]) for name in ("events", "steps", "particles"))
    if min(e, s, p) < 0:
        raise ValueError("negative tree count")
    return sum(int(counts[name]) * dtype.itemsize for name, dtype in DTYPES.items()) + 8 * (2*s + p + 3*e + 2)


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA256")


def _bound(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"changed bound input: {path}")


def _index(table, names):
    return pd.MultiIndex.from_arrays([table[name] for name in names], names=list(names))


def _admit(counts, limits):
    for name, count in counts.items():
        if type(count) is not int or count < 0 or count > getattr(limits, f"max_{name}"):
            raise ValueError(f"{name} row cap exceeded or invalid count")
    if not counts["events"]:
        raise ValueError("empty all-event census")
    size = resident_bytes(counts)
    if size > limits.max_resident_bytes:
        raise ValueError(f"resident array byte cap exceeded: {size} > {limits.max_resident_bytes}")
    return size


def _validate_manifest(manifest):
    """Check JSON shape/types before legacy-schema checks or field access."""
    required = {"schema", "raw_times", "output_sha256", "entries", "events",
                "run", "subrun", "first_event"}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise ValueError("manifest must be an object with all required replay fields")
    if not isinstance(manifest["schema"], str) or type(manifest["raw_times"]) is not bool:
        raise ValueError("invalid manifest schema or raw_times type")
    _digest(manifest["output_sha256"])
    entries = manifest["entries"]
    if not isinstance(entries, dict) or set(entries) != set(TYPES):
        raise ValueError("manifest entries must contain events, steps and particles")
    values = [manifest[name] for name in ("events", "run", "subrun", "first_event")]
    if any(type(value) is not int or value < 0 for value in (*values, *entries.values())):
        raise ValueError("manifest counts and event identities must be nonnegative integers")


def _check_tree(tree, name, limits):
    if tree.classname != "TTree":
        raise ValueError("raw replay requires flat TTrees")
    if set(tree.keys()) != set(TYPES[name]):
        raise ValueError(f"unexpected {name} branches")
    for column, dtype in TYPES[name].items():
        branch = tree[column]
        # Only TBranch.postprocess metadata is safe before embedded-payload admission.
        embedded = getattr(branch, "_embedded_baskets", None)
        normal = getattr(branch, "_num_normal_baskets", None)
        if not isinstance(embedded, list) or embedded:
            raise ValueError(f"embedded or unknown ROOT baskets are unsupported: {name}.{column}")
        if type(normal) is not int or normal < 0:
            raise ValueError(f"unsupported ROOT basket metadata: {name}.{column}")
        actual = np.dtype(branch.interpretation.numpy_dtype)
        if actual.newbyteorder("=") != np.dtype(dtype):
            raise ValueError(f"replay branch dtype disagreement: {name}.{column}")
        for basket in range(normal):
            # Normal basket_key() reads headers without decompressing payloads.
            key = branch.basket_key(basket)
            if key.fKeylen + key.fObjlen > limits.max_basket_bytes:
                raise ValueError(f"ROOT basket byte cap exceeded: {name}.{column}")


def _read(tree, name, limits):
    out = np.empty(tree.num_entries, dtype=DTYPES[name])
    offset = 0
    for chunk in tree.iterate(list(TYPES[name]), step_size=limits.chunk_rows, library="np"):
        size = len(chunk[next(iter(TYPES[name]))])
        if size > limits.chunk_rows or offset + size > len(out):
            raise ValueError("ROOT chunk row cap or tree coverage disagreement")
        for column, dtype in TYPES[name].items():
            values = np.asarray(chunk[column])
            if values.shape != (size,) or values.dtype.newbyteorder("=") != np.dtype(dtype):
                raise ValueError(f"replay branch dtype/shape disagreement: {name}.{column}")
            out[column][offset:offset + size] = values
        offset += size
    if offset != len(out):
        raise ValueError("ROOT row coverage disagreement")
    return out


def _order_offsets(event_rows, n):
    order = np.argsort(event_rows, kind="stable")
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(np.bincount(event_rows, minlength=n), out=offsets[1:])
    return order, offsets


def _gather(order, offsets, event_rows):
    starts = offsets[event_rows]
    counts = offsets[event_rows + 1] - starts
    total = int(counts.sum())
    positions = np.arange(total, dtype=np.int64)
    positions += np.repeat(starts - (np.cumsum(counts) - counts), counts)
    return np.sort(order[positions])


class IndexedReplay:
    """One admitted input only; select repeatedly without ROOT reads or event loops."""

    def __init__(self, tables, frame_ids, step_order, step_offsets, particle_order,
                 particle_offsets, step_particle, binding, limits, size):
        self._tables = tables
        self._frame_ids = frame_ids
        self._step_order, self._step_offsets = step_order, step_offsets
        self._particle_order, self._particle_offsets = particle_order, particle_offsets
        self._step_particle = step_particle
        self._binding, self._limits = binding, limits
        self.resident_array_bytes = size
        for array in (*tables.values(), frame_ids, step_order, step_offsets, particle_order,
                      particle_offsets, step_particle):
            array.flags.writeable = False

    @property
    def binding(self):
        return dict(self._binding)

    def close(self):
        """Release this input's cached arrays; previously returned tables remain caller-owned."""
        self._tables = None
        self._frame_ids = self._step_order = self._step_offsets = None
        self._particle_order = self._particle_offsets = self._step_particle = None

    def __enter__(self):
        if self._tables is None:
            raise ValueError("closed replay cache")
        return self

    def __exit__(self, *exc):
        self.close()

    def select(self, selected_registry, *, expected_registry_sha256):
        """Select exact rows of the bound full registry, never rebuild labels from a subset."""
        if self._tables is None:
            raise ValueError("closed replay cache")
        if expected_registry_sha256 != self._binding["registry_sha256"]:
            raise ValueError("selected registry digest mismatch")
        selected = selected_registry
        if list(selected) != list(F.COLUMNS):
            raise ValueError("selected registry columns mismatch")
        for column in (*IDS, "frame_id"):
            if selected[column].dtype.kind not in "iu" or np.any(selected[column] < 0):
                raise ValueError("invalid selected registry integer")
        if (selected.duplicated(list(F.FRAME_KEYS)).any() or selected.frame_id.duplicated().any()
                or not (selected.input_sha256 == self._binding["input_sha256"]).all()):
            raise ValueError("duplicate or foreign selected registry frame")
        events = self._tables["events"]
        event_index = _index(events, IDS)
        er = event_index.get_indexer(pd.MultiIndex.from_frame(selected[list(IDS)]))
        if np.any(er < 0):
            raise ValueError("selected frame outside input census")
        if (not np.array_equal(selected.frame_id, self._frame_ids[er])
                or any(not np.array_equal(selected[name], events[name][er]) for name in ("pbtime", "intensity"))):
            raise ValueError("selected registry identity or exposure mismatch")
        ns = int((self._step_offsets[er + 1] - self._step_offsets[er]).sum())
        npart = int((self._particle_offsets[er + 1] - self._particle_offsets[er]).sum())
        counts = {"events": len(er), "steps": ns, "particles": npart}
        size = sum(n * (DTYPES[name].itemsize + 272 + (8 if name == "steps" else 0)) for name, n in counts.items())
        if size > self._limits.max_selection_bytes:
            raise ValueError("selected typed payload byte cap exceeded")
        sr = _gather(self._step_order, self._step_offsets, er)
        pr = _gather(self._particle_order, self._particle_offsets, er)
        result = {}
        for name, rows in (("events", er), ("steps", sr), ("particles", pr)):
            table = pd.DataFrame.from_records(self._tables[name][rows])
            table["source_row"] = rows
            table["input_sha256"] = self._binding["input_sha256"]
            locations = event_index.get_indexer(pd.MultiIndex.from_frame(table[list(IDS)]))
            table["frame_id"] = self._frame_ids[locations]
            result[name] = table
        result["steps"]["pdg"] = self._tables["particles"]["pdg"][self._step_particle[sr]].astype(np.int64)
        return Selection(**result, registry_sha256=expected_registry_sha256,
                         input_sha256=self._binding["input_sha256"], typed_payload_bytes=size)


def load_input(path, registry, *, expected_registry_sha256, expected_input_sha256,
               expected_manifest_sha256, limits=Limits()):
    """Admit and read one local legacy raw replay, checking hashes before and after reading."""
    for value in (expected_registry_sha256, expected_input_sha256, expected_manifest_sha256):
        _digest(value)
    try:
        registry_digest = F.digest(registry)
    except (AssertionError, KeyError, TypeError) as exc:
        raise ValueError("invalid full registry") from exc
    if registry_digest != expected_registry_sha256:
        raise ValueError("full registry digest mismatch")
    local = registry[registry.input_sha256 == expected_input_sha256]
    if not len(local):
        raise ValueError("input absent from full registry")
    path = Path(path)
    if not path.is_file():
        raise ValueError("requires a local replay file")
    sidecar = path.with_suffix(".json")
    _bound(path, expected_input_sha256)
    _bound(sidecar, expected_manifest_sha256)
    manifest = json.loads(sidecar.read_text())
    _validate_manifest(manifest)
    require_legacy_response(manifest)
    if manifest.get("output_sha256") != expected_input_sha256 or manifest.get("raw_times") is not True:
        raise ValueError("changed input or non-raw replay")
    with uproot.open(path, array_cache=None, object_cache=None,
                     minimal_ttree_metadata=True) as root:
        trees = {name: root[f"dump/{name}"] for name in TYPES}
        counts = {name: int(tree.num_entries) for name, tree in trees.items()}
        size = _admit(counts, limits)
        if counts != manifest["entries"] or counts["events"] != manifest["events"] or counts["events"] != len(local):
            raise ValueError("manifest/tree/full registry census disagreement")
        for name, tree in trees.items():
            _check_tree(tree, name, limits)
        tables = {name: _read(tree, name, limits) for name, tree in trees.items()}
    events, steps, parts = (tables[name] for name in ("events", "steps", "particles"))
    event_index = _index(events, IDS)
    if event_index.has_duplicates:
        raise ValueError("duplicate all-event census identity")
    if (not np.isfinite(events["pbtime"]).all() or not np.isfinite(events["intensity"]).all()
            or np.any(events["intensity"] < 0)):
        raise ValueError("invalid saved event exposure")
    if (not np.all(events["run"] == manifest["run"]) or not np.all(events["subrun"] == manifest["subrun"])
            or not np.array_equal(events["event"], np.arange(manifest["first_event"], manifest["first_event"] + len(events)))):
        raise ValueError("manifest event identity disagreement")
    locations = pd.MultiIndex.from_frame(local[list(IDS)]).get_indexer(event_index)
    if np.any(locations < 0):
        raise ValueError("all-event census outside full registry")
    for name in ("pbtime", "intensity"):
        if not np.array_equal(events[name], local[name].to_numpy()[locations]):
            raise ValueError("all-event census/full registry exposure mismatch")
    frame_ids = local.frame_id.to_numpy()[locations].astype(np.int64)
    for name in STEP_TYPES:
        values = steps[name]
        if (np.isinf(values).any() if name in ("t0", "t1") else not np.isfinite(values).all()):
            raise ValueError(f"invalid raw step value: {name}")
    if np.any(steps["bar"] < 0) or np.any(steps["sp"] <= 0) or np.any(steps["edep"] < 0):
        raise ValueError("invalid raw step identity or energy")
    if (np.any(parts["key"] <= 0) or np.any(parts["pdg"] == 0)
            or np.any(parts["parent"] != -1) or np.any(parts["isprim"] != 1)):
        raise ValueError("invalid synthetic replay particle")
    se = event_index.get_indexer(_index(steps, IDS))
    pe = event_index.get_indexer(_index(parts, IDS))
    if np.any(se < 0) or np.any(pe < 0):
        raise ValueError("step or particle outside all-event census")
    particle_index = _index(parts, (*IDS, "key"))
    if particle_index.has_duplicates:
        raise ValueError("duplicate particle identity")
    step_particle = particle_index.get_indexer(_index(steps, (*IDS, "sp")))
    if np.any(step_particle < 0):
        raise ValueError("missing particle identity for raw step")
    step_order, step_offsets = _order_offsets(se, len(events))
    particle_order, particle_offsets = _order_offsets(pe, len(events))
    _bound(path, expected_input_sha256)
    _bound(sidecar, expected_manifest_sha256)
    actual = sum(a.nbytes for a in (*tables.values(), frame_ids, step_order, step_offsets,
                                  particle_order, particle_offsets, step_particle))
    if actual != size:
        raise ValueError("resident array accounting disagreement")
    binding = {"path": str(path.resolve()), "input_sha256": expected_input_sha256,
               "manifest_sha256": expected_manifest_sha256, "registry_sha256": expected_registry_sha256,
               "execution_admitted": False}
    return IndexedReplay(tables, frame_ids, step_order, step_offsets, particle_order,
                         particle_offsets, step_particle, binding, limits, size)
