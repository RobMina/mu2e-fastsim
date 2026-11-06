"""Read existing flat detector-step dumps and overlay joint frames before response."""
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

from .sampling import FrameSelection


def _signed_indices(values, name):
    values = ak.to_numpy(ak.flatten(values))
    if values.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer keys")
    if np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds int64")
    return values.astype(np.int64, copy=False)


IDS = ("run", "subrun", "event")
PARTICLE_FIELDS = ("key", "parent", "pdg", "ccode", "scode", "isprim", "genidx",
                   "sx", "sy", "sz", "ex", "ey", "ez", "spx", "spy", "spz", "se",
                   "epx", "epy", "epz", "ee", "st", "et")
GEN_FIELDS = ("idx", "pdg", "genid", "x", "y", "z", "px", "py", "pz", "e", "t")
TRACKER_FIELDS = ("sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp",
                  "eion", "len", "width", "t", "x0", "y0", "z0", "x1", "y1", "z1",
                  "px", "py", "pz")
CRV_FIELDS = ("bar", "sp", "t0", "t1", "edep", "x0", "y0", "z0", "x1", "y1", "z1",
              "px0", "py0", "pz0", "p1", "len")
TIME_FIELDS = {"particles": ("st", "et"), "gen": ("t",), "tracker": ("t",),
               "crv": ("t0", "t1"), "calo": ("t",)}


def _event_keys(ids):
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.shape[1] != 3 or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("event IDs must be nonnegative integer (run, subrun, event) rows")
    return np.rec.fromarrays(ids.T.astype(np.uint64), names=IDS)


def _flat_keys(frame, key):
    return np.rec.fromarrays([frame, key], names=("frame", "key"))


def _check_reference(target, query, name):
    if query.size == 0:
        return
    order = np.argsort(target)
    sorted_target = target[order]
    index = np.searchsorted(sorted_target, query)
    if target.size == 0 or np.any(index == target.size):
        raise ValueError(f"dangling {name} reference")
    if np.any(sorted_target[index] != query):
        raise ValueError(f"dangling {name} reference")


@dataclass(frozen=True)
class FrameBatch:
    """One joint tracker/CRV frame per event, with empty collections retained."""

    event_ids: np.ndarray
    collections: Mapping[str, ak.Array]

    def __post_init__(self):
        ids = np.asarray(self.event_ids)
        keys = _event_keys(ids)
        if len(np.unique(keys)) != len(keys):
            raise ValueError("duplicate event IDs in frame pool")
        if not {"particles", "gen", "tracker", "crv"} <= self.collections.keys():
            raise ValueError("particles, gen, tracker and crv collections are required")
        for name, rows in self.collections.items():
            if name not in TIME_FIELDS or len(rows) != len(ids):
                raise ValueError(f"invalid collection or event count: {name}")
            ak.num(rows, axis=1)
        object.__setattr__(self, "event_ids", ids)

    def validate_references(self):
        """Check event-local SimParticle keys and GenParticle indices before remapping."""
        parts, gen = self.collections["particles"], self.collections["gen"]
        frame = np.repeat(np.arange(len(self.event_ids)), ak.to_numpy(ak.num(parts)))
        key = _signed_indices(parts.key, "particle keys")
        pkeys = _flat_keys(frame, key)
        if np.any(key < 0) or len(np.unique(pkeys)) != len(pkeys):
            raise ValueError("SimParticle keys must be nonnegative and event-unique")
        parent = _signed_indices(parts.parent, "parent keys")
        if np.any(parent < -1):
            raise ValueError("null parent must be -1")
        _check_reference(pkeys, _flat_keys(frame[parent >= 0], parent[parent >= 0]), "parent")
        gi = _signed_indices(parts.genidx, "generator references")
        _signed_indices(gen.idx, "generator indices")
        ngen = ak.to_numpy(ak.num(gen))
        if np.any(gi < -1) or np.any(gi >= ngen[frame]):
            raise ValueError("dangling generator index")
        if not bool(ak.all(gen.idx == ak.local_index(gen, axis=1))):
            raise ValueError("generator indices must follow collection order")
        for name in self.collections.keys() - {"particles", "gen"}:
            rows = self.collections[name]
            event = np.repeat(np.arange(len(self.event_ids)), ak.to_numpy(ak.num(rows)))
            _check_reference(pkeys, _flat_keys(event, _signed_indices(rows.sp, name)), name)


def _group(tree, event_ids, fields):
    arrays = tree.arrays([*IDS, *fields], library="ak")
    ids = np.column_stack([ak.to_numpy(arrays[k]) for k in IDS])
    census = _event_keys(event_ids)
    keys = _event_keys(ids)
    order = np.argsort(census)
    where = np.searchsorted(census[order], keys)
    if len(keys) and (len(census) == 0 or np.any(where == len(census))):
        raise ValueError("step/particle event absent from census")
    if len(keys) and np.any(census[order[where]] != keys):
        raise ValueError("step/particle event absent from census")
    event = order[where]
    row_order = np.argsort(event, kind="stable")
    return ak.unflatten(arrays[list(fields)][row_order], np.bincount(event, minlength=len(census)))


def read_event_log(path):
    """Read the stock EventLister census, including events with no detector steps."""
    matches = re.findall(r"^Event:\s*\d+ run:\s*(\d+) subRun:\s*(\d+) event:\s*(\d+)\s*$",
                         Path(path).read_text(), flags=re.MULTILINE)
    return np.asarray(matches, dtype=np.uint64).reshape(-1, 3)


def read_frames(path, *, tracker_directory="tracker", crv_directory="crv", event_ids=None):
    """Read highest-cycle existing StrawGasStepTableDumper + CrvStepTableDumper trees.

    Configure both analyzers on the same dts event stream. The CRV events tree is
    the all-event census, not the union of nonempty detector or generator rows.
    With older dumpers, supply event_ids from the same successful job's EventLister
    log. Calorimeter steps are not extracted by these two analyzers.
    """
    with uproot.open(path) as root:
        census = None
        if event_ids is None:
            census = root[f"{crv_directory}/events"].arrays([*IDS, "n_steps"], library="np")
            ids = np.column_stack([census[k] for k in IDS])
        else:
            ids = np.asarray(event_ids)
            _event_keys(ids)
        collections = {
            "particles": _group(root[f"{crv_directory}/particles"], ids, PARTICLE_FIELDS),
            "gen": _group(root[f"{crv_directory}/gen"], ids, GEN_FIELDS),
            "tracker": _group(root[f"{tracker_directory}/sgsteps"], ids, TRACKER_FIELDS),
            "crv": _group(root[f"{crv_directory}/steps"], ids, CRV_FIELDS),
        }
        if census is not None and not np.array_equal(ak.to_numpy(ak.num(collections["crv"])), census["n_steps"]):
            raise ValueError("CRV census and extracted steps disagree")
    batch = FrameBatch(ids, collections)
    batch.validate_references()
    return batch


def _local_offsets(lengths, event, nevents):
    prefix = np.r_[0, np.cumsum(lengths)]
    first = np.searchsorted(event, np.arange(nevents))
    return prefix[:-1] - prefix[first[event]]


def overlay(signal, backgrounds, *, time_offset_ns=None):
    """Merge signal with (FrameBatch, FrameSelection) pairs, preserving all step rows.

    IDs are remapped per donor occurrence, including donor reuse in one event.
    time_offset_ns is a configured extra SimTimeOffset per PRIMARY event; None is
    stock ordinary mixing. It shifts donor global times only, never proper times
    or the primary. Source pulse offsets must not be supplied a second time.
    """
    nevents = len(signal.event_ids)
    names = set(signal.collections)
    signal.validate_references()
    arrays = {name: [rows] for name, rows in signal.collections.items()}
    event = [np.arange(nevents)]
    shift = [np.zeros(nevents)]
    delta = np.zeros(nevents) if time_offset_ns is None else np.asarray(time_offset_ns, dtype=float)
    if delta.shape != (nevents,) or np.any(~np.isfinite(delta)):
        raise ValueError("time_offset_ns must be a finite primary-event vector")
    for pool, selection in backgrounds:
        if not isinstance(selection, FrameSelection) or selection.counts.size != nevents:
            raise ValueError("one frame multiplicity per primary event required")
        if names != set(pool.collections):
            raise ValueError("signal and donor collection sets must match")
        if np.any(selection.donors >= len(pool.event_ids)):
            raise ValueError("donor index exceeds pool")
        pool.validate_references()
        event.append(selection.event_index)
        shift.append(delta[selection.event_index])
        for name in names:
            if set(ak.fields(signal.collections[name])) != set(ak.fields(pool.collections[name])):
                raise ValueError(f"signal/donor {name} schemas differ")
            arrays[name].append(pool.collections[name][selection.donors])
    event = np.concatenate(event)
    order = np.argsort(event, kind="stable")
    event = event[order]
    shift = np.concatenate(shift)[order]
    rows = {name: ak.concatenate(parts, axis=0)[order] for name, parts in arrays.items()}
    maxkey = ak.to_numpy(ak.fill_none(ak.max(rows["particles"].key, axis=1), -1))
    poffset = _local_offsets(maxkey + 1, event, nevents)
    goffset = _local_offsets(ak.to_numpy(ak.num(rows["gen"])), event, nevents)
    output = {}
    for name, collection in rows.items():
        if name == "particles":
            collection = ak.with_field(collection, collection.key + poffset, "key")
            collection = ak.with_field(collection, ak.where(collection.parent >= 0,
                                       collection.parent + poffset, -1), "parent")
            collection = ak.with_field(collection, ak.where(collection.genidx >= 0,
                                       collection.genidx + goffset, -1), "genidx")
        elif name == "gen":
            collection = ak.with_field(collection, collection.idx + goffset, "idx")
        else:
            collection = ak.with_field(collection, collection.sp + poffset, "sp")
        if time_offset_ns is not None:
            for field in TIME_FIELDS[name]:
                collection = ak.with_field(collection, collection[field] + shift, field)
        nrows = ak.to_numpy(ak.num(collection))
        counts = np.zeros(nevents, dtype=np.int64)
        np.add.at(counts, event, nrows)
        output[name] = ak.unflatten(ak.flatten(collection, axis=1), counts)
    result = FrameBatch(signal.event_ids.copy(), output)
    result.validate_references()
    return result
