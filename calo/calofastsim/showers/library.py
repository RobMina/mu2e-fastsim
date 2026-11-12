"""Binned whole-event shower draws; no energy rescaling or leakage correction.

The libraries follow the shared library core (mu2efastsim.library, core/docs/libraries.md): save() writes the arrays,
the JSON metadata and an embedded manifest; load() checks the manifest, the arrays and their content hash. A library
written before the manifests loads only when its file sha256 is registered (showers/registered.json: the libraries
in use that predate them, e.g. the CAL2 production candidate); any other file without a manifest is refused, and
write_manifest() adds a sidecar to it. load() returns (library, metadata) as before.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from mu2efastsim.library import axes as LA
from mu2efastsim.library import store as LS
from mu2efastsim.library.domain import DomainCounter
from mu2efastsim.library.hashing import file_sha256

REGISTERED = Path(__file__).with_name("registered.json")


@dataclass(frozen=True)
class Binning:
    """Total entry energy [MeV], forward cosine, and disk-axis radius [mm]."""

    energy: tuple = (0., 50., 80., 95., 110.)
    cosine: tuple = (0., .4, .7, 1.000001)
    radius: tuple = (0., 450., 550., 750., 1000.)

    def arrays(self):
        edges = tuple(np.asarray(v, dtype=float) for v in (self.energy, self.cosine, self.radius))
        if any(v.ndim != 1 or len(v) < 2 or not np.all(np.isfinite(v)) or np.any(np.diff(v) <= 0)
               for v in edges):
            raise ValueError("bin edges must be finite and strictly increasing")
        return edges


def frame(direction):
    """Right-handed column frames with z along each incident momentum."""
    u = np.asarray(direction, dtype=float)
    if u.ndim != 2 or u.shape[1] != 3 or not np.all(np.isfinite(u)):
        raise ValueError("direction must have finite shape (N,3)")
    norm = np.linalg.norm(u, axis=1)
    if np.any(norm == 0):
        raise ValueError("direction must be nonzero")
    z = u / norm[:, None]
    reference = np.zeros_like(z)
    reference[:, 2] = 1
    reference[np.abs(z[:, 2]) > .9] = (1, 0, 0)
    x = np.cross(reference, z)
    x /= np.linalg.norm(x, axis=1)[:, None]
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=2)


def cells(energy, cosine, radius, binning):
    """Return clipped bin coordinates and an explicit out-of-range flag."""
    values = np.column_stack((energy, cosine, radius))
    if not np.all(np.isfinite(values)):
        raise ValueError("entry features must be finite")
    edges = binning.arrays()
    index = np.column_stack([np.clip(np.searchsorted(e, values[:, k], side="right") - 1, 0, len(e) - 2)
                             for k, e in enumerate(edges)])
    outside = np.any(np.column_stack([(values[:, k] < e[0]) | (values[:, k] > e[-1])
                                     for k, e in enumerate(edges)]), axis=1)
    return index, outside


def _check(entries, steps):
    n = len(entries["energy"])
    for name in ("energy", "cosine", "radius", "time", "disk", "event_row"):
        a = np.asarray(entries[name])
        if a.shape != (n,) or not np.all(np.isfinite(a)):
            raise ValueError(f"invalid entry {name}")
    for name in ("position", "direction"):
        a = np.asarray(entries[name])
        if a.shape != (n, 3) or not np.all(np.isfinite(a)):
            raise ValueError(f"invalid entry {name}")
    if np.any(entries["energy"] <= 0) or np.any(entries["radius"] < 0):
        raise ValueError("invalid entry energy or radius")
    if np.any((entries["cosine"] < 0) | (entries["cosine"] > 1)):
        raise ValueError("only forward disk entries are supported")
    frame(entries["direction"])
    row = np.asarray(steps["entry"])
    if row.ndim != 1 or row.dtype.kind not in "iu" or np.any((row < 0) | (row >= n)):
        raise ValueError("invalid step entry index")
    for name in ("energy_g4", "energy_birks", "time", "crystal", "disk"):
        a = np.asarray(steps[name])
        if a.shape != row.shape or not np.all(np.isfinite(a)):
            raise ValueError(f"invalid step {name}")
    if np.asarray(steps["position"]).shape != (len(row), 3) or not np.all(np.isfinite(steps["position"])):
        raise ValueError("invalid step position")
    if np.any(steps["energy_g4"] < 0) or np.any(steps["energy_birks"] < 0):
        raise ValueError("negative deposited energy")
    if len(np.unique(entries["event_row"])) != n:
        raise ValueError("one entry per source event is required")


def build(entries, steps, *, development, binning=Binning()):
    """Pack the selected development events, retaining whole showers and zeros.

    Entries have position/direction in global coordinates and total energy.
    Steps carry a zero-based ``entry`` row, global position and absolute time.
    ``development`` is one boolean per entry, chosen independently of showers.
    """
    _check(entries, steps)
    development = np.asarray(development)
    if development.dtype.kind != "b" or development.shape != np.shape(entries["energy"]):
        raise ValueError("development must be a boolean entry mask")
    selected = np.flatnonzero(development)
    if len(selected) == 0:
        raise ValueError("no development entries")
    lookup = np.full(len(development), -1, dtype=np.int64)
    lookup[selected] = np.arange(len(selected))
    row = lookup[steps["entry"]]
    take = np.flatnonzero(row >= 0)
    take = take[np.argsort(row[take], kind="stable")]
    row = row[take]
    offsets = np.r_[0, np.cumsum(np.bincount(row, minlength=len(selected)))]
    packed = {"entry_" + k: np.asarray(v)[selected] for k, v in entries.items()}
    packed.update({"step_" + k: np.asarray(v)[take] for k, v in steps.items() if k != "entry"})
    rotation = frame(packed["entry_direction"])
    delta = packed.pop("step_position") - packed["entry_position"][row]
    packed["step_relative_position"] = np.einsum("nji,nj->ni", rotation[row], delta)
    packed["step_relative_time"] = packed.pop("step_time") - packed["entry_time"][row]
    packed["offsets"] = offsets
    index, outside = cells(packed["entry_energy"], packed["entry_cosine"], packed["entry_radius"], binning)
    packed["entry_cell"] = index
    packed["entry_outside_bins"] = outside
    packed.update(dict(zip(("bins_energy", "bins_cosine", "bins_radius"), binning.arrays())))
    return packed


def _radius_pool(rows, radius, query_radius, neighbors):
    """Nearest rows with bounded query-by-K storage and library-row tie breaks.

    Radius-sorted neighbors on either side bracket the nearest K. Left-hand
    duplicate radii are canonicalized to the first library rows of each tie,
    including when the candidate window begins partway through that tie.
    """
    sorted_rows = rows[np.argsort(radius[rows], kind="stable")]
    sorted_radius = radius[sorted_rows]
    split = np.searchsorted(sorted_radius, query_radius, side="left")
    left = split[:, None] - neighbors + np.arange(neighbors)
    right = split[:, None] + np.arange(neighbors)
    valid_left = left >= 0
    valid_right = right < len(rows)
    left = np.maximum(left, 0)
    first = np.searchsorted(sorted_radius, sorted_radius[left], side="left")
    left = first + left - np.maximum(first, np.maximum(split - neighbors, 0)[:, None])
    candidates = sorted_rows[np.column_stack((left, np.minimum(right, len(rows) - 1)))]
    distance = np.abs(radius[candidates] - query_radius[:, None])
    distance[~np.column_stack((valid_left, valid_right))] = np.inf
    nearest = np.lexsort((candidates, distance), axis=1)[:, :neighbors]
    return np.sort(np.take_along_axis(candidates, nearest, axis=1), axis=1)


def sample(library, entries, rng, *, orientation="incident", radius_neighbors=0):
    """Draw one whole pattern per query; flag empty-bin/out-of-range fallback.

    Sparse bins use the closest occupied bin in bin-index distance, in the same
    entry disk only. No support in a disk returns donor=-1 and zero steps;
    ``supported`` distinguishes that outcome from a sampled empty shower.
    Energies are copied, not scaled to the query entry energy.

    ``radius_neighbors=0`` preserves the original draw and output schema. A
    positive integer restricts each selected cell to up to that many nearest
    entry radii, breaking distance ties by library row. The eligible rows are
    sorted by library row before the same per-disk uniform draws are applied.
    Empty showers remain eligible. Only this opt-in adds per-query
    ``radius_pool_size`` and ``bin_pool_size`` (both zero without disk support).
    """
    if (isinstance(radius_neighbors, (bool, np.bool_)) or
            not isinstance(radius_neighbors, (int, np.integer)) or radius_neighbors < 0):
        raise ValueError("radius_neighbors must be a nonnegative integer")
    binning = Binning(*(library["bins_" + k] for k in ("energy", "cosine", "radius")))
    query, outside = cells(entries["energy"], entries["cosine"], entries["radius"], binning)
    rotation = frame(entries["direction"])
    n = len(query)
    for name, shape in (("position", (n, 3)), ("time", (n,)), ("disk", (n,))):
        if np.shape(entries[name]) != shape or not np.all(np.isfinite(entries[name])):
            raise ValueError(f"invalid query {name}")
    if len(rotation) != n:
        raise ValueError("query direction length mismatch")
    labels = np.column_stack((library["entry_disk"], library["entry_cell"]))
    occupied, inverse = np.unique(labels, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(occupied))
    starts = np.r_[0, np.cumsum(counts[:-1])]
    donors = np.full(n, -1, dtype=np.int64)
    fallback = outside.copy()
    if radius_neighbors:
        radius_pool_size = np.zeros(n, dtype=np.int64)
        bin_pool_size = np.zeros(n, dtype=np.int64)
    for disk in np.unique(entries["disk"]):
        where = np.flatnonzero(np.asarray(entries["disk"]) == disk)
        pool = np.flatnonzero(occupied[:, 0] == disk)
        if not len(pool):
            fallback[where] = True
            continue
        distance = np.sum((query[where, None, :] - occupied[pool, 1:][None, :, :]) ** 2, axis=2)
        near = np.argmin(distance, axis=1)
        bins = pool[near]
        fallback[where] |= distance[np.arange(len(where)), near] != 0
        if not radius_neighbors:
            donors[where] = order[starts[bins] + (rng.random(len(where)) * counts[bins]).astype(np.int64)]
            continue
        uniform = rng.random(len(where))
        bin_pool_size[where] = counts[bins]
        for cell in np.unique(bins):
            local = np.flatnonzero(bins == cell)
            target = where[local]
            rows = order[starts[cell]:starts[cell] + counts[cell]]
            size = min(radius_neighbors, len(rows))
            radius_pool_size[target] = size
            draw = (uniform[local] * size).astype(np.int64)
            if size == len(rows):
                donors[target] = rows[draw]
            else:
                eligible = _radius_pool(rows, library["entry_radius"], np.asarray(entries["radius"])[target], size)
                donors[target] = eligible[np.arange(len(target)), draw]
    supported = donors >= 0
    length = np.zeros(n, dtype=np.int64)
    length[supported] = np.diff(library["offsets"])[donors[supported]]
    offsets = np.r_[0, np.cumsum(length)]
    row = np.repeat(np.arange(n), length)
    source = library["offsets"][donors[row]] + np.arange(offsets[-1]) - np.repeat(offsets[:-1], length)
    result = {k.removeprefix("step_"): v[source] for k, v in library.items() if k.startswith("step_")}
    relative = result.pop("relative_position")
    if orientation == "incident":
        delta = np.einsum("nij,nj->ni", rotation[row], relative)
    elif orientation == "disk":
        delta = np.einsum("nij,nj->ni", frame(library["entry_direction"])[donors[row]], relative)
        origin = np.column_stack([library["geometry_disk_origin_" + a] for a in "xyz"])
        disk_ids = library["geometry_disk_disk"]
        slot = np.searchsorted(disk_ids, np.asarray(entries["disk"])[row])
        axis = np.column_stack([library["geometry_disk_axis_z_" + a] for a in "xyz"])
        if (np.any(np.diff(disk_ids) <= 0) or np.any(slot >= len(disk_ids)) or
                not np.allclose(axis, (0, 0, 1)) or not np.array_equal(disk_ids[slot], np.asarray(entries["disk"])[row])):
            raise ValueError("disk orientation requires sorted, z-aligned disk geometry")
        source_xy = library["entry_position"][donors[row], :2] - origin[slot, :2]
        target_xy = entries["position"][row, :2] - origin[slot, :2]
        angle = np.arctan2(target_xy[:, 1], target_xy[:, 0]) - np.arctan2(source_xy[:, 1], source_xy[:, 0])
        c, s = np.cos(angle), np.sin(angle)
        x, y = delta[:, 0].copy(), delta[:, 1].copy()
        delta[:, 0], delta[:, 1] = c * x - s * y, s * x + c * y
    else:
        raise ValueError("orientation must be incident or disk")
    result["position"] = entries["position"][row] + delta
    result["time"] = entries["time"][row] + result.pop("relative_time")
    result.update(entry=row, offsets=offsets, donor=donors, supported=supported, fallback=fallback)
    if radius_neighbors:
        result.update(radius_pool_size=radius_pool_size, bin_pool_size=bin_pool_size)
    return result


def _name(metadata):
    tag = str(metadata.get("candidate") or metadata.get("name") or "unnamed")
    return "calo.showers." + "".join(c if c.isalnum() or c in "_-" else "_" for c in tag)


def axes(library):
    """The cell keys (mu2efastsim.library.axes.Axis) of a library's binning."""
    frame = "conversion-electron entry at the disk front (incident frame)"
    return (LA.Axis("energy", "MeV", library["bins_energy"], frame=frame, array="bins_energy", policy="clip",
                    description="total entry energy"),
            LA.Axis("cosine", "1", library["bins_cosine"], frame=frame, array="bins_cosine", policy="clip",
                    description="forward cosine of the entry direction (only forward entries are supported)"),
            LA.Axis("radius", "mm", library["bins_radius"], frame=frame, array="bins_radius", policy="clip",
                    description="entry radius from the disk axis"))


def manifest_fields(library, metadata, **provenance):
    """Manifest fields of a library (mu2efastsim.library.manifest); provenance fields not given are unrecorded."""
    fields = dict(
        name=_name(metadata), subsystem="calo", version=1, kind="pool",
        description="Whole-event Geant4 calorimeter shower patterns per entry cell (energy, cosine, radius), drawn "
                    "whole for a query entry and placed in its frame (sample); empty cells use the closest occupied "
                    "cell of the same disk.",
        axes=[a.describe() for a in axes(library)],
        outputs=[dict(name="steps", unit="mm, ns, MeV", dtype="float64",
                      description="the donor's steps placed at the query entry; donor -1 and no steps without "
                                  "support in the disk"),
                 dict(name="donor", unit="1", dtype="int64", description="library row drawn, -1 when unsupported"),
                 dict(name="fallback", unit="1", dtype="bool", description="drawn from another cell")],
        dtype=dict(storage={"steps": "as built (float64)"}, compute=["float32", "float64"],
                   notes="host draws in float64; the device hand-off casts to the compute dtype"),
        rng=dict(host="numpy.random.Generator (sample): uniform draws per query, the same with or without the "
                      "radius_neighbors option", device="none (host draw)"),
        domain=dict(coverage="forward entries; cells clipped to the end bins (entry_outside_bins flags it)",
                    fallbacks=[dict(name="fallback", levels={"0": "the entry's own cell",
                                                             "1": "the closest occupied cell of the disk"},
                                    description="sample() returns fallback per query")]),
        config_hash=None,
        source=dict(description="metadata: " + json.dumps(metadata, sort_keys=True)[:2000], samples=[]),
    )
    fields.update(provenance)
    if "validation" not in provenance:
        fields["validation"] = dict(status="not validated", record=None)
    unrecorded = dict(provenance.get("unrecorded", {}))
    for k in ("generator", "release", "seeds", "geometry", "split"):
        if fields.get(k) is None:
            fields[k] = None
            unrecorded.setdefault(k, "not given to save()")
    unrecorded.setdefault("config_hash", "no build configuration manifest")
    fields["unrecorded"] = unrecorded
    return fields


def save(path, library, metadata, **provenance):
    """Write numeric arrays, small JSON metadata and the library manifest without pickle. Returns the manifest."""
    arrays = dict(library, metadata=np.array(json.dumps(metadata, sort_keys=True)))
    return LS.save(Path(path), arrays, manifest_fields(library, metadata, **provenance))


def write_manifest(path, **provenance):
    """Sidecar manifest for a library written before the manifests; the .npz is not touched."""
    arrays = LS.load_arrays(path)
    metadata = json.loads(str(arrays["metadata"]))
    library = {k: v for k, v in arrays.items() if k != "metadata"}
    return LS.write_sidecar(path, manifest_fields(library, metadata, **provenance))


def registered(path=REGISTERED):
    """{sha256: entry} of the registered libraries that predate the manifests."""
    p = Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {e["sha256"]: dict(e, id=k) for k, e in d["entries"].items()}


def load(path, *, registry=REGISTERED):
    """(library, metadata) after the library checks: a manifest (embedded or sidecar) with the arrays and content hash
    it declares; or, for a library written before the manifests, a registered file sha256. Anything else is refused
    (mu2efastsim.library.store.LibraryError)."""
    try:
        arrays = dict(LS.load(Path(path)).arrays)
    except LS.LibraryError as exc:
        if not any("no manifest" in p for p in exc.problems) or file_sha256(path) not in registered(registry):
            raise
        with np.load(path, allow_pickle=False) as source:          # registered legacy file: read as before
            library = {k: source[k] for k in source.files if k != "metadata"}
            metadata = json.loads(str(source["metadata"]))
        return library, metadata
    metadata = json.loads(str(arrays.pop("metadata")))
    return arrays, metadata


def domain_counts(library, energy, cosine, radius, counter=None):
    """DomainCounter of a set of queries (host): entry values outside the cell bins (clipped). The fallback to another
    cell is per draw: count it from sample()'s 'fallback' with counter.add_levels('fallback', ...)."""
    counter = counter or DomainCounter("calo.showers")
    counter.add_lookups(len(np.asarray(energy)))
    for a, v in zip(axes(library), (energy, cosine, radius)):
        counter.add_values(a.name, a.edges, np.asarray(v, float))
    return counter
