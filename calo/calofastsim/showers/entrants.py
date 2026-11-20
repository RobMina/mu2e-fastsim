"""Per-entrant calorimeter shower library (opt-in; P2 of the g4fastsim plan).

A fast-simulation entrant is a particle whose first crossing into a calorimeter disk comes before any of its ancestors
entered a disk: with a fast-simulation model on the disks, the entrant stops at the disk and its descendants never
exist, so its pattern is every CaloShowerStep (CSS) of its subtree, in both disks. Entry crossings are disk
virtual-detector steps moving into the disk: front 73/75 (along the disk axis), inner edge 77/79 (outward), back
74/76 (against the axis) and outer edge 78/80 (inward). The library keeps the front and inner-edge entrants; the
others are counted in the census.

Each pattern is stored in its entrant's disk frame turned to the entry azimuth (x radial, y tangential, z along the
disk axis), so ``sample`` places a donor at a query entry by the rotation about the disk axis that carries the donor
azimuth to the query azimuth, plus the translation between the entries. Energies are never rescaled.

Cells: species (e-, e+, gamma, other) x face x disk x total energy x face cosine x position, where the face cosine is
along the disk axis (front) or along the outward radius (inner edge), and the position is the entry radius (front)
or the entry depth along the disk axis from the disk origin (inner edge). Sparse cells draw from the closest occupied
cell (bin-index distance) of the same species, face and disk; a (species, face, disk) group without entrants is
unsupported (donor -1), which a fast-simulation model treats as "leave the particle to full Geant4".

The file layout follows the shared library core (mu2efastsim.library): ``save`` embeds the manifest and content hash;
``export_root`` writes the same arrays as ROOT trees, with that content hash, for the Mu2eG4 model.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import uproot

from mu2efastsim.library import axes as LA
from mu2efastsim.library import store as LS
from mu2efastsim.library.hashing import content_hash

from .export import EVENT, IDENTITY, columns, join, keys, vector

FRONT, INNER, BACK, OUTER = 0, 1, 2, 3
FACE_NAMES = ("front", "inner", "back", "outer")
SUPPORTED_FACES = (FRONT, INNER)
VD_FACES = {73: (0, FRONT), 75: (1, FRONT), 77: (0, INNER), 79: (1, INNER),
            74: (0, BACK), 76: (1, BACK), 78: (0, OUTER), 80: (1, OUTER)}
SPECIES_PDG = (11, -11, 22)        # species codes 0, 1, 2; anything else is 3 ("other")
SPECIES_NAMES = ("e-", "e+", "gamma", "other")
MASS = {11: .51099895, -11: .51099895}


def _find(reference, query):
    """Row of each query key in reference, -1 when absent."""
    if len(reference) == 0:
        return np.full(len(query), -1, dtype=np.int64)
    order = np.argsort(reference)
    index = np.minimum(np.searchsorted(reference[order], query), len(reference) - 1)
    return np.where(reference[order[index]] == query, order[index], -1)


def species_code(pdg):
    pdg = np.asarray(pdg)
    out = np.full(pdg.shape, 3, dtype=np.int64)
    for code, value in enumerate(SPECIES_PDG):
        out[pdg == value] = code
    return out


@dataclass(frozen=True)
class Binning:
    """Total entry energy [MeV], face cosine, front radius [mm] and inner-edge depth along the disk axis [mm]."""

    energy: tuple = (0., 50., 80., 95., 110.)
    cosine: tuple = (0., .4, .7, 1.000001)
    front_radius: tuple = (0., 450., 550., 750., 1000.)
    inner_depth: tuple = (-200., -100., -50., 0., 50., 100., 200.)

    def arrays(self):
        edges = tuple(np.asarray(v, dtype=float) for v in (self.energy, self.cosine, self.front_radius, self.inner_depth))
        if any(v.ndim != 1 or len(v) < 2 or not np.all(np.isfinite(v)) or np.any(np.diff(v) <= 0) for v in edges):
            raise ValueError("bin edges must be finite and strictly increasing")
        return edges


def _locate(edges, values):
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    return index, (values < edges[0]) | (values > edges[-1])


def cells(face, energy, cosine, position, binning):
    """(N,3) clipped bin indices (energy, cosine, position) and an out-of-range flag; position bins depend on face."""
    face = np.asarray(face)
    values = np.column_stack((energy, cosine, position)).astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("entry features must be finite")
    e, c, r, z = binning.arrays()
    ie, oe = _locate(e, values[:, 0])
    ic, oc = _locate(c, values[:, 1])
    ir, orr = _locate(r, values[:, 2])
    iz, oz = _locate(z, values[:, 2])
    ip = np.where(face == INNER, iz, ir)
    op = np.where(face == INNER, oz, orr)
    return np.column_stack((ie, ic, ip)), oe | oc | op


def _disk_frames(disks):
    """Per-disk origin (D,3) and rotation (D,3,3) whose columns are the disk x, y, z axes in global coordinates."""
    origin = vector(disks, "origin")
    rotation = np.stack([vector(disks, "axis_" + a) for a in "xyz"], axis=2)
    return origin, rotation


def _azimuth_rotation(phi):
    c, s = np.cos(phi), np.sin(phi)
    rot = np.zeros((len(phi), 3, 3))
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1.
    return rot


def entry_features(face, local_position, local_momentum, pdg):
    """Total energy, face cosine, position key and azimuth of entries given in disk-local coordinates."""
    p = np.linalg.norm(local_momentum, axis=1)
    mass = np.array([MASS.get(int(v), 0.) for v in np.asarray(pdg)])
    energy = np.sqrt(p ** 2 + mass ** 2)
    phi = np.arctan2(local_position[:, 1], local_position[:, 0])
    rho = np.hypot(local_position[:, 0], local_position[:, 1])
    radial = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(len(phi))))
    along = np.divide(local_momentum[:, 2], p, out=np.zeros_like(p), where=p > 0)
    outward = np.divide(np.sum(local_momentum * radial, axis=1), p, out=np.zeros_like(p), where=p > 0)
    face = np.asarray(face)
    cosine = np.clip(np.where(face == INNER, outward, along), 0., 1.)
    key = np.where(face == INNER, local_position[:, 2], rho)
    return energy, cosine, key, phi


def transverse_angle(local_position, local_momentum):
    """psi: angle of the transverse entry momentum from the outward radius at the entry, in (-pi, pi] (disk frame)."""
    lp, lm = np.asarray(local_position, float), np.asarray(local_momentum, float)
    return np.angle(np.exp(1j * (np.arctan2(lm[:, 1], lm[:, 0]) - np.arctan2(lp[:, 1], lp[:, 0]))))


PSI_BINS_15 = tuple(np.radians(np.arange(-180, 181, 15)))   # 15 degree psi bins (opt-in cell axis of sample)


def psi_cells(face, psi, psi_bins):
    """psi bin index of front entries (clipped to the edges); inner-edge entries are not split (index 0)."""
    edges = np.asarray(psi_bins, float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("psi_bins must be increasing edges")
    index = np.clip(np.searchsorted(edges, np.asarray(psi, float), side="right") - 1, 0, len(edges) - 2)
    return np.where(np.asarray(face) == FRONT, index, 0)


def read(path, *, skip_failed=False):
    """Entrants and their subtree CSS from one CaloLibraryDumper export (calo_library_tables.root).

    Returns (entrants, css, geometry, census). Entrants: one row per fast-simulation entrant with event_row, pdg,
    species, face, disk, global position/momentum/time and local (disk-frame) position/momentum. CSS: rows of
    supported-face entrants with entrant row, global position, absolute time, energies, crystal and disk.
    ``skip_failed`` (opt-in) drops events whose Geant4 transport failed (``g4_status != 0``) with all their rows
    instead of raising; event rows then index the completed events, and the census counts the dropped ones.
    """
    with uproot.open(path) as source:
        events = columns(source, "events", EVENT + ("g4_status",))
        failed = events["g4_status"] != 0
        if np.any(failed) and not skip_failed:
            raise ValueError("input contains failed transport events; select completed input events first")
        particles = columns(source, "particles", IDENTITY + ("pdg", "parent_product_id", "parent_key", "parent_valid"))
        steps = columns(source, "steps", IDENTITY + ("kind", "volume", "time") +
                        tuple(p + "_" + a for p in ("position", "momentum") for a in "xyz"))
        showers = columns(source, "showers", IDENTITY + ("sim_valid", "crystal", "time", "energy_g4", "energy_birks") +
                          tuple("position_" + a for a in "xyz"))
        disks = columns(source, "disks", ("run", "disk") + tuple(p + "_" + a for p in ("origin", "size", "axis_x", "axis_y", "axis_z") for a in "xyz"))
        crystals = columns(source, "crystals", ("run", "crystal", "disk") + tuple(p + "_" + a for p in ("position", "size") for a in "xyz"))
        vd = columns(source, "virtualdetectors", ("run", "id") + tuple("position_" + a for a in "xyz"))
    if np.any(failed):
        bad = keys({k: v[failed] for k, v in events.items()}, EVENT)
        events = {k: v[~failed] for k, v in events.items()}
        particles, steps, showers = ({k: v[~np.isin(keys(t, EVENT), bad)] for k, v in t.items()}
                                     for t in (particles, steps, showers))
    event_ids = keys(events, EVENT)

    pid = keys(particles, IDENTITY)
    n = len(pid)
    parent = np.full(n, -1, dtype=np.int64)
    linked = particles["parent_valid"].astype(bool)
    target = {k: v[linked] for k, v in particles.items()}
    target.update(sim_product_id=target["parent_product_id"], sim_key=target["parent_key"])
    parent[linked] = join(pid, keys(target, IDENTITY))

    origin, rotation = _disk_frames(disks)
    vd_mask = (steps["kind"] == 0) & np.isin(steps["volume"], list(VD_FACES))
    steps = {k: v[vd_mask] for k, v in steps.items()}
    disk = np.array([VD_FACES[int(v)][0] for v in steps["volume"]], dtype=np.int64)
    face = np.array([VD_FACES[int(v)][1] for v in steps["volume"]], dtype=np.int64)
    drow = join(keys(disks, ("run", "disk")), keys({**steps, "disk": disk}, ("run", "disk")))
    local_pos = np.einsum("nji,nj->ni", rotation[drow], vector(steps, "position") - origin[drow])
    local_mom = np.einsum("nji,nj->ni", rotation[drow], vector(steps, "momentum"))
    radial = local_pos[:, :2] / np.maximum(np.hypot(local_pos[:, 0], local_pos[:, 1]), 1e-9)[:, None]
    outward = np.sum(local_mom[:, :2] * radial, axis=1)
    entering = np.select([face == FRONT, face == BACK, face == INNER, face == OUTER],
                         [local_mom[:, 2] > 0, local_mom[:, 2] < 0, outward > 0, outward < 0], False)
    srow = join(pid, keys(steps, IDENTITY))
    if np.any(srow < 0):
        raise ValueError("virtual-detector step without its SimParticle")

    # each particle's first entry crossing (earliest time; ties broken by step order)
    cand = np.flatnonzero(entering)
    cand = cand[np.lexsort((cand, steps["time"][cand], srow[cand]))]
    first = cand[np.r_[True, np.diff(srow[cand]) != 0]] if len(cand) else cand
    crossing = np.full(n, -1, dtype=np.int64)
    crossing[srow[first]] = first
    has_entry = crossing >= 0

    # the entrant of each particle: its topmost ancestor-or-self with an entry crossing
    top = np.where(has_entry, np.arange(n), -1)
    cur = parent.copy()
    for _ in range(n + 1):
        live = cur >= 0
        if not live.any():
            break
        safe = np.where(live, cur, 0)
        hit = live & has_entry[safe]
        top[hit] = cur[hit]
        cur = np.where(live, parent[safe], -1)
    else:
        raise ValueError("cyclic SimParticle ancestry")
    entrant_particles = np.flatnonzero(has_entry & (top == np.arange(n)))
    c = crossing[entrant_particles]
    erow_all = join(event_ids, keys(particles, EVENT))
    entrants = dict(event_row=erow_all[entrant_particles], pdg=particles["pdg"][entrant_particles],
                    face=face[c], disk=disk[c], time=steps["time"][c],
                    position=vector(steps, "position")[c], momentum=vector(steps, "momentum")[c],
                    local_position=local_pos[c], local_momentum=local_mom[c])
    entrants["species"] = species_code(entrants["pdg"])
    entrant_row = np.full(n, -1, dtype=np.int64)
    entrant_row[entrant_particles] = np.arange(len(entrant_particles))

    # CSS: global position from the crystal frame, entrant from the CSS particle
    crow = join(keys(crystals, ("run", "crystal")), keys(showers, ("run", "crystal")))
    cdisk = join(keys(disks, ("run", "disk")), keys(crystals, ("run", "disk")))
    position = vector(crystals, "position")[crow] + np.einsum("nij,nj->ni", rotation[cdisk[crow]], vector(showers, "position"))
    prow = np.where(showers["sim_valid"].astype(bool), _find(pid, keys(showers, IDENTITY)), -1)
    safe = np.maximum(prow, 0)
    owner = np.where((prow >= 0) & (top[safe] >= 0), entrant_row[top[safe]], -1)
    eb = showers["energy_birks"].astype(float)
    supported = np.isin(entrants["face"], SUPPORTED_FACES)
    keep = owner >= 0
    keep[keep] = supported[owner[keep]]
    css = dict(entrant=owner[keep], position=position[keep], time=showers["time"][keep].astype(float),
               energy_g4=showers["energy_g4"][keep].astype(float), energy_birks=eb[keep],
               crystal=showers["crystal"][keep], disk=crystals["disk"][crow[keep]])
    total = float(eb.sum())
    census = dict(events=len(event_ids), failed_events_skipped=int(np.sum(failed)), particles=n, entrants=len(entrant_particles), css_rows=len(eb),
                  css_energy_birks=total,
                  css_fraction_unresolved_particle=float(eb[prow < 0].sum()) / total if total else 0.,
                  css_fraction_no_entrant=float(eb[(prow >= 0) & (owner < 0)].sum()) / total if total else 0.,
                  by_face={})
    for f in range(4):
        m = np.flatnonzero(entrants["face"] == f)
        e = float(eb[np.isin(owner, m)].sum())
        census["by_face"][FACE_NAMES[f]] = dict(entrants=int(len(m)), energy_fraction=e / total if total else 0.)
    geometry = {"crystal_" + k: v for k, v in crystals.items()}
    geometry.update({"disk_" + k: v for k, v in disks.items()})
    geometry.update({"vd_" + k: v for k, v in vd.items()})
    sel = np.flatnonzero(supported)
    remap = np.full(len(supported), -1, dtype=np.int64)
    remap[sel] = np.arange(len(sel))
    entrants = {k: v[sel] for k, v in entrants.items()}
    css["entrant"] = remap[css["entrant"]]
    return entrants, css, geometry, census


def read_many(paths, *, skip_failed=False):
    """Concatenate several exports (one geometry); event rows are offset per file. ``skip_failed`` as in :func:`read`
    (the offsets count the completed events)."""
    parts, geometry, census = [], None, []
    offset = 0
    for path in paths:
        entrants, css, geo, cen = read(path, skip_failed=skip_failed)
        if geometry is None:
            geometry = geo
        elif any(not np.array_equal(geometry[k], geo[k]) for k in geometry if not k.endswith("_run")):
            raise ValueError(f"{path}: geometry differs from the first export")
        entrants = dict(entrants, event_row=entrants["event_row"] + offset)
        parts.append((entrants, css))
        offset += cen["events"]
        census.append(dict(cen, file=str(path)))
    entrants = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    shift = np.cumsum([0] + [len(p[0]["face"]) for p in parts[:-1]])
    css = {k: np.concatenate([p[1][k] + (s if k == "entrant" else 0) for p, s in zip(parts, shift)]) for k in parts[0][1]}
    return entrants, css, geometry, census


def build(entrants, css, geometry, *, binning=Binning()):
    """Pack entrants and their CSS: one row per entrant, steps relative to the entry in the turned disk frame."""
    n = len(entrants["face"])
    row = np.asarray(css["entrant"])
    if row.ndim != 1 or np.any((row < 0) | (row >= n)):
        raise ValueError("invalid CSS entrant index")
    order = np.argsort(row, kind="stable")
    row = row[order]
    disk_ids = np.asarray(geometry["disk_disk"])
    if np.any(np.diff(disk_ids) <= 0):
        raise ValueError("disk geometry must be sorted by disk id")
    origin = np.column_stack([geometry["disk_origin_" + a] for a in "xyz"])
    rotation = np.stack([np.column_stack([geometry[f"disk_axis_{ax}_{a}"] for a in "xyz"]) for ax in "xyz"], axis=2)
    energy, cosine, key, phi = entry_features(entrants["face"], entrants["local_position"], entrants["local_momentum"],
                                              entrants["pdg"])
    slot = np.searchsorted(disk_ids, np.asarray(css["disk"])[order])
    local = np.einsum("nji,nj->ni", rotation[slot], np.asarray(css["position"])[order] - origin[slot])
    entry_slot = np.searchsorted(disk_ids, entrants["disk"])
    # the entry point expressed in the frame of the deposit's disk, so both disks share one placement rule
    entry_in_slot = np.einsum("nji,nj->ni", rotation[slot],
                              origin[entry_slot[row]] + np.einsum("nij,nj->ni", rotation[entry_slot[row]],
                                                                  entrants["local_position"][row]) - origin[slot])
    turn = _azimuth_rotation(-phi[row])
    relative = np.einsum("nij,nj->ni", turn, local - entry_in_slot)
    index, outside = cells(entrants["face"], energy, cosine, key, binning)
    e, c, r, z = binning.arrays()
    library = dict(
        entry_event_row=np.asarray(entrants["event_row"], dtype=np.int64), entry_pdg=np.asarray(entrants["pdg"], dtype=np.int64),
        entry_species=np.asarray(entrants["species"], dtype=np.int64), entry_face=np.asarray(entrants["face"], dtype=np.int64),
        entry_disk=np.asarray(entrants["disk"], dtype=np.int64), entry_energy=energy, entry_cosine=cosine,
        entry_key=key, entry_azimuth=phi, entry_time=np.asarray(entrants["time"], dtype=float),
        entry_local_position=np.asarray(entrants["local_position"], dtype=float), entry_cell=index, entry_outside_bins=outside,
        entry_local_momentum=np.asarray(entrants["local_momentum"], dtype=float),
        entry_psi=transverse_angle(entrants["local_position"], entrants["local_momentum"]),
        step_relative_position=relative, step_relative_time=np.asarray(css["time"])[order] - np.asarray(entrants["time"])[row],
        step_energy_g4=np.asarray(css["energy_g4"])[order], step_energy_birks=np.asarray(css["energy_birks"])[order],
        step_crystal=np.asarray(css["crystal"])[order].astype(np.int64), step_disk=np.asarray(css["disk"])[order].astype(np.int64),
        offsets=np.r_[0, np.cumsum(np.bincount(row, minlength=n))].astype(np.int64),
        bins_energy=e, bins_cosine=c, bins_front_radius=r, bins_inner_depth=z)
    library.update({"geometry_" + k: np.asarray(v) for k, v in geometry.items()})
    return library


def _binning(library):
    return Binning(*(tuple(library["bins_" + k]) for k in ("energy", "cosine", "front_radius", "inner_depth")))


def entry_distance2(face, key, azimuth, rho, donor_key, donor_azimuth):
    """Squared distance [mm^2] between query entries (N,) and donor entries (M,) of one face, (N, M).

    The azimuth difference is folded by the 180-degree turn about the disk axis, a symmetry of the crystal lattice (the
    x and y mirrors are too, but would flip the shower's handedness in the solenoid field), so a donor and its turned
    image count alike. Front face: the distance between the two entry points on the face (radius ``key``); inner
    edge: sqrt(dz^2 + (rho dphi)^2) with the depth ``key`` and the query radius ``rho``. The rule of the Mu2eG4 model
    (Offline ``Mu2eG4CaloEntrantLibrary::selectNearest``)."""
    key = np.asarray(key, float)[:, None]
    dk = np.asarray(donor_key, float)[None, :]
    dphi = np.asarray(azimuth, float)[:, None] - np.asarray(donor_azimuth, float)[None, :]
    dphi = np.mod(dphi + np.pi / 2, np.pi) - np.pi / 2
    if face == INNER:
        return (key - dk) ** 2 + (np.asarray(rho, float)[:, None] * dphi) ** 2
    return np.maximum(key ** 2 + dk ** 2 - 2. * key * dk * np.cos(dphi), 0.)


def nearest_entries(face, key, azimuth, rho, rows, donor_key, donor_azimuth, neighbors, uniform, *, chunk=4_000_000):
    """One donor per query among the ``neighbors`` donor ``rows`` (ascending library rows) whose entries are nearest
    (:func:`entry_distance2`); the candidates are ordered by distance, ties by library row, and ``uniform`` picks
    ``min(floor(u n), n - 1)`` of the n = min(neighbors, len(rows)). Returns (donor rows, distances [mm])."""
    n = len(key)
    rows = np.asarray(rows, np.int64)
    k = min(int(neighbors), len(rows))
    donors = np.full(n, -1, np.int64)
    distance = np.full(n, np.nan)
    if not n or not k:
        return donors, distance
    step = max(1, chunk // len(rows))
    for lo in range(0, n, step):
        sl = slice(lo, min(n, lo + step))
        d2 = entry_distance2(face, key[sl], azimuth[sl], rho[sl], donor_key, donor_azimuth)
        near = np.argpartition(d2, k - 1, axis=1)[:, :k] if k < len(rows) else np.tile(np.arange(len(rows)), (d2.shape[0], 1))
        nd2 = np.take_along_axis(d2, near, axis=1)
        order = np.lexsort((rows[near], nd2), axis=1)
        near, nd2 = np.take_along_axis(near, order, axis=1), np.take_along_axis(nd2, order, axis=1)
        pick = np.minimum((np.asarray(uniform[sl]) * k).astype(np.int64), k - 1)
        donors[sl] = rows[near[np.arange(len(pick)), pick]]
        distance[sl] = np.sqrt(nd2[np.arange(len(pick)), pick])
    return donors, distance


def sample(library, queries, rng, *, dtype=np.float64, key_neighbors=0, psi_bins=None, entry_neighbors=0,
           max_entry_distance=None, max_energy_deviation=None):
    """Draw one donor per query and place its CSS at the query entry.

    queries: disk, face, pdg, time, local_position (N,3) and local_momentum (N,3) in the disk frame. Returns the
    placed steps (global position, absolute time, energies, donor crystal), per-query donor (-1 when the
    species/face/disk group has no entrants), supported, fallback (another cell or outside the bins) and offsets.

    ``key_neighbors=0`` keeps the uniform draw over the cell. A positive K (opt-in) draws, in a cell with more than K
    entrants, among the K whose position key (front radius or inner-edge depth) is nearest the query's (ties by
    library row; the eligible rows sorted by library row, then the same uniform), the rule of
    :func:`calofastsim.showers.library.sample`'s ``radius_neighbors``: the donor pattern then moves by at most the
    key spread of those K instead of the cell width (a front entry near the inner edge or an inner-edge entry near
    the crystals' back face otherwise lands partly outside the crystals).

    ``psi_bins`` (opt-in, edges in radians, e.g. :data:`PSI_BINS_15`) adds the angle of the transverse entry momentum
    from the outward radius (:func:`transverse_angle`) as a cell axis of front entries: a front shower's offset from
    its entry follows that direction (CE on the W7 dev files: +21 mm outward-going vs -31 mm inward-going at 450-1000
    mm), which the energy, cosine and radius cells do not carry. Needs a library built with ``entry_psi``. The
    nearest occupied cell uses the bin-index distance over all axes, psi included.

    ``entry_neighbors`` (opt-in, K > 0) is the Mu2eG4 model's default donor rule (``donorSelection: nearest``) instead
    of the cells: among the entrants of the query's species, face, disk, energy bin and cosine bin, the K whose entry
    points are nearest the query's (:func:`entry_distance2`, which counts a donor's image turned by 180 degrees), one
    drawn with the query's uniform (:func:`nearest_entries`). Placement then turns a donor by about 0 or 180 degrees
    and moves it about that distance, so its shower keeps its place on the crystal lattice near the stepped inner and
    outer edges of the crystal array, which an arbitrary turn does not (Mu2eG4 test: 14.5 % of the placed energy
    landed outside the crystals with the cell rule, 3.4 % with this one). Queries outside the bins, or whose nearest
    usable donor is farther than ``max_entry_distance`` [mm], get no donor (left to full simulation); ``distance``
    returns the donor's entry distance. Not combined with ``key_neighbors`` or ``psi_bins``. The Mu2eG4 defaults
    are ``entry_neighbors=1, max_entry_distance=20., max_energy_deviation=0.1``.
    The device chain (:mod:`calofastsim.entrant_chain`) does not implement this rule.

    ``max_energy_deviation`` (opt-in, either rule): no donor when |E_donor - E| > max_energy_deviation * E, with the
    total entry energies (energies are never rescaled).
    """
    if (isinstance(key_neighbors, (bool, np.bool_)) or not isinstance(key_neighbors, (int, np.integer))
            or key_neighbors < 0):
        raise ValueError("key_neighbors must be a nonnegative integer")
    if (isinstance(entry_neighbors, (bool, np.bool_)) or not isinstance(entry_neighbors, (int, np.integer))
            or entry_neighbors < 0):
        raise ValueError("entry_neighbors must be a nonnegative integer")
    if entry_neighbors and (key_neighbors or psi_bins is not None):
        raise ValueError("entry_neighbors is not combined with key_neighbors or psi_bins")
    for name, value in (("max_entry_distance", max_entry_distance), ("max_energy_deviation", max_energy_deviation)):
        if value is not None and not (np.isfinite(value) and value >= 0):
            raise ValueError(f"{name} must be a nonnegative number or None")
    face = np.asarray(queries["face"])
    n = len(face)
    disk = np.asarray(queries["disk"])
    species = species_code(queries["pdg"])
    energy, cosine, key, phi = entry_features(face, np.asarray(queries["local_position"], float),
                                              np.asarray(queries["local_momentum"], float), queries["pdg"])
    query, outside = cells(face, energy, cosine, key, _binning(library))
    group = np.column_stack((library["entry_species"], library["entry_face"], library["entry_disk"]))
    labels = np.column_stack((group, library["entry_cell"]))
    if psi_bins is not None:
        if "entry_psi" not in library:
            raise ValueError("psi_bins needs a library built with entry_psi")
        labels = np.column_stack((labels, psi_cells(library["entry_face"], library["entry_psi"], psi_bins)))
        query = np.column_stack((query, psi_cells(face, transverse_angle(queries["local_position"],
                                                                         queries["local_momentum"]), psi_bins)))
    occupied, inverse = np.unique(labels, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(occupied))
    starts = np.r_[0, np.cumsum(counts[:-1])]
    donors = np.full(n, -1, dtype=np.int64)
    fallback = outside.copy()
    uniform = rng.random(n)
    entry_distance = np.full(n, np.nan)
    qgroup = np.column_stack((species, face, disk))
    if entry_neighbors:
        rho = np.hypot(np.asarray(queries["local_position"], float)[:, 0], np.asarray(queries["local_position"], float)[:, 1])
        lgroup = np.column_stack((group, library["entry_cell"][:, :2]))
        qcell = np.column_stack((qgroup, query[:, :2]))
        fallback = np.zeros(n, bool)
        ok = ~outside
        for g in np.unique(qcell[ok], axis=0):
            where = np.flatnonzero(ok & np.all(qcell == g, axis=1))
            rows = np.flatnonzero(np.all(lgroup == g, axis=1))
            if not len(rows):
                continue
            donors[where], entry_distance[where] = nearest_entries(
                int(g[1]), key[where], phi[where], rho[where], rows, np.asarray(library["entry_key"], float)[rows],
                np.asarray(library["entry_azimuth"], float)[rows], entry_neighbors, uniform[where])
        if max_entry_distance is not None:
            far = donors >= 0
            far[far] = entry_distance[far] > max_entry_distance
            donors[far] = -1
    for g in (np.unique(qgroup, axis=0) if not entry_neighbors else ()):
        where = np.flatnonzero(np.all(qgroup == g, axis=1))
        pool = np.flatnonzero(np.all(occupied[:, :3] == g, axis=1))
        if not len(pool):
            continue
        distance = np.sum((query[where, None, :] - occupied[pool, 3:][None, :, :]) ** 2, axis=2)
        near = np.argmin(distance, axis=1)
        bins = pool[near]
        fallback[where] |= distance[np.arange(len(where)), near] != 0
        donors[where] = order[starts[bins] + (uniform[where] * counts[bins]).astype(np.int64)]
        if key_neighbors:
            from .library import _radius_pool
            for cell in np.unique(bins[counts[bins] > key_neighbors]):
                local = np.flatnonzero(bins == cell)
                target = where[local]
                rows = order[starts[cell]:starts[cell] + counts[cell]]
                eligible = _radius_pool(rows, np.asarray(library["entry_key"], float), key[target], key_neighbors)
                draw = (uniform[target] * key_neighbors).astype(np.int64)
                donors[target] = eligible[np.arange(len(target)), draw]
    if max_energy_deviation is not None:
        has = donors >= 0
        has[has] = (np.abs(np.asarray(library["entry_energy"], float)[donors[has]] - energy[has])
                    > max_energy_deviation * energy[has])
        donors[has] = -1
    supported = donors >= 0
    length = np.zeros(n, dtype=np.int64)
    length[supported] = np.diff(library["offsets"])[donors[supported]]
    offsets = np.r_[0, np.cumsum(length)]
    row = np.repeat(np.arange(n), length)
    source = library["offsets"][donors[row]] + np.arange(offsets[-1]) - np.repeat(offsets[:-1], length)

    disk_ids = np.asarray(library["geometry_disk_disk"])
    origin = np.column_stack([library["geometry_disk_origin_" + a] for a in "xyz"]).astype(dtype)
    rotation = np.stack([np.column_stack([library[f"geometry_disk_axis_{ax}_{a}"] for a in "xyz"])
                         for ax in "xyz"], axis=2).astype(dtype)
    step_slot = np.searchsorted(disk_ids, library["step_disk"][source])
    entry_slot = np.searchsorted(disk_ids, disk[row])
    entry_global = origin[entry_slot] + np.einsum("nij,nj->ni", rotation[entry_slot],
                                                  np.asarray(queries["local_position"], dtype)[row])
    entry_local = np.einsum("nji,nj->ni", rotation[step_slot], entry_global - origin[step_slot])
    turn = _azimuth_rotation(phi[row]).astype(dtype)
    local = entry_local + np.einsum("nij,nj->ni", turn, library["step_relative_position"][source].astype(dtype))
    position = origin[step_slot] + np.einsum("nij,nj->ni", rotation[step_slot], local)
    return dict(position=position, time=(np.asarray(queries["time"], dtype)[row] +
                                         library["step_relative_time"][source].astype(dtype)),
                energy_g4=library["step_energy_g4"][source].astype(dtype),
                energy_birks=library["step_energy_birks"][source].astype(dtype),
                donor_crystal=library["step_crystal"][source], disk=library["step_disk"][source],
                entry=row, offsets=offsets, donor=donors, supported=supported, fallback=fallback,
                distance=entry_distance)


def axes(library):
    """Cell keys (mu2efastsim.library.axes.Axis) besides species, face and disk."""
    frame = "entrant's disk frame turned to the entry azimuth"
    return (LA.Axis("energy", "MeV", library["bins_energy"], frame=frame, array="bins_energy", policy="clip",
                    description="total entry energy"),
            LA.Axis("cosine", "1", library["bins_cosine"], frame=frame, array="bins_cosine", policy="clip",
                    description="entry direction along the face normal (disk axis or outward radius)"),
            LA.Axis("front_radius", "mm", library["bins_front_radius"], frame=frame, array="bins_front_radius",
                    policy="clip", description="front-face entry radius from the disk axis"),
            LA.Axis("inner_depth", "mm", library["bins_inner_depth"], frame=frame, array="bins_inner_depth",
                    policy="clip", description="inner-edge entry position along the disk axis from the disk origin"))


def manifest_fields(library, metadata, **provenance):
    fields = dict(
        name="calo.showers.entrants." + "".join(ch if ch.isalnum() or ch in "_-" else "_"
                                                for ch in str(metadata.get("name", "unnamed"))),
        subsystem="calo", version=1, kind="pool",
        description="Per-entrant Geant4 calorimeter shower patterns (all CSS of an entrant's subtree) per cell "
                    "(species, face, disk, energy, face cosine, radius or depth), placed at a query entry by the "
                    "rotation about the disk axis and translation (sample); sparse cells use the closest occupied "
                    "cell of the same species, face and disk.",
        axes=[a.describe() for a in axes(library)],
        outputs=[dict(name="steps", unit="mm, ns, MeV", dtype="float64", description="placed CSS of the donor"),
                 dict(name="donor", unit="1", dtype="int64", description="library row, -1 when unsupported"),
                 dict(name="fallback", unit="1", dtype="bool", description="another cell or outside the bins")],
        dtype=dict(storage={"steps": "float64"}, compute=["float32", "float64"],
                   notes="sample(dtype=...) places in either precision"),
        rng=dict(host="numpy.random.Generator: one uniform per query", device="none"),
        domain=dict(coverage="front and inner-edge entrants; back and outer-edge entrants are not modelled",
                    fallbacks=[dict(name="fallback", levels={"0": "own cell", "1": "closest occupied cell"})]),
        config_hash=None, source=dict(description="metadata: " + json.dumps(metadata, sort_keys=True)[:2000], samples=[]),
    )
    fields.update(provenance)
    fields.setdefault("validation", dict(status="not validated", record=None))
    unrecorded = dict(provenance.get("unrecorded", {}))
    for k in ("generator", "release", "seeds", "geometry", "split"):
        if fields.get(k) is None:
            fields[k] = None
            unrecorded.setdefault(k, "not given to save()")
    unrecorded.setdefault("config_hash", "no build configuration manifest")
    fields["unrecorded"] = unrecorded
    return fields


def save(path, library, metadata, **provenance):
    arrays = dict(library, metadata=np.array(json.dumps(metadata, sort_keys=True)))
    return LS.save(Path(path), arrays, manifest_fields(library, metadata, **provenance))


def load(path):
    arrays = dict(LS.load(Path(path)).arrays)
    return arrays, json.loads(str(arrays.pop("metadata")))


ROOT_ENTRY = ("species", "face", "disk", "energy", "cosine", "key", "azimuth", "time")


def export_root(library, path):
    """Write the library as ROOT trees for the Mu2eG4 model; returns the library content hash it records.

    Trees: ``entries`` (one row per entrant: species, face, disk, energy, cosine, key, azimuth, time, psi when the
    library has it (:func:`transverse_angle`), cell_energy,
    cell_cosine, cell_key, outside, step_begin, step_end), ``steps`` (x, y, z relative, t relative, energy_g4,
    energy_birks, disk, crystal), ``bins`` (one tree per axis, branch ``edge``), ``disks`` (disk, origin and axes) and
    the TObjString ``content_hash`` (sha256 of the npz arrays, mu2efastsim.library.hashing.content_hash).
    """
    digest = content_hash({k: v for k, v in library.items() if k != "metadata"})
    entries = {k: np.asarray(library["entry_" + k]) for k in ROOT_ENTRY + (("psi",) if "entry_psi" in library else ())}
    entries.update(cell_energy=library["entry_cell"][:, 0], cell_cosine=library["entry_cell"][:, 1],
                   cell_key=library["entry_cell"][:, 2], outside=library["entry_outside_bins"].astype(np.int32),
                   step_begin=library["offsets"][:-1], step_end=library["offsets"][1:])
    rel = library["step_relative_position"]
    steps = dict(x=rel[:, 0], y=rel[:, 1], z=rel[:, 2], t=library["step_relative_time"],
                 energy_g4=library["step_energy_g4"], energy_birks=library["step_energy_birks"],
                 disk=library["step_disk"], crystal=library["step_crystal"])
    disks = {"disk": np.asarray(library["geometry_disk_disk"], dtype=np.int64)}
    for prefix in ("origin", "axis_x", "axis_y", "axis_z"):
        for a in "xyz":
            disks[f"{prefix}_{a}"] = np.asarray(library[f"geometry_disk_{prefix}_{a}"], dtype=float)
    with uproot.recreate(path) as out:
        out.mktree("entries", {k: np.asarray(v).dtype for k, v in entries.items()})
        out["entries"].extend({k: np.asarray(v) for k, v in entries.items()})
        out.mktree("steps", {k: np.asarray(v).dtype for k, v in steps.items()})
        out["steps"].extend({k: np.asarray(v) for k, v in steps.items()})
        out.mktree("disks", {k: v.dtype for k, v in disks.items()})
        out["disks"].extend(disks)
        for axis in ("energy", "cosine", "front_radius", "inner_depth"):
            out.mktree("bins_" + axis, {"edge": np.float64})
            out["bins_" + axis].extend({"edge": np.asarray(library["bins_" + axis], dtype=float)})
        out["content_hash"] = digest
    return digest


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Build a per-entrant shower library from CaloLibraryDumper exports")
    parser.add_argument("out", type=Path, help="library .npz")
    parser.add_argument("exports", type=Path, nargs="+")
    parser.add_argument("--root", type=Path, help="also write the ROOT export for Mu2eG4")
    parser.add_argument("--name", default="unnamed")
    parser.add_argument("--skip-failed", action="store_true",
                        help="drop failed-transport events instead of raising (counted in the census)")
    args = parser.parse_args(argv)
    entrants, css, geometry, census = read_many(args.exports, skip_failed=args.skip_failed)
    library = build(entrants, css, geometry)
    metadata = dict(name=args.name, exports=[str(p) for p in args.exports], census=census,
                    entrants=int(len(entrants["face"])), steps=int(len(css["entrant"])))
    save(args.out, library, metadata)
    if args.root:
        print("content_hash", export_root(library, args.root))
    print(json.dumps(dict(entrants=metadata["entrants"], steps=metadata["steps"],
                          by_face=[c["by_face"] for c in census]), indent=1)[:4000])


if __name__ == "__main__":
    main()
