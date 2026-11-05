"""Read saved CaloLibraryDumper numeric ROOT tables; never run art or Geant4."""
from __future__ import annotations

import numpy as np
import uproot


EVENT = ("run", "subrun", "event")
IDENTITY = EVENT + ("sim_product_id", "sim_key")


def keys(table, fields):
    return np.rec.fromarrays([np.asarray(table[k], dtype=np.int64) for k in fields], names=fields)


def vector(table, prefix):
    return np.column_stack([table[prefix + "_" + axis] for axis in "xyz"])


def join(reference, query):
    order = np.argsort(reference)
    index = np.searchsorted(reference[order], query)
    if len(reference) == 0 and len(query):
        raise ValueError("missing reference rows")
    if np.any(index >= len(reference)) or not np.array_equal(reference[order[index]], query):
        raise ValueError("unresolved table identity")
    return order[index]


def columns(source, table, fields):
    return source["dump/" + table].arrays(list(fields), library="np")


def read(path):
    """One first forward direct-root front VD entry and the entire event CSS.

    The supported input is the single-CE StageParticle pilot, with the upstream
    parent reseated into g4run under its original key. Photon-only and side-only
    arrivals remain in the event census but are not direct-electron entries.
    """
    with uproot.open(path) as source:
        events = columns(source, "events", EVENT + ("g4_status", "n_showers"))
        generated = columns(source, "generated", EVENT + ("pdg", "parent_key", "parent_valid"))
        products = columns(source, "products", EVENT + ("kind", "product_id"))
        names = source["dump/products"]["branch_name"].array(library="np").astype(str)
        transport = (products["kind"] == -1) & np.char.endswith(names, "_g4run__CaloLibraryCE.")
        products = {k: v[transport] for k, v in products.items()}
        event_ids = keys(events, EVENT)
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError("duplicate source event")
        if len(generated["pdg"]) != len(event_ids) or np.any(generated["pdg"] != 11) or not np.all(generated["parent_valid"]):
            raise ValueError("expected one parent-linked CE per event")
        if len(products["product_id"]) != len(event_ids):
            raise ValueError("expected one raw g4run product per event")
        gi = join(keys(generated, EVENT), event_ids)
        pi = join(keys(products, EVENT), event_ids)
        parent = {**events, "parent_product_id": products["product_id"][pi], "parent_key": generated["parent_key"][gi]}
        particles = columns(source, "particles", IDENTITY + ("pdg", "parent_product_id", "parent_key", "parent_valid"))
        pfields = EVENT + ("parent_product_id", "parent_key")
        root = ((particles["pdg"] == 11) & particles["parent_valid"] &
                (particles["sim_product_id"] == particles["parent_product_id"]) &
                np.isin(keys(particles, pfields), keys(parent, pfields)))
        roots = {k: v[root] for k, v in particles.items()}
        if len(roots["event"]) != len(event_ids) or not np.array_equal(np.sort(keys(roots, EVENT)), np.sort(event_ids)):
            raise ValueError("generated parent did not identify exactly one transport CE per event")
        root_ids = keys(roots, IDENTITY)
        del particles
        fields = IDENTITY + ("kind", "volume", "time") + tuple(p + "_" + a for p in ("position", "momentum") for a in "xyz")
        steps = columns(source, "steps", fields)
        front = ((steps["kind"] == 0) & np.isin(steps["volume"], (73, 75)) & np.isin(keys(steps, IDENTITY), root_ids))
        steps = {k: v[front] for k, v in steps.items()}
        disks = columns(source, "disks", ("run", "disk") + tuple(p + "_" + a for p in ("origin", "size", "axis_x", "axis_y", "axis_z") for a in "xyz"))
        crystals = columns(source, "crystals", ("run", "crystal", "disk") + tuple(p + "_" + a for p in ("position", "size") for a in "xyz"))
        showers = columns(source, "showers", EVENT + ("crystal", "time", "energy_g4", "energy_birks") + tuple("position_" + a for a in "xyz"))
        vd = columns(source, "virtualdetectors", ("run", "id") + tuple("position_" + a for a in "xyz"))
    if np.any(events["g4_status"] != 0):
        raise ValueError("input contains failed transport events; select completed input events first")
    rotation = np.stack([vector(disks, "axis_" + a) for a in "xyz"], axis=2)
    disk_id = (steps["volume"] - 73) // 2
    drow = join(keys(disks, ("run", "disk")), keys({**steps, "disk": disk_id}, ("run", "disk")))
    momentum = vector(steps, "momentum")
    magnitude = np.linalg.norm(momentum, axis=1)
    cosine = np.divide(np.sum(momentum * rotation[drow, :, 2], axis=1), magnitude,
                       out=np.zeros_like(magnitude), where=magnitude > 0)
    forward = np.flatnonzero(cosine > 0)
    erow = join(event_ids, keys(steps, EVENT))
    order = forward[np.lexsort((steps["time"][forward], erow[forward]))]
    first = order[np.r_[True, np.diff(erow[order]) != 0]] if len(order) else order
    first = first[np.argsort(erow[first])]
    position = vector(steps, "position")[first]
    local = np.einsum("nji,nj->ni", rotation[drow[first]], position - vector(disks, "origin")[drow[first]])
    entries = dict(event_row=erow[first], disk=disk_id[first], position=position,
                   direction=momentum[first] / magnitude[first, None], energy=np.sqrt(magnitude[first] ** 2 + .51099895 ** 2),
                   cosine=np.clip(cosine[first], 0, 1), radius=np.linalg.norm(local[:, :2], axis=1), time=steps["time"][first])
    srow = join(event_ids, keys(showers, EVENT))
    lookup = np.full(len(event_ids), -1, dtype=np.int64)
    lookup[entries["event_row"]] = np.arange(len(first))
    take = lookup[srow] >= 0
    crystal_row = join(keys(crystals, ("run", "crystal")), keys(showers, ("run", "crystal")))
    crystal_disk = join(keys(disks, ("run", "disk")), keys(crystals, ("run", "disk")))
    global_position = vector(crystals, "position")[crystal_row] + np.einsum("nij,nj->ni", rotation[crystal_disk[crystal_row]], vector(showers, "position"))
    css = {k: showers[k][take] for k in ("energy_g4", "energy_birks", "time", "crystal")}
    css.update(entry=lookup[srow[take]], position=global_position[take], disk=crystals["disk"][crystal_row[take]])
    total = np.bincount(srow, weights=showers["energy_birks"], minlength=len(event_ids))
    census = dict(events=len(event_ids), front_entry_events=len(first), empty_css_events=int(np.sum(events["n_showers"] == 0)),
                  non_front_entry_events=int(np.sum(lookup < 0)),
                  non_front_entry_positive_events=int(np.sum((lookup < 0) & (total > 0))),
                  css_rows=len(srow), retained_css_rows=int(np.sum(take)))
    geometry = {"crystal_" + k: v for k, v in crystals.items()}
    geometry.update({"disk_" + k: v for k, v in disks.items()})
    geometry.update({"vd_" + k: v for k, v in vd.items()})
    return entries, css, geometry, census, total
