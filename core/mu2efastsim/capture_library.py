"""Audited component captures to a component_response bank (docs/component-training.md, "Captures to a bank").

Every capture must pass the strict linked audit (capture.audit with links) before it is converted; nothing is
repaired, clamped or dropped. A trial is one capture event: its primary's local entrance state, every recorded track
as the local particle tree (root first, parents before children), its in-component steps and its first-boundary
exits (first exits and exits at birth), with times relative to the primary's entrance. All files must describe one
component type: geometry and physics labels, envelope half sizes and the volume inventory below the envelope. The
bank is written by component_response.save_library; bank_admissible stays false, and the generator and energy
accounting stay unrecorded (docs/libraries.md, section 8).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import capture, component_response as cr
from .library import hashing
from .saved_g4 import frame, match, require, unique

EVENT = capture.EVENT
VOLUME_LABELS = ("name", "copy", "logical", "motherLogical", "material", "solid")
PLACEMENT_ATOL = 1e-7
EXIT_REASONS = (2, 4)


def _xyz(table, prefix):
    return capture.vectors(table, prefix, "xyz")


def _volumes(tables):
    """Rows of the envelope and its descendants (first event), checked identical in every event of the file."""
    e, v = tables["events"], tables["volumes"]
    envelope = str(e["envelope"][0])
    require(np.all(e["envelope"] == envelope), "envelope differs between events")
    events = unique(e, EVENT, "event").assign(_event=np.arange(len(e["event"])))
    vr = match(frame(v, EVENT), events, EVENT, "volume")["_event"].to_numpy()
    rows = [np.flatnonzero(vr == i)[np.argsort(v["index"][vr == i], kind="stable")] for i in range(len(events))]
    for field in ("id", "density", *VOLUME_LABELS):
        require(all(np.array_equal(v[field][r], v[field][rows[0]]) for r in rows), f"volume {field} differs between events")
    first = rows[0]
    names, logical, mother = (v[f][first].astype(str) for f in ("name", "logical", "motherLogical"))
    keep = names == envelope
    require(keep.sum() == 1, "envelope census mismatch")
    while True:
        grown = keep | np.isin(mother, logical[keep])
        if np.array_equal(grown, keep):
            break
        keep = grown
    daughters = keep & (names != envelope)
    for field in ("translation", "rotation"):
        require(all(np.array_equal(v[field][r][daughters], v[field][first][daughters]) for r in rows),
                f"volume {field} below the envelope differs between events")
    return first[keep], envelope


def _inventory(tables, rows, envelope):
    """The component's volume inventory: labels, densities and mother-relative placements (envelope placement zeroed)."""
    v = tables["volumes"]
    out = {f: v[f][rows].astype(str) if f != "copy" else v[f][rows].astype(np.int64) for f in VOLUME_LABELS}
    out["density"] = v["density"][rows].astype(np.float64)
    is_env = out["name"] == envelope
    out["translation"] = np.where(is_env[:, None], 0., v["translation"][rows].astype(np.float64))
    out["rotation"] = np.where(is_env[:, None], np.eye(3).ravel(), v["rotation"][rows].astype(np.float64))
    return out


def inventory_hash(inventory):
    """Digest of a volume inventory: labels and densities exactly, placements rounded to 1e-9 mm."""
    rows = [[str(inventory[f][i]) for f in VOLUME_LABELS] + [repr(float(inventory["density"][i]))]
            + [repr(round(float(x), 9) + 0.) for x in (*inventory["translation"][i], *inventory["rotation"][i])]
            for i in range(len(inventory["name"]))]
    return hashing.digest(rows)


def _same_inventory(a, b):
    labels = all(np.array_equal(a[f], b[f]) for f in (*VOLUME_LABELS, "density"))
    placed = all(a[f].shape == b[f].shape and np.allclose(a[f], b[f], rtol=0, atol=PLACEMENT_ATOL)
                 for f in ("translation", "rotation"))
    return labels and placed


def trials(tables, links, expected_events):
    """One audited capture as bank pieces: (component, inventory, entrances, particles, steps, exits, counts, info)."""
    require(links is not None, "a bank needs the capture's links and manifest")
    audit = capture.audit(tables, expected_events, links=links)
    birth, _, _ = capture.check_links(tables, links)
    e, t, s = tables["events"], tables["tracks"], tables["steps"]
    for field in ("geometryTag", "physicsTag"):
        require(np.all(e[field] == e[field][0]), f"{field} differs between events")
    half = np.asarray(e["halfSize"], dtype=np.float64)
    require(np.all(half == half[0]), "halfSize differs between events")
    vrows, envelope = _volumes(tables)
    inventory = _inventory(tables, vrows, envelope)
    local_volume = {int(i): k for k, i in enumerate(tables["volumes"]["id"][vrows])}
    component = cr.ComponentType(str(e["geometryTag"][0]), str(e["physicsTag"][0]), tuple(half[0]), len(vrows))

    events = unique(e, EVENT, "event").assign(_event=np.arange(len(e["event"])))
    tr = match(frame(t, EVENT), events, EVENT, "track")["_event"].to_numpy()
    order = np.lexsort((t["index"], tr))
    tr_sorted = tr[order]
    particle_counts = np.bincount(tr, minlength=len(events))
    particle_offsets = np.r_[0, np.cumsum(particle_counts)]
    local = np.empty(len(order), dtype=np.int64)
    local[order] = np.arange(len(order)) - particle_offsets[tr_sorted]
    keys = frame(t, EVENT).assign(track=t["track"], _local=local)
    primary = t["parent"] == 0
    parent_local = np.full(len(order), -1, dtype=np.int64)
    parents = frame(t, EVENT).assign(track=t["parent"])
    parent_local[~primary] = match(parents.loc[~primary], keys, [*EVENT, "track"], "parent")["_local"].to_numpy()
    require(np.all(local[primary] == 0), "primary is not the first track of its event")
    proot = np.flatnonzero(primary)[np.argsort(tr[primary])]
    require(np.array_equal(tr[proot], np.arange(len(events))), "expected one primary per event")
    t0 = t["entrance_time"][proot]

    particles = dict(pdg=t["pdg"][order].astype(np.int64), parent=parent_local[order],
                     exited=np.isin(t["reason"][order], EXIT_REASONS))
    entrances = dict(pdg=t["pdg"][proot].astype(np.int64), position=_xyz(t, "entrance_l")[proot],
                     momentum=_xyz(t, "entrance_lp")[proot], energy=t["entrance_energy"][proot].astype(np.float64))

    inside = np.flatnonzero(s["preInside"])
    step_keys = frame(s, EVENT).assign(track=s["track"])
    sl = match(step_keys.iloc[inside], keys, [*EVENT, "track"], "step")["_local"].to_numpy()
    se = match(frame(s, EVENT).iloc[inside], events, EVENT, "step")["_event"].to_numpy()
    sorder = np.lexsort((s["index"][inside], se))
    rows, se, sl = inside[sorder], se[sorder], sl[sorder]
    volume_ids = s["pre_volume"][rows]
    require(np.all(np.isin(volume_ids, list(local_volume))), "in-component step outside the envelope inventory")
    steps = dict(particle=sl, volume=np.array([local_volume[int(i)] for i in volume_ids], dtype=np.int64),
                 x0=_xyz(s, "pre_l")[rows], x1=_xyz(s, "post_l")[rows],
                 p0=_xyz(s, "pre_lp")[rows], p1=_xyz(s, "post_lp")[rows],
                 t0=s["pre_time"][rows] - t0[se], t1=s["post_time"][rows] - t0[se],
                 length=s["length"][rows].astype(np.float64), edep=s["deposit"][rows].astype(np.float64))

    first = np.flatnonzero(s["firstExit"])
    fl = match(step_keys.iloc[first], keys, [*EVENT, "track"], "exit")["_local"].to_numpy()
    fe = match(frame(s, EVENT).iloc[first], events, EVENT, "exit")["_event"].to_numpy()
    born = np.flatnonzero(birth)
    ex_event = np.r_[fe, tr[born]].astype(np.int64)
    ex_local = np.r_[fl, local[born]].astype(np.int64)
    position = np.vstack([_xyz(s, "post_l")[first], _xyz(t, "begin_l")[born]])
    momentum = np.vstack([_xyz(s, "post_lp")[first], _xyz(t, "begin_lp")[born]])
    energy = np.r_[s["post_energy"][first], t["begin_energy"][born]].astype(np.float64)
    time = np.r_[s["post_time"][first], t["begin_time"][born]] - t0[ex_event]
    eorder = np.lexsort((ex_local, ex_event))
    exits = dict(particle=ex_local[eorder], position=position[eorder], momentum=momentum[eorder],
                 energy=energy[eorder], time=time[eorder])
    counts = dict(particles=particle_counts, steps=np.bincount(se, minlength=len(events)),
                  exits=np.bincount(ex_event, minlength=len(events)))
    info = dict(events=len(events), g4Version=sorted(set(map(str, links["manifest"]["g4Version"]))),
                process_manifest_hash=sorted(set(map(str, links["manifest"]["manifestHash"]))),
                run_subrun_event=[[int(x) for x in row] for row in frame(e, EVENT).to_numpy()],
                exits_at_birth=int(birth.sum()), component_deposit_MeV=audit["component_deposit"])
    return component, inventory, entrances, particles, steps, exits, counts, info


def _concat(parts, schema):
    return {k: np.concatenate([p[k] for p in parts]) for k in schema}


def _dataset(path, relative_to):
    """The capture's name in the manifest: its path below relative_to (no site paths), refused outside it."""
    path, root = Path(path).resolve(), Path(relative_to or Path.cwd()).resolve()
    require(path.is_relative_to(root), f"capture {path.name} is outside {root.name}; name datasets relative to their tree")
    return path.relative_to(root).as_posix()


def library_from_captures(paths, expected_events, *, relative_to=None, sample_info=None):
    """The Library of one component type from audited capture files, and the source/geometry provenance.

    Datasets are named relative to relative_to (default: the working directory). sample_info, one dict per path,
    adds the caller's description of each sample (e.g. its campaign state) without replacing recorded keys.
    """
    require(len(paths) > 0, "no capture files")
    sample_info = list(sample_info) if sample_info is not None else [{}] * len(paths)
    require(len(sample_info) == len(paths), "sample_info needs one entry per capture")
    pieces, samples = [], []
    for path, extra in zip(paths, sample_info):
        tables, links = capture.read(path), capture.read_links(path)
        component, inventory, *rest, info = trials(tables, links, expected_events)
        if pieces:
            require(component == pieces[0][0], f"{path}: different component type")
            require(_same_inventory(inventory, pieces[0][1]), f"{path}: different volume inventory")
            require(info["process_manifest_hash"] == samples[0]["process_manifest_hash"] and
                    info["g4Version"] == samples[0]["g4Version"], f"{path}: different Geant4 process manifest")
        pieces.append((component, inventory, *rest))
        sample = dict(dataset=_dataset(path, relative_to), selection="every event (strict linked capture audit)",
                      sha256=hashing.file_sha256(path), **info)
        require(not set(extra) & {*sample, "supplied"},
                f"sample_info cannot replace recorded keys {sorted(set(extra) & {*sample, 'supplied'})}")
        samples.append({**sample, **extra, **({"supplied": sorted(extra)} if extra else {})})
    component, inventory = pieces[0][0], pieces[0][1]
    entrances, particles, steps, exits = (_concat([p[i] for p in pieces], schema) for i, schema in
                                          ((2, cr.ENTRANCE), (3, cr.PARTICLES), (4, cr.STEPS), (5, cr.EXITS)))
    offsets = {name: np.r_[0, np.cumsum(np.concatenate([p[6][name] for p in pieces]))]
               for name in ("particles", "steps", "exits")}
    library = cr.Library(component, entrances, particles, offsets["particles"], steps, offsets["steps"],
                         exits, offsets["exits"])
    provenance = dict(
        source=dict(samples=samples, description="component captures (schema 1, links and manifest), one trial per "
                                                 "capture event, in file then event order"),
        geometry=dict(tag=component.geometry, hash=inventory_hash(inventory),
                      hash_of="the volume inventory below the capture envelope: names, copies, logical and mother "
                              "volumes, materials, solids, densities and mother-relative placements (to 1e-9)",
                      physics=component.physics, half_size=list(component.half_size), volumes=component.volumes,
                      volume_names=[str(x) for x in inventory["name"]]))
    return library, provenance


def convert(paths, output, expected_events, *, relative_to=None, sample_info=None, **provenance):
    """Write the bank of the given captures with component_response.save_library; returns its manifest.

    provenance gives further manifest fields (seeds, release, split, config_hash, notes); its source and geometry
    entries add keys beside the recorded ones and cannot replace them. Each of source and geometry lists its keys
    read from the captures ("recorded") and those the caller supplied ("supplied"); a sample lists its supplied keys.
    """
    library, recorded = library_from_captures(paths, expected_events, relative_to=relative_to, sample_info=sample_info)
    for field in ("source", "geometry"):
        extra = dict(provenance.pop(field, None) or {})
        own = {**recorded[field], "recorded": sorted(recorded[field])}
        clash = set(extra) & {*own, "supplied"}
        require(not clash, f"{field} cannot replace recorded keys {sorted(clash)}")
        recorded[field] = {**own, **extra, **({"supplied": sorted(extra)} if extra else {})}
    return cr.save_library(output, library, **{**recorded, **provenance})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--expected-events", type=int, required=True, help="events in every capture file")
    parser.add_argument("--output", type=Path, required=True, help="bank .npz")
    parser.add_argument("--relative-to", type=Path, help="name datasets relative to this directory (default: cwd)")
    parser.add_argument("--provenance", type=Path, help="JSON object of further manifest fields (seeds, release, ...)")
    parser.add_argument("--sample-info", type=Path, help="JSON list, one object per capture, added to its sample")
    args = parser.parse_args(argv)
    require(args.output.suffix == ".npz" and all(args.output.resolve() != p.resolve() for p in args.captures),
            "output must be a new .npz, not a capture")
    provenance = json.loads(args.provenance.read_text()) if args.provenance else {}
    info = json.loads(args.sample_info.read_text()) if args.sample_info else None
    manifest = convert(args.captures, args.output, args.expected_events, relative_to=args.relative_to,
                       sample_info=info, **provenance)
    print(json.dumps({k: manifest[k] for k in ("name", "kind", "content_hash", "unrecorded", "validation")}, indent=2))


if __name__ == "__main__":
    main()
