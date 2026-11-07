"""Reconcile saved production art CSS with numeric export; no building or sampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from calofastsim import artsteps
from . import export

SOURCE_LABEL = "compressDetStepMCs__Primary"
FIELDS = {"sim_product": "sim_product_id", "sim_key": "sim_key", "n_compress": "n_compress",
          "volume": "crystal", "time": "time", "energy_g4": "energy_g4",
          "energy_birks": "energy_birks", "momentum_in": "momentum_in",
          **{"pos_" + a: "position_" + a for a in "xyz"}}


def _equal(left, right, label):
    # Do not narrow exported doubles back to source floats: that hides changed bits.
    if not np.array_equal(left, right):
        detail = ""
        if np.shape(left) == np.shape(right) and np.ndim(left) == 1:
            bad = np.flatnonzero(np.asarray(left) != np.asarray(right))
            detail = f" at {len(bad)} rows; first source/export values {np.asarray(left)[bad[:4]]} / {np.asarray(right)[bad[:4]]}"
        raise ValueError(f"{label} mismatch{detail}")


def reconcile(raw, ids, events, showers):
    """Check all ordered source events and CSS columns exactly, including empty events."""
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.shape[1] != 3:
        raise ValueError("source event IDs must have shape (events, 3)")
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError("duplicate source event identity")
    _equal(ids, np.column_stack([events[k] for k in export.EVENT]), "ordered event IDs")
    entry = np.asarray(raw["entry"])
    if entry.ndim != 1 or entry.dtype.kind not in "iu":
        raise ValueError("source CSS entry must be a one-dimensional integer column")
    if np.any(entry >= len(ids)) or np.any(entry < 0):
        raise ValueError("source CSS entry out of range")
    if np.any(entry[1:] < entry[:-1]):
        raise ValueError("source CSS entry order mismatch")
    counts = np.bincount(entry.astype(np.int64), minlength=len(ids))
    _equal(counts, events["n_showers"], "source CSS counts versus events.n_showers")
    for axis, field in enumerate(export.EVENT):
        _equal(ids[entry, axis], showers[field], f"ordered shower {field}")
    ordinal = np.arange(len(entry)) - np.repeat(np.cumsum(counts) - counts, counts)
    _equal(ordinal, showers["index"], "shower local index")
    for source, target in FIELDS.items():
        if np.shape(raw[source]) != entry.shape:
            raise ValueError(f"source CSS {source} row count mismatch")
        _equal(raw[source], showers[target], f"CSS {source} -> {target}")
    return dict(events=len(ids), css_rows=len(entry), all_fields_exact=True)


def geometry_check(source):
    """Check the exported disk/crystal frames and saved VD crossings, not material parity."""
    xyz = lambda prefix: tuple(prefix + "_" + a for a in "xyz")
    disks = export.columns(source, "disks", ("run", "disk") + xyz("origin") + xyz("size") +
                           xyz("axis_x") + xyz("axis_y") + xyz("axis_z"))
    crystals = export.columns(source, "crystals", ("run", "crystal", "disk") + xyz("position") +
                              xyz("local_position") + xyz("size"))
    vd = export.columns(source, "virtualdetectors", ("run", "id", "half_length") + xyz("position"))
    steps = export.columns(source, "steps", ("run", "kind", "volume") + xyz("position"))
    for table, fields in ((disks, ("run", "disk")), (crystals, ("run", "crystal")), (vd, ("run", "id"))):
        ids = export.keys(table, fields)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate geometry identity")
    rotation = np.stack([export.vector(disks, "axis_" + a) for a in "xyz"], axis=2)
    if not np.allclose(np.einsum("nji,njk->nik", rotation, rotation), np.eye(3), atol=1e-12, rtol=0):
        raise ValueError("nonorthonormal disk frame")
    if np.any(export.vector(crystals, "size") <= 0) or np.any(export.vector(disks, "size") <= 0):
        raise ValueError("invalid geometry sizes")
    cr = export.join(export.keys(disks, ("run", "disk")), export.keys(crystals, ("run", "disk")))
    expected = export.vector(disks, "origin")[cr] + np.einsum(
        "nij,nj->ni", rotation[cr], export.vector(crystals, "local_position"))
    if not np.allclose(expected, export.vector(crystals, "position"), atol=1e-9, rtol=0):
        raise ValueError("crystal local/global frame mismatch")
    front = np.isin(vd["id"], (73, 75))
    faces = {k: v[front] for k, v in vd.items()}
    faces["disk"] = (faces["id"] - 73) // 2
    vr = export.join(export.keys(disks, ("run", "disk")), export.keys(faces, ("run", "disk")))
    if not np.array_equal(np.sort(export.keys(faces, ("run", "disk"))),
                          np.sort(export.keys(disks, ("run", "disk")))):
        raise ValueError("missing disk-front VD")
    expected_front = export.vector(disks, "origin")[vr] - rotation[vr, :, 2] * (
        disks["size_z"][vr] / 2 - faces["half_length"])[:, None]
    if not np.allclose(expected_front, export.vector(faces, "position"), atol=1e-9, rtol=0):
        raise ValueError("disk-front VD frame mismatch")
    at_front = (steps["kind"] == 0) & np.isin(steps["volume"], (73, 75))
    crossings = {k: v[at_front] for k, v in steps.items()}
    crossings["id"] = crossings["volume"]
    sr = export.join(export.keys(faces, ("run", "id")), export.keys(crossings, ("run", "id")))
    residual = np.abs(np.sum((export.vector(crossings, "position") -
                              export.vector(faces, "position")[sr]) * rotation[vr[sr], :, 2], axis=1))
    if np.any(residual > faces["half_length"][sr] + 1e-4):
        raise ValueError("saved front crossing outside exported VD slab")
    return dict(disks=len(disks["disk"]), crystals=len(crystals["crystal"]),
                crystals_per_disk=np.bincount(cr, minlength=len(disks["disk"])).tolist(),
                front_vd_z_mm=faces["position_z"].tolist(), checked_front_crossings=len(residual),
                max_front_plane_distance_mm=float(np.max(residual)) if len(residual) else None,
                scope="table frames and saved crossings only; release/configuration parity checked separately")


def main(argv=None):
    """Check a bounded saved development extraction, without fitting or sampling."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--art", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-geometry", action="store_true", help="check table frames and saved front-VD slabs")
    args = parser.parse_args(argv)
    # Reject graph/census defects before invoking the independent raw decoder.
    _, _, _, census, _ = export.read(args.tables, input_mode="primary")
    raw, ids = artsteps.read_steps(args.art, label=SOURCE_LABEL)
    with uproot.open(args.tables) as source:
        events = export.columns(source, "events", export.EVENT + ("n_showers",))
        showers = export.columns(source, "showers", export.EVENT + ("index",) + tuple(FIELDS.values()))
        geometry = geometry_check(source) if args.check_geometry else None
    exact = reconcile(raw, ids, events, showers)
    report = dict(art=str(args.art), tables=str(args.tables), input_mode="primary",
                  source_label=SOURCE_LABEL, reconciliation=exact, census=census,
                  denominator="stored/selected-population events; not unconditional generated-event acceptance",
                  geometry=geometry if geometry is not None else "not checked here; actual sample/exporter geometry parity is a separate gate")
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
