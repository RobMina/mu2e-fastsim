"""Opt-in G4Beamline ROOT crossing extraction, not an inclusive beam/exposure model.

Stored rows (including recrossings) are preserved in file/plane/stored-row order.
No earliest-hit, forward-direction or G4BL-to-Mu2e coordinate transform is inferred.
Only local/staged inputs are accepted; uproot is imported only when read is called.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .tsreference import _integers, _source, _state, _tree

PDGS = (13, -13, 211, -211)
PLANE_PATHS = {
    "COL1in": "VirtualDetector/Coll_01_DetIn",
    "COL1out": "VirtualDetector/Coll_01_DetOut",
    "COL3in": "VirtualDetector/Coll_03_DetIn",
    "COL3mid": "VirtualDetector/Coll_03_DetMid",
    "COL3out": "VirtualDetector/Coll_03_DetOut",
    "COL5in": "VirtualDetector/Coll_05_DetIn",
    "COL5out": "VirtualDetector/Coll_05_DetOut",
}
PLANES = tuple(PLANE_PATHS)
INTEGER = ("EventID", "TrackID", "ParentID", "PDGid")
STATE = ("Weight", "x", "y", "z", "Px", "Py", "Pz", "t")
BRANCHES = INTEGER + STATE
COLUMNS = ("ifile", "iplane", "source_row") + BRANCHES
IDENTITY = ("ifile", "EventID", "TrackID")


def _local_path(value):
    text = str(value)
    if urlsplit(text).scheme or "://" in text:
        raise ValueError("Inputs must be local/staged files, not remote URLs")
    path = Path(value)
    try:
        if not path.is_file():
            raise ValueError(f"Missing or non-file ROOT input: {path}")
        return path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Cannot access local ROOT input: {path}") from exc


def _pdg_codes(values, n):
    """Validate stored codes, flagging rounded float32 nuclei rather than inventing isotope identity."""
    a = np.asarray(values)
    if a.shape != (n,) or a.dtype.kind not in "iuf":
        raise ValueError("PDGid: expected a one-dimensional numeric integer column")
    if a.dtype.kind == "f":
        if not np.all(np.isfinite(a)) or not np.all(a == np.floor(a)):
            raise ValueError("PDGid: nonfinite or nonintegral code")
        if np.any(a < -(2 ** 63)) or np.any(a >= 2 ** 63):
            raise ValueError("PDGid: outside int64 range")
        unsafe = np.abs(a) >= 2 ** (np.finfo(a.dtype).nmant + 1)
    else:
        if a.dtype.kind == "u" and np.any(a > np.iinfo(np.int64).max):
            raise ValueError("PDGid: outside int64 range")
        unsafe = np.zeros(n, dtype=bool)
    return a.astype(np.int64), unsafe


def _planes(planes):
    planes = [planes] if isinstance(planes, str) else list(planes)
    if not planes or any(not isinstance(p, str) or not p.strip() or ";" in p for p in planes):
        raise ValueError("planes must be nonempty tree paths/aliases without cycles")
    paths = [PLANE_PATHS.get(p, p) for p in planes]
    if any(p != p.strip() or p.startswith("/") or p.endswith("/") or "//" in p
           or any(part in (".", "..") for part in p.split("/")) for p in paths):
        raise ValueError("planes must be canonical relative ROOT tree paths")
    if len(set(paths)) != len(paths):
        raise ValueError("planes must be unique, including resolved aliases")
    return paths


def _keys(data, names=IDENTITY):
    """Structured exact-int keys: no float coercion or Python row iteration."""
    keys = np.empty(len(data["EventID"]), dtype=[(name, np.int64) for name in names])
    for name in names:
        keys[name] = data[name]
    return keys


def _track_index(data):
    return np.unique(_keys(data), return_index=True, return_inverse=True)


def _weight_summary(weights):
    return dict(count=int(weights.size), unitweights=bool(np.all(weights == 1)),
                nonunit_count=int(np.count_nonzero(weights != 1)),
                zero_count=int(np.count_nonzero(weights == 0)),
                min=float(weights.min()) if weights.size else None,
                max=float(weights.max()) if weights.size else None)


def _weight_ambiguity(data, inverse, ntracks):
    low = np.full(ntracks, np.inf)
    high = np.full(ntracks, -np.inf)
    np.minimum.at(low, inverse, data["Weight"])
    np.maximum.at(high, inverse, data["Weight"])
    return int(np.count_nonzero(low != high))


def _observations(data):
    """Raw scalar ranges/sign counts only; units, coordinates and forward direction are unverified."""
    return dict(
        event_id_range=[int(data["EventID"].min()), int(data["EventID"].max())]
        if len(data["EventID"]) else None,
        raw_ranges={k: dict(min=float(data[k].min()), max=float(data[k].max()))
                    if len(data[k]) else dict(min=None, max=None) for k in STATE[1:]},
        raw_momentum_signs={k: dict(negative=int(np.count_nonzero(data[k] < 0)),
                                    zero=int(np.count_nonzero(data[k] == 0)),
                                    positive=int(np.count_nonzero(data[k] > 0)))
                            for k in ("Px", "Py", "Pz")})


def _selected_summary(selected, pdgs):
    tracks, first, inverse = _track_index(selected)
    plane_tracks = np.unique(_keys(selected, IDENTITY + ("iplane",)))
    events = np.unique(_keys(selected, ("ifile", "EventID")))
    return dict(
        selected_crossings=int(len(selected["EventID"])),
        unique_selected_tracks=int(len(tracks)),
        selected_species={str(pdg): dict(
            crossings=int(np.count_nonzero(selected["PDGid"] == pdg)),
            unique_tracks=int(np.count_nonzero(selected["PDGid"][first] == pdg))) for pdg in pdgs},
        recrossing_count=int(len(selected["EventID"]) - len(plane_tracks)),
        represented_eventIDs=int(len(events)),
        selected_weights=_weight_summary(selected["Weight"]),
        selected_tracks_with_varying_crossing_weights=_weight_ambiguity(selected, inverse, len(tracks)),
        selected_raw_observations=_observations(selected))


def _summary(data, pdgs):
    mask = np.isin(data["PDGid"], pdgs)
    selected = {k: v[mask] for k, v in data.items()}
    tracks, _, inverse = _track_index(data)
    return dict(_selected_summary(selected, pdgs), entries=int(len(mask)),
                all_species_unique_tracks=int(len(tracks)),
                all_species_recrossing_count=int(len(mask) - len(np.unique(
                    _keys(data, IDENTITY + ("iplane",))))),
                all_species_represented_eventIDs=int(len(np.unique(_keys(data, ("ifile", "EventID"))))),
                all_crossing_weights=_weight_summary(data["Weight"]),
                all_species_raw_observations=_observations(data),
                all_species_tracks_with_varying_crossing_weights=_weight_ambiguity(data, inverse, len(tracks)))


def _validate_tracks(data, path):
    """Repeated crossings are valid; PDG and parent must agree across ALL planes/species."""
    _, first, inverse = _track_index(data)
    for name in ("PDGid", "ParentID"):
        if np.any(data[name] != data[name][first][inverse]):
            raise ValueError(f"{path}: inconsistent {name} for repeated (EventID, TrackID) across crossings/planes")


def read(paths, planes=PLANES, pdgs=PDGS):
    """Return (row-aligned NumPy columns, JSON-safe receipt), validating before selection.

    planes accepts default aliases or arbitrary tree paths, e.g. NTuple/Z3712.
    Each requested plane is mandatory; its highest persisted cycle is used. Identity
    is (ifile, EventID, TrackID); row keys additionally contain iplane/source_row.
    ParentID is preserved but not resolved; absent parents do not imply a parent PDG.
    EventID >= 0, TrackID > 0, ParentID >= 0, and no self-parent are required.
    Weight must be finite and NONNEGATIVE: zero is explicitly allowed for killed
    weights. Raw weights are never normalised, deduplicated or assumed constant.
    All requested-plane species are validated, including species not selected.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("At least one ROOT input is required")
    plane_paths = _planes(planes)
    pdgs = _integers(pdgs, "pdgs")
    if not len(pdgs) or len(np.unique(pdgs)) != len(pdgs) or not np.all(np.isin(pdgs, PDGS)):
        raise ValueError("pdgs must be a nonempty distinct subset of 13, -13, 211, -211")
    sources = []
    seen_paths, seen_hashes = set(), set()
    for path in paths:
        local = _local_path(path)
        if local in seen_paths:
            raise ValueError(f"Duplicate input path: {local}")
        source = _source(local)
        if source["sha256"] in seen_hashes:
            raise ValueError(f"Byte-identical input files would double count crossings: {local}")
        seen_paths.add(local)
        seen_hashes.add(source["sha256"])
        sources.append(source)

    import uproot

    columns = {k: [] for k in COLUMNS}
    files = []
    for ifile, source in enumerate(sources):
        path = source["path"]
        with uproot.open(path) as root:
            # Inventory every tree and cycle, including unrequested trees and empty trees.
            inventory = {key: dict(classname=classname, num_entries=int(root[key].num_entries))
                         for key, classname in root.classnames(recursive=True, cycle=True).items()
                         if classname in ("TTree", "TNtuple", "TNtupleD") or "RNTuple" in classname}
            raw = {k: [] for k in COLUMNS}
            plane_receipts = []
            for iplane, tree_path in enumerate(plane_paths):
                tree, key, cycles = _tree(root, tree_path)
                if key not in inventory:
                    raise ValueError(f"{path}, {key}: requested plane is not a ROOT tree")
                missing = set(BRANCHES) - set(tree.keys())
                if missing:
                    raise ValueError(f"{path}, {key}: missing required branches {sorted(missing)}")
                arrays = tree.arrays(list(BRANCHES), library="np", how=dict)
                n = int(tree.num_entries)
                data = {k: _integers(arrays[k], k, n) for k in INTEGER if k != "PDGid"}
                data["PDGid"], unsafe_codes = _pdg_codes(arrays["PDGid"], n)
                data.update({k: _state(arrays[k], k, n) for k in STATE})
                if np.any(data["EventID"] < 0):
                    raise ValueError(f"{path}, {key}: EventID must be nonnegative")
                if np.any(data["TrackID"] <= 0):
                    raise ValueError(f"{path}, {key}: TrackID must be positive")
                if np.any(data["ParentID"] < 0):
                    raise ValueError(f"{path}, {key}: ParentID must be nonnegative")
                if np.any(data["ParentID"] == data["TrackID"]):
                    raise ValueError(f"{path}, {key}: self-parent TrackID is invalid")
                if np.any(data["Weight"] < 0):
                    raise ValueError(f"{path}, {key}: Weight must be nonnegative (zero is allowed)")
                data.update(ifile=np.full(n, ifile, dtype=np.int64),
                            iplane=np.full(n, iplane, dtype=np.int64),
                            source_row=np.arange(n, dtype=np.int64))
                for name in COLUMNS:
                    raw[name].append(data[name])
                plane_receipts.append(dict(iplane=iplane, tree=tree_path, selected_key=key,
                                           cycle=int(key.rsplit(";", 1)[1]), cycles=cycles,
                                           pdg_precision_unsafe_rows=int(np.count_nonzero(unsafe_codes)),
                                           **_summary(data, pdgs)))
            data = {k: np.concatenate(v) for k, v in raw.items()}
            _validate_tracks(data, path)
            selected = np.isin(data["PDGid"], pdgs)
            for name in COLUMNS:
                columns[name].append(data[name][selected])
            files.append(dict(source, ifile=ifile, tree_inventory=inventory,
                              planes=plane_receipts, **_summary(data, pdgs)))

    output = {k: np.concatenate(v) for k, v in columns.items()}
    receipt = dict(
        schema="ts-g4bl-crossings-v1", files=files, planes=plane_paths,
        plane_index={str(i): path for i, path in enumerate(plane_paths)}, pdgs=pdgs.tolist(),
        entries=sum(f["entries"] for f in files), **_selected_summary(output, pdgs),
        identity=list(IDENTITY), row_key=[*IDENTITY, "iplane", "source_row"],
        row_order="Input file order, requested plane order, stored source_row order; no deduplication",
        parent_scope="Same file/EventID; parents are not resolved and may be absent; parent PDG cannot be inferred",
        geometry_unverified=True, cuts_unverified=True, direction_unverified=True,
        normalization_unverified=True,
        coordinates="Raw input x/y/z and Px/Py/Pz; no G4BL-to-Mu2e transform inferred; units not certified",
        direction="All stored crossings retained; no earliest-hit, forward or time ordering selection",
        cuts="Only the requested PDG subset is selected; upstream production cuts/filters are unverified",
        pdg_precision="Muon/pion codes are exact; large stored float species codes may already be rounded, flagged per plane, not certified isotope identities",
        pdg_precision_unsafe_rows=sum(p["pdg_precision_unsafe_rows"] for f in files for p in f["planes"]),
        weight_policy="Finite nonnegative raw Weight, including zero; repeated crossings may have different weights",
        weight_ambiguity="Varying crossing weights are reported per track, never merged or treated as exposure",
        recrossing_definition="Selected crossings minus distinct (ifile, EventID, TrackID, iplane); cross-plane appearances are not recrossings",
        represented_eventIDs_definition="Count of distinct file-scoped EventIDs represented by selected crossings, not generated events",
        count_warning="Crossings, represented identities and unitweights are not POT exposure or generated-event counts",
        scope="Stored crossing reference only; no inclusive population, transmission, efficiency or normalization claim")
    # Enforce JSON safety as part of the public reader contract, not only at CLI time.
    json.dumps(receipt, allow_nan=False)
    return output, receipt


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.tsg4bl", description=__doc__)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--plane", action="append", help="Alias or tree path, repeatable; default seven collimator planes")
    ap.add_argument("--pdg", type=int, action="append", help="PDG 13, -13, 211 or -211, repeatable")
    ap.add_argument("--out", required=True, help="NPZ path; receipt is written alongside as .json")
    args = ap.parse_args(argv)
    output = Path(args.out)
    if output.suffix != ".npz":
        ap.error("--out must end in lowercase .npz")
    receipt_path = output.with_suffix(".json")
    try:
        if (output.resolve() == receipt_path.resolve()
                or (output.exists() and receipt_path.exists() and output.samefile(receipt_path))):
            raise ValueError("NPZ and receipt outputs must not alias one another")
        inputs = [_local_path(p) for p in args.roots]
        for target in (output, receipt_path):
            if target.resolve() in inputs or (target.exists() and any(target.samefile(p) for p in inputs)):
                raise ValueError("outputs must not overwrite ROOT inputs")
        columns, receipt = read(inputs, planes=args.plane if args.plane is not None else PLANES,
                                pdgs=args.pdg if args.pdg is not None else PDGS)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)
    np.savez_compressed(output, metadata_json=np.array(metadata), **columns)
    receipt["output"] = _source(output)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"Saved {receipt['selected_crossings']} stored crossings: {output}; receipt: {receipt_path}")


if __name__ == "__main__":
    main()
