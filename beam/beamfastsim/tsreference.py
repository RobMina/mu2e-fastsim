"""Opt-in extraction of retained ReadVirtualDetector ntpart rows, not an inclusive TS sample.

The upstream production filter preserves downstream survivors and ancestors. These rows
cannot establish transmission, efficiency, generated exposure or POT-normalised rates.
VD state is the first STORED hit, not necessarily the earliest or forward crossing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PDGS = (13, -13, 211, -211)
IDENTITY = ("run", "subrun", "evt", "trk")
INTEGER = IDENTITY + ("parent_id", "parent_pdg", "pdg")
BORN = ("x", "y", "z", "px", "py", "pz", "time")
END = ("xstop", "ystop", "zstop", "pxstop", "pystop", "pzstop", "tstop")
OPTIONAL = ("code", "codestop", "p", "pstop")
VD = ("isvd", "xvd", "yvd", "zvd", "pxvd", "pyvd", "pzvd", "tvd")
VD_STATE = ("x", "y", "z", "px", "py", "pz", "time")


def _integers(values, name, n=None):
    a = np.asarray(values)
    if a.ndim != 1 or (n is not None and len(a) != n) or a.dtype.kind not in "iuf":
        raise ValueError(f"{name}: expected a one-dimensional numeric integer column")
    if a.dtype.kind == "f":
        if not np.all(np.isfinite(a)) or not np.all(a == np.floor(a)):
            raise ValueError(f"{name}: nonfinite or nonintegral identity/code")
        limit = 2 ** (np.finfo(a.dtype).nmant + 1)
        if np.any(np.abs(a) >= limit):
            raise ValueError(f"{name}: floating-point integer precision is unsafe at or beyond {limit}")
        if np.any(a < -(2 ** 63)) or np.any(a >= 2 ** 63):
            raise ValueError(f"{name}: outside int64 range")
    elif a.dtype.kind == "u" and np.any(a > np.iinfo(np.int64).max):
        raise ValueError(f"{name}: outside int64 range")
    return a.astype(np.int64)


def _state(values, name, n):
    a = np.asarray(values)
    if a.shape != (n,) or a.dtype.kind not in "iuf" or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: expected a finite numeric scalar per particle")
    return a.astype(np.float64)


def _matrix(values, name, n, requested_width):
    a = np.asarray(values)
    if n == 0 and a.shape == (0,) and a.dtype.kind == "O":
        return np.empty((0, requested_width), dtype=np.float64)
    if a.dtype.kind == "O":
        try:
            a = np.stack(a)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name}: malformed or ragged VD dimensions") from exc
    if a.ndim != 2 or a.shape[0] != n or a.dtype.kind not in "biuf":
        raise ValueError(f"{name}: expected rectangular numeric VD dimensions")
    return a


def _source(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def _tree(root, name):
    if ";" in name:
        raise ValueError("tree must not specify a cycle; the highest cycle is mandatory")
    directory, _, leaf = name.rpartition("/")
    try:
        parent = root[directory] if directory else root
        cycles = sorted(int(k.rsplit(";", 1)[1]) for k in parent.keys(cycle=True, recursive=False)
                        if k.rsplit(";", 1)[0] == leaf)
    except KeyError as exc:
        raise ValueError(f"Missing tree {name}") from exc
    if not cycles:
        raise ValueError(f"Missing tree {name}")
    key = f"{name};{cycles[-1]}"
    return root[key], key, {str(c): int(root[f"{name};{c}"].num_entries) for c in cycles}


def read(paths, tree="mutree/ntpart", vds=tuple(range(1, 9)), pdgs=PDGS):
    """Return (row-aligned NumPy columns, JSON-safe receipt); validate whole files before selection.

    Identity is (ifile, run, subrun, evt, trk); duplicate keys within a file fail,
    even for unselected species. Parent IDs are preserved, not resolved or renumbered.
    Start/end state and ancestry are required; code, codestop, p and pstop are optional
    with explicit has_* masks. VD columns use vdN_hit and vdN_stored_first_{x,...,time}.
    No-hit state is NaN. Neither selection nor VD presence drops no-hit particles.
    """
    import uproot

    paths = list(paths)
    if not paths:
        raise ValueError("At least one ROOT input is required")
    vds = _integers(vds, "vds")
    pdgs = _integers(pdgs, "pdgs")
    if not len(vds) or len(np.unique(vds)) != len(vds) or not np.all(np.isin(vds, range(1, 9))):
        raise ValueError("vds must be distinct IDs in [1, 8]")
    if not len(pdgs) or len(np.unique(pdgs)) != len(pdgs) or not np.all(np.isin(pdgs, PDGS)):
        raise ValueError("pdgs must be a nonempty distinct subset of 13, -13, 211, -211")
    vd_names = [f"vd{v}_hit" for v in vds]
    vd_names += [f"vd{v}_stored_first_{k}" for v in vds for k in VD_STATE]
    names = INTEGER + BORN + END + OPTIONAL + tuple(f"has_{k}" for k in OPTIONAL)
    cols = {k: [] for k in names + ("ifile", "source_row", "q") + tuple(vd_names)}
    files = []
    required = INTEGER + BORN + END + VD
    for ifile, path in enumerate(paths):
        with uproot.open(path) as root:
            source_tree, key, cycles = _tree(root, tree)
            branches = set(source_tree.keys())
            missing = set(required) - branches
            if missing:
                raise ValueError(f"{path}: missing required branches {sorted(missing)}")
            present = [k for k in OPTIONAL if k in branches]
            arrays = source_tree.arrays(list(required) + present, library="np", how=dict)
            n = int(source_tree.num_entries)
            data = {k: _integers(arrays[k], k, n) for k in INTEGER}
            ids = np.column_stack([data[k] for k in IDENTITY])
            if len(np.unique(ids, axis=0)) != n:
                raise ValueError(f"{path}: duplicate particle identity (run, subrun, evt, trk)")
            data.update({k: _state(arrays[k], k, n) for k in BORN + END})
            for k in OPTIONAL:
                if k in present:
                    data[k] = (_integers(arrays[k], k, n) if k in ("code", "codestop")
                               else _state(arrays[k], k, n))
                else:
                    data[k] = np.zeros(n, dtype=np.int64) if k in ("code", "codestop") else np.full(n, np.nan)
                data[f"has_{k}"] = np.full(n, k in present, dtype=bool)
            vd = {k: _matrix(arrays[k], k, n, int(vds.max())) for k in VD}
            shape = vd["isvd"].shape
            if any(a.shape != shape for a in vd.values()) or shape[1] < int(vds.max()):
                raise ValueError(f"{path}: inconsistent or insufficient VD dimensions for requested IDs")
            status = vd["isvd"][:, vds - 1]
            if not np.all(np.isin(status, [0, 1])):
                raise ValueError(f"{path}: isvd must contain finite binary hit status")
            hit = status.astype(bool)
            for j, v in enumerate(vds):
                data[f"vd{v}_hit"] = hit[:, j]
                for k, branch in zip(VD_STATE, VD[1:]):
                    state = vd[branch][:, v - 1].astype(np.float64)
                    if not np.all(np.isfinite(state[hit[:, j]])):
                        raise ValueError(f"{path}: nonfinite hit state in {branch} at VD{v}")
                    data[f"vd{v}_stored_first_{k}"] = np.where(hit[:, j], state, np.nan)
            selected = np.isin(data["pdg"], pdgs)
            count = int(selected.sum())
            data["ifile"] = np.full(n, ifile, dtype=np.int64)
            data["source_row"] = np.arange(n, dtype=np.int64)
            data["q"] = np.where(np.isin(data["pdg"], [13, -211]), -1, 1).astype(np.int8)
            for k in cols:
                cols[k].append(data[k][selected])
            files.append(dict(_source(path), ifile=ifile, selected_key=key,
                              cycle=int(key.rsplit(";", 1)[1]), cycles=cycles,
                              rows=n, selected_rows=count, optional_branches=present))
    receipt = dict(schema="ts-retained-reference-v1", survivor_conditioned=True,
                   normalisation_unverified=True, files=files, tree=tree,
                   pdgs=pdgs.tolist(), vds=vds.tolist(), rows=sum(f["rows"] for f in files),
                   selected_rows=sum(f["selected_rows"] for f in files),
                   identity=["ifile", *IDENTITY], parent_scope="Same file/run/subrun/evt; ancestors may be absent",
                   state_columns=dict(born=list(BORN), end=list(END)),
                   optional_missing="has_* is false; integer placeholder 0, floating placeholder NaN",
                   vd_state="First stored hit only; chronology and forward direction are not certified",
                   scope="Retained particle rows only; not inclusive, not generated counts, not efficiency or POT rates")
    output = {k: np.concatenate(v) for k, v in cols.items()}
    receipt["retained_species_counts"] = {str(pdg): int(np.count_nonzero(output["pdg"] == pdg))
                                          for pdg in pdgs}
    receipt["stored_vd_hit_counts"] = {str(v): {str(pdg): int(np.count_nonzero(
        output[f"vd{v}_hit"] & (output["pdg"] == pdg))) for pdg in pdgs} for v in vds}
    receipt["count_warning"] = "Retained particle/hit counts only; these are not efficiencies or POT yields"
    return output, receipt


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.tsreference", description=__doc__)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--tree", default="mutree/ntpart")
    ap.add_argument("--vd", type=int, action="append", help="VD ID 1..8, repeatable; default all eight")
    ap.add_argument("--pdg", type=int, action="append", help="PDG 13, -13, 211 or -211, repeatable")
    ap.add_argument("--out", required=True, help="NPZ path; receipt is written alongside as .json")
    args = ap.parse_args(argv)
    output = Path(args.out)
    if output.suffix.lower() != ".npz":
        ap.error("--out must end in .npz")
    receipt_path = output.with_suffix(".json")
    inputs = {Path(p).resolve() for p in args.roots}
    if output.resolve() in inputs or receipt_path.resolve() in inputs:
        ap.error("outputs must not overwrite ROOT inputs")
    try:
        columns, receipt = read(args.roots, tree=args.tree,
                                vds=args.vd if args.vd is not None else tuple(range(1, 9)),
                                pdgs=args.pdg if args.pdg is not None else PDGS)
    except ValueError as exc:
        ap.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)
    np.savez_compressed(output, metadata_json=np.array(metadata), **columns)
    receipt["output"] = _source(output)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"Saved {receipt['selected_rows']} retained rows: {output}; receipt: {receipt_path}")


if __name__ == "__main__":
    main()
