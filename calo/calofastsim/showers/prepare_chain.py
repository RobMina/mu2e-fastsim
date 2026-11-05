"""Prepare compact same-primary inputs for the CE-to-calorimeter measurement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from .compare import response_inputs
from .export import EVENT, join, keys


def prepare(tables, response, sim_crystal, backend, *, start, end, buffer):
    """Keep every generated event and stock response, including side/photon-only arrivals."""
    with uproot.open(tables) as source:
        events = source["dump/events"].arrays(list(EVENT) + ["g4_status"], library="np")
        generated = source["dump/generated"].arrays(list(EVENT) + ["pdg", "time"] +
                    ["position_" + a for a in "xyz"] + ["momentum_p" + a for a in "xyz"], library="np")
        showers = source["dump/showers"].arrays(list(EVENT) + ["crystal", "energy_g4", "energy_birks", "time", "position_z"], library="np")
    ids = keys(events, EVENT)
    if len(np.unique(ids)) != len(ids) or np.any(events["g4_status"] != 0):
        raise ValueError("expected unique, completed source events")
    if len(generated["event"]) != len(ids) or np.any(generated["pdg"] != 11):
        raise ValueError("expected one generated electron per event")
    order = join(keys(generated, EVENT), ids)
    p = np.column_stack([generated["momentum_p" + a][order] for a in "xyz"])
    magnitude = np.linalg.norm(p, axis=1)
    if np.any(magnitude <= 0):
        raise ValueError("generated electron has zero momentum")
    conditions, observed_lru = response_inputs(response)
    order_c = join(keys(conditions, EVENT), ids)
    conditions = {k: v[order_c] for k, v in conditions.items()}
    calibration = np.loadtxt(sim_crystal)
    calibration = calibration[np.argsort(calibration[:, 0])]
    if not np.array_equal(calibration[:, 0], np.arange(len(calibration))):
        raise ValueError("Sim_crystal IDs must be contiguous")
    lru = calibration[:, 1].astype(np.float32)
    present = np.flatnonzero(np.isfinite(observed_lru))
    if np.any(present >= len(lru)) or not np.array_equal(lru[present], observed_lru[present].astype(np.float32)):
        raise ValueError("saved LRU differs from the supplied complete crystal conditions")
    row = join(ids, keys(showers, EVENT))
    n = len(ids)
    crystal = showers["crystal"]
    energy = backend.corrected_energy(showers["energy_g4"], showers["energy_birks"], showers["position_z"],
                                       conditions["crystal_length"][row], lru[crystal])
    timing = {name: conditions[field][row] for name, field in (("pbtime", "pbtime"), ("mbtime", "mbtime"),
              ("event_length", "event_length"), ("time_from_protons_to_dr_marker", "marker"))}
    _, accepted = backend.fold_step_time(showers["time"], onspill=conditions["spill"][row] == 1,
                                         digitization_start=start, digitization_end=end, digitization_buffer=buffer, **timing)
    arrays = dict(x0=np.column_stack([generated["position_" + a][order] for a in "xyz"]),
                  u0=p / magnitude[:, None], p0=magnitude, t0=generated["time"][order],
                  stock_visible=np.bincount(row, weights=showers["energy_birks"], minlength=n),
                  stock_response=np.bincount(row, weights=np.where(accepted, energy, 0.), minlength=n), lru=lru)
    arrays.update({"condition_" + k: v for k, v in conditions.items()})
    return arrays


def stock_clusters(path, event_ids):
    """Leading production cluster for every generated event, aligned by full event identity."""
    import awkward as ak

    with uproot.open(path) as source:
        tree = source["prd/ntuple"]
        if tree.classname == "ROOT::RNTuple":
            data = tree.arrays(filter_name=["evtinfo", "evtinfo.*", "caloclusters", "caloclusters.*"])
        else:
            data = tree.arrays(["evtinfo", "caloclusters"])
    evt = data["evtinfo"]
    fields = {k: ak.to_numpy(evt[k]) for k in EVENT}
    rows = join(event_ids, keys(fields, EVENT))
    if len(rows) != len(event_ids) or len(np.unique(rows)) != len(rows):
        raise ValueError("stock ntuple does not contain every input event once")
    cl = data["caloclusters"]
    energy = cl["caloclusters.energyDep_"]
    best = ak.argmax(energy, axis=1, keepdims=True)
    result = {}
    mapping = dict(energy="energyDep_", time="time_", x="cog_.fCoordinates.fX", y="cog_.fCoordinates.fY", z="cog_.fCoordinates.fZ")
    for name, field in mapping.items():
        fill = 0. if name == "energy" else np.nan
        values = ak.to_numpy(ak.fill_none(ak.firsts(cl["caloclusters." + field][best]), fill))
        result[name] = np.full(len(event_ids), fill)
        result[name][rows] = values
    return result


def main():
    import importlib.util

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--response-source", required=True, type=Path)
    parser.add_argument("--sim-crystal", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--stock-ntuple", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--digitization-start", required=True, type=float)
    parser.add_argument("--digitization-end", required=True, type=float)
    parser.add_argument("--digitization-buffer", required=True, type=float)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("saved_response", args.response_source)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    arrays = prepare(args.tables, args.response, args.sim_crystal, backend,
                     start=args.digitization_start, end=args.digitization_end, buffer=args.digitization_buffer)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stock_ntuple:
        ids = keys({k: arrays["condition_" + k] for k in EVENT}, EVENT)
        arrays.update({"stock_cluster_" + k: v for k, v in stock_clusters(args.stock_ntuple, ids).items()})
    np.savez_compressed(args.out / "chain_inputs.npz", **arrays)
    geometry = args.geometry.read_text()
    geometry = "\n".join(line.removeprefix("Geom: ") for line in geometry.splitlines()) + "\n"
    (args.out / "geometry.txt").write_text(geometry)
    info = {k: str(v) for k, v in vars(args).items()}
    info.update(events=len(arrays["p0"]), stock_positive=int(np.sum(arrays["stock_visible"] > 0)),
                stock_visible_above10=int(np.sum(arrays["stock_visible"] > 10)),
                stock_response_above10=int(np.sum(arrays["stock_response"] > 10)))
    (args.out / "inputs.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
