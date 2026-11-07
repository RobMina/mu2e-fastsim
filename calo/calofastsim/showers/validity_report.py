"""Saved-only development diagnostics and CPU/GPU profiles; never sample or reconstruct."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compare import paired_hist, paired_mean, paired_ratio, plot
from .library import load


ARMS = ("stock", "baseline", "crystal", "stock2")


def observables(cluster, entry, folded_entry_time, origin):
    """Incident-relative diagnostics in the common disk transverse frame, not track resolution."""
    xy = entry["position"][:, :2] - origin
    radius = np.linalg.norm(xy, axis=1)
    radial = np.divide(xy, radius[:, None], out=np.tile([1., 0.], (len(xy), 1)), where=radius[:, None] > 0)
    delta = np.column_stack([cluster["x"], cluster["y"]]) - xy
    momentum = np.sqrt(entry["energy"] ** 2 - .51099895 ** 2)
    return dict(energy=cluster["energy"], radial=np.sum(delta * radial, axis=1),
                tangent=delta[:, 1] * radial[:, 0] - delta[:, 0] * radial[:, 1],
                time=cluster["time"] - folded_entry_time, truth_ep=cluster["energy"] / momentum)


def compare_observables(fast, stock, rows):
    """Marginals use the same incident queries; shape errors are paired source-block diagnostics."""
    group = rows // 100
    selected = (fast["energy"] > 10) & (stock["energy"] > 10)
    definitions = (("energy", "Leading-cluster energy [MeV]", np.arange(0., 111., 5.), np.ones(len(rows), bool)),
                   ("radial", "Centroid minus entry: radial [mm]", np.arange(-1400., 1401., 40.), selected),
                   ("tangent", "Centroid minus entry: tangential [mm]", np.arange(-1400., 1401., 40.), selected),
                   ("time", "Cluster time minus folded entry [ns]", np.arange(-20., 21., 1.), selected),
                   ("truth_ep", "E/p at entry (truth diagnostic)", np.linspace(0., 1.2, 25), np.ones(len(rows), bool)))
    shapes = {}
    for name, label, edges, mask in definitions:
        f, s, g = fast[name][mask], stock[name][mask], group[mask]
        shapes[label] = paired_hist(f, s, g, edges)
        shapes[label]["outside_range"] = dict(fast=int(((f < edges[0]) | (f > edges[-1])).sum()),
                                               stock=int(((s < edges[0]) | (s > edges[-1])).sum()))
        shapes[label]["paired_mean"] = paired_mean(f, s, g)
    return shapes


def diagnostics(library, saved, conditions):
    rows, source_rows = saved["query_event_rows"], saved["event_row"]
    if not np.array_equal(source_rows, np.arange(1000)) or not np.array_equal(rows, library["entry_event_row"]):
        raise ValueError("saved clusters and development-library query identities must match")
    origin = np.column_stack([library["geometry_disk_origin_" + a] for a in "xy"])
    if not np.allclose(origin, origin[:1]):
        raise ValueError("saved leading centroids need a common disk transverse origin")
    axes = np.stack([np.column_stack([library["geometry_disk_axis_" + a + "_" + b] for b in "xyz"]) for a in "xyz"], axis=2)
    if not np.allclose(axes, np.eye(3)):
        raise ValueError("saved leading centroids require aligned disk axes")
    entry = {k: library["entry_" + k] for k in ("position", "direction", "energy", "time", "radius", "disk", "cosine")}
    from calofastsim.response import fold_step_time
    from calofastsim.readout import ReadoutConfig
    cfg = ReadoutConfig()
    folded, _ = fold_step_time(entry["time"], pbtime=conditions["condition_pbtime"][rows],
        mbtime=conditions["condition_mbtime"][rows], event_length=conditions["condition_event_length"][rows],
        onspill=conditions["condition_spill"][rows] == 1, time_from_protons_to_dr_marker=conditions["condition_marker"][rows],
        digitization_start=cfg.digitization_start, digitization_end=cfg.digitization_end, digitization_buffer=cfg.digitization_buffer)
    names = ("stock1",) + ARMS
    clusters = {name: {k: saved[name + "_" + k][rows] for k in ("energy", "time", "x", "y", "z")} for name in names}
    obs = {name: observables(value, entry, folded, origin[0]) for name, value in clusters.items()}
    common = np.logical_and.reduce([clusters[name]["energy"] > 10 for name in names])
    result = {}
    for name in ARMS:
        fast, stock = clusters[name], clusters["stock1"]
        distance = np.hypot(fast["x"][common] - stock["x"][common], fast["y"][common] - stock["y"][common])
        masks = {"all": np.ones(len(rows), bool)}
        masks.update({f"disk{disk}": entry["disk"] == disk for disk in np.unique(entry["disk"])})
        masks.update({f"radius_{lo}_{hi}": (entry["radius"] >= lo) & (entry["radius"] < hi)
                      for lo, hi in ((0, 450), (450, 550), (550, 1000))})
        for feature, edges in (("energy", (0, 50, 80, 95, 110)), ("cosine", (0, .4, .7, 1.000001))):
            masks.update({f"{feature}_{lo}_{hi}": (entry[feature] >= lo) & (entry[feature] < hi)
                          for lo, hi in zip(edges[:-1], edges[1:])})
        strata = {}
        for key, mask in masks.items():
            strata[key] = dict(entries=int(mask.sum()),
                               acceptance=paired_ratio(fast["energy"][mask] > 10, stock["energy"][mask] > 10, rows[mask] // 100),
                               energy=paired_mean(fast["energy"][mask], stock["energy"][mask], rows[mask] // 100))
        result[name] = dict(shapes=compare_observables(obs[name], obs["stock1"], rows), strata=strata,
                            common_selected=int(common.sum()),
                            common_distance_mm=paired_mean(distance, np.zeros(len(distance)), rows[common] // 100))
    return dict(entries=len(rows), arms=result, caveat="Development conditional front entries, not independent confirmation. "
                "Incident p and positions are simulation diagnostics, not reconstructed E/p or track resolution. "
                "Time residual includes independent reconstructed PBT smearing. Library/cross-fold uncertainty excluded.")


def replay_comparison(cpu, gpu):
    reports = {device: json.loads((path / "numbers.json").read_text()) for device, path in (("CPU", cpu), ("MIG", gpu))}
    result = {}
    with np.load(cpu / "clusters.npz") as a, np.load(gpu / "clusters.npz") as b:
        if not np.array_equal(a["event_row"], b["event_row"]) or not np.array_equal(a["query_event_rows"], b["query_event_rows"]):
            raise ValueError("CPU/GPU replay event identities differ")
        for name in ("stock", "baseline", "crystal"):
            checks = {key: bool(np.array_equal(a[name + "_" + key], b[name + "_" + key], equal_nan=True))
                      for key in ("energy", "time", "x", "y", "z")}
            differences = {}
            for key in checks:
                x, y = a[name + "_" + key], b[name + "_" + key]
                both = np.isfinite(x) & np.isfinite(y)
                delta = np.abs(x[both] - y[both])
                differences[key] = dict(nonfinite_mask_equal=bool(np.array_equal(np.isfinite(x), np.isfinite(y))),
                                        changed_finite_values=int(np.count_nonzero(delta)), max_abs=float(delta.max(initial=0)))
            result[name] = dict(exact=checks, differences=differences,
                                selection_equal=bool(np.array_equal(a[name + "_energy"] > 10, b[name + "_energy"] > 10)),
                                warm_seconds={device: report["arms"][name]["timing"]["warm_reco_seconds"]
                                              for device, report in reports.items()})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "inputs", "cpu", "gpu", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    library, _ = load(args.library)
    with np.load(args.gpu / "clusters.npz") as saved, np.load(args.inputs) as conditions:
        result = diagnostics(library, saved, conditions)
    result["backend_replay"] = replay_comparison(args.cpu, args.gpu)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "numbers.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name in ("stock", "baseline", "crystal"):
        plot(result["arms"][name]["shapes"], args.out / (name + ".png"),
             name + ": development-only direct-front queries; paired diagnostics", labels=("Stock reconstruction", name))
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
