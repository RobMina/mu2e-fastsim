"""Saved-development loss accounting; no reconstruction, tuning or corrected model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

from .compare import paired_hist, paired_mean, paired_ratio
from .crystal_placement import place_crystals
from .export import EVENT, columns, keys, read
from .library import load, sample
from .production_compare import _json, check_geometry, check_independence
from .response import cluster_sum, place

ARMS = ("geometric", "crystal")
LOSS = ("raw_inner", "raw_outer", "raw_axial", "interior_unmapped",
        "center_or_proposed_envelope", "center_too_far", "containment", "postmap")


def exact(label, a, b):
    """Reject stale saved identities, entries or draw arrays, not just mismatched counts."""
    if not np.array_equal(a, b):
        raise ValueError("saved comparison mismatch: " + label)


def reproduce(tables, library, metadata, saved, seed):
    """Replay only the frozen draw and placements against the complete saved row universe."""
    entries, original, geometry, census, _ = read(tables, input_mode="primary")
    with uproot.open(tables) as source:
        full_ids = keys(columns(source, "events", EVENT), EVENT)
        for prefix, table in (("crystal", "crystals"), ("disk", "disks"), ("vd", "virtualdetectors")):
            extra = [k.removeprefix("geometry_" + prefix + "_") for k in library
                     if k.startswith("geometry_" + prefix + "_") and k.removeprefix("geometry_") not in geometry]
            geometry.update({prefix + "_" + k: v for k, v in columns(source, table, extra).items()} if extra else {})
    exact("full_ids", full_ids, saved["full_ids"])
    exact("event_row", np.arange(len(full_ids)), saved["event_row"])
    exact("query_event_rows", entries["event_row"], saved["query_event_rows"])
    for name, value in entries.items():
        exact("entry_" + name, value, saved["entry_" + name])
    for name, value in geometry.items():
        exact("geometry_" + name, value, saved["geometry_" + name])
    check_geometry(library, geometry)
    independence = check_independence(tables, library, metadata, full_ids)
    for arm in ("stockref", "control", *ARMS):
        energy = saved[arm + "_energy"]
        if energy.shape != (len(full_ids),) or not np.all(np.isfinite(energy)) or np.any(energy < 0):
            raise ValueError("invalid saved leading energy: " + arm)
    original_placed = place(library, original)
    if not np.all(original_placed["mapped"]) or not np.array_equal(original_placed["crystal"], original["crystal"]):
        raise ValueError("original CSS does not round-trip to its saved crystals")
    draw = sample(library, entries, np.random.default_rng(seed), orientation="disk")
    for name in ("donor", "supported", "fallback", "offsets"):
        exact("sampling_" + name, draw[name], saved["sampling_" + name])
    placed = dict(geometric=place(library, draw), crystal=place_crystals(library, entries, draw))
    return entries, original, draw, placed, dict(census=census, independence=independence)


def loss_categories(library, draw, placed, arm):
    """Exclusive accounting labels: raw radial, raw axial, then placement guard priority."""
    reason = np.full(len(draw["entry"]), -1, dtype=np.int8)
    for disk in np.unique(draw["disk"]):
        take = np.flatnonzero(draw["disk"] == disk)
        ds = np.flatnonzero(library["geometry_disk_disk"] == disk)
        pool = library["geometry_crystal_disk"] == disk
        if len(ds) != 1 or not np.any(pool):
            raise ValueError("loss accounting requires one geometry row per disk")
        ds = ds[0]
        origin = np.array([library["geometry_disk_origin_" + a][ds] for a in "xyz"])
        rotation = np.column_stack([np.array([library["geometry_disk_axis_" + a + "_" + b][ds]
                                              for b in "xyz"]) for a in "xyz"])
        local = (draw["position"][take] - origin) @ rotation
        radius = np.linalg.norm(local[:, :2], axis=1)
        center = np.column_stack([library["geometry_crystal_position_" + a][pool] for a in "xyz"])
        front = ((center - origin) @ rotation)[:, 2]
        inner = radius < library["geometry_disk_size_x"][ds]
        outer = radius > library["geometry_disk_size_y"][ds]
        axial = ((local[:, 2] < front.min() - 1e-6) |
                 (local[:, 2] > np.max(front + library["geometry_crystal_size_z"][pool]) + 1e-6))
        interior = np.full(len(take), 3, dtype=np.int8)
        if arm == "crystal":
            guard = placed["crystal_loss_reason"][take]
            if not np.all(np.isin(guard, (0, 1, 2, 3))):
                raise ValueError("unknown crystal loss reason")
            interior = np.select([guard == 1, guard == 2, guard == 3], [4, 5, 6], default=7)
        reason[take] = np.where(inner, 0, np.where(outer, 1, np.where(axial, 2, interior)))
    reason[placed["mapped"]] = -1
    if np.any((reason < 0) != placed["mapped"]):
        raise ValueError("unclassified placement loss")
    return reason


def candidate_box_distance(library, draw):
    """Distance to the four nearest-center crystal boxes used by geometric placement."""
    from scipy.spatial import cKDTree

    distance = np.full(len(draw["entry"]), np.nan)
    centers = np.column_stack([library["geometry_crystal_position_" + a] for a in "xyz"])
    sizes = np.column_stack([library["geometry_crystal_size_" + a] for a in "xyz"])
    for disk in np.unique(draw["disk"]):
        take = np.flatnonzero(draw["disk"] == disk)
        pool = np.flatnonzero(library["geometry_crystal_disk"] == disk)
        ds = np.flatnonzero(library["geometry_disk_disk"] == disk)[0]
        rotation = np.column_stack([np.array([library["geometry_disk_axis_" + a + "_" + b][ds]
                                              for b in "xyz"]) for a in "xyz"])
        xyz, center = draw["position"][take] @ rotation, centers[pool] @ rotation
        _, near = cKDTree(center[:, :2]).query(xyz[:, :2], k=[1, 2, 3, 4])
        valid = near < len(pool)
        near = np.minimum(near, len(pool) - 1)
        delta = xyz[:, None, :] - center[near]
        outside = np.maximum(np.abs(delta[:, :, :2]) - sizes[pool[near], :2] / 2, 0)
        depth = np.maximum(np.maximum(-delta[:, :, 2], delta[:, :, 2] - sizes[pool[near], 2]), 0)
        d = np.sqrt(np.sum(outside ** 2, axis=2) + depth ** 2)
        distance[take] = np.min(np.where(valid, d, np.inf), axis=1)
    return distance


def energy_sum(steps, n, mask=None):
    """Whole-event CSS energy, including both disks and zero-energy query rows."""
    weights = steps["energy_birks"] if mask is None else np.where(mask, steps["energy_birks"], 0.)
    return np.bincount(steps["entry"], weights=weights, minlength=n)


def donor_features(library, entries, draw):
    """Incident-state mismatch diagnostics, never response-model conditioning."""
    n = len(entries["energy"])
    donor, supported = draw["donor"], draw["supported"]
    result = {name: np.full(n, np.nan) for name in ("radius_delta", "energy_delta", "cosine_delta", "angle_deg")}
    d = donor[supported]
    for name in ("radius", "energy", "cosine"):
        result[name + "_delta"][supported] = library["entry_" + name][d] - entries[name][supported]
    ds = np.searchsorted(library["geometry_disk_disk"], entries["disk"][supported])
    origin = np.column_stack([library["geometry_disk_origin_" + a] for a in "xy"])[ds]
    target_xy, source_xy = entries["position"][supported, :2] - origin, library["entry_position"][d, :2] - origin
    angle = np.arctan2(target_xy[:, 1], target_xy[:, 0]) - np.arctan2(source_xy[:, 1], source_xy[:, 0])
    u = library["entry_direction"][d].copy()
    x, y = u[:, 0].copy(), u[:, 1].copy()
    u[:, 0], u[:, 1] = np.cos(angle) * x - np.sin(angle) * y, np.sin(angle) * x + np.cos(angle) * y
    dot = np.sum(u * entries["direction"][supported], axis=1)
    result["angle_deg"][supported] = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))
    return result


def strata(entries):
    radius = entries["radius"]
    masks = {"all": np.ones(len(radius), bool)}
    masks.update({f"radius_{lo}_{hi}": (radius >= lo) & (radius < hi)
                  for lo, hi in ((0, 450), (450, 550), (550, 1000), (1000, np.inf))})
    masks.update({f"disk{disk}": entries["disk"] == disk for disk in (0, 1)})
    return masks


def summarize(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return dict(count=len(finite), mean=float(finite.mean()) if len(finite) else None,
                quantiles=dict(zip(("p10", "p50", "p90"), np.quantile(finite, [.1, .5, .9]).tolist())) if len(finite) else None)


def selection_accounting(original, donor, mapped, lost, stock, fast, control, supported=None):
    """CSS gates are not reconstructed selection; saved miss partitions are associations only."""
    source_gate, donor_gate, mapped_gate = original > 10, donor > 10, mapped > 10
    misses, gains = (stock > 10) & (fast <= 10), (stock <= 10) & (fast > 10)
    control_miss = (control > 10) & (fast <= 10)
    supported = np.ones(len(donor), bool) if supported is None else np.asarray(supported, bool)
    categories = dict(no_donor_energy=supported & (donor == 0), unsupported=~supported,
                      positive_donor_no_loss=supported & (donor > 0) & (lost <= 1e-12),
                      positive_donor_with_loss=supported & (donor > 0) & (lost > 1e-12))
    return dict(css_gate=dict(original=int(source_gate.sum()), donor=int(donor_gate.sum()), mapped=int(mapped_gate.sum()),
            donor_to_mapped_down=int((donor_gate & ~mapped_gate).sum()),
            source_to_donor_down=int((source_gate & ~donor_gate).sum()),
            source_to_donor_up=int((~source_gate & donor_gate).sum())),
        reco=dict(stock_selected=int((stock > 10).sum()), fast_selected=int((fast > 10).sum()),
            misses=int(misses.sum()), gains=int(gains.sum()), net_deficit=int(misses.sum() - gains.sum()),
            miss_partition={k: int((misses & m).sum()) for k, m in categories.items()},
            control_misses=int(control_miss.sum()),
            control_miss_partition={k: int((control_miss & m).sum()) for k, m in categories.items()},
            misses_with_css_placement_gate_down=int((misses & donor_gate & ~mapped_gate).sum())))


def centroid_coordinates(steps, entries, origin, accepted=None):
    """Both-disk Birks centroid relative to the query, not a reconstructed leading cluster."""
    cluster = cluster_sum(steps, len(entries["energy"]), accepted=accepted)
    xy = entries["position"][:, :2] - origin
    radius = np.linalg.norm(xy, axis=1)
    radial = np.divide(xy, radius[:, None], out=np.tile([1., 0.], (len(xy), 1)), where=radius[:, None] > 0)
    delta = cluster["position"][:, :2] - entries["position"][:, :2]
    return dict(energy=cluster["energy"], radial=np.sum(delta * radial, axis=1),
                tangent=delta[:, 1] * radial[:, 0] - delta[:, 0] * radial[:, 1])


def diagnose(library, entries, original, draw, placed, saved):
    n, rows = len(entries["energy"]), entries["event_row"]
    source, donor = energy_sum(original, n), energy_sum(draw, n)
    features = donor_features(library, entries, draw)
    arrays = dict(event_row=rows, original=source, donor=donor, **features)
    result = dict(strata={}, shapes={}, closure={}, associations={}, geometry={})
    result["geometry"] = {"disk_" + k: library["geometry_disk_" + k].tolist() for k in ("disk", "size_x", "size_y")}
    result["geometry"]["loss_category_precedence"] = "raw inner, raw outer, raw axial, then crystal guard; mapped always excluded"
    association_masks = {"radius_delta_lt_minus25": features["radius_delta"] < -25,
        "radius_delta_within25": np.abs(features["radius_delta"]) <= 25,
        "radius_delta_gt25": features["radius_delta"] > 25,
        "angle_lt10": features["angle_deg"] < 10,
        "angle_10_30": (features["angle_deg"] >= 10) & (features["angle_deg"] < 30),
        "angle_ge30": features["angle_deg"] >= 30}
    masks = strata(entries)
    box_distance = candidate_box_distance(library, draw)
    if not np.all(np.isfinite(box_distance)) or np.any(box_distance < 0):
        raise ValueError("invalid candidate box distance")
    for arm, steps in placed.items():
        mapped, lost = energy_sum(steps, n, steps["mapped"]), energy_sum(steps, n, ~steps["mapped"])
        if not np.allclose(mapped + lost, donor, rtol=1e-12, atol=1e-10):
            raise ValueError("placement violates per-query energy balance")
        arrays.update({arm + "_mapped": mapped, arm + "_lost": lost})
        reasons = loss_categories(library, draw, steps, arm)
        losses = {k: energy_sum(steps, n, reasons == i) for i, k in enumerate(LOSS)}
        if not np.allclose(sum(losses.values()), lost, rtol=1e-12, atol=1e-10):
            raise ValueError("loss categories do not partition unmapped energy")
        arrays.update({arm + "_loss_" + k: v for k, v in losses.items()})
        result["closure"][arm] = dict(max_abs_energy_balance_mev=float(np.max(np.abs(mapped + lost - donor), initial=0)),
            max_abs_loss_partition_mev=float(np.max(np.abs(sum(losses.values()) - lost), initial=0)))
        for stratum in ("all", "radius_0_450", "radius_450_550", "radius_550_1000"):
            associations = result["associations"].setdefault(stratum, {}).setdefault(arm, {})
            for label, feature_mask in association_masks.items():
                m = masks[stratum] & feature_mask
                group = rows[m] // 100
                associations[label] = dict(entries=int(m.sum()), fallback=int((m & draw["fallback"]).sum()),
                    empty=int((m & draw["supported"] & (np.diff(draw["offsets"]) == 0)).sum()),
                    donor_minus_original=paired_mean(donor[m], source[m], group),
                    lost=paired_mean(lost[m], np.zeros(int(m.sum())), group),
                    reco_ratio=paired_ratio(saved[arm + "_energy"][rows[m]] > 10,
                                            saved["stockref_energy"][rows[m]] > 10, group))
        for name, mask in masks.items():
            group = rows[mask] // 100
            zero = np.zeros(int(mask.sum()))
            stats = result["strata"].setdefault(name, dict(entries=int(mask.sum()), arms={}))
            stats["donor_features"] = {k: summarize(v[mask]) for k, v in features.items()}
            stats["sampling"] = dict(unsupported=int((mask & ~draw["supported"]).sum()),
                fallback=int((mask & draw["fallback"]).sum()),
                empty=int((mask & draw["supported"] & (np.diff(draw["offsets"]) == 0)).sum()))
            stats["arms"][arm] = dict(
                donor_minus_original=paired_mean(donor[mask], source[mask], group),
                placement_loss=paired_mean(lost[mask], zero, group),
                mapped_minus_original=paired_mean(mapped[mask], source[mask], group),
                lost_fraction_of_donor=paired_ratio(lost[mask], donor[mask], group),
                losses={k: paired_mean(v[mask], zero, group) for k, v in losses.items()},
                selection=selection_accounting(source[mask], donor[mask], mapped[mask], lost[mask],
                    saved["stockref_energy"][rows[mask]], saved[arm + "_energy"][rows[mask]],
                    saved["control_energy"][rows[mask]], draw["supported"][mask]),
                saved_reco_delta=paired_mean(saved[arm + "_energy"][rows[mask]], saved["stockref_energy"][rows[mask]], group))
            stats["arms"][arm]["lost_by_candidate_box_distance"] = {}
            for lo, hi in ((0., 1.), (1., 10.), (10., np.inf)):
                loss = energy_sum(steps, n, (~steps["mapped"]) & (box_distance >= lo) & (box_distance < hi))
                stats["arms"][arm]["lost_by_candidate_box_distance"][f"{lo}_{hi}_mm"] = paired_mean(loss[mask], zero, group)
    origin = np.column_stack([library["geometry_disk_origin_" + a] for a in "xy"])
    axes = np.stack([np.column_stack([library["geometry_disk_axis_" + a + "_" + b] for b in "xyz"])
                     for a in "xyz"], axis=2)
    if not np.allclose(origin, origin[:1]) or not np.allclose(axes, np.eye(3)):
        raise ValueError("centroid diagnostics require common transverse origins and aligned disk axes")
    centroids = dict(original=centroid_coordinates(original, entries, origin[0]),
                     donor=centroid_coordinates(draw, entries, origin[0]))
    centroids.update({k: centroid_coordinates(v, entries, origin[0], v["mapped"]) for k, v in placed.items()})
    common = np.logical_and.reduce([v["energy"] > 10 for v in centroids.values()])
    for name in ("all", "radius_0_450", "radius_450_550", "radius_550_1000"):
        mask = masks[name] & common
        result["shapes"][name] = dict(common_css_gate_entries=int(mask.sum()), arms={})
        for arm in ("donor", *ARMS):
            shape = {}
            for coordinate in ("radial", "tangent"):
                f, s = centroids[arm][coordinate][mask], centroids["original"][coordinate][mask]
                edges = np.arange(-1400., 1401., 40.)
                hist = paired_hist(f, s, rows[mask] // 100, edges)
                hist["outside_range"] = dict(fast=int(((f < edges[0]) | (f > edges[-1])).sum()),
                                             stock=int(((s < edges[0]) | (s > edges[-1])).sum()))
                hist["mean"] = paired_mean(f, s, rows[mask] // 100)
                shape[coordinate] = hist
            result["shapes"][name]["arms"][arm] = shape
    arrays.update({arm + "_centroid_" + k: v for arm, values in centroids.items() for k, v in values.items()})
    return result, arrays


def plot_report(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("radius_0_450", "radius_450_550", "radius_550_1000")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    x = np.arange(len(names))
    def series(field, arm):
        values = [result["strata"][k]["arms"][arm][field] for k in names]
        return [np.nan if v["difference"] is None else v["difference"] for v in values], [
            np.nan if v["error"] is None else v["error"] for v in values]
    y, err = series("donor_minus_original", "geometric")
    axes[0].errorbar(x, y, yerr=err, marker="o", label="Donor − original, before placement")
    for arm, offset in (("geometric", -.08), ("crystal", .08)):
        y, err = series("mapped_minus_original", arm)
        axes[0].errorbar(x + offset, y, yerr=err, marker="o", label=arm + " mapped − original")
        losses = result["strata"]["radius_0_450"]["arms"][arm]["losses"]
        values = {k: np.nan if v["fast"] is None else v["fast"] for k, v in losses.items()}
        grouped = [values["raw_inner"], sum(values[k] for k in ("raw_outer", "raw_axial")),
                   sum(values[k] for k in LOSS[3:])]
        axes[1].bar(np.arange(3) + offset * 2, grouped, width=.3, label=arm)
    for i, name in enumerate(names):
        if not result["strata"][name]["entries"]:
            axes[0].text(i, .05, "no queries", ha="center", transform=axes[0].get_xaxis_transform())
    if not result["strata"]["radius_0_450"]["entries"]:
        axes[1].text(.5, .5, "No inner-radius queries", ha="center", transform=axes[1].transAxes)
    axes[0].axhline(0, color="gray", lw=1)
    axes[0].set_xticks(x, ("<450", "450–550", "550–1000"))
    axes[0].set_xlabel("Incident radius [mm]")
    axes[0].set_ylabel("Mean both-disk Birks CSS energy difference [MeV]")
    axes[0].legend(fontsize=8)
    axes[1].set_xticks(np.arange(3), ("Raw inner hole", "Raw outer/axial", "Remaining mapping"))
    axes[1].set_ylabel("Mean lost Birks CSS energy [MeV/query]")
    axes[1].set_title("Inner-radius queries only (<450 mm)")
    axes[1].legend()
    fig.suptitle("Frozen development draw: CSS accounting, not reconstructed-energy recovery")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("tables", "library", "comparison", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261105)
    args = p.parse_args()
    library, metadata = load(args.library)
    with np.load(args.comparison / "clusters.npz", allow_pickle=False) as saved:
        entries, original, draw, placed, audit = reproduce(args.tables, library, metadata, saved, args.seed)
        result, arrays = diagnose(library, entries, original, draw, placed, saved)
    result.update(audit=audit, seed=args.seed,
        scope="Frozen development draw, both disks, all direct-front queries including zeros; filtered stored population. "
              "Exact Birks CSS balance: mapped-original=(donor-original)-lost. Not a decomposition of leading-reco energy. "
              "CSS>10 gates and miss partitions cannot predict reconstructed selection recovery. "
              "No-loss selection failures exclude discarded energy, not remapping/time/response effects. "
              "Donor-minus-original includes finite sampling and incident/population mismatch, not certified model bias. "
              "Candidate-box distances use RAW positions and the geometric placer's four transverse-center neighbors, "
              "also for the crystal arm; these are not crystal-remapping displacements. "
              "Centroid shapes require all four CSS totals>10; diagonal paired histogram chi2 is descriptive, not a p-value. "
              "Exact saved IDs/entries/geometry/donor indices/offsets are checked; original comparison did not save payload/code hashes. "
              "Current hashes identify this diagnostic, not retrospective certification of reconstructed input payloads. "
              "Paired 100-source-row errors exclude library training/draw uncertainty. No new reconstruction or tuning.")
    files = dict(tables=args.tables, library=args.library, clusters=args.comparison / "clusters.npz")
    result["inputs"] = {}
    for name, path in files.items():
        with path.open("rb") as source:
            result["inputs"][name] = dict(path=str(path), sha256=hashlib.file_digest(source, "sha256").hexdigest())
    args.out.mkdir(parents=True, exist_ok=True)
    _json(args.out / "numbers.json", result)
    np.savez_compressed(args.out / "arrays.npz", **arrays)
    plot_report(result, args.out / "energy_balance.png")
    summary = {k: v for k, v in result.items() if k != "shapes"}
    _json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
