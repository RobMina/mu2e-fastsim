"""Saved-only conditional response diagnosis of the completed production-candidate trial."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import candidate_support as auth, production_compare as pc, source_overlap_report
from .compare import paired_hist, paired_mean, paired_ratio
from .library import Binning, cells
from .validity_report import observables

TRIAL_SHA = "dee6f99faa4882a3ac98ded6a9a7d160aa011e1d739aab78235c443adc4eaf5a"
ARMS = ("control", "geometric", "crystal")
EDGES = dict(energy=np.arange(0., 111., 5.), truth_ep=np.linspace(0, 1.2, 25),
             radial=np.arange(-1400., 1401., 40.), tangent=np.arange(-1400., 1401., 40.),
             time=np.arange(-20., 21., 1.))
CAVEAT = ("Saved exposed-development conditional diagnostics, not confirmation or inclusive acceptance. "
          "Incident momentum/direction and truth E/p are simulation diagnostics, not measured-variable corrections. "
          "Energy and E/p include all front-query zeros. Spatial/time own-selected and jointly-selected samples are separate. "
          "CSS-to-leading-cluster telescoping terms are algebra, NOT causal attribution or calibrated response factors. "
          "Conditional shapes/widths and direction correlations, not independent-shower eventwise distances, test distributional agreement. "
          "Stored-row//100 paired delete-block errors omit library/model/population uncertainty. No fitted normalization or nominal p-values. "
          "Own-selected shape fractions use observed selected denominators including overflow, not a fitted yield. "
          "Small/empty cells remain reported; no model selection by largest observed pull. "
          "Selected-production versus unfiltered-pilot/release confounding and 165 development/pilot parent overlaps remain.")


def equal(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"):
        raise ValueError("saved residual mismatch: " + name)


def coordinates(saved):
    entry = {k[6:]: v for k, v in saved.items() if k.startswith("entry_")}
    geom = {k[9:]: v for k, v in saved.items() if k.startswith("geometry_")}
    origin = np.column_stack([geom["disk_origin_" + a] for a in "xy"])
    axes = np.stack([np.column_stack([geom["disk_axis_" + a + "_" + b] for b in "xyz"]) for a in "xyz"], axis=2)
    if not np.allclose(origin, origin[:1], atol=1e-6, rtol=0) or not np.allclose(axes, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("requires aligned disks with common transverse origin")
    xy = entry["position"][:, :2] - origin[0]
    r = np.linalg.norm(xy, axis=1)
    axis = np.divide(xy, r[:, None], out=np.tile([1., 0.], (len(r), 1)), where=r[:, None] > 0)
    direction = entry["direction"]
    ur = np.sum(direction[:, :2] * axis, axis=1)
    ut = direction[:, 1] * axis[:, 0] - direction[:, 0] * axis[:, 1]
    return entry, origin[0], ur, ut


def masks(entry, ur, ut, diagnostic):
    n = len(ur)
    result = dict(all=np.ones(n, bool), disk0=entry["disk"] == 0, disk1=entry["disk"] == 1)
    for name, value, edges in (("radius", entry["radius"], (0, 450, 550, 1000, np.inf)),
                              ("energy", entry["energy"], (0, 50, 80, 95, 110, np.inf)),
                              ("cosine", entry["cosine"], (0, .4, .7, 1.000001)),
                              ("angle", diagnostic["residual_angle_deg"], (0, 10, 30, np.inf))):
        result.update({f"{name}_{lo}_{hi}": (value >= lo) & (value < hi) for lo, hi in zip(edges[:-1], edges[1:])})
    result.update(radial_direction_negative=ur < 0, radial_direction_nonnegative=ur >= 0,
                  tangent_direction_negative=ut < 0, tangent_direction_nonnegative=ut >= 0)
    for disk in (0, 1):
        for lo, hi in ((0, 450), (450, 550), (550, 1000)):
            result[f"disk{disk}_radius_{lo}_{hi}"] = (entry["disk"] == disk) & (entry["radius"] >= lo) & (entry["radius"] < hi)
    high = (entry["disk"] == 1) & (entry["radius"] < 450) & (entry["energy"] >= 95)
    result.update(highEinnerdisk1=high, problem109=high & (entry["cosine"] >= .7), fallback=diagnostic["pool_fallback"])
    return result


def telescoping(saved, diagnostic, rows):
    original = diagnostic["residual_original"]
    donor = diagnostic["residual_donor"]
    oaccepted = diagnostic["residual_original_time_accepted"]
    control, stock = saved["control_energy"][rows], saved["stockref_energy"][rows]
    result = {}
    for arm in ("geometric", "crystal"):
        fast = saved[arm + "_energy"][rows]
        accepted = diagnostic[f"residual_{arm}_time_accepted"]
        result[arm] = dict(donor=donor - original, spatial=-diagnostic[f"residual_{arm}_lost"],
            timing=-(diagnostic[f"residual_{arm}_time_rejected"] - (original - oaccepted)),
            downstream=(fast - accepted) - (control - oaccepted), control=control - stock)
        if not np.allclose(sum(result[arm].values()), fast - stock, atol=1e-10, rtol=0):
            raise ValueError("telescoping response identity failed")
    return result


def hist_own(fast, stock, group, edges, fm, sm):
    """Own-selected histograms retain flow bins and delete blocks from the original query universe."""
    unique, inverse = np.unique(group, return_inverse=True)
    nb = len(edges) + 1
    def hist(value, mask):
        if not np.all(np.isfinite(value[mask])):
            raise ValueError("nonfinite selected histogram value")
        slot = np.searchsorted(edges, value[mask], side="right")
        slot[value[mask] == edges[-1]] = nb - 2
        return np.bincount(inverse[mask] * nb + slot, minlength=len(unique) * nb).reshape(len(unique), nb)
    f, s = hist(fast, fm), hist(stock, sm)
    ft, st = f.sum(axis=0), s.sum(axis=0)
    nf, ns = ft.sum(), st.sum()
    fd = ft / nf if nf else np.full(nb, np.nan)
    sd = st / ns if ns else np.full(nb, np.nan)
    delta = fd - sd
    if len(unique) > 1 and np.all(nf - f.sum(axis=1) > 0) and np.all(ns - s.sum(axis=1) > 0):
        leave = (ft - f) / (nf - f.sum(axis=1))[:, None] - (st - s) / (ns - s.sum(axis=1))[:, None]
        error = np.sqrt((len(unique) - 1) / len(unique) * np.sum((leave - leave.mean(axis=0)) ** 2, axis=0))
    else:
        error = np.full(nb, np.nan)
    return dict(edges=edges, fast=ft, stock=st, fast_selected=int(nf), stock_selected=int(ns),
                fraction_difference=delta, fraction_error=error, flow_bins=True)


def cell_contributions(fast, stock, cell, edges):
    nb = len(edges) + 1
    def count(value):
        if not np.all(np.isfinite(value)):
            raise ValueError("nonfinite inclusive cell value")
        slot = np.searchsorted(edges, value, side="right")
        slot[value == edges[-1]] = nb - 2
        return np.bincount(cell * nb + slot, minlength=96 * nb).reshape(96, nb)
    f, s = count(fast), count(stock)
    return dict(fast=f, stock=s, difference=f - s)


def analyze(saved, diagnostic):
    from .conditional_stats import moments

    entry, origin, ur, ut = coordinates(saved)
    rows = entry["event_row"]
    if rows.dtype.kind not in "iu" or np.any(rows < 0) or np.any(rows[1:] <= rows[:-1]) or np.any(rows >= len(saved["event_row"])):
        raise ValueError("invalid query rows")
    equal("query rows", rows, saved["query_event_rows"])
    equal("diagnostic query rows", rows, diagnostic["query_event_rows"])
    equal("full IDs", saved["full_ids"], diagnostic["full_ids"])
    equal("full rows", saved["event_row"], diagnostic["event_row"])
    equal("CSS rows", rows, diagnostic["residual_event_row"])
    p = np.sqrt(entry["energy"] ** 2 - .51099895 ** 2)
    if not np.all(np.isfinite(p) & (p > 0)):
        raise ValueError("invalid incident momentum")
    obs = {}
    for arm in ("stockref", *ARMS):
        cluster = {k: saved[arm + "_" + k][rows] for k in ("energy", "time", "x", "y", "z")}
        if np.any(~np.isfinite(cluster["energy"]) | (cluster["energy"] < 0)):
            raise ValueError("invalid saved leading energy")
        obs[arm] = observables(cluster, entry, diagnostic["residual_entry_folded_time"], origin)
    stock = obs["stockref"]
    selected = {k: v["energy"] > 10 for k, v in obs.items()}
    terms = telescoping(saved, diagnostic, rows)
    cell, outside = cells(entry["energy"], entry["cosine"], entry["radius"], Binning())
    if np.any(outside) or np.any(~np.isin(entry["disk"], [0, 1])):
        raise ValueError("query outside fixed diagnostic cells")
    cell_id = np.ravel_multi_index((entry["disk"].astype(int), *cell.T), (2, 4, 3, 4))
    categories = masks(entry, ur, ut, diagnostic)
    for i in range(96):
        categories[f"cell_{i:02d}"] = cell_id == i
    result = dict(caveat=CAVEAT, queries=len(rows), stored_events=len(saved["event_row"]),
                  cell_coordinates=np.column_stack(np.unravel_index(np.arange(96), (2, 4, 3, 4))),
                  regions={}, global_histograms={}, cell_histograms={}, downstream_term_is_causal=False)
    arrays = dict(event_row=rows, momentum=p, direction_radial=ur, direction_tangent=ut, cell_id=cell_id)
    for arm in ARMS:
        result["global_histograms"][arm] = {}
        for name, edges in EDGES.items():
            f, s = obs[arm][name], stock[name]
            joint = selected[arm] & selected["stockref"] if name in ("radial", "tangent", "time") else np.ones(len(rows), bool)
            h = paired_hist(f[joint], s[joint], rows[joint] // 100, edges)
            h["underflow"] = dict(fast=int((f[joint] < edges[0]).sum()), stock=int((s[joint] < edges[0]).sum()))
            h["overflow"] = dict(fast=int((f[joint] > edges[-1]).sum()), stock=int((s[joint] > edges[-1]).sum()))
            result["global_histograms"][arm][name] = h
        result["cell_histograms"][arm] = {}
        for name in ("energy", "truth_ep"):
            result["cell_histograms"][arm][name] = cell_contributions(obs[arm][name], stock[name], cell_id, EDGES[name])
    for label, mask in categories.items():
        group = rows[mask] // 100
        zero = np.zeros(int(mask.sum()))
        record = dict(entries=int(mask.sum()), source_blocks=len(np.unique(group)), arms={})
        for arm in ARMS:
            f, s = obs[arm], stock
            fs, ss = selected[arm][mask], selected["stockref"][mask]
            item = dict(selected=int(fs.sum()), stock_selected=int(ss.sum()), gains=int((fs & ~ss).sum()),
                misses=int((ss & ~fs).sum()), ratio=paired_ratio(fs, ss, group),
                energy=paired_mean(f["energy"][mask], s["energy"][mask], group),
                truth_ep=paired_mean(f["truth_ep"][mask], s["truth_ep"][mask], group),
                energy_momentum=moments(f["energy"][mask], s["energy"][mask], p[mask], group),
                ep_momentum=moments(f["truth_ep"][mask], s["truth_ep"][mask], p[mask], group),
                ep_ge1=paired_mean(f["truth_ep"][mask] >= 1, s["truth_ep"][mask] >= 1, group), spatial={})
            if arm in terms:
                item["energy_terms"] = {k: paired_mean(v[mask], zero, group) for k, v in terms[arm].items()}
                item["ep_terms"] = {k: paired_mean(v[mask] / p[mask], zero, group) for k, v in terms[arm].items()}
            for name, x in (("radial", ur), ("tangent", ut), ("time", diagnostic["residual_entry_folded_time"])):
                item["spatial"][name] = dict(
                    own=moments(f[name][mask], s[name][mask], x[mask], group, fast_mask=fs, stock_mask=ss),
                    joint=moments(f[name][mask], s[name][mask], x[mask], group, fast_mask=fs & ss, stock_mask=fs & ss))
            if not label.startswith("cell_"):
                item["shapes"] = {name: hist_own(f[name][mask], s[name][mask], group, EDGES[name],
                    fs if name in ("radial", "tangent", "time") else np.ones(len(fs), bool),
                    ss if name in ("radial", "tangent", "time") else np.ones(len(ss), bool)) for name in EDGES}
            record["arms"][arm] = item
        result["regions"][label] = record
    for arm, value in obs.items():
        arrays.update({arm + "_" + k: v for k, v in value.items()})
    for arm, value in terms.items():
        arrays.update({arm + "_term_" + k: v for k, v in value.items()})
    return result, arrays


def stage_correlations(diagnostic, arrays):
    """CSS-centroid diagnostics localize conditional structure, not reconstructed-resolution closure."""
    from .conditional_stats import moments

    group = arrays["event_row"] // 100
    result = {}
    for arm in ("donor", "geometric", "crystal"):
        energy = diagnostic[f"css_{arm}_centroid_energy"]
        original = diagnostic["css_original_centroid_energy"]
        item = dict(energy_radial_direction=moments(energy, original, arrays["direction_radial"], group),
                    energy_tangent_direction=moments(energy, original, arrays["direction_tangent"], group))
        for name, key in (("radial", "direction_radial"), ("tangent", "direction_tangent")):
            item[name] = moments(diagnostic[f"css_{arm}_centroid_{name}"], diagnostic[f"css_original_centroid_{name}"],
                arrays[key], group, fast_mask=energy > 0, stock_mask=original > 0)
        result[arm] = item
    return dict(caveat="Whole-event CSS centroids, own positive-CSS denominators; not reconstructed leading-cluster quantities or closure.", arms=result)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [f"disk{d}_radius_{lo}_{hi}" for d in (0, 1) for lo, hi in ((0, 450), (450, 550), (550, 1000))]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, variable in zip(axes, ("energy", "truth_ep")):
        for arm, shift in (("control", -.2), ("geometric", 0), ("crystal", .2)):
            values = [result["regions"][k]["arms"][arm][variable] for k in names]
            ax.errorbar(x + shift, [v["difference"] for v in values], yerr=[v["error"] for v in values], fmt="o", label=arm)
        ax.axhline(0, color="gray", lw=1)
        ax.set_xticks(x, [k.replace("_radius_", "\nr=") for k in names], fontsize=8)
        ax.set_ylabel("Mean difference vs stock [MeV]" if variable == "energy" else "Mean truth E/p difference vs stock")
        ax.legend(fontsize=8)
    fig.suptitle("Fixed production-candidate draw: incident-defined regions, all zeros retained")
    fig.tight_layout()
    fig.savefig(out / "conditional_bias.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, arm in enumerate(("geometric", "crystal")):
        for ax, name in zip(axes[row], ("truth_ep", "radial", "tangent")):
            h = result["regions"]["all"]["arms"][arm]["shapes"][name]
            edges = np.asarray(h["edges"])
            for field, label in (("stock", "Stock"), ("fast", arm)):
                counts = np.asarray(h[field])
                ax.stairs(counts[1:-1] / counts.sum(), edges, label=label)
            ax.set_xlabel(name + (" [mm]" if name in ("radial", "tangent") else ""))
            ax.set_ylabel("Fraction / bin")
            ax.legend(fontsize=8)
    fig.suptitle("Truth E/p: all queries; spatial: each arm's own E>10 sample (flows retained in denominators)")
    fig.tight_layout()
    fig.savefig(out / "conditional_shapes.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name in zip(axes, ("radial", "tangent")):
        for arm, shift in (("control", -.2), ("geometric", 0), ("crystal", .2)):
            values = [result["regions"][k]["arms"][arm]["spatial"][name]["own"] for k in names]
            ax.errorbar(x + shift, [v["difference"]["correlation"] for v in values],
                yerr=[v["error"]["correlation"] for v in values], fmt="o", label=arm)
        ax.axhline(0, color="gray", lw=1)
        ax.set_xticks(x, [k.replace("_radius_", "\nr=") for k in names], fontsize=8)
        ax.set_ylabel(name + " centroid–direction correlation minus stock")
        ax.legend(fontsize=8)
    fig.suptitle("Conditional response: own-selected E>10, paired stored-row block errors")
    fig.tight_layout()
    fig.savefig(out / "direction_correlation.png", dpi=150)
    plt.close(fig)


def run(args):
    if args.out.exists():
        raise ValueError("new saved-only output directory required")
    inputs = source_overlap_report.Inputs()
    summary = auth.pinned_json(inputs, args.trial / "summary.json", TRIAL_SHA)
    values = {name: auth.pinned_npz(inputs, args.trial / name, summary["output_sha256"][name])
              for name in ("clusters.npz", "diagnostics.npz")}
    previous = auth.pinned_json(inputs, args.trial / "numbers.json", summary["output_sha256"]["numbers.json"])
    result, arrays = analyze(values["clusters.npz"], values["diagnostics.npz"])
    result["css_stage_correlations"] = stage_correlations(values["diagnostics.npz"], arrays)
    for arm in ARMS:
        for name, title in (("energy", "Leading-cluster energy [MeV]"), ("truth_ep", "E/p at entry (truth diagnostic)")):
            old = previous["diagnostics"]["candidate"][arm]["observables"][title]
            for key in ("fast", "stock", "variance", "pull", "chi2", "ndf"):
                if result["global_histograms"][arm][name][key] != old[key]:
                    raise ValueError("saved marginal differs: " + arm + ":" + name + ":" + key)
    inputs.unchanged()
    result["input_sha256"] = inputs.sha256
    args.out.mkdir(parents=True)
    pc._json(args.out / "numbers.json", clean(result))
    np.savez_compressed(args.out / "arrays.npz", **arrays)
    plot(result, args.out)
    compact = {name: {arm: {k: v for k, v in item.items() if k in ("selected", "stock_selected", "energy", "truth_ep", "ep_ge1", "energy_terms", "ep_terms")}
                      for arm, item in region["arms"].items()} for name, region in result["regions"].items()
               if not name.startswith("cell_")}
    pc._json(args.out / "summary.json", clean(dict(scope=CAVEAT, regions=compact)))
    print("Saved-only conditional diagnosis complete; no sampling or reconstruction.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trial", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
