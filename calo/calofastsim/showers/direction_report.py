"""Saved-only capped-direction assessment against frozen PRODUCTIONK8 and stock."""
from __future__ import annotations

import numpy as np

from . import candidate_report as inclusive
from . import candidate_residual as residual
from .compare import paired_mean, paired_ratio
from .conditional_stats import moments

ARMS = inclusive.ARMS
SCOPE = (
    "One saved capped-direction development trial; no sampling, placement or reconstruction. "
    "In inherited inclusive keys k8 means frozen PRODUCTIONK8, NOT old505K8; candidate means new direction. "
    "Direct differences are new direction minus PRODUCTIONK8, paired by original stored-row//100. "
    "Own E>10 masks have separate denominators; joint_two_way requires new and PRODUCTIONK8 E>10. "
    "Common_three_way requires new, PRODUCTIONK8 and stock E>10 on ONE identical subset. "
    "Only common_three_way comparisons support a joint multivariant improvement assessment; never subtract "
    "new-stock and production-stock statistics formed on different intersections. Signed changes alone are not improvement. "
    "All supplied original blocks in each region remain in conditional-moment errors, including unselected blocks. "
    "Energy and truth E/p retain zeros. No eventwise stochastic distance is resolution; no nominal p-values, "
    "confirmation, promotion or model/training uncertainty claim. Conditional pool draw SD is separate. "
    "Full stored nonfront sampled zeros are bookkeeping, not inclusive generated acceptance. "
    "Selected-production/unfiltered-pilot confounding and the 165 development/40 donor overlap flag remain."
)


def _flags(prepared, output, pools):
    n = len(prepared["entries"]["event_row"])
    flags = {}
    for name in ("direction_used", "direction_fallback"):
        value = np.asarray(prepared["draw"][name])
        if value.shape != (n,) or value.dtype.kind != "b":
            raise ValueError("invalid direction flag: " + name)
        flags[name] = value
        if "sampling_" + name in output:
            inclusive._equal(name, output["sampling_" + name], value)
        if name in pools:
            inclusive._equal("pool_" + name, pools[name], value)
    supported = np.asarray(prepared["draw"]["supported"])
    if np.any(flags["direction_used"] & flags["direction_fallback"]) or np.any(flags["direction_used"] & ~supported):
        raise ValueError("inconsistent direction flags")
    if not np.array_equal(supported, flags["direction_used"] | flags["direction_fallback"]):
        raise ValueError("direction branch flags do not partition supported queries")
    return flags


def _coordinates(saved):
    entry = {k[6:]: v for k, v in saved.items() if k.startswith("entry_")}
    n = len(entry["event_row"])
    for name in ("position", "direction"):
        value = np.asarray(entry[name])
        if value.shape != (n, 3) or not np.all(np.isfinite(value)):
            raise ValueError("invalid incident coordinate: " + name)
    for name in ("energy", "radius", "cosine", "time", "disk"):
        value = np.asarray(entry[name])
        if value.shape != (n,) or not np.all(np.isfinite(value)):
            raise ValueError("invalid incident feature: " + name)
    if np.any(entry["energy"] <= .51099895):
        raise ValueError("invalid incident momentum")
    _, _, ur, ut = residual.coordinates(saved)
    return entry, ur, ut


def _diagnostics(saved, diagnostic):
    rows = saved["query_event_rows"]
    n = len(rows)
    for name in ("event_row", "full_ids", "query_event_rows"):
        inclusive._equal("diagnostic_" + name, diagnostic[name], saved[name])
    for name in ("residual_event_row", "css_event_row", "pool_event_row"):
        inclusive._equal(name, diagnostic[name], rows)
    for name in ("donor", "supported", "fallback"):
        inclusive._equal("production_pool_" + name, diagnostic["pool_" + name], saved["sampling_" + name])
    for name, value in diagnostic.items():
        if name.startswith(("residual_", "css_", "pool_")) and np.asarray(value).shape != (n,):
            raise ValueError("invalid diagnostic shape: " + name)
    for name in ("original", "donor"):
        inclusive._equal("CSS energy " + name, diagnostic["residual_" + name], diagnostic["css_" + name])
        inclusive._equal("CSS centroid energy " + name, diagnostic["residual_" + name], diagnostic[f"css_{name}_centroid_energy"])
    for name in ("original", "donor", *ARMS):
        energy = np.asarray(diagnostic[f"css_{name}_centroid_energy"])
        if np.any(~np.isfinite(energy) | (energy < 0)):
            raise ValueError("invalid CSS centroid energy")
        for coordinate in ("radial", "tangent"):
            if not np.all(np.isfinite(diagnostic[f"css_{name}_centroid_{coordinate}"][energy > 0])):
                raise ValueError("nonfinite positive CSS centroid")
    for name in ("original", "donor", "original_time_accepted"):
        value = diagnostic["residual_" + name]
        if np.any(~np.isfinite(value) | (value < 0)):
            raise ValueError("invalid diagnostic energy: " + name)
    for arm in ARMS:
        parts = [diagnostic[f"residual_{arm}_{key}"] for key in ("lost", "time_rejected", "time_accepted")]
        if any(np.any(~np.isfinite(value) | (value < 0)) for value in parts):
            raise ValueError("invalid diagnostic energy partition")
        if not np.allclose(sum(parts), diagnostic["residual_donor"], rtol=0, atol=1e-10):
            raise ValueError("diagnostic donor partition mismatch")
        inclusive._equal("mapped CSS energy", diagnostic[f"css_{arm}_centroid_energy"], diagnostic[f"residual_{arm}_mapped"])
    if not np.all(np.isfinite(diagnostic["residual_entry_folded_time"])):
        raise ValueError("invalid folded entry time")


def _pair(fast, prior, x, group, fm, pm):
    return dict(own=moments(fast, prior, x, group, fast_mask=fm, stock_mask=pm),
                joint_two_way=moments(fast, prior, x, group, fast_mask=fm & pm, stock_mask=fm & pm))


def _direct(new, old, diagnostic, previous, categories):
    rows, p = new["event_row"], new["momentum"]
    result = dict(scope=SCOPE, labels=dict(fast="new_direction", stock="PRODUCTIONK8"), regions={}, cell_histograms={})
    for arm in ARMS:
        result["cell_histograms"][arm] = {name: residual.cell_contributions(
            new[arm + "_" + name], old[arm + "_" + name], new["cell_id"], residual.EDGES[name])
            for name in ("energy", "truth_ep")}
    for label, mask in categories.items():
        group = rows[mask] // 100
        record = dict(entries=int(mask.sum()), source_blocks=len(np.unique(group)), arms={})
        for arm in ARMS:
            f = {key: new[arm + "_" + key][mask] for key in residual.EDGES}
            o = {key: old[arm + "_" + key][mask] for key in residual.EDGES}
            s = {key: new["stockref_" + key][mask] for key in residual.EDGES}
            fm, om, sm = f["energy"] > 10, o["energy"] > 10, s["energy"] > 10
            common = fm & om & sm
            item = dict(selected=int(fm.sum()), production_selected=int(om.sum()), stock_selected=int(sm.sum()),
                        gains=int((fm & ~om).sum()), misses=int((om & ~fm).sum()),
                        selection=paired_mean(fm, om, group), ratio=paired_ratio(fm, om, group),
                        energy=paired_mean(f["energy"], o["energy"], group),
                        truth_ep=paired_mean(f["truth_ep"], o["truth_ep"], group),
                        ep_ge1=paired_mean(f["truth_ep"] >= 1, o["truth_ep"] >= 1, group),
                        energy_momentum=moments(f["energy"], o["energy"], p[mask], group),
                        ep_momentum=moments(f["truth_ep"], o["truth_ep"], p[mask], group), spatial={},
                        joint_two_way_selected=int((fm & om).sum()), common_three_way_selected=int(common.sum()),
                        common_three_way_scope="new_direction, PRODUCTIONK8 and stock E>10; identical mask for all three contrasts",
                        common_three_way={})
            for name, x in (("radial", new["direction_radial"]), ("tangent", new["direction_tangent"]),
                            ("time", diagnostic["residual_entry_folded_time"])):
                item["spatial"][name] = _pair(f[name], o[name], x[mask], group, fm, om)
                item["common_three_way"][name] = {key: moments(a, b, x[mask], group,
                    fast_mask=common, stock_mask=common) for key, a, b in (
                    ("new_vs_stock", f[name], s[name]), ("production_vs_stock", o[name], s[name]),
                    ("new_vs_production", f[name], o[name]))}
            item["energy_terms"] = {key: paired_mean(new[f"{arm}_term_{key}"][mask],
                old[f"{arm}_term_{key}"][mask], group) for key in ("donor", "spatial", "timing", "downstream", "control")}
            item["ep_terms"] = {key: paired_mean(new[f"{arm}_term_{key}"][mask] / p[mask],
                old[f"{arm}_term_{key}"][mask] / p[mask], group) for key in item["energy_terms"]}
            item["css_energy_partition"] = {key: paired_mean(diagnostic["residual_" + key][mask],
                previous["residual_" + key][mask], group) for key in
                ("donor", arm + "_lost", arm + "_time_rejected", arm + "_time_accepted")}
            if not label.startswith("cell_"):
                item["shapes"] = {name: residual.hist_own(f[name], o[name], group, edges,
                    fm if name in ("radial", "tangent", "time") else np.ones(len(fm), bool),
                    om if name in ("radial", "tangent", "time") else np.ones(len(om), bool))
                    for name, edges in residual.EDGES.items()}
            record["arms"][arm] = item
        result["regions"][label] = record
    return result


def report(library, prepared, baseline, production, output, pools, production_diagnostics):
    """Return inclusive JSON, diagnostic arrays, conditional JSON and conditional arrays; no I/O."""
    ids = np.asarray(prepared["full_ids"])
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("full stored IDs must be a unique one-dimensional array")
    flags = _flags(prepared, output, pools)
    entry, ur, ut = _coordinates(output)
    inclusive._saved(prepared, baseline, production, output)
    _diagnostics(production, production_diagnostics)
    result, diagnostic = inclusive.report(library, prepared, baseline, production, output, pools)
    _diagnostics(output, diagnostic)
    for name, value in diagnostic.items():
        if name.startswith(("css_original", "residual_original", "residual_entry_")):
            inclusive._equal("original diagnostic " + name, production_diagnostics[name], value)
    conditional, arrays = residual.analyze(output, diagnostic)
    _, old = residual.analyze(production, production_diagnostics)
    for name in ("event_row", "momentum", "direction_radial", "direction_tangent", "cell_id"):
        inclusive._equal("conditional " + name, arrays[name], old[name])
    categories = inclusive.fixed_strata(entry, prepared["draw"])
    categories.update(residual.masks(entry, ur, ut, diagnostic))
    categories.update(flags)
    categories.update({f"cell_{i:02d}": arrays["cell_id"] == i for i in range(96)})
    conditional["direction_scope"] = SCOPE
    conditional["direct_production"] = _direct(arrays, old, diagnostic, production_diagnostics, categories)
    conditional["css_stage_correlations"] = residual.stage_correlations(diagnostic, arrays)
    conditional["production_css_stage_correlations"] = residual.stage_correlations(production_diagnostics, old)
    conditional["conditional_css_stage_correlations"] = {label: residual.stage_correlations(
        {k: v[mask] for k, v in diagnostic.items() if k.startswith("css_")},
        {k: arrays[k][mask] for k in ("event_row", "direction_radial", "direction_tangent")})
        for label, mask in categories.items()}
    result["direction_scope"] = SCOPE
    result["archive_labels"] = dict(legacy="original legacy", k8="PRODUCTIONK8 (not old505K8)", candidate="new_direction")
    result["direction_strata_counts"] = {k: int(v.sum()) for k, v in categories.items()}
    result["direction_draw"] = inclusive._draw_summary(prepared, {
        k.removeprefix("pool_"): v for k, v in diagnostic.items() if k.startswith("pool_")}, flags)
    for name, value in flags.items():
        diagnostic[name] = value.copy()
        arrays[name] = value.copy()
    arrays.update({"production_" + k: v for k, v in old.items()})
    for arm in ARMS:
        arrays[arm + "_common_three_way_selected"] = ((arrays[arm + "_energy"] > 10) &
            (old[arm + "_energy"] > 10) & (arrays["stockref_energy"] > 10))
    return inclusive._json_safe(result), diagnostic, residual.clean(conditional), arrays


def plot(conditional_result, out):
    """Write the inherited PNG conditional plots only; the caller owns output directories."""
    residual.plot(conditional_result, out)
