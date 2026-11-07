"""Saved-only assessment of one production-candidate draw and two saved placements."""
from __future__ import annotations

import numpy as np

from . import placement_diagnostic as pd
from . import production_compare as pc
from . import residual_diagnostic as rd
from .compare import paired_mean

ARMS = ("geometric", "crystal")
FIELDS = ("energy", "time", "x", "y", "z")
SCOPE = (
    "One saved production-candidate trial on exposed development queries; no sampling or reconstruction. "
    "Selection is leading-cluster E>10 MeV. All energy means retain zeros; time and spatial marginals use "
    "jointly selected pairs. Eight-arm distances use one identical common-selected subset. "
    "Full stored sampled-arm nonfront zeros are bookkeeping, NOT an inclusive acceptance model. "
    "Paired errors use original stored-row//100 blocks, not donor or compact query indices, and omit "
    "training/model/draw uncertainty. Conditional uniform-pool draw SDs are separate, fixed-library quantities. "
    "CSS, donor occupancy, mismatch, timing and reconstructed-miss strata are descriptive output-conditioned "
    "associations, not closure, causal recovery or a correction model; no nominal p-values. "
    "Selected production versus unfiltered legacy and release differences preclude a library-size-only claim. "
    "No independent confirmation, replacement promotion or throughput claim; the 165 development/pilot "
    "shared-parent flag remains, and original-proton independence is not certified."
)


def _equal(name, actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind in "fc" and b.dtype.kind in "fc"):
        raise ValueError("saved report mismatch: " + name)


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def fixed_strata(entries, draw):
    """Incident-defined thresholds retain the previously exposed 190/109-query regions."""
    masks = pd.strata(entries)
    for feature, edges in (("energy", (0, 50, 80, 95, 110, np.inf)),
                           ("cosine", (0, .4, .7, 1.000001))):
        masks.update({f"{feature}_{lo}_{hi}": (entries[feature] >= lo) & (entries[feature] < hi)
                      for lo, hi in zip(edges[:-1], edges[1:])})
    for disk in (0, 1):
        high = (entries["disk"] == disk) & (entries["radius"] < 450) & (entries["energy"] >= 95)
        masks[f"inner_highE_disk{disk}"] = high
        masks[f"inner_highE_disk{disk}_coslow"] = high & (entries["cosine"] < .7)
        masks[f"inner_highE_disk{disk}_coshigh"] = high & (entries["cosine"] >= .7)
    masks.update(fallback=draw["fallback"], exact_no_fallback=~draw["fallback"],
                 unsupported=~draw["supported"])
    return masks


def _saved(prepared, baseline, k8, output):
    """Check full-row alignment before any query subsetting; inputs are never mutated."""
    rows = np.asarray(prepared["entries"]["event_row"])
    n = len(prepared["full_ids"])
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < 0) or
            np.any(rows >= n) or np.any(np.diff(rows) <= 0)):
        raise ValueError("invalid original stored query rows")
    expected = dict(event_row=np.arange(n), full_ids=prepared["full_ids"], query_event_rows=rows)
    for prefix, name in (("entry_", "entries"), ("geometry_", "geometry"), ("timing_", "timing")):
        expected.update({prefix + k: v for k, v in prepared[name].items()})
    collections = {}
    front = np.zeros(n, bool)
    front[rows] = True
    for label, saved in (("legacy", baseline), ("k8", k8), ("candidate", output)):
        for name, value in expected.items():
            _equal(label + ":" + name, saved[name], value)
        clusters = {arm: {f: np.asarray(saved[arm + "_" + f]) for f in FIELDS}
                    for arm in ("stockref", "control", *ARMS)}
        for arm, cluster in clusters.items():
            energy = cluster["energy"]
            if energy.shape != (n,) or np.any(~np.isfinite(energy) | (energy < 0)):
                raise ValueError("invalid full stored energy: " + label + ":" + arm)
            for field in FIELDS[1:]:
                value = cluster[field]
                if value.shape != (n,) or not np.all(np.isfinite(value[energy > 0])):
                    raise ValueError("invalid positive cluster field: " + label + ":" + arm + ":" + field)
            if arm in ARMS and np.any(energy[~front] != 0):
                raise ValueError("sampled-arm nonfront energy must be zero: " + label + ":" + arm)
            if label != "legacy" and arm in ("stockref", "control"):
                for field in FIELDS:
                    _equal(label + ":" + arm + "_" + field, cluster[field], collections["legacy"][arm][field])
        collections[label] = clusters
    for name in ("donor", "supported", "fallback", "offsets", "radius_pool_size", "bin_pool_size"):
        _equal("sampling_" + name, output["sampling_" + name], prepared["draw"][name])
    return collections, front


def _pool_arrays(library, prepared, pools):
    """Validate the frozen eligible pools and chosen outcomes without enumerating or drawing."""
    draw, entries = prepared["draw"], prepared["entries"]
    n, nd = len(entries["energy"]), len(library["entry_energy"])
    eligible = np.asarray(pools["eligible"])
    if eligible.shape != (n, 8) or eligible.dtype.kind not in "iu" or np.any((eligible < -1) | (eligible >= nd)):
        raise ValueError("invalid fixed K8 eligible pool")
    valid = eligible >= 0
    size = valid.sum(axis=1)
    for name, value in (("event_row", entries["event_row"]), ("size", size),
                        ("supported", size > 0)):
        _equal("pool_" + name, pools[name], value)
    for name, target in (("supported", "supported"), ("fallback", "fallback"),
                         ("radius_pool_size", "size"), ("bin_pool_size", "full_size")):
        _equal("draw_" + name, draw[name], pools[target])
    supported, donor = np.asarray(draw["supported"]), np.asarray(draw["donor"])
    if (supported.dtype.kind != "b" or np.asarray(draw["fallback"]).dtype.kind != "b" or
            donor.shape != (n,) or donor.dtype.kind not in "iu" or
            np.any(donor[~supported] != -1) or np.any((donor[supported] < 0) | (donor[supported] >= nd)) or
            not np.all(np.any(eligible[supported] == donor[supported, None], axis=1))):
        raise ValueError("chosen donor is not in authenticated pool")
    if np.any(pools["full_size"] < size):
        raise ValueError("invalid full-cell pool size")
    ordered = np.sort(np.where(valid, eligible, nd), axis=1)
    if np.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] < nd)):
        raise ValueError("duplicate eligible donor")
    offsets = np.asarray(library["offsets"])
    lengths = np.diff(offsets)
    if (offsets.shape != (nd + 1,) or offsets.dtype.kind not in "iu" or offsets[0] != 0 or
            np.any(lengths < 0) or offsets[-1] != len(library["step_energy_birks"])):
        raise ValueError("invalid library offsets")
    energy = np.asarray(library["step_energy_birks"])
    if np.any(~np.isfinite(energy) | (energy < 0)):
        raise ValueError("invalid library CSS energy")
    totals = np.bincount(np.repeat(np.arange(nd), lengths), weights=energy, minlength=nd)
    outcomes = dict(energy=totals, empty=lengths == 0, weak=totals <= 10)
    expected_lengths = np.zeros(n, np.int64)
    expected_lengths[supported] = lengths[donor[supported]]
    _equal("draw_offsets", draw["offsets"], np.r_[0, np.cumsum(expected_lengths)])
    _equal("draw_entry", draw["entry"], np.repeat(np.arange(n), expected_lengths))
    expected_energy = np.zeros(n)
    expected_energy[supported] = totals[donor[supported]]
    starts = np.repeat(np.where(supported, offsets[np.maximum(donor, 0)], 0), expected_lengths)
    local = np.arange(len(draw["entry"])) - np.repeat(draw["offsets"][:-1], expected_lengths)
    _equal("draw_step_energy_birks", draw["energy_birks"], energy[starts + local])
    if not np.allclose(pd.energy_sum(draw, n), expected_energy, rtol=1e-12, atol=1e-10):
        raise ValueError("chosen donor CSS energy mismatch")
    arrays = dict(event_row=entries["event_row"], supported=supported, fallback=draw["fallback"], donor=donor)
    for name, value in outcomes.items():
        _equal("outcome_" + name, pools["outcome_" + name], value)
        gathered = np.zeros(eligible.shape)
        gathered[valid] = value[eligible[valid]]
        mean = np.divide(gathered.sum(axis=1), size, out=np.full(n, np.nan), where=supported)
        _equal("prob_" + name, pools["prob_" + name], mean)
        variance = np.divide(np.sum(np.where(valid, (gathered - mean[:, None]) ** 2, 0), axis=1),
                             size, out=np.full(n, np.nan), where=supported)
        actual = np.zeros(n)
        actual[supported] = value[donor[supported]]
        arrays.update({"expected_" + name: mean, "variance_" + name: variance, "actual_" + name: actual})
    for arm in ARMS:
        steps = prepared["placed"][arm]
        for name in ("entry", "energy_birks", "time"):
            _equal(arm + "_" + name, steps[name], draw[name])
        if np.asarray(steps["mapped"]).shape != draw["entry"].shape or np.asarray(steps["mapped"]).dtype.kind != "b":
            raise ValueError("invalid saved placement mask")
    return arrays


def _draw_summary(prepared, arrays, masks):
    original = prepared["original"]
    n = len(prepared["entries"]["energy"])
    original_energy = pd.energy_sum(original, n)
    outcomes = dict(energy=original_energy, empty=np.bincount(original["entry"], minlength=n) == 0,
                    weak=original_energy <= 10)
    result = {}
    for label, mask in masks.items():
        use = mask & arrays["supported"]
        rows = arrays["event_row"][use]
        count = int(use.sum())
        record = dict(entries=int(mask.sum()), supported=count, unsupported=int((mask & ~arrays["supported"]).sum()),
                      fallback=int((mask & arrays["fallback"]).sum()), outcomes={})
        for name, original_value in outcomes.items():
            expected, actual = arrays["expected_" + name][use], arrays["actual_" + name][use]
            sd = float(np.sqrt(arrays["variance_" + name][use].sum()))
            record["outcomes"][name] = dict(actual_total=float(actual.sum()), expected_total=float(expected.sum()),
                original_total=float(original_value[use].sum()), conditional_draw_sd_total=sd,
                conditional_draw_sd_mean=sd / count if count else None,
                actual_minus_expected=paired_mean(actual, expected, rows // 100),
                actual_minus_original=paired_mean(actual, original_value[use], rows // 100),
                expected_minus_original=paired_mean(expected, original_value[use], rows // 100))
        result[label] = record
    return result


def report(library, prepared, baseline, k8, output, pools):
    """Return (strict-JSON report, numeric arrays); callers authenticate archives and use pc._json."""
    from .reconstruct import report as cluster_report

    collections, front = _saved(prepared, baseline, k8, output)
    pool_arrays = _pool_arrays(library, prepared, pools)
    entries, rows = prepared["entries"], prepared["entries"]["event_row"]
    masks = fixed_strata(entries, prepared["draw"])
    clusters = {name: collections["legacy"][name] for name in ("stockref", "control")}
    clusters.update({label + "_" + arm: collection[arm] for label, collection in collections.items() for arm in ARMS})
    common = np.logical_and.reduce([v["energy"][rows] > 10 for v in clusters.values()])
    common_rows = rows[common]
    distances = {name: np.hypot(v["x"][common_rows] - clusters["stockref"]["x"][common_rows],
                               v["y"][common_rows] - clusters["stockref"]["y"][common_rows])
                 for name, v in clusters.items()}
    result = dict(scope=SCOPE, selection="leading_cluster_energy > 10 MeV", source_block="original stored event_row // 100",
                  queries=len(rows), full_stored_rows=len(front), strata_counts={k: int(m.sum()) for k, m in masks.items()},
                  diagnostics={}, comparisons={}, change={}, draw=_draw_summary(prepared, pool_arrays, masks))
    result["eight_arm_common_selected"] = int(common.sum())
    result["eight_arm_distance_mm"] = {k: paired_mean(v, np.zeros(len(v)), common_rows // 100) for k, v in distances.items()}
    result["eight_arm_distance_change_mm"] = {label: {arm: paired_mean(distances["candidate_" + arm],
        distances[label + "_" + arm], common_rows // 100) for arm in ARMS} for label in ("legacy", "k8")}
    for label, collection in collections.items():
        result["diagnostics"][label] = pc.diagnostics(prepared, collection)
        for arm, item in result["diagnostics"][label].items():
            item["common_selected"] = int(common.sum())
            item["common_distance_mm"] = result["eight_arm_distance_mm"]["control" if arm == "control" else label + "_" + arm]
            item["common_selection_scope"] = "same eight-arm subset"
    for name, cluster in clusters.items():
        result["comparisons"][name] = {label: cluster_report({k: v[rows[mask]] for k, v in cluster.items()},
            {k: v[rows[mask]] for k, v in clusters["stockref"].items()}, rows[mask]) for label, mask in masks.items()}
    for reference in ("legacy", "k8"):
        result["change"][reference] = {arm: {label: cluster_report(
            {k: v[rows[mask]] for k, v in collections["candidate"][arm].items()},
            {k: v[rows[mask]] for k, v in collections[reference][arm].items()}, rows[mask])
            for label, mask in masks.items()} for arm in ARMS}
    result["full_stored_accounting"] = dict(rows=len(front), front=int(front.sum()), nonfront=int((~front).sum()),
        inclusive_sampled_acceptance_model=False,
        original_control_all_rows=cluster_report(clusters["control"], clusters["stockref"], np.arange(len(front))), arms={})
    for name, cluster in clusters.items():
        result["full_stored_accounting"]["arms"][name] = {label: dict(rows=int(mask.sum()),
            selected=int((cluster["energy"][mask] > 10).sum()), zero=int((cluster["energy"][mask] == 0).sum()),
            energy_sum=float(cluster["energy"][mask].sum()))
            for label, mask in (("all", np.ones(len(front), bool)), ("front", front), ("nonfront", ~front))}
    css, css_arrays = pd.diagnose(library, entries, prepared["original"], prepared["draw"], prepared["placed"], output)
    context = dict(prepared, census=dict(events=len(front), front_entries=len(rows), nonfront_events=int((~front).sum())),
                   independence=dict(source_independence_certified=False, authentication="caller-authenticated saved inputs"))
    residual, residual_arrays = rd.diagnose(library, context, output, css_arrays)
    residual["caveat"] = SCOPE + " Exact float32 lower timing gate; no upper cut or second PBT subtraction. Nominal end exposure is not photon/waveform loss."
    result.update(css=css, residual=residual)
    result["draw_uncertainty"] = dict(
        conditional="Independent uniform eligible-pool draws with the authenticated library and queries held fixed; not training/model/population uncertainty.",
        paired="actual_minus_expected, actual_minus_original and expected_minus_original errors are stored-row block jackknives, not draw SDs.",
        weak="Whole-event both-disk Birks energy <=10 MeV, including empty donors; unsupported is separate.")
    result["css"]["caveat"] = "Output-conditioned CSS>10 centroid subsets and loss/miss associations are descriptive, not closure."
    arrays = dict(event_row=np.arange(len(front)), full_ids=prepared["full_ids"], query_event_rows=rows,
                  front=front, eight_arm_common_selected=common, eight_arm_common_event_rows=common_rows)
    for prefix, values in (("pool_", pool_arrays), ("css_", css_arrays), ("residual_", residual_arrays)):
        arrays.update({prefix + k: v for k, v in values.items()})
    arrays.update({name + "_selected": cluster["energy"] > 10 for name, cluster in clusters.items()})
    return _json_safe(result), arrays
