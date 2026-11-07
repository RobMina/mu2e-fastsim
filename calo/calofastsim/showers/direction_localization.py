"""Pure saved-array residual localization; the caller owns authentication and I/O."""
from __future__ import annotations

import numpy as np

from .candidate_residual import clean, equal
from .compare import paired_mean
from .conditional_stats import moments

ARMS = ("geometric", "crystal")
TERMS = ("donor", "spatial", "timing", "downstream", "control")
EP_EDGES = np.linspace(0., 1.2, 25)
TAILS = {"radial": (100., 200., 400.), "tangent": (50., 100., 200.)}
SCOPE = (
    "Saved exposed-file development diagnostics only; no sampling, replay, placement or reconstruction. "
    "Energy/Ep include all front-query zeros. Ep count errors are sqrt(B*sample variance of signed "
    "original-row//100 block counts), using the FULL query block universe even for individual cells. "
    "Partition counts, not regional chi-squares or errors, are additive. Moment and fraction errors "
    "are paired delete-original-block errors retaining unselected blocks within each complete region; "
    "undefined deletions are null. Widths are population standard deviations, not detector/track resolution. "
    "Common_three_way is identical new/PRODUCTIONK8/stock E>10 for each placement; control is not "
    "silently inserted when its centroid is absent. Separate common_four_way also requires control E>10 "
    "and compares all four on ONE identical mask. Absolute quantiles/maxima and own-mean variance "
    "partitions are descriptive; no central-only replacement width. Stock-Ep windows are outcome-conditioned "
    "diagnostics, not fitted cuts. Five terms divided by the same saved query momentum are algebra, "
    "not causal mechanisms/calibration or evidence for energy creation. CSS positive-energy populations "
    "are distinct from leading clusters; stage differences are not causal. Conditional saved-pool draw "
    "SD is separate from block error. No nominal p-values, tuning, uncertainty closure, confirmation or "
    "promotion; finite-library/model/stock-seed uncertainty and inclusive acceptance remain open."
)


def _vector(data, key, n, kind=None, finite=False):
    value = np.asarray(data[key])
    if value.shape != (n,) or (kind and value.dtype.kind not in kind):
        raise ValueError("invalid saved vector: " + key)
    if finite and not np.all(np.isfinite(value)):
        raise ValueError("nonfinite saved vector: " + key)
    return value


def _close(name, actual, expected):
    if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, rtol=1e-12, atol=1e-10):
        raise ValueError("saved identity mismatch: " + name)


def _validate(a, d, old, c):
    ids = np.asarray(c["full_ids"])
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("full IDs must be unique and one-dimensional")
    full, n = len(ids), len(a["event_row"])
    rows = _vector(a, "event_row", n, "iu")
    if np.any(rows < 0) or np.any(rows >= full) or np.any(rows[1:] <= rows[:-1]):
        raise ValueError("invalid original query rows")
    front = np.zeros(full, bool)
    front[rows] = True
    for saved in (c, d, old):
        if np.asarray(saved["full_ids"]).dtype != ids.dtype:
            raise ValueError("full ID dtype mismatch")
        _vector(saved, "event_row", full, "iu")
        _vector(saved, "query_event_rows", n, "iu")
        equal("full IDs", saved["full_ids"], ids)
        equal("full rows", saved["event_row"], np.arange(full))
        equal("query rows", saved["query_event_rows"], rows)
    equal("entry rows", c["entry_event_row"], rows)
    for key, value in a.items():
        _vector(a, key, n)
    for key in ("event_row", "momentum", "direction_radial", "direction_tangent", "cell_id"):
        equal("production " + key, a["production_" + key], a[key])
    p = _vector(a, "momentum", n, finite=True)
    if np.any(p <= 0):
        raise ValueError("invalid saved momentum")
    energy = _vector(c, "entry_energy", n, finite=True)
    _close("saved momentum", p * p, energy * energy - .51099895 ** 2)
    cell = _vector(a, "cell_id", n, "iu")
    disk = _vector(c, "entry_disk", n, "iu")
    if np.any((cell < 0) | (cell >= 96)) or np.any(~np.isin(disk, (0, 1))):
        raise ValueError("invalid fixed cell/disk")
    indices = []
    for key, edges in (("energy", (0., 50., 80., 95., 110.)),
                       ("cosine", (0., .4, .7, 1.000001)),
                       ("radius", (0., 450., 550., 750., 1000.))):
        v = _vector(c, "entry_" + key, n, finite=True)
        if np.any((v < edges[0]) | (v > edges[-1])):
            raise ValueError("query outside fixed cells")
        indices.append(np.minimum(np.searchsorted(edges, v, side="right") - 1, len(edges) - 2))
    equal("fixed cell assignment", cell, np.ravel_multi_index((disk, *indices), (2, 4, 3, 4)))
    for coordinate in TAILS:
        _vector(a, "direction_" + coordinate, n, finite=True)
    used = _vector(a, "direction_used", n, "b")
    fallback = _vector(a, "direction_fallback", n, "b")
    supported = _vector(c, "sampling_supported", n, "b")
    if np.any(used & fallback) or not np.all(supported) or not np.array_equal(used | fallback, supported):
        raise ValueError("direction flags must partition all supported front queries")
    for flag in ("direction_used", "direction_fallback"):
        equal(flag, a[flag], _vector(d, flag, n, "b"))
        if "sampling_" + flag in c:
            equal(flag, a[flag], c["sampling_" + flag])
    if "sampling_direction_count" in c:
        count = _vector(c, "sampling_direction_count", n, "iu")
        if np.any(count < 0) or not np.array_equal(count >= 8, used):
            raise ValueError("saved capped support count disagrees with flags")
    for key in ("donor", "supported", "fallback"):
        _vector(c, "sampling_" + key, n, "iu" if key == "donor" else "b")
        equal("saved pool " + key, d["pool_" + key], c["sampling_" + key])
    for key in ("radius_pool_size", "bin_pool_size"):
        size = _vector(c, "sampling_" + key, n, "iu")
        if np.any(size < 1):
            raise ValueError("invalid supported pool size")
    if np.any(c["sampling_radius_pool_size"] > 8) or np.any(c["sampling_bin_pool_size"] < c["sampling_radius_pool_size"]):
        raise ValueError("invalid fixed K8 pool sizes")
    equal("fallback donor identity", d["pool_donor"][fallback], old["pool_donor"][fallback])
    for key in d:
        if key.startswith(("css_donor_centroid_", "pool_expected_", "pool_variance_", "pool_actual_")) or key == "residual_donor_steps":
            equal("fallback donor/pool invariant " + key, d[key][fallback], old[key][fallback])
    for key in d:
        if key.startswith(("css_original", "residual_original", "residual_entry_")):
            equal("original invariant " + key, d[key], old[key])
    for prefix, diagnostic in (("", d), ("production_", old)):
        for key in ("residual_event_row", "css_event_row", "pool_event_row"):
            equal(key, diagnostic[key], rows)
        equal("front", _vector(diagnostic, "front", full, "b"), front)
        for key in ("pool_supported", "pool_fallback"):
            _vector(diagnostic, key, n, "b")
        if np.any(_vector(diagnostic, "pool_donor", n, "iu") < 0):
            raise ValueError("invalid supported donor")
        for key in ("residual_donor_steps", "residual_original_steps"):
            if np.any(_vector(diagnostic, key, n, "iu") < 0):
                raise ValueError("invalid CSS step count")
        for key in diagnostic:
            if key.startswith(("residual_", "css_", "pool_")):
                _vector(diagnostic, key, n)
        equal("support invariant", diagnostic["pool_supported"], supported)
        for stage in ("original", "donor", *ARMS):
            e = _vector(diagnostic, f"css_{stage}_centroid_energy", n, finite=True)
            if np.any(e < 0):
                raise ValueError("negative CSS energy")
            for coordinate in TAILS:
                v = _vector(diagnostic, f"css_{stage}_centroid_{coordinate}", n)
                if not np.all(np.isfinite(v[e > 0])):
                    raise ValueError("nonfinite positive CSS coordinate")
        for arm in ("stockref", "control", *ARMS):
            base = prefix + arm + "_"
            e = _vector(a, base + "energy", n, finite=True)
            ep = _vector(a, base + "truth_ep", n, finite=True)
            if np.any(e < 0):
                raise ValueError("negative leading energy")
            _close(base + "Ep", ep, e / p)
            equal(base + "selected", _vector(diagnostic, f"residual_{arm}_selected", n, "b"), e > 10)
            for field in ("radial", "tangent", "time"):
                v = _vector(a, base + field, n)
                if not np.all(np.isfinite(v[e > 10])):
                    raise ValueError("nonfinite selected leading field")
            if arm in ("stockref", "control"):
                for field in ("energy", "truth_ep", "radial", "tangent", "time"):
                    equal("production invariant " + field, a["production_" + arm + "_" + field], a[arm + "_" + field])
        original, donor = (diagnostic["residual_" + k] for k in ("original", "donor"))
        accepted = diagnostic["residual_original_time_accepted"]
        if np.any(accepted > original + 1e-10):
            raise ValueError("original time accepted energy exceeds original")
        for key in ("original", "donor"):
            equal("CSS energy", diagnostic["residual_" + key], diagnostic["css_" + key])
            equal("CSS centroid energy", diagnostic["residual_" + key], diagnostic[f"css_{key}_centroid_energy"])
        for arm in ARMS:
            base = prefix + arm + "_"
            lost, rejected, kept = (diagnostic[f"residual_{arm}_" + k] for k in ("lost", "time_rejected", "time_accepted"))
            for v in (original, donor, accepted, lost, rejected, kept):
                if np.any(~np.isfinite(v) | (v < 0)):
                    raise ValueError("invalid CSS energy partition")
            _close("donor partition", lost + rejected + kept, donor)
            _close("mapped partition", rejected + kept, diagnostic[f"residual_{arm}_mapped"])
            equal("mapped centroid", diagnostic[f"residual_{arm}_mapped"], diagnostic[f"css_{arm}_centroid_energy"])
            control, stock = a["control_energy"], a["stockref_energy"]
            expected = (donor - original, -lost, -(rejected - (original - accepted)),
                        (a[base + "energy"] - kept) - (control - accepted), control - stock)
            for term, value in zip(TERMS, expected):
                _close(base + term, _vector(a, base + "term_" + term, n, finite=True), value)
            total = sum(a[base + "term_" + term] for term in TERMS)
            _close("energy telescoping", total, a[base + "energy"] - stock)
            _close("Ep telescoping", total / p, a[base + "truth_ep"] - a["stockref_truth_ep"])
        for outcome in ("energy", "empty", "weak"):
            expected = _vector(diagnostic, "pool_expected_" + outcome, n, finite=True)
            variance = _vector(diagnostic, "pool_variance_" + outcome, n, finite=True)
            actual = _vector(diagnostic, "pool_actual_" + outcome, n, finite=True)
            if np.any(expected < 0) or np.any(variance < 0):
                raise ValueError("negative pool moment")
            if outcome != "energy":
                if np.any(expected > 1) or np.any(~np.isin(actual, (0., 1.))):
                    raise ValueError("invalid pool probability/outcome")
                _close("Bernoulli variance", variance, expected * (1 - expected))
            observed = donor if outcome == "energy" else (diagnostic["residual_donor_steps"] == 0 if outcome == "empty" else donor <= 10)
            equal("actual pool outcome", actual, observed)
    for arm in ("stockref", "control", *ARMS):
        e = _vector(c, arm + "_energy", full, finite=True)
        if np.any(e < 0) or (arm in ARMS and np.any(e[~front] != 0)):
            raise ValueError("invalid full stored leading energy")
        equal("saved leading energy", e[rows], a[arm + "_energy"])
        for field in ("x", "y", "z", "time"):
            value = _vector(c, arm + "_" + field, full)
            if not np.all(np.isfinite(value[e > 0])):
                raise ValueError("nonfinite positive full stored leading field")
    for arm in ARMS:
        common = (a[arm + "_energy"] > 10) & (a["production_" + arm + "_energy"] > 10) & (a["stockref_energy"] > 10)
        equal("saved common selection", _vector(a, arm + "_common_three_way_selected", n, "b"), common)
    return rows


def _slots(values):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite inclusive histogram value")
    slot = np.searchsorted(EP_EDGES, values, side="right")
    return np.where(values == EP_EDGES[-1], len(EP_EDGES) - 1, slot)


def count_contributions(fast, stock, group, labels, size):
    """Additive flow-bin counts with ALL supplied original blocks, including absent-cell blocks."""
    unique, inverse = np.unique(group, return_inverse=True)
    labels = np.asarray(labels)
    if labels.shape != np.shape(fast) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= size)):
        raise ValueError("invalid disjoint count labels")
    nb, blocks = len(EP_EDGES) + 1, len(unique)
    def histogram(value):
        return np.bincount((labels * blocks + inverse) * nb + _slots(value), minlength=size * blocks * nb).reshape(size, blocks, nb)
    f, s = histogram(fast), histogram(stock)
    delta = f - s
    variance = blocks * np.var(delta, axis=1, ddof=1) if blocks > 1 else np.full((size, nb), np.nan)
    return dict(fast=f.sum(axis=1), stock=s.sum(axis=1), difference=delta.sum(axis=1),
                error=np.sqrt(variance), source_blocks=blocks)


def _regions(a, c):
    n = len(a["event_row"])
    result = {"all": np.ones(n, bool)}
    for disk in (0, 1):
        result[f"disk{disk}"] = c["entry_disk"] == disk
    for key, edges in (("radius", (0, 450, 550, 1000, np.inf)),
                       ("energy", (0, 50, 80, 95, 110, np.inf)),
                       ("cosine", (0, .4, .7, 1.000001))):
        result.update({f"{key}_{lo}_{hi}": (c["entry_" + key] >= lo) & (c["entry_" + key] < hi)
                       for lo, hi in zip(edges[:-1], edges[1:])})
    for name in ("direction_used", "direction_fallback"):
        result[name] = a[name]
    for name in TAILS:
        result[name + "_direction_negative"] = a["direction_" + name] < 0
        result[name + "_direction_nonnegative"] = a["direction_" + name] >= 0
    for disk in (0, 1):
        for label, mask in list(result.items()):
            if label.startswith("radius_"):
                result[f"disk{disk}_" + label] = result[f"disk{disk}"] & mask
    result.update({f"cell_{i:02d}": a["cell_id"] == i for i in range(96)})
    return result


def _ep_counts(a, c):
    group = a["event_row"] // 100
    partitions = dict(cell=(a["cell_id"], 96), disk=(c["entry_disk"].astype(int), 2),
                      radius=(np.searchsorted((450, 550, 1000), c["entry_radius"], side="right"), 4),
                      direction_flag=(a["direction_fallback"].astype(int), 2))
    prefixes = {"control": "control", **{label + "_" + arm: prefix + arm
                for label, prefix in (("new", ""), ("production", "production_")) for arm in ARMS}}
    stock = a["stockref_truth_ep"]
    result = dict(edges=EP_EDGES, flow_bins=True, bin_order="underflow, 24 regular bins (inclusive final edge), overflow",
                  partition_labels=dict(cell=np.arange(96), disk=[0, 1], radius=["[0,450)", "[450,550)", "[550,1000)", "[1000,inf)"],
                                        direction_flag=["direction_used", "direction_fallback"]), arms={})
    for label, prefix in prefixes.items():
        fast = a[prefix + "_truth_ep"]
        global_count = count_contributions(fast, stock, group, np.zeros(len(group), int), 1)
        global_count = {k: v[0] if isinstance(v, np.ndarray) else v for k, v in global_count.items()}
        parts = {key: count_contributions(fast, stock, group, ids, size) for key, (ids, size) in partitions.items()}
        for part in parts.values():
            for key in ("fast", "stock", "difference"):
                equal("additive count partition", part[key].sum(axis=0), global_count[key])
        nb = len(EP_EDGES) + 1
        migration = np.bincount(_slots(stock) * nb + _slots(fast), minlength=nb * nb).reshape(nb, nb)
        equal("migration stock marginal", migration.sum(axis=1), global_count["stock"])
        equal("migration arm marginal", migration.sum(axis=0), global_count["fast"])
        result["arms"][label] = dict(global_counts=global_count, partitions=parts, stock_to_arm=migration)
    return result


def _windows(a):
    ep = a["stockref_truth_ep"]
    masks = {f"{lo:.2f}_{hi:.2f}": (ep >= lo) & (ep < hi) for lo, hi in ((.8, .85), (.85, .9), (.9, .95), (.95, 1.))}
    masks.update(ge1=ep >= 1., gt1p2=ep > 1.2)
    result = {}
    for name, mask in masks.items():
        group, p = a["event_row"][mask] // 100, a["momentum"][mask]
        record = dict(entries=int(mask.sum()), source_blocks=len(np.unique(group)), arms={})
        for label, prefix in (("control", "control"), *((label + "_" + arm, prefix + arm)
                              for label, prefix in (("new", ""), ("production", "production_")) for arm in ARMS)):
            item = {key: paired_mean(a[prefix + "_" + key][mask], a["stockref_" + key][mask], group)
                    for key in ("energy", "truth_ep")}
            if label != "control":
                item["energy_terms"] = {term: paired_mean(a[prefix + "_term_" + term][mask], np.zeros(len(p)), group) for term in TERMS}
                item["ep_terms"] = {term: paired_mean(a[prefix + "_term_" + term][mask] / p, np.zeros(len(p)), group) for term in TERMS}
            record["arms"][label] = item
        result[name] = record
    return result


def _jackknife(deleted):
    b = len(deleted)
    if b < 2:
        return np.full(deleted.shape[1:], np.nan)
    return np.sqrt((b - 1) / b * np.sum((deleted - deleted.mean(axis=0)) ** 2, axis=0))


def _tails(y, mask, group, thresholds):
    unique, inverse = np.unique(group, return_inverse=True)
    b, t, n = len(unique), len(thresholds), int(mask.sum())
    counts = np.sum(np.abs(y[mask, None]) > thresholds, axis=0)
    selected = np.bincount(inverse[mask], minlength=b)
    hits = np.abs(y[mask, None]) > thresholds
    by_block = np.zeros((b, t), int)
    np.add.at(by_block, inverse[mask], hits)
    denominator = n - selected
    deleted = np.divide(counts - by_block, denominator[:, None], out=np.full((b, t), np.nan), where=denominator[:, None] > 0)
    fractions = counts / n if n else np.full(t, np.nan)
    return dict(n=n, counts=counts, fraction=fractions, error=_jackknife(deleted)), deleted


def _description(y, mask, split):
    value = y[mask]
    n = len(value)
    if not n:
        return dict(abs_quantiles=None, abs_max=None, variance_parts=dict(central_n=0, tail_n=0, central=None, tail=None, total=None))
    centered = (value - value.mean()) ** 2
    central = np.abs(value) <= split
    parts = dict(central_n=int(central.sum()), tail_n=int((~central).sum()), central=float(centered[central].sum() / n),
                 tail=float(centered[~central].sum() / n), total=float(centered.mean()))
    return dict(abs_quantiles=np.quantile(np.abs(value), (.5, .9, .95, .99)), abs_max=float(np.abs(value).max()), variance_parts=parts)


def _population(values, masks, x, group, coordinate, reference="stock", tails=True):
    pairs = [(key, reference) for key in values if key != reference] + [("new", "production")]
    result = dict(arms={}, contrasts={})
    for fast, stock in pairs:
        stats = moments(values[fast], values[stock], x, group, fast_mask=masks[fast], stock_mask=masks[stock])
        for key, side in ((fast, "fast"), (stock, "stock")):
            if key not in result["arms"]:
                result["arms"][key] = dict(moments=stats[side])
        result["contrasts"][fast + "_minus_" + stock] = dict(moments={k: stats[k] for k in ("difference", "error")})
    if tails:
        threshold = np.array(TAILS[coordinate])
        tail = {key: _tails(value, masks[key], group, threshold) for key, value in values.items()}
        for key, value in values.items():
            result["arms"][key].update(tails=tail[key][0], **_description(value, masks[key], threshold[1]))
        for fast, stock in pairs:
            result["contrasts"][fast + "_minus_" + stock]["tails"] = dict(
                difference=tail[fast][0]["fraction"] - tail[stock][0]["fraction"], error=_jackknife(tail[fast][1] - tail[stock][1]))
    return result


def _spatial(a, regions):
    result = {}
    for label, region in regions.items():
        if label.startswith(("energy_", "cosine_")):
            continue
        group = a["event_row"][region] // 100
        record = dict(entries=int(region.sum()), source_blocks=len(np.unique(group)), arms={})
        for arm in ARMS:
            prefixes = dict(new=arm, production="production_" + arm, stock="stockref", control="control")
            own = {key: a[prefix + "_energy"][region] > 10 for key, prefix in prefixes.items()}
            common = own["new"] & own["production"] & own["stock"]
            item = {}
            for coordinate in TAILS:
                values = {key: a[prefix + "_" + coordinate][region] for key, prefix in prefixes.items()}
                x = a["direction_" + coordinate][region]
                three = {key: value for key, value in values.items() if key != "control"}
                four = common & own["control"]
                item[coordinate] = dict(own=_population(values, own, x, group, coordinate),
                    common_three_way=_population(three, {key: common for key in three}, x, group, coordinate),
                    common_four_way=_population(values, {key: four for key in values}, x, group, coordinate),
                    control_unselected_in_three_way=int((common & ~own["control"]).sum()))
            record["arms"][arm] = item
        result[label] = record
    return result


def _stages(a, d, old, regions):
    result = {}
    for label, region in regions.items():
        if label.startswith(("cell_", "energy_", "cosine_", "disk0_radius", "disk1_radius")):
            continue
        group = a["event_row"][region] // 100
        record = dict(entries=int(region.sum()), source_blocks=len(np.unique(group)), stages={})
        for stage in ("donor", *ARMS):
            sources = dict(new=(d, stage), production=(old, stage), original=(d, "original"))
            own = {key: source[f"css_{name}_centroid_energy"][region] > 0 for key, (source, name) in sources.items()}
            common = own["new"] & own["production"] & own["original"]
            item = {}
            for coordinate in TAILS:
                values = {key: source[f"css_{name}_centroid_{coordinate}"][region] for key, (source, name) in sources.items()}
                x = a["direction_" + coordinate][region]
                item[coordinate] = dict(own=_population(values, own, x, group, coordinate, "original", False),
                    common_three_way=_population(values, {key: common for key in values}, x, group, coordinate, "original", False))
            record["stages"][stage] = item
        result[label] = record
    return result


def _fallback(a, d, old, c, regions):
    fallback = a["direction_fallback"]
    count = c.get("sampling_direction_count")
    records = {}
    for label, mask in regions.items():
        use = fallback & mask
        group = a["event_row"][use] // 100
        item = dict(entries=int(use.sum()), full_region_entries=int(mask.sum()), direction_used=int((mask & a["direction_used"]).sum()),
                    source_blocks=len(np.unique(group)), outcomes={}, pool={})
        for key, prefix in (("stock", "stockref"), ("control", "control"), *((label + "_" + arm, prefix + arm)
                           for label, prefix in (("new", ""), ("production", "production_")) for arm in ARMS)):
            e = a[prefix + "_energy"][use]
            item["outcomes"][key] = dict(zero=int((e == 0).sum()), weak_le10=int((e <= 10).sum()), selected=int((e > 10).sum()),
                                         energy=paired_mean(e, a["stockref_energy"][use], group),
                                         truth_ep=paired_mean(a[prefix + "_truth_ep"][use], a["stockref_truth_ep"][use], group))
        for label_arm, diagnostic in (("new", d), ("production", old)):
            outcomes = {}
            for outcome in ("energy", "empty", "weak"):
                actual = diagnostic["pool_actual_" + outcome][use]
                expected = diagnostic["pool_expected_" + outcome][use]
                variance = diagnostic["pool_variance_" + outcome][use]
                original = diagnostic["residual_original"][use]
                if outcome == "empty":
                    original = diagnostic["residual_original_steps"][use] == 0
                elif outcome == "weak":
                    original = original <= 10
                outcomes[outcome] = dict(actual_total=float(actual.sum()), expected_total=float(expected.sum()), original_total=float(original.sum()),
                    conditional_draw_sd_total=float(np.sqrt(variance.sum())),
                    conditional_draw_sd_mean=float(np.sqrt(variance.sum()) / len(actual)) if len(actual) else None,
                    actual_minus_expected=paired_mean(actual, expected, group), actual_minus_original=paired_mean(actual, original, group),
                    expected_minus_original=paired_mean(expected, original, group))
            item["pool"][label_arm] = outcomes
        item["capped_count_histogram_0_to_7"] = np.bincount(count[use], minlength=8) if count is not None else None
        records[label] = item
    columns = {"event_row": a["event_row"][fallback], "cell_id": a["cell_id"][fallback],
               "direction_count": count[fallback] if count is not None else None}
    columns.update({key: c["entry_" + key][fallback] for key in ("disk", "energy", "cosine", "radius")})
    columns.update({key: a[key][fallback] for key in ("direction_radial", "direction_tangent")})
    columns.update({key: c["sampling_" + key][fallback] for key in ("radius_pool_size", "bin_pool_size")})
    for key, value in a.items():
        if key.endswith(("_energy", "_truth_ep")):
            columns[key] = value[fallback]
    for prefix, diagnostic in (("", d), ("production_", old)):
        for key, value in diagnostic.items():
            if key.startswith(("pool_expected_", "pool_actual_", "pool_variance_")) or key in (
                    "residual_original", "residual_donor", "residual_original_steps", "residual_donor_steps"):
                columns[prefix + key] = value[fallback]
    return dict(entries=int(fallback.sum()), negative_tangent_queries=int((fallback & (a["direction_tangent"] < 0)).sum()),
                capped_counts_available=count is not None,
                capped_count_note="Exact saved capped support counts are optional; no pool enumeration or inference from outcomes.",
                regions=records, queries=columns)


def analyze(arrays, diagnostic, previous_diagnostic, clusters):
    """Return a JSON-able report without I/O or mutation; optional clusters sampling_direction_count is saved-only."""
    a, d, old, c = ({key: np.asarray(value) for key, value in source.items()}
                    for source in (arrays, diagnostic, previous_diagnostic, clusters))
    rows = _validate(a, d, old, c)
    regions = _regions(a, c)
    report = dict(schema="direction-localization-v1", scope=SCOPE, queries=len(rows), stored_events=len(c["event_row"]),
                  source_blocks=len(np.unique(rows // 100)), cell_coordinates=np.column_stack(np.unravel_index(np.arange(96), (2, 4, 3, 4))),
                  tail_thresholds_mm=TAILS, absolute_quantile_probabilities=[.5, .9, .95, .99],
                  variance_partition_cut_mm=dict(radial=200., tangent=100.), variance_partition_center="each arm's own selected mean",
                  stage_region_scope="all, disks, radius, signed direction and direction flags; no cell-stage cross product",
                  ep_counts=_ep_counts(a, c), stock_ep_windows=_windows(a), spatial=_spatial(a, regions),
                  css_stages=_stages(a, d, old, regions), fallback=_fallback(a, d, old, c, regions))
    return clean(report)
