"""Deterministic fixed-K8 support diagnostics; no draws, reconstruction or correction."""
from __future__ import annotations

import numpy as np

from . import coverage as coverage_validation
from . import placement_diagnostic as pd
from .compare import paired_mean
from .library import Binning, _radius_pool, cells


NEIGHBORS = 8
_FEATURES = ("energy", "cosine", "radius")
_SHAPE = (2, 4, 3, 4)
_CAVEAT = (
    "Exact uniform K8 expectations conditional on the fixed library and query population; no draws or reconstruction. "
    "Empty means no CSS steps, not zero Birks energy; weak means whole-event both-disk Birks <= 10 MeV. "
    "Unsupported queries are retained separately and all outcome comparisons use identical supported queries. "
    "Outside-bin coordinates are clipped and retain an explicit outside/fallback flag, matching the sampler; "
    "unlike coverage occupancy, outside donors and queries are NOT excluded from clipped-cell counts. "
    "Conditional Bernoulli SD = sqrt(sum p(1-p)) assumes independent uniform donor choices and excludes "
    "training, model, population and original-sample uncertainty. Source-row//100 paired jackknife errors "
    "are separate conditional diagnostics, not independent-production uncertainties. Donor exposure, eligible "
    "union and expected unique donors are support diagnostics, NOT effective N. Incident-truth radius, energy, "
    "cosine and azimuth-rotated direction mismatches are diagnostics, not measured-variable corrections. "
    "Old pilot and production candidate differ in selected population and runtime; differences cannot be "
    "causally attributed to library size alone. No closure, promotion, inclusive acceptance or independence claim."
)


def _numeric(value, shape, name, *, integer=False, nonnegative=False):
    a = coverage_validation._numeric(value, shape, name, integer=integer, nonnegative=nonnegative)
    if integer and a.size and np.any(a > np.iinfo(np.int64).max):
        raise ValueError("integer exceeds int64: " + name)
    return a.astype(np.int64 if integer else np.float64, copy=False)


def _entries(entries):
    if np.ndim(entries["energy"]) != 1:
        raise ValueError("entry energy must be a vector")
    n = len(entries["energy"])
    result = {}
    for name in ("energy", "cosine", "radius", "time", "disk", "event_row"):
        result[name] = _numeric(entries[name], (n,), "entry " + name,
                                integer=name in ("disk", "event_row"), nonnegative=name != "time")
    for name in ("position", "direction"):
        result[name] = _numeric(entries[name], (n, 3), "entry " + name)
    if np.any(result["disk"] > 1) or np.any(result["cosine"] > 1):
        raise ValueError("invalid entry disk or cosine")
    if not np.allclose(np.linalg.norm(result["direction"], axis=1), 1., rtol=0, atol=1e-6):
        raise ValueError("entry directions must be unit vectors")
    if len(np.unique(result["event_row"])) != n:
        raise ValueError("one entry per source event is required")
    return result


def _geometry(library):
    disk = _numeric(library["geometry_disk_disk"], (2,), "geometry disk", integer=True)
    if not np.array_equal(disk, [0, 1]):
        raise ValueError("sorted both-disk geometry [0, 1] is required")
    for stem in ("origin", "axis_z"):
        value = np.column_stack([_numeric(library[f"geometry_disk_{stem}_{a}"], (2,),
                                          "geometry " + stem + " " + a) for a in "xyz"])
        if stem == "axis_z" and not np.allclose(value, [0., 0., 1.], rtol=0, atol=1e-6):
            raise ValueError("azimuth rotation requires unit z-aligned disk axes")
    for stem in ("axis_x", "axis_y"):
        keys = [f"geometry_disk_{stem}_{a}" for a in "xyz"]
        if any(key in library for key in keys):
            if not all(key in library for key in keys):
                raise ValueError("incomplete geometry " + stem)
            axis = np.column_stack([_numeric(library[key], (2,), key) for key in keys])
            if not np.allclose(np.linalg.norm(axis, axis=1), 1., rtol=0, atol=1e-6):
                raise ValueError("geometry axes must be unit vectors")


def _validate_library(library):
    _entries({name: library["entry_" + name] for name in
              ("energy", "cosine", "radius", "time", "disk", "event_row", "position", "direction")})
    _geometry(library)
    n = len(library["entry_energy"])
    _numeric(library["offsets"], (n + 1,), "library offsets", integer=True, nonnegative=True)
    coverage_validation.frozen(library)


def enumerate_pools(library, entries):
    """Return the pool_census pool schema without requiring or fabricating a saved draw."""
    _validate_library(library)
    entries = _entries(entries)
    binning = Binning(*(library["bins_" + name] for name in _FEATURES))
    query, outside = cells(*(entries[name] for name in _FEATURES), binning)
    labels = np.column_stack((library["entry_disk"], library["entry_cell"])).astype(np.int64)
    occupied, inverse = np.unique(labels, axis=0, return_inverse=True)
    count = np.bincount(inverse, minlength=len(occupied))
    n = len(query)
    eligible = np.full((n, NEIGHBORS), -1, np.int64)
    chosen = np.full(n, -1, np.int64)
    size, full_size, exact_size = (np.zeros(n, np.int64) for _ in range(3))
    fallback = outside.copy()
    for disk in (0, 1):
        where = np.flatnonzero(entries["disk"] == disk)
        pool = np.flatnonzero(occupied[:, 0] == disk)
        if not len(pool):
            fallback[where] = True
            continue
        distance = np.sum((query[where, None, :] - occupied[pool, 1:][None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distance, axis=1)
        chosen[where] = pool[nearest]
        full_size[where] = count[chosen[where]]
        exact = distance[np.arange(len(where)), nearest] == 0
        exact_size[where] = np.where(exact, full_size[where], 0)
        fallback[where] |= ~exact
        for cell in np.unique(chosen[where]):
            target = where[chosen[where] == cell]
            rows = np.flatnonzero(inverse == cell)
            k = min(NEIGHBORS, len(rows))
            size[target] = k
            eligible[target, :k] = (rows if k == len(rows) else
                                    _radius_pool(rows, np.asarray(library["entry_radius"], float), entries["radius"][target], k))
    supported = chosen >= 0
    return dict(eligible=eligible, query_cell=np.column_stack((entries["disk"], query)), chosen_cell=chosen,
                occupied=occupied, library_cell=inverse, counts=count, size=size, full_size=full_size,
                exact_size=exact_size, supported=supported, fallback=fallback, outside=outside,
                substituted=supported & (exact_size == 0))


def _outcomes(row, energy, n):
    total = np.bincount(row, weights=energy, minlength=n)
    if not np.all(np.isfinite(total)):
        raise ValueError("nonfinite accumulated Birks energy")
    return dict(energy=total, empty=np.bincount(row, minlength=n) == 0, weak=total <= 10)


def _original(steps, n):
    if np.ndim(steps["entry"]) != 1:
        raise ValueError("CSS entry must be a vector")
    m = len(steps["entry"])
    row = _numeric(steps["entry"], (m,), "CSS entry", integer=True, nonnegative=True)
    if np.any(row >= n):
        raise ValueError("CSS entry outside query rows")
    energy = _numeric(steps["energy_birks"], (m,), "CSS energy_birks", nonnegative=True)
    for name in ("energy_g4", "time", "disk", "crystal", "position"):
        if name in steps:
            value = _numeric(steps[name], (m, 3) if name == "position" else (m,), "CSS " + name,
                             integer=name in ("disk", "crystal"), nonnegative=name not in ("time", "position"))
            if name == "disk" and np.any(value > 1):
                raise ValueError("invalid CSS disk")
    if "offsets" in steps:
        offsets = _numeric(steps["offsets"], (n + 1,), "CSS offsets", integer=True, nonnegative=True)
        if offsets[0] != 0 or offsets[-1] != m or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("invalid CSS offsets")
        if not np.array_equal(row, np.repeat(np.arange(n), np.diff(offsets))):
            raise ValueError("CSS offsets disagree with packed entry rows")
    return _outcomes(row, energy, n)


def _coverage(library, entries, pools):
    query, slot = np.nonzero(pools["eligible"] >= 0)
    donor = pools["eligible"][query, slot]
    result = {name: np.full(pools["eligible"].shape, np.nan) for name in
              ("radius_delta", "angle_deg", "energy_delta", "cosine_delta")}
    for name in _FEATURES:
        result[name + "_delta"][query, slot] = library["entry_" + name][donor] - entries[name][query]
    origin = np.column_stack([library["geometry_disk_origin_" + a] for a in "xy"])[entries["disk"][query]]
    source = library["entry_position"][donor, :2] - origin
    target = entries["position"][query, :2] - origin
    phi = np.arctan2(target[:, 1], target[:, 0]) - np.arctan2(source[:, 1], source[:, 0])
    direction = np.asarray(library["entry_direction"][donor], float).copy()
    x, y = direction[:, 0].copy(), direction[:, 1].copy()
    direction[:, 0] = np.cos(phi) * x - np.sin(phi) * y
    direction[:, 1] = np.sin(phi) * x + np.cos(phi) * y
    dot = np.sum(direction * entries["direction"][query], axis=1)
    result["angle_deg"][query, slot] = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))
    return result


def _probabilities(values, pools):
    valid = pools["eligible"] >= 0
    gathered = np.zeros(valid.shape)
    gathered[valid] = values[pools["eligible"][valid]]
    return np.divide(gathered.sum(axis=1), pools["size"], out=np.full(len(valid), np.nan),
                     where=pools["supported"])


def _distribution(values):
    value, count = np.unique(values, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(value, count)}


def _region(mask, entries, pools, outcomes, probability, original, cover):
    supported = mask & pools["supported"]
    ids = pools["eligible"][supported]
    valid = ids >= 0
    weights = np.broadcast_to(1. / pools["size"][supported, None], ids.shape)
    exposure = np.bincount(ids[valid], weights=weights[valid], minlength=len(outcomes["energy"]))
    with np.errstate(divide="ignore"):
        log_unused = np.bincount(ids[valid], weights=np.log1p(-weights[valid]), minlength=len(exposure))
    union = exposure > 0
    rows = entries["event_row"][supported]
    result = dict(queries=int(mask.sum()), supported=int(supported.sum()), unsupported=int((mask & ~pools["supported"]).sum()),
                  fallback=int((mask & pools["fallback"]).sum()), exact_no_fallback=int((mask & ~pools["fallback"]).sum()),
                  exact_cell_empty=int((mask & (pools["exact_size"] == 0)).sum()),
                  outside_bins=int((mask & pools["outside"]).sum()), substituted=int((mask & pools["substituted"]).sum()),
                  unique_eligible_donors=int(union.sum()), eligible_empty=int((union & outcomes["empty"]).sum()),
                  eligible_weak=int((union & outcomes["weak"]).sum()), expected_unique_drawn=float((-np.expm1(log_unused)).sum()),
                  maximum_expected_exposure=float(exposure.max(initial=0)), expected_exposure=pd.summarize(exposure[union]),
                  exposure_by_outcome={k: float(exposure[outcomes[k]].sum()) for k in ("empty", "weak")},
                  cell_occupancy=pd.summarize(pools["full_size"][supported]),
                  exact_cell_occupancy=pd.summarize(pools["exact_size"][mask]),
                  radius_pool_size=pd.summarize(pools["size"][supported]),
                  count_distributions={k: _distribution(pools[k][mask]) for k in ("size", "full_size", "exact_size")},
                  outcomes={}, coverage={})
    for kind in ("empty", "weak"):
        p, o = probability[kind][supported], original[kind][supported]
        result["outcomes"][kind] = dict(queries=len(rows), original=int(o.sum()),
            original_fraction=float(o.mean()) if len(o) else None, expected_count=float(p.sum()),
            expected_fraction=float(p.mean()) if len(p) else None,
            conditional_draw_sd=float(np.sqrt(np.sum(p * (1 - p)))), probability=pd.summarize(p),
            expected_minus_original=paired_mean(p, o, rows // 100),
            deterministic_queries=int(((p == 0) | (p == 1)).sum()),
            zero_probability_queries=int((p == 0).sum()), unit_probability_queries=int((p == 1).sum()))
    p, o = probability["energy"][supported], original["energy"][supported]
    result["outcomes"]["energy"] = dict(queries=len(rows), expected_mean=float(p.mean()) if len(p) else None,
        original_mean=float(o.mean()) if len(o) else None, expected_minus_original=paired_mean(p, o, rows // 100))
    for name, values in cover.items():
        value = values[supported]
        absolute = np.abs(value)
        best = np.min(np.where(valid, absolute, np.inf), axis=1)
        worst = np.max(np.where(valid, absolute, -np.inf), axis=1)
        mean = np.sum(np.where(valid, absolute, 0), axis=1) / pools["size"][supported]
        result["coverage"][name] = dict(best=pd.summarize(best), worst=pd.summarize(worst), mean=pd.summarize(mean),
                                          signed_pool_values=pd.summarize(value[valid]))
    return result, exposure


def _cell_census(library, entries, pools, outcomes, original):
    labels = np.indices(_SHAPE, dtype=np.int64).reshape(4, -1).T
    query = np.ravel_multi_index(pools["query_cell"].T, _SHAPE)
    donor = np.ravel_multi_index(np.column_stack((library["entry_disk"].astype(np.int64), library["entry_cell"].astype(np.int64))).T, _SHAPE)
    count = lambda rows, weights=None: np.bincount(rows, weights=weights, minlength=len(labels)).astype(np.int64)
    fields = dict(queries=count(query), donors=count(donor),
                  outside_queries=count(query, pools["outside"]),
                  outside_donors=count(donor, library["entry_outside_bins"]),
                  supported_queries=count(query, pools["supported"]),
                  unsupported_queries=count(query, ~pools["supported"]),
                  fallback_queries=count(query, pools["fallback"]),
                  substituted_queries=count(query, pools["substituted"]))
    for kind in ("empty", "weak"):
        fields["original_" + kind] = count(query, original[kind])
        fields["donor_" + kind] = count(donor, outcomes[kind])
    return [dict(cell=label.tolist(), **{k: int(v[i]) for k, v in fields.items()}) for i, label in enumerate(labels)]


def audit(library, entries, original_css):
    """Return (JSON-safe report, numeric arrays) for fixed K8 and complete supplied queries."""
    pools = enumerate_pools(library, entries)
    entries = _entries(entries)
    original = _original(original_css, len(entries["energy"]))
    lengths = np.diff(np.asarray(library["offsets"], dtype=np.int64))
    outcomes = _outcomes(np.repeat(np.arange(len(lengths)), lengths), library["step_energy_birks"], len(lengths))
    probability = {name: _probabilities(value, pools) for name, value in outcomes.items()}
    cover = _coverage(library, entries, pools)
    masks = pd.strata(entries)
    for disk in (0, 1):
        mask = (entries["disk"] == disk) & (entries["radius"] < 450) & (entries["energy"] >= 95)
        masks[f"inner_highE_disk{disk}"] = mask
        for label, lo, hi in (("coslow", 0, .7), ("coshigh", .7, 1.000001)):
            masks[f"inner_highE_disk{disk}_{label}"] = mask & (entries["cosine"] >= lo) & (entries["cosine"] < hi)
    masks.update(fallback=pools["fallback"], exact_no_fallback=~pools["fallback"])
    arrays = dict(pools)
    arrays.update(event_row=entries["event_row"].copy(),
                  **{"prob_" + k: v for k, v in probability.items()},
                  **{"original_" + k: v for k, v in original.items()},
                  **{"pool_" + k: v for k, v in cover.items()},
                  **{"outcome_" + k: v for k, v in outcomes.items()})
    regions = {}
    for name, mask in masks.items():
        regions[name], arrays["exposure_" + name] = _region(mask, entries, pools, outcomes, probability, original, cover)
    report = dict(neighbors=NEIGHBORS, queries=len(entries["energy"]), library_donors=len(lengths),
                  library_empty=int(outcomes["empty"].sum()), library_weak=int(outcomes["weak"].sum()),
                  regions=regions, cells=_cell_census(library, entries, pools, outcomes, original),
                  cell_count=int(np.prod(_SHAPE)), bins={k: library["bins_" + k].tolist() for k in _FEATURES},
                  cell_order=["disk", "energy", "cosine", "radius"],
                  coverage_metric="Best/worst/mean are per-query absolute mismatches; signed_pool_values retains signed deltas.",
                  bin_convention="Sampler parity: clipped coordinates, terminal edges included, outside explicitly flagged and retained.",
                  caveat=_CAVEAT)
    return report, arrays
