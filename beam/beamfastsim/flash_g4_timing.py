"""Conservative time brackets for already resolved spatial photon exits (ns).

The exporter has float64 step pre-times but no native post/crossing time;
SimParticle birth/end times are round-to-nearest float32. Collection indices
order a photon's steps within one event and step product. These assumptions,
resolved particle pointers, and the spatial extractor's exit/continuation flags
are required; positions are neither required nor reconstructed here. This is a
physically resolved subset, not export completeness or S2 certification.

The lower bound is the matched raw first-segment pre-time, never the NPZ time
or chord flight diagnostic. The upper bound is the next same-photon,
same-product pre-time AFTER the crossing segment (after the continuation for
two-segment exits), or the upper float32 rounding midpoint of a defined particle
end time when no such next step exists. Equal times are allowed; nonfinite or
decreasing pre-times invalidate their entire same-photon/product sequence.
Birth rounding intervals must be valid and not wholly after the first pre-time.
Only fallback bounds require a defined, finite, float32-compatible end time.
Closed rounding intervals conservatively include ties; even a zero-width
bracket is not a claim to have exported a native crossing timestamp.
"""
from __future__ import annotations

import numpy as np

from .flash_inclusive_check import EVENT, PRODUCT, keys, join

PARTICLE_FIELDS = (*PRODUCT, "key", "time", "end_time", "end_defined")
STEP_FIELDS = (*PRODUCT, "index", "particle_state", "particle_product", "particle_key", "time")
FIELDS = {"particles": PARTICLE_FIELDS, "steps": STEP_FIELDS}
EXTRA_FIELDS = {"particles": ("time", "end_time", "end_defined")}
PHOTON_FIELDS = (*PRODUCT, "key", "first_step_product", "first_stepindex",
                 "two_segment_reconstruction", "continuation_stepindex",
                 "exit_candidate", "exit", "ambiguous")
UPPER_INVALID, UPPER_NEXT_STEP, UPPER_END_TIME = 0, 1, 2


def float32_interval(values):
    """Return float64 closed rounding bounds; invalid elements become NaN.

    Accept float32 or float64 arrays only. A float64 value must be an EXACT
    widening of float32, not merely close to one. Other dtypes raise ValueError.
    The two adjacent-float32 midpoints differ at powers of two. At the largest
    finite magnitudes the virtual next value is +/-2**128, giving the finite
    round-to-infinity threshold rather than an infinite rounding interval.
    """
    values = np.asarray(values)
    if values.dtype.kind != "f" or values.dtype.itemsize not in (4, 8):
        raise ValueError("times must have floating dtype float32 or float64")
    with np.errstate(over="ignore", invalid="ignore"):
        stored = values.astype(np.float32)
        center = stored.astype(np.float64)
        previous = np.nextafter(stored, np.float32(-np.inf)).astype(np.float64)
        following = np.nextafter(stored, np.float32(np.inf)).astype(np.float64)
        previous = np.where(np.isneginf(previous), 2 * center - following, previous)
        following = np.where(np.isposinf(following), 2 * center - previous, following)
        lower = (previous + center) / 2
        upper = (center + following) / 2
    valid = np.isfinite(values) & np.isfinite(center) & (values == center)
    return np.where(valid, lower, np.nan), np.where(valid, upper, np.nan)


def _columns(table, fields, label):
    missing = set(fields) - table.keys()
    if missing:
        raise ValueError(f"{label}: missing required fields: {sorted(missing)}")
    first = np.asarray(table[fields[0]])
    if first.ndim != 1:
        raise ValueError(f"{label}.{fields[0]}: expected one-dimensional columns")
    size = len(first)
    for field in fields:
        if np.asarray(table[field]).shape != (size,):
            raise ValueError(f"{label}.{field}: expected one-dimensional equal-length columns")
    return {field: np.asarray(table[field]) for field in fields}


def _take(values, index, fill):
    out = np.full(len(index), fill, dtype=np.asarray(values).dtype)
    present = index >= 0
    out[present] = values[index[present]]
    return out


def crossing_intervals(tables, photons):
    """Return lower_ns, upper_ns, valid, upper_source arrays in photon order.

    Source codes are 0 invalid, 1 next step, 2 particle end-time midpoint.
    Missing/mismatched identities or inconsistent per-row evidence yield NaN
    bounds and source 0. Duplicate raw particle or step keys, missing columns,
    malformed columns and invalid dtypes raise ValueError rather than guessing.
    Full unsigned run/subrun/event/product/key joins preserve uint64 identities.
    Raw invalid next steps cannot be bypassed using an end-time fallback.
    """
    a = _columns(tables["particles"], PARTICLE_FIELDS, "particles")
    s = _columns(tables["steps"], STEP_FIELDS, "steps")
    p = _columns(photons, PHOTON_FIELDS, "photons")
    if s["time"].dtype.kind != "f":
        raise ValueError("steps.time must have floating dtype")
    birth_lower, _ = float32_interval(a["time"])
    end_lower, end_upper = float32_interval(a["end_time"])
    particle_keys = keys(a, (*PRODUCT, "key"))
    step_keys = keys(s, (*PRODUCT, "index"))
    join(particle_keys, particle_keys)
    join(step_keys, step_keys)
    particle = join(keys(p, (*PRODUCT, "key")), particle_keys)
    owner = join(keys(s, (*EVENT, "particle_product", "particle_key")), particle_keys)
    resolved = (owner >= 0) & (s["particle_state"] == 1)
    rows = np.flatnonzero(resolved)
    order = rows[np.lexsort((s["index"][rows], s["product"][rows], owner[rows]))]
    same = ((owner[order[1:]] == owner[order[:-1]])
            & (s["product"][order[1:]] == s["product"][order[:-1]]))
    following = np.full(len(owner), -1, np.int64)
    following[order[:-1][same]] = order[1:][same]
    good_sequence = np.zeros(len(owner), bool)
    if len(order):
        times = s["time"][order]
        bad = ~np.isfinite(times)
        bad[1:] |= same & (times[1:] < times[:-1])
        groups = np.cumsum(np.r_[True, ~same]) - 1
        group_bad = np.bincount(groups, weights=bad)
        good_sequence[order] = group_bad[groups] == 0
    query = {**p, "product": p["first_step_product"], "index": p["first_stepindex"]}
    first = join(keys(query, (*PRODUCT, "index")), step_keys)
    query["index"] = p["continuation_stepindex"]
    continuation = join(keys(query, (*PRODUCT, "index")), step_keys)
    two = p["two_segment_reconstruction"] == 1
    first_next = _take(following, first, -1)
    continuation_ok = ((continuation >= 0) & (continuation == first_next))
    anchor = np.where(two, continuation, first)
    next_step = _take(following, anchor, -1)
    lower = _take(s["time"].astype(np.float64), first, np.nan)
    upper = _take(s["time"].astype(np.float64), next_step, np.nan)
    birth = _take(birth_lower, particle, np.nan)
    fallback_upper = _take(end_upper, particle, np.nan)
    fallback_lower = _take(end_lower, particle, np.nan)
    fallback = next_step < 0
    upper = np.where(fallback, fallback_upper, upper)
    fallback_ok = ((_take(a["end_defined"], particle, 0) == 1)
                   & np.isfinite(fallback_lower) & (fallback_upper >= birth)
                   & (fallback_upper >= _take(s["time"], anchor, np.nan)))
    valid = ((particle >= 0) & (first >= 0)
             & (_take(owner, first, -1) == particle)
             & _take(good_sequence, first, False)
             & (p["exit_candidate"] == 1) & (p["exit"] == 1) & (p["ambiguous"] == 0)
             & ((p["two_segment_reconstruction"] == 0) | two)
             & (~two | continuation_ok) & (~fallback | fallback_ok)
             & np.isfinite(birth) & (lower >= birth)
             & np.isfinite(lower) & np.isfinite(upper) & (upper >= lower))
    source = np.where(valid, np.where(fallback, UPPER_END_TIME, UPPER_NEXT_STEP), UPPER_INVALID)
    return dict(lower_ns=np.where(valid, lower, np.nan),
                upper_ns=np.where(valid, upper, np.nan), valid=valid,
                upper_source=source.astype(np.uint8))
