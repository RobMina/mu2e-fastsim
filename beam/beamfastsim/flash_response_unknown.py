"""Saved-only evidence for UNKNOWN tracker attribution; never reclassify a step.

The origins archive is the compact NPZ from flash_response_origins, and response
is the ORIGINAL, unmerged foil_response.npz. No ROOT, geometry, or transport is
read. Full event/product/key identities are joined as unsigned integers. Saved
crossing brackets and exact float32 birth rounding bounds are compared strictly;
no flight-time estimate, tolerance, or rounded time is used.

Output consists of columnar all-event, unknown-step, observed-chain and candidate
tables. Candidate relations describe comparisons, NOT revised ownership. In
particular an excluded same-track candidate still triggers UNKNOWN in attribute.
The original response does not save raw tracker identities or the ancestry depth
limit. An original depth limit may be explicitly supplied from verified original
configuration, never guessed from the diagnostic trace or current code defaults.
Without it, ancestor rules are conditional, not proven original triggers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .flash_g4_timing import float32_interval
from .flash_inclusive_check import EVENT, PRODUCT, keys, join

UNKNOWN, FRONTIER = 2, 1
EXTRACTION_CHECKS = (
    "new_photons_without_volume", "photons_without_stage", "photons_after_destination_stage",
    "particles_without_event", "steps_without_event", "particles_without_product",
    "steps_without_product", "unresolved_photon_step_pointers",
)
PARTICLES = (*PRODUCT, "key", "pdg", "parent_state", "parent_product", "parent_key", "time")
TRACKER = (*PRODUCT, "index", "particle_state", "particle_product", "particle_key", "time",
           "total_edep", "nonionizing_edep", "ownership_status", "ionizing_mev")
PHOTONS = (*PRODUCT, "key", "exit", "birth_inside_geometry", "ambiguous",
           "recorded_invalid_segments", "crossing_lower_ns", "crossing_upper_ns",
           "crossing_valid", "frontier_status")
PROVENANCE = (
    "foil_index", "stage", "creation", "start_volume", "birth_volume_product", "birth_energy_mev",
    "parent_state", "parent_product", "parent_key", "recorded_steps", "exit_candidate",
    "first_step_product", "first_stepindex", "first_pre_time_ns", "first_end_process",
    "continuation_stepindex", "two_segment_reconstruction", "endpoint_ambiguous", "boundary_ambiguous",
    "first_segment_boundary_ambiguous", "first_segment_endpoint_ambiguous", "recorded_ambiguous_segments",
    "recorded_outside_start_segments", "crossing_upper_source", "covering_photon_row", "ancestry_depth",
    "frontier_own_invalid", "frontier_missing_parent", "frontier_uncertain_ancestor",
    "frontier_time_overlap", "frontier_depth_exhausted",
)


def _columns(source, fields, prefix=""):
    missing = [prefix + name for name in fields if prefix + name not in source]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = {name: np.asarray(source[prefix + name]) for name in fields}
    shape = out[fields[0]].shape
    if len(shape) != 1 or any(v.shape != shape for v in out.values()):
        raise ValueError(f"{prefix or 'table'} columns must be one-dimensional and equal length")
    return out


def _indices(value, size, label):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value >= size):
        raise ValueError(f"{label}: invalid row indices")
    return value.astype(np.int64)


def _json_columns(table):
    """Keep integer identities as JSON integers, never float-promote for nulls."""
    out = {}
    for name, values in table.items():
        values = np.asarray(values)
        if values.dtype.kind == "f":
            data = values.astype(object)
            data[~np.isfinite(values)] = None
            out[name] = data.tolist()
        else:
            out[name] = values.tolist()
    return out


def _optional(source, names, size, prefix=""):
    out = {}
    for name in names:
        if prefix + name in source:
            value = np.asarray(source[prefix + name])
            if value.shape != (size,):
                raise ValueError(f"{prefix + name}: malformed optional column")
            out[name] = value
    return out


def diagnose(origins, response, *, response_max_depth=None):
    """Validate saved alignment and return JSON-ready, columnar UNKNOWN evidence.

    No event/particle loop or ancestry rerun: sorted saved trace rows supply the
    immediate child at depth-1. A trace ending at a resolvable parent is reported
    as truncated/depth-limited; it is not invented into a complete ancestry.
    """
    if response_max_depth is not None:
        if (isinstance(response_max_depth, (bool, np.bool_))
                or not isinstance(response_max_depth, (int, np.integer)) or response_max_depth < 1):
            raise ValueError("response_max_depth must be a positive integer or None")
        response_max_depth = int(response_max_depth)
    a = _columns(origins, PARTICLES, "particle_")
    s = _columns(origins, TRACKER, "tracker_")
    p = _columns(response, PHOTONS)
    e = _columns(response, (*EVENT, "ordinal"), "event_")
    n, na, npht, ne = len(s["index"]), len(a["key"]), len(p["key"]), len(e["event"])
    ak, sk, pk, ek = (keys(a, (*PRODUCT, "key")), keys(s, (*PRODUCT, "index")),
                      keys(p, (*PRODUCT, "key")), keys(e, EVENT))
    for identity in (ak, sk, pk, ek):
        join(identity, identity)  # Every full identity must be unique, even with no UNKNOWN rows.
    keys(e, ("ordinal",))
    if len(np.unique(e["ordinal"])) != ne:
        raise ValueError("Duplicate event ordinal")
    if "input_event_count" in response and int(response["input_event_count"]) != ne:
        raise ValueError("input event count disagrees with full ledger")
    sr = np.asarray(response["tracker_step_row"])
    if sr.shape != (n,) or sr.dtype.kind not in "iu" or np.any(sr < 0):
        raise ValueError("original tracker row mapping is invalid")
    if np.any(sr[1:] <= sr[:-1]):
        raise ValueError("Duplicate or unordered original tracker rows")
    if "tracker_step_row" in origins and not np.array_equal(origins["tracker_step_row"], sr):
        raise ValueError("tracker row mapping disagrees with original response")
    for name in ("ownership_status", "ionizing_mev"):
        if not np.array_equal(s[name], response["tracker_" + name]):
            raise ValueError(f"tracker {name} disagrees with original response")
    status, energy = s["ownership_status"], s["ionizing_mev"]
    if status.dtype.kind not in "iu" or np.any(~np.isin(status, [0, 1, 2])):
        raise ValueError("invalid tracker ownership status")
    raw_energy = s["total_edep"].astype(np.float64) - s["nonionizing_edep"].astype(np.float64)
    if not np.array_equal(raw_energy, energy) or np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("tracker raw energy disagrees with saved ionization or is invalid")
    # Independent tracker identities are verified only when actually saved.
    identity_fields = (*PRODUCT, "index", "particle_state", "particle_product", "particle_key", "time")
    verified_identity_fields = []
    for name in identity_fields:
        field = "tracker_" + name
        if field in response:
            if not np.array_equal(s[name], response[field], equal_nan=True):
                raise ValueError(f"tracker identity {name} disagrees with original response")
            verified_identity_fields.append(name)
    se = join(keys(s, EVENT), ek)
    if np.any(se < 0) or np.any(join(keys(a, EVENT), ek) < 0) or np.any(join(keys(p, EVENT), ek) < 0):
        raise ValueError("tracker, particle or photon identity absent from all-event ledger")
    # Compact origins contains every particle in tracker-active events only.
    pe = join(keys(p, EVENT), ek)
    photon_particle = join(pk, ak)
    tracker_active = np.bincount(se, minlength=ne) > 0
    if np.any(tracker_active[pe] & (photon_particle < 0)):
        raise ValueError("response photon full identities absent from origins particles in tracker-active events")
    event_columns = dict(e)
    totals = {}
    for name, code in (("tracker", None), ("residual", 0), ("owned", 1), ("unknown", UNKNOWN)):
        use = np.ones(n, bool) if code is None else status == code
        sums = np.bincount(se[use], weights=energy[use], minlength=ne)
        saved = np.asarray(response["event_" + name + "_ionizing_mev"])
        if not np.array_equal(sums, saved):
            raise ValueError(f"event {name} energy disagrees with original response")
        event_columns[name + "_ionizing_mev"] = saved
        totals[name + "_ionizing_mev"] = float(saved.sum())
    event_columns.update(_optional(response, ("tracker_product_complete", "ownership_unknown",
        "recording_checks_passed", "exposure_checks_passed"), ne, "event_"))

    parent = join(keys(a, (*EVENT, "parent_product", "parent_key")), ak)
    leaf = join(keys(s, (*EVENT, "particle_product", "particle_key")), ak)
    particle_photon = join(ak, pk)
    if np.any((particle_photon >= 0) & (a["pdg"] != 22)):
        raise ValueError("saved foil photon identity is not a photon in origins")
    trace = _columns(origins, ("hit_row", "particle_row", "depth"), "trace_")
    h = _indices(trace["hit_row"], n, "trace_hit_row")
    ar = _indices(trace["particle_row"], na, "trace_particle_row")
    d = trace["depth"]
    if d.dtype.kind not in "iu" or np.any(d < 0) or np.any(d > np.iinfo(np.int64).max):
        raise ValueError("invalid trace depth")
    d = d.astype(np.int64)
    order = np.lexsort((d, h))
    h, ar, d = h[order], ar[order], d[order]
    head = np.r_[True, h[1:] != h[:-1]] if len(h) else np.zeros(0, bool)
    tail = np.r_[h[1:] != h[:-1], True] if len(h) else np.zeros(0, bool)
    continuation = np.flatnonzero(~head)
    if np.any(d[head] != 0) or np.any(d[continuation] != d[continuation - 1] + 1):
        raise ValueError("Duplicate or noncontiguous trace depth")
    if (np.any(leaf[h[head]] != ar[head]) or np.any(s["particle_state"][h[head]] != 1)
            or np.any(a["parent_state"][ar[continuation - 1]] != 1)
            or np.any(parent[ar[continuation - 1]] != ar[continuation])):
        raise ValueError("observed trace disagrees with full tracker/parent identities")
    trace_count = np.bincount(h, minlength=n)
    expected_trace = (leaf >= 0) & (s["particle_state"] == 1)
    if np.any((trace_count > 0) != expected_trace):
        raise ValueError("observed trace missing or present for inconsistent tracker pointer")
    last = np.full(n, -1, np.int64)
    last[h[tail]] = ar[tail]
    root = np.zeros(n, bool)
    missing = ~expected_trace
    truncated = np.zeros(n, bool)
    rows = np.flatnonzero(last >= 0)
    terminal = last[rows]
    root[rows] = a["parent_state"][terminal] == 0
    missing[rows] = (~root[rows] & ((a["parent_state"][terminal] != 1) | (parent[terminal] < 0)))
    truncated[rows] = ~root[rows] & ~missing[rows]

    # Observed chains are retained only for UNKNOWN steps; all input events remain above.
    u = np.flatnonzero(status == UNKNOWN)
    unknown_index = np.full(n, -1, np.int64)
    unknown_index[u] = np.arange(len(u))
    use = status[h] == UNKNOWN
    h, ar, d, head = h[use], ar[use], d[use], head[use]
    chain = dict(unknown_row=unknown_index[h], tracker_row=h, particle_row=ar, depth=d,
                 parent_particle_row=parent[ar], photon_row=particle_photon[ar])
    for name in (*PARTICLES, "creation", "stage", "start_volume", "energy", "x", "y", "z"):
        field = "particle_" + name
        if field in origins:
            value = _optional(origins, (name,), na, "particle_")[name]
            chain[name] = value[ar]
    birth_lo, birth_hi = float32_interval(a["time"])
    chain.update(birth_lower_ns=birth_lo[ar], birth_upper_ns=birth_hi[ar])

    found = particle_photon[ar] >= 0
    ci = np.flatnonzero(found)
    ch, ca, cd = h[ci], ar[ci], d[ci]
    photon = particle_photon[ca]
    same = cd == 0
    child = np.full(len(ci), -1, np.int64)
    child[~same] = ar[ci[~same] - 1]
    blo, bhi = np.full(len(ci), np.nan), np.full(len(ci), np.nan)
    blo[~same], bhi[~same] = birth_lo[child[~same]], birth_hi[child[~same]]
    lo, hi = p["crossing_lower_ns"][photon], p["crossing_upper_ns"][photon]
    finite_interval = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
    spatial = (p["birth_inside_geometry"][photon].astype(bool) & ~p["ambiguous"][photon].astype(bool)
               & (p["recorded_invalid_segments"][photon] == 0))
    usable = p["exit"][photon].astype(bool) & p["crossing_valid"][photon].astype(bool) & finite_interval & spatial
    step_time = s["time"][ch]
    comparison_lo = np.where(same, step_time, blo)
    comparison_hi = np.where(same, step_time, bhi)
    comparison_valid = np.isfinite(comparison_lo) & np.isfinite(comparison_hi)
    valid = usable & comparison_valid
    eligible = valid & (hi < comparison_lo)
    excluded = valid & (lo > comparison_hi)
    overlap = valid & ~(eligible | excluded)
    relation = np.full(len(ci), "invalid", dtype="U8")
    relation[eligible], relation[excluded], relation[overlap] = "eligible", "excluded", "overlap"
    comparison_unknown = np.where(same, ~eligible, ~(eligible | excluded))
    # A nonfinite tracker pre-time prevents the original candidate walk.
    walk_active = np.isfinite(step_time)
    # Ancestor depth d is compared on iteration d; same-track precedes the loop.
    visited = np.full(len(ci), None, object)
    visited[~walk_active] = False
    visited[walk_active & same] = True
    if response_max_depth is not None:
        visited[:] = walk_active & (cd <= response_max_depth)
    trigger = np.full(len(ci), None, object)
    trigger[~comparison_unknown | (visited == False)] = False
    trigger[comparison_unknown & (visited == True)] = True
    candidate_columns = dict(unknown_row=unknown_index[ch], tracker_row=ch, depth=cd,
        particle_row=ca, photon_row=photon, immediate_child_particle_row=child, same_track=same,
        step_pre_time_ns=step_time, child_birth_lower_ns=blo, child_birth_upper_ns=bhi,
        comparison_lower_ns=comparison_lo, comparison_upper_ns=comparison_hi,
        spatial_usable=spatial, interval_finite_ordered=finite_interval, candidate_usable=usable,
        comparison_valid=comparison_valid, relation=relation, original_walk_initially_active=walk_active,
        comparison_unknown_rule=comparison_unknown, original_candidate_visited=visited,
        triggers_original_unknown=trigger,
        conditional_original_unknown_rule=comparison_unknown & (visited == None))
    for name in PHOTONS:
        candidate_columns[name] = p[name][photon]
    provenance = _optional(response, PROVENANCE, npht)
    for name, values in provenance.items():
        candidate_columns[name] = values[photon]
    # Nullable child identities preserve all unsigned bits (no -1 uint64 sentinel).
    for name in (*PRODUCT, "key", "time"):
        value = np.empty(len(ci), object)
        value[:] = None
        value[~same] = a[name][child[~same]]
        if name == "time":
            bad_child_time = np.zeros(len(ci), bool)
            bad_child_time[~same] = ~np.isfinite(a[name][child[~same]])
            value[bad_child_time] = None
        candidate_columns["child_" + name] = value

    candidate_count = np.bincount(ch, minlength=n)
    candidate_trigger = np.bincount(ch[trigger == True], minlength=n) > 0
    candidate_conditional = np.bincount(ch[comparison_unknown & (visited == None)], minlength=n) > 0
    # Last eligible = rootmost *observed* eligible, not a newly assigned owner.
    rootmost = np.full(n, -1, np.int64)
    last_eligible = np.full(n, -1, np.int64)
    np.maximum.at(last_eligible, ch[eligible], np.flatnonzero(eligible))
    has_eligible = last_eligible >= 0
    rootmost[has_eligible] = photon[last_eligible[has_eligible]]
    frontier_blocked = np.zeros(n, bool)
    frontier_blocked[has_eligible] = p["frontier_status"][rootmost[has_eligible]] != FRONTIER
    report = json.loads(str(response["report_json"]))
    checks = {name: report.get(name) for name in EXTRACTION_CHECKS}
    extraction_ok = all(value == 0 for value in checks.values())
    nonfinite_step = ~np.isfinite(s["time"])
    initially_active = expected_trace & ~nonfinite_step
    observed_depth = trace_count - 1
    # Inspecting parent state at depth d requires iteration d+1.
    exhausted = np.full(n, None, object)
    exhausted[~initially_active] = False
    parent_inspected = np.full(n, None, object)
    parent_inspected[~initially_active] = False
    parent_inspected[initially_active & (observed_depth == 0)] = True
    if response_max_depth is not None:
        enough_walk = (observed_depth >= response_max_depth) | (
            truncated & (observed_depth + 1 >= response_max_depth))
        exhausted[initially_active & enough_walk] = True
        exhausted[initially_active & (root | missing) & (observed_depth < response_max_depth)] = False
        parent_inspected[initially_active] = observed_depth[initially_active] < response_max_depth
    else:
        exhausted[initially_active & (root | missing) & (observed_depth == 0)] = False
    pointer_trigger = np.full(n, None, object)
    pointer_trigger[~initially_active | root | (exhausted == True)] = False
    pointer_trigger[~expected_trace] = True
    parent_missing = missing & initially_active
    pointer_trigger[parent_missing] = parent_inspected[parent_missing]

    # The original frontier gate requires its walk's final eligible candidate.
    original_frontier = np.full(n, None, object)
    original_frontier[~initially_active] = False
    if response_max_depth is not None:
        covered = initially_active & ((observed_depth >= response_max_depth) |
                    ((root | missing) & (observed_depth < response_max_depth)))
        last_original_eligible = np.full(n, -1, np.int64)
        in_walk = eligible & (visited == True)
        np.maximum.at(last_original_eligible, ch[in_walk], np.flatnonzero(in_walk))
        original_frontier[covered] = False
        gate = covered & (last_original_eligible >= 0)
        original_frontier[gate] = p["frontier_status"][photon[last_original_eligible[gate]]] != FRONTIER
    verified_reason = (nonfinite_step | (pointer_trigger == True) | candidate_trigger |
                       (exhausted == True) | (original_frontier == True) | ~np.bool_(extraction_ok))
    observed_reason = (missing | truncated | nonfinite_step | candidate_trigger | candidate_conditional |
                       frontier_blocked | (exhausted == True) | ~np.bool_(extraction_ok))
    unknown_columns = dict(tracker_row=u, original_step_row=sr[u], event_row=se[u],
        event_ordinal=e["ordinal"][se[u]], original_status=status[u], ionizing_mev=energy[u],
        leaf_particle_row=leaf[u], trace_rows=trace_count[u], observed_max_depth=trace_count[u] - 1,
        last_particle_row=last[u], observed_root_reached=root[u], missing_pointer=missing[u],
        missing_leaf_pointer=(~expected_trace)[u], missing_parent_pointer=(missing & expected_trace)[u],
        global_extraction_unknown=np.full(len(u), not extraction_ok, bool),
        saved_trace_truncated_or_depth_exhausted=truncated[u], nonfinite_step_time=nonfinite_step[u],
        candidate_count=candidate_count[u], candidate_unknown_trigger=candidate_trigger[u],
        candidate_conditional_unknown_rule=candidate_conditional[u],
        original_missing_pointer_trigger=pointer_trigger[u], original_depth_exhausted=exhausted[u],
        original_frontier_unknown_trigger=original_frontier[u],
        original_unknown_cause_verified=verified_reason[u], unknown_without_verified_original_cause=~verified_reason[u],
        rootmost_observed_eligible_photon_row=rootmost[u], rootmost_observed_frontier_blocked=frontier_blocked[u],
        observed_unknown_evidence=observed_reason[u], unknown_without_observed_explanation=~observed_reason[u])
    for name in (*PRODUCT, "index", "particle_state", "particle_product", "particle_key", "time"):
        unknown_columns[name] = s[name][u]
    for name, values in _optional(origins, ("ancestry_unresolved", "ancestry_metadata_unknown",
            "foil_ancestry_complete", "has_new_foil_photon_ancestor", "has_new_foil_born_ancestor"), n).items():
        unknown_columns[name] = values[u]
    event_columns["unknown_tracker_steps"] = np.bincount(se[u], minlength=ne)
    return dict(schema="flash_response_unknown/v2", scope="Saved-only comparison evidence; no ownership changes or residual reassignment",
        input_events=ne, tracker_steps=n, unknown_steps=len(u), unknown_events=int(len(np.unique(se[u]))),
        **totals, energy_per_input_event_mev={name: value / ne if ne else None for name, value in totals.items()},
        status_codes={"residual": 0, "owned": 1, "unknown": UNKNOWN},
        relation_note="Strict bounds only. Excluded same-track candidates still trigger original UNKNOWN when the walk is active. Ancestor comparisons with unknown visitation are conditional. Rootmost means observed, not an owner assignment.",
        null_note="Null original-cause/visitation booleans mean unverified, not false. Null numeric evidence denotes nonfinite or not applicable; same_track and validity columns distinguish these cases. Integer keys are exact JSON integers; consumers must not parse them as float64.",
        validation=dict(exact_tracker_energy_status=True, exact_raw_ionization=True, exact_all_event_energy=True,
            unique_full_identities=True, trace_pointer_identities=True, tracker_active_response_photons_in_origins=True,
            independently_verified_tracker_fields=verified_identity_fields,
            independent_tracker_identity_complete=all(name in verified_identity_fields for name in identity_fields),
            limitation="Original response normally saves only tracker_step_row, energy and status, not raw tracker identities. Compact origins has no independent event ledger or raw row mapping. Equal-energy/status identity substitutions cannot be excluded without the original ROOT; none is read here."),
        response_depth_limit=response_max_depth,
        response_depth_limit_provenance=("user-supplied-original-config" if response_max_depth is not None else "unknown"),
        depth_note="Original response does not save max_depth; no default is inferred. Same-track is checked before the parent loop; an ancestor at depth d is compared iff d <= the original limit and the walk starts. Parent state at d needs iteration d+1. Diagnostic trace truncation is not proof of original exhaustion. With a supplied limit, successful saved parent edges (including the resolvable terminal edge) may prove exhaustion; a shorter incomplete trace leaves it unknown. Cycles can exhaust depth.",
        extraction_checks=checks, extraction_checks_passed=extraction_ok,
        available_candidate_provenance=list(provenance), events=_json_columns(event_columns),
        unknown_steps_table=_json_columns(unknown_columns), observed_chain=_json_columns(chain),
        candidates=_json_columns(candidate_columns))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origins", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--response-max-depth", type=int, default=None,
                        help="positive ancestry limit from verified ORIGINAL response configuration (no inferred default)")
    args = parser.parse_args(argv)
    if args.response_max_depth is not None and args.response_max_depth < 1:
        parser.error("--response-max-depth must be a positive integer")
    if args.out.resolve() in (args.origins.resolve(), args.response.resolve()):
        parser.error("output must not overwrite either input archive")
    start = perf_counter()
    with np.load(args.origins, allow_pickle=False) as archive:
        origins = {name: archive[name] for name in archive.files}
    with np.load(args.response, allow_pickle=False) as archive:
        response = {name: archive[name] for name in archive.files}
    result = diagnose(origins, response, response_max_depth=args.response_max_depth)
    result.update(origins=str(args.origins), response=str(args.response), elapsed_seconds=perf_counter() - start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"{result['unknown_steps']} UNKNOWN tracker steps in {result['unknown_events']}/{result['input_events']} input events; "
          f"{result['unknown_ionizing_mev']:.17g} MeV; {result['elapsed_seconds']:.3f} s/file. {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
