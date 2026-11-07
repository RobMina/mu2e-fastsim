"""Historical EarlyEleBeamFlash observable from retained aggregated records.

This reducer neither reproduces the detector filter nor converts raw G4 steps
into StrawGasSteps. Inputs must already represent the retained early stream.
Generated resampler draws come from independent, complete SubRun counters.
"""
from __future__ import annotations

import numpy as np

from .flash_inclusive_check import EVENT, keys, join

BUDGET_MEV_PER_GENERATED_EVENT = 7.506758e-6
DOCUMENTED_POT_PER_GENERATED_EVENT = 11.536718606512062
SUBRUN = ("run", "subrun")
STEP = (*EVENT, "product", "index")


def _columns(table, identities, values=()):
    fields = (*identities, *values)
    if missing := set(fields) - table.keys():
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if any(np.ma.isMaskedArray(table[k]) for k in fields):
        raise ValueError("Masked columns are unsupported; missing records are not zeros")
    arrays = {k: np.asarray(table[k]) for k in fields}
    shape = arrays[fields[0]].shape
    if len(shape) != 1 or any(a.shape != shape for a in arrays.values()):
        raise ValueError("Expected equal-length one-dimensional columns")
    identity = keys(arrays, identities)
    join(identity, identity)
    return arrays, identity


def summarize(retained_events, straw_steps, generated_subruns):
    """Sum ALL retained StrawGasStep ionization using supplied generated counters.

    retained_events has unique run/subrun/event columns, including retained
    events with no tracker deposit. straw_steps has run/subrun/event/product/
    index and ionizing_mev, with unique step identities. generated_subruns has
    unique run/subrun and positive integer generated_events, including SubRuns
    with zero retained events. It must cover the full completed input selection,
    not just step-bearing or saved-event SubRuns. Caller completeness is required.

    Returns event totals in ledger order and scalar normalization diagnostics.
    There is no per-step threshold, time mask, photon weighting, selection
    efficiency correction, inferred uncertainty, or inferred missing event ID.
    """
    events, event_keys = _columns(retained_events, EVENT)
    steps, _ = _columns(straw_steps, STEP, ("ionizing_mev",))
    subruns, subrun_keys = _columns(generated_subruns, SUBRUN, ("generated_events",))
    counts = subruns["generated_events"]
    if counts.dtype.kind not in "iu" or np.any(counts <= 0) or not len(counts):
        raise ValueError("Generated SubRun counters must be positive integers")
    # Object reduction avoids overflowing a uint64 total before normalization.
    generated = int(np.sum(counts.astype(object)))
    if generated > 2**53:
        raise ValueError("Generated total exceeds exact float64 integer range")
    event_subrun = join(keys(events, SUBRUN), subrun_keys)
    if np.any(event_subrun < 0):
        raise ValueError("Retained event has no generated SubRun counter")
    retained_counts = np.bincount(event_subrun, minlength=len(counts))
    if np.any(retained_counts.astype(np.uint64) > counts.astype(np.uint64)):
        raise ValueError("Retained event count exceeds generated SubRun count")
    step_event = join(keys(steps, EVENT), event_keys)
    if np.any(step_event < 0):
        raise ValueError("StrawGasStep is outside the retained event ledger")
    energy = steps["ionizing_mev"]
    if energy.dtype.kind not in "iuf" or not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("Ionizing energy must be finite, nonnegative and numeric")
    energy = energy.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        per_event = np.bincount(step_event, weights=energy, minlength=len(event_keys))
        total = float(per_event.sum())
    if not np.isfinite(total):
        raise ValueError("Ionizing energy total is nonfinite")
    mean = total / generated
    return {
        "observable": "early_flash_tracker_ionization_per_generated_resampler_event",
        "generated_events": generated,
        "generated_subruns": len(counts),
        "retained_events": len(event_keys),
        "tracker_steps": len(energy),
        "tracker_ionizing_mev": total,
        "event_tracker_ionizing_mev": per_event,
        "mev_per_generated_event": mean,
        "mev_per_retained_event": total / len(event_keys) if len(event_keys) else None,
        "retained_fraction": len(event_keys) / generated,
        "budget_mev_per_generated_event": BUDGET_MEV_PER_GENERATED_EVENT,
        "central_within_budget": mean <= BUDGET_MEV_PER_GENERATED_EVENT,
        "uncertainty": None,
        "optimizer_feasibility": None,
    }
