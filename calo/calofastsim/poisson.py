"""All accepted saved-step/SiPM inputs for a no-travel PE-statistics control; no RNG."""
from __future__ import annotations

import numpy as np

from . import response
from .compression import lookup
from .identities import EVENT_FIELDS, keys


STEP_FIELDS = EVENT_FIELDS + ("product_id", "index")
RO_FIELDS = STEP_FIELDS + ("sipm",)


def ledger(events, steps, options):
    """Predict float32 means/times for every accepted CSS times both SiPMs, including zeros."""
    ei = lookup(keys(events, EVENT_FIELDS), keys(steps, EVENT_FIELDS))
    timing = {name: events[field][ei] for name, field in (
        ("pbtime", "pbtime"), ("mbtime", "mbtime"), ("event_length", "event_length"),
        ("time_from_protons_to_dr_marker", "marker"))}
    timing.update(onspill=events["spill"][ei] == 1,
                  **{name: options[name] for name in ("digitization_start", "digitization_end", "digitization_buffer")})
    folded, accepted = response.fold_step_time(steps["time"], **timing)
    selected = np.flatnonzero(accepted)
    energy = response.corrected_energy(
        steps["energy_g4"][accepted], steps["energy_birks"][accepted], steps["z"][accepted],
        events["crystal_length"][ei][accepted], steps["lru"][accepted],
        birks=options["birks"], longitudinal=options["longitudinal"])
    means = response.photoelectron_mean(energy, np.column_stack((steps["yield0"][accepted], steps["yield1"][accepted])))
    if np.any(~np.isfinite(means) | (means < 0) | (means > 2e9)):
        raise ValueError("finite nonnegative means at most 2e9 required for this bounded control")
    fields = {name: np.repeat(steps[name][accepted], 2) for name in STEP_FIELDS}
    sipm = (steps["sipm0"][accepted, None] + np.arange(2)).ravel()
    fields["sipm"] = sipm
    identity = keys(fields, RO_FIELDS)
    if len(np.unique(identity)) != len(identity):
        raise ValueError("duplicate accepted step/SiPM ledger identity")
    return dict(identity=identity, mean=means.ravel(), time=np.repeat(folded[accepted].astype(np.float32), 2),
                sipm=sipm, step_index=np.repeat(selected, 2), accepted=accepted)


def zero_fill(expected_keys, observed_keys, counts):
    """Join strictly positive persisted counts into a complete zero-initialized input ledger."""
    counts = np.asarray(counts)
    if (counts.ndim != 1 or len(counts) != len(observed_keys) or
            not np.issubdtype(counts.dtype, np.integer) or np.any((counts <= 0) | (counts > np.iinfo(np.int32).max))):
        raise ValueError("persisted readouts require positive int32-range counts")
    if len(np.unique(observed_keys)) != len(observed_keys):
        raise ValueError("duplicate stock CSS/SiPM readout")
    order = lookup(expected_keys, observed_keys)
    filled = np.zeros(len(expected_keys), np.int64)
    filled[order] = counts
    return filled, order
