"""Historical EarlyFlash detector OR on complete, already aggregated products.

This is not raw-step aggregation, an upstream status filter, or a source model.
Collection identities and order must describe the persisted detector products.
"""
from __future__ import annotations

import numpy as np

from . import flash_board_observable as observable
from .flash_inclusive_check import EVENT, PRODUCT, keys, join

TRACKER, CALO, CRV = 1, 2, 3
PARTICLE = (*PRODUCT, "particle_product", "particle_key")


def _nonnegative_integer(value, name):
    if value.dtype.kind not in "iu" or np.any(value < 0):
        raise ValueError(f"{name} must be nonnegative integers")


def _stored_float(value, name):
    """Require finite nonnegative accessor values losslessly stored as float32."""
    if value.dtype.kind not in "iuf" or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{name} must be finite nonnegative numeric values")
    if value.dtype.kind in "iu" and np.any(value > 2**53):
        raise ValueError(f"{name} integer exceeds exact float64 comparison range")
    with np.errstate(over="ignore", invalid="ignore"):
        stored = value.astype(np.float32)
    if not np.isfinite(stored).all() or np.any(stored.astype(np.float64) != value):
        raise ValueError(f"{name} must be losslessly representable as stored float32")
    return stored


def _steps(table, energy_name, role, products, product_keys, product_events):
    arrays, _ = observable._columns(
        table, observable.STEP, ("particle_product", "particle_key", energy_name, "momentum_mev_c"))
    keys(arrays, PARTICLE)
    product_index = join(keys(arrays, PRODUCT), product_keys)
    if np.any(product_index < 0):
        raise ValueError("Detector step has no declared product")
    if np.any(products["detector"][product_index] != role):
        raise ValueError("Detector step belongs to the wrong product role")
    if np.any(arrays["index"].astype(np.uint64) >= products["size"][product_index].astype(np.uint64)):
        raise ValueError("Detector step index is outside the declared product size")
    counts = np.bincount(product_index, minlength=len(product_keys))
    used = products["detector"] == role
    if np.any(counts[used].astype(np.uint64) != products["size"][used].astype(np.uint64)):
        raise ValueError("Detector product is incomplete: step count differs from declared size")
    energy = _stored_float(arrays[energy_name], energy_name)
    momentum = _stored_float(arrays["momentum_mev_c"], "momentum_mev_c").astype(np.float64)
    threshold = 0. if role == CALO else 1e-6
    good = (energy.astype(np.float64) > threshold) & (momentum > 0.) & (momentum < 1e6)
    return arrays, energy, good, product_events[product_index]


def _accepted_events(arrays, energy, good, event_index, n_events, role):
    accepted = np.zeros(n_events, dtype=bool)
    if not good.any():
        return accepted
    group_keys = keys(arrays, PARTICLE)[good]
    unique, first, group = np.unique(group_keys, return_index=True, return_inverse=True)
    if role == CALO:
        order = np.lexsort((arrays["index"][good], group))
        totals = np.zeros(len(unique), dtype=np.float32)
        # Unbuffered indexed adds preserve sequential float32 accumulation within each collection/particle.
        with np.errstate(over="ignore", invalid="ignore"):
            np.add.at(totals, group[order], energy[good][order])
        if not np.isfinite(totals).all():
            raise ValueError("Calorimeter particle sum is nonfinite")
        total = totals.astype(np.float64)
        passes = (total >= .01) & (total <= 1e6)
    else:
        counts = np.bincount(group, minlength=len(unique))
        if np.any(counts > np.iinfo(np.uint32).max):
            raise ValueError("Qualifying step count exceeds the historical unsigned counter range")
        passes = (counts >= 1) & (counts <= 1_000_000)
    accepted[event_index[good][first][passes]] = True
    return accepted


def select(events, products, straw_steps, calo_steps, crv_steps):
    """Select detector-OR events in ledger order without discarding small deposits.

    events has unique run/subrun/event. products has those keys plus product,
    detector (TRACKER/CALO/CRV) and size, including every empty input collection.
    Each event needs all three roles. Steps have event/product/index identities,
    particle_product/particle_key, momentum_mev_c and ionizing_mev (tracker),
    birks_mev (calo), or visible_mev (CRV). index is the collection position.
    Stored float32 accessors may be supplied as exact float64 promotions.
    Caller supplies complete historical aggregated products and resolved pointers.
    """
    event_columns, event_keys = observable._columns(events, EVENT)
    product_columns, product_keys = observable._columns(products, PRODUCT, ("detector", "size"))
    for name in ("detector", "size"):
        _nonnegative_integer(product_columns[name], name)
    if not np.isin(product_columns["detector"], (TRACKER, CALO, CRV)).all():
        raise ValueError("Unknown detector product role")
    product_events = join(keys(product_columns, EVENT), event_keys)
    if np.any(product_events < 0):
        raise ValueError("Detector product is outside the event ledger")
    masks, tables = {}, {}
    for role, name, table, energy_name in (
        (TRACKER, "tracker", straw_steps, "ionizing_mev"),
        (CALO, "calo", calo_steps, "birks_mev"),
        (CRV, "crv", crv_steps, "visible_mev"),
    ):
        present = np.bincount(product_events[product_columns["detector"] == role], minlength=len(event_keys))
        if np.any(present == 0):
            raise ValueError(f"Missing {name} product; absent is not an empty collection")
        arrays, energy, good, step_events = _steps(
            table, energy_name, role, product_columns, product_keys, product_events)
        masks[name] = _accepted_events(arrays, energy, good, step_events, len(event_keys), role)
        tables[name] = (arrays, step_events)
    retained = masks["tracker"] | masks["calo"] | masks["crv"]
    straw, straw_events = tables["tracker"]
    return {
        "retained": retained,
        "tracker": masks["tracker"], "calo": masks["calo"], "crv": masks["crv"],
        "retained_events": {k: v[retained] for k, v in event_columns.items()},
        "retained_straw_steps": {k: v[retained[straw_events]] for k, v in straw.items()},
    }


def summarize(events, products, straw_steps, calo_steps, crv_steps, generated_subruns):
    """Compose the historical detector OR and Q reducer with complete exposure.

    events is the full ledger reaching the detector filter after upstream gates,
    not an inferred generated universe. Counters also cover generated events that
    never reached this filter. Failed/truncated simulation remains an external
    validity concern; this function cannot certify upstream completion.
    """
    selected = select(events, products, straw_steps, calo_steps, crv_steps)
    empty = {k: v[:0] for k, v in selected["retained_straw_steps"].items()}
    observable.summarize(events, empty, generated_subruns)
    result = observable.summarize(selected["retained_events"], selected["retained_straw_steps"], generated_subruns)
    result["detector_selection"] = {k: selected[k] for k in ("retained", "tracker", "calo", "crv")}
    result["events_at_detector_filter"] = len(selected["retained"])
    return result
