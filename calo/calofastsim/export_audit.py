"""Vectorized numeric-export census; never infer incidence from deposited energy."""
from __future__ import annotations

import numpy as np


from .ancestry import forest
from .identities import EVENT_FIELDS, contains, keys


def audit(tables, *, expected_events, expected_step_products):
    """Reconcile events/registry/rows/pointers, reporting failed transport separately.

    Requires the draft exporter's exact numeric schema. Events must include all
    failed transport; absent shower products on status-zero events are errors.
    This does not establish boundary coverage, root assignment, unfiltered source
    selection or a complete independent incident manifest.
    """
    if (isinstance(expected_step_products, (bool, np.bool_))
            or not isinstance(expected_step_products, (int, np.integer)) or expected_step_products <= 0):
        raise ValueError("positive configured raw-step product count required")
    events = tables["events"]
    event_keys = keys(events, EVENT_FIELDS)
    if len(event_keys) != expected_events or len(np.unique(event_keys)) != expected_events:
        raise ValueError("event census is incomplete or duplicated")
    ordered_events = np.sort(event_keys)
    status = np.asarray(events["g4_status"])
    if status.shape != (expected_events,) or not np.issubdtype(status.dtype, np.integer) or np.any(status < 0):
        raise ValueError("invalid G4 status column")
    success = status == 0
    for field in ("showers_present", "transport_particles_present", "generated_present"):
        present = np.asarray(events[field])
        if present.shape != success.shape or present.dtype.kind != "b" or np.any(success & ~present):
            raise ValueError(f"missing required status-zero product: {field}")
    unresolved = np.asarray(events["unresolved_links"])
    if unresolved.shape != success.shape or np.any(unresolved[success] != 0):
        raise ValueError("unresolved identity in status-zero transport")
    for name in ("particles", "genparticles", "generated", "steps", "showers"):
        keys(tables[name], EVENT_FIELDS)
    registry = tables["products"]
    reg_keys = keys(registry, EVENT_FIELDS + ("product_id",))
    reg_events = keys(registry, EVENT_FIELDS)
    if not np.all(contains(ordered_events, reg_events)):
        raise ValueError("product registry has unknown events")
    reg_kind, reg_size = np.asarray(registry["kind"]), np.asarray(registry["size"])
    present = np.asarray(registry["present"])
    if any(array.shape != (len(reg_keys),) for array in (reg_kind, reg_size, present)) or present.dtype.kind != "b":
        raise ValueError("invalid registry column shapes/types")
    if (not np.issubdtype(reg_kind.dtype, np.integer) or not np.issubdtype(reg_size.dtype, np.integer)
            or np.any(reg_size < 0) or np.any(~present & (reg_size != 0))
            or np.any(present & (np.asarray(registry["product_id"]) == 0))):
        raise ValueError("invalid registry size/presence")
    if len(np.unique(reg_keys[present])) != np.count_nonzero(present):
        raise ValueError("duplicate present product identity")
    if np.any((reg_kind < -4) | (reg_kind >= expected_step_products)):
        raise ValueError("unexpected product registry kind")
    reg_event_indices = np.searchsorted(ordered_events, reg_events)
    good_registry = success[np.argsort(event_keys)][reg_event_indices]
    for kind in range(expected_step_products):
        selected = good_registry & (reg_kind == kind)
        counts = np.bincount(reg_event_indices[selected], minlength=expected_events)
        if not np.array_equal(counts, success[np.argsort(event_keys)].astype(np.int64)) or np.any(selected & ~present):
            raise ValueError("missing or duplicated configured raw-step product on status-zero event")
    for kind, field in ((-2, "generated_present"), (-3, "showers_present")):
        selected = (reg_kind == kind) & present
        counts = np.bincount(reg_event_indices[selected], minlength=expected_events)
        if not np.array_equal(counts, np.asarray(events[field])[np.argsort(event_keys)].astype(np.int64)):
            raise ValueError("product registry disagrees with event product presence")
    for name, kind, event_count, index_name in (("particles", -1, "n_particles", "sim_key"),
                                               ("generated", -2, "n_generated", "index"),
                                               ("showers", -3, "n_showers", "index"),
                                               ("steps", None, "n_steps", "index"),
                                               ("genparticles", -4, None, "index")):
        table = tables[name]
        event_rows = keys(table, EVENT_FIELDS)
        if not np.all(contains(ordered_events, event_rows)):
            raise ValueError(f"{name} has unknown event identities")
        event_indices = np.searchsorted(ordered_events, event_rows)
        if event_count is not None:
            wanted = np.asarray(events[event_count])[np.argsort(event_keys)]
            if not np.array_equal(np.bincount(event_indices, minlength=expected_events), wanted):
                raise ValueError(f"{name} row count disagrees with events")
        product_field = "sim_product_id" if name == "particles" else "product_id"
        row_keys = keys({**table, "product_id": table[product_field]}, EVENT_FIELDS + ("product_id",))
        unique, counts = np.unique(row_keys, return_counts=True)
        select = (reg_kind >= 0) if kind is None else (reg_kind == kind)
        selected_keys, selected_sizes = reg_keys[select & present], reg_size[select & present]
        order = np.argsort(selected_keys)
        selected_keys, selected_sizes = selected_keys[order], selected_sizes[order]
        if not np.all(contains(selected_keys, unique)):
            raise ValueError(f"{name} refers to an unregistered product")
        observed = np.zeros(len(selected_keys), dtype=np.int64)
        observed[np.searchsorted(selected_keys, unique)] = counts
        if not np.array_equal(observed, selected_sizes):
            raise ValueError(f"{name} row count disagrees with product registry")
        identities = keys({**table, "product_id": table[product_field]}, EVENT_FIELDS + ("product_id", index_name))
        if len(np.unique(identities)) != len(identities):
            raise ValueError(f"duplicate {name} row identity")
        if name != "particles":
            row_order = np.argsort(identities)
            ordered_products = row_keys[row_order]
            starts = np.r_[True, ordered_products[1:] != ordered_products[:-1]][:len(ordered_products)]
            sequence = np.arange(len(row_order))
            first = np.maximum.accumulate(np.where(starts, sequence, 0))
            if not np.array_equal(identities[index_name][row_order], sequence - first):
                raise ValueError(f"noncontiguous {name} row indices")
    particles = tables["particles"]
    sim_keys = np.sort(keys(particles, EVENT_FIELDS + ("sim_product_id", "sim_key")))
    for name, link in (("particles", "parent"), ("generated", "parent"), ("steps", "sim"), ("showers", "sim")):
        table = tables[name]
        valid = np.asarray(table[link + "_valid"])
        if valid.dtype.kind != "b":
            raise ValueError("identity presence flag must be boolean")
        aliases = {**table, "sim_product_id": table[link + "_product_id"], "sim_key": table[link + "_key"]}
        references = keys(aliases, EVENT_FIELDS + ("sim_product_id", "sim_key"))
        event_indices = np.searchsorted(ordered_events, keys(table, EVENT_FIELDS))
        ok_event = success[np.argsort(event_keys)][event_indices]
        if np.any(valid & ok_event & ~contains(sim_keys, references)):
            raise ValueError(f"unresolved {name} {link} link")
        if link == "sim" and np.any(ok_event & ~valid):
            raise ValueError(f"null {name} sim link")
    gen = tables["genparticles"]
    gen_keys = np.sort(keys({**gen, "generator_product_id": gen["product_id"], "generator_key": gen["index"]},
                           EVENT_FIELDS + ("generator_product_id", "generator_key")))
    gen_valid = np.asarray(particles["generator_valid"])
    gen_refs = keys(particles, EVENT_FIELDS + ("generator_product_id", "generator_key"))
    if gen_valid.dtype.kind != "b":
        raise ValueError("generator presence flag must be boolean")
    particle_events = np.searchsorted(ordered_events, keys(particles, EVENT_FIELDS))
    good = success[np.argsort(event_keys)][particle_events]
    if np.any(good & gen_valid & ~contains(gen_keys, gen_refs)):
        raise ValueError("unresolved generator identity")
    forest({field: np.asarray(particles[field])[good] for field in
            EVENT_FIELDS + ("sim_product_id", "sim_key", "parent_valid", "parent_product_id", "parent_key")})
    codes, counts = np.unique(status, return_counts=True)
    return {"events": expected_events, "status_counts": {str(int(k)): int(v) for k, v in zip(codes, counts)},
            "numeric_export_reconciled": True, "status_zero_parent_forest_acyclic": True,
            "complete_incident_census": False, "zero_response_certified": False,
            "scope": "event/product/row/SimParticle-link and status-zero parent-forest consistency only"}
