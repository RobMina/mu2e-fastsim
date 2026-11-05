"""Describe a complete numeric pilot export; never certify boundary incidence or zeros."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from calofastsim.ancestry import assign, forest
from calofastsim.identities import EVENT_FIELDS, contains, keys


SIM_FIELDS = EVENT_FIELDS + ("sim_product_id", "sim_key")
BOUNDARIES = np.array([73, 74, 75, 76, 77, 78, 79, 80, 102, 103, 104, 105, 106, 107, 108, 109])


def histogram(values):
    unique, counts = np.unique(values, return_counts=True)
    return {str(value): int(count) for value, count in zip(unique, counts)}


def columns(source, name, fields):
    return source[f"dump/{name}"].arrays(fields, library="np")


def boundary_gap_report(events, particles, roots, steps, showers, boundary, direct):
    """Keep shower-positive boundary gaps as columnar diagnostics, never discard them."""
    event_ids = keys(events, EVENT_FIELDS)
    gap = (events["n_showers"] > 0) & ~contains(np.unique(keys(steps, EVENT_FIELDS)[direct]), event_ids)
    gap_ids = event_ids[gap]
    selected = contains(gap_ids[np.argsort(gap_ids)], keys(particles, EVENT_FIELDS))
    subset = {name: value[selected] for name, value in particles.items()}
    parent = forest(subset)
    identities = keys(subset, SIM_FIELDS)
    is_root = contains(np.sort(keys(roots, SIM_FIELDS)), identities)
    label = is_root.copy()
    parent[is_root] = np.flatnonzero(is_root)
    for _ in range(len(identities).bit_length() + 1):
        label |= label[parent]
        parent = parent[parent]
    selected_boundary = boundary & contains(np.sort(gap_ids), keys(steps, EVENT_FIELDS))
    references = keys(steps, SIM_FIELDS)[selected_boundary]
    order = np.argsort(identities)
    if not np.all(contains(identities[order], references)):
        raise ValueError("boundary gap has unresolved particle identity")
    slots = order[np.searchsorted(identities[order], references)]
    shower_refs = keys(showers, SIM_FIELDS)[contains(np.sort(gap_ids), keys(showers, EVENT_FIELDS))]
    referenced = contains(np.unique(shower_refs), identities)
    referenced_boundary = contains(np.unique(references), identities[referenced])
    states = {name: value[is_root].tolist() for name, value in subset.items()}
    shower_states = {name: value[referenced].tolist() for name, value in subset.items()}
    shower_states["has_exact_boundary_row"] = referenced_boundary.tolist()
    boundary_rows = {name: value[selected_boundary].tolist() for name, value in steps.items()}
    boundary_rows["particle_pdg"] = subset["pdg"][slots].tolist()
    boundary_rows["descends_from_generated_root"] = label[slots].tolist()
    return {"event_columns": {name: value[gap].tolist() for name, value in events.items()},
            "root_particle_columns": states, "shower_ancestor_particle_columns": shower_states,
            "boundary_step_columns": boundary_rows,
            "summary": {"events": len(gap_ids), "boundary_rows": len(references),
                        "boundary_particle_pdg_counts": histogram(subset["pdg"][slots]),
                        "all_boundary_rows_descend_from_generated_root": bool(np.all(label[slots])),
                        "shower_ancestor_particles": int(np.count_nonzero(referenced)),
                        "shower_ancestor_pdg_counts": histogram(subset["pdg"][referenced]),
                        "shower_ancestors_with_exact_boundary_row": int(np.count_nonzero(referenced_boundary)),
                        "events_without_any_boundary_row": int(np.count_nonzero(~contains(
                            np.unique(keys(steps, EVENT_FIELDS)[selected_boundary]), gap_ids)))},
            "scope": "shower-positive events lacking direct root VD rows; not an incident or zero certificate"}


def census(path):
    with uproot.open(path) as source:
        events = columns(source, "events", EVENT_FIELDS + ("g4_status", "n_showers", "n_steps"))
        registry = columns(source, "products", EVENT_FIELDS + ("kind", "product_id", "size"))
        branch_names = ak.to_numpy(source["dump/products"]["branch_name"].array(library="ak"))
        particles = columns(source, "particles", SIM_FIELDS + ("parent_valid", "parent_product_id", "parent_key", "pdg", "creation",
                                                               "start_x", "start_y", "start_z", "start_px", "start_py", "start_pz", "start_e", "start_time",
                                                               "end_x", "end_y", "end_z", "end_px", "end_py", "end_pz", "end_e", "end_time", "stopping"))
        generated = columns(source, "generated", EVENT_FIELDS + ("product_id", "index", "pdg", "creation", "parent_valid", "parent_product_id", "parent_key",
                                                                "position_x", "position_y", "position_z", "momentum_px", "momentum_py", "momentum_pz", "momentum_e", "time"))
        steps = columns(source, "steps", SIM_FIELDS + ("kind", "volume", "time", "position_x", "position_y", "position_z",
                                                       "momentum_x", "momentum_y", "momentum_z"))
        showers = columns(source, "showers", SIM_FIELDS + ("crystal", "energy_g4", "energy_birks"))
    return summarize(events, registry, branch_names, particles, generated, steps, showers)


def summarize(events, registry, branch_names, particles, generated, steps, showers):
    """Census an already reconciled status-zero export without an incidence claim."""
    if np.any(events["g4_status"] != 0):
        raise ValueError("this diagnostic requires a separately audited status-zero file")
    event_keys = keys(events, EVENT_FIELDS)
    generated_events = keys(generated, EVENT_FIELDS)
    if (len(np.unique(event_keys)) != len(event_keys) or len(generated_events) != len(event_keys)
            or not np.array_equal(np.sort(event_keys), np.sort(generated_events))
            or np.any(generated["pdg"] != 11) or np.any(~generated["parent_valid"])):
        raise ValueError("expected exactly one parent-linked generated CE electron per event")
    transport = (registry["kind"] == -1) & np.char.endswith(branch_names.astype(str), "_g4run__CaloLibraryCE.")
    transport_ids = keys({**registry, "sim_product_id": registry["product_id"]}, EVENT_FIELDS + ("sim_product_id",))[transport]
    if len(transport_ids) != len(event_keys):
        raise ValueError("expected one raw g4run SimParticle product per event")
    particle_products = keys(particles, EVENT_FIELDS + ("sim_product_id",))
    root_candidate = np.isin(particle_products, transport_ids) & particles["parent_valid"]
    parent_fields = EVENT_FIELDS + ("parent_product_id", "parent_key")
    registry_events = keys(transport_ids, EVENT_FIELDS)
    if not np.array_equal(np.sort(registry_events), np.sort(event_keys)):
        raise ValueError("transport registry event population does not match events")
    registry_order = np.argsort(registry_events)
    gen_slot = registry_order[np.searchsorted(registry_events[registry_order], keys(generated, EVENT_FIELDS))]
    mapped_parent = {**generated, "parent_product_id": transport_ids["sim_product_id"][gen_slot]}
    source_parents = keys({**generated, "sim_product_id": generated["parent_product_id"],
                          "sim_key": generated["parent_key"]}, SIM_FIELDS)
    mapped_parents = keys({**mapped_parent, "sim_product_id": mapped_parent["parent_product_id"],
                          "sim_key": mapped_parent["parent_key"]}, SIM_FIELDS)
    if (not np.all(np.isin(source_parents, keys(particles, SIM_FIELDS)))
            or not np.all(np.isin(mapped_parents, keys(particles, SIM_FIELDS)))):
        raise ValueError("missing original or transport-reseated StageParticle parent")
    root_candidate &= (particles["pdg"] == 11) & np.isin(keys(particles, parent_fields), keys(mapped_parent, parent_fields))
    roots = {field: particles[field][root_candidate] for field in SIM_FIELDS}
    root_events = keys(roots, EVENT_FIELDS)
    if not np.array_equal(np.sort(root_events), np.sort(event_keys)):
        raise ValueError("generated-parent identity does not select one unique transport root per event")
    gen_order = np.argsort(generated_events)
    aligned_gen = gen_order[np.searchsorted(generated_events[gen_order], root_events)]
    if not np.array_equal(particles["creation"][root_candidate], generated["creation"][aligned_gen]):
        raise ValueError("transport root creation code does not match StageParticle")
    state_delta = {}
    for target, source_name in (("start_x", "position_x"), ("start_y", "position_y"), ("start_z", "position_z"),
                                ("start_px", "momentum_px"), ("start_py", "momentum_py"), ("start_pz", "momentum_pz"),
                                ("start_e", "momentum_e"), ("start_time", "time")):
        state_delta[target] = float(np.max(np.abs(particles[target][root_candidate] - generated[source_name][aligned_gen]), initial=0))
    incidents = {**roots, "product_id": roots["sim_product_id"], "particle_key": roots["sim_key"]}
    root_keys = keys(roots, SIM_FIELDS)
    boundary = (steps["kind"] == 0) & np.isin(steps["volume"], BOUNDARIES)
    direct = boundary & np.isin(keys(steps, SIM_FIELDS), root_keys)
    boundary_keys = keys(steps, EVENT_FIELDS)[boundary]
    direct_keys = keys(steps, EVENT_FIELDS)[direct]
    assign(particles, incidents, showers)
    calo_steps = {name: values[steps["kind"] == 1] for name, values in steps.items()}
    assign(particles, incidents, calo_steps)
    energy = np.zeros(len(event_keys), dtype=np.float64)
    order = np.argsort(event_keys)
    slot = order[np.searchsorted(event_keys[order], keys(showers, EVENT_FIELDS))]
    np.add.at(energy, slot, showers["energy_g4"])
    return {
        "events": len(event_keys), "status_counts": histogram(events["g4_status"]),
        "table_rows": {"particles": len(particles["event"]), "generated": len(generated_events),
                       "steps": len(steps["event"]), "showers": len(showers["event"])},
        "transport_roots_by_generated_parent_identity": len(root_keys),
        "transport_root_minus_generated_max_abs_state_difference": state_delta,
        "parent_reseating_policy": "Mu2eG4 StageParticle input parent is copied into g4run with same key, then child links target g4run ProductID",
        "all_shower_and_raw_crystal_step_identities_descend_from_generated_root": True,
        "events_with_any_boundary_row": len(np.unique(boundary_keys)),
        "events_with_direct_generated_root_boundary_rows": len(np.unique(direct_keys)),
        "direct_generated_root_boundary_rows": int(np.count_nonzero(direct)),
        "direct_root_boundary_vd_counts": histogram(steps["volume"][direct]),
        "empty_shower_collection_events": int(np.count_nonzero(events["n_showers"] == 0)),
        "empty_shower_events_with_direct_root_boundary_rows": int(np.count_nonzero(
            (events["n_showers"] == 0) & np.isin(event_keys, direct_keys))),
        "shower_events_without_direct_root_boundary_rows": int(np.count_nonzero(
            (events["n_showers"] > 0) & ~np.isin(event_keys, direct_keys))),
        "shower_positive_boundary_gaps": boundary_gap_report(events, particles, roots, steps, showers, boundary, direct),
        "shower_rows_by_crystal": histogram(showers["crystal"]),
        "event_total_shower_energy_g4_quantiles": np.quantile(energy, [0, .5, .9, .99, 1]).tolist(),
        "product_branch_names": np.unique(branch_names).tolist(),
        "certified_unbiased": False, "complete_incident_census": False, "zero_response_certified": False,
        "warning": "VD rows are not deduplicated incoming exterior crossings; empty events are not certified incident zeros",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = census(args.input)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: value for name, value in result.items() if name != "shower_rows_by_crystal"}, indent=2))


if __name__ == "__main__":
    main()
