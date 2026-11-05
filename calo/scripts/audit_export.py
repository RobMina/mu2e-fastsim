"""Audit finished numeric library-export tables without certifying incidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from calofastsim.export_audit import audit
from audit_status import read_status_rows


FIELDS = {
    "events": ("run", "subrun", "event", "g4_status", "showers_present", "transport_particles_present",
               "generated_present", "unresolved_links", "n_particles", "n_generated", "n_showers", "n_steps"),
    "products": ("run", "subrun", "event", "product_id", "kind", "size", "present"),
    "particles": ("run", "subrun", "event", "sim_product_id", "sim_key", "parent_valid", "parent_product_id",
                  "parent_key", "generator_valid", "generator_product_id", "generator_key"),
    "genparticles": ("run", "subrun", "event", "product_id", "index"),
    "generated": ("run", "subrun", "event", "product_id", "index", "parent_valid", "parent_product_id", "parent_key"),
    "steps": ("run", "subrun", "event", "product_id", "index", "sim_valid", "sim_product_id", "sim_key"),
    "showers": ("run", "subrun", "event", "product_id", "index", "sim_valid", "sim_product_id", "sim_key"),
}


def reconcile_status(events, status_log, expected_events):
    printed = read_status_rows(status_log, expected_events)
    exported = np.column_stack([events[field] for field in ("run", "subrun", "event", "g4_status")])
    if not np.array_equal(exported, printed[:, :4]):
        raise ValueError("exported event identities/order/status do not match independent art PrintModule")


def check(path, *, directory, expected_events, expected_step_products, status_log=None):
    with uproot.open(path) as source:
        tables = {name: source[f"{directory}/{name}"].arrays(fields, library="np")
                  for name, fields in FIELDS.items()}
    result = audit(tables, expected_events=expected_events, expected_step_products=expected_step_products)
    if status_log is not None:
        reconcile_status(tables["events"], status_log, expected_events)
    result["independent_printmodule_event_id_order_status_verified"] = status_log is not None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--directory", default="dump")
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--expected-step-products", type=int, required=True)
    parser.add_argument("--status-log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.input, directory=args.directory, expected_events=args.expected_events,
                   expected_step_products=args.expected_step_products, status_log=args.status_log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
