"""Independent analytic expectations for the deterministic art extraction fixture.

This validates exported fixture values, not an arbitrary art input or transported
sample. Run only after the user-built fixture and read-only exporter jobs succeed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .flash_reference_export import audit_export, read_export


def check_fixture(path, *, directory="export", fail_status=False, unavailable_ptr=False):
    """Check counts, every step field, particle precision and ordered ancestry."""
    export = read_export(path, directory)
    audit = audit_export(export, expected_events=4, photon_role="fixture:photon", photon_cap=1000000,
                         particle_tag="fixture:continued:flashReferenceFixture", particle_cap=1000000)
    failures = []

    def check(ok, code):
        if not bool(ok):
            failures.append(code)

    def bits(actual, expected, code):
        expected = np.asarray(expected, dtype=actual.dtype)
        expected = np.broadcast_to(expected, actual.shape)
        check(actual.tobytes() == expected.tobytes(), code)

    ev, products = export["events"], export["products"]
    check(audit["table_structure_valid"], "table_structure")
    expected_physics = not fail_status and not unavailable_ptr
    check(audit["physics_checks_passed"] == expected_physics, "physics_expected_result")
    expected_failures = ({"status_status", "status_killed_field"} if fail_status else set())
    if unavailable_ptr:
        expected_failures.add("steps_particle_unavailable")
    actual_failures = {f["code"] for f in audit["failures"] if f["category"] == "physics"}
    check(actual_failures == expected_failures, "physics_failure_codes")
    check(len(ev["run"]) == 4, "event_count")
    check(len(np.unique(np.stack([ev["run"], ev["subrun"]], axis=1), axis=0)) == 2, "subrun_count")
    check(np.count_nonzero(ev["nsteps"] == 0) == 2, "two_empty_events")
    nonempty = (ev["event"] % 2) == 1
    check(np.all(ev["nsteps"][nonempty] == 4), "nonempty_step_count")
    check(np.all(ev["nparticles"][nonempty] == 5), "nonempty_particle_count")
    check(np.all(ev["ndaughters"][nonempty] == 3), "daughter_count")
    source_x = 1.0 + 2.0**-40
    source_t = 10331490143436802.
    energy = np.sqrt(16.078125 + .51099895**2)

    def pid(instance, kind):
        selected = (products["label"] == "fixture") & (products["instance"] == instance) & (products["kind"] == kind)
        values = np.unique(products["product"][selected])
        check(len(values) == 1, f"identity_{kind}_{instance}")
        return values[0] if len(values) == 1 else np.uint64(0)

    input_pid, continued_pid, gen_pid = pid("input", 1), pid("continued", 1), pid("", 2)
    for role in ("source", "photon", "tracker"):
        s = {name: a[export["steps"]["product"] == pid(role, 3)] for name, a in export["steps"].items()}
        count = 4 if role == "source" else 2
        check(len(s["index"]) == count, role + "_rows")
        values = {
            "volume": 123 if role == "tracker" else 9,
            "x": source_x, "y": -0., "z": 4179.,
            "post_x": source_x, "post_y": 0., "post_z": 4179. if role == "tracker" else 4179.5,
            "px": .125, "py": -.25, "pz": 4., "post_px": .125, "post_py": -.25, "post_pz": 4.,
            "time": source_t, "proper_time": .25 if role == "tracker" else .125,
            "total_edep": .25 if role == "tracker" else 0.,
            "nonionizing_edep": .125 if role == "tracker" else 0.,
            "visible_edep": .0625 if role == "tracker" else 0., "length": .5,
            "end_process": 17 if role == "tracker" else 49,
            "particle_product": input_pid if role == "source" else continued_pid,
            "particle_key": 17 if role == "source" else 101, "particle_state": 1,
        }
        if role == "source":
            values["time"] = source_t + 2 * s["index"]
        if role == "tracker":
            values["post_y"] = -0.
            if unavailable_ptr:
                values["particle_key"] = 999
                check(np.all(np.isin(s["particle_state"], (2, 3))), "unavailable_ptr_state")
                values.pop("particle_state")
        if role == "photon":
            values.update(post_x=source_x + .25, post_y=-.5, post_z=4181., px=0., py=0., pz=.5,
                          post_px=0., post_py=.25, post_pz=.25, proper_time=0., total_edep=.125,
                          nonionizing_edep=.03125, visible_edep=.015625, length=2.125, end_process=12,
                          particle_key=102)
        for field, value in values.items():
            bits(s[field], value, role + "_" + field)

    p = export["particles"]
    check(np.all(p["key"] == p["id"]), "stored_particle_ids")
    input_rows = p["product"] == input_pid
    check(np.count_nonzero(input_rows) == 2, "input_particle_rows")
    bits(p["stage"][input_rows], 0, "input_stage")
    bits(p["generator_state"][input_rows], 1, "input_generator")
    bits(p["generator_product"][input_rows], gen_pid, "input_generator_product")
    check(np.all(p["parent_state"][input_rows] == 0), "input_null_parent")
    new = (p["product"] == continued_pid) & (p["key"] != 17)
    check(np.count_nonzero(new) == 6, "new_stage_rows")
    bits(p["stage"][new], 2, "new_stage")
    bits(p["time"], np.float32(source_t), "particle_float_times")
    bits(p["x"], source_x, "particle_double_position")
    bits(p["energy"][p["pdg"] == 11], np.float32(energy), "particle_float_energy")
    photon = p["pdg"] == 22
    bits(p["parent_key"][photon], 101, "photon_parent")
    bits(p["ion_excitation"][photon], .0625, "ion_excitation")
    bits(p["ion_float_level"][photon], 3, "ion_level")
    key = p["key"]
    old, gamma_rows = key == 17, key == 102
    continued = p["product"] == continued_pid
    full_particle_values = {
        "stage": np.where(old, 0, 2), "pdg": np.where(gamma_rows, 22, 11),
        "creation": np.where(gamma_rows, 16, 56), "stopping": np.where(key == 101, 16, 49),
        "ndaughters": np.where(continued & old, 2, (key == 101).astype(int)),
        "parent_state": (~old).astype(int), "generator_state": old.astype(int),
        "end_x": source_x, "y": -0., "z": 4179.,
        "end_y": np.where((key == 101) | gamma_rows, 0., -0.),
        "end_z": np.where(key == 101, 4180., np.where(gamma_rows, 4181., 4179.)),
        "px": np.where(gamma_rows, 0., .125), "py": np.where(gamma_rows, 0., -.25),
        "pz": np.where(gamma_rows, .5, 4.), "end_px": np.where(gamma_rows, 0., .125),
        "end_py": np.where(gamma_rows, 0., -.25), "end_pz": np.where(gamma_rows, .5, 4.),
        "end_energy": np.where(gamma_rows, .5, np.float32(energy)),
        "proper_time": np.where(gamma_rows, 0., .125), "end_time": np.float32(source_t),
        "end_proper_time": np.where(gamma_rows, 0., .25), "start_volume": np.where(old, 7, 11),
        "end_volume": np.where(old, 7, 11), "start_g4status": 0, "end_g4status": 2,
        "end_defined": 1, "pre_last_ke": -1., "end_ke": np.where(old, 3.5, np.where(gamma_rows, .5, 3.)),
        "nsteps": np.where(old, 2, np.where(key == 101, 3, 1)),
        "track_length": np.where(key == 101, .75, np.where(gamma_rows, 2., .5)),
        "ion_excitation": np.where(gamma_rows, .0625, 0.), "ion_float_level": np.where(gamma_rows, 3, 0),
    }
    for field, expected in full_particle_values.items():
        bits(p[field], expected, "particle_full_" + field)
    check(np.all(p["parent_product"][~old] == continued_pid), "parent_products")
    check(np.all(p["parent_key"][~old] == np.where(gamma_rows[~old], 101, 17)), "parent_keys")
    check(np.all(p["generator_product"][old] == gen_pid) and np.all(p["generator_key"][old] == 0), "generator_ptrs")
    d = export["daughters"]
    copied = d["owner_key"] == 17
    check(np.all(d["daughter_key"][copied] == np.where(d["index"][copied] == 0, 103, 101)), "daughter_order")
    check(np.all(d["daughter_product"] == continued_pid), "daughter_product")
    statuses = export["statuses"]
    for field, expected in {"status": int(fail_status), "ntracks": np.where(statuses["event"] % 2, 3, 0),
                            "overflow": 0, "killed_step": 0, "killed_field": int(fail_status),
                            "cpu": .0625, "real": .125}.items():
        bits(statuses[field], expected, "status_" + field)
    g = export["generators"]
    bits(g["time"], source_t, "generator_double_time")
    bits(g["proper_time"], .125, "generator_proper_time")
    bits(g["energy"], energy, "generator_energy")
    for field, expected in {"pdg": 11, "gen_id": 1, "x": source_x, "y": -0., "z": 4179.,
                            "px": .125, "py": -.25, "pz": 4.}.items():
        bits(g[field], expected, "generator_full_" + field)
    v = export["volumes"]
    check(np.all(v["copy"] == 9), "repeated_volume_copy")
    check(set(v["index"]) == {7, 11}, "distinct_volume_keys")
    check(np.all(v["stage"] == np.where(v["index"] == 7, 0, 2)), "volume_stages")
    check(np.all(v["name"] == np.where(v["index"] == 7, "upstream", "foil")), "volume_names")
    check(np.all(v["material"] == np.where(v["index"] == 7, "Vacuum", "Aluminum")), "volume_materials")
    stages = export["stages"]
    check(np.count_nonzero((stages["stage"] == 1) & (stages["size"] == 0)) == 2, "empty_stage")
    return {"fixture_values_valid": not failures, "failures": failures, "audit": audit,
            "scope": "Deterministic extraction-fixture expectations only; not physics or arbitrary-art extraction certification."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--directory", default="export")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fail-status", action="store_true")
    parser.add_argument("--unavailable-ptr", action="store_true")
    args = parser.parse_args(argv)
    report = check_fixture(args.input, directory=args.directory, fail_status=args.fail_status, unavailable_ptr=args.unavailable_ptr)
    with Path(args.out).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report["fixture_values_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
