"""Field-sensitive thin-surface diagnostic with the existing h5 adapter; no builds."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import validate_runtime as runtime

H5_LIBRARIES = {
    "libceleritas.so": "f11aa6c5bf290329e51956f5948f12933f51cf26a256b24b186d19e4e1c19f0d",
    "libceleritas_final.so": "043039fdef917a2bd7fc04645aa42327fe2ae1d00043f58cd0cc79d8b720e810",
    "libaccel_final.so": "48ec8354046c1e9f6228f40cdac95575337ee963946f9b92a4cb91089ac15410",
}
SETTINGS = {
    "default": {},
    "chord_loose": {"delta_chord": 0.05},
    "chord_fine": {"delta_chord": 0.0005},
    "delta_loose": {"delta_intersection": 0.001},
    "delta_fine": {"delta_intersection": 1e-7},
    "epsilon_loose": {"epsilon_rel_max": 1e-4},
    "epsilon_fine": {"epsilon_rel_max": 1e-8},
    "joint_fine": {"delta_chord": 0.0005, "delta_intersection": 1e-7, "epsilon_rel_max": 1e-8},
}
EVENTS = 4
TIME_BUDGET_S = 1e-14


def oracle(pdg, field):
    """First z=20 mm plane intersection in vacuum: analytic helix, not transport replay."""
    mass, kinetic, length = 0.51099895, 10.0, 25.0
    momentum = math.sqrt(kinetic * (kinetic + 2 * mass))
    charge = -1 if pdg == 11 else 1
    omega = charge * 0.299792458 * field / momentum
    if omega:
        x = 0.6 * math.sin(omega * length) / omega
        y = 0.6 * (math.cos(omega * length) - 1) / omega
    else:
        x, y = 15.0, 0.0
    beta = momentum / (kinetic + mass)
    return {"position_mm": [x, y, 20.0], "path_mm": length,
            "time_s": length / (299792458000.0 * beta)}


def geometry(split):
    """Two nonoverlapping sensitive tiles share one planar boundary; 10 micrometre lead."""
    volumes, placements, solids = [], [], []
    for copy, lo, hi in ((1, -99.0, split), (2, split, 99.0)):
        solids.append(f'<box name="tile{copy}_box" x="198" y="{hi-lo:.16g}" z="0.01" lunit="mm"/>')
        volumes.append(f'<volume name="tile{copy}"><materialref ref="Lead"/><solidref ref="tile{copy}_box"/>'
                       '<auxiliary auxtype="SensDet" auxvalue="surface"/></volume>')
        placements.append(f'<physvol copynumber="{copy}"><volumeref ref="tile{copy}"/>'
                          f'<position name="place{copy}" x="0" y="{(hi+lo)/2:.16g}" z="20.005" unit="mm"/></physvol>')
    return ('<?xml version="1.0"?><gdml><define/><materials>'
            '<element name="H" formula="H" Z="1"><atom value="1.00794"/></element>'
            '<element name="Pb" formula="Pb" Z="82"><atom value="207.2"/></element>'
            '<material name="Vacuum" state="gas"><D value="1e-25" unit="g/cm3"/><fraction n="1" ref="H"/></material>'
            '<material name="Lead" state="solid"><D value="11.35" unit="g/cm3"/><fraction n="1" ref="Pb"/></material>'
            '</materials><solids><box name="world_box" x="200" y="200" z="200" lunit="mm"/>'
            + ''.join(solids) + '</solids><structure>' + ''.join(volumes)
            + '<volume name="world"><materialref ref="Vacuum"/><solidref ref="world_box"/>'
            + ''.join(placements) + '</volume></structure><setup name="Default" version="1.0">'
            '<world ref="world"/></setup></gdml>\n')


def map_data(field):
    return dict(format="m2fs-cartmap-v1", length_unit="mm", field_unit="tesla", coordinates="gdml-world",
                outside_policy="world-contained", order="xyz-component",
                axes=[dict(min_mm=-200, max_mm=200, num=9) for _ in range(3)], field=[0, 0, field] * 729)


def cases():
    rows = []
    for pdg in (11, -11):
        for shift in (-0.001, 0.001):
            split = oracle(pdg, 1)["position_mm"][1] + shift
            for setting in SETTINGS:
                rows.append(dict(name=f"p{pdg}_shift{shift:g}_{setting}_gpu", pdg=pdg, split_mm=split,
                                 shift_mm=shift, setting=setting, mode="gpu", field=1))
            for mode, field in (("g4", 1), ("gpu", 0)):
                rows.append(dict(name=f"p{pdg}_shift{shift:g}_control_{mode}_b{field}", pdg=pdg, split_mm=split,
                                 shift_mm=shift, setting="joint_fine", mode=mode, field=field))
    return rows


def make_card(row):
    card = runtime.smoke.make_input(row["pdg"], EVENTS, 1)
    card.update(geometry_file="surface.gdml", sd_type="event_hit", field_type="cartmap", field_file="map.json",
                field_options=dict(SETTINGS[row["setting"]]), physics_options={"msc": "none"})
    card.pop("field")
    card["primary_options"]["direction"]["params"] = [0.6, 0, 0.8]
    return card


def check_events(rows, case):
    runtime.require(len(rows) == EVENTS, "Wrong event denominator")
    runtime.require([r["event_id"] for r in rows] == list(range(EVENTS)), "Missing/duplicate event")
    expected = oracle(case["pdg"], case["field"])
    copy = 1 if expected["position_mm"][1] < case["split_mm"] else 2
    first = []
    for row in rows:
        runtime.require(row["format"] == "m2fs-event-total-edep-v1" and row["primary_count"] == 1, "Wrong score contract")
        runtime.require(len(row["collections"]) == 1 and row["collections"][0]["sd"] == "surface", "Wrong SD")
        hits = row["collections"][0]["hits"]
        runtime.require(hits and row["hit_count"] == len(hits), "No surface hits or wrong count")
        runtime.require(all(math.isfinite(h["pre_time_s"]) and math.isfinite(h["total_edep_mev"])
                            and h["total_edep_mev"] > 0 for h in hits), "Invalid hit")
        energy = math.fsum(h["total_edep_mev"] for h in hits)
        runtime.require(math.isclose(energy, row["total_edep_mev"], rel_tol=1e-12) and
                        math.isclose(energy, row["collections"][0]["total_edep_mev"], rel_tol=1e-12), "Energy sum mismatch")
        hit = min(hits, key=lambda h: h["pre_time_s"])
        first.append(dict(event=row["event_id"], copy=hit["copy_num"], time_s=hit["pre_time_s"],
                          time_error_s=hit["pre_time_s"]-expected["time_s"]))
    correct = sum(h["copy"] == copy for h in first)
    time_error = max(abs(h["time_error_s"]) for h in first)
    energies = [r["total_edep_mev"] for r in rows]
    return dict(expected_copy=copy, oracle=expected, first_hits=first, correct_first_tiles=correct,
                event_total_edep_mev=energies, mean_total_edep_mev=math.fsum(energies) / EVENTS,
                events=EVENTS, max_time_error_s=time_error,
                passed=correct == EVENTS and time_error <= TIME_BUDGET_S)


def check(out):
    manifest = runtime.read_json(out / "manifest.json")
    runtime.require(manifest["executable_sha256"] == "6061c8f14695725f416e8e0e88a3fea5b8ea048afe84cb1455b651a50112c31c", "Changed h5 executable")
    libraries = manifest["library_provenance"]["executables"]["m2fs-celer-g4"]
    for name in ("libceleritas.so", "libaccel_final.so"):
        runtime.require(libraries[name]["sha256"] == H5_LIBRARIES[name], "Changed loaded h5 library")
    runtime.require(manifest["cases"] == cases() and len(manifest["runs"]) == len(cases()), "Missing/changed cases")
    summaries = []
    for row, run in zip(cases(), manifest["runs"]):
        work = out / row["name"]
        runtime.require(run["name"] == row["name"] and run["returncode"] == 0, "Failed/mismatched process")
        for name, digest in run["sha256"].items():
            runtime.require(runtime.digest(work / name) == digest, "Changed executed input/output")
        card = runtime.read_json(work / "input.json")
        runtime.require(card == make_card(row), "Changed card")
        runtime.require(runtime.read_json(work / "map.json") == map_data(row["field"]), "Changed map")
        runtime.require((work / "surface.gdml").read_text() == geometry(row["split_mm"]), "Changed geometry")
        data = runtime.read_json(work / "result.json")
        runtime.check_transport(data, dict(card=card, mode=row["mode"]))
        runtime.require(data["input"]["physics_options"]["msc"] == "none", "MSC control not applied")
        opts = data["input"]["field_options"]
        runtime.require(all(opts[k] == v for k, v in SETTINGS[row["setting"]].items()), "Field controls not applied")
        events = [json.loads(line) for line in (work / "events.jsonl").read_text().splitlines()]
        summaries.append(dict(**row, **check_events(events, row)))
    result = dict(format="m2fs-surface-diagnostic-v1", passed=all(r["passed"] for r in summaries), rows=summaries,
                  scope="Thin-surface first deposited hit identity/time, not position/path or full navigator certification",
                  time_budget_s=TIME_BUDGET_S, internal_exhaustion_status="not exposed")
    return result


def run(args):
    runtime.verify_source_build(args.source, args.build)
    for name, digest in H5_LIBRARIES.items():
        runtime.require(runtime.digest(args.prefix / "lib64" / name) == digest, "Changed h5 library")
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    provenance = runtime.library_provenance(args.build, args.prefix, out)
    manifest = dict(cases=cases(), runs=[], library_provenance=provenance,
                    executable_sha256=runtime.digest(args.build / "m2fs-celer-g4"), script_sha256=runtime.digest(Path(__file__)))
    runtime.write_json(out / "manifest.json", manifest)
    for row in cases():
        work = out / row["name"]
        work.mkdir()
        (work / "surface.gdml").write_text(geometry(row["split_mm"]))
        runtime.write_json(work / "map.json", map_data(row["field"]))
        runtime.write_json(work / "input.json", make_card(row))
        env = runtime.smoke.clean_environment(row["mode"])
        env["M2FS_EVENT_OUTPUT"] = str(work / "events.jsonl")
        start = time.monotonic()
        with (work / "run.log").open("x") as log:
            result = subprocess.run([str(args.build / "m2fs-celer-g4"), "input.json"], cwd=work, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=90, check=False)
        record = dict(name=row["name"], returncode=result.returncode, seconds=time.monotonic()-start,
                      sha256={p.name: runtime.digest(p) for p in work.iterdir() if p.is_file()})
        manifest["runs"].append(record)
        runtime.write_json(out / "manifest.json", manifest)
        print(json.dumps(record), flush=True)
        runtime.require(result.returncode == 0, f"Failed {row['name']}")
    report = check(out)
    runtime.write_json(out / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--prefix", type=Path)
    args = parser.parse_args()
    report = run(args) if args.action == "run" else check(args.output)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
