"""Run a bounded synthetic Geant4/Celeritas EM smoke test, not Mu2e validation."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


GEOMETRY = '''<?xml version="1.0" encoding="UTF-8"?>
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="http://cern.ch/2001/Schemas/GDML/gdml.xsd">
  <define/>
  <materials>
    <element name="H" formula="H" Z="1"><atom value="1.00794"/></element>
    <element name="Pb" formula="Pb" Z="82"><atom value="207.2"/></element>
    <material name="Vacuum" state="gas">
      <D value="1e-25" unit="g/cm3"/><fraction n="1" ref="H"/>
    </material>
    <material name="Lead" state="solid">
      <D value="11.35" unit="g/cm3"/><fraction n="1" ref="Pb"/>
    </material>
  </materials>
  <solids>
    <box name="world_box" x="200" y="200" z="200" lunit="cm"/>
    <box name="absorber_box" x="100" y="100" z="100" lunit="cm"/>
  </solids>
  <structure>
    <volume name="absorber">
      <materialref ref="Lead"/><solidref ref="absorber_box"/>
      <auxiliary auxtype="SensDet" auxvalue="smoke_calo"/>
    </volume>
    <volume name="world">
      <materialref ref="Vacuum"/><solidref ref="world_box"/>
      <physvol copynumber="1"><volumeref ref="absorber"/></physvol>
    </volume>
  </structure>
  <setup name="Default" version="1.0"><world ref="world"/></setup>
</gdml>
'''


def make_input(pdg, events, field_tesla):
    """Use v0.6.3 JSON units: positions in cm, energies in MeV, field in tesla."""
    if pdg not in (11, -11, 22) or not 1 <= events <= 1024:
        raise ValueError("Smoke test requires e-/e+/gamma and 1..1024 events")
    if field_tesla not in (0, 1):
        raise ValueError("Smoke field must be 0 or 1 tesla")
    return {
        "_format": "celer-g4", "_version": "0.6.3",
        "geometry_file": "../smoke.gdml", "output_file": "result.json",
        "physics_list": "celer_em", "sd_type": "simple_calo",
        "field_type": "uniform", "field": [0, 0, field_tesla],
        "num_track_slots": 1024, "initializer_capacity": 16384,
        "secondary_stack_factor": 4, "max_steps": 10000,
        "action_times": True,
        "primary_options": {
            "_format": "primary-generator", "_units": "cgs",
            "seed": 20261025, "pdg": [pdg], "num_events": events,
            "primaries_per_event": 1,
            "energy": {"distribution": "delta", "params": [10.0]},
            "position": {"distribution": "delta", "params": [0, 0, 0]},
            "direction": {"distribution": "delta", "params": [0, 0, 1]},
        },
    }


def clean_environment(mode):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CELER_")}
    env.update(G4FORCENUMBEROFTHREADS="1", G4FORCE_RUN_MANAGER_TYPE="Serial",
               CELER_LOG="info")
    if mode == "g4":
        env.update(CELER_DISABLE="1", CELER_DISABLE_DEVICE="1")
    elif mode != "gpu":
        raise ValueError(f"Unsupported mode {mode}")
    return env


def inspect_result(data, pdg, mode, events):
    """Check persisted end-event records, scorer and GPU evidence; ignore the buggy hit log."""
    primary = data["input"]["primary_options"]
    if (primary["pdg"] != [pdg] or primary["num_events"] != events
            or primary["primaries_per_event"] != 1
            or primary["energy"] != {"distribution": "delta", "params": [10.0]}):
        raise ValueError("Result primary contract does not match the requested smoke arm")
    recorded = sum(map(len, data["result"]["time"]["events"]))
    if recorded != events:
        raise ValueError(f"Expected {events} completed event records, got {recorded}")
    calo = data["result"]["smoke_calo"]
    if calo["_units"]["energy_deposition"] != "MeV" or calo["volume_labels"] != ["absorber"]:
        raise ValueError("Unexpected score units or sensitive volume")
    energy = sum(calo["energy_deposition"])
    expected = events * (10.0 + (2 * 0.51099891 if pdg == -11 else 0))
    if not math.isfinite(energy) or abs(energy / expected - 1) > 1e-5:
        raise ValueError(f"Contained-energy smoke check failed: {energy} versus {expected} MeV")
    system = data["system"]
    actions = data["result"]["time"]["actions"]
    if mode == "gpu":
        if system["device"]["platform"] != "CUDA" or "A100" not in system["device"]["name"]:
            raise ValueError("Expected CUDA on an A100")
        kernels = {k["name"] for k in system["kernels"]}
        if "extend-from-primaries" not in kernels or "step-gather-post" not in kernels:
            raise ValueError("No GPU primary/hit kernel evidence")
        if not any(a.get("along-step-uniform-msc", 0) > 0 for a in actions):
            raise ValueError("No timed GPU transport action")
        if system["environ"].get("CELER_DISABLE") not in (None, "", "0"):
            raise ValueError("Offload was disabled")
    elif system["environ"].get("CELER_DISABLE") != "1":
        raise ValueError("Geant4 control did not disable offload")
    return {"pdg": pdg, "mode": mode, "completed_event_records": recorded,
            "energy_mev": energy, "expected_contained_energy_mev": expected,
            "relative_energy_error": energy / expected - 1,
            "transport_seconds": data["result"]["time"]["total"]}


def check_output(out):
    manifest = json.loads((out / "manifest.json").read_text())
    expected = {(pdg, mode) for pdg in (11, -11, 22) for mode in ("g4", "gpu")}
    runs = manifest["runs"]
    if len(runs) != 6 or {(r["pdg"], r["mode"]) for r in runs} != expected:
        raise ValueError("Incomplete or duplicate smoke arms")
    rows = []
    for run in runs:
        if run["returncode"] != 0:
            raise ValueError(f"Failed process: {run['name']}")
        result = json.loads((out / run["name"] / "result.json").read_text())
        rows.append(inspect_result(result, run["pdg"], run["mode"], manifest["events_per_arm"]))
    summary = {"scope": "contained-energy and execution smoke only; not Mu2e validation",
               "pass": True, "rows": rows,
               "hit_counter": "Excluded: v0.6.3 HitProcessor::num_hits_ lacks initialization"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--events", type=int, default=64)
    parser.add_argument("--field-tesla", type=int, choices=(0, 1), default=1)
    parser.add_argument("--check-only", action="store_true", help="Inspect existing outputs without transport")
    args = parser.parse_args()
    if args.check_only:
        check_output(args.output)
        return
    make_input(11, args.events, args.field_tesla)
    out = args.output.resolve()
    out.mkdir(parents=False, exist_ok=False)
    env = clean_environment("gpu")
    version = subprocess.check_output(["celer-g4", "--version"], env=env, text=True).strip()
    if version != "0.6.3":
        raise RuntimeError(f"Expected Celeritas 0.6.3, got {version}")
    (out / "smoke.gdml").write_text(GEOMETRY)
    manifest = {
        "scope": "synthetic EM smoke only; not Mu2e flash or a speed benchmark",
        "version": version, "events_per_arm": args.events, "field_tesla": args.field_tesla,
        "geometry_sha256": hashlib.sha256(GEOMETRY.encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": [],
    }
    for pdg in (11, -11, 22):
        for mode in ("g4", "gpu"):
            name = f"pdg{pdg}_{mode}"
            work = out / name
            work.mkdir()
            card = make_input(pdg, args.events, args.field_tesla)
            (work / "input.json").write_text(json.dumps(card, indent=2) + "\n")
            start = time.monotonic()
            with (work / "run.log").open("w") as log:
                result = subprocess.run(["celer-g4", "input.json"], cwd=work,
                                        env=clean_environment(mode), stdout=log, stderr=subprocess.STDOUT)
            record = {"name": name, "pdg": pdg, "mode": mode,
                      "returncode": result.returncode, "wall_seconds": time.monotonic() - start}
            manifest["runs"].append(record)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if result.returncode:
                raise RuntimeError(f"{name} failed; see {work / 'run.log'}")
    check_output(out)


if __name__ == "__main__":
    main()
