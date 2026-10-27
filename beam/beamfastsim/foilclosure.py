"""Bounded G4 single-foil closure driver using the published study geometry/readback modules.

This is a zero-field NIST-Al slab diagnostic, not the full detector or a replacement reference. The exported
material/cut tables must be compared with StoppingTarget_Al before assigning material equivalence. Run in a
SimJob MDC2025ac shell on one idle GPVM CPU. No build, implicit input copy or baseline overwrite is performed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .reference_audit import _identity

RELEASE = "/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ac"
OFFLINE = Path("/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v10_47_00/Offline")
MMU = 105.6583755


def macro_errors(text):
    return any(message in text.lower() for message in ("command not found", "illegal application state",
        "command refused", "not found in the process table", "illegal process", "illegal parameter",
        "command aborted", "batch is interrupted"))


def validate_case(energy, thickness, cosine, pdg, events, seed, energy_max=None, cosine_min=None):
    energy_max = energy if energy_max is None else energy_max
    cosine_min = cosine if cosine_min is None else cosine_min
    for value in (energy, thickness, cosine, energy_max, cosine_min):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("positive finite energy, thickness and cosine required")
    if not .1 <= energy <= energy_max <= 40 or thickness > 1 or not .5 <= cosine_min <= cosine <= 1:
        raise ValueError("pilot domain: 0.1-40 MeV, thickness <=1 mm, cosine >=0.5 (ranges ordered)")
    if pdg not in (13, -13):
        raise ValueError("muon charge required")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (events, seed)) or events >= 2**24:
        raise ValueError("positive integer events/seed required; float ntuple IDs must be exact")


def render(energy, thickness, cosine, pdg, energy_max=None, cosine_min=None):
    """One-muon gun just upstream of an effectively infinite foil, in a small vacuum world.

    Fixed energy/direction by default; energy_max / cosine_min give a gun flat in momentum and in cos(theta)."""
    validate_case(energy, thickness, cosine, pdg, 1, 1, energy_max, cosine_min)
    energy_max = energy if energy_max is None else energy_max
    cosine_min = cosine if cosine_min is None else cosine_min
    half = thickness / 2
    geom = f'''string detector.name = "g4study";
bool g4.visible = false;
bool g4.solid = false;
bool g4.forceAuxEdgeVisible = false;
bool g4.doSurfaceCheck = true;
bool g4.placePV = true;
string world.materialName = "G4_Galactic";
bool world.boxVisible = false;
bool world.boxSolid = false;
double world.halfLength = 20.0;
double world.outerLayerThickness = 0.001;
bool box.visible = false;
bool box.solid = false;
string box.wallMaterialName = "G4_Al";
vector<double> box.halfLengths = {{19., 19., {half:.17g}}};
int box.copyNumber = 2;
vector<double> box.centerInWorld = {{0., 0., 0.}};
'''
    p = math.sqrt(energy * (energy + 2 * MMU))
    pmax = math.sqrt(energy_max * (energy_max + 2 * MMU))
    gun = f'''bool particleGun.do = true;
double particleGun.czmin = {cosine_min:.17g};
double particleGun.czmax = {cosine:.17g};
double particleGun.phimin = 0.;
double particleGun.phimax = 0.;
double particleGun.tmin = 0.;
double particleGun.tmax = 0.;
vector<double> particleGun.point = {{0., 0., {-half - .001:.17g}}};
vector<double> particleGun.halfLength = {{0., 0., 0.}};
double particleGun.pmin = {p:.17g};
double particleGun.pmax = {pmax:.17g};
int particleGun.id = {pdg};
double particleGun.mean = -1;
bool particleGun.doHistograms = false;
bool particleGun.verbose = false;
'''
    return geom, gun


def run_case(out, *, energy, thickness, cosine=1., pdg=13, events=10, seed=20261026,
             export_tables=False, step_fraction=None, no_msc=False, final_range=.001, bins=None,
             energy_max=None, cosine_min=None):
    """New directory, immutable inputs, subprocess outcome, ROOT receipt and explicit unfinished marker."""
    validate_case(energy, thickness, cosine, pdg, events, seed, energy_max, cosine_min)
    if step_fraction is not None and (not math.isfinite(step_fraction) or not 0 < step_fraction < 1):
        raise ValueError("step fraction must lie in (0,1)")
    if not math.isfinite(final_range) or final_range <= 0:
        raise ValueError("positive final range required")
    if bins is not None and (isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0):
        raise ValueError("positive integer bins required")
    if not isinstance(no_msc, bool):
        raise ValueError("no_msc must be a boolean")
    name = "mu-" if pdg == 13 else "mu+"
    if os.environ.get("MUSE_WORK_DIR") != RELEASE:
        raise ValueError("published SimJob MDC2025ac setup required")
    out = Path(out).resolve()
    data = Path(os.environ.get("BEAMFS_DATA_ROOT", "/exp/mu2e/data/users/ram2aq/beam_fastsim")).resolve()
    if data not in out.parents:
        raise ValueError(f"output must be inside the data area {data} (BEAMFS_DATA_ROOT on grid workers)")
    template = Path(__file__).resolve().parents[1] / "fcl/foilclosure.fcl"
    binary = Path(shutil.which("mu2e") or "missing-mu2e").resolve()
    geometry, gun = render(energy, thickness, cosine, pdg, energy_max, cosine_min)
    out.mkdir(parents=True, exist_ok=False)
    (out / "foil_geom.txt").write_text(geometry)
    (out / "foil_gun.txt").write_text(gun)
    (out / "base.fcl").write_text(template.read_text())
    macro = ["/control/verbose 2"]
    if export_tables:
        (out / "tables").mkdir()
        macro.extend(["/run/particle/setStoredInAscii 1", "/process/eLoss/CSDARange true", "/process/em/binsPerDecade 168"])
    if bins is not None:
        macro.append(f"/process/em/binsPerDecade {bins}")
    if step_fraction is not None:
        macro.append(f"/process/eLoss/StepFunctionMuHad {step_fraction:.17g} {final_range:.17g} mm")
    if no_msc:
        macro.extend(["/run/initialize", "/process/list all", f"/process/inactivate msc {name}",
                      f"/process/inactivate CoulombScat {name}", f"/process/dump msc {name}",
                      f"/process/dump CoulombScat {name}"])
    (out / "preinit.mac").write_text("\n".join(macro) + "\n")
    config = f'''#include "base.fcl"
source.maxEvents: {events}
services.SeedService.baseSeed: {seed}
physics.producers.g4run.g4Macro: "preinit.mac"
'''
    if export_tables:
        config += 'physics.producers.g4run.debug.storePhysicsTablesDir: "tables"\n'
    (out / "job.fcl").write_text(config)
    env = os.environ.copy()
    for env_name in ("MU2E_SEARCH_PATH", "FHICL_FILE_PATH"):
        env[env_name] = str(out) + ":" + env.get(env_name, "")
    artifacts = [template, Path(__file__), binary]
    artifacts.extend(OFFLINE / p for p in ("Mu2eG4/src/ConstructEnvBox_tool.cc", "Mu2eG4/src/Mu2eStudyWorld.cc",
        "Mu2eG4/src/Mu2eG4Cuts.cc", "Analyses/src/Mu2eG4StudyReadBack_module.cc", "Mu2eG4/fcl/prolog.fcl"))
    receipt = dict(complete=False, release=RELEASE, muse_stub=os.environ.get("MUSE_STUB"),
        case=dict(energy_MeV=energy, thickness_mm=thickness, cosine=cosine, pdg=pdg, events=events, seed=seed,
                  export_tables=export_tables, step_fraction=step_fraction, no_msc=no_msc, final_range=final_range, bins=bins,
                  energy_max_MeV=energy if energy_max is None else energy_max,
                  cosine_min=cosine if cosine_min is None else cosine_min),
        sources=[_identity(p) for p in artifacts],
        scope="Zero-field G4_Al study box; material equivalence and stochastic closure require separate checks")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    with (out / "expand.log").open("w") as log:
        expanded = subprocess.run([str(binary), "-c", "job.fcl", "--debug-config", "expanded.fcl"],
                                  cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
    if expanded.returncode:
        raise RuntimeError(f"configuration expansion failed: {out / 'expand.log'}")
    started = time.monotonic()
    with (out / "job.log").open("w") as log:
        result = subprocess.run([str(binary), "-c", "job.fcl"], cwd=out, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    receipt.update(exit_code=result.returncode, wall_s=time.monotonic() - started)
    text = (out / "job.log").read_text()
    receipt["complete"] = result.returncode == 0 and "Art has completed and will exit with status 0" in text
    if macro_errors(text):
        receipt["complete"] = False
    if no_msc and not all(command in text for command in (f"/process/dump msc {name}", f"/process/dump CoulombScat {name}")):
        receipt["complete"] = False
    receipt["artifacts"] = [_identity(out / n) for n in ("base.fcl", "job.fcl", "foil_geom.txt", "foil_gun.txt",
                                                          "preinit.mac", "expanded.fcl", "job.log")]
    if (out / "foil.root").exists():
        receipt["output"] = _identity(out / "foil.root")
    if export_tables:
        receipt["tables"] = [_identity(p) for p in sorted((out / "tables").iterdir()) if p.is_file()]
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ("complete", "exit_code", "wall_s", "case")}), flush=True)
    if not receipt["complete"]:
        raise RuntimeError(f"G4 run failed: {out / 'job.log'}")
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--energy", type=float, required=True)
    ap.add_argument("--thickness", type=float, required=True)
    ap.add_argument("--cosine", type=float, default=1.)
    ap.add_argument("--pdg", type=int, default=13)
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--export-tables", action="store_true")
    ap.add_argument("--step-fraction", type=float)
    ap.add_argument("--no-msc", action="store_true")
    ap.add_argument("--final-range", type=float, default=.001)
    ap.add_argument("--bins", type=int)
    ap.add_argument("--energy-max", type=float, help="flat-in-momentum gun from --energy to this kinetic energy")
    ap.add_argument("--cosine-min", type=float, help="flat-in-cos(theta) gun from this cosine to --cosine")
    a = ap.parse_args(argv)
    run_case(a.out, energy=a.energy, thickness=a.thickness, cosine=a.cosine, pdg=a.pdg,
             events=a.events, seed=a.seed, export_tables=a.export_tables, step_fraction=a.step_fraction, no_msc=a.no_msc,
             final_range=a.final_range, bins=a.bins, energy_max=a.energy_max, cosine_min=a.cosine_min)


if __name__ == "__main__":
    main()
