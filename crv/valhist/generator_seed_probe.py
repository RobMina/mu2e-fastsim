"""Observe stock module seed assignment with zero generated events; no RNG independence claim.

The unchanged generator/reference FCLs are initialized in their historical CVMFS Musing.
Only seed verbosity, run/subrun/baseSeed and output paths are overridden. No events are
produced; a zero-event probe is implementation evidence, not a historical RNG-state capture.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time

from . import genassoc_campaign as C, genassoc_fullgen as F
from . import generator_bank_capture as B

run_process = B.run_process

MUSING = "/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ay"


def configuration(mode, subrun, out):
    if mode not in ("crygen", "reference") or not isinstance(subrun, int) or subrun < 1:
        raise ValueError("invalid mode/subrun")
    if mode == "crygen" and not 2001 <= subrun <= 2100:
        raise ValueError("crygen subrun outside historical campaign")
    if mode == "reference" and not 1 <= subrun <= 220:
        raise ValueError("reference subrun outside historical campaign")
    base = 20263000 + subrun - 2001 if mode == "crygen" else 20261016 + subrun - 1
    parent = "gen_only_extracted.fcl" if mode == "crygen" else "ref_sim_extracted.fcl"
    lines = [f'#include "{parent}"', f"source.firstSubRun: {subrun}",
             f"services.SeedService.baseSeed: {base}", "services.SeedService.verbosity: 1",
             f"outputs.GenOutput.fileName: {json.dumps(str(out / 'unused_gen.art'))}"]
    if mode == "reference":
        lines.append(f"outputs.PrimaryOutput.fileName: {json.dumps(str(out / 'unused_dts.art'))}")
    return "\n".join(lines) + "\n", base


def verify_zero(text):
    exits = re.findall(r"Art has completed and will exit with status (-?\d+)\b", text)
    counts = re.findall(r"TrigReport Events total = (\d+) passed = (\d+) failed = (\d+)", text)
    if not exits or any(v != "0" for v in exits) or counts != [("0", "0", "0")]:
        raise ValueError("seed probe must complete with exactly zero generated events")
    headings = list(re.finditer(r"(?m)^\s*Seed Value\s+ModuleLabel.InstanceName\s*$", text))
    if len(headings) != 1:
        raise ValueError("require exactly one populated seed summary")
    block = text[headings[0].end():].split("=============================================", 1)[0]
    rows = re.findall(r"(?m)^\s*(\d+)\s+(\w+)\s*$", block)
    seeds = {name: int(seed) for seed, name in rows}
    if len(rows) != 3 or set(seeds) != {"generate", "cosmicTimeOffset", "g4run"}:
        raise ValueError("unexpected seed assignment summary")
    bases = re.findall(r"(?m)^\s*Base Seed:\s*(\d+)\s*$", text)
    if not bases or len(set(bases)) != 1:
        raise ValueError("ambiguous base seed")
    return int(bases[0]), seeds


def run(mode, subrun, outdir):
    out = Path(outdir).absolute()
    out.mkdir()
    try:
        return _run(mode, subrun, out)
    except BaseException as exc:
        C._new_json(out / "failed.json", {"error": str(exc), "downstream_authorized": False})
        raise


def _run(mode, subrun, out):
    if os.environ.get("MUSE_WORK_DIR") != MUSING:
        raise ValueError("seed probe requires historical CVMFS Musing")
    code = Path(__file__).resolve().parents[1]
    source = code / "grid/fcl" / ("gen_only_extracted.fcl" if mode == "crygen" else "ref_sim_extracted.fcl")
    config, base = configuration(mode, subrun, out)
    fcl = out / "seed_probe.fcl"
    fcl.write_text(config)
    executable = shutil.which("mu2e")
    if not executable:
        raise ValueError("mu2e missing")
    env = dict(os.environ, FHICL_FILE_PATH=str(source.parent) + ":" + os.environ.get("FHICL_FILE_PATH", ""))
    old = os.environ.get("FHICL_FILE_PATH")
    os.environ["FHICL_FILE_PATH"] = env["FHICL_FILE_PATH"]
    try:
        configs = F._config_files(fcl, Path(MUSING))
    finally:
        if old is None:
            os.environ.pop("FHICL_FILE_PATH", None)
        else:
            os.environ["FHICL_FILE_PATH"] = old
    inputs = F._hashes([*configs, Path(__file__).resolve(), Path(executable).resolve(),
                        code / "scripts/run_generator_seed_probe.sh", Path(F.__file__).resolve(),
                        Path(B.__file__).resolve(), Path(C.__file__).resolve()])
    command = [executable, "-c", str(fcl), "-n", "0"]
    manifest = {"mode": mode, "subrun": subrun, "base_seed": base, "musing": MUSING,
                "command": command, "input_sha256": inputs,
                "environment": {k: env.get(k) for k in ("MUSE_WORK_DIR", "FHICL_FILE_PATH", "CET_PLUGIN_PATH",
                    "LD_LIBRARY_PATH", "MU2E_SEARCH_PATH", "CRYDATAPATH")}, "downstream_authorized": False}
    C._new_json(out / "inputs.json", manifest)
    start = time.monotonic()
    result = run_process(command, out / "probe.log", out, env)
    text = (out / "probe.log").read_text()
    observed_base, seeds = verify_zero(text)
    if observed_base != base:
        raise ValueError("observed base seed differs from requested configuration")
    report = {**manifest, "execution": result, "generated_events": 0, "seconds": time.monotonic() - start,
              "observed_seed_assignment": seeds,
              "input_sha256_post": F._unchanged(inputs), "log_sha256": F._hashes([out / "probe.log"]),
              "historical_rng_state_certified": False,
              "scope": "Current initialization of historical immutable Musing/config; not an original-job state dump."}
    C._new_json(out / "certificate.json", report)
    print(text, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("crygen", "reference"), required=True)
    p.add_argument("--subrun", type=int, required=True)
    p.add_argument("--outdir", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
