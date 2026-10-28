"""Bounded strict re-dump of an existing crygen candidate file; never admit or publish a bank.

The candidate inventory binds the legacy table. Only art tools open the exact pnfs
path, with no namespace queries, staging or content hashing. Every attempt gets a new
output directory. A file certificate is not producer/selection or whole-bank admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import uproot

from crvfastsim import genassoc, tables
from . import genassoc_campaign as C, genassoc_fullgen as F, genassoc_smoke as S
from . import library_variants as V

ART_TEMPLATE = ("/pnfs/mu2e/scratch/users/ram2aq/crv_fastsim/crygen_v1/"
                "gen.ram2aq.CosmicCRYExtractedGen.v1.001400_{subrun:08d}.art")
SOURCE_SHA256 = "9127327c865c39d1a006bc703787066171ffe6fdbda374335f3d29ef6634f1f2"
PLUGIN_SHA256 = "3a884c6ac90836f752c96263d571c38cc4a4a1520df179b20928c73d7eaa5125"
MISSING = {2011, 2028, 2031, 2033, 2034, 2037}
POLICY = "One-file census/payload certificate only; downstream_authorized remains false."
CONFIG_SHA256 = {
    "CrvFastSim/fcl/dump_gen.fcl": "cb9e07791464e3f0190c7dfcaa393471c6e0f088ac86f6cd0f111e63efcda49b",
    "CrvFastSim/fcl/dump_crvsteps.fcl": "1cef585cbc98b0b1eb36949812dca797b0631d51fe1d4232fe574925021991f3",
    "Offline/fcl/minimalMessageService.fcl": "d05b2b517fc2e1598814175b1d0a2b2c4fff8239a7da7607e9ab4477f50fd48c",
    "Offline/fcl/messageService.fcl": "3d462bcfe90d0be57c85f45e5d137f50a296a596d36c6fc0cbbd58891136c64b",
    "Offline/fcl/standardMessageDestinations.fcl": "d4f8d5f07573d4265484b7ec5da97823d5a9c4d017a34846f4dab0c06718cdca",
}


def approved_configs(hashes):
    found = {}
    for path, digest in hashes.items():
        names = [name for name in CONFIG_SHA256 if str(path).endswith("/" + name)]
        if len(names) != 1 or names[0] in found or CONFIG_SHA256[names[0]] != digest:
            raise ValueError("unapproved dumper configuration/include: " + str(path))
        found[names[0]] = digest
    if found != CONFIG_SHA256:
        raise ValueError("incomplete approved dumper configuration closure")
    return found


def select_record(candidate, subrun):
    if candidate.get("candidate_only") is not True or candidate.get("downstream_authorized") is not False:
        raise ValueError("require isolated candidate inventory")
    records = candidate["files"]
    partitions = [(r["run"], r["subrun"]) for r in records]
    expected = {(1400, s) for s in range(2001, 2101) if s not in MISSING}
    if len(records) != 94 or set(partitions) != expected or len(set(partitions)) != len(partitions):
        raise ValueError("candidate must inventory all 94 distinct historical partitions")
    if candidate.get("events") != 94_000_000 or any(r["events_inferred"] != 1_000_000 for r in records):
        raise ValueError("unexpected candidate exposure")
    matches = [r for r in records if r["subrun"] == subrun]
    if len(matches) != 1:
        raise ValueError("requested subrun absent from candidate")
    row = dict(matches[0])
    old = Path(row["path"])
    if not old.is_absolute() or old.name != f"gen.001400_{subrun:08d}.root" or str(old).startswith("/pnfs/"):
        raise ValueError("require exact local legacy table path")
    row["art"] = ART_TEMPLATE.format(subrun=subrun)
    return row


def compare_file(old, new, run, subrun, expected_events):
    report, keys = F.compare_payload(old, new, expected_events=expected_events)
    if (not np.all(keys[:, 0] == run) or not np.all(keys[:, 1] == subrun)
            or not np.array_equal(np.sort(keys[:, 2]), np.arange(1, expected_events + 1))):
        raise ValueError("strict census must contain the complete expected run/subrun/event range")
    with uproot.open(new) as root:
        e = root["dump/events"].arrays(["ngen", "nsteps", "n_gen"], library="np")
    if not np.array_equal(e["ngen"], e["n_gen"]) or not np.all(e["nsteps"] == -1):
        raise ValueError("merged census aliases or unextracted-step sentinel mismatch")
    report.update(complete_event_range=True, merged_census_aliases=True,
                  event_keys_sha256=hashlib.sha256(keys.astype("<i8").tobytes()).hexdigest())
    return report


def run_process(command, log, cwd, env):
    start = time.monotonic()
    with Path(log).open("x") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=cwd, env=env)
        try:
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    result = {"command": command, "exit": code, "pid": proc.pid, "seconds": time.monotonic() - start}
    if code:
        raise RuntimeError(f"command exited {code}; evidence retained in {log}")
    return result


def capture(candidate, subrun, area, outdir):
    out = V._new_output(outdir)
    out.mkdir()
    try:
        return _capture(candidate, subrun, area, out)
    except BaseException as exc:
        C._new_json(out / "failed.json", {"error": str(exc), "exception": type(exc).__name__,
                                         "policy": POLICY, "downstream_authorized": False})
        raise


def _capture(candidate, subrun, area, out):
    candidate = Path(candidate).resolve(strict=True)
    pre = F._hashes([candidate])
    row = select_record(json.loads(candidate.read_text()), subrun)
    pre.update(F._hashes([row["path"]]))
    if pre[row["path"]] != row["sha256"]:
        raise ValueError("legacy input differs from candidate hash")
    area = Path(area).resolve(strict=True)
    if os.environ.get("MUSE_WORK_DIR") != str(area):
        raise ValueError("wrong MUSE_WORK_DIR for requested user-built dumper")
    plugin = area / "build/al9-prof-e29-p107/CrvFastSim/lib/libcrvfastsim_CrvStepTableDumper_module.so"
    source = area / "CrvFastSim/src/CrvStepTableDumper_module.cc"
    config = area / "CrvFastSim/fcl/dump_gen.fcl"
    if V.sha256(plugin) != PLUGIN_SHA256 or V.sha256(source) != SOURCE_SHA256:
        raise ValueError("dumper differs from user-built and smoke-tested merged artifact")
    if plugin.stat().st_mtime < source.stat().st_mtime:
        raise ValueError("dumper predates source")
    tools = {name: shutil.which(name) for name in ("mu2e", "config_dumper")}
    if not all(tools.values()):
        raise ValueError("mu2e/config_dumper missing from configured environment")
    config_files = F._config_files(config, area)
    approved_configs(F._hashes(config_files))
    files = [source, plugin, Path(sys.executable).resolve(), *map(Path, tools.values()),
             *config_files, Path(__file__).resolve()]
    for module in (genassoc, tables, tables._core, C, F, S, V):
        files.append(Path(module.__file__).resolve())
    driver = Path(__file__).resolve().parents[1] / "scripts/run_generator_bank_capture.sh"
    files.append(driver)
    code = F._hashes(files)
    environment = {k: os.environ.get(k) for k in ("MUSE_WORK_DIR", "FHICL_FILE_PATH", "CET_PLUGIN_PATH",
                                                "LD_LIBRARY_PATH", "MU2E_SEARCH_PATH", "PATH", "PYTHONPATH")}
    env = dict(os.environ, FHICL_FILE_PATH=str(area) + ":" + (environment["FHICL_FILE_PATH"] or ""))
    environment["FHICL_FILE_PATH"] = env["FHICL_FILE_PATH"]
    oldlog = Path(row["path"]).parent / "logs" / f"dumpgen.001400_{subrun:08d}.log"
    history = {"status": "derived exact path; strict payload comparison required"}
    if oldlog.is_file():
        pre.update(F._hashes([oldlog]))
        try:
            F.parse_dump_log(oldlog.read_text(), row["art"])
            history = {"status": "historical successful input log; completeness not inferred", "path": str(oldlog)}
        except ValueError:
            history = {"status": "historical log not successful; strict capture required", "path": str(oldlog)}
    manifest = {"schema": "valhist.generator_bank_capture.v1", "candidate": str(candidate), "source": row,
                "input_sha256": pre, "code_sha256": code, "environment": environment, "area": str(area),
                "backing": str((area / "backing").resolve(strict=True)), "historical_log": history,
                "policy": POLICY, "downstream_authorized": False,
                "art_checksum_policy": "Exact known path only; no namespace queries or art content hashing."}
    C._new_json(out / "inputs.json", manifest)
    F._unchanged(pre)
    F._unchanged(code)
    cfg = out / "original_processes.fcl"
    cfg_run = run_process([tools["config_dumper"], "-P", row["art"]], cfg, area, env)
    target, log = out / "fullgen.root", out / "dump.log"
    prefix = out / "loader"
    command = [tools["mu2e"], "-c", str(config), "-s", row["art"], "-T", str(target)]
    C._new_json(out / "attempt.json", {"command": command, "source": row, "loader_prefix": str(prefix)})
    art_run = run_process(command, log, area, dict(env, LD_DEBUG="libs", LD_DEBUG_OUTPUT=str(prefix)))
    F.parse_dump_log(log.read_text(), row["art"])
    trace = Path(str(prefix) + "." + str(art_run["pid"]))
    loaded = S.loaded_plugin(trace.read_text(), plugin)
    artifacts = F._hashes([target, log, trace, cfg, out / "inputs.json", out / "attempt.json"])
    report = compare_file(row["path"], target, row["run"], row["subrun"], row["events_inferred"])
    result = {**manifest, "report": report, "config_capture": cfg_run, "art_run": art_run,
              "loaded_plugin": loaded, "artifact_sha256": F._unchanged(artifacts),
              "input_sha256_post": F._unchanged(pre), "code_sha256_post": F._unchanged(code),
              "strict_file_certified": True, "whole_bank_certified": False,
              "producer_compatibility_certified": False, "selection_independence_certified": False}
    C._new_json(out / "certificate.json", result)
    print(json.dumps({"subrun": subrun, "events": report["event_count"], "genrows": report["genrows"],
                      "art_seconds": art_run["seconds"], "downstream_authorized": False}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--subrun", required=True, type=int)
    p.add_argument("--area", required=True)
    p.add_argument("--outdir", required=True)
    capture(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
