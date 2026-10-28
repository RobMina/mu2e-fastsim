"""Bind retrieved scheduler/worker evidence to the retained candidate, without authorizing it."""
import argparse
import json
from pathlib import Path
import re
import shlex

from . import genassoc_campaign as C, genassoc_fullgen as F, generator_bank_capture as B

CLUSTER = 87215765


def worker(text, proc):
    starts = re.findall(r"=== grid_crygen: shard (\d+) subrun (\d+) seed (\d+) nevts (\d+) host (\S+)", text)
    expected = (str(proc), str(2001 + proc), str(20263000 + proc), "1000000")
    if len(starts) != 1 or starts[0][:4] != expected:
        raise ValueError("worker identity/configuration mismatch")
    exits = re.findall(r"grid_crygen.sh COMPLETED with exit status (-?\d+)", text)
    done = re.findall(r"=== grid_crygen: done shard (\d+)\b", text)
    totals = re.findall(r"TrigReport Events total = (\d+) passed = (\d+) failed = (\d+)", text)
    if exits != ["0"] or done != [str(proc)] or totals != [("1000000", "1000000", "0")]:
        raise ValueError("worker lacks complete successful generation evidence")
    return {"proc": proc, "subrun": 2001 + proc, "seed": 20263000 + proc, "host": starts[0][4],
            "logged_events": 1000000, "physics_table_census_certified": False}


def history(records):
    seen, result = set(), []
    for r in records:
        proc = r["ProcId"]
        if r["ClusterId"] != CLUSTER or r["Owner"] != "ram2aq" or proc in seen or not 0 <= proc < 100:
            raise ValueError("wrong/duplicate scheduler identity")
        seen.add(proc)
        env = dict(word.split("=", 1) for word in shlex.split(r["Environment"]))
        expected = {"NEVTS": "1000000", "BASESEED": "20263000", "SUBRUN0": "2001",
                    "MUSING": "MDC2025ay", "PROCESS": str(proc)}
        if any(env.get(k) != v for k, v in expected.items()):
            raise ValueError("scheduler environment mismatch")
        status = r["JobStatus"]
        if status == 3:
            if (not re.fullmatch(r"\s*SYSTEM_PERIODIC_HOLD\s+Run Time/limit 1085[23]/1\.080000000000000E\+04",
                                r.get("LastHoldReason", ""))
                    or r.get("RemoveReason") != "via condor_rm (by user ram2aq)" or r.get("NumJobStarts") != 1):
                raise ValueError("removed job lacks expected hold/start/removal evidence")
        elif status != 4 or r.get("ExitCode") != 0:
            raise ValueError("unexpected scheduler completion")
        wall = r["RemoteWallClockTime"]
        cpu = r["RemoteUserCpu"] + r["RemoteSysCpu"]
        if wall <= 0 or cpu < 0:
            raise ValueError("invalid scheduler resource accounting")
        result.append({"proc": proc, "subrun": 2001 + proc, "status": status,
                       "starts": r.get("NumJobStarts"), "host": r["LastRemoteHost"].split("@")[-1],
                       "wall_seconds": wall, "cpu_seconds": cpu, "cpu_wall_fraction": cpu / wall,
                       "last_hold": r.get("LastHoldReason"), "remove_reason": r.get("RemoveReason")})
    if {r["subrun"] for r in result if r["status"] == 3} != B.MISSING:
        raise ValueError("removed job inventory does not match the six omitted subruns")
    return sorted(result, key=lambda r: r["proc"])


def run(history_file, archive, candidate, out):
    output = Path(out)
    if output.exists():
        raise FileExistsError(output)
    archive = Path(archive)
    paths = [Path(history_file), Path(candidate), archive / "simple.cmd", archive / "grid_crygen.sh"]
    logs = sorted(archive.glob(f"*cluster.{CLUSTER}.*.out"))
    bound = F._hashes([*paths, *logs, __file__, B.__file__, F.__file__, C.__file__])
    ads = history(json.loads(Path(history_file).read_text()))
    source = json.loads(Path(candidate).read_text())
    B.select_record(source, 2001)
    submit = (archive / "simple.cmd").read_text()
    if re.findall(r"(?im)^queue\s+(\d+)\s*$", submit) != ["100"]:
        raise ValueError("original submit does not queue exactly 100 jobs")
    for setting in ("NEVTS=1000000", "BASESEED=20263000", "SUBRUN0=2001", "MUSING=MDC2025ay"):
        if setting not in submit:
            raise ValueError("missing original submission setting")
    workers = []
    for path in logs:
        match = re.search(rf"cluster\.{CLUSTER}\.(\d+)\.out$", path.name)
        workers.append({**worker(path.read_text(), int(match[1])), "path": str(path)})
    retained = {r["subrun"] for r in source["files"]}
    if len(workers) != 94 or {r["subrun"] for r in workers} != retained:
        raise ValueError("worker successes differ from exact candidate inventory")
    if any((r["status"] == 4) != (r["subrun"] in retained) for r in ads):
        raise ValueError("scheduler/table outcome contradiction")
    failures = [r for r in ads if r["status"] == 3]
    result = {"schema": "valhist.generator_history.v1", "cluster": CLUSTER, "submitted_jobs": 100,
              "retained_workers": workers, "scheduler_records": ads,
              "missing_scheduler_procs": sorted(set(range(100)) - {r["proc"] for r in ads}),
              "removed_subruns": sorted(r["subrun"] for r in failures),
              "failure_hosts": sorted({r["host"] for r in failures}),
              "failed_worker_logs_present": any(list(archive.glob(f"*cluster.{CLUSTER}.{s - 2001}.*")) for s in B.MISSING),
              "input_sha256": bound, "input_sha256_post": F._unchanged(bound),
              "downstream_authorized": False, "selection_independence_certified": False,
              "conclusion": "All omissions match recorded one-start lifetime holds and user removals; common host and low CPU/wall support infrastructure slowness.",
              "limits": ["Final ads are not a complete retry/event history; missing scheduler records are explicit.",
                         "Successful worker event logs do not replace schema-v2 event censuses.",
                         "Absent failed-worker progress cannot exclude every content-dependent runtime contribution.",
                         "Conditional use requires an explicit host-assignment/missingness assumption; no automatic admission."]}
    C._new_json(output, result)
    print(json.dumps({"successes": len(workers), "scheduler_records": len(ads),
                      "failure_hosts": result["failure_hosts"], "removed_subruns": result["removed_subruns"]}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("history-file", "archive", "candidate", "out"):
        p.add_argument("--" + name, required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
