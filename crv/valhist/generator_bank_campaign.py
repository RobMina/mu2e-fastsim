"""Versioned conditional species/KE-only capture campaign; never publish a denominator.

Each host runs its assigned files sequentially. Reuse the two completed captures,
retain every failed attempt, and validate full censuses one partition at a time.
The completion receipt distinguishes observations from explicitly accepted assumptions.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import socket

from . import generator_bank_capture as B, generator_bank_provenance as P
from . import generator_seed_report as R
from . import generator_capture_verify as V, generator_history as H
from . import genassoc_campaign as C, genassoc_fullgen as F

SCHEMA = "valhist.generator_bank_campaign.v1"
SCOPE = "absolute-PDG/native-KE counts and full-census generated-event exposure only"
ASSUMPTIONS = [
    "Host assignment and lifetime-related missingness are independent of generated species/KE content.",
    "Distinct CRY generator seed streams are independent for species/KE count statistics.",
]
LIMITS = [
    "No unconditional unbiasedness or historical RNG-state certification.",
    "Only 65 scheduler final ads are retained; failed-worker progress logs are unavailable.",
    "Joint timing/energy independence and nonlinear response are outside scope.",
    "Tail, rounded reach, saturation, donor uncertainty and nonlinear DQM remain separate gates.",
    "No library overwrite, asset alias/default promotion, build, grid submission or commit is authorized.",
]
TASK = "R2 crv-tier1b-descendants; conditional94 strict captures; r2_cond94_20261027"
LOCKS = Path("/exp/mu2e/data/users/ram2aq/agent_locks")
DATA = Path("/exp/mu2e/data/users/ram2aq")
LOG_TEMPLATE = ("/pnfs/mu2e/scratch/users/ram2aq/crv_fastsim/crygen_v1/logs/"
                "mu2e.001400_{subrun:08d}.log.gz")
RESERVE_BYTES = 4 * 1024**3
PER_FILE_BUDGET = 96 * 1024**2


def read(path):
    return json.loads(Path(path).read_text())


def merge_hashes(*closures):
    result = {}
    for closure in closures:
        for path, digest in closure.items():
            if path in result and result[path] != digest:
                raise ValueError("conflicting campaign hash binding: " + path)
            result[path] = digest
    return result


def probe_hashes(probes):
    closures = []
    for probe in probes:
        cert = read(probe["path"])
        closures.extend([F._hashes([probe["path"]]), cert["input_sha256"], cert["log_sha256"]])
    result = merge_hashes(*closures)
    F._unchanged(result)
    return result


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    modules = (B, P, V, H, R, R.P, C, F, B.S, B.V, B.genassoc, B.tables, B.tables._core)
    return F._hashes([Path(__file__).resolve(), *(Path(m.__file__).resolve() for m in modules),
                      root / "scripts/run_generator_bank_capture.sh",
                      root / "scripts/run_generator_bank_campaign.sh"])


def quota(required):
    used = int(os.getxattr(DATA, "ceph.dir.rbytes"))
    limit = int(os.getxattr(DATA, "ceph.quota.max_bytes"))
    if limit <= 0 or used < 0 or limit - used < required + RESERVE_BYTES:
        raise ValueError(f"insufficient quota: used={used} limit={limit} required={required} reserve={RESERVE_BYTES}")
    return {"used_bytes": used, "limit_bytes": limit, "required_bytes": required,
            "reserve_bytes": RESERVE_BYTES}


def schedule(candidate, hosts, reused):
    B.select_record(candidate, 2001)
    if not hosts or hosts != sorted(set(hosts)) or any(h not in {f"mu2egpvm{i:02d}" for i in range(1, 8)} for h in hosts):
        raise ValueError("require distinct allowed GPVM hosts")
    if set(reused) != {2001, 2003}:
        raise ValueError("reuse exactly the two completed historical captures")
    pending = sorted(r["subrun"] for r in candidate["files"] if r["subrun"] not in reused)
    return {host: pending[i::len(hosts)] for i, host in enumerate(hosts)}


def validate(plan):
    if (plan.get("schema") != SCHEMA or plan.get("scope") != SCOPE or plan.get("assumptions") != ASSUMPTIONS
            or plan.get("downstream_authorized") is not False or plan.get("capture_authorized") is not True
            or plan.get("selection_independence_certified") is not False):
        raise ValueError("invalid conditional campaign contract")
    candidate = read(plan["candidate"])
    reused = {int(k): v for k, v in plan["reused"].items()}
    expected = schedule(candidate, list(plan["assignments"]), reused)
    if plan["assignments"] != expected:
        raise ValueError("campaign assignments differ from exact inventory partition")
    root = Path(plan["outdir"])
    if (not root.is_absolute() or root != root.resolve() or not root.is_relative_to(DATA.resolve())
            or plan["limits"] != LIMITS or not plan.get("authorization")):
        raise ValueError("invalid campaign location/limitations/authorization")
    return candidate


def prepare(candidate, history, seed_report, certificates, area, hosts, outdir):
    out = B.V._new_output(outdir)
    if not out.is_relative_to(DATA):
        raise ValueError("campaign outputs must be in the data area")
    candidate, history, seed_report, area = (Path(p).resolve(strict=True) for p in (candidate, history, seed_report, area))
    source = read(candidate)
    reused = {}
    for name in certificates:
        certpath = Path(name).resolve(strict=True)
        cert = read(certpath)
        subrun = cert["source"]["subrun"]
        if subrun in reused or Path(cert["candidate"]) != candidate or Path(cert["area"]) != area:
            raise ValueError("duplicate or incompatible reused capture")
        reused[subrun] = str(certpath)
    assignments = schedule(source, hosts, reused)
    hist = read(history)
    if (hist.get("downstream_authorized") is not False or hist.get("selection_independence_certified") is not False
            or hist.get("cluster") != H.CLUSTER or hist.get("submitted_jobs") != 100
            or hist.get("removed_subruns") != sorted(B.MISSING) or hist.get("failure_hosts") != ["fnpc9029.fnal.gov"]
            or len(hist.get("retained_workers", [])) != 94 or len(hist.get("scheduler_records", [])) != 65
            or hist.get("failed_worker_logs_present") is not False):
        raise ValueError("unexpected historical evidence contract")
    if hist["input_sha256"] != hist["input_sha256_post"] or hist["input_sha256"].get(str(candidate)) != B.V.sha256(candidate):
        raise ValueError("history/candidate binding drift")
    F._unchanged(hist["input_sha256"])
    seed = read(seed_report)
    if (seed.get("downstream_authorized") is not False or seed.get("historical_rng_state_certified") is not False
            or seed["input_sha256"] != seed["input_sha256_post"]
            or seed["input_sha256"].get(str(candidate)) != B.V.sha256(candidate)):
        raise ValueError("seed evidence must remain bound and nonauthorizing")
    F._unchanged(seed["input_sha256"])
    probes = [R.read_probe(r["path"]) for r in seed["probes"]]
    if (probes != seed["probes"] or {(r["mode"], r["subrun"]) for r in probes}
            != {("crygen", 2001), ("crygen", 2003), ("reference", 2)}
            or any(r["offsets"] != {"cosmicTimeOffset": 0, "g4run": 1, "generate": 2} for r in probes)):
        raise ValueError("unexpected seed probe evidence")
    code = code_hashes()
    pre = merge_hashes(F._hashes([candidate, history, seed_report, *reused.values()]),
                       hist["input_sha256"], seed["input_sha256"], probe_hashes(probes))
    space = quota(92 * PER_FILE_BUDGET)
    out.mkdir()
    checked = V.verify(list(reused.values()), out / "reused_verification.json")
    pre.update(F._hashes([out / "reused_verification.json"]))
    pre.update(checked["input_sha256"])
    root = candidate.parent
    reference = {"config": str(root / "crygen_provenance_v1/g4ref_2.fcl"),
                 "log": str(root / "crygen_provenance_v1/g4ref_2_generation.log")}
    pre.update(F._hashes(reference.values()))
    plan = {"schema": SCHEMA, "outdir": str(out), "candidate": str(candidate), "area": str(area),
            "history": str(history), "seed_report": str(seed_report), "reference": reference,
            "reused": {str(k): v for k, v in reused.items()}, "assignments": assignments,
            "scope": SCOPE, "assumptions": ASSUMPTIONS, "limits": LIMITS,
            "authorization": "User explicitly approved conditional denominator-only scope and remaining 92 strict captures/producer checks on 2026-10-27.",
            "capture_authorized": True, "selection_independence_certified": False,
            "downstream_authorized": False, "quota_at_preparation": space,
            "input_sha256": pre, "input_sha256_post": F._unchanged(pre),
            "code_sha256": code, "code_sha256_post": F._unchanged(code)}
    validate(plan)
    P.write_report(out / "manifest.json", plan)
    print(json.dumps({"manifest": str(out / "manifest.json"), "sha256": B.V.sha256(out / "manifest.json"),
                      "assignments": assignments}), flush=True)
    return plan


def campaign_inputs(plan, manifest):
    return merge_hashes(plan["input_sha256"], plan["code_sha256"], F._hashes([manifest]))


def load_plan(manifest, manifest_sha256):
    manifest = Path(manifest).resolve(strict=True)
    if B.V.sha256(manifest) != manifest_sha256:
        raise ValueError("campaign manifest differs from launch digest")
    plan = read(manifest)
    validate(plan)
    if manifest != Path(plan["outdir"]) / "manifest.json":
        raise ValueError("manifest must reside in its output directory")
    if code_hashes() != plan["code_sha256"]:
        raise ValueError("executing code differs from campaign code closure")
    if (plan["input_sha256"] != plan["input_sha256_post"]
            or plan["code_sha256"] != plan["code_sha256_post"]):
        raise ValueError("campaign preparation input/code drift")
    F._unchanged(campaign_inputs(plan, manifest))
    return manifest, plan


def certificate_path(plan, subrun):
    return (Path(plan["reused"][str(subrun)]) if str(subrun) in plan["reused"] else
            Path(plan["outdir"]) / f"capture_{subrun}/certificate.json")


def captured_path(plan, subrun):
    return certificate_path(plan, subrun).parent


def check_certificate(plan, subrun):
    path = certificate_path(plan, subrun)
    cert = read(path)
    expected = B.select_record(read(plan["candidate"]), subrun)
    if (cert.get("candidate") != plan["candidate"] or cert.get("area") != plan["area"]
            or cert.get("source") != expected or cert.get("strict_file_certified") is not True
            or cert.get("downstream_authorized") is not False or cert["report"]["event_count"] != 1_000_000):
        raise ValueError("capture differs from exact campaign candidate/area/partition")
    return cert


def producer(plan, subrun):
    out = Path(plan["outdir"]) / f"producer_{subrun}"
    out.mkdir()
    log = out / "generation.log.gz"
    with open(LOG_TEMPLATE.format(subrun=subrun), "rb") as source, log.open("xb") as target:
        while block := source.read(1024 * 1024):
            target.write(block)
    gzip.decompress(log.read_bytes()).decode("utf-8", errors="strict")
    report = P.compare(captured_path(plan, subrun) / "original_processes.fcl", log,
                       plan["reference"]["config"], plan["reference"]["log"])
    if report["configurations"]["crygen"]["subrun"] != subrun:
        raise ValueError("producer report partition mismatch")
    P.write_report(out / "report.json", report)
    return str(out / "report.json")


def run(manifest, manifest_sha256, host):
    manifest, plan = load_plan(manifest, manifest_sha256)
    if socket.gethostname().split(".")[0] != host or host not in plan["assignments"]:
        raise ValueError("wrong host/assignment")
    if (LOCKS / host / "task").read_text().strip() != TASK:
        raise ValueError("missing matching R2 host lock")
    pre = campaign_inputs(plan, manifest)
    F._unchanged(pre)
    out = Path(plan["outdir"]) / host
    out.mkdir()
    C._new_json(out / "attempt.json", {"host": host, "subruns": plan["assignments"][host], "manifest_sha256": pre[str(manifest)]})
    records = []
    try:
        for subrun in plan["assignments"][host]:
            space = quota(PER_FILE_BUDGET)
            F._unchanged(plan["code_sha256"])
            print(f"begin {host} subrun={subrun}", flush=True)
            capture = B.capture(plan["candidate"], subrun, plan["area"], captured_path(plan, subrun))
            record = {"subrun": subrun, "certificate": str(captured_path(plan, subrun) / "certificate.json"),
                      "producer": producer(plan, subrun), "events": capture["report"]["event_count"],
                      "genrows": capture["report"]["genrows"], "quota": space}
            records.append(record)
            C._new_json(out / f"completed_{subrun}.json", record)
            print(json.dumps({"completed": subrun, "host": host, "files_done": len(records)}), flush=True)
        result = {"host": host, "records": records, "input_sha256": pre,
                  "input_sha256_post": F._unchanged(pre), "downstream_authorized": False}
        C._new_json(out / "complete.json", result)
        return result
    except BaseException as exc:
        C._new_json(out / "failed.json", {"error": str(exc), "exception": type(exc).__name__,
                                         "completed": records, "downstream_authorized": False})
        raise


def aggregate(manifest, manifest_sha256):
    manifest, plan = load_plan(manifest, manifest_sha256)
    source = validate(plan)
    out = Path(plan["outdir"]) / "aggregate"
    out.mkdir()
    expected_inputs = campaign_inputs(plan, manifest)
    bound = dict(expected_inputs)
    F._unchanged(bound)
    for host, assigned in plan["assignments"].items():
        path = Path(plan["outdir"]) / host / "complete.json"
        done = read(path)
        if (done.get("host") != host or done.get("downstream_authorized") is not False
                or [r["subrun"] for r in done["records"]] != assigned
                or done["input_sha256"] != expected_inputs or done["input_sha256_post"] != expected_inputs):
            raise ValueError("missing/inconsistent worker completion")
        for record in done["records"]:
            subrun = record["subrun"]
            cert = check_certificate(plan, subrun)
            if (record["certificate"] != str(certificate_path(plan, subrun))
                    or record["producer"] != str(Path(plan["outdir"]) / f"producer_{subrun}/report.json")
                    or record["events"] != cert["report"]["event_count"]
                    or record["genrows"] != cert["report"]["genrows"]):
                raise ValueError("worker record differs from planned capture")
        bound.update(F._hashes([path]))
    for subrun in sorted(map(int, plan["reused"])):
        producer(plan, subrun)
    certs, producers = [], []
    for row in sorted(source["files"], key=lambda r: r["subrun"]):
        subrun = row["subrun"]
        check_certificate(plan, subrun)
        certs.append(certificate_path(plan, subrun))
        path = Path(plan["outdir"]) / f"producer_{subrun}/report.json"
        report = read(path)
        if (report.get("status") != "observed_pair_matches" or report.get("downstream_authorized") is not False
                or report["configurations"]["crygen"]["subrun"] != subrun):
            raise ValueError("producer status/partition mismatch")
        config, log = captured_path(plan, subrun) / "original_processes.fcl", path.parent / "generation.log.gz"
        current = P.compare(config, log, plan["reference"]["config"], plan["reference"]["log"])
        if report != current:
            raise ValueError("producer evidence changed")
        bound.update(F._hashes([path, log, config]))
        producers.append(str(path))
    checked = V.verify(certs, out / "capture_verification.json")
    if len(checked["captures"]) != 94 or checked["event_count"] != 94_000_000:
        raise ValueError("full bank strict census incomplete")
    bound.update(checked["input_sha256"])
    bound.update(F._hashes([out / "capture_verification.json"]))
    result = {"schema": SCHEMA, "scope": SCOPE, "assumptions": ASSUMPTIONS, "limits": LIMITS,
              "manifest": str(manifest), "strict_event_census_certified": True,
              "observed_producer_pairs_match": True, "files": 94, "events": checked["event_count"],
              "genrows": sum(r["report"]["genrows"] for r in checked["captures"]),
              "producers": producers, "selection_independence_certified": False,
              "conditional_capture_evidence_complete": True, "bank_admitted_for_propagation": False,
              "downstream_authorized": False, "input_sha256": bound, "input_sha256_post": F._unchanged(bound)}
    C._new_json(out / "report.json", result)
    print(json.dumps({k: result[k] for k in ("files", "events", "genrows", "downstream_authorized")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for key in ("candidate", "history", "seed-report", "area", "outdir"):
        prep.add_argument("--" + key, required=True)
    prep.add_argument("--certificates", nargs="+", required=True)
    prep.add_argument("--hosts", nargs="+", required=True)
    worker = commands.add_parser("run")
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--manifest-sha256", required=True)
    worker.add_argument("--host", required=True)
    finish = commands.add_parser("aggregate")
    finish.add_argument("--manifest", required=True)
    finish.add_argument("--manifest-sha256", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "run": run, "aggregate": aggregate}[command](**args)


if __name__ == "__main__":
    main()
