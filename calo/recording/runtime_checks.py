"""Run bounded built-plugin checks with no scheduled Geant4 event processing."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import importlib.util

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        h = hashlib.file_digest(stream, "sha256")
    return h.hexdigest()


def run(command, log, timeout=180):
    with log.open("xb") as stream:
        try:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, cwd=log.parent)
        except subprocess.TimeoutExpired:
            return 124
    return process.returncode


def check_environment(overlay, build_receipt):
    report = json.loads(build_receipt.read_text())
    if os.environ.get("MUSE_WORK_DIR") != str(overlay) or os.environ.get("MUSE_ENVSET") != "p110":
        raise ValueError("wrong built workDir/environment")
    if report.get("source_manifest_sha256") != digest(HERE / "current_head/source-manifest.json"):
        raise ValueError("source manifest changed after the verified user build; rebuild and reverify first")
    if report.get("overlay_receipt_sha256") != digest(overlay / "overlay-receipt.json"):
        raise ValueError("overlay pins changed after verified build")
    if report["work_dir"] != str(overlay):
        raise ValueError("build receipt belongs to another area")
    for name, entry in report["binaries"].items():
        path = overlay / "build/al9-prof-e29-p110/Offline/lib" / name
        if digest(path) != entry["sha256"]:
            raise ValueError("verified binary changed: " + name)
    return report


def fixture_validator():
    spec = importlib.util.spec_from_file_location("fixture_manifest_validator", HERE / "manifest/validate_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_manifest_fixture(out, artifact, source_log, mode):
    validator = fixture_validator()
    pins = HERE / "manifest/source_manifest.json"
    validator.check_snapshot()
    source_sha = digest(artifact)
    if source_sha != "5f3dda4f8645e2f8b802bdef5b9993af4a002250456ee8a71de2142d33dd1271":
        raise ValueError("fixture requires the immutable audited stopped-muon pool")
    triples = re.findall(r"Begin processing the .*? record\. run: (\d+) subRun: (\d+) event: (\d+)", source_log.read_text())
    if len(triples) < 3:
        raise ValueError("independent source copy log does not identify first three input EventIDs")
    expected = [[int(value) for value in triple] for triple in triples[:3]]
    (out / "expected_event_ids.json").write_text(json.dumps(expected) + "\n")
    template = (HERE / "manifest_fixture.fcl").read_text()
    if '  filters : {' not in template or '  analyzers : {' in template:
        raise ValueError("manifest writers must be trigger-path filters")
    if mode in ("empty", "strict_empty"):
        template = template.replace("module_type : CeEndpoint", "module_type : Pileup")
        template = template.replace("      pdgId : 11\n", "      captureProducts : []\n      decayProducts : []\n")
    elif mode == "missing":
        template = template.replace("generate::CaloManifestFixture", "missing::MuBeamResampler")
    elif mode == "generator_error":
        template = template.replace("TargetStopFilter::MuBeamResampler", "IPAStopFilter::MuBeamResampler")
    if mode in ("strict", "strict_empty"):
        template = template.replace('rootMode : "record_all"', 'rootMode : "require_one"')
    for key, value in {
        "REVIEW_ROOT_OUTPUT": str(out / "incident_manifest.root"),
        "REVIEW_SOURCE_ID": "immutable-stopped-muon-pool-native-fixture",
        "REVIEW_SOURCE_SHA256": source_sha,
        "REVIEW_SNAPSHOT_SHA256": digest(pins),
        "REVIEW_NORMALIZED_SHA256": "0" * 64,
    }.items():
        template = template.replace(key, value)
    fcl = out / "executed.fcl"
    fcl.write_text(template)
    expanded = out / "draft.expanded.fcl"
    if run(["fhicl-dump", "-c", str(fcl)], expanded) != 0:
        raise ValueError("manifest fixture expansion failed")
    normalized = validator.normalized_fcl_sha256(expanded)
    final = out / "executed.expanded.fcl"
    text = expanded.read_text()
    old = 'expandedFclSha256: "' + "0" * 64 + '"'
    if text.count(old) != 2:
        raise ValueError("expanded config does not contain exactly two digest placeholders")
    final.write_text(text.replace(old, 'expandedFclSha256: "' + normalized + '"'))
    if validator.normalized_fcl_sha256(final) != normalized:
        raise ValueError("normalized FCL hash did not stabilize")
    return final, {"artifact_sha256": source_sha, "snapshot_sha256": digest(pins),
                   "independent_source_log_sha256": digest(source_log), "expected_event_ids": expected,
                   "expanded_sha256": digest(final), "normalized_expanded_sha256": normalized,
                   "fixture_mode": mode, "generator_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("describe", "geometry", "manifest"))
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--source-log", type=Path)
    parser.add_argument("--fixture", choices=("single", "strict", "empty", "strict_empty", "missing", "generator_error"), default="single")
    args = parser.parse_args()
    overlay = args.overlay.resolve(strict=True)
    report = check_environment(overlay, args.build_receipt)
    args.out.mkdir(parents=True, exist_ok=False)
    record = {"mode": args.mode, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "build_receipt_sha256": digest(args.build_receipt), "work_dir": str(overlay),
              "transport_requested": False, "results": {}}
    (args.out / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.mode == "describe":
        for module in ("CaloInputEventManifest", "CaloIncidentManifest", "CaloShowerStepMaker", "Mu2eG4"):
            result = run(["mu2e", "--print-description", module], args.out / (module + ".txt"))
            record["results"][module] = result
    elif args.mode == "manifest":
        if args.artifact is None or args.source_log is None:
            parser.error("manifest fixture requires --artifact and --source-log")
        fcl, claims = prepare_manifest_fixture(args.out, args.artifact, args.source_log, args.fixture)
        record.update(claims)
        record["results"]["manifest_job"] = run(["mu2e", "-c", str(fcl), "-s", str(args.artifact), "-n", "3"],
                                                args.out / "manifest.log")
        record["source_unchanged"] = digest(args.artifact) == claims["artifact_sha256"]
        record["root_file_present"] = (args.out / "incident_manifest.root").is_file()
        record["complete_manifest_certified"] = False
    else:
        template = (HERE / "zero_transport.fcl").read_text()
        if template.count("REVIEW_OUTPUT_PREFIX") != 1:
            raise ValueError("zero-transport template drift")
        text = template.replace("REVIEW_OUTPUT_PREFIX", str(args.out / "native"))
        fcl = args.out / "executed.fcl"
        fcl.write_text(text)
        expanded = args.out / "executed.expanded.fcl"
        result = run(["fhicl-dump", "-c", str(fcl)], expanded)
        record["results"]["expand"] = result
        if result == 0:
            record["expanded_sha256"] = digest(expanded)
            result = run(["mu2e", "-c", str(expanded), "-n", "1"], args.out / "geometry.log", timeout=240)
            record["results"]["native_initialize"] = result
        record["template_sha256"] = digest(HERE / "zero_transport.fcl")
    record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record["all_commands_exit_zero"] = all(value == 0 for value in record["results"].values())
    record["native_geometry_certified"] = False
    record["physics_validated"] = False
    (args.out / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record["all_commands_exit_zero"] else 1)


if __name__ == "__main__":
    main()
