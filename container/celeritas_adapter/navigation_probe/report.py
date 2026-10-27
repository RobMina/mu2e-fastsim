"""Replay the corrected synthetic suite across two retained runs without rewriting either run."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

import fixtures


FIRST_CASES = ("affine", "constant", "guard")
REMAINING_CASES = ("outer_limit", "record_limit", "tangent", "world_exit", "zero")
IDENTITY_KEYS = ("executable_sha256", "source_manifest_sha256", "runner_sha256", "library_provenance")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify_identity(manifests):
    for key in IDENTITY_KEYS:
        fixtures.require(manifests[0].get(key) and all(m.get(key) == manifests[0][key] for m in manifests),
                         "different/missing build identity: " + key)
    for manifest in manifests:
        fixtures.require(manifest.get("format") == "m2fs-navigation-run-v1" and manifest.get("sanitizer") is True,
                         "native sanitizer run required")
    return {key: manifests[0][key] for key in IDENTITY_KEYS}


def verify_source(directory, manifest):
    source = directory.parent / "replay-source"
    fixtures.require(sha(source / "source-manifest.json") == manifest["source_manifest_sha256"], "source manifest identity")
    pin = read(source / "source-manifest.json")
    for name, digest in pin["sha256"].items():
        fixtures.require(sha(fixtures._local(source, name)) == digest, "changed replay source: " + name)
    fixtures.require(sha(source / "run_probe.py") == manifest["runner_sha256"], "runner identity")


def verify_fixture(directory, reference, case):
    request = read(reference / case["request"])
    for name in (case["request"], case["expected"], request["map_file"], request["geometry_file"]):
        fixtures.require(sha(fixtures._local(directory, name)) == sha(reference / name), "not the revised fixture: " + name)


def verify_outer_revision(first, remaining):
    old = read(first / "outer_limit.request.json")
    new = read(remaining / "outer_limit.request.json")
    fixtures.require(old["tracks"][0]["max_step_mm"] == 100.0 and new["tracks"][0]["max_step_mm"] == 4.0,
                     "unexpected outer-limit trial-cap revision")
    old["tracks"][0]["max_step_mm"] = 4.0
    fixtures.require(old == new, "outer-limit revision changed more than trial cap")
    old_expected = read(first / "outer_limit.expected.json")
    new_expected = read(remaining / "outer_limit.expected.json")
    old_expected["request_sha256"] = new_expected["request_sha256"]
    fixtures.require(old_expected == new_expected, "outer-limit oracle/budgets changed")


def report(first, remaining):
    directories = [Path(first).resolve(), Path(remaining).resolve()]
    fixtures.require(directories[0] != directories[1], "need distinct original and continuation runs")
    selections = [FIRST_CASES, REMAINING_CASES]
    manifests = [read(path / "run-manifest.json") for path in directories]
    identity = verify_identity(manifests)
    failures = [run for run in manifests[0]["runs"] if run.get("accepted_process") is not True]
    fixtures.require(len(failures) == 1 and failures[0]["request"] == "outer_limit.request.json",
                     "expected preserved original outer-limit failure")
    fixtures.require(all(run.get("accepted_process") is True for run in manifests[1]["runs"]), "continuation failure")
    for directory, manifest in zip(directories, manifests):
        verify_source(directory, manifest)
    verify_outer_revision(*directories)
    first_log = fixtures._local(directories[0], failures[0]["log"])
    fixtures.require(sha(first_log) == failures[0]["log_sha256"] and "M2FS_NAV_MAP_GUARD" in first_log.read_text(),
                     "original failure log changed")
    fixtures.require(sha(fixtures._local(directories[0], failures[0]["sanitizer_log"])) == failures[0]["sanitizer_sha256"],
                     "original failure sanitizer log changed")
    checks = [fixtures.check(path, cases) for path, cases in zip(directories, selections)]
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        reference = Path(temporary) / "revised"
        index = fixtures.prepare(reference)
        case_by_id = {case["id"]: case for case in index["cases"]}
        coverage = [case for selection in selections for case in selection]
        fixtures.require(len(coverage) == len(set(coverage)) and set(coverage) == set(case_by_id), "incomplete/duplicate revised coverage")
        for directory, selection, check in zip(directories, selections, checks):
            for case_id in selection:
                case = case_by_id[case_id]
                verify_fixture(directory, reference, case)
                if case["kind"] == "guard":
                    continue
                request = read(directory / case["request"])
                result = read(directory / case["result"])
                track_ids = {track["id"] for track in request["tracks"]}
                for backend in ("host", "device"):
                    values = [row for row in check["metrics"] if row["id"] in track_ids and row["backend"] == backend]
                    outputs = result["tracks"][backend]
                    fixtures.require(len(values) == len(request["tracks"]) == len(outputs), "incomplete backend coverage")
                    row = dict(case=case_id, kind=case["kind"], backend=backend, tracks=len(values),
                               steps=sum(track["count"] for track in outputs), crossings=sum(track["crossings"] for track in outputs),
                               terminal_statuses=sorted({track["status"] for track in outputs}))
                    for key in ("position_max_mm", "direction_max", "crossing_path_max_mm", "crossing_position_max_mm", "face_max_mm"):
                        row[key] = float(np.max([value[key] for value in values]))
                    rows.append(row)
    original = dict(directory=str(directories[0]), run_manifest_sha256=sha(directories[0] / "run-manifest.json"),
                    accepted_process=False, failed_case="outer_limit", trial_cap_mm=100.0,
                    returncode=failures[0]["returncode"], preserved=True)
    return dict(format="m2fs-navigation-combined-replay-v1", revised_suite_pass=True, single_run_pass=False,
                scope="synthetic field plus navigation only; no material/scoring/real-Mu2e or performance qualification",
                original_failure=original, revision=dict(case="outer_limit", key="max_step_mm", before=100.0, after=4.0,
                                                        other_request_fields_and_budgets_unchanged=True),
                identity=identity, checker_sha256=sha(Path(fixtures.__file__)), reporter_sha256=sha(Path(__file__)),
                runs=[dict(directory=str(path), run_manifest_sha256=sha(path / "run-manifest.json"), selected_cases=list(cases),
                           replay=check) for path, cases, check in zip(directories, selections, checks)],
                counts=dict(fixture_cases=len(coverage), positive_track_backend_checks=sum(row["tracks"] for row in rows if row["kind"] == "positive"),
                            exhaustion_track_backend_checks=sum(row["tracks"] for row in rows if row["kind"] != "positive"),
                            guard_process_checks=sum(row["kind"] == "guard" for check in checks for row in check["metrics"]),
                            crossings=sum(row["crossings"] for row in rows), sanitizer_clean_cases=len(rows) // 2),
                cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_results", type=Path)
    parser.add_argument("continuation_results", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.original_results, args.continuation_results), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
