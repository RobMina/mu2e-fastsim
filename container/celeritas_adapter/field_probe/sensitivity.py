"""Prepare/check h6b control-isolation fixtures for the existing h6 executable; no builds."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import oracle


EPSILONS = (1e-4, 1e-6, 1e-8, 1e-10)
CHORDS_MM = (0.5, 0.02, 0.005, 0.0005)
CAPS_MM = (16.0, 8.0, 4.0, 1.0)
GROUPS = ("p5_q-1_L600", "p5_q+1_L600", "p25_q-1_L600", "p25_q+1_L600")
FORMAT = "m2fs-field-sensitivity-fixtures-v1"
GATES = dict(position_mm=2e-5, direction=2e-6, fine_to_coarse=0.1, floor_mm=1e-7)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variants():
    """Independent axes and four crossed settings; thresholds are fixed before execution."""
    result = [("epsilon", 16.0, 20.0, eps) for eps in EPSILONS]
    result += [("chord", 16.0, chord, 1e-3) for chord in CHORDS_MM]
    result += [("cap", cap, 20.0, 1e-4) for cap in CAPS_MM]
    result += [("cross", 16.0, chord, eps) for chord in (0.5, 0.005) for eps in (1e-4, 1e-10)]
    return result


def build_case(family, resolution):
    data, base, expected = oracle.build_case(family, resolution)
    first = {}
    for track, meta in zip(base["tracks"], expected["tracks"]):
        first.setdefault(meta["group"], (track, meta))
    tracks, metadata, controls = [], [], []
    for group in GROUPS:
        initial, meta = first[group]
        for axis, cap, chord, eps in variants():
            identity = f"{group}_{axis}_cap{cap:g}_chord{chord:g}_eps{eps:g}"
            track = dict(initial, id=identity, method="adaptive", max_step_mm=cap,
                         delta_chord_mm=chord, epsilon_rel_max=eps)
            tracks.append(track)
            metadata.append(dict(meta, id=identity, group=f"{group}/{axis}", sweep="adaptive"))
            controls.append(dict(id=identity, physical_group=group, axis=axis))
    for group in ("p5_q+0_L600", "p25_q+0_L600", "p105_q+0_L600"):
        initial, meta = first[group]
        identity = f"{group}_neutral"
        tracks.append(dict(initial, id=identity, method="adaptive", max_step_mm=16.0,
                           delta_chord_mm=20.0, epsilon_rel_max=1e-10))
        metadata.append(dict(meta, id=identity, group=group, sweep="adaptive"))
        controls.append(dict(id=identity, physical_group=group, axis="neutral"))
    request = dict(base, tracks=tracks)
    expected = dict(expected, request_sha256=oracle.digest(request), tracks=metadata)
    oracle.validate_request(request)
    oracle.require(len(tracks) == 67, "Unexpected sensitivity track count")
    return data, request, expected, controls


def prepare(directory):
    directory = Path(directory)
    oracle.require(not directory.exists(), "prepare requires NEW_OUTPUT")
    cases = [(f, r, build_case(f, r)) for f in oracle.FAMILIES for r in oracle.RESOLUTIONS]
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for family, resolution, (data, request, expected, controls) in cases:
        stem = f"{family}-{resolution}"
        entry = dict(name=stem, map=f"{stem}.map.json", request=f"{stem}.request.json",
                     expected=f"{stem}.expected.json", result=f"{stem}.result.json", controls=controls)
        for key, value in (("map", data), ("request", request), ("expected", expected)):
            oracle.write_json(directory / entry[key], value)
        entries.append(entry)
    index = dict(format=FORMAT, cases=entries, gates=GATES,
                 scope="Control-isolation diagnostic, not geometry/Mu2e validation")
    oracle.write_json(directory / "fixtures.json", index)
    return index


def control_report(report, controls):
    """Report outer substeps separately from internal integration work (field evaluations)."""
    result = {}
    for backend in ("host", "device"):
        by_id = {t["id"]: t for t in report["trajectories"][backend]}
        grouped = {}
        for control in controls:
            row = dict(by_id[control["id"]])
            grouped.setdefault(f"{control['physical_group']}/{control['axis']}", []).append(row)
        result[backend] = grouped
    return result


def diagnostic_metrics(request, result, expected):
    """Compute raw endpoint diagnostics without granting acceptance to failed records."""
    oracle.require(oracle.canonical(result.get("input")) == oracle.canonical(request), "Changed result input")
    metrics = {}
    for backend in ("host", "device"):
        rows = result["tracks"][backend]
        oracle.require(len(rows) == len(request["tracks"]), "Missing track records")
        positions = np.asarray([r["end_position_mm"] for r in rows], dtype=float)
        momenta = np.asarray([r["end_momentum_mev_c"] for r in rows], dtype=float)
        truth_x = np.asarray([m["end_position_mm"] for m in expected["tracks"]])
        truth_p = np.asarray([m["end_momentum_mev_c"] for m in expected["tracks"]])
        norms = np.linalg.norm(momenta, axis=1)
        p0 = np.linalg.norm([t["momentum_mev_c"] for t in request["tracks"]], axis=1)
        oracle.require(np.isfinite(positions).all() and np.isfinite(momenta).all() and np.all(norms > 0),
                       "Nonfinite/zero diagnostic endpoint")
        errors = np.linalg.norm(positions - truth_x, axis=1)
        angles = np.linalg.norm(momenta / norms[:, None] - truth_p / np.linalg.norm(truth_p, axis=1)[:, None], axis=1)
        drift = np.abs(norms / p0 - 1)
        metrics[backend] = [dict(id=t["id"], method=t["method"], max_step_mm=t["max_step_mm"],
                                epsilon_rel_max=t["epsilon_rel_max"], delta_chord_mm=t["delta_chord_mm"],
                                position_error_mm=float(e), direction_error=float(a),
                                momentum_relative_drift=float(d), steps=r["steps"],
                                field_evaluations=r["field_evaluations"])
                            for t, r, e, a, d in zip(request["tracks"], rows, errors, angles, drift)]
    return metrics


def collect_checks(request, result, expected):
    """Validate every record with unchanged h6 limits, retaining failures instead of stopping at the first."""
    metrics = diagnostic_metrics(request, result, expected)
    failures = []
    fields = None
    for i, track in enumerate(request["tracks"]):
        selected = dict(request, tracks=[track])
        sidecar = dict(expected, tracks=[expected["tracks"][i]], request_sha256=oracle.digest(selected))
        sliced = dict(result, input=selected, tracks={b: [result["tracks"][b][i]] for b in ("host", "device")})
        try:
            checked = oracle.check_result(selected, sliced, sidecar)
            fields = checked["fields"]
            accepted = True
        except ValueError as error:
            failures.append(dict(id=track["id"], error=str(error)))
            accepted = False
        for backend in ("host", "device"):
            metrics[backend][i]["paired_endpoint_budget_pass"] = accepted
    return dict(fields=fields, trajectories=metrics, failures=failures)


def response_gates(controlled):
    """Require control response on smooth GPU helices, not monotonicity on quantized affine fields."""
    rows = []
    for group in GROUPS[:2]:
        for axis in ("epsilon", "chord"):
            sweep = controlled["device"][f"{group}/{axis}"]
            coarse, fine = sweep[0], sweep[-1]
            ratio = fine["position_error_mm"] / max(coarse["position_error_mm"], GATES["floor_mm"])
            endpoint_ok = (fine["position_error_mm"] <= GATES["position_mm"]
                           and fine["direction_error"] <= GATES["direction"]
                           and fine.get("paired_endpoint_budget_pass", True))
            improvement = fine["position_error_mm"] <= GATES["floor_mm"] or ratio <= GATES["fine_to_coarse"]
            work_key = "field_evaluations" if axis == "epsilon" else "steps"
            response = fine[work_key] > coarse[work_key]
            rows.append(dict(group=group, axis=axis, coarse_position_mm=coarse["position_error_mm"],
                             fine_position_mm=fine["position_error_mm"], fine_direction_error=fine["direction_error"],
                             fine_to_coarse=ratio, work_metric=work_key, coarse_work=coarse[work_key],
                             fine_work=fine[work_key], endpoint_pass=bool(endpoint_ok),
                             improvement_pass=bool(improvement), response_pass=bool(response),
                             passed=bool(endpoint_ok and improvement and response)))
    return rows


def check_directory(directory, baseline):
    directory, baseline = Path(directory), Path(baseline)
    index = oracle.read_json(directory / "fixtures.json")
    oracle.require(index.get("format") == FORMAT and index.get("gates") == GATES, "Wrong sensitivity index/gates")
    baseline_report = oracle.check_directory(baseline)
    manifest = oracle.read_json(directory / "run-manifest.json")
    base_manifest = oracle.read_json(baseline / "run-manifest.json")
    oracle.require(manifest["executable_sha256"] == base_manifest["executable_sha256"], "Different h6 executable")
    oracle.require(manifest["source_manifest_sha256"] == base_manifest["source_manifest_sha256"], "Different h6 source")
    oracle.require(manifest["library_sha256"] == base_manifest["library_sha256"], "Different h5 libraries")
    names = {f"{f}-{r}" for f in oracle.FAMILIES for r in oracle.RESOLUTIONS}
    oracle.require(len(index["cases"]) == 6 and {e["name"] for e in index["cases"]} == names, "Incomplete sensitivity cases")
    runs = manifest["runs"]
    oracle.require(len(runs) == 6 and {r["request"] for r in runs} == {n + ".request.json" for n in names}, "Incomplete run records")
    oracle.require(all(r["returncode"] == 0 and r["sanitizer_clean"] is True for r in runs), "Failed runtime/memcheck")
    for filename, expected_hash in manifest["fixture_sha256"].items():
        oracle.require(Path(filename).name == filename and sha(directory / filename) == expected_hash, "Modified executed fixture")
    for run in runs:
        oracle.require(Path(run["result"]).name == run["result"] and sha(directory / run["result"]) == run["result_sha256"], "Modified runtime output")
    reports, gates = {}, []
    for entry in index["cases"]:
        stem = entry["name"]
        family, resolution = stem.split("-")
        data, request, expected, controls = build_case(family, resolution)
        wanted = dict(name=stem, map=f"{stem}.map.json", request=f"{stem}.request.json",
                      expected=f"{stem}.expected.json", result=f"{stem}.result.json", controls=controls)
        oracle.require(entry == wanted, "Modified sensitivity entry")
        for key, fresh in (("map", data), ("request", request), ("expected", expected)):
            oracle.require(oracle.canonical(oracle.read_json(directory / entry[key])) == oracle.canonical(fresh),
                           f"Modified sensitivity {key}")
        result = oracle.read_json(directory / entry["result"])
        checked = collect_checks(request, result, expected)
        controlled = control_report(checked, controls)
        if family == "constant":
            gates.extend(dict(case=stem, **g) for g in response_gates(controlled))
        baseline_rows = {}
        for backend in ("host", "device"):
            baseline_rows[backend] = [r for r in baseline_report["cases"][stem]["trajectories"][backend]
                                      if r.get("group") in GROUPS]
        reports[stem] = dict(fields=checked["fields"], controls=controlled, retained_h6=baseline_rows,
                             endpoint_budget_failures=checked["failures"])
    all_endpoints = all(not r["endpoint_budget_failures"] for r in reports.values())
    return dict(format="m2fs-field-sensitivity-check-v1", passed=all_endpoints and all(g["passed"] for g in gates),
                endpoint_budget_checks_passed=all_endpoints, response_gates_passed=all(g["passed"] for g in gates),
                gates=gates, cases=reports,
                baseline=str(baseline), baseline_manifest_sha256=sha(baseline / "run-manifest.json"),
                scope="Control sensitivity and independent endpoint budgets, not internal-iteration certification")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    try:
        oracle.require(args.action == "prepare" or args.baseline is not None, "check needs retained --baseline results")
        result = prepare(args.directory) if args.action == "prepare" else check_directory(args.directory, args.baseline)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("passed", True) else 1
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"field sensitivity: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
