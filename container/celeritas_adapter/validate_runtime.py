"""Bounded synthetic checks of an already-built adapter; never configure or build."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import smoke_celeritas as smoke


MANIFEST_SHA256 = "774e93dc8e2f8aec55b0e76947cb237c72300d5e1d122ccc89ca08b57fe28933"
MATRIX = ((1e-5, 2e-5, 3e-5), (4e-5, -2e-5, 6e-5), (7e-5, 8e-5, 1e-5))
POINTS = [[0, 0, 0], [10, 50, 100], [-800, 500, -1000], [333, 555, 777],
          [-1000, -1200, -1000], [1000, 1200, 1000], [0, 400, 0]]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_source_build(source, build):
    pin = source / "adapter-manifest.json"
    require(digest(pin) == MANIFEST_SHA256, "Not the pinned h4 prepared source")
    prepared = read_json(pin)["prepared_sha256"]
    for name, sha in prepared.items():
        require(digest(source / name) == sha, f"Modified prepared source: {name}")
    cache = (build / "CMakeCache.txt").read_text()
    require(f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n" in cache, "Build points to a different source tree")
    newest_source = max((source / name).stat().st_mtime for name in prepared)
    for name in ("m2fs-celer-g4", "m2fs-field-check"):
        require(os.access(build / name, os.X_OK), f"Missing built executable: {name}")
        require((build / name).stat().st_mtime >= newest_source, f"Binary predates prepared source: {name}")


def check_library_resolution(text, prefix):
    require("not found" not in text, "Missing dynamic library")
    families = {"accel", "celeritas", "geocel", "corecel", "orange"}
    found = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[1] != "=>":
            continue
        name, path = parts[0], Path(parts[2])
        family = name.removeprefix("lib").split(".")[0].removesuffix("_final")
        if family in families:
            require(path.is_absolute() and path.resolve().is_relative_to(prefix.resolve()),
                    f"Mixed Celeritas libraries: {name} resolves outside candidate prefix: {path}")
            found[name] = str(path)
    required = {f"lib{name}.so" for name in ("accel_final", "accel", "celeritas", "geocel", "corecel")}
    require(required <= found.keys(), "Incomplete Celeritas library resolution")
    return found


def library_provenance(build, prefix, out):
    if prefix is None:
        return None
    prefix = prefix.resolve()
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    expected = str(prefix / "lib64/cmake/Celeritas")
    require(any(line.startswith("Celeritas_DIR:") and line.split("=", 1)[1] == expected for line in cache),
            "Adapter was not configured against the candidate Celeritas prefix")
    provenance = {"prefix": str(prefix), "executables": {}}
    for name in ("m2fs-celer-g4", "m2fs-field-check"):
        text = subprocess.check_output(["ldd", str(build / name)], text=True, stderr=subprocess.STDOUT)
        (out / f"{name}.ldd.txt").write_text(text)
        paths = check_library_resolution(text, prefix)
        provenance["executables"][name] = {lib: {"path": path, "sha256": digest(Path(path))}
                                           for lib, path in paths.items()}
    return provenance


def build_evidence(build, log_dir=None):
    if log_dir is None:
        files = {name: build / name for name in ("configure.log", "build.log")}
    else:
        files = {name: log_dir / name for name in ("adapter-configure.log", "adapter-build.log",
                                                  "configure.log", "build.log", "install.log")}
    files["CMakeCache.txt"] = build / "CMakeCache.txt"
    return {name: digest(path) for name, path in files.items()}


def field_values(points, affine):
    return [[(1.0 if j == 2 else 0.0) +
             (math.fsum(p[i] * MATRIX[i][j] for i in range(3)) if affine else 0.0)
             for j in range(3)] for p in points]


def check_field(data, points, affine):
    require(data.get("field_unit") == "tesla", "Wrong field output units")
    values = data["field"]
    require(len(values) == len(points) and all(len(v) == 3 for v in values), "Wrong field shape")
    errors = [abs(v - e) for row, expected in zip(values, field_values(points, affine))
              for v, e in zip(row, expected)]
    require(all(math.isfinite(e) for e in errors) and max(errors) < 2e-6,
            "Host field differs from analytic constant/affine oracle")
    return {"points": len(points), "max_abs_error_tesla": max(errors)}


def check_transport(data, record):
    card = data["input"]
    primary = card["primary_options"]
    expected = record["card"]["primary_options"]
    require(all(primary.get(k) == v for k, v in expected.items()),
            "Persisted primary input differs from request")
    require(card["sd_type"] == record["card"]["sd_type"], "Wrong sensitive detector type")
    require(card["field_type"] == record["card"]["field_type"], "Wrong field type")
    if card["field_type"] == "cartmap":
        require(card["field_file"] == record["card"]["field_file"], "Wrong map file")
    else:
        require(card["field"] == [0, 0, 1], "Wrong uniform field")
    completed = sum(map(len, data["result"]["time"]["events"]))
    require(completed == primary["num_events"], "Incomplete finished-event records")
    system = data["system"]
    require(system["build"]["version"] == "0.6.3", "Wrong runtime Celeritas version")
    if record["mode"] == "gpu":
        require(system["device"]["platform"] == "CUDA" and "A100" in system["device"]["name"],
                "Expected A100 CUDA device")
        names = {k["name"] for k in system["kernels"]}
        require({"extend-from-primaries", "step-gather-post"} <= names, "Missing GPU kernel evidence")
        actions = data["result"]["time"]["actions"]
        prefix = "along-step-cart" if card["field_type"] == "cartmap" else "along-step-uniform"
        if not record.get("empty", False):
            require(any(value > 0 for action in actions for key, value in action.items()
                        if key.startswith(prefix)), "No timed field-specific GPU charged transport action")
        else:
            require(any(a.get("along-step-neutral", 0) > 0 for a in actions), "No neutral GPU transport action")
        require(system["environ"].get("CELER_DISABLE") in (None, "", "0"), "GPU offload disabled")
        require(system["environ"].get("CELER_DISABLE_DEVICE") in (None, "", "0"), "GPU device disabled")
    else:
        require(system["environ"].get("CELER_DISABLE") == "1", "G4 control offload enabled")
    return completed


def check_events(path, events, primaries, energy_per_primary, empty=False):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(rows) == events, "Wrong JSONL event count")
    require([r["event_id"] for r in rows] == list(range(events)), "Missing/duplicate/unordered event IDs")
    require(all(r["format"] == "m2fs-event-total-edep-v1" for r in rows), "Wrong event format")
    require(all(r["primary_count"] == primaries for r in rows), "Wrong primary denominator")
    require(all(len(r["collections"]) == 1 for r in rows), "Expected exactly one sensitive collection")
    collections = [c for r in rows for c in r["collections"]]
    hits = [h for c in collections for h in c["hits"]]
    require(all(type(h.get("volume_instance")) is int and h["volume_instance"] >= 0
                and type(h.get("copy_num")) is int and h["copy_num"] == 1 for h in hits),
            "Missing or wrong hit volume/copy identity")
    require(len({h["volume_instance"] for h in hits}) <= 1, "Multiple absorber volume identities")
    require(all(c["sd"] == "smoke_calo" and c["collection"] == "smoke_calo" for c in collections),
            "Wrong sensitive collection identity")
    require(all(math.isfinite(h["total_edep_mev"]) and h["total_edep_mev"] >= 0
                and math.isfinite(h["pre_time_s"]) for h in hits), "Invalid hit energy/time")
    require(all(math.isclose(c["total_edep_mev"], math.fsum(h["total_edep_mev"] for h in c["hits"]),
                            rel_tol=1e-12, abs_tol=1e-10) for c in collections), "Hit/collection sum mismatch")
    require(all(r["hit_count"] == sum(len(c["hits"]) for c in r["collections"]) for r in rows),
            "Hit count mismatch")
    require(all(math.isclose(r["total_edep_mev"], math.fsum(c["total_edep_mev"] for c in r["collections"]),
                            rel_tol=1e-12, abs_tol=1e-10) for r in rows), "Collection/event sum mismatch")
    expected = 0 if empty else primaries * energy_per_primary
    require(all(math.isfinite(r["total_edep_mev"]) and
                math.isclose(r["total_edep_mev"], expected, rel_tol=1e-5, abs_tol=1e-10) for r in rows),
            "Per-event contained-energy check failed")
    if empty:
        require(not hits and all(r["hit_count"] == 0 for r in rows), "Expected genuinely zero-hit events")
    total = math.fsum(r["total_edep_mev"] for r in rows)
    return {"events": len(rows), "primaries": sum(r["primary_count"] for r in rows),
            "hits": len(hits), "total_edep_mev": total, "zero_hit_events": sum(r["hit_count"] == 0 for r in rows)}


def inspect(out, record):
    work = out / record["name"]
    log = (work / "run.log").read_text()
    if record["kind"] == "reject":
        require(record["returncode"] != 0 and not record.get("timeout"), "Expected bounded explicit rejection")
        messages = record.get("messages", [record["message"]])
        require(any(message in log for message in messages), "Rejection reason did not match expected guard")
        if record.get("preserve"):
            require((work / "events.jsonl").read_text() == "preserve-me\n", "Existing output was modified")
        return {"rejection": record["message"], "returncode": record["returncode"]}
    require(record["returncode"] == 0, "Process failed")
    if record["kind"] == "field":
        return check_field(json.loads(log), POINTS, record["affine"])
    data = read_json(work / "result.json")
    completed = check_transport(data, record)
    if record["kind"] == "calo":
        result = smoke.inspect_result(data, record["pdg"], record["mode"], completed)
        require(math.isclose(result["energy_mev"], record["baseline_energy_mev"], rel_tol=1e-10),
                "Uniform adapter differs from retained stock energy")
        return result
    energy = 10 + (2 * 0.51099891 if record["pdg"] == -11 else 0)
    result = check_events(work / "events.jsonl", completed,
                          record["card"]["primary_options"]["primaries_per_event"], energy,
                          empty=record.get("empty", False))
    if record.get("baseline_energy_mev") is not None:
        require(math.isclose(result["total_edep_mev"], record["baseline_energy_mev"], rel_tol=1e-5),
                "Event output differs from retained simple_calo energy")
    return result


def expected_cases():
    names = {"field_affine", "field_constant"}
    names.update(f"reject_point_{axis}_{edge}" for axis, edge in
                 ((0, -1000.01), (0, 1000.01), (1, -1200.01), (1, 1200.01), (2, -1400), (2, 1400)))
    names.update(f"events_{pdg}_{mode}_{field}" for pdg in (11, -11, 22)
                 for mode in ("g4", "gpu") for field in ("uniform", "constant", "affine"))
    names.update(f"calo_{pdg}_{mode}_uniform" for pdg in (11, -11, 22) for mode in ("g4", "gpu"))
    names.update(f"{kind}_{mode}" for kind in ("empty", "multiprimary") for mode in ("g4", "gpu"))
    names.update(f"reject_{guard}" for guard in ("existing", "missing_parent", "no_sd", "world", "kill", "nonfatal", "write"))
    return names


def check_output(out):
    manifest = read_json(out / "manifest.json")
    require(len(manifest["runs"]) == len(expected_cases()), "Incomplete runtime suite")
    require({r["name"] for r in manifest["runs"]} == expected_cases(), "Missing/duplicate/substituted runtime cases")
    rows = []
    for record in manifest["runs"]:
        row = {"name": record["name"]}
        try:
            row.update(inspect(out, record), passed=True)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            row.update(passed=False, error=str(exc))
        rows.append(row)
    summary = {"scope": "synthetic host-field/execution/event-accounting checks, not Mu2e or field-convergence validation",
               "passed": all(r["passed"] for r in rows), "rows": rows}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary["passed"]


def limited_output():
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (1, 1))
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)


def run_case(out, manifest, record, command, env, limit_output=False):
    work = out / record["name"]
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=work, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=60,
                                preexec_fn=limited_output if limit_output else None)
        record["returncode"] = result.returncode
        output = result.stdout
    except subprocess.TimeoutExpired as exc:
        record.update(returncode=124, timeout=True)
        output = exc.stdout or b""
    (work / "run.log").write_bytes(output)
    record["wall_seconds"] = time.monotonic() - start
    manifest["runs"].append(record)
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: record[k] for k in ("name", "returncode", "wall_seconds")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--prefix", type=Path, help="Require every loaded Celeritas library under this prefix")
    parser.add_argument("--build-log-dir", type=Path, help="h5 build root containing adapter and library logs")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.check_only:
        return 0 if check_output(out) else 1
    require(all((args.source, args.build, args.baseline)), "Source/build/baseline required")
    source, build, baseline = args.source.resolve(), args.build.resolve(), args.baseline.resolve()
    pin = source / "adapter-manifest.json"
    verify_source_build(source, build)
    retained = {}
    for pdg in (11, -11, 22):
        for mode in ("g4", "gpu"):
            path = baseline / f"pdg{pdg}_{mode}" / "result.json"
            retained[pdg, mode] = smoke.inspect_result(read_json(path), pdg, mode, 64)
    out.mkdir(exist_ok=False)
    (out / "smoke.gdml").write_text(smoke.GEOMETRY)
    (out / "no-sd.gdml").write_text(smoke.GEOMETRY.replace('      <auxiliary auxtype="SensDet" auxvalue="smoke_calo"/>\n', ''))
    manifest = {"source": str(source), "build": str(build), "baseline": str(baseline),
                "source_manifest_sha256": digest(pin), "runner_sha256": digest(Path(__file__)),
                "smoke_helper_sha256": digest(Path(smoke.__file__)),
                "binary_sha256": {n: digest(build / n) for n in ("m2fs-celer-g4", "m2fs-field-check")},
                "build_evidence_sha256": build_evidence(build, args.build_log_dir),
                "library_provenance": library_provenance(build, args.prefix, out),
                "baseline_sha256": {f"pdg{p}_{m}": digest(baseline / f"pdg{p}_{m}" / "result.json") for p, m in retained},
                "map_sha256": {n: digest(source / n) for n in ("constant-map.json", "affine-map.json")},
                "planned_runs": 43, "runs": []}
    env = smoke.clean_environment("g4")
    env.pop("M2FS_EVENT_OUTPUT", None)
    for affine in (False, True):
        record = {"name": "field_affine" if affine else "field_constant", "kind": "field", "affine": affine}
        work = out / record["name"]
        work.mkdir()
        write_json(work / "points.json", POINTS)
        run_case(out, manifest, record, [str(build / "m2fs-field-check"),
                 str(source / ("affine-map.json" if affine else "constant-map.json")), "points.json"], env)
    for axis, edge in ((0, -1000.01), (0, 1000.01), (1, -1200.01), (1, 1200.01),
                       (2, -1400), (2, 1400)):
        record = {"name": f"reject_point_{axis}_{edge}", "kind": "reject",
                  "message": "Point must have a one-cell interpolation margin"}
        work = out / record["name"]
        work.mkdir()
        point = [0, 0, 0]
        point[axis] = edge
        write_json(work / "points.json", [point])
        run_case(out, manifest, record, [str(build / "m2fs-field-check"), str(source / "affine-map.json"), "points.json"], env)
    cases = []
    for pdg in (11, -11, 22):
        for mode in ("g4", "gpu"):
            for field in ("uniform", "constant", "affine"):
                card = smoke.make_input(pdg, 64, 1)
                card["sd_type"] = "event_hit"
                if field != "uniform":
                    card.update(field_type="cartmap", field_file=str(source / f"{field}-map.json"))
                    card.pop("field")
                cases.append({"name": f"events_{pdg}_{mode}_{field}", "kind": "events", "mode": mode,
                              "pdg": pdg, "card": card, "baseline_energy_mev": retained[pdg, mode]["energy_mev"]})
            cases.append({"name": f"calo_{pdg}_{mode}_uniform", "kind": "calo", "mode": mode, "pdg": pdg,
                          "card": smoke.make_input(pdg, 64, 1), "baseline_energy_mev": retained[pdg, mode]["energy_mev"]})
    for mode in ("g4", "gpu"):
        for empty in (False, True):
            card = smoke.make_input(22 if empty else 11, 8, 1)
            card.update(sd_type="event_hit", field_type="cartmap", field_file=str(source / "constant-map.json"))
            card.pop("field")
            card["primary_options"]["primaries_per_event"] = 2
            if empty:
                card["primary_options"]["position"]["params"] = [90, 90, 90]
            cases.append({"name": f"{'empty' if empty else 'multiprimary'}_{mode}", "kind": "events",
                          "mode": mode, "pdg": 22 if empty else 11, "empty": empty, "card": card})
    guards = [("existing", "Cannot create exclusive event output file"),
              ("missing_parent", "Cannot create exclusive event output file"),
              ("no_sd", "Event has no sensitive hit collections"),
              ("world", "World must fit inside Cartesian map with a one-cell margin"),
              ("kill", "Event output forbids kill-offload mode"),
              ("nonfatal", "Event output requires fatal transport failures"),
              ("write", "Event output flush failed")]
    for guard, message in guards:
        card = smoke.make_input(11, 1, 1)
        card["sd_type"] = "event_hit"
        if guard == "no_sd":
            card["geometry_file"] = "../no-sd.gdml"
        if guard == "world":
            small = read_json(source / "constant-map.json")
            small["axes"][0].update(min_mm=-1000, max_mm=1000)
            write_json(out / "small-map.json", small)
            card.update(field_type="cartmap", field_file="../small-map.json")
            card.pop("field")
        cases.append({"name": f"reject_{guard}", "kind": "reject", "message": message,
                      "mode": "gpu", "pdg": 11, "card": card, "guard": guard, "preserve": guard == "existing"})
    for record in cases:
        work = out / record["name"]
        work.mkdir()
        write_json(work / "input.json", record["card"])
        env = smoke.clean_environment(record["mode"])
        env.pop("M2FS_EVENT_OUTPUT", None)
        if record["kind"] != "calo":
            env["M2FS_EVENT_OUTPUT"] = str(work / "events.jsonl")
        guard = record.get("guard")
        if guard == "write":
            record["messages"] = ["Event output write failed", "Event output flush failed"]
        if guard == "existing":
            (work / "events.jsonl").write_text("preserve-me\n")
        elif guard == "missing_parent":
            env["M2FS_EVENT_OUTPUT"] = str(work / "absent" / "events.jsonl")
        elif guard == "kill":
            env["CELER_KILL_OFFLOAD"] = "1"
        elif guard == "nonfatal":
            env["CELER_NONFATAL_FLUSH"] = "1"
        run_case(out, manifest, record, [str(build / "m2fs-celer-g4"), "input.json"], env, guard == "write")
    return 0 if check_output(out) else 1


if __name__ == "__main__":
    sys.exit(main())
