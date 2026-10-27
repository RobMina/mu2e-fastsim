"""Run an already-built field probe and save evidence; never configure or compile."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time


LIBRARIES = {
    "libceleritas.so": "f11aa6c5bf290329e51956f5948f12933f51cf26a256b24b186d19e4e1c19f0d",
    "libceleritas_final.so": "043039fdef917a2bd7fc04645aa42327fe2ae1d00043f58cd0cc79d8b720e810",
    "libaccel_final.so": "48ec8354046c1e9f6228f40cdac95575337ee963946f9b92a4cb91089ac15410",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def resolve_libraries(text, prefix):
    require("not found" not in text, "Missing linked library")
    result = {}
    families = {"celeritas", "corecel", "geocel", "accel", "orange"}
    for line in text.splitlines():
        cols = line.split()
        if len(cols) < 3 or cols[1] != "=>":
            continue
        name, value = cols[0], Path(cols[2])
        family = name.removeprefix("lib").split(".")[0].removesuffix("_final")
        if family in families:
            require(value.is_absolute() and value.resolve().is_relative_to(prefix),
                    f"Mixed candidate/image libraries: {name} => {value}")
            result[name] = str(value)
    require({"libceleritas.so", "libgeocel.so", "libcorecel.so"} <= result.keys(),
            "Incomplete linked Celeritas library family")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--sanitizer", action="store_true")
    args = parser.parse_args()
    source, build, fixtures, prefix = (p.resolve() for p in (args.source, args.build, args.fixtures, args.prefix))
    exe = build / "m2fs-field-probe"
    require(os.access(exe, os.X_OK), "Missing built probe")
    pin = json.loads((source / "probe-manifest.json").read_text())
    require(pin["format"] == "m2fs-field-probe-source-v1", "Wrong preparation manifest")
    for name, sha in pin["sha256"].items():
        require(digest(source / name) == sha, f"Modified prepared source: {name}")
    require(exe.stat().st_mtime >= max((source / n).stat().st_mtime for n in pin["sha256"]),
            "Probe binary predates prepared source")
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    require(f"CMAKE_HOME_DIRECTORY:INTERNAL={source}" in cache, "Build has different source tree")
    require(any(line.startswith("M2FS_CARTMAP_DIR:") and Path(line.split("=", 1)[1]).resolve() == source
                for line in cache), "Build uses a different map parser directory")
    commands = json.loads((build / "compile_commands.json").read_text())
    parsers = [Path(c["file"]).resolve() for c in commands if Path(c["file"]).name == "CartMapIO.cc"]
    require(parsers == [source / "CartMapIO.cc"], "Compilation database does not use the pinned parser")
    require(any(line.startswith("Celeritas_DIR:") and line.split("=", 1)[1] == str(prefix / "lib64/cmake/Celeritas")
                for line in cache), "Build has different Celeritas installation")
    for name, sha in LIBRARIES.items():
        require(digest(prefix / "lib64" / name) == sha, f"h5 library changed: {name}")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("CELER_")}
    environment["LD_LIBRARY_PATH"] = f"{prefix}/lib64:{prefix}/lib:" + environment.get("LD_LIBRARY_PATH", "")
    environment["CUDA_LAUNCH_BLOCKING"] = "1"
    ldd = subprocess.check_output(["ldd", str(exe)], env=environment, text=True, stderr=subprocess.STDOUT)
    linked = resolve_libraries(ldd, prefix)
    sanitizer = shutil.which("compute-sanitizer") if args.sanitizer else None
    require(not args.sanitizer or sanitizer, "Compute Sanitizer missing")
    requests = sorted(fixtures.glob("*.request.json"))
    names = {f"{family}-{res}.request.json" for family in ("zero", "constant", "affine")
             for res in ("coarse", "refined")}
    require({p.name for p in requests} == names, "Incomplete prepared requests")
    fixture_files = [fixtures / "fixtures.json"]
    for path in requests:
        fixture_files.extend([path, path.with_name(path.name.replace(".request.json", ".map.json")),
                              path.with_name(path.name.replace(".request.json", ".expected.json"))])
    fixture_hashes = {p.name: digest(p) for p in fixture_files}
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    for path in fixture_files:
        shutil.copyfile(path, out / path.name)
    (out / "ldd.txt").write_text(ldd)
    manifest = {"scope": "field/ODE probe only, not full transport, geometry or Mu2e validation",
                "source_manifest_sha256": digest(source / "probe-manifest.json"),
                "executable_sha256": digest(exe), "runner_sha256": digest(Path(__file__)),
                "cache_sha256": digest(build / "CMakeCache.txt"), "fixture_sha256": fixture_hashes,
                "compile_commands_sha256": digest(build / "compile_commands.json"),
                "library_sha256": {n: digest(Path(p)) for n, p in linked.items()},
                "library_paths": linked, "sanitizer": bool(sanitizer), "runs": []}
    if sanitizer:
        manifest["sanitizer_version"] = subprocess.check_output([sanitizer, "--version"], text=True)
    for request_path in requests:
        card = json.loads(request_path.read_text())
        map_path = (fixtures / card["map_file"]).resolve()
        require(map_path.is_relative_to(fixtures), "Map must be in the fixture directory")
        command = [str(exe), str(request_path)]
        if sanitizer:
            command = [sanitizer, "--tool", "memcheck", "--error-exitcode", "99",
                       "--log-file", str(out / (request_path.stem + ".memcheck.txt"))] + command
        start = time.monotonic()
        result_path = out / request_path.name.replace(".request.json", ".result.json")
        log_path = out / request_path.name.replace(".request.json", ".stderr.txt")
        code = 124
        with result_path.open("xb") as stdout, log_path.open("xb") as stderr:
            try:
                code = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr,
                                      timeout=180, check=False).returncode
            except subprocess.TimeoutExpired:
                stderr.write(b"\nProbe timed out\n")
        record = {"request": request_path.name, "request_sha256": digest(request_path),
                  "map_sha256": digest(map_path), "command": command, "returncode": code,
                  "wall_seconds": time.monotonic() - start, "result": result_path.name,
                  "result_sha256": digest(result_path)}
        if sanitizer:
            sanitizer_path = out / (request_path.stem + ".memcheck.txt")
            text = sanitizer_path.read_text() if sanitizer_path.exists() else ""
            errors = re.findall(r"ERROR SUMMARY: (\d+) errors?", text)
            record["sanitizer_clean"] = (bool(errors) and all(int(n) == 0 for n in errors)
                                         and "COMPUTE-SANITIZER" in text and "========= ERROR:" not in text)
        manifest["runs"].append(record)
        write_json(out / "run-manifest.json", manifest)
        print(json.dumps(record), flush=True)
        if code != 0 or (sanitizer and not record["sanitizer_clean"]):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
