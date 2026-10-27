"""Run a user-built navigation probe with pinned inputs/libraries; never configure or compile."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def libraries(text, prefix):
    prefix = prefix.resolve()
    require("not found" not in text, "Unresolved dynamic library")
    found = {}
    for line in text.splitlines():
        cols = line.split()
        if len(cols) < 3 or cols[1] != "=>":
            continue
        family = cols[0].removeprefix("lib").split(".")[0].removesuffix("_final")
        if family in {"celeritas", "corecel", "geocel", "orange", "accel"}:
            path = Path(cols[2])
            require(path.is_absolute() and path.resolve().is_relative_to(prefix), f"Mixed library: {line}")
            found[cols[0]] = dict(path=str(path), sha256=sha(path))
    require({"libceleritas.so", "libgeocel.so", "libcorecel.so"} <= found.keys(), "Incomplete library family")
    return found


def verify(source, build, prefix):
    pin = read_json(source / "source-manifest.json")
    require(pin["format"] == "m2fs-navigation-source-v1", "Wrong source manifest")
    for name, digest in pin["sha256"].items():
        require(Path(name).name == name and sha(source / name) == digest, f"Changed source {name}")
    for name, digest in pin["installed_api_sha256"].items():
        require(sha(prefix / "include" / name) == digest, f"Changed h5 API {name}")
    for name, digest in LIBRARIES.items():
        require(sha(prefix / "lib64" / name) == digest, f"Changed h5 library {name}")
    exe = build / "m2fs-navigation-probe"
    require(os.access(exe, os.X_OK), "User-built navigation executable missing")
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    require(f"CMAKE_HOME_DIRECTORY:INTERNAL={source}" in cache, "Different build source")
    for key, wanted in (("Celeritas_DIR", prefix / "lib64/cmake/Celeritas"), ("M2FS_CARTMAP_DIR", source)):
        require(any(line.startswith(key + ":") and Path(line.split("=", 1)[1]).resolve() == wanted for line in cache),
                f"Wrong build binding for {key}")
    compiled = read_json(build / "compile_commands.json")
    parser_files = [Path(c["file"]).resolve() for c in compiled if Path(c["file"]).name == "CartMapIO.cc"]
    require(parser_files == [source / "CartMapIO.cc"], "Unpinned map parser")
    cuda = [c for c in compiled if Path(c["file"]).name == "NavKernels.cu"]
    require(len(cuda) == 1 and "-rdc=true" in cuda[0].get("command", " ".join(cuda[0].get("arguments", []))),
            "Navigation kernel was not compiled with RDC")
    require(exe.stat().st_mtime >= max((source / name).stat().st_mtime for name in pin["sha256"]), "Binary predates snapshot")
    return exe


def accepted_process(code, result_exists, log, guard, sanitizer_text=None):
    if guard:
        return code != 0 and code != 124 and not result_exists and "M2FS_NAV_MAP_GUARD" in log
    errors = re.findall(r"ERROR SUMMARY: (\d+) errors?", sanitizer_text or "")
    clean = sanitizer_text is None or (bool(errors) and all(int(n) == 0 for n in errors)
                                       and "COMPUTE-SANITIZER" in sanitizer_text and "========= ERROR:" not in sanitizer_text)
    return code == 0 and result_exists and clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--sanitizer", action="store_true")
    args = parser.parse_args()
    source, fixtures, build, prefix = (p.resolve() for p in (args.source, args.fixtures, args.build, args.prefix))
    exe = verify(source, build, prefix)
    out = args.output.resolve()
    require(not out.exists(), "Output must be new")
    index = read_json(fixtures / "fixtures.json")
    require(index.get("format") == "m2fs-navigation-fixtures-v1" and index.get("cases"), "Prepared fixtures missing")
    cases = index["cases"]
    require(len({c["id"] for c in cases}) == len(cases), "Duplicate fixture IDs")
    requests = sorted(fixtures.glob("*.request.json"))
    require({p.name for p in requests} == {c["request"] for c in cases}, "Fixture index/request mismatch")
    by_request = {c["request"]: c for c in cases}
    for case in cases:
        for key in ("request", "expected", "result"):
            require(Path(case[key]).name == case[key] and case[key] not in {".", "..", ""}, "Unsafe case filename")
        expected = read_json(fixtures / case["expected"])
        require(expected["kind"] == case["kind"], "Fixture kind mismatch")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("CELER_")}
    environment.update(LD_LIBRARY_PATH=f"{prefix}/lib64:{prefix}/lib:" + environment.get("LD_LIBRARY_PATH", ""),
                       CUDA_LAUNCH_BLOCKING="1")
    ldd = subprocess.check_output(["ldd", str(exe)], env=environment, text=True, stderr=subprocess.STDOUT)
    linked = libraries(ldd, prefix)
    sanitizer = shutil.which("compute-sanitizer") if args.sanitizer else None
    require(not args.sanitizer or sanitizer, "Compute Sanitizer missing")
    out.mkdir(exist_ok=False)
    copied = []
    for path in sorted(fixtures.iterdir()):
        require(path.is_file() and path.suffix in {".json", ".gdml"}, "Unexpected fixture entry")
        shutil.copyfile(path, out / path.name)
        copied.append(path.name)
    (out / "ldd.txt").write_text(ldd)
    for name in ("configure.log", "build.log", "CMakeCache.txt", "compile_commands.json"):
        shutil.copyfile(build / name, out / name)
    manifest = dict(format="m2fs-navigation-run-v1", executable_sha256=sha(exe), library_provenance=linked,
                    source_manifest_sha256=sha(source / "source-manifest.json"), runner_sha256=sha(Path(__file__)),
                    fixture_sha256={name: sha(out / name) for name in copied}, runs=[], sanitizer=bool(sanitizer),
                    scope="Field plus navigation only; no interactions, scorer, real Mu2e or speed validation")
    for request in requests:
        stem = request.name.removesuffix(".request.json")
        guard = by_request[request.name]["kind"] == "guard"
        for backend in (("host", "device") if guard else ("both",)):
            name = f"{stem}-{backend}" if guard else stem
            result = out / f"{name}.result.json"
            log = out / f"{name}.log"
            command = [str(exe), str(out / request.name), str(result), backend]
            memcheck = out / f"{name}.memcheck.txt"
            if sanitizer and not guard:
                command = [sanitizer, "--tool", "memcheck", "--error-exitcode", "99", "--log-file", str(memcheck)] + command
            start = time.monotonic()
            code = 124
            with log.open("x") as stdout:
                try:
                    code = subprocess.run(command, env=environment, stdout=stdout, stderr=subprocess.STDOUT,
                                          timeout=180, check=False).returncode
                except subprocess.TimeoutExpired:
                    stdout.write("\nNavigation probe timed out\n")
            sanitizer_text = memcheck.read_text() if sanitizer and not guard and memcheck.exists() else None
            if sanitizer and not guard and sanitizer_text is None:
                sanitizer_text = "missing memcheck log"
            ok = accepted_process(code, result.is_file(), log.read_text(), guard, sanitizer_text)
            if ok and not guard:
                data = read_json(result)
                expectation = read_json(out / by_request[request.name]["expected"])
                statuses = [entry["status"] for entry in expectation["tracks"]]
                ok = (data.get("format") == "m2fs-navigation-result-v1"
                      and data.get("input") == read_json(out / request.name)
                      and data.get("backend") == "both" and set(data.get("tracks", {})) == {"host", "device"}
                      and all([entry.get("status") for entry in data["tracks"][b]] == statuses for b in ("host", "device")))
            record = dict(name=name, request=request.name, result=result.name, backend=backend, log=log.name,
                          returncode=code, seconds=time.monotonic()-start, command=command, negative_guard=guard,
                          accepted_process=ok, log_sha256=sha(log), result_sha256=sha(result) if result.exists() else None,
                          sanitizer_clean=ok if sanitizer and not guard else None,
                          sanitizer_log=memcheck.name if sanitizer and not guard else None,
                          sanitizer_sha256=sha(memcheck) if sanitizer and not guard and memcheck.exists() else None)
            manifest["runs"].append(record)
            write_json(out / "run-manifest.json", manifest)
            print(json.dumps(record), flush=True)
            if not ok:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
