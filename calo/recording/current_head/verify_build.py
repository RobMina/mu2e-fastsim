"""Record a user-completed overlay build using hashes, mtimes and read-only ELF checks."""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import subprocess

HERE = Path(__file__).resolve().parent
LIBRARIES = {
    "libmu2e_Mu2eG4.so": "Mu2eG4",
    "libmu2e_Mu2eG4_Mu2eG4_module.so": "Mu2eG4",
    "libmu2e_Mu2eG4_Mu2eG4MT_module.so": "Mu2eG4",
    "libmu2e_CaloMC.so": "CaloMC",
    "libmu2e_CaloMC_CaloShowerStepMaker_module.so": "CaloMC",
    "libmu2e_CaloRecording_CaloInputEventManifest_module.so": "CaloRecording",
    "libmu2e_CaloRecording_CaloIncidentManifest_module.so": "CaloRecording",
}
STUB = "al9-prof-e29-p110"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=True)
    return result.stdout + result.stderr


def relocation(text):
    missing = re.findall(r"^.*not found.*$", text, re.M)
    undefined = sorted(set(re.findall(r"^undefined symbol:\s+(\S+)", text, re.M)))
    resolved = dict(re.findall(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(", text, re.M))
    return {"missing": missing, "undefined": undefined, "resolved": resolved}


def verify(overlay):
    overlay = overlay.resolve(strict=True)
    if os.environ.get("MUSE_WORK_DIR") != str(overlay) or os.environ.get("MUSE_STUB") != STUB:
        raise ValueError("wrong configured workDir or build stub")
    spec = importlib.util.spec_from_file_location("built_overlay_guard", HERE / "prepare_overlay.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    preparation = guard.verify(overlay)
    backing = Path(preparation["backing"])
    source = Path(preparation["source_offline"])
    library_dir = overlay / "build" / STUB / "Offline/lib"
    backing_libraries = backing / "build" / STUB / "Offline/lib"
    completion = (overlay / "build" / STUB / ".musebuild").read_text().strip()
    if not re.fullmatch(r"\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} to\s+\d{2}:\d{2}:\d{2}", completion):
        raise ValueError("no completed build receipt")
    for variable in ("MUSE_LIBRARY_PATH", "CET_PLUGIN_PATH"):
        paths = os.environ.get(variable, "").split(":")
        if str(library_dir) not in paths or str(backing_libraries) not in paths:
            raise ValueError("missing local/backing lookup path: " + variable)
        if paths.index(str(library_dir)) >= paths.index(str(backing_libraries)):
            raise ValueError("backing precedes local libraries: " + variable)
    report = {
        "schema": 1,
        "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "work_dir": str(overlay),
        "build_stub": STUB,
        "source_revision": preparation["source_revision"],
        "source_manifest_sha256": digest(HERE / "source-manifest.json"),
        "overlay_receipt_sha256": digest(overlay / "overlay-receipt.json"),
        "backing": str(backing),
        "completion": completion,
        "checks": {},
        "binaries": {},
        "native_execution_performed": False,
        "physics_validation_performed": False,
        "user_reported_build_success": True,
    }
    for name, package in LIBRARIES.items():
        path = library_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("local binary missing or a symlink: " + name)
        inputs = [source / item for item in preparation["package_files"]
                  if item.startswith(package + "/") and (Path(item).suffix in (".cc", ".hh")
                                                        or Path(item).name == "SConscript")]
        latest = max(item.stat().st_mtime_ns for item in inputs)
        if path.stat().st_mtime_ns <= latest:
            raise ValueError("binary predates package source/header/build inputs: " + name)
        elf = command("readelf", "-d", str(path))
        linkage_text = command("ldd", "-r", str(path))
        linkage = relocation(linkage_text)
        if linkage["missing"]:
            raise ValueError("missing shared dependency: " + name)
        for soname, resolved in linkage["resolved"].items():
            if soname.startswith("libmu2e_"):
                actual = Path(resolved).resolve(strict=True)
                expected = library_dir if soname in LIBRARIES else backing_libraries
                if actual.parent != expected:
                    raise ValueError("unexpected Offline dependency location: " + soname)
        stock = None
        if (backing_libraries / name).is_file():
            stock = relocation(command("ldd", "-r", str(backing_libraries / name)))
        if linkage["undefined"]:
            if (stock is None or stock["missing"] or stock["undefined"] != linkage["undefined"]
                    or not all(symbol.startswith("cblas_") for symbol in linkage["undefined"])):
                raise ValueError("new or unexplained isolated relocation error: " + name)
        if package == "CaloRecording" and "libcetlib.so" not in linkage["resolved"]:
            raise ValueError("manifest direct cetlib dependency unresolved")
        if package == "CaloRecording" and not re.search(r"NEEDED.*\[libcetlib\.so\]", elf):
            raise ValueError("manifest does not directly declare cetlib")
        report["binaries"][name] = {
            "sha256": digest(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "latest_package_input_mtime_ns": latest,
            "needed": re.findall(r"NEEDED.*\[(.*?)\]", elf),
            "elf_dynamic": elf,
            "ldd_r": linkage_text,
            "missing_dependencies": linkage["missing"],
            "isolated_undefined_symbols": linkage["undefined"],
            "stock_isolated_undefined_symbols": None if stock is None else stock["undefined"],
        }
    for module, main in (("libmu2e_Mu2eG4_Mu2eG4_module.so", "libmu2e_Mu2eG4.so"),
                         ("libmu2e_Mu2eG4_Mu2eG4MT_module.so", "libmu2e_Mu2eG4.so"),
                         ("libmu2e_CaloMC_CaloShowerStepMaker_module.so", "libmu2e_CaloMC.so")):
        resolved = relocation(report["binaries"][module]["ldd_r"])["resolved"]
        if main not in resolved or Path(resolved[main]).resolve(strict=True) != library_dir / main:
            raise ValueError("plugin does not resolve its modified main library locally: " + module)
    executable = Path(command("which", "mu2e").strip()).resolve(strict=True)
    if executable != backing / "build" / STUB / "Offline/bin/mu2e":
        raise ValueError("unexpected executable")
    executable_relocations = relocation(command("ldd", "-r", str(executable)))
    if executable_relocations["missing"] or executable_relocations["undefined"]:
        raise ValueError("backing executable dependency check failed")
    report["executable"] = {"path": str(executable), "sha256": digest(executable)}
    report["checks"] = {
        "completed_build_receipt": True,
        "source_hashes_and_overlay_links": True,
        "binary_freshness": True,
        "local_library_and_plugin_path_precedence": True,
        "modified_plugin_mainlibs_resolve_locally": True,
        "no_missing_shared_dependencies": True,
        "manifest_isolated_relocations_clean": True,
        "mu2eg4_isolated_relocations_clean": False,
        "mu2eg4_isolated_cblas_symbols_match_exact_stock": True,
        "backing_executable_isolated_relocations_clean": True,
    }
    report["limitation"] = (
        "Mu2eG4 isolated ldd -r exposes stock-identical GSL cblas symbols; not a clean standalone "
        "relocation check or a proven runnable art global-loading context. No plugins scheduled, "
        "FHiCL expanded, geometry initialized, events transported or physics invariance checked."
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("verification output must be new")
    result = verify(args.overlay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"checks": result["checks"], "limitation": result["limitation"]}, indent=2))
    for name, entry in result["binaries"].items():
        print(name, entry["sha256"])
    print("Verification receipt SHA256:", digest(args.output))
