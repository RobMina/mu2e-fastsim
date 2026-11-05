"""Prepare a three-package, exact-CI-backed source overlay without compiling anything."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os

HERE = Path(__file__).resolve().parent
PACKAGES = ("CaloMC", "CaloRecording", "Mu2eG4")
STUB = "al9-prof-e29-p110"
REQUIRED_LIBRARIES = (
    "libmu2e_Mu2eG4.so",
    "libmu2e_Mu2eG4_Mu2eG4_module.so",
    "libmu2e_CaloMC.so",
    "libmu2e_CaloMC_CaloShowerStepMaker_module.so",
    "libmu2e_MCDataProducts.so",
    "libmu2e_SeedService.so",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verifier():
    spec = importlib.util.spec_from_file_location("prepared_source_verifier", HERE / "verify_area.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_files(root):
    result = {}
    for package in PACKAGES:
        for directory, dirs, files in os.walk(root / package, followlinks=False):
            if any((Path(directory) / name).is_symlink() for name in dirs):
                raise ValueError("source directory links are unsupported")
            for name in files:
                path = Path(directory) / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError("source must contain ordinary files: " + str(path))
                result[str(path.relative_to(root))] = sha(path)
    return dict(sorted(result.items()))


def discover(offline):
    return sorted(
        str((Path(directory) / "SConscript").relative_to(offline.parent))
        for directory, dirs, files in os.walk(offline, followlinks=False)
        if "SConscript" in files
    )


def expected_scripts():
    return sorted("Offline/" + package + "/src/SConscript" for package in PACKAGES)


def verify(overlay):
    receipt = json.loads((overlay / "overlay-receipt.json").read_text())
    source = Path(receipt["source_offline"])
    backing = Path(receipt["backing"])
    pins = json.loads((HERE / "source-manifest.json").read_text())
    if sha(HERE / "source-manifest.json") != receipt["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    load_verifier().verify(source)
    if (backing / "Offline/.git/HEAD").read_text().strip() != pins["base_revision"]:
        raise ValueError("backing revision changed")
    if (overlay / "backing").resolve(strict=True) != backing:
        raise ValueError("backing link changed")
    if set(path.name for path in (overlay / "Offline").iterdir()) != set(PACKAGES) | {".muse"}:
        raise ValueError("overlay package scope changed")
    if not (overlay / "Offline/.muse").is_symlink():
        raise ValueError("overlay environment must link to prepared source")
    if (overlay / "Offline/.muse").resolve(strict=True) != source / ".muse":
        raise ValueError("overlay environment link changed")
    if sha(source / ".muse") != pins["base_sources"][".muse"]:
        raise ValueError("prepared environment changed")
    actual = {}
    for directory, dirs, files in os.walk(overlay / "Offline", followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise ValueError("overlay directories must be real for build discovery")
        for name in files:
            path = Path(directory) / name
            relative = str(path.relative_to(overlay / "Offline"))
            if relative == ".muse":
                continue
            if not path.is_symlink() or path.resolve(strict=True) != source / relative:
                raise ValueError("overlay source link changed: " + relative)
            actual[relative] = sha(path)
    if actual != receipt["package_files"] or actual != package_files(source):
        raise ValueError("overlay source inventory changed")
    if discover(overlay / "Offline") != expected_scripts():
        raise ValueError("overlay build discovery is not exactly three packages")
    print("Exact backing, prepared-source links and three-package discovery verified.")
    print("No compilation, linkage, plugin loading or native execution performed.")
    return receipt


def prepare(source, backing, overlay):
    if overlay.exists() or overlay.is_symlink():
        raise ValueError("overlay destination must be new")
    source = source.resolve(strict=True)
    backing = backing.resolve(strict=True)
    overlay = overlay.absolute()
    if overlay == source or source in overlay.parents or overlay in source.parents:
        raise ValueError("overlay and source must be separate trees")
    load_verifier().verify(source)
    pins = json.loads((HERE / "source-manifest.json").read_text())
    if (backing / "Offline/.git/HEAD").read_text().strip() != pins["base_revision"]:
        raise ValueError("backing is not the exact prepared Offline revision")
    if sha(backing / "Offline/.muse") != pins["base_sources"][".muse"]:
        raise ValueError("backing environment does not match prepared p110 source")
    build = backing / "build" / STUB
    completion = (build / ".musebuild").read_text().strip()
    if "error" in completion.lower() or " to " not in completion:
        raise ValueError("backing build completion receipt is invalid")
    for name in REQUIRED_LIBRARIES:
        if not (build / "Offline/lib" / name).is_file():
            raise ValueError("backing library absent: " + name)
    if not (build / "Offline/bin/mu2e").is_file():
        raise ValueError("backing executable absent")
    files = package_files(source)
    if not files or any(not any(name.startswith(package + "/") for name in files) for package in PACKAGES):
        raise ValueError("prepared package absent")
    if sorted("Offline/" + name for name in files if Path(name).name == "SConscript") != expected_scripts():
        raise ValueError("unexpected source build discovery")
    overrides = pins["patched_sources"]
    for name, digest in files.items():
        if name in overrides:
            if digest != overrides[name]:
                raise ValueError("prepared override drift: " + name)
        elif not (backing / "Offline" / name).is_file() or sha(backing / "Offline" / name) != digest:
            raise ValueError("non-override source differs from backing: " + name)
    receipt = {
        "schema": 1,
        "source_offline": str(source),
        "source_revision": pins["base_revision"],
        "source_manifest_sha256": sha(HERE / "source-manifest.json"),
        "backing": str(backing),
        "backing_build_stub": STUB,
        "backing_completion": completion,
        "packages": list(PACKAGES),
        "sconscripts": expected_scripts(),
        "package_files": files,
        "source_modified": False,
        "build_performed": False,
        "native_execution_performed": False,
    }
    overlay.mkdir(parents=True)
    (overlay / "Offline").mkdir()
    (overlay / "Offline/.muse").symlink_to(source / ".muse")
    for name in files:
        target = overlay / "Offline" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source / name)
    (overlay / "backing").symlink_to(backing, target_is_directory=True)
    with (overlay / "overlay-receipt.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    verify(overlay)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", type=Path)
    parser.add_argument("--backing", type=Path)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.overlay.resolve(strict=True))
    elif args.offline is None or args.backing is None:
        parser.error("--offline and --backing are required for preparation")
    else:
        prepare(args.offline, args.backing, args.overlay)
