"""Stage a new standalone field-probe source snapshot; no configure or compiler calls."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


SOURCES = ("CMakeLists.txt", "probe.cc", "ProbeTypes.hh", "ProbeAlgorithms.hh", "ProbeKernels.cu",
           "run_probe.py", "oracle.py", "README.md")
CARTMAP_FILES = ("CartMapIO.hh", "CartMapIO.cc")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(adapter, output):
    here = Path(__file__).resolve().parent
    pin = adapter / "adapter-manifest.json"
    if digest(pin) != "774e93dc8e2f8aec55b0e76947cb237c72300d5e1d122ccc89ca08b57fe28933":
        raise ValueError("Require the pinned h4 adapter snapshot")
    prepared = json.loads(pin.read_text())["prepared_sha256"]
    for name in CARTMAP_FILES:
        if digest(adapter / name) != prepared[name]:
            raise ValueError(f"Pinned h4 map parser changed: {name}")
    files = {name: here / name for name in SOURCES}
    files["field_contract.py"] = here.parent / "field_contract.py"
    files.update({name: adapter / name for name in CARTMAP_FILES})
    hashes = {name: digest(path) for name, path in files.items()}
    output.mkdir(exist_ok=False)
    for name, path in files.items():
        shutil.copyfile(path, output / name)
    manifest = {"format": "m2fs-field-probe-source-v1", "upstream_version": "0.6.3",
                "adapter_manifest_sha256": digest(pin), "sha256": hashes,
                "scope": "prepared source only; no C++/CUDA build or runtime validation"}
    (output / "probe-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.adapter.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
