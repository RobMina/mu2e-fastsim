"""Stage a new navigation-probe source snapshot and API pins without configuring or building."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCES = ("CMakeLists.txt", "probe.cc", "NavTypes.hh", "NavAlgorithms.hh", "NavKernels.cu",
           "README.md", "fixtures.py", "run_probe.py", "prepare.py")
API_PINS = {
    "celeritas/field/FieldPropagator.hh": "e2d40443a45590c3f31048ebe531182a1fa316305070f89aabab322fe96edd7b",
    "celeritas/field/FieldSubstepper.hh": "89eca96f51df7b9efd4abc474e8c933fe98f0f38fc1d9bc0285e6ddebce7d9e7",
    "celeritas/field/FieldDriverOptions.hh": "753d88e242dda4230d35e26548f29796b4c0f20059daa1e5b6650988e6c82964",
    "celeritas/field/CartMapField.hh": "bf2213f2e8a60724f11de4815dab17e5970596fdd631c2075c7a662ac3b3301f",
    "celeritas/field/MakeMagFieldPropagator.hh": "caabccb4084d8816dcfb5e7f6ea0d32277342b23cf4a2e503d110827de9afc3f",
    "celeritas/field/DormandPrinceIntegrator.hh": "bec36959c2cc1ef3b3d753ec09b0e3d41495b2bd7f147b868c7c2d1e1a231ce7",
    "geocel/vg/VecgeomParams.hh": "3bc1449214a735dc172f8f9fbd42196110ffb28871fea0a9285e6d836697dc8e",
    "geocel/vg/VecgeomTrackView.hh": "27ed34b50c0fc241ad52f2fca5a0ff3b157207e7dc301e8efd757bb84a467db6",
    "geocel/vg/VecgeomData.hh": "556f80cfeee406f867ef223e741d0efbde76e1f281fe851d1122993f49b126ab",
    "geocel/GeantGeoParams.hh": "f00c3cd5f1a975d27e3a38c4fc777f902476e5c08beebaf61b6c89b69b10133a",
}
ADAPTER_PIN = "774e93dc8e2f8aec55b0e76947cb237c72300d5e1d122ccc89ca08b57fe28933"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prepare(adapter, upstream, output):
    here = Path(__file__).resolve().parent
    require(not output.exists(), "Source destination must be new")
    require(digest(adapter / "adapter-manifest.json") == ADAPTER_PIN, "Not pinned h4 parser source")
    manifest = json.loads((adapter / "adapter-manifest.json").read_text())
    files = {name: here / name for name in SOURCES}
    for name in ("CartMapIO.cc", "CartMapIO.hh"):
        require(digest(adapter / name) == manifest["prepared_sha256"][name], "Modified h4 parser")
        files[name] = adapter / name
    for name, sha in API_PINS.items():
        require(digest(upstream / "src" / name) == sha, f"API mismatch: {name}")
    hashes = {name: digest(path) for name, path in files.items()}
    output.mkdir(parents=True, exist_ok=False)
    for name, path in files.items():
        shutil.copyfile(path, output / name)
    (output / "api-pins.sha256").write_text("".join(f"{sha}  {name}\n" for name, sha in API_PINS.items()))
    hashes["api-pins.sha256"] = digest(output / "api-pins.sha256")
    result = dict(format="m2fs-navigation-source-v1", upstream_version="0.6.3",
                  upstream_commit="ddf6c690e37d53b56d001e6bf0c26cb1b14963fa", adapter_manifest_sha256=ADAPTER_PIN,
                  sha256=hashes, installed_api_sha256=API_PINS,
                  scope="Prepared only; no C++/CUDA configuration, compilation or runtime")
    (output / "source-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "source.sha256").write_text("".join(f"{sha}  {name}\n" for name, sha in hashes.items())
                                        + f"{digest(output / 'source-manifest.json')}  source-manifest.json\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("new_output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.adapter.resolve(), args.upstream.resolve(), args.new_output.resolve()), indent=2))


if __name__ == "__main__":
    main()
