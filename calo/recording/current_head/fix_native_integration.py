"""Apply reviewed trigger-path filter and inventory cap-diagnostic corrections; never build."""
from pathlib import Path
import argparse
import difflib
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
OLD_MANIFEST = "0b08b0f3babb5416baecf02ae77a14cafc0236feb684dc7a95c69dffb4c0f69e"
OLD_RECEIPT = "e97172afaf60f5348a7d6c8afc6447e8a94eeb32e22b7f86c3940ff19824ee04"
INVENTORY = "Mu2eG4/inc/Mu2eG4PlacementInventory.hh"
MANIFEST_FILES = ("src/CaloInputEventManifest_module.cc", "src/CaloIncidentManifest_module.cc",
                  "fcl/manifest_ce_review.fcl")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def apply(source, overlay):
    source = source.resolve(strict=True)
    overlay = overlay.resolve(strict=True)
    pins_path = HERE / "source-manifest.json"
    receipt_path = overlay / "overlay-receipt.json"
    pins_bytes, receipt_bytes = pins_path.read_bytes(), receipt_path.read_bytes()
    if sha(pins_bytes) != OLD_MANIFEST or sha(receipt_bytes) != OLD_RECEIPT:
        raise ValueError("native-integration baseline drift")
    spec = importlib.util.spec_from_file_location("native_integration_overlay_guard", HERE / "prepare_overlay.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.verify(overlay)
    review = HERE.parent / "manifest"
    review_pins = json.loads((review / "source_manifest.json").read_text())
    for name, expected in review_pins["proposal_files"].items():
        if sha((review / name).read_bytes()) != expected:
            raise ValueError("review manifest drift: " + name)
    inventory_pins = json.loads((HERE.parent / "inventory/source-manifest.json").read_text())
    header = (HERE.parent / "inventory/payload/Mu2eG4PlacementInventory.hh").read_bytes()
    if sha(header) != inventory_pins["patched"][INVENTORY]:
        raise ValueError("inventory review header drift")
    replacements = {"CaloRecording/" + name: (review / name).read_bytes() for name in MANIFEST_FILES}
    replacements[INVENTORY] = header
    for name in MANIFEST_FILES[:2]:
        text = replacements["CaloRecording/" + name].decode()
        if "public art::EDFilter" not in text or "return true;" not in text or "EDAnalyzer" in text:
            raise ValueError("manifest must be a nonselecting filter")
    if (b'static constexpr Count byteLimit = 1073741824;' not in header
            or b'static constexpr Count textLimit = 65536;' not in header):
        raise ValueError("inventory hard limits must not change")
    patch = ""
    for relative, data in sorted(replacements.items()):
        old = (source / relative).read_text()
        patch += "".join(difflib.unified_diff(old.splitlines(True), data.decode().splitlines(True),
                                            fromfile="a/" + relative, tofile="b/" + relative))
    with (HERE / "native-integration-fix.patch").open("x") as stream:
        stream.write(patch)
    for name, data in (("pre-native-integration-source-manifest.json", pins_bytes),
                       ("pre-native-integration-overlay-receipt.json", receipt_bytes)):
        with (overlay / name).open("xb") as stream:
            stream.write(data)
    pins, receipt = json.loads(pins_bytes), json.loads(receipt_bytes)
    for relative, data in replacements.items():
        (source / relative).write_bytes(data)
        pins["patched_sources"][relative] = sha(data)
        receipt["package_files"][relative] = sha(data)
    pins["inventory_patch_sha256"] = inventory_pins["patch_sha256"]
    pins["manifest_review_pins_sha256"] = sha((review / "source_manifest.json").read_bytes())
    pins["native_integration_fix"] = {
        "prior_source_manifest_sha256": OLD_MANIFEST,
        "patch_sha256": sha(patch.encode()),
        "changes": ["manifest EDAnalyzers to always-pass EDFilters on trigger path",
                    "inventory cap diagnostics distinguish serialization stages; limits unchanged"],
        "rebuild_required": True,
    }
    pins["status"] = "native checks exposed analyzer trigger-path rejection and inventory cap; corrected sources await user rebuild"
    pins_path.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
    receipt["source_manifest_sha256"] = sha(pins_path.read_bytes())
    receipt["native_integration_fix"] = {"prior_receipt_sha256": OLD_RECEIPT, "user_rebuild_pending": True}
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    guard.verify(overlay)
    for relative, data in replacements.items():
        print(relative, sha(data))
    print("Patch SHA256:", sha(patch.encode()))
    print("Source manifest SHA256:", sha(pins_path.read_bytes()))
    print("Overlay receipt SHA256:", sha(receipt_path.read_bytes()))
    print("User rebuild required. No compiler/linker invocation or new transport.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    args = parser.parse_args()
    apply(args.offline, args.overlay)
