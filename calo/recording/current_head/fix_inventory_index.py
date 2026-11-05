"""Apply the reviewed daughter-index compile fix and refresh source/overlay pins; never build."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
REL = "Mu2eG4/inc/Mu2eG4PlacementInventory.hh"
OLD_HEADER = "5d5fcd91980778f33b3c665f25db64df35fb1a27a6068fc0a32d7498f271f213"
OLD_MANIFEST = "794c95ad90ba77c517b7c42ded10fec26097b5e93d25ea0a4cf687b004a72e5a"
OLD_RECEIPT = "5a35eda1fd90e08f75190e4a977bec314b5a5b82a040b5d38b5a7254b5a2e0a0"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def apply(source, overlay):
    source = source.resolve(strict=True)
    overlay = overlay.resolve(strict=True)
    manifest_path = HERE / "source-manifest.json"
    receipt_path = overlay / "overlay-receipt.json"
    manifest_bytes = manifest_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    old_bytes = (source / REL).read_bytes()
    if sha(manifest_bytes) != OLD_MANIFEST or sha(receipt_bytes) != OLD_RECEIPT or sha(old_bytes) != OLD_HEADER:
        raise ValueError("compile-fix baseline drift")
    spec = importlib.util.spec_from_file_location("inventory_overlay_guard", HERE / "prepare_overlay.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.verify(overlay)
    text = old_bytes.decode()
    old = "    for (int i = 0; i < daughters; ++i) walk(lv->GetDaughter(i), path, chain, depth + 1, selected);"
    new = old.replace("int i = 0", "std::size_t i = 0")
    include = "#include <cmath>\n#include <cstdint>"
    if text.count(old) != 1 or text.count(include) != 1:
        raise ValueError("nonunique reviewed compile-fix anchors")
    replacement = text.replace(old, new).replace(include, "#include <cmath>\n#include <cstddef>\n#include <cstdint>").encode()
    review = HERE.parent / "inventory"
    inventory = json.loads((review / "source-manifest.json").read_text())
    if replacement != (review / "payload/Mu2eG4PlacementInventory.hh").read_bytes():
        raise ValueError("review payload differs from the two reviewed edits")
    if sha(replacement) != inventory["patched"][REL]:
        raise ValueError("inventory pins do not identify the corrected header")
    if sha((review / "placement-inventory.patch").read_bytes()) != inventory["patch_sha256"]:
        raise ValueError("inventory patch hash mismatch")
    for name, expected in inventory["patched"].items():
        if name != REL and sha((source / name).read_bytes()) != expected:
            raise ValueError("inventory companion source drift")
    for name, data in (("pre-index-fix-source-manifest.json", manifest_bytes),
                       ("pre-index-fix-overlay-receipt.json", receipt_bytes)):
        with (overlay / name).open("xb") as stream:
            stream.write(data)
    (source / REL).write_bytes(replacement)
    manifest = json.loads(manifest_bytes)
    manifest["patched_sources"][REL] = sha(replacement)
    manifest["inventory_patch_sha256"] = inventory["patch_sha256"]
    manifest["inventory_index_compile_fix"] = {
        "prior_header_sha256": OLD_HEADER,
        "prior_source_manifest_sha256": OLD_MANIFEST,
        "change": "std::size_t daughter index and explicit cstddef include",
        "native_api": "p110 G4LogicalVolume GetNoDaughters and GetDaughter use std::size_t",
        "rebuild_required": True,
    }
    manifest["status"] = "user build reported sign-compare error; index fix applied, user rebuild pending"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt = json.loads(receipt_bytes)
    receipt["package_files"][REL] = sha(replacement)
    receipt["source_manifest_sha256"] = sha(manifest_path.read_bytes())
    receipt["inventory_index_compile_fix"] = {
        "prior_receipt_sha256": OLD_RECEIPT,
        "user_rebuild_pending": True,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    guard.verify(overlay)
    print("Corrected header SHA256:", sha(replacement))
    print("Updated source manifest SHA256:", sha(manifest_path.read_bytes()))
    print("Updated overlay receipt SHA256:", sha(receipt_path.read_bytes()))
    print("No compilation or native validation performed; user rebuild required.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    args = parser.parse_args()
    apply(args.offline, args.overlay)
