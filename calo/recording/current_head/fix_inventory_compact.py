"""Install the reviewed compact inventory header into the isolated source/overlay; never build."""
from pathlib import Path
import argparse
import difflib
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
OLD_MANIFEST = "35d91a0a328206579e54cf820f0ed87874816d504e2cff176f9489cd276dbeca"
OLD_RECEIPT = "5a101a27cd3b6c2afbf9f6af5ee8652e37f5cda2fa06d3b6182d0aaf24821eff"
OLD_HEADER = "b4a247f69542dc654ec796dfcf9941e62f9e13b91f889fea171f25f81e4fed3f"
HEADER = "Mu2eG4/inc/Mu2eG4PlacementInventory.hh"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_header(header):
    required = (
        '"mu2e-placement-inventory-v2',
        'static constexpr Count byteLimit = 1073741824;',
        'static constexpr Count textLimit = 65536;',
        'static constexpr Count memoryLimit = 268435456;',
        'static constexpr Count placementLimit = 1000000;',
        'static constexpr unsigned depthLimit = 128;',
        'std::min(maxBytes_, memoryLimit)',
        'inventory.capture();', 'inventory.serialize(g4Run, -1)',
        'inventory.writeExclusive(g4Run, expected)',
        'std::pmr::vector<Placement>', 'unsupported_metadata_only',
    )
    text = header.decode()
    for marker in required:
        if marker not in text:
            raise ValueError("compact inventory contract absent: " + marker)
    for forbidden in ('LocateGlobalPoint', 'G4UniformRand', 'SetTranslation(', 'SetRotation('):
        if forbidden in text:
            raise ValueError("nonobservational compact inventory")


def apply(source, overlay):
    source, overlay = source.resolve(strict=True), overlay.resolve(strict=True)
    pins_path, receipt_path = HERE / "source-manifest.json", overlay / "overlay-receipt.json"
    pins_bytes, receipt_bytes = pins_path.read_bytes(), receipt_path.read_bytes()
    if sha(pins_bytes) != OLD_MANIFEST or sha(receipt_bytes) != OLD_RECEIPT:
        raise ValueError("compact-inventory baseline drift")
    spec = importlib.util.spec_from_file_location("compact_overlay_guard", HERE / "prepare_overlay.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    before = guard.verify(overlay)
    if Path(before["source_offline"]).resolve() != source:
        raise ValueError("overlay belongs to another source")
    inventory = HERE.parent / "inventory"
    inventory_pins = json.loads((inventory / "source-manifest.json").read_text())
    header = (inventory / "payload/Mu2eG4PlacementInventory.hh").read_bytes()
    if sha(header) != inventory_pins["patched"][HEADER]:
        raise ValueError("compact review payload hash drift")
    if sha((inventory / "placement-inventory.patch").read_bytes()) != inventory_pins["patch_sha256"]:
        raise ValueError("compact review patch drift")
    if sha((source / HEADER).read_bytes()) != OLD_HEADER:
        raise ValueError("compact inventory prior header drift")
    for name, expected in inventory_pins["patched"].items():
        if name != HEADER and sha((source / name).read_bytes()) != expected:
            raise ValueError("compact inventory must not change installed hook")
    check_header(header)
    patch = "".join(difflib.unified_diff((source / HEADER).read_text().splitlines(True),
                                       header.decode().splitlines(True), fromfile="a/" + HEADER, tofile="b/" + HEADER))
    archives = {
        HERE / "compact-inventory-fix.patch": patch.encode(),
        overlay / "pre-compact-inventory-source-manifest.json": pins_bytes,
        overlay / "pre-compact-inventory-overlay-receipt.json": receipt_bytes,
        overlay / "pre-compact-inventory-header.hh": (source / HEADER).read_bytes(),
    }
    if any(path.exists() for path in archives):
        raise ValueError("compact inventory archive collision")
    pins, receipt = json.loads(pins_bytes), json.loads(receipt_bytes)
    pins["patched_sources"][HEADER] = sha(header)
    pins["inventory_patch_sha256"] = inventory_pins["patch_sha256"]
    pins["compact_inventory_fix"] = {
        "prior_source_manifest_sha256": OLD_MANIFEST,
        "prior_header_sha256": OLD_HEADER, "patch_sha256": sha(patch.encode()),
        "schema": "mu2e-placement-inventory-v2",
        "configured_file_row_caps_unchanged": True,
        "helper_allocator_hard_cap_bytes": 268435456,
        "rebuild_required": True,
    }
    pins["status"] = "compact v2 inventory source prepared; prior native manifest evidence retained; user rebuild required"
    new_pins = (json.dumps(pins, indent=2, sort_keys=True) + "\n").encode()
    receipt["package_files"][HEADER] = sha(header)
    receipt["source_manifest_sha256"] = sha(new_pins)
    receipt["compact_inventory_fix"] = {"prior_receipt_sha256": OLD_RECEIPT, "user_rebuild_pending": True}
    for path, data in archives.items():
        with path.open("xb") as stream:
            stream.write(data)
    (source / HEADER).write_bytes(header)
    pins_path.write_bytes(new_pins)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    guard.verify(overlay)
    print("Compact header SHA256:", sha(header))
    print("Compact delta patch SHA256:", sha(patch.encode()))
    print("Source manifest SHA256:", sha(pins_path.read_bytes()))
    print("Overlay receipt SHA256:", sha(receipt_path.read_bytes()))
    print("User rebuild required; no compiler/linker invocation or native execution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    args = parser.parse_args()
    apply(args.offline, args.overlay)
