"""Add the observed missing direct cetlib dependency to both build descriptions; never build."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
OLD_MANIFEST = "451124bf577577639528dd2b379d720664b489100cb8d7687a46913307fc535e"
OLD_RECEIPT = "60565e8487f512e054959aed8e2e4b4d21b251845cd99d994f007a3a84967545"
CHANGES = {
    "CMakeLists.txt": ("      fhiclcpp::fhiclcpp\n      cetlib_except::cetlib_except",
                      "      fhiclcpp::fhiclcpp\n      cetlib::cetlib\n      cetlib_except::cetlib_except"),
    "src/SConscript": ("    'fhiclcpp',\n    'cetlib_except',",
                      "    'fhiclcpp',\n    'cetlib',\n    'cetlib_except',"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def apply(source, overlay):
    source = source.resolve(strict=True)
    overlay = overlay.resolve(strict=True)
    manifest_path = HERE / "source-manifest.json"
    receipt_path = overlay / "overlay-receipt.json"
    manifest_bytes = manifest_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    if sha(manifest_bytes) != OLD_MANIFEST or sha(receipt_bytes) != OLD_RECEIPT:
        raise ValueError("manifest-linkage baseline drift")
    spec = importlib.util.spec_from_file_location("manifest_overlay_guard", HERE / "prepare_overlay.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.verify(overlay)
    review = HERE.parent / "manifest"
    review_pins = json.loads((review / "source_manifest.json").read_text())
    for name, expected in review_pins["proposal_files"].items():
        if sha((review / name).read_bytes()) != expected:
            raise ValueError("review proposal drift: " + name)
    replacements = {}
    for name, (old, new) in CHANGES.items():
        relative = "CaloRecording/" + name
        text = (source / relative).read_text()
        if text.count(old) != 1:
            raise ValueError("nonunique direct-library anchor: " + name)
        replacement = text.replace(old, new).encode()
        if replacement != (review / name).read_bytes():
            raise ValueError("review contains changes beyond the direct-library addition: " + name)
        replacements[relative] = replacement
    for name, data in (("pre-cetlib-fix-source-manifest.json", manifest_bytes),
                       ("pre-cetlib-fix-overlay-receipt.json", receipt_bytes)):
        with (overlay / name).open("xb") as stream:
            stream.write(data)
    manifest = json.loads(manifest_bytes)
    receipt = json.loads(receipt_bytes)
    for relative, data in replacements.items():
        (source / relative).write_bytes(data)
        manifest["patched_sources"][relative] = sha(data)
        receipt["package_files"][relative] = sha(data)
    manifest["manifest_review_pins_sha256"] = sha((review / "source_manifest.json").read_bytes())
    manifest["manifest_cetlib_link_fix"] = {
        "prior_source_manifest_sha256": OLD_MANIFEST,
        "symbol": "cet::detail::provide_file_path[abi:cxx11](char const*)",
        "direct_library": "cetlib / cetlib::cetlib",
        "rebuild_required": True,
    }
    manifest["status"] = "user link failure diagnosed; mirrored direct cetlib dependency added, user rebuild pending"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt["source_manifest_sha256"] = sha(manifest_path.read_bytes())
    receipt["manifest_cetlib_link_fix"] = {"prior_receipt_sha256": OLD_RECEIPT, "user_rebuild_pending": True}
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    guard.verify(overlay)
    for relative, data in replacements.items():
        print(relative, sha(data))
    print("Updated source manifest SHA256:", sha(manifest_path.read_bytes()))
    print("Updated overlay receipt SHA256:", sha(receipt_path.read_bytes()))
    print("No compiler/linker invocation or native validation performed; user rebuild required.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    args = parser.parse_args()
    apply(args.offline, args.overlay)
