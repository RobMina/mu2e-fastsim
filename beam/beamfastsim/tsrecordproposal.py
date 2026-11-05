"""Verify and assemble an unapplied native-recording proposal; never build or run it."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import re
from pathlib import Path, PurePosixPath


class ProposalError(ValueError):
    """A source, destination or proposal binding is unsupported."""


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProposalError("invalid relative source path")
    name = PurePosixPath(value)
    if name.is_absolute() or ".." in name.parts or str(name) != value:
        raise ProposalError("unsafe relative source path")
    return value


def _bound_file(root, name):
    path = root / _relative(name)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ProposalError(f"missing, linked or escaping source: {name}")
    return path


def verify(source_root, proposal_root):
    """Verify exact base files and proposal assets, without modifying either source."""
    source_root, proposal_root = Path(source_root), Path(proposal_root)
    pins = json.loads(_bound_file(proposal_root, "source_pins.json").read_text())
    if pins.get("schema") != "target-recording-source-proposal-v1" or pins.get("library_eligible") is not False:
        raise ProposalError("unsupported proposal schema or eligibility")
    for section, root in (("source_sha256", source_root), ("proposal_sha256", proposal_root)):
        entries = pins.get(section)
        if not isinstance(entries, dict) or not entries:
            raise ProposalError(f"missing {section}")
        for name, digest in entries.items():
            if not isinstance(digest, str) or len(digest) != 64 or _digest(_bound_file(root, name)) != digest:
                raise ProposalError(f"source/proposal hash mismatch: {name}")
    required_assets = {"target_recording.patch", "TargetRecording.hh", "TargetRecording.cc", "README.md"}
    if not required_assets <= set(pins["proposal_sha256"]):
        raise ProposalError("unbound proposal asset")
    revision = pins.get("offline_revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ProposalError("invalid pinned Offline revision")
    generated = {"Mu2eG4/inc/TargetRecording.hh", "Mu2eG4/src/TargetRecording.cc",
                 "PROPOSAL_STATUS.json", "ASSEMBLY_FAILED"}
    if generated & set(pins["source_sha256"]):
        raise ProposalError("base source collides with generated proposal outputs")
    changes = pins.get("patched_sha256")
    if not isinstance(changes, dict) or not changes or not set(changes) <= set(pins["source_sha256"]):
        raise ProposalError("unbound patch target")
    for name in changes:
        _relative(name)
    patch_lines = _bound_file(proposal_root, "target_recording.patch").read_text().splitlines()
    old = [line[6:] for line in patch_lines if line.startswith("--- a/")]
    new = [line[6:] for line in patch_lines if line.startswith("+++ b/")]
    if (old != new or len(old) != len(set(old)) or set(old) != set(changes)
            or any(line.startswith(("diff --git", "rename ", "copy ", "new file mode", "deleted file mode"))
                   for line in patch_lines)
            or sum(line.startswith("--- ") for line in patch_lines) != len(old)
            or sum(line.startswith("+++ ") for line in patch_lines) != len(new)):
        raise ProposalError("patch paths differ from the exact reviewed target set")
    return pins


def _apply_exact_patch(destination, patch):
    lines = patch.read_text().splitlines(keepends=True)
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- a/"):
            raise ProposalError("expected exact unified-diff file header")
        name = _relative(lines[index][6:].rstrip("\n"))
        index += 1
        if index >= len(lines) or lines[index] != f"+++ b/{name}\n":
            raise ProposalError("mismatched patch file headers")
        index += 1
        path = _bound_file(destination, name)
        original = path.read_text().splitlines(keepends=True)
        output, cursor = [], 0
        while index < len(lines) and not lines[index].startswith("--- a/"):
            header = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n", lines[index])
            if not header:
                raise ProposalError("unsupported patch hunk header")
            old_start, old_count, new_start, new_count = (int(header.group(1)), int(header.group(2) or 1),
                                                         int(header.group(3)), int(header.group(4) or 1))
            start = old_start - 1 if old_count else old_start
            if start < cursor or start > len(original):
                raise ProposalError("overlapping or out-of-range patch hunk")
            output.extend(original[cursor:start])
            if new_start != len(output) + (1 if new_count else 0):
                raise ProposalError("incorrect new-file hunk offset")
            cursor, used, produced = start, 0, 0
            index += 1
            while index < len(lines) and not lines[index].startswith(("@@ ", "--- a/")):
                line = lines[index]
                if not line or line[0] not in " +-":
                    raise ProposalError("unsupported patch line")
                if line[0] in " -":
                    if cursor >= len(original) or original[cursor] != line[1:]:
                        raise ProposalError("patch context does not exactly match base")
                    cursor += 1
                    used += 1
                if line[0] in " +":
                    output.append(line[1:])
                    produced += 1
                index += 1
            if used != old_count or produced != new_count:
                raise ProposalError("patch hunk line counts disagree")
        output.extend(original[cursor:])
        path.write_text("".join(output))


def assemble(source_root, proposal_root, destination):
    """Create only a new isolated review tree after all input bindings pass."""
    source_root, proposal_root, destination = Path(source_root), Path(proposal_root), Path(destination)
    pins = verify(source_root, proposal_root)
    if destination.exists() or destination.is_symlink():
        raise ProposalError("destination must not exist")
    resolved = destination.resolve()
    if any(resolved.is_relative_to(root.resolve()) or root.resolve().is_relative_to(resolved)
           for root in (source_root, proposal_root)):
        raise ProposalError("destination overlaps a source/proposal tree")
    destination.mkdir(parents=False)
    try:
        for name in pins["source_sha256"]:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_bound_file(source_root, name), target)
        patch = _bound_file(proposal_root, "target_recording.patch").resolve()
        _apply_exact_patch(destination, patch)
        for name, digest in pins["source_sha256"].items():
            expected = pins["patched_sha256"].get(name, digest)
            if _digest(_bound_file(destination, name)) != expected:
                raise ProposalError(f"assembled source hash mismatch: {name}")
        for name, target_name in (("TargetRecording.hh", "Mu2eG4/inc/TargetRecording.hh"),
                                  ("TargetRecording.cc", "Mu2eG4/src/TargetRecording.cc")):
            shutil.copyfile(_bound_file(proposal_root, name), destination / target_name)
        receipt = {"schema": pins["schema"], "offline_revision": pins["offline_revision"],
                   "source_files_verified": len(pins["source_sha256"]),
                   "patch_files_verified": len(pins["patched_sha256"]),
                   "proposal_manifest_sha256": _digest(proposal_root / "source_pins.json"),
                   "library_eligible": False, "build_performed": False,
                   "native_execution_performed": False, "simulation_performed": False}
        (destination / "PROPOSAL_STATUS.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    except Exception:
        (destination / "ASSEMBLY_FAILED").write_text("Incomplete review tree; do not build or execute.\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = args.out.resolve()
    protected = (args.source_root.resolve(), args.proposal_root.resolve(), args.destination.resolve())
    if args.out.exists() or args.out.is_symlink() or any(report.is_relative_to(root) for root in protected):
        raise ProposalError("report must be new and outside source, proposal and assembled trees")
    with args.out.open("x") as stream:
        result = assemble(args.source_root, args.proposal_root, args.destination)
        stream.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
