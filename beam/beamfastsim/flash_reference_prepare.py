"""Prepare an unfiltered reference configuration without running transport.

The manifest identifies prepared inputs, not a completed simulation or a certified
normalization. Geometry includes are expanded against one explicit runtime root;
field-map payloads and the executable still require a separate runtime audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"\s*(?://[^\n]*)?$', re.M)


def identity(path):
    """Hash one explicitly named local file, retaining its resolved identity."""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def flatten_geometry(path, root):
    """Preserve repeated includes and override order; refuse cycles or root escape."""
    root = Path(root).resolve(strict=True)
    sources = {}

    def expand(current, ancestors):
        current = current.resolve(strict=True)
        if current in ancestors:
            raise ValueError("Cyclic geometry include")
        text = current.read_text(encoding="ascii")
        sources[str(current)] = identity(current)
        if len(re.findall(r'^\s*#include\b', text, re.M)) != len(_INCLUDE.findall(text)):
            raise ValueError("Unsupported geometry include syntax")

        def replace(match):
            relative = Path(match.group(1))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Geometry includes must remain under offline_root")
            included = (root / relative).resolve(strict=True)
            if not included.is_relative_to(root):
                raise ValueError("Geometry include escapes offline_root")
            return expand(included, ancestors + (current,))

        return _INCLUDE.sub(replace, text)

    return expand(Path(path), ()), list(sources.values())


def _quote(value):
    value = str(value)
    if any(ord(c) < 32 or ord(c) > 126 for c in value):
        raise ValueError("FHiCL paths must be printable ASCII")
    return json.dumps(value, ensure_ascii=True)


def prepare(*, input_path, geom, offline_root, field, template, events, seed, physics_list, out):
    """Create a fresh configuration bundle; no shell, simulation or submission."""
    if type(events) is not int or not 1 <= events <= 1000:
        raise ValueError("Pilot events must be an integer in [1, 1000]")
    if type(seed) is not int or not 1 <= seed < 900000000:
        raise ValueError("Seed must be an integer in [1, 900000000)")
    if physics_list not in ("FTFP_BERT", "ShieldingM"):
        raise ValueError("Choose the declared FTFP_BERT or ShieldingM reference")
    out = Path(out).absolute()
    if out.exists():
        raise FileExistsError(out)
    inputs = {name: identity(path) for name, path in
              (("source_art", input_path), ("field_config", field), ("template", template))}
    geometry, closure = flatten_geometry(geom, offline_root)
    geometry += '\n// Disable sensitive-detector hit truncation for this reference.\nint g4.stepsSizeLimit = 0;\n'
    if "#include" in re.sub(r"//[^\n]*", "", geometry):
        raise ValueError("Unexpanded geometry include")
    raw = Path(template).read_text(encoding="ascii")
    job = '\n'.join([
        raw,
        f'source.fileNames: [{_quote(inputs["source_art"]["path"])}]',
        f'source.maxEvents: {events}',
        f'services.SeedService.baseSeed: {seed}',
        f'services.GeometryService.inputFile: {_quote(out / "geometry.txt")}',
        f'services.GeometryService.bFieldFile: {_quote(inputs["field_config"]["path"])}',
        f'physics.producers.g4run.physics.physicsListName: {_quote(physics_list)}',
        f'outputs.raw.fileName: {_quote(out / "raw.art")}',
        f'services.TFileService.fileName: {_quote(out / "audit.root")}',
        '',
    ])
    payloads = {"raw.fcl": raw, "geometry.txt": geometry, "job.fcl": job}
    manifest = {
        "schema": "flash-reference-preparation-v1", "transport_executed": False,
        "events_requested": events, "seed": seed, "physics_list": physics_list,
        "source_assignment": "Ordered RootInput prefix; all source hits, no resampling or added time offset",
        "normalization": "Selected source events and DS states only; no POT or board normalization certified",
        "offline_root": str(Path(offline_root).resolve()), "inputs": inputs, "geometry_sources": closure,
        "prepared_sha256": {name: hashlib.sha256(text.encode("ascii")).hexdigest()
                            for name, text in payloads.items()},
        "unverified": ["Installed module compatibility and expanded configuration",
                       "Executable/libraries and field-map payload identities",
                       "Input EventIDs, species and primary-state correspondence",
                       "Recorded event/particle/step completeness and cap headroom",
                       "Closed-surface photon-exit reconstruction and normalization"],
    }
    out.mkdir(parents=True, exist_ok=False)
    for name, text in payloads.items():
        with (out / name).open("x", encoding="ascii") as stream:
            stream.write(text)
    with (out / "preparation.json").open("x", encoding="ascii") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "geom", "offline-root", "field", "template", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physics-list", choices=("FTFP_BERT", "ShieldingM"), required=True)
    args = parser.parse_args(argv)
    prepare(input_path=args.input, geom=args.geom, offline_root=args.offline_root,
            field=args.field, template=args.template, events=args.events, seed=args.seed,
            physics_list=args.physics_list, out=args.out)


if __name__ == "__main__":
    main()
