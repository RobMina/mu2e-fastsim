"""Audited whole-donor-file tier-1b diagnostics; never infer generated exposure."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import uproot

KEYS = ("run", "subrun", "event")
SCHEMA = "valhist.library_variants.v1"
CODE_ROOT = Path(__file__).resolve().parents[1]
LIMITATIONS = [
    "Half libraries are disjoint-source diagnostics, NOT a one-sigma uncertainty.",
    "Whole-file bootstrap assumes exchangeable independent donor files; repeated files are intentional.",
    "Quantile edges are re-estimated for each variant by tier1b.build, not frozen to nominal edges.",
    "The fixed gen denominator is shared across variants and with validation shards 1-10; "
    "its uncertainty and that validation dependence are NOT resampled or removed.",
    "Donor identity certification covers all persisted dump/particles event keys, not unpersisted "
    "generated events. Nominal lib ROOT has lost donor keys; it cannot certify source independence.",
    "Source independence is not a held-out validation certificate; audit validation event IDs separately.",
]


def sha256(path):
    """Streaming hash, including on Python 3.9 (no hashlib.file_digest)."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sealed_json(path, value):
    path.write_bytes(_json_bytes(value))
    path.with_suffix(path.suffix + ".sha256").write_text(sha256(path) + "\n")


def _read_sealed(path):
    if sha256(path) != path.with_suffix(path.suffix + ".sha256").read_text().strip():
        raise ValueError("manifest SHA256 mismatch: " + str(path))
    return json.loads(path.read_text())


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("%s must be an integer >= %s" % (name, minimum))
    return value


def _record(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("not a regular file: " + str(path))
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _code_identity():
    """Content identity of the actual build code; not a claim about historical nominal code."""
    paths = sorted((CODE_ROOT / "crvfastsim").rglob("*.py")) + [Path(__file__).resolve()]
    spec = importlib.util.find_spec("mu2efastsim")
    if spec is None or spec.origin is None:
        raise ValueError("mu2efastsim must be on PYTHONPATH to fingerprint build code")
    paths += sorted(Path(spec.origin).parent.rglob("*.py"))
    return {str(p.resolve()): sha256(p) for p in paths}


def _new_output(outdir):
    # lexists also refuses dangling symlinks. Do not create parents on failed preflight.
    raw = Path(outdir).expanduser()
    if os.path.lexists(str(raw)):
        raise FileExistsError("refusing existing output: " + str(raw))
    out = raw.resolve()
    if out == CODE_ROOT or CODE_ROOT in out.parents:
        raise ValueError("outputs must be outside the code tree")
    if not out.parent.is_dir():
        raise FileNotFoundError("output parent must already exist: " + str(out.parent))
    return out


def _publish(stage, out):
    # Exclusive reservation: unlike rename(stage, out), cannot replace an existing empty directory.
    out.mkdir()
    try:
        for source in stage.iterdir():
            source.chmod(0o444)
            source.rename(out / source.name)
    except BaseException:
        shutil.rmtree(out)
        raise


def _paths(metadata, field, parent):
    entries = metadata.get(field)
    if not isinstance(entries, list) or not entries or not all(isinstance(x, str) and x for x in entries):
        raise ValueError("nominal metadata requires a nonempty path list: " + field)
    paths = [str((parent / x).resolve(strict=True)) for x in entries]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate nominal source path in " + field)
    return paths


def _nominal(nominal_json, generated_per_file):
    _positive_int(generated_per_file, "generated_per_file")
    path = Path(nominal_json).resolve(strict=True)
    metadata = json.loads(path.read_text())
    donors, gen = _paths(metadata, "inputs", path.parent), _paths(metadata, "gen", path.parent)
    if len(donors) < 2 or len(donors) % 2:
        raise ValueError("alternating halves require an even donor-file count >= 2")
    expected = _positive_int(metadata.get("generated_scaled_to"), "nominal generated_scaled_to")
    if len(donors) * generated_per_file != expected:
        raise ValueError("explicit per-file exposure times donor-file count does not equal nominal generated_scaled_to")
    return path, metadata, donors, gen


def _event_inventory(path, tree):
    """All rows, full integer tuples, vectorized within-file deduplication; no packed keys."""
    with uproot.open(path) as root:
        cycles = sorted(int(k.rsplit(";", 1)[1]) for k in root.keys(recursive=True, cycle=True)
                        if k.rsplit(";", 1)[0] == tree)
        if not cycles:
            raise ValueError("%s: missing %s" % (path, tree))
        table = root["%s;%s" % (tree, cycles[-1])]
        arrays = table.arrays(list(KEYS), library="np")
        columns = [arrays[k] for k in KEYS]
        for column in columns:
            if (column.ndim != 1 or column.dtype.kind not in "iu" or
                    np.any(column < 0) or np.any(column > np.iinfo(np.int64).max)):
                raise ValueError("invalid integer event identity in " + str(path))
        keys = np.unique(np.stack(columns, axis=1).astype("<i8"), axis=0)
        if not len(keys):
            raise ValueError("empty event identity inventory: " + str(path))
        return keys, {"tree": tree, "cycles": cycles, "selected_cycle": cycles[-1],
                      "rows": int(table.num_entries), "unique_events": len(keys),
                      "event_keys_sha256": hashlib.sha256(keys.tobytes()).hexdigest()}


def _audit_bank(paths, tree):
    records, populations = [], []
    for path in paths:
        start = time.monotonic()
        record = _record(path)
        keys, inventory = _event_inventory(path, tree)
        if sha256(path) != record["sha256"]:
            raise ValueError("input changed during audit: " + path)
        record["events"] = inventory
        records.append(record)
        populations.append(keys)
        print("audited %s in %.3f s" % (path, time.monotonic() - start), file=sys.stderr, flush=True)
    all_keys = np.concatenate(populations)
    unique = np.unique(all_keys, axis=0)
    if len(unique) != len(all_keys):
        raise ValueError("cross-file duplicate event identities in " + tree)
    return records, unique


def _variants(nfiles, generated_per_file, bootstrap, seed):
    _positive_int(bootstrap, "bootstrap", 0)
    _positive_int(seed, "seed", 0)
    choices = {"half_even": list(range(0, nfiles, 2)), "half_odd": list(range(1, nfiles, 2))}
    # Each replica has its own deterministic stream; adding replicas preserves old ones.
    for index in range(1, bootstrap + 1):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, index])))
        choices["bootstrap_%04d" % index] = rng.integers(0, nfiles, size=nfiles).tolist()
    return {name: {"donor_indices": indices, "generated": len(indices) * generated_per_file,
                   "file_multiplicity": np.bincount(indices, minlength=nfiles).tolist()}
            for name, indices in choices.items()}


def plan(nominal_json, outdir, generated_per_file, bootstrap=0, seed=0):
    """Validate everything before publishing an immutable plan; return its content."""
    out = _new_output(outdir)
    path, metadata, donors, gen = _nominal(nominal_json, generated_per_file)
    variants = _variants(len(donors), generated_per_file, bootstrap, seed)
    nominal_record = _record(path)
    root_record = _record(path.with_suffix(".root"))
    code = _code_identity()
    donor_records, donor_keys = _audit_bank(donors, "dump/particles")
    gen_records, gen_keys = _audit_bank(gen, "dump/gen")
    overlap = len(donor_keys) + len(gen_keys) - len(np.unique(np.concatenate([donor_keys, gen_keys]), axis=0))
    if overlap:
        raise ValueError("donor/gen cross-file duplicate event identities")
    historical = {k: metadata[k] for k in ("code_identity", "revision", "commit", "git") if k in metadata}
    result = {"schema": SCHEMA, "nominal_json": nominal_record, "nominal_root": root_record,
              "nominal_build_code_identity": {"status": "reported_not_verified" if historical else "unavailable",
                                              "metadata_claims": historical},
              "build_code_sha256": code, "generated_per_file": generated_per_file,
              "nominal_generated": metadata["generated_scaled_to"],
              "donors": donor_records, "gen": gen_records, "bootstrap": bootstrap, "seed": seed,
              "rng": "numpy.PCG64/SeedSequence([seed, 1-based replica])", "numpy_version": np.__version__,
              "ordering": "nominal inputs list order; zero-based alternating indices",
              "certification": {"donor_cross_file_duplicates": 0, "gen_cross_file_duplicates": 0,
                                "donor_gen_overlap": int(overlap), "donor_unique_events": len(donor_keys),
                                "gen_unique_events": len(gen_keys)},
              "variants": variants, "limitations": LIMITATIONS}
    _verify_inputs(result)
    with tempfile.TemporaryDirectory(prefix=".library-plan-", dir=str(out.parent)) as tmp:
        stage = Path(tmp)
        _sealed_json(stage / "plan.json", result)
        _publish(stage, out)
    return result


def _verify_inputs(manifest):
    for record in [manifest["nominal_json"], manifest["nominal_root"]] + manifest["donors"] + manifest["gen"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("input SHA256 mismatch: " + record["path"])
    if _code_identity() != manifest["build_code_sha256"]:
        raise ValueError("build code changed since planning; create a new plan")


def _validate_plan(manifest):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported plan schema")
    _verify_inputs(manifest)
    _, metadata, donors, gen = _nominal(manifest["nominal_json"]["path"], manifest["generated_per_file"])
    if donors != [r["path"] for r in manifest["donors"]] or gen != [r["path"] for r in manifest["gen"]]:
        raise ValueError("plan source lists differ from nominal metadata")
    if manifest["nominal_generated"] != metadata["generated_scaled_to"]:
        raise ValueError("plan nominal exposure mismatch")
    if manifest["variants"] != _variants(len(donors), manifest["generated_per_file"],
                                        manifest["bootstrap"], manifest["seed"]):
        raise ValueError("plan variants or exposure do not match deterministic specification")
    if manifest["limitations"] != LIMITATIONS:
        raise ValueError("plan limitations were changed")


def _build_library(donors, gen, counters, generated, output):
    # Explicit existing local counters: no find_counters lookup/download/cache writes.
    from crvfastsim import geometry, tier1b
    geom = geometry.from_offline_config(str(counters))
    library, info = tier1b.build(donors, gen, geom, generated=generated)
    tier1b.write_library(str(output), library)
    return info


def build(plan_json, variant, counters, outdir):
    """Build exactly one explicit variant in a NEW directory, with no implicit all-variants run."""
    out = _new_output(outdir)
    path = Path(plan_json).resolve(strict=True)
    manifest = _read_sealed(path)
    plan_hash = sha256(path)
    _validate_plan(manifest)
    if variant not in manifest["variants"]:
        raise ValueError("unknown variant: " + variant)
    counters_record = _record(counters)
    selection = manifest["variants"][variant]
    donors = [manifest["donors"][i]["path"] for i in selection["donor_indices"]]
    gen = [r["path"] for r in manifest["gen"]]
    with tempfile.TemporaryDirectory(prefix=".library-build-", dir=str(out.parent)) as tmp:
        stage = Path(tmp)
        info = _build_library(donors, gen, counters_record["path"], selection["generated"], stage / "library.root")
        if info.get("generated_scaled_to") != selection["generated"] or info.get("files") != len(donors):
            raise ValueError("builder reported inconsistent exposure/file count")
        # Refuse publication if any source, code, geometry or plan drifted during the heavy build.
        _verify_inputs(manifest)
        if sha256(counters_record["path"]) != counters_record["sha256"] or sha256(path) != plan_hash:
            raise ValueError("geometry or plan changed during build")
        result = {**info, "inputs": donors, "gen": gen,
                  "provenance": {"schema": SCHEMA, "variant": variant, "selection": selection,
                                 "plan_path": str(path), "plan_sha256": plan_hash,
                                 "counters": counters_record, "build_code_sha256": manifest["build_code_sha256"],
                                 "nominal_build_code_identity": manifest["nominal_build_code_identity"],
                                 "library_sha256": sha256(stage / "library.root"), "limitations": LIMITATIONS}}
        _sealed_json(stage / "library.json", result)
        # A frozen copy keeps the build independently auditable if the original plan is moved.
        _sealed_json(stage / "plan.json", manifest)
        _publish(stage, out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="hash and audit all source event identities, then freeze variants")
    p.add_argument("nominal_json")
    p.add_argument("--outdir", required=True)
    p.add_argument("--generated-per-file", type=int, required=True)
    p.add_argument("--bootstrap", type=int, default=0, metavar="B")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("build", help="build one named variant, rechecking all hashes")
    p.add_argument("plan_json")
    p.add_argument("--variant", required=True)
    p.add_argument("--counters", required=True, help="existing local Offline counters text file")
    p.add_argument("--outdir", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = (plan if command == "plan" else build)(**args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, "%s: %s\n" % (command, error))
    print("planned " + ", ".join(result["variants"]) if command == "plan" else "built " + args["variant"])


if __name__ == "__main__":
    main()
