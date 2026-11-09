"""Shared whole-generator-file denominator bootstraps of an audited tier-1b library.

No donor rebuild, transport, or evaluation: one immutable library per global realization.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import uproot

from crvfastsim import tables, tier1b
from . import library_variants as V

SCHEMA = "valhist.denominator_variants.v1"
BIN = ["apdg", "kebin"]
LIMITATIONS = [
    "Denominator-only, conditional on the audited input donor library; NOT a joint bootstrap.",
    "One whole-file realization is shared globally across all evaluation shards using this library.",
    "File bootstrap assumes exchangeable independent generator blocks and the same generator configuration.",
    "Event inventories certify persisted IDs, not missing events or independence of production conditions.",
    "Held-out checks cover the supplied complete gen inventories and the entire audited donor bank only.",
    "Explicit event counts assert complete gen coverage, including events with no nonmuon primaries.",
    "Unsupported shower bins or probabilities above one fail; no clipping, pseudocounts or redraws.",
    "Finite donors, validation exposure, response noise and covariance are not a calibrated error bar here.",
]


def _digest(value):
    return hashlib.sha256(V._json_bytes(value)).hexdigest()


def _code_identity():
    return {**V._code_identity(), str(Path(__file__).resolve()): V.sha256(__file__)}


def _check_record(record):
    if V._record(record["path"]) != {k: record[k] for k in ("path", "sha256", "bytes")}:
        raise ValueError("input SHA256/size mismatch: " + record["path"])


def _base(library_json):
    """Require a sealed library_variants build and its unchanged, sealed sibling audit."""
    path = Path(library_json).resolve(strict=True)
    metadata = V._read_sealed(path)
    audit_path = path.parent / "plan.json"
    audit = V._read_sealed(audit_path)
    V._validate_plan(audit)
    provenance = metadata["provenance"]
    selection = audit["variants"].get(provenance["variant"])
    expected = {
        "schema": V.SCHEMA, "selection": selection,
        "plan_sha256": V.sha256(audit_path),
        "build_code_sha256": audit["build_code_sha256"],
        "nominal_build_code_identity": audit["nominal_build_code_identity"],
        "limitations": V.LIMITATIONS,
        "library_sha256": V.sha256(path.with_suffix(".root")),
    }
    if selection is None or any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError("base library provenance mismatch")
    if (metadata["inputs"] != [audit["donors"][i]["path"] for i in selection["donor_indices"]]
            or metadata["gen"] != [r["path"] for r in audit["gen"]]
            or metadata["generated_scaled_to"] != selection["generated"]
            or metadata["files"] != len(selection["donor_indices"])):
        raise ValueError("base library selection/exposure mismatch")
    _check_record(provenance["counters"])
    records = {"json": V._record(path), "root": V._record(path.with_suffix(".root")),
               "donor_plan": V._record(audit_path)}
    return metadata, audit, records


def _counts(record):
    """Vectorized sufficient statistics; ALL primary rows of each file remain coupled."""
    frame = tables.read(record["path"], "gen", list(V.KEYS) + ["pdg", "e"])
    if (frame.pdg.dtype.kind not in "iu" or np.any(frame.pdg == np.iinfo(np.int64).min)
            or np.any(frame.pdg > np.iinfo(np.int64).max)
            or frame.e.dtype.kind not in "iuf" or not np.isfinite(frame.e).all()):
        raise ValueError("invalid generator pdg/energy: " + record["path"])
    frame = frame.loc[frame.pdg.abs() != tables.PRIMARY_MUON_PDG].copy()
    frame["apdg"] = frame.pdg.abs().astype(np.int64)
    frame["kebin"] = tier1b._ke_bin((frame.e - frame.apdg.map(tier1b.MASS).fillna(0.0)).to_numpy())
    counts = frame.groupby(BIN).size().rename("count").reset_index()
    return counts.astype(np.int64).to_numpy().tolist()


def _exposures(values, records, label):
    if not isinstance(values, (list, tuple)) or len(values) != len(records):
        raise ValueError(label + " requires one explicit event count per file, in source order")
    values = [V._positive_int(n, label) for n in values]
    if values != [r["events"]["unique_events"] for r in records]:
        raise ValueError(label + " does not match complete persisted gen event inventory")
    return values


def _overlap(left, right):
    return int(len(left) + len(right) - len(np.unique(np.concatenate([left, right]), axis=0)))


def _variants(events, replicas, seed):
    V._positive_int(replicas, "replicas")
    V._positive_int(seed, "seed", 0)
    result = {}
    for index in range(1, replicas + 1):
        # Separate domain from library_variants' [seed, index] donor stream.
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, index, 1])))
        indices = rng.integers(0, len(events), size=len(events)).tolist()
        selection = {"generator_indices": indices,
                     "file_multiplicity": np.bincount(indices, minlength=len(events)).tolist(),
                     "selected_generator_events": sum(events[i] for i in indices)}
        result["denominator_%04d" % index] = {**selection, "selection_sha256": _digest(selection)}
    return result


def _reach(showers, generators, multiplicity, generated):
    """Ratio of weighted totals; NEVER deduplicate repeated bootstrap event identities.

    The existing _reach(read_many(repeated_paths)) would deduplicate the exposure but
    not primary rows. Instead sum each file's complete event and primary counts.
    """
    V._positive_int(generated, "generated")
    weights = np.asarray(multiplicity)
    if (weights.ndim != 1 or len(weights) != len(generators) or weights.dtype.kind not in "iu"
            or np.any(weights < 0) or not np.any(weights)):
        raise ValueError("invalid file multiplicity")
    events = sum(int(m) * r["events"]["unique_events"] for r, m in zip(generators, weights))
    blocks = [pd.DataFrame(r["bin_counts"], columns=BIN + ["count"]).assign(
        count=lambda x, m=int(m): x["count"].astype(np.int64) * m)
        for r, m in zip(generators, weights) if m and r["bin_counts"]]
    if not blocks:
        raise ValueError("bootstrap contains no nonmuon generator primaries")
    ng = pd.concat(blocks, ignore_index=True).groupby(BIN)["count"].sum().reset_index()
    scaled = np.rint(ng["count"].to_numpy() * (float(generated) / events))
    if np.any(~np.isfinite(scaled)) or np.any(scaled >= float(2 ** 63)):
        raise ValueError("scaled denominator exceeds int64")
    ng["ngen"] = scaled.astype(np.int64)
    ns = showers.groupby(BIN).size().rename("nshower").reset_index()
    checked = ns.merge(ng, on=BIN, how="left")
    if checked.ngen.isna().any() or np.any(checked.nshower > checked.ngen):
        raise ValueError("unsupported shower bin or reach probability above one")
    result = ng.merge(ns, on=BIN, how="left").fillna({"nshower": 0})
    return result[BIN + ["ngen", "nshower"]].astype(np.int64), int(ng["count"].sum()), events


def _same_reach(actual, expected):
    columns = BIN + ["ngen", "nshower"]
    a = actual[columns].sort_values(BIN).reset_index(drop=True)
    b = expected[columns].sort_values(BIN).reset_index(drop=True)
    if not a.equals(b):
        raise ValueError("base libgen is not the source-reviewed denominator formula")


def _prepare(library_json, held_out, generator_events, held_out_events, replicas, seed):
    V._positive_int(replicas, "replicas")
    V._positive_int(seed, "seed", 0)
    metadata, audit, base = _base(library_json)
    code = _code_identity()
    paths = [str(Path(p).resolve(strict=True)) for p in held_out]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("held_out requires distinct complete gen files")
    donors, donor_keys = V._audit_bank([r["path"] for r in audit["donors"]], "dump/particles")
    generators, gen_keys = V._audit_bank([r["path"] for r in audit["gen"]], "dump/gen")
    held, held_keys = V._audit_bank(paths, "dump/gen")
    if donors != audit["donors"] or generators != audit["gen"]:
        raise ValueError("source inventory differs from donor audit")
    overlaps = {"donor_generator": _overlap(donor_keys, gen_keys),
                "held_out_donor": _overlap(held_keys, donor_keys),
                "held_out_generator": _overlap(held_keys, gen_keys)}
    if any(overlaps.values()):
        raise ValueError("event ID overlap: " + str(overlaps))
    exposure = {"donor_generated": metadata["generated_scaled_to"],
                "generator_events": _exposures(generator_events, generators, "generator_events"),
                "held_out_events": _exposures(held_out_events, held, "held_out_events")}
    for record in generators:
        record["bin_counts"] = _counts(record)
    root = base["root"]["path"]
    reach, nonmu, _ = _reach(tables.read(root, "libshower"), generators,
                            [1] * len(generators), exposure["donor_generated"])
    _same_reach(tables.read(root, "libgen"), reach)
    if metadata["generated_nonmu"] != nonmu:
        raise ValueError("base generated_nonmu mismatch")
    sources = {"base": base, "donors": donors, "generators": generators, "held_out": held}
    result = {"schema": SCHEMA, "mode": "denominator_only", **sources,
              "source_sha256": _digest(sources), "exposure": exposure, "exposure_sha256": _digest(exposure),
              "base_donor_selection": metadata["provenance"]["selection"],
              "base_donor_selection_sha256": _digest(metadata["provenance"]["selection"]),
              "audit": {**overlaps, "held_out_unique_events": len(held_keys),
                        "scope": "entire donor bank and original generator bank; full persisted event tuples"},
              "replicas": replicas, "seed": seed,
              "rng": "numpy.PCG64/SeedSequence([seed, one_based_replica, 1])",
              "versions": {"numpy": np.__version__, "pandas": pd.__version__, "uproot": uproot.__version__},
              "code_sha256": code, "variants": _variants(exposure["generator_events"], replicas, seed),
              "limitations": LIMITATIONS}
    _verify(result)
    return result


def _verify(manifest):
    for record in list(manifest["base"].values()) + manifest["donors"] + manifest["generators"] + manifest["held_out"]:
        _check_record(record)
    if _code_identity() != manifest["code_sha256"]:
        raise ValueError("denominator code changed since planning")
    # Includes nominal inputs, geometry and base sidecars, not just the directly used ROOT.
    _base(manifest["base"]["json"]["path"])


def plan(library_json, outdir, held_out, generator_events, held_out_events, replicas=1, seed=0):
    """Seal a denominator-only plan; explicit complete event counts in each bank's file order."""
    out = V._new_output(outdir)
    result = _prepare(library_json, held_out, generator_events, held_out_events, replicas, seed)
    with tempfile.TemporaryDirectory(prefix=".denominator-plan-", dir=str(out.parent)) as tmp:
        stage = Path(tmp)
        V._sealed_json(stage / "plan.json", result)
        V._publish(stage, out)
    return result


def _write_variant(source, output, reach):
    """Copy all donor payloads; only append a new highest-cycle dump/libgen TTree."""
    shutil.copyfile(source, output)
    columns = {c: reach[c].to_numpy() for c in reach.columns}
    with uproot.update(output) as root:
        root.mktree("dump/libgen", {c: a.dtype for c, a in columns.items()})
        root["dump/libgen"].extend(columns)
    _same_reach(tables.read(output, "libgen"), reach)


def build(plan_json, variant, outdir):
    """Publish one global realization. Re-audit all inputs; never run physics or redraw a failure."""
    out = V._new_output(outdir)
    path = Path(plan_json).resolve(strict=True)
    manifest = V._read_sealed(path)
    plan_hash = V.sha256(path)
    if manifest.get("schema") != SCHEMA or manifest.get("mode") != "denominator_only":
        raise ValueError("unsupported denominator plan")
    if variant not in manifest["variants"]:
        raise ValueError("unknown variant: " + variant)
    _verify(manifest)
    fresh = _prepare(manifest["base"]["json"]["path"], [r["path"] for r in manifest["held_out"]],
                     manifest["exposure"]["generator_events"], manifest["exposure"]["held_out_events"],
                     manifest["replicas"], manifest["seed"])
    if fresh != manifest:
        raise ValueError("denominator plan differs from deterministic source/selection/exposure audit")
    selection = manifest["variants"][variant]
    root = manifest["base"]["root"]["path"]
    reach, nonmu, events = _reach(tables.read(root, "libshower"), manifest["generators"],
                                selection["file_multiplicity"], manifest["exposure"]["donor_generated"])
    metadata = V._read_sealed(Path(manifest["base"]["json"]["path"]))
    with tempfile.TemporaryDirectory(prefix=".denominator-build-", dir=str(out.parent)) as tmp:
        stage = Path(tmp)
        _write_variant(root, stage / "library.root", reach)
        _verify(manifest)
        if V.sha256(path) != plan_hash or V._read_sealed(path) != manifest:
            raise ValueError("plan changed during build")
        # Preserve donor statistics but do not present the old build provenance as this product's identity.
        result = {k: v for k, v in metadata.items() if k != "provenance"}
        result.update({"gen": [manifest["generators"][i]["path"] for i in selection["generator_indices"]],
                       "generated_nonmu": nonmu,
                       "provenance": {"schema": SCHEMA, "mode": "denominator_only", "variant": variant,
                                      "plan_sha256": plan_hash, "selection": selection,
                                      "source_sha256": manifest["source_sha256"],
                                      "exposure_sha256": manifest["exposure_sha256"],
                                      "selected_generator_events": events,
                                      "code_sha256": manifest["code_sha256"],
                                      "base": manifest["base"], "audit": manifest["audit"],
                                      "library_sha256": V.sha256(stage / "library.root"),
                                      "limitations": LIMITATIONS}})
        V._sealed_json(stage / "library.json", result)
        V._sealed_json(stage / "plan.json", manifest)
        V._publish(stage, out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("library_json", help="sealed library_variants build, with sibling plan.json")
    p.add_argument("--outdir", required=True)
    p.add_argument("--held-out", nargs="+", required=True, help="complete evaluation dump/gen files")
    p.add_argument("--generator-events", nargs="+", type=int, required=True)
    p.add_argument("--held-out-events", nargs="+", type=int, required=True)
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("build")
    p.add_argument("plan_json")
    p.add_argument("--variant", required=True)
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
