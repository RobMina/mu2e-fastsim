"""Plan versioned donor re-dumps and compare generator association changes without repairing rows."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pandas as pd
import uproot

from crvfastsim import tables
from valhist import denominator_provenance as P
from valhist.library_variants import sha256


def _new_json(path, result):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def summarize(frame):
    """All retained showers, with primary/gen identity and kinematic consistency diagnostics."""
    return {"showers": len(frame), "pdg_mismatch": int(frame.pdg_disagreement.sum()),
            "missing_join": int(frame.gen_join_missing.sum()), "negative_ke": int(frame.negative_ke.sum()),
            "nonfinite": int(frame.nonfinite.sum()),
            "same_pdg_energy_delta_gt_1MeV": int(((~frame.pdg_disagreement) & (frame.start_minus_gen_e.abs() > 1)).sum()),
            "max_abs_start_gen_energy_MeV": float(frame.start_minus_gen_e.abs().max()) if len(frame) else None,
            "max_abs_start_gen_time_ns": float(frame.start_minus_gen_t.abs().max()) if len(frame) else None}


def compare(old, new, out):
    """Require unchanged steps and legacy particle payload; quantify only changed generator associations."""
    hashes = {str(p): sha256(p) for p in (old, new)}
    invariant = {}
    for tree in ("steps", "particles", "subruns"):
        a, b = tables.read(old, tree), tables.read(new, tree)
        pd.testing.assert_frame_equal(a, b[a.columns], check_exact=True, check_dtype=False)
        invariant[tree] = {"rows": len(a), "legacy_columns_identical": True,
                           "added_columns": [c for c in b.columns if c not in a.columns]}
    with uproot.open(new) as root:
        strict = "dump/events" in root
    association = None
    if strict:
        from crvfastsim.genassoc import validate_donor
        association = validate_donor(str(new))
    a, ai = P.donor_showers(old)
    b, bi = P.donor_showers(new)
    key = tables.KEY + ["prim_key"]
    changed = a[key + ["pdg", "genidx", "gen_pdg", "ke", "kebin", "gen_t", "pdg_disagreement"]].merge(
        b[key + ["gen_pdg", "ke", "kebin", "gen_t", "pdg_disagreement"]], on=key,
        suffixes=("_old", "_new"), validate="one_to_one", how="outer", indicator=True)
    if not changed._merge.eq("both").all():
        raise ValueError("shower identity set changed")
    altered = ((changed.kebin_old != changed.kebin_new) | (changed.gen_t_old != changed.gen_t_new) |
               (changed.gen_pdg_old != changed.gen_pdg_new) | (changed.ke_old != changed.ke_new))
    result = {"old": str(old), "new": str(new), "input_sha256": hashes, "invariants": invariant,
              "old_summary": summarize(a), "new_summary": summarize(b),
              "old_tree_inventory": ai, "new_tree_inventory": bi,
              "changed_showers": int(altered.sum()),
              "changed_ke_bins": int((changed.kebin_old != changed.kebin_new).sum()),
              "changed_gen_times": int((changed.gen_t_old != changed.gen_t_new).sum()),
              "changes": P.records(changed.loc[altered].drop(columns="_merge")),
              "association_validation": association,
              "limitation": ("Schema-v2 pointer identity checked; production physics closure remains separate." if strict else
                             "Legacy-schema comparison does not prove ProductID equality; strict re-dump still required.")}
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("input changed during comparison: " + path)
    _new_json(out, result)
    print(json.dumps({k: result[k] for k in ("invariants", "old_summary", "new_summary", "changed_showers",
                                           "changed_ke_bins", "changed_gen_times")}, indent=2))


def plan(metadata, outdir, area, generated_per_file):
    """Resolve exact pnfs input paths from original successful dump logs, never scan dCache."""
    metadata = Path(metadata).resolve(strict=True)
    area = Path(area).resolve()
    meta = json.loads(metadata.read_text())
    donors = meta["inputs"]
    if any(not Path(p).is_absolute() for p in donors + meta["gen"]):
        raise ValueError("source metadata must use absolute donor and generator paths")
    if generated_per_file * len(donors) != meta["generated_scaled_to"]:
        raise ValueError("per-file generated exposure does not reproduce library exposure")
    out = Path(outdir).resolve()
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for donor in donors:
        p = Path(donor)
        seq = p.name.removeprefix("crvsteps.").removesuffix(".root")
        log = p.parent / "logs" / f"dump.{seq}.log"
        text = log.read_text()
        inputs = set(re.findall(r'Opened input file "([^"]+)"', text))
        if len(inputs) != 1 or "Art has completed and will exit with status 0" not in text:
            raise ValueError("cannot certify original input/completion from " + str(log))
        source = inputs.pop()
        if not source.startswith("/pnfs/") or Path(source).name.split(".")[-2] != seq:
            raise ValueError("unexpected original art path: " + source)
        rows.append({"sequence": seq, "art": source, "old_table": str(p), "old_sha256": sha256(p),
                     "old_log": str(log), "old_log_sha256": sha256(log),
                     "new_table": str(out / "donors" / p.name)})
    result = {"source_metadata": str(metadata), "source_metadata_sha256": sha256(metadata), "area": str(area),
              "generated_per_file": generated_per_file, "generated": meta["generated_scaled_to"],
              "gen": meta["gen"], "gen_sha256": {p: sha256(p) for p in meta["gen"]}, "donors": rows,
              "policy": "Full-generator denominator unchanged; no output overwrite; strict schema-v2 tables required."}
    out.mkdir()
    for name in ("donors", "logs", "comparisons", "library", "validation"):
        (out / name).mkdir()
    _new_json(out / "plan.json", result)
    with (out / "donors.tsv").open("x") as stream:
        for r in rows:
            stream.write("\t".join(r[k] for k in ("sequence", "art", "old_table", "new_table")) + "\n")
    print(f"planned {len(rows)} donors, {result['generated']} generated exposure, output {out}")


def _load_plan(path):
    path = Path(path).resolve(strict=True)
    result = json.loads(path.read_text())
    if sha256(result["source_metadata"]) != result["source_metadata_sha256"]:
        raise ValueError("source metadata drift")
    source = json.loads(Path(result["source_metadata"]).read_text())
    if (result["generated"] != source["generated_scaled_to"] or result["gen"] != source["gen"]
            or [r["old_table"] for r in result["donors"]] != source["inputs"]
            or result["generated_per_file"] * len(result["donors"]) != result["generated"]):
        raise ValueError("plan no longer reproduces source donor/exposure metadata")
    for gen, digest in result["gen_sha256"].items():
        if sha256(gen) != digest:
            raise ValueError("generator denominator drift: " + gen)
    return path.parent, result


def redump(path, first=1, last=None):
    """Sequential art re-dump; completed outputs are validated before reuse, incomplete outputs fail closed."""
    from crvfastsim.genassoc import validate_donor
    path = Path(path).resolve(strict=True)
    out, meta = _load_plan(path)
    area = Path(meta["area"])
    if Path(os.environ.get("MUSE_WORK_DIR", "")) != area:
        raise ValueError("MUSE_WORK_DIR must be " + str(area))
    libs = list((area / "build").glob("*/CrvFastSim/lib/libcrvfastsim_CrvStepTableDumper_module.so"))
    if len(libs) != 1:
        raise ValueError("expected exactly one built dumper in isolated area")
    source = area / "CrvFastSim/src/CrvStepTableDumper_module.cc"
    if libs[0].stat().st_mtime < source.stat().st_mtime:
        raise ValueError("dumper binary predates source; user build required")
    fcl = area / "CrvFastSim/fcl/dump_crvsteps_genassoc.fcl"
    includes = area / "CrvFastSim/fcl/dump_crvsteps.fcl"
    bound = {str(p): sha256(p) for p in (Path(path), source, libs[0], fcl, includes)}
    last = len(meta["donors"]) if last is None else last
    if not 1 <= first <= last <= len(meta["donors"]):
        raise ValueError("invalid donor interval")
    for row in meta["donors"][first - 1:last]:
        seq, target = row["sequence"], Path(row["new_table"])
        done = out / "logs" / f"dump.{seq}.done.json"
        if sha256(row["old_table"]) != row["old_sha256"]:
            raise ValueError("old donor drift: " + seq)
        if done.exists():
            saved = json.loads(done.read_text())
            if saved["bound"] != bound or sha256(target) != saved["output_sha256"]:
                raise ValueError("completed output/source drift: " + seq)
            validate_donor(str(target))
            print(f"{seq}: verified completed re-dump", flush=True)
            continue
        if target.exists():
            raise FileExistsError("incomplete output, inspect before retry: " + str(target))
        log = out / "logs" / f"dump.{seq}.log"
        if log.exists():
            raise FileExistsError("previous attempt, inspect before retry: " + str(log))
        cmd = ["mu2e", "-c", str(fcl), "-s", row["art"], "-T", str(target)]
        start = time.monotonic()
        with log.open("x") as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"{seq}: art exit {result.returncode}; see {log}")
        elapsed = time.monotonic() - start
        validation = validate_donor(str(target))
        comparison = out / "comparisons" / f"{seq}.json"
        compare(row["old_table"], str(target), comparison)
        for file, digest in bound.items():
            if sha256(file) != digest:
                raise ValueError("source changed during run: " + file)
        _new_json(done, {"command": cmd, "exit": 0, "wall_seconds": elapsed, "bound": bound,
                         "output_sha256": sha256(target), "validation": validation})
        print(f"{seq}: completed and validated in {elapsed:.3f} s", flush=True)


def rebuild(path):
    """Rebuild from corrected donors only after all re-dumps completed; keep the original exposure denominator."""
    from crvfastsim import tier1b
    path = Path(path).resolve(strict=True)
    out, meta = _load_plan(path)
    donors = []
    donor_hashes = {}
    plan_hash = sha256(path)
    dumper_binding = None
    for row in meta["donors"]:
        done = out / "logs" / f"dump.{row['sequence']}.done.json"
        record = json.loads(done.read_text())
        if record["bound"].get(str(Path(path))) != plan_hash:
            raise ValueError("completed donor is bound to another plan")
        if dumper_binding is None:
            dumper_binding = record["bound"]
        elif record["bound"] != dumper_binding:
            raise ValueError("donors were re-dumped with different source/binary/config bindings")
        if record["exit"] != 0 or sha256(row["new_table"]) != record["output_sha256"]:
            raise ValueError("unverified corrected donor: " + row["sequence"])
        donors.append(row["new_table"])
        donor_hashes[row["new_table"]] = record["output_sha256"]
    output = out / "library/library.root"
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    from crvfastsim import genassoc
    from crvfastsim.analyses.census import primary_of
    import inspect
    code_paths = (Path(__file__), Path(tier1b.__file__), Path(genassoc.__file__), Path(inspect.getfile(primary_of)))
    code = {str(p): sha256(p) for p in code_paths}
    from crvfastsim import geometry
    counters = Path(geometry.find_counters("Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt")).resolve()
    geometry_hash = sha256(counters)
    lib, info = tier1b.build(donors, meta["gen"], geometry.from_offline_config(str(counters)), generated=meta["generated"])
    support = lib["libshower"].groupby(["apdg", "kebin"]).size().rename("actual").reset_index().merge(
        lib["libgen"], on=["apdg", "kebin"], how="left", validate="one_to_one")
    if (support.ngen.isna().any() or (support.ngen <= 0).any() or (support.actual != support.nshower).any()
            or (support.nshower > support.ngen).any()):
        raise ValueError("corrected library still has missing/invalid shower support; no publication")
    for file, digest in {**code, **donor_hashes, **meta["gen_sha256"],
                         str(path): plan_hash, str(counters): geometry_hash}.items():
        if sha256(file) != digest:
            raise ValueError("input/source drift during rebuild: " + file)
    tier1b.write_library(str(output), lib)
    _new_json(output.with_suffix(".json"), {"inputs": donors, "gen": meta["gen"], **info,
                                          "plan": str(path), "plan_sha256": plan_hash, "code_sha256": code,
                                          "geometry": str(counters), "geometry_sha256": geometry_hash,
                                          "dumper_binding": dumper_binding, "output_sha256": sha256(output)})
    print(f"built {output}", flush=True)


def verify_library(root):
    """Require a completed library sidecar and frozen generator/exposure plan before downstream replay."""
    root = Path(root).resolve(strict=True)
    meta = json.loads((root / "library/library.json").read_text())
    if sha256(root / "library/library.root") != meta["output_sha256"]:
        raise ValueError("library hash differs from completed rebuild")
    if sha256(meta["plan"]) != meta["plan_sha256"]:
        raise ValueError("library plan drift")
    _load_plan(meta["plan"])
    if sha256(meta["geometry"]) != meta["geometry_sha256"]:
        raise ValueError("library geometry drift")
    print("verified completed library and bound exposure", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("verify-library")
    a.add_argument("root")
    a = sub.add_parser("redump")
    a.add_argument("plan")
    a.add_argument("--first", type=int, default=1)
    a.add_argument("--last", type=int)
    a = sub.add_parser("rebuild")
    a.add_argument("plan")
    a = sub.add_parser("compare")
    a.add_argument("old")
    a.add_argument("new")
    a.add_argument("--out", required=True)
    a = sub.add_parser("plan")
    a.add_argument("metadata")
    a.add_argument("--outdir", required=True)
    a.add_argument("--area", required=True)
    a.add_argument("--generated-per-file", type=int, required=True)
    args = p.parse_args()
    if args.command == "verify-library":
        verify_library(args.root)
    elif args.command == "redump":
        redump(args.plan, args.first, args.last)
    elif args.command == "rebuild":
        rebuild(args.plan)
    elif args.command == "compare":
        compare(args.old, args.new, args.out)
    else:
        plan(args.metadata, args.outdir, args.area, args.generated_per_file)


if __name__ == "__main__":
    main()
