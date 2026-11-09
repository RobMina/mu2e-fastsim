"""Reproducible tier-1b shower-source diagnostic and primary-step invariance guard."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import tables, tier1b
from valhist.library_variants import _event_inventory, sha256
from valhist.steplayers import shard, shards

SHOWER_TREES = ("libshower", "libshowersteps", "libgen")


def replace_showers(base, source):
    """Keep every descendant tree from base; replace showers and their reach denominator together."""
    return {name: (source if name in SHOWER_TREES else base)[name] for name in tier1b.LIB_TREES}


def event_bank(paths, tree):
    """Inventory all persisted event identities, including highest ROOT cycle, without packed keys."""
    keys, records = [], []
    for path in paths:
        key, record = _event_inventory(path, tree)
        keys.append(key)
        records.append({"path": str(path), **record})
    key = np.concatenate(keys)
    unique = np.unique(key, axis=0)
    if len(key) != len(unique):
        raise ValueError("cross-file event overlap: " + tree)
    return unique, records


def source_library(base, train, gen, evaluation, out):
    """Build a shower-only source variation after auditing training/evaluation event independence."""
    from crvfastsim.analyses.tier1b import _geom
    out = Path(out)
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(out)
    train_keys, train_records = event_bank(train, "dump/particles")
    gen_keys, gen_records = event_bank(gen, "dump/gen")
    eval_keys, eval_records = event_bank(evaluation, "dump/gen")
    overlap = len(gen_keys) + len(eval_keys) - len(np.unique(np.concatenate([gen_keys, eval_keys]), axis=0))
    if overlap:
        raise ValueError("training/evaluation gen overlap")
    if len(np.unique(np.concatenate([gen_keys, train_keys]), axis=0)) != len(gen_keys):
        raise ValueError("training particles outside training gen universe")
    fingerprints = {str(p): sha256(p) for p in [base, *train, *gen, *evaluation]}
    source, info = tier1b.build(train, gen, _geom(None))
    hybrid = replace_showers(tier1b.read_library(base), source)
    for path, digest in fingerprints.items():
        if sha256(path) != digest:
            raise ValueError("input changed during build: " + path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tier1b.write_library(str(out), hybrid)
    report = {"base": base, "training": train_records, "training_gen": gen_records,
              "evaluation_gen": eval_records, "training_generated_events": len(gen_keys),
              "evaluation_generated_events": len(eval_keys), "train_eval_overlap": overlap,
              "source_build": info, "replaced_trees": SHOWER_TREES,
              "input_sha256": fingerprints, "output_sha256": sha256(out),
              "code_sha256": {str(p): sha256(p) for p in (Path(__file__), Path(tier1b.__file__))},
              "limitations": ["Source-change diagnostic, not a calibrated library uncertainty.",
                              "No rate tuning: shower reach denominator comes from training gen only.",
                              "Finite donor statistics are not in event-paired errors."]}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("training_generated_events", "evaluation_generated_events",
                                           "train_eval_overlap", "source_build")}, indent=2), flush=True)


def shower_audit(metadata, out):
    """Audit a campaign library's exact shower projection and generator support, without repairing it."""
    from valhist import denominator_provenance as P
    meta_path = Path(metadata)
    meta = json.loads(meta_path.read_text())
    donors = meta.get("inputs") or [r["path"] for r in meta["training"]]
    gen = meta.get("gen") or [r["path"] for r in meta["training_gen"]]
    exposure = meta.get("generated_scaled_to") or meta["training_generated_events"]
    lib_path = meta_path.with_suffix(".root")
    if Path(out).exists():
        raise FileExistsError(out)
    hashes = {str(p): sha256(p) for p in [meta_path, lib_path, *donors, *gen]}
    all_showers, records, offset = [], [], 0
    for path in donors:
        sh, info = P.donor_showers(path)
        sh["shid"] += offset
        offset += len(sh)
        all_showers.append(sh)
        records.append({"path": path, "showers": len(sh), "pdg_mismatch": int(sh.pdg_disagreement.sum()),
                        "missing_join": int(sh.gen_join_missing.sum()), "negative_ke": int(sh.negative_ke.sum()),
                        "same_pdg_energy_delta_gt_1MeV": int(((~sh.pdg_disagreement) & (sh.start_minus_gen_e.abs() > 1)).sum()),
                        "tree_inventory": {k: info[k] for k in ("particles", "gen", "steps")}})
        print(f"audited {path}: {len(sh)} showers, {int(sh.pdg_disagreement.sum())} PDG mismatches", flush=True)
    showers = pd.concat(all_showers, ignore_index=True)
    lib = tier1b.read_library(str(lib_path))
    cols = ["shid", "apdg", "kebin", "ke"]
    pd.testing.assert_frame_equal(showers[cols], lib["libshower"][cols], check_dtype=False, check_exact=True)
    frames = [tables.read(p, "gen") for p in gen]
    gen_events = len(pd.concat(frames)[tables.KEY].drop_duplicates())
    blocks = [P.energy(g.loc[g.pdg.abs() != 13]) for g in frames]
    cells = P.support(showers, blocks, gen_events, exposure, lib["libgen"])
    missing = cells.loc[cells.missing_raw_support, P.BIN]
    marked = showers.merge(missing.assign(missing=True), on=P.BIN, how="left")
    report = {"metadata": str(meta_path), "library": str(lib_path), "donors": records,
              "showers": len(showers), "pdg_mismatch": int(showers.pdg_disagreement.sum()),
              "missing_join": int(showers.gen_join_missing.sum()), "negative_ke": int(showers.negative_ke.sum()),
              "same_pdg_energy_delta_gt_1MeV": sum(r["same_pdg_energy_delta_gt_1MeV"] for r in records),
              "missing_support_showers": int(marked.missing.eq(True).sum()),
              "missing_support_pdg_mismatch": int((marked.missing.eq(True) & marked.pdg_disagreement).sum()),
              "cells": P.records(cells), "input_sha256": hashes, "projection_identical": True,
              "limitation": "Truth-provenance diagnostic; matching PDG/energy does not certify missing ProductID."}
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("input changed during audit: " + path)
    Path(out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("donors", "cells", "input_sha256")}, indent=2))


def primary_rows(steps, parts, descendants=False):
    """Muon steps selected by ancestry, optionally including descendants, independent of row order."""
    p = parts.loc[(parts.isprim == 1) & (parts.pdg.abs() == 13), tables.KEY + ["key"]]
    if descendants:
        while True:
            children = parts.merge(p.rename(columns={"key": "parent"}), on=tables.KEY + ["parent"])
            expanded = pd.concat([p, children[tables.KEY + ["key"]]]).drop_duplicates()
            if len(expanded) == len(p):
                break
            p = expanded
    s = steps.merge(p.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], validate="many_to_one")
    cols = [c for c in steps.columns if c != "nstep"]
    return s[cols].sort_values(tables.KEY + ["sp", "bar", "t0"], kind="stable").reset_index(drop=True)


def exposure_guard(generated, arms, selected, out):
    """Require complete generated identities in fast tables and containment of reference step identities."""
    records = []
    for k in selected:
        gen_path = generated.format(s=shard(k), k=k)
        population, gen_record = _event_inventory(gen_path, "dump/gen")
        record = {"shard": k, "generated_events": len(population), "generator": gen_record, "arms": {}}
        for name, pattern in arms:
            path = pattern.format(s=shard(k), k=k)
            tree = "dump/particles" if name == "reference" else "dump/gen"
            keys, inventory = _event_inventory(path, tree)
            if name != "reference" and not np.array_equal(population, keys):
                raise ValueError("fast arm generated population differs: " + path)
            if len(np.unique(np.concatenate([population, keys]), axis=0)) != len(population):
                raise ValueError("arm contains events outside generator: " + path)
            record["arms"][name] = {"path": path, **inventory}
        records.append(record)
        print(f"{shard(k)}: {len(population)} generated events; all arm identities consistent", flush=True)
    Path(out).write_text(json.dumps({"shards": records, "pass": True,
                                    "generated_events": sum(r["generated_events"] for r in records)}, indent=2) + "\n")


def primary_guard(base, candidate, selected, out, descendants=False):
    """Require unchanged primary steps in every retained shard; nstep includes added descendants."""
    records = []
    for k in selected:
        paths = [pat.format(s=shard(k), k=k) for pat in (base, candidate)]
        rows = [primary_rows(tables.read(p, "steps"), tables.read(p, "particles"), descendants) for p in paths]
        pd.testing.assert_frame_equal(*rows, check_exact=True, check_dtype=False)
        records.append({"shard": k, "primary_steps": len(rows[0]),
                        "primary_events": len(rows[0][tables.KEY].drop_duplicates()), "identical": True})
        label = "muon-family" if descendants else "primary-muon"
        print(f"{shard(k)}: {len(rows[0])} {label} steps identical", flush=True)
    Path(out).write_text(json.dumps({"base": base, "candidate": candidate, "shards": records,
                                    "ignored_column": "nstep", "includes_descendants": descendants,
                                    "pass": True}, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("shower-source")
    p.add_argument("--base", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--gen", nargs="+", required=True)
    p.add_argument("--evaluation", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("exposure-guard")
    p.add_argument("--generated", required=True)
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--shards", default="11-20")
    p.add_argument("--out", required=True)
    p = sp.add_parser("shower-audit")
    p.add_argument("metadata")
    p.add_argument("--out", required=True)
    p = sp.add_parser("primary-guard")
    p.add_argument("--base", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--shards", default="1-20")
    p.add_argument("--descendants", action="store_true")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "shower-source":
        source_library(a.base, tables.expand(a.train), tables.expand(a.gen), tables.expand(a.evaluation), a.out)
    elif a.cmd == "exposure-guard":
        exposure_guard(a.generated, [tuple(x.split("=", 1)) for x in a.arm], shards(a.shards), a.out)
    elif a.cmd == "shower-audit":
        shower_audit(a.metadata, a.out)
    else:
        primary_guard(a.base, a.candidate, shards(a.shards), a.out, a.descendants)


if __name__ == "__main__":
    main()
