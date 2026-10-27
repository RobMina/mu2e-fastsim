"""Summarize immutable generator-repair evidence and held-out physics comparisons."""
import argparse
import json
from pathlib import Path
import re

import pandas as pd

from crvfastsim import tier1b
from valhist.genassoc_campaign import _load_plan, _new_json
from valhist.library_variants import sha256
from valhist.steplayers import clusters


SUM_METRICS = ("showers", "pdg_mismatch", "missing_join", "negative_ke", "nonfinite", "same_pdg_energy_delta_gt_1MeV")
MAX_METRICS = ("max_abs_start_gen_energy_MeV", "max_abs_start_gen_time_ns")


def aggregate(summaries):
    """Combine per-file counts and extrema without averaging away individual defects."""
    return {**{k: sum(s[k] for s in summaries) for k in SUM_METRICS},
            **{k: max((s[k] for s in summaries if s[k] is not None), default=None) for k in MAX_METRICS}}


def donors(plan, out):
    root, meta = _load_plan(plan)
    binding = None
    comparisons, records = [], []
    evidence = {str(Path(plan).resolve()): sha256(plan)}
    for row in meta["donors"]:
        record_path = root / "logs" / f"dump.{row['sequence']}.done.json"
        comparison_path = root / "comparisons" / f"{row['sequence']}.json"
        record = json.loads(record_path.read_text())
        comparison = json.loads(comparison_path.read_text())
        binding = record["bound"] if binding is None else binding
        if record["exit"] != 0 or record["bound"] != binding or binding[str(Path(plan).resolve())] != sha256(plan):
            raise ValueError("incomplete or inconsistent donor bindings")
        if comparison["input_sha256"][row["new_table"]] != record["output_sha256"]:
            raise ValueError("comparison is not for completed donor")
        if comparison["input_sha256"][row["old_table"]] != row["old_sha256"]:
            raise ValueError("comparison is not for planned baseline")
        if not all(v["legacy_columns_identical"] for v in comparison["invariants"].values()):
            raise ValueError("legacy donor physics payload changed")
        if not record["validation"]["associations_checked"] or record["validation"]["ancestry_cycles"]:
            raise ValueError("donor association validation failed")
        evidence.update({str(record_path): sha256(record_path), str(comparison_path): sha256(comparison_path),
                         row["new_table"]: record["output_sha256"], row["old_table"]: row["old_sha256"]})
        comparisons.append(comparison)
        records.append(record)
    plugin = next(p for p in binding if p.endswith("_module.so"))
    traces = []
    for path in sorted((root / "logs").glob("loader_*.*")):
        paths = re.findall(r"calling init:\s*(\S*libcrvfastsim_CrvStepTableDumper_module\.so)", path.read_text())
        if not paths:
            continue
        if paths != [plugin]:
            raise ValueError("wrong or ambiguous loaded dumper: " + str(path))
        traces.append(str(path))
        evidence[str(path)] = sha256(path)
    if len(traces) != len(records):
        raise ValueError("expected exactly one successful dumper loader trace per donor")
    result = {"plan": str(Path(plan).resolve()), "files": len(records), "generated_exposure": meta["generated"],
              "accepted_counts": {k: sum(r["validation"]["counts"][k] for r in records)
                                  for k in ("events", "gen", "particles", "steps")},
              "old": aggregate([c["old_summary"] for c in comparisons]),
              "corrected": aggregate([c["new_summary"] for c in comparisons]),
              "changed": {k: sum(c[k] for c in comparisons) for k in ("changed_showers", "changed_ke_bins", "changed_gen_times")},
              "legacy_payload_identical": True, "all_associations_checked": True,
              "loaded_plugin": plugin, "loader_traces": traces,
              "art_wall_seconds": [r["wall_seconds"] for r in records], "bound": binding,
              "evidence_sha256": evidence, "limitation": "Extraction checks do not establish physics closure."}
    for path, digest in {**evidence, **binding}.items():
        if sha256(path) != digest:
            raise ValueError("evidence drift: " + path)
    _new_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("evidence_sha256", "bound", "loader_traces")}, indent=2))
    if result["old"]["showers"] != result["corrected"]["showers"]:
        raise ValueError("retained shower count changed")
    if any(result["corrected"][k] for k in SUM_METRICS if k != "showers"):
        raise ValueError("corrected donors retain association/kinematic defects; inspect donor summary")


def library_invariants(plan, out):
    root, meta = _load_plan(plan)
    before = Path(meta["source_metadata"]).with_suffix(".root")
    after = root / "library/library.root"
    hashes = {str(p): sha256(p) for p in (before, after)}
    a, b = tier1b.read_library(str(before)), tier1b.read_library(str(after))
    invariant = {}
    for tree in ("libbins", "libxing", "libdesc"):
        pd.testing.assert_frame_equal(a[tree], b[tree], check_exact=True, check_dtype=False)
        invariant[tree] = {"rows": len(a[tree]), "identical": True}
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("library drift: " + path)
    result = {"invariants": invariant, "input_sha256": hashes}
    _new_json(out, result)
    print(json.dumps(result, indent=2))


def closure(root, baseline, out):
    root, baseline = Path(root), Path(baseline)
    validation = root / "validation/s11-20"
    position = json.loads((validation / "position_comparison.json").read_text())
    a = json.loads((baseline / "v3_ccp_s11_20/tier1b_check.json").read_text())
    b = json.loads((validation / "v3/tier1b_check.json").read_text())
    metrics = ("nonmu_edep_fast_over_ref", "nonmu_bar_hits_fast_over_ref",
               "desc_outside_muon_bars_per_crossing_fast", "desc_outside_muon_bars_per_crossing_ref")
    result = {"step_positions": position["cmp"],
              "v3": {g: {k: {"old": a[g][k], "corrected": b[g][k]} for k in metrics}
                     for g in ("EX", "T1", "T2", "M")},
              "clusters": clusters(validation / "response.json"),
              "muon_family": json.loads((validation / "muon_family_guard.json").read_text()),
              "exposure": json.loads((validation / "exposure.json").read_text()),
              "limitation": "Old worktree NumPy response; diagonal scores omit cross-bin and finite-library covariance."}
    _new_json(out, result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("donors", "library-invariants"):
        s = sub.add_parser(name)
        s.add_argument("plan")
        s.add_argument("--out", required=True)
    s = sub.add_parser("closure")
    s.add_argument("root")
    s.add_argument("--baseline", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "donors":
        donors(args.plan, args.out)
    elif args.command == "library-invariants":
        library_invariants(args.plan, args.out)
    else:
        closure(args.root, args.baseline, args.out)


if __name__ == "__main__":
    main()
