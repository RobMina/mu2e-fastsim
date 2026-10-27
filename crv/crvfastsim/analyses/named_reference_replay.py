"""Reverify a completed named campaign and replay its saved V1/paired statistics without transport."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from . import named_reference_campaign as N
from . import reference_certification as C
from . import transport_paths as P
from . import transport_validation as V
from . import v1compare as B


def validate_saved(campaign, certificates):
    """Bind fresh certificates to the historical inputs, per-shard evidence and final results."""
    campaign = C._local_path(campaign)
    shards, evidence = N.validate_certificates(certificates)
    manifest = N.read_json(campaign / "manifest.json")
    summary = N.read_json(campaign / "summary.json")
    frozen = manifest["snapshot"]
    N.require(summary.get("execution_complete") is True, "campaign execution incomplete")
    N.require(summary["requested_configs"] == manifest["configs"] == list(N.CONFIGS), "require both configurations")
    N.require(manifest["subruns"] == list(N.SUBRUNS) and manifest["generated_events_per_shard"] == N.EVENTS
              and manifest["generated_events_per_config"] == N.TOTAL, "campaign coverage mismatch")
    N.assert_frozen(frozen, N.read_json(campaign / "source_final.json"))
    for module, key in ((V, "diagnostic_sha256"), (B, "v1_sha256"), (C, "certifier_sha256")):
        N.require(C.fingerprint(module.__file__)["sha256"] == frozen[key], f"replay implementation changed: {key}")
    N.require(C.fingerprint(P.__file__)["sha256"] == frozen["path_sources"][P.__name__], "path implementation changed")
    old_shards = manifest["shards"]
    N.require(len(old_shards) == len(shards), "historical certificate coverage mismatch")
    seen = set()
    for old in old_shards:
        key = (old["config"], old["subrun"])
        N.require(key in shards and key not in seen, "historical certificate duplicate or extra shard")
        seen.add(key)
        fresh = shards[key]
        N.require(all(old[k] == fresh[k] for k in ("run", "counts", "artifacts")),
                  "fresh certificate disagrees with historical persisted evidence")
        evidence.append(old["certificate"])
    evidence.extend(manifest["inputs"])
    files = {}
    for cfg in N.CONFIGS:
        result = summary["configs"][cfg]
        accepted = sum(shards[cfg, s]["counts"]["accepted_events"] for s in N.SUBRUNS)
        N.require(result["execution_complete"] is True and result["shards"] == 20
                  and result["generated_events"] == N.TOTAL and result["accepted_events"] == accepted,
                  "saved result count mismatch")
        N.require(N.read_json(campaign / cfg / "summary.json") == result, "per-config summary mismatch")
        files[cfg] = {"audits": [], "emits": []}
        for subrun in N.SUBRUNS:
            shard = shards[cfg, subrun]
            saved = N.read_json(campaign / cfg / str(subrun) / "shard.json")
            N.require(saved["config"] == cfg and saved["subrun"] == subrun and saved["seed"] == subrun - 100
                      and saved["transport_reused"] == (subrun == 101)
                      and saved["generated_events"] == N.EVENTS
                      and saved["accepted_events"] == shard["counts"]["accepted_events"], "shard summary mismatch")
            N.require(set(saved["files"]) == {"candidate", "audit", "paths", "v1_emit"}, "incomplete shard evidence")
            evidence.extend(saved["files"].values())
            evidence.append(saved["certificate"])
            audit_path, path_path = (saved["files"][k]["path"] for k in ("audit", "paths"))
            audit, paths = N.read_json(audit_path), N.read_json(path_path)
            N.validate_provenance(audit, paths, shard, frozen)
            N.require(N.geometry_signature([audit["inputs"]["candidate"]]) ==
                      N.geometry_signature([saved["files"]["candidate"]]), "shard candidate fingerprint mismatch")
            evidence.extend(audit["artifacts"])
            if subrun == 101:
                for name, path in (("candidate", saved["files"]["candidate"]["path"]),
                                   ("audit", audit_path), ("paths", path_path)):
                    N.require(path == manifest["retained"][cfg][name], "retained shard was replaced")
            else:
                for name in ("source_before.json", "source_after.json"):
                    N.assert_frozen(frozen, N.read_json(campaign / cfg / str(subrun) / name))
            files[cfg]["audits"].append(audit_path)
            files[cfg]["emits"].append(saved["files"]["v1_emit"]["path"])
            evidence.append(C.fingerprint(campaign / cfg / str(subrun) / "shard.json"))
        N.validate_shard_set([N.read_json(p) for p in files[cfg]["audits"]], cfg)
        for name in ("audit", "paths", "v1_stats"):
            evidence.append(result[name])
        audit = N.read_json(result["audit"]["path"])
        paths = N.read_json(result["paths"]["path"])
        N.require(audit["accepted_count_gate"] is True and audit["expected_accepted"] == accepted
                  and audit["persisted_reference_accepted_events"] == accepted
                  and audit["persisted_generated_events"] == paths["persisted_generated_events"] == N.TOTAL,
                  "merged count evidence mismatch")
        N.require(audit["sectors"] == result["all_sector_paired_diagnostics"], "paired summary mismatch")
        evidence.extend(audit["source_audits"])
        evidence.extend(paths["source_audits"])
        N.require(N.read_json(result["v1_stats"]["path"]) == result["v1"], "V1 summary mismatch")
        N.require(type(result["v1_pass"]) is bool and result["v1_pass"] == result["v1"]["pass"]
                  and result["v1_status"] == ("PASS" if result["v1_pass"] else "FAIL"), "V1 verdict mismatch")
        for geometry in frozen["configs"][cfg]["geometry"]:
            evidence.append(geometry)
    N.require(summary["v1_pass"] == all(summary["configs"][cfg]["v1_pass"] for cfg in N.CONFIGS),
              "overall V1 verdict mismatch")
    for name in ("manifest.json", "summary.json", "source_final.json"):
        evidence.append(C.fingerprint(campaign / name))
    N.verify_evidence(evidence)
    return summary, frozen, shards, files, evidence


def percent(value):
    return "unavailable" if value is None else f"{100 * value:+.3f}"


def paired(value, field, error):
    return f"{percent(value[field])} ± {100 * value[error]:.3f}" if value[error] is not None else percent(value[field])


def markdown(summary):
    lines = ["# Frozen named-reference certification and V1 replay", "",
             "Shifts and event-cluster paired SE are percentages; no model/systematic uncertainty is included.",
             "V1 groups retain their historical event/bar aggregation; paired diagnostics retain genidx as well.", ""]
    for cfg, result in summary["configs"].items():
        lines += [f"## {cfg}: {result['v1_status']}", "",
                  f"Generated: {result['generated_events']:,}; accepted: {result['accepted_events']:,}; shards: 20.", "",
                  "| Group | V1 G4 crossings | Paired crossing shift ± SE | Length mean ± SE | Energy mean ± SE | Length/energy bins | Failed V1 gates |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for group, value in result["v1"]["groups"].items():
            diag = result["all_sector_paired_diagnostics"]["group:" + group]["candidate_vs_reference"]
            failed = ", ".join(k for k, ok in value["checks"].items() if not ok) or "none (PASS)"
            lines.append(f"| {group} | {value['n_ref']:,} | {paired(diag, 'relative', 'se')} | "
                         f"{paired(diag, 'length_mean_relative', 'length_mean_se')} | "
                         f"{paired(diag, 'edep_mean_relative', 'edep_mean_se')} | "
                         f"{value['len_ratio']['nbins_used']}/{value['edep_ratio']['nbins_used']} | {failed} |")
        lines += ["", "### Individual sectors (diagnostics, not independent V1 gates)", "",
                  "| Sector | G4 crossings | Crossing shift ± SE | Length mean ± SE | Energy mean ± SE |",
                  "|---|---:|---:|---:|---:|"]
        for sector, values in result["all_sector_paired_diagnostics"].items():
            if sector.startswith("group:"):
                continue
            value = values["candidate_vs_reference"]
            lines.append(f"| {sector} | {value['b']:,} | {paired(value, 'relative', 'se')} | "
                         f"{paired(value, 'length_mean_relative', 'length_mean_se')} | "
                         f"{paired(value, 'edep_mean_relative', 'edep_mean_se')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def replay(campaign, certificates, outdir):
    outdir = C._local_path(outdir)
    N.require(not outdir.exists(), "refusing existing output directory")
    summary, frozen, shards, files, evidence = validate_saved(campaign, certificates)
    outdir.mkdir()
    results = {}
    for cfg in N.CONFIGS:
        base = outdir / cfg
        base.mkdir()
        merged = V.merge(files[cfg]["audits"], base / "audit1M.json")
        original = N.read_json(summary["configs"][cfg]["audit"]["path"])
        N.require(merged == original, "paired replay differs from saved result")
        v1 = B.run(files[cfg]["emits"], str(base / "v1"), merge=True, run_config=N.CONFIGS[cfg])
        N.require(v1 == summary["configs"][cfg]["v1"], "V1 replay differs from saved result")
        results[cfg] = {"paired_exact": True, "v1_exact": True,
                        "audit": C.fingerprint(base / "audit1M.json"),
                        "v1_stats": C.fingerprint(base / "v1" / "v1compare_stats.json")}
    N.verify_evidence(evidence)
    inventories = {cfg: {"generated_events": sum(shards[cfg, s]["counts"]["generated_events"] for s in N.SUBRUNS),
                         "accepted_events": sum(shards[cfg, s]["counts"]["accepted_events"] for s in N.SUBRUNS),
                         "all_selected_cycles_one": all(a["selected_cycle"] == 1
                             for s in N.SUBRUNS for name in ("gen", "crvsteps")
                             for a in shards[cfg, s]["artifacts"][name]["trees"].values())} for cfg in N.CONFIGS}
    report = {"schema": "named-reference-replay-v1", "status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Fresh persisted certification and replay of historical transport outputs; not current transport certification.",
              "campaign": C.fingerprint(Path(campaign) / "summary.json"), "frozen_source": frozen,
              "replay_source": {m.__name__: C.fingerprint(m.__file__) for m in (N, V, P, B, C)},
              "replay_driver": C.fingerprint(__file__), "evidence": evidence, "inventories": inventories,
              "results": results, "v1_pass": summary["v1_pass"]}
    N.write_json(outdir / "verification.json", report)
    with (outdir / "tables.md").open("x") as stream:
        stream.write(markdown(summary))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--certificates", type=Path, nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = replay(**vars(args))
    print(f"Verification PASS; frozen physics V1 {'PASS' if report['v1_pass'] else 'FAIL'}; {args.outdir}")


if __name__ == "__main__":
    main()
