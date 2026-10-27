"""Require exact output identity with a completed saved baseline, allowing relocated immutable assets."""
import argparse
import json
from pathlib import Path
import statistics


SETUP_SOURCES = {"crvfastsim/analyses/runchain.py", "crvfastsim/analyses/e2e_bench.py",
                 "crvfastsim/analyses/context_audit.py"}
DEFAULTS = {"deviation": None, "gdml": None, "counters": None, "jax": False, "max_events": None,
            "reco_cpu": False, "msc_tail": False, "tail_k": None, "sliced_msc": False,
            "first_muon_only": False, "no_absorbers": False, "run_config": None, "window": None,
            "fused_response": False, "prefetch": False, "unbanded_world": False, "no_below_world": False}


def named_hashes(mapping, source=False):
    result = {}
    for path, digest in mapping.items():
        if source:
            parts = Path(path).parts
            roots = [i for i, p in enumerate(parts) if p in ("crvfastsim", "mu2efastsim")]
            if not roots:
                raise ValueError(f"unrecognized package source: {path}")
            name = "/".join(parts[roots[-1]:])
        else:
            name = Path(path).name
        if name in result:
            raise ValueError(f"ambiguous asset/source basename: {name}")
        result[name] = digest
    return result


def settings(report):
    args = {**DEFAULTS, **report["arguments"]}
    assets = report["provenance"]["assets_sha256"]
    for key in ("library", "calib", "deviation", "gdml", "counters"):
        value = args.get(key)
        if value and value != "nominal":
            resolved = report["provenance"].get("argument_asset_paths", {}).get(key)
            if resolved is None and Path(value).is_absolute():
                resolved = str(Path(value))
            if resolved not in assets:
                raise ValueError(f"argument {key} not bound to an asset hash; relative legacy paths need recorded bindings")
            args[key] = assets[resolved]
    args["inputs"] = [report["provenance"]["input_sha256"][p] for p in args["inputs"]]
    args.pop("repeat", None)
    return args


def effective_config(report, field):
    config = dict(report["provenance"][field])
    if field == "effective_run_config" and report["arguments"].get("counters"):
        config["counters"] = settings(report)["counters"]
    return config


def removed_appledouble(reference, candidate):
    removed = {}
    for name, digest in reference.items():
        path = Path(name)
        if not path.name.startswith("._") or name in candidate:
            continue
        counterpart = str(path.with_name(path.name[2:]))
        if counterpart not in reference or counterpart not in candidate:
            raise ValueError(f"metadata exception requires retained paired source: {name}")
        removed[name] = digest
    return removed


def compare(reference, candidate, allow_removed_appledouble=False):
    a, b = (json.loads(Path(p).read_text()) for p in (reference, candidate))
    for report in (a, b):
        if report.get("status") != "complete" or not report.get("repeat_fingerprints_identical"):
            raise ValueError("completed repeat-identical reports required")
    if b.get("context_policy") != "prepared_once" or not b.get("context_audits_passed"):
        raise ValueError("candidate must have audited prepared context")
    for key in ("boundary", "device"):
        if a[key] != b[key]:
            raise ValueError(f"incompatible {key}")
    if settings(a) != settings(b):
        raise ValueError("physics arguments/input ordering differ")
    if named_hashes(a["provenance"]["assets_sha256"]) != named_hashes(b["provenance"]["assets_sha256"]):
        raise ValueError("asset content differs")
    for field in ("effective_run_config", "effective_chain_config"):
        if effective_config(a, field) != effective_config(b, field):
            raise ValueError(f"incompatible {field}")
    for report in (a, b):
        report["hardware_software"].setdefault("environment", {})
    env_a, env_b = (x["hardware_software"]["environment"] for x in (a, b))
    if env_a.get("CRVFASTSIM_COMPACT_DISPATCH", "0") != env_b.get("CRVFASTSIM_COMPACT_DISPATCH", "0"):
        raise ValueError("compact dispatch differs")
    if a["hardware_software"]["versions"] != b["hardware_software"]["versions"]:
        raise ValueError("software versions differ")
    sa, sb = (named_hashes(r["provenance"]["source_sha256"], source=True) for r in (a, b))
    metadata = removed_appledouble(sa, sb) if allow_removed_appledouble else {}
    changed = sorted(k for k in (sa.keys() | sb.keys()) - metadata.keys() if sa.get(k) != sb.get(k))
    if set(changed) - SETUP_SOURCES:
        raise ValueError(f"non-setup source changed: {sorted(set(changed) - SETUP_SOURCES)}")
    reference_fps = [r["fingerprint"] for r in a["passes"][0]["files"]]
    for report in (a, b):
        for p in report["passes"]:
            if [r["fingerprint"] for r in p["files"]] != reference_fps:
                raise ValueError("prepared output differs from saved baseline: no tolerance allowed")
    aw = [p["elapsed_s"] for p in a["passes"][1:]]
    bw = [p["elapsed_s"] for p in b["passes"][1:]]
    return {"exact_output_identity": True, "source_changes": changed,
            "source_gate": "explicit removed AppleDouble filename exception" if metadata else "strict",
            "removed_appledouble_sha256": metadata,
            "reference": str(reference), "candidate": str(candidate),
            "reference_context_policy": a.get("context_policy", "reload_each_pass"),
            "candidate_context_policy": b["context_policy"],
            "reference_warm_elapsed_s": aw, "candidate_warm_elapsed_s": bw,
            "median_warm_speedup": statistics.median(aw) / statistics.median(bw),
            "preparation_elapsed_s": b["preparation_elapsed_s"],
            "candidate_invocation_elapsed_before_report_s": b["invocation_elapsed_before_report_s"],
            "comparison_limit": "saved earlier baseline, not interleaved timing; pre/post audit included in candidate passes",
            "stock_physics_validation": "not evaluated"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-removed-appledouble", action="store_true",
                   help="explicit packaging exception for removed ._* files with retained paired sources; recorded by hash")
    a = p.parse_args(argv)
    result = compare(a.reference, a.candidate, allow_removed_appledouble=a.allow_removed_appledouble)
    with open(a.out, "x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
