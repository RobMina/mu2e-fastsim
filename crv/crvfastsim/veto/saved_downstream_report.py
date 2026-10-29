"""Compact saved downstream results and exact extreme-photon lineage; never rerun response."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import saved_photon_downstream as D
from .replay import sha256


def extreme_lineage(q, attempts):
    """Select optical primary attempts by immutable photon root, not observed time or distance."""
    roots = q.loc[q.endpoint_roundoff_step, ["event", "chan", "photon_id"]].rename(columns={"photon_id": "root"})
    if roots.duplicated(["event", "chan", "root"]).any():
        raise ValueError("duplicate extreme photon identity")
    primary = attempts[attempts.primary & ~attempts.dark]
    selected = primary.merge(roots, on=["event", "chan", "root"], how="inner", validate="one_to_one")
    return selected


def extreme_channels(q, clusters, digis, pulses):
    """Observed cluster prechecks on extreme-photon channels, not hypothetical analog maxima."""
    keys = ["event", "chan"]
    selected = q.loc[q.endpoint_roundoff_step, keys].drop_duplicates()
    c = clusters.merge(selected, on=keys, how="inner", validate="many_to_one")
    d = digis.merge(selected, on=keys, how="inner", validate="many_to_one")
    p = pulses.merge(selected, on=keys, how="inner", validate="many_to_one")
    return {"clusters": c.to_dict(orient="records"), "digis": len(d[keys + ["starttdc"]].drop_duplicates()),
            "pulses": len(p), "scope": "All saved charge clusters on these channels, not a truth association to fitted pulses."}


def run(directory, out):
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise ValueError("choose a new saved-report directory")
    out.mkdir(parents=True)
    D.verify(directory, out / "verified")
    summary_path = directory / "summary.json"
    digest = sha256(summary_path)
    saved = json.loads(summary_path.read_text())
    optical_path = D.OP.bound(saved["input_summary"], saved["input_summary_sha256"])
    optical = json.loads(optical_path.read_text())
    results = []
    for i, block in enumerate(saved["blocks"]):
        ob = optical["blocks"][i]
        if ob["input_sha256"] != block["input_sha256"]:
            raise ValueError("optical/downstream input identity differs")
        artifact = ob["artifacts"]["photons"]
        q = pd.read_parquet(D.OP.bound(optical_path.parent / artifact["path"], artifact["sha256"]))
        models = {}
        for prefix, model in block["models"].items():
            lineage, analog, channel_stages = {}, {}, {}
            for arm in D.OP.ARMS:
                name = f"{prefix}_{arm}_attempts.parquet"
                attempt = pd.read_parquet(D.OP.bound(directory / name, saved["output_sha256"][name]))
                lineage[arm] = extreme_lineage(q, attempt)
                lineage[arm].to_parquet(out / f"{prefix}_{arm}_extreme_primaries.parquet", index=False)
                name = f"{prefix}_{arm}_samples.parquet"
                samples = pd.read_parquet(D.OP.bound(directory / name, saved["output_sha256"][name]))
                analog[arm] = samples[samples.valid]
                stage = []
                for label in ("clusters", "digis", "pulses"):
                    name = f"{prefix}_{arm}_{label}.parquet"
                    stage.append(pd.read_parquet(D.OP.bound(directory / name, saved["output_sha256"][name])))
                channel_stages[arm] = extreme_channels(q, *stage)
            numeric = {}
            for arm in ("arrival_only", "rejection_only", "stock_timing"):
                stats, _ = D.compare_rows(analog["historical"], analog[arm], ["event", "chan", "tdc"],
                                         ["voltage_noiseless", "voltage", "cluster_prekeep"])
                numeric[f"historical_vs_{arm}"] = stats
            a, b = lineage["rejection_only"], lineage["stock_timing"]
            keys = ["event", "chan", "root"]
            joined = a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True, validate="one_to_one")
            both = joined._merge.eq("both")
            rows = joined.loc[both, keys].copy()
            rows["time_a"], rows["time_b"] = joined.loc[both, "time_a"], joined.loc[both, "time_b"]
            for name in ("uniform", "fired", "charge", "pe"):
                rows[f"{name}_a"], rows[f"{name}_b"] = joined.loc[both, f"{name}_a"], joined.loc[both, f"{name}_b"]
            rows.to_parquet(out / f"{prefix}_extreme_primary_contrast.parquet", index=False)
            models[prefix] = {"arms": model["arms"], "stage_comparisons": model["comparisons"],
                "analog_comparisons": numeric, "extreme_channel_stages": channel_stages, "extreme_primaries": {arm: {"attempts": len(t), "fired": int(t.fired.sum())}
                                                                     for arm, t in lineage.items()},
                "extreme_common_primary_rows": rows.to_dict(orient="records"),
                "extreme_primary_only_rejection": int(joined._merge.eq("left_only").sum()),
                "extreme_primary_only_stock": int(joined._merge.eq("right_only").sum())}
        results.append({"input_sha256": block["input_sha256"], "events": block["events"], "models": models})
    D.OP.bound(summary_path, digest)
    report = {"schema": "crv-saved-downstream-report-v1", "input_summary_sha256": digest, "blocks": results,
              "scope": D.SCOPE, "regenerated_response": False, "code_sha256": sha256(__file__),
              "output_sha256": {p.name: sha256(p) for p in out.glob("*.parquet")}}
    with (out / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    report = run(a.directory, a.out)
    for b in report["blocks"]:
        for name, m in b["models"].items():
            print(json.dumps({"model": name, "arms": m["arms"], "comparisons": m["stage_comparisons"],
                              "analog": m["analog_comparisons"], "extreme_primaries": m["extreme_primaries"],
                              "extreme_common_primary_rows": m["extreme_common_primary_rows"]}, allow_nan=False))


if __name__ == "__main__":
    main()
