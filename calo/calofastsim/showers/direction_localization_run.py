"""Authenticate and publish a saved-only localization of the fixed direction trial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import candidate_support as auth, source_overlap_report

TRIAL_SHA = "4bf2eb82fa744fef6aac2d22932cabe2e3820de8039525a0de2758fc5cf5918e"
PREVIOUS_DIAGNOSTICS_SHA = "0385c0634055ec59cc97f88c9d1a4a10c5843bbbf0e4a99bd15908d8fcc5e403"
SCOPE = (
    "Saved-only exposed-file localization, exact direction candidate fixed. "
    "No sampling, random replay, placement, response, reconstruction, ROOT read or new physics input. "
    "Truth-E/p and stock-outcome-conditioned terms are diagnostics, not measured-variable corrections. "
    "No threshold scan, event veto, closure/promotion, nominal p-value or resolution claim. "
    "Original stored-row block errors omit model, library and stock-seed uncertainty."
)


def authenticate(trial, production):
    """Reuse the completed trial hash chain, without reopening its physics inputs."""
    inputs = source_overlap_report.Inputs()
    receipt = auth.pinned_json(inputs, trial / "summary.json", TRIAL_SHA)
    if receipt.get("completed") is not True or receipt.get("sampler_calls") != 1 or receipt.get("reconstruction_calls") != 2:
        raise ValueError("a completed one-draw/two-call receipt is required")
    names = ("conditional_arrays.npz", "diagnostics.npz", "clusters.npz", "pools.npz")
    saved = {name: auth.pinned_npz(inputs, trial / name, receipt["output_sha256"][name]) for name in names}
    old_path = production / "diagnostics.npz"
    anchors = [v for k, v in receipt["input_sha256"].items()
               if k.endswith("/production_candidate_trial/diagnostics.npz")]
    if anchors != [PREVIOUS_DIAGNOSTICS_SHA]:
        raise ValueError("prior diagnostics are not uniquely anchored by the completed trial")
    old = auth.pinned_npz(inputs, old_path, PREVIOUS_DIAGNOSTICS_SHA)
    return inputs, saved, old


def support_counts(pools, arrays):
    """Tabulate saved capped-pool counts, including every fallback query."""
    rows, count = np.asarray(arrays["event_row"]), np.asarray(pools["direction_count"])
    if not np.array_equal(rows, pools["event_row"]) or count.shape != rows.shape or count.dtype.kind not in "iu" or np.any(count < 0):
        raise ValueError("invalid saved capped-pool rows/counts")
    for name in ("direction_used", "direction_fallback"):
        if not np.array_equal(arrays[name], pools[name]):
            raise ValueError("saved capped-pool flag mismatch")
    used, fallback = arrays["direction_used"], arrays["direction_fallback"]
    if used.dtype.kind != "b" or fallback.dtype.kind != "b" or np.any(used & fallback):
        raise ValueError("invalid direction branch masks")
    if not np.array_equal(used | fallback, pools["supported"]) or not np.array_equal(used, (count >= 8) & pools["supported"]):
        raise ValueError("saved capped-pool eligibility mismatch")
    masks = dict(all=np.ones(len(rows), bool), direction_used=used, direction_fallback=fallback)
    masks.update({f"cell_{i:02d}": arrays["cell_id"] == i for i in range(96)})
    return {name: dict(entries=int(mask.sum()), used=int((used & mask).sum()),
                      fallback=int((fallback & mask).sum()),
                      count_categories=[int(np.sum(mask & (count == 0))), int(np.sum(mask & (count > 0) & (count < 8))),
                                        int(np.sum(mask & (count >= 8)))]) for name, mask in masks.items()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args):
    if args.out.exists():
        raise FileExistsError("new saved-only output directory required")
    from . import direction_localization as analysis
    from . import conditional_stats, compare, candidate_residual

    inputs, saved, old = authenticate(args.trial, args.production)
    for module in (analysis, conditional_stats, compare, candidate_residual):
        inputs.read(Path(module.__file__))
    inputs.read(Path(__file__))
    census = support_counts(saved["pools.npz"], saved["conditional_arrays.npz"])
    clusters = dict(saved["clusters.npz"], sampling_direction_count=saved["pools.npz"]["direction_count"])
    report = analysis.analyze(saved["conditional_arrays.npz"], saved["diagnostics.npz"], old, clusters)
    report["saved_capped_support"] = dict(categories=["zero", "one_to_seven", "at_least_eight"], regions=census)
    inputs.unchanged()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "numbers.json", report)
    output_sha = hashlib.sha256((args.out / "numbers.json").read_bytes()).hexdigest()
    write_json(args.out / "summary.json", dict(scope=SCOPE, completed=True, input_sha256=inputs.sha256,
        output_sha256={"numbers.json": output_sha}, trial_sha256=TRIAL_SHA,
        sampler_calls=0, placement_calls=0, reconstruction_calls=0, root_reads=0))
    inputs.unchanged()
    print("Saved-only direction localization complete; zero physics calls.")
    print("numbers.json SHA256 " + output_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trial", "production", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
