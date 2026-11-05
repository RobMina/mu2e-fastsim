"""Compare retained-photon attribution across two independently simulated designs.

A source-region contribution is not a fitted response or a causal attribution to
material. The input bins, retained genealogy and generated-event denominators must
be certified before interpreting any design difference. No automatic verdict or
paired error is supplied without paired-source certification.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def aggregate(report):
    """Sum explicit birth histograms, including energy outside the represented bins."""
    if not report.get("files"):
        raise ValueError("An ancestry report with input files is required")
    first = report["files"][0]["summary"]["photon_birth"]
    z, r = first["z_edges_mm"], first["r_edges_mm"]
    total = np.zeros((len(z) - 1, len(r) - 1))
    accounted, unresolved, outside = 0.0, 0.0, 0.0
    for file in report["files"]:
        summary = file["summary"]
        source = summary["photon_birth"]
        if source["z_edges_mm"] != z or source["r_edges_mm"] != r:
            raise ValueError("Photon birth bins differ within an input report")
        total += np.asarray(source["eion_sum_mev"], float)
        unresolved += source["no_photon"]["eion_sum_mev"] + source["nonfinite_birth"]["eion_sum_mev"]
        outside += source["outside_bins"]["eion_sum_mev"]
        accounted += summary["eion_sum_mev"]
    if not np.isclose(total.sum() + unresolved + outside, accounted, rtol=1e-12, atol=1e-12):
        raise ValueError("Photon-bin energy accounting does not close")
    return dict(z_edges_mm=z, r_edges_mm=r, eion_sum_mev=total, total_mev=accounted,
                unresolved_mev=unresolved, outside_bins_mev=outside)


def compare(low, high):
    """A normalized difference ledger, not a primary/crossing-matched comparison."""
    a, b = aggregate(low), aggregate(high)
    if a["z_edges_mm"] != b["z_edges_mm"] or a["r_edges_mm"] != b["r_edges_mm"]:
        raise ValueError("Photon birth bins differ across designs")
    na = low["normalization"]["certified_generated_events"]
    nb = high["normalization"]["certified_generated_events"]
    if na is None or nb is None or min(na, nb) <= 0:
        raise ValueError("Explicitly certified generated-event denominators are required")
    av, bv = a["total_mev"] / na, b["total_mev"] / nb
    delta = bv - av
    birth_delta = b["eion_sum_mev"] / nb - a["eion_sum_mev"] / na
    return dict(schema="flash-design-difference-v1", low_mev_per_generated=av, high_mev_per_generated=bv,
                high_over_low=bv / av if av > 0 else None, delta_mev_per_generated=delta,
                z_edges_mm=a["z_edges_mm"], r_edges_mm=a["r_edges_mm"],
                birth_delta_mev_per_generated=birth_delta.tolist(),
                fraction_of_design_delta_by_birth_bin=(birth_delta / delta).tolist() if delta != 0 else None,
                unresolved_delta_mev_per_generated=b["unresolved_mev"] / nb - a["unresolved_mev"] / na,
                outside_bins_delta_mev_per_generated=b["outside_bins_mev"] / nb - a["outside_bins_mev"] / na,
                error_method="None: no paired source identity certificate supplied; no statistical significance claimed",
                scope="Nearest retained-photon birth bins; not material labels, full-shower origins or a response fit")


def compare_regions(low, high):
    """Central-value geometric ledger only; coordinate membership is not a material tag."""
    na = low["normalization"]["certified_generated_events"]
    nb = high["normalization"]["certified_generated_events"]
    if na is None or nb is None or min(na, nb) <= 0:
        raise ValueError("Certified generated denominators are required")
    names = low["by_region"]["region"]
    if names != high["by_region"]["region"]:
        raise ValueError("Region labels differ")
    av = np.asarray(low["by_region"]["eion_sum_mev"], float) / na
    bv = np.asarray(high["by_region"]["eion_sum_mev"], float) / nb
    delta = float(bv.sum() - av.sum())
    return dict(schema="flash-region-difference-v1", region=names,
                low_mev_per_generated=av.tolist(), high_mev_per_generated=bv.tolist(),
                delta_mev_per_generated=(bv - av).tolist(),
                fraction_of_design_delta=((bv - av) / delta).tolist() if delta != 0 else None,
                scope="Central-value truth geometry diagnostic; no paired-source/geometry-material certification or statistical significance claimed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("low", type=Path)
    parser.add_argument("high", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regions", action="store_true", help="Compare geometric region ledgers")
    args = parser.parse_args(argv)
    function = compare_regions if args.regions else compare
    result = function(json.loads(args.low.read_text()), json.loads(args.high.read_text()))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
