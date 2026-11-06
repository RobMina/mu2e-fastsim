"""Matched ordered-input primary-brems first-foil S1 comparison, not full flash.

Compare saved fast uncollided weights with process-selected G4 original photons.
All event rows must match; unknown G4 exits and fast loopers remain caveats, not
absorbed photons or a changed denominator. No fitted correction is applied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .flash_inclusive_check import EVENT, keys, join
from .flash_flux_compare import paired


def compare(fast, fast_report, references):
    """Return count/energy comparisons for one design and disjoint G4 shards."""
    fk = keys({k: fast["event"][:, i] for i, k in enumerate(EVENT)}, EVENT)
    join(fk, fk)
    ek = keys({k: np.concatenate([r["event_" + k] for r in references]) for k in EVENT}, EVENT)
    join(ek, ek)
    index = join(fk, ek)
    if len(ek) != len(fk) or np.any(index < 0) or fast_report["source_events"] != len(ek):
        raise ValueError("fast and G4 must contain identical complete input event universes")
    n = len(ek)
    if n < 1:
        raise ValueError("empty input event ledger")
    if fast_report["dropped_births"] or fast_report["capped_parents"]:
        raise ValueError("fast source is truncated; do not compare as a completed flux")
    _, group = np.unique(fast["event"][:, :2], axis=0, return_inverse=True)
    result = dict(scope="Matched input primary-brems/uncollided first birth-foil exit; NOT full S1 or flash closure",
                  input_events=n, g4_shards=len(references), fast_loopers=fast_report["loopers"],
                  fast_scope=fast_report["scope"], metrics={})
    for name, fkey, gkey in (("photons", "uncollided_photons", "evt_primary_exits"),
                              ("energy_mev", "uncollided_energy_mev", "evt_primary_energy")):
        f = np.asarray(fast[fkey], float)
        g = np.concatenate([r[gkey] for r in references])[index].astype(float)
        if f.shape != (n,) or g.shape != (n,):
            raise ValueError("event totals must align with ledger")
        item = paired(g, f, n)
        delta = f - g
        diff_se = float(delta.std(ddof=1) * np.sqrt(n)) if n > 1 else None
        fg, gg = np.bincount(group, weights=f), np.bincount(group, weights=g)
        result["metrics"][name] = dict(g4_total=item["low_sum"], fast_total=item["high_sum"],
            fast_over_g4=item["ratio"], paired_ratio_se=item["ratio_se"],
            subrun_ratio_se=paired(gg, fg, len(gg))["ratio_se"],
            g4_mean_per_event=item["low_per_event"], fast_mean_per_event=item["high_per_event"],
            g4_mean_event_se=item["low_se"], fast_mean_event_se=item["high_se"],
            difference_total=float(delta.sum()), difference_total_event_se=diff_se,
            difference_pull=float(delta.sum()/diff_se) if diff_se else None)
    reports = [json.loads(str(r["report_json"])) for r in references]
    result["g4_diagnostics"] = {k: sum(r[k] for r in reports) for k in (
        "primary_birth_count", "primary_uncollided_exit_count", "ambiguous_count", "unresolved_count",
        "unknown_ancestry_count", "unknown_history_count", "primary_no_recorded_exit_count",
        "primary_prior_non_geometry_count", "source_states_declared", "source_missing_events", "source_size_mismatch_events")}
    result["caveats"] = ["Both fast and G4 use the same input events, but random streams differ.",
        "Source primaries only: no annihilation or secondary brems/cascade closure.",
        "Unknown or unrecorded exits remain unresolved, not proven absorbed.",
        "Fast charged layers are planar; finite-annulus photon rays do not repair charged rim/hole escape.",
        "Fast uncollided attenuation uses XCOM; no target-wide or downstream detector prediction.",
        "Event and SubRun SEs are sampling diagnostics, not full model/systematic uncertainty."]
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fast", required=True, type=Path)
    p.add_argument("--g4", required=True, type=Path, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    with np.load(a.fast/"event_totals.npz", allow_pickle=False) as z:
        fast = dict(z)
    references = []
    for path in a.g4:
        with np.load(path, allow_pickle=False) as z:
            references.append(dict(z))
    result = compare(fast, json.loads((a.fast/"flux.json").read_text()), references)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
