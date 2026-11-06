"""Merge saved ancestry censuses against an independently merged all-event ledger.

Local particle-row and trace indices are never concatenated. Original tracker
identities must be unique, and all three per-event ownership totals must match
the reference. This is a birth-ancestry census, not transport independence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .flash_inclusive_check import EVENT, PRODUCT, keys, join
from .flash_response_origins import summarize, group_energy

LABELS = ("leaf", "nearest_new_photon", "earliest_new_photon", "nearest_photon", "stage_entry", "root")
ANCESTRY = ("ancestry_unresolved", "ancestry_metadata_unknown", "foil_ancestry_complete",
            "has_new_foil_photon_ancestor", "has_new_foil_born_ancestor", "leaf_is_stage_entry") + tuple(
    label + "_" + field for label in LABELS for field in
    ("pdg", "creation", "stage", "birth_volume", "birth_material", "present", "birth_known"))
TRACKER = (*PRODUCT, "index", "total_edep", "nonionizing_edep")


def group_event_energy(energy, event_row, events, **columns):
    """Category totals with event sampling SE, retaining zero events in variance."""
    result = group_energy(energy, **columns)
    labels = np.rec.fromarrays(list(columns.values()), names=list(columns))
    unique, category = np.unique(labels, return_inverse=True)
    pairs, inverse = np.unique(np.rec.fromarrays([category, event_row], names=["category", "event"]),
                               return_inverse=True)
    sums = np.bincount(inverse, weights=energy, minlength=len(pairs))
    total = np.bincount(pairs["category"], weights=sums, minlength=len(unique))
    square = np.bincount(pairs["category"], weights=sums * sums, minlength=len(unique))
    se = np.sqrt(np.maximum(0., (events * square - total * total) / (events - 1))) if events > 1 else None
    result.update(nonzero_events=np.bincount(pairs["category"][sums > 0], minlength=len(unique)).tolist(),
                  total_event_se_mev=se.tolist() if se is not None else [None] * len(unique),
                  effective_events=np.divide(total * total, square, out=np.zeros_like(total, dtype=float), where=square > 0).tolist())
    return result


def merge(paths, reference):
    """Validate full tracker identities and event energy before merging categories."""
    paths = [Path(p) for p in paths]
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("origins inputs must be nonempty and unique")
    fields = (*ANCESTRY, *("tracker_" + k for k in TRACKER),
              "tracker_ownership_status", "tracker_ionizing_mev")
    pieces = {k: [] for k in fields}
    source_inputs, declared_events = [], 0
    identities = {"tracker_" + k for k in (*PRODUCT, "index")}
    for path in paths:
        sidecar = json.loads(path.with_suffix(".json").read_text())
        source_inputs.append(str(Path(sidecar["input"]).resolve()))
        count = sidecar["input_events"]
        if type(count) is not int or count < 1:
            raise ValueError("invalid sidecar event count")
        declared_events += count
        with np.load(path, allow_pickle=False) as data:
            n = len(data["tracker_ionizing_mev"])
            if sidecar["tracker_steps"] != n:
                raise ValueError("tracker census disagrees with original ancestry sidecar")
            for k in fields:
                column = data[k]
                if column.shape != (n,):
                    raise ValueError("origins column shapes must match tracker rows")
                if k in identities:
                    if column.dtype.kind not in "iu" or np.any(column < 0):
                        raise ValueError("invalid tracker identity")
                    column = column.astype(np.uint64)
                pieces[k].append(column)
    if len(set(source_inputs)) != len(source_inputs):
        raise ValueError("duplicate source ROOT inputs")
    data = {k: np.concatenate(v) for k, v in pieces.items()}
    steps = {k: data["tracker_" + k] for k in TRACKER}
    sk = keys(steps, (*PRODUCT, "index"))
    join(sk, sk)
    energy = data["tracker_ionizing_mev"]
    expected = steps["total_edep"].astype(float) - steps["nonionizing_edep"].astype(float)
    if not np.array_equal(energy, expected) or np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("invalid or mismatched tracker energy")
    status = data["tracker_ownership_status"]
    if status.dtype.kind not in "iu" or np.any(~np.isin(status, (0, 1, 2))):
        raise ValueError("invalid tracker ownership status")
    events = {k: np.asarray(reference["event_" + k]) for k in (*EVENT, "ordinal")}
    n = len(events["run"])
    if n < 1 or declared_events != n:
        raise ValueError("source sidecar event counts do not cover reference denominator")
    if any(v.shape != (n,) for v in events.values()):
        raise ValueError("reference identity columns must align with event ledger")
    for name in ("tracker", "owned", "residual", "unknown"):
        value = np.asarray(reference["event_" + name + "_ionizing_mev"])
        if value.shape != (n,) or value.dtype.kind not in "iuf" or np.any(~np.isfinite(value)) or np.any(value < 0):
            raise ValueError("reference energy columns must be finite nonnegative and align with event ledger")
    ek = keys(events, EVENT)
    join(ek, ek)
    ordinal = np.asarray(events["ordinal"])
    if ordinal.shape != (len(ek),) or ordinal.dtype.kind not in "iu" or np.any(ordinal < 0):
        raise ValueError("invalid event ordinals")
    if len(np.unique(ordinal)) != len(ordinal):
        raise ValueError("duplicate event ordinals")
    se = join(keys(steps, EVENT), ek)
    if np.any(se < 0):
        raise ValueError("tracker event absent from reference")
    raw = np.bincount(se, weights=energy, minlength=len(ek))
    if not np.allclose(raw, reference["event_tracker_ionizing_mev"], rtol=1e-13, atol=1e-15):
        raise ValueError("raw event totals disagree with reference")
    response = {**reference, "tracker_ownership_status": status, "tracker_ionizing_mev": energy}
    report = summarize(dict(events=events, steps=steps), response, {k: data[k] for k in ANCESTRY})
    report["source_foil_photon_census"] = group_event_energy(energy, se, n, status=status,
        source_pdg=data["stage_entry_pdg"], source_present=data["stage_entry_present"],
        foil_born_observed=data["has_new_foil_born_ancestor"],
        foil_photon_observed=data["has_new_foil_photon_ancestor"],
        ancestry_complete=data["foil_ancestry_complete"])
    residual = status == 0
    report["residual_event_origins"] = group_energy(energy[residual],
        **{k: steps[k][residual] for k in EVENT}, ordinal=ordinal[se[residual]],
        source_pdg=data["stage_entry_pdg"][residual],
        earliest_photon_present=data["earliest_new_photon_present"][residual],
        earliest_photon_volume=data["earliest_new_photon_birth_volume"][residual],
        foil_born_observed=data["has_new_foil_born_ancestor"][residual],
        foil_photon_observed=data["has_new_foil_photon_ancestor"][residual],
        ancestry_complete=data["foil_ancestry_complete"][residual])
    report.update(shards=len(paths), inputs=[str(p) for p in paths], source_inputs=source_inputs,
                  validation="Unique full tracker identities and source paths; sidecar event/step census checked; raw and categorized event totals match independent merged reference",
                  census_scope="Tracker counts cover supplied ancestry files; sidecars checked, not a new raw ROOT re-export or per-event zero-step census")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origins", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    with np.load(args.reference, allow_pickle=False) as data:
        reference = {k: data[k] for k in data.files if k.startswith("event_")}
    report = merge(args.origins, reference)
    report["reference"] = str(args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['shards']} shards; {report['input_events']} events; {report['tracker_steps']} tracker steps; {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
