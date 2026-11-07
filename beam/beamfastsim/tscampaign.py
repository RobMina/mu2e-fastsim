"""Read finished Run 1A shards; held-out reads require explicit cohort selection."""
import argparse
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import uproot

from . import tssmoke
from .tsreference import _tree


def status_g4(path):
    """Decode the stock StatusG4 object with its file-provided ROOT streamer."""
    with uproot.open(path) as f:
        events, meta = _tree(f, "Events")
        names = [k for k in events.keys() if "StatusG4_g4run__" in k and k.endswith(".obj")]
        if len(names) != 1:
            return {"decoded": False, "reason": f"expected one StatusG4 object branch, got {names}"}
        try:
            a = events[names[0]].array(library="ak")
            fields = ("_status", "_nG4Tracks", "_overflowSimParticles", "_nKilledStepLimit",
                      "_nKilledByFieldPropagator", "_cpuTime", "_realTime")
            missing = set(fields) - set(ak.fields(a))
            if missing:
                return {"decoded": False, "reason": f"missing members: {sorted(missing)}",
                        "fields": ak.fields(a)}
            data = {k: ak.to_numpy(a[k]) for k in fields}
        except (ValueError, TypeError, NotImplementedError, KeyError) as exc:
            return {"decoded": False, "reason": f"{type(exc).__name__}: {exc}"}
        if any(v.dtype.kind not in "biuf" or v.shape != (len(a),) for v in data.values()):
            return {"decoded": False, "reason": "nonnumeric or nonscalar StatusG4 members"}
        valid = (np.all(np.isin(data["_status"], [0, 1, 10]))
                 and all(np.all(np.isfinite(data[k]) & (data[k] >= 0)) for k in fields)
                 and np.all(data["_nG4Tracks"] < 10**8))
        if not valid:
            return {"decoded": False, "reason": "stock object streamer produced invalid status/count/time values; no byte-order correction assumed",
                    "branch": names[0], "fields": ak.fields(a)}
        values, counts = np.unique(data["_status"], return_counts=True)
        return {"decoded": True, "tree": meta["tree"], "events": len(a),
                "status_counts": dict(zip(map(str, values), map(int, counts))),
                "overflow_events": int(np.count_nonzero(data["_overflowSimParticles"])),
                "step_limit_killed_tracks": int(data["_nKilledStepLimit"].sum()),
                "field_propagator_killed_tracks": int(data["_nKilledByFieldPropagator"].sum()),
                "g4_tracks": int(data["_nG4Tracks"].sum()),
                "cpu_seconds": float(data["_cpuTime"].sum(dtype=np.float64)),
                "real_seconds": float(data["_realTime"].sum(dtype=np.float64))}


def status_histograms(path):
    """Read stock StatusG4Analyzer histograms including flow bins; no status-object decoding."""
    with uproot.open(path) as f:
        names = ("statusValue", "numG4Tracks", "overflowSimParticles", "hNKilledStep")
        histograms = {}
        for name in names:
            hist, meta = _tree(f, name)
            histograms[name] = {"key": meta["tree"], "entries": int(hist.member("fEntries")),
                                "edges": hist.axis().edges().tolist(), "counts_with_flow": hist.values(flow=True).tolist()}
        events = histograms["numG4Tracks"]["entries"]
        nonzero = histograms["statusValue"]["entries"]
        return {"events": events, "zero_status_events": events - nonzero, "nonzero_status_events": nonzero,
                "overflow_events": histograms["overflowSimParticles"]["entries"],
                "step_limit_events": histograms["hNKilledStep"]["entries"], "histograms": histograms,
                "limits": "stock histogram summary, not event-keyed statuses; field-propagator kills have no dedicated histogram"}


def _cohort_indices(cohort):
    if cohort == "development":
        return range(0, 100)
    if cohort == "heldout":
        return range(100, 200)
    raise ValueError("cohort must be development or heldout")


def inspect_shard(root, index, out, *, generation_job, cohort="development"):
    """Read one shard from the explicitly selected cohort after successful art exit."""
    indices = _cohort_indices(cohort)
    if index not in indices:
        raise ValueError(f"{cohort} indices must be {indices.start}..{indices.stop - 1}; "
                         "files outside the selected cohort are not opened")
    root, out = Path(root), Path(out)
    shard = root / "shards" / f"{index:03d}"
    if not str(generation_job).isascii() or not str(generation_job).isdecimal() or int(generation_job) <= 0:
        raise ValueError("generation_job must be a positive integer job ID")
    log = root / "logs" / f"{generation_job}_{index}.log"
    if "Art has completed and will exit with status 0." not in log.read_text():
        raise ValueError(f"no successful art completion in {log}")
    art = shard / f"sim.user.TSRun1ARaw.campaign_v1.001430_{920000 + index:08d}.art"
    start = time.perf_counter()
    report = tssmoke.inspect(art, shard / "tsrun1a_raw.root", 5000, compact=True)
    report.update(index=index, cohort=cohort, status_g4=status_g4(art),
                  seconds=time.perf_counter() - start)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def combine(paths, *, cohort="development"):
    """Sum one cohort without converting unknown ancestry into losses."""
    allowed_indices = _cohort_indices(cohort)
    reports = [json.loads(Path(p).read_text()) for p in paths]
    indices = [r["index"] for r in reports]
    if not reports or len(set(indices)) != len(indices):
        raise ValueError(f"nonempty distinct {cohort} shards required")
    if any(r["cohort"] != cohort or r["index"] not in allowed_indices for r in reports):
        raise ValueError(f"{cohort} reports only, with indices {allowed_indices.start}..{allowed_indices.stop - 1}")
    counts = ("finished_art_events", "events_in_ntpart", "primary_proton_events", "particle_rows",
              "coll1_mu_pi_forward_rows", "coll1_mu_pi_forward_tracks", "tracks_with_repeated_forward_records",
              "events_with_forward_mu_pi", "events_without_forward_mu_pi_in_ntpart", "ds_beam_all_species_rows",
              "ds_mu_pi_rows", "ds_mu_pi_linked_to_forward_source", "ds_mu_pi_without_forward_source",
              "ds_mu_pi_ancestry_incomplete", "source_tracks_with_ds_mu_pi_descendant",
              "source_tracks_without_linked_ds_mu_pi_descendant", "source_tracks_without_ds_mu_pi_descendant",
              "source_tracks_with_multiple_ds_descendants", "linked_ds_rows_preceding_source")
    total = {k: None if any(r[k] is None for r in reports) else sum(r[k] for r in reports) for k in counts}
    species = {str(pdg): {k: sum(r["source_species"][str(pdg)][k] for r in reports)
                         for k in ("inputs", "with_ds_mu_pi")} for pdg in (-211, -13, 13, 211)}
    status = {"decoded_shards": sum(r["status_g4"]["decoded"] for r in reports)}
    decoded = [r["status_g4"] for r in reports if r["status_g4"]["decoded"]]
    if decoded:
        status["status_counts"] = {k: sum(r["status_counts"].get(k, 0) for r in decoded)
                                   for k in sorted(set().union(*(r["status_counts"] for r in decoded)))}
        for k in ("events", "overflow_events", "step_limit_killed_tracks", "field_propagator_killed_tracks"):
            status[k] = sum(r[k] for r in decoded)
    return {"cohort": cohort, "indices": sorted(indices), "shards": len(reports),
            "checks_passed": all(r["checks_passed"] for r in reports), **total,
            "source_species": species, "status_g4": status,
            "seconds": sum(r["seconds"] for r in reports),
            "limits": reports[0]["limits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--index", type=int)
    parser.add_argument("--cohort", choices=("development", "heldout"), default="development",
                        help="census cohort; held-out access requires explicit selection")
    parser.add_argument("--generation-job", type=int, help="generation array job ID for completion logs")
    parser.add_argument("--status-art", help="inspect only the StatusG4 object in a finished smoke art")
    parser.add_argument("--combine", nargs="+")
    parser.add_argument("--status-histograms", help="finished output of stock tsstatus.fcl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.status_histograms:
        report = status_histograms(args.status_histograms)
    elif args.status_art:
        report = status_g4(args.status_art)
    elif args.combine:
        report = combine(args.combine, cohort=args.cohort)
    elif args.root is not None and args.index is not None:
        if args.generation_job is None:
            parser.error("--root/--index requires --generation-job")
        report = inspect_shard(args.root, args.index, args.out, generation_job=args.generation_job,
                               cohort=args.cohort)
    else:
        parser.error("use --root/--index/--generation-job, --combine, --status-art, or --status-histograms")
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("trees", "art_event_branches")},
                     indent=2, allow_nan=False))
    raise SystemExit(0 if report.get("checks_passed", True) else 1)


if __name__ == "__main__":
    main()
