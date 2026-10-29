"""Recheck saved common-photon timing evidence without generating optics or downstream response."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import geometry
from . import optical_timing_probe as probe
from .replay import IDS, sha256
from .replay_timing import DIGI_END, stock_step_mask, wrap_arrivals


def validate_saved(steps, events, q, channels, geom):
    """Recompute identity, timing masks and summaries from every saved photon, including rejected ones."""
    if steps.source_row.duplicated().any() or events.duplicated(list(IDS)).any():
        raise ValueError("duplicate source identity")
    for key in ("source_step_row", "step_index", "event_row", "chan", "draw_seed", "ordinal", "photon_id"):
        if q[key].dtype.kind not in "iu" or np.any(q[key] < 0):
            raise ValueError(f"invalid photon identity: {key}")
    for key in (*probe.FLAGS, "live_channel", "stock_step_kept", "endpoint_roundoff_step"):
        if q[key].dtype != bool:
            raise ValueError(f"invalid flag type: {key}")
    if np.any(q.step_index >= len(steps)) or np.any(q.event_row >= len(events)) or np.any(q.chan >= geom.nbars * 4):
        raise ValueError("photon identity outside saved domain")
    if q.duplicated(["source_step_row", "chan", "draw_seed", "ordinal"]).any():
        raise ValueError("duplicate photon key")
    if not np.array_equal(q.photon_id, np.arange(len(q))):
        raise ValueError("photon row ordering changed")
    expected_ordinal = q.groupby(["source_step_row", "chan"], sort=False).cumcount()
    if not np.array_equal(q.ordinal, expected_ordinal):
        raise ValueError("photon ordinal changed")
    si, er = q.step_index.to_numpy(), q.event_row.to_numpy()
    for k in IDS:
        if not np.array_equal(q[k], steps[k].to_numpy()[si]) or not np.array_equal(q[k], events[k].to_numpy()[er]):
            raise ValueError("photon full event identity changed")
    if (not np.array_equal(q.source_step_row, steps.source_row.to_numpy()[si])
            or not np.array_equal(q.chan // 4, steps.bar.to_numpy()[si])
            or not np.array_equal(q.endpoint_roundoff_step, steps.endpoint_roundoff.to_numpy()[si])):
        raise ValueError("photon step identity changed")
    w = -events.pbtime.to_numpy().astype(np.float32).astype(float)[er]
    event_index = pd.MultiIndex.from_frame(events[list(IDS)])
    step_event = event_index.get_indexer(pd.MultiIndex.from_frame(steps[list(IDS)]))
    if np.any(step_event < 0):
        raise ValueError("saved step outside event census")
    step_w = -events.pbtime.to_numpy().astype(np.float32).astype(float)[step_event]
    accepted = stock_step_mask(steps.t0, steps.t1, step_w)[si]
    np.testing.assert_array_equal(q.origin, w)
    np.testing.assert_array_equal(q.stock_step_kept, accepted)
    raw = wrap_arrivals(q.time_raw_absolute, w)
    rel = raw - w
    folded = wrap_arrivals(q.time_historical_relative)
    np.testing.assert_array_equal(q.time_raw_wrapped_absolute, raw)
    np.testing.assert_array_equal(q.time_raw_wrapped_relative, rel)
    np.testing.assert_array_equal(q.time_folded_wrapped_relative, folded)
    residual = rel - folded
    np.testing.assert_array_equal(q.raw_folded_residual_ns, residual)
    np.testing.assert_array_equal(q.raw_folded_circular_residual_ns,
                                  np.minimum(abs(residual), abs(abs(residual) - probe.PERIOD)))
    h = q.live_channel.to_numpy() & (q.time_historical_relative.to_numpy() <= DIGI_END)
    a = q.live_channel.to_numpy() & (raw <= w + DIGI_END)
    f = q.live_channel.to_numpy() & (folded <= DIGI_END)
    masks = dict(historical_kept=h, arrival_only_kept=a, rejection_only_kept=h & accepted,
                 stock_timing_kept=a & accepted, folded_arrival_kept=f,
                 raw_folded_upper_flip=a != f,
                 raw_folded_late_flip=q.live_channel.to_numpy() & ((raw >= w + 1400.) != (folded >= 1400.)),
                 rescued_by_arrival=a & ~h, removed_by_arrival=h & ~a,
                 rescued_survives_step=a & ~h & accepted,
                 accepted_time_changed=h & a & accepted & (q.time_historical_relative.to_numpy() != rel))
    for key, expected in masks.items():
        np.testing.assert_array_equal(q[key], expected)
    rebuilt, rechannels = probe.summaries(q, events[list(probe.EVENT_TYPES)], geom)
    pd.testing.assert_frame_equal(events, rebuilt)
    pd.testing.assert_frame_equal(channels, rechannels)


def block_summary(steps, q, events):
    extreme = q[q.endpoint_roundoff_step]
    column_names = [*IDS, "source_step_row", "chan", "ordinal", "time_raw_absolute", "time_historical_relative",
                    "time_raw_wrapped_relative", "raw_folded_residual_ns", "historical_kept", "stock_timing_kept"]
    selected = q[q.historical_kept & q.stock_timing_kept]
    magnitudes = np.abs(selected.time_raw_wrapped_relative - selected.time_historical_relative)
    return {"frames": len(events), "steps": len(steps), "photons": len(q), "counts": {k: int(q[k].sum()) for k in probe.FLAGS},
            "extreme_step_photons": len(extreme), "extreme_step_counts": {k: int(extreme[k].sum()) for k in probe.FLAGS},
            "extreme_photon_rows_limit": 100, "extreme_photon_rows": extreme[column_names].head(100).to_dict(orient="records"),
            "common_accepted_time_difference_counts": {str(cut): int(np.sum(magnitudes > cut)) for cut in (0., 1e-9, 1e-6, 1e-3, .1, 1.)},
            "max_raw_folded_circular_residual_ns": float(q.raw_folded_circular_residual_ns.max()) if len(q) else None}


def run(directory, out):
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise ValueError("choose a new report path")
    summary_path = directory / "summary.json"
    summary_hash = sha256(summary_path)
    report = json.loads(summary_path.read_text())
    if (report["schema"] != "crv-optical-timing-probe-v1" or report["rng_contract"] != probe.RNG_CONTRACT
            or report["F"] is not None or report["timing_closure"] is not False):
        raise ValueError("unrecognized diagnostic contract")
    hashes = report["assets"] | report["inputs"] | report["code_sha256"]
    for path, digest in hashes.items():
        probe.bound(path, digest)
    counters = [p for p, digest in report["assets"].items() if digest == probe.COUNTERS_SHA]
    if len(counters) != 1:
        raise ValueError("requires pinned counters asset")
    geom = geometry.from_offline_config(counters[0])
    checks = []
    for number, block in enumerate(report["blocks"]):
        for path, digest in block["input_bindings"].items():
            probe.bound(path, digest)
        tables = {}
        for name in ("steps", "photons", "events", "channels"):
            artifact = block["artifacts"][name]
            if artifact["path"] != f"input{number}_{name}.parquet":
                raise ValueError("unexpected artifact path")
            tables[name] = pd.read_parquet(probe.bound(directory / artifact["path"], artifact["sha256"]))
            if len(tables[name]) != artifact["rows"]:
                raise ValueError("artifact row count changed")
        steps, q, events, channels = (tables[n] for n in ("steps", "photons", "events", "channels"))
        if not np.all(q.draw_seed == block["seed"]):
            raise ValueError("photon draw seed changed")
        validate_saved(steps, events, q, channels, geom)
        stats = block_summary(steps, q, events)
        for key in ("photons", "counts", "extreme_step_photons", "extreme_step_counts", "max_raw_folded_circular_residual_ns"):
            if stats[key] != block[key]:
                raise ValueError("saved summary disagrees with photons")
        if events.to_dict(orient="records") != block["selected_events"]:
            raise ValueError("saved event summary disagreement")
        checks.append({"input_sha256": block["input_sha256"], "seed": block["seed"], **stats})
    if sha256(summary_path) != summary_hash:
        raise ValueError("summary changed during verification")
    result = {"schema": "crv-optical-timing-report-v1", "summary_sha256": summary_hash, "blocks": checks,
              "all_saved_fields_verified": True, "new_optical_draw": False, "scope": probe.SCOPE,
              "F": None, "timing_closure": False, "code_sha256": sha256(__file__)}
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.probe, args.out), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
