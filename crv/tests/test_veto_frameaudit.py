"""Saved-frame denominator and conditional job-block uncertainty contracts."""
import copy

import awkward as ak
import numpy as np
import pytest

from crvfastsim.veto.card import VetoCard
from crvfastsim.veto.deadtime import factorised
from crvfastsim.veto.frameaudit import aggregate, audit_frame, main, track_histogram


def fixture_frames():
    co = ak.zip({"time": ak.Array([[1000.0], [], [300.0]]),
                 "timeStart": ak.Array([[995.0], [], [295.0]]),
                 "timeEnd": ak.Array([[1005.0], [], [305.0]]),
                 "PEs": ak.Array([[50.0], [], [500.0]]),
                 "nLayers": ak.Array([[3], [], [3]]),
                 "sectorType": ak.Array([[1], [], [2]])})
    frames = ak.zip({"run": np.full(3, 1430), "subrun": np.zeros(3, dtype=int),
                     "event": np.arange(1, 4), "coincs": co}, depth_limit=1)
    meta = {"events": 3, "frames": 3, "coincs": 2, "frac_frames_with_coinc": 2 / 3, "seed": 1,
            "pools": {"test": {"files": 1, "frames": 5, "stepped": 2}}}
    return frames, meta


def test_empty_frame_denominator_and_existing_method_parity():
    fr, meta = fixture_frames()
    card = VetoCard(gate_start_lo=420.0)
    times = np.array([1050.0])
    weights, edges = track_histogram(times, 0.25)
    row = audit_frame(fr, meta, card, weights, edges)
    old = factorised(times, fr.coincs, card)
    result = aggregate([row])
    assert result["frac"] == old["frac"] == 2 / 3
    assert result["err_iid_frames"] == pytest.approx(old["err"])
    assert result["err_job_blocks"] is None
    assert row["empty_frames"] == 1
    assert row["gate_fail_frac"] == row["passing_gate_coverage_frac"] == 1 / 3


@pytest.mark.parametrize("change,match", [({"frames": 4}, "frames"), ({"events": 4}, "events"),
                                          ({"coincs": 3}, "coincidence count"),
                                          ({"frac_frames_with_coinc": 1.0}, "occupancy")])
def test_sidecar_mismatch_rejected(change, match):
    fr, meta = fixture_frames()
    weights, edges = track_histogram([1050.0], 0.25)
    with pytest.raises(ValueError, match=match):
        audit_frame(fr, meta | change, VetoCard(), weights, edges)


def test_duplicates_and_nonfinite_rejected():
    fr, meta = fixture_frames()
    weights, edges = track_histogram([1050.0], 0.25)
    with pytest.raises(ValueError, match="duplicate"):
        audit_frame(fr[[0, 0, 2]], meta, VetoCard(), weights, edges)
    bad = ak.with_field(fr, ak.with_field(fr.coincs, fr.coincs.time * np.nan, "time"), "coincs")
    with pytest.raises(ValueError, match="nonfinite"):
        audit_frame(bad, meta, VetoCard(), weights, edges)
    badmeta = copy.deepcopy(meta)
    badmeta["pools"]["test"]["frames"] = 1
    with pytest.raises(ValueError, match="pool frame denominator"):
        audit_frame(fr, badmeta, VetoCard(), weights, edges)


def test_block_error_equal_and_unequal_job_sizes():
    base = {"empty_frames": 0, "coincs": 1, "frames_with_coinc": 1, "gate_fail": 0}
    rows = [base | {"frames": 10, "coverage_sum": s, "coverage_sq_sum": s} for s in (1, 2, 6)]
    result = aggregate(rows)
    assert result["err_job_blocks"] == pytest.approx(np.std([0.1, 0.2, 0.6], ddof=1) / np.sqrt(3))
    rows = [base | {"frames": 2, "coverage_sum": 1, "coverage_sq_sum": 1},
            base | {"frames": 4, "coverage_sum": 0, "coverage_sq_sum": 0}]
    result = aggregate(rows)
    assert result["frac"] == 1 / 6
    assert result["err_job_blocks"] == pytest.approx(2 / 9)


@pytest.mark.parametrize("times,step", [([], 0.25), ([np.nan], 0.25), ([1], 0), ([1], np.inf)])
def test_invalid_histogram(times, step):
    with pytest.raises(ValueError):
        track_histogram(times, step)


def test_cli_repeated_ids_across_jobs_are_valid(tmp_path):
    import json
    fr, meta = fixture_frames()
    card = VetoCard(gate_start_lo=420.0)
    selection = tmp_path / "selection.parquet"
    ak.to_parquet(ak.zip({"t_mid": ak.Array([[1050.0], [], []]),
                          "p_mid": ak.Array([[100.0], [], []]), "coincs": fr.coincs}, depth_limit=1), selection)
    ref = {"card": card.__dict__ | {"sha": card.sha()}, "pmin": 85, "pmax": 200, "selected": 1,
           "F": factorised([1050], fr.coincs, card)}
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(ref))
    paths = [tmp_path / "frames1.parquet", tmp_path / "frames2.parquet"]
    for i, path in enumerate(paths):
        ak.to_parquet(fr, path)
        path.with_suffix(".json").write_text(json.dumps(meta | {"seed": i}))
    output = tmp_path / "audit.json"
    args = [*map(str, paths), "--selection", str(selection), "--reference", str(reference), "--out", str(output)]
    main(args)
    report = json.loads(output.read_text())
    assert report["pooled"]["frames"] == 6
    assert report["pooled"]["empty_frames"] == 2
    assert report["pooled"]["err_job_blocks"] == 0
    assert report["delta_from_reference_F"] == 0
    with pytest.raises(FileExistsError):
        main(args)
