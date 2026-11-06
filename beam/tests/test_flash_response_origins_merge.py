"""The census merge must preserve identity and independent event totals."""
import json

import numpy as np
import pytest

from beamfastsim.flash_response_origins import trace
from beamfastsim.flash_response_origins_merge import merge, main, group_event_energy
from test_flash_response_origins import case


def payload(t, r):
    return {**trace(t), **{"tracker_" + k: v for k, v in t["steps"].items()},
            "tracker_ownership_status": r["tracker_ownership_status"],
            "tracker_ionizing_mev": r["tracker_ionizing_mev"]}


def save(path, data, events=2):
    np.savez_compressed(path, **data)
    path.with_suffix(".json").write_text(json.dumps(dict(input=str(path.with_suffix(".root")),
        input_events=events, tracker_steps=len(data["tracker_ionizing_mev"]))))


@pytest.fixture
def saved(case, tmp_path):
    t, r = case
    r["event_tracker_ionizing_mev"] = np.array([1.25, 0.])
    p = tmp_path / "origins.npz"
    save(p, payload(t, r))
    return t, r, p


def test_category_error_uses_event_sums_and_zero_events():
    out = group_event_energy(np.array([1., 2., 4.]), np.array([0, 0, 1]), 4,
                             category=np.array([0, 0, 0]))
    assert out["ionizing_mev"] == [7.]
    assert out["nonzero_events"] == [2]
    assert out["effective_events"] == [49 / 25]
    assert out["total_event_se_mev"][0] == pytest.approx(np.sqrt(4 * np.var([3., 4., 0., 0.], ddof=1)))
    empty = group_event_energy(np.array([]), np.array([], int), 2, category=np.array([], int))
    assert empty["total_event_se_mev"] == []


def test_merge_matches_original_and_cli(saved, tmp_path):
    _, r, p = saved
    report = merge([p], r)
    assert report["input_events"] == 2
    assert report["tracker_steps"] == 2
    assert report["top_residual_events"]["sum_mev"] == 1.
    assert report["top_owned_events"]["sum_mev"] == .25
    assert report["residual_event_origins"]["event"] == [5]
    assert report["residual_event_origins"]["earliest_photon_present"] == [False]
    assert sum(report["source_foil_photon_census"]["ionizing_mev"]) == 1.25
    ref, out = tmp_path / "reference.npz", tmp_path / "merged.json"
    np.savez_compressed(ref, **r)
    argv = ["--origins", str(p), "--reference", str(ref), "--out", str(out)]
    assert main(argv) == 0
    assert json.loads(out.read_text())["input_events"] == 2
    with pytest.raises(FileExistsError):
        main(argv)


def test_two_disjoint_shards_preserve_high_unsigned_identity(saved, tmp_path):
    t, r, p = saved
    d = payload(t, r)
    d["tracker_event"] = d["tracker_event"] + np.uint64(100)
    q = tmp_path / "second.npz"
    save(q, d)
    ref = {k: np.concatenate([v, v + np.uint64(100) if k in
           ("event_event", "event_ordinal") else v]) for k, v in r.items() if k.startswith("event_")}
    report = merge([p, q], ref)
    assert report["input_events"] == 4
    assert report["tracker_steps"] == 4
    assert report["top_residual_events"]["sum_mev"] == 2.
    assert set(report["top_residual_events"]["event"]) == {5, 105}


def test_duplicate_paths_and_tracker_identity_rejected(saved, tmp_path):
    t, r, p = saved
    with pytest.raises(ValueError, match="unique"):
        merge([p, p], r)
    q = tmp_path / "duplicate.npz"
    save(q, payload(t, r))
    with pytest.raises(ValueError, match="Duplicate"):
        merge([p, q], r)


@pytest.mark.parametrize("field", ["event_tracker_ionizing_mev", "event_owned_ionizing_mev"])
def test_reference_energy_mismatch_fails(saved, field):
    _, r, p = saved
    r[field][0] += 1
    with pytest.raises(ValueError, match="disagree"):
        merge([p], r)


def test_missing_event_and_duplicate_ordinal(saved):
    _, r, p = saved
    r["event_event"][0] = 99
    with pytest.raises(ValueError, match="absent"):
        merge([p], r)
    r["event_event"][0] = 5
    r["event_ordinal"][1] = r["event_ordinal"][0]
    with pytest.raises(ValueError, match="duplicate event ordinals"):
        merge([p], r)


def test_empty_tracker_keeps_all_events(case, tmp_path):
    t, r = case
    t["steps"] = {k: v[:0] for k, v in t["steps"].items()}
    r["tracker_ownership_status"] = np.array([], np.int8)
    r["tracker_ionizing_mev"] = np.array([])
    for k in ("tracker", "owned", "residual", "unknown"):
        r["event_" + k + "_ionizing_mev"] = np.zeros(2)
    p = tmp_path / "empty.npz"
    save(p, payload(t, r))
    report = merge([p], r)
    assert report["input_events"] == 2
    assert report["tracker_steps"] == 0
    assert report["top_residual_events"]["sum_mev"] == 0.


@pytest.mark.parametrize("field", ["event_run", "event_unknown_ionizing_mev"])
def test_truncated_reference_cannot_broadcast(saved, field):
    _, r, p = saved
    r[field] = r[field][:1]
    with pytest.raises(ValueError, match="denominator|align"):
        merge([p], r)


def test_missing_zero_shard_and_mixed_empty_identity(saved, tmp_path):
    t, r, p = saved
    d = payload(t, r)
    d = {k: v[:0] for k, v in d.items()}
    for k in ("run", "subrun", "event", "product", "index"):
        d["tracker_" + k] = np.array([], np.int64)
    q = tmp_path / "empty.npz"
    save(q, d)
    ref = {k: np.concatenate([v, v + np.uint64(100) if k in
           ("event_event", "event_ordinal") else v if k in ("event_run", "event_subrun")
           else np.zeros_like(v)]) for k, v in r.items() if k.startswith("event_")}
    assert merge([p, q], ref)["tracker_steps"] == 2
    with pytest.raises(ValueError, match="denominator"):
        merge([p], ref)
    sidecar = json.loads(q.with_suffix(".json").read_text())
    sidecar["tracker_steps"] = 1
    q.with_suffix(".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="census"):
        merge([p, q], ref)


@pytest.mark.parametrize("field,value,match", [("tracker_ionizing_mev", np.nan, "energy"),
                                               ("tracker_ownership_status", 5, "status")])
def test_corrupt_payload_rejected(saved, field, value, match):
    t, r, p = saved
    d = payload(t, r)
    d[field][0] = value
    save(p, d)
    with pytest.raises(ValueError, match=match):
        merge([p], r)
