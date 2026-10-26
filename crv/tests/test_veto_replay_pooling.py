"""Sidecar-bound paired replay pooling rejects stale contracts and invalid denominators."""
import copy
import json

import awkward as ak
import numpy as np
import pytest

from crvfastsim.veto import replay_metrics as metrics
from crvfastsim.veto.card import VetoCard
from crvfastsim.veto.compareframes import COLS
from crvfastsim.veto.fastframes import run_config
from crvfastsim.veto.replay import sha256
from crvfastsim.veto.replay_policy import (
    INTENDED, LEGACY_STOCK, LEGACY_WRITERS, LEGACY_FRAMES, TIMING, apply_policy, metadata_identity,
)


def metadata(policy=INTENDED):
    response, contract = apply_policy(run_config("run1a_v01")[0], policy)
    return {"input_sha256": "a" * 64, "conditions_sha256": "b" * 64,
            "photon_dev_sha256": "c" * 64, "timing": TIMING, "response_contract": contract,
            "stock_shared_table_yields_diagnostic": policy == LEGACY_STOCK,
            "effective_yields": {k: v.yield_per_mev for k, v in response.sectors.items()},
            "frames": 2, "coincs": 1}


def checked(meta, legacy_policy=None):
    identity, digest, recovery = metadata_identity(meta, legacy_policy=legacy_policy)
    return {"input_sha256": meta["input_sha256"], "conditions_sha256": meta["conditions_sha256"],
            "fast_metadata_sha256": "d" * 64, "response_identity": identity,
            "response_identity_sha256": digest, "legacy_policy_recovery": recovery,
            "events_check": {"events": meta["frames"], "coincs": 0},
            "entries_highest_cycle": {"events": meta["frames"], "clusters": 0},
            "comparison": {"common_frames": meta["frames"],
                           "fast": {"frames": meta["frames"], "coincs": meta["coincs"]},
                           "stock": {"frames": meta["frames"], "coincs": 0}}}


def frames(n=2, coincidence=True):
    counts = np.zeros(n, dtype=int)
    counts[0] = int(coincidence)
    values = {"time": 1000., "timeStart": 995., "timeEnd": 1005., "PEs": 50., "nLayers": 3, "sectorType": 1}
    co = ak.zip({k: ak.unflatten(np.full(int(counts.sum()), v), counts) for k, v in values.items()})
    return ak.zip({"run": np.full(n, 1430), "subrun": np.zeros(n, dtype=int),
                   "event": np.arange(1, n + 1), "coincs": co}, depth_limit=1)


def test_sidecar_hash_and_old_checks_require_recheck():
    meta = metadata()
    check = checked(meta)
    assert metrics.verify_metadata(check, meta, "d" * 64)[1] == check["response_identity_sha256"]
    for key in ("fast_metadata_sha256", "conditions_sha256", "response_identity", "response_identity_sha256", "legacy_policy_recovery"):
        old = {k: v for k, v in check.items() if k != key}
        with pytest.raises(ValueError, match="rerun replay_check on retained outputs"):
            metrics.verify_metadata(old, meta, "d" * 64)
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        metrics.verify_metadata(check, meta, "e" * 64)


@pytest.mark.parametrize("key,value,match", [
    ("input_sha256", "e" * 64, "input identity"),
    ("conditions_sha256", "e" * 64, "conditions identity"),
    ("photon_dev_sha256", "e" * 64, "response identity"),
    ("timing", "different folding", "timing"),
    ("effective_yields", {"T1": 1, "T2": 1}, "effective yields"),
])
def test_metadata_identity_changes_rejected_even_with_matching_sidecar_hash(key, value, match):
    meta = metadata()
    check = checked(meta)
    with pytest.raises(ValueError, match=match):
        metrics.verify_metadata(check, meta | {key: value}, "d" * 64)


def test_checked_identity_digest_and_recovery_are_verified():
    meta = metadata()
    check = checked(meta)
    for key, value in (("response_identity_sha256", "0" * 64), ("legacy_policy_recovery", "invented")):
        with pytest.raises(ValueError):
            metrics.verify_metadata(check | {key: value}, meta, "d" * 64)
    changed = copy.deepcopy(check)
    changed["response_identity"]["photon_dev_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="response identity"):
        metrics.verify_metadata(changed, meta, "d" * 64)


def test_legacy_policy_only_recovered_when_check_explicitly_records_it():
    meta = metadata()
    del meta["response_contract"]
    meta["code_sha256"] = {"replay_fast.py": next(k for k, v in LEGACY_WRITERS.items() if v == INTENDED),
                           "fastframes.py": LEGACY_FRAMES}
    check = checked(meta, legacy_policy=INTENDED)
    assert metrics.verify_metadata(check, meta, "d" * 64)[2] is not None
    with pytest.raises(ValueError, match="explicit --legacy-yield-policy"):
        metrics.verify_metadata(check | {"legacy_policy_recovery": None}, meta, "d" * 64)


@pytest.mark.parametrize("change", [
    {"conditions_sha256": "e" * 64}, {"photon_dev_sha256": "e" * 64}, {"policy": LEGACY_STOCK},
])
def test_response_identity_not_just_boolean_controls_pooling(change):
    first = checked(metadata())
    other = metadata(change.get("policy", INTENDED)) | {k: v for k, v in change.items() if k != "policy"}
    second = checked(other | {"input_sha256": "f" * 64})
    with pytest.raises(ValueError, match="different response identities"):
        metrics.validate_pool([first, second])
    metrics.validate_pool([first, checked(metadata() | {"input_sha256": "f" * 64})])
    with pytest.raises(ValueError, match="replicas of one input"):
        metrics.validate_pool([first, first])


@pytest.mark.parametrize("section,key", [
    ("sidecar", "frames"), ("sidecar", "coincs"), ("events_check", "events"),
    ("events_check", "coincs"), ("entries_highest_cycle", "events"),
    ("entries_highest_cycle", "clusters"), ("comparison", "common_frames"),
    ("fast", "frames"), ("fast", "coincs"), ("stock", "frames"), ("stock", "coincs"),
])
def test_actual_counts_match_every_checked_denominator(section, key):
    meta = metadata()
    check = checked(meta)
    assert metrics.validate_frames(frames(), frames(coincidence=False), check, meta) == [1, 0]
    target = meta if section == "sidecar" else check["comparison"][section] if section in ("fast", "stock") else check[section]
    target[key] += 1
    with pytest.raises(ValueError, match="count differs"):
        metrics.validate_frames(frames(), frames(coincidence=False), check, meta)


@pytest.mark.parametrize("value", [True, 2.0, -1, None])
def test_invalid_count_types(value):
    meta = metadata()
    with pytest.raises(ValueError, match="count differs"):
        metrics.validate_frames(frames(), frames(coincidence=False), checked(meta), meta | {"frames": value})


@pytest.mark.parametrize("field", COLS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coincidence_fields_rejected(field, value):
    meta = metadata()
    fast = frames()
    bad = ak.with_field(fast, ak.with_field(fast.coincs, fast.coincs[field] * value, field), "coincs")
    with pytest.raises(ValueError, match="nonfinite fast coincidence field"):
        metrics.validate_frames(bad, frames(coincidence=False), checked(meta), meta)
    with pytest.raises(ValueError, match="nonfinite stock coincidence field"):
        metrics.validate_frames(fast, bad, checked(meta), meta)


@pytest.mark.parametrize("value", [-1e-15, 1 + 1e-15, np.nan, np.inf])
def test_coverage_range_and_finiteness(value):
    for fast, stock in (([value], [0]), ([0], [value])):
        with pytest.raises(ValueError, match="coverage"):
            metrics.paired_stats(fast, stock)


def test_duplicate_or_different_frame_ids():
    meta = metadata()
    check = checked(meta)
    with pytest.raises(ValueError, match="duplicate"):
        metrics.validate_frames(frames()[[0, 0]], frames(coincidence=False), check, meta)
    with pytest.raises(ValueError, match="identities differ"):
        metrics.validate_frames(frames(), frames(coincidence=False)[::-1], check, meta)


def test_cli_unequal_blocks_provenance_and_tampered_sidecar(tmp_path, monkeypatch):
    selection, reference, output = (tmp_path / name for name in ("selection.parquet", "reference.json", "metrics.json"))
    ak.to_parquet(ak.zip({"t_mid": ak.Array([[1050.]]), "p_mid": ak.Array([[100.]]),
                          "coincs": frames(1, coincidence=False).coincs}, depth_limit=1), selection)
    card = VetoCard(gate_start_lo=420.)
    reference.write_text(json.dumps({"card": card.__dict__ | {"sha": card.sha()}, "selected": 1,
                                     "pmin": 85, "pmax": 200, "F": {"grid_ns": 0.25}}))
    paths, checks, stock_frames = [], [], {}
    for i, n in enumerate((2, 4)):
        fast, stock, check_path = (tmp_path / f"{kind}{i}.{suffix}" for kind, suffix in
                                  (("fast", "parquet"), ("stock", "root"), ("check", "json")))
        ak.to_parquet(frames(n, coincidence=i == 0), fast)
        stock.write_bytes(f"mock stock {i}".encode())
        stock_frames[str(stock)] = frames(n, coincidence=False)
        meta = metadata() | {"input_sha256": str(i) * 64, "frames": n, "coincs": int(i == 0)}
        fast.with_suffix(".json").write_text(json.dumps(meta))
        check = checked(meta) | {"conditions_equal": True, "stock": str(stock), "stock_sha256": sha256(stock),
                                 "fast_sha256": sha256(fast), "fast_metadata_sha256": sha256(fast.with_suffix(".json"))}
        check_path.write_text(json.dumps(check))
        paths.append(str(fast))
        checks.append(str(check_path))
    monkeypatch.setattr(metrics, "dump_frames", lambda paths: stock_frames[paths[0]])
    args = ["--checks", *checks, "--fast", *paths, "--selection", str(selection), "--reference", str(reference)]
    metrics.main(args + ["--out", str(output)])
    report = json.loads(output.read_text())
    assert report["pooled"]["frames"] == 6
    assert report["pooled"]["delta"] == pytest.approx(1 / 6)
    assert report["pooled"]["delta_err_blocks"] == pytest.approx(2 / 9)
    assert report["pooled"]["delta_err_paired"] == pytest.approx(np.std([1, 0, 0, 0, 0, 0], ddof=1) / np.sqrt(6))
    assert report["selection_sha256"] == sha256(selection)
    assert report["reference_sha256"] == sha256(reference)
    assert report["grid_ns"] == 0.25
    assert report["blocks"][0]["check_sha256"] == sha256(checks[0])
    sidecar = tmp_path / "fast0.json"
    sidecar.write_text(sidecar.read_text() + "\n")
    rejected = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        metrics.main(args + ["--out", str(rejected)])
    assert not rejected.exists()
