"""Synthetic saved-only authentication/publication; no physics inputs or model calls."""
import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from calofastsim.showers import direction_localization_run as runner


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    trial, production, out = (tmp_path / name for name in ("trial", "production", "report"))
    trial.mkdir()
    production.mkdir()
    names = ("conditional_arrays.npz", "diagnostics.npz", "clusters.npz", "pools.npz")
    arrays = dict(event_row=np.array([1, 2, 3]), cell_id=np.array([0, 0, 48]),
                  direction_used=np.array([False, False, True]), direction_fallback=np.array([True, True, False]))
    pools = dict(arrays, direction_count=np.array([0, 7, 8]), supported=np.ones(3, bool))
    for name in names:
        value = arrays if name == "conditional_arrays.npz" else pools if name == "pools.npz" else dict(synthetic=np.array([1, 2, 3]))
        np.savez(trial / name, **value)
    np.savez(production / "diagnostics.npz", synthetic=np.array([4, 5]))
    old_sha = digest(production / "diagnostics.npz")
    receipt = dict(completed=True, sampler_calls=1, reconstruction_calls=2,
        output_sha256={name: digest(trial / name) for name in names},
        input_sha256={"frozen/production_candidate_trial/diagnostics.npz": old_sha})
    runner.write_json(trial / "summary.json", receipt)
    monkeypatch.setattr(runner, "TRIAL_SHA", digest(trial / "summary.json"))
    monkeypatch.setattr(runner, "PREVIOUS_DIAGNOSTICS_SHA", old_sha)
    return argparse.Namespace(trial=trial, production=production, out=out), receipt


def repin(args, receipt, monkeypatch):
    runner.write_json(args.trial / "summary.json", receipt)
    monkeypatch.setattr(runner, "TRIAL_SHA", digest(args.trial / "summary.json"))


def test_authentication_parses_pinned_saved_bytes_only(bundle):
    args, _ = bundle
    inputs, saved, old = runner.authenticate(args.trial, args.production)
    assert set(saved) == {"conditional_arrays.npz", "diagnostics.npz", "clusters.npz", "pools.npz"}
    np.testing.assert_array_equal(old["synthetic"], [4, 5])
    assert len(inputs.sha256) == 6
    inputs.unchanged()
    assert not args.out.exists()


@pytest.mark.parametrize("name", ["summary.json", "conditional_arrays.npz", "diagnostics.npz", "clusters.npz", "pools.npz"])
def test_changed_trial_bytes_fail_without_outputs(bundle, name):
    args, _ = bundle
    with (args.trial / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.authenticate(args.trial, args.production)
    assert not args.out.exists()


def test_changed_prior_bytes_fail(bundle):
    args, _ = bundle
    with (args.production / "diagnostics.npz").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.authenticate(args.trial, args.production)


@pytest.mark.parametrize("key,value", [("completed", False), ("sampler_calls", 2), ("reconstruction_calls", 1)])
def test_incomplete_or_wrong_call_receipt_rejected(bundle, monkeypatch, key, value):
    args, receipt = bundle
    receipt[key] = value
    repin(args, receipt, monkeypatch)
    with pytest.raises(ValueError, match="completed one-draw/two-call"):
        runner.authenticate(args.trial, args.production)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "wrong"])
def test_prior_anchor_must_be_unique_and_match_literal_pin(bundle, monkeypatch, mode):
    args, receipt = bundle
    if mode == "missing":
        receipt["input_sha256"] = {}
    elif mode == "duplicate":
        receipt["input_sha256"]["second/production_candidate_trial/diagnostics.npz"] = runner.PREVIOUS_DIAGNOSTICS_SHA
    else:
        receipt["input_sha256"]["frozen/production_candidate_trial/diagnostics.npz"] = "0" * 64
    repin(args, receipt, monkeypatch)
    with pytest.raises(ValueError, match="uniquely anchored"):
        runner.authenticate(args.trial, args.production)


def test_existing_output_rejected_before_authentication(bundle, monkeypatch):
    args, _ = bundle
    args.out.mkdir()
    monkeypatch.setattr(runner, "authenticate", lambda *a: pytest.fail("must reject before reading"))
    with pytest.raises(FileExistsError):
        runner.run(args)


def install_analysis(monkeypatch, analyze):
    import calofastsim.showers as package
    fake = types.ModuleType("calofastsim.showers.direction_localization")
    fake.__file__ = __file__
    fake.analyze = analyze
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(package, "direction_localization", fake, raising=False)


def test_publish_new_output_once_and_hash_payload(bundle, monkeypatch):
    args, _ = bundle
    calls = []
    def analyze(arrays, diagnostic, previous, clusters):
        calls.append((arrays, diagnostic, previous, clusters))
        return dict(entries=3, scope="synthetic")
    install_analysis(monkeypatch, analyze)
    runner.run(args)
    assert len(calls) == 1
    summary = json.loads((args.out / "summary.json").read_text())
    assert summary["output_sha256"]["numbers.json"] == digest(args.out / "numbers.json")
    assert summary["sampler_calls"] == summary["placement_calls"] == summary["reconstruction_calls"] == summary["root_reads"] == 0
    with pytest.raises(FileExistsError):
        runner.run(args)
    assert len(calls) == 1


def test_changed_saved_bytes_during_analysis_prevent_publication(bundle, monkeypatch):
    args, _ = bundle
    def analyze(*values):
        with (args.trial / "clusters.npz").open("ab") as stream:
            stream.write(b" ")
        return {}
    install_analysis(monkeypatch, analyze)
    with pytest.raises(ValueError, match="input changed"):
        runner.run(args)
    assert not args.out.exists()


def test_analysis_failure_no_output_and_no_retry(bundle, monkeypatch):
    args, _ = bundle
    calls = []
    def analyze(*values):
        calls.append(True)
        raise ValueError("synthetic analysis failed")
    install_analysis(monkeypatch, analyze)
    with pytest.raises(ValueError, match="synthetic analysis failed"):
        runner.run(args)
    assert calls == [True] and not args.out.exists()


def test_strict_json_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        runner.write_json(tmp_path / "report.json", {"bad": float("nan")})


def test_saved_support_counts_keep_empty_cells_and_partition(bundle):
    args, _ = bundle
    _, saved, _ = runner.authenticate(args.trial, args.production)
    result = runner.support_counts(saved['pools.npz'], saved['conditional_arrays.npz'])
    assert len(result) == 99
    assert result['all']['count_categories'] == [1, 1, 1]
    assert result['direction_fallback']['count_categories'] == [1, 1, 0]
    assert result['cell_01']['entries'] == 0
    assert sum(result[f'cell_{i:02d}']['entries'] for i in range(96)) == 3


@pytest.mark.parametrize('mode', ['row', 'negative', 'float', 'flags', 'eligibility'])
def test_saved_support_rejects_inconsistent_values(bundle, mode):
    args, _ = bundle
    _, saved, _ = runner.authenticate(args.trial, args.production)
    pools, arrays = saved['pools.npz'], saved['conditional_arrays.npz']
    if mode == 'row':
        pools['event_row'][0] = 99
    elif mode == 'negative':
        pools['direction_count'][0] = -1
    elif mode == 'float':
        pools['direction_count'] = pools['direction_count'].astype(float)
    elif mode == 'flags':
        pools['direction_used'][0] = True
    else:
        pools['direction_count'][0] = 8
    with pytest.raises(ValueError):
        runner.support_counts(pools, arrays)
