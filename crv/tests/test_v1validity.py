"""V1 evidence validity is separate from the unchanged physics tolerances."""
import copy
import json
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeWarning

from crvfastsim.analyses import v1compare as V
from test_v1compare import _state


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(V, "GROUPS", {"EX": [0]})
    return _state(1.0)


def test_valid_state_and_zero_chi2(state):
    state["fast"] = copy.deepcopy(state["ref"])
    result = V.summarise(state)
    assert result["pass"]
    assert result["groups"]["EX"]["len_ratio"]["chi2_ndf"] == 0


def test_empty_state_fails_and_retains_groups():
    result = V.summarise(V._new_state())
    assert not result["pass"]
    assert set(result["groups"]) == set(V.GROUPS)
    assert all(not r["pass"] for r in result["groups"].values())


def test_empty_required_group_is_not_skipped():
    result = V.summarise(_state(1.0))
    assert not result["pass"]
    assert "T1" in result["groups"]
    assert not result["groups"]["T1"]["checks"]["population"]


@pytest.mark.parametrize("arm", ["fast", "ref"])
def test_empty_arm_fails(state, arm):
    state[arm] = V._new_arm()
    result = V.summarise(state)
    assert not result["pass"]
    assert not result["groups"]["EX"]["checks"]["population"]


@pytest.mark.parametrize("events", [0, -1, float("nan"), 123])
def test_invalid_event_universe(state, events):
    state["fast"]["events"] = events
    assert not V.summarise(state)["pass"]


@pytest.mark.parametrize("matched", [0, 200_000])
def test_no_in_range_time_evidence_fails(state, matched):
    state["matched"]["EX"] = matched
    state["dt"]["EX"] = [0] * len(state["dt"]["EX"])
    r = V.summarise(state)["groups"]["EX"]
    assert not r["checks"]["t0_bias"]
    assert "t0_bias" in r["invalid_reasons"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_histograms(value):
    h = np.full(600, value)
    assert not V._hist_mpv(h, V.EDGES["edep"])["valid"]
    assert not V._hist_ratio_band(h, h, V.EDGES["edep"])["valid"]


def test_sparse_shapes_and_underconstrained_fit_fail():
    h = np.ones(600)
    result = V._hist_ratio_band(h, h, V.EDGES["edep"])
    assert not result["valid"] and result["nbins_used"] == 0
    h[:] = 0
    h[50:53] = 1000
    assert V._hist_mpv(h, V.EDGES["edep"])["reason"] == "insufficient_fit_bins"


def test_zero_candidate_bin_fails():
    hr = np.full(600, 1000)
    hf = hr.copy()
    hf[:100] = 0
    result = V._hist_ratio_band(hf, hr, V.EDGES["edep"])
    assert not result["valid"]
    assert result["reason"] == "empty_candidate_comparison_bin"


@pytest.mark.parametrize("parameter,value", [(0, -1), (1, -122.06), (1, 0), (1, 20), (2, -1), (2, 20), (1, float("nan"))])
def test_invalid_fit_parameters(state, monkeypatch, parameter, value):
    par = np.array([10000., 3.4, .5])
    par[parameter] = value
    monkeypatch.setattr("scipy.optimize.curve_fit", lambda *a, **k: (par, np.eye(3) * .0001))
    r = V.summarise(state)["groups"]["EX"]
    assert not r["checks"]["edep_mpv"]
    assert not r["edep_mpv_fits"]["fast"]["valid"]


@pytest.mark.parametrize("variance", [-1, 0, float("nan"), float("inf"), 2.31e6 ** 2])
def test_invalid_or_unidentified_covariance(state, monkeypatch, variance):
    cov = np.eye(3) * .0001
    cov[1, 1] = variance
    monkeypatch.setattr("scipy.optimize.curve_fit", lambda *a, **k: (np.array([10000., 3.4, .5]), cov))
    assert not V.summarise(state)["groups"]["EX"]["checks"]["edep_mpv"]


@pytest.mark.parametrize("matched", [float("inf"), float("nan"), 200000.5, 20000000])
def test_invalid_matched_count_fails(state, matched):
    state["matched"]["EX"] = matched
    assert not V.summarise(state)["groups"]["EX"]["checks"]["t0_bias"]


def test_unidentified_scale_error_fails(state, monkeypatch):
    cov = np.diag([.0001, .0001, 1e30])
    monkeypatch.setattr("scipy.optimize.curve_fit", lambda *a, **k: (np.array([10000., 3.4, .5]), cov))
    assert not V.summarise(state)["groups"]["EX"]["checks"]["edep_mpv"]


@pytest.mark.parametrize("failure", [RuntimeError, ValueError, TypeError, OptimizeWarning])
def test_fit_failure_is_reported(state, monkeypatch, failure):
    def fail(*args, **kwargs):
        if failure is OptimizeWarning:
            warnings.warn("covariance unavailable", OptimizeWarning)
        raise failure("fit failed")
    monkeypatch.setattr("scipy.optimize.curve_fit", fail)
    assert V.summarise(state)["groups"]["EX"]["edep_mpv_fits"]["ref"]["reason"] == "fit_failed"


def test_merge_preserves_validity(state):
    state["fast"] = copy.deepcopy(state["ref"])
    merged = V._new_state()
    V._merge_into(merged, state)
    V._merge_into(merged, state)
    assert V.summarise(merged)["pass"]
    broken = copy.deepcopy(state)
    broken["ref"]["groups"]["EX"]["h_edep"] = [0] * 600
    assert not V.summarise(broken)["pass"]


@pytest.mark.parametrize("emit", [[], ["--emit", "unused.json"]])
def test_cli_failure_is_nonzero_and_writes_strict_json(tmp_path, monkeypatch, emit):
    from crvfastsim import cli
    monkeypatch.setattr(V, "GROUPS", {"EX": [0]})
    monkeypatch.setattr(V, "set_groups", lambda *args: None)
    batch = tmp_path / "empty.json"
    batch.write_text(json.dumps(V._new_state()))
    with pytest.raises(SystemExit, match="1"):
        cli.main(["v1compare", str(batch), "--merge", "--outdir", str(tmp_path)] + emit)
    result = json.loads((tmp_path / "v1compare_stats.json").read_text(),
                        parse_constant=lambda s: pytest.fail(f"nonstandard JSON {s}"))
    assert not result["pass"]
    assert result["groups"]["EX"]["edep_mpv_rel"] is None


@pytest.mark.parametrize("passed,emit", [(True, None), (False, "batch.json")])
def test_cli_pass_and_accumulation_exit_zero(monkeypatch, passed, emit):
    from crvfastsim import cli
    monkeypatch.setitem(cli.ANALYSES, "v1compare", (lambda *a, **k: {"pass": passed}, "test"))
    args = ["v1compare", "unused.root", "--compare", "unused.root"]
    if emit:
        args += ["--emit", emit]
    cli.main(args)
