"""Whole-frame exact response admission and comparisons using synthetic assets."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import bounded_response as R, frame_registry as F, qualified_response as Q
from test_veto_qualified_response import domain, configs, calibration


def setup(monkeypatch, thermal=.01):
    r, q, _ = domain()
    cfg, wcfg = configs(monkeypatch)
    args = dict(registry_digest=F.digest(r), optical_seed=3027000, response_seed=4027000,
                arm="historical", pmap=np.ones((40, 40)), sipm_cfg=replace(cfg, thermal_rate=thermal),
                waveform_cfg=wcfg, nbars=1)
    return r, q, calibration(), args


def select(q, r):
    return q[pd.MultiIndex.from_frame(q[list(F.FRAME_KEYS)]).isin(pd.MultiIndex.from_frame(r[list(F.FRAME_KEYS)]))]


def test_all_stage_partition_subset_permutation_with_full_census(monkeypatch):
    r, q, c, args = setup(monkeypatch)
    full, info = R.run_arm(q, r, r, c, **args)
    parts = [R.run_arm(select(q, sub).iloc[::-1], r, sub.iloc[::-1], c.iloc[::-1], **args)[0]
             for sub in (r.iloc[[3, 1]], r.iloc[[0, 2]])]
    result = R.compare_partition(full, parts)
    assert result["all_physical_fields_exact"] and result["frames"] == 4
    assert set(result["stages"]) == set(R.STAGES)
    assert full["frames"].conditions.eq(4).all()
    assert info["live_channels"] == 16 and info["digis"] > 0 and info["pulses"] > 0
    assert info["execution_admitted"] is False
    assert info["pulse_fit_partition_exact_certified"] is False
    one = r.iloc[[0]]
    reduced, _ = R.run_arm(select(q, one), r, one, c, **args)
    assert R.compare_partition(full, [reduced])["frames"] == 1


@pytest.mark.parametrize("arm", ["historical", "arrival_only", "rejection_only", "stock_timing"])
def test_each_timing_arm_preserves_keys_and_complete_denominator(monkeypatch, arm):
    r, q, c, args = setup(monkeypatch, thermal=0.)
    q.loc[q.ordinal % 2 == 0, f"{arm}_kept"] = False
    full, info = R.run_arm(q, r, r, c, **{**args, "arm": arm})
    assert info["photons"] == len(q)//2
    assert len(full["frames"]) == 4 and len(full["phases"]) == 16
    assert set(full["candidates"].root) == set(q.loc[q[f"{arm}_kept"], "photon_id"])
    t = "time_historical_relative" if arm in ("historical", "rejection_only") else "time_raw_wrapped_relative"
    assert full["candidates"].time.eq(q[t].iloc[0]).all()


def test_quiet_frame_and_zero_live_frame_still_have_coverage(monkeypatch):
    r, q, c, args = setup(monkeypatch)
    q = q.iloc[:0]
    c.loc[c.input_sha256 == "b"*64, "status"] = 1
    out, info = R.run_arm(q, r, r, c, **args)
    assert len(out["frames"]) == 4 and info["photons"] == 0
    assert out["candidates"].dark.all()
    dark_frames = r[r.input_sha256 == "a"*64].frame_id
    assert set(out["candidates"].frame_id) == set(dark_frames)
    dead = out["frames"][out["frames"].input_sha256 == "b"*64]
    assert dead.conditions.eq(0).all() and dead.pulses.eq(0).all()


@pytest.mark.parametrize("case", ["digest", "shift", "metadata", "foreign", "seed", "root", "mask", "nan", "duplicate"])
def test_bad_bound_inputs(monkeypatch, case):
    r, q, c, args = setup(monkeypatch)
    selected = r.copy()
    if case == "digest":
        args["registry_digest"] = "0"*64
    elif case == "shift":
        selected["frame_id"] += 1
    elif case == "metadata":
        selected["intensity"] += 1
    elif case == "foreign":
        selected = r.iloc[1:]
    elif case == "seed":
        args["optical_seed"] += 1
    elif case == "root":
        q.loc[0, "photon_id"] += 1
    elif case == "mask":
        q["live_channel"] = False
    elif case == "nan":
        q["time_historical_relative"] = np.nan
    else:
        q = pd.concat([q, q.iloc[:1]], ignore_index=True)
    with pytest.raises((ValueError, AssertionError)):
        R.run_arm(q, r, selected, c, **args)


@pytest.mark.parametrize("limit", ["frames", "live_channels", "photons"])
def test_response_caps_precede_conditions_cross_join(monkeypatch, limit):
    r, q, c, args = setup(monkeypatch)
    monkeypatch.setattr(Q, "conditions", lambda *a, **k: pytest.fail("conditions allocation before cap"))
    with pytest.raises(ValueError, match="cap"):
        R.run_arm(q, r, r, c, limits=replace(R.Limits(), **{limit: 1}), **args)


def test_duplicate_partition_and_missing_stage_cannot_certify(monkeypatch):
    r, q, c, args = setup(monkeypatch, thermal=0.)
    out, _ = R.run_arm(q, r, r, c, **args)
    with pytest.raises(ValueError, match="duplicate"):
        R.compare_partition(out, [out, out])
    bad = dict(out)
    bad.pop("attempts")
    with pytest.raises(ValueError, match="stage"):
        R.compare_partition(out, [bad])
    bad = {k: t.copy() for k, t in out.items()}
    bad["pulses"].loc[0, "pe"] += 1e-10
    with pytest.raises(AssertionError):
        R.compare_partition(out, [bad])


@pytest.mark.parametrize("mutation", ["missing_field", "mask", "frame_count", "sample_deleted", "sample_tdc", "sample_identity"])
def test_symmetric_corruption_cannot_certify(monkeypatch, mutation):
    r, q, c, args = setup(monkeypatch)
    products, _ = R.run_arm(q, r, r, c, **args)
    bad = {k: t.copy() for k, t in products.items()}
    if mutation == "missing_field":
        bad["attempts"] = bad["attempts"].drop(columns=["p", "uniform"])
    elif mutation == "mask":
        bad["samples"].loc[bad["samples"].valid.idxmax(), "valid"] = False
    elif mutation == "frame_count":
        bad["frames"].loc[0, "samples"] += 1
    elif mutation == "sample_deleted":
        bad["fit_samples"] = bad["fit_samples"].iloc[1:]
    elif mutation == "sample_tdc":
        bad["samples"].loc[0, "tdc"] += 1
    else:
        bad["samples"].loc[0, "frame_id"] = 100
    with pytest.raises((ValueError, AssertionError)):
        R.compare_partition(bad, [bad])
