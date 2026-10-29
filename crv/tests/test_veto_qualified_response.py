"""Full source-qualified identity bridge into unchanged diagnostic response helpers."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import frame_registry as F, qualified_response as R, keyed_optical_smoke as M
from crvfastsim.response import sipm as S
from crvfastsim.reco import waveform as W
from test_veto_frame_registry import events


def calibration():
    c = pd.DataFrame({"chan": np.arange(4), "pedestal": 100., "pulsearea": 1000., "pulseheight": 20.,
                      "timeoffset": 2., "status": 0, "roc": 1, "feb": 1})
    return pd.concat([c.assign(input_sha256="a"*64), c.assign(input_sha256="b"*64)], ignore_index=True)


def domain():
    ev = events()
    r = F.freeze(ev)
    q = ev.iloc[np.repeat(np.arange(len(ev)), 60)].reset_index(drop=True)
    q["draw_seed"] = np.uint64(3027000)
    q["source_row"] = np.repeat(np.arange(len(ev)), 60)
    q["chan"], q["substep"], q["photon_type"] = 0, 0, 0
    q["ordinal"] = np.tile(np.arange(60), len(ev))
    q["photon_id"] = M.roots(q)
    q["time_historical_relative"], q["time_raw_wrapped_relative"] = 500., 502.
    q["live_channel"] = True
    for arm in M.P.ARMS:
        q[f"{arm}_kept"] = True
    q = q.drop(columns=["pbtime", "intensity"])
    cond = R.conditions(r, ev, calibration(), F.digest(r), nbars=1)
    return r, q, cond


def configs(monkeypatch):
    cfg = replace(S.SiPMConfig(), p1=1., p2=.001, cross_talk=0.)
    wf = W.WaveformConfig()
    t = np.arange(0., wf.single_pe_max_time, wf.single_pe_precision)
    monkeypatch.setattr(W, "single_pe_template", lambda *a: .06*np.exp(-.5*((t-35.)/14.)**2))
    return cfg, wf


def test_all_live_channels_and_quiet_frame_conditions():
    r, q, cond = domain()
    assert len(cond) == 16
    assert len(cond.merge(q[list(F.FRAME_KEYS)].drop_duplicates(), on=list(F.FRAME_KEYS))) == 16
    assert cond.chan.nunique() == 4 and q.chan.nunique() == 1
    assert np.all(cond.w == -float(np.float32(-224.46)))
    dead = calibration()
    dead.loc[dead.chan == 1, "status"] = 1
    out = R.conditions(r, events(), dead, F.digest(r), nbars=1)
    assert len(out) == 12 and not (out.chan == 1).any()


@pytest.mark.parametrize("case", ["truncated", "duplicate", "missing", "nonfinite"])
def test_bad_calibration(case):
    r, _, _ = domain()
    c = calibration()
    if case == "truncated":
        c = c[c.chan < 3]
    elif case == "duplicate":
        c = pd.concat([c, c.iloc[:1]])
    elif case == "missing":
        c = c[c.input_sha256 == "a"*64]
    else:
        c.loc[0, "pedestal"] = np.nan
    with pytest.raises(ValueError):
        R.conditions(r, events(), c, F.digest(r), nbars=1)


def test_registry_digest_is_required_at_adapter_boundary():
    r, q, cond = domain()
    altered = r.copy()
    altered["intensity"] *= 2
    with pytest.raises(ValueError, match="digest"):
        R.admit(q, cond, altered, F.digest(r), 3027000)


@pytest.mark.parametrize("case", ["seed", "root", "event", "dead_mask", "missing_condition"])
def test_bad_saved_photon_admission(case):
    r, q, cond = domain()
    if case == "seed":
        q["draw_seed"] = 8
    elif case == "root":
        q.loc[0, "photon_id"] += 1
    elif case == "event":
        q["run"] = 99
        q["photon_id"] = M.roots(q)
    elif case == "dead_mask":
        q["live_channel"] = False
    else:
        cond = cond[cond.chan != 0]
    with pytest.raises((ValueError, AssertionError)):
        R.admit(q, cond, r, F.digest(r), 3027000)


def test_real_synthetic_all_stages_restore_full_frame_and_thermal_noise_keys(monkeypatch):
    r, q, cond = domain()
    cfg, wf = configs(monkeypatch)
    cfg = replace(cfg, thermal_rate=.01)
    args = (r, F.digest(r), 3027000, 4027000, "historical", np.ones((40, 40)), cfg, wf)
    products, info = R.run_arm(q, cond, *args)
    assert info["photons"] == len(q) and info["pulses"] > 0 and info["digis"] > 0
    assert info["dark_candidates"] > 0
    for name, frame in products.items():
        assert set((*F.FRAME_KEYS, "frame_id")) <= set(frame), name
        assert pd.MultiIndex.from_frame(frame[list(F.FRAME_KEYS)]).isin(pd.MultiIndex.from_frame(r[list(F.FRAME_KEYS)])).all()
    assert products["phases"].frame_id.nunique() == 4
    assert products["phases"].groupby("frame_id").phase.nunique().eq(1).all()
    assert products["phases"].phase.nunique() == 4
    one = r.iloc[[0]]
    select = pd.MultiIndex.from_frame(q[list(F.FRAME_KEYS)]).isin(pd.MultiIndex.from_frame(one[list(F.FRAME_KEYS)]))
    cond_select = pd.MultiIndex.from_frame(cond[list(F.FRAME_KEYS)]).isin(pd.MultiIndex.from_frame(one[list(F.FRAME_KEYS)]))
    reduced, _ = R.run_arm(q[select].iloc[::-1], cond[cond_select].iloc[::-1], *args)
    stats = R.compare_partition(products, reduced, r, F.digest(r))
    assert stats["charges"]["exact"] and stats["digis"]["exact"]
    assert stats["pulses"]["only_a"] == stats["pulses"]["only_b"] == 0
    print("QUALIFIED_PARTITION_COMPARISON", stats)
    keys = {"charges": ["frame_id", "chan", "root", "dark", "generation", "px", "py"],
            "candidates": ["frame_id", "chan", "root"], "digis": ["frame_id", "chan", "starttdc", "sample"],
            "phases": ["frame_id", "chan"]}
    for name, identity in keys.items():
        expected = products[name][products[name].frame_id == 0].drop(columns=["wf"], errors="ignore")
        actual = reduced[name].drop(columns=["wf"], errors="ignore")
        pd.testing.assert_frame_equal(actual.sort_values(identity).reset_index(drop=True),
                                      expected.sort_values(identity).reset_index(drop=True), check_exact=True)
    assert info["pulse_fit_partition_exact_certified"] is False
    identity = ["frame_id", "chan", "starttdc", "seq"]
    expected = products["pulses"][products["pulses"].frame_id == 0]
    pd.testing.assert_frame_equal(reduced["pulses"][identity].sort_values(identity).reset_index(drop=True),
                                  expected[identity].sort_values(identity).reset_index(drop=True), check_exact=True)


def test_zero_optical_frame_still_gets_thermal_and_identity(monkeypatch):
    r, q, cond = domain()
    cfg, wf = configs(monkeypatch)
    products, info = R.run_arm(q.iloc[:0], cond, r, F.digest(r), 3027000, 4027000,
                              "stock_timing", np.ones((40, 40)), replace(cfg, thermal_rate=.01), wf)
    assert info["photons"] == 0 and info["dark_candidates"] > 0
    assert products["candidates"].dark.all()
    assert products["candidates"].frame_id.nunique() == 4


def test_false_live_mask_cannot_silently_remove_channel():
    r, q, cond = domain()
    q["live_channel"] = False
    for arm in M.P.ARMS:
        q[f"{arm}_kept"] = False
    with pytest.raises(ValueError, match="live mask"):
        R.admit(q, cond, r, F.digest(r), 3027000)
