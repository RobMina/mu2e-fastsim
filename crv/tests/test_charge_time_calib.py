import numpy as np
import pytest

from crvfastsim.objective.charge_time_calib import fit


def test_weighted_profile_matches_dense_solve_with_missing_channel_ids():
    rng = np.random.default_rng(42)
    channel = np.repeat([2, 6, 11], [10, 17, 26])
    index = np.searchsorted([2, 6, 11], channel)
    d = rng.uniform(0, 1000, len(channel)) + 100 * index
    base_pe = rng.uniform(30, 70, len(channel))
    base_time = rng.normal(100, 3, len(channel))
    log_pe = 0.02 + np.array([-0.1, 0.02, 0.08])[index] - 0.0002 * d
    dt = 1.1 + np.array([-0.2, 0.0, 0.2])[index] + 0.0003 * d
    log_pe += rng.normal(0, 0.02, len(channel))
    dt += rng.normal(0, 0.2, len(channel))
    wc, wt = rng.uniform(0.5, 2, (2, len(channel)))
    got = fit(channel, d, base_pe * np.exp(log_pe), base_pe, base_time + dt, base_time,
              charge_weight=wc, time_weight=wt)
    design = np.column_stack((d, np.eye(3)[index]))
    for y, w, slope, g, local in (
        (log_pe, wc, -got["delta_attenuation_per_mm"], got["log_scale"], got["log_gain"]),
        (dt, wt, got["delta_slowness_ns_per_mm"], got["time_zero_ns"], got["time_offset_ns"]),
    ):
        coef = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
        np.testing.assert_allclose(np.r_[slope, g + np.asarray(local)], coef, atol=1e-12)
        assert abs(sum(local)) < 1e-12
    assert got["channel"] == [2, 6, 11]


def test_reject_confounded_distance_and_nonpositive_charge():
    ch = [1, 1, 2, 2]
    with pytest.raises(ValueError, match="within-channel"):
        fit(ch, [10, 10, 30, 30], [1] * 4, [1] * 4, [2] * 4, [2] * 4)
    with pytest.raises(ValueError, match="pe.*positive"):
        fit(ch, [10, 20, 30, 40], [0, 1, 1, 1], [1] * 4, [2] * 4, [2] * 4)
