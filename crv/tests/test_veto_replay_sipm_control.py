"""Keyed SiPM diagnostic contracts; no production RNG or response patching."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from crvfastsim.response.sipm import SiPMConfig
from crvfastsim.veto import replay_sipm_control as S


CFG = SiPMConfig(cross_talk=0)
CERTAIN = replace(CFG, p1=1, p2=0.001)


def candidates(rows=None):
    if rows is None:
        rows = [(1, 4, 0, 200., 0, 0, False), (1, 4, 1, 201., 2, 2, False),
                (1, 4, 2, 210., 0, 0, False), (2, 5, 0, -20., 9, 9, False)]
    return pd.DataFrame(rows, columns=S._CANDIDATE_COLUMNS)


def photons(n=100):
    return pd.DataFrame(dict(event=np.zeros(n, dtype=int), chan=np.zeros(n, dtype=int),
                             photon_id=np.arange(n), time=np.arange(n, dtype=float)))


def test_hash_broadcast_repeat_domain_order_seed_and_normal():
    keys = np.arange(10000)
    u = S.keyed_uniform(42, "check", keys)
    np.testing.assert_array_equal(u, S.keyed_uniform(42, "check", keys))
    assert np.all((u >= 0) & (u < 1))
    assert abs(u.mean() - 0.5) < 0.015
    assert len(np.unique(u)) == len(u)
    assert not np.array_equal(u, S.keyed_uniform(43, "check", keys))
    assert not np.array_equal(u, S.keyed_uniform(42, "other", keys))
    b = S.keyed_uniform(-7, "broadcast", keys[:4, None], np.arange(3))
    assert b.shape == (4, 3)
    assert b[2, 1] == S.keyed_uniform(-7, "broadcast", 2, 1)
    assert S.keyed_uniform(0, "order", 1, 2) != S.keyed_uniform(0, "order", 2, 1)
    n = S.keyed_normal(42, "normal", keys)
    assert np.isfinite(n).all() and abs(n.mean()) < .04 and abs(n.std() - 1) < .04
    assert S.HASH_VERSION == "sipm-keyed-v1"


@pytest.mark.parametrize("seed,domain,keys", [(1.2, "a", [1]), (True, "a", [1]),
                                               (1, 3, [1]), (1, "a", [1.0]), (1, "a", [np.nan])])
def test_hash_rejects_noninteger_keys(seed, domain, keys):
    with pytest.raises(ValueError):
        S.keyed_uniform(seed, domain, np.asarray(keys))


def test_permutation_channel_subset_and_root_subset():
    c = candidates()
    baseline = S.simulate_candidates(c, CFG, 9)
    permuted = S.simulate_candidates(c.sample(frac=1, random_state=41), CFG, 9)
    for left, right in zip(baseline, permuted):
        assert_frame_equal(left, right)
    for mask in (c.chan == 4, c.root != 1):
        reduced = S.simulate_candidates(c[mask], CFG, 9)
        for full, small in zip(baseline, reduced):
            keep = full.chan == 4 if mask.equals(c.chan == 4) else full.root != 1
            assert_frame_equal(full[keep].reset_index(drop=True), small)


def test_ct_permutation_and_unrelated_channel_subset():
    c = candidates()
    cfg = replace(CERTAIN, cross_talk=.8)
    charges, attempts = S.simulate_candidates(c, cfg, 15)
    assert (attempts.generation > 0).any()
    for full, reduced in zip((charges, attempts), S.simulate_candidates(c[c.chan == 4], cfg, 15)):
        assert_frame_equal(full[full.chan == 4].reset_index(drop=True), reduced)
    for original, permuted in zip((charges, attempts), S.simulate_candidates(c.iloc[::-1], cfg, 15)):
        assert_frame_equal(original, permuted)


def test_dark_insertion_different_pixel_does_not_shift_optical_rng():
    c = candidates([(1, 2, 0, 201., 1, 1, False)])
    dark = candidates([(1, 2, -1, 200., 8, 8, True)])
    for off, on in zip(S.simulate_candidates(c, CERTAIN, 22),
                       S.simulate_candidates(pd.concat([c, dark]), CERTAIN, 22)):
        assert_frame_equal(off, on[~on.dark].reset_index(drop=True))


def test_same_pixel_thermal_recovery_changes_charge_not_uniform():
    c = candidates([(1, 2, 0, 201., 1, 1, False)])
    dark = candidates([(1, 2, -1, 200., 1, 1, True)])
    _, off = S.simulate_candidates(c, CERTAIN, 22)
    _, on = S.simulate_candidates(pd.concat([c, dark]), CERTAIN, 22)
    optical = on[~on.dark].iloc[0]
    assert optical.uniform == off.uniform.iloc[0]
    assert optical.v == pytest.approx(CERTAIN.overvoltage * (1 - np.exp(-1 / CERTAIN.time_constant)))
    assert 0 < optical.charge < off.charge.iloc[0]
    assert on.iloc[0].dark


def test_inactive_no_lower_cut_and_inclusive_end():
    c = candidates([(0, 0, 0, -100., 0, 0, False), (0, 0, 1, 1500., 1, 1, False),
                    (0, 0, 2, np.nextafter(1500., np.inf), 2, 2, False),
                    (0, 0, 3, 200., 18, 18, False)])
    charges, attempts = S.simulate_candidates(c, CERTAIN, 3)
    assert set(charges.root) == {0, 1}
    assert set(attempts.root) == {0, 1, 3}
    dead = attempts[attempts.root == 3].iloc[0]
    assert dead.inactive and not dead.fired and dead.p == 0 and dead.charge == 0


def test_zero_ct_analytic_recovery_and_same_time_tie():
    c = candidates([(0, 0, 2, 100., 1, 1, False), (0, 0, 0, 100., 1, 1, False),
                    (0, 0, 1, 100. + CERTAIN.time_constant, 1, 1, False)])
    charges, a = S.simulate_candidates(c, CERTAIN, 1)
    assert list(a.root) == [0, 2, 1]
    np.testing.assert_allclose(a.v, [3, 0, 3 * (1 - np.exp(-1))])
    np.testing.assert_allclose(a.pe, [1, 0, 1 - np.exp(-1)])
    assert list(a.fired) == [True, False, True]
    assert len(charges) == 2 and a.primary.all() and (a.multiplicity == 1).all()
    assert (a.uniform < a.p).equals(a.fired)
    assert (a.root_px == a.px).all()
    np.testing.assert_allclose(a.charge, a.pe * CERTAIN.capacitance * CERTAIN.overvoltage)


def test_zero_avalanche_probability():
    charges, attempts = S.simulate_candidates(candidates(), replace(CFG, p1=0), 2)
    assert charges.empty and len(attempts) == 4
    assert not attempts.fired.any() and (attempts.p == 0).all()


def test_grouped_ct_multiplicity_and_ancestry():
    c = candidates([(0, 0, 42, 100., 0, 0, False)])
    charges, a = S.simulate_candidates(c, replace(CERTAIN, cross_talk=1), 11)
    grouped = a[(a.generation == 2) & (a.px == 1) & (a.py == 1)].iloc[0]
    assert grouped.multiplicity == 2 and grouped.fired
    assert (a.root == 42).all() and (a.root_px == 0).all() and (a.root_py == 0).all()
    assert (a.root_time == 100).all() and (a.photon == 42).all()
    assert not charges.duplicated(["px", "py"]).any()
    assert len(charges) == 40 * 40 - len(CFG.inactive)
    assert not a[a.generation > 0].primary.any()


def test_fail_closed_attempt_limit(monkeypatch):
    monkeypatch.setattr(S, "MAX_ATTEMPTS", 2)
    with pytest.raises(RuntimeError, match="limit"):
        S.simulate_candidates(candidates(), CERTAIN, 1)
    with pytest.raises(RuntimeError, match="limit"):
        S.simulate_candidates(candidates().iloc[:1], replace(CERTAIN, cross_talk=1), 1)
    with pytest.raises(RuntimeError, match="limit"):
        S.thermal_candidates(pd.DataFrame(dict(event=[0], chan=[0])), replace(CFG, thermal_rate=1), 1)


def test_pixel_policy_range_and_common_bins():
    p = photons(10000)
    weights = np.zeros((40, 40)); weights[39, 39] = 1
    fixed = S.optical_candidates(p, weights, CFG, 42, "fixed_integer")
    legacy = S.optical_candidates(p, weights, CFG, 42, "legacy_round")
    assert (fixed.px == 39).all() and (fixed.py == 39).all()
    assert set(legacy.px) == {39, 40} and set(legacy.py) == {39, 40}
    assert (legacy.px == 40).mean() == pytest.approx(.5, abs=.02)
    assert not fixed.dark.any() and np.array_equal(fixed.root, p.photon_id)
    weights = np.ones((40, 40))
    fixed = S.optical_candidates(p, weights, CFG, 42, "fixed_integer")
    legacy = S.optical_candidates(p, weights, CFG, 42, "legacy_round")
    assert ((legacy[["px", "py"]] - fixed[["px", "py"]]).isin([0, 1])).all().all()
    perm = S.optical_candidates(p.iloc[::-1], weights, CFG, 42)
    assert_frame_equal(fixed, perm.iloc[::-1].reset_index(drop=True))
    assert_frame_equal(fixed.iloc[500:].reset_index(drop=True), S.optical_candidates(p.iloc[500:], weights, CFG, 42))


def test_basic_bin_distribution():
    weights = np.zeros((40, 40)); weights[1, 2] = 1; weights[4, 5] = 3
    out = S.optical_candidates(photons(30000), weights, CFG, 123)
    assert set(zip(out.px, out.py)) == {(1, 2), (4, 5)}
    assert (out.px == 1).mean() == pytest.approx(.25, abs=.015)


def test_thermal_poisson_quiet_channels_identity_and_invariance():
    channels = pd.DataFrame(dict(event=np.arange(20000), chan=np.zeros(20000, dtype=int)))
    cfg = replace(CFG, thermal_rate=.4 * CFG.p_fire(CFG.overvoltage) / 1400)
    dark = S.thermal_candidates(channels, cfg, 314)
    assert len(dark) / len(channels) == pytest.approx(.4, abs=.02)
    assert dark.event.nunique() / len(channels) == pytest.approx(1 - np.exp(-.4), abs=.02)
    assert dark.dark.all() and (dark.root < 0).all()
    assert dark.time.between(100, 1500, inclusive="left").all()
    assert dark.px.between(0, 39).all() and dark.py.between(0, 39).all()
    assert not dark.duplicated(["event", "chan", "root"]).any()
    assert_frame_equal(dark, S.thermal_candidates(channels.iloc[::-1], cfg, 314))
    subset = S.thermal_candidates(channels.iloc[:100], cfg, 314)
    assert_frame_equal(dark[dark.event < 100].reset_index(drop=True), subset)
    assert len(dark) > 0


def test_thermal_no_attempt_equals_off_and_zero_window():
    channels = pd.DataFrame(dict(event=[0], chan=[0]))
    cfg = replace(CFG, thermal_rate=np.nextafter(0., 1.))
    dark = S.thermal_candidates(channels, cfg, 1)
    assert dark.empty
    for off, on in zip(S.simulate_candidates(candidates(), CFG, 4),
                       S.simulate_candidates(pd.concat([candidates(), dark]), cfg, 4)):
        assert_frame_equal(off, on)
    assert S.thermal_candidates(channels, CFG, 1).empty
    assert S.thermal_candidates(channels, replace(CFG, thermal_rate=1), 1, start=100, end=100).empty


def test_empty_schemas_and_no_global_rng_consumption():
    state = np.random.get_state()
    charges, attempts = S.simulate_candidates(candidates([]), CFG, 1)
    assert charges.empty and attempts.empty
    assert list(attempts) == S._ATTEMPT_COLUMNS
    empty = S.optical_candidates(photons(0), np.ones((40, 40)), CFG, 1)
    assert empty.empty and list(empty) == S._CANDIDATE_COLUMNS
    assert S.thermal_candidates(pd.DataFrame(columns=["event", "chan"]), CFG, 1).empty
    S.simulate_candidates(candidates(), SiPMConfig(), 1)
    S.thermal_candidates(pd.DataFrame(dict(event=[0], chan=[0])), replace(CFG, thermal_rate=.001), 1)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize("field,bad", [("trap0_prob", .1), ("trap1_prob", .1), ("p1", 1.1),
    ("p1", np.nan), ("p2", 0), ("overvoltage", np.inf), ("time_constant", 0), ("capacitance", -1),
    ("thermal_rate", -1), ("thermal_rate", np.inf), ("cross_talk", 1.1), ("cross_talk", 1),
    ("n_pixels_x", 39), ("n_pixels_y", 41), ("inactive", ((40, 0),)), ("inactive", ((.5, 0),))])
def test_illegal_config(field, bad):
    with pytest.raises(ValueError):
        S.simulate_candidates(candidates(), replace(CFG, **{field: bad}), 1)


@pytest.mark.parametrize("field,bad", [("time", np.nan), ("time", np.inf), ("px", -1), ("py", 41),
    ("root", .5), ("root", -1), ("event", -1), ("chan", -1), ("dark", 2), ("chan", np.inf)])
def test_illegal_candidates(field, bad):
    c = candidates().astype({field: float})
    c.loc[0, field] = bad
    with pytest.raises(ValueError):
        S.simulate_candidates(c, CFG, 1)


def test_duplicate_missing_and_nonfinite_window():
    with pytest.raises(ValueError, match="duplicate"):
        S.simulate_candidates(pd.concat([candidates(), candidates()]), CFG, 1)
    with pytest.raises(ValueError, match="missing"):
        S.simulate_candidates(candidates().drop(columns="px"), CFG, 1)
    for end in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            S.simulate_candidates(candidates(), CFG, 1, end=end)
        with pytest.raises(ValueError):
            S.thermal_candidates(pd.DataFrame(dict(event=[0], chan=[0])), CFG, 1, end=end)
    with pytest.raises(ValueError):
        S.thermal_candidates(pd.DataFrame(dict(event=[0], chan=[0])), CFG, 1, start=200, end=100)
    with pytest.raises(ValueError):
        S.thermal_candidates(pd.DataFrame(dict(event=[0, 0], chan=[0, 0])), CFG, 1)
    with pytest.raises(ValueError):
        S.thermal_candidates(pd.DataFrame(dict(event=[0], chan=[0])), replace(CFG, p1=0, thermal_rate=.1), 1)


@pytest.mark.parametrize("weights", [np.zeros((40, 40)), np.ones((39, 40)),
    np.full((40, 40), -1), np.full((40, 40), np.nan), np.full((40, 40), np.inf)])
def test_bad_map(weights):
    with pytest.raises(ValueError):
        S.optical_candidates(photons(), weights, CFG, 1)


def test_optical_invalid_policy_and_identity():
    with pytest.raises(ValueError):
        S.optical_candidates(photons(), np.ones((40, 40)), CFG, 1, "unknown")
    p = photons(); p.loc[1, "photon_id"] = 0
    with pytest.raises(ValueError):
        S.optical_candidates(p, np.ones((40, 40)), CFG, 1)
    p.loc[1, "photon_id"] = -1
    with pytest.raises(ValueError):
        S.optical_candidates(p, np.ones((40, 40)), CFG, 1)


def test_hash_v1_golden_vector_and_integer_seed_types():
    keys = np.array([-1, 0, 1, 2**63 - 1], dtype=np.int64)
    expected = [float.fromhex(x) for x in [
        "0x1.dbc1cb9f35efbp-1", "0x1.c371685ff669cp-1",
        "0x1.baf35120e95b0p-3", "0x1.70a423f6f5b9fp-1"]]
    np.testing.assert_array_equal(S.keyed_uniform(42, "golden", keys), expected)
    np.testing.assert_array_equal(S.keyed_uniform(np.int64(42), "golden", keys), expected)
    np.testing.assert_array_equal(S.keyed_uniform(np.uint64(42), "golden", keys), expected)


def test_int64_root_extremes_and_unsigned_overflow():
    c = candidates([(0, 0, 2**63 - 1, 1., 1, 1, False),
                    (0, 0, -(2**63), 2., 1, 1, True)])
    _, a = S.simulate_candidates(c, CERTAIN, 1)
    assert list(a.root) == [2**63 - 1, -(2**63)]
    c = candidates().iloc[:1].copy()
    c["root"] = np.array([2**63], dtype=np.uint64)
    with pytest.raises(ValueError, match="int64"):
        S.simulate_candidates(c, CFG, 1)
    c["root"] = float(2**63)
    with pytest.raises(ValueError, match="int64"):
        S.simulate_candidates(c, CFG, 1)


def test_legacy_pixel_40_supported_and_all_late_empty():
    c = candidates([(0, 0, 0, 1500., 40, 40, False)])
    charges, a = S.simulate_candidates(c, CERTAIN, 1)
    assert len(charges) == 1 and a.px.iloc[0] == 40
    charges, a = S.simulate_candidates(c, CERTAIN, 1, end=1499.)
    assert charges.empty and a.empty


def test_nontrivial_grouped_probability_formula():
    cfg = replace(CERTAIN, p1=.5, cross_talk=1 - .5**4)
    c = candidates([(i, 0, 0, 100., 5, 5, False) for i in range(50)])
    _, a = S.simulate_candidates(c, cfg, 777)
    live = a[~a.inactive]
    np.testing.assert_allclose(live.p, 1 - (1 - cfg.p_fire(live.v))**live.multiplicity)
    grouped = live[(live.multiplicity > 1) & (live.v > 0)]
    assert len(grouped) > 0
    assert ((grouped.p > 0) & (grouped.p < 1)).all()


def test_pixel_policy_translation_preserves_primary_and_cascade_variates():
    c = candidates([(i, 0, 0, 100., 5, 5, False) for i in range(200)])
    translated = c.assign(px=6, py=7)
    for cfg in (replace(CFG, cross_talk=0, inactive=()), replace(CFG, inactive=())):
        _, a = S.simulate_candidates(c, cfg, 777)
        _, b = S.simulate_candidates(translated, cfg, 777)
        for frame in (a, b):
            frame["dx"] = frame.px - frame.root_px
            frame["dy"] = frame.py - frame.root_py
        keys = ["event", "chan", "root", "generation", "dx", "dy"]
        cols = keys + ["uniform", "p", "fired", "pe"]
        assert_frame_equal(a[cols].sort_values(keys).reset_index(drop=True), b[cols].sort_values(keys).reset_index(drop=True))


def test_dark_and_optical_key_domains_and_time_interactions():
    optical = candidates([(0, 0, 0, 100., 5, 5, False)])
    dark = optical.assign(dark=True)
    _, a = S.simulate_candidates(optical, CERTAIN, 7)
    _, d = S.simulate_candidates(dark, CERTAIN, 7)
    assert a.uniform.iloc[0] != d.uniform.iloc[0]
    late_dark = candidates([(0, 0, -1, 200., 5, 5, True)])
    _, combined = S.simulate_candidates(pd.concat([late_dark, optical]), CERTAIN, 7)
    assert_frame_equal(a, combined[~combined.dark].reset_index(drop=True))
