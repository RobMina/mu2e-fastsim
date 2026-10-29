"""Exact frozen-helper parity and fail-before-allocation diagnostic admission."""
from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from crvfastsim.response.sipm import SiPMConfig
from crvfastsim.veto import bounded_sipm as B
from crvfastsim.veto import replay_sipm_control as S


CFG = SiPMConfig(cross_talk=0)
CERTAIN = replace(CFG, p1=1, p2=.001)
STATE = 41 * 41 * 8


def candidates(rows=None):
    if rows is None:
        rows = [(1, 4, 0, 200., 0, 0, False), (1, 4, 1, 201., 2, 2, False),
                (1, 4, 2, 210., 0, 0, False), (2, 5, 0, -20., 9, 9, False)]
    return pd.DataFrame(rows, columns=S._CANDIDATE_COLUMNS)


def photons(n=100):
    return pd.DataFrame(dict(event=np.arange(n) % 3, chan=np.arange(n) % 7,
                             photon_id=np.arange(n), time=np.arange(n, dtype=float)))


def exact_products(left, right):
    for a, b in zip(left, right):
        assert_frame_equal(a, b, check_exact=True)
        assert list(a) == S._ATTEMPT_COLUMNS
        assert np.isfinite(a.select_dtypes(include="number").to_numpy()).all()


def proxy(module, **overrides):
    return SimpleNamespace(**{**vars(module), **overrides})


def forbidden(*args, **kwargs):
    pytest.fail("allocation/operation reached before admission")


def test_limits_frozen_defaults_and_explicit_call_validation():
    limits = B.Limits()
    assert limits.photons == 300_000 and limits.attempts == 1_000_000
    assert limits.live_channels == 4 * 5120
    assert limits.live_channels * STATE <= limits.state_bytes <= 300 * 1024**2
    with pytest.raises(FrozenInstanceError):
        limits.photons = 1
    zero = B.Limits(**{field.name: 0 for field in fields(B.Limits)})
    zero.validate()
    for function, args in ((B.optical_candidates, (photons(0), np.ones((40, 40)), CFG, 1)),
                           (B.thermal_candidates, (pd.DataFrame(columns=["event", "chan"]), CFG, 1)),
                           (B.simulate_candidates, (candidates([]), CFG, 1))):
        with pytest.raises(ValueError, match="Limits"):
            function(*args, limits={})


@pytest.mark.parametrize("field", [field.name for field in fields(B.Limits)])
@pytest.mark.parametrize("bad", [-1, True, np.bool_(False), 1.5, np.nan, np.inf, "3", None,
                                  np.int64(4), np.uint64(4), 2**100])
def test_malformed_limits(field, bad):
    with pytest.raises(ValueError, match="resource limit"):
        B.Limits(**{field: bad})


@pytest.mark.parametrize("field,bad", [("photons", 10_000_001), ("thermal_candidates", 10_000_001),
                                      ("candidates", 10_000_001), ("attempts", 10_000_001),
                                      ("generations", 1683), ("live_channels", 10_000_001),
                                      ("state_bytes", int(np.iinfo(np.intp).max) + 1)])
def test_limits_machine_safe_upper_bounds(field, bad):
    with pytest.raises(ValueError, match=field):
        B.Limits(**{field: bad})


@pytest.mark.parametrize("policy", ["fixed_integer", "legacy_round"])
def test_optical_exact_mapping_order_schema_and_limits(policy):
    p = photons(1200).sample(frac=1, random_state=12)
    weights = np.arange(1600).reshape(40, 40)
    before = p.copy(deep=True)
    actual = B.optical_candidates(p, weights, CFG, 42, policy,
                                  limits=B.Limits(photons=len(p), candidates=len(p)))
    assert_frame_equal(actual, S.optical_candidates(p, weights, CFG, 42, policy), check_exact=True)
    assert_frame_equal(p, before, check_exact=True)
    assert not actual.dark.any() and (actual.root >= 0).all()
    assert np.isfinite(actual.select_dtypes(include="number").to_numpy()).all()


@pytest.mark.parametrize("limit", ["photons", "candidates"])
def test_optical_rejected_before_frame_copy_or_mapping(monkeypatch, limit):
    monkeypatch.setattr(B, "S", proxy(S, _frame=forbidden, optical_candidates=forbidden))
    with pytest.raises(RuntimeError, match=limit):
        B.optical_candidates(photons(3), np.ones((40, 40)), CFG, 1, limits=B.Limits(**{limit: 2}))


def test_optical_live_channel_limit_before_mapping(monkeypatch):
    monkeypatch.setattr(B, "S", proxy(S, optical_candidates=forbidden))
    with pytest.raises(RuntimeError, match="live_channels"):
        B.optical_candidates(photons(3), np.ones((40, 40)), CFG, 1, limits=B.Limits(live_channels=2))


@pytest.mark.parametrize("seed", [1, 3, 314])
def test_thermal_exact_poisson_order_negative_roots_and_subset(seed):
    channels = pd.DataFrame(dict(event=np.arange(1000), chan=np.arange(1000) % 17))
    cfg = replace(CFG, thermal_rate=.8 * CFG.p_fire(CFG.overvoltage) / 1400)
    actual = B.thermal_candidates(channels, cfg, seed)
    assert_frame_equal(actual, S.thermal_candidates(channels, cfg, seed), check_exact=True)
    assert actual.dark.all() and (actual.root < 0).all()
    assert actual.time.between(100., 1500., inclusive="left").all()
    assert not actual.duplicated(["event", "chan", "root"]).any()
    assert_frame_equal(actual, B.thermal_candidates(channels.iloc[::-1], cfg, seed), check_exact=True)
    subset = B.thermal_candidates(channels[channels.event < 100], cfg, seed)
    assert_frame_equal(actual[actual.event < 100].reset_index(drop=True), subset, check_exact=True)


def test_thermal_live_limit_before_channel_copy(monkeypatch):
    monkeypatch.setattr(B, "S", proxy(S, _frame=forbidden))
    with pytest.raises(RuntimeError, match="live_channels"):
        B.thermal_candidates(pd.DataFrame(dict(event=[1, 2], chan=[0, 0])), CFG, 1,
                             limits=B.Limits(live_channels=1))


def test_thermal_expected_cap_before_wait_allocations(monkeypatch):
    channels = pd.DataFrame(dict(event=[0], chan=[0]))
    cfg = replace(CFG, thermal_rate=2 * CFG.p_fire(CFG.overvoltage) / 1400)
    monkeypatch.setattr(B, "np", proxy(np, arange=forbidden, zeros=forbidden, full=forbidden))
    with pytest.raises(RuntimeError, match="thermal_candidates expected"):
        B.thermal_candidates(channels, cfg, 3, limits=B.Limits(thermal_candidates=1))


@pytest.mark.parametrize("limit", ["thermal_candidates", "candidates", "attempts"])
def test_thermal_actual_append_cap_not_only_expectation(monkeypatch, limit):
    channels = pd.DataFrame(dict(event=[0], chan=[0]))
    cfg = replace(CFG, thermal_rate=CFG.p_fire(CFG.overvoltage) / 1400)
    assert len(S.thermal_candidates(channels, cfg, 3)) == 3
    appended = []

    def full(shape, value, *args, **kwargs):
        appended.append((shape, value))
        assert len(appended) == 1
        return np.full(shape, value, *args, **kwargs)

    monkeypatch.setattr(B, "np", proxy(np, full=full, concatenate=forbidden))
    with pytest.raises(RuntimeError, match="thermal_candidates actual"):
        B.thermal_candidates(channels, cfg, 3, limits=B.Limits(**{limit: 1}))
    assert appended == [(1, -1)]


@pytest.mark.parametrize("cfg", [CFG, CERTAIN, replace(CERTAIN, cross_talk=.8),
                                  replace(CFG, p1=0)])
def test_recovery_exact_unpermuted_permuted_and_unrelated_subset(cfg):
    c = candidates()
    before = c.copy(deep=True)
    full = B.simulate_candidates(c, cfg, 15)
    exact_products(full, S.simulate_candidates(c, cfg, 15))
    exact_products(full, B.simulate_candidates(c.iloc[::-1], cfg, 15))
    small = B.simulate_candidates(c[c.chan == 4], cfg, 15)
    exact_products(tuple(a[a.chan == 4].reset_index(drop=True) for a in full), small)
    assert_frame_equal(c, before, check_exact=True)


def test_full_cascade_grouping_inactive_and_ancestry_exact():
    c = candidates([(0, 0, 42, 100., 0, 0, False)])
    cfg = replace(CERTAIN, cross_talk=1)
    charges, attempts = B.simulate_candidates(c, cfg, 11)
    exact_products((charges, attempts), S.simulate_candidates(c, cfg, 11))
    assert len(charges) == 40 * 40 - len(cfg.inactive)
    assert attempts.generation.max() > 10
    assert attempts.multiplicity.max() > 1
    assert (attempts.root == 42).all() and (attempts.photon == 42).all()
    assert attempts.inactive.any() and not attempts[attempts.inactive].fired.any()


def test_fixed_integer_plus_thermal_complete_chain_exact():
    p = photons(200)
    weights = np.ones((40, 40))
    channels = pd.DataFrame(dict(event=np.repeat(np.arange(4), 9), chan=np.tile(np.arange(9), 4)))
    cfg = replace(CERTAIN, thermal_rate=.002, cross_talk=.8)
    optical = B.optical_candidates(p, weights, cfg, 35)
    dark = B.thermal_candidates(channels, cfg, 35)
    assert_frame_equal(optical, S.optical_candidates(p, weights, cfg, 35), check_exact=True)
    assert_frame_equal(dark, S.thermal_candidates(channels, cfg, 35), check_exact=True)
    c = pd.concat([optical, dark], ignore_index=True)
    exact_products(B.simulate_candidates(c, cfg, 35), S.simulate_candidates(c, cfg, 35))
    assert (dark.event == 3).any()


def test_same_pixel_dark_recovery_one_unsplit_history_and_ties():
    c = candidates([(1, 2, 0, 201., 1, 1, False), (1, 2, -1, 200., 1, 1, True),
                    (1, 2, 1, 201., 1, 1, False)])
    full = B.simulate_candidates(c, CERTAIN, 22)
    exact_products(full, S.simulate_candidates(c, CERTAIN, 22))
    _, attempts = full
    assert list(attempts.root) == [-1, 0, 1]
    assert 0 < attempts.pe.iloc[1] < 1
    assert attempts.pe.iloc[2] == 0
    with pytest.raises(RuntimeError, match="candidates"):
        B.simulate_candidates(c, CERTAIN, 22, limits=B.Limits(candidates=2))


def test_inclusive_end_negative_time_padded_pixel_and_int64_roots():
    c = candidates([(0, 0, 2**63 - 1, -100., 40, 40, False),
                    (0, 0, -(2**63), 1500., 1, 1, True),
                    (0, 0, 0, np.nextafter(1500., np.inf), 1, 1, False)])
    exact_products(B.simulate_candidates(c, CERTAIN, 1), S.simulate_candidates(c, CERTAIN, 1))
    _, attempts = B.simulate_candidates(c, CERTAIN, 1)
    assert list(attempts.root) == [2**63 - 1, -(2**63)]


def test_candidate_limit_before_validation_copy_even_for_late_input(monkeypatch):
    c = candidates().assign(time=1600.)
    monkeypatch.setattr(B, "S", proxy(S, _frame=forbidden, _config=forbidden))
    with pytest.raises(RuntimeError, match="candidates"):
        B.simulate_candidates(c, CERTAIN, 1, limits=B.Limits(candidates=3))


@pytest.mark.parametrize("field,cap", [("state_bytes", 2 * STATE - 1), ("live_channels", 1),
                                      ("generations", 0), ("attempts", 3), ("photons", 3)])
def test_simulation_preallocation_guards(monkeypatch, field, cap):
    monkeypatch.setattr(B, "np", proxy(np, full=forbidden))
    with pytest.raises(RuntimeError, match=field):
        B.simulate_candidates(candidates(), CERTAIN, 1, limits=B.Limits(**{field: cap}))


def test_thermal_candidate_input_limit_before_state(monkeypatch):
    monkeypatch.setattr(B, "np", proxy(np, full=forbidden))
    with pytest.raises(RuntimeError, match="thermal_candidates"):
        B.simulate_candidates(candidates().assign(dark=True, root=lambda x: -1 - x.root), CFG, 1,
                              limits=B.Limits(thermal_candidates=3))


def test_state_exact_boundary_and_only_active_event_channel_pairs(monkeypatch):
    c = candidates([(0, 7, 0, 200., 1, 1, False), (1, 7, 0, 200., 1, 1, False),
                    (9, 8, 0, 1501., 1, 1, False)])
    allocations = []

    def full(shape, value, *args, **kwargs):
        if isinstance(shape, tuple):
            allocations.append(shape)
        return np.full(shape, value, *args, **kwargs)

    monkeypatch.setattr(B, "np", proxy(np, full=full))
    actual = B.simulate_candidates(c, CERTAIN, 1, limits=B.Limits(state_bytes=2 * STATE, live_channels=2))
    exact_products(actual, S.simulate_candidates(c, CERTAIN, 1))
    assert allocations == [(2, 41 * 41)]


@pytest.mark.parametrize("field,cap", [("attempts", 4), ("generations", 1)])
def test_four_neighbour_guard_before_expansion(monkeypatch, field, cap):
    monkeypatch.setattr(B, "_neighbours", forbidden)
    c = candidates([(0, 0, 0, 100., 0, 0, False)])
    with pytest.raises(RuntimeError, match=field + " neighbour upper"):
        B.simulate_candidates(c, replace(CERTAIN, cross_talk=1), 1, limits=B.Limits(**{field: cap}))


def test_total_ledger_guard_across_root_ranks_before_append(monkeypatch):
    first = candidates([(0, 0, 0, 100., 0, 0, False)])
    cfg = replace(CERTAIN, cross_talk=.8)
    assert len(S.simulate_candidates(first, cfg, 2)[1]) == 3
    rest = candidates([(0, 0, i, 200. + i, 18, 18, False) for i in range(1, 6)])
    c = pd.concat([first, rest], ignore_index=True)
    appended = []

    def dataframe(data, *args, **kwargs):
        appended.append(len(data["event"]))
        assert sum(appended) <= 6
        return pd.DataFrame(data, *args, **kwargs)

    monkeypatch.setattr(B, "pd", proxy(pd, DataFrame=dataframe, concat=forbidden))
    with pytest.raises(RuntimeError, match="attempts ledger"):
        B.simulate_candidates(c, cfg, 2, limits=B.Limits(attempts=6))
    assert sum(appended) == 6


def test_empty_optics_thermal_and_late_inputs_preserve_external_frame_registry():
    registry = pd.DataFrame(dict(event=[10, 20, 30], phase=[1., 2., 3.]))
    before = registry.copy(deep=True)
    channels = registry[["event"]].assign(chan=4)
    zero = B.Limits(photons=0, thermal_candidates=0, candidates=0, attempts=0,
                    generations=0, state_bytes=0, live_channels=3)
    optical = B.optical_candidates(photons(0), np.ones((40, 40)), CFG, 1, limits=zero)
    dark = B.thermal_candidates(channels, CFG, 1, limits=zero)
    assert_frame_equal(optical, S.optical_candidates(photons(0), np.ones((40, 40)), CFG, 1), check_exact=True)
    assert_frame_equal(dark, S.thermal_candidates(channels, CFG, 1), check_exact=True)
    empty = pd.concat([optical, dark], ignore_index=True)
    exact_products(B.simulate_candidates(empty, CFG, 1, limits=zero), S.simulate_candidates(empty, CFG, 1))
    late = candidates().assign(time=1600.)
    exact_products(B.simulate_candidates(late, CFG, 1, limits=B.Limits(state_bytes=0, live_channels=0)),
                   S.simulate_candidates(late, CFG, 1))
    assert_frame_equal(registry, before, check_exact=True)
    assert len(registry) == 3


def test_zero_window_and_nonzero_thermal_rate_with_no_generated_roots():
    channels = pd.DataFrame(dict(event=[0], chan=[0]))
    cfg = replace(CFG, thermal_rate=np.nextafter(0., 1.))
    dark = B.thermal_candidates(channels, cfg, 1)
    assert dark.empty
    exact_products(B.simulate_candidates(pd.concat([candidates(), dark]), cfg, 1),
                   S.simulate_candidates(candidates(), CFG, 1))
    cfg = replace(CFG, thermal_rate=1.)
    assert_frame_equal(B.thermal_candidates(channels, cfg, 1, start=100., end=100.),
                       S.thermal_candidates(channels, cfg, 1, start=100., end=100.), check_exact=True)


@pytest.mark.parametrize("field,bad", [("time", np.nan), ("time", np.inf), ("px", -1),
                                      ("py", 41), ("root", .5), ("root", -1),
                                      ("event", -1), ("chan", -1), ("dark", 2)])
def test_malformed_candidates(field, bad):
    c = candidates().astype({field: float})
    c.loc[0, field] = bad
    with pytest.raises(ValueError):
        B.simulate_candidates(c, CFG, 1)


@pytest.mark.parametrize("field,bad", [("p1", np.nan), ("p2", 0), ("capacitance", -1),
                                      ("thermal_rate", -1), ("cross_talk", 1.1),
                                      ("trap0_prob", .1), ("n_pixels_x", 39),
                                      ("inactive", ((40, 0),))])
def test_shared_configuration_validation(field, bad):
    cfg = replace(CFG, **{field: bad})
    channels = pd.DataFrame(dict(event=[0], chan=[0]))
    for function, args in ((B.simulate_candidates, (candidates(), cfg, 1)),
                           (B.thermal_candidates, (channels, cfg, 1)),
                           (B.optical_candidates, (photons(), np.ones((40, 40)), cfg, 1))):
        with pytest.raises(ValueError):
            function(*args)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_windows_and_photon_time(bad):
    channels = pd.DataFrame(dict(event=[0], chan=[0]))
    with pytest.raises(ValueError):
        B.simulate_candidates(candidates(), CFG, 1, end=bad)
    with pytest.raises(ValueError):
        B.thermal_candidates(channels, CFG, 1, end=bad)
    with pytest.raises(ValueError):
        B.thermal_candidates(channels, CFG, 1, start=bad)
    with pytest.raises(ValueError):
        B.optical_candidates(photons().assign(time=bad), np.ones((40, 40)), CFG, 1)


def test_nonfinite_arithmetic_rejected_not_returned():
    c = candidates([(0, 0, 0, 100., 0, 0, False)])
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="nonfinite"):
            B.simulate_candidates(c, replace(CERTAIN, capacitance=np.finfo(float).max), 1)
        with pytest.raises(ValueError, match="thermal time window"):
            B.thermal_candidates(pd.DataFrame(dict(event=[0], chan=[0])), CFG, 1,
                                 start=-np.finfo(float).max, end=np.finfo(float).max)


def test_duplicate_missing_and_unsigned_id_validation():
    c = candidates()
    with pytest.raises(ValueError, match="duplicate"):
        B.simulate_candidates(pd.concat([c, c]), CFG, 1)
    with pytest.raises(ValueError, match="missing"):
        B.simulate_candidates(c.drop(columns="px"), CFG, 1)
    c = c.iloc[:1].assign(root=np.array([2**63], dtype=np.uint64))
    with pytest.raises(ValueError, match="int64"):
        B.simulate_candidates(c, CFG, 1)
    with pytest.raises(ValueError, match="duplicate"):
        B.thermal_candidates(pd.DataFrame(dict(event=[0, 0], chan=[0, 0])), CFG, 1)
    with pytest.raises(ValueError, match="window"):
        B.thermal_candidates(pd.DataFrame(dict(event=[0], chan=[0])), CFG, 1, start=200., end=100.)


def test_global_rng_state_unchanged():
    before = np.random.get_state()
    B.optical_candidates(photons(), np.ones((40, 40)), CFG, 11)
    B.thermal_candidates(pd.DataFrame(dict(event=[0], chan=[0])), replace(CFG, thermal_rate=.001), 11)
    B.simulate_candidates(candidates(), replace(CERTAIN, cross_talk=.8), 11)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
