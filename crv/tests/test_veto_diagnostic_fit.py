"""Partition-exact diagnostics on identical ADCs, separate from production equivalence."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.reco import pulsefit as P
from crvfastsim.veto import diagnostic_fit as B, qualified_response as R, frame_registry as F
from crvfastsim.veto import replay_waveform_control as W
from test_veto_qualified_response import domain, configs


def matrix():
    n = np.array([24, 36, 48, 60, 216], np.int64)
    t = np.arange(n.max()) * P.PERIOD
    peaks = np.array([75., 105., 135., 165., 75.])
    adc = np.rint(100. + P.gumbel(t[None, :], np.array([400., 500., 800., 900., 600.])[:, None],
                                peaks[:, None], np.array([14., 19., 24., 32., 19.])[:, None])).astype(np.int64)
    adc[4] = np.rint(100. + sum(P.gumbel(t, 600., 75. + 300. * k, 19.) for k in range(9)))
    adc[np.arange(len(t))[None, :] >= n[:, None]] = 0
    return adc, n, np.zeros(5, np.int64), np.full(5, 100.), np.full(5, 1000.), np.full(5, 20.)


def canonical(result, selection):
    out = {k: v.copy() for k, v in result.items() if k != "wf_near_threshold"}
    out["wf"] = selection[out["wf"]]
    order = np.lexsort((out["seq"], out["wf"]))
    return {k: v[order] for k, v in out.items()}


def test_matrix_partition_permutation_padding_and_full_recovery_exact():
    args = matrix()
    full = B.reconstruct_matrix(*args)
    assert np.count_nonzero(full["wf"] == 4) == 9
    for selected in (np.arange(5)[::-1], np.array([0, 2, 4]), np.array([1]), np.array([4])):
        part = [a[selected] for a in args]
        part[0] = part[0][:, :part[1].max()]
        got = B.reconstruct_matrix(*part)
        expected = {k: v[np.isin(full["wf"], selected)] for k, v in full.items() if k != "wf_near_threshold"}
        for k, v in canonical(got, selected).items():
            np.testing.assert_array_equal(v, expected[k], err_msg=k)
        np.testing.assert_array_equal(got["wf_near_threshold"], full["wf_near_threshold"][selected])
    padded = list(args)
    padded[0] = np.pad(args[0], ((0, 0), (0, 100)), constant_values=999)
    for k, v in B.reconstruct_matrix(*padded).items():
        np.testing.assert_array_equal(v, full[k], err_msg=k)


def test_single_waveform_matches_unchanged_production_exactly():
    args = matrix()
    for row in (0, 1, 2, 3, 4):
        one = [a[row:row+1] for a in args]
        expected = P.reconstruct_full(*one)
        for k, value in B.reconstruct_matrix(*one).items():
            np.testing.assert_array_equal(value, expected[k], err_msg=k)


def test_width_buckets_remove_masked_padding_and_prove_fit_intervention(monkeypatch):
    t = np.tile(np.arange(9) * 12.5, (3, 1))
    count = np.array([5, 7, 9])
    y = np.rint(P.gumbel(t, 600., 40., 19.))
    p0 = np.tile([600., 40., 19.], (3, 1))
    lo, hi = np.tile([420., 25., 5.], (3, 1)), np.tile([900., 55., 40.], (3, 1))
    mask = np.arange(9)[None, :] < count[:, None]
    raw = P.fit_gumbel(t, y, mask, p0, lo, hi)
    calls, original = [], B.LM.fit_gumbel
    def capture(t, y, mask, *args):
        calls.append(t.shape)
        assert mask.all()
        return original(t, y, mask, *args)
    monkeypatch.setattr(B.LM, "fit_gumbel", capture)
    got = B.fit_buckets(t, y, mask, p0, lo, hi)
    assert calls == [(1, 5), (1, 7), (1, 9)]
    one = original(t[:1, :5], y[:1, :5], mask[:1, :5], p0[:1], lo[:1], hi[:1])
    for a, b in zip(got, one):
        np.testing.assert_array_equal(a[:1], b)
    print("MASKED_WIDTH_DIAGNOSIS", {"parameters_changed": int(np.count_nonzero(raw[0] != got[0])),
                                        "max_abs_parameter_delta": float(np.max(np.abs(raw[0] - got[0])))})


def as_digis(table):
    keys = ["event", "chan", "starttdc"]
    frame = table.sort_values(keys + ["sample"])
    heads = frame.drop_duplicates(keys)
    count = frame.groupby(keys, sort=False).size().to_numpy()
    assert (count == count[0]).all()
    return {"chan": heads.event.to_numpy() * W.MULT + heads.chan.to_numpy(),
            "starttdc": heads.starttdc.to_numpy(), "adc": frame.adc.to_numpy().reshape(len(heads), count[0])}


def test_known_qualified_partition_identical_adcs_new_fit_exact(monkeypatch):
    registry, q, cond = domain()
    cfg, wf = configs(monkeypatch)
    args = (registry, F.digest(registry), 3027000, 4027000, "historical", np.ones((40, 40)), replace(cfg, thermal_rate=.01), wf)
    products, _ = R.run_arm(q, cond, *args)
    encoded = F.encode(products["digis"].drop(columns="frame_id"), registry)
    conditions = F.encode(cond, registry)
    all_digis = as_digis(encoded)
    full = B.reconstruct(all_digis, conditions)
    keep = all_digis["chan"] // W.MULT == 0
    subset = {k: v[keep][::-1] for k, v in all_digis.items()}
    reduced = B.reconstruct(subset, conditions[conditions.event == 0].iloc[::-1])
    for actual, complete, keys in zip(reduced, full, (["event", "chan", "seq"], ["event", "chan", "starttdc"], ["event", "chan", "tdc"])):
        expected = complete[complete.event == 0].drop(columns="wf")
        pd.testing.assert_frame_equal(actual.drop(columns="wf").sort_values(keys).reset_index(drop=True),
                                      expected.sort_values(keys).reset_index(drop=True), check_exact=True)
    captured, kernel = [], P.fit_gumbel
    def capture(*args):
        result = kernel(*args)
        captured.append((args, result))
        return result
    with monkeypatch.context() as context:
        context.setattr(P, "fit_gumbel", capture)
        original = W.reconstruct(all_digis, conditions)[0]
    width_changed, width_max, padding_changed = 0, 0., 0
    for fit_args, old_result in captured:
        trimmed = B.fit_buckets(*fit_args)
        width_changed += int(np.count_nonzero(old_result[0] != trimmed[0]))
        width_max = max(width_max, float(np.max(np.abs(old_result[0] - trimmed[0]))))
        padded = [np.pad(a, ((0, 0), (0, 17)), mode="edge") for a in fit_args[:3]]
        padded[2][:, -17:] = False
        pad_result = kernel(*padded, *fit_args[3:])
        padding_changed += int(np.count_nonzero(old_result[0] != pad_result[0]))
        stable = B.fit_buckets(*padded, *fit_args[3:])
        for a, b in zip(stable, trimmed):
            np.testing.assert_array_equal(a, b)
    print("IDENTICAL_FIT_INPUT_WIDTH_INTERVENTION", {"changed_parameters": width_changed,
          "max_abs_parameter_delta": width_max, "fixed_batch_masked_padding_changed_parameters": padding_changed})
    assert padding_changed > 0
    compare = original.merge(full[0], on=["wf", "seq"], suffixes=("_old", "_new"), validate="one_to_one")
    print("QUALIFIED_FIT_INTERVENTION", {"pulses": len(compare), "target_pulses": len(reduced[0]),
          "changed_pe": int((compare.pe_old != compare.pe_new).sum()),
          "max_abs_pe": float(np.max(np.abs(compare.pe_old - compare.pe_new))),
          "changed_failed": int((compare.failed_old != compare.failed_new).sum())})
    assert len(compare) == len(original) == len(full[0])
    for name in ("failed", "zerondf", "pedfromdb", "channel_ignored"):
        assert (compare[f"{name}_old"] == compare[f"{name}_new"]).all()


def test_concat_cap_precedes_matrix_allocation(monkeypatch):
    d = {"chan": np.array([0, 1]), "starttdc": np.array([0, 0]), "adc": np.ones((2, 12), np.int64)}
    monkeypatch.setattr(P, "concat_waveforms", lambda *a: pytest.fail("allocated before cap"))
    with pytest.raises(ValueError, match="allocation limit"):
        B.reconstruct(d, F.encode(domain()[2], domain()[0]), limits=B.Limits(matrix_cells=23))


@pytest.mark.parametrize("case", ["length", "cells", "rows", "nan", "calibration", "dtype"])
def test_invalid_matrix_and_allocation_limits(case):
    args = list(matrix())
    limits = B.Limits()
    if case == "length":
        args[1][0] = 0
    elif case == "cells":
        limits = replace(limits, matrix_cells=10)
    elif case == "rows":
        limits = replace(limits, sample_rows=10)
    elif case == "nan":
        args[0] = args[0].astype(float)
        args[0][0, 1] = np.nan
    elif case == "calibration":
        args[4][0] = 0.
    else:
        args[1] = args[1].astype(float)
    with pytest.raises(ValueError):
        B.reconstruct_matrix(*args, limits=limits)


def test_empty_and_nonprefix_fit_masks():
    empty = B.reconstruct_matrix(np.zeros((0, 0), int), np.zeros(0, int), np.zeros(0, int), *[np.zeros(0)]*3)
    assert not any(len(v) for v in empty.values())
    with pytest.raises(ValueError, match="prefixes"):
        B.fit_buckets(np.ones((1, 3)), np.ones((1, 3)), np.array([[True, False, True]]),
                      np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3)))


def test_repeated_widths_are_vectorized_and_solver_failure_aborts(monkeypatch):
    t = np.tile(np.arange(7) * 12.5, (4, 1))
    y = np.rint(P.gumbel(t, 600., 40., 19.))
    mask = np.ones_like(t, dtype=bool)
    p0 = np.tile([600., 40., 19.], (4, 1))
    lo, hi = np.tile([420., 25., 5.], (4, 1)), np.tile([900., 55., 40.], (4, 1))
    original, calls = B.LM.fit_gumbel, []
    def capture(*args):
        calls.append(args[0].shape)
        return original(*args)
    monkeypatch.setattr(B.LM, "fit_gumbel", capture)
    full = B.fit_buckets(t, y, mask, p0, lo, hi)
    assert calls == [(4, 7)]
    for a, b in zip(full, P.fit_gumbel(t, y, mask, p0, lo, hi)):
        np.testing.assert_array_equal(a, b)
    solve = np.linalg.solve
    def problematic_companion(a, b):
        if len(a) > 1:
            raise np.linalg.LinAlgError("injected singular companion")
        return solve(a, b)
    monkeypatch.setattr(np.linalg, "solve", problematic_companion)
    with pytest.raises(ValueError, match="batch incomplete"):
        B.fit_buckets(t, y, mask, p0, lo, hi)
    one = B.fit_buckets(t[:1], y[:1], mask[:1], p0[:1], lo[:1], hi[:1])
    for a, b in zip(one, full):
        np.testing.assert_array_equal(a, b[:1])


@pytest.mark.parametrize("case", ["fractional_adc", "oversize_adc", "tiny_calibration", "config", "condition_nan", "status_float"])
def test_unsafe_numeric_inputs_fail_closed(case):
    args = list(matrix())
    if case == "fractional_adc":
        args[0] = args[0].astype(float) + .5
    elif case == "oversize_adc":
        args[0][0, 0] = 4096
    elif case == "tiny_calibration":
        args[4][0] = 1e-300
    elif case == "config":
        with pytest.raises(ValueError, match="default"):
            B.reconstruct_matrix(*args, cfg=replace(P.RecoConfig(), lm_iterations=1))
        return
    else:
        r, _, c = domain()
        c = F.encode(c, r)
        if case == "condition_nan":
            c.loc[0, "w"] = np.nan
        else:
            c["status"] = c.status.astype(float) + .5
        d = {"chan": np.array([0]), "starttdc": np.array([0]), "adc": np.ones((1, 12), np.int64)}
        with pytest.raises(ValueError):
            B.reconstruct(d, c)
        return
    with pytest.raises(ValueError):
        B.reconstruct_matrix(*args)


def test_overlapping_nonconsecutive_digi_tails_preserve_grouping():
    r, _, c = domain()
    c = F.encode(c, r)
    adc = np.tile(np.array([100, 110, 120, 150, 210, 290, 220, 160, 130, 110, 101, 100]), (3, 1))
    d = {"chan": np.zeros(3, np.int64), "starttdc": np.array([20, 30, 42]), "adc": adc}
    actual = B.reconstruct(d, c)
    expected = W.reconstruct(d, c)
    assert actual[1].n.tolist() == [12, 24]
    pd.testing.assert_frame_equal(actual[1], expected[1], check_exact=True)
    pd.testing.assert_frame_equal(actual[2], expected[2], check_exact=True)
    assert actual[2].duplicated(["event", "chan", "tdc"]).any()
