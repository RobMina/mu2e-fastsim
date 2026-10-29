"""Historical-grid pilot coverage contracts, not an aggregate veto-fraction measurement."""
from dataclasses import asdict, replace

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import pilot_coverage as P
from crvfastsim.veto.card import VetoCard
from crvfastsim.veto.frameaudit import track_histogram


def reference(card=None, selected=3, step=1.0):
    card = card or VetoCard(dt_lo=0.0, dt_hi=2.0)
    return {"card": asdict(card) | {"sha": card.sha()}, "selected": selected,
            "pmin": 85.0, "pmax": 105.0, "F": {"grid_ns": step}}


def selection(times=None, momenta=None):
    return ak.Array({"t_mid": [[0.0, 1.0], [2.0]] if times is None else times,
                     "p_mid": [[100.0, 100.0], [100.0]] if momenta is None else momenta})


def frames(n=3):
    return pd.DataFrame({"input_sha256": pd.Series(["a" * 64] * n, dtype=object),
                         "run": np.ones(n, dtype=np.int64), "subrun": np.zeros(n, dtype=np.int64),
                         "event": np.arange(n, dtype=np.int64), "frame_id": np.arange(n, dtype=np.int64),
                         "pbtime": np.zeros(n), "intensity": np.ones(n)})


def clusters(f, which=(0,), times=None, **kwargs):
    which = np.asarray(which, dtype=np.int64)
    out = f.iloc[which][P.KEYS].reset_index(drop=True).copy()
    out["idx"] = out.groupby("frame_id").cumcount().astype(np.int64)
    out["avgtime"] = np.zeros(len(out)) if times is None else np.asarray(times, dtype=float)
    out["start"], out["end"] = out.avgtime - 0.1, out.avgtime + 0.1
    out["pe"] = np.full(len(out), 10.0)
    out["nLayers"] = np.full(len(out), 3, dtype=np.int64)
    out["sectortype"] = np.full(len(out), 1, dtype=np.int64)
    for key, value in kwargs.items():
        out[key] = value
    return out


def evaluate(f, c, card=None, **kwargs):
    card, hist, edges, _ = P.frozen_grid(reference(card), selection())
    return P.evaluate(f, c, card, hist, edges, **kwargs)


def test_frozen_histogram_exact_and_metadata():
    ref = reference()
    card, hist, edges, meta = P.frozen_grid(ref, selection())
    expected_hist, expected_edges = track_histogram([0.0, 1.0, 2.0], 1.0)
    np.testing.assert_array_equal(hist, expected_hist)
    np.testing.assert_array_equal(edges, expected_edges)
    assert card.sha() == ref["card"]["sha"] == meta["card_sha"]
    assert meta["selected"] == meta["total_tracks"] == 3
    assert meta["grid_ns"] == 1.0 and meta["bins"] == len(hist)
    assert meta["contract"] == "historical-difference-array-edges-v1"
    assert "not continuous strict centre-exact" in meta["boundary_semantics"]
    assert "not full F" in meta["scope"]
    assert "F" not in meta


def test_strict_momentum_bounds_keep_only_interior_and_empty_events():
    sel = selection([[10.0, 20.0, 30.0], [], [40.0]], [[85.0, 100.0, 105.0], [], [110.0]])
    _, hist, edges, meta = P.frozen_grid(reference(selected=1), sel)
    h, e = track_histogram([20.0], 1.0)
    np.testing.assert_array_equal(hist, h)
    np.testing.assert_array_equal(edges, e)
    assert meta["total_tracks"] == 4


def test_source_qualified_nonzero_mixed_empty_and_preserved_frame_columns():
    f = frames()
    f.loc[1, "input_sha256"] = "b" * 64
    f.loc[1, "event"] = 0
    f.index = [8, 4, 6]
    c = clusters(f, (0, 1), [0.0, 20.0]).iloc[::-1]
    out = evaluate(f, c)
    pd.testing.assert_frame_equal(out[f.columns], f)
    np.testing.assert_array_equal(out.n_coincidences, [1, 1, 0])
    np.testing.assert_array_equal(out.n_veto_eligible, [1, 1, 0])
    np.testing.assert_array_equal(out.gate_pass, [True, True, True])
    np.testing.assert_array_equal(out.track_grid_coverage, [2 / 3, 0, 0])
    assert not any(name in out for name in ("F", "fraction", "err", "finite_library_uncertainty"))


@pytest.mark.parametrize("n", [0, 3])
def test_all_empty(n):
    f = frames(n)
    out = evaluate(f, clusters(f, ()))
    assert len(out) == n
    assert (out.n_coincidences == 0).all() and (out.n_veto_eligible == 0).all()
    assert out.gate_pass.all() and (out.track_grid_coverage == 0).all()


@pytest.mark.parametrize("mode,coverage,passed,eligible", [("event", 1.0, False, 1), ("coinc", 0.0, True, 0)])
def test_event_vs_coincidence_gate(mode, coverage, passed, eligible):
    f = frames()
    c = clusters(f, (0,), [0.0], start=[-1.0], end=[1.0])
    card = VetoCard(dt_hi=2.0, gate_start_lo=-1.0, gate_end_hi=3.0, gate_mode=mode)
    out = evaluate(f, c, card)
    assert out.track_grid_coverage[0] == coverage
    assert bool(out.gate_pass[0]) is passed and out.n_veto_eligible[0] == eligible
    assert out.track_grid_coverage[1:].eq(0).all() and out.gate_pass[1:].all()


@pytest.mark.parametrize("mode", ["event", "coinc"])
def test_end_gate_is_strict(mode):
    f = frames(1)
    c = clusters(f, end=[1.0])
    card = VetoCard(dt_hi=2.0, gate_end_hi=1.0, gate_mode=mode)
    out = evaluate(f, c, card)
    assert out.track_grid_coverage[0] == (1.0 if mode == "event" else 0.0)


@pytest.mark.parametrize("changes,eligible", [({"min_pe": 11.0}, 0), ({"min_pe": 10.0}, 1),
    ({"min_layers": 4}, 0), ({"min_layers": 3}, 1), ({"sectors": (2,)}, 0),
    ({"sectors": (1,)}, 1), ({"min_pe_by_sector": {1: 11.0}}, 0),
    ({"min_pe": 11.0, "min_pe_by_sector": {1: 10.0}}, 1)])
def test_sector_pe_and_layer_thresholds(changes, eligible):
    f = frames(1)
    out = evaluate(f, clusters(f), replace(VetoCard(dt_hi=2.0), **changes))
    assert out.n_veto_eligible[0] == eligible
    assert out.track_grid_coverage[0] == (2 / 3 if eligible else 0)


def test_overlapping_windows_are_union_not_sum():
    f = frames()
    out = evaluate(f, clusters(f, (0, 0, 0), [-1.0, -1.0, 0.0]))
    assert out.n_coincidences[0] == 3 and out.track_grid_coverage[0] == 1.0


@pytest.mark.parametrize("time", [-0.5, 0.0, 0.5, 1.0, np.nextafter(0.5, 0.0), np.nextafter(0.5, 1.0)])
def test_exact_boundary_parity(time):
    f = frames(1)
    c = clusters(f, times=[time])
    card, hist, edges, _ = P.frozen_grid(reference(), selection())
    out = P.evaluate(f, c, card, hist, edges)
    lo = np.clip(np.ceil((time + card.dt_lo - edges[0]) / (edges[1] - edges[0])), 0, len(hist))
    hi = np.clip(np.ceil((time + card.dt_hi - edges[0]) / (edges[1] - edges[0])), 0, len(hist))
    cells = np.arange(len(hist))
    assert out.track_grid_coverage[0] == ((cells >= lo) & (cells < hi)) @ hist


def test_historical_grid_differs_from_continuous_strict_centre_boundary():
    f = frames(1)
    card, hist, edges, _ = P.frozen_grid(reference(selected=1), selection([[0.0]], [[100.0]]))
    out = P.evaluate(f, clusters(f, times=[0.0]), card, hist, edges)
    centres = (edges[:-1] + edges[1:]) / 2
    continuous_strict = ((centres > card.dt_lo) & (centres < card.dt_hi)) @ hist
    assert out.track_grid_coverage[0] == 0.0
    assert continuous_strict == 0.0
    out = P.evaluate(f, clusters(f, times=[-0.5]), card, hist, edges)
    continuous_strict = ((centres > -0.5) & (centres < 1.5)) @ hist
    assert out.track_grid_coverage[0] == continuous_strict == 1.0
    out = P.evaluate(f, clusters(f, times=[-2.0]), card, hist, edges)
    continuous_strict = ((centres > -2.0) & (centres < 0.0)) @ hist
    assert out.track_grid_coverage[0] == 1.0 and continuous_strict == 0.0


@pytest.mark.parametrize("key,value", [("pmin", True), ("pmax", "105"), ("pmin", np.nan),
    ("pmax", np.inf), ("pmin", 105.0), ("pmin", 106.0), ("selected", True),
    ("selected", 3.0), ("selected", 2), ("selected", 0)])
def test_bad_reference_selection_parameters(key, value):
    ref = reference()
    ref[key] = value
    with pytest.raises(ValueError):
        P.frozen_grid(ref, selection())


@pytest.mark.parametrize("step", [True, "1", 0.0, -1.0, np.inf, np.nan])
def test_bad_grid_step(step):
    with pytest.raises(ValueError):
        P.frozen_grid(reference(step=step), selection())


@pytest.mark.parametrize("mutation", ["missing_hash", "wrong_hash", "missing_card", "missing_selected", "missing_grid",
    "bad_card_real", "bad_card_layers", "reversed_window", "bad_sector_threshold"])
def test_bad_reference_card_and_schema(mutation):
    ref = reference()
    if mutation == "missing_hash":
        del ref["card"]["sha"]
    elif mutation == "wrong_hash":
        ref["card"]["sha"] = "0" * 16
    elif mutation == "missing_card":
        del ref["card"]
    elif mutation == "missing_selected":
        del ref["selected"]
    elif mutation == "missing_grid":
        del ref["F"]
    elif mutation == "bad_card_real":
        ref["card"]["dt_hi"] = np.inf
    elif mutation == "bad_card_layers":
        ref["card"]["min_layers"] = True
    elif mutation == "reversed_window":
        ref["card"]["dt_hi"] = -2.0
    else:
        ref["card"]["min_pe_by_sector"] = {1: np.nan}
    with pytest.raises(ValueError):
        P.frozen_grid(ref, selection())


@pytest.mark.parametrize("sel", [selection([[1.0], [2.0, 3.0]], [[100.0, 100.0], [100.0]]),
    selection([[1.0, np.nan], [2.0]]), selection([[1.0, 2.0], [np.inf]]),
    selection(momenta=[[100.0, np.nan], [100.0]]), selection(momenta=[[100.0, np.inf], [100.0]]),
    selection([[True, False], [True]]), selection([["1", "2"], ["3"]]),
    selection([[1.0, None], [2.0]]), selection([None, [1.0, 2.0, 3.0]]),
    selection([1.0, 2.0, 3.0], [100.0, 100.0, 100.0]),
    selection([[[1.0, 2.0]], [[3.0]]], [[[100.0, 100.0]], [[100.0]]])])
def test_bad_selection_jagged_types_and_finiteness(sel):
    with pytest.raises(ValueError):
        P.frozen_grid(reference(), sel)


@pytest.mark.parametrize("limit", [{"max_tracks": 2}, {"max_bins": 2}, {"max_tracks": True}, {"max_bins": 2.5}])
def test_caps_precede_histogram_and_arange(monkeypatch, limit):
    sel, ref = selection(), reference()
    def forbidden(*args, **kwargs):
        raise AssertionError("allocation helper called before cap admission")
    monkeypatch.setattr(P, "track_histogram", forbidden)
    monkeypatch.setattr(P.np, "arange", forbidden)
    with pytest.raises(ValueError):
        P.frozen_grid(ref, sel, **limit)


def test_total_track_cap_not_selected_only(monkeypatch):
    sel = selection([[0.0, 1.0, 2.0]], [[80.0, 100.0, 110.0]])
    monkeypatch.setattr(P, "track_histogram", lambda *a: pytest.fail("histogram before total track cap"))
    with pytest.raises(ValueError, match="max_tracks"):
        P.frozen_grid(reference(selected=1), sel, max_tracks=2)


@pytest.mark.parametrize("times,step", [([0.0, 1e200, 2.0], 1.0), ([0.0, 1.0, 2.0], 1e-300),
                                       ([1e300, 1e300, 1e300], 1.0)])
def test_extreme_grid_rejected_before_helper(monkeypatch, times, step):
    monkeypatch.setattr(P, "track_histogram", lambda *a: pytest.fail("unbounded grid"))
    with pytest.raises(ValueError):
        P.frozen_grid(reference(step=step), selection([times], [[100.0] * 3]))


@pytest.mark.parametrize("target,field,value", [("frames", "run", 1.0), ("frames", "frame_id", True),
    ("clusters", "idx", 0.0), ("clusters", "event", True), ("clusters", "sectortype", 1.0),
    ("clusters", "nLayers", False), ("clusters", "avgtime", np.nan), ("clusters", "pe", np.inf),
    ("clusters", "start", "0"), ("clusters", "end", True), ("frames", "intensity", np.nan),
    ("frames", "input_sha256", "bad"), ("clusters", "input_sha256", "b" * 64),
    ("clusters", "run", 2), ("clusters", "frame_id", 100), ("clusters", "pe", -1.0)])
def test_malformed_identity_schema_and_missing_frame(target, field, value):
    f = frames(1)
    c = clusters(f)
    (f if target == "frames" else c)[field] = value
    with pytest.raises(ValueError):
        evaluate(f, c)


@pytest.mark.parametrize("mutation", ["duplicate_frame", "duplicate_cluster", "negative_idx", "missing_field",
    "missing_key", "count", "float_count", "reversed_times", "duplicate_columns"])
def test_ledger_schema_and_counts(mutation):
    f = frames()
    c = clusters(f)
    if mutation == "duplicate_frame":
        f = pd.concat([f, f.iloc[:1]], ignore_index=True)
    elif mutation == "duplicate_cluster":
        c = pd.concat([c, c], ignore_index=True)
    elif mutation == "negative_idx":
        c["idx"] = -1
    elif mutation == "missing_field":
        c = c.drop(columns="avgtime")
    elif mutation == "missing_key":
        f = f.drop(columns="run")
    elif mutation == "count":
        f["n_coincidences"] = [0, 0, 0]
    elif mutation == "float_count":
        f["n_coincidences"] = [1.0, 0.0, 0.0]
    elif mutation == "reversed_times":
        c["start"] = c.end + 1
    else:
        c = pd.concat([c, c[["avgtime"]]], axis=1)
    with pytest.raises(ValueError):
        evaluate(f, c)


def test_saved_coincidence_count_consistent():
    f = frames()
    f["n_coincidences"] = [1, 0, 0]
    out = evaluate(f, clusters(f))
    np.testing.assert_array_equal(out.n_coincidences, f.n_coincidences)


def test_max_cells_before_coverage_or_unflatten(monkeypatch):
    f = frames(1)
    c = clusters(f, (0, 0))
    card, hist, edges, _ = P.frozen_grid(reference(), selection())
    monkeypatch.setattr(P.deadtime, "frame_coverage", lambda *a, **k: pytest.fail("helper before cap"))
    monkeypatch.setattr(P.ak, "unflatten", lambda *a, **k: pytest.fail("allocation before cap"))
    with pytest.raises(ValueError, match="max_cells"):
        P.evaluate(f, c, card, hist, edges, max_cells=3 * len(hist))


def test_bounded_chunk_calls_and_exact_independent_parity(monkeypatch):
    f = frames(17)
    c = clusters(f, tuple(np.repeat(np.arange(17), 2)), np.tile([-1.0, 0.0], 17))
    card, hist, edges, _ = P.frozen_grid(reference(), selection())
    original = P.deadtime.frame_coverage
    calls = []
    limit = 3 * (3 * len(hist) + 1)
    def bounded(coincs, card, hist, edges, chunk):
        counts = ak.to_numpy(ak.num(coincs.time))
        assert (len(coincs) * (len(hist) + 1) + counts.sum() * len(hist)) <= limit
        assert chunk == len(coincs)
        calls.append(len(coincs))
        return original(coincs, card, hist, edges, chunk=chunk)
    monkeypatch.setattr(P.deadtime, "frame_coverage", bounded)
    out = P.evaluate(f, c, card, hist, edges, max_cells=limit)
    assert calls == [3, 3, 3, 3, 3, 2]
    np.testing.assert_array_equal(out.track_grid_coverage, np.ones(17))


def test_independent_check_detects_changed_helper(monkeypatch):
    f = frames(1)
    original = P.deadtime.frame_coverage
    monkeypatch.setattr(P.deadtime, "frame_coverage", lambda *a, **k: original(*a, **k) + 1e-15)
    with pytest.raises(ValueError, match="parity"):
        evaluate(f, clusters(f))


@pytest.mark.parametrize("mutation", ["hist_nan", "hist_negative", "hist_unnormalized", "edges_nonuniform", "edges_nan"])
def test_malformed_grid_evaluate(mutation):
    f = frames(1)
    card, hist, edges, _ = P.frozen_grid(reference(), selection())
    if mutation == "hist_nan":
        hist[0] = np.nan
    elif mutation == "hist_negative":
        hist[0] = -1.0
    elif mutation == "hist_unnormalized":
        hist *= 2
    elif mutation == "edges_nonuniform":
        edges[1] += 0.1
    else:
        edges[0] = np.inf
    with pytest.raises(ValueError):
        P.evaluate(f, clusters(f), card, hist, edges)


def test_nontrivial_histogram_block_reductions_and_clipped_windows():
    rng = np.random.default_rng(17)
    times = rng.normal(1000.0, 10.0, 301).tolist()
    card, hist, edges, _ = P.frozen_grid(reference(selected=301, step=0.25),
                                         selection([times], [[100.0] * len(times)]))
    f = frames(17)
    which = tuple(np.repeat(np.arange(17), 4))
    times = rng.normal(1000.0, 10.0, len(which))
    times[:4] = [-1e6, 1e6, edges[0], edges[-1]]
    c = clusters(f, which, times)
    limit = 3 * (5 * len(hist) + 1)
    out = P.evaluate(f, c, card, hist, edges, max_cells=limit)
    assert ((out.track_grid_coverage >= 0) & (out.track_grid_coverage <= 1)).all()
    assert out.track_grid_coverage.sum() > 0


def test_failing_event_gate_counts_even_threshold_ineligible_cluster():
    f = frames(1)
    c = clusters(f, start=[-1.0], pe=[0.0])
    out = evaluate(f, c, VetoCard(min_pe=10.0, gate_start_lo=0.0))
    assert out.n_veto_eligible[0] == 0 and not out.gate_pass[0]
    assert out.track_grid_coverage[0] == 1.0


@pytest.mark.parametrize("value", [42, True, None])
def test_sha_requires_string_content(value):
    f = frames(1)
    c = clusters(f)
    f["input_sha256"] = value
    with pytest.raises(ValueError, match="input_sha256"):
        evaluate(f, c)


def test_out_of_selection_nonfinite_track_still_rejected():
    sel = selection([[np.nan, 1.0]], [[80.0, 100.0]])
    with pytest.raises(ValueError, match="finite"):
        P.frozen_grid(reference(selected=1), sel)


def test_window_int64_cast_overflow_rejected_before_helper(monkeypatch):
    f = frames(1)
    c = clusters(f, times=[1e300])
    monkeypatch.setattr(P.deadtime, "frame_coverage", lambda *a, **k: pytest.fail("unsafe int64 cast"))
    with pytest.raises(ValueError, match="int64"):
        evaluate(f, c)


def test_cluster_indices_are_unique_labels_not_coverage_positions():
    f = frames(1)
    c = clusters(f, (0, 0), [-1.0, 0.0], idx=[9, 4])
    out = evaluate(f, c)
    assert out.track_grid_coverage[0] == 1.0 and out.n_coincidences[0] == 2


def test_unrepresentable_real_reference_rejected():
    ref = reference()
    ref["pmin"] = -(10**400)
    with pytest.raises(ValueError, match="finite"):
        P.frozen_grid(ref, selection())


@pytest.mark.parametrize("case", ["missing", "distribution", "disagree"])
def test_adapter_frame_cluster_counts_are_independent_census(case):
    f = frames()
    c = clusters(f, (0, 1))
    f["clusters"] = np.array([1, 1, 0], dtype=np.int64)
    if case == "missing":
        c = c.iloc[:1]
    elif case == "distribution":
        f["clusters"] = [2, 0, 0]
    else:
        f["n_coincidences"] = [0, 1, 0]
    with pytest.raises(ValueError, match="inconsistent frame coincidence counts"):
        evaluate(f, c)
