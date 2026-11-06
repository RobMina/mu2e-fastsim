"""Unit tests: beamfastsim.dioreco multi-primary filtering, CE track selection and the paired comparison."""
import json

import numpy as np
import pandas as pd
import pytest

from beamfastsim import dioreco as R


def _events(n):
    return pd.DataFrame(dict(run=np.full(n, 1430, np.uint64), subrun=np.full(n, 50, np.uint64),
                             event=np.arange(1, n + 1, dtype=np.uint64)))


def test_multi_primary_events_are_dropped_from_every_event_tree():
    ev = _events(4)
    parts = pd.DataFrame(dict(run=[1430] * 6, subrun=[50] * 6, event=[1, 2, 2, 3, 4, 4], pdg=[11, 11, 11, 11, 13, 11],
                              isprim=[1, 1, 1, 1, 1, 1], ismcprim=[0] * 6))
    bad = R.multi_primary_events(parts)
    assert bad.event.tolist() == [2]                       # event 4 has one electron and one muon
    steps = ev.assign(e=np.arange(4.0))
    assert R.drop_events(steps, bad).event.tolist() == [1, 3, 4]
    sub = pd.DataFrame(dict(run=[1430], subrun=[50], livetime=[1.0]))
    assert R.drop_events(sub, bad) is sub                  # no event column: unchanged


def _arm(ev, p, sel=None):
    n = len(ev)
    sel = np.ones(n, bool) if sel is None else sel
    d = ev.copy()
    d["fit"] = np.isfinite(p)
    d["p_reco"] = p
    d["pz"] = np.where(np.isfinite(p), 50.0, np.nan)
    d["pdg"] = 11.0
    d["fitcon"] = 0.1
    d["tandip"] = np.where(sel, 0.7, 0.2)
    d["d0"] = 0.0
    d["rmax"] = 500.0
    d["sel"] = R.selected(d)
    return d


def test_selection_and_paired_comparison():
    rng = np.random.default_rng(3)
    n = 20_000
    ev = _events(n)
    p = rng.uniform(95.0, 105.0, n)
    has = rng.random(n) < 0.6
    pr = np.where(has, p, np.nan)
    ref = _arm(ev, pr)
    same = R.compare(ev, ref, ref)
    assert same["efficiency_selected"]["ratio"] == 1.0 and same["efficiency_selected"]["diff_err"] == 0.0
    assert same["above"]["100"]["arm"] == pytest.approx(((pr > 100) & has).mean())
    # the arm loses 5 % of the reference tracks and shifts the momentum by +0.1 MeV/c
    keep = has & (rng.random(n) > 0.05)
    arm = _arm(ev, np.where(keep, p + 0.1, np.nan))
    r = R.compare(ev, arm, ref)
    e = r["efficiency_selected"]
    assert e["ratio"] == pytest.approx(keep.sum() / has.sum())
    assert abs(e["ratio"] - 0.95) < 4 * e["ratio_err"] and e["ratio_err"] < 0.005
    assert r["lost"] == int((has & ~keep).sum()) and r["gained"] == 0
    assert r["same_event_dp"]["mean"] == pytest.approx(0.1, abs=1e-9)
    # tracks failing the fiducial cuts are not selected
    cut = _arm(ev, pr, sel=np.zeros(n, bool))
    assert R.compare(ev, cut, ref)["efficiency_selected"]["arm"] == 0.0


def test_events_outside_the_denominator_raise():
    ev = _events(10)
    a = _arm(_events(12), np.full(12, 101.0))
    with pytest.raises(ValueError):
        R.compare(ev, a, _arm(ev, np.full(10, 101.0)))


def test_single_round_trip(tmp_path):
    pytest.importorskip("uproot")
    from mu2efastsim import tables
    ev = _events(3)
    parts = ev.iloc[[0, 1, 1, 2]].reset_index(drop=True).assign(pdg=11, isprim=1, ismcprim=0, key=[1, 1, 2, 1])
    gen = ev.assign(pdg=13)
    src = tmp_path / "t.root"
    tables.write(str(src), dict(gen=gen, particles=parts, steps=ev.assign(eion=0.1)), as_ttree=True)
    n, names = R.single(str(src), str(tmp_path / "out" / "t.root"))
    assert n == 1 and set(names) == {"gen", "particles", "steps"}
    assert tables.read(str(tmp_path / "out" / "t.root"), "gen").event.tolist() == [1, 3]


def test_arm_from_reduce_csv(tmp_path):
    ev = _events(50)
    a = _arm(ev, np.full(50, 101.0))
    a.drop(columns="sel").to_csv(tmp_path / "a.csv.gz", index=False)
    d, _ = R.read_arm(str(tmp_path / "a.csv.gz"))
    assert d["sel"].all() and d.event.dtype == np.uint64
    assert R.compare(ev, d, a)["efficiency_selected"]["diff"] == 0.0


def test_drop_outside_counts_extra_events():
    ev = _events(10)
    big = _arm(_events(12), np.full(12, 101.0))
    r = R.compare(ev, big, _arm(ev, np.full(10, 101.0)), drop_outside=True)
    assert r["dropped_outside"] == dict(arm=2, ref=0) and r["efficiency_selected"]["diff"] == 0.0


def test_shift_check_predicts_the_ratio_of_a_pure_shift():
    rng = np.random.default_rng(8)
    n = 200_000
    ev = _events(n)
    p = 95.0 + rng.exponential(1.2, n)            # steep falling spectrum above 95
    ref = _arm(ev, p)
    arm = _arm(ev, p + 0.05)
    r = R.compare(ev, arm, ref)
    assert r["shift_check"]["shift"] == pytest.approx(0.05)
    for c in R.P_CUTS:
        assert r["shift_check"]["ratio_predicted"][f"{c:g}"] == pytest.approx(r["above"][f"{c:g}"]["ratio"], rel=1e-12)
    assert json.loads(json.dumps(r["shift_check"], allow_nan=False)) == r["shift_check"]


def test_shift_check_zero_reference_yield_is_undefined():
    ev = _events(2)
    r = R.compare(ev, _arm(ev, [100.03, 100.03]), _arm(ev, [99.98, 99.98]))
    sc = r["shift_check"]
    assert sc["shift"] == pytest.approx(0.05)
    assert sc["ratio_predicted"]["95"] == 1.0
    assert np.isnan(r["above"]["100"]["ratio"])
    assert all(sc["ratio_predicted"][f"{c:g}"] is None for c in R.P_CUTS if c >= 100)
    assert json.loads(json.dumps(sc, allow_nan=False)) == sc


@pytest.mark.parametrize("pa, pr", [([101.0, np.nan], [np.nan, 101.0]), ([94.0, 94.0], [94.0, 94.0])])
def test_shift_check_without_same_event_shift_is_undefined(pa, pr):
    ev = _events(2)
    r = R.compare(ev, _arm(ev, pa), _arm(ev, pr))
    assert r["same_event_dp"]["n"] == 0
    assert r["shift_check"] == dict(shift=None, ratio_predicted={f"{c:g}": None for c in R.P_CUTS})
    assert json.loads(json.dumps(r["shift_check"], allow_nan=False)) == r["shift_check"]


@pytest.mark.parametrize("missing", ["arm", "ref", "both"])
def test_shift_check_missing_arms_preserves_event_denominator(missing):
    ev = _events(3)
    arm = ref = _arm(ev, [99.5, 101.0, 103.0])
    if missing in ("arm", "both"):
        arm = arm.iloc[:0]
    if missing in ("ref", "both"):
        ref = ref.iloc[:0]
    r = R.compare(ev, arm, ref)
    assert r["events"] == 3
    assert r["efficiency_selected"]["arm_count"] == len(arm)
    assert r["efficiency_selected"]["ref_count"] == len(ref)
    assert r["lost"] == max(len(ref) - len(arm), 0)
    assert r["gained"] == max(len(arm) - len(ref), 0)
    assert r["shift_check"] == dict(shift=None, ratio_predicted={f"{c:g}": None for c in R.P_CUTS})
    assert json.loads(json.dumps(r["shift_check"], allow_nan=False)) == r["shift_check"]


def test_shift_check_counts_only_selected_reference_tracks():
    ev = _events(6)
    ref = _arm(ev.iloc[:5], [101.0, 90.0, 200.0, np.nan, 105.0], sel=[True, True, False, True, False])
    arm = _arm(ev, [211.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    r = R.compare(ev, arm, ref)
    assert r["shift_check"]["shift"] == 110.0
    assert r["efficiency_selected"]["ref_count"] == 2
    assert r["above"]["100"]["ref_count"] == 1
    assert r["shift_check"]["ratio_predicted"] == {"95": 2.0, "100": 2.0, "102": None, "103": None, "104": None}
    assert json.loads(json.dumps(r["shift_check"], allow_nan=False)) == r["shift_check"]


def test_shift_check_defined_zero_prediction():
    ev = _events(2)
    r = R.compare(ev, _arm(ev, [96.0, 97.0]), _arm(ev, [101.0, 102.0]))
    assert r["shift_check"]["shift"] == -5.0
    assert r["shift_check"]["ratio_predicted"]["95"] == 1.0
    assert r["shift_check"]["ratio_predicted"]["100"] == 0.0
    assert json.loads(json.dumps(r["shift_check"], allow_nan=False)) == r["shift_check"]


def test_shift_check_nonfinite_mean_is_undefined():
    ev = _events(2)
    with np.errstate(over="ignore"):
        r = R.compare(ev, _arm(ev, [1e308, 1e308]), _arm(ev, [101.0, 101.0]))
    assert np.isinf(r["same_event_dp"]["mean"])
    assert r["shift_check"] == dict(shift=None, ratio_predicted={f"{c:g}": None for c in R.P_CUTS})
    assert json.loads(json.dumps(r["shift_check"], allow_nan=False)) == r["shift_check"]


@pytest.fixture
def pyplot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yield plt
    plt.close("all")


def test_plot_shows_zero_numerator_bins(tmp_path, pyplot):
    ev = _events(8)
    ref = _arm(ev, [100.1] * 4 + [101.1] * 4)
    arm = _arm(ev, [99.9, 99.9, np.nan, np.nan, 101.1, 101.1, np.nan, np.nan])
    r = R.compare(ev, arm, ref)
    path = tmp_path / "zero-yield.png"
    R.plot(r, path)
    ratio_ax = pyplot.gcf().axes[1]
    points = ratio_ax.containers[0].lines[0]
    np.testing.assert_allclose(points.get_xdata(), [100.125, 101.125])
    np.testing.assert_allclose(points.get_ydata(), [0.0, 0.5])
    assert ratio_ax.get_ylim()[0] < 0.0 < ratio_ax.get_ylim()[1]
    assert points.get_marker() == "o" and points.get_markersize() > 0
    assert path.stat().st_size > 0


@pytest.mark.parametrize("missing", ["arm", "ref", "both", "no-pair", "zero-tail"])
@pytest.mark.parametrize("legacy_nan", [False, True])
def test_plot_undefined_shift_predictions(tmp_path, pyplot, missing, legacy_nan):
    ev = _events(2)
    arm = ref = _arm(ev, [101.0, 101.0])
    if missing in ("arm", "both"):
        arm = arm.iloc[:0]
    if missing in ("ref", "both"):
        ref = ref.iloc[:0]
    if missing == "no-pair":
        arm, ref = _arm(ev, [101.0, np.nan]), _arm(ev, [np.nan, 101.0])
    if missing == "zero-tail":
        arm, ref = _arm(ev, [100.03, 100.03]), _arm(ev, [99.98, 99.98])
    r = R.compare(ev, arm, ref)
    sc = r["shift_check"]
    if legacy_nan:
        sc["shift"] = np.nan if sc["shift"] is None else sc["shift"]
        sc["ratio_predicted"] = {c: np.nan if v is None else v for c, v in sc["ratio_predicted"].items()}
    r = json.loads(json.dumps(r))
    path = tmp_path / "undefined.png"
    R.plot(r, path)
    ax = pyplot.gcf().axes[0]
    note = next(t.get_text() for t in ax.texts if "same-event shift" in t.get_text())
    if missing == "zero-tail":
        assert "same-event shift +0.050 MeV/c" in note
        assert "1.000 above 95, unavailable above 100" in note
    else:
        assert "same-event shift unavailable" in note
        assert "unavailable above 95, unavailable above 100" in note
    assert "nan" not in note.lower() and "nan" not in ax.get_title().lower()
    assert path.stat().st_size > 0
