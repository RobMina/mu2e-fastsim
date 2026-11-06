"""Unit tests: beamfastsim.dioreco multi-primary filtering, CE track selection and the paired comparison."""
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
    assert r["shift_check"]["ratio_predicted"]["100"] == pytest.approx(r["above"]["100"]["ratio"], rel=1e-12)
