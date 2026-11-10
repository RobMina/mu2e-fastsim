"""stagecompare: the per-stage matching on synthetic reco-stage dumps (StrawHitTableDumper layout)."""
import awkward as ak
import numpy as np
import pytest

from trkfastsim.reco import stagecompare as SC


def _event(rng, run, ev, nstraw=40, shift_t0=0.0, drop=0):
    """One event: nstraw straw hits on distinct straws, panel hits of two straws each, one time cluster per
    collection (0: panel hits, 2: straw hits), one helix seed, one fit with all straws."""
    sid = rng.choice(20000, nstraw, replace=False).astype(np.int32)
    pind = np.arange(nstraw)                                   # panel hit k = straw hits 2k, 2k+1
    npanel = nstraw // 2
    keep = np.arange(nstraw - drop)                            # straw hits used by clusters, seeds and the fit
    tc0 = np.arange(npanel - (drop + 1) // 2)                  # panel-hit indices
    lh = dict(rad=300 + rng.normal(), lam=200 + rng.normal(), cx=10.0, cy=-5.0, phi0=0.3, t0=600.0 + shift_t0)
    return dict(
        run=run, subrun=1, event=ev,
        d_sid=sid, d_tdc0=sid * 3, d_tdc1=sid * 3 + 1, d_tot0=np.ones(nstraw, np.int32),
        d_tot1=np.ones(nstraw, np.int32), d_pmp=np.full(nstraw, 7, np.int32),
        s_sid=sid, s_nsh=np.ones(nstraw, np.int32), s_ncombo=np.ones(nstraw, np.int32),
        s_eend=np.zeros(nstraw, np.int32), s_ind=np.arange(nstraw), s_flag=np.zeros(nstraw, np.uint32),
        s_time=rng.uniform(500, 1500, nstraw).astype(np.float32),
        p_sid=sid[::2], p_nsh=np.full(npanel, 2, np.int32), p_ncombo=np.full(npanel, 2, np.int32),
        p_eend=np.zeros(npanel, np.int32), p_ind=pind, p_flag=np.zeros(npanel, np.uint32),
        p_time=rng.uniform(500, 1500, npanel).astype(np.float32),
        tc_coll=np.array([0, 2]), tc_n=np.array([len(tc0), len(keep)]), tc_ind=np.concatenate([tc0, keep]),
        tc_t0=np.array([600.0 + shift_t0, 601.0 + shift_t0], np.float32),
        hs_coll=np.array([0]), hs_n=np.array([len(keep)]), hs_ind=keep, hs_status=np.array([3], np.uint32),
        hs_t0=np.array([600.0 + shift_t0], np.float32),
        **{f"hs_{p}": np.array([lh[p] if p in lh else 1.0]) for p in ("rad", "lam", "cx", "cy", "fz0")},
        kl_status=np.array([7], np.uint32), kl_chi2=np.array([40.0], np.float32), kl_ndof=np.array([35]),
        kl_fitcon=np.array([0.4], np.float32), kl_nhit=np.array([len(keep)]), kl_nactive=np.array([len(keep) - 2]),
        kl_segmom=np.array([104.0], np.float32), kl_segmomerr=np.array([0.2], np.float32),
        **{f"kl_lh_{p}": np.array([v]) for p, v in lh.items()}, **{f"kl_lh_{p}err": np.array([1.0]) for p in lh},
        kh_trk=np.zeros(len(keep), np.int32), kh_ind=keep, kh_sid=sid[keep],
        kh_active=(np.arange(len(keep)) >= 2).astype(np.int32), kh_ambig=np.where(keep % 2, 1, -1).astype(np.int32))


def _dump(events):
    ev = ak.Array(events)
    keys = np.stack([ak.to_numpy(ev[k]).astype(np.int64) for k in ("run", "subrun", "event")], 1)
    return {"ev": ev, "keys": keys, "fields": set(ev.fields)}


def _events(seed, n=6, **kw):
    rng = np.random.default_rng(seed)
    return [_event(rng, 1430, i, **kw) for i in range(n)]


def test_identical_arms_match_everything_exactly():
    A = _dump(_events(1))
    B = _dump(_events(1)[::-1])                                # same events, another order
    r = SC.compare(A, B)
    assert r["events"] == [6, 6, 6]
    assert r["hits"]["same_digis"]["frac"] == 1.0 and r["hits"]["s"]["sid_equal"]["frac"] == 1.0
    assert r["hits"]["p"]["time_absdiff"]["q"][-1] == 0.0
    for c in ("0", "2"):
        tc = r["timeclusters"][c]
        assert tc["matched"] == 6 and tc["jaccard"]["q"][0] == 1.0 and tc["abs_dt0_lt_0.1"]["frac"] == 1.0
    hs = r["helices"]["0"]
    assert hs["matched"] == 6 and hs["status_equal"]["frac"] == 1.0 and hs["d_rad"]["q"][-1] == 0.0
    f = r["fits"]
    assert f["matched"] == 6 and f["status_equal"]["frac"] == 1.0 and f["pull_lh_t0"]["q"][-1] == 0.0
    assert f["hit_active_equal"]["frac"] == 1.0 and f["hit_ambig_equal"]["frac"] == 1.0
    assert f["hits_joined"] == 6 * 40


def test_dropped_hits_and_shifted_t0_show_in_jaccard_and_dt0():
    A = _dump(_events(2, drop=4, shift_t0=0.5))
    B = _dump(_events(2))
    r = SC.compare(A, B)
    tc = r["timeclusters"]["2"]
    assert tc["matched"] == 6 and tc["jaccard"]["q"][2] == pytest.approx(36 / 40)
    assert tc["dt0"]["q"][2] == pytest.approx(0.5) and tc["abs_dt0_lt_0.1"]["frac"] == 0.0
    assert r["timeclusters"]["0"]["jaccard"]["q"][2] == pytest.approx(36 / 40)   # panel hits expand to their straws
    assert r["fits"]["pull_lh_t0"]["q"][2] == pytest.approx(0.5)
    assert r["fits"]["abs_pull_lt_0.1_lh_t0"]["frac"] == 0.0
    assert r["hits"]["same_digis"]["frac"] == 1.0                  # the digis are unchanged


def test_other_digis_skip_the_one_to_one_hit_comparison_but_still_match_objects():
    a, b = _events(3), _events(3)
    for e in b[:2]:
        e["d_tdc0"] = e["d_tdc0"] + 1                              # another digitisation of two events
    r = SC.compare(_dump(a), _dump(b))
    assert r["hits"]["same_digis"]["frac"] == pytest.approx(4 / 6)
    assert r["hits"]["s"]["sid_equal"]["n"] == 4 * 40
    assert r["fits"]["matched"] == 6


def test_no_match_below_the_shared_straw_threshold():
    a, b = _events(4, n=2), _events(5, n=2)                       # different straws in the same event ids
    r = SC.compare(_dump(a), _dump(b))
    assert r["fits"]["matched"] == 0 and r["timeclusters"]["2"]["matched"] == 0


def test_load_reads_a_written_dump(tmp_path):
    uproot = pytest.importorskip("uproot")
    ev = ak.Array(_events(6, n=3))
    p = tmp_path / "stages.root"
    with uproot.recreate(p) as f:
        f["events"] = {k: ev[k] for k in ev.fields}
    D = SC.load([str(p)])
    assert len(D["keys"]) == 3 and "kl_lh_rad" in D["fields"]
    r = SC.compare(D, D)
    assert r["fits"]["matched"] == 3
