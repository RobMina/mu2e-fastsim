"""Synthetic paired reference and clustered-error contracts."""
import numpy as np
import pytest

from beamfastsim import target_compare as T


def test_clustered_histogram_errors_cover_pairing_and_flow():
    a = np.array([-2., 0.5, 0.5, 2., 2., 0.5])
    b = np.array([0.5, 0.5, 2., 2., 2., 0.5])
    groups = np.array([0, 0, 1, 1, 2, 2])
    sel = np.ones(6, bool)
    r = T.paired_hist(a, b, sel, sel, groups, np.array([0., 1.]))
    assert r["base"] == [1, 3, 2]
    assert r["cand"] == [0, 3, 3]
    assert r["diff"] == [-1, 0, 1]
    assert np.allclose(r["diff_err"], [1., np.sqrt(3), 1.])
    assert r["clusters"] == 3
    null = T.paired_hist(a, a, sel, sel, groups, np.array([0., 1.]))
    assert null["diff"] == [0, 0, 0] and null["diff_err"] == [0, 0, 0]


def test_one_cluster_does_not_claim_a_variance():
    r = T.paired_hist(np.array([0.5]), np.array([0.5]), np.array([True]), np.array([True]),
                      np.array([0]), np.array([0., 1.]))
    assert r["diff_err"] is None


def test_reference_cli_uses_explicit_process_code_and_highest_cycle(tmp_path):
    uproot = pytest.importorskip("uproot")
    from beamfastsim import cereference as R
    f = tmp_path / "reference.root"
    branches = {k: "float64" for k in R.START}
    branches.update({k: "int32" for k in ("codestop", "evt", "subrun", "code", "pdg")})
    branches.update({k: "13 * float64" for k in R.VD})
    a = {k: np.ones(3) for k in R.START}
    a.update({k: np.arange(3, dtype=np.int32) for k in ("codestop", "evt", "subrun")})
    a.update(code=np.array([167, 166, 179], np.int32), pdg=np.full(3, 11, np.int32))
    a.update({k: np.ones((3, 13)) for k in R.VD})
    with uproot.recreate(f) as root:
        root.mktree("cetree/ntpart", branches).extend(a)
        root.mktree("cetree/ntpart", branches).extend({k: v[1:2] for k, v in a.items()})
    out = tmp_path / "dio.npz"
    R.main([str(f), "--code", "166", "--gencount", "100", "--out", str(out)])
    with np.load(out) as r:
        assert r["rows"].tolist() == [1]
        assert r["gencount"].tolist() == [100]
        assert r["isvd_front"].tolist() == [1]
    with pytest.raises(SystemExit):
        R.main([str(f), "--gencount", "0", "--out", str(out)])


def test_clustered_mean_does_not_treat_shared_rows_as_independent():
    a = T.clustered_mean(np.array([0., 0., 2., 2.]), np.array([0, 0, 1, 1]))
    assert a["mean"] == 1 and a["err"] == 1
    assert a["clusters"] == 2 and a["n"] == 4
    assert T.clustered_mean(np.array([1., 2.]), np.array([0, 0]))["err"] is None
    assert T.clustered_mean(np.array([]), np.array([]))["mean"] is None


def test_same_start_comparison_checks_identity_and_rejects_duplicate_rows(tmp_path, monkeypatch):
    from beamfastsim import cechain as C
    from beamfastsim import cecompare as CC
    n = 4
    ref, fast = tmp_path / "reference.npz", tmp_path / "fast.npz"
    np.savez(ref, x=np.arange(n, dtype=float), y=np.zeros(n), z=np.zeros(n), px=np.zeros(n),
             py=np.zeros(n), pz=np.full(n, 100.), time=np.zeros(n), ifile=np.array([0, 0, 1, 1]))
    state = dict(x=np.c_[np.arange(n), np.zeros(n), np.ones(n)], u=np.tile([0., 0., 1.], (n, 1)),
                 p=np.full(n, 99.), t=np.ones(n), t0=np.zeros(n), ok=np.ones(n, bool), gen=10)
    monkeypatch.setattr(CC, "g4_plane", lambda *a: state)
    def observables(geom, field, s):
        return dict(p=s["p"], cos=np.ones(n), r=s["x"][:, 0], rmax=np.full(n, 600.),
                    tof=s["t"] - s["t0"], t=s["t"], dE=np.zeros(n))
    monkeypatch.setattr(CC, "observables", observables)
    source = dict(idx=np.arange(n), fate=np.full(n, C.PLANE), x=state["x"], u=state["u"],
                  T=np.full(n, np.hypot(99., 0.51099895) - 0.51099895), t=state["t"], t0=state["t0"],
                  nfoil=np.ones(n), nipa=np.zeros(n), nopa=np.zeros(n), path=np.ones(n), wall=1.,
                  x0=np.c_[np.arange(n), np.zeros(n), np.zeros(n)], u0=state["u"], p0=np.full(n, 100.))
    np.savez(fast, **source)
    r = T.compare(ref, str(fast), None, None)
    assert r["start_identity_verified"] and not r["generated_yield_validated"]
    assert r["cuts"]["450.0"]["kept_yield"]["diff"] == 0
    assert r["cuts"]["450.0"]["same_start"]["p"]["err"] == pytest.approx(0)
    source["p0"] = np.full(n, 101.)
    np.savez(fast, **source)
    with pytest.raises(ValueError, match="reference starts"):
        T.compare(ref, str(fast), None, None)
    source["idx"] = np.array([0, 0, 2, 3])
    np.savez(fast, **source)
    with pytest.raises(ValueError, match="unique valid"):
        T.compare(ref, str(fast), None, None)


def test_reference_audit_counts_persisted_events_and_particle_identities(tmp_path):
    uproot = pytest.importorskip("uproot")
    from beamfastsim import target_reference_audit as A
    path = tmp_path / "audit.root"
    a = {k: np.ones(4) for k in ("x", "y", "z", "px", "py", "pz", "time")}
    a.update(run=np.ones(4, np.int32), subrun=np.ones(4, np.int32), evt=np.array([1, 1, 2, 2], np.int32),
             trk=np.array([1, 2, 1, 2], np.int32), code=np.full(4, 179, np.int32),
             pdg=np.array([11, -11, 11, -11], np.int32), isvd=np.ones((4, 13)))
    schema = {k: v.dtype for k, v in a.items() if k != "isvd"}
    schema["isvd"] = "13 * float64"
    with uproot.recreate(path) as root:
        root.mktree("cetree/ntpart", schema).extend(a)
    r = A.audit([path], 179)[0]
    assert r["persisted_events"] == 2 and r["persisted_particle_rows"] == 4
    assert r["primary_electrons"] == r["primary_positrons"] == r["primary_electron_events"] == 2
    assert r["max_electrons_per_event"] == 1 and r["duplicate_particle_ids"] == 0


def test_migrations_distinguish_reach_losses_and_selection_edges():
    from beamfastsim import cechain as C
    sa = np.array([True, True, True, False, False])
    sb = np.array([True, False, False, True, False])
    fate = np.array([C.PLANE, C.PLANE, C.ABSORBED, C.PLANE, C.STOPPED])
    r = T.migration_report(sa, sb, np.array([True, True, True, False, False]), fate, np.array([0, 0, 1, 1, 1]))
    assert r['both'] == 1 and r['base_only'] == 2 and r['cand_only'] == 1 and r['neither'] == 1
    assert r['base_only_at_fast_plane'] == 1 and r['cand_only_without_g4_plane'] == 1
    assert r['base_only_fates']['absorbed'] == 1 and r['base_only_fates']['plane'] == 1
