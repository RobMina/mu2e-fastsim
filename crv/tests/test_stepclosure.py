"""Regression checks for step-layer diagnostic statistics and the shower-source guard."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim import tier1b
from valhist import stepclosure as C
from valhist import steplayers as S


def test_ratio_error_uses_ratio_residual_not_raw_difference():
    r = np.array([1., 2., 3.])
    c = 2 * r
    assert S.ratio_error(r.sum(), c.sum(), r @ r, c @ c, (c - r) @ (c - r)) == 0
    c = np.array([2., 0., 5.])
    expected = np.linalg.norm(c - c.sum() / r.sum() * r) / r.sum()
    assert S.ratio_error(r.sum(), c.sum(), r @ r, c @ c, (c - r) @ (c - r)) == pytest.approx(expected)
    assert np.isnan(S.ratio_error(0., 1., 0., 1., 1.))


def test_accumulate_preserves_event_clustering_and_pairing():
    r = pd.DataFrame({"grp": ["EX"] * 3, "ev": [1, 1, 2], "w": [1., 2., 4.]})
    c = pd.DataFrame({"grp": ["EX"] * 2, "ev": [1, 3], "w": [5., 6.]})
    row = S.accumulate(r, c, ["grp"]).iloc[0]
    assert row.n_ref == 3 and row.n2_ref == 5 and row.n_cand == 2 and row.n2_cand == 2
    assert row.dn2 == 3
    assert row.e_ref == 7 and row.e2_ref == 25 and row.e_cand == 11 and row.e2_cand == 61
    assert row.de2 == 56


def test_posfill_keeps_root_flow_separate():
    counts, second = S.posfill(np.array([1, 1, 2, 2, 3]), np.array([-1., .2, .4, .8, 1.]), np.array([0., .5, 1.]))
    np.testing.assert_array_equal(counts, [1, 2, 1, 1])
    np.testing.assert_array_equal(second, [1, 2, 1, 1])
    score = S.posstat(counts, second, counts + [100, 0, 0, 100], second + [10000, 0, 0, 10000])
    assert score["chi2ndf_w2"] == 0 and score["flow_cand"] == 202


def test_shower_replacement_leaves_descendant_population_unchanged():
    base = {k: pd.DataFrame({"v": [i]}) for i, k in enumerate(tier1b.LIB_TREES)}
    source = {k: pd.DataFrame({"v": [i + 10]}) for i, k in enumerate(tier1b.LIB_TREES)}
    hybrid = C.replace_showers(base, source)
    for name in tier1b.LIB_TREES:
        assert hybrid[name] is (source if name in C.SHOWER_TREES else base)[name]


def test_primary_guard_selects_identity_and_ignores_total_step_count():
    parts = pd.DataFrame({"run": [1400] * 3, "subrun": [1] * 3, "event": [5] * 3,
                          "key": [1, 2, 3], "pdg": [13, -13, 11], "isprim": [1, 1, 0]})
    steps = pd.DataFrame({"run": [1400] * 3, "subrun": [1] * 3, "event": [5] * 3,
                          "sp": [1, 3, 2], "bar": [4, 4, 4], "t0": [1., 2., 3.], "nstep": [3] * 3})
    primary = C.primary_rows(steps, parts)
    assert list(primary.sp) == [1, 2] and "nstep" not in primary.columns
    pd.testing.assert_frame_equal(primary, C.primary_rows(steps.iloc[::-1].assign(nstep=9), parts))
    parts["parent"] = [-1, -1, 2]
    assert set(C.primary_rows(steps, parts, descendants=True).sp) == {1, 2, 3}


def test_campaign_event_keys_fail_closed_outside_supported_identity():
    frame = pd.DataFrame({"run": [1400, 1401], "subrun": [1, 1], "event": [5, 5]})
    with pytest.raises(ValueError, match="run 1400"):
        S.evkey(frame)
    with pytest.raises(ValueError, match="chain-compatible"):
        S.evkey(frame.iloc[:1].assign(event=1 << 20))
    assert S.evkey(frame.iloc[:1])[0] == (1 << 24) + 5


def test_cluster_score_excludes_flow_and_labels_legacy(tmp_path):
    import json
    rows = [[1, 0, 1, 1, 1, 1, 101, 10201, 101, 10201, 10000, 10000],
            [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0]]
    views = {f"cl_ctype_{v}bin": {"cols": ["ctype", f"{v}bin"], "rows": rows} for v in "yz"}
    views["cl_ctype"] = {"cols": ["ctype"], "rows": [[1, 3, 3, 3, 3, 103, 10203, 103, 10203, 10000, 10000]]}
    report = {"meta": {"position_bins": "root_flow", "y_edges": [0, 1, 1], "z_edges": [0, 1, 1]}, "views": views}
    path = tmp_path / "response.json"
    path.write_text(json.dumps(report))
    result = S.clusters(path)["EX/y"]
    assert result["chi2ndf_w2"] == 0 and result["flow_ref"] == 1 and result["flow_cand"] == 101
    del report["meta"]["position_bins"]
    path.write_text(json.dumps(report))
    result = S.clusters(path)["EX/y"]
    assert result["position_bins"] == "legacy_clipped" and result["chi2ndf_w2"] > 0


def test_exposure_guard_rejects_mismatched_fast_population(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_event_inventory", lambda path, tree:
                        (np.array([[1400, 1, 6 if path == "bad" else 5]]), {"rows": 1}))
    out = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="population differs"):
        C.exposure_guard("gen", [("fast", "bad")], [1], out)
    assert not out.exists()
    C.exposure_guard("gen", [("fast", "good"), ("reference", "ref")], [1], out)
    assert out.exists()


def test_shower_audit_keeps_bad_identity_and_checks_projection(tmp_path, monkeypatch):
    import json
    from mu2efastsim import tables as core
    from valhist import denominator_provenance as P
    donor = tmp_path / "donor.root"
    donor.write_bytes(b"donor")
    gen = tmp_path / "gen.root"
    core.write(str(gen), {"gen": pd.DataFrame({"run": [1400], "subrun": [1], "event": [5], "pdg": [22],
                                               "e": [10.], "px": [10.], "py": [0.], "pz": [0.]})}, as_ttree=True)
    meta = tmp_path / "library.json"
    meta.write_text(json.dumps({"inputs": [str(donor)], "gen": [str(gen)], "generated_scaled_to": 1}))
    meta.with_suffix(".root").write_bytes(b"library")
    kb = tier1b._ke_bin(np.array([10.]))[0]
    showers = pd.DataFrame({"shid": [0], "apdg": [22], "kebin": [kb], "ke": [10.],
                            "pdg_disagreement": [True], "gen_join_missing": [False], "negative_ke": [False],
                            "start_minus_gen_e": [1.]})
    monkeypatch.setattr(P, "donor_showers", lambda path: (showers.copy(), dict(particles={}, gen={}, steps={})))
    lib = {"libshower": showers[["shid", "apdg", "kebin", "ke"]].copy(),
           "libgen": pd.DataFrame({"apdg": [22], "kebin": [kb], "ngen": [1], "nshower": [1]})}
    monkeypatch.setattr(tier1b, "read_library", lambda path: lib)
    out = tmp_path / "audit.json"
    C.shower_audit(meta, out)
    report = json.loads(out.read_text())
    assert report["pdg_mismatch"] == 1 and report["showers"] == 1 and report["projection_identical"]
    lib["libshower"].loc[0, "ke"] = 11.
    with pytest.raises(AssertionError):
        C.shower_audit(meta, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_event_bank_rejects_cross_file_duplicates(monkeypatch):
    monkeypatch.setattr(C, "_event_inventory", lambda path, tree: (np.array([[1400, 1, 5]]), {"rows": 1}))
    with pytest.raises(ValueError, match="cross-file event overlap"):
        C.event_bank(["a", "b"], "dump/gen")
