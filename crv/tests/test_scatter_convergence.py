"""The clustered proposal changes shape variance only and retains evidence guards."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim.analyses import scatter_convergence as S
from crvfastsim.analyses import v1compare as VC, v1strata as VS


def test_proposal_only_replaces_shapes():
    group = {"checks": {"crossings": False, "len_ratio": False, "edep_ratio": True},
             "len_ratio": {"valid": True}, "edep_ratio": {"valid": True}}
    shapes = {c: {"valid": True, "chi2_ndf_clustered": 1.} for c in ("len", "edep")}
    got = S.proposal(group, shapes, 1.5)
    assert not got["pass"]
    assert got["checks"] == {"crossings": False, "len_ratio": True, "edep_ratio": True}
    assert not group["checks"]["len_ratio"]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 1.51])
def test_unsupported_or_failed_shape_cannot_pass(value):
    group = {"checks": {"len_ratio": True, "edep_ratio": True},
             "len_ratio": {"valid": True}, "edep_ratio": {"valid": True}}
    shapes = {c: {"valid": True, "chi2_ndf_clustered": value} for c in ("len", "edep")}
    assert not S.proposal(group, shapes, 1.5)["pass"]


def test_exact_moments_uses_v1_event_bar_unit(monkeypatch):
    gen = pd.DataFrame({"run": [1, 1, 1], "subrun": [2, 2, 2], "event": [1, 2, 3]})
    x = pd.DataFrame({"run": [1, 1], "subrun": [2, 2], "event": [1, 2], "bar": [0, 0],
                      "sector": [0, 0], "len": [20., 40.], "edep": [3., 6.]})
    monkeypatch.setattr(VC, "GROUPS", {"test": [0]})
    monkeypatch.setattr(VC, "_arm_crossings", lambda *a: (x, gen))
    geom = type("Geometry", (), {"sector": np.array([0]), "sector_names": ["test_sector"]})()
    edges = {"test": {c: (np.array([0., 30., 100.]), (0., 120.)) for c in ("len", "edep")}}
    n, got = S.exact_moments("a", "b", gen, geom, edges)
    assert n == 3
    mean = VS.summarise(got["group:test"]["means"], n)
    assert mean["n_reference"] == mean["n_candidate"] == 2
    assert mean["crossings_rel"] == mean["crossings_se"] == 0.
    assert got["test_sector"]["means"] == got["group:test"]["means"]


def test_invalid_v1_shape_is_not_repaired_by_proposal():
    group = {"checks": {"len_ratio": False, "edep_ratio": True},
             "len_ratio": {"valid": False}, "edep_ratio": {"valid": True}}
    shapes = {c: {"valid": True, "chi2_ndf_clustered": 0.} for c in ("len", "edep")}
    assert not S.proposal(group, shapes, 1.5)["pass"]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_unsupported_table_cell(value):
    assert S.format_number(value) == "unsupported"


def test_reference_fingerprints_checked_before_use(tmp_path, monkeypatch):
    import json
    from crvfastsim.analyses import transport_validation as TV
    campaign = tmp_path / "thread_a_named1M_20261026"
    campaign.mkdir()
    path = tmp_path / "gen.root"
    path.write_bytes(b"reference")
    record = TV.fingerprint(path)
    monkeypatch.setattr(S, "PREFIX", "/recorded/crv_fastsim/")
    record["path"] = S.PREFIX + "gen.root"
    manifest = {"shards": [{"config": "run2", "subrun": 101, "counts": {"accepted_events": 1},
                            "artifacts": {"gen": record, "crvsteps": record}}]}
    (campaign / "manifest.json").write_text(json.dumps(manifest))
    _, rows = S.reference_shards(tmp_path, [101])
    assert rows[0][1]["expected_accepted"] == 1
    path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        S.reference_shards(tmp_path, [101])


def test_duplicate_subruns_refused_before_output(tmp_path):
    from argparse import Namespace
    with pytest.raises(ValueError, match="duplicate subruns"):
        S.run(Namespace(subruns=[101, 101], outdir=str(tmp_path / "unused")))
    assert not (tmp_path / "unused").exists()


def test_scan_report_rejects_caps(tmp_path):
    import json
    from crvfastsim.analyses import scatter_scan_report as R
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"records": [{"diagnostics": {"complete": False, "cap_hit": True}}]}))
    with pytest.raises(ValueError, match="uncapped"):
        R.run([str(path)], tmp_path / "report")


def test_paired_probes_identity_has_zero_difference(monkeypatch):
    from crvfastsim.analyses import scatter_scan_report as R
    gen = pd.DataFrame({"run": [1, 1, 1], "subrun": [2, 2, 2], "event": [1, 2, 3]})
    x = pd.DataFrame({"run": [1, 1], "subrun": [2, 2], "event": [1, 2], "bar": [0, 0],
                      "sector": [0, 0], "len": [20., 40.], "edep": [3., 6.]})
    geom = type("Geometry", (), {"sector": np.array([0])})()
    rc = type("Config", (), {"geometry": lambda self: geom})()
    monkeypatch.setattr(R.runconfig, "get", lambda *a: rc)
    monkeypatch.setattr(VC, "set_groups", lambda *a: None)
    monkeypatch.setattr(VC, "GROUPS", {"test": [0]})
    monkeypatch.setattr(VC, "_arm_crossings", lambda *a: (x, gen))
    monkeypatch.setattr(R.tables, "read", lambda *a: gen)
    row = {"records": [{"subrun": 2, "paths": {"gen": "gen"}, "candidate": {"path": "a"}}]}
    got = R.paired_probes(row, row)["test"]
    assert got["relative"] == got["se"] == 0.
    assert got["length_mean_relative"] == got["length_mean_se"] == 0.


def test_paired_probes_rejects_unmatched_generators(monkeypatch):
    from crvfastsim.analyses import scatter_scan_report as R
    geom = type("Geometry", (), {})()
    rc = type("Config", (), {"geometry": lambda self: geom})()
    monkeypatch.setattr(R.runconfig, "get", lambda *a: rc)
    monkeypatch.setattr(VC, "set_groups", lambda *a: None)
    a = {"records": [{"subrun": 2, "paths": {"gen": "a"}}]}
    b = {"records": [{"subrun": 2, "paths": {"gen": "b"}}]}
    with pytest.raises(ValueError, match="identical generators"):
        R.paired_probes(a, b)
