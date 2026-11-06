"""Synthetic recorded Beam / exact-key ancestry regressions; no private inputs."""
import json
import sys
import numpy as np
import pytest
import uproot

from beamfastsim import tsreference as ts


def rows(trk, parent=None, pdg=None, code=None, evt=None):
    n = len(trk)
    a = {k: np.zeros(n) for k in ("x", "y", "z", "px", "py", "pz", "time",
         "xstop", "ystop", "zstop", "pxstop", "pystop", "pzstop", "tstop", "pstop")}
    a.update(run=np.ones(n, dtype="i8"), subrun=np.ones(n, dtype="i8"),
             evt=np.ones(n, dtype="i8") if evt is None else np.array(evt, dtype="i8"),
             trk=np.array(trk, dtype="i8"),
             parent_id=np.zeros(n, dtype="i8") if parent is None else np.array(parent, dtype="i8"),
             pdg=np.full(n, 13, dtype="i8") if pdg is None else np.array(pdg, dtype="i8"),
             code=np.full(n, 14, dtype="i8") if code is None else np.array(code, dtype="i8"),
             codestop=np.full(n, 54, dtype="i8"))
    a["xstop"][:] = 3904
    a["zstop"][:] = ts.SOURCE_Z
    a["pstop"][:] = a["pzstop"][:] = 70
    a["tstop"][:] = 42
    return a


def beam(a, take=None, downstream=False):
    """Float32 TNtuple pre-step fixture, distinct from creation AND terminal."""
    if take is None:
        take = np.arange(len(a["trk"]))
    b = {k: a[k][take].astype("f4") for k in ts.IDS + ("pdg",)}
    for k, v in dict(x=-3904 if downstream else 3904, y=12, z=4179 if downstream else -4004,
                     px=3, py=4, pz=55, time=140 if downstream else 40).items():
        b[k] = np.full(len(take), v, dtype="f4")
    return b


def ds(a, indices):
    a["xstop"][indices] = -3904
    a["zstop"][indices] = 4186.25
    a["tstop"][indices] = 142
    return a


def build(s, d, take=None, ends=(), **kwargs):
    return ts.build(s, d, source_beam=beam(s, take),
                    downstream_beam=beam(d, list(ends), downstream=True), **kwargs)


def stock_files(tmp_path, s, d, sb=None, db=None):
    paths = [tmp_path / name for name in ("source.root", "downstream.root")]
    for path, a, b in zip(paths, (s, d), (sb, db)):
        with uproot.recreate(path) as f:
            f.mktree("mutree/ntpart", a)
            if b is not None:
                f.mktree("mutree/nttvd", b)
    return paths


def test_restart_decay_recorded_states_and_zero_loss_denominator():
    s = rows([7, 8, 9], parent=[1, 2, 3], pdg=[13, -211, 13], code=[14, 101, 14])
    d = rows([7, 8, 91, 92, 99], parent=[1, 2, 7, 8, 92],
             pdg=[13, -211, 13, -211, 13], code=[14, 101, 56, 56, 14])
    d["codestop"][3] = 14
    ds(d, [2, 4])
    out, report = build(s, d, ends=[2, 4], event_id=np.array([[1, 1, 1], [1, 1, 2]]))
    np.testing.assert_array_equal(out["id"][:, 3], [7, 8, 9])
    np.testing.assert_array_equal(out["stock_reached"], [True, True, False])
    np.testing.assert_array_equal(out["stock_restarted"], [True, True, False])
    np.testing.assert_array_equal(out["stock_id"][:, 3], [91, 99, -1])
    np.testing.assert_array_equal(out["pdg"], [13, -211, 13])
    np.testing.assert_array_equal(out["stock_pdg"], [13, 13, 0])
    np.testing.assert_array_equal(out["x"][0], [3904, 12, -4004])
    np.testing.assert_array_equal(out["p"][0], [3, 4, 55])
    assert out["t"][0] == 40 != s["time"][0] and out["t"][0] != s["tstop"][0]
    assert out["stock_x"][1, 2] == 4179 != d["zstop"][4]
    assert out["stock_t"][1] == 140 and np.isnan(out["stock_x"][2]).all()
    assert report["max_ancestry_depth"] == 2
    assert report["events_without_source"] == 1
    assert report["state_kind"] == "recorded pre-step Beam"
    assert out["id"].dtype == np.int64


def test_empty_downstream_and_save_load(tmp_path):
    out, report = build(rows([1, 2]), rows([]))
    assert len(out["id"]) == 2 and not out["stock_reached"].any()
    assert not out["stock_restarted"].any()
    assert np.isnan(out["stock_p"]).all() and out["stock_x"].shape == (2, 3)
    assert report["stock_not_reached"] == 2
    path = tmp_path / "ref.npz"
    ts.save(out, path)
    for k, v in ts.load(path).items():
        np.testing.assert_array_equal(v, out[k])


@pytest.mark.parametrize("source", [rows([]), rows([1])])
def test_empty_source_and_ancestors_do_not_enter_cohort(source):
    out, report = build(source, rows([]), take=[])
    assert out["id"].shape == (0, 4) and out["stock_x"].shape == (0, 3)
    assert report["events_without_source"] == len(source["trk"])


def test_recorded_selection_has_no_plane_time_direction_or_momentum_cuts():
    s = rows([1, 2, 3, 4, 5, 6], pdg=[13, -13, 211, -211, 13, 11])
    s["code"][:] = 101
    s["zstop"][1] += 1
    s["pzstop"][2] = -70
    s["codestop"][3] = 14
    sb = beam(s); sb["pz"][:] = [0, -1, 1, 400, 1000, 0]
    sb["z"][:] = np.arange(6); sb["time"][:] = np.arange(6) * 10000
    out, _ = ts.build(s, rows([]), source_beam=sb, downstream_beam=beam(rows([])))
    np.testing.assert_array_equal(out["id"][:, 3], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(out["p"][:, 2], [0, -1, 1, 400, 1000])
    old, report = ts.build(s, rows([]), terminal_only=True)
    np.testing.assert_array_equal(old["id"][:, 3], [1, 5])
    assert report["state_kind"] == "terminal diagnostic"


def test_endpoint_selection_has_no_terminal_region_or_stopcode_cut():
    s = rows([7])
    d = rows([7, 9], parent=[0, 7], code=[14, 56]); d["codestop"][:] = 14
    db = beam(d, [1], downstream=True); db["z"][:] = 123; db["pz"][:] = -1; db["time"][:] = 1e6
    out, _ = ts.build(s, d, source_beam=beam(s), downstream_beam=db)
    assert out["stock_reached"].tolist() == [True]
    assert out["stock_x"][0, 2] == 123 and out["stock_p"][0, 2] == -1


@pytest.mark.parametrize("key", ["run", "subrun", "evt", "parent_id"])
def test_full_identity_required_not_equal_track_id(key):
    s = rows([7], parent=[1])
    d = rows([7, 9], parent=[1, 7], code=[14, 56])
    if key == "parent_id":
        d[key][0] = 999
    else:
        d[key][:] += 1
    with pytest.raises(ValueError, match="missing source root|unmatched"):
        build(s, d, ends=[1])


def test_parent_missing_and_equal_source_endpoint_id_not_a_shortcut():
    with pytest.raises(ValueError, match="unmatched"):
        build(rows([7]), rows([7], parent=[999], code=[56]), ends=[0])


def test_non_mu_pi_rows_are_not_cohort_or_endpoints():
    s = rows([7, 8], pdg=[13, 11]); d = rows([10], pdg=[2212])
    out, report = build(s, d, ends=[0])
    assert out["id"][:, 3].tolist() == [7] and not out["stock_reached"].any()
    assert report["unrelated_new_descendants"] == 1


def test_unrelated_nonendpoint_daughters_and_restarts_are_reported_not_gating():
    d = rows([7, 8, 9, 10, 11], parent=[0, 7, 8, 999, 999], code=[14, 56, 99, 14, 56])
    out, report = build(rows([7, 12]), d, ends=[1])
    assert out["stock_reached"].tolist() == [True, False]
    assert report["unrelated_new_descendants"] == 3
    assert report["unmapped_restarts"] == 1
    with pytest.raises(ValueError, match="unsupported ancestry"):
        build(rows([7, 12]), d, ends=[2])


@pytest.mark.parametrize("pdg", [13, 211])
def test_wrong_restart_or_decay_species_fail(pdg):
    d = rows([7, 8, 9], parent=[0, 7, 8], pdg=[13, 211, pdg], code=[14, 56, 14])
    with pytest.raises(ValueError, match="unsupported ancestry"):
        build(rows([7]), d, ends=[2])


def test_cyclic_endpoint_ancestry_fails():
    d = rows([8, 9], parent=[9, 8], pdg=[13, -211], code=[14, 56])
    with pytest.raises(ValueError, match="unsupported ancestry|cyclic ancestry"):
        build(rows([7]), d, ends=[0])


def test_multiple_endpoints_raise_not_collapse():
    d = rows([7, 8, 9], parent=[0, 7, 7], code=[14, 56, 56])
    with pytest.raises(ValueError, match="multiple DS endpoints"):
        build(rows([7]), d, ends=[1, 2])


def test_duplicate_particle_and_recorded_keys_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build(rows([7, 7]), rows([]))
    with pytest.raises(ValueError, match="duplicate"):
        build(rows([7]), rows([]), take=[0, 0])
    d = rows([7, 8], parent=[0, 7], code=[14, 56])
    with pytest.raises(ValueError, match="duplicate"):
        build(rows([7]), d, ends=[1, 1])


@pytest.mark.parametrize("key", ts.IDS)
@pytest.mark.parametrize("value", [1.5, np.nan, np.inf, 2**24, 2**24 + 2, -(2**24)])
def test_float_tntuple_ids_require_integral_exact_precision_range(key, value):
    s = rows([7]); sb = beam(s); sb[key][0] = value
    with pytest.raises(ValueError, match="nonintegral/nonfinite|precision range"):
        ts.build(s, rows([]), source_beam=sb, downstream_beam=beam(rows([])))


def test_unselected_nuclear_pdg_does_not_block_muon_cohort():
    s = rows([7]); sb = beam(rows([7, 9], pdg=[13, 1000260560]))
    out, report = ts.build(s, rows([]), source_beam=sb, downstream_beam=beam(rows([])))
    assert out["pdg"].tolist() == [13]
    assert report["source_beam_rows"] == 2


def test_matching_scalar_ntpart_does_not_override_float_precision_check():
    s = rows([2**24]); sb = beam(s)
    with pytest.raises(ValueError, match="precision range"):
        ts.build(s, rows([]), source_beam=sb, downstream_beam=beam(rows([])))
    s = rows([2**24 - 1]); out, _ = build(s, rows([]))
    assert out["id"][0, 3] == 2**24 - 1
    s["parent_id"] = np.array([1.5])
    with pytest.raises(ValueError, match="nonintegral"):
        build(s, rows([]))


@pytest.mark.parametrize("change", ["trk", "pdg"])
def test_recorded_rows_must_match_scalar_identity(change):
    s = rows([7]); sb = beam(s); sb[change][0] = 8 if change == "trk" else -13
    with pytest.raises(ValueError, match="identity unmatched|PDG mismatch"):
        ts.build(s, rows([]), source_beam=sb, downstream_beam=beam(rows([])))


def test_default_requires_nttvd_and_explicit_terminal_diagnostic(tmp_path):
    paths = stock_files(tmp_path, rows([7]), rows([]))
    with pytest.raises(ValueError, match="nttvd"):
        ts.read(*paths)
    with pytest.raises(ValueError, match="nttvd required"):
        ts.build(rows([7]), rows([]))
    out, report = ts.read(*paths, terminal_only=True)
    assert out["t"].tolist() == [42] and report["state_kind"] == "terminal diagnostic"


def test_highest_cycle_recorded_read_and_observed_event_union(tmp_path):
    s, d = rows([1]), rows([])
    paths = stock_files(tmp_path, s, d, beam(s), beam(d))
    s = rows([2, 3], evt=[2, 3])
    with uproot.update(paths[0]) as f:
        f.mktree("mutree/ntpart", s)
        f.mktree("mutree/nttvd", beam(s, [0]))
        f.mktree("mutree/ntvd", {"run": np.array([1.]), "subrun": np.array([1.]), "evt": np.array([99.])})
    out, report = ts.read(*paths)
    assert report["source_file"]["tree"] == "mutree/ntpart;2"
    assert report["source_file"]["beam"]["tree"] == "mutree/nttvd;2"
    np.testing.assert_array_equal(out["event_id"][:, 2], [2, 3, 99])
    assert report["events_without_source"] == 2 and len(out["id"]) == 1
    assert not report["event_inventory_complete"] and not report["shard_event_inventory_complete"]


def art_file(tmp_path, subrun, count):
    path = tmp_path / f"sim.test.TS1Beam.MDC2025ay-001.000001_{subrun:08d}.art"
    with uproot.recreate(path) as f:
        f.mktree("Events", {"opaque": np.arange(count)})
    return path


def test_art_denominator_keeps_empty_source_events_zero_shards_and_filtered_ds(tmp_path):
    s, d = rows([7]), rows([])
    paths = stock_files(tmp_path, s, d, beam(s), beam(d))
    art = [art_file(tmp_path, 1, 2), art_file(tmp_path, 2, 4), art_file(tmp_path, 3, 0)]
    with uproot.update(art[0]) as f:
        f.mktree("Events", {"opaque": np.arange(10)})
    out, report = ts.read(*paths, source_art_paths=art)
    np.testing.assert_array_equal(out["parent_shard_ids"], [[1, 1], [1, 2], [1, 3]])
    np.testing.assert_array_equal(out["parent_shard_event_counts"], [10, 4, 0])
    assert report["source_art_files"][0]["tree"] == "Events;2"
    assert report["parent_events_total"] == 14 and report["events_without_source"] == 13
    assert len(out["event_id"]) == 1 and report["parent_events_observed"] == 1
    assert not report["event_inventory_complete"] and report["shard_event_inventory_complete"]
    assert out["stock_reached"].tolist() == [False] and len(out["id"]) == 1


def test_art_inventory_refuses_duplicate_shards_or_insufficient_counts(tmp_path):
    s, d = rows([7]), rows([])
    paths = stock_files(tmp_path, s, d, beam(s), beam(d))
    art = art_file(tmp_path, 1, 0)
    with pytest.raises(ValueError, match="does not cover"):
        ts.read(*paths, source_art_paths=[art])
    with pytest.raises(ValueError, match="duplicate"):
        ts.art_inventory([art, art])
    other = art_file(tmp_path, 2, 10)
    with pytest.raises(ValueError, match="does not cover"):
        ts.read(*paths, source_art_paths=[other])


def test_cli_source_art_nargs_and_json(tmp_path, monkeypatch, capsys):
    s, d = rows([7]), rows([])
    source, downstream = stock_files(tmp_path, s, d, beam(s), beam(d))
    arts = [art_file(tmp_path, 1, 3), art_file(tmp_path, 2, 0)]
    out = tmp_path / "ref.npz"
    monkeypatch.setattr(sys, "argv", ["tsreference", "--source", str(source), "--downstream", str(downstream),
                                    "--out", str(out), "--source-art", *map(str, arts)])
    ts.main()
    assert json.loads(capsys.readouterr().out)["parent_events_total"] == 3
    assert ts.load(out)["parent_shard_event_counts"].tolist() == [3, 0]


def test_empty_cohort_retains_art_shard_denominator(tmp_path):
    s, d = rows([]), rows([])
    paths = stock_files(tmp_path, s, d, beam(s), beam(d))
    with pytest.raises(ValueError, match="at least one file"):
        ts.read(*paths, source_art_paths=[])
    out, report = ts.read(*paths, source_art_paths=[art_file(tmp_path, 1, 8)])
    assert out["id"].shape == (0, 4) and out["event_id"].shape == (0, 3)
    assert report["events_without_source"] == 8
    assert out["parent_shard_event_counts"].tolist() == [8]


def test_shuffled_recorded_rows_align_by_exact_keys_not_table_order():
    s = rows([7, 8], evt=[1, 2])
    d = rows([7, 8, 9, 10], parent=[0, 0, 7, 8], evt=[1, 2, 1, 2], code=[14, 14, 56, 56])
    sb, db = beam(s, [1, 0]), beam(d, [2, 3], downstream=True)
    db["time"][:] = [100, 200]
    out, _ = ts.build(s, d, source_beam=sb, downstream_beam=db)
    assert out["id"][:, 3].tolist() == [8, 7]
    assert out["stock_id"][:, 3].tolist() == [10, 9]
    assert out["stock_t"].tolist() == [200, 100]


def test_lookup_wide_keys_not_float_packed():
    index = np.array([[2**54, 1, 2, 9], [2**54 + 1, 1, 2, 9]], dtype="i8")
    np.testing.assert_array_equal(ts.lookup(index, index[::-1]), [1, 0])
    np.testing.assert_array_equal(ts.ids(dict(zip(ts.IDS, index.T))), index)


def test_analyzer_services_have_concrete_seed():
    from pathlib import Path
    import re
    fcl = Path(ts.__file__).with_name("tsread.fcl").read_text()
    assert "services: @local::Services.Sim" in fcl
    seeds = re.findall(r"^services\.SeedService\.baseSeed:\s*(\d+)\s*$", fcl, re.M)
    assert len(seeds) == 1 and int(seeds[0]) > 0
    assert fcl.index("services.SeedService.baseSeed:") > fcl.index("services: @local::Services.Sim")
    assert "producers:" not in fcl
