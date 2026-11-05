"""Synthetic certificates reject ambiguous identities and preserve full stored input."""
import json

import numpy as np
import pytest
import uproot

from beamfastsim.flash_step_input import _hits, crosscheck, main, parse_event_index


DONE = "Art has completed and will exit with status 0.\n"


def header(event, count, *, subrun=2, short=False):
    eid = f"1:{subrun}:{event}" if short else f"run: 1 subRun: {subrun} event: {event}"
    return f"Hits for BeamFilter:Beam in {eid}: ({count})\n"


def track(key, *, copy=8, pdg=11, product="1:42"):
    return f"     trackId: ( {product},{key})  volumeId: {copy}  pdgId: {pdg}  energy deposit: 0  time: 100\n"


def write_tree(path, name, a, *, update=False):
    with (uproot.update(path) if update else uproot.recreate(path)) as root:
        root.mktree(name, {k: v.dtype if v.ndim == 1 else np.dtype((v.dtype, v.shape[1:])) for k, v in a.items()}).extend(a)


def fixture_files(tmp_path, *, double=True):
    generic, legacy, log = (tmp_path / name for name in ("generic.root", "legacy.root", "job.log"))
    t = np.array([1e12 + 0.125, -4., 1695. + .0625], np.float64)
    hits = {f"hits.{k}": np.array([1., 2., 3.], np.float32) + i for i, k in enumerate(("x", "y", "z", "px", "py", "pz"))}
    hits.update({"hits.time": t.astype(np.float32), "hits.pdgId": np.array([11, 22, -11], np.int32),
                 "hits.particleId": np.array([7, 7, 7], np.uint32), "hits.volumeCopy": np.array([8, 9, 8], np.uint32)})
    if double:
        hits["particle.time_"] = t
        hits["particle.state_.fRep"] = np.zeros((3, 6), np.float64)
    old = {k.removeprefix("hits."): v.copy() for k, v in hits.items() if k.startswith("hits.") and k.split(".")[1] not in ("pdgId", "particleId", "volumeCopy")}
    old.update(run=np.ones(3, np.float32), subrun=np.full(3, 2, np.float32), evt=np.array([4, 4, 5], np.float32),
               trk=np.full(3, 7, np.float32), sid=np.array([8, 9, 8], np.float32), pdg=np.array([11, 11, -11], np.float32))
    hits["hits.pdgId"][1] = 11
    ancestry = {k: np.array(v, np.float32) for k, v in dict(run=[1, 1], evt=[5, 4], trk=[7, 7], pk=[0, 0],
                    pdg=[-11, 11], sgt=[1690, 1e12], scc=[13, 4], rcc=[13, 1], egt=[1696, 1e12]).items()}
    write_tree(generic, "input/nt", hits)
    write_tree(legacy, "input/ntvd", old)
    write_tree(generic, "ancestry/ntpssp", ancestry, update=True)
    log.write_text(header(4, 2) + track(7) + track(7, copy=9) + header(5, 1, short=True) + track(7, pdg=-11) + DONE)
    return generic, legacy, log, hits, old, ancestry


@pytest.mark.parametrize("short", [False, True])
def test_parse_expands_event_ids_without_losing_duplicate_keys(short):
    text = header(4, 2, short=short) + track(7) + track(7, copy=9) + header(5, 0, short=short) + header(6, 1, short=short) + track(7, pdg=-11)
    a = parse_event_index(text)
    np.testing.assert_array_equal(a["event"], [4, 4, 6])
    np.testing.assert_array_equal(a["particle_key"], [7, 7, 7])
    np.testing.assert_array_equal(a["event_hit_counts"], [2, 0, 1])
    assert a["event_particle_id"].dtype.names == ("run", "subrun", "event", "particle_key")
    assert len(np.unique(a["event_particle_id"])) == 2
    assert a["product_id"].tolist() == ["1:42"] * 3


@pytest.mark.parametrize("text", [
    "", header(4, 1), track(7) + header(4, 0), header(4, 0) + track(7),
    header(4, 2) + track(7) + header(5, 0) + track(7),
    header(4, 0) + header(4, 0),
    header(4, 1).replace("BeamFilter:Beam", "BeamFilter:BeamExtra") + track(7),
    header(4, 1).replace("subRun:", "subrun:") + track(7),
    header(4, 1).replace("run: 1 subRun: 2 event: 4", "EventID(1,2,4)") + track(7),
    header(4, 1) + track(7).replace("trackId:", "trackID:"),
    header(4, 1) + track(7).replace("volumeId:", "volumeCopy:"),
    header(4, 1) + track(7).replace("pdgId:", "pdg:"),
    header(4, 1) + track(7).replace(",7)", ",-1)"),
])
def test_parse_fails_closed(text):
    with pytest.raises(ValueError):
        parse_event_index(text)


def test_crosscheck_exports_unchanged_full_states_and_hashes(tmp_path):
    generic, legacy, log, hits, old, _ = fixture_files(tmp_path)
    out = tmp_path / "report.json"
    states = tmp_path / "states.npz"
    assert main(["--generic", str(generic), "--legacy", str(legacy), "--log", str(log), "--out", str(out), "--states", str(states)]) == 0
    report = json.loads(out.read_text())
    assert report["rows"] == 3 and report["events"] == 2
    assert report["certified"] and report["particle_state"]["available"]
    assert report["time_storage_dtype"] == "float64"
    assert set(report["artifacts"]) == {"generic", "legacy", "log", "states"}
    assert all(len(v["sha256"]) == 64 for v in report["artifacts"].values())
    with np.load(states, allow_pickle=False) as a:
        assert a["x"].shape == a["p"].shape == (3, 3)
        np.testing.assert_array_equal(a["time_ns"], hits["particle.time_"])
        assert a["time_ns"][0] > 1e12 and a["time_ns"][1] == -4
        np.testing.assert_array_equal(a["time_float32_ns"], old["time"])
        np.testing.assert_array_equal(a["time_float32_spacing_ns"], np.abs(np.spacing(old["time"])))
        np.testing.assert_array_equal(a["time_storage_spacing_ns"], np.abs(np.spacing(hits["particle.time_"])))
        np.testing.assert_array_equal(a["ancestry_row"], [1, 1, 0])
        np.testing.assert_array_equal(a["ancestry_scc"], [4, 4, 13])
        np.testing.assert_array_equal(a["event"], [4, 4, 5])
        np.testing.assert_array_equal(a["input_row"], [0, 1, 2])


def test_float_only_schema_is_explicitly_diagnostic(tmp_path):
    generic, legacy, log, *_ = fixture_files(tmp_path, double=False)
    states = tmp_path / "states.npz"
    report = crosscheck(generic, legacy, log, states)
    assert not report["particle_state"]["available"]
    assert "Float_t" in report["precision_warning"]
    with np.load(states) as a:
        assert a["time_ns"].dtype == np.float32
        assert a["time_float32_spacing_ns"][0] > 1695


@pytest.mark.parametrize("field", ["run", "subrun", "evt", "trk", "sid", "pdg", "x", "y", "z", "time", "px", "py", "pz"])
def test_crosscheck_rejects_any_legacy_field_mismatch(tmp_path, field):
    generic, legacy, log, _, old, _ = fixture_files(tmp_path)
    old[field][0] = np.nextafter(old[field][0], np.float32(np.inf)) if field in ("x", "y", "z", "time", "px", "py", "pz") else old[field][0] + 1
    write_tree(legacy, "input/ntvd", old, update=True)
    states = tmp_path / "rejected.npz"
    with pytest.raises(ValueError, match="mismatch"):
        crosscheck(generic, legacy, log, states)
    assert not states.exists()


@pytest.mark.parametrize("field", ["particleId", "volumeCopy", "pdgId"])
def test_crosscheck_rejects_generic_identity_mismatch(tmp_path, field):
    generic, legacy, log, hits, *_ = fixture_files(tmp_path)
    hits[f"hits.{field}"][0] += 1
    write_tree(generic, "input/nt", hits, update=True)
    with pytest.raises(ValueError, match="Generic/printer"):
        crosscheck(generic, legacy, log)


def test_highest_cycle_selected_and_all_rows_compared(tmp_path):
    generic, legacy, log, hits, old, _ = fixture_files(tmp_path)
    hits["hits.x"] += 9
    old["x"] += 9
    write_tree(generic, "input/nt", hits, update=True)
    write_tree(legacy, "input/ntvd", old, update=True)
    report = crosscheck(generic, legacy, log)
    assert report["generic_tree"]["object_path"].endswith("nt;2")
    assert report["legacy_tree"]["object_path"].endswith("ntvd;2")
    old["x"] = old["x"][::-1]
    write_tree(legacy, "input/ntvd", old, update=True)
    with pytest.raises(ValueError, match="Exact Float_t"):
        crosscheck(generic, legacy, log)


@pytest.mark.parametrize("value", [np.nan, 1.5, 2**24])
def test_float_identity_ambiguity_rejected(tmp_path, value):
    generic, legacy, log, _, old, _ = fixture_files(tmp_path)
    old["trk"][0] = value
    write_tree(legacy, "input/ntvd", old, update=True)
    with pytest.raises(ValueError, match="Float_t"):
        crosscheck(generic, legacy, log)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "pdg"])
def test_ancestry_join_is_event_qualified_and_strict(tmp_path, kind):
    generic, legacy, log, _, _, a = fixture_files(tmp_path)
    if kind == "missing":
        a["evt"][0] = 999
    elif kind == "duplicate":
        a["evt"][0] = a["evt"][1]
    else:
        a["pdg"][0] = 13
    write_tree(generic, "ancestry/ntpssp", a, update=True)
    with pytest.raises(ValueError, match="[Aa]ncestry"):
        crosscheck(generic, legacy, log)


def test_multisubrun_including_zero_hit_event_is_not_certified(tmp_path):
    generic, legacy, log, *_ = fixture_files(tmp_path)
    log.write_text(log.read_text() + header(20, 0, subrun=3))
    with pytest.raises(ValueError, match="one printer-certified subrun"):
        crosscheck(generic, legacy, log)


@pytest.mark.parametrize("kind", ["missing_success", "double_mismatch", "nan", "wrong_dtype", "row_count"])
def test_crosscheck_schema_and_completion_fail_closed(tmp_path, kind):
    generic, legacy, log, hits, *_ = fixture_files(tmp_path)
    if kind == "missing_success":
        log.write_text(log.read_text().replace(DONE, "Art has completed and will exit with status 1.\n"))
    elif kind == "double_mismatch":
        hits["particle.time_"][0] *= 2
    elif kind == "nan":
        hits["hits.x"][0] = np.nan
    elif kind == "wrong_dtype":
        hits["hits.particleId"] = hits["hits.particleId"].astype(np.float32)
    else:
        hits = {k: v[:2] for k, v in hits.items()}
    if kind != "missing_success":
        write_tree(generic, "input/nt", hits, update=True)
    with pytest.raises(ValueError):
        crosscheck(generic, legacy, log)


def test_leaf_list_hits_are_read_without_guessing():
    a = np.zeros(2, dtype=[(k, "f4") for k in ("x", "y", "z", "time", "px", "py", "pz")] +
                 [("pdgId", "i4"), ("particleId", "u4"), ("volumeCopy", "u4")])

    class Branch:
        def array(self, library):
            return a

    class Tree:
        def keys(self, recursive):
            return ["hits"]

        def __getitem__(self, key):
            return Branch()

    columns, layout = _hits(Tree())
    assert layout == "hits leaf-list"
    assert columns["particleId"].dtype == np.uint32


def test_cli_does_not_overwrite_inputs(tmp_path):
    generic, legacy, log, *_ = fixture_files(tmp_path)
    with pytest.raises(SystemExit):
        main(["--generic", str(generic), "--legacy", str(legacy), "--log", str(log),
              "--out", str(log), "--states", str(tmp_path / "states.npz")])


def test_zero_hit_event_is_retained_in_export(tmp_path):
    generic, legacy, log, hits, old, ancestry = fixture_files(tmp_path)
    write_tree(generic, "input/nt", {k: v[:0] for k, v in hits.items()}, update=True)
    write_tree(legacy, "input/ntvd", {k: v[:0] for k, v in old.items()}, update=True)
    write_tree(generic, "ancestry/ntpssp", {k: v[:0] for k, v in ancestry.items()}, update=True)
    log.write_text(header(9, 0) + DONE)
    states = tmp_path / "empty.npz"
    report = crosscheck(generic, legacy, log, states)
    assert report["rows"] == 0 and report["events"] == 1
    with np.load(states) as a:
        assert a["x"].shape == (0, 3)
        assert a["event_headers"]["event"].tolist() == [9]
        assert a["event_hit_counts"].tolist() == [0]


def test_unrecognized_particle_split_names_are_not_guessed(tmp_path):
    generic, legacy, log, hits, *_ = fixture_files(tmp_path)
    hits["particle.mystery_time"] = hits.pop("particle.time_")
    write_tree(generic, "input/nt", hits, update=True)
    report = crosscheck(generic, legacy, log)
    assert not report["particle_state"]["available"]
    assert "particle.mystery_time" in report["particle_state"]["available_branches"]
    assert report["time_storage_dtype"] == "float32"


def test_missing_ancestry_is_not_certified(tmp_path):
    generic, legacy, log, hits, *_ = fixture_files(tmp_path)
    write_tree(generic, "input/nt", hits)
    with pytest.raises(KeyError):
        crosscheck(generic, legacy, log)


def test_nonexact_collection_label_is_rejected_even_with_extra_process(tmp_path):
    generic, legacy, log, *_ = fixture_files(tmp_path)
    log.write_text(log.read_text().replace("BeamFilter:Beam in", "BeamFilter:Beam:other in"))
    with pytest.raises(ValueError, match="collection label"):
        crosscheck(generic, legacy, log)


def test_explicit_ancestry_omission_keeps_full_event_identity(tmp_path):
    generic, legacy, log, hits, old, _ = fixture_files(tmp_path)
    text = log.read_text().replace(header(5, 1, short=True), header(5, 1, short=True, subrun=3))
    log.write_text(text)
    old["subrun"][-1] = 3
    write_tree(legacy, "input/ntvd", old, update=True)
    with pytest.raises(ValueError, match="subrun"):
        crosscheck(generic, legacy, log)
    report = crosscheck(generic, legacy, log, include_ancestry=False)
    assert report["certified"] and not report["ancestry_tree"]["certified"]
    assert report["rows"] == 3
