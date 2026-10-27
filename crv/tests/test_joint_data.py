"""Local synthetic EventNtuples only: passive data capture, identity and receipts."""
import copy
import hashlib
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from crvfastsim.objective import crvline as CL, crvresponse as R, joint_data as D


def sample():
    """One selected EX cluster and one truly empty event; typed jagged arrays."""
    counts = np.array([1, 0])
    c = ak.Array({
        "crvcoincs.sectorType": ak.unflatten(np.array([1]), counts),
        "crvcoincs.time": ak.unflatten(np.array([100.]), counts),
        "crvcoincs.PEs": ak.unflatten(np.array([60.]), counts),
        "crvcoincs.hitPosAndTimeCalculated": ak.unflatten(np.array([True]), counts),
        "crvcoincs.pos.fCoordinates.fX": ak.unflatten(np.array([0.]), counts),
        "crvcoincs.pos.fCoordinates.fY": ak.unflatten(np.array([CL.Y_REF]), counts),
        "crvcoincs.nLayers": ak.unflatten(np.array([3]), counts),
        "crvcoincs.nHits": ak.unflatten(np.array([3]), counts),
        "crvcoincs.PEsPerLayer[4]": ak.unflatten(np.array([[20., 20., 20., 0.]]), counts),
    })
    pcounts = np.array([3, 0])
    p = ak.Array({
        "crvpulses.barId": ak.unflatten(np.array([0, 16, 32]), pcounts),
        "crvpulses.PEs": ak.unflatten(np.array([20., 20., 20.]), pcounts),
        "crvpulses.crvHitIndex": ak.unflatten(np.array([0, 0, 0]), pcounts),
        "crvpulses.sequenceIndex": ak.unflatten(np.array([0, 0, 0]), pcounts),
        "crvpulses.SiPMId": ak.unflatten(np.array([0, 1, 0]), pcounts),
        "crvpulses.time": ak.unflatten(np.array([100., 101., 102.]), pcounts),
        "crvpulses.pos.fCoordinates.fX": ak.unflatten(np.array([0., 0., 0.]), pcounts),
        "crvpulses.pos.fCoordinates.fY": ak.unflatten(CL.Y_REF + np.array([-30., 0., 30.]), pcounts),
        "crvpulses.pos.fCoordinates.fZ": ak.unflatten(np.array([0., 0., 0.]), pcounts),
    })
    keys = np.array([[125026, 7, 2 ** 53 + 1], [125026, 7, 2 ** 53 + 2]], dtype=np.int64)
    return c, p, keys


def write_root(path, keys=None, prefix="evtinfo/", missing=None, cycles=False, empty=False):
    c, p, default = sample()
    keys = default if keys is None else keys
    if empty:
        c, p, keys = c[:0], p[:0], keys[:0]
    columns = {field: c[field] for field in c.fields}
    columns.update({field: p[field] for field in p.fields})
    columns.update({prefix + field: keys[:, index] for index, field in enumerate(("run", "subrun", "event"))
                    if field != missing})
    with uproot.recreate(path) as root:
        if cycles:
            root.mkrntuple(R.TREE, {name: values[:1] for name, values in columns.items()})
        root.mkrntuple(R.TREE, columns)
    return c, p, keys


def reference(path, root, chunk=1):
    state = R.new_state()
    # Uproot's synthetic RNTuple writer supports var * 4 PE layers; its literal
    # [4] field requires an exact filter rather than accumulate_file's TTree API.
    with uproot.open(root) as file:
        tree = file[R.TREE]
        for start in range(0, tree.num_entries, chunk):
            c = tree.arrays(filter_name=lambda name: name in CL.COINC_BR,
                            entry_start=start, entry_stop=start + chunk)
            p = tree.arrays(filter_name=lambda name: name in CL.PULSE_BR,
                            entry_start=start, entry_stop=start + chunk)
            R.accumulate_arrays(c, p, state, 400000.)
            state.setdefault("window_ns", []).append(400000.)
    state["files"] = 1
    path.write_text(json.dumps(state))
    return state


def test_highest_cycle_passive_capture_and_full_integer_empty_identity(tmp_path, monkeypatch):
    root = tmp_path / "data.root"
    _, _, keys = write_root(root, cycles=True)
    ref = tmp_path / "ref.json"
    expected = reference(ref, root)
    # Frozen scoring references may have absent/zero supplemental timing diagnostics.
    frozen = copy.deepcopy(expected)
    frozen.pop("dtp")
    frozen["ftp"] = R.new_state()["ftp"]
    ref.write_text(json.dumps(frozen))
    monkeypatch.setattr(R, "shift_times", lambda *args: pytest.fail("must never shift DATA times"))
    original = R.accumulate_arrays
    measured = []
    original_add = D.joint.JointCapture.add

    def accumulate(c, p, state, window):
        result = original(c, p, state, window)
        measured.append(result)
        return result

    def add(self, c, p, result, event_keys, source_index=0):
        assert result is measured[-1]  # Reuse, never refit or reconstruct the returned o.
        return original_add(self, c, p, result, event_keys, source_index=source_index)

    monkeypatch.setattr(R, "accumulate_arrays", accumulate)
    monkeypatch.setattr(D.joint.JointCapture, "add", add)
    out = tmp_path / "capture"
    summary = D.capture_data([root], out, chunk=1, reference_accumulator=ref)
    actual = json.loads((out / "response_state.json").read_text())
    assert actual == R.merge_states([expected])
    assert summary["complete"] and summary["counts"]["events"] == 2
    assert len(summary["raw_shards"]) == 2
    with np.load(out / summary["raw_shards"][1]["path"], allow_pickle=False) as rows:
        np.testing.assert_array_equal(rows["event_event"], np.repeat(keys[1, 2], 4))
        np.testing.assert_array_equal(rows["event_run"], np.repeat(keys[1, 0], 4))
        np.testing.assert_array_equal(rows["event_subrun"], np.repeat(keys[1, 1], 4))
        assert len(rows["hit_event"]) == 0
        assert not rows["event_selected"].any()
    provenance = json.loads((out / "provenance.json").read_text())
    source = provenance["inputs"][0]
    assert source["tree_cycle"] == 2 and source["entries_available"] == 2
    assert source["entry_ranges"] == [[0, 1], [1, 2]]
    assert source["reference_comparison"]["status"] == "exact"
    assert source["sha256"] == hashlib.sha256(root.read_bytes()).hexdigest()
    assert summary["event_key_sha256"] == hashlib.sha256(keys.astype("<i8").tobytes()).hexdigest()
    assert provenance["time_shift_ns"] is None
    assert len(provenance["source_code"]) == 6
    assert {p.suffix for p in out.iterdir()} == {".npz", ".json"}


@pytest.mark.parametrize("prefix", ["evtinfo/", "evtinfo.", "evtinfo/evtinfo.", ""])
def test_identity_discovery_real_schemas(tmp_path, prefix):
    path = tmp_path / "data.root"
    _, _, keys = write_root(path, prefix=prefix)
    with uproot.open(path) as root:
        tree, _ = D._highest_tree(root)
        branches = D.discover_event_branches(tree)
        np.testing.assert_array_equal(D.read_event_keys(tree, branches, 0, 2), keys)


@pytest.mark.parametrize("missing", ["run", "subrun", "event"])
def test_missing_identity_fails_closed(tmp_path, missing):
    root = tmp_path / "data.root"
    write_root(root, missing=missing)
    out = tmp_path / "capture"
    with pytest.raises(ValueError, match="full run/subrun/event"):
        D.capture_data([root], out)
    assert not (out / "summary.json").exists()


@pytest.mark.parametrize("kind", ["float", "negative", "overflow"])
def test_invalid_identity_fails_closed(tmp_path, kind):
    root = tmp_path / "data.root"
    keys = sample()[2].copy()
    if kind == "float":
        keys = keys.astype(float)
    elif kind == "negative":
        keys[0, 2] = -1
    else:
        keys = keys.astype(np.uint64)
        keys[0, 2] = 2 ** 63
    write_root(root, keys=keys)
    with pytest.raises(ValueError, match="identity"):
        D.capture_data([root], tmp_path / "capture")


def test_ambiguous_schema_refused():
    class Tree:
        def keys(self, **kwargs):
            return [prefix + field for prefix in ("evtinfo/", "") for field in ("run", "subrun", "event")]
    with pytest.raises(ValueError, match="unambiguous"):
        D.discover_event_branches(Tree())


@pytest.mark.parametrize("across_files", [False, True])
def test_duplicates_across_chunks_and_files_refused(tmp_path, across_files):
    root = tmp_path / "data.root"
    keys = sample()[2].copy()
    if not across_files:
        keys[1] = keys[0]
    write_root(root, keys=keys)
    inputs = [root]
    if across_files:
        other = tmp_path / "other.root"
        write_root(other, keys=keys)
        inputs.append(other)
    out = tmp_path / "capture"
    with pytest.raises(ValueError, match="duplicate full"):
        D.capture_data(inputs, out, chunk=1)
    assert not (out / "summary.json").exists()
    assert not (out / "provenance.json").exists()


def test_reference_mismatch_fails_no_completion_receipt(tmp_path):
    root = tmp_path / "data.root"
    write_root(root)
    ref = tmp_path / "ref.json"
    state = reference(ref, root)
    state["h"]["pe_layer_EX"][0] += 1
    ref.write_text(json.dumps(state))
    out = tmp_path / "capture"
    with pytest.raises(ValueError, match="scoring field h"):
        D.capture_data([root], out, reference_accumulator=ref)
    assert not (out / "summary.json").exists()


def test_entry_stop_and_cli(tmp_path, capsys):
    root = tmp_path / "data.root"
    write_root(root)
    ref = tmp_path / "ref.json"
    reference(ref, root)
    out = tmp_path / "capture"
    report = D.main(["--inputs", str(root), "--out", str(out), "--entry-stop", "1",
                     "--chunk", "1", "--reference-accumulator", str(ref)])
    assert report["counts"]["events"] == 1
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["options"]["window_ns"] == 400000.
    assert not provenance["full_files"]
    assert provenance["inputs"][0]["reference_comparison"]["status"] == "skipped_partial_capture"
    assert json.loads(capsys.readouterr().out)["complete"]
    with pytest.raises(FileExistsError):
        D.capture_data([root], out)


@pytest.mark.parametrize("empty_file", [False, True])
def test_zero_events_preserve_schema(tmp_path, empty_file):
    root = tmp_path / "data.root"
    write_root(root, empty=empty_file)
    out = tmp_path / "capture"
    report = D.capture_data([root], out, entry_stop=None if empty_file else 0)
    assert report["counts"]["events"] == 0 and report["counts"]["files"] == 1
    with np.load(out / report["raw_shards"][0]["path"]) as rows:
        assert rows["event_event"].shape == (0,)


@pytest.mark.parametrize("kwargs", [{"chunk": 0}, {"entry_stop": -1}, {"window_ns": np.nan},
                                    {"window_ns": 0}, {"reference_accumulator": ["a", "b"]}])
def test_invalid_options_no_output_created(tmp_path, kwargs):
    out = tmp_path / "capture"
    with pytest.raises(ValueError):
        D.capture_data([tmp_path / "unused.root"], out, **kwargs)
    assert not out.exists()


def test_ttree_actual_evtinfo_branch_spelling_and_highest_cycle(tmp_path):
    path = tmp_path / "identities.root"
    keys = sample()[2]
    columns = {"evtinfo/" + name: keys[:, i] for i, name in enumerate(("run", "subrun", "event"))}
    with uproot.recreate(path) as root:
        root.mktree(R.TREE, {name: values[:1] for name, values in columns.items()})
        root.mktree(R.TREE, columns)
    with uproot.open(path) as root:
        tree, cycle = D._highest_tree(root)
        assert tree.classname == "TTree" and cycle == 2
        branches = D.discover_event_branches(tree)
        assert branches == {name: "evtinfo/" + name for name in ("run", "subrun", "event")}
        np.testing.assert_array_equal(D.read_event_keys(tree, branches, 0, 2), keys)


def test_jagged_identity_rejected(tmp_path):
    path = tmp_path / "identities.root"
    with uproot.recreate(path) as root:
        root.mktree(R.TREE, {"evtinfo/run": np.array([1, 1]), "evtinfo/subrun": np.array([0, 0]),
                            "evtinfo/event": ak.Array([[3], [4]])})
    with uproot.open(path) as root:
        tree, _ = D._highest_tree(root)
        with pytest.raises(ValueError, match="scalar per event"):
            D.read_event_keys(tree, D.discover_event_branches(tree), 0, 2)


@pytest.mark.parametrize("field", D.SCORING_FIELDS)
def test_every_scoring_field_compared_exactly(field):
    actual = R.new_state()
    reference = copy.deepcopy(actual)
    if field == "h":
        reference[field]["pe_layer_EX"][0] = 1
    elif field == "oot_n":
        reference[field]["EX"] = 1
    else:
        reference[field] += 1
    with pytest.raises(ValueError, match="scoring field " + field):
        D.compare_scoring_fields(actual, reference)


def test_same_subrun_event_different_runs_are_not_duplicates(tmp_path):
    inputs = [tmp_path / "one.root", tmp_path / "two.root"]
    keys = sample()[2]
    write_root(inputs[0], keys=keys)
    other = keys.copy()
    other[:, 0] += 1
    write_root(inputs[1], keys=other)
    refs = [tmp_path / "one.json", tmp_path / "two.json"]
    for path, ref in zip(inputs, refs):
        reference(ref, path)
    out = tmp_path / "capture"
    report = D.capture_data(inputs, out, reference_accumulator=refs)
    assert report["counts"]["events"] == 4 and report["counts"]["files"] == 2
    with np.load(out / report["raw_shards"][1]["path"]) as rows:
        assert np.all(rows["event_source_index"] == 1)
    provenance = json.loads((out / "provenance.json").read_text())
    assert all(source["reference_comparison"]["status"] == "exact" for source in provenance["inputs"])


def test_input_mutation_refused(tmp_path, monkeypatch):
    path = tmp_path / "data.root"
    write_root(path)
    original_digest = D.digest
    calls = 0

    def changing_digest(source):
        nonlocal calls
        value = original_digest(source)
        if str(source) == str(path):
            calls += 1
            if calls > 1:
                return "changed"
        return value

    monkeypatch.setattr(D, "digest", changing_digest)
    out = tmp_path / "capture"
    with pytest.raises(ValueError, match="ROOT input changed"):
        D.capture_data([path], out)
    assert not (out / "summary.json").exists()
