"""Synthetic authentication and orchestration checks; no real physics or donor draws."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calofastsim.showers import candidate_support as cs
from calofastsim.showers import library, source_audit, source_events, source_overlap_report as overlap, source_split


POOL_FIELDS = ("eligible", "query_cell", "chosen_cell", "occupied", "library_cell", "counts", "size", "full_size",
               "exact_size", "supported", "fallback", "outside", "substituted", "event_row", "prob_empty", "prob_weak",
               "prob_energy", "pool_radius_delta", "pool_angle_deg")


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return source_events.file_digest(path)


def npz_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **value)
    return source_events.file_digest(path)


def identities(n, fid="107658739", first=10000):
    expected = dict(output_fid=np.full(n, fid), run=np.ones(n, np.int64), subrun=np.zeros(n, np.int64),
                    event=np.arange(n, dtype=np.int64))
    ledger = dict(expected, parent_fid=np.full(n, source_audit.PARENT_FID), source_run=np.ones(n, np.int64),
                  source_subrun=np.zeros(n, np.int64), source_event=np.arange(first, first + n, dtype=np.int64),
                  source_count=np.ones(n, np.int64), source_present=np.ones(n, bool), source_file_resolved=np.ones(n, bool))
    return expected, ledger


def bound_bundle(tmp_path, n=2813):
    directory = tmp_path / "file0.art" / "bound"
    expected, ledger = identities(n)
    outputs = {name: npz_file(directory / name, data) for name, data in
               (("expected.npz", expected), ("ledger.npz", ledger))}
    summary = dict(role="development", item=directory.parent.name, events=n, source_rows=n,
                   parent_fid=source_audit.PARENT_FID, ordered_full_census_exact=True,
                   input_sha256={"synthetic_proof": "a" * 64}, output_sha256=outputs)
    json_file(directory / "summary.json", summary)
    parent = dict(input_sha256={str(directory / name): source_events.file_digest(directory / name)
                               for name in ("summary.json", "expected.npz", "ledger.npz")})
    return directory, parent, dict(expected=expected, ledger=ledger)


@pytest.fixture(autouse=True)
def no_sampling(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("candidate support must not sample, reconstruct or run subprocesses")
    monkeypatch.setattr(library, "sample", forbidden)
    monkeypatch.setattr(source_audit.subprocess, "run", forbidden)
    for name in ("default_rng", "RandomState", "random", "choice"):
        monkeypatch.setattr(np.random, name, forbidden)


@pytest.mark.parametrize("kind", ["json", "npz"])
def test_external_pins_and_mutation_detection(tmp_path, kind):
    path = tmp_path / ("input." + kind)
    digest = json_file(path, {"answer": 3}) if kind == "json" else npz_file(path, {"answer": np.array([3])})
    inputs = overlap.Inputs()
    reader = getattr(cs, "pinned_" + kind)
    result = reader(inputs, path, digest)
    assert np.asarray(result["answer"]).item() == 3
    with pytest.raises(ValueError, match="hash mismatch"):
        reader(overlap.Inputs(), path, "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        reader(overlap.Inputs(), path, "not-a-digest")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        inputs.unchanged()


def test_parent_bundle_all_three_external_hashes(tmp_path):
    directory, parent, expected = bound_bundle(tmp_path)
    result = cs.parent_bundle(overlap.Inputs(), directory, parent)
    for field in overlap.LEDGER_FIELDS:
        np.testing.assert_array_equal(result["ledger"][field], expected["ledger"][field])
    assert result["stats"]["events"] == 2813


@pytest.mark.parametrize("name", ["summary.json", "expected.npz", "ledger.npz"])
@pytest.mark.parametrize("fault", ["missing", "ambiguous", "modified"])
def test_parent_bundle_rejects_unanchored_or_changed_members(tmp_path, name, fault):
    directory, parent, _ = bound_bundle(tmp_path)
    key = str(directory / name)
    if fault == "missing":
        del parent["input_sha256"][key]
    elif fault == "ambiguous":
        parent["input_sha256"]["/another/" + directory.parent.name + "/bound/" + name] = parent["input_sha256"][key]
    else:
        (directory / name).write_bytes((directory / name).read_bytes() + b" ")
    with pytest.raises(ValueError, match="parent-bundle"):
        cs.parent_bundle(overlap.Inputs(), directory, parent)


@pytest.fixture
def query(tmp_path, monkeypatch):
    path = tmp_path / "synthetic-tables.bin"
    path.write_bytes(b"synthetic numeric export placeholder")
    expected, ledger = identities(4)
    bound = dict(expected=expected, ledger=ledger)
    numeric = {k: v.copy() for k, v in ledger.items()}
    proof = dict(events=4, printer_source_ids_exact=True, ordered_event_census_exact=True,
                 comparison={"all_old_columns_exact": True}, input_sha256={"tables": source_events.file_digest(path)})
    calls = []
    def read(actual, full_expected, **kwargs):
        assert actual == path
        assert full_expected is expected
        assert kwargs == dict(source_tag="TargetStopResampler::Primary", source_branch="art::EventIDs_TargetStopResampler__Primary.")
        calls.append(actual)
        return numeric, {"synthetic_complete": True}
    monkeypatch.setattr(source_events, "read", read)
    return SimpleNamespace(path=path, proof=proof, bound=bound, numeric=numeric, calls=calls)


def test_query_validates_complete_stored_event_order(query):
    assert cs.validate_query(query.path, query.proof, query.bound) == {"synthetic_complete": True}
    assert query.calls == [query.path]


@pytest.mark.parametrize("field", ["run", "subrun", "event", "source_run", "source_subrun", "source_event", "source_count"])
def test_query_rejects_nonfront_identity_difference(query, field):
    query.numeric[field][-1] += 1
    with pytest.raises(ValueError, match="ledger mismatch"):
        cs.validate_query(query.path, query.proof, query.bound)


@pytest.mark.parametrize("fault", ["digest", "events", "printer", "census", "columns"])
def test_query_requires_complete_runtime_proof(query, fault):
    if fault == "digest":
        query.proof["input_sha256"]["tables"] = "0" * 64
    elif fault == "columns":
        query.proof["comparison"]["all_old_columns_exact"] = False
    else:
        key = dict(events="events", printer="printer_source_ids_exact", census="ordered_event_census_exact")[fault]
        query.proof[key] = 3 if fault == "events" else False
    with pytest.raises(ValueError):
        cs.validate_query(query.path, query.proof, query.bound)
    assert not query.calls


def test_query_rejects_mutation_while_reading(query, monkeypatch):
    original = source_events.read
    def read(*args, **kwargs):
        result = original(*args, **kwargs)
        query.path.write_bytes(b"changed during validation")
        return result
    monkeypatch.setattr(source_events, "read", read)
    with pytest.raises(ValueError, match="changed"):
        cs.validate_query(query.path, query.proof, query.bound)


def pools():
    result = {name: np.array([0], np.int64) for name in POOL_FIELDS}
    for name in ("supported", "fallback", "outside", "substituted"):
        result[name] = np.array([name == "supported"])
    for name in ("prob_empty", "prob_weak", "prob_energy", "pool_radius_delta", "pool_angle_deg"):
        result[name] = np.array([.25, np.nan])
    return result


def test_legacy_parity_all_fields_and_nan_padding():
    saved = pools()
    result = cs.legacy_pool_parity(deepcopy(saved), saved)
    assert set(result["fields"]) == set(POOL_FIELDS)
    assert result["exact"] and result["new_draw_performed"] is False


@pytest.mark.parametrize("field", ["query_cell", "occupied"])
def test_legacy_float_query_labels_are_bounded_exact_integers_only(field):
    current, saved = pools(), pools()
    current[field] = np.array([[0, 3, 2, 1], [1, 0, 0, 3]], np.int64)
    saved[field] = current[field].astype(np.float64)
    assert field in cs.legacy_pool_parity(current, saved)["coordinate_dtype_normalization"]
    for bad in (0.1, np.nan, np.inf, 2., float(2 ** 60)):
        changed = deepcopy(saved)
        changed[field][0, 0] = bad
        with pytest.raises(ValueError):
            cs.legacy_pool_parity(current, changed)


@pytest.mark.parametrize("field", POOL_FIELDS)
@pytest.mark.parametrize("fault", ["missing", "dtype", "shape", "value"])
def test_legacy_parity_rejects_any_pool_change(field, fault):
    current, saved = pools(), pools()
    if fault == "missing":
        del current[field]
    elif fault == "dtype":
        current[field] = current[field].astype(np.float32)
    elif fault == "shape":
        current[field] = current[field][:, None]
    elif current[field].dtype.kind == "b":
        current[field][0] = ~current[field][0]
    else:
        current[field][0] += 1
    with pytest.raises(ValueError, match="legacy pool|pool mismatch"):
        cs.legacy_pool_parity(current, saved)


def geometry():
    return dict(crystal_crystal=np.array([2, 1]), crystal_x=np.array([20., 10.]),
                disk_disk=np.array([0, 1]), disk_z=np.array([100., 200.]), vd_id=np.array([73, 74]),
                vd_z=np.array([90., 190.]))


def test_geometry_full_field_union_and_identity_sorting(monkeypatch):
    geom = geometry()
    packed = {"geometry_" + key: value[::-1].copy() for key, value in geom.items()}
    packed["geometry_crystal_width"] = np.array([3., 4.])
    calls = []
    monkeypatch.setattr(cs.uproot, "open", lambda path: nullcontext("synthetic"))
    def columns(source, table, names):
        calls.append((source, table, names))
        return {"width": np.array([4., 3.])}
    monkeypatch.setattr(cs.export, "columns", columns)
    checked = cs.geometry_parity(Path("synthetic"), geom, packed, deepcopy(packed))
    assert "crystal_width" in checked[0] and checked[0] == checked[1]
    assert calls and all(call == ("synthetic", "crystals", ["width"]) for call in calls)
    assert "crystal_width" not in geom
    packed["geometry_disk_z"][0] += 2e-6
    with pytest.raises(ValueError, match="geometry mismatch"):
        cs.geometry_parity(Path("synthetic"), geom, packed)


@pytest.fixture
def run_bundle(tmp_path, monkeypatch):
    support = pytest.importorskip("calofastsim.showers.support_audit")
    directory, parent, bound = bound_bundle(tmp_path)
    parent_dir = tmp_path / "parent"
    expected, ledger = identities(4, fid="111", first=100)
    parent["output_sha256"] = {"training/" + name: npz_file(parent_dir / "training" / name, data)
                               for name, data in (("expected.npz", expected), ("ledger.npz", ledger))}
    parent_path = parent_dir / "summary.json"
    parent_sha = json_file(parent_path, parent)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    packed = dict(entry_event_row=np.array([0, 2], np.int64))
    training = {key: {} for key in ("front", "in_bins", "out_of_bins", "entry_disk", "highEinnerdisk1", "thresholds")}
    metadata = dict(parent_summary_sha256=parent_sha, training_only=True, candidate_library_constructed=True,
                    sampled=False, legacy_donors_included=False, training=training,
                    file_order=[dict(item="training.art", ordinal=1, fid="111", events=4)])
    library.save(candidate_dir / "library.npz", packed, metadata)
    donor = {k: ledger[k][packed["entry_event_row"]].copy() for k in source_split.OUTPUT_KEY + source_split.PARENT_KEY}
    donor.update(donor_row=np.arange(2, dtype=np.int64), global_event_row=packed["entry_event_row"].copy(),
                 file_ordinal=np.ones(2, np.int64), file_event_row=packed["entry_event_row"].copy())
    outputs = {"library.npz": source_events.file_digest(candidate_dir / "library.npz")}
    for name, data in (("training_expected.npz", expected), ("training_ledger.npz", ledger),
                       ("donor_provenance.npz", donor), ("training_cells.npz", {"counts": np.array([2])})):
        outputs[name] = npz_file(candidate_dir / name, data)
    outputs["metadata.json"] = json_file(candidate_dir / "metadata.json", metadata)
    candidate = dict(metadata, output_sha256=outputs)
    candidate_sha = json_file(candidate_dir / "summary.json", candidate)
    tables = tmp_path / "tables.bin"
    tables.write_bytes(b"synthetic query")
    proof = dict(events=2813, printer_source_ids_exact=True, ordered_event_census_exact=True,
                 comparison={"all_old_columns_exact": True},
                 input_sha256={"tables": source_events.file_digest(tables), "baseline": "b" * 64})
    proof_path = tmp_path / "proof.json"
    proof_sha = json_file(proof_path, proof)
    legacy = tmp_path / "legacy.npz"
    library.save(legacy, dict(entry_event_row=np.array([1], np.int64)), {})
    report = dict(inputs={"library": {"sha256": source_events.file_digest(legacy)}, "tables": {"sha256": "b" * 64}})
    report_path = tmp_path / "legacy_report.json"
    report_sha = json_file(report_path, report)
    pool_path = tmp_path / "pools.npz"
    pool_sha = npz_file(pool_path, pools())
    args = SimpleNamespace(candidate=candidate_dir, candidate_sha256=candidate_sha, parent_summary=parent_path,
                           parent_sha256=parent_sha, query_proof=proof_path, query_proof_sha256=proof_sha,
                           legacy_report=report_path, legacy_report_sha256=report_sha, legacy_pools=pool_path,
                           legacy_pools_sha256=pool_sha, query_bound=directory, tables=tables,
                           legacy_library=legacy, out=tmp_path / "output")
    calls = []
    monkeypatch.setattr(source_events, "read", lambda *a, **k: (bound["ledger"], {"synthetic": True}))
    monkeypatch.setattr(cs.export, "read", lambda *a, **k: ({}, {}, {}, {"synthetic": True}, 2813))
    monkeypatch.setattr(cs.coverage, "census", lambda *a: ({}, {"synthetic": True}))
    monkeypatch.setattr(cs.coverage, "frozen", lambda *a: ({"counts": np.array([2])}, training))
    monkeypatch.setattr(cs, "geometry_parity", lambda *a: [[], []])
    def audit(lib, entries, css):
        calls.append(len(lib["entry_event_row"]))
        return {"synthetic": True}, pools()
    monkeypatch.setattr(support, "audit", audit)
    return SimpleNamespace(args=args, candidate=candidate, expected=expected, ledger=ledger, donor=donor,
                           packed=packed, metadata=metadata, calls=calls)


def repin_candidate(bundle, name, data):
    bundle.candidate["output_sha256"][name] = npz_file(bundle.args.candidate / name, data)
    bundle.args.candidate_sha256 = json_file(bundle.args.candidate / "summary.json", bundle.candidate)


def test_run_mocked_end_to_end_publishes_authenticated_arrays_only(run_bundle):
    result = cs.run(run_bundle.args)
    assert run_bundle.calls == [2, 1]
    assert result["sampling_performed"] is result["reconstruction_performed"] is False
    assert result["candidate_query_overlap"]["shared_parent_events"] == 0
    assert set(result["output_sha256"]) == {"candidate_pools.npz", "legacy_pools.npz"}
    for name, digest in result["output_sha256"].items():
        assert source_events.file_digest(run_bundle.args.out / name) == digest
        with np.load(run_bundle.args.out / name, allow_pickle=False) as saved:
            cs.legacy_pool_parity(pools(), dict(saved))
    assert json.loads((run_bundle.args.out / "summary.json").read_text())["output_sha256"] == result["output_sha256"]
    with pytest.raises(FileExistsError):
        cs.run(run_bundle.args)


@pytest.mark.parametrize("field", source_split.OUTPUT_KEY + source_split.PARENT_KEY)
def test_run_rejects_donor_identity_mismatch(run_bundle, field):
    changed = deepcopy(run_bundle.donor)
    changed[field][0] = "999" if changed[field].dtype.kind == "U" else 999
    repin_candidate(run_bundle, "donor_provenance.npz", changed)
    with pytest.raises(ValueError, match="identity mismatch"):
        cs.run(run_bundle.args)
    assert not run_bundle.args.out.exists()


@pytest.mark.parametrize("name", ["library.npz", "training_expected.npz", "training_ledger.npz", "donor_provenance.npz"])
def test_run_rejects_changed_candidate_consumed_output(run_bundle, name):
    path = run_bundle.args.candidate / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        cs.run(run_bundle.args)
    assert not run_bundle.args.out.exists()


def test_run_rejects_self_consistent_training_ledger_replacement(run_bundle):
    run_bundle.ledger["source_event"] += 200
    run_bundle.donor["source_event"] += 200
    repin_candidate(run_bundle, "training_ledger.npz", run_bundle.ledger)
    repin_candidate(run_bundle, "donor_provenance.npz", run_bundle.donor)
    with pytest.raises(ValueError):
        cs.run(run_bundle.args)


@pytest.mark.parametrize("name", ["metadata.json", "training_cells.npz"])
def test_run_rejects_changed_other_candidate_outputs(run_bundle, name):
    path = run_bundle.args.candidate / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        cs.run(run_bundle.args)


@pytest.mark.parametrize("fault", ["negative", "duplicate", "descending", "float_donor_row",
                                  "float_global_row", "bad_file_ordinal", "bad_file_event_row"])
def test_run_rejects_invalid_donor_row_coordinates(run_bundle, fault):
    rows = dict(negative=[-1, 2], duplicate=[0, 0], descending=[2, 0]).get(fault, [0, 2])
    run_bundle.packed["entry_event_row"] = np.asarray(rows, np.int64)
    donor = {k: run_bundle.ledger[k][rows].copy() for k in source_split.OUTPUT_KEY + source_split.PARENT_KEY}
    donor.update(donor_row=np.arange(2, dtype=np.int64), global_event_row=np.asarray(rows, np.int64),
                 file_ordinal=np.ones(2, np.int64), file_event_row=np.asarray(rows, np.int64))
    if fault.startswith("float_"):
        key = "donor_row" if fault == "float_donor_row" else "global_event_row"
        donor[key] = donor[key].astype(float)
    if fault == "bad_file_ordinal":
        donor["file_ordinal"][:] = 0
    if fault == "bad_file_event_row":
        donor["file_event_row"][:] = -1
    path = run_bundle.args.candidate / "library.npz"
    library.save(path, run_bundle.packed, run_bundle.metadata)
    run_bundle.candidate["output_sha256"][path.name] = source_events.file_digest(path)
    repin_candidate(run_bundle, "donor_provenance.npz", donor)
    with pytest.raises(ValueError):
        cs.run(run_bundle.args)


@pytest.mark.parametrize("argument", ["candidate", "parent_summary", "query_proof", "legacy_report", "legacy_pools", "legacy_library"])
def test_run_requires_every_external_trust_anchor(run_bundle, argument):
    path = getattr(run_bundle.args, argument)
    if argument == "candidate":
        path = path / "summary.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        cs.run(run_bundle.args)
    assert not run_bundle.args.out.exists()


def test_run_rechecks_input_changes_before_publishing(run_bundle, monkeypatch):
    support = pytest.importorskip("calofastsim.showers.support_audit")
    def audit(*args):
        run_bundle.args.tables.write_bytes(b"changed after initial authentication")
        return {"synthetic": True}, pools()
    monkeypatch.setattr(support, "audit", audit)
    with pytest.raises(ValueError, match="changed"):
        cs.run(run_bundle.args)
    assert not run_bundle.args.out.exists()


def test_geometry_libraries_may_have_distinct_extra_fields(monkeypatch):
    geom = geometry()
    first = {"geometry_" + key: value.copy() for key, value in geom.items()}
    second = deepcopy(first)
    second["geometry_crystal_width"] = np.array([4., 3.])
    monkeypatch.setattr(cs.uproot, "open", lambda path: nullcontext("synthetic"))
    monkeypatch.setattr(cs.export, "columns", lambda *args: {"width": np.array([4., 3.])})
    checked = cs.geometry_parity(Path("synthetic"), geom, first, second)
    assert "crystal_width" not in checked[0] and "crystal_width" in checked[1]


def test_real_audit_matches_independently_expected_k8_pool_arrays():
    support = pytest.importorskip("calofastsim.showers.support_audit")
    def entries(radii, disks):
        n = len(radii)
        return dict(energy=np.full(n, 100.), cosine=np.ones(n), radius=np.asarray(radii, float),
                    time=np.zeros(n), disk=np.asarray(disks, np.int64), event_row=np.arange(n, dtype=np.int64),
                    position=np.column_stack((radii, np.zeros(n), np.zeros(n))), direction=np.tile([0., 0., 1.], (n, 1)))
    donor_entries = entries([100., 200.], [0, 0])
    steps = dict(entry=np.array([1]), position=np.array([[200., 0., 0.]]), time=np.zeros(1),
                 energy_g4=np.array([30.]), energy_birks=np.array([20.]), disk=np.array([1]), crystal=np.array([1]))
    packed = library.build(donor_entries, steps, development=np.ones(2, bool))
    packed["geometry_disk_disk"] = np.array([0, 1])
    for axis in "xyz":
        packed["geometry_disk_origin_" + axis] = np.zeros(2)
        packed["geometry_disk_axis_z_" + axis] = np.full(2, float(axis == "z"))
    query = entries([100., 500., 100.], [0, 0, 1])
    _, actual = support.audit(packed, query, dict(entry=np.array([1]), energy_birks=np.array([20.])))
    eligible = np.full((3, 8), -1, np.int64)
    eligible[:2, :2] = [0, 1]
    radius, angle = np.full((3, 8), np.nan), np.full((3, 8), np.nan)
    radius[:2, :2] = [[0., 100.], [-400., -300.]]
    angle[:2, :2] = 0.
    expected = dict(eligible=eligible, query_cell=np.array([[0, 3, 2, 0], [0, 3, 2, 1], [1, 3, 2, 0]]),
                    chosen_cell=np.array([0, 0, -1]), occupied=np.array([[0, 3, 2, 0]]), library_cell=np.array([0, 0]),
                    counts=np.array([2]), size=np.array([2, 2, 0]), full_size=np.array([2, 2, 0]),
                    exact_size=np.array([2, 0, 0]), supported=np.array([True, True, False]),
                    fallback=np.array([False, True, True]), outside=np.zeros(3, bool),
                    substituted=np.array([False, True, False]), event_row=np.arange(3),
                    prob_empty=np.array([.5, .5, np.nan]), prob_weak=np.array([.5, .5, np.nan]),
                    prob_energy=np.array([10., 10., np.nan]), pool_radius_delta=radius, pool_angle_deg=angle)
    assert cs.legacy_pool_parity(actual, expected)["exact"]
