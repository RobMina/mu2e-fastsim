"""Synthetic immediate-source provenance contracts, not source-file certification.

The C++/FCL/shell checks below are SOURCE-STATIC ONLY: no compilation, art run,
remote access, real data, geometry verification, or source-independence claim.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest
import uproot

from calofastsim.showers import source_events as source


EVENT = ("run", "subrun", "event")
TAG = "TargetStopResampler::Primary"
BRANCH = "art::EventIDs_TargetStopResampler__Primary."
ART = Path(__file__).resolve().parents[1] / "art"
CPP = ART / "CaloFastSim/src/CaloLibraryDumper_module.cc"
FCL = ART / "CaloFastSim/fcl"
RESULT_FIELDS = set(EVENT) | {
    "source_count", "source_product_id", "source_run", "source_subrun",
    "source_event", "source_present", "source_file_resolved",
}


def expected_census():
    # Independently specified full ordered census, not selected from exporter rows.
    return {"run": np.array([7, 7, 8]), "subrun": np.array([2, 3, 2]),
            "event": np.array([33, 11, 22])}


def tables():
    def ids():
        return {"run": np.array([7, 7, 8]), "subrun": np.array([2, 3, 2]),
                "event": np.array([33, 11, 22])}

    return {
        "events": {**ids(), "source_present": np.ones(3, bool),
                   "source_product_id": np.array([81, 81, 82], dtype=np.uint32),
                   "source_count": np.ones(3, dtype=np.uint64),
                   "source_tag": np.array([TAG] * 3), "source_branch": np.array([BRANCH] * 3),
                   "g4_status": np.array([0, 3, 1]), "n_showers": np.array([4, 0, 0])},
        "source_events": {**ids(), "product_id": np.array([81, 81, 82], dtype=np.uint32),
                          "index": np.zeros(3, dtype=np.uint64), "source_valid": np.ones(3, bool),
                          "source_run": np.array([107, 109, 107], dtype=np.uint64),
                          "source_subrun": np.array([4, 5, 4], dtype=np.uint64),
                          "source_event": np.array([99, 2, 100], dtype=np.uint64)},
        "products": {**ids(), "product_id": np.array([81, 81, 82], dtype=np.uint32),
                     "kind": np.full(3, -6), "size": np.ones(3, dtype=np.uint64),
                     "present": np.ones(3, bool), "branch_name": np.array([BRANCH] * 3)},
    }


def validate(data, expected=None, **kwargs):
    return source.validate(data["events"], data["source_events"], data["products"],
                           expected_census() if expected is None else expected,
                           source_tag=kwargs.get("source_tag", TAG),
                           source_branch=kwargs.get("source_branch", BRANCH))


def take(table, indices):
    return {field: value[indices] for field, value in table.items()}


def write_root(path, data):
    # Numeric TTree columns plus the two explicit provenance strings; no art objects.
    with uproot.recreate(path) as output:
        for name, columns in data.items():
            output.mktree("dump/" + name, columns)


def cli_inputs(tmp_path, data=None):
    root = tmp_path / "tables.root"
    expected = tmp_path / "expected.npz"
    output = tmp_path / "new-output"
    write_root(root, tables() if data is None else data)
    np.savez(expected, **expected_census())
    argv = ["--tables", str(root), "--expected-events", str(expected),
            "--source-tag", TAG, "--source-branch", BRANCH, "--out", str(output)]
    return root, expected, output, argv


def test_valid_arrays_preserve_full_order_and_do_not_mutate_inputs():
    data = tables()
    before = deepcopy(data)
    result, report = validate(data)
    assert set(result) == RESULT_FIELDS
    for field in EVENT:
        np.testing.assert_array_equal(result[field], expected_census()[field])
    for field in ("source_run", "source_subrun", "source_event"):
        np.testing.assert_array_equal(result[field], data["source_events"][field])
    np.testing.assert_array_equal(result["source_product_id"], [81, 81, 82])
    np.testing.assert_array_equal(result["source_count"], [1, 1, 1])
    assert report["events"] == report["source_rows"] == 3
    assert report["ordered_event_census_exact"] is True
    assert (report["source_tag"], report["source_branch"]) == (TAG, BRANCH)
    for name, columns in data.items():
        for field, values in columns.items():
            np.testing.assert_array_equal(values, before[name][field])


@pytest.mark.parametrize("omit_selection_columns", [False, True])
def test_numeric_root_reader_neither_status_nor_shower_selects(tmp_path, omit_selection_columns):
    data = tables()
    if omit_selection_columns:
        del data["events"]["g4_status"]
        del data["events"]["n_showers"]
    root = tmp_path / "synthetic.root"
    write_root(root, data)
    result, report = source.read(root, expected_census(), source_tag=TAG, source_branch=BRANCH)
    assert report["events"] == 3
    np.testing.assert_array_equal(result["event"], [33, 11, 22])
    assert set(result) == RESULT_FIELDS
    assert result["source_present"].dtype == np.dtype(bool)
    assert result["source_present"].all()
    assert result["source_file_resolved"].dtype == np.dtype(bool)
    assert not result["source_file_resolved"].any()
    assert report["source_file_resolution"] == "not_performed"
    assert report["source_independence_certified"] is False
    assert "no original-file or stopped-muon-key inference" in report["scope"]
    assert not any("fid" in field or field.endswith("_key") for field in result)


def test_repeated_immediate_source_id_is_not_independence_or_file_resolution():
    data = tables()
    for field in ("source_run", "source_subrun", "source_event"):
        data["source_events"][field][:] = data["source_events"][field][0]
    # Even tempting extra metadata is not evidence for an original file or key.
    data["source_events"]["source_fid"] = np.array(["unverified-file"] * 3)
    data["source_events"]["sim_key"] = np.array([44, 44, 44])
    result, report = validate(data)
    assert set(result) == RESULT_FIELDS
    assert not result["source_file_resolved"].any()
    assert report["source_independence_certified"] is False


def test_independent_expected_census_is_a_required_argument():
    data = tables()
    with pytest.raises(TypeError, match="expected_events"):
        source.validate(data["events"], data["source_events"], data["products"],
                        source_tag=TAG, source_branch=BRANCH)
    with pytest.raises(TypeError, match="expected_events"):
        source.read("not-opened.root", source_tag=TAG, source_branch=BRANCH)


@pytest.mark.parametrize("change", ["order", "subset", "extra", "duplicate", "run", "subrun", "event"])
def test_expected_census_must_match_every_full_identity_in_order(change):
    expected = expected_census()
    if change in ("order", "subset", "extra", "duplicate"):
        indices = {"order": [2, 1, 0], "subset": [0, 1],
                   "extra": [0, 1, 2, 0], "duplicate": [0, 0, 2]}[change]
        expected = take(expected, indices)
        if change == "extra":
            expected["event"][-1] = 999
    else:
        expected[change][1] += 1
    with pytest.raises(ValueError, match="census mismatch|duplicate output"):
        validate(tables(), expected)


def test_empty_export_cannot_certify_provenance():
    data = {name: take(table, slice(0, 0)) for name, table in tables().items()}
    with pytest.raises(ValueError, match="empty export"):
        validate(data, take(expected_census(), slice(0, 0)))


def test_output_identity_is_full_run_subrun_event_and_unique():
    data = tables()
    expected = expected_census()
    for table in (*data.values(), expected):
        table["event"][:] = 12  # Same event number is valid in distinct subruns/runs.
    result, _ = validate(data, expected)
    assert len(result["event"]) == 3
    for field in EVENT:
        data["events"][field][1] = data["events"][field][0]
    with pytest.raises(ValueError, match="duplicate output"):
        validate(data, expected)


@pytest.mark.parametrize("value", [2**32 - 1, 2**40 + 7, 2**53 + 1, 2**63 - 1])
def test_uint64_event_numbers_in_signed_int64_range_are_not_truncated(tmp_path, value):
    data, expected = tables(), expected_census()
    for table in (*data.values(), expected):
        table["event"] = np.array([value, 11, 22], dtype=np.uint64)
    data["source_events"]["source_event"][0] = value
    root = tmp_path / "wide.root"
    write_root(root, data)
    result, _ = source.read(root, expected, source_tag=TAG, source_branch=BRANCH)
    assert int(result["event"][0]) == value
    assert int(result["source_event"][0]) == value
    assert result["event"].dtype.kind in "iu"
    assert result["source_event"].dtype.kind in "iu"


INTEGER_COLUMNS = (
    [("events", field) for field in EVENT + ("source_product_id", "source_count")]
    + [("source_events", field) for field in EVENT + ("product_id", "index", "source_run", "source_subrun", "source_event")]
    + [("products", field) for field in EVENT + ("product_id", "kind", "size")]
    + [("expected", field) for field in EVENT]
)


@pytest.mark.parametrize("table,field", INTEGER_COLUMNS)
def test_every_identity_count_index_and_registry_kind_requires_integer_dtype(table, field):
    data, expected = tables(), expected_census()
    target = expected if table == "expected" else data[table]
    target[field] = target[field].astype(float)  # Even exactly integral floats are forbidden.
    with pytest.raises(ValueError, match="integer|registry kind"):
        validate(data, expected)


@pytest.mark.parametrize("value", [np.array([-1, 2, 3]),
                                   np.array([2**63, 2, 3], dtype=np.uint64),
                                   np.array([2**64 - 1, 2, 3], dtype=np.uint64),
                                   np.array([True, True, True]),
                                   np.array(["1", "2", "3"]),
                                   np.array([1, 2, 3], dtype=object)])
def test_invalid_identity_domains_are_not_silently_coerced(value):
    data = tables()
    data["source_events"]["source_event"] = value
    with pytest.raises(ValueError, match="integer|overflow|identity"):
        validate(data)


@pytest.mark.parametrize("value", [0, 2**32, 2**40])
def test_product_id_has_its_own_nonzero_uint32_domain(value):
    data = tables()
    # Consistent across tables: rejection must not rely on a join mismatch.
    for name, field in (("events", "source_product_id"), ("source_events", "product_id"),
                        ("products", "product_id")):
        data[name][field] = np.full(3, value, dtype=np.uint64)
    with pytest.raises(ValueError, match="ProductID|art identity|overflow"):
        validate(data)


def test_maximum_uint32_product_id_is_preserved_without_signed_narrowing():
    data = tables()
    for name, field in (("events", "source_product_id"), ("source_events", "product_id"),
                        ("products", "product_id")):
        data[name][field] = np.full(3, 2**32 - 1, dtype=np.uint32)
    result, _ = validate(data)
    np.testing.assert_array_equal(result["source_product_id"], [2**32 - 1] * 3)


@pytest.mark.parametrize("table,field,value", [
    ("events", "source_count", np.ones((3, 1), dtype=int)),
    ("source_events", "source_run", np.ones(2, dtype=int)),
    ("source_events", "index", np.array(0)),
    ("products", "size", np.ones(2, dtype=int)),
])
def test_columns_must_be_one_dimensional_and_aligned(table, field, value):
    data = tables()
    data[table][field] = value
    with pytest.raises(ValueError, match="one-dimensional|lengths differ"):
        validate(data)


@pytest.mark.parametrize("table,field", [("events", "source_present"),
                                         ("source_events", "source_valid"), ("products", "present")])
@pytest.mark.parametrize("value", [np.array([True, False, True]), np.ones(3, dtype=int),
                                   np.ones(2, dtype=bool)])
def test_source_and_registry_presence_require_true_aligned_booleans(table, field, value):
    data = tables()
    data[table][field] = value
    with pytest.raises(ValueError, match="false|booleans|lengths differ"):
        validate(data)


@pytest.mark.parametrize("count", [0, 2])
def test_exactly_one_secondary_required_even_with_consistent_row_counts(count):
    data = tables()
    data["events"]["source_count"][:] = count
    data["products"]["size"][:] = count
    indices = np.repeat(np.arange(3), count)
    data["source_events"] = take(data["source_events"], indices)
    data["source_events"]["index"] = np.tile(np.arange(count), 3)
    with pytest.raises(ValueError, match="exactly one secondary"):
        validate(data)


@pytest.mark.parametrize("field,value", [("index", 1), ("product_id", 99)])
def test_source_sequence_index_and_product_must_match_event(field, value):
    data = tables()
    data["source_events"][field][1] = value
    with pytest.raises(ValueError, match="index or ProductID mismatch"):
        validate(data)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "order", "unknown"])
def test_source_rows_must_be_complete_unique_event_aligned_and_ordered(change):
    data = tables()
    if change == "unknown":
        data["source_events"]["event"][0] = 999
    else:
        indices = {"missing": [0, 2], "extra": [0, 1, 2, 2],
                   "duplicate": [0, 0, 2], "order": [1, 0, 2]}[change]
        data["source_events"] = take(data["source_events"], indices)
    with pytest.raises(ValueError, match="counts disagree|event order|unresolved table identity"):
        validate(data)


@pytest.mark.parametrize("tag,branch", [("TargetStopResampler", BRANCH),
                                       ("TargetStopResampler::", BRANCH),
                                       ("::Primary", BRANCH),
                                       ("TargetStopResampler: :Primary", BRANCH),
                                       ("TargetStopResampler:::Primary", BRANCH),
                                       (TAG, "art::EventIDs_TargetStopResampler__Other."),
                                       (TAG, "artEventIDs_TargetStopResampler__Primary.")])
def test_source_tag_requires_explicit_process_and_matching_expected_branch(tag, branch):
    with pytest.raises(ValueError, match="source_tag|explicit source_tag"):
        validate(tables(), source_tag=tag, source_branch=branch)


@pytest.mark.parametrize("field,value", [("source_tag", "Other::Primary"),
                                         ("source_tag", "TargetStopResampler::Other"),
                                         ("source_branch", "art::EventIDs_Other__Primary."),
                                         ("source_branch", "art::EventIDs_TargetStopResampler__Other.")])
def test_recorded_event_tag_and_branch_cannot_disagree_with_expectation(field, value):
    data = tables()
    data["events"][field] = np.array([value] * 3)
    with pytest.raises(ValueError, match="explicitly expected source product"):
        validate(data)


@pytest.mark.parametrize("change", ["wrong_kind", "missing", "duplicate", "unknown", "id", "size", "branch"])
def test_registry_requires_kind_minus_six_one_matching_present_product_per_event(change):
    data = tables()
    registry = data["products"]
    if change == "wrong_kind":
        registry["kind"][0] = -5
    elif change in ("missing", "duplicate"):
        data["products"] = take(registry, [0, 2] if change == "missing" else [0, 0, 2])
    elif change == "unknown":
        registry["event"][0] = 999
    elif change == "id":
        registry["product_id"][0] = 99
    elif change == "size":
        registry["size"][0] = 2
    else:
        registry["branch_name"] = np.array(["art::EventIDs_Other__Primary."] * 3)
    with pytest.raises(ValueError, match="one row per event|unresolved table identity|ID/count mismatch|explicitly expected"):
        validate(data)


def test_registry_order_is_not_source_sequence_order_and_unrelated_kinds_are_ignored():
    data = tables()
    chosen = take(data["products"], [2, 0, 1])
    unrelated = deepcopy(chosen)
    unrelated["kind"][:] = -1
    unrelated["branch_name"] = np.array(["mu2eSimParticles_g4run__Primary."] * 3)
    unrelated["product_id"][:] = 123
    data["products"] = {field: np.concatenate((chosen[field], unrelated[field])) for field in chosen}
    result, report = validate(data)
    np.testing.assert_array_equal(result["source_product_id"], [81, 81, 82])
    assert report["source_rows"] == 3


@pytest.mark.parametrize("missing", ["old_export", "source_events", "source_present", "source_product_id",
                                     "source_count", "source_tag", "source_branch", "source_valid", "branch_name"])
def test_old_export_or_missing_required_root_branches_fail_closed(tmp_path, missing):
    data = tables()
    if missing == "old_export":
        data["events"] = {field: value for field, value in data["events"].items()
                          if not field.startswith("source_")}
        del data["source_events"]
    elif missing == "source_events":
        del data["source_events"]
    elif missing == "source_valid":
        del data["source_events"][missing]
    elif missing == "branch_name":
        del data["products"][missing]
    else:
        del data["events"][missing]
    root = tmp_path / "incomplete.root"
    write_root(root, data)
    with pytest.raises((KeyError, ValueError)):
        source.read(root, expected_census(), source_tag=TAG, source_branch=BRANCH)


def test_cli_new_output_writes_safe_arrays_strict_json_and_exact_hashes(tmp_path, capsys):
    root, expected, output, argv = cli_inputs(tmp_path)
    root_before, expected_before = root.read_bytes(), expected.read_bytes()
    source.main(argv)
    assert {path.name for path in output.iterdir()} == {"source_events.npz", "summary.json"}

    def reject_constant(value):
        raise AssertionError("non-JSON numeric constant: " + value)

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result
            result[key] = value
        return result

    summary_text = (output / "summary.json").read_text()
    report = json.loads(summary_text, parse_constant=reject_constant, object_pairs_hook=reject_duplicate_keys)
    assert capsys.readouterr().out == summary_text
    assert report["input_sha256"] == {
        "tables": hashlib.sha256(root_before).hexdigest(),
        "expected_events": hashlib.sha256(expected_before).hexdigest(),
    }
    assert report["source_sha256"] == hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()
    assert report["events"] == report["source_rows"] == 3
    assert report["source_independence_certified"] is False
    assert report["source_file_resolution"] == "not_performed"
    with np.load(output / "source_events.npz", allow_pickle=False) as archive:
        assert set(archive.files) == RESULT_FIELDS
        np.testing.assert_array_equal(archive["event"], [33, 11, 22])
        np.testing.assert_array_equal(archive["source_event"], [99, 2, 100])
        assert not archive["source_file_resolved"].any()
        assert all(archive[field].dtype.kind in "iub" for field in archive.files)
    assert root.read_bytes() == root_before
    assert expected.read_bytes() == expected_before


@pytest.mark.parametrize("existing", ["empty_directory", "occupied_directory", "file"])
def test_cli_refuses_any_existing_output_without_overwriting(tmp_path, existing):
    root, expected, output, argv = cli_inputs(tmp_path)
    if existing == "file":
        output.write_bytes(b"do not replace")
    else:
        output.mkdir()
        if existing == "occupied_directory":
            (output / "summary.json").write_bytes(b"do not replace")
    # This must be refused before input parsing, not just at the final mkdir.
    root.unlink()
    expected.unlink()
    with pytest.raises(FileExistsError):
        source.main(argv)
    if existing == "file":
        assert output.read_bytes() == b"do not replace"
    elif existing == "occupied_directory":
        assert (output / "summary.json").read_bytes() == b"do not replace"
        assert len(list(output.iterdir())) == 1
    else:
        assert list(output.iterdir()) == []


@pytest.mark.parametrize("invalid", ["empty_export", "wrong_census", "missing_product"])
def test_cli_validation_failure_does_not_create_output(tmp_path, invalid):
    data = tables()
    if invalid == "empty_export":
        data = {name: take(table, slice(0, 0)) for name, table in data.items()}
    elif invalid == "missing_product":
        data["events"]["source_present"][0] = False
    root, expected, output, argv = cli_inputs(tmp_path, data)
    if invalid == "wrong_census":
        np.savez(expected, **take(expected_census(), [2, 1, 0]))
    with pytest.raises(ValueError):
        source.main(argv)
    assert not output.exists()


def test_cli_denies_pickled_expected_event_arrays(tmp_path):
    root, expected, output, argv = cli_inputs(tmp_path)
    census = expected_census()
    census["event"] = census["event"].astype(object)
    np.savez(expected, **census)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded when allow_pickle=False"):
        source.main(argv)
    assert not output.exists()


def test_cli_detects_tables_changing_during_validation(tmp_path, monkeypatch):
    root, expected, output, argv = cli_inputs(tmp_path)
    original_read = source.read

    def changed_read(path, *args, **kwargs):
        result = original_read(path, *args, **kwargs)
        path.write_bytes(path.read_bytes() + b"changed after read")
        return result

    monkeypatch.setattr(source, "read", changed_read)
    with pytest.raises(ValueError, match="tables changed"):
        source.main(argv)
    assert not output.exists()


def test_cli_streams_root_hash_instead_of_reading_all_bytes(tmp_path, monkeypatch):
    root, expected, output, argv = cli_inputs(tmp_path)
    original = Path.read_bytes

    def guarded(path):
        if path == root:
            raise AssertionError("ROOT hashing must use bounded chunks")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    source.main(argv)
    assert (output / "summary.json").is_file()


# Everything below is a source-text regression contract, NOT compiled-art evidence.
def cpp_text():
    return CPP.read_text()


def block_after(text, marker):
    """Extract these simple source blocks with balanced braces (not a C++ parser)."""
    start = text.index("{", text.index(marker))
    depth = 1
    for end in range(start + 1, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start + 1:end]
    raise AssertionError("unbalanced source block: " + marker)


def test_source_static_cpp_opt_in_defaults_false_and_explicit_process_is_guarded():
    text = cpp_text()
    assert re.search(r'Atom<bool>\s+exportSourceEvents\s*\{\s*fhicl::Name\("exportSourceEvents"\),\s*false\s*\}', text)
    constructor = block_after(text, "explicit CaloLibraryDumper(")
    optional = block_after(constructor, "if (conf_.exportSourceEvents())")
    assert "tag.label().empty() || tag.process().empty()" in optional
    assert 'throw cet::exception("CALOLIBRARY")' in optional
    assert "consumes<art::EventIDSequence>(tag)" in optional
    assert optional.index("tag.process().empty()") < optional.index("consumes<art::EventIDSequence>")


def test_source_static_cpp_new_event_branches_and_table_are_conditional():
    begin = block_after(cpp_text(), "void CaloLibraryDumper::beginJob()")
    optional = block_after(begin, "if (conf_.exportSourceEvents())")
    event_fields = set(re.findall(r'\bevents_->Branch\("(source_[^"]+)"', optional))
    assert event_fields == {"source_present", "source_product_id", "source_count", "source_tag", "source_branch"}
    assert 'tfs->make<TTree>("source_events"' in optional
    assert "ids(source_events_)" in optional
    assert set(re.findall(r'source_events_->Branch\("([^"]+)"', optional)) == {
        "product_id", "index", "source_valid", "source_run", "source_subrun", "source_event",
    }
    remainder = begin.replace(optional, "")
    assert 'events_->Branch("source_' not in remainder
    assert 'tfs->make<TTree>("source_events"' not in remainder
    assert 'source_tag_ = conf_.sourceEvents().encode()' in optional


def test_source_static_cpp_missing_product_is_fatal_regardless_of_g4_status():
    text = cpp_text()
    body = block_after(text, "void CaloLibraryDumper::sourceEvents(")
    assert "if (!conf_.exportSourceEvents()) return;" in body
    assert "art::Handle<art::EventIDSequence> handle;" in body
    assert "if (!event.getByLabel(conf_.sourceEvents(), handle))" in body
    assert 'throw cet::exception("CALOLIBRARY") << "missing required source EventIDSequence "' in body
    assert "getMany" not in body and "status_" not in body and "catch" not in body
    assert body.index("if (!conf_.exportSourceEvents()) return;") < body.index("getByLabel")
    analyze = block_after(text, "void CaloLibraryDumper::analyze(")
    prefix = analyze[:analyze.index("sourceEvents(event);")]
    assert re.search(r"^\s*sourceEvents\(event\);", analyze, re.MULTILINE)
    assert not re.search(r"\b(if|return)\b", prefix)
    assert "stat->status()" in prefix  # Merely records status; it does not select events.
    assert analyze.index("sourceEvents(event);") < analyze.index("status_ == 0")


def test_source_static_cpp_kind_minus_six_registry_and_ordered_secondary_loop():
    body = block_after(cpp_text(), "void CaloLibraryDumper::sourceEvents(")
    for statement in ("source_present_ = false", "source_product_ = 0", "source_count_ = 0",
                      "source_index_ = 0", "source_present_ = true", "source_valid_ = false",
                      "source_product_ = handle.id().value()", "source_count_ = handle->size()",
                      "source_branch_ = handle.provenance()->productDescription().branchName()",
                      "product_id_ = source_product_", "kind_ = -6", "n_collection_ = source_count_",
                      "collection_present_ = true", "branch_name_ = source_branch_", "products_->Fill()"):
        assert statement in body
    loop = block_after(body, "for (auto const& id : *handle)")
    assert "source_valid_ = id.isValid()" in loop
    for target, method in (("run", "run"), ("subrun", "subRun"), ("event", "event")):
        assert f"source_{target}_ = id.{method}()" in loop
    assert loop.index("source_events_->Fill()") < loop.index("++source_index_")
    assert body.index("source_index_ = 0") < body.index("for (auto const& id : *handle)")
    assert body.index("products_->Fill()") < body.index("for (auto const& id : *handle)")
    assert not re.search(r"\b(sort|continue|break|if)\b", loop)


def test_source_static_new_fcls_are_separate_leaves_and_old_configs_unchanged():
    # Frozen hashes of the pre-opt-in leaves: source export must not alter defaults.
    old_hashes = {
        "dump_library.fcl": "90fd2a2e149fbd7eff70dfd247e7be844965dca97900d9be6f7823268e3b35b8",
        "dump_library_primary.fcl": "a59830edf60d4537dfccd5db852a1d7bc6e9adf70a7cf632b88d13dadb25ea5b",
    }
    for name, expected in old_hashes.items():
        assert hashlib.sha256((FCL / name).read_bytes()).hexdigest() == expected
        assert "exportSourceEvents" not in (FCL / name).read_text()
    for name, base, process in (
        ("dump_library_sources.fcl", "dump_library.fcl", "CaloLibraryCE"),
        ("dump_library_primary_sources.fcl", "dump_library_primary.fcl", "Primary"),
    ):
        lines = [line.strip() for line in (FCL / name).read_text().splitlines()
                 if line.strip() and (not line.lstrip().startswith("#") or line.startswith("#include"))]
        assert lines == [f'#include "CaloFastSim/fcl/{base}"',
                         "physics.analyzers.dump.exportSourceEvents : true",
                         f'physics.analyzers.dump.sourceEvents : "TargetStopResampler::{process}"']


def test_source_static_export_primary_selector_is_explicit_and_opt_in():
    script = (ART / "export_primary.sh").read_text()
    assert 'case "${CALO_EXPORT_SOURCE_EVENTS:-0}" in' in script
    selector = script.split('case "${CALO_EXPORT_SOURCE_EVENTS:-0}" in', 1)[1].split("esac", 1)[0]
    assert re.search(r"0\)\s*EXPORT_FCL=dump_library_primary\.fcl\s*;;", selector)
    assert re.search(r"1\)\s*EXPORT_FCL=dump_library_primary_sources\.fcl\s*;;", selector)
    assert "*)" in selector and "exit 2" in selector
    assert script.index("esac") < script.index('mkdir "$CALO_OUT"')
    assert 'fhicl-dump -c "$CALO_AREA/CaloFastSim/fcl/$EXPORT_FCL" > executed.fcl' in script
    assert 'mu2e -c executed.fcl -s "$CALO_INPUT" -n "$CALO_EVENTS"' in script
