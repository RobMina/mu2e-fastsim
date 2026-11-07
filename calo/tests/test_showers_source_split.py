"""Synthetic catalog roles and immediate-parent identities, not physical independence."""
from collections import Counter
from copy import deepcopy
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import pytest

from calofastsim import showers
from calofastsim.showers import library, source_split as split


OUTPUT = ("output_fid", "run", "subrun", "event")
PARENT = ("parent_fid", "source_run", "source_subrun", "source_event")


@pytest.fixture(autouse=True)
def no_rng_or_reconstruction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("catalog/ledger diagnostics must not sample or reconstruct")

    for module in (showers, library):
        monkeypatch.setattr(module, "sample", forbidden)
    for name in ("default_rng", "RandomState", "random", "rand", "randn", "choice", "seed"):
        monkeypatch.setattr(np.random, name, forbidden)
    for name in ("random", "randint", "randrange", "choice", "choices", "sample", "seed"):
        monkeypatch.setattr(random, name, forbidden)
    original_import = builtins.__import__
    blocked = ("ROOT", "uproot", "calofastsim.chain", "calofastsim.cluster",
               "calofastsim.digi", "calofastsim.hits", "calofastsim.templatefit",
               "calofastsim.showers.response")

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        absolute = importlib.util.resolve_name("." * level + name, globals["__package__"]) if level else name
        candidates = (absolute,) + tuple(absolute + "." + item for item in fromlist or ())
        if any(candidate == module or candidate.startswith(module + ".")
               for candidate in candidates for module in blocked):
            forbidden()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.fixture
def catalog():
    dataset = "sim.synthetic.CE.v1.art"
    query = {"dataset_did": "synthetic:" + dataset}
    rows = []
    for subrun in range(1, 5):
        sequence = f"101_{subrun}"
        name = f"sim.synthetic.CE.v1.{sequence}.art"
        rows.append(dict(did="synthetic:" + name, name=name, namespace="synthetic",
                         fid=f"catalog-fid-{subrun}", size=1000 + subrun,
                         checksums={"adler32": f"{subrun:08x}"},
                         metadata={"dh.dataset": dataset, "fn.sequencer": sequence,
                                   "rs.first_run": 101, "rs.last_run": 101,
                                   "rs.first_subrun": subrun, "rs.last_subrun": subrun}))
    pages = [dict(tool="query_dataset_files", arguments=dict(query, offset=offset, limit=2),
                  result=dict(dataset_did=query["dataset_did"], total_matches=4, returned=2,
                              applied_filters=dict(offset=offset, limit=2), warnings=[],
                              results=rows[offset:offset + 2])) for offset in (0, 2)]
    plan = dict(dataset_did=query["dataset_did"], catalog_query=query,
                selection_rule="explicit synthetic filenames only", supersedes="candidate-v0",
                roles=dict(training=[rows[0]["did"]], development=[rows[1]["did"]],
                           confirmation=[rows[2]["did"]]))
    return plan, {"records": pages}


def columns(records, names):
    return {name: np.asarray([record[i] for record in records],
                            dtype=str if name.endswith("fid") else np.int64)
            for i, name in enumerate(names)}


def ledger(outputs, parents):
    return dict(columns(outputs, OUTPUT), **columns(parents, PARENT),
                source_count=np.ones(len(outputs), np.int64),
                source_present=np.ones(len(outputs), bool),
                source_file_resolved=np.ones(len(outputs), bool))


@pytest.fixture
def ledgers():
    a, b, c, d = (("parent-a", 7, 3, 11), ("parent-a", 8, 3, 11),
                  ("parent-a", 7, 4, 11), ("parent-b", 7, 3, 11))
    left = ledger([("out-left", 1, 2, event) for event in (10, 11, 12, 13, 14)], [a, a, b, c, c])
    right = ledger([("out-right", 90, 20, event) for event in (40, 41, 42, 43)], [c, a, a, d])
    # The expected census is independently enumerated, never sliced from a ledger.
    expected_left = columns([("out-left", 1, 2, 10), ("out-left", 1, 2, 11),
                             ("out-left", 1, 2, 12), ("out-left", 1, 2, 13),
                             ("out-left", 1, 2, 14)], OUTPUT)
    expected_right = columns([("out-right", 90, 20, 40), ("out-right", 90, 20, 41),
                              ("out-right", 90, 20, 42), ("out-right", 90, 20, 43)], OUTPUT)
    return left, right, expected_left, expected_right


def reference(left, right):
    """Small independent tuple/Counter oracle, deliberately not vectorized."""
    a, b = (Counter(zip(*(table[key].tolist() for key in PARENT))) for table in (left, right))

    def stats(counts):
        return dict(events=sum(counts.values()), unique_parent_events=len(counts),
                    reused_parent_events=sum(count > 1 for count in counts.values()),
                    repeated_output_events=sum(count - 1 for count in counts.values()),
                    max_parent_reuse=max(counts.values()))

    shared = set(a) & set(b)
    return dict(left=stats(a), right=stats(b), shared_parent_events=len(shared),
                left_events_with_shared_parent=sum(a[key] for key in shared),
                right_events_with_shared_parent=sum(b[key] for key in shared),
                parent_events_disjoint=not shared)


def test_freeze_candidate_only_complete_inventory_no_input_mutation(catalog):
    plan, receipt = catalog
    before = deepcopy(catalog)
    report = split.freeze(plan, receipt)
    assert catalog == before
    assert report == split.freeze(plan, receipt)
    assert report["state"] == "candidate_file_roles_only"
    assert report["schema_version"] == 1
    assert report["dataset_did"] == plan["dataset_did"]
    assert report["selection_rule"] == plan["selection_rule"]
    assert report["supersedes"] == "candidate-v0"
    assert report["inventory_files"] == 4 and report["unallocated_files"] == 1
    assert report["file_roles_disjoint"] is True
    assert report["upstream_event_overlap"] == "not_checked"
    for flag in ("source_independence_certified", "confirmation_event_contents_opened_by_this_operation", "execution_authorized"):
        assert report[flag] is False
    for role in ("training", "development", "confirmation"):
        row, = report["roles"][role]
        assert row["did"] == plan["roles"][role][0]
        assert set(row) == {"did", "fid", "size", "checksums"}
    assert "not donor counts or availability" in report["scope"]
    json.dumps(report, allow_nan=False)


def test_catalog_sorts_pages_and_ignores_other_queries_and_tools(catalog):
    plan, receipt = catalog
    expected = split.catalog_rows(receipt, plan["catalog_query"])
    other = deepcopy(receipt["records"][0])
    other["arguments"]["dataset_did"] = "synthetic:other"
    wrong_tool = deepcopy(receipt["records"][0])
    wrong_tool["tool"] = "not_a_catalog_query"
    receipt["records"] = [other, receipt["records"][1], wrong_tool, receipt["records"][0]]
    assert split.catalog_rows(receipt, plan["catalog_query"]) == expected


@pytest.mark.parametrize("fault, message", [
    ("none", "no matching"), ("missing", "incomplete"), ("duplicate", "gap or overlap"),
    ("gap", "gap or overlap"), ("count", "dataset/count changed"),
])
def test_catalog_missing_duplicate_gapped_or_changed_pages(catalog, fault, message):
    plan, receipt = catalog
    pages = receipt["records"]
    if fault == "none":
        pages.clear()
    elif fault == "missing":
        pages.pop()
    elif fault == "duplicate":
        pages.append(deepcopy(pages[0]))
    elif fault == "gap":
        pages[1]["arguments"]["offset"] = 3
    else:
        pages[1]["result"]["total_matches"] = 5
    with pytest.raises(ValueError, match=message):
        split.freeze(plan, receipt)


def test_catalog_dataset_and_page_size_must_agree(catalog):
    for field, value in (("dataset_did", "synthetic:wrong"), ("returned", 1)):
        plan, receipt = deepcopy(catalog)
        receipt["records"][1]["result"][field] = value
        with pytest.raises(ValueError, match="changed|size mismatch"):
            split.freeze(plan, receipt)
    plan, receipt = deepcopy(catalog)
    receipt["records"][1]["arguments"]["limit"] = 1
    with pytest.raises(ValueError, match="size mismatch"):
        split.freeze(plan, receipt)


def test_catalog_duplicate_did_or_canonical_fid_rejected(catalog):
    for key in ("did", "fid"):
        plan, receipt = deepcopy(catalog)
        receipt["records"][1]["result"]["results"][1][key] = receipt["records"][0]["result"]["results"][0][key]
        with pytest.raises(ValueError, match="duplicate catalog file identity"):
            split.freeze(plan, receipt)


def test_required_file_absent_rejected(catalog):
    plan, receipt = catalog
    plan["roles"]["confirmation"] = ["synthetic:sim.synthetic.CE.v1.101_99.art"]
    with pytest.raises(ValueError, match="required files absent"):
        split.freeze(plan, receipt)


def test_role_overlap_and_within_role_duplicates_rejected(catalog):
    for across_roles in (False, True):
        plan, receipt = deepcopy(catalog)
        role = "confirmation" if across_roles else "training"
        plan["roles"][role].append(plan["roles"]["training"][0])
        with pytest.raises(ValueError, match="role overlap or duplicate"):
            split.freeze(plan, receipt)


def test_all_three_nonempty_roles_required(catalog):
    for fault in ("missing", "extra", "empty"):
        plan, receipt = deepcopy(catalog)
        if fault == "missing":
            del plan["roles"]["confirmation"]
        elif fault == "extra":
            plan["roles"]["test"] = []
        else:
            plan["roles"]["confirmation"] = []
        with pytest.raises(ValueError, match="roles|empty file role"):
            split.freeze(plan, receipt)


def test_plan_dataset_must_match_query(catalog):
    plan, receipt = catalog
    plan["dataset_did"] = "synthetic:sim.other.CE.v1.art"
    with pytest.raises(ValueError, match="plan dataset mismatch"):
        split.freeze(plan, receipt)


def test_catalog_metadata_fields_must_match_filename(catalog):
    fields = ("dh.dataset", "fn.sequencer", "rs.first_run", "rs.last_run", "rs.first_subrun", "rs.last_subrun")
    for field in fields:
        for missing in (False, True):
            plan, receipt = deepcopy(catalog)
            md = receipt["records"][0]["result"]["results"][0]["metadata"]
            if missing:
                del md[field]
            else:
                md[field] = "wrong" if isinstance(md[field], str) else md[field] + 1
            with pytest.raises(ValueError, match="metadata disagrees"):
                split.freeze(plan, receipt)


def test_catalog_filename_namespace_and_sequencer_must_match(catalog):
    for field, value in (("namespace", "other"), ("name", "wrong.art"),
                         ("name", "sim.synthetic.CE.v1.invalid.art")):
        plan, receipt = deepcopy(catalog)
        row = receipt["records"][0]["result"]["results"][0]
        row[field] = value
        if value.endswith("invalid.art"):
            row["did"] = "synthetic:" + value
            plan["roles"]["training"] = [row["did"]]
        with pytest.raises(ValueError, match="identity mismatch|invalid file sequencer"):
            split.freeze(plan, receipt)


def test_catalog_requires_nonempty_checksums_and_valid_fid_size(catalog):
    for field, value in (("checksums", {}), ("checksums", {"adler32": ""}),
                         ("checksums", {"adler32": 1}), ("fid", ""), ("fid", 4),
                         ("size", 0), ("size", -1), ("size", 1.0), ("size", True)):
        plan, receipt = deepcopy(catalog)
        receipt["records"][0]["result"]["results"][0][field] = value
        with pytest.raises(ValueError, match="checksum|FID/size"):
            split.freeze(plan, receipt)


def test_validate_reports_reuse_and_preserves_inputs(ledgers):
    left, _, expected, _ = ledgers
    before = deepcopy((left, expected))
    parents, stats = split.validate_ledger(left, expected)
    assert stats == dict(events=5, unique_parent_events=3, reused_parent_events=2,
                         repeated_output_events=2, max_parent_reuse=2)
    assert list(parents.dtype.names) == list(PARENT)
    assert parents.tolist() == list(zip(*(left[key] for key in PARENT)))
    for actual, saved in zip((left, expected), before):
        for key in saved:
            np.testing.assert_array_equal(actual[key], saved[key])


def test_shared_parents_count_even_with_different_downstream_ids(ledgers):
    report = split.compare_ledgers(*ledgers)
    for key, value in reference(*ledgers[:2]).items():
        assert report[key] == value
    assert report["shared_parent_events"] == 2
    assert report["left_events_with_shared_parent"] == 4
    assert report["right_events_with_shared_parent"] == 3
    assert report["source_independence_certified"] is False
    assert "immediate stop-event" in report["scope"]


def test_same_event_numbers_in_different_parent_files_are_distinct():
    left = ledger([("left", 1, 1, 1)], [("parent-a", 5, 6, 7)])
    right = ledger([("right", 1, 1, 1)], [("parent-b", 5, 6, 7)])
    result = split.compare_ledgers(left, right, columns([("left", 1, 1, 1)], OUTPUT),
                                   columns([("right", 1, 1, 1)], OUTPUT))
    assert result["shared_parent_events"] == 0
    assert result["left_events_with_shared_parent"] == result["right_events_with_shared_parent"] == 0
    assert result["parent_events_disjoint"] is True
    assert result["source_independence_certified"] is False


def test_full_run_subrun_event_parent_identity_is_used():
    parents = [("p", 1, 2, 3), ("p", 9, 2, 3), ("p", 1, 9, 3), ("p", 1, 2, 9)]
    left = ledger([("left", 0, 0, 0)], parents[:1])
    right = ledger([("right", 0, 0, i) for i in (1, 2, 3)], parents[1:])
    result = split.compare_ledgers(left, right, columns([("left", 0, 0, 0)], OUTPUT),
                                   columns([("right", 0, 0, 1), ("right", 0, 0, 2), ("right", 0, 0, 3)], OUTPUT))
    assert result["shared_parent_events"] == 0
    assert result["right"]["unique_parent_events"] == 3


def test_full_output_fid_and_event_identity_and_zero_are_valid():
    outputs = [("a", 0, 0, 0), ("b", 0, 0, 0), ("a", 1, 0, 0), ("a", 0, 1, 0), ("a", 0, 0, 1)]
    expected = columns([("a", 0, 0, 1), ("a", 0, 1, 0), ("a", 1, 0, 0),
                        ("b", 0, 0, 0), ("a", 0, 0, 0)], OUTPUT)
    _, stats = split.validate_ledger(ledger(outputs, [("p", 0, 0, 0)] * 5), expected)
    assert stats["events"] == stats["max_parent_reuse"] == 5
    assert stats["repeated_output_events"] == 4


def test_duplicate_output_and_duplicate_expected_census_rejected(ledgers):
    for target_index in (0, 2):
        left, _, expected, _ = deepcopy(ledgers)
        target = left if target_index == 0 else expected
        for key in OUTPUT:
            target[key][1] = target[key][0]
        with pytest.raises(ValueError, match="duplicate output event identity"):
            split.validate_ledger(left, expected)


def test_expected_census_requires_exact_coverage_not_just_counts(ledgers):
    for fault in ("missing_ledger_row", "missing_expected_row", "same_count_wrong_event", "empty_expected"):
        left, _, expected, _ = deepcopy(ledgers)
        if fault == "missing_ledger_row":
            left = {key: value[:-1] for key, value in left.items()}
        elif fault == "missing_expected_row":
            expected = {key: value[:-1] for key, value in expected.items()}
        elif fault == "same_count_wrong_event":
            expected["event"][-1] = 999
        else:
            expected = columns([], OUTPUT)
        with pytest.raises(ValueError, match="full expected output event census"):
            split.validate_ledger(left, expected)


def test_empty_ledgers_cannot_certify_a_split():
    with pytest.raises(ValueError, match="empty ledger"):
        split.validate_ledger(ledger([], []), columns([], OUTPUT))


def test_missing_required_columns_and_expected_argument_rejected(ledgers):
    for name in (*OUTPUT, *PARENT, "source_count", "source_present", "source_file_resolved"):
        left, _, expected, _ = deepcopy(ledgers)
        del left[name]
        with pytest.raises(KeyError):
            split.validate_ledger(left, expected)
    left, _, expected, _ = ledgers
    for name in OUTPUT:
        incomplete = {key: value for key, value in expected.items() if key != name}
        with pytest.raises(KeyError):
            split.validate_ledger(left, incomplete)
    with pytest.raises(TypeError):
        split.validate_ledger(left)


def test_source_count_exactly_one_is_mandatory(ledgers):
    for value in (0, 2, -1, True, 1.0):
        left, _, expected, _ = deepcopy(ledgers)
        left["source_count"] = np.full(5, value)
        with pytest.raises(ValueError, match="integer identities|source-event provenance"):
            split.validate_ledger(left, expected)


def test_missing_or_ambiguous_provenance_flags_rejected(ledgers):
    for name in ("source_present", "source_file_resolved"):
        for values in (np.array([True, True, False, True, True]), np.ones(5, int),
                       np.ones(5, float), np.array(True), np.ones((5, 1), bool), np.ones(4, bool)):
            left, _, expected, _ = deepcopy(ledgers)
            left[name] = values
            with pytest.raises(ValueError, match="boolean columns|source-event provenance"):
                split.validate_ledger(left, expected)


def test_id_columns_reject_float_boolean_negative_and_out_of_range(ledgers):
    for location, names in ((0, ("run", "subrun", "event", "source_run", "source_subrun", "source_event")),
                            (2, ("run", "subrun", "event"))):
        for name in names:
            for value in (1.0, True, -1, np.uint64(2 ** 63), "1"):
                data = deepcopy(ledgers)
                data[location][name] = np.full(5, value)
                with pytest.raises(ValueError, match="integer identities"):
                    split.validate_ledger(data[0], data[2])


def test_signed_range_boundary_and_unsigned_integer_ids_are_valid():
    maximum = np.iinfo(np.int64).max
    table = ledger([("a", maximum, 0, maximum)], [("p", 0, maximum, 0)])
    expected = columns([("a", maximum, 0, maximum)], OUTPUT)
    table["event"] = table["event"].astype(np.uint64)
    assert split.validate_ledger(table, expected)[1]["events"] == 1


def test_canonical_fids_require_nonempty_string_columns(ledgers):
    for location, name in ((0, "output_fid"), (0, "parent_fid"), (2, "output_fid")):
        for values in (np.array([""] * 5), np.arange(5), np.array([b"fid"] * 5),
                       np.array(["fid"] * 5, dtype=object)):
            data = deepcopy(ledgers)
            data[location][name] = values
            with pytest.raises(ValueError, match="canonical catalog FID strings"):
                split.validate_ledger(data[0], data[2])


def test_column_dimensions_and_alignment_rejected(ledgers):
    for name in (*OUTPUT, *PARENT, "source_count"):
        for change in (lambda a: a[:-1], lambda a: a.reshape(-1, 1), lambda a: a[0]):
            left, _, expected, _ = deepcopy(ledgers)
            left[name] = change(left[name])
            with pytest.raises(ValueError):
                split.validate_ledger(left, expected)
    left, _, expected, _ = deepcopy(ledgers)
    for name in PARENT:
        left[name] = left[name][:-1]
    with pytest.raises(ValueError, match="output/parent lengths differ"):
        split.validate_ledger(left, expected)


def test_vectorized_comparison_matches_independent_reference_and_ordering():
    left_outputs = [("left", 10, i // 10, i) for i in range(90)]
    right_outputs = [("right", 90, i // 10, i + 100) for i in range(70)]
    left_parents = [(f"p{i % 3}", i % 2, i % 5, i % 7) for i in range(90)]
    right_parents = [(f"p{i % 4}", i % 2, i % 5, i % 7) for i in range(70)]
    left, right = ledger(left_outputs, left_parents), ledger(right_outputs, right_parents)
    expected_left = columns([("left", 10, subrun, subrun * 10 + event)
                             for subrun in range(9) for event in range(10)], OUTPUT)
    expected_right = columns([("right", 90, subrun, 100 + subrun * 10 + event)
                              for subrun in range(7) for event in range(10)], OUTPUT)
    oracle = reference(left, right)
    for table in (left, right, expected_left):
        for name in table:
            table[name] = table[name][::-1]
    actual = split.compare_ledgers(left, right, expected_left, expected_right)
    for key, value in oracle.items():
        assert actual[key] == value
    reverse = split.compare_ledgers(right, left, expected_right, expected_left)
    assert reverse["left"] == actual["right"] and reverse["right"] == actual["left"]
    assert reverse["shared_parent_events"] == actual["shared_parent_events"]
    assert reverse["left_events_with_shared_parent"] == actual["right_events_with_shared_parent"]


def strict_json(text):
    def reject_constant(value):
        raise AssertionError(f"non-JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def cli_inputs(tmp_path, mode, catalog, ledgers):
    paths = {}
    values = dict(zip(("plan", "catalog"), catalog)) if mode == "freeze" else dict(zip(
        ("left", "right", "expected_left", "expected_right"), ledgers))
    for name, value in values.items():
        path = tmp_path / (name + (".json" if mode == "freeze" else ".npz"))
        if mode == "freeze":
            path.write_text(json.dumps(value))
        else:
            np.savez(path, **value)
        paths[name] = path
    args = [mode]
    for name, path in paths.items():
        args.extend(["--" + name.replace("_", "-"), str(path)])
    return paths, args


@pytest.mark.parametrize("mode", ["freeze", "overlap"])
def test_cli_strict_json_hash_receipts_and_no_overwrite(tmp_path, capsys, catalog, ledgers, mode):
    paths, args = cli_inputs(tmp_path, mode, catalog, ledgers)
    before = {name: path.read_bytes() for name, path in paths.items()}
    out = tmp_path / "reports" / "candidate.json"
    args += ["--out", str(out)]
    split.main(args)
    report = strict_json(out.read_text())
    assert report["input_sha256"] == {name: hashlib.sha256(content).hexdigest() for name, content in before.items()}
    assert report["source_sha256"] == hashlib.sha256(Path(split.__file__).read_bytes()).hexdigest()
    assert report["source_independence_certified"] is False
    assert strict_json(capsys.readouterr().out) == dict(out=str(out), mode=mode, source_sha256=report["source_sha256"])
    assert {name: path.read_bytes() for name, path in paths.items()} == before
    if mode == "overlap":
        for key, value in reference(*ledgers[:2]).items():
            assert report[key] == value
    saved = out.read_bytes()
    with pytest.raises(FileExistsError):
        split.main(args)
    assert out.read_bytes() == saved
    assert capsys.readouterr().out == ""


def test_cli_existing_output_is_checked_before_reading_inputs(tmp_path):
    out = tmp_path / "frozen.json"
    out.write_bytes(b"immutable existing receipt\n")
    with pytest.raises(FileExistsError):
        split.main(["freeze", "--plan", str(tmp_path / "missing-plan.json"),
                    "--catalog", str(tmp_path / "missing-catalog.json"), "--out", str(out)])
    assert out.read_bytes() == b"immutable existing receipt\n"


def test_cli_invalid_ledger_writes_no_report(tmp_path, catalog, ledgers):
    left, right, expected_left, expected_right = ledgers
    left["source_file_resolved"][0] = False
    _, args = cli_inputs(tmp_path, "overlap", catalog, (left, right, expected_left, expected_right))
    out = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="unresolved"):
        split.main(args + ["--out", str(out)])
    assert not out.exists()


def test_cli_overlap_disallows_pickle_payloads(tmp_path, catalog, ledgers):
    ledgers[0]["parent_fid"] = ledgers[0]["parent_fid"].astype(object)
    _, args = cli_inputs(tmp_path, "overlap", catalog, ledgers)
    out = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        split.main(args + ["--out", str(out)])
    assert not out.exists()


def test_cli_refuses_nonfinite_json_report(tmp_path, catalog, ledgers):
    catalog[0]["selection_rule"] = float("nan")
    _, args = cli_inputs(tmp_path, "freeze", catalog, ledgers)
    out = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        split.main(args + ["--out", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("fid", [" ", "parent-a ", "\tparent-a", "parent/a", "pärent-a"])
def test_malformed_fid_cannot_manufacture_disjointness(ledgers, fid):
    left, _, expected, _ = ledgers
    left["parent_fid"] = np.full(5, fid)
    with pytest.raises(ValueError, match="canonical catalog FID"):
        split.validate_ledger(left, expected)


@pytest.mark.parametrize("conflicting_parent", [False, True])
def test_shared_output_event_rejected_even_with_conflicting_parent(conflicting_parent):
    left = ledger([("out", 1, 2, 3)], [("parent-a", 4, 5, 6)])
    right = ledger([("out", 1, 2, 3)], [("parent-b" if conflicting_parent else "parent-a", 4, 5, 6)])
    expected = columns([("out", 1, 2, 3)], OUTPUT)
    with pytest.raises(ValueError, match="output event identity overlaps"):
        split.compare_ledgers(left, right, expected, expected)


def test_same_output_file_with_disjoint_events_is_reported():
    left = ledger([("out", 1, 2, 3)], [("parent-a", 4, 5, 6)])
    right = ledger([("out", 1, 2, 4)], [("parent-b", 4, 5, 6)])
    result = split.compare_ledgers(left, right, columns([("out", 1, 2, 3)], OUTPUT),
                                  columns([("out", 1, 2, 4)], OUTPUT))
    assert result["shared_output_files"] == 1
    assert result["shared_output_events"] == 0
    assert result["parent_events_disjoint"]
    assert not result["source_independence_certified"]


@pytest.mark.parametrize("fault", ["offset", "limit", "warning", "floating_count", "bool_count"])
def test_effective_filters_and_completeness_warnings_fail_closed(catalog, fault):
    plan, receipt = catalog
    result = receipt["records"][0]["result"]
    if fault in ("offset", "limit"):
        result["applied_filters"][fault] += 1
    elif fault == "warning":
        result["warnings"] = ["partial response"]
    else:
        result["total_matches"] = 4.0 if fault == "floating_count" else True
    with pytest.raises(ValueError, match="effective filters|warnings|integer counts"):
        split.freeze(plan, receipt)


@pytest.mark.parametrize("fault", ["dataset", "bool_metadata", "checksum_key", "checksum_value", "fid"])
def test_unallocated_catalog_records_are_also_validated(catalog, fault):
    plan, receipt = catalog
    row = receipt["records"][1]["result"]["results"][1]
    if fault == "dataset":
        row["metadata"]["dh.dataset"] = "wrong"
    elif fault == "bool_metadata":
        row["metadata"]["rs.first_subrun"] = True
    elif fault == "checksum_key":
        row["checksums"] = {"": "12345678"}
    elif fault == "checksum_value":
        row["checksums"] = {"adler32": " "}
    else:
        row["fid"] = "   "
    with pytest.raises(ValueError, match="metadata disagrees|checksum|FID/size"):
        split.freeze(plan, receipt)


def test_mixed_integer_widths_preserve_large_parent_keys():
    number = 2 ** 60 + 1
    left = ledger([("left", 1, 1, 1)], [("parent", 1, 1, number)])
    right = ledger([("right", 1, 1, 1)], [("parent", 1, 1, number + 1)])
    right["source_event"] = right["source_event"].astype(np.uint64)
    result = split.compare_ledgers(left, right, columns([("left", 1, 1, 1)], OUTPUT),
                                  columns([("right", 1, 1, 1)], OUTPUT))
    assert result["shared_parent_events"] == 0
