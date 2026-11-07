"""Synthetic runtime-check contracts only: no build, art run, or real-file claim."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import awkward as ak
import numpy as np
import pytest
import uproot

from calofastsim.showers import source_event_check as check
from test_showers_source_events import tables as source_tables, write_root

TAG, BRANCH = check.SOURCE_TAG, check.SOURCE_BRANCH
PRINTED_TAG = "InputTag: label = 'TargetStopResampler', instance = '', process = 'Primary'"


def fixtures():
    candidate = source_tables()
    # Repeated immediate IDs are allowed, and are only a diagnostic count.
    for field in check.SOURCE_IDS:
        candidate["source_events"][field][2] = candidate["source_events"][field][0]
    old_products = deepcopy(candidate["products"])
    old_products["kind"][:] = -1
    old_products["product_id"][:] = 23
    old_products["size"][:] = 2
    old_products["branch_name"] = np.array(["mu2eSimParticles_g4run__Primary."] * 3)
    candidate["products"] = {field: np.concatenate((old_products[field], values))
                             for field, values in candidate["products"].items()}
    candidate["showers"] = dict(run=np.array([7, 8]), subrun=np.array([2, 2]),
                                 event=np.array([33, 22]), time=np.array([1., np.nan]),
                                 energy=np.array([0.1, 0.2], dtype=np.float32),
                                 sim_key=np.array([2**63 + 1, 2**64 - 1], dtype=np.uint64))
    candidate["empty"] = dict(key=np.array([], dtype=np.uint32), value=np.array([], dtype=np.float64))
    candidate["geometry"] = dict(name=np.array(["disk", "crystal"]), xyz=np.array([0., 12.]))
    baseline = deepcopy(candidate)
    del baseline["source_events"]
    baseline["events"] = {field: values for field, values in baseline["events"].items()
                          if field not in check.ADDED_EVENT_FIELDS}
    baseline["products"] = old_products
    return baseline, candidate


def printer(candidate):
    # Building tiny synthetic fixtures is not the production parser's hot path.
    lines = ["unrelated initialization"]
    suffix = ("st", "nd", "rd")
    for i in range(3):
        ids = candidate["events"]
        src = candidate["source_events"]
        lines.extend(["%MSG-i ArtReport:  event processing",
                      f"Begin processing the {i+1}{suffix[i]} record. run: {ids['run'][i]} "
                      f"subRun: {ids['subrun'][i]} event: {ids['event'][i]} at 06-Nov-2026 12:00:00 UTC",
                      "%MSG",
                      f"EventIDSequence {PRINTED_TAG} :  ( run: {src['source_run'][i]} "
                      f"subRun: {src['source_subrun'][i]} event: {src['source_event'][i]} )"])
    lines.extend(["TrigReport Events total = 3 passed = 3 failed = 0", check.COMPLETION])
    return "\n".join(lines) + "\n"


def inputs(tmp_path, baseline=None, candidate=None, text=None):
    base, new = fixtures()
    base = base if baseline is None else baseline
    new = new if candidate is None else candidate
    paths = tmp_path / "baseline.root", tmp_path / "tables.root", tmp_path / "printer.log"
    write_root(paths[0], base)
    write_root(paths[1], new)
    paths[2].write_text(printer(new) if text is None else text)
    return paths


def validate(paths):
    return check.validate(*paths, source_tag=TAG, source_branch=BRANCH)


def cli(paths, out):
    return ["validate", "--baseline", str(paths[0]), "--tables", str(paths[1]),
            "--printer-log", str(paths[2]), "--source-tag", TAG,
            "--source-branch", BRANCH, "--out", str(out)]


def test_valid_all_old_columns_nan_empty_and_unselected_events(tmp_path):
    paths = inputs(tmp_path)
    result, report = validate(paths)
    np.testing.assert_array_equal(result["event"], [33, 11, 22])
    np.testing.assert_array_equal(result["source_event"], [99, 2, 99])
    assert report["events"] == report["source_rows"] == 3
    assert report["unique_immediate_source_event_ids"] == 2
    assert report["comparison"]["all_old_columns_exact"]
    assert report["comparison"]["old_table_rows"]["dump/empty"] == 0
    assert report["comparison"]["old_columns_compared"] == sum(len(t) for t in fixtures()[0].values())
    assert report["printer"]["trigreport"] == dict(total=3, passed=3, failed=0)
    assert report["printer_source_ids_exact"]
    assert not report["source_independence_certified"]
    assert report["source_file_resolution"] == "not_performed"
    assert not result["source_file_resolved"].any()


@pytest.mark.parametrize("name,field", [(name, field) for name, table in fixtures()[0].items()
                                        for field in table if name != "empty"])
def test_every_old_column_is_compared(tmp_path, name, field):
    base, new = fixtures()
    value = new[name][field]
    if value.dtype.kind in "USO":
        new[name][field] = np.array(["different"] * len(value))
    elif value.dtype.kind == "b":
        value[0] = not value[0]
    elif value.dtype.kind == "f":
        value[0] = np.nextafter(value[0], np.inf, dtype=value.dtype)
    else:
        value[0] += 1
    with pytest.raises(ValueError, match="mismatch"):
        validate(inputs(tmp_path, base, new))


@pytest.mark.parametrize("defect", ["dtype", "nan-location", "row-order", "column-order", "missing-column",
                                     "extra-column", "missing-table", "extra-table", "extra-source-column",
                                     "baseline-source", "baseline-event-source", "baseline-kind",
                                     "old-kind-filter", "new-non-source-row", "empty-dtype"])
def test_exact_schema_order_and_only_minus_six_filter(tmp_path, defect):
    base, new = fixtures()
    if defect == "dtype":
        new["showers"]["energy"] = new["showers"]["energy"].astype(np.float64)
    elif defect == "nan-location":
        new["showers"]["time"][:] = [np.nan, 1.]
    elif defect == "row-order":
        new["products"] = {k: v[[2, 1, 0, 3, 4, 5]] for k, v in new["products"].items()}
    elif defect == "column-order":
        new["showers"] = dict(reversed(list(new["showers"].items())))
    elif defect == "missing-column":
        del new["events"]["g4_status"]
    elif defect == "extra-column":
        new["events"]["unexpected"] = np.zeros(3)
    elif defect == "missing-table":
        del new["empty"]
    elif defect == "extra-table":
        new["extra"] = dict(x=np.ones(1))
    elif defect == "extra-source-column":
        new["source_events"]["source_fid"] = np.zeros(3)
    elif defect == "baseline-source":
        base["source_events"] = deepcopy(new["source_events"])
    elif defect == "baseline-event-source":
        base["events"]["source_count"] = np.ones(3, dtype=int)
    elif defect == "baseline-kind":
        base["products"]["kind"][0] = -6
    elif defect == "old-kind-filter":
        new["products"]["kind"][0] = -6
    elif defect == "new-non-source-row":
        new["products"]["kind"][-1] = -5
    else:
        new["empty"]["key"] = new["empty"]["key"].astype(np.uint64)
    with pytest.raises(ValueError):
        validate(inputs(tmp_path, base, new))


def test_interleaved_new_products_preserve_old_relative_order(tmp_path):
    base, new = fixtures()
    new["products"] = {k: v[[0, 3, 1, 4, 2, 5]] for k, v in new["products"].items()}
    validate(inputs(tmp_path, base, new))


@pytest.mark.parametrize("value", [2**32 - 1, 2**53 + 1, 2**63 - 1])
def test_no_float_roundtrip_for_output_or_source_ids(tmp_path, value):
    base, new = fixtures()
    for data in (base, new):
        for table in data.values():
            if "event" in table:
                table["event"] = table["event"].astype(np.uint64)
                table["event"][table["event"] == 33] = value
    new["source_events"]["source_event"][0] = value
    result, _ = validate(inputs(tmp_path, base, new))
    assert int(result["event"][0]) == int(result["source_event"][0]) == value


@pytest.mark.parametrize("field", check.SOURCE_IDS)
def test_each_decoded_source_axis_must_match_export(tmp_path, field):
    base, new = fixtures()
    text = printer(new)
    new["source_events"][field][0] += 1
    with pytest.raises(ValueError, match="stock printer/exported " + field):
        validate(inputs(tmp_path, base, new, text))


@pytest.mark.parametrize("defect", ["absent", "count", "invalid", "index", "missing", "duplicate", "order", "orphan", "tag", "branch"])
def test_source_events_reader_remains_required_gate(tmp_path, defect):
    base, new = fixtures()
    text = printer(new)
    if defect == "absent":
        new["events"]["source_present"][0] = False
    elif defect == "count":
        new["events"]["source_count"][0] = 2
    elif defect == "invalid":
        new["source_events"]["source_valid"][0] = False
    elif defect == "index":
        new["source_events"]["index"][0] = 1
    elif defect in ("missing", "duplicate", "order"):
        indices = {"missing": [0, 1], "duplicate": [0, 0, 2], "order": [1, 0, 2]}[defect]
        new["source_events"] = {k: v[indices] for k, v in new["source_events"].items()}
    elif defect == "orphan":
        new["source_events"]["event"][0] = 999
    else:
        new["events"]["source_" + defect] = np.array(["wrong"] * 3)
    with pytest.raises(ValueError):
        validate(inputs(tmp_path, base, new, text))


@pytest.mark.parametrize("defect", ["missing-art", "missing-sequence", "duplicate-art", "duplicate-sequence", "sequence-first",
                                     "wrong-output-run", "wrong-output-subrun", "wrong-output-event", "ordinal", "short-art",
                                     "bad-art", "truncated-art", "bad-sequence", "multi-source", "empty-sequence",
                                     "wrong-tag", "extra-sequence", "negative", "float", "overflow", "very-wide",
                                     "trig-total", "trig-pass", "trig-fail", "trig-duplicate", "trig-missing", "trig-malformed",
                                     "completion-missing", "completion-failed", "completion-duplicate", "completion-not-terminal"])
def test_log_rejects_malformed_incomplete_duplicate_and_orphan_records(defect):
    _, new = fixtures()
    text = printer(new)
    art = next(line for line in text.splitlines() if line.startswith("Begin"))
    seq = next(line for line in text.splitlines() if line.startswith("EventIDSequence"))
    trig = "TrigReport Events total = 3 passed = 3 failed = 0"
    replacements = {
        "missing-art": (art + "\n", ""), "missing-sequence": (seq + "\n", ""),
        "duplicate-art": (art, art + "\n" + art), "duplicate-sequence": (seq, seq + "\n" + seq),
        "sequence-first": (art, seq + "\n" + art),
        "wrong-output-run": (art, art.replace("run: 7", "run: 8")),
        "wrong-output-subrun": (art, art.replace("subRun: 2", "subRun: 8")),
        "wrong-output-event": (art, art.replace("event: 33", "event: 22")),
        "ordinal": ("the 1st", "the 2nd"),
        "short-art": (art, "Begin processing the 1st record."),
        "bad-art": (art, art.replace("run:", "run=")),
        "truncated-art": (art, art.split(" at ")[0]),
        "bad-sequence": (seq, seq + " trailing garbage"),
        "multi-source": (seq, seq + " ( run: 2 subRun: 0 event: 1 )"),
        "empty-sequence": (seq, f"EventIDSequence {PRINTED_TAG} :"),
        "wrong-tag": (PRINTED_TAG, PRINTED_TAG.replace("TargetStopResampler", "Other")),
        "extra-sequence": (trig, seq + "\n" + trig),
        "negative": (seq, seq.replace("event: 99", "event: -1")),
        "float": (seq, seq.replace("event: 99", "event: 99.0")),
        "overflow": (seq, seq.replace("event: 99", f"event: {2**63}")),
        "very-wide": (seq, seq.replace("event: 99", f"event: {2**64}")),
        "trig-total": ("total = 3", "total = 2"), "trig-pass": ("passed = 3", "passed = 2"),
        "trig-fail": ("failed = 0", "failed = 1"), "trig-duplicate": (trig, trig + "\n" + trig),
        "trig-missing": (trig, ""), "trig-malformed": (trig, trig + " extra"),
        "completion-missing": (check.COMPLETION, ""),
        "completion-failed": ("status 0.", "status 1."),
        "completion-duplicate": (check.COMPLETION, check.COMPLETION + "\n" + check.COMPLETION),
        "completion-not-terminal": (check.COMPLETION, check.COMPLETION + "\nmore output"),
    }
    a, b = replacements[defect]
    with pytest.raises(ValueError):
        check.parse_printer(text.replace(a, b, 1), new["events"], source_tag=TAG)


def test_printer_whitespace_crlf_and_unrelated_trig_path_summary():
    _, new = fixtures()
    text = printer(new).replace(" :  (", "\t:\t(").replace("\n", "\r\n")
    text = text.replace(check.COMPLETION, "TrigReport Path Summary\r\n" + check.COMPLETION)
    result, report = check.parse_printer(text, new["events"], source_tag=TAG)
    assert result["source_event"].tolist() == [99, 2, 99]
    assert report["terminal_success"]


@pytest.mark.parametrize("defect", ["empty", "duplicate", "float", "negative", "overflow"])
def test_census_rejects_invalid_full_event_ids(tmp_path, defect):
    base, _ = fixtures()
    if defect == "empty":
        base["events"] = {k: v[:0] for k, v in base["events"].items()}
    elif defect == "duplicate":
        for field in check.EVENT:
            base["events"][field][1] = base["events"][field][0]
    else:
        base["events"]["event"] = {"float": np.array([33., 11., 22.]),
                                    "negative": np.array([-1, 11, 22]),
                                    "overflow": np.array([2**63, 11, 22], dtype=np.uint64)}[defect]
    paths = inputs(tmp_path, baseline=base)
    out = tmp_path / "never" / "events.npz"
    with pytest.raises(ValueError):
        check.main(["census", "--baseline", str(paths[0]), "--out", str(out)])
    assert not out.parent.exists()


def strict_json(text):
    def reject(value):
        pytest.fail("not strict JSON: " + value)
    return json.loads(text, parse_constant=reject)


def test_cli_outputs_strict_json_streamed_hashes_and_integer_arrays(tmp_path, monkeypatch, capsys):
    paths = inputs(tmp_path)
    original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out = tmp_path / "validated"

    def forbidden(path):
        pytest.fail("hashes must stream, not Path.read_bytes: " + str(path))

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    check.main(cli(paths, out))
    report = strict_json((out / "summary.json").read_text())
    assert strict_json(capsys.readouterr().out) == report
    assert report["input_sha256"] == dict(zip(("baseline", "tables", "printer_log"), original.values()))
    assert report["source_sha256"] == check.source_events.file_digest(check.__file__)
    assert set(p.name for p in out.iterdir()) == {"summary.json", "source_events.npz"}
    with np.load(out / "source_events.npz", allow_pickle=False) as archive:
        assert all(archive[field].dtype.kind in "iub" for field in archive.files)
        np.testing.assert_array_equal(archive["event"], [33, 11, 22])
    assert {p: check.source_events.file_digest(p) for p in paths} == original


def test_census_cli_full_order_and_original_integer_dtypes(tmp_path, capsys):
    base, _ = fixtures()
    base["events"]["event"] = np.array([2**53 + 1, 11, 22], dtype=np.uint64)
    path = tmp_path / "baseline.root"
    write_root(path, base)
    out = tmp_path / "new" / "expected.npz"
    check.main(["census", "--baseline", str(path), "--out", str(out)])
    with np.load(out, allow_pickle=False) as archive:
        assert archive.files == list(check.EVENT)
        for field in check.EVENT:
            assert archive[field].dtype == base["events"][field].dtype
            np.testing.assert_array_equal(archive[field], base["events"][field])
    report = strict_json(capsys.readouterr().out)
    assert report["events"] == 3
    assert report["input_sha256"]["baseline"] == check.source_events.file_digest(path)
    assert list(out.parent.iterdir()) == [out]


@pytest.mark.parametrize("command", ["census", "validate"])
@pytest.mark.parametrize("existing", ["file", "directory", "occupied", "dangling-symlink"])
def test_cli_never_overwrites_or_reads_inputs_if_output_exists(tmp_path, command, existing):
    out = tmp_path / "existing.npz"
    if existing == "file":
        out.write_text("keep")
    elif existing == "dangling-symlink":
        out.symlink_to(tmp_path / "absent-target")
    else:
        out.mkdir()
        if existing == "occupied":
            (out / "summary.json").write_text("keep")
    missing = tmp_path / "missing"
    argv = (cli((missing,) * 3, out) if command == "validate" else
            ["census", "--baseline", str(missing), "--out", str(out)])
    with pytest.raises(FileExistsError):
        check.main(argv)
    if existing == "file":
        assert out.read_text() == "keep"
    elif existing == "occupied":
        assert (out / "summary.json").read_text() == "keep"
    elif existing == "dangling-symlink":
        assert out.is_symlink() and not out.exists()
    else:
        assert not list(out.iterdir())


@pytest.mark.parametrize("defect", ["tables", "printer", "source-tag", "source-branch"])
def test_cli_validates_before_mkdir(tmp_path, defect):
    paths = inputs(tmp_path)
    out = tmp_path / "absent-parent" / "never"
    argv = cli(paths, out)
    if defect == "tables":
        base, new = fixtures()
        new["events"]["source_count"][0] = 0
        write_root(paths[1], new)
    elif defect == "printer":
        paths[2].write_text("truncated")
    else:
        argv[argv.index("--" + defect) + 1] = "wrong"
    with pytest.raises(ValueError):
        check.main(argv)
    assert not out.parent.exists()


@pytest.mark.parametrize("which", [0, 1, 2])
def test_validate_detects_each_input_mutation_before_mkdir(tmp_path, monkeypatch, which):
    paths = inputs(tmp_path)
    out = tmp_path / "absent-parent" / "never"
    original = check.validate

    def mutated(*args, **kwargs):
        result = original(*args, **kwargs)
        with paths[which].open("ab") as stream:
            stream.write(b"changed after validation")
        return result

    monkeypatch.setattr(check, "validate", mutated)
    with pytest.raises(ValueError, match="changed during"):
        check.main(cli(paths, out))
    assert not out.parent.exists()


def test_census_detects_baseline_mutation_before_mkdir(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    out = tmp_path / "absent-parent" / "never.npz"
    original = check.read_census

    def mutated(path):
        result = original(path)
        with path.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(check, "read_census", mutated)
    with pytest.raises(ValueError, match="baseline changed"):
        check.main(["census", "--baseline", str(paths[0]), "--out", str(out)])
    assert not out.parent.exists()


def test_cli_module_entrypoint_and_required_explicit_source_arguments(tmp_path):
    paths = inputs(tmp_path)
    out = tmp_path / "module-output"
    command = [sys.executable, "-m", "calofastsim.showers.source_event_check"]
    run = subprocess.run(command + cli(paths, out), text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert strict_json(run.stdout)["events"] == 3
    argv = cli(paths, tmp_path / "never")
    del argv[argv.index("--source-tag"):argv.index("--source-tag") + 2]
    run = subprocess.run(command + argv, text=True, capture_output=True)
    assert run.returncode == 2 and "--source-tag" in run.stderr
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("extra", ["( run: 107 subRun: 4 event: 99 )",
                                    "( run: broken subRun: 4 event: 99 )",
                                    "EventIDSequence broken",
                                    "Begin processing broken"])
def test_log_orphan_continuation_or_malformed_extra_row_is_not_ignored(extra):
    _, new = fixtures()
    text = printer(new).replace("TrigReport Events", extra + "\nTrigReport Events")
    with pytest.raises(ValueError):
        check.parse_printer(text, new["events"], source_tag=TAG)


def test_completion_must_be_whole_line_not_suffix_of_error():
    _, new = fixtures()
    text = printer(new).replace(check.COMPLETION, "ERROR: expected " + check.COMPLETION)
    with pytest.raises(ValueError, match="completion"):
        check.parse_printer(text, new["events"], source_tag=TAG)


@pytest.mark.parametrize("defect", ["duplicate-cycle", "extra-object", "extra-directory"])
def test_root_extra_objects_and_hidden_key_cycles_fail_closed(tmp_path, defect):
    paths = inputs(tmp_path)
    with uproot.update(paths[1]) as root:
        if defect == "duplicate-cycle":
            root.mktree("dump/empty", fixtures()[1]["empty"])
        elif defect == "extra-object":
            root["note"] = "unapproved metadata"
        else:
            root.mkdir("extra")
    with pytest.raises(ValueError, match="cycles|schema|only numeric"):
        validate(paths)


def test_source_reader_receives_baseline_full_census_and_explicit_product(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    original = check.source_events.read
    calls = []

    def audited(path, expected, *, source_tag, source_branch):
        assert path == paths[1]
        assert (source_tag, source_branch) == (TAG, BRANCH)
        for field in check.EVENT:
            np.testing.assert_array_equal(expected[field], fixtures()[0]["events"][field])
        calls.append(True)
        return original(path, expected, source_tag=source_tag, source_branch=source_branch)

    monkeypatch.setattr(check.source_events, "read", audited)
    validate(paths)
    assert calls == [True]


def test_census_requires_npz_suffix_without_implicit_filename_rewrite(tmp_path):
    paths = inputs(tmp_path)
    out = tmp_path / "never" / "census"
    with pytest.raises(ValueError, match=".npz suffix"):
        check.main(["census", "--baseline", str(paths[0]), "--out", str(out)])
    assert not out.parent.exists()


def test_output_race_uses_exclusive_creation(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    out = tmp_path / "race.npz"
    original = check.read_census

    def occupied(path):
        result = original(path)
        out.write_bytes(b"created by another process")
        return result

    monkeypatch.setattr(check, "read_census", occupied)
    with pytest.raises(FileExistsError):
        check.main(["census", "--baseline", str(paths[0]), "--out", str(out)])
    assert out.read_bytes() == b"created by another process"


@pytest.mark.parametrize("defect", [None, "value", "boundary", "dtype"])
def test_jagged_geometry_neighbors_exact(tmp_path, defect):
    base, new = fixtures()
    neighbors = ak.values_astype(ak.Array([[1, 2], [], [3]]), np.uint32)
    base["neighbors"] = dict(values=neighbors)
    changed = neighbors
    if defect == "value":
        changed = ak.values_astype(ak.Array([[1, 9], [], [3]]), np.uint32)
    elif defect == "boundary":
        changed = ak.values_astype(ak.Array([[1], [2], [3]]), np.uint32)
    elif defect == "dtype":
        changed = ak.values_astype(neighbors, np.uint64)
    new["neighbors"] = dict(values=changed)
    paths = inputs(tmp_path, base, new)
    if defect:
        with pytest.raises(ValueError, match="mismatch"):
            validate(paths)
    else:
        assert validate(paths)[1]["comparison"]["all_old_columns_exact"]


def test_exact_comparison_preserves_adjacent_large_integer_bits():
    left = ak.Array(np.array([2**63 + 1], np.uint64))
    right = ak.Array(np.array([2**63 + 2], np.uint64))
    with pytest.raises(ValueError, match="ordered values mismatch"):
        check._equal(left, right, "uint64")
