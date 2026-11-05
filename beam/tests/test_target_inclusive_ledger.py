"""Stock text ledger validates every required event, product and trigger record."""
import hashlib
import json

import numpy as np
import pytest

from beamfastsim.target_inclusive_ledger import main, read_log, summary


def event(event_id=1, status=0, accepted=True, process="R1Inclusive", run=1430, subrun=6000,
          tracks=50, overflow=0, steps=0, triggers=True, timing=True):
    tail = " CPU:    0.125 Wall:    0.150" if timing else ""
    text = (f" ###############  PrintModule Run/Subrun/Event {run:>9} {subrun:>9} {event_id:>9}\n"
            f"\nProductPrint mu2e::StatusG4_g4run__{process}\n"
            f"  status: {status:>3} nG4Tracks: {tracks:>7} overflowSimP: {overflow:>2}"
            f" nStepLimit: {steps:>2}{tail}\n")
    if triggers:
        text += (f"\nProductPrint art::TriggerResults_TriggerResults__{process}\n"
                 f"  {'pass' if accepted else 'fail'}  0  PrimaryPath\n"
                 "  pass  1  MonitorPath\n")
    return text


def read(tmp_path, text, expected=1, **kwargs):
    path = tmp_path / "stock.log"
    path.write_text(text)
    return read_log(path, expected, **kwargs)


def test_preserves_full_integer_ids_status_counts_order_and_source_sha(tmp_path):
    text = ("begin job\n ###############  PrintModule Run/Subrun      1430      6000\n"
            + event(91, 0, True, run=16777217)
            + "unrelated diagnostic\nProductPrint mu2e::GenParticleCollection_generate__R1Inclusive\n"
            + "ordinary product payload\n"
            + event(7, 1, False, steps=3)
            + event(12, 10, True, tracks=900, overflow=1)
            + "end job\n")
    ledger = read(tmp_path, text, 3)
    assert all(isinstance(value, np.ndarray) for value in ledger.values())
    np.testing.assert_array_equal(ledger["identities"], [[16777217, 6000, 91], [1430, 6000, 7], [1430, 6000, 12]])
    assert ledger["identities"].dtype == np.int64
    np.testing.assert_array_equal(ledger["accepted"], [1, 0, 1])
    np.testing.assert_array_equal(ledger["status"], [0, 1, 10])
    np.testing.assert_array_equal(ledger["nG4Tracks"], [50, 50, 900])
    np.testing.assert_array_equal(ledger["overflowSimP"], [0, 0, 1])
    np.testing.assert_array_equal(ledger["nStepLimit"], [0, 3, 0])
    assert ledger["source_sha256"].item() == hashlib.sha256(text.encode()).hexdigest()
    report = summary(ledger)
    assert report["status_counts"] == {"0": 1, "1": 1, "10": 1}
    assert report["status_zero_events"] == 1
    assert report["nonzero_status_events"] == 2
    assert report["accepted_events"] == 2 and report["rejected_events"] == 1
    assert report["missing_trigger_events"] == 0
    assert report["overflow_events"] == 1 and report["killed_step_limit_tracks"] == 3
    assert report["nG4Tracks"] == 1000
    assert report["zero_response_certified"] is False
    json.dumps(report, allow_nan=False)


def test_generation_fixture_without_triggers_is_unknown_not_rejected(tmp_path):
    ledger = read(tmp_path, event(triggers=False, timing=False), require_triggers=False)
    np.testing.assert_array_equal(ledger["accepted"], [-1])
    report = summary(ledger)
    assert report["accepted_events"] == report["rejected_events"] == 0
    assert report["missing_trigger_events"] == 1
    with pytest.raises(ValueError):
        read(tmp_path, event(triggers=False))


def test_optional_trigger_mode_still_parses_present_decisions(tmp_path):
    ledger = read(tmp_path, event(1, accepted=False) + event(2, triggers=False), 2, require_triggers=False)
    np.testing.assert_array_equal(ledger["accepted"], [0, -1])


@pytest.mark.parametrize("type_name", ["mu2e::StatusG4", "mu2eStatusG4", "StatusG4"])
@pytest.mark.parametrize("trigger_type", ["art::TriggerResults", "artTriggerResults", "TriggerResults"])
def test_stock_branch_type_spellings_and_custom_process(tmp_path, type_name, trigger_type):
    text = event(process="OtherProcess").replace("mu2e::StatusG4", type_name).replace("art::TriggerResults", trigger_type)
    ledger = read(tmp_path, text, process="OtherProcess")
    assert ledger["process"].item() == "OtherProcess"
    assert ledger["accepted"].tolist() == [1]


@pytest.mark.parametrize("token", ["1.5", "1e3", "1x", "-1", "nan", "9223372036854775808"])
@pytest.mark.parametrize("field", ["run", "subrun", "event_id"])
def test_noninteger_or_out_of_range_identity_refused(tmp_path, field, token):
    text = event()
    header, rest = text.split("\n", 1)
    ids = ["1430", "6000", "1"]
    ids[["run", "subrun", "event_id"].index(field)] = token
    text = " ###############  PrintModule Run/Subrun/Event " + " ".join(ids) + "\n" + rest
    with pytest.raises(ValueError):
        read(tmp_path, text)


@pytest.mark.parametrize("suffix", [" extra", " 9", ".0"])
def test_header_must_match_complete_line(tmp_path, suffix):
    first, rest = event().split("\n", 1)
    with pytest.raises(ValueError):
        read(tmp_path, first + suffix + "\n" + rest)


def test_duplicate_ids_refused_but_same_event_number_in_different_subrun_valid(tmp_path):
    with pytest.raises(ValueError, match="duplicate event"):
        read(tmp_path, event() + event(), 2)
    ledger = read(tmp_path, event(subrun=6000) + event(subrun=6001), 2)
    assert ledger["identities"].shape == (2, 3)


@pytest.mark.parametrize("expected", [0, 2])
def test_event_count_mismatch(tmp_path, expected):
    with pytest.raises(ValueError, match="count mismatch"):
        read(tmp_path, event(), expected)


@pytest.mark.parametrize("expected", [-1, 1.0, True, "1"])
def test_invalid_expected_events(tmp_path, expected):
    with pytest.raises(ValueError, match="expected_events"):
        read(tmp_path, event(), expected)


@pytest.mark.parametrize("process", ["", "R1 Inclusive", "R1Inclusive\n", "R1.Inclusive", None])
def test_invalid_process_argument(tmp_path, process):
    with pytest.raises(ValueError, match="process"):
        read(tmp_path, event(), process=process)


@pytest.mark.parametrize("replacement", ["2", "9", "11", "2147483647", "-1", "nan", "1.0"])
def test_unknown_or_malformed_status_refused(tmp_path, replacement):
    with pytest.raises(ValueError):
        read(tmp_path, event().replace("status:   0", f"status: {replacement}"))


@pytest.mark.parametrize("old,new", [
    ("nG4Tracks:      50", "nG4Tracks: 50.0"),
    ("nG4Tracks:      50", "nG4Tracks: -1"),
    ("nG4Tracks:      50", "nG4Tracks: 9223372036854775808"),
    ("overflowSimP:  0", "overflowSimP: 2"),
    ("overflowSimP:  0", "overflowSimP: false"),
    ("nStepLimit:  0", "nStepLimit: 0.5"),
    ("nStepLimit:  0", "nStepLimit: -1"),
    ("nStepLimit:  0", "stepLimit: 0"),
    ("Wall:    0.150", "Wall: nan"),
])
def test_malformed_required_status_values_refused(tmp_path, old, new):
    original = event()
    assert old in original
    with pytest.raises(ValueError):
        read(tmp_path, original.replace(old, new))


@pytest.mark.parametrize("text", [
    event().replace("ProductPrint mu2e::StatusG4_g4run__R1Inclusive\n", ""),
    event().replace("mu2e::StatusG4_g4run__R1Inclusive", "mu2e::StatusG4_g4run__Wrong"),
    event().replace("mu2e::StatusG4_g4run__R1Inclusive", "mu2e::StatusG4_other__R1Inclusive"),
    event().replace("mu2e::StatusG4_g4run__R1Inclusive", "mu2e::StatusG4_g4run__R1Inclusive extra"),
    event().replace("  status:", "ProductPrint mu2e::StatusG4_g4run__R1Inclusive\n  status:"),
    event().replace("  status:", "ProductPrint mu2e::StatusG4_g4run__Wrong\n  status:"),
    event().replace(" CPU:", " nStepLimit: 0 CPU:"),
    event().replace("  status:   0", "  status:   0\n  status:   0"),
])
def test_wrong_missing_duplicate_status_product_or_record(tmp_path, text):
    with pytest.raises(ValueError):
        read(tmp_path, text)


def test_status_rows_cannot_slide_across_event_boundaries(tmp_path):
    first = event(1).splitlines(keepends=True)
    first = "".join(line for line in first if "status:" not in line)
    second = event(2).replace("  status:", "  status: 0 nG4Tracks: 50 overflowSimP: 0 nStepLimit: 0\n  status:")
    with pytest.raises(ValueError):
        read(tmp_path, first + second, 2)


@pytest.mark.parametrize("text", [
    event().replace("  pass  0  PrimaryPath\n", ""),
    event().replace("  pass  0  PrimaryPath", "  pass  0  PrimaryPath\n  fail  0  PrimaryPath"),
    event().replace("  pass  0  PrimaryPath", "  true  0  PrimaryPath"),
    event().replace("  pass  0  PrimaryPath", "  pass  0.5  PrimaryPath"),
    event().replace("  pass  0  PrimaryPath", "  pass  -1  PrimaryPath"),
    event().replace("  pass  0  PrimaryPath", "  pass  9223372036854775808  PrimaryPath"),
    event().replace("  pass  0  PrimaryPath", "  pass  0  PrimaryPath extra"),
    event().replace("art::TriggerResults_TriggerResults__R1Inclusive", "art::TriggerResults_TriggerResults__Wrong"),
    event().replace("ProductPrint art::TriggerResults_TriggerResults__R1Inclusive\n", ""),
    event() + "\nProductPrint art::TriggerResults_TriggerResults__R1Inclusive\n  pass  0  PrimaryPath\n",
    event().replace("  pass  0  PrimaryPath", "ProductPrint mu2e::GenParticleCollection_generate__R1Inclusive\n  pass  0  PrimaryPath"),
])
@pytest.mark.parametrize("required", [True, False])
def test_malformed_or_duplicate_present_trigger_refused_even_when_optional(tmp_path, text, required):
    with pytest.raises(ValueError):
        read(tmp_path, text, require_triggers=required)


@pytest.mark.parametrize("extra", [
    "ProductPrint mu2e::StatusG4_g4run__R1Inclusive\n  status: 0 nG4Tracks: 1 overflowSimP: 0 nStepLimit: 0\n",
    "ProductPrint art::TriggerResults_TriggerResults__R1Inclusive\n  pass  0  PrimaryPath\n",
])
def test_required_records_outside_event_or_after_subrun_boundary_refused(tmp_path, extra):
    with pytest.raises(ValueError):
        read(tmp_path, extra + event())
    with pytest.raises(ValueError):
        read(tmp_path, event() + " ###############  PrintModule Run/Subrun 1430 6001\n" + extra)


def test_whitespace_crlf_and_interleaved_text(tmp_path):
    text = event().replace("  status:", "interleaved diagnostic\n\tstatus:")
    text = text.replace("  pass  0  PrimaryPath", "another diagnostic\n\tpass\t0\tPrimaryPath")
    path = tmp_path / "crlf.log"
    path.write_bytes(text.replace("\n", "\r\n").encode())
    ledger = read_log(path, 1)
    assert ledger["accepted"].tolist() == [1]
    assert ledger["source_sha256"].item() == hashlib.sha256(path.read_bytes()).hexdigest()


def test_empty_log_is_valid_only_for_zero_expected_events(tmp_path):
    ledger = read(tmp_path, "ordinary job startup and shutdown\n", 0)
    assert ledger["identities"].shape == (0, 3)
    assert ledger["accepted"].shape == ledger["status"].shape == (0,)
    report = summary(ledger)
    assert report["events"] == 0 and report["status_counts"] == {}
    assert report["event_id_min"] is report["event_id_max"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("suffix", ["", ".npz"])
def test_cli_writes_nonpickled_npz_and_strict_json(tmp_path, suffix, capsys):
    path = tmp_path / "stock.log"
    path.write_text(event(status=1, steps=4, process="ExampleProcess"))
    out = tmp_path / ("ledger" + suffix)
    main(["--log", str(path), "--expected-events", "1", "--process", "ExampleProcess", "--out", str(out)])
    with np.load(tmp_path / "ledger.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["identities"], [[1430, 6000, 1]])
        np.testing.assert_array_equal(saved["status"], [1])
        assert all(saved[name].dtype.kind != "O" for name in saved.files)
        expected = summary(saved)
    report = json.loads((tmp_path / "ledger.json").read_text())
    assert report == expected == json.loads(capsys.readouterr().out)
    assert report["nonzero_status_events"] == 1


@pytest.mark.parametrize("existing", ["ledger.npz", "ledger.json"])
def test_cli_refuses_either_existing_output_without_modification(tmp_path, existing):
    path = tmp_path / existing
    path.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        main(["--log", str(tmp_path / "absent.log"), "--expected-events", "1", "--out", str(tmp_path / "ledger")])
    assert path.read_bytes() == b"do not overwrite"
    other = "ledger.json" if existing.endswith("npz") else "ledger.npz"
    assert not (tmp_path / other).exists()


def test_cli_invalid_input_creates_no_outputs(tmp_path):
    log = tmp_path / "stock.log"
    log.write_text(event(triggers=False))
    with pytest.raises(ValueError):
        main(["--log", str(log), "--expected-events", "1", "--out", str(tmp_path / "ledger")])
    assert not (tmp_path / "ledger.npz").exists()
    assert not (tmp_path / "ledger.json").exists()


def test_art_summary_mentions_of_primary_path_are_not_trigger_records(tmp_path):
    text = event() + '\nTrigReport ---------- Path   Summary ------------\n'
    text += 'TrigReport 0 2000 1000 1000 0 PrimaryPath\n'
    text += 'TimeReport 0.1 0.2 PrimaryPath\n'
    assert read(tmp_path, text)['accepted'].tolist() == [1]


def test_cli_reconciles_saved_read_with_independent_generation_census(tmp_path):
    log, original = tmp_path / 'read.log', tmp_path / 'generation.log'
    log.write_text(event(1) + event(2, status=1, steps=3))
    original.write_text(event(1, triggers=False) + event(2, status=1, steps=3, triggers=False))
    main(['--log', str(log), '--generation-log', str(original), '--expected-events', '2', '--out', str(tmp_path / 'ledger')])
    assert json.loads((tmp_path / 'ledger.json').read_text())['generation_census_reconciled']
    original.write_text(event(1, triggers=False) + event(2, status=0, triggers=False))
    with pytest.raises(ValueError, match='independent generation'):
        main(['--log', str(log), '--generation-log', str(original), '--expected-events', '2', '--out', str(tmp_path / 'bad')])


def test_generation_reconciliation_uses_event_keys_not_read_order(tmp_path):
    log, original = tmp_path / 'read.log', tmp_path / 'generation.log'
    log.write_text(event(2, status=1, steps=3) + event(1))
    original.write_text(event(1, triggers=False) + event(2, status=1, steps=3, triggers=False))
    main(['--log', str(log), '--generation-log', str(original), '--expected-events', '2', '--out', str(tmp_path / 'shuffled')])
    assert json.loads((tmp_path / 'shuffled.json').read_text())['generation_census_reconciled']
