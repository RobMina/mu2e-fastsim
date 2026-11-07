"""Saved-only throughput validation, independent timings and empty-inclusive censuses."""
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from mixing import throughput as tp

IDS = np.array([[1, 2, 7], [1, 3, 7], [2**63 + 1, 2, 7]], dtype=np.uint64)
TIME = {"wall_s": 10., "user_s": 7., "system_s": 1., "max_rss_kb": 2048, "exit_status": 0}


def verbose_time(elapsed="1:02.50", exit_status=0):
    return f"""Command being timed: "mu2e -c saved.fcl"
 User time (seconds): 52.00
 System time (seconds): 2.00
 Elapsed (wall clock) time (h:mm:ss or m:ss): {elapsed}
 Maximum resident set size (kbytes): 4096
 Exit status: {exit_status}
"""


def art_log():
    return """TrigReport Events total = 3 passed = 2 failed = 1
TimeReport CPU = 51.2 Real = 60.25
Art has completed and will exit with status 0.
"""


def driver():
    stages = {"read_validate": .3, "sample": .1, "overlay_validate": .4, "adapter_validate": .2}
    return {"events": 3, "input_mode": "direct_frames", "paired_comparisons_skipped": True,
            "seed": 42, "chunk": 32, "devices": ["recorded GPU"], "setup_s": 2., "input_s": 1.2,
            "sampling_write_s": .05, "write_s": .3, "frame_preparation_s": stages,
            "preparation": {"passes": [{"pass": 0, "seconds": 1., "stages": stages}],
                            "exact_check_s": .001, "sampling_reference_identical": False,
                            "repeats_identical": False},
            "passes": [{"pass": 0, "combined_response_s": 4., "crv": {"seconds": 1.},
                        "tracker": {"seconds": 2.}}]}


def save_sampling(directory, change=False):
    arrays = {"ids": IDS.copy(), "intensity": np.array([1., 2., 3.]),
              "donors": np.array([1, 0, 2], dtype=np.int64)}
    if change:
        arrays["donors"][0] = 0
    def receipt(key):
        return {"npz_key": key, "shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)}
    metadata = {"campaign": {"mode": "sequential"}, "seed": 42, "stream_order": ["pool"],
                "rates": [1.], "intensity": receipt("intensity"), "multiplicities": [1, 1, 1],
                "sources": {"signal": "signal.root"}, "source_fingerprints": {"signal": "saved hash"},
                "signal_events_full": 3, "signal_event_ids": receipt("ids"), "pool_event_counts": {"pool": 3},
                "estimated_selected_rows": {"tracker": 2}, "actual_rows": {"tracker": 2},
                "selections": {"pool": {"donors": receipt("donors")}}, "timing_s": {"read": 1.}}
    (directory / "sampling.json").write_text(json.dumps(metadata))
    np.savez(directory / "sampling.npz", **arrays)


def save_root(path, native=False, ids=None, changed=False, tracker=True):
    ids = IDS if ids is None else ids
    fields = dict(zip(("run", "subrun", "event"), ids.T))
    prefix = "crvdump/" if native else "crvfast/"
    with uproot.recreate(path) as root:
        root.mktree(prefix + "events", {key: values[:1] for key, values in fields.items()})
        counts = {"n" + key: np.array([1, 0, 0]) for key in ("digis", "pulses", "clusters")}
        root.mktree(prefix + "events", fields if native else {**fields, **counts})
        for key in ("digis", "pulses", "clusters"):
            root.mktree(prefix + key, {"value": np.array([1. + changed])})
        if tracker:
            root.mktree("trkdigi/events", {**fields, "d_sid": ak.Array([[1, 2], [], []]),
                                          "d_adc": ak.Array([[4, 5, 6, 7], [], []])})


@pytest.fixture
def saved(tmp_path):
    rootdir, reference, stock = (tmp_path / name for name in ("bench", "v35", "stock"))
    for path in (rootdir, reference, stock):
        path.mkdir()
    for directory in (reference, *(rootdir / name for name in tp.DEFAULT_RUNS)):
        directory.mkdir(exist_ok=True)
        save_root(directory / "fast.root")
        save_sampling(directory)
        (directory / "report.json").write_text(json.dumps(driver()))
    for name in tp.DEFAULT_RUNS:
        (rootdir / (name + ".time.json")).write_text(json.dumps(dict(TIME, wall_s=12.)))
        (rootdir / (name + ".process.time.json")).write_text(json.dumps(TIME))
    save_root(stock / "response.root", native=True)
    (stock / "art.time").write_text(verbose_time())
    (stock / "art.log").write_text(art_log())
    return rootdir, reference, stock


@pytest.mark.parametrize("elapsed,expected", [("1:02.50", 62.5), ("12:34", 754), ("1:02:03.25", 3723.25), ("0:00.01", .01)])
def test_gnu_verbose_elapsed_formats(elapsed, expected):
    result = tp.parse_gnu_time(verbose_time(elapsed))
    assert result["wall_s"] == expected
    assert result["user_s"] == 52
    assert result["system_s"] == 2
    assert result["max_rss_kb"] == 4096
    assert result["exit_status"] == 0


@pytest.mark.parametrize("elapsed", ["0:00", "1:60", "1:60:00", "1:00:60", "nan", "inf", "-1:01", "1:02:03:04", "1.25", "1:2:3"])
def test_bad_verbose_elapsed(elapsed):
    with pytest.raises(ValueError):
        tp.parse_gnu_time(verbose_time(elapsed))


def test_gnu_verbose_requires_one_successful_complete_record():
    for text in (verbose_time(exit_status=2), verbose_time() * 2,
                 verbose_time().replace(" Exit status: 0\n", ""),
                 verbose_time().replace("52.00", "nan"),
                 verbose_time().replace("4096", "0")):
        with pytest.raises(ValueError):
            tp.parse_gnu_time(text)


def test_json_time_accepts_rounded_zero_cpu_but_not_zero_wall():
    assert tp.parse_time_json(json.dumps(dict(TIME, user_s=0., system_s=0.)))["wall_s"] == 10.


@pytest.mark.parametrize("key,value", [("wall_s", 0), ("wall_s", -1), ("wall_s", float("inf")),
                                      ("wall_s", float("nan")), ("wall_s", True), ("user_s", -1),
                                      ("system_s", float("nan")), ("max_rss_kb", 0),
                                      ("max_rss_kb", 4.5), ("exit_status", 1), ("exit_status", False)])
def test_bad_json_time(key, value):
    with pytest.raises(ValueError):
        tp.parse_time_json(json.dumps(dict(TIME, **{key: value})))


@pytest.mark.parametrize("text", ["", "{}", "[]", "null", json.dumps(TIME) * 2,
                                 json.dumps(TIME) + "\n" + json.dumps(TIME),
                                 json.dumps(TIME).replace('"wall_s": 10.0', '"wall_s": 10, "wall_s": 20')])
def test_json_time_rejects_missing_ambiguous_or_nonobject_records(text):
    with pytest.raises(ValueError):
        tp.parse_time_json(text)


def test_stock_log_filters_are_not_process_failures_and_scopes_are_separate():
    result = tp.parse_art_log(art_log(), 3)
    assert result["rejected_all_paths"] == 1
    assert result["cpu_s"] == 51.2
    assert result["real_s"] == 60.25


@pytest.mark.parametrize("old,new", [("status 0.", "status 1."),
                                      ("Art has completed and will exit with status 0.", ""),
                                      ("total = 3", "total = 4"), ("passed = 2", "passed = 3"),
                                      ("Real = 60.25", "Real = nan"), ("Real = 60.25", "Real = 0"),
                                      ("CPU = 51.2", "CPU = -1"), ("CPU = 51.2", "CPU = inf")])
def test_stock_log_rejects_failure_incomplete_census_and_bad_timing(old, new):
    with pytest.raises(ValueError):
        tp.parse_art_log(art_log().replace(old, new), 3)


def test_duplicate_stock_logs_are_not_a_success():
    with pytest.raises(ValueError):
        tp.parse_art_log(art_log() * 2, 3)


def test_report_counts_highest_cycle_and_empty_events_and_records_scopes(saved):
    result = tp.report(*saved)
    assert result["events"] == 3
    assert result["run_order"] == list(tp.DEFAULT_RUNS)
    assert result["stock"]["external_process"]["wall_s"] == 62.5
    assert result["stock"]["art_time_report"]["real_s"] == 60.25
    assert result["stock"]["rerun"] is False
    assert result["hardware"] is None and result["stock"]["hardware"] is None
    assert result["stock"]["endpoint"] == "hardware not recorded"
    assert "empty detector events" in result["denominator"]
    assert "not a cold OS" in result["cache_scope"]
    assert "NOT fully warmed in-process" in result["cache_scope"]
    assert "devices and timing boundaries differ" in result["endpoint_scope"]
    assert "historical_ratio_scope" not in result
    assert "setup_s includes input_s" in result["stage_scope"]
    for run in result["runs"]:
        assert run["census"]["events"] == 3
        assert run["census"]["tracker_empty_events"] == 2
        assert run["census"]["crv"]["empty_digis_events"] == 2
        assert run["python_process"]["events_per_s"] == .3
        assert run["outer_container_process"]["events_per_s"] == .25
        assert run["response_plus_write"]["seconds"] == 4.3
        assert run["validation"]["sampling_identical"] is True
        assert len(run["validation"]["outputs"]["objects"]) == 5
        assert "descriptive_historical_stock_wall_over_python_wall" not in run
        assert len(run["artifacts"]["fast.root"]["sha256"]) == 64
    json.dumps(result, allow_nan=False)
    assert result == tp.report(*saved)


def test_optional_hardware_is_verbatim_not_inferred(saved):
    (saved[0] / "hardware.txt").write_text("captured node description\n")
    (saved[0] / "stock_hardware.txt").write_text("original stock node description\n")
    result = tp.report(*saved)
    assert result["hardware"]["text"] == "captured node description\n"
    assert result["stock"]["hardware"]["text"] == "original stock node description\n"


def test_explicit_ordered_names(saved):
    names = ("private_first", "reuse_first", "reuse_second")
    for old, new in zip(tp.DEFAULT_RUNS, names):
        (saved[0] / old).rename(saved[0] / new)
        for suffix in (".time.json", ".process.time.json"):
            (saved[0] / (old + suffix)).rename(saved[0] / (new + suffix))
    result = tp.report(*saved, run_names=names)
    assert result["run_order"] == list(names)
    assert result["runs"][0]["cache_state"] == "fresh_private_cache"
    assert result["runs"][1]["cache_state"] == "populated_cache_fresh_process"


@pytest.mark.parametrize("names", [("one", "one", "two"), ("../outside", "two", "three"), ("one",)])
def test_run_names_must_be_unique_ordered_direct_children(saved, names):
    with pytest.raises(ValueError):
        tp.report(*saved, run_names=names)


@pytest.mark.parametrize("mutation", ["events", "prepare", "response", "checks", "native", "paired_payload", "reference_check", "setup", "zero", "nonfinite", "missing_tracker"])
def test_driver_single_pass_contract(saved, mutation):
    path = saved[0] / tp.DEFAULT_RUNS[0] / "report.json"
    data = json.loads(path.read_text())
    if mutation == "events":
        data["events"] = 4
    elif mutation == "prepare":
        data["preparation"]["passes"] *= 2
    elif mutation == "response":
        data["passes"] *= 2
    elif mutation == "checks":
        data["paired_comparisons_skipped"] = False
    elif mutation == "native":
        data["input_mode"] = "native_cache"
    elif mutation == "paired_payload":
        data["crv_paired"] = {"ratio": 1}
    elif mutation == "reference_check":
        data["preparation"]["sampling_reference_identical"] = True
    elif mutation == "setup":
        data["setup_s"] = data["input_s"] / 2
    elif mutation == "zero":
        data["passes"][0]["combined_response_s"] = 0
    elif mutation == "nonfinite":
        data["write_s"] = float("nan")
    else:
        del data["passes"][0]["tracker"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        tp.report(*saved)


@pytest.mark.parametrize("name", tp.DEFAULT_RUNS)
def test_every_run_has_real_sampling_comparison(saved, name):
    save_sampling(saved[0] / name, change=True)
    with pytest.raises(ValueError, match="saved sampling array changed"):
        tp.report(*saved)


@pytest.mark.parametrize("name", tp.DEFAULT_RUNS)
def test_every_run_has_real_output_comparison(saved, name):
    save_root(saved[0] / name / "fast.root", changed=True)
    with pytest.raises(ValueError, match="output column changed"):
        tp.report(*saved)


def test_sampling_npz_metadata_must_describe_actual_arrays(saved):
    path = saved[0] / tp.DEFAULT_RUNS[0] / "sampling.json"
    data = json.loads(path.read_text())
    data["signal_event_ids"]["dtype"] = "float64"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="shape/dtype"):
        tp.report(*saved)


def test_changed_sampling_source_identity_is_not_ignored(saved):
    path = saved[0] / tp.DEFAULT_RUNS[0] / "sampling.json"
    data = json.loads(path.read_text())
    data["source_fingerprints"]["signal"] = "other hash"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="saved sampling value changed"):
        tp.report(*saved)


@pytest.mark.parametrize("role", ["candidate", "stock"])
@pytest.mark.parametrize("change", ["subrun", "order", "duplicate"])
def test_full_ids_not_only_event_numbers_must_match(saved, role, change):
    ids = IDS.copy()
    if change == "subrun":
        ids[1, 1] += 1
    elif change == "order":
        ids = ids[::-1]
    else:
        ids[1] = ids[0]
    path = saved[2] / "response.root" if role == "stock" else saved[0] / tp.DEFAULT_RUNS[0] / "fast.root"
    save_root(path, native=role == "stock", ids=ids)
    with pytest.raises(ValueError, match="identities|census"):
        tp.report(*saved)


@pytest.mark.parametrize("name", tp.DEFAULT_RUNS)
@pytest.mark.parametrize("suffix", [".time.json", ".process.time.json"])
def test_both_external_time_records_required_successful_for_every_run(saved, name, suffix):
    (saved[0] / (name + suffix)).write_text(json.dumps(dict(TIME, exit_status=1)))
    with pytest.raises(ValueError, match="exit_status"):
        tp.report(*saved)


def test_impossible_nested_process_timing_is_rejected(saved):
    (saved[0] / (tp.DEFAULT_RUNS[0] + ".process.time.json")).write_text(json.dumps(dict(TIME, wall_s=20.)))
    with pytest.raises(ValueError, match="exceeds outer"):
        tp.report(*saved)


def test_overflow_rate_is_rejected(saved):
    (saved[0] / (tp.DEFAULT_RUNS[0] + ".process.time.json")).write_text(json.dumps(dict(TIME, wall_s=1e-320)))
    with pytest.raises(ValueError, match="finite"):
        tp.report(*saved)


def cli_args(saved, output):
    return [str(saved[0]), "--reference", str(saved[1]), "--stock", str(saved[2]), "--out", str(output)]


def test_cli_publishes_once_and_never_overwrites(saved, tmp_path):
    output = tmp_path / "throughput.json"
    assert tp.main(cli_args(saved, output)) == 0
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        tp.main(cli_args(saved, output))
    assert error.value.code == 1
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".throughput-*"))


def test_failed_validation_preserves_all_state_and_does_not_publish(saved, tmp_path):
    output = tmp_path / "throughput.json"
    (saved[2] / "art.time").write_text(verbose_time(exit_status=1))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit) as error:
        tp.main(cli_args(saved, output))
    assert error.value.code == 1
    assert not output.exists()
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after


def test_atomic_publish_does_not_replace_racing_existing_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        tp.publish(path, "new")
    assert path.read_text() == "original"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["report.json"]


def test_read_ids_does_not_promote_mixed_signed_unsigned_columns_to_float(tmp_path):
    path = tmp_path / "mixed_ids.root"
    with uproot.recreate(path) as root:
        root.mktree("events", {"run": np.array([1, 2], dtype=np.int32),
                               "subrun": np.array([3, 4], dtype=np.int32),
                               "event": np.array([2**63 + 1, 2**63 + 2], dtype=np.uint64)})
    values = tp.read_ids(path, "events")
    assert values.dtype == np.uint64
    assert int(values[0, 2]) == 2**63 + 1
    assert int(values[1, 2]) == 2**63 + 2


@pytest.mark.parametrize("bad", [np.array([[1., 2., 3.]]), np.array([[1, -2, 3]])])
def test_full_identity_comparison_rejects_noninteger_or_negative_ids(bad):
    with pytest.raises(ValueError, match="nonnegative integers"):
        tp.equal_ids(IDS[:1], bad, "bad")


def test_failed_cli_does_not_replace_dangling_symlink(saved, tmp_path):
    output = tmp_path / "throughput.json"
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(SystemExit) as error:
        tp.main(cli_args(saved, output))
    assert error.value.code == 1
    assert output.is_symlink() and not output.exists()
    assert not (tmp_path / "missing").exists()


def test_report_does_not_reassemble_frames_or_invoke_gpu(saved, monkeypatch):
    import mixing.direct_frames
    def forbidden(*args, **kwargs):
        raise AssertionError("post-run analysis must not load full frames")
    monkeypatch.setattr(mixing.direct_frames, "load_manifest", forbidden)
    assert tp.report(*saved)["events"] == 3


def test_public_hardware_labels_come_only_from_metadata(saved):
    (saved[0] / "hardware.json").write_text(json.dumps({"label": "recorded accelerator; 6 host CPUs; 24 GB RAM"}))
    (saved[0] / "stock_hardware.json").write_text(json.dumps({"label": "recorded CPU; 3 cores; 12 GB RAM"}))
    result = tp.report(*saved)
    assert result["hardware"]["label"] == "recorded accelerator; 6 host CPUs; 24 GB RAM"
    assert result["stock"]["endpoint"] == "recorded CPU; 3 cores; 12 GB RAM"
    assert len(result["hardware"]["metadata"]["sha256"]) == 64


@pytest.mark.parametrize("label", [None, "", "  ", 2, "device\nhost"])
def test_invalid_public_hardware_label_rejected(saved, label):
    (saved[0] / "hardware.json").write_text(json.dumps({"label": label}))
    with pytest.raises(ValueError, match="public label"):
        tp.report(*saved)
