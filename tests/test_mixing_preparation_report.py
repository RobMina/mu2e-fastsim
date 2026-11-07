"""Synthetic native preparation reports validate timing, full census and exact ROOT evidence."""
import json
import numpy as np
import pytest
import uproot

from container.offline.compare_root import compare
from mixing.preparation_report import STEP_DIRS, main, report, timing


def log(real="2"):
    return f"""TimeTracker printout (sec)
 Min Avg Max Median RMS nEvts
==================================
Full event 0.1 0.5 0.9 0.5 0.1 3
source:RootInput(read) 0 0.4 0.8 0.4 0 3
p1:MuStopPileupMixer:MixBackgroundFrames 0 0.1 0.2 0.1 0 2
p2:MuStopPileupMixer:MixBackgroundFrames 0 0.2 0.4 0.2 0 1
p1:MuStopPileupFilter:MixingFilter 0 0.01 0.02 0.01 0 3
p1:CrvPhotons:CrvPhotons 0 0.03 0.06 0.03 0 3
p1:CrvSiPMCharges:CrvSiPMCharges 0 0.02 0.04 0.02 0 3
end_path:primarySteps:StrawGasStepTableDumper 0 0.01 0.02 0.01 0 3
p1:EWMProducer:EventWindowMarkerProducer 0 0.01 0.02 0.01 0 3
end_path:Output:RootOutput 0 0.01 0.02 0.01 0 3
end_path:Output:RootOutput(write) 0 0.01 0.02 0.01 0 3
p1:Zero:Producer 0 0 0 0 0 3
p1:Never:Producer 0 0 0 0 0 0
==================================
TrigReport Events total = 3 passed = 3 failed = 0
TimeReport CPU = 1 Real = {real}
Art has completed and will exit with status 0.
"""


def save(path, ids=None, change_steps=False, changed_payload=False, omit=None):
    if ids is None:
        ids = np.array([[1, 2, 7], [1, 3, 7], [2, 2, 7]], dtype=np.uint32)
    with uproot.recreate(path) as root:
        root.mktree("crvdump/events", {name: np.array([99]) for name in ("run", "subrun", "event")})
        root.mktree("crvdump/events", dict(zip(("run", "subrun", "event"), ids.T)))
        for index, name in enumerate(STEP_DIRS):
            if name != omit:
                root.mktree(f"{name}/sgsteps", {"step": np.arange(index + (change_steps and index == 1))})
        for name in ("digis", "pulses", "clusters"):
            root.mktree(f"crvdump/{name}", {"count": np.array([2 + changed_payload])})
        root["diagnostic/hist"] = (np.array([1., 2.]), np.array([0., 1., 2.]))


@pytest.fixture
def inputs(tmp_path):
    reference, candidate = tmp_path / "saved", tmp_path / "new"
    for directory in (reference, candidate):
        directory.mkdir()
        save(directory / "response.root")
        (directory / "art.log").write_text(log())
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(compare(reference / "response.root", candidate / "response.root")))
    return reference, candidate, comparison


def test_complete_highest_cycle_report_and_descriptive_boundary(inputs):
    reference, candidate, comparison = inputs
    (candidate / "art.log").write_text(log("1.5"))
    (reference / "metadata.json").write_text('{"node": "recorded-node"}')
    result = report(*inputs, expected_events=3)
    assert result["equal"] is True
    assert result["reference"]["node"] == "recorded-node"
    assert "node" not in result["candidate"]
    assert result["candidate"]["finished_events"] == 3
    assert result["candidate"]["total_steps"] == 10
    assert result["candidate"]["steps"]["primarySteps"] == 0
    assert "not a controlled speedup" in result["interpretation"]
    assert result["candidate_minus_reference_s"]["real_s"] == -0.5
    assert result["candidate"]["timing"]["real_events_per_s"] == 2
    json.dumps(result, allow_nan=False)


def test_weighted_module_sum_excludes_full_event_source_and_write_subtimer():
    result = timing(log(), 3)
    assert result["module_total_s"] == pytest.approx(0.67)
    assert result["full_event_total_s"] == 1.5
    assert result["by_category_s"] == pytest.approx({"mix": .4, "filters": .03, "CrvPhotons": .09,
                                                   "other CRV": .06, "step export": .03, "misc": .06})
    assert result["by_module"]["MuStopPileupMixer"]["invocations"] == 3
    assert result["by_module"]["MuStopPileupMixer"]["total_s"] == pytest.approx(.4)
    assert result["by_module"]["Zero"]["total_s"] == result["by_module"]["Never"]["total_s"] == 0
    assert {x["label"] for x in result["excluded_subtimers"]} == {"RootInput(read)", "Output(write)"}


@pytest.mark.parametrize("old,new", [
    ("status 0.", "status 1."), ("Art has completed and will exit with status 0.", ""),
    ("CPU = 1", "CPU = 0"), ("Real = 2", "Real = 0"),
    ("Real = 2", "Real = 1e-320"), ("0.1 0.5 0.9", "0.1 0 0.9"),
    ("Full event", "Missing event"), ("Min Avg Max Median RMS nEvts", "Min Avg Max nEvts"),
    ("0 0.1 0.2 0.1 0 2", "0 0.1 0.2 0.1 0 4"),
    ("0 0.1 0.2 0.1 0 2", "0 1e308 1e308 1e308 0 2"),
])
def test_bad_logs_fail(old, new):
    with pytest.raises(ValueError):
        timing(log().replace(old, new), 3)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e999", "-0.1"])
@pytest.mark.parametrize("old", ["CPU = 1", "Real = 2", "0 0.1 0.2", "0.1 0.5 0.9"])
def test_nonfinite_and_negative_statistics_never_hidden(bad, old):
    new = old.rsplit(" ", 1)[0] + " " + bad
    with pytest.raises(ValueError):
        timing(log().replace(old, new), 3)


def test_concatenated_and_truncated_logs_fail():
    for text in (log() + log(), log().rsplit("==================================", 1)[0]):
        with pytest.raises(ValueError):
            timing(text, 3)


@pytest.mark.parametrize("expected", [0, -1, 2, 4, 2741])
def test_expected_finished_event_count_required(inputs, expected):
    with pytest.raises(ValueError):
        report(*inputs, expected_events=expected)


def test_log_count_must_match_root_census(inputs):
    with pytest.raises(ValueError, match="log event count"):
        timing(log(), 4)


@pytest.mark.parametrize("ids", [
    [[1, 2, 7], [1, 2, 7], [2, 2, 7]],
    [[1, 2, 7], [1, 3, 7]],
    [[1, 2, 7], [1, 3, 7], [2, 9, 7]],
    [[2, 2, 7], [1, 3, 7], [1, 2, 7]],
])
def test_duplicate_missing_changed_subrun_and_reordered_ids_fail(inputs, ids):
    save(inputs[1] / "response.root", ids=np.asarray(ids, dtype=np.uint32))
    with pytest.raises(ValueError, match="census|identities"):
        report(*inputs, expected_events=3)


@pytest.mark.parametrize("kwargs", [{"change_steps": True}, *({"omit": name} for name in STEP_DIRS)])
def test_every_step_tree_required_and_count_checked(inputs, kwargs):
    save(inputs[1] / "response.root", **kwargs)
    with pytest.raises((ValueError, KeyError)):
        report(*inputs, expected_events=3)


@pytest.mark.parametrize("mutation", ["flag", "bare", "missing", "entries", "branch", "class", "object_flag",
                                       "path", "nan", "extra", "duplicate"])
def test_comparison_report_must_match_complete_existing_comparator(inputs, mutation):
    data = json.loads(inputs[2].read_text())
    tree = next(x for x in data["objects"] if x["key"] == "crvdump/events")
    if mutation == "flag":
        data["equal"] = False
    elif mutation == "bare":
        data = {"equal": True, "reference": data["reference"], "candidate": data["candidate"]}
    elif mutation == "missing":
        data["objects"] = [tree]
    elif mutation == "entries":
        tree["reference_entries"] = 2
    elif mutation == "branch":
        tree["different_branches"] = ["run"]
    elif mutation == "class":
        tree["class"] = "TH1D"
    elif mutation == "object_flag":
        tree["equal"] = False
    elif mutation == "path":
        data["candidate"] = str(inputs[0] / "response.root")
    elif mutation == "nan":
        tree["reference_entries"] = float("nan")
    elif mutation == "extra":
        data["objects"].append({"key": "unknown", "equal": True})
    elif mutation == "duplicate":
        data["objects"].append(tree)
    inputs[2].write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report(*inputs, expected_events=3)


def test_stale_equal_comparison_cannot_hide_changed_payload(inputs):
    save(inputs[1] / "response.root", changed_payload=True)
    with pytest.raises(ValueError, match="exact ROOT comparison"):
        report(*inputs, expected_events=3)


def test_cli_output_and_failure_without_writing_report(inputs, tmp_path):
    output = tmp_path / "report.json"
    args = ["--reference", str(inputs[0]), "--candidate", str(inputs[1]),
            "--comparison", str(inputs[2]), "--expected-events", "3", "--out", str(output)]
    assert main(args) == 0
    assert json.loads(output.read_text())["equal"] is True
    output.unlink()
    (inputs[1] / "art.log").write_text(log().replace("status 0.", "status 2."))
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 1
    assert not output.exists()


@pytest.mark.parametrize("data", [[], None, {"equal": True}, {"equal": 1}])
def test_non_report_json_is_rejected(inputs, data):
    inputs[2].write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report(*inputs, expected_events=3)


def test_equal_flag_requires_boolean_in_every_object(inputs):
    data = json.loads(inputs[2].read_text())
    data["objects"][0]["equal"] = 1
    inputs[2].write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report(*inputs, expected_events=3)


def test_equal_root_files_do_not_excuse_duplicate_full_id_census(inputs):
    for directory in inputs[:2]:
        save(directory / "response.root", ids=np.array([[1, 2, 7]] * 3, dtype=np.uint32))
    inputs[2].write_text(json.dumps(compare(inputs[0] / "response.root", inputs[1] / "response.root")))
    with pytest.raises(ValueError, match="duplicate event identities"):
        report(*inputs, expected_events=3)


def test_compensating_step_count_changes_fail_even_when_total_is_unchanged(inputs):
    with uproot.update(inputs[1] / "response.root") as root:
        root.mktree("primarySteps/sgsteps", {"step": np.array([0], dtype=np.int64)})
        root.mktree("stopSteps/sgsteps", {"step": np.array([], dtype=np.int64)})
    with pytest.raises(ValueError, match="step-tree counts"):
        report(*inputs, expected_events=3)
