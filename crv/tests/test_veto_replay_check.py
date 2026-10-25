"""Stock replay must preserve event identities, windows and bunch times before comparison."""
import json

import awkward as ak
import numpy as np
import pytest

from crvfastsim.veto.replay_check import check_events


def inputs():
    expected = ak.Array({"run": [1430, 1430], "subrun": [0, 0], "event": [1, 2],
                         "pbtime": [-209.46, -204.46]})
    actual = ak.with_field(expected, ak.values_astype(expected.pbtime, np.float32), "pbtime")
    for key, vals in {"spill": [1, 1], "evlen": [1700.0, 1700.0], "ndigi": [18, 0],
                      "npulse": [19, 0], "ncluster": [0, 0]}.items():
        actual = ak.with_field(actual, ak.Array(vals), key)
    return expected, actual


def test_event_prefix_and_empty_events():
    expected, actual = inputs()
    assert check_events(expected, actual[:1], 1)["pulses"] == 19
    report = check_events(expected, actual, 2)
    assert report["events"] == 2
    assert report["digis"] == 18
    assert report["coincs"] == 0


@pytest.mark.parametrize("key,values,match", [
    ("run", [1431, 1431], "ID mismatch"), ("event", [2, 1], "ID mismatch"),
    ("pbtime", [-209.46, -204.46], "bunch time"),
    ("spill", [1, 0], "window"), ("evlen", [1700.0, 1675.0], "window")])
def test_bad_event_contract(key, values, match):
    expected, actual = inputs()
    actual = ak.with_field(actual, ak.Array(values), key)
    with pytest.raises(ValueError, match=match):
        check_events(expected, actual, 2)


def test_missing_duplicate_or_excess_events():
    expected, actual = inputs()
    for count in (0, 1, 3):
        with pytest.raises(ValueError, match="count"):
            check_events(expected, actual, count)
    with pytest.raises(ValueError, match="duplicate"):
        check_events(expected[[0, 0]], actual[[0, 0]], 2)


def test_empty_comparison_summary_json_is_standard(tmp_path):
    from mu2efastsim.plots import write_json
    from crvfastsim.veto.compareframes import summary
    empty = ak.zip({"coincs": ak.zip({k: ak.Array([[]]) for k in ("time", "PEs", "sectorType")})}, depth_limit=1)[:0]
    write_json(tmp_path, "summary.json", summary(empty))
    text = (tmp_path / "summary.json").read_text()
    assert "NaN" not in text
    assert json.loads(text)["coincs_per_frame"] is None
