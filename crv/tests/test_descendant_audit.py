"""Synthetic crossing-support census and clustered paired-error checks."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from valhist import descendant_audit as A


def fixture():
    geom = SimpleNamespace(nbars=3, sector=np.zeros(3, dtype=int), module=np.zeros(3, dtype=int),
                           layer=np.arange(3), center=np.array([[0., 2., 0.], [0., 1., 0.], [0., 0., 0.]]),
                           local_to_world=np.tile([1, 0, 2], (3, 1)))
    parts = pd.DataFrame({"run": [1400] * 6, "subrun": [1] * 6, "event": [1, 1, 1, 1, 2, 2],
                          "key": [1, 2, 3, 4, 1, 2], "parent": [-1, 1, -1, 3, -1, 1],
                          "pdg": [13, 11, -13, 11, 13, 11], "isprim": [1, 0, 1, 0, 1, 0]})
    steps = pd.DataFrame({"run": [1400] * 5, "subrun": [1] * 5, "event": [1, 1, 1, 1, 2],
                          "sp": [1, 2, 2, 4, 2], "bar": [0, 0, 1, 2, 2], "edep": [10., 1., 2., 3., 4.]})
    return steps, parts, geom


def test_support_partition_uses_own_ancestor_not_event_nearest():
    d, info = A.annotate(*fixture())
    assert list(d.support) == list(A.KINDS)
    assert list(d.lay) == [0, 1, 2, 2]
    assert d.edep.sum() == 10
    assert info["collapsed_crossings"] == 0
    assert info["descendant_steps"] == 4


def test_multiple_muons_sharing_a_bar_are_counted_separately():
    steps, parts, geom = fixture()
    second = steps.iloc[:1].assign(sp=3)
    d, info = A.annotate(pd.concat([steps, second], ignore_index=True), parts, geom)
    assert d.loc[d.sp.eq(4), "support"].item() == "own_other_bar"
    assert info["primary_crossings_by_ancestor"] == 2
    assert info["event_bar_crossings"] == 1
    assert info["collapsed_crossings"] == 1


def test_event_identity_includes_run_and_subrun():
    steps, parts, geom = fixture()
    steps.loc[steps.event.eq(2), "run"] = 1401
    parts.loc[parts.event.eq(2), "run"] = 1401
    steps["event"] = 1
    parts["event"] = 1
    d, _ = A.annotate(steps, parts, geom)
    assert d.loc[d.run.eq(1401), "support"].item() == "no_event_crossing"
    c = d.assign(edep=2 * d.edep, w=2 * d.w)
    result = A.summarize(A.paired_views(d, c))["sector"]
    assert all(row["e_ratio"] == 2 for row in result)
    assert all(row["e_ratio_se"] == pytest.approx(0, abs=1e-14) for row in result)


def test_paired_moments_and_missing_support_are_not_dropped():
    r, _ = A.annotate(*fixture())
    c = r.iloc[:2].copy()
    c["w"] = [3., 4.]
    rows = {row["support"]: row for row in A.summarize(A.paired_views(r, c))["sector"]}
    assert rows["own_bar"]["e_delta"] == 2
    assert rows["own_bar"]["e_delta_se"] == 2
    assert rows["no_event_crossing"]["e_ratio"] == 0
    assert rows["no_event_crossing"]["e_delta"] == -4
    assert rows["no_event_crossing"]["e_delta_se"] == 4


def test_same_event_steps_are_clustered_before_squaring():
    r, _ = A.annotate(*fixture())
    r = pd.concat([r.iloc[:1], r.iloc[:1]], ignore_index=True)
    result = A.paired_views(r, r.iloc[:0])["sector"].iloc[0]
    assert result.n_ref == 2 and result.n2_ref == 4
    assert result.e_ref == 2 and result.e2_ref == 4


@pytest.mark.parametrize("defect", ["duplicate", "parent", "particle", "fractional", "cycle", "bar", "energy"])
def test_invalid_inputs_fail_closed(defect):
    steps, parts, geom = fixture()
    if defect == "duplicate":
        parts = pd.concat([parts, parts.iloc[:1]], ignore_index=True)
    elif defect == "parent":
        parts.loc[1, "parent"] = 99
    elif defect == "particle":
        steps.loc[1, "sp"] = 99
    elif defect == "fractional":
        steps["sp"] = steps.sp.astype(float) + 0.1
    elif defect == "cycle":
        parts.loc[0, "parent"] = 2
    elif defect == "bar":
        steps.loc[0, "bar"] = -1
    else:
        steps.loc[0, "edep"] = np.nan
    with pytest.raises(ValueError):
        A.annotate(steps, parts, geom)


def test_no_descendants_produces_empty_census():
    steps, parts, geom = fixture()
    d, info = A.annotate(steps.iloc[:1], parts, geom)
    assert d.empty and info["descendant_steps"] == 0
    assert all(frame.empty for frame in A.paired_views(d, d).values())


def test_placement_trace_distinguishes_other_own_bar_and_other_muon():
    from valhist.descendant_placement import features
    _, _, geom = fixture()
    fx = pd.DataFrame({"run": [1400] * 3, "subrun": [1] * 3, "event": [5] * 3,
                       "sp": [1, 1, 2], "bar": [0, 1, 2]})
    trace = {"xi": np.zeros(4, dtype=int), "target_bar": np.array([0, 1, 2, -1]),
             "kept": np.array([True, True, True, False]), "source_samebar": np.array([True, False, False, False]),
             "source_edep": np.ones(4)}
    d = features(fx, trace, geom)
    assert list(d.target_kind) == ["anchor", "another_own_bar", "otherbar", "lost"]
    assert d.target_group.iloc[-1] == "lost" and d.layer.iloc[-1] == -1


def test_run_refuses_overwrite_before_read(tmp_path):
    out = tmp_path / "existing.json"
    out.write_text("retained")
    with pytest.raises(FileExistsError):
        A.run(["missing.root"], None, out)
    assert out.read_text() == "retained"
