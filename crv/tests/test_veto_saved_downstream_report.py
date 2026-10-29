"""Exact saved photon-root lineage, independent of time and pixel position."""
import pandas as pd
import pytest

from crvfastsim.veto.saved_downstream_report import extreme_lineage


def test_lineage_uses_identity_not_time_or_nearby_channel():
    q = pd.DataFrame({"event": [1, 1, 2], "chan": [4, 5, 4], "photon_id": [7, 8, 7],
                      "endpoint_roundoff_step": [True, False, False]})
    attempts = pd.DataFrame({"event": [1, 1, 2, 1, 1], "chan": [4, 5, 4, 4, 4], "root": [7, 7, 7, 7, -1],
                             "primary": [True, True, True, False, True], "dark": [False, False, False, False, True],
                             "time": [1., 1., 1., 1., 1.]})
    result = extreme_lineage(q, attempts)
    assert len(result) == 1 and result.root.iloc[0] == 7 and result.event.iloc[0] == 1
    attempts.loc[0, "time"] = 9999.
    assert len(extreme_lineage(q, attempts)) == 1


def test_duplicate_root_fails_and_empty_extremes_supported():
    q = pd.DataFrame({"event": [1], "chan": [4], "photon_id": [7], "endpoint_roundoff_step": [True]})
    a = pd.DataFrame({"event": [1], "chan": [4], "root": [7], "primary": [True], "dark": [False]})
    with pytest.raises(ValueError, match="duplicate"):
        extreme_lineage(pd.concat([q, q]), a)
    assert extreme_lineage(q.assign(endpoint_roundoff_step=False), a).empty


def test_extreme_channel_stages_do_not_infer_truth_or_signal_maximum():
    from crvfastsim.veto.saved_downstream_report import extreme_channels
    q = pd.DataFrame({"event": [1], "chan": [4], "endpoint_roundoff_step": [True]})
    clusters = pd.DataFrame({"event": [1, 1], "chan": [4, 5], "prekeep": [False, True], "charges": [4, 100]})
    digis = pd.DataFrame({"event": [1, 1], "chan": [5, 5], "starttdc": [10, 10], "sample": [0, 1]})
    pulses = pd.DataFrame({"event": [1], "chan": [5]})
    report = extreme_channels(q, clusters, digis, pulses)
    assert report["digis"] == report["pulses"] == 0
    assert len(report["clusters"]) == 1 and report["clusters"][0]["prekeep"] is False
