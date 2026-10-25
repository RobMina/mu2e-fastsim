"""Convergence statistics are paired by saved row and clustered by source event."""
import numpy as np
import pytest
from beamfastsim.convergence import ARMS, cluster_mean, summarize


def test_cluster_error_tracks_events_not_lanes():
    ids = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 2], [1, 0, 2]])
    result = cluster_mean([0, 0, 1, 1], ids)
    assert result == dict(mean=0.5, stderr=0.5, events=2)
    assert cluster_mean([], np.empty((0, 3))) == dict(mean=None, stderr=None, events=0)


def test_summary_counts_cap_and_paired_difference():
    g = dict(run=np.ones(4), subrun=np.zeros(4), evt=np.arange(4), foil=np.array([0, 1, -1, -1]),
             time=np.zeros(4))
    f = dict(fate=np.array([1, 6, 1, 9]), foil=np.array([0, -1, 1, -1]), t=np.array([2., 3., 4., 5.]))
    base = dict(fate=np.array([1, 1, 6, 9]), foil=np.array([0, 1, -1, -1]), t=np.ones(4))
    result = summarize(g, f, base)
    assert result['fates']['9'] == 1
    assert result['minus_nominal']['mean'] == 0
    assert result['minus_nominal']['stderr'] == pytest.approx(np.sqrt(1 / 6))
    assert result['flight_time_delta_ns_given_both']['mean'] == 1
    assert result['foil_profile'] == [1, 1]
    assert ARMS['nominal'][0] * ARMS['nominal'][1] == ARMS['halfstep'][0] * ARMS['halfstep'][1]
