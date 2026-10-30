"""Moment algebra uses event-clustered contributions, not independent adjacent pairs."""
import numpy as np

from valhist.device_tier1b_moments import add, summarize


def test_correlation_and_cluster_error():
    rng = np.random.default_rng(8)
    xy = rng.normal(size=(1000, 2))
    xy[:, 1] += .5 * xy[:, 0]
    x, y = xy.T
    q = np.stack([np.ones(len(x)), x, y, x*x, y*y, x*y], axis=1)
    acc = dict(correlation_first=q.sum(0), correlation_second=q.T @ q,
               tail_thresholds=np.array([0, 1]), tail_counts=np.array([1000, 10]), events=1000)
    got = summarize(acc)
    assert abs(got["correlation"] - np.corrcoef(x, y)[0, 1]) < 1e-13
    assert 0 < got["correlation_se"] < .1
    total = {}
    add(total, acc)
    add(total, acc)
    twice = summarize(total)
    assert abs(twice["correlation"] - got["correlation"]) < 1e-13
    assert abs(twice["correlation_se"] * np.sqrt(2) - got["correlation_se"]) < 1e-13
