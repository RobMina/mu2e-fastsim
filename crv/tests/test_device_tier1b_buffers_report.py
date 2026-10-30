"""Measured summaries separate explicitly warmed repeats from compile-inclusive calls."""
from valhist.device_tier1b_buffers_report import summarize


def test_summary_counts_exact_replay_and_timed_subset():
    data = dict(records=[dict(events=50000, shard=22, metadata=dict(exact=True), replay=dict(max_abs=1e-12),
                              warmup=dict(numpy=9., jax=8.), warm=dict(numpy=[3., 4., 5.], jax=[1., 2., 3.]),
                              warm_boundaries=[dict(main_s=1.), dict(main_s=2.), dict(main_s=3.)]),
                         dict(events=50000, shard=24, metadata=dict(exact=True), replay=dict(max_abs=2e-12))],
                library_load_s=10., device_prepare_s=2., device_array_bytes=123, boundary="test")
    result = summarize(data)
    assert result["events"] == 100000 and result["shards"] == 2
    assert result["timed_shards"] == [22]
    assert result["warm_summed_shard_medians"] == dict(numpy=4., jax=2.)
    assert result["warm_observed_ratio"] == 2.
    assert result["component_summed_shard_medians"] == dict(main_s=2.)
    assert result["metadata_exact"] and result["replay_max_abs"] == 2e-12
    assert result["certified"] is False
