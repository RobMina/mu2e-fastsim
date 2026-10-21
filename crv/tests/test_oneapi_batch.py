"""On oneAPI the batched waveform synthesis uses at least ONEAPI_MIN_BATCH_ROWS rows per call; elsewhere as asked."""


def test_batch_rows(monkeypatch):
    from crvfastsim.reco import cli, coinc_jax
    monkeypatch.setattr(coinc_jax, "is_oneapi", lambda: False)
    assert cli._batch_rows(2000) == 2000
    monkeypatch.setattr(coinc_jax, "is_oneapi", lambda: True)
    assert cli._batch_rows(2000) == cli.ONEAPI_MIN_BATCH_ROWS
    assert cli._batch_rows(10 * cli.ONEAPI_MIN_BATCH_ROWS) == 10 * cli.ONEAPI_MIN_BATCH_ROWS
