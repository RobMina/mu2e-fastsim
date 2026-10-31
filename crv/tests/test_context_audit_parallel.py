"""Parallel leaves retain the serial digest and fail-closed mutation contract."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.analyses.context_audit import content_fingerprint, freeze_arrays


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_parallel_digest_matches_serial_for_complete_nested_payload(workers):
    shared = np.arange(400000, dtype=np.float64)
    data = {"shared": shared, "alias": shared, "matrix": shared.reshape(1000, 400).T,
            "table": pd.DataFrame({"a": shared, "b": shared + 1}),
            "object": np.array(["EX", None, "M"], dtype=object),
            "config": SimpleNamespace(items=(True, None, 4, "field")),
            "small": np.arange(5), "empty": np.empty((0, 2))}
    expected = content_fingerprint(data, workers=1)
    freeze_arrays(data)
    assert content_fingerprint(data, workers=workers) == expected
    data["table"]["a"] = np.zeros(len(shared))
    assert content_fingerprint(data, workers=workers) != expected
    assert content_fingerprint(data, workers=workers) == content_fingerprint(data, workers=1)


def test_parallel_leaves_rehash_writable_aliases_and_root_array():
    data = np.arange(400000, dtype=np.float64)
    alias = data.view()
    expected = content_fingerprint(data, workers=1)
    assert content_fingerprint(data, workers=4) == expected
    freeze_arrays(data)
    alias[-1] = -1
    assert content_fingerprint(data, workers=4) != expected
    assert content_fingerprint(data, workers=4) == content_fingerprint(data, workers=1)


def test_parallel_fails_closed_and_environment_is_explicit(monkeypatch):
    monkeypatch.setenv("CRVFS_CONTEXT_HASH_WORKERS", "4")
    assert content_fingerprint(np.arange(400000)) == content_fingerprint(np.arange(400000), workers=1)
    cyclic = [np.arange(400000)]
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cycle"):
        content_fingerprint(cyclic)
    with pytest.raises(TypeError, match="unsupported"):
        content_fingerprint([np.arange(400000), object()])
    for invalid in (0, 33, -1, True, 1.5):
        with pytest.raises(ValueError, match="workers"):
            content_fingerprint(None, workers=invalid)
    monkeypatch.setenv("CRVFS_CONTEXT_HASH_WORKERS", "bad")
    with pytest.raises(ValueError):
        content_fingerprint(None)


def test_parallel_device_identity_without_host_copy(monkeypatch):
    jax = pytest.importorskip("jax")
    array = jax.device_put(np.arange(3), jax.devices("cpu")[0])
    value = {"host": np.arange(400000), "device": array}
    expected = content_fingerprint(value, workers=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("device-to-host copy requested")

    monkeypatch.setattr(type(array), "__array__", forbidden)
    monkeypatch.setattr(jax, "device_get", forbidden)
    assert content_fingerprint(value, workers=4) == expected
