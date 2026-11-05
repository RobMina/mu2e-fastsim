"""Benchmark bookkeeping with a synthetic transport, not a timing fixture."""
import json

import numpy as np
import pytest


def test_bench_records_cold_warm_and_checks_repeat_identity(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from beamfastsim import target_bench as B
    from beamfastsim import cechain as C
    from beamfastsim import cechain_jax as CJ
    monkeypatch.setattr(C, "from_offline", lambda *a: None)
    monkeypatch.setattr(C, "_starts", lambda *a: (np.zeros((2, 3)), np.tile([0., 0., 1.], (2, 1)),
                                                np.full(2, 100.), np.zeros(2), np.arange(2)))
    monkeypatch.setattr(CJ, "device_maps", lambda *a: ())
    result = C.Result(np.full(2, C.PLANE), np.zeros((2, 3)), np.tile([0., 0., 1.], (2, 1)),
                      np.full(2, 99.), np.zeros(2), np.ones(2), np.zeros(2), np.zeros(2), np.zeros(2))
    monkeypatch.setattr(CJ, "transport", lambda *a, **k: result)
    out = str(tmp_path / "bench")
    B.main(["--ref", "unused", "--offline-root", "unused", "--maps", "unused", "--out", out])
    report = json.loads((tmp_path / "bench.json").read_text())
    assert report["n"] == 2 and report["repeat_identical"] and not report["physics_validated"]
    assert len(report["warm_transport_s"]) == 2
    assert report["first_with_setup_s"] >= report["first_transport_s"]
    with np.load(out + ".npz") as r:
        assert np.array_equal(r["idx"], [0, 1])
