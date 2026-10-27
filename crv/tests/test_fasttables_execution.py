"""Opt-in execution-trace/fused CLI admission, forwarding and fail-before-publication."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

from crvfastsim import response_receipt, runconfig
from crvfastsim.response import execution_trace
from valhist import fasttables as FT


@pytest.mark.parametrize("args", [
    ["--trace-response"],
    ["--trace-response", "--run-config", "run1a_v01", "--write-receipt"],
    ["--fused-response"],
    ["--fused-response", "--trace-response", "--run-config", "run2_v10"],
    ["--fused-response", "--jax", "--run-config", "run2_v10"],
    ["--fused-response", "--jax", "--trace-response", "--run-config", "run2_v10", "--window", "onspill"],
])
def test_bad_options_refused_before_config(monkeypatch, tmp_path, args):
    monkeypatch.setattr(FT, "configure", lambda **kw: pytest.fail("configured before rejection"))
    monkeypatch.setattr(sys, "argv", ["fasttables", "chain", "input", str(tmp_path / "out.root"), *args])
    with pytest.raises(SystemExit) as caught:
        FT.main()
    assert caught.value.code == 2
    assert not (tmp_path / "out.root").exists()


def test_trace_refuses_reco_and_existing_output(monkeypatch, tmp_path):
    monkeypatch.setattr(FT, "configure", lambda **kw: pytest.fail("configured before rejection"))
    output = tmp_path / "out.root"
    for stage in ("reco", "chain"):
        if stage == "chain":
            output.write_bytes(b"original")
        monkeypatch.setattr(sys, "argv", ["fasttables", stage, "input", str(output),
                                         "--run-config", "run2_v10", "--trace-response"])
        with pytest.raises(SystemExit) as caught:
            FT.main()
        assert caught.value.code == 2
    assert output.read_bytes() == b"original"


@pytest.mark.parametrize("backend", ["numpy", "jax", "fused", "disabled"])
@pytest.mark.parametrize("fail", [False, True])
def test_trace_brackets_digis_and_validates_before_write(monkeypatch, tmp_path, backend, fail):
    jax = backend in ("jax", "fused")
    cfg = replace(runconfig.get("run2_v10").chain_config(jax=jax), window=(0., 100000.))
    metadata = {}
    active = []
    seen = []
    monkeypatch.setattr(FT, "configure", lambda **kw: (None, None, None, cfg, metadata))
    from crvfastsim.response import photons, sipm
    monkeypatch.setattr(photons, "load_tables", lambda response: {})
    monkeypatch.setattr(sipm, "photon_map", lambda: None)
    data = pd.DataFrame({"run": [1400], "subrun": [101], "event": [7]})
    monkeypatch.setattr(FT.tables, "read", lambda *a: data)
    evidence = {"test_only": True, "backend": backend}

    class Trace:
        def require_backend(self):
            assert not active
            seen.append("validate")
            if fail:
                raise ValueError("insufficient backend evidence")
            return evidence

    @contextmanager
    def observe(requested):
        assert requested == backend
        active.append(True)
        seen.append("enter")
        try:
            yield Trace()
        finally:
            active.pop()
            seen.append("exit")

    monkeypatch.setattr(execution_trace, "observe", observe)

    def digis(*args):
        actual = args[4]
        assert actual.jax == jax and actual.fused_response == (backend == "fused")
        assert bool(active) == (backend != "disabled")
        seen.append("digis")
        return {}

    monkeypatch.setattr(FT, "fast_digis", digis)
    monkeypatch.setattr(FT, "write", lambda *a: seen.append("write"))
    output = tmp_path / "out.root"
    args = ["fasttables", "digis", "input", str(output), "--run-config", "run2_v10"]
    if backend != "disabled":
        args.append("--trace-response")
    if jax:
        args.append("--jax")
    if backend == "fused":
        args.append("--fused-response")
    monkeypatch.setattr(sys, "argv", args)
    if fail and backend != "disabled":
        with pytest.raises(ValueError, match="insufficient backend evidence"):
            FT.main()
        assert seen == ["enter", "digis", "exit", "validate"]
        assert not Path(str(output) + ".metadata.json").exists()
    else:
        FT.main()
        saved = json.loads(Path(str(output) + ".metadata.json").read_text())
        if backend == "disabled":
            assert seen == ["digis", "write"]
            assert "response_execution" not in saved
        else:
            assert seen == ["enter", "digis", "exit", "validate", "write"]
            assert saved["response_execution"] == evidence
        assert saved["chain_config"]["fused_response"] == (backend == "fused")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_trace_io_scope_restores_uproot_defaults(enabled, fail):
    before = FT.uproot.open.defaults.copy()
    try:
        with FT._trace_local_io(enabled):
            if enabled:
                assert FT.uproot.open.defaults["handler"] is FT.uproot.source.file.MemmapSource
                assert not FT.uproot.open.defaults["use_threads"]
            else:
                assert FT.uproot.open.defaults == before
            if fail:
                raise RuntimeError("test failure")
    except RuntimeError:
        assert fail
    assert FT.uproot.open.defaults == before


def test_local_root_reads_leave_trace_eligible(tmp_path):
    import subprocess
    import os
    script = """
import sys, numpy as np, uproot
from valhist import fasttables as F
from crvfastsim.response import execution_trace
path = sys.argv[1]
with uproot.recreate(path) as f:
    f.mktree('events', {'event': np.arange(10)})
with F._trace_local_io(True):
    with uproot.open(path) as f:
        assert len(f['events'].arrays(library='np')['event']) == 10
    with execution_trace.observe('numpy') as trace:
        pass
assert trace.summary()['status'] == 'insufficient_evidence'
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "test.root")],
                            env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_receipt_rejects_execution_marker_before_admission():
    with pytest.raises(ValueError, match="not normal receipt"):
        response_receipt._metadata({"schema": response_receipt.METADATA_SCHEMA,
                                    "stage": "chain", "response_execution": {}})
