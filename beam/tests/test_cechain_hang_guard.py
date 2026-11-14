"""The CE-chain transport loop compiles without command buffers when XLA_FLAGS has device loops and concurrent graph
regions both on on a GPU (the combination hangs it); otherwise it gets no compiler options."""
import warnings

import pytest

from beamfastsim import cechain_jax as CJ

LOOPS = "--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE"
CONC = "--xla_gpu_graph_enable_concurrent_region=true"


@pytest.mark.parametrize("flags, backend, guarded", [
    (f"{LOOPS} {CONC}", "gpu", True),
    ("--xla_gpu_enable_command_buffer=FUSION,+WHILE --xla_gpu_graph_enable_concurrent_region=1", "gpu", True),
    (LOOPS, "gpu", False), (CONC, "gpu", False), ("", "gpu", False),
    (f"{LOOPS} {CONC}", "cpu", False)])
def test_run_options(monkeypatch, flags, backend, guarded):
    monkeypatch.setenv("XLA_FLAGS", flags)
    monkeypatch.setattr(CJ.jax, "default_backend", lambda: backend)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        opts = CJ._run_options()
    assert opts == ({"xla_gpu_enable_command_buffer": ""} if guarded else None)
    assert any("hangs with both" in str(x.message) for x in w) is guarded
