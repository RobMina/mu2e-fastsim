"""Observer policy tests and untouched synthetic CPU response executions."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import _thread
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import chain, geometry, runconfig
from crvfastsim.response import execution_trace, sipm
from test_readout_execution import _optical_table


SEED = 20261027


def test_numpy_does_not_import_jax():
    code = """
import builtins, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'jax' or name.startswith('jax.'):
        raise AssertionError('NumPy observer imported JAX')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from crvfastsim.response import execution_trace
with execution_trace.observe('numpy') as trace:
    pass
assert 'jax' not in trace.summary()
assert not any(k == 'jax' or k.startswith('jax.') for k in sys.modules)
json.dumps(trace.summary())
"""
    result = subprocess.run([sys.executable, "-c", code], env=os.environ.copy(),
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["gpu", "JAX", None, [], True])
def test_invalid_backend(backend):
    with pytest.raises(ValueError, match="backend must be"):
        execution_trace.observe(backend)


def test_empty_and_unfinished_are_insufficient():
    trace = execution_trace.observe("numpy")
    assert not trace.summary()["backend_confirmed"]
    with trace:
        with pytest.raises(ValueError, match="has not finished"):
            trace.require_backend()
    evidence = trace.summary()
    assert evidence["status"] == "insufficient_evidence"
    assert evidence["calls"] == {}
    assert evidence["sipm_workload_calls"] == 0
    with pytest.raises(ValueError, match="missing required"):
        trace.require_backend()
    with pytest.raises(ValueError, match="single-use"):
        with trace:
            pass


def test_exception_restores_profiler():
    assert sys.getprofile() is None
    with pytest.raises(RuntimeError, match="sentinel"):
        with execution_trace.observe("numpy") as trace:
            raise RuntimeError("sentinel")
    assert sys.getprofile() is None
    assert not trace.summary()["completed"]
    with pytest.raises(ValueError, match="exception"):
        trace.require_backend()


def test_nested_context_rejected_without_clobber():
    with execution_trace.observe("numpy") as trace:
        profile = sys.getprofile()
        with pytest.raises(ValueError, match="existing profilers"):
            with execution_trace.observe("numpy"):
                pass
        assert sys.getprofile() is profile
    assert sys.getprofile() is None
    assert trace.summary()["completed"]


@pytest.mark.parametrize("scope", ["current", "threads"])
def test_existing_profiler_is_preserved(scope):
    setter = sys.setprofile if scope == "current" else threading.setprofile
    getter = sys.getprofile if scope == "current" else threading.getprofile
    previous = getter()
    def profile(frame, event, arg):
        return None
    setter(profile)
    try:
        with pytest.raises(ValueError, match="existing profilers"):
            with execution_trace.observe("numpy"):
                pass
        assert getter() is profile
    finally:
        setter(previous)


def test_profiler_replacement_not_clobbered():
    def other(frame, event, arg):
        return None
    try:
        with pytest.raises(ValueError, match="profiler changed"):
            with execution_trace.observe("numpy") as trace:
                sys.setprofile(other)
        assert sys.getprofile() is other
        assert trace.summary()["status"] == "unsupported"
    finally:
        sys.setprofile(None)


def test_worker_thread_and_active_threads_rejected():
    ready, finish = threading.Event(), threading.Event()
    errors = []
    def worker():
        try:
            with execution_trace.observe("numpy"):
                pass
        except ValueError as error:
            errors.append(str(error))
        ready.set()
        finish.wait(10)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert ready.wait(10)
        assert errors and "main Python thread" in errors[0]
        with pytest.raises(ValueError, match="active Python worker"):
            with execution_trace.observe("numpy"):
                pass
    finally:
        finish.set()
        thread.join(10)
    assert sys.getprofile() is None


def test_thread_creation_rejected_even_if_caught():
    thread = threading.Thread(target=lambda: None)
    with pytest.raises(ValueError, match="thread creation"):
        with execution_trace.observe("numpy") as trace:
            with pytest.raises(ValueError, match="thread creation"):
                thread.start()
    assert thread.ident is None
    assert sys.getprofile() is None
    assert trace.summary()["status"] == "unsupported"
    with pytest.raises(ValueError, match="thread creation"):
        trace.require_backend()


def test_raw_thread_creation_rejected():
    with pytest.raises(ValueError, match="thread creation"):
        with execution_trace.observe("numpy") as trace:
            _thread.start_new_thread(lambda: None, ())
    assert sys.getprofile() is None
    assert trace.summary()["status"] == "unsupported"


def test_active_raw_thread_rejected():
    ready, finish = threading.Event(), threading.Event()
    done = threading.Event()
    def worker():
        ready.set()
        try:
            finish.wait(10)
        finally:
            done.set()
    _thread.start_new_thread(worker, ())
    try:
        assert ready.wait(10)
        with pytest.raises(ValueError, match="active Python worker"):
            with execution_trace.observe("numpy"):
                pass
    finally:
        finish.set()
        assert done.wait(10)
        import time
        deadline = time.monotonic() + 10
        while _thread._count() and time.monotonic() < deadline:
            time.sleep(.001)
    assert not _thread._count()


def test_only_actual_response_code_is_counted():
    trace = execution_trace.observe("numpy")
    namespace = {"__name__": "crvfastsim.response.sipm"}
    exec("def simulate(): pass", namespace)
    with trace:
        namespace["simulate"]()
        sipm.simulate([], [], [], np.ones((40, 40)), sipm.SiPMConfig(), np.random.default_rng(SEED))
    assert trace.summary()["calls"] == {"sipm.simulate": 1}
    assert trace.summary()["status"] == "insufficient_evidence"
    snapshot = trace.summary()
    snapshot["calls"]["sipm.simulate"] = 99
    assert trace.summary()["calls"]["sipm.simulate"] == 1


@pytest.mark.parametrize("backend", ["jax", "fused"])
def test_real_numpy_call_is_reported_as_fallback(backend):
    jax = pytest.importorskip("jax")
    with jax.default_device(jax.devices("cpu")[0]):
        jax.numpy.zeros(1).block_until_ready()
        with execution_trace.observe(backend) as trace:
            sipm.simulate([], [], [], np.ones((40, 40)), sipm.SiPMConfig(), np.random.default_rng(SEED))
    evidence = trace.summary()
    assert evidence["status"] == "fallback"
    assert evidence["fallback_detected"]
    assert evidence["fallback_observation"] == "observed"
    with pytest.raises(ValueError, match="NumPy sipm.simulate"):
        trace.require_backend()


def test_wrapper_only_callback_is_not_kernel_evidence():
    trace = execution_trace.observe("numpy")
    frame = SimpleNamespace(f_globals=sipm.simulate.__globals__, f_code=sipm.simulate.__code__,
                            f_locals={}, f_back=None)
    with trace:
        trace._profile(frame, "call", None)
        trace._profile(frame, "return", None)
        trace._profile(frame, "c_call", None)
    evidence = trace.summary()
    assert evidence["calls"] == {"sipm.simulate": 1}
    assert "sipm._simulate_chunk" in evidence["missing_calls"]
    with pytest.raises(ValueError, match="missing required"):
        trace.require_backend()


@pytest.fixture(scope="module", params=["numpy", "jax", "fused"])
def real_case(request):
    backend = request.param
    rc = runconfig.get("run1a_v01")
    names = list(rc.sectors)
    geom = geometry.CRVGeometry(
        names, np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64),
        np.zeros((1, 3)), np.array([[9.64, 25.42, 1000.]]), np.array([[0, 1, 2]]))
    cfg = rc.chain_config(jax=backend != "numpy", fused_response=backend == "fused")
    cfg = replace(cfg, response=replace(cfg.response, sectors={
        s: replace(v, table=f"readout_execution_{abs(v.reflector)}") for s, v in rc.sectors.items()}))
    tabs = {f"readout_execution_{r}": _optical_table(r) for r in {abs(v.reflector) for v in rc.sectors.values()}}
    cpu = None
    if backend != "numpy":
        jax = pytest.importorskip("jax")
        cpu = jax.devices("cpu")[0]
        from crvfastsim.response.photons_jax import DeviceTable
        with jax.default_device(cpu):
            tabs = {k: DeviceTable(v) for k, v in tabs.items()}
    steps = pd.DataFrame({
        "subrun": [1], "event": [7], "bar": [0], "edep": [.003], "len": [.05],
        "t0": [500.], "t1": [500.01], "x0": [2.], "x1": [2.05], "y0": [0.], "y1": [0.],
        "z0": [0.], "z1": [0.], "px0": [1000.], "py0": [0.], "pz0": [0.], "p1": [1000.]})
    return backend, cpu, geom, cfg, tabs, steps


def _run(case, absent=False, traced=True):
    backend, cpu, geom, cfg, tabs, steps = case
    context = nullcontext()
    if cpu is not None:
        import jax
        context = jax.default_device(cpu)
    cond = runconfig.nominal_conditions(geom)
    if absent:
        cond = replace(cond, readout_mask=np.zeros(4, bool))
    rng = np.random.default_rng(SEED)
    with context:
        key = None if cpu is None else jax.random.PRNGKey(SEED)
        with execution_trace.observe(backend) if traced else nullcontext() as trace:
            out = chain.charges(steps, None, geom, cond, cfg, rng, key=key, tabs=tabs,
                                pmap=np.ones((40, 40)), pdg=np.array([13]))
    assert sys.getprofile() is None
    return out, deepcopy(rng.bit_generator.state), trace


def test_real_backend_dispatch_and_unchanged_outputs(real_case):
    plain, plain_state, _ = _run(real_case, traced=False)
    out, state, trace = _run(real_case)
    assert len(out["chan"]) > 0
    for key in out:
        np.testing.assert_array_equal(out[key], plain[key])
    assert state == plain_state
    evidence = trace.require_backend()
    assert evidence["status"] == "confirmed"
    assert not evidence["fallback_detected"]
    assert evidence["photon_workload_calls"] > 0
    assert evidence["sipm_workload_calls"] > 0
    if real_case[0] == "numpy":
        assert "jax" not in evidence
    else:
        devices = evidence["jax"]["observed_argument_devices"]
        assert devices and {d["platform"] for d in devices} == {"cpu"}
        assert evidence["jax"]["version"]
    print(f"\nreal execution evidence: {json.dumps(evidence, sort_keys=True)}")


def test_real_no_sipm_workload_is_not_confirmed(real_case):
    out, _, trace = _run(real_case, absent=True)
    assert len(out["chan"]) == 0
    evidence = trace.summary()
    assert evidence["status"] == "insufficient_evidence"
    assert not evidence["fallback_detected"]
    assert evidence["sipm_workload_calls"] == 0
    if real_case[0] != "numpy":
        assert evidence["fallback_observation"] == "insufficient_evidence"
    with pytest.raises(ValueError, match="no nonempty SiPM workload"):
        trace.require_backend()
