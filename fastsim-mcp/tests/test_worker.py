"""Detached point workers: a point survives the client and the server, and a worker killed mid-stream is respawned.

The workers here are real processes (python -m fastsim_tools.worker in their own session) driving the file-based
stand-in kit (fakekit.py), so a client or worker drop is a real process going away.
"""
import json
import os
import signal
import time

import pytest

import fastsim_tools
from fastsim_tools import backend, worker

pytest.importorskip("mu2efastsim.endpoints.geometry")

FACTORY = "fakekit:make"


def wait_for(pred, timeout=30.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("condition not reached in time")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTSIM_MCP_WORKER_POLL_S", "0.1")
    monkeypatch.setenv("FASTSIM_MCP_SUBMIT_WAIT_S", "20")
    monkeypatch.setenv("FAKEKIT_STREAM_STEPS", "2")
    monkeypatch.setenv("FAKEKIT_STREAM_SLEEP_S", "0")
    monkeypatch.setenv("FAKEKIT_RUN_S", "0.5")
    monkeypatch.setenv("FAKEKIT_WAIT_S", "60")
    monkeypatch.delenv("FASTSIM_MCP_KIT_FACTORY", raising=False)
    yield tmp_path
    be = backend.Backend(state_dir=tmp_path, kit_factory=FACTORY)
    for name in be.list_points():                    # no stray workers (never this process: in-process runs)
        pid = be.heartbeat(name).get("pid")
        if pid and pid != os.getpid() and be.worker_alive(name):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    backend.set_backend(None)


def session(root):
    """A fresh 'server process': its own Backend, nothing shared in memory with earlier sessions."""
    be = backend.Backend(state_dir=root, kit_factory=FACTORY)
    backend.set_backend(be)
    return be


def chunks(root, handle):
    return json.loads((root / "fake" / f"{handle}.json").read_text())["chunks"]


def test_backend_detached_by_default_only_for_a_built_kit(tmp_path):
    assert backend.Backend(state_dir=tmp_path).detached
    assert not backend.Backend(kit=object(), state_dir=tmp_path).detached


def test_point_survives_the_client(env):
    be = session(env)
    r = fastsim_tools.run_design_point("d1", electrons=3000)
    assert r.status == "submitted", r.message
    assert all(j["job_id"] for j in r.metadata["jobs"]) and "detached worker" in r.message
    pid = r.metadata["worker"]["pid"]
    assert pid and pid != os.getpid()
    backend.set_backend(None)                       # the client and its server go away
    del be
    wait_for(lambda: backend.Backend(state_dir=env, kit_factory=FACTORY).heartbeat("d1").get("phase") == "done")
    session(env)                                     # a later session reattaches
    st = fastsim_tools.job_status("d1")
    assert st.status == "success", st.message
    assert all(j["inputs_streamed"] and j["fetched"] for j in st.metadata["jobs"])
    for h in ("d1.edep.float64", "d1.edep.float32"):
        assert set(chunks(env, h)) == {pid}          # only the worker streamed, never a server
    out = fastsim_tools.fetch_outputs("d1")
    assert out.status == "success" and {o["dtype"] for o in out.metadata["outputs"]} == {"float64", "float32"}


def test_worker_killed_mid_stream_is_respawned(env, monkeypatch):
    monkeypatch.setenv("FAKEKIT_STREAM_STEPS", "8")
    monkeypatch.setenv("FAKEKIT_STREAM_SLEEP_S", "0.3")
    session(env)
    r = fastsim_tools.run_design_point("d2", electrons=3000, dtypes=["float64"])
    assert r.status == "submitted", r.message
    h = "d2.edep.float64"
    first = r.metadata["worker"]["pid"]
    wait_for(lambda: len(chunks(env, h)) >= 2)                       # mid-stream
    assert not (env / "beam_edep_cpu" / f"{h}.streamed").exists()
    assert os.getsid(first) == first != os.getsid(0)                 # the worker runs in its own session
    os.kill(first, signal.SIGKILL)                                   # the worker dies mid-stream
    wait_for(lambda: not backend.Backend(state_dir=env, kit_factory=FACTORY).worker_alive("d2"))
    session(env)
    st = fastsim_tools.job_status("d2")
    assert st.metadata["worker"]["restarted"] and "Worker restarted" in st.message
    wait_for(lambda: backend.Backend(state_dir=env, kit_factory=FACTORY).heartbeat("d2").get("phase") == "done")
    done = fastsim_tools.job_status("d2")
    assert done.status == "success" and done.metadata["jobs"][0]["inputs_streamed"]
    assert done.metadata["worker"]["starts"] == 2
    pids = set(chunks(env, h))
    assert first in pids and len(pids) == 2 and os.getpid() not in pids
    fake = json.loads((env / "fake" / f"{h}.json").read_text())
    assert fake["submits"] == 2                                      # resubmitted idempotently, one job


def test_one_worker_per_point(env):
    be = session(env)
    be.save_point({"name": "d3", "detached": True, "request": {}, "files": [],
                   "jobs": [{"handle": "d3.edep.float64", "dtype": "float64", "task": "beam_edep_cpu",
                             "params": {"task": "beam_edep_cpu", "seed": 1}}]})
    held = worker.try_lock(env, "d3")
    assert held is not None and be.worker_alive("d3")
    assert worker.run("d3", backend.Backend(state_dir=env, kit_factory=FACTORY, detached=False), once=True) == 0
    assert be.heartbeat("d3") == {}                                  # the second worker did nothing
    held.close()
    assert not be.worker_alive("d3")


def test_worker_round_in_process(env):
    be = backend.Backend(state_dir=env, kit_factory=FACTORY, detached=False)
    be.save_point({"name": "d4", "detached": True, "request": {}, "files": [],
                   "jobs": [{"handle": "d4.edep.float64", "dtype": "float64", "task": "beam_edep_cpu",
                             "params": {"task": "beam_edep_cpu", "seed": 1}}]})
    assert worker.run("d4", be, once=True) == 0
    hb = be.heartbeat("d4")
    assert hb["phase"] == "tracking" and hb["jobs"]["d4.edep.float64"]["job_id"] and hb["starts"] == 1


def test_submit_error_is_reported_not_respawned(env):
    session(env)
    r = fastsim_tools.run_design_point("d5", seed=13, dtypes=["float64"])
    assert r.status == "error" and "could not submit" in r.message and "seed 13" in r.message
    st = fastsim_tools.job_status("d5")
    assert st.status == "error" and not st.metadata["worker"]["restarted"]
    assert st.metadata["jobs"][0]["state"] == "failed"


def test_restart_limit(env):
    be = session(env)
    be.save_point({"name": "d6", "detached": True, "request": {"stage": "edep"}, "files": [],
                   "jobs": [{"handle": "d6.edep.float64", "dtype": "float64", "task": "beam_edep_cpu",
                             "params": {"task": "beam_edep_cpu", "seed": 1}}]})
    worker.write_json(worker.heartbeat_path(env, "d6"), {"phase": "tracking", "starts": backend.MAX_WORKER_STARTS,
                                                          "jobs": {}})
    st = fastsim_tools.job_status("d6")
    assert "without finishing" in st.message and not be.worker_alive("d6")


def test_job_failing_its_input_wait_is_reported(env, monkeypatch):
    """A stream too slow for the job's input wait: the job fails as a real one does and the worker reports it."""
    monkeypatch.setenv("FAKEKIT_STREAM_STEPS", "1000")
    monkeypatch.setenv("FAKEKIT_STREAM_SLEEP_S", "0.05")
    monkeypatch.setenv("FAKEKIT_WAIT_S", "1.5")
    session(env)
    r = fastsim_tools.run_design_point("d7", dtypes=["float64"])
    assert r.status == "submitted"
    wait_for(lambda: backend.Backend(state_dir=env, kit_factory=FACTORY).heartbeat("d7").get("phase") == "done")
    st = fastsim_tools.job_status("d7")
    assert st.status == "error" and st.metadata["jobs"][0]["state"] == "failed"
    assert "no input arrived" in st.metadata["jobs"][0]["message"]
