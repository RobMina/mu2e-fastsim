"""The compute side of the tools: the fastsim kit (``mu2efastsim.endpoints.kit.FastsimKit``) and the point records.

The kit does the work: it renders a job body, submits it to the configured endpoint (Slurm over ssh by default),
streams the Mu2e-internal inputs into the running job and fetches ``out/`` when it completes. Everything about the
site (hosts, account, work areas, the image) comes from the environment, as for any kit user (site.env.example).
This module keeps one JSON record per design point beside the kit's own state, so a later server process can answer
job_status and fetch_outputs for a point submitted earlier.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


STREAMED_TASKS = ("beam_point", "beam_sob", "beam_edep", "beam_edep_cpu")


MAX_WORKER_STARTS = 5     # respawns of a point's worker before job_status gives up and points at its log


class Backend:
    """The kit plus a state directory. ``kit`` is built on first use (tests pass a stand-in).

    ``detached`` (default: True when the kit is built here, False for an injected kit object): design points run
    in a detached worker process (fastsim_tools.worker) instead of this process. ``kit_factory`` ("module:function",
    default $FASTSIM_MCP_KIT_FACTORY, else mu2efastsim.endpoints.kit.FastsimKit) builds the kit from the state
    directory, here and in the worker."""

    def __init__(self, kit=None, state_dir: Optional[str | Path] = None, detached: Optional[bool] = None,
                 kit_factory: Optional[str] = None):
        self._kit = kit
        root = state_dir or os.environ.get("FASTSIMKIT_STATE_DIR") or "~/.mu2efastsim/kit"
        self.root = Path(root).expanduser().resolve()
        self.kit_factory = kit_factory or os.environ.get("FASTSIM_MCP_KIT_FACTORY") or ""
        self.detached = (kit is None) if detached is None else detached

    @property
    def kit(self):
        if self._kit is None:
            if self.kit_factory:
                import importlib
                module, _, func = self.kit_factory.partition(":")
                self._kit = getattr(importlib.import_module(module), func)(self.root)
            else:
                from mu2efastsim.endpoints.kit import FastsimKit
                self._kit = FastsimKit(state_dir=self.root)
        return self._kit

    # -- detached workers --------------------------------------------------------------------------------------
    def heartbeat(self, name: str) -> Dict[str, Any]:
        from .worker import heartbeat_path
        p = heartbeat_path(self.root, name)
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return {}

    def worker_alive(self, name: str) -> bool:
        """True while a worker holds the point's lock (a dead process releases it)."""
        from .worker import try_lock
        fh = try_lock(self.root, name)
        if fh is None:
            return True
        fh.close()
        return False

    def worker_log(self, name: str) -> Path:
        return self.root / "mcp_workers" / f"{name}.log"

    def spawn_worker(self, name: str) -> int:
        """Start the point's worker in its own session (it outlives this process and the MCP client)."""
        import subprocess
        import sys
        here = Path(__file__).resolve().parents[1]
        paths = [str(here)] + [p for p in sys.path if p and Path(p).is_dir() and str(p) != str(here)]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths), FASTSIMKIT_STATE_DIR=str(self.root))
        if self.kit_factory:
            env["FASTSIM_MCP_KIT_FACTORY"] = self.kit_factory
        log = self.worker_log(name)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as out:
            out.write(f"--- {now()} start\n")
            out.flush()
            proc = subprocess.Popen([sys.executable, "-m", "fastsim_tools.worker", "--state-dir", str(self.root),
                                     "--point", name], cwd=str(here), env=env, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        return proc.pid

    def ensure_worker(self, name: str) -> Dict[str, Any]:
        """Respawn the worker of an unfinished detached point when none runs. Returns what was done."""
        hb = self.heartbeat(name)
        if self.worker_alive(name):
            return {"alive": True, "restarted": False}
        if hb.get("phase") in ("done", "error", "expired"):
            return {"alive": False, "restarted": False}
        if hb.get("starts", 0) >= MAX_WORKER_STARTS:
            return {"alive": False, "restarted": False,
                    "error": f"worker started {hb['starts']} times without finishing; see {self.worker_log(name)}"}
        return {"alive": True, "restarted": True, "pid": self.spawn_worker(name)}

    def track(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Job states of a point without streaming from this process.

        Detached point: from the worker's heartbeat (the final states once it has finished), respawning a worker
        that died before the point was done. Points recorded before detached workers: straight from the kit, as
        before."""
        name = record["name"]
        if not record.get("detached"):
            return {"jobs": {j["handle"]: dict(self.status(j["handle"]), **self.stream_state(j["task"], j["handle"]))
                             for j in record["jobs"]}, "worker": None}
        worker = self.ensure_worker(name)
        hb = self.heartbeat(name)
        jobs = {}
        for j in record["jobs"]:
            h = j["handle"]
            beat = (hb.get("jobs") or {}).get(h, {})
            if beat.get("state"):          # a finished worker's beat holds the final states
                jobs[h] = dict(beat)
            elif hb.get("phase") == "error" or worker.get("error"):
                jobs[h] = dict(beat, state="failed",
                               message=f"worker: {hb.get('error') or worker.get('error')}")
            else:
                jobs[h] = dict(beat, state="working", message="submitting")
        info = {"alive": worker["alive"], "restarted": worker.get("restarted", False), "pid": hb.get("pid"),
                "phase": hb.get("phase"), "starts": hb.get("starts", 0), "log": str(self.worker_log(name)),
                "beat_age_s": round(time.time() - hb["beat"], 1) if hb.get("beat") else None}
        if worker.get("error") or hb.get("error"):
            info["error"] = worker.get("error") or hb.get("error")
        return {"jobs": jobs, "worker": info}

    # -- point records ---------------------------------------------------------------------------------------
    @property
    def points_dir(self) -> Path:
        return self.root / "mcp_points"

    def load_point(self, name: str) -> Optional[Dict[str, Any]]:
        p = self.points_dir / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def save_point(self, record: Dict[str, Any]) -> Path:
        self.points_dir.mkdir(parents=True, exist_ok=True)
        p = self.points_dir / f"{record['name']}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True))
        tmp.replace(p)
        return p

    def list_points(self) -> List[str]:
        return sorted(p.stem for p in self.points_dir.glob("*.json")) if self.points_dir.exists() else []

    def write_geometry(self, text: str, sha256: str) -> Path:
        d = self.root / "mcp_geometry"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{sha256}.txt"
        if not p.exists():
            p.write_text(text, encoding="ascii")
        return p

    # -- the kit ---------------------------------------------------------------------------------------------
    def check(self, handle: str, params: Dict[str, Any], files: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.kit.check(handle, params, files, [])

    def submit(self, handle: str, params: Dict[str, Any], files: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.kit.submit(handle, params, files, [])

    def status(self, handle: str) -> Dict[str, Any]:
        return self.kit.status(handle)

    def results(self, handle: str) -> Dict[str, Any]:
        return self.kit.results(handle)

    def cancel(self, handle: str) -> Dict[str, Any]:
        return self.kit.cancel(handle)

    def job_record(self, handle: str) -> Dict[str, Any]:
        """The endpoint's local job record (job id, remote directory), {} before submission."""
        p = self.root / "jobs" / f"{handle}.json"
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return {}

    def stream_state(self, task: str, handle: str) -> Dict[str, Any]:
        """Whether the kit has streamed the job's inputs in (the beam streaming tasks keep markers per task; None
        for other tasks: chain_point streams per sub-job and reports it in its status message)."""
        if task not in STREAMED_TASKS:
            return {"inputs_streamed": None}
        d = self.root / task
        if (d / f"{handle}.streamerr").exists():
            return {"inputs_streamed": False, "stream_error": (d / f"{handle}.streamerr").read_text()[:300]}
        if (d / f"{handle}.streamed").exists():
            return {"inputs_streamed": True}
        return {"inputs_streamed": False}


_BACKEND: Optional[Backend] = None


def get() -> Backend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = Backend()
    return _BACKEND


def set_backend(backend: Optional[Backend]) -> None:
    """Replace the process-wide backend (tests; a server configured with another state directory)."""
    global _BACKEND
    _BACKEND = backend


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
