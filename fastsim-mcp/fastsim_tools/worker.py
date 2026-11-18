"""The detached point worker: submits a design point's jobs, streams their inputs, tracks them and fetches outputs.

A design point must not depend on the MCP client staying connected: the kit streams the Mu2e-internal inputs into a
running job from the process that holds the kit, and the job gives up after its input wait. So the server starts
one worker per point in its own session (``python -m fastsim_tools.worker --state-dir DIR --point NAME``), and the
worker does everything that needs a live kit:

  1. ``kit.submit`` of every job (idempotent per handle: a restarted worker gets the same jobs back);
  2. ``kit.status`` of every job still in the queue every ``poll`` seconds, which keeps the kit's streaming going;
  3. ``kit.results`` of every job that completes, so the outputs are on the kit host without a client;
  4. a heartbeat file (``mcp_workers/<name>.json``) after each round, read by job_status and fetch_outputs.

It holds ``mcp_workers/<name>.lock`` (flock) for its lifetime, so there is at most one worker per point; a second
one exits at once. It ends when every job is terminal and fetched, or after ``max_hours``. The server respawns a
worker that died before the point was done (``Backend.track``).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

TERMINAL = ("completed", "failed", "cancelled")


def heartbeat_path(root: Path, name: str) -> Path:
    return Path(root) / "mcp_workers" / f"{name}.json"


def lock_path(root: Path, name: str) -> Path:
    return Path(root) / "mcp_workers" / f"{name}.lock"


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, default=str))
    tmp.replace(path)


def try_lock(root: Path, name: str):
    """An open, exclusively locked file object, or None when another process holds the point's lock."""
    p = lock_path(root, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def done(job: Dict[str, Any]) -> bool:
    """A job the worker no longer needs to watch: terminal, and fetched (or its fetch failed) when completed."""
    if job.get("state") not in TERMINAL:
        return False
    return job["state"] != "completed" or job.get("fetched") or bool(job.get("results_error"))


def run(name: str, backend, poll_s: float = 30.0, max_hours: float = 48.0, once: bool = False,
        handle_sigterm: bool = False) -> int:
    """Run the worker of point ``name`` with ``backend`` (fastsim_tools.backend.Backend). Returns an exit code.
    ``handle_sigterm``: record a polite stop in the heartbeat (the worker process; not for in-process callers)."""
    root = backend.root
    lock = try_lock(root, name)
    if lock is None:
        return 0                                     # another worker has the point
    record = backend.load_point(name)
    if record is None:
        return 2
    hb_file = heartbeat_path(root, name)
    old = json.loads(hb_file.read_text()) if hb_file.exists() else {}
    hb = {"name": name, "pid": os.getpid(), "host": socket.gethostname(), "started": time.time(),
          "starts": old.get("starts", 0) + 1, "phase": "submitting", "error": None,
          "jobs": {j["handle"]: dict(old.get("jobs", {}).get(j["handle"], {}), dtype=j.get("dtype"), task=j["task"])
                   for j in record["jobs"]}}

    def beat(phase=None):
        if phase:
            hb["phase"] = phase
        hb["beat"] = time.time()
        write_json(hb_file, hb)

    def stop(signum, frame):                          # a polite stop: say so, the server may respawn
        beat("stopped")
        sys.exit(0)

    if handle_sigterm:
        signal.signal(signal.SIGTERM, stop)
    beat()
    try:
        files = record.get("files") or []
        for j in record["jobs"]:
            backend.submit(j["handle"], j["params"], files)
            hb["jobs"][j["handle"]]["job_id"] = backend.job_record(j["handle"]).get("job_id")
        beat("tracking")
        deadline = time.time() + 3600 * max_hours
        while True:
            for j in record["jobs"]:
                h, job = j["handle"], hb["jobs"][j["handle"]]
                if done(job):
                    continue
                s = backend.status(h)                 # keeps the kit's input streaming going for this job
                job.update(state=s.get("state"), message=s.get("message", ""),
                           job_id=job.get("job_id") or backend.job_record(h).get("job_id"),
                           **backend.stream_state(j["task"], h))
                if job["state"] == "completed" and not job.get("fetched"):
                    try:
                        backend.results(h)
                        job["fetched"] = True
                    except Exception as exc:          # recorded, fetch_outputs retries in the server
                        job["results_error"] = f"{type(exc).__name__}: {exc}"[:300]
            if all(done(job) for job in hb["jobs"].values()):
                beat("done")
                return 0
            if time.time() > deadline:
                beat("expired")
                return 3
            beat()
            if once:
                return 0
            time.sleep(poll_s)
    except Exception as exc:
        hb["error"] = f"{type(exc).__name__}: {exc}"[:500]
        traceback.print_exc()
        beat("error")
        return 1
    finally:
        lock.close()


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m fastsim_tools.worker", description=__doc__.split("\n")[0])
    p.add_argument("--state-dir", required=True)
    p.add_argument("--point", required=True)
    p.add_argument("--poll-s", type=float, default=float(os.environ.get("FASTSIM_MCP_WORKER_POLL_S", "30")))
    p.add_argument("--max-hours", type=float, default=float(os.environ.get("FASTSIM_MCP_WORKER_MAX_H", "48")))
    a = p.parse_args(argv)
    from .backend import Backend
    return run(a.point, Backend(state_dir=a.state_dir, detached=False), a.poll_s, a.max_hours, handle_sigterm=True)


if __name__ == "__main__":
    raise SystemExit(main())
