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


class Backend:
    """The kit plus a state directory. ``kit`` is built on first use (tests pass a stand-in)."""

    def __init__(self, kit=None, state_dir: Optional[str | Path] = None):
        self._kit = kit
        root = state_dir or os.environ.get("FASTSIMKIT_STATE_DIR") or "~/.mu2efastsim/kit"
        self.root = Path(root).expanduser().resolve()

    @property
    def kit(self):
        if self._kit is None:
            from mu2efastsim.endpoints.kit import FastsimKit
            self._kit = FastsimKit(state_dir=self.root)
        return self._kit

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
        """Whether the kit has streamed the job's inputs in (streaming tasks keep markers per task)."""
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
