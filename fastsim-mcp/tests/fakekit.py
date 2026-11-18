"""A file-based stand-in for the fastsim kit, shared by processes (the test and detached workers).

Each job is a JSON file under <root>/fake/. Streaming is simulated the way the kit does it: a ``status`` call from
the process holding the kit advances the job's input stream by one chunk (FAKEKIT_STREAM_SLEEP_S each, recording the
pid that streamed it); after FAKEKIT_STREAM_STEPS chunks it writes the kit's ``<task>/<handle>.streamed`` marker.
The job then completes after FAKEKIT_RUN_S of wall time. A job not streamed within FAKEKIT_WAIT_S of its creation
fails like a real one ("no input arrived"). submit is idempotent per handle, as the kit's is.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastsim_tools import design_point as dp


def _env(name, default):
    return float(os.environ.get(name, default))


class FileKit:
    def __init__(self, root):
        self.root = Path(root)
        self.dir = self.root / "fake"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name):
        return self.dir / f"{name}.json"

    def load(self, name):
        return json.loads(self._path(name).read_text())

    def _save(self, name, rec):
        tmp = self._path(name).with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(rec))
        tmp.replace(self._path(name))

    def check(self, name, params, files, inputs):
        return {"ok": True, "message": ""}

    def submit(self, name, params, files, inputs):
        if params.get("seed") == 13:
            raise ValueError("fake submit refused (seed 13)")
        if self._path(name).exists():
            rec = self.load(name)
            if rec["params"] != params:
                raise ValueError("request differs")
            rec["submits"] += 1
        else:
            n = len(list(self.dir.glob("*.json")))
            rec = {"params": params, "state": "working", "created": time.time(), "chunks": [], "streamed_at": None,
                   "submits": 1, "job_id": str(5000 + n)}
        self._save(name, rec)
        jobs = self.root / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        (jobs / f"{name}.json").write_text(json.dumps({"job_id": rec["job_id"]}))
        return {"handle": name}

    def status(self, name):
        rec = self.load(name)
        task = rec["params"]["task"]
        now = time.time()
        if rec["state"] == "working":
            if rec["streamed_at"] is None:
                if now - rec["created"] > _env("FAKEKIT_WAIT_S", 60):
                    rec["state"] = "failed"
                    rec["message"] = "no input arrived"
                else:                                  # one chunk of the input stream, from this process
                    time.sleep(_env("FAKEKIT_STREAM_SLEEP_S", 0))
                    rec = self.load(name)              # another process may have streamed meanwhile
                    rec["chunks"].append(os.getpid())
                    if len(rec["chunks"]) >= _env("FAKEKIT_STREAM_STEPS", 2):
                        rec["streamed_at"] = now
                        marker = self.root / task / f"{name}.streamed"
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.write_text(str(now))
            elif now - rec["streamed_at"] >= _env("FAKEKIT_RUN_S", 0.5):
                rec["state"] = "completed"
            self._save(name, rec)
        return {"state": rec["state"], "message": rec.get("message", rec["state"].upper()), "poll_ms": 0}

    def results(self, name):
        rec = self.load(name)
        if rec["state"] != "completed":
            raise ValueError(f"{name} is {rec['state']}")
        out = self.root / "out" / name / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / dp.EDEP_FILE).write_bytes(b"root")
        (out / "metrics.json").write_text(json.dumps({"ce_acceptance": 0.52, "stops_per_pot": 9e-4}))
        return {"metrics": {"ce_acceptance": 0.52, "stops_per_pot": 9e-4},
                "files": [{"name": dp.EDEP_FILE, "uri": (out / dp.EDEP_FILE).as_uri(), "kind": "root"}],
                "metadata": {"job_id": rec["job_id"]}}

    def cancel(self, name):
        rec = self.load(name)
        if rec["state"] == "working":
            rec["state"] = "cancelled"
            self._save(name, rec)
        return {"state": rec["state"]}


def make(root):
    return FileKit(root)
