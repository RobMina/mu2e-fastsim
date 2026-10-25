"""beam_point on an endpoint without FNAL storage (UVA, Slurm over SSH): one one-GPU job, inputs streamed in.

The Mu2e-internal input (the DS-entrance muon table g4ref.npz, made from MuBeamCat) is never stored on the endpoint:
when the job starts it writes ``<job dir>/inbox`` (node, tmpfs dir) and the kit pipes the file from the FNAL host
through the kit host into that tmpfs (``ssh <fnal> cat`` -> ``srun --overlap ... cat >``), checks its md5 and touches
``ready``; the job's EXIT trap deletes the tmpfs dir and writes ``out/cleanup.txt``, without which ``results`` refuses.
An absolute FNAL geometry path (a rendered Mu2eBO overlay) is streamed the same way. The public inputs (DS field map,
Offline geometry tree) are fetched inside the job from the CVMFS Stratum-1 (``beamfastsim.stage``). Code comes from a
checkout of the umbrella on the endpoint (FASTSIMKIT_BEAM_CODE, FASTSIMKIT_CODE_REF) run in the image (FASTSIMKIT_SIF).
The kit host must reach both the endpoint and the FNAL host by ssh for the point's lifetime.

Environment: FASTSIMKIT_FNAL_HOST (default mu2egpvm03), FASTSIMKIT_BEAM_DATA (FNAL dir holding g4ref_full/g4ref.npz),
FASTSIMKIT_BEAM_CODE, FASTSIMKIT_CODE_REF.
"""
from __future__ import annotations

import json
import os
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Dict

from mu2efastsim.endpoints import EndpointError, Resources
from mu2efastsim.endpoints.chainpoint import FnalSource

G4REF = "g4ref_full/g4ref.npz"
WAIT_S = 900
RESOURCES = Resources(cores=8, mem_gb=64, gpus=1, gpu_type="a100", minutes=30)
LIMITS = {"muons": (1, 10 ** 7), "seed": (0, 2 ** 31 - 1), "repeat": (0, 5)}

BODY = """@common@CODE=@code@
L=${TMPDIR:-/tmp}/beamfs_kit_j${SLURM_JOB_ID:-0}_$$; mkdir -p "$L"; chmod 700 "$L"
trap 'rm -rf "$L"; echo "$(date -Is) $(hostname) deleted $L" >> "$M2FS_OUT/cleanup.txt"' EXIT
trap 'exit 143' TERM INT
B() { apptainer exec --nv --cleanenv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},PYTHONPATH=$CODE/core:$CODE/beam,XLA_PYTHON_CLIENT_PREALLOCATE=false,BEAMFS_OFFLINE_ROOT=$L/pub/offline,BEAMFS_MAPS=$L/pub/maps" -B "$L" -B "$M2FS_OUT" -B "$CODE" --pwd "$CODE" @sif@ "$@"; }
B python -c "import beamfastsim.stage, beamfastsim.transport" || { echo "beamfastsim missing in $CODE"; exit 5; }
echo "$(hostname) $L" > "$(dirname "$M2FS_OUT")/inbox"      # read by the kit, which then streams the inputs in
tw=$SECONDS; while [ $((SECONDS-tw)) -lt @waits@ ]; do [ -f "$L/ready" ] && break; sleep 3; done
[ -f "$L/ready" ] || { echo "no input arrived within @waits@ s"; exit 3; }
echo "{\\"input_wait_s\\": $((SECONDS-tw))}" > "$M2FS_OUT/wait.json"
B python -m beamfastsim.stage "$L/pub" @stagegeom@ > "$M2FS_OUT/stage.txt" 2>&1 || { tail -5 "$M2FS_OUT/stage.txt"; exit 4; }
tail -1 "$M2FS_OUT/stage.txt"
B python -m beamfastsim point --g4 "$L/g4ref.npz" @geomarg@ --n @muons@ --seed @seed@ --repeat @repeat@ \\
  --batch 131072 --chunk 400 --out "$M2FS_OUT" > "$M2FS_OUT/point.txt" 2>&1 || { tail -20 "$M2FS_OUT/point.txt"; exit 6; }
tail -1 "$M2FS_OUT/point.txt"
"""


class _Source(FnalSource):
    """FnalSource that also takes absolute paths (a rendered geometry overlay on FNAL storage)."""

    def path(self, rel: str) -> str:
        return rel if rel.startswith("/") else super().path(rel)


def validate(params: Dict[str, Any]) -> Dict[str, Any]:
    from . import kit_task
    from mu2efastsim.endpoints.parameters import validate as validate_parameters
    return validate_parameters("beam_point", params, kit_task.PARAMS)


class BeamPointStream:
    def __init__(self, kit, source=None, code=None, background=True, poll_s=10.0):
        self.kit, self.ep = kit, kit.ep
        self.source = source or _Source(os.environ.get("FASTSIMKIT_FNAL_HOST", "mu2egpvm03"),
                                        os.environ.get("FASTSIMKIT_BEAM_DATA", "/exp/mu2e/data/users/ram2aq/beam_fastsim"))
        self.code = code or os.environ.get("FASTSIMKIT_BEAM_CODE", "/scratch/ram2aq/mu2efastsim/beam_code_kit")
        self.code_ref = os.environ.get("FASTSIMKIT_CODE_REF", "main")
        self.background, self.poll_s = background, poll_s
        self.dir = Path(kit.out_root).parent / "beam_point"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def _mark(self, name, what, text=""):
        (self.dir / f"{name}.{what}").write_text(text)

    def _has(self, name, what) -> bool:
        return (self.dir / f"{name}.{what}").exists()

    def _rec(self, name):
        p = self.dir / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    @staticmethod
    def inputs(p) -> Dict[str, str]:
        """Files the kit streams: {FNAL rel or abs path: name in the job's tmpfs}."""
        out = {G4REF: "g4ref.npz"}
        if p["geom"].startswith("/"):
            out[p["geom"]] = "geom_overlay.txt"
        return out

    def render(self, p):
        from mu2efastsim.endpoints.kit import _COMMON, APPTAINER_RUNTIME, admit_resources
        admit_resources(self.ep, RESOURCES)
        if not hasattr(self.ep, "stream_in"):
            raise ValueError(f"fastsimkit: beam_point streams its input into the job and needs an endpoint that can "
                             f"(slurm_ssh); {getattr(self.ep, 'name', '?')!r} cannot")
        g = p["geom"]
        if g == "nominal":
            stagegeom = geomarg = ""
        elif g.startswith("/"):
            stagegeom = geomarg = '--geom "$L/geom_overlay.txt"'
        else:
            stagegeom = geomarg = f'--geom "$CODE/beam/geom/{g}"'
        body = BODY.replace("@common@", _COMMON).replace("@runtime_a@", APPTAINER_RUNTIME["a"])
        subs = {"sif": shlex.quote(self.kit.sif), "code": shlex.quote(self.code), "waits": str(WAIT_S),
                "stagegeom": stagegeom, "geomarg": geomarg, "muons": str(p["muons"]), "seed": str(p["seed"]),
                "repeat": str(p["repeat"])}
        for a, b in subs.items():
            body = body.replace(f"@{a}@", b)
        import re
        if re.search(r"@[a-z_]+@", body):
            raise ValueError("fastsimkit: beam_point: unfilled placeholder")
        return body, RESOURCES

    # -- contract ------------------------------------------------------------------------------------------------
    def check(self, name, params) -> Dict[str, Any]:
        try:
            self.render(validate(params))
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def ensure_code(self):
        q = shlex.quote
        self.ep._sh(f"set -e; C={q(self.code)}; R={q(self.code_ref)}; "
                    f"if [ -d $C/.git ]; then git -C $C fetch -q origin && git -C $C checkout -q $R && "
                    f"git -C $C reset -q --hard origin/$R; else git clone -q https://github.com/RobMina/mu2e-fastsim.git $C "
                    f"&& git -C $C checkout -q $R; fi; git -C $C submodule update -q --init core; "
                    f"test -d $C/beam/beamfastsim", timeout=900)

    def submit(self, name, params) -> Dict[str, Any]:
        p = validate(params)
        old = self._rec(name)
        if old is not None and old["params"] != p:
            raise ValueError(f"fastsimkit: {name!r} was already submitted with different params")
        body, res = self.render(p)
        if old is None:
            self.ensure_code()
            (self.dir / f"{name}.json").write_text(json.dumps({"params": p, "submitted_at": time.time(),
                                                               "code": self.code, "code_ref": self.code_ref}))
        try:
            self.ep.submit(name, body, res)
        except EndpointError as e:
            raise ValueError(str(e))
        self.ensure_streaming(name)
        return {"handle": name}

    def ensure_streaming(self, name):
        if self._has(name, "streamed") or self._has(name, "streamerr"):
            return
        if not self.background:
            self.stream_step(name)
            return
        with self._lock:
            t = self._threads.get(name)
            if t is None or not t.is_alive():
                t = threading.Thread(target=self._loop, args=(name,), daemon=True)
                self._threads[name] = t
                t.start()

    def _loop(self, name):
        while not (self._has(name, "streamed") or self._has(name, "streamerr")):
            if self.stream_step(name) == "gone":
                return
            time.sleep(self.poll_s)

    def stream_step(self, name) -> str:
        """One attempt: 'waiting' (job not started), 'gone' (left the queue), 'streamed', 'error'."""
        st = self.ep.status(name)
        if st.state != "working":
            return "gone"
        inbox = self.ep.read_file(name, "inbox").split()
        if len(inbox) < 2:
            return "waiting"
        host, tmp = inbox[0], inbox[1]
        p = self._rec(name)["params"]
        try:
            for src, dest in self.inputs(p).items():
                want = self.source.md5(src)
                got = self.ep.stream_in(name, host, f"{tmp}/{dest}", self.source.argv(src))
                if got != want:
                    raise EndpointError(f"md5 mismatch for {src}: {got} != {want}")
            self.ep.run_on_node(name, host, f"touch {shlex.quote(tmp)}/ready")
        except EndpointError as e:
            self._mark(name, "streamerr", str(e))
            self.ep.cancel(name)
            return "error"
        self._mark(name, "streamed", time.strftime("%Y-%m-%dT%H:%M:%S"))
        return "streamed"

    def status(self, name) -> Dict[str, Any]:
        from mu2efastsim.endpoints.kit import POLL_MS
        s = self.ep.status(name)
        self.ensure_streaming(name)
        msg = f"{s.site_state} {s.message}".strip()
        if self._has(name, "streamerr"):
            msg += f"; stream error: {(self.dir / f'{name}.streamerr').read_text()[:200]}"
        return {"state": s.state, "message": msg, "poll_ms": POLL_MS if s.state == "working" else 0, "progress": None}

    def results(self, name) -> Dict[str, Any]:
        s = self.ep.status(name)
        if s.state != "completed":
            raise ValueError(f"fastsimkit: {name!r} is {s.state}, not completed")
        files = self.ep.fetch(name, self.kit.out_root)
        out = Path(self.kit.out_root) / name / "out"
        if not (out / "cleanup.txt").exists():
            raise ValueError(f"fastsimkit: {name!r} staged Mu2e-internal inputs but wrote no cleanup record")
        mpath = out / "metrics.json"
        if not mpath.exists():
            raise ValueError(f"fastsimkit: {name!r} wrote no out/metrics.json")
        metrics = {k: float(v) for k, v in json.loads(mpath.read_text()).items()}
        w = out / "wait.json"
        if w.exists():
            metrics["input_wait_s"] = float(json.loads(w.read_text()).get("input_wait_s", 0))
        return {"metrics": metrics,
                "files": [{"name": p.name, "uri": p.resolve().as_uri(), "kind": "text"} for p in files
                          if p.suffix in (".json", ".txt", ".out", ".log")],
                "metadata": {"endpoint": self.ep.name, "job_id": s.job_id, "site_state": s.site_state,
                             "code_ref": self._rec(name)["code_ref"]}}

    def cancel(self, name) -> Dict[str, Any]:
        return {"state": self.ep.cancel(name)}
