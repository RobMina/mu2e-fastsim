"""crv_response on an endpoint without FNAL storage (UVA, Slurm over SSH): one one-GPU job, inputs streamed in.

The Mu2e-internal inputs (`kit_task.inputs(params)`: the five CrvStep tables, CRVCalib/CRVPhoton CSVs, the period's time
calibration and data reference, the background model; ~530 MB) are never stored on the endpoint: when the job starts it
writes ``<job dir>/inbox`` (node, tmpfs dir) and the kit pipes each file from the FNAL host through the kit host into
that tmpfs (``ssh <fnal> cat`` -> ``srun --overlap ... cat >``), checks its md5 and touches ``ready``; the job's EXIT trap
deletes the tmpfs dir and writes ``out/cleanup.txt``, without which ``results`` refuses. The lookup tables come from the
CVMFS Stratum-1 inside the job. Code: a checkout of the umbrella on the endpoint (FASTSIMKIT_CRV_CODE at
FASTSIMKIT_CRV_CODE_REF, submodules at their pins, or FASTSIMKIT_CRV_BRANCH checked out in core and crv) run in the
image (FASTSIMKIT_SIF). Inputs are cached once per node in /dev/shm (see WAIT), so later points on a node stream nothing. The kit host must
reach both the endpoint and the FNAL host by ssh for the point's lifetime.

Environment: FASTSIMKIT_FNAL_HOST (default mu2egpvm03), FASTSIMKIT_CRV_DATA (default .../crv_fastsim),
FASTSIMKIT_CRV_CODE, FASTSIMKIT_CRV_CODE_REF (umbrella ref, default main), FASTSIMKIT_CRV_BRANCH (optional).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Dict

from mu2efastsim.endpoints import EndpointError
from mu2efastsim.endpoints.chainpoint import FnalSource

from . import kit_task

# Each stream is an ssh channel to the FNAL host and one to the endpoint, both on one ControlMaster per host, which sshd
# caps at MaxSessions (10, see chainpoint): at most FASTSIMKIT_MAX_CHANNELS streams across all points of the kit process.
CHANNELS = threading.BoundedSemaphore(int(os.environ.get("FASTSIMKIT_MAX_CHANNELS", 6)))
WAIT_S = 1800          # input wait: ~530 MB through the kit host at >= 1 MB/s (measured 4-6.6 MB/s for chain_point)
# runs after the kit's _TABLES block (which defines L, the EXIT trap and the lookup tables in $L): announce the input
# dir, wait for the kit to stream the inputs in
# Inputs once per node: /dev/shm/crvfs_resp_<user>_<key> (key = md5 of the point's input set), one holder file per job,
# deleted by the last holder (flock); holders of jobs that left the queue are dropped first. Knobs do not change the
# inputs, so every later point of the same period on that node finds them ready and streams nothing.
WAIT = r'''S=/dev/shm/crvfs_resp_${USER}_@inkey@; LOCK=/dev/shm/crvfs_resp_${USER}.lock; J=${SLURM_JOB_ID:-0}
in_hold() { ( flock 9; mkdir -p "$S/holders" "$S/in"; chmod 700 "$S"
  for h in "$S"/holders/*; do [ -e "$h" ] || continue; squeue -h -j "${h##*/}" -o %i 2>/dev/null | grep -q . || rm -f "$h"; done
  touch "$S/holders/$J" ) 9>"$LOCK"; }
in_release() { ( flock 9; rm -f "$S/holders/$J"; [ -z "$(ls -A "$S/holders" 2>/dev/null)" ] && { rm -rf "$S"; echo "$(date -Is) $(hostname) deleted node inputs $S" >> "$M2FS_OUT/cleanup.txt"; } ) 9>"$LOCK"; }
trap 'in_release; rm -rf "$L"; echo "$(date -Is) $(hostname) deleted $L" >> "$M2FS_OUT/cleanup.txt"' EXIT
in_hold; ln -sfn "$S/in" "$L/in"
INSTATE=$( [ -f "$S/in/ready" ] && echo have || echo need )
echo "$(hostname) $S/in $INSTATE" > "$(dirname "$M2FS_OUT")/inbox"     # read by the kit, which streams the inputs if needed
tw=$SECONDS; while [ $((SECONDS-tw)) -lt @waits@ ]; do [ -f "$S/in/ready" ] && break; sleep 3; done
[ -f "$S/in/ready" ] || { echo "no input arrived within @waits@ s"; exit 3; }
echo "{\"input_wait_s\": $((SECONDS-tw)), \"inputs\": \"$INSTATE\"}" > "$M2FS_OUT/wait.json"
'''


def validate(params: Dict[str, Any]) -> Dict[str, Any]:
    """The point's params (defaults filled), as the kit's Task check would; raises ValueError."""
    extra = sorted(set(params) - set(kit_task.PARAMS) - {"task"})
    if extra or "seed" not in params:
        raise ValueError(f"fastsimkit: task 'crv_response' takes {list(kit_task.PARAMS)}; extra {extra}"
                         + ("" if "seed" in params else ", missing ['seed']"))
    p = {**kit_task.DEFAULTS, **{k: v for k, v in params.items() if k != "task"}}
    for k in kit_task.KNOB_PARAMS:
        v = p[k]
        if v not in ("", None) and (isinstance(v, bool) or not isinstance(v, (int, float, str))
                                    or not re.fullmatch(r"[0-9.eE+-]+", str(v))):
            raise ValueError(f"fastsimkit: crv_response {k} must be a number, got {v!r}")
    kit_task.inputs(p)                       # period / tables checks
    return p


class Delegate:
    """kit.DELEGATES entry: applies on endpoints without FNAL storage that can stream into a job."""

    @staticmethod
    def applies(endpoint):
        return not getattr(endpoint, "fnal_data", None) and hasattr(endpoint, "stream_in")

    @staticmethod
    def make(kit):
        return CrvResponseStream(kit)


class CrvResponseStream:
    def __init__(self, kit, source=None, code=None, background=True, poll_s=10.0):
        self.kit, self.ep = kit, kit.ep
        self.source = source or FnalSource(os.environ.get("FASTSIMKIT_FNAL_HOST", "mu2egpvm03"),
                                           os.environ.get("FASTSIMKIT_CRV_DATA", "/exp/mu2e/data/users/ram2aq/crv_fastsim"))
        self.code = code or os.environ.get("FASTSIMKIT_CRV_CODE", "/scratch/ram2aq/mu2efastsim/crv_response_code_kit")
        self.code_ref = os.environ.get("FASTSIMKIT_CRV_CODE_REF", "main")
        self.branch = os.environ.get("FASTSIMKIT_CRV_BRANCH", "")
        self.background, self.poll_s = background, poll_s
        self.dir = Path(kit.out_root).parent / "crv_response"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._keys: Dict[tuple, str] = {}
        self._md5: Dict[str, str] = {}
        self._node: Dict[str, threading.Lock] = {}      # one input stream per (node, cache) at a time
        self.streams = int(os.environ.get("FASTSIMKIT_STREAMS", 3))      # parallel file streams per point

    def _mark(self, name, what, text=""):
        (self.dir / f"{name}.{what}").write_text(text)

    def _has(self, name, what) -> bool:
        return (self.dir / f"{name}.{what}").exists()

    def _rec(self, name):
        p = self.dir / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def render(self, p):
        from mu2efastsim.endpoints import kit as K
        if not hasattr(self.ep, "stream_in"):
            raise ValueError(f"fastsimkit: crv_response streams its inputs into the job and needs an endpoint that can "
                             f"(slurm_ssh); {getattr(self.ep, 'name', '?')!r} cannot")
        code = shlex.quote(self.code)
        rt = {"a": f'CODE={code}\n'
                   'A() { apptainer exec --nv --cleanenv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},'
                   'PYTHONPATH=$CODE/core:$CODE/crv,XLA_PYTHON_CLIENT_PREALLOCATE=false" -B "$M2FS_OUT" -B "$CODE" '
                   '${L:+-B "$L"} --pwd "$CODE" @sif@ "$@"; }\n',
              "t": 'T() { apptainer exec --nv --cleanenv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},'
                   'PYTHONPATH=$CODE/core:$CODE/crv,XLA_PYTHON_CLIENT_PREALLOCATE=false,CRVFS_LOOKUP_DIR=$L" '
                   '-B "$L" -B "$M2FS_OUT" -B "$CODE" --pwd "$CODE" @sif@ "$@"; }'}
        tables = K._TABLES.replace("@runtime_t@", rt["t"] + "\n" + WAIT)
        body = kit_task.body(K._COMMON.replace("@runtime_a@", rt["a"]), tables)
        body = body.replace('A python -c "import jax', 'A python -c "import crvfastsim.kit_task, jax', 1)
        subs = {"sif": shlex.quote(self.kit.sif), "fnal_data": "/nonexistent_fnal", "inputs_dir": '"$L/in"',
                "waits": str(WAIT_S), "inkey": self.input_key(p),
                **{f"p_{k}": shlex.quote(K._arg(p[k])) for k in kit_task.PARAMS}}
        for a, b in subs.items():
            body = body.replace(f"@{a}@", b)
        if re.search(r"@[a-z_0-9]+@", body):
            raise ValueError("fastsimkit: crv_response: unfilled placeholder")
        return body, K.TASKS["crv_response"].resources

    def input_key(self, p) -> str:
        """Node-cache key of a point's inputs: the md5 of their names and md5s (a changed input file is a new cache)."""
        import hashlib
        key = tuple(sorted(kit_task.inputs(p).items()))
        if key not in self._keys:
            h = hashlib.md5()
            for rel, dest in key:
                h.update(f"{rel} {dest} {self.md5(rel)}\n".encode())
            self._keys[key] = h.hexdigest()[:12]
        return self._keys[key]

    def md5(self, rel) -> str:
        if rel not in self._md5:
            self._md5[rel] = self.source.md5(rel)
        return self._md5[rel]

    # -- contract ------------------------------------------------------------------------------------------------
    def check(self, name, params) -> Dict[str, Any]:
        try:
            self.render(validate(params))
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def ensure_code(self):
        q = shlex.quote
        br = (f"for s in core crv; do git -C $C/$s fetch -q origin {q(self.branch)} && "
              f"git -C $C/$s checkout -q FETCH_HEAD; done; " if self.branch else "")
        self.ep._sh(f"set -e; C={q(self.code)}; R={q(self.code_ref)}; "
                    f"if [ -d $C/.git ]; then git -C $C fetch -q origin && git -C $C checkout -q $R && "
                    f"git -C $C reset -q --hard origin/$R; else git clone -q https://github.com/RobMina/mu2e-fastsim.git $C "
                    f"&& git -C $C checkout -q $R; fi; git -C $C submodule update -q --init core crv; {br}"
                    f"test -f $C/crv/crvfastsim/kit_task.py; git -C $C/crv log -1 --format=%h", timeout=900)

    def submit(self, name, params) -> Dict[str, Any]:
        p = validate(params)
        old = self._rec(name)
        if old is not None and old["params"] != p:
            raise ValueError(f"fastsimkit: {name!r} was already submitted with different params")
        body, res = self.render(p)
        if old is None:
            self.ensure_code()
            (self.dir / f"{name}.json").write_text(json.dumps({"params": p, "submitted_at": time.time(),
                                                               "code": self.code, "code_ref": self.code_ref,
                                                               "branch": self.branch}))
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
            with self._lock:
                node_lock = self._node.setdefault(f"{host} {tmp}", threading.Lock())
            with node_lock:   # a second point on the node waits for the first one's stream, then finds it ready
                have = self.ep.run_on_node(name, host, f"test -f {shlex.quote(tmp)}/ready && echo have || true").strip()
                if have == "have":
                    self._mark(name, "inputsreused", host)
                else:
                    from concurrent.futures import ThreadPoolExecutor

                    def one(item):      # largest first, `streams` at a time; ssh channels capped across points
                        src, dest = item
                        with CHANNELS:
                            got = self.ep.stream_in(name, host, f"{tmp}/{dest}", self.source.argv(src))
                        if got != self.md5(src):
                            raise EndpointError(f"md5 mismatch for {src}: {got} != {self.md5(src)}")
                    items = sorted(kit_task.inputs(p).items(), key=lambda kv: -kit_task.SIZE_MB.get(kv[0], 1))
                    with ThreadPoolExecutor(max_workers=self.streams) as ex:
                        for f in [ex.submit(one, it) for it in items]:
                            f.result()
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
            wj = json.loads(w.read_text())
            metrics["input_wait_s"] = float(wj.get("input_wait_s", 0))
            metrics["inputs_reused"] = 1.0 if wj.get("inputs") == "have" else 0.0
        rec = self._rec(name)
        return {"metrics": metrics,
                "files": [{"name": p.name, "uri": p.resolve().as_uri(), "kind": "text"} for p in files
                          if p.suffix in (".json", ".txt", ".out", ".log")],
                "metadata": {"endpoint": self.ep.name, "job_id": s.job_id, "site_state": s.site_state,
                             "code_ref": rec["code_ref"], "branch": rec.get("branch", "")}}

    def cancel(self, name) -> Dict[str, Any]:
        return {"state": self.ep.cancel(name)}
