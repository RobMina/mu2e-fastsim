"""crv_response on an endpoint without FNAL storage (UVA, Slurm over SSH): one one-GPU job, inputs streamed in.

The Mu2e-internal inputs (`kit_task.inputs(params)`: the five CrvStep tables, CRVCalib/CRVPhoton CSVs, the period's time
calibration and data reference, the background model; ~530 MB) are never stored on the endpoint: when the job starts it
writes ``<job dir>/inbox`` (node, tmpfs dir) and the kit pipes each file from the FNAL host through the kit host into
that tmpfs (``ssh <fnal> cat`` -> ``srun --overlap ... cat >``), checks its md5 and touches ``ready``; the job's EXIT trap
deletes the tmpfs dir and writes ``out/cleanup.txt``, without which ``results`` refuses. The lookup tables come from the
CVMFS Stratum-1 inside the job. Code: a retained read-only snapshot of the umbrella on the endpoint (FASTSIMKIT_CRV_CODE at
FASTSIMKIT_CRV_CODE_REF, submodules at their pins, or FASTSIMKIT_CRV_BRANCH checked out in core and crv) run in the
image (FASTSIMKIT_SIF). Inputs are cached once per node in /dev/shm (see WAIT), so later points on a node stream nothing. The kit host must
reach both the endpoint and the FNAL host by ssh for the point's lifetime.

Environment (see site.env.example): FASTSIMKIT_FNAL_HOST (required: the FNAL host the inputs stream from),
FASTSIMKIT_CRV_DATA (default mu2efastsim.site.crv_data()), FASTSIMKIT_CRV_CODE (default
<FASTSIMKIT_SCRATCH>/crv_response_code_kit), FASTSIMKIT_CRV_CODE_REF (umbrella ref, default main),
FASTSIMKIT_CRV_BRANCH (optional). A missing required variable raises mu2efastsim.site.SiteConfigError.
"""
from __future__ import annotations

from mu2efastsim import site

import json
from dataclasses import asdict
import os
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Dict

from mu2efastsim.endpoints import EndpointError, Resources
from mu2efastsim.endpoints.chainpoint import FnalSource
from mu2efastsim.endpoints.artifacts import (binding, checked_record, input_digest, make_manifest, resolve_image,
                                           resolve_inputs, runtime_prefix, verify_cache, verify_result)
from mu2efastsim.endpoints.snapshots import prepare_code
from mu2efastsim.endpoints.submission import submission_lock

from . import kit_task

# Each stream is an ssh channel to the FNAL host and one to the endpoint, both on one ControlMaster per host, which sshd
# caps at MaxSessions (10, see chainpoint): at most FASTSIMKIT_MAX_CHANNELS streams and md5 calls across all points of the
# kit process (4: q=4 points starting together refused sessions at 6 streams plus md5 calls plus polling, crvpb03/04).
class _Slots:
    """Cross-process semaphore: N lock files under FASTSIMKIT_CHANNEL_DIR, one held (flock) per open channel. Every
    point is its own kit process (one graph.run per point), so an in-process cap cannot bound the ssh sessions."""

    def __init__(self, n, root):
        self.n, self.root = max(1, n), Path(os.path.expanduser(root))
        self._local = threading.local()

    def __enter__(self):
        import fcntl
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            for i in range(self.n):
                fh = open(self.root / f"slot{i}", "a")
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    fh.close()
                    continue
                self._local.__dict__.setdefault("held", []).append(fh)
                return self
            time.sleep(0.5)

    def __exit__(self, *exc):
        import fcntl
        fh = self._local.held.pop()
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


CHANNELS = _Slots(int(os.environ.get("FASTSIMKIT_MAX_CHANNELS", 4)),
                  os.environ.get("FASTSIMKIT_CHANNEL_DIR", "~/.mu2efastsim/ssh_slots"))
RETRIES, RETRY_S = int(os.environ.get("FASTSIMKIT_STREAM_RETRIES", 3)), 20.0   # per file, before the point fails
WAIT_S = 1800          # input wait: ~530 MB through the kit host at >= 1 MB/s (measured 4-6.6 MB/s for chain_point)
# runs after the kit's _TABLES block (which defines L, the EXIT trap and the lookup tables in $L): announce the input
# dir, wait for the kit to stream the inputs in
# Inputs once per node: /dev/shm/crvfs_resp_<user>_<key> (key = SHA256 of the pinned input set), one holder per job,
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
    from mu2efastsim.endpoints.parameters import validate as validate_parameters
    p = validate_parameters("crv_response", params, kit_task.PARAMS, kit_task.DEFAULTS)
    kit_task.inputs(p)
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
        self.source = source or FnalSource(site.require("FASTSIMKIT_FNAL_HOST"),
                                           site.setting("FASTSIMKIT_CRV_DATA") or site.crv_data())
        self.code = code or site.setting("FASTSIMKIT_CRV_CODE") or site.scratch() + "/crv_response_code_kit"
        self.code_ref = os.environ.get("FASTSIMKIT_CRV_CODE_REF", "main")
        self.branch = os.environ.get("FASTSIMKIT_CRV_BRANCH", "")
        self.background, self.poll_s = background, poll_s
        self.dir = Path(kit.out_root).parent / "crv_response"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._node: Dict[str, threading.Lock] = {}      # one input stream per (node, cache) at a time
        self.streams = int(os.environ.get("FASTSIMKIT_STREAMS", 3))      # parallel file streams per point

    def _mark(self, name, what, text=""):
        (self.dir / f"{name}.{what}").write_text(text)

    def _has(self, name, what) -> bool:
        return (self.dir / f"{name}.{what}").exists()

    def _rec(self, name):
        p = self.dir / f"{name}.json"
        return checked_record(p, allow_legacy=True)

    def render(self, p, manifest=None):
        from mu2efastsim.endpoints import kit as K
        K.admit_resources(self.ep, K.TASKS["crv_response"].resources)
        if not hasattr(self.ep, "stream_in"):
            raise ValueError(f"fastsimkit: crv_response streams its inputs into the job and needs an endpoint that can "
                             f"(slurm_ssh); {getattr(self.ep, 'name', '?')!r} cannot")
        code = shlex.quote(manifest["code"]["path"] if manifest else self.code)
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
        subs = {"sif": shlex.quote(manifest["image"]["path"] if manifest else self.kit.sif),
                "fnal_data": "/nonexistent_fnal", "inputs_dir": '"$L/in"',
                "waits": str(WAIT_S), "inkey": input_digest(manifest["inputs"]) if manifest else "preview",
                **{f"p_{k}": shlex.quote(K._arg(p[k])) for k in kit_task.PARAMS}}
        for a, b in subs.items():
            body = body.replace(f"@{a}@", b)
        if re.search(r"@[a-z_0-9]+@", body):
            raise ValueError("fastsimkit: crv_response: unfilled placeholder")
        if manifest is not None:
            from mu2efastsim.endpoints.artifacts import consumption_guard
            marker = 'echo "{\\"input_wait_s'
            at = body.index(marker)
            body = body[:at] + consumption_guard('"$L/in"', manifest["inputs"]["files"]) + body[at:]
            body = runtime_prefix(manifest) + body
        return body, K.TASKS["crv_response"].resources

    def input_key(self, p) -> str:
        """Resolve afresh: replacing bytes at the same source path must change the node-cache identity."""
        with CHANNELS:
            return input_digest(resolve_inputs(self.source, kit_task.inputs(p)))

    def md5(self, rel) -> str:
        """Compatibility helper; never memoize a mutable source path."""
        with CHANNELS:
            return self.source.md5(rel)

    # -- contract ------------------------------------------------------------------------------------------------
    def check(self, name, params) -> Dict[str, Any]:
        try:
            self.render(validate(params))
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def ensure_code(self):
        return prepare_code(self.ep, self.code, self.code_ref, modules=('core', 'crv'), branch=self.branch)

    def submit(self, name, params) -> Dict[str, Any]:
        p = validate(params)
        self.render(p)
        target = {"endpoint": self.ep.name, "host": getattr(self.ep, "host", ""),
                  "source_host": self.source.host, "source_root": self.source.root}
        target.update({k: getattr(self.ep, k) for k in ("account", "partition", "workdir") if hasattr(self.ep, k)})

        def prepare():
            with CHANNELS:
                inputs = resolve_inputs(self.source, kit_task.inputs(p))
            code = self.ensure_code()
            image = resolve_image(self.ep, self.kit.sif)
            manifest = make_manifest("crv_response", code, image, inputs, p)
            body, res = self.render(p, manifest)
            return {"params": p, "submitted_at": time.time(), "code": code["path"],
                    "code_ref": self.code_ref, "branch": self.branch, "manifest": manifest,
                    "body": body, "resources": asdict(res), "target": target}

        try:
            rec = binding(self.dir, name, p, target, prepare)
            self.ep.submit(name, rec["body"], Resources(**rec["resources"]))
        except EndpointError as e:
            raise ValueError(str(e)) from e
        self.ensure_streaming(name)
        return {"handle": name}

    def ensure_streaming(self, name):
        if "manifest" not in (self._rec(name) or {}):
            return
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
        try:
            rec = self._rec(name)
            if not rec or "manifest" not in rec:
                raise ValueError("legacy streaming record has no artifact manifest")
            entries = rec["manifest"]["inputs"]["files"]
            import hashlib
            # State-root locks serialize cooperating cache writers on the same endpoint and node.
            key = hashlib.sha256(f"{getattr(self.ep, 'host', '')} {host} {tmp}".encode()).hexdigest()
            with self._lock:
                node_lock = self._node.setdefault(key, threading.Lock())
            with node_lock, submission_lock(self.dir / ".cache-writers", key):
                have = self.ep.run_on_node(name, host, f"test -f {shlex.quote(tmp)}/ready && echo have || true").strip()
                if have == "have":
                    if not verify_cache(self.ep, name, host, tmp, entries):
                        raise ValueError("ready input cache does not match the bound manifest; refusing overwrite")
                    self._mark(name, "inputsreused", host)
                else:
                    from concurrent.futures import ThreadPoolExecutor

                    def one(entry):
                        for attempt in range(RETRIES + 1):
                            try:
                                with CHANNELS:
                                    got = self.ep.stream_in(name, host, f"{tmp}/{entry['dest']}", entry["argv"])
                            except EndpointError:
                                if attempt == RETRIES:
                                    raise
                                time.sleep(RETRY_S * (attempt + 1))
                                continue
                            if got != entry["md5"]:
                                raise ValueError(f"md5 mismatch for {entry['src']}: {got} != {entry['md5']}")
                            return
                    items = sorted(entries, key=lambda entry: -entry["size"])
                    with ThreadPoolExecutor(max_workers=self.streams) as ex:
                        for future in [ex.submit(one, entry) for entry in items]:
                            future.result()
                    if not verify_cache(self.ep, name, host, tmp, entries):
                        raise ValueError("streamed input cache does not match the bound manifest")
                    self.ep.run_on_node(name, host, f"touch {shlex.quote(tmp)}/ready")
        except (EndpointError, ValueError, KeyError) as e:
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
        rec = self._rec(name)
        if rec is None:
            raise ValueError(f"fastsimkit: {name!r} has no task record")
        if "manifest" in rec:
            verify_result(out, rec["manifest"])
        metrics = {k: float(v) for k, v in json.loads(mpath.read_text()).items()}
        w = out / "wait.json"
        if w.exists():
            wj = json.loads(w.read_text())
            metrics["input_wait_s"] = float(wj.get("input_wait_s", 0))
            metrics["inputs_reused"] = 1.0 if wj.get("inputs") == "have" else 0.0
        return {"metrics": metrics,
                "files": [{"name": p.name, "uri": p.resolve().as_uri(), "kind": "text"} for p in files
                          if p.suffix in (".json", ".txt", ".out", ".log")],
                "metadata": {"endpoint": self.ep.name, "job_id": s.job_id, "site_state": s.site_state,
                             "artifact_manifest": rec.get("manifest"), "manifest_sha": rec.get("manifest", {}).get("manifest_sha"),
                             "artifact_binding": "verified" if "manifest" in rec else "legacy-unverified",
                             "code_ref": rec["code_ref"], "branch": rec.get("branch", "")}}

    def cancel(self, name) -> Dict[str, Any]:
        return {"state": self.ep.cancel(name)}
