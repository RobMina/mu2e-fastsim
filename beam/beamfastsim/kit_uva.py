"""beam_point (and beam_sob, beam_edep) on an endpoint without FNAL storage (UVA, Slurm over SSH): one job, inputs streamed in.

beam_point is one one-GPU job; beam_sob (BeamSobStream) one CPU job on the same route with the design studies' beam table;
beam_edep (BeamEdepStream) one GPU job with that table, the calorimeter shower library and the Run 1A GDML, returning
the EdepAna file.

The Mu2e-internal input (the DS-entrance muon table g4ref.npz, made from MuBeamCat) is never stored on the endpoint:
when the job starts it writes ``<job dir>/inbox`` (node, tmpfs dir) and the kit pipes the file from the FNAL host
through the kit host into that tmpfs (``ssh <fnal> cat`` -> ``srun --overlap ... cat >``), checks its md5 and touches
``ready``; the job's EXIT trap deletes the tmpfs dir and writes ``out/cleanup.txt``, without which ``results`` refuses.
An absolute FNAL geometry path (a rendered Mu2eBO overlay) is streamed the same way. The public inputs (DS field map,
Offline geometry tree) are fetched inside the job from the CVMFS Stratum-1 (``beamfastsim.stage``). Code comes from a
retained read-only snapshot of the umbrella on the endpoint (FASTSIMKIT_BEAM_CODE, FASTSIMKIT_CODE_REF) run in the image (FASTSIMKIT_SIF).
The kit host must reach both the endpoint and the FNAL host by ssh for the point's lifetime.

Environment (site.env): FASTSIMKIT_FNAL_HOST (required: the FNAL host the input is streamed from),
FASTSIMKIT_BEAM_DATA (FNAL dir holding g4ref_full/g4ref.npz; default <M2FS_INPUTS or M2FS_DATA>/beam_fastsim),
FASTSIMKIT_BEAM_CODE (default <FASTSIMKIT_SCRATCH>/beam_code_kit), FASTSIMKIT_CODE_REF (default main),
FASTSIMKIT_STREAM_POLL_S (seconds between queue checks until the inputs are streamed; default 10).
They are read when the source or the code snapshot is first needed; a missing one raises site.SiteConfigError.
"""
from __future__ import annotations

from mu2efastsim import site

import json
from dataclasses import asdict
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict

from mu2efastsim.endpoints import EndpointError, Resources
from mu2efastsim.endpoints.chainpoint import FnalSource
from mu2efastsim.endpoints.artifacts import (binding, checked_record, make_manifest, resolve_image,
                                           resolve_inputs, runtime_prefix, verify_cache, verify_result)
from mu2efastsim.endpoints.snapshots import prepare_code
from mu2efastsim.endpoints.submission import submission_lock
from mu2efastsim.endpoints import geometry as geometry_artifacts

from . import kit_task

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


def validate(params: Dict[str, Any], task: str = "beam_point") -> Dict[str, Any]:
    from mu2efastsim.endpoints.parameters import validate as validate_parameters
    if task in ("beam_edep", "beam_edep_cpu"):
        return validate_parameters(task, params, kit_task.EDEP_PARAMS, {"dtype": "float64"})
    if task == "beam_centuple":
        return validate_parameters(task, params, kit_task.CENT_PARAMS, {"dtype": "float64"})
    return validate_parameters(task, params, kit_task.SOB_PARAMS if task == "beam_sob" else kit_task.PARAMS)


class BeamPointStream:
    TASK = "beam_point"
    TABLE = G4REF
    RESOURCES = RESOURCES
    BODY = BODY

    def __init__(self, kit, source=None, code=None, background=True, poll_s=None):
        self.kit, self.ep = kit, kit.ep
        self._source, self._code = source, code
        self.code_ref = os.environ.get("FASTSIMKIT_CODE_REF", "main")
        self.background = background
        self.poll_s = float(os.environ.get("FASTSIMKIT_STREAM_POLL_S", 10.0)) if poll_s is None else poll_s
        self.dir = Path(kit.out_root).parent / self.TASK
        self.dir.mkdir(parents=True, exist_ok=True)
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    @property
    def source(self):
        if self._source is None:
            self._source = _Source(site.require("FASTSIMKIT_FNAL_HOST"),
                                   site.setting("FASTSIMKIT_BEAM_DATA") or site.data_root() + "/beam_fastsim")
        return self._source

    @source.setter
    def source(self, value):
        self._source = value

    @property
    def code(self):
        if self._code is None:
            self._code = site.setting("FASTSIMKIT_BEAM_CODE") or site.scratch() + "/beam_code_kit"
        return self._code

    @code.setter
    def code(self, value):
        self._code = value

    def _mark(self, name, what, text=""):
        (self.dir / f"{name}.{what}").write_text(text)

    def _has(self, name, what) -> bool:
        return (self.dir / f"{name}.{what}").exists()

    def _rec(self, name):
        p = self.dir / f"{name}.json"
        return checked_record(p, allow_legacy=True)

    @classmethod
    def inputs(cls, p) -> Dict[str, str]:
        """Files the kit streams: {FNAL rel or abs path: name in the job's tmpfs}."""
        out = {cls.TABLE: "g4ref.npz"}
        if p["geom"].startswith("/"):
            out[p["geom"]] = "geom_overlay.txt"
        return out

    def render(self, p, manifest=None, geometry=None):
        from mu2efastsim.endpoints.kit import _COMMON, APPTAINER_RUNTIME, admit_resources
        admit_resources(self.ep, self.RESOURCES)
        if not hasattr(self.ep, "stream_in"):
            raise ValueError(f"fastsimkit: {self.TASK} streams its input into the job and needs an endpoint that can "
                             f"(slurm_ssh); {getattr(self.ep, 'name', '?')!r} cannot")
        g = p["geom"]
        if geometry is not None:
            geometry_artifacts.validate_artifact(geometry)
            stagegeom = geomarg = '--geom "$M2FS_GEOMETRY"'
        elif g == "nominal":
            stagegeom = geomarg = ""
        elif g.startswith("/"):
            stagegeom = geomarg = '--geom "$L/geom_overlay.txt"'
        else:
            stagegeom = geomarg = f'--geom "$CODE/beam/geom/{g}"'
        body = self.BODY.replace("@common@", _COMMON).replace("@runtime_a@", APPTAINER_RUNTIME["a"])
        subs = {"sif": shlex.quote(manifest["image"]["path"] if manifest else self.kit.sif),
                "code": shlex.quote(manifest["code"]["path"] if manifest else self.code), "waits": str(WAIT_S),
                "stagegeom": stagegeom, "geomarg": geomarg, **{k: str(v) for k, v in p.items() if k != "geom"}}
        for a, b in subs.items():
            body = body.replace(f"@{a}@", b)
        import re
        if re.search(r"@[a-z_]+@", body):
            raise ValueError(f"fastsimkit: {self.TASK}: unfilled placeholder")
        if geometry is not None:
            body = geometry_artifacts.stage_shell(geometry) + body
        if manifest is not None:
            from mu2efastsim.endpoints.artifacts import consumption_guard
            body = body.replace('B python -m beamfastsim.stage',
                                consumption_guard('"$L"', manifest["inputs"]["files"]) + 'B python -m beamfastsim.stage', 1)
            body = runtime_prefix(manifest) + body
        return body, self.RESOURCES

    # -- contract ------------------------------------------------------------------------------------------------
    def check(self, name, params) -> Dict[str, Any]:
        try:
            self.render(validate(params, self.TASK))
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def check_binding(self, name, params, geometry=None):
        old = self._rec(name)
        if old is None:
            return None
        if "manifest" not in old:
            raise ValueError("fastsimkit: legacy task has no pinned artifact binding")
        p = validate(params, self.TASK)
        target = {"endpoint": self.ep.name, "host": getattr(self.ep, "host", ""),
                  "source_host": self.source.host, "source_root": self.source.root}
        target.update({k: getattr(self.ep, k) for k in ("account", "partition", "workdir") if hasattr(self.ep, k)})
        if p != old["params"] or target != old["target"]:
            raise ValueError("fastsimkit: different params or target/source from artifact binding")
        if geometry_artifacts.identity(geometry) != geometry_artifacts.identity(old["manifest"].get("geometry")):
            raise ValueError("fastsimkit: geometry differs from artifact binding")
        return old

    def ensure_code(self):
        return prepare_code(self.ep, self.code, self.code_ref)

    def submit(self, name, params, geometry=None) -> Dict[str, Any]:
        p = validate(params, self.TASK)
        if geometry is not None and p["geom"] != "nominal":
            raise ValueError("fastsimkit: geometry FileRef conflicts with geom selector")
        old = self.check_binding(name, p, geometry)
        if old is None:
            self.render(p, geometry=geometry)
        target = {"endpoint": self.ep.name, "host": getattr(self.ep, "host", ""),
                  "source_host": self.source.host, "source_root": self.source.root}
        target.update({k: getattr(self.ep, k) for k in ("account", "partition", "workdir") if hasattr(self.ep, k)})

        def prepare():
            inputs = resolve_inputs(self.source, self.inputs(p))
            code = self.ensure_code()
            image = resolve_image(self.ep, self.kit.sif)
            manifest = make_manifest(self.TASK, code, image, inputs, p, geometry=geometry)
            body, res = self.render(p, manifest, geometry=geometry)
            return {"params": p, "submitted_at": time.time(), "code": code["path"],
                    "code_ref": self.code_ref, "manifest": manifest,
                    "body": body, "resources": asdict(res), "target": target}

        try:
            rec = binding(self.dir, name, p, target, prepare, geometry=geometry)
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
            # Also serializes recovery callers that observed the inbox before the streamed marker.
            with submission_lock(self.dir / ".streams", name):
                have = self.ep.run_on_node(name, host, f"test -f {shlex.quote(tmp)}/ready && echo have || true").strip()
                if have == "have":
                    if not verify_cache(self.ep, name, host, tmp, entries):
                        raise ValueError("ready input cache does not match the bound manifest; refusing overwrite")
                    self._mark(name, "inputsreused", host)
                else:
                    for entry in entries:
                        got = self.ep.stream_in(name, host, f"{tmp}/{entry['dest']}", entry["argv"])
                        if got != entry["md5"]:
                            raise ValueError(f"md5 mismatch for {entry['src']}: {got} != {entry['md5']}")
                    if not verify_cache(self.ep, name, host, tmp, entries):
                        raise ValueError("streamed input cache does not match the bound manifest")
                    self.ep.run_on_node(name, host, f"touch {shlex.quote(tmp)}/ready")
        except subprocess.TimeoutExpired as e:
            # A stalled stream (ssh source or srun sink) raises from subprocess.run: without this the stream thread
            # dies unmarked and the job only fails after its input wait. Mark and cancel, like any other stream error.
            self._mark(name, "streamerr", f"input stream timed out after {e.timeout} s: {' '.join(map(str, e.cmd))[:120]}")
            self.ep.cancel(name)
            return "error"
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
        return self._assemble(name, s, self.ep.fetch(name, self.kit.out_root))

    def _assemble(self, name, s, files) -> Dict[str, Any]:
        """The result of a completed job whose out/ is fetched to <out_root>/<name>/out (files: the fetched files)."""
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
            metrics["input_wait_s"] = float(json.loads(w.read_text()).get("input_wait_s", 0))
        return {"metrics": metrics,
                "files": [{"name": p.name, "uri": p.resolve().as_uri(), "kind": "text"} for p in files
                          if p.suffix in (".json", ".txt", ".out", ".log")],
                "metadata": {"endpoint": self.ep.name, "job_id": s.job_id, "site_state": s.site_state,
                             "artifact_manifest": rec.get("manifest"), "manifest_sha": rec.get("manifest", {}).get("manifest_sha"),
                             "artifact_binding": "verified" if "manifest" in rec else "legacy-unverified",
                             "code_ref": rec["code_ref"]}}

    def cancel(self, name) -> Dict[str, Any]:
        return {"state": self.ep.cancel(name)}


SOB_RESOURCES = Resources(cores=32, mem_gb=96, gpus=0, gpu_type="", minutes=40)   # ~1.7 GB per CE process
SOB_BODY = """module load apptainer >/dev/null 2>&1
M2FS_OUT=$(readlink -f "$M2FS_OUT")   # the scratch area may be a symlink (UVA): bind the real path
echo "host $(hostname) job ${SLURM_JOB_ID:-none} cpus ${SLURM_CPUS_PER_TASK:-?} out $M2FS_OUT"
CODE=@code@
L=${TMPDIR:-/tmp}/beamfs_kit_j${SLURM_JOB_ID:-0}_$$; mkdir -p "$L"; chmod 700 "$L"
trap 'rm -rf "$L"; echo "$(date -Is) $(hostname) deleted $L" >> "$M2FS_OUT/cleanup.txt"' EXIT
trap 'exit 143' TERM INT
B() { apptainer exec --cleanenv --env "JAX_PLATFORMS=cpu,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$CODE/core:$CODE/beam,BEAMFS_OFFLINE_ROOT=$L/pub/offline,BEAMFS_MAPS=$L/pub/maps" -B "$L" -B "$M2FS_OUT" -B "$CODE" --pwd "$CODE" @sif@ "$@"; }
B python -c "import beamfastsim.stage, beamfastsim.cesob" || { echo "beamfastsim.cesob missing in $CODE"; exit 5; }
echo "$(hostname) $L" > "$(dirname "$M2FS_OUT")/inbox"      # read by the kit, which then streams the inputs in
tw=$SECONDS; while [ $((SECONDS-tw)) -lt @waits@ ]; do [ -f "$L/ready" ] && break; sleep 3; done
[ -f "$L/ready" ] || { echo "no input arrived within @waits@ s"; exit 3; }
echo "{\\"input_wait_s\\": $((SECONDS-tw))}" > "$M2FS_OUT/wait.json"
B python -m beamfastsim.stage "$L/pub" @stagegeom@ --extra-map TSdMap --file @dio@ > "$M2FS_OUT/stage.txt" 2>&1 || { tail -5 "$M2FS_OUT/stage.txt"; exit 4; }
tail -1 "$M2FS_OUT/stage.txt"
B python -m beamfastsim.cesob point --g4 "$L/g4ref.npz" --z-up @zup@ @geomarg@ --muons @muons@ --electrons @electrons@ \\
  --seed @seed@ --procs ${SLURM_CPUS_PER_TASK:-@cores@} --out "$M2FS_OUT" > "$M2FS_OUT/point.txt" 2>&1 || { tail -20 "$M2FS_OUT/point.txt"; exit 6; }
tail -c 300 "$M2FS_OUT/point.txt"
"""


class BeamSobStream(BeamPointStream):
    """beam_sob (beamfastsim.cesob point) on the streaming route: the design studies' beam table streamed in, one CPU job
    (stops on JAX-CPU, then the CE chain in one process per core)."""
    TASK = "beam_sob"
    TABLE = kit_task.SOB_TABLE
    RESOURCES = SOB_RESOURCES

    def __init__(self, kit, **kw):
        from . import cesob
        super().__init__(kit, **kw)
        self.BODY = SOB_BODY.replace("@zup@", str(kit_task.SOB_Z_UP)).replace(
            "@cores@", str(SOB_RESOURCES.cores)).replace("@dio@", cesob.DIO_TABLE)


EDEP_RESOURCES = Resources(cores=kit_task.EDEP_CORES, mem_gb=96, gpus=1, gpu_type="a100", minutes=60)
EDEP_BODY = """@common@CODE=@code@
L=${TMPDIR:-/tmp}/beamfs_kit_j${SLURM_JOB_ID:-0}_$$; mkdir -p "$L"; chmod 700 "$L"
trap 'rm -rf "$L"; echo "$(date -Is) $(hostname) deleted $L" >> "$M2FS_OUT/cleanup.txt"' EXIT
trap 'exit 143' TERM INT
B() { apptainer exec --nv --cleanenv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$CODE/core:$CODE/beam:$CODE/calo:$CODE/tracker,XLA_PYTHON_CLIENT_PREALLOCATE=false,BEAMFS_OFFLINE_ROOT=$L/pub/offline,BEAMFS_MAPS=$L/pub/maps" -B "$L" -B "$M2FS_OUT" -B "$CODE" --pwd "$CODE" @sif@ "$@"; }
B python -c "import beamfastsim.stage, beamfastsim.edepana" || { echo "beamfastsim.edepana missing in $CODE"; exit 5; }
echo "$(hostname) $L" > "$(dirname "$M2FS_OUT")/inbox"      # read by the kit, which then streams the inputs in
tw=$SECONDS; while [ $((SECONDS-tw)) -lt @waits@ ]; do [ -f "$L/ready" ] && break; sleep 3; done
[ -f "$L/ready" ] || { echo "no input arrived within @waits@ s"; exit 3; }
echo "{\\"input_wait_s\\": $((SECONDS-tw))}" > "$M2FS_OUT/wait.json"
B python -m beamfastsim.stage "$L/pub" @stagegeom@ --extra-map TSdMap > "$M2FS_OUT/stage.txt" 2>&1 || { tail -5 "$M2FS_OUT/stage.txt"; exit 4; }
tail -1 "$M2FS_OUT/stage.txt"
B python -m beamfastsim.edepana point --g4 "$L/g4ref.npz" --z-up @zup@ @geomarg@ --muons @muons@ --electrons @electrons@ \\
  --seed @seed@ --dtype @dtype@ --library "$L/shower_library.npz" --edge-library "$L/edge_library.npz" --gdml "$L/tracker.gdml" \\
  --procs $(( ${SLURM_CPUS_PER_TASK:-@cores@} - 1 )) --out "$M2FS_OUT" > "$M2FS_OUT/point.txt" 2>&1 || { tail -20 "$M2FS_OUT/point.txt"; exit 6; }
tail -c 300 "$M2FS_OUT/point.txt"
"""


class BeamEdepStream(BeamPointStream):
    """beam_edep (beamfastsim.edepana point) on the streaming route: the design studies' beam table, the calorimeter
    shower library, its inner-edge energy table and the Run 1A GDML streamed in, one GPU job. Its EdepAna file comes back with the results (kind
    "root"), with the edep run record and summary beside it."""
    TASK = "beam_edep"
    TABLE = kit_task.SOB_TABLE
    RESOURCES = EDEP_RESOURCES

    def __init__(self, kit, **kw):
        super().__init__(kit, **kw)
        self.BODY = EDEP_BODY.replace("@zup@", str(kit_task.SOB_Z_UP)).replace("@cores@", str(EDEP_RESOURCES.cores))

    @classmethod
    def inputs(cls, p) -> Dict[str, str]:
        out = super().inputs(p)
        out.update({kit_task.EDEP_LIBRARY: "shower_library.npz", kit_task.EDEP_GDML: "tracker.gdml",
                    kit_task.EDEP_EDGE: "edge_library.npz"})
        return out

    def results(self, name) -> Dict[str, Any]:
        """The EdepAna file as the only FileRef (kind "root"): a chained analysis step takes its upstream files as
        inputs and approx_ce_sensitivity takes one ROOT file. Its run record and summary stay beside it in the
        fetched out/ and are named in metadata."""
        r = super().results(name)
        out = Path(self.kit.out_root) / name / "out"
        nts = out / kit_task.EDEP_FILE
        if not nts.exists():
            raise ValueError(f"fastsimkit: {name!r} wrote no {kit_task.EDEP_FILE}")
        r["metadata"]["text_files"] = [f["name"] for f in r["files"]]
        r["metadata"]["edep_run_record"] = (nts.parent / (nts.name + ".edep.json")).name
        r["files"] = [{"name": nts.name, "uri": nts.resolve().as_uri(), "kind": "root"}]
        return r


EDEP_CPU_RESOURCES = Resources(cores=1, mem_gb=kit_task.EDEP_CPU_MEM_GB, gpus=0, gpu_type="",
                               minutes=kit_task.EDEP_CPU_MINUTES)
# the CPU preamble of beam_sob (no GPU helper, no nvidia-smi, no GPU check), then beam_edep's steps on the CPU
EDEP_CPU_BODY = (SOB_BODY[:SOB_BODY.index("B() {")] + EDEP_BODY[EDEP_BODY.index("B() {"):]).replace(
    'apptainer exec --nv --cleanenv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},OPENBLAS_NUM_THREADS=1,',
    'apptainer exec --cleanenv --env "JAX_PLATFORMS=cpu,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,'
    'XLA_FLAGS=--xla_cpu_multi_thread_eigen=false,').replace(
    "XLA_PYTHON_CLIENT_PREALLOCATE=false,", "").replace(
    "--procs $(( ${SLURM_CPUS_PER_TASK:-@cores@} - 1 ))", "--procs 1")


class BeamEdepCpuStream(BeamEdepStream):
    """beam_edep_cpu: beam_edep's streamed inputs and point on one CPU core (the CPU partition), JAX single-threaded,
    float64 by default."""
    TASK = "beam_edep_cpu"
    RESOURCES = EDEP_CPU_RESOURCES

    def __init__(self, kit, **kw):
        super().__init__(kit, **kw)
        self.BODY = EDEP_CPU_BODY.replace("@zup@", str(kit_task.SOB_Z_UP)).replace(
            "@cores@", str(EDEP_CPU_RESOURCES.cores))


# beam_centuple on the streaming route: two Slurm jobs per point. The GPU job (in the image) streams the beam table and
# the centuple inputs into tmpfs and runs `centuple point --writer none` (stops, CE births, record, trigger, assemble);
# the kit then submits the write stage as a one-core CPU job, `<name>.w`, which runs `centuple write` in the stock
# release's Mu2e environment (mu2e_env.sh -> container/offline/offline_uva.sh: cvmfsexec + the FNAL worker image, so
# the node needs no /cvmfs) and `centuple finish` (metrics.json). The write stage reads the donor EventNtuple files in
# place from FASTSIMKIT_CENT_DONORS on the endpoint (an exception to the stream-and-delete rule: the donor set is ~2 GB
# and lives on the endpoint) and the GPU job's plan from its out/ directory.
CENT_RESOURCES = Resources(cores=kit_task.CENT_CORES, mem_gb=64, gpus=1, gpu_type="a100", minutes=60)
CENT_WRITER_RESOURCES = Resources(cores=1, mem_gb=16, gpus=0, gpu_type="", minutes=150)
CENT_CALO_FILES = ("Sim_crystal.txt", "CsI-waveform-2020-02-12.root", "ReadoutNoise.root", "CsI-propag-2020-07-29.root",
                   "calo_geometry_run1430.npz")
# streamed name (flat in tmpfs) -> path under the point's data directory (beamfastsim.centuple.__main__.DATA)
CENT_LAYOUT = {"shower_library.npz": "shower_library.npz", "edge_library.npz": "edge_library.npz",
               "tracker.gdml": "tracker.gdml", "digicond_onspill.npz": "digicond_onspill.npz",
               "trigger__replay.fcl": "trigger/replay.fcl", "trigger__trgdump.root": "trigger/trgdump.root",
               "trigger__TimeCluster.weights.xml": "trigger/TimeCluster.weights.xml",
               "donor__donor.npz": "donor/donor.npz",
               **{f"calo_response__{f}": f"calo_response/{f}" for f in CENT_CALO_FILES}}
CENT_BODY = EDEP_BODY.replace('python -c "import beamfastsim.stage, beamfastsim.edepana" || { echo "beamfastsim.edepana missing',
                              'python -c "import beamfastsim.stage, beamfastsim.centuple" || { echo "beamfastsim.centuple missing')
CENT_BODY = CENT_BODY.replace("PYTHONPATH=$CODE/core:$CODE/beam:$CODE/calo:$CODE/tracker,",
                              "PYTHONPATH=$CODE/core:$CODE/beam:$CODE/calo:$CODE/tracker:$CODE,")
CENT_BODY = CENT_BODY[:CENT_BODY.index("B python -m beamfastsim.edepana point")] + """mkdir -p "$L/data/trigger" "$L/data/donor" "$L/data/calo_response"
@layout@
B python -m beamfastsim.centuple point --g4 "$L/g4ref.npz" --z-up @zup@ @geomarg@ --data "$L/data" --muons @muons@ \\
  --electrons @electrons@ --seed @seed@ --dtype @dtype@ --writer none --out "$M2FS_OUT" > "$M2FS_OUT/point.txt" 2>&1 || { tail -20 "$M2FS_OUT/point.txt"; exit 6; }
tail -c 300 "$M2FS_OUT/point.txt"
"""
CENT_WRITER_BODY = """module load apptainer >/dev/null 2>&1
M2FS_OUT=$(readlink -f "$M2FS_OUT")
CODE=@code@
G=$(readlink -f "$(dirname "$(dirname "$M2FS_OUT")")/@front@/out")
DON=$(readlink -f @donors@)
echo "host $(hostname) job ${SLURM_JOB_ID:-none} cpus ${SLURM_CPUS_PER_TASK:-?} out $M2FS_OUT front $G"
[ -f "$G/centuple_point.json" ] || { echo "no centuple_point.json in $G"; exit 3; }
[ -d "$DON" ] || { echo "no donor directory $DON"; exit 3; }
# plan, generated count and the donor files (names resolved under $DON) from the front stage's record
ARGS=$(python3 - "$G/centuple_point.json" "$DON" <<'PY'
import json, os, sys
rec, root = json.load(open(sys.argv[1])), sys.argv[2]
found = {}
for d, _, fs in os.walk(root, followlinks=True):
    for f in fs:
        found.setdefault(f, os.path.join(d, f))
missing = [f for f in rec["donor_files"] if f not in found]
if missing:
    sys.exit("missing donor files under %s: %s" % (root, missing[:5]))
print(rec["plan"], rec["n_generated"], " ".join(os.path.realpath(found[f]) for f in rec["donor_files"]))
PY
) || exit 3
read -r PLAN NGEN FILES <<< "$ARGS"
cd "$M2FS_OUT"
tw=$SECONDS
CVMFSEXEC_DIST=@cvmfsexec@ OFFLINE_BINDS="$G,$DON" OFFLINE_LOG_PREFIX=centuple_write \\
CENTUPLE_PYTHONPATH="$CODE/core:$CODE/beam:$CODE/calo:$CODE/tracker:$CODE" \\
  bash "$CODE/beam/beamfastsim/centuple/mu2e_env.sh" python3 -m beamfastsim.centuple write --plan "$G/$PLAN" \\
  --donor-files $FILES --out "$M2FS_OUT/@file@" --n-generated "$NGEN" > "$M2FS_OUT/write.txt" 2>&1 || { tail -20 "$M2FS_OUT/write.txt"; exit 6; }
WS=$((SECONDS-tw))
apptainer exec --cleanenv --env "JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$CODE/core:$CODE/beam" \\
  -B "$G" -B "$M2FS_OUT" -B "$CODE" --pwd "$CODE" @sif@ python -m beamfastsim.centuple finish \\
  --point "$G/centuple_point.json" --file "$M2FS_OUT/@file@" --out "$M2FS_OUT" --writer-s "$WS" \\
  > "$M2FS_OUT/finish.txt" 2>&1 || { tail -20 "$M2FS_OUT/finish.txt"; exit 7; }
tail -c 300 "$M2FS_OUT/finish.txt"
"""


class BeamCentupleStream(BeamPointStream):
    """beam_centuple (beamfastsim.centuple point) on the streaming route: a GPU job for the front stages (inputs streamed
    in, deleted at exit), then a one-core CPU job `<name>.w` for the EventNtuple writer in the Mu2e environment through
    cvmfsexec. Its EventNtuple file comes back with the results (kind "root").

    Site settings: FASTSIMKIT_CENT_DONORS (endpoint directory holding the donor EventNtuple files named in the donor
    library, searched recursively; required), CVMFSEXEC_DIST (endpoint path of a prepared cvmfsexec tree; required)."""
    TASK = "beam_centuple"
    TABLE = kit_task.SOB_TABLE
    RESOURCES = CENT_RESOURCES
    WRITER = ".w"

    def __init__(self, kit, donors=None, cvmfsexec=None, **kw):
        super().__init__(kit, **kw)
        layout = "\n".join(f'ln -s "$L/{k}" "$L/data/{v}"' for k, v in CENT_LAYOUT.items())
        self.BODY = CENT_BODY.replace("@zup@", str(kit_task.SOB_Z_UP)).replace(
            "@cores@", str(CENT_RESOURCES.cores)).replace("@layout@", layout)
        self._donors, self._cvmfsexec = donors, cvmfsexec

    @property
    def donors(self):
        return self._donors or site.require("FASTSIMKIT_CENT_DONORS")

    @property
    def cvmfsexec(self):
        return self._cvmfsexec or site.require("CVMFSEXEC_DIST")

    @classmethod
    def inputs(cls, p) -> Dict[str, str]:
        out = super().inputs(p)
        out.update({f"{kit_task.CENT_DIR}/{v}": k for k, v in CENT_LAYOUT.items()})
        return out

    def writer_body(self, name) -> str:
        rec = self._rec(name)
        man = rec.get("manifest") if rec else None
        subs = {"code": shlex.quote(man["code"]["path"] if man else self.code),
                "sif": shlex.quote(man["image"]["path"] if man else self.kit.sif),
                "front": shlex.quote(name), "donors": shlex.quote(self.donors),
                "cvmfsexec": shlex.quote(self.cvmfsexec), "file": kit_task.CENT_FILE}
        body = CENT_WRITER_BODY
        for a, b in subs.items():
            body = body.replace(f"@{a}@", b)
        return body

    def prestaged(self) -> bool:
        """True when CVMFSEXEC_DIST on the endpoint is a prestaged runtime (its PRESTAGED marker exists): the write
        stages of parallel points must not each fill a cold CVMFS cache over the network (skill
        prestaging-cvmfs-on-uva); the tree's config reads a read-only prestaged cache below each job's own."""
        marker = shlex.quote(self.cvmfsexec.rstrip("/") + "/PRESTAGED")
        return self.ep._sh(f"test -f {marker} && echo yes || true").strip() == "yes"

    def ensure_writer(self, name) -> bool:
        """Submit the write stage once the front job has completed and the CVMFS runtime is prestaged (idempotent,
        under the stream lock). False while it waits for the prestaged runtime."""
        with submission_lock(self.dir / ".streams", name + self.WRITER):
            if self._has(name, "writer"):
                return True
            if not self.prestaged():
                return False
            body = self.writer_body(name)
            try:
                job = self.ep.submit(name + self.WRITER, body, CENT_WRITER_RESOURCES)
            except EndpointError as e:
                self._mark(name, "writererr", str(e))
                raise ValueError(str(e)) from e
            self._mark(name, "writer", str(getattr(job, "job_id", "")))
            return True

    def check(self, name, params) -> Dict[str, Any]:
        r = super().check(name, params)
        if r["ok"]:
            try:
                self.donors, self.cvmfsexec   # noqa: B018 (site settings the write stage needs)
            except site.SiteConfigError as e:
                return {"ok": False, "message": str(e)}
        return r

    def status(self, name) -> Dict[str, Any]:
        from mu2efastsim.endpoints.kit import POLL_MS
        front = super().status(name)
        if front["state"] != "completed":
            if front["state"] == "working":
                front["message"] = "front: " + front["message"]
            return front
        if self._has(name, "writererr"):
            return {"state": "failed", "message": "writer submit failed: " + (self.dir / f"{name}.writererr").read_text()[:200],
                    "poll_ms": 0, "progress": None}
        try:
            submitted = self.ensure_writer(name)
        except ValueError as e:
            return {"state": "failed", "message": f"writer submit failed: {e}", "poll_ms": 0, "progress": None}
        if not submitted:
            return {"state": "working", "message": f"front done; writer waits for the prestaged CVMFS runtime "
                    f"({self.cvmfsexec}/PRESTAGED)", "poll_ms": POLL_MS, "progress": None}
        w = self.ep.status(name + self.WRITER)
        return {"state": w.state, "message": f"writer: {w.site_state} {w.message}".strip(),
                "poll_ms": POLL_MS if w.state == "working" else 0, "progress": None}

    def results(self, name) -> Dict[str, Any]:
        """The EventNtuple file as a FileRef (kind "root"); metrics from the write stage's `finish`. The writer's out/
        is fetched next to the front job's and its files are copied into the front job's out/ (whose manifest and
        geometry evidence the kit verifies)."""
        s = self.ep.status(name)
        if s.state != "completed":
            raise ValueError(f"fastsimkit: {name!r} is {s.state}, not completed")
        w = self.ep.status(name + self.WRITER)
        if w.state != "completed":
            raise ValueError(f"fastsimkit: {name!r} write stage is {w.state}, not completed")
        files = self.ep.fetch(name, self.kit.out_root)          # the front first: its out/ has the beam metrics.json
        wfiles = self.ep.fetch(name + self.WRITER, self.kit.out_root)
        out = Path(self.kit.out_root) / name / "out"
        wout = Path(self.kit.out_root) / (name + self.WRITER) / "out"
        import shutil
        for f in wfiles:                                          # then the writer's (EventNtuple, metrics.json) over it
            if f.parent == wout:
                shutil.copy2(f, out / f.name)
        r = self._assemble(name, s, sorted(set(files) | {out / f.name for f in wfiles if f.parent == wout}))
        nts = out / kit_task.CENT_FILE
        if not nts.exists():
            raise ValueError(f"fastsimkit: {name!r} wrote no {kit_task.CENT_FILE}")
        r["metadata"]["text_files"] = [f["name"] for f in r["files"]]
        r["metadata"]["writer_job_id"] = w.job_id
        r["files"] = [{"name": nts.name, "uri": nts.resolve().as_uri(), "kind": "root"}]
        return r

    def cancel(self, name) -> Dict[str, Any]:
        state = self.ep.cancel(name)
        if self._has(name, "writer"):
            self.ep.cancel(name + self.WRITER)
        return {"state": state}
