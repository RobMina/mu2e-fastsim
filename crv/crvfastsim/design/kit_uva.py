"""crv_design on a Slurm-over-SSH endpoint: one CPU job per design point, nothing streamed.

Code: a retained read-only snapshot of the repository on the endpoint (FASTSIMKIT_CRV_DESIGN_CODE at
FASTSIMKIT_CRV_DESIGN_BRANCH if set, else at FASTSIMKIT_CRV_DESIGN_CODE_REF, default main), run in the project image
(FASTSIMKIT_SIF) on the CPU partition. The muon sample is a file in the endpoint's crv_design data directory
(FASTSIMKIT_CRV_DESIGN_DATA, default <FASTSIMKIT_SCRATCH>/crv_design_data); its sha256 is pinned in the submission
manifest and checked by the job before use. The Run 2 lookup tables are fetched per job from the CVMFS Stratum-1 into
node tmpfs and deleted on exit (the kit's _TABLES block; out/cleanup.txt is required by `results`).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from mu2efastsim import site
from mu2efastsim.endpoints import EndpointError, Resources
from mu2efastsim.endpoints.artifacts import (binding, checked_record, fingerprint_command, input_digest,
                                           make_manifest, parse_fingerprint, resolve_image, runtime_prefix,
                                           verify_result)
from mu2efastsim.endpoints.snapshots import prepare_code

from . import kit_task

TASK = "crv_design"


def validate(params: Dict[str, Any]) -> Dict[str, Any]:
    """The point's params with defaults filled, as the kit's own check; raises ValueError."""
    from mu2efastsim.endpoints.parameters import validate as validate_parameters
    p = validate_parameters(TASK, params, kit_task.PARAMS, kit_task.DEFAULTS)
    if p["sample"] not in kit_task.SAMPLES:
        raise ValueError(f"crv_design: sample must be one of {sorted(kit_task.SAMPLES)}")
    if p["coincidence_layers"] != "" and p["n_layers"] != "" and p["coincidence_layers"] > p["n_layers"]:
        raise ValueError("crv_design: coincidence_layers must not exceed n_layers")
    if p["coincidence_layers"] != "" and p["spare_layers"] != "":
        raise ValueError("crv_design: give coincidence_layers or spare_layers, not both")
    if p["spare_layers"] != "" and p["n_layers"] != "" and p["spare_layers"] >= p["n_layers"]:
        raise ValueError("crv_design: spare_layers must be below n_layers")
    given = [k for k in kit_task.COST_WEIGHTS if p[k] != ""]
    if given and len(given) != len(kit_task.COST_WEIGHTS):
        raise ValueError(f"crv_design: set all of {tuple(kit_task.COST_WEIGHTS)} or none")
    if given and (any(float(p[k]) < 0 for k in given) or sum(float(p[k]) for k in given) <= 0):
        raise ValueError("crv_design: cost weights must be >= 0 with a positive sum")
    return p


class Delegate:
    """kit.DELEGATES entry: every endpoint that runs shell commands (the job needs the endpoint-resident sample)."""

    @staticmethod
    def applies(endpoint):
        return hasattr(endpoint, "_sh")

    @staticmethod
    def make(kit):
        return CrvDesignJob(kit)


class CrvDesignJob:
    def __init__(self, kit, code=None, data=None):
        self.kit, self.ep = kit, kit.ep
        self.code = code or site.setting("FASTSIMKIT_CRV_DESIGN_CODE") or site.scratch() + "/crv_design_code_kit"
        self.data = data or site.setting("FASTSIMKIT_CRV_DESIGN_DATA") or site.scratch() + "/crv_design_data"
        self.code_ref = os.environ.get("FASTSIMKIT_CRV_DESIGN_CODE_REF", "main")
        self.branch = os.environ.get("FASTSIMKIT_CRV_DESIGN_BRANCH", "")
        self.dir = Path(kit.out_root).parent / TASK
        self.dir.mkdir(parents=True, exist_ok=True)

    def _rec(self, name):
        return checked_record(self.dir / f"{name}.json")

    def sample_path(self, p):
        return f"{self.data.rstrip('/')}/{kit_task.SAMPLES[p['sample']]}"

    def inputs(self, p):
        """The pinned endpoint-resident input set (the muon sample): fingerprinted on the endpoint now."""
        path = self.sample_path(p)
        fp = parse_fingerprint(self.ep._sh(fingerprint_command(path), timeout=600))
        name = kit_task.SAMPLES[p["sample"]]
        return {"source": {"host": getattr(self.ep, "host", ""), "root": self.data},
                "files": [{"src": name, "dest": name, **fp, "argv": []}],
                "policy": "endpoint-resident; sha256 pinned at submit and checked by the job"}

    def render(self, p, manifest=None):
        from mu2efastsim.endpoints import kit as K
        res = Resources(**kit_task.RESOURCES)
        K.admit_resources(self.ep, res)
        code = shlex.quote(manifest["code"]["path"] if manifest else self.code)
        sha = manifest["inputs"]["files"][0]["sha256"] if manifest else "0" * 64
        env = "PYTHONPATH=$CODE/core:$CODE/crv,JAX_PLATFORMS=cpu,OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}"
        head = ('module load apptainer >/dev/null 2>&1\n'
                'M2FS_OUT=$(readlink -f "$M2FS_OUT")\n'
                f'CODE={code}; MUD=$(dirname "$(readlink -f @sample_path@)")\n'
                f'A() {{ apptainer exec --cleanenv --env "{env}" -B "$M2FS_OUT" -B "$CODE" --pwd "$CODE" @sif@ "$@"; }}\n'
                'echo "host $(hostname) job ${SLURM_JOB_ID:-none} cpus ${SLURM_CPUS_PER_TASK:-?} out $M2FS_OUT"\n')
        rt_t = (f'T() {{ apptainer exec --cleanenv --env "{env},CRVFS_LOOKUP_DIR=$L" -B "$L" -B "$M2FS_OUT" '
                '-B "$CODE" -B "$MUD" --pwd "$CODE" @sif@ "$@"; }')
        body = head + K._TABLES.replace("@runtime_t@", rt_t).replace("@tables@", " ".join(kit_task.TABLES)) \
            + kit_task.BODY
        subs = {"sif": shlex.quote(manifest["image"]["path"] if manifest else self.kit.sif),
                "sample_path": shlex.quote(self.sample_path(p)), "sample_sha256": sha,
                **{f"p_{k}": shlex.quote(K._arg(p[k])) for k in kit_task.PARAMS}}
        for a, b in subs.items():
            body = body.replace(f"@{a}@", b)
        if re.search(r"@[a-z_0-9]+@", body):
            raise ValueError("fastsimkit: crv_design: unfilled placeholder")
        if manifest is not None:
            body = runtime_prefix(manifest) + body
        return body, res

    # -- contract ------------------------------------------------------------------------------------------------
    def check(self, name, params) -> Dict[str, Any]:
        try:
            self.render(validate(params))
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def submit(self, name, params) -> Dict[str, Any]:
        p = validate(params)
        self.render(p)
        target = {"endpoint": self.ep.name, "host": getattr(self.ep, "host", ""), "data": self.data}
        target.update({k: getattr(self.ep, k) for k in ("account", "partition", "workdir") if hasattr(self.ep, k)})

        def prepare():
            inputs = self.inputs(p)
            code = prepare_code(self.ep, self.code, self.branch or self.code_ref)
            image = resolve_image(self.ep, self.kit.sif)
            manifest = make_manifest(TASK, code, image, inputs, p)
            body, res = self.render(p, manifest)
            return {"params": p, "submitted_at": time.time(), "code": code["path"], "code_ref": self.code_ref,
                    "branch": self.branch, "manifest": manifest, "body": body, "resources": asdict(res),
                    "target": target}

        try:
            rec = binding(self.dir, name, p, target, prepare)
            self.ep.submit(name, rec["body"], Resources(**rec["resources"]))
        except EndpointError as e:
            raise ValueError(str(e)) from e
        return {"handle": name}

    def status(self, name) -> Dict[str, Any]:
        from mu2efastsim.endpoints.kit import POLL_MS
        s = self.ep.status(name)
        msg = f"{s.site_state} {s.message}".strip()
        return {"state": s.state, "message": msg, "poll_ms": POLL_MS if s.state == "working" else 0, "progress": None}

    def results(self, name) -> Dict[str, Any]:
        s = self.ep.status(name)
        if s.state != "completed":
            raise ValueError(f"fastsimkit: {name!r} is {s.state}, not completed")
        files = self.ep.fetch(name, self.kit.out_root)
        out = Path(self.kit.out_root) / name / "out"
        if not (out / "cleanup.txt").exists():
            raise ValueError(f"fastsimkit: {name!r} staged Mu2e-internal tables but wrote no cleanup record")
        mpath = out / "metrics.json"
        if not mpath.exists():
            raise ValueError(f"fastsimkit: {name!r} wrote no out/metrics.json")
        rec = self._rec(name)
        if rec is None:
            raise ValueError(f"fastsimkit: {name!r} has no task record")
        verify_result(out, rec["manifest"])
        metrics = {k: float(v) for k, v in json.loads(mpath.read_text()).items()}
        return {"metrics": metrics,
                "files": [{"name": f.name, "uri": f.resolve().as_uri(), "kind": "text"} for f in files
                          if f.suffix in (".json", ".txt", ".out", ".log")],
                "metadata": {"endpoint": self.ep.name, "job_id": s.job_id, "site_state": s.site_state,
                             "artifact_manifest": rec["manifest"], "manifest_sha": rec["manifest"]["manifest_sha"],
                             "artifact_binding": "verified", "input_sha": input_digest(rec["manifest"]["inputs"]),
                             "code_ref": rec["code_ref"], "branch": rec.get("branch", "")}}

    def cancel(self, name) -> Dict[str, Any]:
        return {"state": self.ep.cancel(name)}
