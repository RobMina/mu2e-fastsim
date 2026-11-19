"""beam_edep_cpu on the FNAL grid (FermiGrid, jobsub_lite): one single-core grid job per design point, inputs staged.

The route of the CPU fast tier for grid slots. At submit the kit, through the jobsub endpoint's submit host
(mu2efastsim.endpoints.jobsub.FermiGridEndpoint):
  1. pins the Mu2e-internal inputs on FNAL storage (the design studies' beam table, the calorimeter shower library, its
     inner-edge table and the Run 1A GDML; the same files as the streaming route, kit_uva.BeamEdepStream.inputs) by
     sha256/md5/size, and stages each to dCache scratch with ``ifdh cp`` (content-addressed, verified, reused);
  2. makes a retained snapshot of the umbrella at FASTSIMKIT_CODE_REF on the submit host, packs it into a
     deterministic tarball (no .git) and stages that;
  3. stages the CPU image (FASTSIMKIT_JOBSUB_SIF, a file on the submit host; default the kit's image);
  4. binds code, image and inputs in the artifact manifest and submits one job.
The worker copies the three in with ``ifdh cp``, checks every digest against the manifest, runs the CPU image with the
cvmfs apptainer nested in the jobsub container (FermiGrid workers have no /dev/fuse, so when the SIF cannot be mounted
it is unpacked once to a sandbox directory, about 20 s), and runs the same ``beamfastsim.edepana point`` command line
as the streaming route on one core. The public inputs (field maps, Offline geometry) are read from the worker's /cvmfs
(bound into the container); if that is not visible they are fetched from a Stratum-1 as on UVA. On exit the inputs,
image and code are deleted from the worker and the deletion recorded in out/cleanup.txt (``results`` refuses
without it); the jobsub endpoint then copies out/ to dCache, and ``results`` fetches it like the streaming route's.

Settings (site.env): the jobsub endpoint's (FASTSIMKIT_JOBSUB_*), FASTSIMKIT_FNAL_HOST and FASTSIMKIT_BEAM_DATA (the
input root, read on the submit host), FASTSIMKIT_CODE_REF, FASTSIMKIT_JOBSUB_SIF (submit-host path of the CPU image),
FASTSIMKIT_JOBSUB_CODE (submit-host snapshot base; default <workdir>/beam_code_kit), FASTSIMKIT_JOBSUB_APPTAINER
(worker apptainer; default the OSG cvmfs build).
"""
from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from mu2efastsim import site
from mu2efastsim.endpoints import EndpointError, Resources
from mu2efastsim.endpoints import geometry as geometry_artifacts
from mu2efastsim.endpoints.artifacts import (binding, consumption_guard, make_manifest, parse_fingerprint,
                                           fingerprint_command, resolve_inputs, validate_manifest)
from mu2efastsim.endpoints.snapshots import prepare_code

from . import kit_task
from .kit_uva import EDEP_CPU_BODY, BeamEdepCpuStream, _head, validate

APPTAINER = "/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer"
CVMFS_PROBE = "/cvmfs/mu2e.opensciencegrid.org/DataFiles/BFieldMaps/Mau13/DSMap.header"
_RUN_MARK = "B python -m beamfastsim.edepana point"
# The point's command line, exactly as on the streaming route (kit_uva.EDEP_CPU_BODY): same flags, one process.
RUN = EDEP_CPU_BODY[EDEP_CPU_BODY.index(_RUN_MARK):] if _RUN_MARK in EDEP_CPU_BODY else _head("", _RUN_MARK)

# The caller's script inside the jobsub job (hependpoints.jobsub.JobsubEndpoint.body): $M2FS_JOB_DIR is node-local
# scratch, $M2FS_OUT is copied to dCache on any exit by the endpoint's EXIT trap (hepep_finish), which the cleanup
# trap below chains to (rc kept), so the deletion is recorded before the copy-out.
BODY = """W="$M2FS_JOB_DIR"; L="$W/in"; CODE="$W/code"
mkdir -p "$L" "$CODE" && chmod 700 "$L" || exit 3
M2FS_OUT=$(readlink -f "$M2FS_OUT")
APPT=@apptainer@
m2fs_cleanup() { rm -rf "$L" "$CODE" "$W/img" "$W/img.sif" "$W/code.tgz" "$W/apptmp"; echo "$(date -u +%FT%TZ) $(hostname) deleted $L" >> "$M2FS_OUT/cleanup.txt"; }
trap 'm2fs_rc=$?; m2fs_cleanup; (exit $m2fs_rc); hepep_finish' EXIT
echo "host $(hostname) job ${JOBSUBJOBID:-none} cores_allowed $(nproc) out $M2FS_OUT"
grep -m1 "model name" /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//' > "$M2FS_OUT/cpu.txt" 2>/dev/null || true
[ -x "$APPT" ] || { echo "no apptainer at $APPT"; exit 2; }
t0=$SECONDS
ifdh cp @image_pnfs@ "$W/img.sif" >/dev/null || { echo "image copy failed"; exit 3; }
ifdh cp @code_pnfs@ "$W/code.tgz" >/dev/null || { echo "code copy failed"; exit 3; }
@copy_inputs@printf '%s  %s\\n' @image_sha@ "$W/img.sif" @code_sha@ "$W/code.tgz" | sha256sum -c - >/dev/null || { echo "image or code differs from the bound manifest"; exit 5; }
@input_guard@printf '%s\\n' @manifest@ > "$M2FS_OUT/artifact-manifest.json" || exit 5
tar -xzf "$W/code.tgz" -C "$CODE" && rm -f "$W/code.tgz" || { echo "code unpack failed"; exit 3; }
CHECK="import jax; print('nested apptainer ok', jax.__version__, jax.devices())"
IMG="$W/img.sif"; MODE=sif
if ! "$APPT" exec --cleanenv "$IMG" python3 -c "$CHECK"; then
  echo "SIF mount failed (no /dev/fuse on this worker?): building a sandbox"; t1=$SECONDS
  export APPTAINER_TMPDIR="$W/apptmp"; mkdir -p "$APPTAINER_TMPDIR"
  "$APPT" build --sandbox "$W/img" "$IMG" > "$W/sandbox.log" 2>&1 || { tail -5 "$W/sandbox.log"; echo "sandbox build failed"; exit 4; }
  rm -f "$IMG"; rm -rf "$APPTAINER_TMPDIR"; IMG="$W/img"; MODE=sandbox
  echo "sandbox $((SECONDS-t1)) s"
  "$APPT" exec --cleanenv "$IMG" python3 -c "$CHECK" || { echo "nested apptainer failed (sandbox)"; exit 4; }
fi
echo "{\\"stage_in_s\\": $((SECONDS-t0)), \\"image_mode\\": \\"$MODE\\"}" > "$M2FS_OUT/grid.json"
BENV="PYTHONDONTWRITEBYTECODE=1"; RUNPFX=()
B() { "${RUNPFX[@]}" "$APPT" exec --cleanenv --env "JAX_PLATFORMS=cpu,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,XLA_FLAGS=--xla_cpu_multi_thread_eigen=false,PYTHONPATH=$CODE/core:$CODE/beam:$CODE/calo:$CODE/tracker,$BENV" -B /cvmfs -B "$L" -B "$M2FS_OUT" -B "$CODE" --pwd "$CODE" "$IMG" "$@"; }
B python -c "import beamfastsim.stage, beamfastsim.edepana" || { echo "beamfastsim.edepana missing in $CODE"; exit 5; }
if B test -r @cvmfs_probe@; then
  echo "public inputs from /cvmfs" > "$M2FS_OUT/stage.txt"
else
  B python -m beamfastsim.stage "$L/pub" @stagegeom@ --extra-map TSdMap > "$M2FS_OUT/stage.txt" 2>&1 || { tail -5 "$M2FS_OUT/stage.txt"; exit 4; }
  BENV="$BENV,BEAMFS_OFFLINE_ROOT=$L/pub/offline,BEAMFS_MAPS=$L/pub/maps"
fi
tail -1 "$M2FS_OUT/stage.txt"
[ -x /usr/bin/time ] && RUNPFX=(/usr/bin/time -v -o "$M2FS_OUT/time.txt")
"""


class BeamEdepCpuGrid(BeamEdepCpuStream):
    """beam_edep_cpu on a jobsub endpoint (FermiGrid): inputs, code tarball and image staged to dCache scratch, one
    single-core job; results, cleanup gate and manifest check as on the streaming route."""
    TASK = "beam_edep_cpu"
    BODY = BODY + RUN

    def __init__(self, kit, source=None, code=None, sif=None, apptainer=None, **kw):
        kw.pop("background", None)
        super().__init__(kit, source=source, code=code, background=False, **kw)
        self.BODY = (BODY + RUN).replace("@zup@", str(kit_task.SOB_Z_UP)).replace("@cores@", "1")
        self._sif, self._apptainer = sif, apptainer

    @property
    def code(self):
        if self._code is None:
            self._code = site.setting("FASTSIMKIT_JOBSUB_CODE") or self.ep.workdir + "/beam_code_kit"
        return self._code

    @code.setter
    def code(self, value):
        self._code = value

    @property
    def sif(self) -> str:
        return self._sif or site.setting("FASTSIMKIT_JOBSUB_SIF") or self.kit.sif

    @property
    def apptainer(self) -> str:
        return self._apptainer or site.setting("FASTSIMKIT_JOBSUB_APPTAINER") or APPTAINER

    def resources(self, p) -> Resources:
        return kit_task.edep_cpu_grid_resources(p, Resources)

    def target(self) -> Dict[str, Any]:
        ep = self.ep
        return {"endpoint": ep.name, "host": ep.host, "group": ep.group, "role": ep.role, "workdir": ep.workdir,
                "output_dir": ep.output_dir, "stage_dir": ep.stage_dir,
                "source_host": self.source.host, "source_root": self.source.root}

    # -- rendering ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _geomargs(p, geometry):
        g = p["geom"]
        if geometry is not None:
            geometry_artifacts.validate_artifact(geometry)
            return '--geom "$M2FS_GEOMETRY"'
        if g == "nominal":
            return ""
        if g.startswith("/"):
            return '--geom "$L/geom_overlay.txt"'
        return f'--geom "$CODE/beam/geom/{g}"'

    def render(self, p, record=None, geometry=None):
        """The job script for validated params, bound to the staged artifacts of `record` (its manifest and dCache
        paths); with record=None only the admission checks run (check) and no body is returned."""
        from mu2efastsim.endpoints.kit import admit_resources
        res = self.resources(p)
        admit_resources(self.ep, res)
        if any(p.get(k, 1.0) != 1.0 for k in kit_task.SCALE_DEFAULTS):
            raise ValueError(f"fastsimkit: {self.TASK}: field-map scales run on the streaming route (UVA) only; "
                             "the grid route renders none yet")
        if not hasattr(self.ep, "stage_file"):
            raise ValueError(f"fastsimkit: {self.TASK} on the grid needs a jobsub endpoint with a dCache stage area")
        geomarg = self._geomargs(p, geometry)
        if record is None:
            return None, res
        manifest, staged = record["manifest"], record["staged"]
        validate_manifest(manifest)
        entries = manifest["inputs"]["files"]
        copies = "".join(f'ifdh cp {shlex.quote(staged["inputs"][e["dest"]])} "$L/{e["dest"]}" >/dev/null || '
                         f'{{ echo "input copy failed: {e["dest"]}"; exit 3; }}\n' for e in entries)
        subs = {"apptainer": shlex.quote(self.apptainer), "image_pnfs": shlex.quote(staged["image"]),
                "code_pnfs": shlex.quote(staged["code"]), "image_sha": manifest["image"]["sha256"],
                "code_sha": manifest["code"]["archive"]["sha256"], "copy_inputs": copies,
                "input_guard": consumption_guard('"$L"', entries),
                "manifest": shlex.quote(json.dumps(manifest, sort_keys=True, allow_nan=False)),
                "cvmfs_probe": CVMFS_PROBE, "stagegeom": geomarg, "geomarg": geomarg,
                **{k: str(v) for k, v in p.items() if k != "geom" and k not in kit_task.SCALE_DEFAULTS}}
        body = self.BODY
        for a, b in subs.items():
            body = body.replace(f"@{a}@", b)
        if re.search(r"@[a-z_]+@", body):
            raise ValueError(f"fastsimkit: {self.TASK}: unfilled placeholder in the grid body")
        if geometry is not None:
            body = geometry_artifacts.stage_shell(geometry) + body
        return body, res

    # -- staging -----------------------------------------------------------------------------------------------------
    def ensure_code(self):
        """A retained snapshot at the code ref on the submit host, packed into a deterministic tarball there; returns
        the snapshot record with its archive (sha256, size, submit-host path)."""
        snap = prepare_code(self.ep, self.code, self.code_ref)
        q = shlex.quote(snap["path"])
        tgz = f"{self.code}.archives/{snap['umbrella_commit']}.tgz"
        out = self.ep._sh(
            f"set -e -o pipefail; mkdir -p {shlex.quote(self.code + '.archives')}; "
            f"tar -C {q} --exclude=.git --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - . "
            f"| gzip -n > {shlex.quote(tgz)}.tmp; mv {shlex.quote(tgz)}.tmp {shlex.quote(tgz)}; "
            f"{fingerprint_command(tgz)}", timeout=900)
        fp = parse_fingerprint(out.strip().splitlines()[-1])
        return {**snap, "archive": {"path": tgz, "sha256": fp["sha256"], "size": fp["size"],
                                    "format": "tar.gz, no .git, sorted, mtime 0"}}

    def resolve_image(self):
        fp = parse_fingerprint(self.ep._sh(fingerprint_command(self.sif), timeout=900))
        return {"path": self.sif, "sha256": fp["sha256"], "size": fp["size"], "policy": "verify-before-run-not-snapshot"}

    def stage(self, manifest) -> Dict[str, Any]:
        """Stage image, code archive and inputs to dCache scratch (verified, reused); returns their dCache paths."""
        ep, img, arc = self.ep, manifest["image"], manifest["code"]["archive"]
        return {"image": ep.stage_file(img["path"], "image", Path(img["path"]).name, img["sha256"], img["size"]),
                "code": ep.stage_file(arc["path"], "code", "code.tgz", arc["sha256"], arc["size"]),
                "inputs": {e["dest"]: ep.stage_file(self.source.path(e["src"]), "inputs", e["dest"], e["sha256"],
                                                    e["size"]) for e in manifest["inputs"]["files"]}}

    # -- contract ----------------------------------------------------------------------------------------------------
    def check(self, name, params) -> Dict[str, Any]:
        try:
            self.render(validate(params, self.TASK))
        except (ValueError, EndpointError) as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def check_binding(self, name, params, geometry=None):
        old = self._rec(name)
        if old is None:
            return None
        p = validate(params, self.TASK)
        if p != old["params"] or self.target() != old["target"]:
            raise ValueError("fastsimkit: different params or target/source from artifact binding")
        if geometry_artifacts.identity(geometry) != geometry_artifacts.identity(old["manifest"].get("geometry")):
            raise ValueError("fastsimkit: geometry differs from artifact binding")
        return old

    def submit(self, name, params, geometry=None) -> Dict[str, Any]:
        p = validate(params, self.TASK)
        if geometry is not None and p["geom"] != "nominal":
            raise ValueError("fastsimkit: geometry FileRef conflicts with geom selector")
        if self.check_binding(name, p, geometry) is None:
            self.render(p, geometry=geometry)
        target = self.target()

        def prepare():
            inputs = resolve_inputs(self.source, self.inputs(p))
            manifest = make_manifest(self.TASK, self.ensure_code(), self.resolve_image(), inputs, p, geometry=geometry)
            rec = {"manifest": manifest, "staged": self.stage(manifest)}
            body, res = self.render(p, rec, geometry=geometry)
            return {**rec, "params": p, "submitted_at": time.time(), "code": manifest["code"]["archive"]["path"],
                    "code_ref": self.code_ref, "body": body, "resources": asdict(res), "target": target}

        try:
            rec = binding(self.dir, name, p, target, prepare, geometry=geometry)
            self.ep.submit(name, rec["body"], Resources(**rec["resources"]))
        except EndpointError as e:
            raise ValueError(str(e)) from e
        return {"handle": name}

    def ensure_streaming(self, name):
        """Nothing streams on the grid: the job copies its staged inputs in itself."""

    def status(self, name) -> Dict[str, Any]:
        from mu2efastsim.endpoints.kit import POLL_MS
        try:
            s = self.ep.status(name)
        except EndpointError as e:
            raise ValueError(str(e)) from e
        return {"state": s.state, "message": f"{s.site_state} {s.message}".strip(),
                "poll_ms": POLL_MS if s.state == "working" else 0, "progress": None}

    def results(self, name) -> Dict[str, Any]:
        r = super().results(name)
        g = Path(self.kit.out_root) / name / "out" / "grid.json"
        if g.exists():
            info = json.loads(g.read_text())
            r["metrics"]["stage_in_s"] = float(info.get("stage_in_s", 0))
            r["metadata"]["image_mode"] = info.get("image_mode")
        cpu = Path(self.kit.out_root) / name / "out" / "cpu.txt"
        if cpu.exists():
            r["metadata"]["cpu_model"] = cpu.read_text().strip()
        return r
