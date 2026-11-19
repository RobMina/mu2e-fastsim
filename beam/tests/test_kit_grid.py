"""beam_edep_cpu on the FNAL grid route (kit_grid + endpoints.jobsub) against a fake grid: the jobsub_* and ifdh
commands are stand-ins on PATH (fakegrid.py), the worker's apptainer is fakeapptainer.py, and the endpoint's
commands, the staging and the rendered job run in a real local bash. No schedd, token, ssh, dCache or image is
contacted; the code snapshot is a local directory (prepare_code is the only mock)."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mu2efastsim.endpoints import kit as kitmod
from mu2efastsim.endpoints import capabilities, chainpoint
from mu2efastsim.endpoints import jobsub as jobsubmod
from mu2efastsim.endpoints.jobsub import FermiGridEndpoint, token_path
from mu2efastsim.endpoints.kit import FastsimKit
from hependpoints import runners as hepep_runners
from hependpoints.runners import local_runner

from beamfastsim import kit_grid, kit_task, kit_uva

if "beam_edep_cpu" not in kitmod.DELEGATES:          # kit.py imported before beam/ was on the path
    kit_task.register(kitmod.TASKS, kitmod.Task, kitmod.Resources, kitmod.FNAL_TASKS)
    kit_task.register_delegate(kitmod.DELEGATES)

HERE = Path(__file__).parent
PE = {"task": "beam_edep_cpu", "geom": "nominal", "muons": 0.0, "electrons": 50000.0, "seed": 20261111.0,
      "dtype": "float64"}
INPUTS = {kit_task.SOB_TABLE: b"beam table", kit_task.EDEP_LIBRARY: b"shower library", kit_task.EDEP_GDML: b"gdml",
          kit_task.EDEP_EDGE: b"edge library"}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class Grid:
    """The fake grid (dCache = a local directory), its commands on PATH, the worker side and a call log."""

    def __init__(self, root: Path):
        self.root, self.bin, self.dcache = root / "grid", root / "bin", root / "dcache"
        for d in (self.root, self.bin, self.dcache):
            d.mkdir(parents=True)
        for name in ("jobsub_submit", "jobsub_q", "jobsub_rm", "jobsub_fetchlog", "ifdh"):
            p = self.bin / name
            p.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fakegrid.py'} {name} \"$@\"\n")
            p.chmod(0o755)
        self.apptainer = self.bin / "apptainer"
        self.apptainer.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fakeapptainer.py'} \"$@\"\n")
        self.apptainer.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}", FAKEGRID_DIR=str(self.root),
                        FAKEGRID_DCACHE=str(self.dcache), FAKEAPPT_CVMFS="1")
        self.calls = []
        self._run = local_runner(self.env)

    def runner(self, cmd, stdin, timeout):
        self.calls.append(cmd[-1])
        return self._run(cmd, stdin, timeout)

    def state(self):
        path = self.root / "state.json"
        return json.loads(path.read_text()) if path.exists() else {"jobs": {}}

    def job(self, jid):
        return self.state()["jobs"][jid]

    def set(self, jid, st):
        data = self.state()
        data["jobs"][jid]["state"] = st
        (self.root / "state.json").write_text(json.dumps(data))

    def run(self, jid, tmp_path, **env):
        """Run the job as a worker would (the env of its -e options, a scratch dir), then drop it from the queue."""
        scratch = tmp_path / f"worker-{jid.split('.')[0]}"
        scratch.mkdir()
        self.set(jid, "R")
        job = self.job(jid)
        cluster, proc = jid.split("@")[0].split(".")
        wenv = dict(self.env, **job["env"], JOBSUBJOBID=jid, CLUSTER=cluster, PROCESS=proc,
                    _CONDOR_SCRATCH_DIR=str(scratch), **env)
        p = subprocess.run(["bash", job["script"]], env=wenv, cwd=scratch, capture_output=True, text=True)
        self.set(jid, "done")
        return p, scratch


class Source(kit_uva._Source):
    """The FNAL input root as a local directory (paths and fingerprints local; nothing is read over ssh)."""

    def __init__(self, root):
        super().__init__("fnal-host.example", str(root))

    def fingerprint(self, rel):
        data = Path(self.path(rel)).read_bytes()
        return {"md5": hashlib.md5(data).hexdigest(), "sha256": _sha(data), "size": len(data)}


@pytest.fixture
def grid(tmp_path):
    return Grid(tmp_path)


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    for k in [k for k in os.environ if k.startswith("FASTSIMKIT_")]:
        monkeypatch.delenv(k)


@pytest.fixture
def code_tree(tmp_path, monkeypatch):
    """prepare_code (a git snapshot on the submit host) replaced by a local directory; everything after it (the tar,
    its digest, the staging, the worker's unpack and check) is production code."""
    snap = tmp_path / "submit" / "beam_code_kit.snapshots" / ("e" * 32)
    (snap / "beam" / "beamfastsim").mkdir(parents=True)
    (snap / "core").mkdir()
    (snap / "beam" / "beamfastsim" / "__init__.py").write_text("# code\n")
    calls = []

    def prepare(ep, base, ref, **kw):
        calls.append((base, ref))
        return {"path": str(snap), "umbrella_commit": "a" * 40, "submodules": {}, "requested_ref": ref,
                "policy": {"name": "retained-readonly-git-v1", "modules": [], "branch": ""}}
    monkeypatch.setattr(kit_grid, "prepare_code", prepare)
    return calls


def make(tmp_path, grid, **kw):
    root = tmp_path / "fnal"
    for rel, data in INPUTS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    sif = tmp_path / "submit" / "mu2efastsim-cpu.sif"
    sif.parent.mkdir(parents=True, exist_ok=True)
    sif.write_bytes(b"cpu image")
    # neutral setups: a test host may have a /cvmfs whose Mu2e setup would replace the fake ifdh on the PATH
    cfg = dict(host="fnal-host.example", workdir=str(tmp_path / "submit"), output_dir=str(grid.dcache / "out"),
               stage_dir=str(grid.dcache / "stage"), state_dir=tmp_path / "state", runner=grid.runner,
               host_setup="", worker_setup="command -v ifdh >/dev/null || exit 89\n")
    cfg.update(kw)
    ep = FermiGridEndpoint(**cfg)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit")
    kit._delegates["beam_edep_cpu"] = kit_grid.BeamEdepCpuGrid(kit, source=Source(root), sif=str(sif),
                                                              apptainer=str(grid.apptainer))
    return ep, kit


def _jid(ep, name):
    return ep._record(name).job_id


# -- sizing, routing, endpoint configuration -----------------------------------------------------------------------

def test_grid_request_is_sized_from_the_measured_uva_envelope():
    size = lambda n: (lambda r: (r.minutes, r.mem_gb, r.cores, r.gpus))(  # noqa: E731
        kit_task.edep_cpu_grid_resources({"electrons": n}, kitmod.Resources))
    # 2 (grid node) x 1.5 (margin) x the longest UVA elapsed (22 / 42 / 115 min at 5k / 10k / 50k CE), 30-min steps
    assert [size(n)[0] for n in (1, 5_000, 10_000, 20_000, 50_000)] == [60, 90, 150, 210, 360]
    assert {size(n)[1:] for n in (1, 10_000, 50_000)} == {(13, 1, 0)}       # MaxRSS <= 11.1 GB at 50k CE
    assert size(0)[:2] == size(50_001)[:2] == (720, 16)                    # not measured: the flat request


def test_beam_edep_cpu_routes_to_the_grid_on_a_jobsub_endpoint(grid, tmp_path):
    ep, kit = make(tmp_path, grid)
    d = kitmod.DELEGATES["beam_edep_cpu"]
    assert d.applies(ep) and isinstance(d.make(kit), kit_grid.BeamEdepCpuGrid)
    uva = type("UVA", (), {"stream_in": 1, "fnal_data": None})()
    eaf = type("EAF", (), {"fnal_data": "/path/to/fnal"})()
    assert d.applies(uva) and not d.applies(eaf)
    assert "fermigrid" in FastsimKit.describe()["tasks"]["beam_edep_cpu"]["endpoints"]
    p = kit_uva.validate(PE, "beam_edep_cpu")
    cap = capabilities.resolve("beam_edep_cpu", p, ep, kitmod.TASKS, chainpoint)
    assert cap["route"]["name"] == "staged" and cap["route"]["supported"]
    assert cap["metrics"]["input_wait_s"]["availability"] == "unavailable"   # nothing streams on the grid
    assert capabilities.resolve("beam_edep_cpu", p, uva, kitmod.TASKS, chainpoint)["route"]["name"] == "streaming"
    assert ["staged"] == [r["name"] for r in capabilities._routes("beam_edep_cpu") if "stage_file" in r["requires"]]
    assert not any("stage_file" in r["requires"] for r in capabilities._routes("beam_edep"))


def test_endpoint_settings_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTSIMKIT_FNAL_HOST", "fnal-host.example")
    monkeypatch.setenv("FASTSIMKIT_JOBSUB_WORKDIR", "/path/to/submit")
    mount = "/pnfs/"                                                     # a dCache mount path, rewritten for the token
    monkeypatch.setenv("FASTSIMKIT_JOBSUB_OUTPUT_DIR", mount + "vo/scratch/area/out")
    with pytest.raises(Exception, match="FASTSIMKIT_JOBSUB_STAGE_DIR"):
        FermiGridEndpoint(state_dir=tmp_path)
    monkeypatch.setenv("FASTSIMKIT_JOBSUB_STAGE_DIR", mount + "vo/scratch/area/stage")
    monkeypatch.setenv("FASTSIMKIT_ENDPOINT", "fermigrid")
    kit = FastsimKit(state_dir=tmp_path / "kit")                          # through endpoints.get, with the UVA cap
    ep = kit.ep
    assert isinstance(ep, FermiGridEndpoint) and ep.name == "fermigrid"
    assert (ep.host, ep.group, ep.role, ep.disk) == ("fnal-host.example", "mu2e", "Analysis", "4GB")
    assert ep.storage_modify == "/vo/scratch/area/out" == token_path(ep.output_dir)
    assert jobsubmod.AUTO_RELEASE in ep.extra_args
    argv = ep.submit_argv("p1", "m2fs-v1-0", kit_task.edep_cpu_grid_resources({"electrons": 50000}, kitmod.Resources))
    for flag in ("-G", "mu2e", "--role=Analysis", "--cpu=1", "--memory=13GB", "--expected-lifetime=360m",
                 "--disk=4GB", "--need-storage-modify=/vo/scratch/area/out", "--lines=+FERMIHTC_AutoRelease=True",
                 "--resource-provides=usage_model=DEDICATED,OPPORTUNISTIC"):
        assert flag in argv, flag


def test_submit_host_commands_run_after_the_mu2e_setup_and_never_read_stdin(monkeypatch, tmp_path):
    seen = []

    def fake_run(argv, **kw):
        seen.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, b"ok\n", b"")
    monkeypatch.setattr(hepep_runners.subprocess, "run", fake_run)        # hependpoints' ssh runner
    ep = FermiGridEndpoint(host="fnal-host.example", workdir="/path/to/submit", output_dir="/path/to/out",
                           stage_dir="/path/to/stage", state_dir=tmp_path)
    assert ep._sh("echo hi") == "ok\n"
    argv, kw = seen[-1]
    assert argv[:3] == ["ssh", "-o", "BatchMode=yes"] and "fnal-host.example" in argv
    assert kw["stdin"] is subprocess.DEVNULL and "input" not in kw         # an MCP client's requests stay unread
    remote = argv[-1]
    assert remote.index("setupmu2e-art.sh") < remote.index("muse setup ops") < remote.index("echo hi")


def test_the_grid_body_runs_the_image_through_the_upstream_sandbox_fallback():
    from hependpoints.apptainer import sandbox_fallback
    assert sandbox_fallback(check='python3 -c "$CHECK"') in kit_grid.BODY
    assert kit_grid.BODY.index('CHECK="import jax') < kit_grid.BODY.index('IMG="$W/img.sif"; MODE=sif')
    assert kit_grid.BODY.index("MODE=sandbox") < kit_grid.BODY.index('"image_mode\\": \\"$MODE')


# -- end to end on the fake grid ------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_submit_stage_run_results_end_to_end(grid, tmp_path, code_tree, dtype):
    ep, kit = make(tmp_path, grid)
    p = {**PE, "dtype": dtype}
    assert kit.check("g1", p, [], [])["ok"]
    assert kit.submit("g1", p, [], []) == {"handle": "g1"}
    rec = kit._delegates["beam_edep_cpu"]._rec("g1")
    staged = rec["staged"]
    # image, code archive and the four inputs staged by content, each one verified
    assert staged["image"].startswith(f"{grid.dcache}/stage/image/{_sha(b'cpu image')[:16]}/")
    for e in rec["manifest"]["inputs"]["files"]:
        path = Path(staged["inputs"][e["dest"]])
        assert path.read_bytes() == INPUTS[e["src"]] and path.parent.name == e["sha256"][:16]
    arc = rec["manifest"]["code"]["archive"]
    assert Path(staged["code"]).read_bytes() == Path(arc["path"]).read_bytes() and _sha(
        Path(arc["path"]).read_bytes()) == arc["sha256"]
    assert (rec["resources"]["minutes"], rec["resources"]["mem_gb"]) == (360, 13)
    body = rec["body"]
    assert subprocess.run(["bash", "-n"], input=body.encode()).returncode == 0
    assert kit_grid.RUN in kit_uva.EDEP_CPU_BODY                     # the streaming route's command line
    for line in kit_grid.RUN.splitlines()[:2]:
        for k, v in {"zup": "3900.0", "geomarg": "", "muons": "0", "electrons": "50000", "seed": "20261111",
                     "dtype": dtype}.items():
            line = line.replace(f"@{k}@", v)
        assert line in body, line
    jid = _jid(ep, "g1")
    job = grid.job(jid)
    assert f"--need-storage-modify={grid.dcache}/out" in job["argv"] and "--memory=13GB" in job["argv"]
    assert kit.status("g1")["state"] == "working"
    proc, scratch = grid.run(jid, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nested apptainer ok" in proc.stdout
    assert not (scratch / "hepep_job" / "in").exists() and not (scratch / "hepep_job" / "img.sif").exists()
    assert kit.status("g1")["state"] == "completed"
    res = kit.results("g1")
    out = kit.out_root / "g1" / "out"
    assert res["metrics"]["n_gen_events"] == 50000 and res["metrics"]["stage_in_s"] >= 0
    assert res["files"] == [{"name": "nts.mu2efastsim.edep.root", "uri": (out / "nts.mu2efastsim.edep.root").resolve().as_uri(),
                             "kind": "root"}]
    assert res["metadata"]["endpoint"] == "fermigrid" and res["metadata"]["image_mode"] == "sif"
    assert res["metadata"]["artifact_binding"] == "verified"
    assert "deleted" in (out / "cleanup.txt").read_text()
    point = json.loads((out / "fake_point.json").read_text())
    assert point["argv"][point["argv"].index("--dtype") + 1] == dtype and "--procs" in point["argv"]
    assert point["argv"][point["argv"].index("--procs") + 1] == "1"
    assert "JAX_PLATFORMS=cpu" in point["env"] and "BEAMFS_MAPS" not in point["env"]    # maps from /cvmfs
    assert "/cvmfs" in point["binds"]


def test_worker_without_fuse_runs_from_a_sandbox_and_without_cvmfs_stages_over_http(grid, tmp_path, code_tree):
    ep, kit = make(tmp_path, grid)
    kit.submit("g2", PE, [], [])
    proc, _ = grid.run(_jid(ep, "g2"), tmp_path, FAKEAPPT_NO_FUSE="1", FAKEAPPT_CVMFS="")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    res = kit.results("g2")
    assert res["metadata"]["image_mode"] == "sandbox"
    point = json.loads((kit.out_root / "g2" / "out" / "fake_point.json").read_text())
    assert point["image"].endswith("/img") and "BEAMFS_MAPS=" in point["env"]


def test_a_staged_input_changed_on_dcache_stops_the_job_before_the_point(grid, tmp_path, code_tree):
    ep, kit = make(tmp_path, grid)
    kit.submit("g3", PE, [], [])
    staged = kit._delegates["beam_edep_cpu"]._rec("g3")["staged"]["inputs"]["g4ref.npz"]
    Path(staged).write_bytes(b"other table")
    proc, _ = grid.run(_jid(ep, "g3"), tmp_path)
    assert proc.returncode == 3
    assert kit.status("g3")["state"] == "failed"
    with pytest.raises(ValueError, match="not completed"):
        kit.results("g3")


def test_a_failing_point_keeps_its_exit_code_and_still_records_the_cleanup(grid, tmp_path, code_tree):
    ep, kit = make(tmp_path, grid)
    kit.submit("g4", PE, [], [])
    proc, _ = grid.run(_jid(ep, "g4"), tmp_path, FAKEAPPT_POINT_RC="1")
    assert proc.returncode == 6
    rec = ep._record("g4")
    assert Path(rec.remote_dir, "job.done").read_text() == "exit=6\n"
    files = ep.fetch("g4", tmp_path / "fetched")
    assert any(f.name == "cleanup.txt" for f in files) and kit.status("g4")["state"] == "failed"


def test_staging_is_reused_and_a_changed_source_is_refused(grid, tmp_path, code_tree):
    ep, kit = make(tmp_path, grid)
    kit.submit("g5", PE, [], [])
    n_copies = sum("ifdh cp" in c and "/stage/" in c and "mktemp" not in c for c in grid.calls)
    assert n_copies == 6                                                   # image, code, four inputs
    kit.submit("g6", {**PE, "seed": 7.0}, [], [])
    assert sum("ifdh cp" in c and "/stage/" in c and "mktemp" not in c for c in grid.calls) == n_copies
    assert _jid(ep, "g6") != _jid(ep, "g5")
    root = Path(kit._delegates["beam_edep_cpu"].source.root)
    pinned = kit._delegates["beam_edep_cpu"]._rec("g5")["manifest"]["inputs"]["files"][0]
    with pytest.raises(Exception, match="not the pinned file"):
        ep.stage_file(str(root / pinned["src"]), "inputs", "x.npz", "0" * 64, pinned["size"])


def test_retry_is_idempotent_and_changed_params_are_refused(grid, tmp_path, code_tree):
    ep, kit = make(tmp_path, grid)
    kit.submit("g7", PE, [], [])
    assert kit.submit("g7", PE, [], []) == {"handle": "g7"}
    assert len(grid.state()["jobs"]) == 1
    with pytest.raises(ValueError, match="different params"):
        kit.submit("g7", {**PE, "seed": 2.0}, [], [])


def test_cancel_removes_the_queued_job(grid, tmp_path, code_tree):
    ep, kit = make(tmp_path, grid)
    kit.submit("g8", PE, [], [])
    assert kit.cancel("g8") == {"state": "cancelled"}
    assert grid.job(_jid(ep, "g8"))["state"] == "X"
    assert kit.status("g8")["state"] == "cancelled"


def test_check_refuses_field_scales_and_needs_no_remote_call(grid, tmp_path):
    ep, kit = make(tmp_path, grid)
    r = kit.check("g9", {**PE, "ds_bscale": 1.01}, [], [])
    assert not r["ok"] and "streaming route" in r["message"]
    with pytest.raises(ValueError, match="streaming route"):
        kit.submit("g9", {**PE, "ds_bscale": 1.01}, [], [])
    assert kit.check("g10", PE, [], [])["ok"]
    assert not grid.calls                                               # admission never stages or submits
