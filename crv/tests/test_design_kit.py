"""fastsimkit task crv_design (crvfastsim.design.kit_task / kit_uva): registration, parameter domain, the rendered CPU
job body (tables to tmpfs, pinned sample, no GPU) and the results contract, on a fake Slurm site. No Mu2e inputs."""
import dataclasses
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim.endpoints import kit as kitmod                          # noqa: E402
from mu2efastsim.endpoints.kit import FastsimKit                         # noqa: E402
from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint              # noqa: E402

from crvfastsim import runconfig as RC                                   # noqa: E402
from crvfastsim.design import kit_task as KT                             # noqa: E402
from crvfastsim.design import kit_uva as KU                              # noqa: E402
from crvfastsim.design import module as M                                # noqa: E402

if "crv_design" not in kitmod.TASKS:          # kit.py imported before crv/ was on the path
    KT.register(kitmod.TASKS, kitmod.Task, kitmod.Resources, delegates=kitmod.DELEGATES)
    kitmod.TABLE_TASKS.add("crv_design")
    kitmod.DELEGATED_ONLY.add("crv_design")

SITE = {"FASTSIMKIT_SLURM_HOST": "slurm-host.example", "FASTSIMKIT_SLURM_ACCOUNT": "test_account",
        "FASTSIMKIT_SCRATCH": "/path/to/slurm"}
SHA = "ab" * 32


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    for name, value in SITE.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(KU, "prepare_code", lambda ep, base, ref: {"path": base + "/snap", "ref": ref})
    monkeypatch.setattr(KU, "resolve_image", lambda ep, sif: {"path": sif, "sha256": "cd" * 32, "size": 1,
                                                               "policy": "test"})
    monkeypatch.setattr(KU, "runtime_prefix", lambda m: "printf '%s\\n' " + shlex.quote(json.dumps(m, sort_keys=True))
                        + ' > "$M2FS_OUT/artifact-manifest.json"\n')


class Site:
    """Fake ssh runner: sbatch, squeue/sacct, the sample fingerprint and the result tar."""

    def __init__(self, cleanup=True):
        self.next_id, self.state, self.dirs, self.manifests, self.cleanup = 8000, {}, {}, {}, cleanup
        self.body = ""

    def __call__(self, cmd, stdin, timeout):
        s = cmd[-1]
        out = b""
        if "sbatch --parsable" in s:
            self.next_id += 1
            j = str(self.next_id)
            self.state[j] = "PENDING"
            self.dirs[j] = s.split("cd ")[1].split(" &&")[0].strip("'")
            self.body = stdin.decode()
            line = next(ln for ln in self.body.splitlines() if "artifact-manifest.json" in ln)
            self.manifests[self.dirs[j]] = json.loads(shlex.split(line)[2])
            out = f"{j}\n".encode()
        elif s.startswith("python3 -c"):
            out = json.dumps({"md5": "ef" * 16, "sha256": SHA, "size": 123}).encode()
        elif s.startswith("squeue"):
            j = s.split("-j ")[1].split()[0]
            out = f"{self.state[j]}|None\n".encode() if self.state[j] in ("PENDING", "RUNNING") else b""
        elif s.startswith("sacct"):
            j = s.split("-j ")[1].split()[0]
            out = f"{self.state[j]}|0:0\n".encode()
        elif "tar -cf" in s:
            d = s.split("cd ")[1].split(" &&")[0].strip("'")
            files = {"out/metrics.json": json.dumps({"inefficiency": 2.3e-3, "cost": 1.0}).encode(),
                     "out/artifact-manifest.json": json.dumps(self.manifests[d]).encode(), "slurm.out": b"ok\n"}
            if self.cleanup:
                files["out/cleanup.txt"] = b"deleted\n"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as t:
                for name, data in files.items():
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
            out = buf.getvalue()
        return subprocess.CompletedProcess(cmd, 0, out, b"")


def make(tmp_path, cleanup=True):
    site = Site(cleanup)
    ep = SlurmSSHEndpoint(runner=site, state_dir=tmp_path / "jobs", workdir="/path/to/slurm/kit")
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    return site, ep, kit


P = {"task": "crv_design", "n_layers": 5.0, "absorber": 20.0, "tilt_deg": 10.0, "coincidence_layers": 3.0}


def test_static_names_match_the_package():
    assert KT.ABSORBER_MATERIALS == M.ABSORBER_MATERIALS
    fields = {f.name for f in dataclasses.fields(M.Design)}
    assert set(KT.KNOBS) | {"absorber_material"} <= fields
    assert set(KT.TABLES) == {f"LookupTable_{s.table}" for s in RC.RUN2_SECTORS.values()}
    assert "crv_design" in kitmod.TASKS and "crv_design" in kitmod.DELEGATES and "crv_design" in kitmod.TABLE_TASKS
    assert kitmod.TASKS["crv_design"].resources.gpus == 0 and "crv_design" in kitmod.DELEGATED_ONLY


def test_plain_render_refuses_the_delegated_task(tmp_path):
    _, _, kit = make(tmp_path)
    with pytest.raises(ValueError, match="delegate"):
        kit.render(P)


def test_design_words():
    assert KT.design_dict(["n_layers=5.0", "thickness=", "absorber_material=MildSteel", "tilt_deg=12.5"]) == \
        {"n_layers": 5, "absorber_material": "MildSteel", "tilt_deg": 12.5}
    with pytest.raises(ValueError):
        KT.design_dict(["layers=4"])


def test_parameter_domain():
    p = KU.validate(P)
    assert p["n_layers"] == 5 and p["thickness"] == "" and p["dtype"] == "float64" and p["pe_scale"] == 0.847
    for bad in ({**P, "n_layers": 0.0}, {**P, "n_layers": 4.5}, {**P, "tilt_deg": 85.0}, {**P, "dtype": "float16"},
                {**P, "absorber_material": "Wood"}, {**P, "coincidence_layers": 6.0}, {**P, "sample": "other"},
                {**P, "colour": 1.0}, {**P, "spare_layers": 1.0}, {k: v for k, v in {**P, "spare_layers": 5.0}.items()
                                                                   if k != "coincidence_layers"}):
        with pytest.raises(ValueError):
            KU.validate(bad)


def test_submit_renders_a_cpu_job_and_results_need_the_cleanup(tmp_path):
    site, ep, kit = make(tmp_path)
    assert kit.check("d1", P, [], [])["ok"]
    assert kit.submit("d1", P, [], []) == {"handle": "d1"}
    body = site.body
    assert subprocess.run(["bash", "-n"], input=body.encode(), capture_output=True).returncode == 0
    assert "--nv" not in body and "nvidia-smi" not in body and "JAX_PLATFORMS=cpu" in body
    assert f'echo "{SHA}  $MU" | sha256sum -c -' in body and "/path/to/slurm/crv_design_data/run2_ce_proxy_v1.npz" in body
    assert all(t in body for t in KT.TABLES) and "cleanup.txt" in body
    assert "n_layers=5 " in body and "thickness=''" in body and "--dtype float64" in body
    assert "/path/to/slurm/crv_design_code_kit/snap" in body
    j = ep.state.get("d1").job_id
    assert kit.status("d1")["state"] == "working"
    site.state[j] = "COMPLETED"
    r = kit.results("d1")
    assert r["metrics"]["inefficiency"] == pytest.approx(2.3e-3) and r["metadata"]["artifact_binding"] == "verified"
    assert kit.submit("d1", P, [], []) == {"handle": "d1"}                 # idempotent
    with pytest.raises(ValueError):
        kit.submit("d1", {**P, "n_layers": 6.0}, [], [])                    # same name, other params


def test_results_refused_without_cleanup_record(tmp_path):
    site, ep, kit = make(tmp_path, cleanup=False)
    kit.submit("d2", P, [], [])
    site.state[ep.state.get("d2").job_id] = "COMPLETED"
    with pytest.raises(ValueError, match="cleanup"):
        kit.results("d2")
