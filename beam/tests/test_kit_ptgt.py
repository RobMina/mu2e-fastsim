"""fastsimkit task ptgt_point: registration, parameters, the rendered CPU / GPU job and the Slurm-over-SSH round trip
against the fake site of test_kit_uva (nothing streamed)."""
import subprocess

import pytest

from mu2efastsim.endpoints import kit as kitmod
from mu2efastsim.endpoints.kit import FastsimKit
from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint

from beamfastsim import kit_task, kit_uva
from test_kit_uva import SITE, Site, _artifacts  # noqa: F401  (autouse: code snapshot and image mocked)

if "ptgt_point" not in kitmod.TASKS:          # kit.py imported before beam/ was on the path
    kit_task.register(kitmod.TASKS, kitmod.Task, kitmod.Resources, kitmod.FNAL_TASKS)
    kit_task.register_delegate(kitmod.DELEGATES)
    kitmod.DELEGATED_ONLY.add("ptgt_point")

P = {"task": "ptgt_point", "Tlength": 160.0, "R_up": 3.1495, "R_mid": 3.1495, "R_dn": 3.1495, "pot": 2000.0,
     "seed": 1.0, "dtype": "float32", "platform": "cpu"}


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    for name, value in {**SITE, "FASTSIMKIT_PTGT_DATA": "/path/to/ptgt_data"}.items():
        monkeypatch.setenv(name, value)


def make(tmp_path):
    site = Site()
    ep = SlurmSSHEndpoint(runner=site, state_dir=tmp_path / "jobs", workdir="/path/to/slurm/kit")
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    kit._delegates["ptgt_point"] = kit_uva.PtgtPointStream(kit, code="/path/to/slurm/beam", background=False)
    return site, ep, kit


def test_registered_with_a_refusing_mounted_body_and_a_uva_delegate():
    t = kitmod.TASKS["ptgt_point"]
    assert t.params == kit_task.PTGT_PARAMS and "stops_per_pot" in t.metrics and "ptgt_point" in kitmod.DELEGATED_ONLY
    assert "exit 3" in t.body and kitmod.DELEGATES["ptgt_point"] is kit_task._UVAPtgtDelegate
    assert t.resources_for({**kit_task.PTGT_DEFAULTS, "platform": "gpu"}).gpus == 1


def test_validate_defaults_and_bounds():
    v = kit_uva.validate({k: P[k] for k in ("Tlength", "R_up", "R_mid", "R_dn")}, "ptgt_point")
    assert v == {"Tlength": 160.0, "R_up": 3.1495, "R_mid": 3.1495, "R_dn": 3.1495, **kit_task.PTGT_DEFAULTS}
    assert kit_uva.validate(P, "ptgt_point")["pot"] == 2000
    for bad in ({**P, "pot": 10}, {**P, "pot": 1500.5}, {**P, "platform": "tpu"}, {**P, "dtype": "float16"},
                {**P, "R_up": 0.0}, {**P, "Tlength": 5000.0}, {**P, "geom": "nominal"}):
        with pytest.raises(ValueError):
            kit_uva.validate(bad, "ptgt_point")


def test_sizing_by_platform_and_pot():
    cpu = kit_task.ptgt_resources({"pot": 100_000, "platform": "cpu"}, kitmod.Resources)
    gpu = kit_task.ptgt_resources({"pot": 100_000, "platform": "gpu"}, kitmod.Resources)
    assert cpu.gpus == 0 and cpu.cores == kit_task.PTGT_CPU_CORES and cpu.minutes % 5 == 0
    assert gpu.gpus == 1 and gpu.gpu_type == "a100" and gpu.minutes < cpu.minutes
    big = kit_task.ptgt_resources({"pot": 10 ** 8, "platform": "cpu"}, kitmod.Resources)
    assert big.minutes == kit_task.PTGT_MAX_MINUTES


def test_cpu_body_submit_and_results(tmp_path):
    site, ep, kit = make(tmp_path)
    assert kit.check("p1", P, [], [])["ok"]
    assert kit.submit("p1", P, [], []) == {"handle": "p1"}
    body = site.body
    assert "-m beamfastsim.ptgt_point --rod 160.0 3.1495 3.1495 3.1495 --pot 2000 --seed 1 --dtype float32" in body
    assert "--platform cpu" in body and "JAX_PLATFORMS=cpu" in body and "apptainer exec --nv" not in body
    assert "gpu:" not in body and "nvidia-smi" not in body
    assert "--extra-map GA05/PSMap" in body and '--ps-map "$L/pub/maps/GA05/PSMap.header"' in body
    assert all(f"{h}  {n}" in body for n, h in kit_task.PTGT_DATA_SHA256.items()) and "/path/to/ptgt_data" in body
    assert "--file Offline/Mu2eG4/geom/bfgeom_v01.txt" in body and "/path/to/slurm/ptgt_cache" in body
    r = subprocess.run(["bash", "-n"], input=body.encode(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    j = ep.state.get("p1").job_id
    site.state[j] = "RUNNING"
    assert kit.status("p1")["state"] == "working"
    site.state[j] = "COMPLETED"
    assert "stop_frac" in kit.results("p1")["metrics"]      # the fake site's metrics file comes back
    with pytest.raises(ValueError, match="different params"):
        kit.submit("p1", {**P, "Tlength": 170.0}, [], [])


def test_gpu_body(tmp_path):
    site, ep, kit = make(tmp_path)
    assert kit.submit("p2", {**P, "platform": "gpu", "dtype": "float64"}, [], []) == {"handle": "p2"}
    body = site.body
    assert "apptainer exec --nv" in body and "JAX_PLATFORMS=cuda" in body and "gpu:a100:1" in body
    assert "--platform gpu" in body and "JAX not on the GPU" in body
    r = subprocess.run(["bash", "-n"], input=body.encode(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode()


def test_refuses_a_geometry_file_and_a_missing_data_directory(tmp_path, monkeypatch):
    _, _, kit = make(tmp_path)
    d = kit._delegates["ptgt_point"]
    with pytest.raises(ValueError, match="no geometry"):
        d.render(kit_uva.validate(P, "ptgt_point"), geometry={"name": "g"})
    monkeypatch.delenv("FASTSIMKIT_PTGT_DATA")
    assert not kit.check("p3", P, [], [])["ok"]
