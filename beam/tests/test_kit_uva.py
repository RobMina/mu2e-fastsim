"""beam_point on a Slurm-over-SSH endpoint against a fake site: delegation, streamed input, md5 check, cleanup gate."""
import io
import json
import subprocess
import tarfile

import pytest

from mu2efastsim.endpoints import kit as kitmod
from mu2efastsim.endpoints.kit import FastsimKit
from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint

from beamfastsim import kit_task, kit_uva

if "beam_point" not in kitmod.DELEGATES:          # kit.py imported before beam/ was on the path
    kit_task.register(kitmod.TASKS, kitmod.Task, kitmod.Resources, kitmod.FNAL_TASKS)
    kit_task.register_delegate(kitmod.DELEGATES)


class Site:
    def __init__(self, cleanup=True):
        self.next_id, self.state, self.files, self.dirs, self.cmds, self.cleanup = 7000, {}, {}, {}, [], cleanup

    def __call__(self, cmd, stdin, timeout):
        s = cmd[-1]
        self.cmds.append(s)
        out = b""
        if "sbatch --parsable" in s:
            self.next_id += 1
            j = str(self.next_id)
            self.state[j] = "PENDING"
            self.dirs[j] = s.split("cd ")[1].split(" &&")[0].strip("'")
            self.body = stdin.decode()
            out = f"{j}\n".encode()
        elif s.startswith("squeue"):
            j = s.split("-j ")[1].split()[0]
            out = f"{self.state[j]}|None\n".encode() if self.state[j] in ("PENDING", "RUNNING") else b""
        elif s.startswith("sacct"):
            j = s.split("-j ")[1].split()[0]
            out = f"{self.state[j]}|0:0\n".encode()
        elif s.startswith("cat "):
            out = self.files.get(s.split()[1].strip("'"), "").encode()
        elif s.startswith("scancel"):
            self.state[s.split()[1]] = "CANCELLED"
        elif "tar -cf" in s:
            buf = io.BytesIO()
            files = {"out/metrics.json": json.dumps({"stop_frac": 0.358, "muons": 1000}).encode(),
                     "out/wait.json": b'{"input_wait_s": 12}', "slurm.out": b"ok\n"}
            if self.cleanup:
                files["out/cleanup.txt"] = b"deleted\n"
            with tarfile.open(fileobj=buf, mode="w") as t:
                for name, data in files.items():
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
            out = buf.getvalue()
        return subprocess.CompletedProcess(cmd, 0, out, b"")


class Source(kit_uva._Source):
    def __init__(self):
        super().__init__("gpvm", "/exp/beam")

    def md5(self, rel):
        return "md5:" + rel


def make(tmp_path, cleanup=True, md5_ok=True):
    site = Site(cleanup)
    streamed = []

    def stream_runner(src, host, remote, timeout):
        path = src[-1].split("cat ", 1)[1].strip("'")
        rel = path.split("/exp/beam/", 1)[1] if path.startswith("/exp/beam/") else path
        streamed.append((rel, remote.split("cat > ")[1].split()[0].strip("'\"")))
        return ("md5:" + rel if md5_ok else "bad") + " x"

    ep = SlurmSSHEndpoint(runner=site, state_dir=tmp_path / "jobs", workdir="/scratch/u/kit", stream_runner=stream_runner)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    kit._delegates["beam_point"] = kit_uva.BeamPointStream(kit, source=Source(), code="/scratch/u/beam", background=False)
    return site, ep, kit, streamed


P = {"task": "beam_point", "geom": "move_extras12.txt", "muons": 105350.0, "seed": 1.0, "repeat": 1.0}


def test_validate():
    assert kit_uva.validate(P)["muons"] == 105350
    for bad in ({**P, "geom": "../x.txt"}, {**P, "geom": "/etc/passwd"}, {**P, "geom": "a b.txt"},
                {**P, "muons": 0}, {**P, "repeat": 1.5}, {k: v for k, v in P.items() if k != "seed"}):
        with pytest.raises(ValueError):
            kit_uva.validate(bad)
    assert kit_uva.validate({**P, "geom": "/exp/mu2e/data/users/x/geom/pt01.txt"})["geom"].endswith("pt01.txt")


def test_uva_delegates_and_eaf_does_not(tmp_path):
    _, ep, kit, _ = make(tmp_path)
    assert kit.check("b", P, [], [])["ok"]                       # FNAL-only as a plain task, but delegated on UVA
    assert kit_task._UVADelegate.applies(ep)
    assert not kit_task._UVADelegate.applies(type("E", (), {"fnal_data": "/exp/x", "stream_in": 1})())


def test_submit_stream_results(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    assert kit.submit("bp1", P, [], []) == {"handle": "bp1"}
    j = ep.state.get("bp1").job_id
    body = site.body
    assert "--geom \"$CODE/beam/geom/move_extras12.txt\"" in body and "--n 105350 " in body and "gpu:a100:1" in body
    r = subprocess.run(["bash", "-n"], input=body.encode(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    assert kit.status("bp1")["state"] == "working" and not streamed
    site.state[j] = "RUNNING"
    site.files[f"{site.dirs[j]}/inbox"] = "udc-a1 /tmp/beamfs_kit_j7001_9\n"
    kit.status("bp1")
    assert streamed == [("g4ref_full/g4ref.npz", "/tmp/beamfs_kit_j7001_9/g4ref.npz")]
    assert any("touch" in c and "/tmp/beamfs_kit_j7001_9/ready" in c for c in site.cmds)
    site.state[j] = "COMPLETED"
    res = kit.results("bp1")
    assert res["metrics"]["stop_frac"] == 0.358 and res["metrics"]["input_wait_s"] == 12
    with pytest.raises(ValueError, match="different params"):
        kit.submit("bp1", {**P, "seed": 2.0}, [], [])


def test_overlay_path_streamed_and_md5_mismatch_cancels(tmp_path):
    site, ep, kit, streamed = make(tmp_path, md5_ok=False)
    g = "/exp/mu2e/data/users/x/geom/pt01.txt"
    kit.submit("bp2", {**P, "geom": g}, [], [])
    j = ep.state.get("bp2").job_id
    assert '--geom "$L/geom_overlay.txt"' in site.body
    site.state[j] = "RUNNING"
    site.files[f"{site.dirs[j]}/inbox"] = "udc-a1 /tmp/t\n"
    kit.status("bp2")
    assert site.state[j] == "CANCELLED"
    assert "stream error" in kit.status("bp2")["message"]


def test_results_refused_without_cleanup(tmp_path):
    site, ep, kit, _ = make(tmp_path, cleanup=False)
    kit.submit("bp3", P, [], [])
    j = ep.state.get("bp3").job_id
    site.state[j] = "COMPLETED"
    with pytest.raises(ValueError, match="cleanup record"):
        kit.results("bp3")
