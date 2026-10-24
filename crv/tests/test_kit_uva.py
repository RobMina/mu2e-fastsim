"""crv_response on a Slurm-over-SSH endpoint (UVA): render, the streamed input set, and a full point against a fake site
(core/tests/test_chainpoint.py's Site), without UVA or the network."""
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("mu2efastsim.endpoints.slurm_ssh")
from crvfastsim import kit_task as KT, kit_uva as KU                     # noqa: E402
from mu2efastsim.endpoints.chainpoint import FnalSource                  # noqa: E402
from mu2efastsim.endpoints.kit import FastsimKit                         # noqa: E402
from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint             # noqa: E402


class Site:
    """Fake UVA: sbatch, squeue/sacct, cat of job files, srun touch, tar of out/."""

    def __init__(self):
        self.next_id, self.state, self.files, self.dirs, self.node_files = 7000, {}, {}, {}, set()

    def __call__(self, cmd, stdin, timeout):
        s = cmd[-1]
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
            out = f"{self.state[j]}|None\n".encode() if self.state.get(j) in ("PENDING", "RUNNING") else b""
        elif s.startswith("sacct") and "ElapsedRaw" in s:
            out = b"2026-10-23T18:00:00|2026-10-23T18:10:00|600\n"
        elif s.startswith("sacct"):
            j = s.split("-j ")[1].split()[0]
            out = f"{self.state[j]}|0:0\n".encode()
        elif s.startswith("cat "):
            out = self.files.get(s.split()[1].strip("'"), "").encode()
        elif s.startswith("srun") and "touch" in s:
            self.node_files.add(s.split("touch ")[1].split()[0].strip("'\""))
        elif s.startswith("srun") and "test -f" in s:
            path = s.split("test -f ")[1].split()[0].strip("'\"")
            out = b"have\n" if path in self.node_files else b""
        elif "tar -cf" in s:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as t:
                for name, data in {"out/metrics.json": json.dumps({"objective": 30.1, "objective_sigma": 0.6}).encode(),
                                   "out/cleanup.txt": b"deleted\n", "out/wait.json": b'{"input_wait_s": 95}'}.items():
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
            out = buf.getvalue()
        return subprocess.CompletedProcess(cmd, 0, out, b"")


class Source(FnalSource):
    def __init__(self):
        super().__init__("gpvm", "/exp/data")

    def md5(self, rel):
        return "md5:" + rel


def make(tmp_path):
    site, streamed = Site(), []

    def stream_runner(src, host, remote, timeout):
        rel = src[-1].split("/exp/data/", 1)[1].strip("'")
        dest = remote.split("cat > ")[1].split()[0].strip("'\"")
        streamed.append((rel, dest))
        return "md5:" + rel + " " + dest

    ep = SlurmSSHEndpoint(runner=site, state_dir=tmp_path / "jobs", workdir="/scratch/u/kit", max_su_per_job=500.0,
                          stream_runner=stream_runner)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    kit._delegates["crv_response"] = KU.CrvResponseStream(kit, source=Source(), code="/scratch/u/crvcode",
                                                          background=False)
    return site, ep, kit, streamed


def test_inputs_per_period_and_background():
    p4 = KT.inputs({"period": "P4"})
    assert set(KT.STEPS) <= set(p4) and KT.BACKGROUND in p4 and "r3/dataref/dataref_P4.json" in p4
    assert "timecal/corr_P4_v2mod.json" in p4 and len(set(p4.values())) == len(p4)       # flat names do not collide
    raw = KT.inputs({"period": "raw", "background": 0})
    assert not any("timecal" in k for k in raw) and KT.BACKGROUND not in raw
    for bad in ({"period": "P9"}, {"tables": "all"}):
        with pytest.raises(ValueError):
            KT.inputs(bad)


def test_render_uva_body(tmp_path):
    _, _, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    body, res = d.render(KU.validate({"task": "crv_response", "seed": 3.0, "speed_EX": 1.08, "period": "P3"}))
    assert not re.search(r"@[a-z_0-9]+@", body) and res.gpus == 1
    assert "apptainer exec" in body and "/x/m.sif" in body and "/scratch/u/crvcode" in body
    assert 'X="$L/in"' in body and "inbox" in body and "corr_P3_v2mod.json" in body and "dataref_P3.json" in body
    assert "/dev/shm/crvfs_resp_${USER}_" in body and "in_release" in body
    assert 'KN+=("speed_EX=1.08")' in body and "CRVFS_LOOKUP_DIR=$L" in body
    assert body.index('ln -sfn "$S/in" "$L/in"') < body.index("inbox") < body.index("crvfastsim reco-r3")
    subprocess.run(["bash", "-n"], input=body.encode(), check=True)
    for bad in ({"task": "crv_response", "seed": 1, "bogus": 1}, {"task": "crv_response", "seed": 1, "speed_EX": "1;rm"},
                {"task": "crv_response"}):
        with pytest.raises(ValueError):
            KU.validate(bad)


def test_point_streams_inputs_and_returns_metrics(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    assert kit.check("pt", {"task": "crv_response", "seed": 1, "atten_T1": 0.58}, [], [])["ok"]
    assert kit.submit("pt", {"task": "crv_response", "seed": 1, "atten_T1": 0.58}, [], [])["handle"] == "pt"
    jid = max(site.state)
    d = kit._delegates["crv_response"]
    assert d.stream_step("pt") == "waiting"                               # pending: no inbox yet
    site.state[jid] = "RUNNING"
    shm = "/dev/shm/crvfs_resp_u_abc/in"
    site.files[f"{site.dirs[jid]}/inbox"] = f"udc-x {shm} need\n"
    assert d.stream_step("pt") == "streamed"
    want = KT.inputs({"period": "P4", "background": 1})
    assert sorted(r for r, _ in streamed) == sorted(want)
    assert all(dest.startswith(f"{shm}/") for _, dest in streamed)
    # a second point on the same node (other knobs, same inputs) streams nothing
    n0 = len(streamed)
    kit.submit("pt2", {"task": "crv_response", "seed": 2, "speed_EX": 1.05}, [], [])
    j2 = max(site.state)
    site.state[j2] = "RUNNING"
    site.files[f"{site.dirs[j2]}/inbox"] = f"udc-x {shm} have\n"
    assert d.stream_step("pt2") == "streamed" and len(streamed) == n0
    assert d._has("pt2", "inputsreused")
    site.state[jid] = "COMPLETED"
    assert kit.status("pt")["state"] == "completed"
    r = kit.results("pt")
    assert r["metrics"]["objective"] == 30.1 and r["metrics"]["input_wait_s"] == 95.0
    assert r["metrics"]["inputs_reused"] == 0.0


def test_input_key_depends_on_inputs_not_knobs(tmp_path):
    _, _, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    k = d.input_key(KU.validate({"task": "crv_response", "seed": 1}))
    assert k == d.input_key(KU.validate({"task": "crv_response", "seed": 9, "speed_EX": 1.1}))
    assert k != d.input_key(KU.validate({"task": "crv_response", "seed": 1, "period": "P3"}))
    assert k != d.input_key(KU.validate({"task": "crv_response", "seed": 1, "background": 0}))


def test_registered_as_delegate():
    from mu2efastsim.endpoints import kit as K
    assert K.DELEGATES.get("crv_response") is KU.Delegate
    assert KU.Delegate.applies(type("U", (), {"stream_in": lambda *a: None})())
    assert not KU.Delegate.applies(type("E", (), {"fnal_data": "/exp"})())


def test_static_names_match_the_chain_and_objective():
    """kit_task carries the knob and term names statically (importable without numpy/awkward): keep them in sync."""
    from crvfastsim import chain
    from crvfastsim.objective import crvresponse as R
    assert set(KT.KNOB_PARAMS) == set(chain.KNOBS)
    assert KT._SLOTS == tuple(R.SLOTS)
    assert {n: tuple(s) for n, s in KT._SHAPES.items()} == {n: tuple(R.SLOTS_OF.get(n, R.SLOTS)) for n in R.SHAPES}
