"""crv_response on a Slurm-over-SSH endpoint (UVA): render, the streamed input set, and a full point against a fake site
(core/tests/test_chainpoint.py's Site), without UVA or the network."""
import io
import hashlib
import shlex
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

        self.cache_ok, self.result_manifest = True, None
        self.manifests = {}

    def __call__(self, cmd, stdin, timeout):
        s = cmd[-1]
        out = b""
        if "sbatch --parsable" in s:
            self.next_id += 1
            j = str(self.next_id)
            self.state[j] = "PENDING"
            self.dirs[j] = s.split("cd ")[1].split(" &&")[0].strip("'")
            self.body = stdin.decode()
            line = next(line for line in self.body.splitlines() if "artifact-manifest.json" in line)
            self.manifests[self.dirs[j]] = json.loads(shlex.split(line)[2])
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
        elif s.startswith("srun") and "VERIFIED" in s:
            out = b"VERIFIED\n" if self.cache_ok else b"MISMATCH\n"
        elif s.startswith("srun") and "touch" in s:
            self.node_files.add(s.split("touch ")[1].split()[0].strip("'\""))
        elif s.startswith("srun") and "test -f" in s:
            path = s.split("test -f ")[1].split()[0].strip("'\"")
            out = b"have\n" if path in self.node_files else b""
        elif s.startswith("scancel"):
            self.state[s.split()[1]] = "CANCELLED"
        elif "tar -cf" in s:
            buf = io.BytesIO()
            manifest = self.result_manifest if self.result_manifest is not None else self.manifests[s.split("cd ")[1].split(" &&")[0].strip("'")]
            with tarfile.open(fileobj=buf, mode="w") as t:
                for name, data in {"out/artifact-manifest.json": json.dumps(manifest).encode(),
                                   "out/metrics.json": json.dumps({"objective": 30.1, "objective_sigma": 0.6}).encode(),
                                   "out/cleanup.txt": b"deleted\n", "out/wait.json": b'{"input_wait_s": 95}'}.items():
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
            out = buf.getvalue()
        return subprocess.CompletedProcess(cmd, 0, out, b"")


class Source(FnalSource):
    def __init__(self):
        super().__init__("gpvm", "/exp/data")

    revision = "original"

    def fingerprint(self, rel):
        data = (rel + self.revision).encode()
        return {"md5": hashlib.md5(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}

    def md5(self, rel):
        return self.fingerprint(rel)["md5"]


@pytest.fixture(autouse=True)
def _own_slots(tmp_path, monkeypatch):
    """Tests never take the operator's real ssh slots (a live campaign holds them)."""
    monkeypatch.setattr(KU, "CHANNELS", KU._Slots(4, str(tmp_path / "test_slots")))


def make(tmp_path):
    site, streamed = Site(), []
    source = Source()

    def stream_runner(src, host, remote, timeout):
        rel = src[-1].split("/exp/data/", 1)[1].strip("'")
        dest = remote.split("cat > ")[1].split()[0].strip("'\"")
        streamed.append((rel, dest))
        return source.md5(rel) + " " + dest

    ep = SlurmSSHEndpoint(runner=site, state_dir=tmp_path / "jobs", workdir="/scratch/u/kit", max_su_per_job=500.0,
                          stream_runner=stream_runner)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    kit._delegates["crv_response"] = KU.CrvResponseStream(kit, source=source, code="/scratch/u/crvcode",
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
    assert 'KN+=("speed_EX="1.08)' in body and "CRVFS_LOOKUP_DIR=$L" in body
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


def test_channel_slots_cap_across_processes(tmp_path):
    """At most N holders at once, counted across processes (each graph.run point is its own kit process)."""
    import multiprocessing as mp
    import time as _t
    root = str(tmp_path / "slots")
    peak = mp.Value("i", 0)
    now = mp.Value("i", 0)

    def hold():
        with KU._Slots(2, root):
            with now.get_lock():
                now.value += 1
                peak.value = max(peak.value, now.value)
            _t.sleep(0.3)
            with now.get_lock():
                now.value -= 1

    ps = [mp.get_context("fork").Process(target=hold) for _ in range(5)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(10)
    assert peak.value == 2 and all(p.exitcode == 0 for p in ps)


def test_stream_retries_a_dropped_file(tmp_path, monkeypatch):
    """A stream that fails once (ssh rc 255) is retried; the point streams and is not cancelled."""
    site, ep, kit, streamed = make(tmp_path)
    monkeypatch.setattr(KU, "RETRY_S", 0.0)
    real, fails = ep.stream_in, {"n": 0}

    def flaky(handle, node, dest, argv, timeout=3600):
        if dest.endswith("00001056.root") and fails["n"] == 0:
            fails["n"] += 1
            from mu2efastsim.endpoints import EndpointError
            raise EndpointError("uva: streaming failed (rc 255/255)")
        return real(handle, node, dest, argv, timeout)
    monkeypatch.setattr(ep, "stream_in", flaky)
    kit.submit("rt", {"task": "crv_response", "seed": 1}, [], [])
    jid = max(site.state)
    site.state[jid] = "RUNNING"
    site.files[f"{site.dirs[jid]}/inbox"] = f"udc-x /dev/shm/crvfs_resp_u_r/in need\n"
    assert kit._delegates["crv_response"].stream_step("rt") == "streamed" and fails["n"] == 1
    assert site.state[jid] == "RUNNING"


@pytest.fixture(autouse=True)
def _artifacts(monkeypatch):
    """Only remote artifact preparation is mocked; binding/manifests/verification remain production code."""
    monkeypatch.setattr(KU, "prepare_code", lambda ep, path, ref, **kw: {
        "path": path + ".snapshots/" + "e" * 32, "umbrella_commit": "a" * 40,
        "submodules": {m: "b" * 40 for m in kw["modules"]}, "requested_ref": ref,
        "policy": {"name": "retained-readonly-git-v1", "modules": list(kw["modules"]),
                   "branch": kw.get("branch", "")}})
    monkeypatch.setattr(KU, "resolve_image", lambda ep, path: {
        "path": path, "sha256": "d" * 64, "size": 100, "policy": "verify-before-run-not-snapshot"})


def _start(site, ep, name, tmp="/dev/shm/test-inputs"):
    jid = ep.state.get(name).job_id
    site.state[jid] = "RUNNING"
    site.files[f"{site.dirs[jid]}/inbox"] = f"node1 {tmp} need\n"
    return jid


def test_stable_source_path_replacement_changes_manifest(tmp_path):
    _, _, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    d.submit("first", {"task": "crv_response", "seed": 1})
    first = d._rec("first")["manifest"]
    d.source.revision = "replaced bytes at same path"
    d.submit("second", {"task": "crv_response", "seed": 1})
    second = d._rec("second")["manifest"]
    assert first["inputs"]["source"] == second["inputs"]["source"]
    assert first["input_sha"] != second["input_sha"]
    assert first["manifest_sha"] != second["manifest_sha"]


def test_retry_uses_frozen_body_ref_image_and_inputs(tmp_path, monkeypatch):
    site, ep, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    d.submit("retry", {"task": "crv_response", "seed": 1})
    original = d._rec("retry")
    body = site.body
    d.code_ref, d.code, kit.sif = "different-ref", "/different/base", "/different/image"
    d.source.revision = "different bytes"
    d.branch = "different-branch"

    def forbidden(*args, **kwargs):
        raise AssertionError("retry must not resolve current artifacts")

    monkeypatch.setattr(d, "ensure_code", forbidden)
    monkeypatch.setattr(d.source, "fingerprint", forbidden)
    monkeypatch.setattr(KU, "resolve_image", forbidden)
    d.submit("retry", {"task": "crv_response", "seed": 1})
    assert d._rec("retry") == original and site.body == body
    assert site.next_id == 7001


def test_stream_mismatch_cancels_against_original_expectations(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    d = kit._delegates["crv_response"]
    d.submit("changed", {"task": "crv_response", "seed": 1})
    d.source.revision = "changed after binding"
    jid = _start(site, ep, "changed")
    assert d.stream_step("changed") == "error"
    assert site.state[jid] == "CANCELLED" and d._has("changed", "streamerr")
    assert "/dev/shm/test-inputs/ready" not in site.node_files


def test_stream_uses_recorded_source_argv_without_remote_resolution(tmp_path, monkeypatch):
    site, ep, kit, streamed = make(tmp_path)
    d = kit._delegates["crv_response"]
    d.submit("pinned", {"task": "crv_response", "seed": 1})
    entries = d._rec("pinned")["manifest"]["inputs"]["files"]
    expected = {e["dest"]: e for e in entries}
    seen = []

    def stream(name, host, dest, argv):
        e = expected[dest.rsplit("/", 1)[1]]
        assert argv == e["argv"]
        seen.append(e["dest"])
        return e["md5"]

    def forbidden(*args):
        raise AssertionError("streaming must not resolve a current source")

    monkeypatch.setattr(ep, "stream_in", stream)
    monkeypatch.setattr(d.source, "fingerprint", forbidden)
    monkeypatch.setattr(d.source, "md5", forbidden)
    monkeypatch.setattr(d.source, "argv", forbidden)
    _start(site, ep, "pinned")
    assert d.stream_step("pinned") == "streamed"
    assert sorted(seen) == sorted(expected)


def test_corrupt_ready_cache_refused_without_overwrite(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    d = kit._delegates["crv_response"]
    d.submit("corrupt", {"task": "crv_response", "seed": 1})
    jid = _start(site, ep, "corrupt")
    site.node_files.add("/dev/shm/test-inputs/ready")
    site.cache_ok = False
    assert d.stream_step("corrupt") == "error"
    assert not streamed and not d._has("corrupt", "inputsreused")
    assert site.state[jid] == "CANCELLED"
    assert "/dev/shm/test-inputs/ready" in site.node_files


def test_fresh_cache_requires_full_verification_before_ready(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    d = kit._delegates["crv_response"]
    d.submit("badsha", {"task": "crv_response", "seed": 1})
    jid = _start(site, ep, "badsha")
    site.cache_ok = False
    assert d.stream_step("badsha") == "error"
    assert streamed and site.state[jid] == "CANCELLED"
    assert "/dev/shm/test-inputs/ready" not in site.node_files


def test_results_manifest_mismatch_refused(tmp_path):
    site, ep, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    d.submit("wrong-result", {"task": "crv_response", "seed": 1})
    site.state[ep.state.get("wrong-result").job_id] = "COMPLETED"
    site.result_manifest = {"other": "manifest"}
    with pytest.raises(ValueError, match="manifest"):
        d.results("wrong-result")


def test_legacy_record_retry_refused(tmp_path):
    _, _, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    (d.dir / "legacy.json").write_text(json.dumps({"params": KU.validate({"task": "crv_response", "seed": 1}), "code_ref": "main"}))
    with pytest.raises(ValueError, match="legacy"):
        d.submit("legacy", {"task": "crv_response", "seed": 1})


def test_check_never_resolves_source_bytes(tmp_path, monkeypatch):
    _, _, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]

    def forbidden(*args):
        raise AssertionError("check must be pure")

    monkeypatch.setattr(d.source, "fingerprint", forbidden)
    assert d.check("preview", {"task": "crv_response", "seed": 1})["ok"]


def test_result_metadata_exposes_bound_manifest(tmp_path):
    site, ep, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    d.submit("metadata", {"task": "crv_response", "seed": 1})
    site.state[ep.state.get("metadata").job_id] = "COMPLETED"
    result = d.results("metadata")
    manifest = d._rec("metadata")["manifest"]
    assert result["metadata"]["artifact_manifest"] == manifest
    assert result["metadata"]["manifest_sha"] == manifest["manifest_sha"]


def test_input_key_rehashes_same_path_after_replacement(tmp_path):
    _, _, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    p = KU.validate({"task": "crv_response", "seed": 1})
    before = d.input_key(p)
    d.source.revision = "new bytes"
    assert d.input_key(p) != before


def test_artifact_binding_is_durable_before_dispatch_and_retry(tmp_path, monkeypatch):
    from mu2efastsim.endpoints import EndpointError
    site, ep, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    real = ep.submit
    attempted = []

    def fail(name, body, resources):
        rec = d._rec(name)
        assert rec["body"] == body and rec["manifest"]
        attempted.append(body)
        raise EndpointError("lost dispatch acknowledgement")

    monkeypatch.setattr(ep, "submit", fail)
    with pytest.raises(ValueError, match="lost dispatch"):
        d.submit("interrupted", {"task": "crv_response", "seed": 1})
    frozen = d._rec("interrupted")
    d.source.revision, d.code_ref, kit.sif = "replacement", "new-ref", "/new.sif"
    monkeypatch.setattr(ep, "submit", real)
    d.submit("interrupted", {"task": "crv_response", "seed": 1})
    assert d._rec("interrupted") == frozen
    assert attempted == [frozen["body"]]
    assert frozen["body"] in site.body


@pytest.mark.parametrize("field", ["host", "account", "partition", "workdir", "source_host", "source_root"])
def test_retry_target_changes_refused(tmp_path, field):
    _, ep, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    d.submit("target", {"task": "crv_response", "seed": 1})
    if field.startswith("source_"):
        setattr(d.source, field.removeprefix("source_"), "changed")
    else:
        setattr(ep, field, "changed")
    with pytest.raises(ValueError, match="target|source"):
        d.submit("target", {"task": "crv_response", "seed": 1})


def test_cache_writers_across_delegate_instances_are_serialized(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time
    site, ep, kit, streamed = make(tmp_path)
    d1 = kit._delegates["crv_response"]
    d1.submit("writer1", {"task": "crv_response", "seed": 1})
    d2 = KU.CrvResponseStream(kit, source=Source(), background=False)
    d2.submit("writer2", {"task": "crv_response", "seed": 2})
    _start(site, ep, "writer1")
    _start(site, ep, "writer2")
    real = ep.stream_in

    def slow(*args):
        time.sleep(0.01)
        return real(*args)

    monkeypatch.setattr(ep, "stream_in", slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(d1.stream_step, "writer1"), pool.submit(d2.stream_step, "writer2")]
        assert [f.result() for f in futures] == ["streamed", "streamed"]
    assert len(streamed) == len(d1._rec("writer1")["manifest"]["inputs"]["files"])
    assert d1._has("writer1", "inputsreused") or d2._has("writer2", "inputsreused")


def test_cache_writers_across_processes_are_serialized(tmp_path, monkeypatch):
    import multiprocessing as mp
    import time
    site, ep, kit, _ = make(tmp_path)
    d = kit._delegates["crv_response"]
    for name, seed in (("process1", 1), ("process2", 2)):
        d.submit(name, {"task": "crv_response", "seed": seed})
        _start(site, ep, name)
    entries = d._rec("process1")["manifest"]["inputs"]["files"]
    expected = {e["dest"]: e for e in entries}
    ctx = mp.get_context("fork")
    ready, streams = ctx.Value("i", 0), ctx.Value("i", 0)

    def stream(name, host, dest, argv):
        with streams.get_lock():
            streams.value += 1
        time.sleep(0.01)
        return expected[dest.rsplit("/", 1)[1]]["md5"]

    def node(name, host, cmd, timeout=120):
        if "VERIFIED" in cmd:
            return "VERIFIED"
        if cmd.startswith("test -f"):
            return "have" if ready.value else ""
        if cmd.startswith("touch"):
            ready.value = 1
        return ""

    monkeypatch.setattr(ep, "stream_in", stream)
    monkeypatch.setattr(ep, "run_on_node", node)

    def run(name):
        assert d.stream_step(name) == "streamed"

    children = [ctx.Process(target=run, args=(name,)) for name in ("process1", "process2")]
    for child in children:
        child.start()
    for child in children:
        child.join(10)
        if child.is_alive():
            child.kill()
            child.join()
    assert all(child.exitcode == 0 for child in children)
    assert streams.value == len(entries)
    assert d._has("process1", "inputsreused") or d._has("process2", "inputsreused")


def test_completed_legacy_results_remain_explicitly_unverified(tmp_path):
    site, ep, kit, _ = make(tmp_path)
    params = {"task": "crv_response", "seed": 1}
    task = "crv_response"
    kit.submit("history", params, [], [])
    rec = ep.state.get("history")
    site.state[rec.job_id] = "COMPLETED"
    delegate = kit._delegates[task]
    old = delegate._rec("history")
    (delegate.dir / "history.json").write_text(json.dumps({k: old[k] for k in ("params", "code_ref")}))
    result = kit.results("history")
    assert result["metadata"]["artifact_binding"] == "legacy-unverified"
    assert result["metadata"]["artifact_manifest"] is None
    with pytest.raises(ValueError, match="legacy"):
        kit.submit("history", params, [], [])
