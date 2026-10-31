"""beam_point on a Slurm-over-SSH endpoint against a fake site: delegation, streamed input, md5 check, cleanup gate."""
import io
import hashlib
import shlex
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

        self.cache_ok, self.result_manifest = True, None
        self.manifests = {}
        self.node_files = set()

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
            line = next(line for line in self.body.splitlines() if "artifact-manifest.json" in line)
            self.manifests[self.dirs[j]] = json.loads(shlex.split(line)[2])
            out = f"{j}\n".encode()
        elif s.startswith("squeue"):
            j = s.split("-j ")[1].split()[0]
            out = f"{self.state[j]}|None\n".encode() if self.state[j] in ("PENDING", "RUNNING") else b""
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
            files = {"out/metrics.json": json.dumps({"stop_frac": 0.358, "muons": 1000}).encode(),
                     "out/wait.json": b'{"input_wait_s": 12}', "slurm.out": b"ok\n"}
            if self.cleanup:
                files["out/cleanup.txt"] = b"deleted\n"
            manifest = self.result_manifest if self.result_manifest is not None else self.manifests[s.split("cd ")[1].split(" &&")[0].strip("'")]
            files["out/artifact-manifest.json"] = json.dumps(manifest).encode()
            with tarfile.open(fileobj=buf, mode="w") as t:
                for name, data in files.items():
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
            out = buf.getvalue()
        return subprocess.CompletedProcess(cmd, 0, out, b"")


GEO = "/fnal/geom-area"          # placeholder FASTSIMKIT_GEOMETRY_PREFIX for absolute overlay paths


# Placeholder site configuration (mu2efastsim.site); nothing here names a real site, user or storage area.
SITE = {"FASTSIMKIT_GEOMETRY_PREFIX": GEO, "FASTSIMKIT_FNAL_HOST": "fnal-host.example",
        "FASTSIMKIT_SLURM_HOST": "slurm-host.example", "FASTSIMKIT_SLURM_ACCOUNT": "test_account",
        "FASTSIMKIT_SCRATCH": "/path/to/slurm"}


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    for name, value in SITE.items():
        monkeypatch.setenv(name, value)


class Source(kit_uva._Source):
    def __init__(self):
        super().__init__("gpvm", "/fnal/beam")

    revision = "original"

    def fingerprint(self, rel):
        data = (rel + self.revision).encode()
        return {"md5": hashlib.md5(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}

    def md5(self, rel):
        return self.fingerprint(rel)["md5"]


def make(tmp_path, cleanup=True, md5_ok=True):
    site = Site(cleanup)
    source = Source()
    streamed = []

    def stream_runner(src, host, remote, timeout):
        path = src[-1].split("cat ", 1)[1].strip("'")
        rel = path.split("/fnal/beam/", 1)[1] if path.startswith("/fnal/beam/") else path
        streamed.append((rel, remote.split("cat > ")[1].split()[0].strip("'\"")))
        return (source.md5(rel) if md5_ok else "bad") + " x"

    ep = SlurmSSHEndpoint(runner=site, state_dir=tmp_path / "jobs", workdir="/path/to/slurm/kit", stream_runner=stream_runner)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    kit._delegates["beam_point"] = kit_uva.BeamPointStream(kit, source=source, code="/path/to/slurm/beam", background=False)
    return site, ep, kit, streamed


P = {"task": "beam_point", "geom": "move_extras12.txt", "muons": 105350.0, "seed": 1.0, "repeat": 1.0}


def test_validate():
    assert kit_uva.validate(P)["muons"] == 105350
    for bad in ({**P, "geom": "../x.txt"}, {**P, "geom": "/etc/passwd"}, {**P, "geom": "a b.txt"},
                {**P, "muons": 0}, {**P, "repeat": 1.5}, {k: v for k, v in P.items() if k != "seed"}):
        with pytest.raises(ValueError):
            kit_uva.validate(bad)
    assert kit_uva.validate({**P, "geom": f"{GEO}/x/geom/pt01.txt"})["geom"].endswith("pt01.txt")


def test_uva_delegates_and_eaf_does_not(tmp_path):
    _, ep, kit, _ = make(tmp_path)
    assert kit.check("b", P, [], [])["ok"]                       # FNAL-only as a plain task, but delegated on UVA
    assert kit_task._UVADelegate.applies(ep)
    assert not kit_task._UVADelegate.applies(type("E", (), {"fnal_data": "/fnal/x", "stream_in": 1})())


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
    site.files[f"{site.dirs[j]}/inbox"] = "udc-a1 /dev/shm/beamfs_kit_j7001_9\n"
    kit.status("bp1")
    assert streamed == [("g4ref_full/g4ref.npz", "/dev/shm/beamfs_kit_j7001_9/g4ref.npz")]
    assert any("touch" in c and "/dev/shm/beamfs_kit_j7001_9/ready" in c for c in site.cmds)
    site.state[j] = "COMPLETED"
    res = kit.results("bp1")
    assert res["metrics"]["stop_frac"] == 0.358 and res["metrics"]["input_wait_s"] == 12
    with pytest.raises(ValueError, match="different params"):
        kit.submit("bp1", {**P, "seed": 2.0}, [], [])


def test_overlay_path_streamed_and_md5_mismatch_cancels(tmp_path):
    site, ep, kit, streamed = make(tmp_path, md5_ok=False)
    g = f"{GEO}/x/geom/pt01.txt"
    kit.submit("bp2", {**P, "geom": g}, [], [])
    j = ep.state.get("bp2").job_id
    assert '--geom "$L/geom_overlay.txt"' in site.body
    site.state[j] = "RUNNING"
    site.files[f"{site.dirs[j]}/inbox"] = "udc-a1 /dev/shm/t\n"
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


@pytest.fixture(autouse=True)
def _artifacts(monkeypatch):
    """Only remote artifact preparation is mocked; binding/manifests/verification remain production code."""
    monkeypatch.setattr(kit_uva, "prepare_code", lambda ep, path, ref, **kw: {
        "path": path + ".snapshots/" + "e" * 32, "umbrella_commit": "a" * 40,
        "submodules": {m: "b" * 40 for m in kw["modules"]}, "requested_ref": ref,
        "policy": {"name": "retained-readonly-git-v1", "modules": list(kw["modules"]),
                   "branch": kw.get("branch", "")}})
    monkeypatch.setattr(kit_uva, "resolve_image", lambda ep, path: {
        "path": path, "sha256": "d" * 64, "size": 100, "policy": "verify-before-run-not-snapshot"})


def _start(site, ep, name, tmp="/dev/shm/test-inputs"):
    jid = ep.state.get(name).job_id
    site.state[jid] = "RUNNING"
    site.files[f"{site.dirs[jid]}/inbox"] = f"node1 {tmp} need\n"
    return jid


def test_stable_source_path_replacement_changes_manifest(tmp_path):
    _, _, kit, _ = make(tmp_path)
    d = kit._delegates["beam_point"]
    d.submit("first", P)
    first = d._rec("first")["manifest"]
    d.source.revision = "replaced bytes at same path"
    d.submit("second", P)
    second = d._rec("second")["manifest"]
    assert first["inputs"]["source"] == second["inputs"]["source"]
    assert first["input_sha"] != second["input_sha"]
    assert first["manifest_sha"] != second["manifest_sha"]


def test_retry_uses_frozen_body_ref_image_and_inputs(tmp_path, monkeypatch):
    site, ep, kit, _ = make(tmp_path)
    d = kit._delegates["beam_point"]
    d.submit("retry", P)
    original = d._rec("retry")
    body = site.body
    d.code_ref, d.code, kit.sif = "different-ref", "/different/base", "/different/image"
    d.source.revision = "different bytes"

    def forbidden(*args, **kwargs):
        raise AssertionError("retry must not resolve current artifacts")

    monkeypatch.setattr(d, "ensure_code", forbidden)
    monkeypatch.setattr(d.source, "fingerprint", forbidden)
    monkeypatch.setattr(kit_uva, "resolve_image", forbidden)
    d.submit("retry", P)
    assert d._rec("retry") == original and site.body == body
    assert site.next_id == 7001


def test_stream_mismatch_cancels_against_original_expectations(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    d = kit._delegates["beam_point"]
    d.submit("changed", P)
    d.source.revision = "changed after binding"
    jid = _start(site, ep, "changed")
    assert d.stream_step("changed") == "error"
    assert site.state[jid] == "CANCELLED" and d._has("changed", "streamerr")
    assert "/dev/shm/test-inputs/ready" not in site.node_files


def test_stream_uses_recorded_source_argv_without_remote_resolution(tmp_path, monkeypatch):
    site, ep, kit, streamed = make(tmp_path)
    d = kit._delegates["beam_point"]
    d.submit("pinned", P)
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
    d = kit._delegates["beam_point"]
    d.submit("corrupt", P)
    jid = _start(site, ep, "corrupt")
    site.node_files.add("/dev/shm/test-inputs/ready")
    site.cache_ok = False
    assert d.stream_step("corrupt") == "error"
    assert not streamed and not d._has("corrupt", "inputsreused")
    assert site.state[jid] == "CANCELLED"
    assert "/dev/shm/test-inputs/ready" in site.node_files


def test_fresh_cache_requires_full_verification_before_ready(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    d = kit._delegates["beam_point"]
    d.submit("badsha", P)
    jid = _start(site, ep, "badsha")
    site.cache_ok = False
    assert d.stream_step("badsha") == "error"
    assert streamed and site.state[jid] == "CANCELLED"
    assert "/dev/shm/test-inputs/ready" not in site.node_files


def test_results_manifest_mismatch_refused(tmp_path):
    site, ep, kit, _ = make(tmp_path)
    d = kit._delegates["beam_point"]
    d.submit("wrong-result", P)
    site.state[ep.state.get("wrong-result").job_id] = "COMPLETED"
    site.result_manifest = {"other": "manifest"}
    with pytest.raises(ValueError, match="manifest"):
        d.results("wrong-result")


def test_legacy_record_retry_refused(tmp_path):
    _, _, kit, _ = make(tmp_path)
    d = kit._delegates["beam_point"]
    (d.dir / "legacy.json").write_text(json.dumps({"params": kit_uva.validate(P), "code_ref": "main"}))
    with pytest.raises(ValueError, match="legacy"):
        d.submit("legacy", P)


def test_check_never_resolves_source_bytes(tmp_path, monkeypatch):
    _, _, kit, _ = make(tmp_path)
    d = kit._delegates["beam_point"]

    def forbidden(*args):
        raise AssertionError("check must be pure")

    monkeypatch.setattr(d.source, "fingerprint", forbidden)
    assert d.check("preview", P)["ok"]


def test_result_metadata_exposes_bound_manifest(tmp_path):
    site, ep, kit, _ = make(tmp_path)
    d = kit._delegates["beam_point"]
    d.submit("metadata", P)
    site.state[ep.state.get("metadata").job_id] = "COMPLETED"
    result = d.results("metadata")
    manifest = d._rec("metadata")["manifest"]
    assert result["metadata"]["artifact_manifest"] == manifest
    assert result["metadata"]["manifest_sha"] == manifest["manifest_sha"]


def test_artifact_binding_is_durable_before_dispatch_and_retry(tmp_path, monkeypatch):
    from mu2efastsim.endpoints import EndpointError
    site, ep, kit, _ = make(tmp_path)
    d = kit._delegates["beam_point"]
    real = ep.submit
    attempted = []

    def fail(name, body, resources):
        rec = d._rec(name)
        assert rec["body"] == body and rec["manifest"]
        attempted.append(body)
        raise EndpointError("lost dispatch acknowledgement")

    monkeypatch.setattr(ep, "submit", fail)
    with pytest.raises(ValueError, match="lost dispatch"):
        d.submit("interrupted", P)
    frozen = d._rec("interrupted")
    d.source.revision, d.code_ref, kit.sif = "replacement", "new-ref", "/new.sif"
    monkeypatch.setattr(ep, "submit", real)
    d.submit("interrupted", P)
    assert d._rec("interrupted") == frozen
    assert attempted == [frozen["body"]]
    assert frozen["body"] in site.body


@pytest.mark.parametrize("field", ["host", "account", "partition", "workdir", "source_host", "source_root"])
def test_retry_target_changes_refused(tmp_path, monkeypatch, field):
    _, ep, kit, _ = make(tmp_path)
    d = kit._delegates["beam_point"]
    d.submit("target", P)
    obj, name = (d.source, field.removeprefix("source_")) if field.startswith("source_") else (ep, field)
    try:
        setattr(obj, name, "changed")
    except AttributeError:                     # a read-only property resolved from the site configuration
        monkeypatch.setattr(type(obj), name, "changed")
    with pytest.raises(ValueError, match="target|source"):
        d.submit("target", P)


def test_completed_legacy_results_remain_explicitly_unverified(tmp_path):
    site, ep, kit, _ = make(tmp_path)
    params = P
    task = "beam_point"
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


def geometry_ref(path, gap):
    data = f'#include "Offline/Mu2eG4/geom/geom_common.txt"\ndouble stoppingTarget.deltaZ = {gap};\n'.encode()
    path.write_bytes(data)
    return {"name": "geom", "kind": "geom", "uri": path.resolve().as_uri(),
            "sha256": hashlib.sha256(data).hexdigest(), "metadata": {
                "format": "offline_simpleconfig", "schema_version": 1, "renderer": "offline_simpleconfig",
                "renderer_sha256": "a" * 64}}


def test_geometry_fileref_embedded_in_each_uva_binding(tmp_path):
    site, ep, kit, streams = make(tmp_path)
    params = {k: v for k, v in P.items() if k != "geom"}
    refs = [geometry_ref(tmp_path / f"candidate{i}.txt", 25 + 5*i) for i in range(2)]
    bodies = []
    for i, ref in enumerate(refs):
        name = f"geometry{i}"
        assert kit.check(name, params, [ref], [], ["stop_frac"])["ok"]
        kit.submit(name, params, [ref], [], ["stop_frac"])
        rec = kit._delegates["beam_point"]._rec(name)
        assert rec["manifest"]["geometry"]["reference"]["sha256"] == ref["sha256"]
        assert len(rec["manifest"]["inputs"]["files"]) == 1
        assert '--geom "$M2FS_GEOMETRY"' in rec["body"]
        assert f"stoppingTarget.deltaZ = {25 + 5*i}" in rec["body"]
        result = subprocess.run(["bash", "-n"], input=rec["body"], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        bodies.append(rec["body"])
    assert bodies[0] != bodies[1]
    (tmp_path / "candidate0.txt").unlink()
    kit.submit("geometry0", params, [refs[0]], [], ["stop_frac"])
    assert kit._delegates["beam_point"]._rec("geometry0")["body"] == bodies[0]
    with pytest.raises(ValueError, match="geometry differs"):
        kit.submit("geometry0", params, [refs[1]], [], ["stop_frac"])
    assert len(site.state) == 2 and not streams


def test_uva_geometry_result_evidence_and_files_from(tmp_path, monkeypatch):
    from mu2efastsim.endpoints import geometry as G
    site, ep, kit, _ = make(tmp_path)
    params = {k: v for k, v in P.items() if k != "geom"}
    ref = geometry_ref(tmp_path / "geom.txt", 25)
    kit.submit("geometry", params, [ref], [], ["stop_frac"])
    site.state[ep.state.get("geometry").job_id] = "COMPLETED"
    fetch = ep.fetch
    artifact = kit._request("geometry")["geometry"]
    def with_geometry(name, dest):
        files = fetch(name, dest)
        out = dest / name / "out"
        (out / "geometry.txt").write_text(artifact["text"])
        (out / "geometry-artifact.json").write_text(json.dumps({k:v for k,v in artifact.items() if k != "text"}))
        return files
    monkeypatch.setattr(ep, "fetch", with_geometry)
    result = kit.results("geometry")
    assert len(result["files"]) == 1 and result["files"][0]["kind"] == "geom"
    assert result["metadata"]["diagnostic_files"]
    kit.submit("downstream", params, [], result["files"], ["stop_frac"])
    assert G.identity(kit._request("downstream")["geometry"]) == G.identity(artifact)


def test_uva_geometry_retry_never_renders_current_body(tmp_path, monkeypatch):
    site, ep, kit, _ = make(tmp_path)
    params = {k: v for k, v in P.items() if k != "geom"}
    ref = geometry_ref(tmp_path / "geom.txt", 25)
    kit.submit("geometry", params, [ref], [], ["stop_frac"])
    delegate = kit._delegates["beam_point"]
    body = delegate._rec("geometry")["body"]
    monkeypatch.setattr(delegate, "render", lambda *a, **kw: pytest.fail("retry must use bound body"))
    (tmp_path / "geom.txt").unlink()
    assert kit.check("geometry", params, [ref], [], ["stop_frac"])["ok"]
    kit.submit("geometry", params, [ref], [], ["stop_frac"])
    assert delegate._rec("geometry")["body"] == body and len(site.state) == 1
    delegate.source.root = "/other"
    check = kit.check("geometry", params, [ref], [], ["stop_frac"])
    assert not check["ok"] and "target/source" in check["message"]
    with pytest.raises(ValueError, match="target/source"):
        kit.submit("geometry", params, [ref], [], ["stop_frac"])


def test_source_and_code_fail_closed_without_site_variables(tmp_path, monkeypatch):
    from mu2efastsim import site
    for v in ("FASTSIMKIT_FNAL_HOST", "FASTSIMKIT_BEAM_DATA", "FASTSIMKIT_BEAM_CODE", "FASTSIMKIT_SCRATCH",
              "M2FS_INPUTS", "M2FS_DATA"):
        monkeypatch.delenv(v, raising=False)
    _, _, kit, _ = make(tmp_path)
    d = kit_uva.BeamPointStream(kit, background=False)
    with pytest.raises(site.SiteConfigError, match="FASTSIMKIT_FNAL_HOST"):
        d.source
    with pytest.raises(site.SiteConfigError, match="FASTSIMKIT_SCRATCH"):
        d.code
    monkeypatch.setenv("FASTSIMKIT_FNAL_HOST", "fnal-host")
    monkeypatch.setenv("M2FS_INPUTS", "/inputs")
    monkeypatch.setenv("FASTSIMKIT_SCRATCH", "/path/to/slurm")
    assert (d.source.host, d.source.root, d.code) == ("fnal-host", "/inputs/beam_fastsim", "/path/to/slurm/beam_code_kit")
