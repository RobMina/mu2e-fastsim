"""Conditional campaign controls without remote execution or production inputs."""
import copy
import json

import pytest

from test_generator_bank_capture import inventory
from valhist import generator_bank_campaign as G

pytestmark = pytest.mark.usefixtures("dcache_site")


def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "DATA", tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(inventory()))
    reused = {2001: "/data/old2001/certificate.json", 2003: "/data/old2003/certificate.json"}
    return {"schema": G.SCHEMA, "scope": G.SCOPE, "assumptions": G.ASSUMPTIONS.copy(),
            "downstream_authorized": False, "capture_authorized": True,
            "selection_independence_certified": False, "candidate": str(candidate),
            "reused": {str(k): v for k, v in reused.items()}, "limits": G.LIMITS.copy(),
            "assignments": G.schedule(inventory(), ["mu2egpvm01", "mu2egpvm02", "mu2egpvm05"], reused),
            "outdir": str(tmp_path / "new"), "authorization": "User approved conditional captures."}


def test_schedule_exact_pending_partitions():
    schedule = G.schedule(inventory(), ["mu2egpvm01", "mu2egpvm02", "mu2egpvm05"], {2001, 2003})
    parts = [s for values in schedule.values() for s in values]
    assert len(parts) == len(set(parts)) == 92
    assert set(parts) == set(range(2001, 2101)) - G.B.MISSING - {2001, 2003}
    assert [len(v) for v in schedule.values()] == [31, 31, 30]


@pytest.mark.parametrize("hosts,reused", [([], {2001, 2003}), (["mu2egpvm01"] * 2, {2001, 2003}),
    (["other"], {2001, 2003}), (["mu2egpvm01"], {2001}), (["mu2egpvm01"], {2001, 2002})])
def test_bad_schedule_rejected(hosts, reused):
    with pytest.raises(ValueError):
        G.schedule(inventory(), hosts, reused)


def test_conditional_contract_and_reuse(tmp_path, monkeypatch):
    p = plan(tmp_path, monkeypatch)
    assert G.validate(p)["events"] == 94_000_000
    assert str(G.captured_path(p, 2001)) == "/data/old2001"
    assert str(G.captured_path(p, 2002)) == str(tmp_path / "new/capture_2002")


@pytest.mark.parametrize("change", ["promote", "selection", "assumption", "scope", "authorize", "omission", "duplicate", "path", "limits"])
def test_contract_tampering_rejected(tmp_path, monkeypatch, change):
    p = copy.deepcopy(plan(tmp_path, monkeypatch))
    if change == "promote":
        p["downstream_authorized"] = True
    elif change == "selection":
        p["selection_independence_certified"] = True
    elif change == "assumption":
        p["assumptions"].pop()
    elif change == "scope":
        p["scope"] = "joint time/energy"
    elif change == "authorize":
        p["capture_authorized"] = False
    elif change == "omission":
        p["assignments"]["mu2egpvm01"].pop()
    elif change == "duplicate":
        p["assignments"]["mu2egpvm02"][0] = p["assignments"]["mu2egpvm01"][0]
    elif change == "path":
        p["outdir"] = "/outside/wrong"
    else:
        p["limits"].pop()
    with pytest.raises(ValueError):
        G.validate(p)


def test_quota_reserve_and_no_disable(monkeypatch):
    monkeypatch.setattr(G, "DATA", "/data-test")
    monkeypatch.setattr(G.os, "getxattr", lambda path, attr: b"100" if attr.endswith("rbytes") else str(G.RESERVE_BYTES + 200).encode(), raising=False)
    assert G.quota(100)["used_bytes"] == 100
    with pytest.raises(ValueError, match="quota"):
        G.quota(101)
    monkeypatch.setattr(G.os, "getxattr", lambda *args: b"0")
    with pytest.raises(ValueError, match="quota"):
        G.quota(0)


def test_wrong_host_cannot_create_attempt(tmp_path, monkeypatch):
    p = plan(tmp_path, monkeypatch)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(p))
    monkeypatch.setattr(G.socket, "gethostname", lambda: "mu2egpvm03.fnal.gov")
    monkeypatch.setattr(G, "load_plan", lambda *args: (path, p))
    with pytest.raises(ValueError, match="host"):
        G.run(path, "digest", "mu2egpvm01")
    assert not (tmp_path / "new").exists()


def test_producer_existing_evidence_never_overwritten(tmp_path):
    (tmp_path / "producer_2001").mkdir()
    protected = tmp_path / "producer_2001/report.json"
    protected.write_text("protected")
    with pytest.raises(FileExistsError):
        G.producer({"outdir": str(tmp_path)}, 2001)
    assert protected.read_text() == "protected"


@pytest.mark.parametrize("change", ["digest", "code", "manifest_path", "drift", "symlink"])
def test_launch_binding_guards(tmp_path, monkeypatch, change):
    p = plan(tmp_path, monkeypatch)
    root = tmp_path / "new"
    root.mkdir()
    path = root / "manifest.json"
    p.update(input_sha256={}, input_sha256_post={}, code_sha256={}, code_sha256_post={})
    monkeypatch.setattr(G, "code_hashes", lambda: {"wrong": "code"} if change == "code" else {})
    if change == "manifest_path":
        path = tmp_path / "elsewhere.json"
    elif change == "drift":
        p["input_sha256_post"] = {"different": "hash"}
    elif change == "symlink":
        link = tmp_path / "link"
        link.symlink_to(root, target_is_directory=True)
        p["outdir"] = str(link)
    path.write_text(json.dumps(p))
    digest = "wrong" if change == "digest" else G.B.V.sha256(path)
    with pytest.raises(ValueError):
        G.load_plan(path, digest)


def test_conflicting_hash_closure_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="conflicting"):
        G.campaign_inputs({"input_sha256": {"same": "a"}, "code_sha256": {"same": "b"}}, path)


@pytest.mark.parametrize("change", ["candidate", "area", "partition", "count", "promotion"])
def test_capture_plan_membership(tmp_path, monkeypatch, change):
    p = plan(tmp_path, monkeypatch)
    p["area"] = "/area"
    path = tmp_path / "new/capture_2002"
    path.mkdir(parents=True)
    cert = {"candidate": p["candidate"], "area": p["area"], "strict_file_certified": True,
            "source": G.B.select_record(inventory(), 2002), "downstream_authorized": False,
            "report": {"event_count": 1_000_000}}
    if change in {"candidate", "area"}:
        cert[change] = "/other"
    elif change == "partition":
        cert["source"]["subrun"] = 2003
    elif change == "count":
        cert["report"]["event_count"] -= 1
    else:
        cert["downstream_authorized"] = True
    (path / "certificate.json").write_text(json.dumps(cert))
    with pytest.raises(ValueError, match="campaign"):
        G.check_certificate(p, 2002)


@pytest.mark.parametrize("changed", ["probe.log", "probe.fcl"])
def test_probe_transitive_dependency_drift_blocks_launch(tmp_path, monkeypatch, changed):
    p = plan(tmp_path, monkeypatch)
    root = tmp_path / "new"
    root.mkdir()
    log, fcl = tmp_path / "probe.log", tmp_path / "probe.fcl"
    log.write_text("evidence")
    fcl.write_text("scope")
    cert = tmp_path / "probe.json"
    cert.write_text(json.dumps({"input_sha256": G.F._hashes([fcl]), "log_sha256": G.F._hashes([log])}))
    bound = G.probe_hashes([{"path": str(cert)}])
    assert set(bound) == {str(cert), str(log), str(fcl)}
    p.update(input_sha256=bound, input_sha256_post=bound.copy(), code_sha256={}, code_sha256_post={})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(p))
    digest = G.B.V.sha256(manifest)
    monkeypatch.setattr(G, "code_hashes", lambda: {})
    (tmp_path / changed).write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        G.load_plan(manifest, digest)
