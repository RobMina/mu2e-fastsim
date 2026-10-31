"""Frozen grid setup stays dry by default, preserves identity ranges, and refuses unsafe promotion."""
import json
from pathlib import Path
import subprocess

import pytest

from crvfastsim.analyses import run2_grid_plan as P

pytestmark = pytest.mark.usefixtures("dcache_site")

CODE = Path(__file__).resolve().parents[1]


def prepared(tmp_path):
    state = tmp_path / "prepared"
    campaign = P.prepare(state, "/dcache-test/mu2e/scratch/users/test/crv/reference", "test")
    return state, campaign


def test_preparation_is_disjoint_and_frozen(tmp_path):
    state, c = prepared(tmp_path)
    assert len(c["shards"]) == 380
    assert sum(s["generated_events"] for s in c["shards"]) == 19000000
    assert c["shards"][0] == {"subrun": 121, "seed": 20265220, "generated_events": 50000}
    assert c["shards"][-1]["seed"] == 20265599
    assert c["phases"]["remainder"]["jobs"] == 377
    assert c["musing"] == "MDC2025ay"
    assert c["write_scope"] == "/mu2e/scratch/users/test/crv/reference"
    assert set(c["payload_sha256"]) == set(P.FILES) | {"grid_run2_extension.sh"}
    assert not (state / "test.attempted").exists()
    assert subprocess.run(["sha256sum", "-c", "payload.sha256"], cwd=state / "payload").returncode == 0
    with pytest.raises(FileExistsError):
        P.prepare(state, c["outdir"], "test")


@pytest.mark.parametrize("out", ["/local/data/test", "/dcache-test/mu2e/scratch/users/test/../bad", "x;touch bad"])
def test_unsafe_output_refused_before_creation(tmp_path, out):
    with pytest.raises(ValueError):
        P.prepare(tmp_path / "unused", out, "test")
    assert not (tmp_path / "unused").exists()


def test_dry_run_needs_no_credentials_or_grid_commands(tmp_path):
    state, _ = prepared(tmp_path)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(["bash", str(CODE / "grid/submit_run2_extension.sh"), str(state), "test"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY RUN" in result.stdout
    assert "--need-storage-modify=/mu2e/scratch/users/test/crv/reference" in result.stdout
    assert "FIRST_SUBRUN=121" in result.stdout and "--memory=3000MB" in result.stdout
    assert not (state / "test.attempted").exists()
    assert not (state / "submission.lock").exists()


def test_mutated_payload_refuses_dry_run(tmp_path):
    state, _ = prepared(tmp_path)
    (state / "payload/ref_sim_run2.fcl").write_text("changed")
    result = subprocess.run(["bash", str(CODE / "grid/submit_run2_extension.sh"), str(state), "test"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "jobsub_submit" not in result.stdout


def test_gate_requires_persisted_evidence(tmp_path):
    state, _ = prepared(tmp_path)
    sha = P.digest(state / "payload/campaign.json")
    evidence = tmp_path / "evidence.json"
    evidence.write_text("measured test evidence")
    shard = {"subrun": 121, "seed": 20265220, "generated_events": 50000, "accepted_events": 100,
             "highest_cycle_counts_verified": True, "generator_values_verified": True,
             "copyout_verified": True, "runtime_verified": True,
             "evidence": [{"path": str(evidence), "sha256": P.digest(evidence)}]}
    gate = tmp_path / "gate.json"
    row = {"schema": "crv-run2-extension-gate-v1", "campaign_sha256": sha, "shards": [shard]}
    gate.write_text(json.dumps(row))
    assert P.validate_gate(gate, sha, [121])["shards"][0]["accepted_events"] == 100
    shard["highest_cycle_counts_verified"] = False
    gate.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="persisted"):
        P.validate_gate(gate, sha, [121])
    shard["highest_cycle_counts_verified"] = True
    gate.write_text(json.dumps(row))
    evidence.write_text("mutated")
    with pytest.raises(ValueError, match="evidence"):
        P.validate_gate(gate, sha, [121])


def test_gate_rejects_missing_or_duplicate_shards(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"schema": "crv-run2-extension-gate-v1", "campaign_sha256": "sha", "shards": []}))
    with pytest.raises(ValueError, match="every prior"):
        P.validate_gate(gate, "sha", [121])


def test_worker_policy_no_exp_build_or_timeout():
    worker = (CODE / "grid/grid_run2_extension.sh").read_text()
    assert "muse build" not in worker and "muse tarball" not in worker
    assert "timeout " not in worker   # site paths: tests/test_no_site_paths.py
    assert "partial unreceipted" in worker
    assert "persisted_event_counts_verified': False" in worker
    assert worker.index('ifdh cp "$GEN"') < worker.index("ifdh cp complete.json")
    assert 'cmp "$0" fcl/grid_run2_extension.sh' in worker
