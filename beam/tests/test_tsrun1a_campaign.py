"""Exercise the campaign launcher with a fake runner, never art or Slurm."""
import os
from pathlib import Path
import subprocess

import pytest

from beamfastsim import tssmoke

SCRIPT = Path(tssmoke.__file__).with_name("tsrun1a_campaign.sbatch")


def campaign(tmp_path):
    root = tmp_path / "campaign"
    (root / "config").mkdir(parents=True)
    (root / "shards").mkdir()
    (root / "config/tsrun1a_raw.fcl").write_text(SCRIPT.with_name("tsrun1a_raw.fcl").read_text())
    (root / "config/tsrun1a_raw.fcl").chmod(0o400)
    (root / "config/offline_uva.sh").write_text(
        'printf "%s\\n" "$@" > invoked.txt\n'
        'printf "%s\\n" "$OFFLINE_EVENTS" "$OFFLINE_OUTPUT_FILE" > environment.txt\n'
        'exit "${MOCK_RUNNER_EXIT:-0}"\n')
    init = tmp_path / "bash_init"
    init.write_text("module() { return 0; }\n")
    return root, {**os.environ, "TS_CAMPAIGN_ROOT": str(root), "CVMFSEXEC_DIST": str(tmp_path),
                  "BASH_ENV": str(init), "SLURM_JOB_ID": "12345", "MOCK_RUNNER_EXIT": "0"}


def run(env, index):
    return subprocess.run(["bash", str(SCRIPT)], env={**env, "SLURM_ARRAY_TASK_ID": str(index)},
                          capture_output=True, text=True)


@pytest.mark.parametrize("index,cohort", [(0, "development"), (99, "development"),
                                          (100, "heldout"), (199, "heldout")])
def test_distinct_shards_seed_spacing_and_runtime(tmp_path, index, cohort):
    root, env = campaign(tmp_path)
    result = run(env, index)
    assert result.returncode == 0, result.stderr
    shard = root / f"shards/{index:03d}"
    fcl = (shard / "shard.fcl").read_text()
    assert f"source.firstSubRun: {920000 + index}" in fcl
    assert f"services.SeedService.baseSeed: {920001 + 1000 * index}" in fcl
    assert "source.maxEvents: 5000" in fcl
    assert f"001430_{920000 + index:08d}.art" in fcl
    assert "SelectEvents" not in fcl and "saveAllPDG: false" in fcl
    row = (shard / "shard.tsv").read_text().splitlines()[1].split("\t")
    assert row == [str(index), cohort, "1430", str(920000 + index),
                   str(920001 + 1000 * index), "5000", "12345"]
    assert (shard / "environment.txt").read_text().splitlines() == ["5000", str(shard / "tsrun1a_raw.root")]
    assert (shard / "invoked.txt").read_text().splitlines() == [
        "/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ac", str(shard / "shard.fcl")]
    assert run(env, index).returncode != 0
    assert (shard / "shard.fcl").read_text() == fcl


@pytest.mark.parametrize("index", [-1, 200, "00", "abc", "1; exit 0"])
def test_invalid_array_index_rejected(tmp_path, index):
    root, env = campaign(tmp_path)
    assert run(env, index).returncode != 0
    assert not list((root / "shards").iterdir())


def test_runner_failure_propagates(tmp_path):
    _, env = campaign(tmp_path)
    assert run({**env, "MOCK_RUNNER_EXIT": "42"}, 0).returncode == 42


def test_bounded_array_and_no_build():
    script = SCRIPT.read_text()
    assert "#SBATCH --array=0-199%32" in script
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --no-requeue" in script
    assert "muse build" not in script and "sbatch " not in script
    subprocess.run(["bash", "-n", SCRIPT], check=True)


@pytest.mark.parametrize("launcher", ["campaign", "smoke"])
@pytest.mark.parametrize("inherited", [None, "0", "1"])
def test_generation_disables_inherited_warm_repeat(tmp_path, launcher, inherited):
    root, env = campaign(tmp_path)
    runner = root / "config/offline_uva.sh"
    runner.write_text('printf "%s\\n" "${OFFLINE_WARM_REPEAT:-unset}" > warm-repeat.txt\n')
    env.pop("OFFLINE_WARM_REPEAT", None)
    if inherited is not None:
        env["OFFLINE_WARM_REPEAT"] = inherited
    if launcher == "campaign":
        result = run(env, 0)
        out = root / "shards/000"
    else:
        out = tmp_path / "smoke"
        env.update(TS_SMOKE_OUT=str(out), TS_SMOKE_FCL=str(root / "config/tsrun1a_raw.fcl"),
                   OFFLINE_RUNNER=str(runner))
        result = subprocess.run(["bash", str(SCRIPT.with_name("tsrun1a_smoke.sbatch"))],
                                env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (out / "warm-repeat.txt").read_text() == "0\n"
