"""site/{eaf,gpvm,slurm}.env: each derives the site variables from the release, MANIFEST and M2FS_DATA; site/check.py
reports them and fails when a needed one is missing."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SLURM = {"FASTSIMKIT_FNAL_HOST": "fnal", "FASTSIMKIT_SLURM_HOST": "login", "FASTSIMKIT_SLURM_ACCOUNT": "acct",
         "FASTSIMKIT_SCRATCH": "/cluster/work"}


@pytest.fixture
def release(tmp_path):
    inputs = tmp_path / "release"
    for d in ("crv_fastsim/lookup_v6_0", "offline", "beam_fastsim"):
        (inputs / d).mkdir(parents=True)
    manifest = tmp_path / "RELEASE.json"
    manifest.write_text(json.dumps({"schema_version": 1, "release": "test-release", "files": []}))
    data = tmp_path / "data"
    data.mkdir()
    return {"M2FS_INPUTS": str(inputs), "MANIFEST": str(manifest), "M2FS_DATA": str(data), "HOME": str(tmp_path),
            "USER": "someone", "PATH": os.environ["PATH"]}


def source(preset, env, then="env"):
    cmd = f'cd "{ROOT}" && . site/{preset}.env && {then}'
    return subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)


@pytest.mark.parametrize("preset", ["eaf", "gpvm", "slurm"])
def test_preset_derives_paths(preset, release):
    env = dict(release, **(SLURM if preset == "slurm" else {}))
    r = source(preset, env)
    assert r.returncode == 0, r.stderr
    got = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    assert got["M2FS_CODE"] == str(ROOT)
    assert got["CRVFS_DATA"] == release["M2FS_INPUTS"] + "/crv_fastsim"
    assert got["MU2E_SEARCH_PATH"] == release["M2FS_INPUTS"] + "/offline"
    assert got["PYTHONPATH"].startswith(f"{ROOT}/core:{ROOT}/crv:{ROOT}/tracker:{ROOT}/beam")
    assert got["M2FS_VENV"] == release["M2FS_DATA"] + "/venv_fastsim"
    if preset == "slurm":
        assert got["FASTSIMKIT_ENDPOINT"] == "uva" and got["FASTSIMKIT_SIF"] == "/cluster/work/mu2efastsim.sif"
    r = source(preset, env, f"{sys.executable} site/check.py {preset}")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "test-release (0 files)" in r.stdout


@pytest.mark.parametrize("missing", ["M2FS_INPUTS", "MANIFEST", "M2FS_DATA"])
def test_preset_names_a_missing_variable(missing, release):
    env = {k: v for k, v in release.items() if k != missing}
    r = source("gpvm", env)
    assert r.returncode != 0 and missing in r.stderr


def test_slurm_preset_needs_the_account(release):
    r = source("slurm", release)
    assert r.returncode != 0 and "FASTSIMKIT_FNAL_HOST" in r.stderr


def test_check_fails_on_unreadable_input(release):
    env = dict(release, M2FS_INPUTS=release["M2FS_INPUTS"] + "/nonexistent")
    r = subprocess.run(["bash", "-c", f'cd "{ROOT}" && . site/eaf.env && {sys.executable} site/check.py eaf'],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "MISSING" in r.stdout
