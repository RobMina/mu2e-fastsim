"""Worker failure publication and side-effect-free submission planning, with no grid access."""
import os
from pathlib import Path
import subprocess

import pytest

ART = Path(__file__).parents[1] / "art"


def executable(path, text):
    path.write_text("#!/usr/bin/env bash\n" + text)
    path.chmod(0o755)


@pytest.mark.parametrize("rc", [0, 1])
def test_worker_only_publishes_success(tmp_path, rc):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    code = tmp_path / "code"
    (code / "beam/fcl").mkdir(parents=True)
    (code / "tasks.txt").write_text("chain smoke /input/gun.txt 0 1 9002 -\n")
    (code / "beam/fcl/export_physics_tables.mac").write_text("")
    subprocess.run(["tar", "czf", str(tmp_path / "input.tgz"), "code"], cwd=tmp_path, check=True)
    executable(bindir / "ifdh", '''case "$1" in
  ls) exit 1 ;;
  mkdir_p) exit 0 ;;
  cp)
    if [ "$2" = /input/code.tgz ]; then cp "$FIXTURE/input.tgz" "$3"
    elif [ "$2" = /input/gun.txt ]; then echo gun > "$3"
    else printf '%s\n' "$3" >> "$FIXTURE/published.txt"; fi ;;
esac
''')
    executable(bindir / "mu2e", '''if [[ "$*" = *--debug-config* ]]; then echo config > job_config.txt; exit 0; fi
printf 'Art has completed and will exit with status %s.\n' "$MOCK_RC"
exit "$MOCK_RC"
''')
    text = (ART / "grid_beam_closure.sh").read_text()
    text = text.replace("source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true", "muse() { return 0; }; setup() { return 0; }")
    text = text.replace("/usr/bin/time -v mu2e", "mu2e").replace("sha256sum", "shasum -a 256")
    script = tmp_path / "worker.sh"
    script.write_text(text)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", FIXTURE=str(tmp_path),
               CODE_TARBALL="/input/code.tgz", PROCESS="0", MOCK_RC=str(rc))
    result = subprocess.run(["bash", str(script), "/out"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == rc, result.stdout + result.stderr
    if rc:
        assert not (tmp_path / "published.txt").exists()
        assert "no success archive" in result.stdout
    else:
        assert (tmp_path / "published.txt").read_text().splitlines() == ["/out/chain_smoke_0000000.tgz", "/out/chain_smoke_0000000.ok"]
        assert (tmp_path / "work/chain_smoke_0000000/completion.txt").is_file()


def test_submit_dry_run_does_not_stage_or_contact_grid(tmp_path):
    core = tmp_path / "core"
    (core / "mu2efastsim").mkdir(parents=True)
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("\nfoilscan moved_edge -13 1000 9 full_finest\n")
    stage = f"/exp/mu2e/data/users/ram2aq/beam_dry_test_{tmp_path.name}"
    env = dict(os.environ, BEAMFS_CORE=str(core))
    result = subprocess.run(["bash", str(ART / "submit_beam_closure.sh"), stage, str(tasks), "--test"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tasks: 1" in result.stdout and "jobsub_submit" in result.stdout
    assert not Path(stage).exists()
