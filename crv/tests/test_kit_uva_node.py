"""crv_response's per-node input cache (kit_uva.WAIT) in a real bash with flock (Linux; skipped elsewhere): the first
job creates the cache, a second one shares it, and the last one to exit deletes it and records the deletion."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import kit_uva as KU          # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("flock") is None, reason="flock (util-linux) not available")


def _job(d, jid, extra):
    w = KU.WAIT.replace("@inkey@", "k1").replace("@waits@", "5").replace("/dev/shm", f"{d}/shm")
    head = w[:w.index("INSTATE=")]
    sh = (f'squeue() {{ for j in $RUNNING; do [ "$3" = "$j" ] && echo $j; done; return 0; }}\n'
          f'USER=t; SLURM_JOB_ID={jid}; M2FS_OUT={d}/out{jid}; mkdir -p $M2FS_OUT; L={d}/L{jid}; mkdir -p $L\n'
          + head + extra)
    return subprocess.run(["bash", "-c", sh], capture_output=True, text=True, env={**os.environ, "RUNNING": "1 2"})


def test_node_cache_shared_then_deleted_by_last_holder(tmp_path):
    (tmp_path / "shm").mkdir()
    cache = tmp_path / "shm" / "crvfs_resp_t_k1"
    # job 1 holds the cache while job 2 runs: model it by keeping job 1's holder file (its trap fires at its own exit)
    r1 = _job(tmp_path, 1, 'touch "$S/in/ready"; trap - EXIT\n')
    assert r1.returncode == 0, r1.stderr
    assert (cache / "in" / "ready").exists() and (cache / "holders" / "1").exists()
    r2 = _job(tmp_path, 2, '[ -L "$L/in" ] && [ -f "$L/in/ready" ] && echo shared\n')
    assert r2.returncode == 0 and "shared" in r2.stdout, r2.stderr
    assert cache.exists() and not (cache / "holders" / "2").exists()          # job 1 still holds it
    # job 1 left the queue without releasing (killed): the next job drops its stale holder and, exiting, deletes
    env_running = "3"
    w = KU.WAIT.replace("@inkey@", "k1").replace("@waits@", "5").replace("/dev/shm", f"{tmp_path}/shm")
    sh = (f'squeue() {{ for j in $RUNNING; do [ "$3" = "$j" ] && echo $j; done; return 0; }}\n'
          f'USER=t; SLURM_JOB_ID=3; M2FS_OUT={tmp_path}/out3; mkdir -p $M2FS_OUT; L={tmp_path}/L3; mkdir -p $L\n'
          + w[:w.index("INSTATE=")])
    r3 = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, env={**os.environ, "RUNNING": env_running})
    assert r3.returncode == 0, r3.stderr
    assert not cache.exists()
    assert "deleted node inputs" in (tmp_path / "out3" / "cleanup.txt").read_text()
