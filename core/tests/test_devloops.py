"""Device-loop setting (mu2efastsim.devloops): off by default; on, importing the package appends the XLA
command-buffer flag unless the user set it; each case runs in a subprocess because the setting acts at import."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBE = ("import json, os, mu2efastsim\nfrom mu2efastsim import devloops as d\n"
         "print(json.dumps({'flags': os.environ.get('XLA_FLAGS'), 'state': d.state()}))")
CLEAR = ("XLA_FLAGS", "MU2EFS_DEVICE_LOOPS", "MU2EFS_DEVICE_DETERMINISTIC")


def run(code=PROBE, **env):
    e = {k: v for k, v in os.environ.items() if k not in CLEAR}
    e.update(PYTHONPATH=str(ROOT), MU2EFS_DEVICE_DETERMINISTIC="0", **env)
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=300)


def probe(**env):
    r = run(**env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_default_leaves_flags():
    out = probe(XLA_FLAGS="--a=1")
    assert out["flags"] == "--a=1"
    assert out["state"] == {"requested": False, "flag_source": "none", "late": False,
                            "xla_gpu_enable_command_buffer": None}


@pytest.mark.parametrize("on", ["1", "true", "on"])
def test_setting_on_appends_flag(on):
    out = probe(MU2EFS_DEVICE_LOOPS=on, XLA_FLAGS="--a=1")
    assert out["flags"] == "--a=1 --xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE"
    assert out["state"] == {"requested": True, "flag_source": "setting", "late": False,
                            "xla_gpu_enable_command_buffer": "+CONDITIONAL,+WHILE"}


def test_user_value_of_the_flag_wins():
    out = probe(MU2EFS_DEVICE_LOOPS="1", XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION")
    assert out["flags"] == "--xla_gpu_enable_command_buffer=FUSION"
    assert out["state"]["flag_source"] == "user" and out["state"]["xla_gpu_enable_command_buffer"] == "FUSION"


def test_with_the_determinism_flag():
    out = probe(MU2EFS_DEVICE_LOOPS="1", MU2EFS_DEVICE_DETERMINISTIC="1")
    assert out["flags"] == "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE"


def test_bad_value_refused():
    r = run(MU2EFS_DEVICE_LOOPS="maybe")
    assert r.returncode != 0 and "MU2EFS_DEVICE_LOOPS" in r.stderr


def test_provenance_records_the_setting():
    out = probe(code="import json\nfrom mu2efastsim import precision\nprint(json.dumps(precision.provenance()['device_loops']))",
                MU2EFS_DEVICE_LOOPS="1")
    assert out["requested"] is True and out["xla_gpu_enable_command_buffer"] == "+CONDITIONAL,+WHILE"
