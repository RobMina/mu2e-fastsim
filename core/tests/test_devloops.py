"""Device-loop and concurrent-region settings (mu2efastsim.devloops): both on by default, importing the package appends
each XLA command-buffer flag unless the user set it or its setting is off; each case runs in a subprocess because the
settings act at import."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBE = ("import json, os, mu2efastsim\nfrom mu2efastsim import devloops as d\n"
         "print(json.dumps({'flags': os.environ.get('XLA_FLAGS'), 'state': d.state()}))")
CLEAR = ("XLA_FLAGS", "MU2EFS_DEVICE_LOOPS", "MU2EFS_GRAPH_CONCURRENCY", "MU2EFS_DEVICE_DETERMINISTIC")
LOOPS = "--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE"
CONC = "--xla_gpu_graph_enable_concurrent_region=true"
CONC_OFF = {"concurrency_requested": False, "concurrency_source": "none", "concurrency_late": False,
            "xla_gpu_graph_enable_concurrent_region": None}
CONC_ON = {"concurrency_requested": True, "concurrency_source": "setting", "concurrency_late": False,
           "xla_gpu_graph_enable_concurrent_region": "true"}


def run(code=PROBE, **env):
    e = {k: v for k, v in os.environ.items() if k not in CLEAR}
    e.update({"PYTHONPATH": str(ROOT), "MU2EFS_DEVICE_DETERMINISTIC": "0", **env})
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=300)


def probe(**env):
    r = run(**env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("off", ["0", "false", "off"])
def test_setting_off_leaves_flags(off):
    out = probe(MU2EFS_DEVICE_LOOPS=off, MU2EFS_GRAPH_CONCURRENCY=off, XLA_FLAGS="--a=1")
    assert out["flags"] == "--a=1"
    assert out["state"] == {"requested": False, "flag_source": "none", "late": False,
                            "xla_gpu_enable_command_buffer": None, **CONC_OFF}


@pytest.mark.parametrize("on", [None, "1", "true", "on"])
def test_default_and_on_append_flags(on):
    env = {} if on is None else {"MU2EFS_DEVICE_LOOPS": on, "MU2EFS_GRAPH_CONCURRENCY": on}
    out = probe(XLA_FLAGS="--a=1", **env)
    assert out["flags"] == f"--a=1 {LOOPS} {CONC}"
    assert out["state"] == {"requested": True, "flag_source": "setting", "late": False,
                            "xla_gpu_enable_command_buffer": "+CONDITIONAL,+WHILE", **CONC_ON}


def test_no_existing_flags():
    out = probe()
    assert out["flags"] == f"{LOOPS} {CONC}"


def test_settings_are_independent():
    out = probe(MU2EFS_DEVICE_LOOPS="0")
    assert out["flags"] == CONC
    assert out["state"]["requested"] is False and out["state"]["concurrency_requested"] is True
    out = probe(MU2EFS_GRAPH_CONCURRENCY="0")
    assert out["flags"] == LOOPS
    assert {k: out["state"][k] for k in CONC_OFF} == CONC_OFF


def test_user_value_of_the_flag_wins():
    out = probe(MU2EFS_DEVICE_LOOPS="1", XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION")
    assert out["flags"] == f"--xla_gpu_enable_command_buffer=FUSION {CONC}"
    assert out["state"]["flag_source"] == "user" and out["state"]["xla_gpu_enable_command_buffer"] == "FUSION"
    out = probe(XLA_FLAGS="--xla_gpu_graph_enable_concurrent_region=false")
    assert out["flags"] == f"--xla_gpu_graph_enable_concurrent_region=false {LOOPS}"
    assert out["state"]["concurrency_source"] == "user"
    assert out["state"]["xla_gpu_graph_enable_concurrent_region"] == "false"


def test_with_the_determinism_flag():
    out = probe(MU2EFS_DEVICE_LOOPS="1", MU2EFS_DEVICE_DETERMINISTIC="1")
    assert out["flags"] == f"--xla_gpu_autotune_level=0 {LOOPS} {CONC}"


@pytest.mark.parametrize("name", ["MU2EFS_DEVICE_LOOPS", "MU2EFS_GRAPH_CONCURRENCY"])
def test_bad_value_refused(name):
    r = run(**{name: "maybe"})
    assert r.returncode != 0 and name in r.stderr


def test_provenance_records_the_settings():
    out = probe(code="import json\nfrom mu2efastsim import precision\nprint(json.dumps(precision.provenance()['device_loops']))",
                MU2EFS_DEVICE_LOOPS="1")
    assert out["requested"] is True and out["xla_gpu_enable_command_buffer"] == "+CONDITIONAL,+WHILE"
    assert out["concurrency_requested"] is True and out["xla_gpu_graph_enable_concurrent_region"] == "true"
