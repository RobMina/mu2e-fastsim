"""Device-loop setting (mu2efastsim.devloops): importing the package appends the XLA command-buffer flag unless the
user set it or the setting is off; a workflow's select (auto by default) then keeps or removes it for that workflow.
Each case runs in a subprocess because the setting acts before the first JAX backend."""
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
    e.update({"PYTHONPATH": str(ROOT), "MU2EFS_DEVICE_DETERMINISTIC": "0", **env})
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=300)


def probe(**env):
    r = run(**env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("off", ["0", "false", "off"])
def test_setting_off_leaves_flags(off):
    out = probe(MU2EFS_DEVICE_LOOPS=off, XLA_FLAGS="--a=1")
    assert out["flags"] == "--a=1"
    assert out["state"] == {"setting": "off", "workflow": None, "requested": False, "flag_source": "none",
                            "late": False, "xla_gpu_enable_command_buffer": None}


@pytest.mark.parametrize("on", [None, "1", "true", "on"])
def test_default_and_on_append_flag(on):
    out = probe(XLA_FLAGS="--a=1", **({} if on is None else {"MU2EFS_DEVICE_LOOPS": on}))
    assert out["flags"] == "--a=1 --xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE"
    assert out["state"] == {"setting": "auto" if on is None else "on", "workflow": None, "requested": True,
                            "flag_source": "setting", "late": False,
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


def select_probe(calls, **env):
    code = ("import json, os, warnings, mu2efastsim\nfrom mu2efastsim import devloops as d\n"
            "warnings.simplefilter('error')\n" + "".join(f"d.select{c}\n" for c in calls) +
            "print(json.dumps({'flags': os.environ.get('XLA_FLAGS'), 'state': d.state()}))")
    return probe(code=code, **env)


@pytest.mark.parametrize("workflow, on", [("W2", False), ("W6", True), ("mixing", True), ("W5", True), (None, True)])
def test_auto_follows_the_workflow(workflow, on):
    out = select_probe([f"({workflow!r})"], XLA_FLAGS="--a=1")
    assert out["flags"] == ("--a=1 --xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE" if on else "--a=1")
    assert out["state"]["workflow"] == workflow and out["state"]["requested"] is on
    assert out["state"]["decided_by"] == "workflow" and out["state"]["late"] is False


@pytest.mark.parametrize("env, flag, on", [("1", None, True), ("0", None, False), ("auto", "on", True),
                                           ("auto", "off", False), ("0", "on", True), ("1", "off", False)])
def test_explicit_setting_and_flag_override_the_workflow(env, flag, on):
    workflow = "W2" if on else "mixing"           # the opposite of the workflow's own choice
    out = select_probe([f"({workflow!r}, {flag!r})"], MU2EFS_DEVICE_LOOPS=env)
    assert ("+CONDITIONAL,+WHILE" in (out["flags"] or "")) is on
    assert out["state"]["decided_by"] == ("flag" if flag else "setting")


def test_the_first_select_stands_unless_forced():
    out = select_probe(['("W2", "on")', '("W2")'])
    assert out["state"]["requested"] is True and out["state"]["decided_by"] == "flag"
    out = select_probe(['("W2")', '("mixing")'])
    assert out["state"]["requested"] is False and out["state"]["workflow"] == "W2"
    out = select_probe(['("W2")', '("W2", "on")'])
    assert out["state"]["requested"] is True


def test_user_value_of_the_flag_stays_for_every_workflow():
    out = select_probe(['("W2")'], XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION")
    assert out["flags"] == "--xla_gpu_enable_command_buffer=FUSION" and out["state"]["flag_source"] == "user"


def test_a_select_after_the_backend_exists_is_late_and_warned():
    code = ("import json, os, warnings, mu2efastsim, jax\nfrom mu2efastsim import devloops as d\njax.devices()\n"
            "with warnings.catch_warnings(record=True) as w:\n    warnings.simplefilter('always')\n"
            "    d.select('W2')\n"
            "print(json.dumps({'warned': any('already exists' in str(x.message) for x in w), 'state': d.state()}))")
    pytest.importorskip("jax")
    out = probe(code=code, JAX_PLATFORMS="cpu")
    assert out["warned"] and out["state"]["late"] is True
    assert out["state"]["xla_gpu_enable_command_buffer"] == "+CONDITIONAL,+WHILE"   # unchanged


def test_bench_selects_for_its_workflow(tmp_path):
    (tmp_path / "toyloops.py").write_text("def step(n):\n    return n\n")
    code = ("import json, sys\nfrom mu2efastsim import bench\n"
            f"bench.main(['toyloops:step', '--events', '2', '--passes', '2', '--workflow', 'W2', '--out', {str(tmp_path / 'b.json')!r}])\n"
            f"print(json.dumps(json.load(open({str(tmp_path / 'b.json')!r}))['device_loops']))")
    r = run(code=code, PYTHONPATH=f"{ROOT}:{tmp_path}", JAX_PLATFORMS="cpu")
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["workflow"] == "W2" and out["requested"] is False and out["xla_gpu_enable_command_buffer"] is None

