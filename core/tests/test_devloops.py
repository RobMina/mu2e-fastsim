"""Command-buffer settings (mu2efastsim.devloops): importing the package appends the device-loop and concurrent-region
flags only when their settings force them on (1); a workflow's select (auto by default) then sets both from the
per-workflow table, with explicit overrides, user XLA_FLAGS values left alone, and the CE chain (W2-W4) never running
with both on. Each case runs in a subprocess because the settings act before the first JAX backend."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLEAR = ("XLA_FLAGS", "MU2EFS_DEVICE_LOOPS", "MU2EFS_GRAPH_CONCURRENCY", "MU2EFS_DEVICE_DETERMINISTIC")
LOOPS = "--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE"
CONC = "--xla_gpu_graph_enable_concurrent_region=true"


def run(code, **env):
    e = {k: v for k, v in os.environ.items() if k not in CLEAR}
    e.update({"PYTHONPATH": str(ROOT), "MU2EFS_DEVICE_DETERMINISTIC": "0", **env})
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=300)


def probe(calls=(), jax_first=False, **env):
    """Import the package, optionally start a JAX backend, run the selects; flags, both states and warnings."""
    code = ("import json, os, warnings, mu2efastsim\nfrom mu2efastsim import devloops as d\n"
            + ("import jax\njax.devices()\n" if jax_first else "") +
            "with warnings.catch_warnings(record=True) as w:\n    warnings.simplefilter('always')\n    pass\n"
            + "".join(f"    d.select{c}\n" for c in calls) +
            "print(json.dumps({'flags': os.environ.get('XLA_FLAGS') or '', 'loops': d.state(),\n"
            "                  'conc': d.concurrency_state(), 'warnings': [str(x.message) for x in w]}))")
    r = run(code, **env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def toks(out):
    return out["flags"].split()


# --- import (no workflow named) -------------------------------------------------------------------------------------

def test_import_appends_nothing_by_default():
    out = probe(XLA_FLAGS="--a=1")
    assert out["flags"] == "--a=1"
    assert out["loops"]["requested"] is False and out["loops"]["setting"] == "auto"
    assert out["conc"]["requested"] is False and out["conc"]["setting"] == "auto"


@pytest.mark.parametrize("on", ["1", "true", "on"])
def test_import_appends_forced_flags(on):
    out = probe(XLA_FLAGS="--a=1", MU2EFS_DEVICE_LOOPS=on, MU2EFS_GRAPH_CONCURRENCY=on)
    assert out["flags"] == f"--a=1 {LOOPS} {CONC}"
    assert out["loops"]["flag_source"] == "setting" and out["conc"]["flag_source"] == "setting"


@pytest.mark.parametrize("name", ["MU2EFS_DEVICE_LOOPS", "MU2EFS_GRAPH_CONCURRENCY"])
def test_bad_value_refused(name):
    r = run("import mu2efastsim", **{name: "maybe"})
    assert r.returncode != 0 and name in r.stderr


def test_with_the_determinism_flag():
    out = probe(MU2EFS_DEVICE_LOOPS="1", MU2EFS_DEVICE_DETERMINISTIC="1")
    assert out["flags"] == f"--xla_gpu_autotune_level=0 {LOOPS}"


def test_provenance_records_both():
    code = ("import json\nfrom mu2efastsim import precision\np = precision.provenance()\n"
            "print(json.dumps([p['device_loops'], p['graph_concurrency']]))")
    r = run(code, MU2EFS_DEVICE_LOOPS="1", MU2EFS_GRAPH_CONCURRENCY="0")
    loops, conc = json.loads(r.stdout.strip().splitlines()[-1])
    assert loops["xla_gpu_enable_command_buffer"] == "+CONDITIONAL,+WHILE" and loops["requested"] is True
    assert conc["xla_gpu_graph_enable_concurrent_region"] is None and conc["requested"] is False


# --- select: the per-workflow table ---------------------------------------------------------------------------------

@pytest.mark.parametrize("workflow, loops, conc", [
    ("W1", True, False), ("W2", False, False), ("W3", False, False), ("W4", False, False), ("W5", False, False),
    ("W6", True, True), ("mixing", True, True), ("W7", False, False), ("W8", False, False),
    ("other", False, False), (None, False, False)])
def test_auto_follows_the_workflow(workflow, loops, conc):
    out = probe([f"({workflow!r})"], XLA_FLAGS="--a=1")
    assert (LOOPS in toks(out)) is loops and (CONC in toks(out)) is conc and "--a=1" in toks(out)
    for rec, on in ((out["loops"], loops), (out["conc"], conc)):
        assert rec["workflow"] == workflow and rec["requested"] is on
        assert rec["decided_by"] == "workflow" and rec["late"] is False
    assert out["warnings"] == []


@pytest.mark.parametrize("env, arg, on", [("1", None, True), ("0", None, False), ("auto", "on", True),
                                          ("auto", "off", False), ("0", "on", True), ("1", "off", False)])
def test_explicit_setting_and_argument_override_the_workflow(env, arg, on):
    workflow = "W8" if on else "W6"           # the opposite of the workflow's own choice for both settings
    out = probe([f"({workflow!r}, {arg!r}, {arg!r})"], MU2EFS_DEVICE_LOOPS=env, MU2EFS_GRAPH_CONCURRENCY=env)
    assert (LOOPS in toks(out)) is on and (CONC in toks(out)) is on
    assert out["loops"]["decided_by"] == out["conc"]["decided_by"] == ("flag" if arg else "setting")


def test_the_first_select_stands_unless_forced():
    out = probe(['("W8", "on")', '("W8")'])
    assert out["loops"]["requested"] is True and out["loops"]["decided_by"] == "flag"
    out = probe(['("W2")', '("mixing")'])
    assert toks(out) == [] and out["loops"]["workflow"] == "W2" and out["conc"]["workflow"] == "W2"
    out = probe(['("W6")', '("W6", None, "off")'])
    assert toks(out) == [LOOPS] and out["conc"]["decided_by"] == "flag"


def test_user_values_stay_for_every_workflow():
    flags = "--xla_gpu_enable_command_buffer=FUSION --xla_gpu_graph_enable_concurrent_region=false"
    for wf in ("W6", "W8"):
        out = probe([f"({wf!r})"], XLA_FLAGS=flags)
        assert out["flags"] == flags
        assert out["loops"]["flag_source"] == "user" and out["conc"]["flag_source"] == "user"


def test_a_select_after_the_backend_exists_is_late_and_warned():
    pytest.importorskip("jax")
    out = probe(['("W6")'], jax_first=True, JAX_PLATFORMS="cpu")
    assert any("already exists" in m for m in out["warnings"])
    assert out["loops"]["late"] is True and out["conc"]["late"] is True and toks(out) == []    # unchanged


# --- the CE chain never runs with device loops and concurrency both on -----------------------------------------------

@pytest.mark.parametrize("workflow", ["W2", "W3", "W4"])
@pytest.mark.parametrize("calls, env", [
    (['({wf!r}, "on", "on")'], {}),
    (['({wf!r})'], {"MU2EFS_DEVICE_LOOPS": "1", "MU2EFS_GRAPH_CONCURRENCY": "1"}),
    (['({wf!r}, "on")'], {"MU2EFS_GRAPH_CONCURRENCY": "1"}),
    (['({wf!r})', '({wf!r}, "on", "on")'], {}),
])
def test_ce_chain_with_loops_drops_concurrency_with_a_warning(workflow, calls, env):
    out = probe([c.format(wf=workflow) for c in calls], **env)
    assert LOOPS in toks(out) and CONC not in toks(out)
    assert out["conc"]["dropped_for"] == workflow and out["conc"]["requested"] is False
    assert any("hangs with both" in m for m in out["warnings"])


def test_ce_chain_concurrency_alone_is_allowed():
    out = probe(['("W2", None, "on")'])
    assert toks(out) == [CONC] and "dropped_for" not in out["conc"] and out["warnings"] == []


def test_ce_chain_user_values_are_not_exempt():
    out = probe(['("W2")'], XLA_FLAGS=f"--xla_gpu_enable_command_buffer=FUSION,+WHILE {CONC}")
    assert toks(out) == ["--xla_gpu_enable_command_buffer=FUSION,+WHILE"] and out["conc"]["dropped_for"] == "W2"


def test_the_combination_stays_for_other_workflows():
    out = probe(['("W6", "on", "on")'], MU2EFS_GRAPH_CONCURRENCY="1")
    assert LOOPS in toks(out) and CONC in toks(out) and out["warnings"] == []


def test_late_ce_chain_select_with_both_on_warns_of_the_hang():
    pytest.importorskip("jax")
    out = probe(['("W2")'], jax_first=True, JAX_PLATFORMS="cpu", MU2EFS_DEVICE_LOOPS="1", MU2EFS_GRAPH_CONCURRENCY="1")
    assert any("hangs the CE" in m for m in out["warnings"]) and CONC in toks(out)


# --- entry points ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("workflow, loops, conc", [("W2", None, None), ("W6", "+CONDITIONAL,+WHILE", "true"),
                                                   ("W8", None, None)])
def test_bench_selects_for_its_workflow(tmp_path, workflow, loops, conc):
    (tmp_path / "toyloops.py").write_text("def step(n):\n    return n\n")
    out_json = tmp_path / "b.json"
    code = ("import json, sys\nfrom mu2efastsim import bench\n"
            f"bench.main(['toyloops:step', '--events', '2', '--passes', '2', '--workflow', {workflow!r}, "
            f"'--out', {str(out_json)!r}])\n"
            f"r = json.load(open({str(out_json)!r}))\nprint(json.dumps([r['device_loops'], r['graph_concurrency']]))")
    r = run(code, PYTHONPATH=f"{ROOT}:{tmp_path}", JAX_PLATFORMS="cpu")
    assert r.returncode == 0, r.stderr[-2000:]
    dl, gc = json.loads(r.stdout.strip().splitlines()[-1])
    assert dl["workflow"] == gc["workflow"] == workflow
    assert dl["xla_gpu_enable_command_buffer"] == loops and gc["xla_gpu_graph_enable_concurrent_region"] == conc


def test_bench_graph_concurrency_flag(tmp_path):
    (tmp_path / "toyloops.py").write_text("def step(n):\n    return n\n")
    out_json = tmp_path / "b.json"
    code = ("import json\nfrom mu2efastsim import bench\n"
            f"bench.main(['toyloops:step', '--events', '2', '--passes', '2', '--workflow', 'W2', '--device-loops', "
            f"'on', '--graph-concurrency', 'on', '--out', {str(out_json)!r}])\n"
            f"print(json.dumps(json.load(open({str(out_json)!r}))['graph_concurrency']))")
    r = run(code, PYTHONPATH=f"{ROOT}:{tmp_path}", JAX_PLATFORMS="cpu")
    assert r.returncode == 0, r.stderr[-2000:]
    gc = json.loads(r.stdout.strip().splitlines()[-1])
    assert gc["dropped_for"] == "W2" and gc["xla_gpu_graph_enable_concurrent_region"] is None
