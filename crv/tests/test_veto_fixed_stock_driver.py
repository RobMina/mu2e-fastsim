"""Shell-driver contracts, using tiny fake artifacts (no ROOT, CVMFS or physics)."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


DRIVER = Path(__file__).resolve().parents[1] / "scripts/compare_veto_fixed_stock.sh"
BLOCKS = ("block0", "block1", "block2_fullinventory")
INTENDED = "intended_per_sector_v1"
WRONG = "legacy_stock_shared_lut_v1"
REPORT = "paired_intended_per_sector_v1.json"


def put(path, text="fixture\n", executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put_json(path, value):
    return put(path, json.dumps(value))


@pytest.fixture
def rig(tmp_path):
    for tool in ("bash", "jq", "sha256sum"):
        if not shutil.which(tool):
            pytest.skip(f"shell regression requires {tool}")
    code = tmp_path / "worktree"
    area = tmp_path / "satellite"
    backing = tmp_path / "backing"
    base = tmp_path / "retained"
    out = tmp_path / "data/candidate"
    out.mkdir(parents=True)
    conditions = put(tmp_path / "conditions.root")
    selection = put(tmp_path / "selected.parquet")
    reference = put(tmp_path / "reference.json", "{}")
    script = put(code / "scripts/compare_veto_fixed_stock.sh", DRIVER.read_text().replace(
        "/exp/mu2e/data/users/", str(tmp_path / "data") + "/").replace(
        "/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_41_00", str(backing)))
    for name in ("check", "metrics", "policy"):
        put(code / f"crvfastsim/veto/replay_{name}.py", f"# fake {name}\n")
    source = put(area / "CrvFastSim/src/CrvReplayEventFromTable_module.cc")
    put(area / "adapter_source_sha256.txt", f"{digest(source)}  {source.relative_to(area)}\n")
    for path in (area / "build/al9-prof-e29-p109/CrvFastSim/lib/libadapter.so",
                 backing / "build/al9-prof-e29-p109/Offline/lib/libmu2e_CRVResponse.so",
                 backing / "build/al9-prof-e29-p109/Offline/lib/libmu2e_CRVReco.so"):
        put(path)
    (area / "backing").symlink_to(backing, target_is_directory=True)
    for block in BLOCKS:
        table = put(base / block / "mixed.root", block + " input\n")
        put_json(base / block / "mixed.json", {"output_sha256": digest(table)})
        put(base / block / "fast.parquet")
        put_json(base / block / "fast.json", {
            "response_contract": {"policy": INTENDED},
            "input_sha256": digest(table), "conditions_sha256": digest(conditions),
            "stock_shared_table_yields_diagnostic": False,
        })
    for n in (1, 10):
        control = out / f"control{n}"
        stock = put(control / "stock.root", f"control {n}\n")
        table = base / "block0/mixed.root"
        put_json(control / "check.json", {
            "conditions_equal": True, "events_check": {"events": n},
            "entries_highest_cycle": {"events": n},
            "input_sha256": digest(table), "conditions_sha256": digest(conditions),
            "stock_sha256": digest(stock), "stock": str(stock),
        })
        put(control / "provenance.txt", f"area {area}\nbacking {backing}\ntable {table}\nevents {n}\n"
            f"{digest(table)}  {table}\n{digest(source)}  {source}\n")
    bin_dir = tmp_path / "bin"
    launches = tmp_path / "launches.jsonl"
    calls = tmp_path / "python.jsonl"
    put(area / "run_veto_stock_replay.sh", '''#!/bin/bash
set -e
printf '%s\\n' "$*" >> "$MOCK_LAUNCHES"
mkdir "$3"
printf 'replay\\n' > "$3/replay.fcl"
printf 'stock\\n' > "$3/stock.root"
''', executable=True)
    put(bin_dir / "ldd", "#!/bin/bash\necho 'mock dependency => /mock/lib.so'\n", executable=True)
    put(bin_dir / "mu2e", '''#!/bin/bash
set -e
while [ "$#" -gt 0 ]; do
    if [ "$1" = --debug-config ]; then printf 'expanded\\n' > "$2"; shift; fi
    shift
done
''', executable=True)
    put(bin_dir / "python3", f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["MOCK_PYTHON_CALLS"], "a") as log:
    log.write(json.dumps({"args": args, "cwd": os.getcwd(),
                         "pythonpath": os.environ.get("PYTHONPATH", "")}) + "\\n")
assert args[:1] == ["-m"], args
out = pathlib.Path(args[args.index("--out") + 1])
result = {}
if args[1] == "crvfastsim.veto.replay_check":
    result = {"response_identity": {"response_contract": {
        "policy": os.environ.get("MOCK_CHECK_POLICY", "intended_per_sector_v1")}}}
else:
    assert args[1] == "crvfastsim.veto.replay_metrics", args
out.write_text(json.dumps(result))
''', executable=True)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               MUSE_WORK_DIR=str(area), MUSE_STUB="al9-prof-e29-p109", MUSE_ENVSET="p109",
               PYTHONPATH=str(tmp_path / "unrelated-worktree"),
               MOCK_LAUNCHES=str(launches), MOCK_PYTHON_CALLS=str(calls))
    return SimpleNamespace(code=code, area=area, backing=backing, base=base, out=out,
                           conditions=conditions, selection=selection, reference=reference,
                           script=script, env=env, launches=launches, calls=calls, root=tmp_path)


def run(rig, mode="stock", overrides=None, args=None):
    env = dict(rig.env)
    if mode == "check":
        env.pop("MUSE_WORK_DIR", None)
    env.update(overrides or {})
    paths = args or [str(getattr(rig, key)) for key in
                     ("area", "base", "out", "conditions", "selection", "reference")]
    # Deliberately relative: the stock branch changes into the satellite before hashing.
    return subprocess.run(["bash", "worktree/scripts/compare_veto_fixed_stock.sh", mode, *paths],
                          cwd=rig.root, env=env, text=True, capture_output=True, timeout=15)


def retained_outputs(rig):
    for block in BLOCKS:
        put(rig.out / block / "stock.root")


def assert_rejected_before_launch(rig, result):
    assert result.returncode != 0, result.stdout + result.stderr
    assert not rig.launches.exists()
    assert not rig.calls.exists()
    assert not (rig.out / REPORT).exists()


def test_relative_driver_invocation_and_absolute_runtime_hash(rig):
    result = run(rig)
    assert result.returncode == 0, result.stdout + result.stderr
    launches = rig.launches.read_text().splitlines()
    assert len(launches) == 3
    for block, line in zip(BLOCKS, launches):
        assert line.split()[:3] == [str(rig.area), str(rig.base / block / "mixed.root"),
                                    str(rig.out / block)]
        assert (rig.out / block / "completed_sha256.txt").exists()
    assert str(rig.script) in (rig.out / "runtime_libraries.txt").read_text()


@pytest.mark.parametrize("mode,missing", [
    ("stock", "mixed.root"), ("stock", "fast.parquet"),
    ("check", "mixed.root"), ("check", "fast.parquet"),
    ("check", "fast.json"), ("check", "stock.root"),
])
def test_missing_late_block_input_blocks_every_launch(rig, mode, missing):
    if mode == "check":
        retained_outputs(rig)
    root = rig.out if missing == "stock.root" else rig.base
    (root / BLOCKS[-1] / missing).unlink()
    assert_rejected_before_launch(rig, run(rig, mode))


def test_modern_wrong_policy_rejected_in_preflight(rig):
    retained_outputs(rig)
    sidecar = rig.base / BLOCKS[-1] / "fast.json"
    meta = json.loads(sidecar.read_text())
    meta["response_contract"]["policy"] = WRONG
    meta["stock_shared_table_yields_diagnostic"] = True
    put_json(sidecar, meta)
    assert_rejected_before_launch(rig, run(rig, "check"))


def test_bound_check_wrong_policy_cannot_write_intended_report(rig):
    retained_outputs(rig)
    result = run(rig, "check", {"MOCK_CHECK_POLICY": WRONG})
    assert result.returncode != 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in rig.calls.read_text().splitlines()]
    assert calls and all(call["args"][1] == "crvfastsim.veto.replay_check" for call in calls)
    assert not (rig.out / REPORT).exists()


def test_check_uses_worktree_cwd_import_path_and_absolute_arguments(rig):
    retained_outputs(rig)
    result = run(rig, "check")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in rig.calls.read_text().splitlines()]
    assert len(calls) == 4
    for call in calls:
        assert call["cwd"] == str(rig.code)
        assert call["pythonpath"].split(os.pathsep) == [str(rig.code), rig.env["PYTHONPATH"]]
        args = call["args"]
        for flag in ("--input", "--stock", "--conditions", "--fast", "--out", "--selection", "--reference", "--checks"):
            if flag not in args:
                continue
            index = args.index(flag) + 1
            while index < len(args) and not args[index].startswith("--"):
                assert Path(args[index]).is_absolute(), args
                index += 1
    hashes = (rig.out / "check_code_inputs_sha256.txt").read_text()
    assert str(rig.script) in hashes
    assert str(rig.code / "crvfastsim/veto/replay_check.py") in hashes
    assert (rig.out / REPORT).exists()


@pytest.mark.parametrize("target", ["block", "check", "report"])
def test_overwrite_refused_before_work(rig, target):
    mode = "stock" if target == "block" else "check"
    if mode == "check":
        retained_outputs(rig)
    path = rig.out / (f"{BLOCKS[-1]}/sentinel" if target == "block" else
                      f"{BLOCKS[-1]}/check.json" if target == "check" else REPORT)
    put(path, "do not replace\n")
    result = run(rig, mode)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not rig.launches.exists()
    assert not rig.calls.exists()
    assert path.read_text() == "do not replace\n"


@pytest.mark.parametrize("key", ["input_sha256", "conditions_sha256", "stock_sha256", "stock"])
def test_control_is_bound_to_current_inputs_and_exact_stock(rig, key):
    path = rig.out / "control10/check.json"
    control = json.loads(path.read_text())
    control[key] = str(rig.root / "other-stock.root") if key == "stock" else "0" * 64
    put_json(path, control)
    assert_rejected_before_launch(rig, run(rig))


@pytest.mark.parametrize("key", ["area", "backing", "table", "events"])
def test_control_provenance_must_match(rig, key):
    path = rig.out / "control10/provenance.txt"
    lines = path.read_text().splitlines()
    put(path, "\n".join(f"{key} wrong" if line.startswith(key + " ") else line for line in lines) + "\n")
    assert_rejected_before_launch(rig, run(rig))


@pytest.mark.parametrize("key,value", [("MUSE_STUB", "al9-prof-e29-p108"), ("MUSE_ENVSET", "p108")])
def test_wrong_runtime_environment_rejected(rig, key, value):
    assert_rejected_before_launch(rig, run(rig, overrides={key: value}))


@pytest.mark.parametrize("index", range(6))
def test_relative_input_or_output_arguments_rejected(rig, index):
    args = [str(getattr(rig, key)) for key in
            ("area", "base", "out", "conditions", "selection", "reference")]
    args[index] = os.path.relpath(args[index], rig.root)
    assert_rejected_before_launch(rig, run(rig, args=args))
