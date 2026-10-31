"""Run the real shell control flow against fake local setup, art and copy commands, never CVMFS or pnfs."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from crvfastsim.analyses import run2_grid_plan as P

pytestmark = pytest.mark.usefixtures("dcache_site")

CODE = Path(__file__).resolve().parents[1]


def worker_fixture(tmp_path):
    state = tmp_path / "state"
    P.prepare(state, "/dcache-test/mu2e/scratch/users/test/crv/reference", "test")
    binpath = tmp_path / "bin"
    binpath.mkdir()
    setup = tmp_path / "setup.sh"
    work = tmp_path / "Musings/SimJob/MDC2025ay"
    work.mkdir(parents=True)
    offline = tmp_path / "Musings/Offline/v13_39_00"
    offline.mkdir(parents=True)
    (work / "backing").symlink_to(offline, target_is_directory=True)
    setup.write_text(f'export MUSE_STUB=al9-prof-e29-p107 MUSE_WORK_DIR="{work}"\n'
                     'muse() { :; }\nsetup() { :; }\n')
    script = (CODE / "grid/grid_run2_extension.sh").read_text().replace(
        "source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh", f'source "{setup}"').replace(
        "/usr/bin/time -v", "fake_time -v")
    frozen = state / "payload/grid_run2_extension.sh"
    frozen.write_text(script)
    frozen.chmod(0o755)
    hashes = (state / "payload/payload.sha256").read_text().splitlines()
    campaign_path = state / "payload/campaign.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["payload_sha256"]["grid_run2_extension.sh"] = P.digest(frozen)
    out = tmp_path / "output"
    campaign["outdir"] = str(out)
    campaign_path.write_text(json.dumps(campaign))
    replacements = {"grid_run2_extension.sh": P.digest(frozen), "campaign.json": P.digest(campaign_path)}
    (state / "payload/payload.sha256").write_text("\n".join(
        f'{replacements[x.split()[-1]]}  {x.split()[-1]}' if x.split()[-1] in replacements else x
        for x in hashes) + "\n")
    ifdh = binpath / "ifdh"
    ifdh.write_text('''#!/usr/bin/env python3
import pathlib, shutil, sys
op, *args = sys.argv[1:]
if op == 'cp':
    shutil.copyfile(args[0], args[1])
elif op == 'ls':
    p = pathlib.Path(args[0])
    if not p.exists():
        sys.exit(1)
    print('\\n'.join(str(x) for x in p.iterdir()) if p.is_dir() else p)
else:
    raise ValueError(op)
''')
    mu2e = binpath / "mu2e"
    mu2e.write_text('''#!/usr/bin/env python3
import pathlib, re, sys
if '--version' in sys.argv:
    print('synthetic runtime')
elif '--debug-config' in sys.argv:
    pathlib.Path(sys.argv[sys.argv.index('--debug-config') + 1]).write_text('resolved synthetic')
else:
    for name in re.findall(r'fileName : "([^"]+)"', pathlib.Path('fcl/shard.fcl').read_text()):
        pathlib.Path(name).write_bytes(b'fake art, not physics')
    print('TrigReport Events total = 50000 passed = 100 failed = 49900')
''')
    timing = binpath / "fake_time"
    timing.write_text('#!/bin/bash\nshift\n"$@"\nrc=$?\nprintf "Exit status: %s\\n" "$rc"\nexit "$rc"\n')
    for path in (ifdh, mu2e, timing):
        path.chmod(0o755)
    out = tmp_path / "output"
    out.mkdir()
    for name in ("logs", "receipts"):
        (out / name).mkdir()
    env = dict(os.environ, PATH=str(binpath) + os.pathsep + os.environ["PATH"], PROCESS="0", FIRST_SUBRUN="121",
               NEVTS="50000", BASESEED="20265100", OWNER="test", TAG="v1", PAYLOAD_PNFS=str(state / "payload"),
               CAMPAIGN_SHA256=P.digest(state / "payload/campaign.json"), FHICL_FILE_PATH="", MU2E_SEARCH_PATH="")
    return frozen, out, env


def launch(script, out, env, work):
    work.mkdir()
    return subprocess.run(["bash", str(script), str(out)], cwd=work, env=env, text=True, capture_output=True)


def test_worker_publishes_last_then_skips_retry(tmp_path):
    script, out, env = worker_fixture(tmp_path)
    first = launch(script, out, env, tmp_path / "first")
    assert first.returncode == 0, first.stdout + first.stderr
    receipt = json.loads((out / "receipts/complete.001400_00000121.json").read_text())
    assert receipt["copyout_complete"] and receipt["generated_events"] == 50000
    assert not receipt["persisted_event_counts_verified"]
    assert receipt["seed"] == 20265220
    second = launch(script, out, env, tmp_path / "second")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "SKIP:" in second.stdout
    assert not (tmp_path / "second/mu2e.log").exists()


def test_worker_refuses_partial_existing_output(tmp_path):
    script, out, env = worker_fixture(tmp_path)
    path = out / "dts.test.CosmicCRYrun2Ref.v1.001400_00000121.art"
    path.write_bytes(b'partial')
    result = launch(script, out, env, tmp_path / "work")
    assert result.returncode != 0
    assert "partial unreceipted" in result.stdout
    assert path.read_bytes() == b'partial'
    assert not (out / "receipts/complete.001400_00000121.json").exists()
