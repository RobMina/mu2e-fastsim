"""Quiet pair orchestration only: no real art, containers, or Slurm submission."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = r'''import hashlib, json, os, pathlib, sys
kind, args = sys.argv[1], sys.argv[2:]
e = os.environ
with open(e['CALLS'], 'a') as out:
    out.write(json.dumps({'kind': kind, 'args': args, 'cwd': os.getcwd(), 'events': e.get('OFFLINE_EVENTS')})+'\n')
if kind == 'module':
    assert args == ['load', 'apptainer']
elif kind == 'sha256sum':
    expected, path = sys.stdin.read().strip().split('  ', 1)
    sys.exit(0 if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == expected else 2)
elif kind == 'apptainer':
    assert args[:3] == ['exec', '--cleanenv', '-B']
    assert pathlib.Path(args[6]).name == 'quiet_replay.py'
    stage, opts = args[7], args[8:]
    def value(key):
        return opts[opts.index(key)+1]
    assert value('--expected-events') == '28044'
    assert value('--runtime-area') == e['TRIGGER_AREA']
    assert value('--runtime-image') == e['OFFLINE_IMAGE']
    if stage == e.get('FAIL_STAGE'):
        sys.exit(31)
    if stage == 'prepare':
        dest = pathlib.Path(value('--prepared')); dest.mkdir()
        (dest/'replay.fcl').write_text('fixture')
    elif stage == 'validate':
        assert pathlib.Path(value('--art')).exists()
        pathlib.Path(value('--output')).write_text('{}')
    else:
        assert stage == 'preflight'
elif kind == 'offline':
    assert args[0] == e['TRIGGER_AREA']
    assert e['OFFLINE_EVENTS'] == '28044'
    pathlib.Path('decisions.art').write_bytes(b'fixture')
    pathlib.Path('replay.expanded.fcl').write_text('fixture')
    sys.exit(int(e.get('ART_EXIT', '0')))
else:
    raise AssertionError(kind)
'''


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / 'private'
    root.mkdir()
    code = root / 'code'
    offline = code / 'container/offline'
    offline.mkdir(parents=True)
    driver = root / 'driver.py'
    driver.write_text(DRIVER)
    binary = root / 'bin'
    binary.mkdir()
    command = shlex.quote(sys.executable) + ' ' + shlex.quote(str(driver))
    for name in ('module', 'apptainer', 'sha256sum'):
        path = binary / name
        path.write_text('#!/bin/bash\nexec ' + command + ' ' + name + ' "$@"\n')
        path.chmod(0o755)
    (offline / 'offline_uva.sh').write_text('#!/bin/bash\nexec ' + command + ' offline "$@"\n')
    image = root / 'image.sif'
    image.write_bytes(b'image')
    env = dict(os.environ, PATH=str(binary)+':'+os.environ['PATH'], CALLS=str(root/'calls.jsonl'),
        TRIGGER_ROOT=str(root), TRIGGER_CODE=str(code), TRIGGER_CONFIG=str(root/'config'),
        TRIGGER_ANALYSIS_IMAGE=str(image), TRIGGER_ANALYSIS_SHA256=hashlib.sha256(b'image').hexdigest(),
        TRIGGER_PYTHON='/usr/local/bin/python3', TRIGGER_AREA='/cvmfs/released', OFFLINE_IMAGE='/cvmfs/pinned',
        CVMFSEXEC_DIST=str(root/'cvmfsexec'), TRIGGER_QUIET_ROOT=str(root/'quiet'),
        TRIGGER_EXPECTED_EVENTS='28044', SLURM_JOB_ID='fixture', OFFLINE_WARM_REPEAT='0', OFFLINE_OUTPUT_FILE='')
    for arm in ('STOCK', 'FAST'):
        for field in ('PREPARED', 'ART', 'COMPARISON'):
            env['TRIGGER_'+arm+'_'+field] = str(root/(arm.lower()+'-'+field.lower()))
    return env


def run(env):
    return subprocess.run(['bash', str(ROOT/'trigger/quiet_pair.sbatch')], env=env, capture_output=True, text=True)


def calls(env):
    path = Path(env['CALLS'])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_two_arms_sequential_fullcount_and_exact_cli(batch):
    result = run(batch)
    assert result.returncode == 0, result.stderr
    rows = calls(batch)
    stages = [row['args'][7] for row in rows if row['kind'] == 'apptainer']
    assert stages == ['prepare', 'preflight', 'validate'] * 2
    art = [row for row in rows if row['kind'] == 'offline']
    assert [Path(row['cwd']).parent.name for row in art] == ['stock', 'fast']
    assert all(row['events'] == '28044' for row in art)
    assert (Path(batch['TRIGGER_QUIET_ROOT'])/'stock/output/art.exit').read_text().strip() == '0'
    assert (Path(batch['TRIGGER_QUIET_ROOT'])/'fast/output/comparison.json').is_file()


@pytest.mark.parametrize('stage', ['prepare', 'preflight', 'validate'])
def test_admission_or_validation_failure_stops_next_arm(batch, stage):
    batch['FAIL_STAGE'] = stage
    assert run(batch).returncode == 31
    assert len([r for r in calls(batch) if r['kind'] == 'offline']) == int(stage == 'validate')


def test_art_failure_preserves_exit_and_never_runs_fast(batch):
    batch['ART_EXIT'] = '17'
    assert run(batch).returncode == 17
    assert (Path(batch['TRIGGER_QUIET_ROOT'])/'stock/output/art.exit').read_text().strip() == '17'
    assert not (Path(batch['TRIGGER_QUIET_ROOT'])/'fast').exists()


@pytest.mark.parametrize('change', ['exists', 'hash', 'count', 'warm', 'output'])
def test_guard_rejects_overwrite_badimage_count_and_repeat(batch, change):
    if change == 'exists':
        Path(batch['TRIGGER_QUIET_ROOT']).mkdir()
    elif change == 'hash':
        batch['TRIGGER_ANALYSIS_SHA256'] = '0'*64
    elif change == 'count':
        batch['TRIGGER_EXPECTED_EVENTS'] = '0'
    elif change == 'warm':
        batch['OFFLINE_WARM_REPEAT'] = '1'
    else:
        batch['OFFLINE_OUTPUT_FILE'] = 'other.root'
    assert run(batch).returncode != 0
    assert not any(row['kind'] == 'offline' for row in calls(batch))
