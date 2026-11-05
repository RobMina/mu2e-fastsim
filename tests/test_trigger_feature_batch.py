"""Feature batch orchestration only: no art, Slurm, containers, or builds."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'trigger/feature_replay.sbatch'
DRIVER = r'''import hashlib, json, os, pathlib, sys
kind, args = sys.argv[1], sys.argv[2:]
e = os.environ
with open(e['CALLS'], 'a') as out:
    out.write(json.dumps({'kind': kind, 'args': args, 'cwd': os.getcwd(),
        'events': e.get('OFFLINE_EVENTS'), 'prefix': e.get('OFFLINE_LOG_PREFIX'),
        'binds': e.get('OFFLINE_BINDS'), 'image': e.get('OFFLINE_IMAGE')})+'\n')
if kind == 'module':
    assert args == ['load', 'apptainer']
elif kind == 'sha256sum':
    assert args == ['-c', '-']
    expected, path = sys.stdin.read().strip().split('  ', 1)
    try:
        actual = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except FileNotFoundError:
        sys.exit(2)
    sys.exit(0 if actual == expected else 2)
elif kind == 'apptainer':
    assert args[:3] == ['exec', '--cleanenv', '-B']
    script = pathlib.Path(args[6]).name
    stage = 'validate' if script == 'feature_validate.py' else args[7]
    if stage == e.get('FAIL_STAGE'):
        sys.exit(31)
    opts = args[7:] if stage == 'validate' else args[8:]
    def value(key):
        return opts[opts.index(key)+1]
    if stage == 'verify-deployment':
        assert script == 'feature_runtime.py'
    elif stage == 'prepare':
        assert script == 'feature_replay.py'
        dest = pathlib.Path(value('--output')); dest.mkdir()
        (dest/'replay.fcl').write_text('fixture full-history configuration')
    elif stage == 'preflight':
        assert script == 'feature_replay.py'
        assert (pathlib.Path(value('--output'))/'replay.fcl').is_file()
    else:
        assert stage == 'validate' and script == 'feature_validate.py'
        for flag in ('--art', '--table', '--log'):
            assert pathlib.Path(value(flag)).is_file()
        assert pathlib.Path('art.exit').read_text() == '0\n'
        pathlib.Path(value('--output')).write_text('{}\n')
elif kind == 'offline':
    assert args == [e['TRIGGER_AREA'], e['TRIGGER_PREPARED']+'/replay.fcl']
    assert e['OFFLINE_EVENTS'] == '28044'
    pathlib.Path('decisions.art').write_bytes(b'fixture decisions')
    pathlib.Path('r7-selection-features.root').write_bytes(b'fixture features')
    pathlib.Path('replay.expanded.fcl').write_text('fixture full-history configuration')
    print('fixture art log', flush=True)
    sys.exit(int(e.get('ART_EXIT', '0')))
else:
    raise AssertionError(kind)
'''

REQUIRED = (
    'TRIGGER_ROOT', 'TRIGGER_CODE', 'TRIGGER_CONFIG', 'TRIGGER_ANALYSIS_IMAGE',
    'TRIGGER_ANALYSIS_SHA256', 'TRIGGER_PYTHON', 'TRIGGER_AREA', 'OFFLINE_IMAGE',
    'CVMFSEXEC_DIST', 'TRIGGER_FEATURE_BUNDLE', 'TRIGGER_FEATURE_LIBRARY_SHA256',
    'TRIGGER_FEATURE_MANIFEST_SHA256', 'TRIGGER_BASELINE_PREPARED',
    'TRIGGER_BASELINE_ART', 'TRIGGER_BASELINE_COMPARISON', 'TRIGGER_EXPECTED_EVENTS',
    'TRIGGER_PREPARED', 'TRIGGER_OUTPUT', 'TRIGGER_EXPORTER_SHA256',
)


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / 'private feature run'
    root.mkdir()
    code = root / 'code snapshot'
    code.mkdir()
    bundle = root / 'frozen bundle'
    bundle.mkdir()
    driver = root / 'driver.py'
    driver.write_text(DRIVER)
    binary = root / 'bin'
    binary.mkdir()
    command = shlex.quote(sys.executable) + ' ' + shlex.quote(str(driver))
    for name in ('module', 'apptainer', 'sha256sum'):
        path = binary / name
        path.write_text('#!/bin/bash\nexec ' + command + ' ' + name + ' "$@"\n')
        path.chmod(0o755)
    (bundle / 'offline_features.sh').write_text(
        '#!/bin/bash\nexec ' + command + ' offline "$@"\n')
    image = root / 'analysis image.sif'
    image.write_bytes(b'analysis image fixture')
    # Do not inherit a login shell, job, or real runtime configuration.
    return dict(
        PATH=str(binary) + os.pathsep + '/usr/bin:/bin', HOME=str(root),
        PYTHONDONTWRITEBYTECODE='1', CALLS=str(root / 'calls.jsonl'),
        TRIGGER_ROOT=str(root), TRIGGER_CODE=str(code),
        TRIGGER_CONFIG=str(root / 'full history config.json'),
        TRIGGER_ANALYSIS_IMAGE=str(image),
        TRIGGER_ANALYSIS_SHA256=hashlib.sha256(image.read_bytes()).hexdigest(),
        TRIGGER_PYTHON='/explicit/container/python3',
        TRIGGER_AREA='/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ba',
        OFFLINE_IMAGE='/cvmfs/pinned/offline-image',
        CVMFSEXEC_DIST=str(root / 'unused cvmfsexec'),
        TRIGGER_FEATURE_BUNDLE=str(bundle),
        TRIGGER_FEATURE_LIBRARY_SHA256='1' * 64,
        TRIGGER_FEATURE_MANIFEST_SHA256='2' * 64,
        TRIGGER_EXPORTER_SHA256='3' * 64,
        TRIGGER_BASELINE_PREPARED=str(root / 'saved fullhybrid prepared'),
        TRIGGER_BASELINE_ART=str(root / 'saved fullhybrid decisions.art'),
        TRIGGER_BASELINE_COMPARISON=str(root / 'saved fullhybrid comparison.json'),
        TRIGGER_EXPECTED_EVENTS='28044',
        TRIGGER_PREPARED=str(root / 'feature prepared'),
        TRIGGER_OUTPUT=str(root / 'feature output'),
        OFFLINE_WARM_REPEAT='0', OFFLINE_OUTPUT_FILE='',
        OFFLINE_EVENTS='1', OFFLINE_LOG_PREFIX='inherited-prefix',
        OFFLINE_BINDS='/inherited-bind-must-not-replace-private-root',
    )


def run(env):
    return subprocess.run(['bash', str(BATCH)], env=env, cwd=env['HOME'],
                          capture_output=True, text=True, timeout=15)


def calls(env):
    path = Path(env['CALLS'])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stages(env):
    result = []
    for row in calls(env):
        if row['kind'] == 'offline':
            result.append('art')
        elif row['kind'] == 'apptainer':
            result.append('validate' if Path(row['args'][6]).name == 'feature_validate.py'
                          else row['args'][7])
    return result


def expected_analysis(env):
    binds = env['TRIGGER_ROOT'] + ',' + str(Path(env['TRIGGER_ROOT']).resolve())
    prefix = ['exec', '--cleanenv', '-B', binds,
              env['TRIGGER_ANALYSIS_IMAGE'], env['TRIGGER_PYTHON']]
    script = lambda name: str(Path(env['TRIGGER_CODE']) / 'trigger' / name)
    pins = ['--bundle', env['TRIGGER_FEATURE_BUNDLE'],
            '--expected-library-sha256', env['TRIGGER_FEATURE_LIBRARY_SHA256'],
            '--expected-manifest-sha256', env['TRIGGER_FEATURE_MANIFEST_SHA256']]
    common = ['--config', env['TRIGGER_CONFIG'], '--exporter-source',
              script('art/R7Trigger/src/KalSeedSelectionFeatures_module.cc'),
              '--exporter-sha256', env['TRIGGER_EXPORTER_SHA256'],
              '--output', env['TRIGGER_PREPARED'], '--expected-events', '28044',
              '--runtime-area', env['TRIGGER_AREA'], '--runtime-image', env['OFFLINE_IMAGE']]
    return [
        prefix + [script('feature_runtime.py'), 'verify-deployment'] + pins,
        prefix + [script('feature_replay.py'), 'prepare'] + common + [
            '--baseline-prepared', env['TRIGGER_BASELINE_PREPARED'],
            '--baseline-art', env['TRIGGER_BASELINE_ART'],
            '--baseline-comparison', env['TRIGGER_BASELINE_COMPARISON']],
        prefix + [script('feature_replay.py'), 'preflight'] + common,
        prefix + [script('feature_validate.py'), '--config', env['TRIGGER_CONFIG'],
                  '--prepared', env['TRIGGER_PREPARED'], '--art', 'decisions.art',
                  '--table', 'r7-selection-features.root', '--log', 'replay.log'] +
        pins + ['--output', 'comparison.json'],
    ]


def test_fullcount_exact_cli_pins_fullhistory_and_sequential_execution(batch):
    result = run(batch)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = calls(batch)
    assert [r['kind'] for r in rows] == [
        'module', 'sha256sum', 'apptainer', 'apptainer', 'apptainer', 'offline', 'apptainer']
    assert stages(batch) == ['verify-deployment', 'prepare', 'preflight', 'art', 'validate']
    analysis = [r for r in rows if r['kind'] == 'apptainer']
    assert [r['args'] for r in analysis] == expected_analysis(batch)
    assert [r['cwd'] for r in analysis] == [batch['HOME']] * 3 + [batch['TRIGGER_OUTPUT']]
    art = next(r for r in rows if r['kind'] == 'offline')
    assert art['args'] == [batch['TRIGGER_AREA'], batch['TRIGGER_PREPARED'] + '/replay.fcl']
    assert art['cwd'] == batch['TRIGGER_OUTPUT']
    assert art['events'] == '28044'
    assert art['prefix'] == 'replay'
    assert art['binds'] == batch['TRIGGER_ROOT'] + ',' + str(Path(batch['TRIGGER_ROOT']).resolve())
    assert art['image'] == batch['OFFLINE_IMAGE']
    output = Path(batch['TRIGGER_OUTPUT'])
    assert (output / 'art.exit').read_text() == '0\n'
    assert (output / 'replay.log').read_text() == 'fixture art log\n'
    assert (output / 'comparison.json').is_file()


@pytest.mark.parametrize('key', REQUIRED)
@pytest.mark.parametrize('value', [None, ''], ids=['unset', 'empty'])
def test_all_runtime_baseline_and_pin_inputs_are_explicit(batch, key, value):
    if value is None:
        batch.pop(key)
    else:
        batch[key] = value
    result = run(batch)
    assert result.returncode != 0
    assert key in result.stderr
    assert calls(batch) == []


@pytest.mark.parametrize('stage', ['verify-deployment', 'prepare', 'preflight', 'validate'])
def test_admission_or_validation_failure_stops_and_keeps_art_exit(batch, stage):
    batch['FAIL_STAGE'] = stage
    result = run(batch)
    assert result.returncode == 31, result.stdout + result.stderr
    sequence = ['verify-deployment', 'prepare', 'preflight', 'art', 'validate']
    assert stages(batch) == sequence[:sequence.index(stage) + 1]
    output = Path(batch['TRIGGER_OUTPUT'])
    if stage == 'validate':
        assert (output / 'art.exit').read_text() == '0\n'
        assert (output / 'replay.log').read_text() == 'fixture art log\n'
        assert (output / 'decisions.art').is_file()
        assert (output / 'r7-selection-features.root').is_file()
        assert not (output / 'comparison.json').exists()
    else:
        assert not output.exists()


def test_art_failure_retains_status_and_log_and_skips_validator(batch):
    batch['ART_EXIT'] = '17'
    result = run(batch)
    assert result.returncode == 17
    assert stages(batch) == ['verify-deployment', 'prepare', 'preflight', 'art']
    output = Path(batch['TRIGGER_OUTPUT'])
    assert (output / 'art.exit').read_text() == '17\n'
    assert (output / 'replay.log').read_text() == 'fixture art log\n'
    assert 'fixture art log' in result.stdout
    assert not (output / 'comparison.json').exists()


@pytest.mark.parametrize('key', ['TRIGGER_PREPARED', 'TRIGGER_OUTPUT'])
@pytest.mark.parametrize('kind', ['directory', 'file'])
def test_refuses_overwriting_prepared_or_output_without_touching_sentinel(batch, key, kind):
    path = Path(batch[key])
    if kind == 'directory':
        path.mkdir()
        sentinel = path / 'previous-artifact'
    else:
        sentinel = path
    sentinel.write_bytes(b'keep previous run\n')
    assert run(batch).returncode == 2
    assert calls(batch) == []
    assert sentinel.read_bytes() == b'keep previous run\n'


@pytest.mark.parametrize('value', ['0', '1', '28043', '28045', '028044', '-1'])
def test_rejects_any_count_except_exact_full_28044_before_admission(batch, value):
    batch['TRIGGER_EXPECTED_EVENTS'] = value
    assert run(batch).returncode == 2
    assert calls(batch) == []


@pytest.mark.parametrize('key,value', [
    ('OFFLINE_WARM_REPEAT', '1'), ('OFFLINE_WARM_REPEAT', '2'),
    ('OFFLINE_OUTPUT_FILE', 'other.root'),
])
def test_rejects_warm_repeat_or_output_override_before_admission(batch, key, value):
    batch[key] = value
    assert run(batch).returncode == 2
    assert calls(batch) == []


def test_unset_optional_warm_and_output_are_safe(batch):
    batch.pop('OFFLINE_WARM_REPEAT')
    batch.pop('OFFLINE_OUTPUT_FILE')
    result = run(batch)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', ['pin', 'bytes', 'missing'])
def test_analysis_image_must_match_explicit_sha_before_any_analysis_or_art(batch, change):
    image = Path(batch['TRIGGER_ANALYSIS_IMAGE'])
    if change == 'pin':
        batch['TRIGGER_ANALYSIS_SHA256'] = '0' * 64
    elif change == 'bytes':
        image.write_bytes(b'changed analysis image')
    else:
        image.unlink()
    assert run(batch).returncode != 0
    assert [r['kind'] for r in calls(batch)] == ['module', 'sha256sum']
    assert not Path(batch['TRIGGER_PREPARED']).exists()
    assert not Path(batch['TRIGGER_OUTPUT']).exists()


def test_source_static_inherited_plugin_flags_are_not_forwarded_as_runtime_authority():
    """The frozen runner owns plugin paths/pins; inherited flags are not proof."""
    runner = (ROOT / 'container/offline/offline_uva.sh').read_text()
    runtime = (ROOT / 'trigger/feature_runtime.py').read_text()
    forwarded = runner.split('for key in ', 1)[1].split('; do', 1)[0].split()
    assert not {'CET_PLUGIN_PATH', 'LD_LIBRARY_PATH', 'OFFLINE_FEATURE_BUNDLE',
                'OFFLINE_FEATURE_RUNNER_SHA256'} & set(forwarded)
    assert '--cleanenv' in runner
    assert 'OFFLINE_FEATURE_BUNDLE=$(dirname "$SELF")' in runtime
    assert 'OFFLINE_FEATURE_RUNNER_SHA256=$(sha256sum "$SELF"' in runtime
    assert 'OFFLINE_FEATURE_BUNDLE OFFLINE_FEATURE_RUNNER_SHA256;' in runtime
    assert 'export CET_PLUGIN_PATH="$OFFLINE_FEATURE_BUNDLE/plugins:' in runtime
    assert 'export LD_LIBRARY_PATH="$OFFLINE_FEATURE_BUNDLE/plugins:' in runtime
