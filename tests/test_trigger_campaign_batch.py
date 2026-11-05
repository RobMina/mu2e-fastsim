"""Local campaign-wrapper control flow; no Slurm, containers, art or remote IO.

Analysis commands are doubles: these tests check orchestration, not the scientific
admission/validation logic exercised by the campaign and hybrid unit tests.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HYBRID = ROOT / 'trigger' / 'hybrid_campaign.sbatch'
INJECTION = ROOT / 'trigger' / 'injection_array.sbatch'

# One driver records structured calls so paths containing spaces stay unambiguous.
DRIVER = r'''import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

kind, args = sys.argv[1], sys.argv[2:]
env = os.environ
keys = ('OFFLINE_EVENTS', 'OFFLINE_LOG_PREFIX', 'OFFLINE_BINDS',
        'TRIGGER_SOURCE', 'TRIGGER_TRANSPORT', 'TRIGGER_OUTPUT',
        'TRIGGER_PREPARED', 'TRIGGER_CONFIG', 'TRIGGER_AREA')
with open(env['FAKE_CALLS'], 'a') as stream:
    stream.write(json.dumps(dict(kind=kind, args=args, cwd=os.getcwd(),
                                 env={key: env.get(key) for key in keys})) + '\n')

if kind == 'module':
    assert args == ['load', 'apptainer']
elif kind == 'sha256sum':
    # A portable implementation of just the wrapper's checksum invocation.
    # Actually hash the fixture image; do not unconditionally admit bad images.
    assert args == ['-c', '-']
    expected, filename = sys.stdin.read().rstrip('\n').split('  ', 1)
    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if actual != expected:
        print('analysis image checksum mismatch', file=sys.stderr)
        sys.exit(1)
elif kind == 'apptainer':
    assert args[:3] == ['exec', '--cleanenv', '-B']
    assert args[4:6] == [env['TRIGGER_ANALYSIS_IMAGE'], env['TRIGGER_PYTHON']]
    script, options = args[6], args[7:]
    def option(name):
        return options[options.index(name) + 1]
    if Path(script).name == 'hybrid_campaign.py':
        stage = 'prepare'
    elif Path(script).name == 'injection_campaign.py':
        assert options[0] == 'prepare-parent'
        stage = 'prepare-parent'
    else:
        assert Path(script).name == 'hybrid_replay.py'
        stage = options[0]
        assert stage in ('preflight', 'validate')
    if env.get('FAKE_FAIL_STAGE') == stage:
        print('mock ' + stage + ' failed', file=sys.stderr)
        sys.exit(31)
    if stage == 'prepare':
        prepared = Path(option('--output'))
        prepared.mkdir()
        (prepared / 'replay.fcl').write_text('# mock full-history replay\n')
    elif stage == 'prepare-parent':
        values = json.loads(env['FAKE_PARENT_ENV'])
        Path(option('--env-output')).write_text(''.join(
            'export ' + key + '=' + shlex.quote(value) + '\n'
            for key, value in values.items()))
    else:
        assert (Path(option('--prepared')) / 'replay.fcl').is_file()
        if stage == 'validate':
            assert Path(option('--art')).is_file()
            assert Path(option('--log')).is_file()
            Path(option('--output')).write_text('{"mock_validation": true}\n')
elif kind == 'offline':
    assert args == [env['TRIGGER_AREA'], env['TRIGGER_PREPARED'] + '/replay.fcl']
    print('mock Offline process over ' + env['OFFLINE_EVENTS'] + ' events')
    Path('decisions.art').write_bytes(b'mock art product')
    sys.exit(int(env.get('FAKE_OFFLINE_STATUS', '0')))
elif kind == 'step-inject':
    assert args == []
    sys.exit(int(env.get('FAKE_STEP_STATUS', '0')))
else:
    raise AssertionError('unexpected mocked command: ' + kind)
'''


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def campaign_batch(tmp_path):
    real_root = tmp_path / 'physical campaign root'
    real_root.mkdir()
    alias = tmp_path / 'campaign root alias'
    alias.symlink_to(real_root, target_is_directory=True)
    code = alias / 'code snapshot'
    tools = tmp_path / 'fake bin'
    for directory in (tools, code / 'trigger', code / 'container' / 'offline',
                      alias / 'prepared', alias / 'history receipts'):
        directory.mkdir(parents=True, exist_ok=True)
    driver = tmp_path / 'fake driver.py'
    driver.write_text(DRIVER)
    calls = tmp_path / 'calls.jsonl'
    image = alias / 'analysis image.sif'
    image.write_bytes(b'pinned mock analysis image\n')
    prepared = alias / 'prepared' / 'full history'
    output = alias / 'full history output'
    plan = alias / 'campaign plan.json'
    plan.write_text('{"events": 54000, "parents": []}\n')
    regeneration = alias / 'history receipts' / 'regeneration manifest.json'
    comparison = alias / 'history receipts' / 'production comparison.json'
    config = alias / 'production config.fcl'
    for path in (regeneration, comparison, config):
        path.write_text('mock input\n')
    stock = alias / 'stock history'
    stock.mkdir()
    parent_env = {
        'TRIGGER_SOURCE': str(stock / 'parent 2.art'),
        'TRIGGER_TRANSPORT': str(alias / 'parent transport'),
        'TRIGGER_OUTPUT': str(alias / 'parent injection'),
        'TRIGGER_PREPARED': str(alias / 'prepared' / 'parent injection'),
        'TRIGGER_CONFIG': str(config),
        'TRIGGER_AREA': str(alias / 'parent release'),
        'OFFLINE_EVENTS': '4321',
    }
    env = {
        'PATH': f'{tools}:/usr/bin:/bin',
        'HOME': str(tmp_path),
        'PYTHONDONTWRITEBYTECODE': '1',
        'TRIGGER_CODE': str(code),
        'TRIGGER_ROOT': str(alias),
        'TRIGGER_CAMPAIGN': str(plan),
        'TRIGGER_REGENERATION_MANIFEST': str(regeneration),
        'TRIGGER_PRODUCTION_COMPARISON': str(comparison),
        'TRIGGER_PRODUCTION_LOG': str(alias / 'production.log'),
        'OFFLINE_IMAGE': '/cvmfs/synthetic/pinned-image',
        'TRIGGER_STOCK_DIRECTORY': str(stock),
        'TRIGGER_PREPARED': str(prepared),
        'TRIGGER_OUTPUT': str(output),
        'TRIGGER_CONFIG': str(config),
        'TRIGGER_AREA': str(alias / 'built release'),
        'CVMFSEXEC_DIST': str(alias / 'unused cvmfs client'),
        'TRIGGER_ANALYSIS_IMAGE': str(image),
        'TRIGGER_ANALYSIS_SHA256': hashlib.sha256(image.read_bytes()).hexdigest(),
        'TRIGGER_PYTHON': sys.executable,
        'TRIGGER_EXPECTED_EVENTS': '54000',
        # Stale per-parent state must not reduce the full-history replay count.
        'OFFLINE_EVENTS': '1',
        'SLURM_ARRAY_TASK_ID': '2',
        'SLURM_ARRAY_JOB_ID': '12345',
        'FAKE_DRIVER': str(driver),
        'FAKE_CALLS': str(calls),
        'FAKE_PARENT_ENV': json.dumps(parent_env),
    }
    for path, kind in ((tools / 'module', 'module'),
                       (tools / 'apptainer', 'apptainer'),
                       (tools / 'sha256sum', 'sha256sum'),
                       (code / 'container' / 'offline' / 'offline_uva.sh', 'offline'),
                       (code / 'trigger' / 'step_inject.sbatch', 'step-inject')):
        executable(path, '#!/bin/bash\nset -eu\n'
                   f'exec "$TRIGGER_PYTHON" "$FAKE_DRIVER" {kind} "$@"\n')
    return SimpleNamespace(env=env, cwd=tmp_path, code=code, root=alias,
                           real_root=real_root, prepared=prepared, output=output,
                           image=image, calls=calls, parent_env=parent_env)


def run_batch(batch, wrapper=HYBRID):
    return subprocess.run(['/bin/bash', str(wrapper)], env=batch.env, cwd=batch.cwd,
                          capture_output=True, text=True, timeout=15)


def recorded(batch, kind=None):
    calls = ([json.loads(line) for line in batch.calls.read_text().splitlines()]
             if batch.calls.exists() else [])
    return calls if kind is None else [call for call in calls if call['kind'] == kind]


def assert_no_offline_or_reports(batch):
    assert not recorded(batch, 'offline')
    for name in ('replay.log', 'art.exit', 'comparison.json', 'decisions.art'):
        assert not (batch.output / name).exists(), f'{name} created before admission'


def test_hybrid_runs_one_full_count_process_and_forwards_history(campaign_batch):
    batch = campaign_batch
    result = run_batch(batch)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call['kind'] for call in recorded(batch)] == [
        'module', 'sha256sum', 'apptainer', 'apptainer', 'offline', 'apptainer',
    ]
    calls = recorded(batch, 'apptainer')
    assert len(calls) == 3
    binds = f'{batch.root},{batch.root.resolve()}'
    assert str(batch.root) != str(batch.root.resolve()), 'fixture must exercise a root alias'
    for call in calls:
        assert call['args'][:6] == [
            'exec', '--cleanenv', '-B', binds,
            str(batch.image), sys.executable,
        ]
    assert calls[0]['args'][6:] == [
        str(batch.code / 'trigger' / 'hybrid_campaign.py'),
        '--plan', batch.env['TRIGGER_CAMPAIGN'],
        '--regeneration', batch.env['TRIGGER_REGENERATION_MANIFEST'],
        '--stock-directory', batch.env['TRIGGER_STOCK_DIRECTORY'],
        '--comparison', batch.env['TRIGGER_PRODUCTION_COMPARISON'],
        '--production-log', batch.env['TRIGGER_PRODUCTION_LOG'],
        '--output', str(batch.prepared),
    ]
    assert calls[1]['args'][6:] == [
        str(batch.code / 'trigger' / 'hybrid_replay.py'), 'preflight',
        '--config', batch.env['TRIGGER_CONFIG'], '--prepared', str(batch.prepared),
        '--expected-events', batch.env['TRIGGER_EXPECTED_EVENTS'],
        '--runtime-area', batch.env['TRIGGER_AREA'], '--runtime-image', batch.env['OFFLINE_IMAGE'],
    ]
    assert calls[2]['args'][6:] == [
        str(batch.code / 'trigger' / 'hybrid_replay.py'), 'validate',
        '--config', batch.env['TRIGGER_CONFIG'], '--prepared', str(batch.prepared),
        '--art', 'decisions.art', '--log', 'replay.log', '--output', 'comparison.json',
    ]
    offline, = recorded(batch, 'offline')
    assert offline['args'] == [batch.env['TRIGGER_AREA'], str(batch.prepared / 'replay.fcl')]
    assert Path(offline['cwd']) == batch.output.resolve()
    assert offline['env']['OFFLINE_EVENTS'] == batch.env['TRIGGER_EXPECTED_EVENTS']
    assert offline['env']['OFFLINE_LOG_PREFIX'] == 'replay'
    assert offline['env']['OFFLINE_BINDS'] == binds
    assert Path(calls[2]['cwd']) == batch.output.resolve()
    assert (batch.output / 'art.exit').read_text() == '0\n'
    assert (batch.output / 'replay.log').read_text() == 'mock Offline process over 54000 events\n'
    assert json.loads((batch.output / 'comparison.json').read_text()) == {'mock_validation': True}


@pytest.mark.parametrize('key,value', [
    ('OFFLINE_WARM_REPEAT', '1'), ('OFFLINE_OUTPUT_FILE', 'overwrite.art'),
    ('TRIGGER_EXPECTED_EVENTS', '0'), ('TRIGGER_EXPECTED_EVENTS', '-1'),
    ('TRIGGER_EXPECTED_EVENTS', '1; exit 0'), ('TRIGGER_EXPECTED_EVENTS', '1000000'),
])
def test_hybrid_refuses_unsafe_run_before_any_commands(campaign_batch, key, value):
    batch = campaign_batch
    batch.env[key] = value
    result = run_batch(batch)
    assert result.returncode == 2, result.stdout + result.stderr
    assert recorded(batch) == []
    assert not batch.output.exists()
    assert not batch.prepared.exists()
    assert_no_offline_or_reports(batch)


@pytest.mark.parametrize('destination', ['output', 'prepared'])
@pytest.mark.parametrize('existing_kind', ['directory', 'file'])
def test_hybrid_refuses_to_overwrite_output_or_preparation(campaign_batch, destination, existing_kind):
    batch = campaign_batch
    path = getattr(batch, destination)
    sentinel = path
    if existing_kind == 'directory':
        path.mkdir()
        sentinel = path / 'previous evidence'
    sentinel.write_bytes(b'preserve previous campaign\n')
    result = run_batch(batch)
    assert result.returncode == 2, result.stdout + result.stderr
    assert recorded(batch) == []
    assert sentinel.read_bytes() == b'preserve previous campaign\n'
    if existing_kind == 'directory':
        assert list(path.iterdir()) == [sentinel]
    assert_no_offline_or_reports(batch)


@pytest.mark.parametrize('wrapper', [HYBRID, INJECTION], ids=['hybrid', 'injection'])
def test_image_checksum_mismatch_stops_before_analysis_or_offline(campaign_batch, wrapper):
    batch = campaign_batch
    batch.image.write_bytes(b'changed image after pinning\n')
    result = run_batch(batch, wrapper)
    assert result.returncode != 0
    assert 'checksum mismatch' in result.stderr
    assert [call['kind'] for call in recorded(batch)] == ['module', 'sha256sum']
    assert not batch.prepared.exists()
    assert not batch.output.exists()
    assert not recorded(batch, 'step-inject')
    assert_no_offline_or_reports(batch)


@pytest.mark.parametrize('stage,analysis_calls', [('prepare', 1), ('preflight', 2)])
def test_hybrid_preparation_or_preflight_failure_stops_before_offline(campaign_batch, stage, analysis_calls):
    batch = campaign_batch
    batch.env['FAKE_FAIL_STAGE'] = stage
    result = run_batch(batch)
    assert result.returncode == 31, result.stdout + result.stderr
    assert f'mock {stage} failed' in result.stderr
    assert len(recorded(batch, 'apptainer')) == analysis_calls
    assert_no_offline_or_reports(batch)


def test_hybrid_offline_failure_preserves_status_and_skips_validate(campaign_batch):
    batch = campaign_batch
    batch.env['FAKE_OFFLINE_STATUS'] = '23'
    result = run_batch(batch)
    assert result.returncode == 23, result.stdout + result.stderr
    assert len(recorded(batch, 'offline')) == 1
    assert len(recorded(batch, 'apptainer')) == 2
    assert (batch.output / 'art.exit').read_text() == '23\n'
    assert 'mock Offline process over 54000 events' in result.stdout
    assert not (batch.output / 'comparison.json').exists()


def test_hybrid_validation_failure_is_not_reported_as_success(campaign_batch):
    batch = campaign_batch
    batch.env['FAKE_FAIL_STAGE'] = 'validate'
    result = run_batch(batch)
    assert result.returncode == 31, result.stdout + result.stderr
    assert len(recorded(batch, 'offline')) == 1
    assert len(recorded(batch, 'apptainer')) == 3
    assert (batch.output / 'art.exit').read_text() == '0\n'
    assert not (batch.output / 'comparison.json').exists()


@pytest.mark.parametrize('ordinal', range(2, 11))
def test_injection_runs_one_parent_preflight_then_one_step_dispatch(campaign_batch, ordinal):
    batch = campaign_batch
    batch.env['SLURM_ARRAY_TASK_ID'] = str(ordinal)
    result = run_batch(batch, INJECTION)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call['kind'] for call in recorded(batch)] == [
        'module', 'sha256sum', 'apptainer', 'step-inject',
    ]
    call, = recorded(batch, 'apptainer')
    env_file = batch.root / 'prepared' / f'injection-parent-{ordinal}-12345.env'
    assert call['args'] == [
        'exec', '--cleanenv', '-B', f'{batch.root},{batch.root.resolve()}',
        str(batch.image), sys.executable,
        str(batch.code / 'trigger' / 'injection_campaign.py'), 'prepare-parent',
        '--plan', batch.env['TRIGGER_CAMPAIGN'], '--ordinal', str(ordinal),
        '--env-output', str(env_file),
    ]
    assert env_file.is_file()
    dispatch, = recorded(batch, 'step-inject')
    assert dispatch['args'] == []
    for key, value in batch.parent_env.items():
        assert dispatch['env'][key] == value, f'prepared parent environment not sourced: {key}'
    assert not recorded(batch, 'offline'), 'array wrapper must delegate, not launch Offline itself'


@pytest.mark.parametrize('ordinal', ['0', '1', '11', '02', '-2', '2; exit 0'])
def test_injection_rejects_reused_first_parent_or_invalid_ordinal(campaign_batch, ordinal):
    batch = campaign_batch
    batch.env['SLURM_ARRAY_TASK_ID'] = ordinal
    result = run_batch(batch, INJECTION)
    assert result.returncode == 2, result.stdout + result.stderr
    assert recorded(batch) == []
    assert list((batch.root / 'prepared').iterdir()) == []


def test_injection_parent_preflight_failure_prevents_step_dispatch(campaign_batch):
    batch = campaign_batch
    batch.env['FAKE_FAIL_STAGE'] = 'prepare-parent'
    result = run_batch(batch, INJECTION)
    assert result.returncode == 31, result.stdout + result.stderr
    assert len(recorded(batch, 'apptainer')) == 1
    assert not recorded(batch, 'step-inject')
    assert not recorded(batch, 'offline')
    assert list((batch.root / 'prepared').iterdir()) == []


def test_injection_propagates_step_dispatch_failure(campaign_batch):
    batch = campaign_batch
    batch.env['FAKE_STEP_STATUS'] = '27'
    result = run_batch(batch, INJECTION)
    assert result.returncode == 27, result.stdout + result.stderr
    assert len(recorded(batch, 'apptainer')) == 1
    assert len(recorded(batch, 'step-inject')) == 1
