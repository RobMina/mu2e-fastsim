"""Local Bash integration of the batch wrapper; no Slurm, art or containers."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'trigger' / 'replay.sbatch'

CONFIG = '''physics: {
  DigitizePath: [ EWMProducer, makeSD ]
  calo_cluster: [ EWMProducer, makeSD, Cut ]
  trigger_paths: [ DigitizePath, "425:calo_cluster" ]
  producers: {
    EWMProducer: { module_type: EventWindowMarkerProducer }
    makeSD: { module_type: StrawDigisFromStepPointMCs }
  }
  filters: { Cut: { module_type: CaloClusterFilter } }
}
outputs: { Output: { SelectEvents: [ calo_cluster ] } }
services: {
  GeometryService: { inputFile: "geometry.txt" }
  DbService: { purpose: Sim_best version: v1_5 }
}
'''

DECISIONS = '''###############  PrintModule Run/Subrun/Event 1 2 3
ProductPrint art::TriggerResults_TriggerResults__Digitize
  pass 425 calo_cluster
ProductPrint art::TriggerResults_TriggerResults__TriggerReplay
  pass 425 calo_cluster

Art has completed and will exit with status 0.
'''

TIMING = '''TimeTracker printout (sec) Min Avg Max Median RMS nEvts
Full event 0.01 0.01 0.01 0.01 0 1
calo_cluster:Cut:CaloClusterFilter 0.002 0.002 0.002 0.002 0 1
==============================
TrigReport Events total = 1 passed = 1 failed = 0
TimeReport CPU = 0.1 Real = 0.2
Art has completed and will exit with status 0.
'''


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def batch(tmp_path):
    code = tmp_path / 'code snapshot'
    prepared = tmp_path / 'prepared config'
    output = tmp_path / 'private output'
    tools = tmp_path / 'fake bin'
    events = tmp_path / 'event lists' / 'input.events'
    for directory in (code / 'container' / 'offline', tools, events.parent):
        directory.mkdir(parents=True)
    (code / 'trigger').symlink_to(ROOT / 'trigger', target_is_directory=True)
    source = tmp_path / 'embedded.fcl'
    source.write_text('Digitize: {\n' + CONFIG + '}\n')
    events.write_text('Run SubRun Event\n1 2 3\n')
    input_file = tmp_path / 'stock input.art'
    input_file.touch()
    area = tmp_path / 'built release'
    area.mkdir()
    image = tmp_path / 'analysis image.sif'
    image.touch()
    payload = tmp_path / 'fake art.log'
    payload.write_text(DECISIONS)
    trace = tmp_path / 'calls'
    art_call = tmp_path / 'art-call'
    apptainer_calls = tmp_path / 'apptainer-calls.jsonl'
    env = {
        'PATH': f'{tools}:/usr/bin:/bin',
        'HOME': str(tmp_path),
        'PYTHONDONTWRITEBYTECODE': '1',
        'TRIGGER_CODE': str(code),
        'TRIGGER_PREPARED': str(prepared),
        'TRIGGER_OUTPUT': str(output),
        'TRIGGER_INPUT': str(input_file),
        'TRIGGER_AREA': str(area),
        'CVMFSEXEC_DIST': str(tmp_path / 'unused cvmfs client'),
        'TRIGGER_PYTHON': sys.executable,
        'TRIGGER_EVENT_LIST': str(events),
        'OFFLINE_EVENTS': '1',
        'OFFLINE_BINDS': '/synthetic/seed:/synthetic/seed',
        'FAKE_TRACE': str(trace),
        'FAKE_ART_CALL': str(art_call),
        'FAKE_ART_LOG': str(payload),
        'FAKE_APPTAINER_CALLS': str(apptainer_calls),
    }
    preparation = subprocess.run(
        [sys.executable, '-B', str(ROOT / 'trigger' / 'replay.py'), str(source),
         '--output', str(prepared)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert preparation.returncode == 0, preparation.stdout + preparation.stderr
    assert all((prepared / name).is_file() for name in
               ('replay.fcl', 'manifest.json', 'inventory.json', 'replay_inventory.json'))

    executable(tools / 'module', '''#!/bin/bash
set -eu
[[ $# == 2 && $1 == load && $2 == apptainer ]]
printf 'module\n' >> "$FAKE_TRACE"
''')
    executable(tools / 'apptainer', f'''#!{sys.executable}
import json
import os
from pathlib import Path
import sys
args = sys.argv[1:]
with open(os.environ['FAKE_APPTAINER_CALLS'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
with open(os.environ['FAKE_TRACE'], 'a') as stream:
    stream.write('analysis\\n')
assert args[:3] == ['exec', '--cleanenv', '-B']
assert Path(args[4]).is_file()
os.execv(args[5], args[5:])
''')
    executable(code / 'container' / 'offline' / 'offline_uva.sh', '''#!/bin/bash
set -euo pipefail
printf 'art\n' >> "$FAKE_TRACE"
printf '%s\n' "$PWD" "$@" "${OFFLINE_EVENTS:-all}" "$OFFLINE_LOG_PREFIX" "$OFFLINE_BINDS" > "$FAKE_ART_CALL"
[[ $# == 3 && $1 == "$TRIGGER_AREA" && $2 == "$TRIGGER_PREPARED/replay.fcl" && $3 == "$TRIGGER_INPUT" ]]
cat "$FAKE_ART_LOG"
exit "${FAKE_ART_STATUS:-0}"
''')
    return SimpleNamespace(env=env, cwd=tmp_path, code=code, prepared=prepared,
                           output=output, tools=tools, events=events, image=image,
                           payload=payload, trace=trace, art_call=art_call,
                           apptainer_calls=apptainer_calls)


def run_batch(batch):
    return subprocess.run(['/bin/bash', str(BATCH)], env=batch.env, cwd=batch.cwd,
                          capture_output=True, text=True, timeout=15)


def assert_no_art_or_reports(batch):
    assert not batch.art_call.exists()
    for name in ('replay.log', 'art.exit', 'comparison.json', 'timing.json'):
        assert not (batch.output / name).exists()


@pytest.mark.parametrize('python', [None, ''])
def test_requires_explicit_python_before_art(batch, python):
    if python is None:
        batch.env.pop('TRIGGER_PYTHON')
    else:
        batch.env['TRIGGER_PYTHON'] = python
    result = run_batch(batch)
    assert result.returncode != 0
    assert 'TRIGGER_PYTHON' in result.stderr
    assert 'Python 3.10+' in result.stderr
    assert not batch.trace.exists()
    assert_no_art_or_reports(batch)


def test_rejects_old_python_during_preflight_before_art(batch):
    old_python = executable(batch.tools / 'old-python', f'''#!{sys.executable}
import sys
assert sys.argv[1] == '-c'
sys.version_info = (3, 9, 0)
exec(sys.argv[2])
''')
    batch.env['TRIGGER_PYTHON'] = str(old_python)
    result = run_batch(batch)
    assert result.returncode != 0
    assert 'Python 3.10+ required' in result.stderr
    assert batch.trace.read_text().splitlines() == ['module']
    assert_no_art_or_reports(batch)


@pytest.mark.parametrize('previous', ['art.exit', 'replay.log'])
def test_refuses_to_overwrite_either_previous_artifact(batch, previous):
    batch.output.mkdir()
    sentinel = batch.output / previous
    sentinel.write_bytes(b'previous replay must survive\n')
    result = run_batch(batch)
    assert result.returncode == 2
    assert 'refuse to overwrite a previous replay' in result.stderr
    assert sentinel.read_bytes() == b'previous replay must survive\n'
    assert {path.name for path in batch.output.iterdir()} == {previous}
    assert not batch.art_call.exists()
    batch.env['TRIGGER_FORCE'] = '1'
    result = run_batch(batch)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (batch.output / 'art.exit').read_text() == '0\n'


def test_zero_events_replays_whole_file_and_counts_from_event_list(batch):
    batch.env['OFFLINE_EVENTS'] = '0'
    result = run_batch(batch)
    assert result.returncode == 0, result.stdout + result.stderr
    assert batch.art_call.read_text().splitlines()[4] == 'all'
    assert json.loads((batch.output / 'comparison.json').read_text())['events'] == 1


def test_success_runs_real_comparator_with_event_ids(batch):
    result = run_batch(batch)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (batch.output / 'art.exit').read_text() == '0\n'
    assert (batch.output / 'replay.log').read_text() == DECISIONS
    report = json.loads((batch.output / 'comparison.json').read_text())
    assert report['events'] == 1
    assert report['equal'] is True
    assert report['input_event_ids_verified'] is True
    assert report['mismatches'] == 0
    assert report['stored_output_selection_mismatches'] == 0
    assert report['paths'] == {'calo_cluster': {
        'bit': 425, 'stock_pass': 1, 'replay_pass': 1, 'mismatches': 0,
        'stock_only': 0, 'replay_only': 0, 'paired_efficiency_difference': 0.0,
    }}
    assert 'events=1 path_mismatches=0' in result.stdout
    assert not (batch.output / 'timing.json').exists()
    assert batch.art_call.read_text().splitlines() == [
        str(batch.output), batch.env['TRIGGER_AREA'], str(batch.prepared / 'replay.fcl'),
        batch.env['TRIGGER_INPUT'], '1', 'trigger',
        f'/synthetic/seed:/synthetic/seed,{batch.prepared}:{batch.prepared}',
    ]


def test_art_failure_preserves_status_and_skips_comparison_even_with_valid_log(batch):
    batch.env['FAKE_ART_STATUS'] = '23'
    result = run_batch(batch)
    assert result.returncode == 23
    assert batch.art_call.exists()
    assert (batch.output / 'art.exit').read_text() == '23\n'
    assert (batch.output / 'replay.log').read_text() == DECISIONS
    assert 'PrintModule Run/Subrun/Event 1 2 3' in result.stdout
    assert not (batch.output / 'comparison.json').exists()
    assert not (batch.output / 'timing.json').exists()


def test_analysis_image_binds_event_list_parent_and_all_analysis_directories(batch):
    batch.env['TRIGGER_ANALYSIS_IMAGE'] = str(batch.image)
    result = run_batch(batch)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in batch.apptainer_calls.read_text().splitlines()]
    assert len(calls) == 2
    for args in calls:
        assert args[:3] == ['exec', '--cleanenv', '-B']
        assert args[3].split(',') == [str(batch.code), str(batch.prepared),
                                      str(batch.output), str(batch.events.parent)]
        assert args[4:6] == [str(batch.image), sys.executable]
    assert calls[0][6] == '-c'
    assert 'sys.version_info >= (3, 10)' in calls[0][7]
    assert calls[1][6:8] == [str(batch.code / 'trigger' / 'compare.py'), 'replay.log']
    event_option = calls[1].index('--input-event-list')
    assert calls[1][event_option + 1] == str(batch.events)
    assert batch.trace.read_text().splitlines() == ['module', 'analysis', 'art', 'analysis']
    report = json.loads((batch.output / 'comparison.json').read_text())
    assert report['equal'] is True and report['input_event_ids_verified'] is True


def test_timing_mode_runs_real_diagnostic_instead_of_comparison(batch):
    batch.env['TRIGGER_MODE'] = 'timing'
    batch.payload.write_text(TIMING)
    result = run_batch(batch)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (batch.output / 'art.exit').read_text() == '0\n'
    assert (batch.output / 'replay.log').read_text() == TIMING
    assert not (batch.output / 'comparison.json').exists()
    report = json.loads((batch.output / 'timing.json').read_text())
    assert report['events'] == 1
    assert report['trigger']['ms_per_input_event'] == pytest.approx(2)
    assert report['coverage']['executed_modules'] == 1
    assert report['by_module']['Cut']['invocations'] == 1
    assert 'events=1 trigger_ms/input=2' in result.stdout
