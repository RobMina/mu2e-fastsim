"""Complete-parent regeneration preserves production configuration and fails closed."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import zlib

import pytest

TRIGGER = Path(__file__).resolve().parents[1] / 'trigger'
sys.path.insert(0, str(TRIGGER))
import regenerate
import replay
import stage_parents


def fixture():
    return '''Digitize: {
      source: { module_type: RootInput fileNames: [ "root://door//pnfs/fnal.gov/usr/mu2e/test/dts.test.a.art" ] }
      physics: { DigitizePath: [ Digi ] physicsPath: [ Digi, Cut ] trigger_paths: [ DigitizePath, "10:physicsPath" ]
        end_paths: [ End ] End: [ Output ] producers: { Digi: { module_type: TestDigi } }
        filters: { Cut: { module_type: TestFilter } } }
      services: { SeedService: { policy: autoIncrement baseSeed: 9 }
        RandomNumberGenerator: { defaultEngineKind: MixMaxRng }
        scheduler: { num_schedules: 1 num_threads: 1 } }
      outputs: { Output: { module_type: RootOutput fileName: "original.art" SelectEvents: [ physicsPath ] } }
      trigger_paths: { trigger_paths: [ "0:DigitizePath", "10:physicsPath" ] }
    }'''


def history():
    return '''Opened input file "root://door//pnfs/fnal.gov/usr/mu2e/test/dts.test.a.art"
services.SeedService.baseSeed: 9
TrigReport Events total = 20 passed = 19 failed = 1
Art has completed and will exit with status 0.
'''


def test_only_permitted_regeneration_changes():
    original = replay.production_config(fixture(), 'Digitize')
    prepared, receipt = regenerate.prepare(fixture(), history(), '/private/parents')
    assert prepared['physics'] == original['physics'] and prepared['services'] == original['services']
    assert prepared['process_name'] == 'Digitize' and 'trigger_paths' not in prepared
    assert prepared['source']['fileNames'] == ['/private/parents/dts.test.a.art']
    assert replay.plain(prepared['outputs']['Output']) == {'module_type': 'RootOutput', 'fileName': 'unfiltered.art'}
    assert receipt['original_processed_events'] == 20 and receipt['opened_parent_order_verified']


@pytest.mark.parametrize('old,new', [('baseSeed: 9', 'baseSeed: 10'), ('status 0.', 'status 1.'),
    ('dts.test.a.art', 'dts.other.art'), ('passed = 19', 'passed = 20')])
def test_mismatched_history_rejected(old, new):
    with pytest.raises(ValueError):
        regenerate.prepare(fixture(), history().replace(old, new), '/private')


@pytest.mark.parametrize('old,new', [('num_threads: 1', 'num_threads: 2'),
    ('"10:physicsPath" ] }', '"11:physicsPath" ] }'), ('TestFilter', 'Mu2eG4')])
def test_bad_schedule_or_simulation_modules_rejected(old, new):
    with pytest.raises(ValueError):
        regenerate.prepare(fixture().replace(old, new), history(), '/private')


def report(payload):
    return {'parents': [{'did': 'mu2e:dts.test.a.art', 'url': 'root://door//pnfs/fnal.gov/usr/mu2e/test/dts.test.a.art'}],
        'catalog': {'files': [{'name': 'dts.test.a.art', 'size': len(payload), 'retired': False,
                             'checksums': {'adler32': f'{zlib.adler32(payload):08x}'}}]}}


def test_stage_validates_namespace_and_checksum(tmp_path):
    payload = b'abc'
    item = report(payload)
    assert stage_parents.records(item)[0][1] == '//pnfs/fnal.gov/usr/mu2e/test/dts.test.a.art'
    (tmp_path / 'dts.test.a.art').write_bytes(payload)
    reader = stage_parents.CheckedReader(io.BytesIO(payload))
    assert reader.read() == payload
    receipt = [{'name': 'dts.test.a.art', **stage_parents.checksum(reader, 3, f'{zlib.adler32(payload):08x}')}]
    (tmp_path / 'stream-receipt.json').write_text(json.dumps(receipt))
    assert stage_parents.verify(item, tmp_path)['equal']
    (tmp_path / 'dts.test.a.art').write_bytes(b'bad')
    with pytest.raises(ValueError, match='adler32'):
        stage_parents.verify(item, tmp_path)


def test_source_stream_read_only_archive_is_checksummed(monkeypatch):
    payload = b'test-bytes'
    monkeypatch.setattr('builtins.open', lambda path, mode: io.BytesIO(payload))
    output = io.BytesIO()
    stage_parents.stream(report(payload), output)
    output.seek(0)
    with tarfile.open(fileobj=output) as archive:
        assert archive.extractfile('dts.test.a.art').read() == payload
        assert json.load(archive.extractfile('stream-receipt.json'))[0]['size'] == len(payload)


@pytest.mark.parametrize('field,value', [('name', '../bad'), ('retired', True), ('size', 0)])
def test_bad_parent_records_fail(field, value):
    item = report(b'abc')
    item['catalog']['files'][0][field] = value
    with pytest.raises(ValueError):
        stage_parents.records(item)


def test_split_output_preserves_physics_and_parent_order():
    result, receipt = regenerate.prepare(fixture(), history(), '/private', split_input_files=True)
    assert result['outputs']['Output']['fileName'] == 'unfiltered_%#.art'
    assert replay.plain(result['outputs']['Output']['fileProperties']) == {'maxInputFiles': '1', 'granularity': 'InputFile'}
    assert receipt['split_input_files']
    assert result['physics'] == replay.production_config(fixture(), 'Digitize')['physics']


@pytest.mark.parametrize('value', ['-1', '0', 'all', '01', '10000000000'])
def test_regeneration_batch_rejects_unbounded_count_before_modules(tmp_path, value):
    import subprocess
    env = {key: 'unused' for key in ('TRIGGER_CODE', 'TRIGGER_PREPARED', 'TRIGGER_PARENTS', 'TRIGGER_SOURCE_AUDIT',
        'TRIGGER_OUTPUT', 'TRIGGER_AREA', 'CVMFSEXEC_DIST', 'TRIGGER_ANALYSIS_IMAGE', 'TRIGGER_PYTHON')}
    env['OFFLINE_EVENTS'] = value
    result = subprocess.run(['/bin/bash', str(TRIGGER / 'regenerate.sbatch')], env=env, text=True, capture_output=True)
    assert result.returncode == 2 and 'bounded positive' in result.stderr


@pytest.mark.parametrize('key,value', [('OFFLINE_WARM_REPEAT', '1'), ('OFFLINE_OUTPUT_FILE', '/synthetic.root')])
def test_regeneration_batch_rejects_inherited_output_repeat_before_art(key, value):
    import subprocess
    env = {key: 'unused' for key in ('TRIGGER_CODE', 'TRIGGER_PREPARED', 'TRIGGER_PARENTS', 'TRIGGER_SOURCE_AUDIT',
        'TRIGGER_OUTPUT', 'TRIGGER_AREA', 'CVMFSEXEC_DIST', 'TRIGGER_ANALYSIS_IMAGE', 'TRIGGER_PYTHON')}
    env.update(OFFLINE_EVENTS='100')
    env[key] = value
    result = subprocess.run(['/bin/bash', str(TRIGGER / 'regenerate.sbatch')], env=env, text=True, capture_output=True)
    assert result.returncode == 2 and 'forbids' in result.stderr
