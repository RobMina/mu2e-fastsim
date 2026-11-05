"""Primary-only step bridge retains stock context and never imports extracted filtering/geometry."""
from pathlib import Path
import sys
import json
import hashlib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import step_handoff
import replay


CONFIG = '''Digitize:{services:{GeometryService:{inputFile:"production-geometry"} SeedService:{baseSeed:9 policy:autoIncrement}}
physics:{producers:{makeSD:{module_type:StrawDigisFromStrawGasSteps module_label:makeSD StrawGasStepModules:["compressDetStepMCs"] AllHitsStraw:91}}}}'''


def test_distinct_fast_steps_never_shadow_stock_calo_or_timing():
    result, receipt = step_handoff.prepare(CONFIG, ['/private/a.art', '/private/b.art'], '/private/fast.root')
    assert set(result['physics']['producers']) == {'R7FastSteps', 'R7FastSD'}
    assert result['physics']['producers']['R7FastSD']['StrawGasStepModules'] == ['R7FastSteps::R7TrackerInject']
    assert 'module_label' not in result['physics']['producers']['R7FastSD']
    assert 'filters' not in result['physics']
    assert 'SelectEvents' not in result['outputs']['Combined']
    assert result['outputs']['Combined']['outputCommands'] == ['keep *']
    assert replay.plain(result['services']['GeometryService']) == {'inputFile': 'production-geometry'}
    assert result['source']['fileNames'] == ['/private/a.art', '/private/b.art']
    assert replay.plain(result['outputs']['Combined']['fileProperties']) == {'maxInputFiles': '1', 'granularity': 'InputFile'}
    assert result['physics']['producers']['R7FastSD']['EventWindowMarker'] == 'EWMProducer::Digitize'
    assert result['physics']['producers']['R7FastSD']['ProtonBunchTimeMC'] == 'EWMProducer::Digitize'
    assert 'primary-only' in receipt['scope']


def test_truth_dump_does_not_digitize_or_filter():
    result, receipt = step_handoff.prepare(CONFIG, ['/private/a.art'])
    assert 'producers' not in result['physics'] and 'trigger_paths' not in result['physics']
    assert result['physics']['analyzers']['dump']['genParticles'] == 'compressDetStepMCs'
    assert receipt['table_file'] is None


@pytest.mark.parametrize('inputs,seed', [([], 1), (['a', 'a'], 1), (['a'], 0), (['a'], True)])
def test_invalid_inputs_or_seeds_rejected(inputs, seed):
    with pytest.raises(ValueError):
        step_handoff.prepare(CONFIG, inputs, seed=seed)


def test_injection_preflight_binds_source_table_config_and_seed(tmp_path):
    source, table = tmp_path / 'source.art', tmp_path / 'table.root'
    source.write_bytes(b'original-source')
    table.write_bytes(b'audited-fast-table')
    result, receipt = step_handoff.prepare(CONFIG, [str(source)], str(table))
    text = '\n'.join(f'{key}: {replay.render(value)}' for key, value in result.items()) + '\n'
    receipt['fcl_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    (tmp_path / 'job.fcl').write_text(text)
    (tmp_path / 'manifest.json').write_text(json.dumps(receipt))
    report = step_handoff.preflight(tmp_path, CONFIG, table, source)
    assert report['table']['sha256'] == hashlib.sha256(b'audited-fast-table').hexdigest()
    admission = {'inputs': [report['source'], report['table']]}
    step_handoff.preflight(tmp_path, CONFIG, table, source, admission)
    table.write_bytes(b'changed-after-admission')
    with pytest.raises(ValueError, match='bytes differ'):
        step_handoff.preflight(tmp_path, CONFIG, table, source, admission)
    with pytest.raises(ValueError, match='audited fast table'):
        step_handoff.preflight(tmp_path, CONFIG, tmp_path / 'other.root', source)
    (tmp_path / 'job.fcl').write_text(text + '\n')
    with pytest.raises(ValueError, match='FCL changed'):
        step_handoff.preflight(tmp_path, CONFIG, table, source)
