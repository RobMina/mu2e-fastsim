"""Parent plans preserve explicit numeric order and reset electronics without rerunning a completed first parent."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import injection_campaign as campaign
import integrity
import replay
from test_trigger_step_handoff import CONFIG


def plan(tmp_path, n=10):
    config = tmp_path / 'config.fcl'
    config.write_text(CONFIG)
    sources = []
    for ordinal in range(1, n + 1):
        path = tmp_path / f'stock_{ordinal}.art'
        path.write_bytes(f'parent{ordinal}'.encode())
        sources.append(path)
    return campaign.create(sources, list(range(1, n + 1)),
        [tmp_path / f'transport_{i}' for i in range(1, n + 1)],
        [tmp_path / f'inject_{i}' for i in range(1, n + 1)],
        [tmp_path / f'prepare_{i}' for i in range(1, n + 1)], config, 20261104, {})


def test_ten_parent_plan_order_and_prepare_only_remaining(tmp_path):
    value = plan(tmp_path)
    assert value['events'] == 55
    assert [row['ordinal'] for row in campaign.check_plan(value)] == list(range(1, 11))
    assert value['parents'][9]['source']['path'].endswith('stock_10.art')
    with pytest.raises(ValueError, match='first parent'):
        campaign.prepare_parent(value, 1, tmp_path / 'env')
    row = value['parents'][9]
    transport = Path(row['transport'])
    transport.mkdir()
    (transport / 'step-admission.json').write_text(json.dumps({'events': 10, 'source_birth_keys_complete': True}))
    campaign.prepare_parent(value, 10, tmp_path / 'env')
    manifest = campaign.load(Path(row['prepared']) / 'manifest.json')
    fcl = replay.parse_expanded((Path(row['prepared']) / 'job.fcl').read_text())
    assert manifest['digitizer_seed'] == 20261104
    assert fcl['source']['fileNames'] == [row['source']['path']]
    assert 'OFFLINE_EVENTS=10' in (tmp_path / 'env').read_text()
    with pytest.raises(ValueError, match='reuse'):
        campaign.prepare_parent(value, 10, tmp_path / 'env2')


def test_campaign_rejects_reorder_or_count_changes(tmp_path):
    value = plan(tmp_path)
    value['parents'][0], value['parents'][1] = value['parents'][1], value['parents'][0]
    with pytest.raises(ValueError, match='parent order'):
        campaign.check_plan(value)


def test_campaign_rejects_changed_source_before_preparation(tmp_path):
    value = plan(tmp_path)
    Path(value['parents'][1]['source']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='stock parent bytes'):
        campaign.prepare_parent(value, 2, tmp_path / 'env')


def test_campaign_rejects_extra_injection_split(tmp_path):
    value = plan(tmp_path)
    directory = Path(value['parents'][0]['injection'])
    directory.mkdir()
    (directory / 'unfiltered_1.art').touch()
    (directory / 'unfiltered_2.art').touch()
    with pytest.raises(ValueError, match='boundaries'):
        campaign.completed(value, [1])


def test_expanded_fhicl_numeric_spellings_are_semantic_not_tolerance():
    assert campaign.semantic(replay.parse_expanded('a:20261104 b:0.03')) == campaign.semantic(
        replay.parse_expanded('a:2.0261104e7 b:3e-2'))
    assert campaign.semantic(replay.parse_expanded('a:1.000000001')) != campaign.semantic(
        replay.parse_expanded('a:1.000000002'))


@pytest.fixture
def finished(tmp_path):
    config = tmp_path / 'config.fcl'
    config.write_text(CONFIG.replace('GeometryService:', 'DbService:{purpose:Sim_best} RandomNumberGenerator:{defaultEngineKind:MixMaxRng} GeometryService:'))
    source = tmp_path / 'source.art'
    source.write_bytes(b'stock')
    directory, transport, prepared = [tmp_path / name for name in ('injection', 'transport', 'prepared')]
    (transport / 'transport').mkdir(parents=True)
    table = transport / 'transport/fasttrk_field.root'
    table.write_bytes(b'audited steps')
    value = campaign.create([source], [2], [transport], [directory], [prepared], config, 20261104,
        {'OFFLINE_IMAGE': '/cvmfs/pinned/image', 'TRIGGER_AREA': '/cvmfs/released/area'})
    result, manifest = campaign.step_handoff.prepare(config.read_text(), [str(source)], str(table), 20261104)
    text = '\n'.join(f'{key}: {replay.render(item)}' for key, item in result.items()) + '\n'
    import hashlib
    manifest['fcl_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    prepared.mkdir()
    (prepared / 'job.fcl').write_text(text)
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    (directory / 'runtime/plugins').mkdir(parents=True)
    plugin = directory / 'runtime/plugins/libtrkfastsim_fake_module.so'
    plugin.write_bytes(b'plugin')
    runner = directory / 'runtime/offline_plugins.sh'
    runner.write_bytes(b'runner')
    candidate = directory / 'unfiltered_1.art'
    candidate.write_bytes(b'replacement')
    step = {'events': 2, 'source_birth_keys_complete': True, 'zero_step_events': 1,
        'inputs': integrity.files([source, table])}
    admission = {'events': 2, 'same_event_handoff_metadata_passed': True,
        'reference_files': integrity.files(source), 'candidate_files': integrity.files(candidate),
        'config_sha256': value['config']['sha256'], 'straw_digi_tag': 'R7FastSD::R7TrackerInject', 'calo_digi_tag': None}
    records = {'step-admission.json': step, 'admission.json': admission,
        'event-count.json': {'equal': True, 'events': 2},
        'runtime/runtime.json': {'plugins': integrity.files(plugin), 'adapted_runner_sha256': integrity.digest(runner)['sha256']},
        'execution-binding.json': campaign.step_handoff.preflight(prepared, config.read_text(), table, source, step)}
    for name, record in records.items():
        (directory / name).write_text(json.dumps(record))
    (directory / 'art.exit').write_text('0\n')
    (directory / 'inject.log').write_text('image=/cvmfs/pinned/image\nMUSE_WORK_DIR=/cvmfs/released/area stub\nArt has completed and will exit with status 0.\n')
    (directory / 'inject.expanded.fcl').write_text(text.replace('20261104', '2.0261104e7'))
    return value, directory, table


def test_completed_injection_binds_reused_parent_and_zeros(finished):
    value, directory, table = finished
    report = campaign.completed(value)
    assert report['all_planned_parents_verified']
    assert report['parents'][0]['zero_step_events'] == 1
    assert report['events'] == 2
    assert report['candidate'] == [str(directory / 'unfiltered_1.art')]
    table.write_bytes(b'changed steps')
    with pytest.raises(ValueError, match='bytes differ'):
        campaign.completed(value)


@pytest.mark.parametrize('kind', ['seed', 'importer', 'source', 'runtime', 'candidate', 'plugin'])
def test_completed_injection_rejects_changed_execution(finished, kind):
    value, directory, table = finished
    expanded = directory / 'inject.expanded.fcl'
    if kind == 'seed':
        expanded.write_text(expanded.read_text().replace('2.0261104e7', '2.0261105e7'))
    elif kind == 'importer':
        expanded.write_text(expanded.read_text().replace('primaryKey: 1', 'primaryKey: 2'))
    elif kind == 'source':
        expanded.write_text(expanded.read_text().replace('source.art', 'other.art'))
    elif kind == 'runtime':
        log = directory / 'inject.log'
        log.write_text(log.read_text().replace('/cvmfs/pinned/image', '/cvmfs/wrong/image'))
    elif kind == 'candidate':
        (directory / 'unfiltered_1.art').write_bytes(b'changed replacement')
    else:
        (directory / 'runtime/plugins/libtrkfastsim_fake_module.so').write_bytes(b'changed plugin')
    with pytest.raises(ValueError):
        campaign.completed(value)
