"""Full-stock trigger preparation demands all ten completed parents and one own counter history."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import response_campaign as campaign


def test_rejects_reused_trigger_preparation_before_reading_inputs(tmp_path):
    output = tmp_path / 'prepared'
    output.mkdir()
    with pytest.raises(ValueError, match='overwrite'):
        campaign.prepare(tmp_path / 'plan', tmp_path, tmp_path, output)


@pytest.mark.parametrize('ordinals', [[1], list(range(10)), list(range(1, 10)) + [9], list(range(10, 0, -1))])
def test_requires_every_original_parent_once_in_numeric_order(tmp_path, monkeypatch, ordinals):
    monkeypatch.setattr(campaign.control, 'load', lambda p: {'parents': [{'ordinal': n} for n in ordinals]})
    with pytest.raises(ValueError, match='every original'):
        campaign.prepare(tmp_path / 'plan', tmp_path, tmp_path, tmp_path / 'new')


def test_batch_separates_response_and_single_history_and_no_builds():
    trigger = Path(campaign.__file__).parent
    inject = (trigger / 'response_inject.sbatch').read_text()
    history = (trigger / 'response_trigger.sbatch').read_text()
    assert 'SLURM_ARRAY_TASK_ID' in inject and 'roundtrip' in inject
    assert 'response_validate.py' in inject
    assert history.count('bash "$TRIGGER_CODE/container/offline/offline_uva.sh"') == 1
    assert 'SLURM_ARRAY_TASK_ID' not in history
    assert '--expected-events 28044' in history
    for text in (inject, history):
        assert 'muse build' not in text and 'scons' not in text
        assert 'OFFLINE_WARM_REPEAT:-0' in text
        assert 'art.exit' in text


@pytest.fixture
def completed_campaign(tmp_path, monkeypatch):
    """Keep orchestration, file hashes, replay preparation and preflight real.

    Only ROOT/art readers are replaced: parent postflight, handoff audit, and
    independent history recertification. Every evidence row binds a local file.
    """
    from copy import deepcopy
    import hashlib
    from types import SimpleNamespace

    replay = campaign.hybrid_replay.replay

    def put(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        campaign.control.save(path, value)
        return path

    # A literal all-52 menu, including one shared stateful filter. This is not
    # runnable physics and never launches art or accesses external inputs.
    production = replay.parse_expanded('''
services: { GeometryService: { inputFile: "fixture-geometry" }
  DbService: { purpose: Sim_best version: v1_5 } }
physics: {
  DigitizePath: [ makeSD ]
  producers: { makeSD: { module_type: StrawDigisFromStrawGasSteps } }
  filters: { PS: { module_type: PrescaleEvent
    eventModeConfig: [ { eventMode: OnSpill prescale: 2 } ] }
    Cut: { module_type: CaloClusterFilter } }
  trigger_paths: [ DigitizePath ]
}
outputs: { Output: { SelectEvents: [ "physics_*" ] } }
''')
    physics_paths = [f'physics_{n}' for n in range(52)]
    for bit, name in enumerate(physics_paths, 100):
        production['physics'][name] = ['PS', 'Cut']
        production['physics']['trigger_paths'].append(f'{bit}:{name}')
    config_text = 'Digitize: ' + replay.render(production) + '\n'
    config = put(tmp_path / 'config.fcl', config_text)
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    plan_path = tmp_path / 'plan.json'
    output_root, fast_prepared, output = [tmp_path / name for name in ('parents', 'fast', 'prepared')]
    rows, sources, candidates = [], [], []
    for ordinal in range(1, 11):
        source = put(tmp_path / 'sources' / f'parent_{ordinal}.art', f'stock parent {ordinal}')
        table = put(tmp_path / 'tables' / f'parent_{ordinal}.root', f'original SGS {ordinal}')
        prepared = tmp_path / 'injection' / f'parent_{ordinal}'
        put(prepared / 'job.fcl', f'prepared injection {ordinal}')
        save(prepared / 'manifest.json', {'ordinal': ordinal})
        table_admission = save(prepared / 'table-admission.json', {'events': 2805 if ordinal <= 4 else 2804})
        row = {'ordinal': ordinal, 'events': 2805 if ordinal <= 4 else 2804,
               'source': campaign.integrity.digest(source), 'table': campaign.integrity.digest(table),
               'prepared': str(prepared), 'table_admission': campaign.integrity.digest(table_admission)}
        rows.append(row)
        sources.append(str(source))
        directory = output_root / f'parent_{ordinal}'
        candidate = put(directory / 'unfiltered_1.art', f'candidate parent {ordinal}')
        candidates.append(str(candidate))
        for relative in ('execution-binding.json', 'runtime-selection.json', 'art.exit',
                         'inject.log', 'inject.expanded.fcl', 'dump-prepared/job.fcl',
                         'dump-prepared/manifest.json', 'roundtrip/runtime-selection.json',
                         'roundtrip/art.exit', 'roundtrip/dump.log', 'roundtrip/dump.expanded.fcl',
                         'roundtrip/source-tables.root'):
            put(directory / relative, f'local evidence {ordinal}: {relative}')
    plan = {'parents': rows, 'events': 28044, 'config': campaign.integrity.digest(config),
            'runtime': {'OFFLINE_IMAGE': 'fixture-image', 'TRIGGER_AREA': 'fixture-release'},
            'admission_kind': campaign.control.KIND, 'scope': campaign.control.SCOPE,
            'primary_only_admission_used': False, 'electronics_only': False,
            'particle_state_lossless': False, 'source_simparticle_product_ids_authenticated': False}
    save(plan_path, plan)
    history_manifest = save(tmp_path / 'history/regeneration.json', {'fixture': True})
    history_comparison = save(tmp_path / 'history/comparison.json', {'fixture': True})
    history = {'equal': True, 'all_reference_events_present': True, 'root_event_ids_verified': True,
               'original_output_selection_event_set_equal': True, 'full_parent_history_verified': True,
               'regenerated_events': plan['events'], 'config_sha256': config_hash,
               'candidate_files': campaign.integrity.files(sources),
               'reference_files': campaign.integrity.files(sources),
               'evidence_files': campaign.integrity.files([history_manifest, history_comparison]),
               'regeneration_manifest_path': str(history_manifest),
               'production_comparison_path': str(history_comparison),
               'regenerated_directory': str(tmp_path / 'sources'),
               'parent_census': {'equal': True, 'root_event_ids_decoded': True,
                   'complete_parent_event_ids_verified': True, 'parent_order_verified': True,
                   'events': plan['events'], 'expected_events': plan['events'],
                   'original_source_events': plan['events'],
                   'original_parent_files': [{'path': row['source']['path'], 'events': row['events']}
                                             for row in rows]}}
    save(fast_prepared / 'manifest.json', {'full_production_history_certified': True,
                                         'production_history_receipt': history})
    put(fast_prepared / 'replay.fcl', 'saved F replay configuration')
    reports = {}
    false_claims = dict.fromkeys(('actual_consumed_ProductID', 'original_SimParticle_ProductID_resolved',
        'source_simparticle_product_ids_authenticated', 'particle_state_lossless', 'electronics_only',
        'primary_only_admission_used'), False)
    for row in rows:
        ordinal = row['ordinal']
        directory = output_root / f'parent_{ordinal}'
        before = campaign.integrity.files([plan_path, config, row['source']['path'], row['table']['path'],
            row['table_admission']['path'], Path(row['prepared']) / 'job.fcl',
            Path(row['prepared']) / 'manifest.json', directory / 'unfiltered_1.art',
            directory / 'execution-binding.json', directory / 'runtime-selection.json',
            directory / 'art.exit', directory / 'inject.log', directory / 'inject.expanded.fcl'])
        dump_before = campaign.integrity.files([directory / name for name in (
            'unfiltered_1.art', 'dump-prepared/job.fcl', 'dump-prepared/manifest.json',
            'roundtrip/runtime-selection.json', 'roundtrip/art.exit', 'roundtrip/dump.log',
            'roundtrip/dump.expanded.fcl', 'roundtrip/source-tables.root')])
        reports[ordinal] = {'schema': 'full-stock-response-postflight-v1', 'validated': True,
            'ordinal': ordinal, 'events': row['events'], 'parent': row,
            'admission_kind': campaign.control.KIND, 'scope': campaign.control.SCOPE,
            'plan': campaign.integrity.digest(plan_path), 'numericalSGS': True,
            'complete_parent_order_verified': True, 'original_SGS_serialized_bytes_preserved': True,
            'consumes_configuration_exact': True,
            'injected_SGS': {'tag': 'R7FastSteps::R7TrackerInject', 'present_all_events': True,
                             'typename': 'std::vector<mu2e::StrawGasStep>'},
            'roundtrip': {'equal': True, 'events': row['events'], 'numericalSGS': True,
                'comparison': {'equal': True, 'events': row['events']},
                'scope': campaign.response_validate.DUMP_SCOPE,
                'inputs_before': dump_before, 'inputs_after': deepcopy(dump_before),
                'output': campaign.integrity.digest(directory / 'roundtrip/source-tables.root'),
                'actual_consumed_ProductID': False, 'particle_state_lossless': False,
                'electronics_only': False},
            'inputs_before': before, 'inputs_after': deepcopy(before),
            'outputs': campaign.integrity.files([directory / 'unfiltered_1.art',
                                                 directory / 'roundtrip/source-tables.root']),
            **false_claims}
        save(directory / 'completed.json', reports[ordinal])
    calls = []

    def validate_parent(path, ordinal, directory):
        assert path == plan_path
        assert directory == output_root / f'parent_{ordinal}'
        assert directory.is_dir()
        calls.append(ordinal)
        return deepcopy(reports[ordinal])

    def recertify(text, manifest, directory, comparison, expected):
        assert (text, manifest, directory, comparison, expected) == (
            config_text, str(history_manifest), str(tmp_path / 'sources'),
            str(history_comparison), plan['events'])
        return deepcopy(history)

    def audit(references, replacements, text, expected, straw_tag):
        assert references == sources
        assert replacements == candidates
        assert (text, expected, straw_tag) == (config_text, 28044, 'R7FastSD::R7TrackerInject')
        return {'same_event_handoff_metadata_passed': True, 'events': expected,
                'reference_files': campaign.integrity.files(references),
                'candidate_files': campaign.integrity.files(replacements),
                'config_sha256': config_hash, 'straw_digi_tag': straw_tag, 'calo_digi_tag': None}

    monkeypatch.setattr(campaign.response_validate, 'validate', validate_parent)
    monkeypatch.setattr(campaign.hybrid_history, 'certify', recertify)
    monkeypatch.setattr(campaign.hybrid_input, 'audit', audit)
    return SimpleNamespace(plan_path=plan_path, plan=plan, output_root=output_root,
        fast_prepared=fast_prepared, output=output, config=config_text, reports=reports,
        sources=sources, candidates=candidates, paths=physics_paths, calls=calls,
        prepare=lambda: campaign.prepare(plan_path, output_root, fast_prepared, output))


def test_prepares_all_completed_parents_with_validator_schema_and_dump_evidence(completed_campaign):
    case = completed_campaign
    manifest = case.prepare()
    assert case.calls == list(range(1, 11))
    assert campaign.control.load(case.output / 'manifest.json') == manifest
    checked, executed = campaign.hybrid_replay.preflight(case.output, case.config)
    assert checked == manifest
    assert manifest['paths'] == case.paths and len(manifest['paths']) == 52
    assert manifest['expected_events'] == 28044
    assert manifest['full_production_history_certified'] is True
    assert manifest['injected_physics_validated'] is False
    assert executed['source']['fileNames'] == case.candidates
    assert manifest['reference_integrity'] == campaign.integrity.files(case.sources)
    assert manifest['input_integrity'] == campaign.integrity.files(case.candidates)
    assert manifest['injection_campaign'] == {'runtime': case.plan['runtime'],
        'scope': campaign.control.SCOPE, 'events': 28044, 'all_planned_parents_verified': True}
    control = manifest['response_control']
    assert control['admission_kind'] == campaign.control.KIND
    assert control['scope'] == campaign.control.SCOPE
    assert 'not electronics-only' in control['scope'] and 'F-C not pure secondaries' in control['scope']
    assert control['trigger_counter_policy'] == 'one process, own continuous all-parent filter invocation history'
    for flag in ('electronics_only', 'particle_state_lossless', 'source_simparticle_product_ids_authenticated'):
        assert control[flag] is False
    assert control['plan'] == campaign.integrity.digest(case.plan_path)
    evidence = []
    for ordinal, report in case.reports.items():
        receipt = case.output_root / f'parent_{ordinal}/completed.json'
        assert control['parents'][ordinal - 1] == report == campaign.control.load(receipt)
        assert 'inputs' not in report  # Actual validator schema, not an obsolete fixture shortcut.
        evidence += campaign.integrity.files([receipt])
        evidence += report['inputs_after'] + report['roundtrip']['inputs_after'] + report['outputs']
    evidence += campaign.integrity.files([case.plan_path, case.fast_prepared / 'manifest.json',
                                        case.fast_prepared / 'replay.fcl'])
    assert manifest['injection_campaign_evidence'] == evidence
    assert evidence == campaign.integrity.files([row['path'] for row in evidence])
    admission = campaign.control.load(case.output / 'admission.json')
    assert admission['candidate_files'] == manifest['input_integrity']
    assert admission['reference_files'] == manifest['reference_integrity']
    # The shared filter appears once as a module in one combined source job,
    # not one replay job/counter history per completed injection parent.
    assert list(executed['physics']['filters']).count('PS') == 1
    assert len(executed['physics']['trigger_paths']) == 52


def test_missing_completed_parent_is_not_admitted(completed_campaign):
    case = completed_campaign
    (case.output_root / 'parent_6/completed.json').unlink()
    with pytest.raises(FileNotFoundError):
        case.prepare()
    assert case.calls == list(range(1, 7))
    assert not case.output.exists()


def test_completed_receipt_must_equal_fresh_postflight(completed_campaign):
    case = completed_campaign
    receipt = case.output_root / 'parent_4/completed.json'
    changed = campaign.control.load(receipt)
    changed['roundtrip']['equal'] = False
    receipt.unlink()
    campaign.control.save(receipt, changed)
    with pytest.raises(ValueError, match='postflight no longer equals completed receipt'):
        case.prepare()
    assert case.calls == [1, 2, 3, 4]
    assert not case.output.exists()


def test_saved_F_must_certify_full_history(completed_campaign):
    case = completed_campaign
    path = case.fast_prepared / 'manifest.json'
    saved = campaign.control.load(path)
    saved['full_production_history_certified'] = False
    path.unlink()
    campaign.control.save(path, saved)
    with pytest.raises(ValueError, match='complete certified original stored history'):
        case.prepare()
    assert case.calls == list(range(1, 11))
    assert not case.output.exists()


@pytest.mark.parametrize('field', ['reference_integrity', 'input_integrity'])
def test_preflight_rejects_reordered_parent_inputs(completed_campaign, field):
    case = completed_campaign
    manifest = case.prepare()
    manifest[field].reverse()
    path = case.output / 'manifest.json'
    path.unlink()
    campaign.control.save(path, manifest)
    with pytest.raises(ValueError, match='source bytes or order changed|ordered hybrid stock reference.*order changed'):
        campaign.hybrid_replay.preflight(case.output, case.config)


@pytest.mark.parametrize('relative', ['completed.json', 'dump-prepared/job.fcl',
    'dump-prepared/manifest.json', 'roundtrip/dump.log', 'roundtrip/dump.expanded.fcl',
    'roundtrip/source-tables.root'])
def test_preflight_rechecks_completed_and_roundtrip_evidence(completed_campaign, relative):
    case = completed_campaign
    case.prepare()
    path = case.output_root / 'parent_10' / relative
    path.write_text(path.read_text() + '\nchanged after preparation')
    with pytest.raises(ValueError, match='injection campaign evidence changed'):
        campaign.hybrid_replay.preflight(case.output, case.config)
