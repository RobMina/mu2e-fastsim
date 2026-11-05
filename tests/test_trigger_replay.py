"""Synthetic trigger provenance and decision-log contract tests."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f'trigger_{name}', ROOT / 'trigger' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = load('replay')
comparison = load('compare')


def fixture():
    return replay.parse_expanded('''
physics: {
  DigitizePath: [ EWMProducer, makeSD ]
  TriggerablePath: [ Truth ]
  calo_cluster: [ EWMProducer, makeSD, PS, TrigMakeSD, Hits, Match, Cut ]
  trigger_paths: [ DigitizePath, "20:TriggerablePath", "425:calo_cluster" ]
  producers: {
    EWMProducer: { module_type: EventWindowMarkerProducer }
    makeSD: { module_type: StrawDigisFromStepPointMCs }
    TrigMakeSD: { module_type: StrawDigisFromArtdaqFragments }
    Hits: { module_type: CaloHitMakerFast caloDigiCollection: "CaloDigiMaker" ProtonBunchTimeTag: "EWMProducer" }
    Match: { module_type: SelectRecoMC }
  }
  filters: {
    PS: { module_type: PrescaleEvent eventModeConfig: [ { eventMode: OnSpill prescale: 1 } ] }
    Cut: { module_type: CaloClusterFilter }
    Truth: { module_type: TruthFilter }
  }
}
outputs: { Output: { SelectEvents: [ "calo_*", TriggerablePath ] } }
services: { GeometryService: { inputFile: "geometry.txt" } DbService: { purpose: Sim_best version: v1_5 } }
''')


def test_roundtrip_and_comments():
    config = fixture()
    assert replay.parse_expanded(replay.render(config)[1:-1]) == config
    assert replay.parse_expanded('a: "# is literal" # ignore\nb: -1')['a'] == '# is literal'


@pytest.mark.parametrize('text', ['a: 1 a: 2', 'a: @local::x', 'a: [ 1', 'a: {', 'a: "bad'])
def test_parser_rejects_unsupported_or_truncated(text):
    with pytest.raises((ValueError, IndexError)):
        replay.parse_expanded(text)


def test_embedded_process_excludes_dumper_preamble_and_parent():
    text = 'Processing file: input.art\nDigitize: ' + replay.render(fixture()) + '\nPrimary: { source: {} }'
    assert replay.production_config(text, 'Digitize') == fixture()
    with pytest.raises(ValueError):
        replay.production_config(text, 'absent')


def test_inventory_and_replay_preserve_bits_digi_timing_and_services():
    config = fixture()
    original = deepcopy(config)
    inv = replay.inventory(config)
    assert inv['paths'][0]['bit'] == '425'
    assert inv['paths'][0]['prescales'][0]['eventModeConfig'][0]['prescale'] == '1'
    result, manifest = replay.prepare(config)
    assert config == original
    assert result['physics']['trigger_paths'] == ['425:calo_cluster']
    assert result['physics']['calo_cluster'] == ['PS', 'TrigMakeSD', 'Hits', 'Cut']
    assert replay.plain(result['physics']['producers']['TrigMakeSD']['module_type']) == 'NullProducer'
    assert 'EWMProducer' not in result['physics']['producers']
    assert result['services'] == config['services']
    assert manifest['removed_mc_matching'] == ['Match']
    assert result['source']['inputCommands'][0] == 'keep *'
    assert not any('TriggerResults' in s for s in result['source']['inputCommands'])
    assert 'SelectEvents' not in result['outputs']['DecisionProducts']


def test_stateful_path_is_labelled_but_mismatches_are_not_excused():
    config = fixture()
    config['physics']['filters']['Cut']['module_type'] = 'LumiStreamFilter'
    _, prepared = replay.prepare(config)
    assert prepared['stateful_paths'] == ['calo_cluster']
    text = log(candidate='fail').replace('ProductPrint art::TriggerResults_TriggerResults__TriggerReplay',
        '  fail 20 TriggerablePath\nProductPrint art::TriggerResults_TriggerResults__TriggerReplay')
    report = comparison.compare(text, prepared, 1)
    assert not report['equal'] and report['mismatches'] == 1
    assert report['nonstateful_mismatches'] == 0


def test_duplicate_path_bits_refused():
    config = fixture()
    config['physics']['trigger_paths'].append('425:other')
    with pytest.raises(ValueError, match='duplicate'):
        replay.paths(config)


def log(stock='pass', candidate='pass', event=3):
    return f'''###############  PrintModule Run/Subrun/Event 1 2 {event}
ProductPrint art::TriggerResults_TriggerResults__Digitize
  {stock} 425 calo_cluster
ProductPrint art::TriggerResults_TriggerResults__TriggerReplay
  {candidate} 425 calo_cluster
'''


def manifest():
    return {'original_process': 'Digitize', 'replay_process': 'TriggerReplay',
            'paths': ['calo_cluster'], 'bits': ['425']}


def test_exact_and_mismatch():
    report = comparison.compare(log() + log('fail', 'fail', 4), manifest(), 2)
    assert report['equal'] and report['events'] == 2
    report = comparison.compare(log(candidate='fail'), manifest(), 1)
    assert not report['equal']
    assert report['paths']['calo_cluster']['mismatches'] == 1


@pytest.mark.parametrize('text', ['', log() + log(), log().replace('__TriggerReplay', '__Other'),
    log().replace('425 calo_cluster', '426 calo_cluster'),
    log().replace('pass 425 calo_cluster', 'pass 425 other'),
    log() + '  pass 425 calo_cluster\n'])
def test_incomplete_duplicate_or_wrong_paths_rejected(text):
    with pytest.raises(ValueError):
        comparison.compare(text, manifest(), 1)


def test_expected_count_required():
    with pytest.raises(ValueError, match='finished event count'):
        comparison.compare(log(), manifest(), 2)
    with pytest.raises(ValueError, match='positive'):
        comparison.compare(log(), manifest(), 0)
    assert comparison.compare(log(), manifest(), 0, [(1, 2, 3)])['events'] == 1
    with pytest.raises(ValueError, match='positive'):
        comparison.compare(log(), manifest(), 0)


@pytest.mark.parametrize('bit', ['0425', '-1', 'abc', '2147483648', '+1'])
def test_preparation_rejects_noncanonical_or_unsupported_bits(bit):
    config = fixture()
    config['physics']['trigger_paths'][-1] = f'{bit}:calo_cluster'
    with pytest.raises(ValueError, match='trigger bit'):
        replay.prepare(config)


@pytest.mark.parametrize('event', [99999999, 100000000, 1234567890])
def test_printer_fixed_width_event_boundary(event):
    text = log(event=event).replace(f'1 2 {event}', f'{1:9d}{2:9d}{event:9d}')
    assert comparison.compare(text, manifest(), 1)['equal']
    assert comparison.event_key(f'{1:9d}{2:9d}{event:9d}') == (1, 2, event)


@pytest.mark.parametrize('extra', ['  exception 425 calo_cluster', '  pass abc calo_cluster',
    '  fail 425', '  notrun 425 calo_cluster', '  pass 0425 calo_cluster',
    'ProductPrint art::TriggerResults_BROKEN'])
def test_unsupported_records_rejected(extra):
    with pytest.raises(ValueError, match='decision|TriggerResults'):
        comparison.compare(log() + extra + '\n', manifest(), 1)


@pytest.mark.parametrize('change', [{'bits': ['0425']}, {'bits': [-1]},
    {'bits': ['2147483648']}, {'stateful_paths': ['absent']}, {'replay_process': 'Digitize'}])
def test_manifest_validation(change):
    item = manifest()
    item.update(change)
    with pytest.raises(ValueError):
        comparison.compare(log(), item, 1)


def test_input_bindings_and_removed_mc_dependency():
    result, prepared = replay.prepare(fixture())
    binding = next(b for b in prepared['external_bindings'] if b['product'] == 'CaloDigis')
    assert binding['tag'] == 'CaloDigiMaker'
    config = fixture()
    config['physics']['producers']['Hits']['caloDigiCollection'] = 'Match'
    with pytest.raises(ValueError, match='removed MC'):
        replay.prepare(config)
    config = fixture()
    config['physics']['filters']['Cut']['clusterTag'] = 'Hits::Digitize'
    with pytest.raises(ValueError, match='old reconstruction'):
        replay.prepare(config)


def test_fragment_unpacking_replaced_by_type_not_label():
    config = fixture()
    config['physics']['producers']['Unpack'] = config['physics']['producers'].pop('TrigMakeSD')
    config['physics']['calo_cluster'][3] = 'Unpack'
    result, _ = replay.prepare(config)
    assert replay.plain(result['physics']['producers']['Unpack']['module_type']) == 'NullProducer'


def test_input_event_ids_and_order_not_just_counts():
    text = 'Run SubRun Event\n 1\n 1 2\n 1 2 3\n 1 2 4\n'
    assert comparison.input_event_keys(text) == [(1, 2, 3), (1, 2, 4)]
    report = comparison.compare(log(), manifest(), 1, [(1, 2, 3)])
    assert report['input_event_ids_verified']
    with pytest.raises(ValueError, match='IDs/order'):
        comparison.compare(log(), manifest(), 1, [(1, 2, 4)])
    with pytest.raises(ValueError, match='duplicate'):
        comparison.input_event_keys(text + '1 2 3\n')


def test_list_and_tuple_identities_preserve_the_same_order_contract():
    text = log() + log(event=4)
    expected = [(1, 2, 3), (1, 2, 4)]
    report = comparison.compare(text, manifest(), 2, expected)
    assert comparison.compare(text, manifest(), 2, [list(key) for key in expected]) == report
    for wrong in (expected[::-1], expected[:1], [expected[0]] * 2, [(1, 2, 3), (1, 2, 5)]):
        with pytest.raises(ValueError, match='IDs/order'):
            comparison.compare(text, manifest(), 2, [list(key) for key in wrong])


def test_blank_block_boundary_does_not_parse_art_summary_as_decisions():
    assert comparison.compare(log() + '\n    1 RANDOM ModuleEndJob\n', manifest(), 1)['equal']


def test_compare_cli_needs_successful_art_completion(tmp_path, monkeypatch):
    import json
    source, prepared, output = [tmp_path / name for name in ('art.log', 'manifest.json', 'report.json')]
    prepared.write_text(json.dumps(manifest()))
    monkeypatch.setattr(sys, 'argv', ['compare.py', str(source), '--manifest', str(prepared),
        '--expected-events', '1', '--output', str(output)])
    source.write_text(log())
    with pytest.raises(ValueError, match='completion'):
        comparison.main()
    assert not output.exists()
    source.write_text(log() + '\nArt has completed and will exit with status 0.\n')
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 0
    assert json.loads(output.read_text())['equal']


def test_compare_cli_exit_code_ignores_stateful_paths_unless_strict(tmp_path, monkeypatch):
    import json
    source, prepared, output, events = [tmp_path / name for name in
                                        ('art.log', 'manifest.json', 'report.json', 'input.events')]
    item = manifest()
    item['stateful_paths'] = ['calo_cluster']
    prepared.write_text(json.dumps(item))
    source.write_text(log(candidate='fail') + '\nArt has completed and will exit with status 0.\n')
    events.write_text('1 2 3\n')
    base = ['compare.py', str(source), '--manifest', str(prepared), '--output', str(output)]
    for extra, code in ((['--expected-events', '1'], 0), (['--expected-events', '1', '--strict'], 1),
                        (['--expected-events', '0', '--input-event-list', str(events)], 0)):
        monkeypatch.setattr(sys, 'argv', base + extra)
        with pytest.raises(SystemExit) as exc:
            comparison.main()
        assert exc.value.code == code
        report = json.loads(output.read_text())
        assert report['events'] == 1 and report['mismatches'] == 1 and report['nonstateful_mismatches'] == 0
    prepared.write_text(json.dumps(manifest()))
    monkeypatch.setattr(sys, 'argv', base + ['--expected-events', '1'])
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 1


def test_direct_digi_count_bindings_are_not_missed():
    config = fixture()
    config['physics']['filters']['Cut'] = {'module_type': 'DigiFilter', 'useStrawDigi': replay.Atom('true'),
        'useCaloDigi': replay.Atom('false'), 'strawDigiCollection': 'fastSD'}
    _, prepared = replay.prepare(config)
    assert any(b['consumer'] == 'Cut' and b['tag'] == 'fastSD' for b in prepared['external_bindings'])


def test_quiet_mode_removes_only_printing_not_trigger_schedule():
    config = fixture()
    original = deepcopy(config)
    verbose, _ = replay.prepare(config)
    quiet, prepared = replay.prepare(config, quiet=True)
    assert config == original
    assert prepared['mode'] == 'quiet_timing'
    assert quiet['physics']['analyzers'] == {}
    assert quiet['physics']['DecisionOutput'] == ['DecisionProducts']
    assert quiet['physics']['trigger_paths'] == verbose['physics']['trigger_paths']
    assert quiet['physics']['producers'] == verbose['physics']['producers']
    assert quiet['physics']['filters'] == verbose['physics']['filters']
    assert quiet['physics']['calo_cluster'] == verbose['physics']['calo_cluster']


def test_digi_substitution_updates_all_known_direct_consumers_not_timing():
    config = fixture()
    config['physics']['producers']['SH'] = {'module_type': 'StrawHitReco',
        'StrawDigiCollectionTag': 'makeSD', 'EventWindowMarker': 'EWMProducer',
        'ProtonBunchTimeTag': 'EWMProducer'}
    config['physics']['calo_cluster'].insert(4, 'SH')
    config['physics']['filters']['Cut'] = {'module_type': 'DigiFilter',
        'useStrawDigi': replay.Atom('true'), 'useCaloDigi': replay.Atom('true'),
        'strawDigiCollection': 'makeSD', 'caloDigiCollection': 'CaloDigiMaker'}
    original = deepcopy(config)
    result, prepared = replay.prepare(config, straw_tag='fastSD::Injected', calo_tag='fastCD::Injected')
    assert config == original
    modules = {**result['physics']['producers'], **result['physics']['filters']}
    assert modules['SH']['StrawDigiCollectionTag'] == modules['Cut']['strawDigiCollection'] == 'fastSD::Injected'
    assert modules['Hits']['caloDigiCollection'] == modules['Cut']['caloDigiCollection'] == 'fastCD::Injected'
    assert replay.plain(modules['SH']['EventWindowMarker']) == 'EWMProducer'
    assert replay.plain(modules['SH']['ProtonBunchTimeTag']) == 'EWMProducer'
    assert len(prepared['digi_substitutions']) == 4
    assert result['physics']['calo_cluster'][-1] == 'Cut'


@pytest.mark.parametrize('tag', ['fastSD', 'fastSD:inst', 'fastSD::', 'Hits::Injected'])
def test_substitution_requires_unambiguous_external_tag(tag):
    with pytest.raises(ValueError, match='explicit|replay module'):
        replay.prepare(fixture(), calo_tag=tag)


def test_output_selection_is_not_the_full_menu_or():
    item = manifest()
    item['stock_output_selection'] = ['calo_*', 'TriggerablePath']
    source = log('pass', 'fail').replace('ProductPrint art::TriggerResults_TriggerResults__TriggerReplay',
        '  pass 20 TriggerablePath\nProductPrint art::TriggerResults_TriggerResults__TriggerReplay')
    report = comparison.compare(source, item, 1)
    assert report['mismatches'] == 1
    assert report['stored_output_selection_mismatches'] == 0
    source = source.replace('pass 20 TriggerablePath', 'fail 20 TriggerablePath')
    assert comparison.compare(source, item, 1)['stored_output_selection_mismatches'] == 1
    item['stock_output_selection'] = ['missing_*']
    with pytest.raises(ValueError, match='matches no original'):
        comparison.compare(source, item, 1)


def test_mismatch_examples_are_bounded_and_identified():
    text = ''.join(log('pass', 'fail', event) for event in range(1, 106))
    report = comparison.compare(text, manifest(), 105)
    assert report['mismatches'] == 105
    assert len(report['mismatch_examples']) == 100
    assert report['mismatch_examples'][0] == {'event': [1, 2, 1], 'path': 'calo_cluster', 'stock': True, 'replay': False}


def test_stored_event_must_pass_original_output_selection():
    item = manifest()
    item['stock_output_selection'] = ['calo_cluster']
    with pytest.raises(ValueError, match='fails its original output'):
        comparison.compare(log('fail', 'fail'), item, 1)


def test_phi_prescale_is_stateful_by_released_filter_invocation_counter():
    config = fixture()
    config['physics']['filters']['Cut'] = {'module_type': 'HelixFilter',
        'posHelicitySelection': {'prescaleUsingD0Phi': replay.Atom('true')},
        'negHelicitySelection': {'configured': replay.Atom('false')}}
    _, prepared = replay.prepare(config)
    assert prepared['stateful_paths'] == ['calo_cluster']
    assert 'invocation count' in prepared['stateful_modules']['Cut']
    config['physics']['filters']['Cut']['posHelicitySelection']['prescaleUsingD0Phi'] = replay.Atom('false')
    _, prepared = replay.prepare(config)
    assert prepared['stateful_paths'] == []
