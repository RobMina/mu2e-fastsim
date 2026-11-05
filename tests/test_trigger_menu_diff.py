"""Synthetic embedded-production menu comparison tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trigger_menu_diff', ROOT / 'trigger' / 'menu_diff.py')
menu_diff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(menu_diff)


def inventory():
    modules = {
        'Shared': {'module_type': 'Fit', 'module_label': 'Shared',
                   'settings': {'threshold': '5', 'weights': ['1', '2'], 'optional': None}},
        'PS': {'module_type': 'PrescaleEvent',
               'eventModeConfig': [{'eventMode': 'OnSpill', 'prescale': '10'}]},
        'Cut': {'module_type': 'Filter', 'minimum': '2'},
    }
    return {
        'paths': [
            {'path': 'A', 'bit': '425', 'sequence': ['Shared', 'PS', 'Cut'],
             'prescales': [{'label': 'PS', 'eventModeConfig': deepcopy(modules['PS']['eventModeConfig'])}],
             'modules': deepcopy(modules)},
            {'path': 'B', 'bit': None, 'sequence': ['Shared'], 'prescales': [],
             'modules': {'Shared': deepcopy(modules['Shared'])}},
        ],
        'digitize_modules': ['EWMProducer', 'makeSD', 'CaloDigiMaker'],
        'output_selection': ['A', 'B'], 'geometry': {'inputFile': 'private-geometry.txt'},
        'conditions': {'purpose': 'Sim_best', 'version': 'v1_5'},
    }


def test_reordered_dicts_equal_and_deterministic_without_mutation():
    before = inventory()
    original = deepcopy(before)

    def reverse_objects(value):
        if isinstance(value, dict):
            return {key: reverse_objects(value[key]) for key in reversed(value)}
        if isinstance(value, list):
            return [reverse_objects(item) for item in value]
        return value

    after = reverse_objects(before)
    report = menu_diff.compare(before, after)
    assert report['equal']
    assert report['summary']['paths_changed'] == report['summary']['modules_changed'] == 0
    assert before == original
    assert menu_diff.canonical(report) == menu_diff.canonical(menu_diff.compare(before, after))
    json.dumps(report, allow_nan=False)


def test_path_add_remove_bit_and_sequence_order_are_distinct():
    before, after = inventory(), inventory()
    after['paths'][0]['bit'] = '426'
    after['paths'][0]['sequence'] = ['Cut', 'Shared', 'PS']
    after['paths'][1]['path'] = 'C'
    report = menu_diff.compare(before, after)
    assert not report['equal']
    assert [row['path'] for row in report['paths']['added']] == ['C']
    assert [row['path'] for row in report['paths']['removed']] == ['B']
    assert report['paths']['changed']['A']['bit'] == {'reference': '425', 'candidate': '426'}
    assert report['paths']['changed']['A']['sequence']['reference'] == ['Shared', 'PS', 'Cut']
    assert report['summary']['sequences_changed'] == report['summary']['bits_changed'] == 1
    assert report['summary']['paths_changed'] == 1
    assert report['modules']['changed'] == []
    assert 'modules' not in report['paths']['added'][0]
    assert report['paths']['added'][0]['module_labels'] == ['Shared']


def test_prescale_configuration_changes_are_exact_and_globally_recorded():
    before, after = inventory(), inventory()
    after['paths'][0]['prescales'][0]['eventModeConfig'][0]['prescale'] = '20'
    after['paths'][0]['modules']['PS']['eventModeConfig'][0]['prescale'] = '20'
    report = menu_diff.compare(before, after)
    assert report['summary']['prescales_changed'] == 1
    assert report['summary']['modules_changed'] == 1
    assert report['paths']['changed']['A']['prescales']['candidate'][0]['eventModeConfig'][0]['prescale'] == '20'
    row = report['modules']['changed'][0]
    assert row['label'] == 'PS'
    assert row['changes'] == [{'op': 'replace', 'pointer': '/eventModeConfig/0/prescale',
                               'reference': '10', 'candidate': '20'}]
    assert row['reference'] != row['candidate']


def test_nested_settings_shared_once_with_affected_paths_and_no_ignored_label():
    before, after = inventory(), inventory()
    for path in after['paths']:
        module = path['modules']['Shared']
        module['settings']['weights'].reverse()
        del module['settings']['optional']
        module['settings']['new/key~'] = {'enabled': True}
        module['module_label'] = 'OtherLabel'
    report = menu_diff.compare(before, after)
    assert report['summary']['modules_changed'] == 1
    assert report['summary']['paths_changed'] == 2
    assert report['summary']['paths_locally_changed'] == 0
    assert report['paths']['changed'] == {}
    row = report['modules']['changed'][0]
    assert row['affected_paths'] == row['reference_paths'] == row['candidate_paths'] == ['A', 'B']
    assert row['reference'] == before['paths'][0]['modules']['Shared']
    assert row['candidate'] == after['paths'][0]['modules']['Shared']
    by_pointer = {change['pointer']: change for change in row['changes']}
    assert by_pointer['/module_label']['candidate'] == 'OtherLabel'
    assert by_pointer['/settings/optional'] == {'op': 'remove', 'pointer': '/settings/optional', 'reference': None}
    assert by_pointer['/settings/new~1key~0']['op'] == 'add'
    assert by_pointer['/settings/weights/0']['candidate'] == '2'
    # JSON booleans, integers and decimal/string scalars are not conflated.
    assert menu_diff.changes(True, 1)[0]['op'] == 'replace'
    assert menu_diff.changes('1', 1)[0]['op'] == 'replace'


def test_module_add_remove_type_and_context_comparisons():
    before, after = inventory(), inventory()
    path = after['paths'][0]
    path['sequence'][-1] = 'NewCut'
    del path['modules']['Cut']
    path['modules']['NewCut'] = {'module_type': 'NewFilter', 'minimum': '3'}
    for row in after['paths']:
        row['modules']['Shared']['module_type'] = 'NewFit'
    after['digitize_modules'].reverse()
    after['output_selection'] = ['A']
    after['geometry']['inputFile'] = 'another-private-file.txt'
    after['conditions']['version'] = 'v2'
    report = menu_diff.compare(before, after)
    assert [row['label'] for row in report['modules']['added']] == ['NewCut']
    assert [row['label'] for row in report['modules']['removed']] == ['Cut']
    assert report['modules']['added'][0]['settings'] == path['modules']['NewCut']
    assert report['modules']['changed'][0]['type'] == {'reference': 'Fit', 'candidate': 'NewFit'}
    for key in menu_diff.CONTEXT_FIELDS:
        assert not report[key]['equal']
        assert report[key]['changes']
    assert report['summary']['modules_added'] == report['summary']['modules_removed'] == 1
    assert report['summary']['paths_changed'] == 2


@pytest.mark.parametrize('kind', ['name', 'bit', 'shared', 'undefined', 'missing', 'module_type',
                                  'prescales', 'nonfinite', 'bit_type', 'extra_module'])
def test_malformed_inventory_fails_closed(kind):
    source = inventory()
    if kind == 'name':
        source['paths'][1]['path'] = 'A'
    elif kind == 'bit':
        source['paths'][1]['bit'] = '425'
    elif kind == 'shared':
        source['paths'][1]['modules']['Shared']['settings']['threshold'] = '9'
    elif kind == 'undefined':
        source['paths'][1]['sequence'].append('Missing')
    elif kind == 'missing':
        del source['paths'][0]['prescales']
    elif kind == 'module_type':
        del source['paths'][0]['modules']['Shared']['module_type']
    elif kind == 'prescales':
        source['paths'][0]['prescales'][0]['eventModeConfig'][0]['prescale'] = '999'
    elif kind == 'nonfinite':
        source['geometry']['bad'] = float('nan')
    elif kind == 'bit_type':
        source['paths'][0]['bit'] = True
    else:
        source['paths'][0]['modules']['Unscheduled'] = {'module_type': 'Fit'}
    with pytest.raises(ValueError):
        menu_diff.compare(inventory(), source)


def test_path_order_unknown_metadata_and_json_array_changes_not_silently_ignored():
    before, after = inventory(), inventory()
    after['paths'].reverse()
    after['new_metadata'] = None
    after['paths'][0]['annotation'] = 'new'
    report = menu_diff.compare(before, after)
    assert not report['equal']
    assert report['paths']['order'] == {'equal': False, 'reference': ['A', 'B'], 'candidate': ['B', 'A']}
    assert report['metadata_changes'] == [{'op': 'add', 'pointer': '/new_metadata', 'candidate': None}]
    assert report['paths']['changed']['B']['metadata_changes'][0]['pointer'] == '/annotation'
    assert menu_diff.changes([None], []) == [{'op': 'remove', 'pointer': '/0', 'reference': None}]
    assert menu_diff.changes([], [None]) == [{'op': 'add', 'pointer': '/0', 'candidate': None}]


def test_cli_report_and_bounded_preview_without_values(tmp_path, monkeypatch, capsys):
    before, after = inventory(), inventory()
    secret = '/PRIVATE/raw/results/DO_NOT_PRINT'
    after['geometry']['inputFile'] = secret
    for index in range(20):
        label = f'Extra{index:02d}'
        for source, value in ((before, 'old'), (after, secret)):
            for row in source['paths']:
                row['sequence'].append(label)
                row['modules'][label] = {'module_type': 'Fit', 'private_setting': value}
    reference, candidate, output = (tmp_path / name for name in ('reference.json', 'candidate.json', 'report.json'))
    reference.write_text(json.dumps(before))
    candidate.write_text(json.dumps(after))
    monkeypatch.setattr(sys, 'argv', ['menu_diff.py', str(reference), str(candidate), '--output', str(output)])
    menu_diff.main()
    report = json.loads(output.read_text())
    console = capsys.readouterr().out
    assert report['summary']['modules_changed'] == 20
    assert secret in output.read_text()
    assert secret not in console and str(tmp_path) not in console
    assert console.count('module changed:') == 12
    assert '8 additional module change records' in console
    assert 'keys=private_setting' in console
    assert 'geometry_equal=false' in console
    output.unlink()
    candidate.write_text('{"paths": [], "paths": []}')
    with pytest.raises(SystemExit) as exc:
        menu_diff.main()
    assert exc.value.code == 1
    assert not output.exists()
    # Refuse to overwrite either source inventory.
    original = reference.read_text()
    monkeypatch.setattr(sys, 'argv', ['menu_diff.py', str(reference), str(candidate), '--output', str(reference)])
    with pytest.raises(SystemExit):
        menu_diff.main()
    assert reference.read_text() == original
