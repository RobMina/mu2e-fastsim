"""Synthetic production TimeTracker parsing and attribution contracts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trigger_timing', ROOT / 'trigger' / 'timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def inventory():
    modules = {
        'EWMProducer': {'module_type': 'EventWindowMarkerProducer'},
        'makeSD': {'module_type': 'StrawDigisFromStepPointMCs'},
        'CaloDigiMaker': {'module_type': 'CaloDigiMaker'},
        'TruthMatch': {'module_type': 'SelectRecoMC'},
        'Shared': {'module_type': 'Fit'},
        'Cut': {'module_type': 'Filter'},
        'Conditional': {'module_type': 'Fit'},
    }
    return {'paths': [
        {'path': 'A', 'sequence': list(modules), 'modules': deepcopy(modules)},
        {'path': 'B', 'sequence': ['Shared', 'Conditional'],
         'modules': {label: deepcopy(modules[label]) for label in ('Shared', 'Conditional')}},
    ]}


def log(rows=None, events=10):
    if rows is None:
        rows = '''A:Shared:Fit 0.01 0.02 0.04 0.02 0.001 8
B:Shared:Fit 0.02 0.04 0.06 0.04 0.001 2
A:Cut:Filter 0.001 0.002 0.003 0.002 0.0001 10
DigitizePath:makeSD:StrawDigisFromStepPointMCs 0.01 0.02 0.04 0.02 0.001 10
A:EWMProducer:EventWindowMarkerProducer 0.001 0.002 0.003 0.002 0.0001 10
A:TruthMatch:SelectRecoMC 0.001 0.002 0.003 0.002 0.0001 10
RecoPath:Reco:Reconstructor 0.01 0.02 0.04 0.02 0.001 10'''
    return f'''ordinary startup chatter
TimeTracker printout (sec) Min Avg Max Median RMS nEvts

------------------------------
Full event 0.00381553 0.0298794 2.17031 0.0312165 0.0217807 {events}
{rows}
==============================
TrigReport Events total = {events} passed = {events} failed = 0
TimeReport CPU = 0.5 Real = 1
Art has completed and will exit with status 0.
'''


def test_shared_module_split_is_weighted_not_charged_to_each_path():
    source = inventory()
    original = deepcopy(source)
    report = timing.diagnose(log(), source)
    assert source == original
    shared = report['by_module']['Shared']
    assert shared['type'] == 'Fit'
    assert shared['invocations'] == 10
    assert shared['total_s'] == pytest.approx(0.24)
    assert shared['amortized_s'] == pytest.approx(0.024)
    assert shared['invocation_avg_s'] == pytest.approx(0.024)
    assert shared['fraction_trigger_time'] == pytest.approx(0.24 / 0.26)
    assert report['trigger']['total_s'] == pytest.approx(0.26)
    assert report['trigger']['ms_per_input_event'] == pytest.approx(26)
    assert report['by_first_execution_path']['A']['total_s'] == pytest.approx(0.18)
    assert report['by_first_execution_path']['B']['total_s'] == pytest.approx(0.08)
    assert 'NOT independent path latency' in report['path_interpretation']
    assert report['coverage']['executed_modules'] == 2
    assert report['coverage']['expected_modules'] == 3
    assert report['coverage']['absent_or_zero_execution_modules'] == ['Conditional']
    assert report['excluded']['row_count'] == 4
    assert report['excluded']['invocations'] == 40
    assert {row['reason'] for row in report['excluded']['rows']} == {
        'original_digitization', 'truth_decoration', 'outside_physics_trigger_inventory'}
    assert report['job']['cpu_events_per_s'] == 20
    assert report['job']['real_events_per_s'] == 10
    assert report['full_event_wall']['avg_s'] == 0.0298794
    assert report['job']['full_event_wall_events_per_s'] == pytest.approx(1 / 0.0298794)
    json.dumps(report, allow_nan=False)


def test_zero_execution_and_absent_conditional_modules_are_allowed():
    text = log().replace('A:Cut:Filter 0.001 0.002 0.003 0.002 0.0001 10',
                         'A:Cut:Filter 0 0 0 0 0 0')
    report = timing.diagnose(text, inventory())
    assert report['by_module']['Cut']['total_s'] == 0
    assert report['by_module']['Cut']['invocation_avg_s'] is None
    assert report['coverage']['executed_modules'] == 1
    assert report['coverage']['absent_or_zero_execution_modules'] == ['Conditional', 'Cut']
    text = log('A:Shared:Fit 0 0 0 0 0 10').replace('CPU = 0.5 Real = 1', 'CPU = 0 Real = 0')
    report = timing.diagnose(text, inventory())
    assert report['trigger']['total_s'] == 0
    assert report['by_module']['Shared']['fraction_trigger_time'] == 0
    assert report['job']['cpu_events_per_s'] is None


def test_realistic_statistics_scientific_notation_and_separate_header():
    source = {'paths': [{'path': 'Physics', 'sequence': ['Intensity', 'Helix'], 'modules': {
        'Intensity': {'module_type': 'IntensityFlat'}, 'Helix': {'module_type': 'LoopHelixFit'}}}]}
    text = log('''Physics:Intensity:IntensityFlat 5.791e-06 7.62108e-06 0.000198686 7.544e-06 2.26958e-06 28148
Physics:Helix:LoopHelixFit 0.000193155 0.00183157 0.00438756 0.00177719 0.000570088 18038''', events=28148)
    text = text.replace('(sec) Min', '(sec)\n\n Min')
    report = timing.diagnose(text, source)
    assert report['by_module']['Helix']['total_s'] == pytest.approx(0.00183157 * 18038)
    assert report['by_module']['Helix']['amortized_s'] == pytest.approx(0.00183157 * 18038 / 28148)


def test_digitize_path_labels_are_excluded_even_with_unusual_names():
    source = inventory()
    source['paths'][0]['sequence'].append('DigiAlias')
    source['paths'][0]['modules']['DigiAlias'] = {'module_type': 'CustomDigi'}
    text = log().replace('A:Shared:Fit', 'DigitizePath:DigiAlias:CustomDigi 0 0.1 0.2 0.1 0 10\nA:Shared:Fit', 1)
    assert 'DigiAlias' not in timing.diagnose(text, source)['by_module']
    source['paths'].append({'path': 'DigitizePath', 'sequence': ['DigiAlias'],
                            'modules': {'DigiAlias': {'module_type': 'CustomDigi'}}})
    report = timing.diagnose(log(), source)
    assert 'DigiAlias' not in report['by_module']


@pytest.mark.parametrize('source', [
    {}, {'modules': {'Shared': {'module_type': 'Fit'}}},
    {'paths': [{'path': 'A', 'modules': {'Shared': {'module_type': 'Fit'}}}]},
    {'paths': [{'path': 'A', 'sequence': ['Missing'], 'modules': {}}]},
])
def test_unscoped_universe_or_incomplete_inventory_rejected(source):
    with pytest.raises(ValueError, match='inventory'):
        timing.diagnose(log(), source)


def test_reco_only_log_and_unexecuted_trigger_only_log_rejected():
    for rows in ('RecoPath:Reco:Reconstructor 0 0.1 0.2 0.1 0 10',
                 'A:Shared:Fit 0 0 0 0 0 0'):
        with pytest.raises(ValueError, match='coverage'):
            timing.diagnose(log(rows), inventory())


@pytest.mark.parametrize('old,new', [
    ('TimeReport CPU = 0.5 Real = 1\n', ''),
    ('TrigReport Events total = 10 passed = 10 failed = 0\n', ''),
    ('Art has completed and will exit with status 0.', ''),
    ('status 0.', 'status 1.'),
    ('passed = 10 failed = 0', 'passed = 9 failed = 0'),
    ('Full event 0.00381553 0.0298794 2.17031 0.0312165 0.0217807 10',
     'Full event 0.00381553 0.0298794 2.17031 0.0312165 0.0217807 9'),
    ('==============================', ''),
    ('0.01 0.02 0.04 0.02 0.001 8', '0.01 0.02 0.04 0.02 8'),
    ('0.01 0.02 0.04 0.02 0.001 8', '0.01 0.02 0.04 0.02 0.001 8.0'),
    ('Min Avg Max Median RMS nEvts', 'Min Avg Max RMS nEvts'),
    ('CPU = 0.5', 'CPU = 1e-320'),
    ('Real = 1', 'Real = 1e-320'),
])
def test_incomplete_malformed_nonzero_or_truncated_log_rejected(old, new):
    with pytest.raises(ValueError):
        timing.diagnose(log().replace(old, new), inventory())


@pytest.mark.parametrize('suffix', [
    log(), '\nTrigReport Events total = 10 passed = 10 failed = 0',
    '\nTimeReport CPU = 1 Real = 2', '\nArt has completed and will exit with status 0.',
])
def test_multiple_job_or_event_summaries_are_ambiguous(suffix):
    with pytest.raises(ValueError, match='ambiguous'):
        timing.diagnose(log() + suffix, inventory())


@pytest.mark.parametrize('bad', ['nan', 'inf', '-inf', '-0.1', '1e999'])
def test_nonfinite_negative_module_full_event_or_job_statistics_rejected(bad):
    for old in ('0.001 8', '0.0298794', 'CPU = 0.5'):
        replacement = bad + ' 8' if old == '0.001 8' else 'CPU = ' + bad if old == 'CPU = 0.5' else bad
        with pytest.raises(ValueError):
            timing.diagnose(log().replace(old, replacement), inventory())


@pytest.mark.parametrize('rows', [
    'A:Shared:Fit 0 0.01 0.02 0.01 0 11',
    'A:Shared:Fit 0 0.01 0.02 0.01 0 8\nB:Shared:Fit 0 0.01 0.02 0.01 0 3',
    'A:Shared:Fit 0 0.01 0.02 0.01 0 2\nA:Shared:Fit 0 0.01 0.02 0.01 0 2',
    'A:Shared:WrongType 0 0.01 0.02 0.01 0 10',
    'A:Shared:Fit 0 0.01 0.02 0.01 0 8\nB:Shared:Other 0 0.01 0.02 0.01 0 2',
    'RecoPath:Reco:Reconstructor 0 0.01 0.02 0.01 0 11',
    'A:Shared:Fit 0 1e308 1e308 1e308 0 10',
])
def test_overcount_duplicate_type_mismatch_or_overflow_rejected(rows):
    with pytest.raises(ValueError):
        timing.diagnose(log(rows), inventory())


def test_cli_writes_json_and_does_not_write_report_on_invalid_input(tmp_path, monkeypatch):
    input_log, input_inventory, output = (tmp_path / name for name in ('job.log', 'inventory.json', 'report.json'))
    input_log.write_text(log())
    input_inventory.write_text(json.dumps(inventory()))
    monkeypatch.setattr(sys, 'argv', ['timing.py', str(input_log), '--inventory', str(input_inventory),
                                    '--output', str(output)])
    timing.main()
    assert json.loads(output.read_text())['trigger']['ms_per_input_event'] == pytest.approx(26)
    output.unlink()
    input_log.write_text(log().replace('status 0.', 'status 2.'))
    with pytest.raises(SystemExit) as exc:
        timing.main()
    assert exc.value.code == 1
    assert not output.exists()


def test_stock_table_separator_source_and_output_write_subtimers():
    text = log().replace('(sec) Min Avg Max Median RMS nEvts\n',
        '(sec) Min Avg Max Median RMS nEvts\n==================================\n')
    text = text.replace('A:Shared:Fit', 'source:RootInput(read) 0 0.001 0.002 0.001 0 10\nA:Shared:Fit', 1)
    text = text.replace('RecoPath:Reco:Reconstructor',
        'end_path:Output:RootOutput 0 0.001 0.002 0.001 0 10\nend_path:Output:RootOutput(write) 0 0.001 0.002 0.001 0 10\nRecoPath:Reco:Reconstructor')
    report = timing.diagnose(text, inventory())
    assert report['trigger']['ms_per_input_event'] == pytest.approx(26)
    assert report['excluded']['row_count'] == 7


def test_digitizers_excluded_from_explicit_inventory_even_when_attributed_to_physics():
    source = inventory()
    source['digitize_modules'] = ['Shared']
    report = timing.diagnose(log(), source)
    assert 'Shared' not in report['by_module']
    assert report['trigger']['ms_per_input_event'] == pytest.approx(2)


def test_cli_expected_finished_event_count(tmp_path, monkeypatch):
    source, prepared, output = (tmp_path / name for name in ('art.log', 'inventory.json', 'timing.json'))
    source.write_text(log())
    prepared.write_text(json.dumps(inventory()))
    monkeypatch.setattr(sys, 'argv', ['timing.py', str(source), '--inventory', str(prepared),
        '--output', str(output), '--expected-events', '11'])
    with pytest.raises(SystemExit) as exc:
        timing.main()
    assert exc.value.code == 1 and not output.exists()


def test_rejected_trigger_events_are_not_art_failures_and_stay_in_denominator():
    report = timing.diagnose(log().replace('passed = 10 failed = 0', 'passed = 9 failed = 1'), inventory())
    assert report['events'] == 10
    assert report['accepted_any_path'] == 9 and report['rejected_all_paths'] == 1
    assert report['trigger']['ms_per_input_event'] == pytest.approx(26)
