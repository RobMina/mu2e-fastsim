"""Paired fast-vs-reference trigger acceptance from synthetic replay logs."""
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f'trigger_hybrid_{name}', ROOT / 'trigger' / 'hybrid' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pair = load('pair')
plot_paired = load('plot_paired')

MANIFEST = {'original_process': 'Digitize', 'replay_process': 'TriggerReplay',
            'paths': ['tpr_Trk', 'cpr_Trk', 'off_path'], 'bits': ['100', '150', '600'], 'stateful_paths': []}
INVENTORY = {'paths': [
    {'path': 'tpr_Trk', 'prescales': [{'label': 'P1', 'eventModeConfig': [{'eventMode': 'OnSpill', 'prescale': '1'},
                                                                        {'eventMode': 'OffSpill', 'prescale': '1'}]}]},
    {'path': 'cpr_Trk', 'prescales': []},
    {'path': 'off_path', 'prescales': [{'label': 'P2', 'eventModeConfig': [{'eventMode': 'OnSpill', 'prescale': '1'}]}]}]}


def record(event, tpr, cpr='fail', off='fail'):
    block = ''.join(f'  {tpr if p == "tpr_Trk" else cpr if p == "cpr_Trk" else off} {b} {p}\n'
                    for p, b in zip(MANIFEST['paths'], MANIFEST['bits']))
    return (f'###############  PrintModule Run/Subrun/Event 1 2 {event}\n'
            f'ProductPrint art::TriggerResults_TriggerResults__Digitize\n{block}'
            f'ProductPrint art::TriggerResults_TriggerResults__TriggerReplay\n{block}\n')


TRIGREPORT = '''TrigReport ---------- Modules in path: tpr_Trk ------------
TrigReport    Path ID    Visited     Passed     Failed      Error Name
TrigReport        100          {n}          {h}          0          0 HelixFilter
TrigReport        100          {h}          {t}          0          0 TrackFilter

Art has completed and will exit with status 0.
'''


def write_log(path, decisions, h, t):
    text = ''.join(record(event, tpr) for event, tpr in decisions)
    path.write_text(text + TRIGREPORT.format(n=len(decisions), h=h, t=t))
    return path


@pytest.fixture
def arms(tmp_path):
    ref = [(e, 'pass') for e in range(1, 9)] + [(9, 'fail')]
    cand = [(e, 'pass') for e in range(1, 6)] + [(6, 'fail'), (7, 'fail'), (9, 'pass')]  # event 8 missing
    return (write_log(tmp_path / 'ref.log', ref, 9, 8), write_log(tmp_path / 'cand.log', cand, 7, 6))


def test_paired_difference_error_and_status(arms):
    ref = ('g4', pair.read_arm([arms[0]], MANIFEST))
    cand = ('fast', pair.read_arm([arms[1]], MANIFEST))
    report = pair.pair(ref, cand, MANIFEST, INVENTORY, 'OffSpill', total_events=10)
    row = report['paths']['tpr_Trk']
    assert (row['g4_pass'], row['fast_pass'], row['fast_only'], row['g4_only']) == (8, 6, 1, 3)
    assert row['paired_difference'] == pytest.approx(-0.2)
    assert row['paired_error'] == pytest.approx(math.sqrt((4 - 4 / 10) / (10 * 9)))
    assert row['status'] == 'tested' and abs(row['pull']) < 2 and 'stages' not in row
    assert report['paths']['off_path']['status'] == 'not tested: disabled in OffSpill'
    assert report['paths']['cpr_Trk']['status'] == 'not tested: no event accepted in either arm'
    assert report['events_in_logs'] == {'g4': 9, 'fast': 8, 'union': 9}
    common = pair.pair(ref, cand, MANIFEST, INVENTORY, 'OffSpill', common=True)
    assert common['events'] == 8 and common['paths']['tpr_Trk']['g4_only'] == 2
    with pytest.raises(ValueError, match='common'):
        pair.pair(ref, cand, MANIFEST, INVENTORY, total_events=10, common=True)


def test_stage_counts_attached_beyond_two_sigma(arms, tmp_path):
    many_ref = [(e, 'pass') for e in range(1, 201)]
    many_cand = [(e, 'pass' if e > 40 else 'fail') for e in range(1, 201)]
    ref = ('g4', pair.read_arm([write_log(tmp_path / 'r.log', many_ref, 200, 200)], MANIFEST))
    cand = ('fast', pair.read_arm([write_log(tmp_path / 'c.log', many_cand, 170, 160)], MANIFEST))
    row = pair.pair(ref, cand, MANIFEST, INVENTORY)['paths']['tpr_Trk']
    assert row['pull'] is None or row['pull'] < -2
    assert row['stages'][0] == {'module': 'HelixFilter', 'g4_passed': 200, 'fast_passed': 170, 'difference': -30}


def test_duplicate_events_and_small_denominator_rejected(arms):
    with pytest.raises(ValueError, match='two logs'):
        pair.read_arm([arms[0], arms[0]], MANIFEST)
    ref = ('g4', pair.read_arm([arms[0]], MANIFEST))
    with pytest.raises(ValueError, match='below'):
        pair.pair(ref, ref, MANIFEST, INVENTORY, total_events=3)


def test_cli_and_plot(arms, tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'manifest.json').write_text(json.dumps(MANIFEST))
    (prepared / 'replay_inventory.json').write_text(json.dumps(INVENTORY))
    out = tmp_path / 'paired.json'
    monkeypatch.setattr(sys, 'argv', ['pair.py', '--reference', f'g4={arms[0]}', '--candidate', f'fast={arms[1]}',
                                      '--prepared', str(prepared), '--event-mode', 'OffSpill', '--output', str(out)])
    pair.main()
    report = json.loads(out.read_text())
    assert report['events'] == 9 and report['paths']['tpr_Trk']['fast_only'] == 1
    png = tmp_path / 'paired.png'
    plot_paired.plot(report, png)
    assert png.stat().st_size > 0
