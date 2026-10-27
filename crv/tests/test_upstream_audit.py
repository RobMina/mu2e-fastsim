"""Retained-product audit isolates historical processes and does not simulate or overwrite."""
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from crvfastsim import response_config as RC
from crvfastsim.objective import upstream_audit as A


def config_text():
    cfg = RC.defaults()
    fields = ' '.join(f'{stock}: {json.dumps(cfg["pulse_reco"][fast])}'
                      for stock, fast in A.PULSE_FIELDS.items())
    return ('CrvPass1: { physics: { CrvRecoPulses: { ' + fields + ' } } '
            'DbService: { purpose: "CRV_COMMISSIONING" version: "v1_0" nearestMatch: false } } '
            'EvBCrv01: { CrvRecoPulses: { defaultBeta: 12.6 } '
            'DbService: { purpose: "OTHER" version: "v2" nearestMatch: true } }')


def test_exact_process_only():
    report = A.compare_config(config_text(), RC.defaults())
    assert report['all_compared_pulse_fields_equal']
    assert report['compared_pulse_field_count'] == 14
    assert report['retained_db']['nearestMatch'] is False
    assert not report['historical_conditions_or_executable_certified']


def test_mismatch_reported_not_fixed():
    report = A.compare_config(config_text().replace('defaultBeta: 19.0', 'defaultBeta: 12.6'), RC.defaults())
    assert not report['all_compared_pulse_fields_equal']
    assert report['pulse_fields']['defaultBeta']['fast'] == 19


@pytest.mark.parametrize('text', ['', 'CrvPass1: {', 'CrvPass1: {} CrvPass1: {}'])
def test_missing_duplicate_unclosed_process_rejected(text):
    with pytest.raises(ValueError):
        A.compare_config(text, RC.defaults())


def test_quoted_braces():
    assert A.named_block('A: { x: "}" B: { y: 2 } }', 'A') == ' x: "}" B: { y: 2 } '


def test_highest_tree_cycle(tmp_path):
    path = tmp_path / 'saved.root'
    with uproot.recreate(path) as f:
        f.mktree('EventNtuple/ntuple', {'event': np.arange(2)})
    with uproot.update(path) as f:
        f.mktree('EventNtuple/ntuple', {'event': np.arange(3)})
    result = A.root_inventory(str(path))
    row = result['trees']['EventNtuple/ntuple']
    assert row['entries'] == 3
    assert ';2' in row['selected_key']
    assert not result['content_checksum_verified']


def test_audit_roundtrip_and_exclusive_output(tmp_path):
    cfg = tmp_path / 'config.txt'
    cfg.write_text(config_text())
    frozen = tmp_path / 'r3.json'
    effective = RC.defaults()
    effective['timing']['phase_scope'] = 'per_event'
    frozen.write_text(json.dumps({'resolved_config': {'values': effective}}))
    capture = tmp_path / 'capture.npz'
    np.savez(capture, hit_time=np.arange(3))
    out = tmp_path / 'audit.json'
    result = A.run(cfg, out, captures=[capture], frozen=frozen)
    assert result['completed'] and not result['new_capture']
    assert result['captures'][0]['columns'] == ['hit_time']
    assert result['frozen']['fast_timing']['phase_scope'] == 'per_event'
    with pytest.raises(FileExistsError):
        A.run(cfg, out)


def test_first_pulse_all_body_summary_and_nonfinite():
    bars = [0, 256, 320, 384, 512, 575]
    p = {name: ak.Array([[value] for value in bars]) for name in ('barId',)}
    p['sequenceIndex'] = ak.Array([[0], [0], [0], [0], [0], [1]])
    for field in ('PEs', 'PEsPulseHeight', 'pulseHeight', 'pulseBeta', 'pulseFitChi2'):
        p[field] = ak.Array([[19.], [np.nan], [4.], [40.], [12.], [20.]])
    result = A.summarize_pulses(p)
    assert len(result) == 16
    assert result['T10']['fields']['pulseBeta']['nonfinite'] == 1
    assert result['T11']['beta_near_bound'] == 1
    assert result['EX0']['beta_equal_19'] == 1
    assert result['M8']['first_pulses'] == 0


def test_payload_read_without_capture(tmp_path):
    path = tmp_path / 'input.root'
    arrays = {f'crvpulses.{name}': ak.Array([[value], [], [value]]) for name, value in
              [('barId', 256), ('sequenceIndex', 0), ('PEs', 20.), ('PEsPulseHeight', 21.),
               ('pulseHeight', 70.), ('pulseBeta', 19.), ('pulseFitChi2', 2.)]}
    arrays['crvdigis.adc'] = ak.Array([[[100, 200]], [], [[99, 4095]]])
    with uproot.recreate(path) as f:
        f['EventNtuple/ntuple'] = arrays
    result = A.payload_inventory(path, 2)
    assert result['entry_stop'] == 2
    assert result['adc']['samples'] == 2 and result['adc']['max'] == 200
    assert result['pulse_summaries']['T10']['first_pulses'] == 1
    assert not result['fallback_fraction_measured']


def test_frozen_missing_config_rejected(tmp_path):
    cfg, frozen = tmp_path / 'config.txt', tmp_path / 'bad.json'
    cfg.write_text(config_text())
    frozen.write_text('{}')
    with pytest.raises(ValueError, match='resolved response'):
        A.run(cfg, tmp_path / 'out.json', frozen=frozen)
