"""Read-only retained-product/configuration inventory; no response replay or calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

from .. import response_config as RC
from . import validation as V

PULSE_FIELDS = {
    'defaultBeta': 'default_beta', 'minBeta': 'min_beta', 'maxBeta': 'max_beta',
    'minADCdifference': 'min_adc_difference', 'maxTimeDifference': 'max_time_difference',
    'minPulseHeightRatio': 'min_pulse_height_ratio', 'maxPulseHeightRatio': 'max_pulse_height_ratio',
    'LEtimeFactor': 'le_time_factor', 'samplesBefore': 'samples_before', 'samplesAfter': 'samples_after',
    'pedestalUndershootThreshold': 'pedestal_undershoot_threshold', 'useTimeOffsetDB': 'use_time_offset_db',
    'timeOffsetScale': 'time_offset_scale', 'ignoreChannels': 'ignore_channels',
}


def named_block(text, name):
    """Extract a unique balanced block without matching braces inside quoted strings."""
    token = re.compile(r'"(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z_0-9]*|[{}:]|[^\s{}:]+')
    tokens = list(token.finditer(text))
    found = []
    for i in range(len(tokens) - 2):
        if [t.group() for t in tokens[i:i + 3]] != [name, ':', '{']:
            continue
        depth = 1
        for end in tokens[i + 3:]:
            depth += (end.group() == '{') - (end.group() == '}')
            if depth == 0:
                found.append(text[tokens[i + 2].end():end.start()])
                break
        else:
            raise ValueError(f'unclosed {name} block')
    if len(found) != 1:
        raise ValueError(f'ambiguous or missing {name} block')
    return found[0]


def scalar(block, name):
    values = re.findall(r'\b' + re.escape(name) + r':\s*("[^"\n]*"|[^\s{},]+)', block)
    if len(values) != 1:
        raise ValueError(f'ambiguous or missing {name}')
    value = values[0]
    if value in ('true', 'false'):
        return value == 'true'
    if value.startswith('"'):
        return json.loads(value)
    return float(value)


def compare_config(text, config):
    """Compare the retained reconstruction process only, not neighboring process histories."""
    process = named_block(text, 'CrvPass1')
    pulse = named_block(process, 'CrvRecoPulses')
    rows = {stock: {'retained': scalar(pulse, stock), 'fast': config['pulse_reco'][fast]}
            for stock, fast in PULSE_FIELDS.items()}
    for row in rows.values():
        row['equal'] = row['retained'] == row['fast']
    db = named_block(process, 'DbService')
    return {'pulse_fields': rows, 'all_compared_pulse_fields_equal': all(x['equal'] for x in rows.values()),
            'retained_db': {k: scalar(db, k) for k in ('purpose', 'version', 'nearestMatch')},
            'fast_timing': config['timing'], 'fast_waveform': config['waveform'],
            'fast_assets': config['assets'], 'compared_pulse_field_count': len(rows),
            'historical_conditions_or_executable_certified': False}


def root_inventory(path):
    """Open an exact known path; select the highest tree cycle without namespace listing/stat."""
    trees = {}
    with open(path, 'rb') as stream, uproot.open(stream) as f:
        for key, cls in f.classnames(recursive=True, cycle=False).items():
            if cls not in ('TTree', 'ROOT::RNTuple'):
                continue
            tree = f[key]
            trees[key] = {'class': cls, 'entries': tree.num_entries,
                          'selected_key': tree.object_path, 'branches': list(tree.keys())}
    return {'path': str(path), 'trees': trees, 'content_checksum_verified': False,
            'scope': 'Retained ROOT metadata only; branch presence is not populated-content or lineage certification.'}


def npz_inventory(path):
    with np.load(path, allow_pickle=False) as f:
        keys = sorted(f.files)
    return {'path': str(path), 'sha256': V.digest(path), 'columns': keys,
            'scope': 'Retained capture column inventory; no rows are reselected or regenerated.'}


def summarize_pulses(p):
    """All first pulses, not the selected common-support timing population."""
    from . import bars
    flat = {name: ak.to_numpy(ak.flatten(value, axis=None)) for name, value in p.items()}
    if len({len(value) for value in flat.values()}) != 1:
        raise ValueError('pulse fields have inconsistent lengths')
    bar = flat['barId']
    if np.any((bar < 0) | (bar >= 576)):
        raise ValueError('pulse bar outside extracted geometry')
    _, bodies, _ = bars.bar_slot_body_layer(bar)
    first = flat['sequenceIndex'] == 0
    result = {}
    for body, name in enumerate(bars.BODY_NAMES):
        selected = (bodies == body) & first
        row = {'first_pulses': int(selected.sum()), 'fields': {}}
        for field in ('PEs', 'PEsPulseHeight', 'pulseHeight', 'pulseBeta', 'pulseFitChi2'):
            values = flat[field][selected]
            finite = values[np.isfinite(values)]
            row['fields'][field] = {'finite': len(finite), 'nonfinite': len(values) - len(finite),
                                    'nonzero': int(np.count_nonzero(finite)),
                                    'q16_q50_q84': np.quantile(finite, [.16, .5, .84]).tolist() if len(finite) else None}
        beta = flat['pulseBeta'][selected]
        row['beta_equal_19'] = int(np.count_nonzero(beta == 19.))
        row['beta_near_bound'] = int(np.count_nonzero((beta <= 5.35) | (beta >= 39.65)))
        result[name] = row
    return result


def payload_inventory(path, entries):
    """Bounded read of existing fit fields/ADC payload; no fit, pairing or row sidecar."""
    fields = ('barId', 'sequenceIndex', 'PEs', 'PEsPulseHeight', 'pulseHeight', 'pulseBeta', 'pulseFitChi2')
    with open(path, 'rb') as stream, uproot.open(stream) as f:
        tree = f['EventNtuple/ntuple']
        stop = min(entries, tree.num_entries)
        names = ['crvpulses.' + name for name in fields] + ['crvdigis.adc']
        if tree.classname == 'ROOT::RNTuple':
            arrays = tree.arrays(filter_name=lambda name: name in names, entry_stop=stop, library='ak')
        else:
            arrays = tree.arrays(names, entry_stop=stop, library='ak')
        p = {name: arrays['crvpulses.' + name] for name in fields}
        adc = arrays['crvdigis.adc']
        values = ak.to_numpy(ak.flatten(adc, axis=None))
        return {'path': str(path), 'selected_key': tree.object_path, 'entry_start': 0, 'entry_stop': stop,
                'available_entries': tree.num_entries, 'pulse_summaries': summarize_pulses(p),
                'adc': {'samples': len(values), 'nonzero': int(np.count_nonzero(values)),
                        'min': float(values.min()) if len(values) else None,
                        'max': float(values.max()) if len(values) else None},
                'scope': 'Data-only all first pulses; not common-support, independent holdout or causal timing attribution.',
                'fallback_fraction_measured': False, 'content_checksum_verified': False}


def run(config_path, out, roots=(), captures=(), frozen=None, sources=(), payload_entries=0, joint_points=()):
    if Path(out).exists():
        raise FileExistsError(out)
    if not isinstance(payload_entries, int) or payload_entries < 0:
        raise ValueError('payload entries must be a nonnegative integer')
    receipts = {}
    def record(path):
        receipts[str(path)] = V.digest(path)
    for source in (config_path, __file__, V.__file__, RC.__file__, RC.DEFAULT_PATH, *sources):
        record(source)
    text = Path(config_path).read_text()
    preset = RC.defaults()
    report = {'schema': 1, 'comparison': 'upstream-retained-audit', 'completed': False,
              'current_preset': compare_config(text, preset), 'roots': [], 'captures': [],
              'new_capture': False, 'response_or_conditions_changed': False}
    if frozen:
        record(frozen)
        saved = json.loads(Path(frozen).read_text())
        config = saved.get('resolved_config', saved.get('response_config'))
        if config is None:
            raise ValueError('frozen receipt lacks a resolved response configuration')
        config = config.get('values', config)
        report['frozen'] = compare_config(text, config)
        report['frozen_config_sha256'] = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    report['joint_config_checks'] = []
    if joint_points:
        from . import joint_compare as J
        record(J.__file__)
        for directory in joint_points:
            root = Path(directory)
            arm = root.name
            if arm not in J.ARMS:
                raise ValueError('joint point directory must name one of the four fixed arms')
            point = {}
            for name in ('provenance', 'r3'):
                path = root / (name + '.json')
                record(path)
                point[name] = json.loads(path.read_text())
            J._config(point, arm)
            report['joint_config_checks'].append({'path': str(root), 'arm': arm, 'passed': True,
                'legacy_attachment_field_absent': 'add_opposite_side_pulses' not in
                    point['provenance']['response_config']['values']['finder'],
                'scope': 'Configuration/hash/exposure contract only, not the full capture replay/identity gate.'})
    for path in roots:
        report['roots'].append(root_inventory(path))
        if payload_entries:
            report['roots'][-1]['payload'] = payload_inventory(path, payload_entries)
    for path in captures:
        record(path)
        report['captures'].append(npz_inventory(path))
    for path, sha in receipts.items():
        if V.digest(path) != sha:
            raise ValueError(f'audit input/source changed: {path}')
    report['files'] = [{'path': p, 'sha256': s} for p, s in receipts.items()]
    report['completed'] = True
    with Path(out).open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--root', action='append', default=[])
    p.add_argument('--capture', action='append', default=[])
    p.add_argument('--source', action='append', default=[])
    p.add_argument('--frozen')
    p.add_argument('--payload-entries', type=int, default=0)
    p.add_argument('--joint-point', action='append', default=[])
    a = p.parse_args(argv)
    report = run(a.config, a.out, a.root, a.capture, a.frozen, a.source, a.payload_entries, a.joint_point)
    print(json.dumps({'completed': report['completed'], 'roots': len(report['roots']),
                      'pulse_fields_equal': report['current_preset']['all_compared_pulse_fields_equal']}))


if __name__ == '__main__':
    main()
