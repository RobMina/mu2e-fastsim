"""Fail-closed diagnostic of production art TimeTracker wall-time attribution.

Use: python3 trigger/timing.py LOG --inventory INVENTORY --output REPORT
The inventory must contain embedded scheduled paths, sequences and module maps,
not an unscoped universe of module definitions. Shared labels are counted once
per invocation using sum(Avg * nEvts), including disjoint first-execution paths.
Path attribution is NOT independent path latency. Rates describe this finished
job, not an extrapolated runtime, trigger efficiency or replay speedup.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

HEADER = re.compile(r'TimeTracker printout\s*\(sec\)\s*:?\s*(.*?)\s*$')
COLUMNS = ['Min', 'Avg', 'Max', 'Median', 'RMS', 'nEvts']
EVENTS = re.compile(r'TrigReport\s+Events\s+total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)\s*$')
TIME = re.compile(r'TimeReport\s+CPU\s*=\s*(\S+)\s+Real\s*=\s*(\S+)\s*$')
EXIT = re.compile(r'Art has completed and will exit with status\s+(\d+)\.?\s*$')
DIGITIZER_LABELS = {'EWMProducer', 'makeSD', 'CaloDigiMaker'}
DIGITIZER_TYPES = {'EventWindowMarkerProducer', 'StrawDigisFromStepPointMCs', 'CaloDigiMaker'}


def nonnegative(token):
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid timing number: {token}') from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'nonfinite or negative timing number: {token}')
    return value


def stats(tokens):
    if len(tokens) != 6 or not re.fullmatch(r'\d+', tokens[-1]):
        raise ValueError('timing row needs five statistics and an integer nEvts')
    values = [nonnegative(token) for token in tokens[:5]]
    return dict(zip(('min_s', 'avg_s', 'max_s', 'median_s', 'rms_s'), values)), int(tokens[-1])


def parse_log(text):
    """Read exactly one complete production timing table and job summary."""
    headers, event_summaries, time_summaries, exits = [], [], [], []
    lines = text.splitlines()
    for index, raw in enumerate(lines):
        line = raw.strip()
        if 'TimeTracker printout' in line:
            if not HEADER.search(line):
                raise ValueError('unsupported TimeTracker header')
            headers.append(index)
        for marker, pattern, found in (
            ('TrigReport Events', EVENTS, event_summaries),
            ('TimeReport CPU', TIME, time_summaries),
            ('Art has completed and will exit with status', EXIT, exits),
        ):
            if marker in ' '.join(line.split()):
                match = pattern.search(line)
                if not match:
                    raise ValueError(f'malformed {marker} summary')
                found.append(match.groups())
    if any(len(found) != 1 for found in (headers, event_summaries, time_summaries, exits)):
        raise ValueError('need exactly one TimeTracker, Events, CPU/Real and art exit summary; missing or ambiguous job')
    events, passed, failed = map(int, event_summaries[0])
    if events <= 0 or passed + failed != events:
        raise ValueError('Events summary must have positive total and passed+failed=total')
    if int(exits[0][0]) != 0:
        raise ValueError('art exit status is not zero')
    cpu, real = map(nonnegative, time_summaries[0])
    inline_columns = HEADER.search(lines[headers[0]].strip())[1].split()
    if inline_columns and inline_columns != COLUMNS:
        raise ValueError('unsupported TimeTracker columns')
    full, rows, seen, columns, ended = None, [], set(), bool(inline_columns), False
    for raw in lines[headers[0] + 1:]:
        line = raw.strip()
        if not line or set(line) <= {'-', ' '}:
            continue
        if set(line) <= {'='}:
            if full is None:
                continue
            ended = True
            break
        if not columns:
            if line.split() != COLUMNS:
                raise ValueError('missing or unsupported TimeTracker columns')
            columns = True
            continue
        if line.startswith('Full event'):
            if full is not None:
                raise ValueError('multiple Full event rows')
            full, count = stats(line[len('Full event'):].split())
            if count != events:
                raise ValueError('Full event nEvts differs from Events total')
            continue
        parts = line.split()
        identity = parts[0].split(':', 2)
        if parts[0] == 'source:RootInput(read)':
            identity = ['source', 'RootInput(read)', 'RootInput']
        if len(identity) != 3 or not all(identity) or full is None:
            raise ValueError(f'bad or truncated TimeTracker row: {line}')
        path, label, module_type = identity
        if module_type == 'RootOutput(write)':
            label += '(write)'
        values, count = stats(parts[1:])
        if (path, label) in seen:
            raise ValueError(f'duplicate timing row: {path}:{label}')
        seen.add((path, label))
        rows.append({'path': path, 'label': label, 'type': module_type,
                     'stats': values, 'invocations': count})
    if not ended or not columns or full is None or not rows:
        raise ValueError('missing or truncated TimeTracker table')
    counts, types = {}, {}
    for row in rows:
        label = row['label']
        counts[label] = counts.get(label, 0) + row['invocations']
        if counts[label] > events:
            raise ValueError(f'label invocation count exceeds input events: {label}')
        if label in types and types[label] != row['type']:
            raise ValueError(f'inconsistent timing module type: {label}')
        types[label] = row['type']
    return {'events': events, 'accepted_any_path': passed, 'rejected_all_paths': failed,
            'cpu_s': cpu, 'real_s': real, 'full_event': full, 'rows': rows}


def trigger_modules(inventory, rows):
    """Map only scheduled physics sequences; never infer membership from timing."""
    if not isinstance(inventory, dict) or not isinstance(inventory.get('paths'), list) or not inventory['paths']:
        raise ValueError('inventory needs embedded scheduled paths, not a full modules universe')
    digitizers = set(DIGITIZER_LABELS)
    digitizers.update(inventory.get('digitize_modules', []))
    digitizers.update(row['label'] for row in rows if row['path'] == 'DigitizePath')
    paths, definitions, seen_paths = [], {}, set()
    for path in inventory['paths']:
        if not isinstance(path, dict):
            raise ValueError('invalid inventory path')
        name, sequence, modules = path.get('path'), path.get('sequence'), path.get('modules')
        if not isinstance(name, str) or not name or name in seen_paths:
            raise ValueError('missing or duplicate inventory path name')
        seen_paths.add(name)
        if not isinstance(sequence, list) or not sequence or not all(isinstance(label, str) and label for label in sequence):
            raise ValueError('inventory needs literal nonempty path sequences')
        if not isinstance(modules, dict) or any(label not in modules for label in sequence):
            raise ValueError('inventory sequence has missing module definitions')
        for label in sequence:
            module = modules[label]
            module_type = module.get('module_type') if isinstance(module, dict) else None
            if not isinstance(module_type, str) or not module_type:
                raise ValueError('inventory needs plain-string module_type definitions')
            if label in definitions and definitions[label] != module_type:
                raise ValueError(f'inconsistent inventory module type: {label}')
            definitions[label] = module_type
            if module_type in DIGITIZER_TYPES:
                digitizers.add(label)
        if name == 'DigitizePath':
            digitizers.update(sequence)
        elif not name.startswith('Triggerable'):
            paths.append(path)
    expected = {label: definitions[label] for path in paths for label in path['sequence']
                if label not in digitizers and definitions[label] != 'SelectRecoMC'}
    if not expected:
        raise ValueError('inventory contains no physics trigger timing labels')
    return expected, digitizers


def finite_sum(values):
    try:
        value = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('timing total overflow') from exc
    if not math.isfinite(value):
        raise ValueError('nonfinite timing total')
    return value


def rate(numerator, denominator):
    if not denominator:
        return None
    try:
        return nonnegative(numerator / denominator)
    except OverflowError as exc:
        raise ValueError('timing rate overflow') from exc


def diagnose(text, inventory):
    parsed = parse_log(text)
    events, rows = parsed['events'], parsed['rows']
    expected, digitizers = trigger_modules(inventory, rows)
    by_module = {label: {'label': label, 'type': kind, 'invocations': 0, 'total_s': 0.0}
                 for label, kind in sorted(expected.items())}
    by_path, excluded, pieces = {}, [], {label: [] for label in expected}
    for row in rows:
        label, kind, count = row['label'], row['type'], row['invocations']
        total = finite_sum([row['stats']['avg_s'] * count])
        if label not in expected:
            reason = ('original_digitization' if label in digitizers else
                      'truth_decoration' if kind == 'SelectRecoMC' else 'outside_physics_trigger_inventory')
            excluded.append({'path': row['path'], 'label': label, 'type': kind,
                             'invocations': count, 'total_s': total, 'reason': reason})
            continue
        if kind != expected[label]:
            raise ValueError(f'timing/inventory module type mismatch: {label}')
        by_module[label]['invocations'] += count
        pieces[label].append(total)
        by_path.setdefault(row['path'], []).append(total)
    executed = [label for label, row in by_module.items() if row['invocations']]
    if not executed:
        raise ValueError('no executed trigger timing coverage; inventory/log have no timing labels intersection')
    for label, row in by_module.items():
        row['total_s'] = finite_sum(pieces[label])
    total = finite_sum(row['total_s'] for row in by_module.values())
    for row in by_module.values():
        row['amortized_s'] = row['total_s'] / events
        row['invocation_avg_s'] = row['total_s'] / row['invocations'] if row['invocations'] else None
        row['fraction_trigger_time'] = row['total_s'] / total if total else 0.0
    full = parsed['full_event']
    full['total_s'] = finite_sum([full['avg_s'] * events])
    return {
        'events': events, 'accepted_any_path': parsed['accepted_any_path'],
        'rejected_all_paths': parsed['rejected_all_paths'],
        'denominator': 'all input events processed by this finished job',
        'full_event_wall': full,
        'job': {'cpu_s': parsed['cpu_s'], 'real_s': parsed['real_s'],
                'cpu_events_per_s': rate(events, parsed['cpu_s']),
                'real_events_per_s': rate(events, parsed['real_s']),
                'full_event_wall_events_per_s': rate(1, full['avg_s'])},
        'trigger': {'total_s': total, 'ms_per_input_event': nonnegative(1000 * (total / events))},
        'by_module': by_module,
        'by_first_execution_path': {path: {'total_s': finite_sum(values),
                                          'amortized_s': finite_sum(values) / events}
                                    for path, values in sorted(by_path.items())},
        'path_interpretation': 'first-execution attribution only; NOT independent path latency',
        'coverage': {'expected_modules': len(expected), 'executed_modules': len(executed),
                     'absent_or_zero_execution_modules': sorted(set(expected) - set(executed)),
                     'note': 'absent modules can be conditional; absence does not prove timing-table completeness'},
        'excluded': {'row_count': len(excluded), 'invocations': sum(row['invocations'] for row in excluded),
                     'total_s': finite_sum(row['total_s'] for row in excluded), 'rows': excluded},
        'caveats': ['wall-time attribution, not trigger CPU time or isolated trigger latency',
                    'no efficiency, replay closure or throughput-improvement claim',
                    'zero CPU/Real or wall averages have undefined rates (null)'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top', type=int, default=0, help='print the N costliest shared module labels')
    parser.add_argument('--expected-events', type=int, help='require this finished event count')
    args = parser.parse_args()
    try:
        report = diagnose(args.log.read_text(), json.loads(args.inventory.read_text()))
        if args.expected_events is not None and report['events'] != args.expected_events:
            raise ValueError('finished timing event count differs from expected events')
        rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
    except (OSError, ValueError) as exc:
        parser.exit(1, f'timing diagnostic failed: {exc}\n')
    args.output.write_text(rendered)
    print(f"events={report['events']} trigger_ms/input={report['trigger']['ms_per_input_event']:.6g} "
          f"executed_modules={report['coverage']['executed_modules']}; first-execution attribution only")
    print(f"full_event_ms={1000 * report['full_event_wall']['avg_s']:.6g} "
          f"job_cpu_s={report['job']['cpu_s']:.6g} job_real_s={report['job']['real_s']:.6g}")
    for row in sorted(report['by_module'].values(), key=lambda r: r['total_s'], reverse=True)[:args.top]:
        if args.top > 0:
            print(f"{row['label']} {row['type']} ms/input={1000 * row['amortized_s']:.6g} "
                  f"fraction={row['fraction_trigger_time']:.4f} n={row['invocations']}")


if __name__ == '__main__':
    main()
