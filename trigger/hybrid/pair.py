"""Paired per-path trigger acceptance of two replays of the same events (e.g. fast vs Geant4 digis).

Use: python3 trigger/hybrid/pair.py --reference g4=a.log,b.log --candidate fast=c.log,d.log \
         --prepared PREPARED_DIR [--event-mode OffSpill] [--total-events N] --output paired.json
Each log is a replay.sbatch decision log (PrintModule records plus the art TrigReport). Events present in only one
arm count as rejected by every path in the other arm; --total-events adds events absent from both arms.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare  # noqa: E402

MODULES = re.compile(r'^TrigReport -+ Modules in path: ([A-Za-z_][A-Za-z_0-9]*) -+$')
MODULE_ROW = re.compile(r'^TrigReport\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([A-Za-z_][A-Za-z_0-9]*)\s*$')


def parse_arm(spec):
    label, _, paths = spec.partition('=')
    if not label or not paths:
        raise ValueError(f'expected LABEL=LOG[,LOG...], got {spec!r}')
    return label, [Path(p) for p in paths.split(',')]


def module_counts(text):
    """Per path, per module visited/passed from the art TrigReport."""
    result, path = {}, None
    for line in text.splitlines():
        if match := MODULES.match(line):
            path = match[1]
            result[path] = {}
        elif path and (match := MODULE_ROW.match(line)):
            visited, passed, failed, error, label = match.groups()
            result[path][label] = {'visited': int(visited), 'passed': int(passed), 'error': int(error)}
        elif path and not line.startswith('TrigReport'):
            path = None
    return result


def read_arm(logs, manifest):
    """Replay decisions {event: {path: accepted}} and summed module counts over all logs of one arm."""
    processes = (manifest['original_process'], manifest['replay_process'])
    decisions, modules = {}, {}
    for log in logs:
        text = log.read_text()
        if compare.COMPLETION.findall(text) != ['0']:
            raise ValueError(f'{log}: need exactly one successful art completion marker')
        for event, products in compare.read_decisions(text, processes).items():
            if event in decisions:
                raise ValueError(f'{log}: event {event} appears in two logs')
            decisions[event] = {path: value[1] for path, value in products[processes[1]].items()}
        for path, rows in module_counts(text).items():
            for label, row in rows.items():
                total = modules.setdefault(path, {}).setdefault(label, {'visited': 0, 'passed': 0, 'error': 0})
                for key in total:
                    total[key] += row[key]
    return decisions, modules


def mode_prescale(row, event_mode):
    """Product of the path's PrescaleEvent factors for this event mode; <= 0 means disabled, None no prescaler."""
    factors = []
    for prescale in row['prescales']:
        modes = {entry.get('eventMode'): entry.get('prescale') for entry in prescale['eventModeConfig']}
        factors.append(int(modes.get(event_mode, 0)))
    return math.prod(factors) if factors else None


def pair(reference, candidate, manifest, inventory, event_mode=None, total_events=None, common=False):
    (ref_label, ref), (cand_label, cand) = reference, candidate
    events = sorted(set(ref[0]) & set(cand[0]) if common else set(ref[0]) | set(cand[0]))
    if common and total_events:
        raise ValueError('--total-events does not apply to the events common to both arms')
    n = total_events or len(events)
    if n < len(events):
        raise ValueError(f'total events {n} is below the {len(events)} events seen in the logs')
    rows = {row['path']: row for row in inventory['paths']}
    stateful = set(manifest.get('stateful_paths', []))
    paths = {}
    for name, bit in zip(manifest['paths'], manifest['bits']):
        r = sum(bool(ref[0].get(e, {}).get(name)) for e in events)
        c = sum(bool(cand[0].get(e, {}).get(name)) for e in events)
        c_only = sum(bool(cand[0].get(e, {}).get(name)) and not ref[0].get(e, {}).get(name) for e in events)
        r_only = sum(bool(ref[0].get(e, {}).get(name)) and not cand[0].get(e, {}).get(name) for e in events)
        diff = (c_only - r_only) / n
        var = ((c_only + r_only) - (c_only - r_only) ** 2 / n) / (n * (n - 1)) if n > 1 else 0.0
        error = math.sqrt(max(var, 0.0))
        prescale = mode_prescale(rows[name], event_mode) if event_mode else None
        if prescale is not None and prescale <= 0:
            status = f'not tested: disabled in {event_mode}'
        elif r == 0 and c == 0:
            status = 'not tested: no event accepted in either arm'
        else:
            status = 'tested'
        paths[name] = {'bit': int(bit), 'status': status, 'stateful': name in stateful,
                       'prescale': prescale, f'{ref_label}_pass': r, f'{cand_label}_pass': c,
                       f'{ref_label}_acceptance': r / n, f'{cand_label}_acceptance': c / n,
                       f'{cand_label}_only': c_only, f'{ref_label}_only': r_only,
                       'paired_difference': diff, 'paired_error': error,
                       'pull': diff / error if error > 0 else (0.0 if c_only == r_only else None)}
        if status == 'tested' and paths[name]['pull'] is not None and abs(paths[name]['pull']) > 2:
            paths[name]['stages'] = stages(name, ref[1], cand[1], ref_label, cand_label)
    return {'reference': ref_label, 'candidate': cand_label, 'events': n,
            'events_in_logs': {ref_label: len(ref[0]), cand_label: len(cand[0]), 'union': len(events)},
            'event_mode': event_mode, 'events_used': 'common to both arms' if common else 'union of both arms',
            'difference': f'{cand_label} - {ref_label}, per event, paired',
            'paths': paths}


def stages(path, ref_modules, cand_modules, ref_label, cand_label):
    """Module-by-module passed counts in both arms, for locating the stage where a path diverges."""
    result = []
    for label, row in ref_modules.get(path, {}).items():
        other = cand_modules.get(path, {}).get(label, {'visited': 0, 'passed': 0})
        result.append({'module': label, f'{ref_label}_passed': row['passed'], f'{cand_label}_passed': other['passed'],
                       'difference': other['passed'] - row['passed']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='LABEL=LOG[,LOG...] reference arm replay logs')
    parser.add_argument('--candidate', required=True, help='LABEL=LOG[,LOG...] candidate arm replay logs')
    parser.add_argument('--prepared', type=Path, required=True, help='replay.py output directory (manifest, inventory)')
    parser.add_argument('--event-mode', help='EventWindowMarker mode of the digis, e.g. OnSpill or OffSpill')
    parser.add_argument('--total-events', type=int, help='denominator including events absent from both arms')
    parser.add_argument('--common', action='store_true', help='only events present in both arms (conditional)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    inventory = json.loads((args.prepared / 'replay_inventory.json').read_text())
    arms = []
    for spec in (args.reference, args.candidate):
        label, logs = parse_arm(spec)
        arms.append((label, read_arm(logs, manifest)))
    report = pair(arms[0], arms[1], manifest, inventory, args.event_mode, args.total_events, args.common)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    ref, cand = report['reference'], report['candidate']
    print(f"events={report['events']} {ref}={report['events_in_logs'][ref]} {cand}={report['events_in_logs'][cand]}")
    for name, row in report['paths'].items():
        if row['status'] == 'tested':
            pull = 'n/a' if row['pull'] is None else f"{row['pull']:+.1f}"
            print(f"{name:32s} {ref} {row[f'{ref}_acceptance']:.4f} {cand} {row[f'{cand}_acceptance']:.4f} "
                  f"diff {100 * row['paired_difference']:+.2f} +- {100 * row['paired_error']:.2f} pp pull {pull}")


if __name__ == '__main__':
    main()
