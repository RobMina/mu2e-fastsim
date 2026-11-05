"""Fail-closed, event-keyed comparison of stock PrintModule trigger decisions."""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import re

EVENT = re.compile(r'PrintModule Run/Subrun/Event (.*)$')
PRODUCT = re.compile(r'^ProductPrint art::TriggerResults_TriggerResults__([A-Za-z_][A-Za-z_0-9]*)$')
DECISION = re.compile(r'^\s+(pass|fail)\s+(0|[1-9][0-9]*)\s+([A-Za-z_][A-Za-z_0-9]*)\s*$')
COMPLETION = re.compile(r'Art has completed and will exit with status (\d+)\.')


def event_key(fields):
    if len(fields) >= 27:
        parts = [fields[:9].strip(), fields[9:18].strip(), fields[18:].strip()]
    else:
        parts = fields.split()
    if len(parts) != 3 or not all(re.fullmatch(r'[0-9]+', p) for p in parts):
        raise ValueError(f"malformed or ambiguous PrintModule event header: {fields!r}")
    return tuple(map(int, parts))


def validate_manifest(manifest):
    original, replay = manifest['original_process'], manifest['replay_process']
    if original == replay or not all(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', p) for p in (original, replay)):
        raise ValueError("invalid or identical process names")
    names, raw_bits = manifest['paths'], manifest['bits']
    if not names or len(set(names)) != len(names) or len(raw_bits) != len(names):
        raise ValueError("invalid path manifest")
    if not all(isinstance(b, str) and re.fullmatch(r'0|[1-9][0-9]*', b) for b in raw_bits):
        raise ValueError("invalid trigger bits")
    bits = list(map(int, raw_bits))
    if len(set(bits)) != len(bits) or max(bits) > 2147483647:
        raise ValueError("duplicate or unsupported trigger bit")
    if not all(isinstance(p, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', p) for p in names):
        raise ValueError("invalid trigger path name")
    if 'stock_output_selection' in manifest and (not isinstance(manifest['stock_output_selection'], list)
            or not manifest['stock_output_selection']
            or not all(isinstance(p, str) for p in manifest['stock_output_selection'])):
        raise ValueError('invalid or empty stored-output selection')
    if not set(manifest.get('stateful_paths', [])).issubset(names):
        raise ValueError("unknown stateful path")
    return original, replay, names, bits


def read_decisions(text, processes):
    events = {}
    event = None
    process = None
    for line in text.splitlines():
        if match := EVENT.search(line):
            event = event_key(match[1])
            if event in events:
                raise ValueError(f"duplicate event {event}")
            events[event] = {}
            process = None
        elif line.startswith('ProductPrint '):
            process = None
            if line.startswith('ProductPrint art::TriggerResults') and not PRODUCT.match(line):
                raise ValueError(f"malformed TriggerResults header: {line}")
            if match := PRODUCT.match(line):
                name = match[1]
                if name in processes:
                    if event is None:
                        raise ValueError("product precedes event header")
                    if name in events[event]:
                        raise ValueError(f"duplicate process {name} in {event}")
                    process = name
                    events[event][process] = {}
        elif match := DECISION.match(line):
            if process is not None:
                passed, bit, path = match.groups()
                if path in events[event][process]:
                    raise ValueError(f"duplicate path {path} in {event}")
                events[event][process][path] = (int(bit), passed == 'pass')
        elif not line.strip():
            process = None
        elif process is not None and line.strip():
            if re.match(r'^\s+\S+\s+\S+\s+[A-Za-z_][A-Za-z_0-9]*\s*$', line):
                raise ValueError(f"unsupported decision record in {process}: {line}")
            if line.lstrip().startswith(('pass ', 'fail ', 'exception ', 'notrun ')):
                raise ValueError(f"malformed decision record in {process}: {line}")
    if not events:
        raise ValueError("no complete PrintModule event headers")
    for event, products in events.items():
        for process in processes:
            if not products.get(process):
                raise ValueError(f"missing decision product {process} in {event}")
    return events


def input_event_keys(text):
    keys = []
    for line in text.splitlines():
        if re.fullmatch(r'\s*[0-9]+\s+[0-9]+\s+[0-9]+\s*', line):
            keys.append(tuple(map(int, line.split())))
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("empty or duplicate input event list")
    return keys


def compare(text, manifest, expected_events, expected_keys=None):
    if expected_events <= 0:
        raise ValueError("expected event count must be positive")
    original, replay, names, bits = validate_manifest(manifest)
    events = read_decisions(text, (original, replay))
    if len(events) != expected_events:
        raise ValueError(f"finished event count {len(events)} != expected {expected_events}")
    if expected_keys is not None and list(events) != [tuple(key) for key in expected_keys]:
        raise ValueError("replayed event IDs/order differ from the input file event list")
    output_selection = manifest.get('stock_output_selection')
    selection_mismatches = 0
    mismatch_examples = []
    totals = {name: {'bit': bit, 'stock_pass': 0, 'replay_pass': 0, 'mismatches': 0,
                     'stock_only': 0, 'replay_only': 0} for name, bit in zip(names, bits)}
    for event, products in events.items():
        for name, bit in zip(names, bits):
            for process in (original, replay):
                value = products[process].get(name)
                if value is None or value[0] != bit:
                    raise ValueError(f"missing path or wrong bit {name}/{bit} in {process}, event {event}")
            stock, candidate = products[original][name][1], products[replay][name][1]
            totals[name]['stock_pass'] += int(stock)
            totals[name]['replay_pass'] += int(candidate)
            totals[name]['mismatches'] += int(stock != candidate)
            totals[name]['stock_only'] += int(stock and not candidate)
            totals[name]['replay_only'] += int(candidate and not stock)
            if stock != candidate and len(mismatch_examples) < 100:
                mismatch_examples.append({'event': list(event), 'path': name,
                                          'stock': stock, 'replay': candidate})
        if set(products[replay]) != set(names):
            raise ValueError(f"unexpected replay paths in {event}")
        if output_selection is not None:
            def selected(process):
                passed = False
                for pattern in output_selection:
                    if not re.fullmatch(r'[A-Za-z_0-9*?]+', pattern):
                        raise ValueError(f"unsupported output selector: {pattern}")
                    matches = [name for name in products[original] if fnmatch.fnmatchcase(name, pattern)]
                    if not matches:
                        raise ValueError(f"output selector matches no original path: {pattern}")
                    for path in matches:
                        if process == replay and path not in names:
                            decision = products[original][path][1]
                        else:
                            decision = products[process][path][1]
                        passed |= decision
                return passed
            if not selected(original):
                raise ValueError(f"stored event {event} fails its original output selection")
            selection_mismatches += int(not selected(replay))
    for row in totals.values():
        row['paired_efficiency_difference'] = (row['replay_pass'] - row['stock_pass']) / len(events)
    mismatches = sum(row['mismatches'] for row in totals.values())
    stateful = set(manifest.get('stateful_paths', []))
    nonstateful_mismatches = sum(row['mismatches'] for name, row in totals.items() if name not in stateful)
    return {'events': len(events), 'paths': totals, 'mismatches': mismatches,
            'mismatch_examples': mismatch_examples,
            'stateful_paths': sorted(stateful), 'nonstateful_mismatches': nonstateful_mismatches,
            'equal': mismatches == 0, 'comparison': 'accepted_boolean_only',
            'full_art_state_compared': False, 'input_event_ids_verified': expected_keys is not None,
            'stored_output_selection_mismatches': selection_mismatches if output_selection is not None else None,
            'selection_caveat': 'original truth-only path bits retained; conditional stored-input selection only',
            'denominator': 'stored input events, not generated events'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-event-list', type=Path, help='file_info_dumper --event-list output')
    args = parser.parse_args()
    text = args.log.read_text()
    statuses = COMPLETION.findall(text)
    if statuses != ['0']:
        raise ValueError("need exactly one successful art completion marker")
    expected = None
    if args.input_event_list:
        expected = input_event_keys(args.input_event_list.read_text())[:args.expected_events]
    report = compare(text, json.loads(args.manifest.read_text()), args.expected_events, expected)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"events={report['events']} path_mismatches={report['mismatches']} accepted decisions only")
    for name, row in report['paths'].items():
        if row['mismatches']:
            print(f"{name} mismatches={row['mismatches']} stock_only={row['stock_only']} replay_only={row['replay_only']}")
    raise SystemExit(0 if report['equal'] else 1)


if __name__ == '__main__':
    main()
