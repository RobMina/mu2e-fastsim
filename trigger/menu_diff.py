"""Compare two replay.py inventories from embedded production configurations.

Use: python3 trigger/menu_diff.py REFERENCE CANDIDATE --output REPORT
Reports exact JSON settings (including module_label), not physics equivalence.
Object key order is immaterial; array order and JSON scalar types are retained.
Shared module definitions and their changes are recorded once, with path users.
Exit 0 means the comparison completed (even if different); invalid inputs exit 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

CONTEXT_FIELDS = ('digitize_modules', 'output_selection', 'geometry', 'conditions')
PATH_FIELDS = ('bit', 'path', 'prescales', 'sequence', 'modules')
NAME = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
PREVIEW_LIMIT = 12


def canonical(value):
    """Canonical JSON: sorted object keys; preserve arrays and scalar types."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False)


def changes(reference, candidate, pointer=''):
    """Lossless leaf changes using escaped RFC 6901 JSON pointers.

    Absent keys/indices use add/remove, distinct from an explicit JSON null.
    Array indices refer to the original inputs, not a sequential JSON Patch.
    """
    if canonical(reference) == canonical(candidate):
        return []
    result = []
    if isinstance(reference, dict) and isinstance(candidate, dict):
        for key in sorted(reference.keys() | candidate.keys()):
            location = pointer + '/' + key.replace('~', '~0').replace('/', '~1')
            if key not in reference:
                result.append({'op': 'add', 'pointer': location, 'candidate': candidate[key]})
            elif key not in candidate:
                result.append({'op': 'remove', 'pointer': location, 'reference': reference[key]})
            else:
                result.extend(changes(reference[key], candidate[key], location))
    elif isinstance(reference, list) and isinstance(candidate, list):
        for index in range(max(len(reference), len(candidate))):
            location = f'{pointer}/{index}'
            if index >= len(reference):
                result.append({'op': 'add', 'pointer': location, 'candidate': candidate[index]})
            elif index >= len(candidate):
                result.append({'op': 'remove', 'pointer': location, 'reference': reference[index]})
            else:
                result.extend(changes(reference[index], candidate[index], location))
    else:
        result.append({'op': 'replace', 'pointer': pointer,
                       'reference': reference, 'candidate': candidate})
    return result


def _require(condition, message):
    if not condition:
        raise ValueError('inventory: ' + message)


def _names(values):
    return isinstance(values, list) and all(isinstance(x, str) and NAME.fullmatch(x) for x in values)


def validate(inventory):
    """Validate replay's scoped inventory and index globally consistent labels."""
    _require(isinstance(inventory, dict), 'expected an object')
    _require(all(key in inventory for key in ('paths', *CONTEXT_FIELDS)), 'missing required keys')
    _require(isinstance(inventory['paths'], list), 'paths must be a list')
    _require(_names(inventory['digitize_modules']), 'digitize_modules must be a label list')
    _require(isinstance(inventory['output_selection'], list) and
             all(isinstance(x, str) for x in inventory['output_selection']),
             'output_selection must be a string list')
    _require(all(isinstance(inventory[key], dict) for key in ('geometry', 'conditions')),
             'geometry and conditions must be objects')
    canonical(inventory)  # Reject nonfinite values even in otherwise unused metadata.
    paths, modules, users, bits = {}, {}, {}, set()
    for row in inventory['paths']:
        _require(isinstance(row, dict) and all(key in row for key in PATH_FIELDS),
                 'path missing required keys')
        name, bit = row['path'], row['bit']
        _require(isinstance(name, str) and NAME.fullmatch(name), 'invalid path name')
        _require(name not in paths, 'duplicate path name')
        _require(bit is None or (isinstance(bit, str) and re.fullmatch(r'0|[1-9][0-9]*', bit)
                                and int(bit) <= 2147483647), 'invalid scheduled bit')
        _require(bit is None or bit not in bits, 'duplicate path bit')
        if bit is not None:
            bits.add(bit)
        _require(_names(row['sequence']), 'sequence must be a label list')
        _require(isinstance(row['modules'], dict), 'modules must be an object')
        _require(all(isinstance(label, str) and NAME.fullmatch(label) for label in row['modules']),
                 'invalid module label')
        _require(set(row['sequence']) <= row['modules'].keys(), 'undefined sequence label')
        _require(set(row['sequence']) == row['modules'].keys(), 'unscheduled module definition')
        expected_prescales = []
        for label in row['sequence']:
            settings = row['modules'][label]
            _require(isinstance(settings, dict) and isinstance(settings.get('module_type'), str)
                     and bool(settings['module_type']), 'missing module_type')
            _require(label not in modules or canonical(modules[label]) == canonical(settings),
                     'inconsistent shared module definitions')
            modules[label] = settings
            users.setdefault(label, set()).add(name)
            if settings['module_type'] == 'PrescaleEvent':
                _require('eventModeConfig' in settings, 'prescale missing eventModeConfig')
                expected_prescales.append({'label': label, 'eventModeConfig': settings['eventModeConfig']})
        _require(isinstance(row['prescales'], list), 'prescales must be a list')
        _require(canonical(row['prescales']) == canonical(expected_prescales),
                 'prescales inconsistent with scheduled module settings')
        paths[name] = row
    return paths, modules, users


def _pair(reference, candidate):
    return {'reference': reference, 'candidate': candidate}


def _extras(value, known):
    return {key: item for key, item in value.items() if key not in known}


def _path_record(row):
    # Definitions live only in the global module section, not once per path.
    return {**_extras(row, ('modules',)), 'module_labels': sorted(row['modules'])}


def compare(reference, candidate):
    """Return a deterministic, JSON-serializable comparison; inputs are unmodified."""
    ref_paths, ref_modules, ref_users = validate(reference)
    new_paths, new_modules, new_users = validate(candidate)
    ref_names, new_names = set(ref_paths), set(new_paths)
    common_paths = ref_names & new_names
    local_changes = {}
    for name in sorted(common_paths):
        before, after = ref_paths[name], new_paths[name]
        fields = {key: _pair(before[key], after[key]) for key in ('bit', 'sequence', 'prescales')
                  if canonical(before[key]) != canonical(after[key])}
        if before['modules'].keys() != after['modules'].keys():
            fields['module_labels'] = _pair(sorted(before['modules']), sorted(after['modules']))
        extra = changes(_extras(before, PATH_FIELDS), _extras(after, PATH_FIELDS))
        if extra:
            fields['metadata_changes'] = extra
        if fields:
            local_changes[name] = fields
    ref_labels, new_labels = set(ref_modules), set(new_modules)
    module_changes = []
    affected = set(local_changes)
    for label in sorted(ref_labels & new_labels):
        if canonical(ref_modules[label]) == canonical(new_modules[label]):
            continue
        affected_paths = sorted(ref_users[label] | new_users[label])
        affected.update(common_paths & set(affected_paths))
        module_changes.append({
            'label': label,
            'type': _pair(ref_modules[label]['module_type'], new_modules[label]['module_type']),
            'reference_paths': sorted(ref_users[label]), 'candidate_paths': sorted(new_users[label]),
            'affected_paths': affected_paths,
            **_pair(ref_modules[label], new_modules[label]),
            'changes': changes(ref_modules[label], new_modules[label]),
        })
    added_modules = [{'label': label, 'type': new_modules[label]['module_type'],
                      'paths': sorted(new_users[label]),
                      'settings': new_modules[label]} for label in sorted(new_labels - ref_labels)]
    removed_modules = [{'label': label, 'type': ref_modules[label]['module_type'],
                        'paths': sorted(ref_users[label]),
                        'settings': ref_modules[label]} for label in sorted(ref_labels - new_labels)]
    contexts = {key: {'equal': canonical(reference[key]) == canonical(candidate[key]),
                      'changes': changes(reference[key], candidate[key])} for key in CONTEXT_FIELDS}
    return {
        'schema_version': 1,
        'equal': canonical(reference) == canonical(candidate),
        'summary': {
            'paths_reference': len(ref_paths), 'paths_candidate': len(new_paths),
            'paths_added': len(new_names - ref_names), 'paths_removed': len(ref_names - new_names),
            'paths_changed': len(affected), 'paths_locally_changed': len(local_changes),
            'bits_changed': sum('bit' in row for row in local_changes.values()),
            'sequences_changed': sum('sequence' in row for row in local_changes.values()),
            'prescales_changed': sum('prescales' in row for row in local_changes.values()),
            'modules_reference': len(ref_modules), 'modules_candidate': len(new_modules),
            'modules_added': len(added_modules), 'modules_removed': len(removed_modules),
            'modules_changed': len(module_changes),
        },
        'paths': {
            'added': [_path_record(new_paths[name]) for name in sorted(new_names - ref_names)],
            'removed': [_path_record(ref_paths[name]) for name in sorted(ref_names - new_names)],
            'changed': local_changes,
            'order': {'equal': list(ref_paths) == list(new_paths),
                      **_pair(list(ref_paths), list(new_paths))},
        },
        'modules': {'added': added_modules, 'removed': removed_modules,
                    'changed': module_changes},
        **contexts,
        'metadata_changes': changes(_extras(reference, ('paths', *CONTEXT_FIELDS)),
                                    _extras(candidate, ('paths', *CONTEXT_FIELDS))),
        'interpretation': ('Exact embedded configuration comparison, not physics equivalence, '
                           'replay closure or efficiency. Paths changed counts common paths '
                           'with local edits or changed shared settings; arrays retain order.'),
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON object key')
        result[key] = value
    return result


def _load(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_object)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True,
                        help='JSON report (contains the original parameter values)')
    args = parser.parse_args()
    try:
        if args.output.resolve() in {args.reference.resolve(), args.candidate.resolve()}:
            raise ValueError('output must not overwrite an input')
        report = compare(_load(args.reference), _load(args.candidate))
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        args.output.write_text(rendered)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        parser.exit(1, f'menu comparison failed: {error}\n')
    counts = report['summary']
    print(f"equal={str(report['equal']).lower()} "
          f"paths +{counts['paths_added']} -{counts['paths_removed']} changed={counts['paths_changed']} "
          f"(bit={counts['bits_changed']} sequence={counts['sequences_changed']} "
          f"prescale={counts['prescales_changed']}) "
          f"modules +{counts['modules_added']} -{counts['modules_removed']} changed={counts['modules_changed']}")
    print(' '.join(f"{key}_equal={str(report[key]['equal']).lower()}" for key in CONTEXT_FIELDS))
    previewed = 0
    for kind in ('changed', 'added', 'removed'):
        for row in report['modules'][kind]:
            if previewed >= PREVIEW_LIMIT:
                break
            module_type = row['type']
            if isinstance(module_type, dict):
                module_type = f"{module_type['reference']} -> {module_type['candidate']}"
            keys = (sorted({change['pointer'].split('/')[1] for change in row['changes']})
                    if kind == 'changed' else sorted(row['settings']))
            text = f"module {kind}: {row['label']} type={module_type} keys={','.join(keys[:12])}"
            if len(keys) > 12:
                text += f",... (+{len(keys) - 12})"
            print(text.encode('unicode_escape').decode('ascii')[:600])
            previewed += 1
    total = counts['modules_changed'] + counts['modules_added'] + counts['modules_removed']
    if total > previewed:
        print(f'{total - previewed} additional module change records in the JSON report')


if __name__ == '__main__':
    main()
