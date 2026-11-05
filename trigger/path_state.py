"""Compare retained TriggerResults states and indices without an art rerun or ROOT runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import awkward as ak
import numpy as np
import uproot

import compare as decisions
import event_keys
import replay
import schedule_registry


class Model_art_3a3a_HLTGlobalStatus_v10(uproot.model.VersionedModel):
    def read_members(self, chunk, cursor, context, file):
        start = cursor.copy()
        size, version, memberwise = uproot.deserialization.numbytes_version(chunk, cursor, context)
        if not memberwise or size is None or version != 10:
            raise ValueError('expected version-10 byte-counted memberwise HLTPathStatus vector')
        header = cursor.bytes(chunk, 2, context)
        length = cursor.field(chunk, struct.Struct('>I'), context)
        if bytes(header) != b'\x00\x0a' or length > 16384 or size != 12 + 2 * length:
            raise ValueError(f'unsupported HLTPathStatus vector framing: size={size}, length={length}')
        self._members['paths_'] = cursor.array(chunk, length, np.dtype('>u2'), context)
        uproot.deserialization.numbytes_check(chunk, start, cursor, size, 'vector<art::HLTPathStatus>', context, file.file_path)

    @classmethod
    def awkward_form(cls, file, context):
        return ak.forms.RecordForm([ak.forms.ListOffsetForm('i64', ak.forms.NumpyForm('uint16'))], ['paths_'])


class Model_art_3a3a_HLTGlobalStatus(uproot.model.DispatchByVersion):
    known_versions = {10: Model_art_3a3a_HLTGlobalStatus_v10}


def check_schema(root):
    expected = {'art::HLTGlobalStatus': (10, [('paths_', 'vector<art::HLTPathStatus>')]),
                'art::HLTPathStatus': (10, [('status_', 'unsigned short')]),
                'art::TriggerResults': (12, [('art::HLTGlobalStatus', 'BASE'), ('psetid_', 'fhicl::ParameterSetID')])}
    for name, (version, fields) in expected.items():
        streamers = root.file.streamers.get(name, {})
        if set(streamers) != {version}:
            raise ValueError(f'unsupported {name} versions: {sorted(streamers)}')
        if [(item.name, item.typename) for item in streamers[version].elements] != fields:
            raise ValueError(f'unsupported {name} member schema')


def read_status(tree, process, expected_paths):
    prefix = f'art::TriggerResults_TriggerResults__{process}'
    parent = tree[prefix + '.']
    present = parent[prefix + '.present'].array(library='np')
    if not np.all(present):
        raise ValueError(f'missing TriggerResults entries in {process}')
    branch = parent[prefix + '.obj']
    interpretation = uproot.interpretation.objects.AsObjects(branch.interpretation.model, branch)
    interpretation._forth = False
    values = branch.array(library='ak', interpretation=interpretation)
    if not ak.all(ak.num(values.paths_, axis=1) == expected_paths):
        raise ValueError(f'path count differs from embedded schedule in {process}')
    if not ak.all(values.psetid_.valid_):
        raise ValueError(f'invalid TriggerResults parameter-set ID in {process}')
    ids = ak.to_numpy(values.psetid_.id_)
    if ids.shape != (tree.num_entries, 20) or not np.all(ids == ids[0]):
        raise ValueError(f'changing or malformed TriggerResults parameter-set ID in {process}')
    return ak.to_numpy(values.paths_), bytes(ids[0]).hex()


def decode(words):
    words = np.asarray(words)
    if words.ndim != 2 or words.dtype.kind not in 'ui' or np.any(words < 0) or np.any(words > 65535):
        raise ValueError('expected a two-dimensional uint16 packed-status matrix')
    state = words & 3
    return state, words >> 2, (state == 0) | (state == 1)


def endpoints(state, index, sequence):
    if len(set(sequence)) != len(sequence):
        raise ValueError('repeated module in path is unsupported for endpoint comparison')
    ran = (state == 1) | (state == 2)
    invalid_zero = (index == 0) & ((state == 2) | ((state == 1) & (len(sequence) != 0)))
    if np.any(ran & ((index > len(sequence)) | invalid_zero)):
        raise ValueError('path index is outside its sequence')
    if np.any((state == 1) & (index != len(sequence))):
        raise ValueError('accepted path index is not its one-past-last executed module')
    labels = np.asarray(['<ready>'] + sequence, dtype=str)
    result = labels[np.where(ran, index, 0)]
    result = np.where(state == 1, '<end-of-path>', result)
    return np.where(state == 3, '<exception-unmapped>', result)


def compare_matrices(stock, candidate, names, original_rows, replay_rows):
    if stock.shape != candidate.shape or stock.shape[1] != len(names):
        raise ValueError('unequal status matrix shapes')
    states_a, indices_a, accepted_a = decode(stock)
    states_b, indices_b, accepted_b = decode(candidate)
    rows = {}
    for col, name in enumerate(names):
        a, b = states_a[:, col], states_b[:, col]
        endpoints_a = endpoints(a, indices_a[:, col], original_rows[name])
        endpoints_b = endpoints(b, indices_b[:, col], replay_rows[name])
        rows[name] = {'accepted_mismatches': int(np.count_nonzero(accepted_a[:, col] != accepted_b[:, col])),
            'state_mismatches': int(np.count_nonzero(a != b)),
            'wasrun_mismatches': int(np.count_nonzero((a != 0) != (b != 0))),
            'raw_index_mismatches': int(np.count_nonzero(indices_a[:, col] != indices_b[:, col])),
            'packed_status_mismatches': int(np.count_nonzero(stock[:, col] != candidate[:, col])),
            'mapped_endpoint_mismatches': int(np.count_nonzero(endpoints_a != endpoints_b)),
            'stock_states': dict(zip(('Ready', 'Pass', 'Fail', 'Exception'), np.bincount(a, minlength=4).tolist())),
            'replay_states': dict(zip(('Ready', 'Pass', 'Fail', 'Exception'), np.bincount(b, minlength=4).tolist()))}
    return rows


def compare_file(path, config_text, replay_text, manifest, log_text, event_list, expected_events):
    original, candidate, names, bits = decisions.validate_manifest(manifest)
    if hashlib.sha256(config_text.encode()).hexdigest() != manifest['config_sha256']:
        raise ValueError('embedded production configuration hash differs from manifest')
    config = replay.production_config(config_text, original)
    replay_config = replay.parse_expanded(replay_text)
    original_paths, replay_paths = replay.paths(config), replay.paths(replay_config)
    if [(bit, name) for bit, name, _ in replay_paths] != list(zip(manifest['bits'], names)):
        raise ValueError('replay configuration schedule differs from manifest')
    original_names = [name for _, name, _ in original_paths]
    original_rows = {name: seq for _, name, seq in original_paths}
    replay_rows = {name: seq for _, name, seq in replay_paths}
    text_report = decisions.compare(log_text, manifest, expected_events, event_list)
    if decisions.COMPLETION.findall(log_text) != ['0']:
        raise ValueError('need exactly one successful art completion marker')
    printed = decisions.read_decisions(log_text, (original, candidate))
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': Model_art_3a3a_HLTGlobalStatus})
    with uproot.open(path, custom_classes=classes) as root:
        check_schema(root)
        tree = root['Events']
        if tree.classname != 'TTree' or tree.num_entries != expected_events:
            raise ValueError('wrong Events type or entry count')
        root_keys = event_keys.read(tree)
        if not np.array_equal(root_keys, np.asarray(event_list, dtype=np.uint64)):
            raise ValueError('ROOT event IDs/order differ from independent input event list')
        stock, stock_id = read_status(tree, original, len(original_names))
        fast, fast_id = read_status(tree, candidate, len(names))
        registry = {original: schedule_registry.verify(root, stock_id, original_paths),
                    candidate: schedule_registry.verify(root, fast_id, replay_paths)}
        cycles = [int(key.rsplit(';', 1)[1]) for key in root.keys(recursive=False) if key.startswith('Events;')]
    stock = stock[:, [original_names.index(name) for name in names]]
    for process, matrix in ((original, stock), (candidate, fast)):
        accepted = decode(matrix)[2]
        for col, (name, bit) in enumerate(zip(names, bits)):
            if process == original and int(original_paths[original_names.index(name)][0]) != bit:
                raise ValueError(f'original bit differs from manifest for {name}')
            logged = np.fromiter((value[process][name][1] for value in printed.values()), dtype=bool, count=expected_events)
            if not np.array_equal(logged, accepted[:, col]):
                raise ValueError(f'ROOT accepted bits differ from validated log for {process}/{name}')
    rows = compare_matrices(stock, fast, names, original_rows, replay_rows)
    totals = {field: sum(row[field] for row in rows.values()) for field in rows[names[0]] if field.endswith('mismatches')}
    if totals['accepted_mismatches'] != text_report['mismatches']:
        raise ValueError('ROOT/log mismatch total differs')
    examples = []
    keys = np.asarray(event_list, dtype=np.uint64)
    state_a, index_a, _ = decode(stock)
    state_b, index_b, _ = decode(fast)
    for col, name in enumerate(names):
        locations = np.flatnonzero(state_a[:, col] != state_b[:, col])[:max(0, 100 - len(examples))]
        examples.extend({'event': key.tolist(), 'path': name, 'stock_state': int(a), 'replay_state': int(b),
                         'stock_index': int(ia), 'replay_index': int(ib)}
                        for key, a, b, ia, ib in zip(keys[locations], state_a[locations, col], state_b[locations, col],
                                                    index_a[locations, col], index_b[locations, col]))
    return {'events': expected_events, 'highest_cycle': max(cycles), 'paths': rows, **totals,
            'state_mismatch_examples': examples, 'parameter_set_ids': {original: stock_id, candidate: fast_id},
            'schedule_registry': registry, 'trigger_schedule_registry_verified': True,
            'raw_packed_equal': totals['packed_status_mismatches'] == 0,
            'states_equal': totals['state_mismatches'] == 0,
            'accepted_bits_crosschecked_with_log': True, 'input_event_ids_verified_by_log': True,
            'root_row_identities_assumed': False, 'state_examples_use_assumed_entry_alignment': False,
            'root_event_ids_independently_decoded': True,
            'comparison': 'retained packed TriggerResults state and raw index, plus mapped endpoint diagnostic',
            'index_convention': 'art 3.15 successful/rejected completion uses executed-worker count (one-past-last)',
            'caveats': ['manifest-path accepted bits match validated log; decoded ROOT IDs/order match independent input event list',
                        'EventAuxiliary decoder supports only the observed fixed nested framing and fails closed otherwise',
                        'removed digitizer/truth-decoration modules change raw indices; mapped endpoints are not serialization identity',
                        'Ready/Exception states are retained, not collapsed into fail; exception endpoints are not mapped',
                        'selected stored input cannot generally restore original invocation counters']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('art', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--input-event-list', type=Path, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare_file(args.art, args.config.read_text(), (args.prepared / 'replay.fcl').read_text(),
        json.loads((args.prepared / 'manifest.json').read_text()), args.log.read_text(),
        decisions.input_event_keys(args.input_event_list.read_text())[:args.expected_events], args.expected_events)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"events={report['events']} state_mismatches={report['state_mismatches']} raw_index_mismatches={report['raw_index_mismatches']}")
    print(f"mapped_endpoint_mismatches={report['mapped_endpoint_mismatches']} raw_packed_equal={report['raw_packed_equal']}")
    raise SystemExit(0 if report['raw_packed_equal'] else 1)


if __name__ == '__main__':
    main()
