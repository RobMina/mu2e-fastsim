"""Bind TriggerResults schedule IDs to literal trigger paths in the embedded SQLite registry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import replay


def resolve(payload, parameter_set_id, expected_paths):
    if not payload.startswith(b'SQLite format 3\x00') or len(payload) > 512 * 1024 * 1024:
        raise ValueError('unsupported embedded registry payload')
    if len(parameter_set_id) != 40 or any(c not in '0123456789abcdef' for c in parameter_set_id):
        raise ValueError('unsupported trigger schedule parameter-set ID')
    with sqlite3.connect(':memory:') as database:
        database.deserialize(payload)
        database.execute('PRAGMA query_only=ON')
        columns = database.execute('PRAGMA table_info(ParameterSets)').fetchall()
        if [column[1] for column in columns] != ['ID', 'PSetBlob']:
            raise ValueError('unsupported embedded ParameterSets schema')
        rows = database.execute('SELECT PSetBlob FROM ParameterSets WHERE ID IN (?, ?)',
                                (parameter_set_id, parameter_set_id + '\x00')).fetchall()
    if len(rows) != 1 or not isinstance(rows[0][0], str):
        raise ValueError('missing, ambiguous or unsupported trigger schedule registry entry')
    blob = rows[0][0]
    text = blob[:-1] if blob.endswith('\x00') else blob
    if '\x00' in text:
        raise ValueError('unsupported embedded registry string framing')
    parsed = replay.parse_expanded(text)
    expected = [f'{bit if bit is not None else "0"}:{name}' for bit, name, _ in expected_paths]
    if set(parsed) != {'trigger_paths'} or replay.plain(parsed['trigger_paths']) != expected:
        raise ValueError('embedded trigger schedule differs from configuration path names, bits or order')
    return {'parameter_set_id': parameter_set_id, 'registry_schedule_verified': True,
            'paths': len(expected), 'registry_blob_sha256': hashlib.sha256(blob.encode()).hexdigest(),
            'binding': 'TriggerResults parameter-set ID resolves to exact ordered bit:path schedule in RootFileDB'}


def verify(root, parameter_set_id, expected_paths):
    key = root.key('RootFileDB')
    if key.fClassName != 'TKey' or key.fObjlen > 512 * 1024 * 1024:
        raise ValueError('unsupported embedded registry key class or size')
    chunk, cursor = key.get_uncompressed_chunk_cursor()
    if cursor.index != 0 or chunk.start != 0 or len(chunk.raw_data) != key.fObjlen:
        raise ValueError('unsupported embedded registry key framing')
    return resolve(bytes(chunk.raw_data), parameter_set_id, expected_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('art', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--replay-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import path_state
    import uproot

    config = replay.production_config(args.config.read_text(), 'Digitize')
    candidate = replay.parse_expanded(args.replay_config.read_text())
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    report = {}
    with uproot.open(args.art, custom_classes=classes) as root:
        path_state.check_schema(root)
        for process, parsed in [('Digitize', config), ('TriggerReplay', candidate)]:
            paths = replay.paths(parsed)
            _, parameter_set_id = path_state.read_status(root['Events'], process, len(paths))
            report[process] = verify(root, parameter_set_id, paths)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('embedded trigger schedules verified for both processes')


if __name__ == '__main__':
    main()
