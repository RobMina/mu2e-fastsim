"""Validate complete-history trigger replay against unchanged first-pass decisions and input identity order."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

import event_keys
import full_replay
import integrity
import path_state
import replay


def validate(path, prepared, directory, config_text, log_text):
    manifest = json.loads((prepared / 'manifest.json').read_text())
    full_replay.preflight(prepared, directory, config_text, manifest['expected_events'])
    census = json.loads((prepared / 'regeneration-census.json').read_text())
    replay_config = replay.parse_expanded((prepared / 'replay.fcl').read_text())
    census_paths = [(directory / row['filename']).resolve() for row in census['files']]
    if census_paths != [Path(name).resolve() for name in replay_config['source']['fileNames']]:
        raise ValueError('census file order differs from bound replay source')
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    input_keys, input_words, ids = [], [], []
    config = replay.production_config(config_text, manifest['original_process'])
    original_paths = replay.paths(config)
    expected = manifest['expected_events']
    for row in census['files']:
        with uproot.open(directory / row['filename'], custom_classes=classes) as root:
            path_state.check_schema(root)
            tree = root['Events']
            keys = event_keys.read(tree)
            words, pset = path_state.read_status(tree, manifest['original_process'], len(original_paths))
            order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
            input_keys.append(keys[order])
            input_words.append(words[order])
            ids.append(pset)
    keys = np.concatenate(input_keys)
    words = np.concatenate(input_words)
    if len(keys) != expected or len(set(ids)) != 1:
        raise ValueError('input counts or trigger schedule IDs differ across regenerated files')
    with uproot.open(path, custom_classes=classes) as root:
        path_state.check_schema(root)
        tree = root['Events']
        retained, pset = path_state.read_status(tree, manifest['original_process'], len(original_paths))
        if pset != ids[0] or not np.array_equal(words, retained):
            raise ValueError('original first-pass packed statuses changed or input order was not preserved')
        if not np.array_equal(event_keys.read(tree), keys):
            raise ValueError('second-pass event identity order differs from original source order')
    report = path_state.compare_file(path, config_text, (prepared / 'replay.fcl').read_text(), manifest,
                                     log_text, keys.tolist(), expected)
    report['saved_replay_file'] = integrity.digest(path)
    report['decision_log_sha256'] = hashlib.sha256(log_text.encode()).hexdigest()
    report['config_sha256'] = manifest['config_sha256']
    report['replay_fcl_sha256'] = manifest['replay_fcl_sha256']
    report['complete_first_pass_words_preserved'] = True
    report['complete_parent_order_verified'] = True
    report['all_menu_accepted_equal'] = report['accepted_mismatches'] == 0
    report['all_menu_states_and_endpoints_equal'] = report['state_mismatches'] == 0 and report['mapped_endpoint_mismatches'] == 0
    report['scope'] = 'all stored preselection-parent events; no generated-event efficiency or independent-shard role inferred by this validator'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('art', type=Path)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.art, args.prepared, args.directory, args.config.read_text(), args.log.read_text())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"events={report['events']} accepted={report['accepted_mismatches']} states={report['state_mismatches']} endpoints={report['mapped_endpoint_mismatches']} raw_packed_equal={report['raw_packed_equal']}")
    raise SystemExit(0 if report['all_menu_accepted_equal'] and report['all_menu_states_and_endpoints_equal'] else 1)


if __name__ == '__main__':
    main()
