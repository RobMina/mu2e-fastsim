"""Second-layer release-source consistency gate for feature_table.read output.

This does not load ROOT files, attest a loaded plugin, independently extract
features, or establish measured parity. Every call checks the exact released
source bytes before interpreting status bits; table declarations are not proof.
The input and its structural-loader metadata are never modified or promoted.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

import awkward as ak
import numpy as np

if __package__:
    from . import feature_table as table
else:
    import feature_table as table

# Exact released bytes bind the reviewed semantics, never checkout/head or schema substitutions.
SOURCE_SHA256 = {
    'TrkFitFlag.hh': '7daa714d20ad862673c3a13a8f482b4253d4bfaec931b642810124b78757e26d',
    'TrkFitFlag.cc': '5dc9f78248421b69784827b4f8fc80471419b96d4249331e4c7d077d541ae97a',
    'KalSeed.hh': 'fc6a22b985a7ca6b11b989152d98d5375cb15eb3912239c8aaec56a8a04176cc',
    'BitMap.hh': 'fa3ff7e181fdd75674fc0e4bd44ad4265bded75d22e1fcc7a063d05a888ef595',
    'TrkFitDirection.hh': '5372e70c04af2caab516d8cd19f23a2b90924443efd62b545e1d50d082c90a63',
    'TrkFitDirection.cc': '83b7d43c5622375c9eba9c7cc7cf5727562183424dafbf12bcc0480a90ea6579',
    'KalSeedFilter_module.cc': '51b6dbfe6f5dcd115afdfb8b9ada7aa187b8610f717cdbbc1527e7b573fabd39',
}
BIT_POSITIONS = {'SeedOK': 4, 'KKLoopHelix': 21, 'KKCentralHelix': 22, 'KKLine': 23}
MASKS = {name: 1 << bit for name, bit in BIT_POSITIONS.items()}
FIT_PRECEDENCE = ('KKLoopHelix', 'KKCentralHelix', 'KKLine')
SUPPORTED_FIT_MASK = MASKS['KKLoopHelix'] | MASKS['KKCentralHelix'] | MASKS['KKLine']
PATHS = table.PATHS


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _sources(source_directory):
    try:
        directory = Path(source_directory).resolve(strict=True)
        if not directory.is_dir():
            raise ValueError('source_directory must be a directory')
        receipts = {}
        for name, expected in SOURCE_SHA256.items():
            path = directory / name
            content = path.read_bytes()
            actual = _sha256(content)
            if actual != expected:
                raise ValueError(f'{name}: released source SHA256 mismatch')
            receipts[name] = {'path': str(path), 'sha256': actual, 'size_bytes': len(content)}
        return receipts
    except (OSError, TypeError, RuntimeError) as exc:
        raise ValueError('required released source files are absent or unreadable') from exc


def _mapping(value, name, paths=False):
    if not isinstance(value, Mapping) or (paths and set(value) != set(PATHS)):
        raise ValueError(f'{name}: expected mapping' + (' of the two canonical paths' if paths else ''))
    return value


def _required(mapping, name):
    if name not in mapping:
        raise ValueError(f'missing required field: {name}')
    return mapping[name]


def validate(data, source_directory):
    """Return a JSON-ready report or raise ValueError, without changing data.

    data is the output of feature_table.read, which remains responsible for the
    complete structural/schema gate. This additional gate rechecks the typed,
    nonoptional status/predicate/key arrays and their track alignment, including
    tracks in events where neither filter was reached. Strict ascending accepted
    keys follow from the released filter's single-collection traversal and break
    after the first passing cut. Multiple fit bits are legitimate: branch order
    is loop, central, line, and supported_fit is their OR, independent of SeedOK.

    Exact release-source consistency is not loaded-plugin attestation or
    independent feature extraction. No runtime/parity admission is performed.
    """
    receipts = _sources(source_directory)
    data = _mapping(data, 'data')
    keys = np.asarray(_required(data, 'keys'))
    if keys.ndim != 2 or keys.shape[1] != 3 or keys.dtype.kind not in 'iu':
        raise ValueError('keys must have integer (N,3) shape')
    n = len(keys)
    metadata = _mapping(_required(data, 'metadata'), 'metadata')
    features = _mapping(_required(data, 'features'), 'features')
    counts, tracks = table._ragged(_required(data, 'track_index'), n, 'track_index', 'uint64')
    local = ak.to_numpy(ak.flatten(ak.local_index(data['track_index'], axis=1)), allow_missing=False)
    if not np.array_equal(tracks, local):
        raise ValueError('track_index must be contiguous original collection indices')
    fields = {}
    for name, owner, dtype in (
            ('raw_status', metadata, 'uint32'),
            ('status_has_SeedOK', features, 'uint8'),
            ('supported_fit', features, 'uint8')):
        field_counts, flat = table._ragged(_required(owner, name), n, name, dtype)
        if not np.array_equal(field_counts, counts):
            raise ValueError(f'{name}: per-track alignment differs from track_index')
        if dtype == 'uint8' and np.any((flat != 0) & (flat != 1)):
            raise ValueError(f'{name}: byte boolean must be strict 0/1')
        fields[name] = flat

    raw = fields['raw_status']
    seed = (raw & np.uint32(MASKS['SeedOK'])) != 0
    supported = (raw & np.uint32(SUPPORTED_FIT_MASK)) != 0
    for name, expected in (('status_has_SeedOK', seed), ('supported_fit', supported)):
        mismatches = int(np.count_nonzero(fields[name] != expected))
        if mismatches:
            raise ValueError(f'{name}: {mismatches} tracks contradict released raw_status masks')

    accepted = _mapping(_required(data, 'accepted_keys'), 'accepted_keys', paths=True)
    info = _mapping(_required(data, 'info_present'), 'info_present', paths=True)
    path_counts = {}
    for path in PATHS:
        present = table._vector(info[path], n, f'{path} info_present', 'bool')
        selected_counts, selected = table._ragged(accepted[path], n, f'{path} accepted_keys', 'uint64')
        if np.any((~present) & (selected_counts != 0)):
            raise ValueError(f'{path}: absent TriggerInfo must have zero accepted keys')
        if np.any(selected >= np.repeat(counts, selected_counts)):
            raise ValueError(f'{path}: accepted key outside original collection range')
        rows = np.repeat(np.arange(n), selected_counts)
        if np.any((rows[1:] == rows[:-1]) & (selected[1:] <= selected[:-1])):
            raise ValueError(f'{path}: accepted keys must be strictly ascending within each event')
        path_counts[path] = {
            'accepted_tracks': int(selected.size),
            'events_with_accepted_tracks': int(np.count_nonzero(selected_counts)),
            'tracks_checked_in_unreached_events': int(np.sum(counts[~present])),
        }

    loop = (raw & np.uint32(MASKS['KKLoopHelix'])) != 0
    central = (raw & np.uint32(MASKS['KKCentralHelix'])) != 0
    line = (raw & np.uint32(MASKS['KKLine'])) != 0
    return {
        'source_bound_status_consistent': True,
        'accepted_keys_strictly_ascending': True,
        'offline_release': 'v13_41_01',
        'offline_commit': 'b132ab6567ac021851353e05b7a1feb45bfb7c14',
        'source_files': receipts,
        'bit_positions': dict(BIT_POSITIONS), 'masks': dict(MASKS),
        'supported_fit_mask': SUPPORTED_FIT_MASK,
        'fit_precedence': list(FIT_PRECEDENCE),
        'direction_signs': {'downstream': 1, 'upstream': -1},
        'counts': {
            'events': n, 'tracks': int(raw.size),
            'zero_track_events': int(np.count_nonzero(counts == 0)),
            'status_has_SeedOK': int(np.count_nonzero(seed)),
            'supported_fit': int(np.count_nonzero(supported)),
            'multiple_fit_bits': int(np.count_nonzero(loop.astype(np.uint8) + central + line > 1)),
            'fit_branch_precedence': {
                'KKLoopHelix': int(np.count_nonzero(loop)),
                'KKCentralHelix': int(np.count_nonzero(~loop & central)),
                'KKLine': int(np.count_nonzero(~loop & ~central & line)),
            },
        },
        'paths': path_counts,
        'scope': 'release-source consistency only, for all actual tracks including non-reached events; '
                 'not loaded-plugin attestation or independent feature extraction; '
                 'no runtime admission or measured parity claim; input metadata is unchanged',
    }


def main():
    """Check a completed saved table against decision-art identities, without art."""
    import argparse
    import json
    import uproot
    import event_keys
    import integrity

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('table', 'decision-art', 'source-directory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse saved status report overwrite')
    evidence = integrity.files([args.table, args.decision_art])
    with uproot.open(args.decision_art) as root:
        cycles = [int(key.rsplit(';', 1)[1]) for key in root.keys(recursive=False)
                  if key.startswith('Events;')]
        if not cycles:
            raise ValueError('missing saved Events TTree')
        tree = root[f'Events;{max(cycles)}']
        if tree.classname != 'TTree':
            raise ValueError('highest saved Events object must be TTree')
        keys = event_keys.read(tree)
    data = table.read(args.table, expected_keys=keys)
    report = validate(data, args.source_directory)
    if evidence != integrity.files([args.table, args.decision_art]):
        raise ValueError('saved table or art changed during status check')
    report.update(evidence=evidence, highest_decision_cycle=max(cycles),
                  measured_parity_established=False, runtime_admitted=False,
                  identity_scope='table matches supplied decision art IDs; no independent source traversal admission')
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"saved status check: {len(keys)} events, {report['counts']['tracks']} tracks; no measured parity claim")


if __name__ == '__main__':
    main()
