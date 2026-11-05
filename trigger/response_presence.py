"""Fail-closed persisted SGS presence metadata; never decode SGS or art pointer payloads.

inspect discovers a unique compressDetStepMCs product unless given an exact tag.
verify(source, receipt, tag=...) re-inspects immutable bytes and requires that exact
label:instance:process. admit keeps supplied parent order, sorting only within each
parent for numeric traversal. No process default is inferred from a dumper label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

import event_keys
import integrity
import serialized


FRIENDLY_TYPE = 'mu2e::StrawGasSteps'
LABEL = 'compressDetStepMCs'
TAG = re.compile(r'[A-Za-z_][A-Za-z_0-9]*:[A-Za-z_0-9]*:[A-Za-z_][A-Za-z_0-9]*\Z')
OBJECT_TYPES = {'vector<mu2e::StrawGasStep>', 'mu2e::StrawGasSteps', 'mu2e::StrawGasStepCollection'}
LIMITS = [
    'Persisted wrapper presence only; no SGS numerical values, collection sizes or particle identity are decoded.',
    'Nonempty serialized entries include framing; present-empty SGS collections are permitted and not identified.',
    'SimParticles candidates are metadata only, not a particle-presence or pointer-resolution certification.',
    'Does not prove dumper runtime tag resolution, complete production population, geometry, conditions or response equivalence.',
    'Digests detect before/after changes, not adversarial concurrent modification and restoration.',
]


def _canonical(typename):
    return ''.join(typename.split()).replace('std::', '')


def _tag(tag):
    if not isinstance(tag, str) or not TAG.fullmatch(tag):
        raise ValueError('need exact label:instance:process SGS tag')
    return tag


def _latest(root):
    cycles = [int(name.rsplit(';', 1)[1]) for name in root.keys(recursive=False)
              if re.fullmatch(r'Events;[1-9][0-9]*', name)]
    if not cycles:
        raise ValueError('missing Events TTree cycle')
    cycle = max(cycles)
    tree = root[f'Events;{cycle}']
    if tree.classname != 'TTree':
        raise ValueError('highest Events cycle is not a TTree')
    if type(tree.num_entries) is not int or tree.num_entries <= 0:
        raise ValueError('need positive finished Events entries')
    return tree, cycle


def _product(tree, tag):
    label = LABEL if tag is None else _tag(tag).split(':')[0]
    prefix = f'{FRIENDLY_TYPE}_{label}_'
    names = list(tree.keys(recursive=False))
    candidates = [name for name in names if name.startswith(prefix) and name.endswith('.')]
    if tag is None:
        if len(candidates) != 1:
            raise ValueError('need unique SGS branch for discovery; missing or ambiguous product')
        branch_name = candidates[0]
        empty_instance = f'{FRIENDLY_TYPE}_{LABEL}__'
        if not branch_name.startswith(empty_instance):
            raise ValueError('nonempty instance needs an explicit SGS tag')
        tag = _tag(f'{LABEL}::{branch_name[len(empty_instance):-1]}')
    else:
        tag = _tag(tag)
        branch_name = f'{FRIENDLY_TYPE}_{tag.replace(":", "_")}.'
        if candidates.count(branch_name) != 1:
            raise ValueError('missing or ambiguous exact SGS branch')
    parent = tree[branch_name]
    stem = branch_name[:-1]
    children = list(parent.keys(recursive=False))
    for suffix in ('.present', '.obj'):
        if children.count(stem + suffix) != 1:
            raise ValueError(f'missing or ambiguous wrapper child: {stem + suffix}')
    present_branch, obj = parent[stem + '.present'], parent[stem + '.obj']
    count = tree.num_entries
    if parent.num_entries != count or present_branch.num_entries != count or obj.num_entries != count:
        raise ValueError('wrapper/object/present entry count differs from Events')
    object_type = _canonical(obj.typename)
    if object_type not in OBJECT_TYPES or _canonical(parent.typename) != f'art::Wrapper<{object_type}>':
        raise ValueError('unsupported SGS wrapper or object typename')
    present = present_branch.array(library='np')
    if (not isinstance(present, np.ndarray) or np.ma.isMaskedArray(present)
            or present.dtype != np.dtype(np.bool_) or present.shape != (count,)):
        raise ValueError('wrapper present must have exact boolean dtype and event count')
    if not np.all(present):
        raise ValueError('absent SGS product in one or more events')
    values = serialized.entry_bytes(obj)
    lengths = ak.to_numpy(ak.num(values, axis=1))
    if len(values) != count or lengths.shape != (count,) or np.any(lengths <= 0):
        raise ValueError('empty serialized SGS entry or incorrect count')
    return {'friendly_type': FRIENDLY_TYPE, 'tag': tag, 'process': tag.split(':')[2],
            'branch': branch_name, 'wrapper_typename': parent.typename, 'object_typename': obj.typename,
            'present_typename': present_branch.typename, 'entries': count, 'all_present': True,
            'serialized_entries_nonempty': True}


def inspect(source, expected_events=None, tag=None):
    """Inspect one local source; optional exact tag, otherwise unique empty-instance discovery."""
    if expected_events is not None and (type(expected_events) is not int or expected_events <= 0):
        raise ValueError('expected_events must be a positive integer')
    if tag is not None:
        _tag(tag)
    source = Path(source)
    before = integrity.digest(source)
    with uproot.open(source) as root:
        tree, cycle = _latest(root)
        if expected_events is not None and tree.num_entries != expected_events:
            raise ValueError('Events count differs from expected_events')
        keys = event_keys.read(tree)
        if (not isinstance(keys, np.ndarray) or keys.shape != (tree.num_entries, 3)
                or keys.dtype.kind not in 'iu' or np.any(keys < 0)
                or np.any(keys[:, [0, 2]] == 0) or np.any(keys >= 2**32 - 1)
                or len(np.unique(keys, axis=0)) != tree.num_entries):
            raise ValueError('invalid or duplicate event keys')
        product = _product(tree, tag)
        streamers = {name: {str(version): [{'name': item.name, 'typename': item.typename}
                                         for item in streamer.elements]
                            for version, streamer in versions.items()}
                     for name, versions in root.file.streamers.items()
                     if any(token in name for token in ('StrawGasStep', 'SimParticle', 'art::Ptr',
                                                       'EventAuxiliary', 'EventID', 'SubRunID', 'RunID'))}
        process = product['process']
        particles = [name for name in tree.keys(recursive=False)
                     if name == f'mu2e::SimParticles_{LABEL}__{process}.']
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        report = {'schema': 'response-presence-v1', 'events': tree.num_entries,
                  'highest_cycle': cycle, 'product': product, 'tag': product['tag'],
                  'all_present': True, 'event_keys_physical': keys.tolist(),
                  'event_keys_numeric': keys[order].tolist(), 'streamers': streamers,
                  'simparticles_candidates': particles, 'simparticles_presence_checked': False,
                  'scope': 'persisted SGS product presence metadata only; not physics validation',
                  'limits': list(LIMITS)}
    after = integrity.digest(source)
    if before != after:
        raise ValueError('source bytes changed during presence inspection')
    return dict(report, source_before=before, source_after=after)


def verify(source, receipt, expected_events=None, *, tag):
    """Reinspect and bind a single-file receipt; caller must pin the exact SGS tag."""
    _tag(tag)
    if not isinstance(receipt, dict) or receipt.get('tag') != tag:
        raise ValueError('receipt does not bind the caller exact SGS tag')
    current = inspect(source, expected_events, tag)
    if current != receipt:
        raise ValueError('presence receipt differs from current source inspection')
    return current


def admit(source, counts=None, tag=None):
    """Inspect sequential parents; counts, when supplied, are positive per-parent counts."""
    sources = [Path(source)] if isinstance(source, (str, Path)) else list(map(Path, source))
    if not sources or len(set(path.resolve() for path in sources)) != len(sources):
        raise ValueError('need nonempty unique sequential source files')
    counts = [None] * len(sources) if counts is None else list(counts)
    if len(counts) != len(sources):
        raise ValueError('counts must correspond one-to-one with source parents')
    rows = [inspect(path, count, tag) for path, count in zip(sources, counts)]
    if len({row['tag'] for row in rows}) != 1:
        raise ValueError('source parents have different exact SGS tags')
    physical = np.concatenate([np.asarray(row['event_keys_physical'], dtype=np.uint64) for row in rows])
    numeric = np.concatenate([np.asarray(row['event_keys_numeric'], dtype=np.uint64) for row in rows])
    if len(np.unique(physical, axis=0)) != len(physical):
        raise ValueError('duplicate event keys across source parents')
    return {'schema': 'response-presence-parents-v1', 'events': len(physical), 'total_events': len(physical),
            'parents': len(rows), 'files': rows, 'tag': rows[0]['tag'], 'all_present': True,
            'event_keys_physical': physical.tolist(), 'event_keys_numeric': numeric.tolist(),
            'source_traversal': 'supplied parent order; numeric (run,subrun,event) only within each parent',
            'scope': 'persisted SGS product presence metadata only; not physics validation', 'limits': list(LIMITS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, nargs='+', required=True)
    parser.add_argument('--counts', type=int, nargs='+')
    parser.add_argument('--tag', help='exact compressDetStepMCs:instance:process; otherwise discover uniquely')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise ValueError('refuse presence report overwrite')
    report = admit(args.source, args.counts, args.tag)
    text = json.dumps(report, indent=2) + '\n'
    if args.output is not None:
        with args.output.open('x') as output:
            output.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
