"""Bind a primary-only fast step table to every source event, including explicit no-handoff zeros."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

import event_keys
import integrity


KEYS = ['run', 'subrun', 'event']


def keys(tree, label, unique=False):
    columns = tree.arrays(KEYS, library='np')
    values = np.column_stack([columns[key] for key in KEYS])
    if values.dtype.kind not in 'iu' or np.any(values < 0) or np.any(values[:, [0, 2]] == 0):
        raise ValueError(f'invalid integer event identities: {label}')
    if unique and len(np.unique(values, axis=0)) != len(values):
        raise ValueError(f'duplicate event identities: {label}')
    return values.astype(np.uint64)


def ordered(values):
    return values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))]


def subset(values, denominator, label):
    values = np.unique(values, axis=0)
    combined = np.concatenate([denominator, values])
    if len(np.unique(combined, axis=0)) != len(denominator):
        raise ValueError(f'{label} contains identities outside source population')


def audit(source, states, table, handoff, expected):
    if type(expected) is not int or expected <= 0:
        raise ValueError('need positive source count')
    sources = [Path(source)] if isinstance(source, (str, Path)) else list(map(Path, source))
    source_rows = []
    for path in sources:
        with uproot.open(path) as root:
            source_rows.append(event_keys.read(root['Events']))
    source_keys = np.concatenate(source_rows)
    if len(source_keys) != expected or len(np.unique(source_keys, axis=0)) != expected:
        raise ValueError('source count or uniqueness differs')
    with uproot.open(states) as root:
        births = keys(root['dump/gen'], 'births', unique=True)
        entries = keys(root['dump/entries'], 'fast entries', unique=True)
        birth_pdg = root['dump/gen']['pdg'].array(library='np')
        entry_pdg = root['dump/entries']['pdg'].array(library='np')
        if np.any(birth_pdg != 11) or np.any(entry_pdg != 11):
            raise ValueError('transport is not the single-electron diagnostic')
    if not np.array_equal(ordered(source_keys), ordered(births)):
        raise ValueError('fast transport lost source births, including zero-response events')
    subset(entries, source_keys, 'fast entries')
    with uproot.open(table) as root:
        if 'dump/sgsteps' not in root or 'dump/particles' not in root:
            raise ValueError('injector table must carry explicit step and particle schemas, including empty trees')
        steps = keys(root['dump/sgsteps'], 'fast steps')
        particles = keys(root['dump/particles'], 'injector particles', unique=True)
        particle_values = root['dump/particles'].arrays(['key', 'pdg', 'parent', 'isprim'], library='np')
        if (np.any(particle_values['key'] != 1) or np.any(particle_values['pdg'] != 11)
                or np.any(particle_values['parent'] != -1) or np.any(particle_values['isprim'] != 1)):
            raise ValueError('injector table violates single primary electron key-1 contract')
        step_values = root['dump/sgsteps'].arrays(['sp', 'eion', 'len', 'width', 't',
            'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'px', 'py', 'pz'], library='np')
        if (np.any(step_values['sp'] != 1) or any(not np.all(np.isfinite(value)) for value in step_values.values())
                or np.any(step_values['eion'] < 0) or np.any(step_values['len'] <= 0) or np.any(step_values['width'] < 0)):
            raise ValueError('invalid primary step payload')
    if not np.array_equal(ordered(entries), ordered(particles)):
        raise ValueError('injector particle population differs from fast handoffs')
    subset(steps, entries, 'fast steps')
    receipt = json.loads(Path(handoff).read_text())
    if receipt.get('generated_starts') != expected or receipt.get('entries') != len(entries) or receipt.get('steps') != len(steps):
        raise ValueError('transport receipt counts differ from saved output')
    return {'events': expected, 'fast_handoffs': len(entries), 'events_with_fast_steps': len(np.unique(steps, axis=0)),
        'no_handoff_events': expected - len(entries), 'zero_step_events': expected - len(np.unique(steps, axis=0)),
        'steps': len(steps), 'source_birth_keys_complete': True,
        'inputs': integrity.files([*sources, states, table, handoff]), 'transport_seed': receipt['seed'],
        'scope': 'primary-only fast first passage, conditional on existing source selection; no secondaries/returns',
        'zero_policy': 'source events without handoff/steps are explicitly audited zeros; importer must not filter them'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs='+', type=Path, required=True)
    for name in ('states', 'table', 'handoff', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse step admission overwrite')
    report = audit(args.source, args.states, args.table, args.handoff, args.expected_events)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"events={report['events']} explicit_zero_step_events={report['zero_step_events']}")


if __name__ == '__main__':
    main()
