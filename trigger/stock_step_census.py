"""Saved-only original SGS truth census; migration strata are DIAGNOSTIC, not corrections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

import event_keys
import integrity
import path_state
import replay
import stage_flow as flow

KEYS = ['run', 'subrun', 'event']
INTS = ['nsgs', 'sp', 'shape', 'ion', 'sid', 'upanel', 'straw']
FLOATS = ['eion', 'len', 'width', 't', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'px', 'py', 'pz']


def records(values):
    values = np.ascontiguousarray(values, dtype=np.int64)
    return values.view(np.dtype([(str(i), '<i8') for i in range(values.shape[1])])).ravel()


def join(rows, population):
    """Exact vectorized foreign-key lookup, independent of row sorting."""
    a, b = records(rows), records(population)
    order = np.argsort(b)
    pos = np.searchsorted(b[order], a)
    if len(a) and (not len(b) or np.any(pos >= len(b))):
        raise ValueError('foreign identity')
    if len(a) and np.any(b[order[pos]] != a):
        raise ValueError('foreign identity')
    return order[pos]


def integer(data, names):
    for name in names:
        if (name not in data or data[name].ndim != 1 or data[name].dtype.kind not in 'iu'
                or np.any(data[name] > np.iinfo('i8').max)):
            raise ValueError(f'invalid integer field: {name}')


def identities(data):
    integer(data, KEYS)
    rows = np.column_stack([data[k] for k in KEYS]).astype('i8')
    if np.any(rows < 0) or np.any(rows[:, [0, 2]] == 0):
        raise ValueError('invalid event identities')
    return rows


def census(tables, denominator):
    """One table at a time; only event-indexed aggregates survive each file."""
    if len(np.unique(denominator, axis=0)) != len(denominator):
        raise ValueError('duplicate denominator identities')
    n = len(denominator)
    fields = ('steps', 'eion', 'length', 'primary_steps', 'primary_eion', 'primary_length',
              'isprim_electron_steps', 'ismcprim_electron_steps', 'isprim_electron_eion',
              'ismcprim_electron_eion', 'isprim_electron_length', 'ismcprim_electron_length')
    totals = {k: np.zeros(n, dtype='f8' if 'eion' in k or 'length' in k else 'i8') for k in fields}
    seen = np.zeros(n, dtype=bool)
    species, dangling, repeated, multi, mode = {}, 0, 0, 0, None
    for table in tables:
        with uproot.open(table) as root:
            gen = root['dump/gen'].arrays(library='np')
            particles = root['dump/particles'].arrays(library='np')
            steps = root['dump/sgsteps'].arrays(library='np')
        births = identities(gen)
        idx = join(births, denominator)
        if len(np.unique(idx)) != len(idx) or np.any(seen[idx]):
            raise ValueError('duplicate source birth identities')
        seen[idx] = True
        pk, sk = identities(particles), identities(steps)
        join(pk, births)
        si = join(sk, births)
        integer(particles, ['key', 'pdg', 'parent', 'isprim'])
        has_mc = 'ismcprim' in particles
        if mode is not None and mode != has_mc:
            raise ValueError('inconsistent ismcprim schema')
        mode = has_mc
        if has_mc:
            integer(particles, ['ismcprim'])
        flags = ['isprim'] + (['ismcprim'] if has_mc else [])
        if (np.any(particles['key'] < 0) or np.any(particles['parent'] < -1)
                or any(np.any(~np.isin(particles[k], [0, 1])) for k in flags)):
            raise ValueError('invalid particle key/parent/primary flag')
        particle_keys = np.column_stack([pk, particles['key']])
        if len(np.unique(particle_keys, axis=0)) != len(pk):
            raise ValueError('duplicate particle keys')
        parents = np.column_stack([pk, particles['parent']])
        dangling += int(np.count_nonzero((particles['parent'] >= 0) & ~np.isin(records(parents), records(particle_keys))))
        integer(steps, INTS)
        for key in FLOATS:
            if key not in steps or steps[key].dtype.kind not in 'iuf' or not np.all(np.isfinite(steps[key])):
                raise ValueError(f'invalid finite numeric payload: {key}')
        if (any(np.any(steps[k] < 0) for k in ['sp', 'eion', 'len', 'width', 'sid', 'upanel', 'straw'])
                or np.any(steps['sid'] > 65535) or np.any(steps['upanel'] >= 216)
                or np.any(steps['straw'] >= 96) or np.any(~np.isin(steps['shape'], [0, 1, 2, 3]))
                or np.any(~np.isin(steps['ion'], [0, 1, 2]))):
            raise ValueError('SGS payload range violation')
        plane, panel = steps['sid'] >> 10, (steps['sid'] >> 7) & 7
        if (np.any(plane >= 36) or np.any(panel >= 6)
                or np.any(steps['upanel'] != plane * 6 + panel) or np.any(steps['straw'] != (steps['sid'] & 127))):
            raise ValueError('inconsistent StrawId mapping')
        for key, expected in [('plane', plane), ('panel', panel)]:
            if key in steps:
                integer(steps, [key])
                if np.any(steps[key] != expected):
                    raise ValueError('inconsistent plane/panel mapping')
        counts = np.bincount(si, minlength=len(births))
        if np.any(steps['nsgs'] != counts[si]):
            raise ValueError('nsgs differs from event row count')
        association = join(np.column_stack([sk, steps['sp']]), particle_keys)
        pdg = particles['pdg'][association]
        primary = (pdg == 11) & (particles['isprim'][association] == 1)
        mc = (pdg == 11) & (particles['ismcprim'][association] == 1) if has_mc else np.zeros(len(sk), bool)
        chosen = mc if has_mc else primary
        gi = idx[si]
        weights = [None, steps['eion'], steps['len'], chosen, chosen * steps['eion'],
                   chosen * steps['len'], primary, mc, primary * steps['eion'], mc * steps['eion'],
                   primary * steps['len'], mc * steps['len']]
        for key, weight in zip(fields, weights):
            totals[key] += np.bincount(gi, weights=weight, minlength=n).astype(totals[key].dtype)
        ids, inv = np.unique(pdg, return_inverse=True)
        counts = np.bincount(inv)
        energies, lengths = (np.bincount(inv, weights=steps[k]) for k in ('eion', 'len'))
        for i, code in enumerate(ids):
            row = species.setdefault(str(int(code)), {'steps': 0, 'eion': 0., 'length': 0.})
            row['steps'] += int(counts[i])
            row['eion'] += float(energies[i])
            row['length'] += float(lengths[i])
        _, multiplicity = np.unique(np.column_stack([sk, steps['sp'], steps['sid']]), axis=0, return_counts=True)
        repeated += int(np.maximum(multiplicity - 1, 0).sum())
        multi += int(np.count_nonzero(multiplicity > 1))
    if not np.all(seen):
        raise ValueError('source births do not cover exact saved denominator, including zero steps')
    totals['other_steps'] = totals['steps'] - totals['primary_steps']
    totals['other_eion'] = totals['eion'] - totals['primary_eion']
    totals['other_length'] = totals['length'] - totals['primary_length']
    return totals, {'species_pdg': species, 'dangling_particle_parents': dangling,
        'same_particle_same_sid_extra_rows': repeated, 'same_particle_same_sid_multirow_groups': multi,
        'primary_definition': 'pdg==11 and ' + ('ismcprim==1' if mode else 'isprim==1'),
        'ismcprim_available': bool(mode), 'units': {'eion': 'MeV', 'length': 'mm'},
        'zero_policy': 'zero exported SGS rows with complete gen identity coverage; exporter does not persist source collection presence',
        'repeat_policy': 'descriptive multiplicity only; not physical returns'}


def aggregate(values, mask):
    row = {'events': int(np.count_nonzero(mask))}
    row.update({k: (float(v[mask].sum()) if v.dtype.kind == 'f' else int(v[mask].sum())) for k, v in values.items()})
    row.update({'events_with_' + k: int(np.count_nonzero(mask & (values[k] > 0)))
                for k in ('steps', 'primary_steps', 'other_steps')})
    row['zero_step_events'] = row['events'] - row['events_with_steps']
    return row


def migrations(values, stock, fast, names):
    a, b = path_state.decode(stock)[0] == 1, path_state.decode(fast)[0] == 1
    bins = {'0': values['other_steps'] == 0, '1-9': (values['other_steps'] > 0) & (values['other_steps'] < 10),
            '10+': values['other_steps'] >= 10}
    return {name: {label: {'totals': aggregate(values, mask),
        'omitted_nonprimary_step_count': {key: aggregate(values, mask & cut) for key, cut in bins.items()}}
        for label, mask in [('both_pass', a[:, col] & b[:, col]), ('stock_only', a[:, col] & ~b[:, col]),
                            ('fast_only', ~a[:, col] & b[:, col]), ('neither_pass', ~a[:, col] & ~b[:, col])]}
        for col, name in enumerate(names)}


def analyze(tables, decision_art, comparison, config, prepared, table_receipt=None):
    tables = list(map(Path, tables))
    art, comparison, config, prepared = map(Path, (decision_art, comparison, config, prepared))
    if not tables or len(set(p.resolve() for p in tables)) != len(tables):
        raise ValueError('need unique original source tables')
    paths = [art, comparison, config, prepared / 'manifest.json', prepared / 'replay.fcl', *tables]
    if table_receipt is not None:
        paths.append(Path(table_receipt))
    before = integrity.files(paths)
    flowreport = flow.analyze(art, config, prepared, comparison)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    actual = integrity.files(tables)
    evidence = manifest.get('injection_campaign_evidence', [])
    bound = lambda rows: all(any(flow._same_bytes([r], [t]) for r in rows) for t in actual)
    if not bound(evidence):
        if table_receipt is None:
            raise ValueError('original tables require manifest hashes or --table-receipt')
        receipt = json.loads(Path(table_receipt).read_text())
        if (receipt.get('source_birth_identities_verified') is not True or receipt.get('events') != flowreport['events']
                or not bound(receipt.get('inputs', []))):
            raise ValueError('source-table receipt does not bind exact original table bytes/count')
    stock_paths = replay.paths(replay.production_config(config.read_text(), manifest['original_process']))
    names = list(flowreport['paths'])
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    with uproot.open(art, custom_classes=classes) as root:
        tree = root[f"Events;{flowreport['highest_cycle']}"]
        keys = event_keys.read(tree)
        stock = path_state.read_status(tree, manifest['original_process'], len(stock_paths))[0]
        fast = path_state.read_status(tree, manifest['replay_process'], len(names))[0]
    stock = stock[:, [[n for _, n, _ in stock_paths].index(name) for name in names]]
    values, details = census(tables, keys)
    report = {'events': len(keys), 'census': aggregate(values, np.ones(len(keys), bool)), **details,
        'event_keys_in_saved_order': keys.tolist(), 'per_event': {k: v.tolist() for k, v in values.items()},
        'paths': migrations(values, stock, fast, names), 'flowreport': flowreport, 'inputs': before,
        'scope': 'DIAGNOSTIC original SGS truth populations on exact saved denominator; no response conditioning',
        'limitations': ['no fitted correction, efficiency correction model, truth model or causal attribution',
            'no SEM or independent common-seed replicas; no full import promise or physical return classification',
            'zero exported SGS rows include every gen event, but absent source collection vs present-empty is not persisted; parent closure is reported, not required']}
    if before != integrity.files(paths):
        raise ValueError('input bytes changed during census')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', nargs='+', type=Path, required=True)
    for name in ('decision-art', 'comparison', 'config', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--table-receipt', type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('refuse census overwrite')
    report = analyze(args.tables, args.decision_art, args.comparison, args.config, args.prepared, args.table_receipt)
    with args.output.open('x') as target:
        target.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
