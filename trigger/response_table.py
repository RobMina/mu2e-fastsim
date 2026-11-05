"""Full-stock TABLE admission, not a lossless-import or response-runtime certificate.

Only reads inputs. Event traversal is supplied independently, never inferred from
fast handoffs. Numerical SGS rows (including repeats and all species) are neither
sorted nor filtered. An ordered payload hash binds the observed encounter order;
it cannot authenticate an earlier, unprovided version of the source tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

import event_keys
import integrity
from stock_step_census import KEYS, INTS, FLOATS, identities, integer, join

PART_INTS = ['key', 'parent', 'pdg', 'isprim', 'ismcprim', 'genidx', 'ccode', 'scode']
PART_STATE = ['sx', 'sy', 'sz', 'ex', 'ey', 'ez', 'spx', 'spy', 'spz', 'st', 'epx', 'epy', 'epz', 'et', 'se', 'ee']
PART_FLOAT32 = ['spx', 'spy', 'spz', 'st', 'epx', 'epy', 'epz', 'et']
GEN_INTS = ['idx', 'pdg', 'genid']
GEN_STATE = ['x', 'y', 'z', 'px', 'py', 'pz', 'e', 't']
I32 = np.iinfo('i4').max
U32 = np.iinfo('u4').max


def _integers(data, names, low, high=I32):
    integer(data, names)
    for name in names:
        if np.any(data[name] < low) or np.any(data[name] > high):
            raise ValueError(f'integer range violation: {name}')


def _finite(data, names, float32=()):
    for name in names:
        if (name not in data or data[name].ndim != 1 or data[name].dtype.kind not in 'iuf'
                or not np.all(np.isfinite(data[name]))):
            raise ValueError(f'invalid finite scalar field: {name}')
        if name in float32 and np.any(np.abs(data[name]) > np.finfo('f4').max):
            raise ValueError(f'imported float32 overflow: {name}')


def _ids(data):
    values = identities(data)
    if np.any(values > U32):
        raise ValueError('event identity exceeds unsigned 32-bit importer range')
    return values


def _read(root, name, required):
    if 'dump/' + name not in root or root['dump/' + name].classname != 'TTree':
        raise ValueError(f'need explicit dump/{name} TTree, including empty schema')
    tree = root['dump/' + name]
    if not set(required).issubset(tree.keys()):
        raise ValueError(f'missing {name} schema fields: {sorted(set(required) - set(tree.keys()))}')
    data = tree.arrays(required, library='np')
    if any(v.ndim != 1 or len(v) != tree.num_entries for v in data.values()):
        raise ValueError(f'non-scalar or inconsistent {name} columns')
    return data


def _ordered_hash(data):
    """Hash names, dtypes, shapes and raw arrays, in row order; never modify data."""
    sha = hashlib.sha256()
    for name in sorted(data):
        array = np.ascontiguousarray(data[name])
        sha.update(json.dumps([name, array.dtype.str, array.shape]).encode() + b'\0')
        sha.update(array.tobytes())
    return sha.hexdigest()


def _traversal(rows, population):
    index = join(rows, population)
    if np.any(np.diff(index) < 0):
        raise ValueError('table event encounter order differs from original traversal')
    return index


def _parents(particles, keys, rows):
    parent = particles['parent']
    active = parent >= 0
    if np.any(parent == particles['key']):
        raise ValueError('self-parent particle')
    n = len(keys)
    pointers = np.full(n + 1, n, dtype=np.int64)
    try:
        pointers[np.flatnonzero(active)] = join(np.column_stack([rows[active], parent[active]]), keys)
    except ValueError as exc:
        raise ValueError('dangling particle parent') from exc
    # Pointer doubling detects cycles in O(log N) vectorized passes, not event loops.
    for _ in range((n + 1).bit_length()):
        pointers = pointers[pointers]
    if np.any(pointers[:n] != n):
        raise ValueError('particle parent cycle')


def audit(tables, denominator):
    """Admit full-stock exported tables against original Nx3 event traversal keys.

    Requires the actual dumper schema, including ismcprim and explicit particle
    state. ismcprim is descriptive only: it need not equal isprim or identify a
    genealogy root. Non-primary generator links and non-generator roots are legal.
    Multi-generator events are supported with unique contiguous collection idx.
    """
    tables = [Path(p) for p in tables]
    if not tables or len({p.resolve() for p in tables}) != len(tables):
        raise ValueError('need unique original source tables')
    if (not isinstance(denominator, np.ndarray) or denominator.ndim != 2
            or denominator.shape[1] != 3 or denominator.dtype.kind not in 'iu' or not len(denominator)):
        raise ValueError('denominator must be a nonempty integer numpy Nx3 array')
    if (np.any(denominator > U32) or np.any(denominator < 0)
            or np.any(denominator[:, [0, 2]] == 0)):
        raise ValueError('invalid denominator event identities')
    denominator = denominator.astype('i8', copy=True)
    if len(np.unique(denominator, axis=0)) != len(denominator):
        raise ValueError('duplicate denominator identities')
    before = integrity.files(tables)
    n = len(denominator)
    counts = {name: np.zeros(n, dtype='i8') for name in ('gen', 'particles', 'steps')}
    sums = {name: np.zeros(n, dtype='f8') for name in ('eion', 'length')}
    species, file_reports, traversals = {}, [], []
    repeats = 0
    for path in tables:
        with uproot.open(path) as root:
            g = _read(root, 'gen', KEYS + GEN_INTS + GEN_STATE)
            p = _read(root, 'particles', KEYS + PART_INTS + PART_STATE)
            s = _read(root, 'sgsteps', KEYS + INTS + ['plane', 'panel'] + FLOATS)
            # These optional trees are also consumed by the existing importer.
            extras = {}
            for name, ints, floats in (
                ('surfsteps', ['sp', 'surf', 'sindex'], ['edep', *FLOATS[3:]]),
                ('trajpts', ['sp', 'ipt'], ['x', 'y', 'z', 't', 'ke'])):
                if 'dump/' + name in root:
                    extras[name] = (_read(root, name, KEYS + ints + floats), ints, floats)
        births, pr, sr = _ids(g), _ids(p), _ids(s)
        if not len(births):
            raise ValueError('table has no generated-event coverage')
        _, first = np.unique(births, axis=0, return_index=True)
        events = births[np.sort(first)]
        ei = _traversal(events, denominator)
        traversals.append(ei)
        gi, pi, si = (_traversal(rows, events) for rows in (births, pr, sr))
        _integers(g, ['idx', 'genid'], 0)
        _integers(g, ['pdg'], -I32 - 1)
        _finite(g, GEN_STATE)
        if np.any(g['e'] < 0):
            raise ValueError('negative generator energy')
        gc = np.bincount(gi, minlength=len(events))
        if np.any(g['idx'] != np.arange(len(births)) - np.repeat(np.cumsum(gc) - gc, gc)):
            raise ValueError('duplicate, missing or out-of-order generator idx')
        gkeys = np.column_stack([births, g['idx']])
        _integers(p, ['key'], 0)
        _integers(p, ['parent', 'genidx'], -1)
        _integers(p, ['pdg'], -I32 - 1)
        _integers(p, ['ccode', 'scode'], 0)
        _integers(p, ['isprim', 'ismcprim'], 0, 1)
        _finite(p, PART_STATE, PART_FLOAT32)
        if np.any(p['se'] < 0) or np.any(p['ee'] < 0):
            raise ValueError('negative particle energy')
        pkeys = np.column_stack([pr, p['key']])
        if len(np.unique(pkeys, axis=0)) != len(pr):
            raise ValueError('duplicate event/particle keys')
        _parents(p, pkeys, pr)
        linked = p['genidx'] >= 0
        if np.any((p['isprim'] == 1) & ~linked):
            raise ValueError('isprim requires an explicit generator link (no importer fallback)')
        lookup = join(np.column_stack([pr[linked], p['genidx'][linked]]), gkeys)
        if np.any(p['pdg'][linked] != g['pdg'][lookup]):
            raise ValueError('particle/generator PDG link mismatch')
        _integers(s, INTS + ['plane', 'panel'], 0)
        _finite(s, FLOATS, [k for k in FLOATS if k != 't'])
        if (any(np.any(s[k] < 0) for k in ('eion', 'len', 'width'))
                or np.any(s['sid'] > 65535) or np.any(s['upanel'] >= 216)
                or np.any(s['straw'] >= 96) or np.any(s['plane'] >= 36) or np.any(s['panel'] >= 6)
                or np.any(s['shape'] > 3) or np.any(s['ion'] > 2)):
            raise ValueError('SGS payload range violation')
        if (np.any(s['sid'] != (s['plane'] << 10) + (s['panel'] << 7) + s['straw'])
                or np.any(s['upanel'] != s['plane'] * 6 + s['panel'])):
            raise ValueError('inconsistent StrawId mapping')
        sc = np.bincount(si, minlength=len(events))
        if np.any(s['nsgs'] != sc[si]):
            raise ValueError('nsgs differs from event row count')
        association = join(np.column_stack([sr, s['sp']]), pkeys)
        for name, (data, ints, floats) in extras.items():
            er = _ids(data)
            _traversal(er, events)
            _integers(data, ints, 0)
            _finite(data, floats, [k for k in floats if k != 't' or name == 'trajpts'])
            join(np.column_stack([er, data['sp']]), pkeys)
            energy = 'edep' if name == 'surfsteps' else 'ke'
            if np.any(data[energy] < 0):
                raise ValueError(f'negative {name} energy')
        counts['gen'][ei] = gc
        counts['particles'][ei] = np.bincount(pi, minlength=len(events))
        counts['steps'][ei] = sc
        for name, field in [('eion', 'eion'), ('length', 'len')]:
            sums[name][ei] = np.bincount(si, weights=s[field], minlength=len(events))
        pdgs, inverse = np.unique(p['pdg'], return_inverse=True)
        pc = np.bincount(inverse, minlength=len(pdgs))
        step_species = join(p['pdg'][association, None], pdgs[:, None])
        ns = np.bincount(step_species, minlength=len(pdgs))
        energies = np.bincount(step_species, weights=s['eion'], minlength=len(pdgs))
        lengths = np.bincount(step_species, weights=s['len'], minlength=len(pdgs))
        for i, pdg in enumerate(pdgs):
            row = species.setdefault(str(int(pdg)), {'particles': 0, 'steps': 0, 'eion': 0., 'length': 0.})
            for field, value in [('particles', int(pc[i])), ('steps', int(ns[i])), ('eion', float(energies[i])), ('length', float(lengths[i]))]:
                row[field] += value
        _, multiplicity = np.unique(np.column_stack([sr, s['sp'], s['sid']]), axis=0, return_counts=True)
        repeats += int(np.maximum(multiplicity - 1, 0).sum())
        file_reports.append({'path': str(path.resolve()), 'events': len(events),
            'rows': {'gen': len(births), 'particles': len(pr), 'sgsteps': len(sr)},
            'ordered_payload_sha256': {name: _ordered_hash(data) for name, data in
                [('gen', g), ('particles', p), ('sgsteps', s), *((name, item[0]) for name, item in extras.items())]}})
    if not np.array_equal(np.concatenate(traversals), np.arange(n)):
        raise ValueError('source gen coverage/order differs from exact original denominator, including zero-row events')
    after = integrity.files(tables)
    if before != after:
        raise ValueError('table input bytes changed during admission')
    report = {'schema': 'full-stock-response-table-v1', 'admitted': True, 'events': n,
        'event_keys_in_original_order': denominator.tolist(), 'source_birth_identities_verified': True,
        'particles': int(counts['particles'].sum()), 'steps': int(counts['steps'].sum()),
        'generator_rows': int(counts['gen'].sum()), 'zero_step_events': int(np.count_nonzero(counts['steps'] == 0)),
        'per_event': {k: v.tolist() for k, v in {**counts, **sums}.items()},
        'species_pdg': species, 'same_particle_same_sid_extra_rows': repeats,
        'ismcprim_required': True, 'dangling_particle_parents': 0, 'parent_cycles': 0,
        'inputs': before, 'table_hashes_before': before, 'table_hashes_after': after, 'tables': file_reports,
        'source_collection_presence_proven': False, 'particle_productid_lossless': False,
        'particle_state_lossless': False, 'sgs_rows_filtered': 0, 'sgs_order': 'unchanged table encounter order',
        'scope': 'TABLE admission only; all species, original numeric keys, key offset zero; not campaign/runtime authorization',
        'limitations': ['Caller must bind these observed source-table sizes/SHA256 to the completed original export receipts; this API has no historical receipt input.',
            'Source collection presence is NOT proven: absent and present-empty SGS are indistinguishable in this export.',
            'Particle ProductID/state are NOT lossless: new product pointers, rebuilt daughters, unrestored constructor/process fields and restricted mass-based energies.',
            'Ordered hashes bind observed bytes/rows, not historical source order, source product identity or numerical SGS runtime round-trip equality.',
            'No built-importer/conditions/RNG/provenance, matching parent context, preserved-product or all-52 trigger verification.',
            'ismcprim is not isprim or a root test; generator links are checked numerically, not by art ProductID.',
            'ccode/scode/genid are checked as nonnegative signed-32 codes, not against a runtime enum dictionary; vdsteps/subruns are not imported or certified.']}
    # Also fail closed on overflow in floating sums; the receipt must be strict JSON.
    json.dumps(report, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', nargs='+', type=Path, required=True)
    parser.add_argument('--denominator-art', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('refuse response table receipt overwrite')
    before = integrity.files(args.denominator_art)
    rows = []
    for path in args.denominator_art:
        with uproot.open(path) as root:
            rows.append(event_keys.read(root['Events']))  # EventAuxiliary metadata only.
    report = audit(args.tables, np.concatenate(rows))
    if before != integrity.files(args.denominator_art):
        raise ValueError('denominator art bytes changed during admission')
    report['denominator_inputs'] = before
    with args.output.open('x') as target:
        target.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
