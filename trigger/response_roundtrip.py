"""Exact numerical SGS round trip, not an art ProductID or lossless-particle audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

import event_keys
import integrity

KEYS = ('run', 'subrun', 'event')
SGS_INTS = ('nsgs', 'sid', 'plane', 'panel', 'straw', 'upanel', 'shape', 'ion', 'sp')
SGS_FLOATS = ('eion', 'len', 'width', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'px', 'py', 'pz')
SGS_SCHEMA = {**dict.fromkeys(KEYS, 'u4'), **dict.fromkeys(SGS_INTS, 'i4'),
              **dict.fromkeys(SGS_FLOATS, 'f4'), 't': 'f8'}
GEN_SCHEMA = {**dict.fromkeys(KEYS, 'u4'), **dict.fromkeys(('idx', 'pdg', 'genid'), 'i4'),
              **dict.fromkeys(('x', 'y', 'z', 'px', 'py', 'pz', 'e', 't'), 'f8')}
PARTICLE_SCHEMA = {**dict.fromkeys(KEYS, 'u4'),
    **dict.fromkeys(('key', 'parent', 'pdg', 'ccode', 'scode', 'isprim', 'ismcprim', 'genidx'), 'i4'),
    **dict.fromkeys(('sx', 'sy', 'sz', 'ex', 'ey', 'ez'), 'f8'),
    **dict.fromkeys(('spx', 'spy', 'spz', 'se', 'epx', 'epy', 'epz', 'ee', 'st', 'et'), 'f4')}


def _records(rows):
    rows = np.ascontiguousarray(rows, dtype='<i8')
    return rows.view(np.dtype([(str(i), '<i8') for i in range(rows.shape[1])])).ravel()


def _identities(data):
    rows = np.column_stack([data[k] for k in KEYS]).astype('i8')
    if np.any(rows < 0) or np.any(rows[:, [0, 2]] == 0):
        raise ValueError('invalid event identities')
    return rows


def _denominator(rows):
    if (not isinstance(rows, np.ndarray) or rows.ndim != 2 or rows.shape[1] != 3
            or rows.dtype.kind not in 'iu' or np.any(rows > np.iinfo('u4').max)):
        raise ValueError('denominator must be an integer Nx3 event identity array')
    rows = rows.astype('i8')
    if (np.any(rows < 0) or np.any(rows[:, [0, 2]] == 0)
            or len(np.unique(rows, axis=0)) != len(rows)):
        raise ValueError('invalid or duplicate denominator identities')
    return rows


def _read(root, name, schema, optional=False):
    """Uproot resolves the highest cycle; do not silently fall back to an older tree."""
    path = 'dump/' + name
    if path not in root:
        if optional:
            return None, None
        raise ValueError(f'missing {path} TTree')
    tree = root[path]
    if tree.classname != 'TTree':
        raise ValueError(f'{path} must be a TTree')
    data = tree.arrays(library='np')
    required = set(KEYS) | {'key', 'pdg', 'parent'} if optional else set(schema)
    if not required <= set(data) or not set(data) <= set(schema):
        raise ValueError(f'unsupported {path} schema')
    for field, values in data.items():
        dtype = np.dtype(schema[field])
        if (values.ndim != 1 or len(values) != tree.num_entries
                or (values.dtype.kind, values.dtype.itemsize) != (dtype.kind, dtype.itemsize)):
            raise ValueError(f'invalid {path} dtype/shape: {field}; expected {dtype}')
        if not np.all(np.isfinite(values)):
            raise ValueError(f'nonfinite {path} payload: {field}')
    _identities(data)
    return data, {'tree': tree.object_path, 'entries': int(tree.num_entries),
                  'schema': {k: np.dtype(schema[k]).name for k in data}}


def _contains(rows, population, label):
    if np.any(~np.isin(_records(rows), _records(population))):
        raise ValueError(f'{label}: foreign identity/reference')


def _validate(gen, steps, particles):
    births = _identities(gen)
    gen_keys = np.column_stack([births, gen['idx']])
    if np.any(gen['idx'] < 0) or len(np.unique(gen_keys, axis=0)) != len(births):
        raise ValueError('duplicate or invalid gen index')
    coverage = np.unique(births, axis=0)
    sk = _identities(steps)
    _contains(sk, coverage, 'SGS event')
    if (any(np.any(steps[k] < 0) for k in ('nsgs', 'sp', 'sid', 'eion', 'len', 'width'))
            or np.any(steps['sid'] > 65535)
            or np.any(~np.isin(steps['shape'], [0, 1, 2, 3]))
            or np.any(~np.isin(steps['ion'], [0, 1, 2]))):
        raise ValueError('SGS payload range violation')
    plane, panel, straw = steps['sid'] >> 10, (steps['sid'] >> 7) & 7, steps['sid'] & 127
    if (np.any(plane >= 36) or np.any(panel >= 6) or np.any(straw >= 96)
            or np.any(steps['plane'] != plane) or np.any(steps['panel'] != panel)
            or np.any(steps['straw'] != straw) or np.any(steps['upanel'] != plane * 6 + panel)):
        raise ValueError('inconsistent StrawId mapping')
    _, inverse, counts = np.unique(sk, axis=0, return_inverse=True, return_counts=True)
    if np.any(steps['nsgs'] != counts[inverse]):
        raise ValueError('nsgs differs from per-event row count')
    if particles is not None:
        pk = _identities(particles)
        _contains(pk, coverage, 'particle event')
        keys = np.column_stack([pk, particles['key']])
        if np.any(particles['key'] < 0) or len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError('duplicate or invalid particle key')
        if np.any(particles['parent'] < -1):
            raise ValueError('invalid particle parent')
        _contains(np.column_stack([sk, steps['sp']]), keys, 'SGS particle')
    return coverage


def _load(path):
    with uproot.open(path) as root:
        gen, gmeta = _read(root, 'gen', GEN_SCHEMA)
        steps, smeta = _read(root, 'sgsteps', SGS_SCHEMA)
        particles, pmeta = _read(root, 'particles', PARTICLE_SCHEMA, optional=True)
    coverage = _validate(gen, steps, particles)
    return {'gen': gen, 'sgsteps': steps, 'particles': particles, 'coverage': coverage,
            'trees': {'gen': gmeta, 'sgsteps': smeta, 'particles': pmeta}}


def _ordinal_keys(data):
    """Stable within-event rank, including interleaved events and repeated SGS rows."""
    rows = _identities(data)
    _, inverse, counts = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    ordinal = np.empty(len(rows), dtype='i8')
    ordinal[order] = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.column_stack([rows, ordinal])


def _comparison(original, imported, akeys, bkeys):
    _, ai, bi = np.intersect1d(_records(akeys), _records(bkeys), return_indices=True)
    fields = {}
    for name in sorted(set(original) & set(imported)):
        a, b = original[name][ai], imported[name][bi]
        fields[name] = {'numeric_mismatches': int(np.count_nonzero(a != b)),
                        'dtype': a.dtype.name}
        if a.dtype.kind == 'f':
            unsigned = np.dtype(f'u{a.dtype.itemsize}')
            fields[name]['bit_mismatches'] = int(np.count_nonzero(a.view(unsigned) != b.view(unsigned)))
    missing, extra = len(akeys) - len(ai), len(bkeys) - len(bi)
    return {'equal': not (missing or extra or any(v['numeric_mismatches'] for v in fields.values())),
            'original_rows': len(akeys), 'imported_rows': len(bkeys), 'matched_rows': len(ai),
            'missing_rows': missing, 'extra_rows': extra, 'fields': fields,
            'original_only_fields': sorted(set(original) - set(imported)),
            'imported_only_fields': sorted(set(imported) - set(original))}


def _coverage(rows, denominator):
    a, b = _records(rows), _records(denominator)
    return {'missing_events': int(np.count_nonzero(~np.isin(b, a))),
            'extra_events': int(np.count_nonzero(~np.isin(a, b)))}


def _concatenate(parts, name):
    data = [part[name] for part in parts]
    if any(d is None for d in data):
        return None
    if any(set(d) != set(data[0]) for d in data):
        raise ValueError(f'inconsistent original {name} schemas')
    return {k: np.concatenate([d[k] for d in data]) for k in data[0]}


def compare(original_tables, imported_table, denominator):
    """Compare frozen tables exactly; validation failures raise, valid differences report false.

    Source gen coverage must equal the independent denominator (unique events, not
    gen row count). Duplicate (event,gen index), including across files, is invalid.
    Particles are optional diagnostics and NEVER a lossless-admission gate.
    """
    denominator = _denominator(denominator)
    originals = list(map(Path, original_tables))
    imported_table = Path(imported_table)
    paths = [*originals, imported_table]
    if not originals or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('need distinct original tables and one separate imported table')
    before = integrity.files(paths)
    parts = [_load(p) for p in originals]
    original = {name: _concatenate(parts, name) for name in ('gen', 'sgsteps', 'particles')}
    # Source files are independent event partitions, not fragments of the same event.
    source_events = np.concatenate([p['coverage'] for p in parts])
    if len(np.unique(source_events, axis=0)) != len(source_events):
        raise ValueError('duplicate original gen event coverage across tables')
    if any(_coverage(source_events, denominator).values()):
        raise ValueError('original gen coverage differs from denominator, including zero-row events')
    imported = _load(imported_table)
    coverage = _coverage(imported['coverage'], denominator)
    sgs = _comparison(original['sgsteps'], imported['sgsteps'],
                      _ordinal_keys(original['sgsteps']), _ordinal_keys(imported['sgsteps']))
    gen = _comparison(original['gen'], imported['gen'],
        np.column_stack([_identities(original['gen']), original['gen']['idx']]),
        np.column_stack([_identities(imported['gen']), imported['gen']['idx']]))
    particles = {'available': False, 'gating': False}
    a, b = original['particles'], imported['particles']
    if a is not None and b is not None:
        particles = {'available': True, 'gating': False, **_comparison(a, b,
            np.column_stack([_identities(a), a['key']]), np.column_stack([_identities(b), b['key']]))}
    zero_original = len(source_events) - len(np.unique(_identities(original['sgsteps']), axis=0))
    zero_imported = len(imported['coverage']) - len(np.unique(_identities(imported['sgsteps']), axis=0))
    after = integrity.files(paths)
    if before != after:
        raise ValueError('input bytes changed during round-trip comparison')
    return {'equal': sgs['equal'] and gen['equal'] and not any(coverage.values()),
        'events': len(denominator), 'sgs': sgs, 'gen': gen, 'particles': particles,
        'imported_coverage': coverage, 'zero_step_events': {'original': zero_original, 'imported': zero_imported},
        'inputs_before': before, 'inputs_after': after,
        'trees': {'original': [p['trees'] for p in parts], 'imported': imported['trees']},
        'alignment': '(run, subrun, event, within-event SGS encounter ordinal); no straw/particle sort or deduplication',
        'scope': 'exact dtype-aware numerical SGS and generated-particle comparison only',
        'claims': {'original_ProductID_preserved': False, 'SimParticle_lossless': False, 'electronics_only': False},
        'limitations': [
            'No original art ProductID identity or provenance/consumed-product validation.',
            'SimParticle reconstruction is deliberately lossy: codes and energy need not agree; particle differences are diagnostic only.',
            'ismcprim is recomputed from the configured original PrimaryParticle pointer keys; equality is not proof of imported primary identity.',
            'Zero exported SGS rows do not distinguish absent source collection from present-empty; collection presence requires independent evidence.',
            'Numerical SGS agreement does not establish an electronics-only control, executable compatibility, or lossless SimParticle round trip.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-tables', nargs='+', type=Path, required=True)
    for name in ('imported-table', 'denominator-art', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('refuse round-trip report overwrite')
    paths = [*args.original_tables, args.imported_table, args.denominator_art]
    before = integrity.files(paths)
    with uproot.open(args.denominator_art) as root:
        tree = root['Events']  # highest cycle, as for the dump TTrees
        if tree.classname != 'TTree':
            raise ValueError('denominator Events must be a TTree')
        denominator = event_keys.read(tree)
        denominator_tree = tree.object_path
    report = compare(args.original_tables, args.imported_table, denominator)
    after = integrity.files(paths)
    if before != after:
        raise ValueError('input bytes changed during round-trip CLI')
    report.update(inputs_before=before, inputs_after=after, denominator_tree=denominator_tree)
    with args.output.open('x') as target:
        target.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not report['equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
