"""Opt-in event-preserving DIO source from an audited stopped-muon parent table."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import target_sources as S
from .target_parent_audit import checksum


def _integers(values, name, shape=None):
    values = np.asarray(values)
    if values.dtype.kind not in 'iuf' or (shape is not None and values.shape != shape):
        raise ValueError(f'{name} must have integer values and declared shape')
    if values.dtype.kind == 'f':
        limit = min(2 ** (np.finfo(values.dtype).nmant + 1), 2 ** 63)
        if (not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any(values >= limit)):
            raise ValueError(f'{name} has fractional, nonfinite or unsafe floating identities')
    if values.size and (np.any(values < 0) or np.any(values > np.iinfo(np.int64).max)):
        raise ValueError(f'{name} must fit nonnegative int64')
    return values.astype(np.int64, copy=False)


def parent_layout(parent):
    """Require a complete eligible parent with explicit identities and multiplicities."""
    required = ('run', 'subrun', 'evt', 'trk', 'pdg', 'codestop', 'event_index',
                'event_keys', 'event_multiplicity', 'at_rest', 'foil',
                'xstop', 'ystop', 'zstop', 'tstop')
    if any(k not in parent for k in required):
        raise ValueError('audited DIO parent columns, identities and multiplicities required')
    if 'weight' in parent:
        raise ValueError('weighted parents require a separate event-normalization adapter')
    counts = np.asarray(parent['event_multiplicity'])
    if counts.ndim != 1 or not counts.size:
        raise ValueError('nonempty parent event multiplicities required')
    counts = _integers(counts, 'event_multiplicity')
    if np.any(counts < 1):
        raise ValueError('every parent event must contain eligible muons')
    keys = _integers(parent['event_keys'], 'event_keys', (counts.size, 3))
    if np.unique(keys, axis=0).shape[0] != counts.size:
        raise ValueError('duplicate parent event keys')
    n = np.asarray(parent['pdg']).size
    vectors = tuple(k for k in required if k not in ('event_keys', 'event_multiplicity'))
    if not n or any(np.shape(parent[k]) != (n,) for k in vectors):
        raise ValueError('parent columns must be nonempty equal-length vectors')
    groups = _integers(parent['event_index'], 'event_index', (n,))
    identity = np.column_stack([_integers(parent[k], k, (n,)) for k in ('run', 'subrun', 'evt', 'trk')])
    if (groups.max() >= counts.size or not np.array_equal(np.bincount(groups, minlength=counts.size), counts)
            or not np.array_equal(identity[:, :3], keys[groups])):
        raise ValueError('parent event identity/multiplicity mismatch')
    if np.unique(identity, axis=0).shape[0] != n:
        raise ValueError('duplicate parent stop identity')
    if (np.asarray(parent['at_rest']).dtype.kind != 'b' or not np.all(parent['at_rest'])
            or not np.all(np.asarray(parent['pdg']) == 13)
            or not np.all(np.asarray(parent['codestop']) == 32)):
        raise ValueError('parent must contain only stopped negative muons with process 32')
    _integers(parent['foil'], 'foil', (n,))
    if any(not np.isfinite(parent[k]).all() for k in ('xstop', 'ystop', 'zstop', 'tstop')):
        raise ValueError('finite parent positions and times required')
    return counts, groups, keys


def draw_schedule(rng, event_population, n_events, parent_events):
    """Sample declared event occurrences, retaining duplicate population exposure."""
    if (not isinstance(n_events, (int, np.integer)) or isinstance(n_events, bool) or n_events < 1):
        raise ValueError('n_events must be a positive integer')
    if (not isinstance(parent_events, (int, np.integer)) or isinstance(parent_events, bool) or parent_events < 1):
        raise ValueError('parent_events must be a positive integer')
    population = np.asarray(event_population)
    if population.ndim != 1 or not population.size:
        raise ValueError('nonempty explicit event occurrence population required')
    population = _integers(population, 'event_population')
    if np.any(population >= parent_events):
        raise ValueError('event population outside declared parent')
    return population[rng.integers(0, population.size, int(n_events))]


def dio_events(rng, parent, event_schedule, spectrum, tau=S.C.TAU_MU_AL):
    """Emit one DIO electron per eligible muon in every scheduled parent event."""
    counts, groups, keys = parent_layout(parent)
    schedule = np.asarray(event_schedule)
    if schedule.ndim != 1 or not schedule.size:
        raise ValueError('nonempty explicit event schedule required')
    schedule = _integers(schedule, 'event_schedule')
    if schedule.max() >= counts.size:
        raise ValueError('event schedule outside declared parent')
    if not np.isfinite(tau) or tau < 0:
        raise ValueError('finite nonnegative muon lifetime required')
    emitted_counts = counts[schedule]
    event_index = np.repeat(np.arange(schedule.size, dtype=np.int64), emitted_counts)
    starts = np.r_[0, np.cumsum(emitted_counts[:-1], dtype=np.int64)]
    within = np.arange(event_index.size) - starts[event_index]
    offsets = np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
    order = np.argsort(groups, kind='stable')
    idx = order[offsets[schedule[event_index]] + within]
    energy = spectrum.sample(rng, idx.size)
    if not np.isfinite(energy).all() or np.any(energy <= S.ME):
        raise ValueError('finite DIO total energies above electron mass required')
    momentum = np.sqrt(energy ** 2 - S.ME ** 2)
    direction = S.C.isotropic(rng, idx.size)
    out = dict(p=momentum, u=direction, pdg=np.full(idx.size, 11, np.int32), stop_idx=idx,
               time=np.asarray(parent['tstop'])[idx] + rng.exponential(tau, idx.size),
               event_index=event_index, parent_event_index=schedule[event_index],
               scheduled_parent_event_index=schedule.copy(), event_multiplicity=emitted_counts,
               scheduled_parent_event_keys=keys[schedule], gencount=np.array([schedule.size], np.int64))
    for axis, position, component in zip(('x', 'y', 'z'), ('xstop', 'ystop', 'zstop'), range(3)):
        out[axis] = np.asarray(parent[position])[idx]
        out['p' + axis] = momentum * direction[:, component]
    for k in ('run', 'subrun', 'evt', 'trk', 'foil'):
        out['parent_' + k] = np.asarray(parent[k])[idx]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', required=True, help='audited complete stopped-muon NPZ table')
    ap.add_argument('--population', required=True, help='NPZ event_index vector of predeclared parent event occurrences')
    ap.add_argument('--n-events', type=int, required=True)
    ap.add_argument('--schedule-seed', type=int, required=True)
    ap.add_argument('--source-seed', type=int, required=True)
    ap.add_argument('--spectrum-table', required=True)
    ap.add_argument('--elow', type=float, required=True)
    ap.add_argument('--ehi', type=float, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise FileExistsError('refusing to overwrite event source')
    if a.schedule_seed < 0 or a.source_seed < 0 or a.schedule_seed == a.source_seed:
        raise ValueError('distinct nonnegative schedule/source seeds required')
    spectrum = S.BinnedSpectrum.table(a.spectrum_table, a.elow, a.ehi)
    with np.load(a.parent, allow_pickle=False) as parent, np.load(a.population, allow_pickle=False) as population:
        counts, _, _ = parent_layout(parent)
        schedule = draw_schedule(np.random.default_rng(a.schedule_seed), population['event_index'],
                                 a.n_events, counts.size)
        out = dio_events(np.random.default_rng(a.source_seed), parent, schedule, spectrum)
    metadata = dict(process='dio', mode='explicit-event-population-all-eligible-muons',
        generated_events=int(a.n_events), generated_electrons=int(out['pdg'].size),
        schedule_seed=a.schedule_seed, source_seed=a.source_seed, parent_sha256=checksum(a.parent),
        population_sha256=checksum(a.population), spectrum_sha256=checksum(a.spectrum_table),
        elow=float(spectrum.edges[0]), ehi=float(spectrum.edges[-1]), bin_center=False,
        normalization='one electron per eligible muon; explicit generated-event and emitted-electron denominators',
        population_scope='declared empirical parent event occurrences, not absolute decay/POT normalization',
        source_validated=False, generated_yield_validated=False, certificate_established=False)
    out['metadata'] = json.dumps(metadata, allow_nan=False)
    with open(a.out, 'xb') as stream:
        np.savez_compressed(stream, **out)
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
