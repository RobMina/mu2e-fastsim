"""Read-only full candidate JAX finder replay of sealed retained A1/hybrid pulse tables."""
import argparse
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from . import membership_boundary as B
from . import reco_detail as R
from .pilot import digest

PROPERTIES = ['pe', 'start', 'end', 'slope', 'avgtime', 'poscalc', 'x', 'y', 'z']


def event_index(events):
    """Assign dense integer row IDs to the entire census, including empty events."""
    census = events[R.IDS].copy()
    R._integers(census, R.IDS)
    R._unique(census, R.IDS, 'event census')
    census = census.sort_values(R.IDS).reset_index(drop=True)
    return census.assign(ev=np.arange(len(census), dtype=np.int64))


def _indexed(frame, census):
    result = frame.merge(census, on=R.IDS, how='left', validate='many_to_one')
    if result.ev.isna().any():
        raise ValueError('row outside full event census')
    result['ev'] = result.ev.astype(np.int64)
    return result


def retained_tables(tables, links, census):
    c = _indexed(tables['clusters'], census).reset_index(drop=True)
    c['cid'] = np.arange(len(c), dtype=np.int64)
    m = links['clusters'].merge(c[R.KEY + ['cid']], on=R.KEY, how='left', validate='many_to_one')
    if m.cid.isna().any():
        raise ValueError('membership outside cluster table')
    return c[['ev', 'cid', 'sectortype'] + [n for n in PROPERTIES if n in c]], m[['cid', 'target']]


def _canonical(clusters, members):
    """Canonical comma-delimited integer sets; no collection-order cluster pairing."""
    c, m = clusters.copy(), members.copy()
    R._integers(c, ['ev', 'cid'])
    R._integers(c, ['sectortype'], nonnegative=False)
    R._integers(m, ['cid', 'target'])
    R._unique(c, ['cid'], 'cluster row')
    R._unique(m, ['cid', 'target'], 'member edge')
    if not m.cid.isin(c.cid).all():
        raise ValueError('member references missing cluster')
    ordered = m.sort_values(['cid', 'target'])
    tokens = ordered.assign(token=ordered.target.astype(str) + ',')
    signatures = tokens.groupby('cid', sort=False).token.sum().rename('signature')
    c = c.merge(signatures, on='cid', how='left', validate='one_to_one')
    c['signature'] = c.signature.fillna('')
    c['nlinks'] = c.cid.map(m.groupby('cid').size()).fillna(0).astype(np.int64)
    identity = ['ev', 'sectortype', 'signature']
    c['multiplicity'] = c.groupby(identity).cid.transform('size')
    c['occurrence'] = c.groupby(identity).cumcount()
    return c


def _numerical(reference, candidate, name):
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    delta = b[finite] - a[finite]
    tolerance = R.PE_ATOL + R.PE_RTOL * np.maximum(np.abs(a[finite]), np.abs(b[finite])) if name == 'pe' else 0.
    return dict(rows=len(a), exact_mismatches=int((a != b).sum()), nonfinite_pairs=int((~finite).sum()),
        beyond_tolerance=int((np.abs(delta) > tolerance).sum()),
        atol=R.PE_ATOL if name == 'pe' else 0., rtol=R.PE_RTOL if name == 'pe' else 0.,
        delta_min=float(delta.min()) if len(delta) else None,
        delta_max=float(delta.max()) if len(delta) else None,
        delta_mean=float(delta.mean()) if len(delta) else None,
        delta_max_abs=float(np.abs(delta).max()) if len(delta) else None)


def _comparison(a, ma, b, mb, events):
    identity = ['ev', 'sectortype', 'signature', 'occurrence']
    paired = a.merge(b, on=identity, how='outer', suffixes=('_ref', '_candidate'),
                     indicator=True, validate='one_to_one')
    exact = paired._merge == 'both'
    changed = paired.loc[~exact, 'ev'].to_numpy(dtype=np.int64)
    ca = np.bincount(a.ev.to_numpy(), minlength=events)
    cb = np.bincount(b.ev.to_numpy(), minlength=events)
    def occupancy(c, m):
        edges = m.merge(c[['cid', 'ev', 'sectortype']], on='cid', validate='many_to_one')
        return edges.groupby(['ev', 'sectortype', 'target']).size().rename('links').reset_index()
    oa, ob = occupancy(a, ma), occupancy(b, mb)
    occupancy_delta = oa.merge(ob, on=['ev', 'sectortype', 'target'], how='outer',
                               suffixes=('_ref', '_candidate')).fillna(0)
    delta = occupancy_delta.links_candidate - occupancy_delta.links_ref
    numeric = paired[exact & (paired.multiplicity_ref == 1) & (paired.multiplicity_candidate == 1)]
    fields = [n for n in PROPERTIES if n in a and n in b]
    return dict(events=events, both_empty_events=int(((ca == 0) & (cb == 0)).sum()),
        reference_clusters=len(a), candidate_clusters=len(b), exact_member_set_matches=int(exact.sum()),
        unmatched_reference_clusters=int((paired._merge == 'left_only').sum()),
        unmatched_candidate_clusters=int((paired._merge == 'right_only').sum()),
        cluster_count_mismatch_events=int((ca != cb).sum()),
        membership_mismatch_events=len(np.unique(changed)), membership_equal=bool(exact.all()),
        reference_member_links=len(ma), candidate_member_links=len(mb),
        removed_membership_links=int((-delta.clip(upper=0)).sum()),
        added_membership_links=int(delta.clip(lower=0).sum()),
        unmatched_reference_member_links=int(paired.loc[paired._merge == 'left_only', 'nlinks_ref'].sum()),
        unmatched_candidate_member_links=int(paired.loc[paired._merge == 'right_only', 'nlinks_candidate'].sum()),
        numerical_matched_clusters=len(numeric), ambiguous_exact_matches=int(exact.sum()) - len(numeric),
        properties={n: _numerical(numeric[n + '_ref'], numeric[n + '_candidate'], n) for n in fields})


def compare_tables(reference, candidate, events):
    """Count full-census mismatches and deltas only for identical native member sets."""
    a, ma = reference
    b, mb = candidate
    a, b = _canonical(a, ma), _canonical(b, mb)
    if (a.ev >= events).any() or (b.ev >= events).any() or events < 0:
        raise ValueError('cluster event row outside census')
    result = _comparison(a, ma, b, mb, events)
    result['sectors'] = {}
    for sector in np.union1d(a.sectortype, b.sectortype):
        sa, sb = a[a.sectortype == sector], b[b.sectortype == sector]
        result['sectors'][str(int(sector))] = _comparison(
            sa, ma[ma.cid.isin(sa.cid)], sb, mb[mb.cid.isin(sb.cid)], events)
    return result


def _integer_array(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iuf':
        raise ValueError(f'invalid finder integer array {name}')
    if values.dtype.kind == 'f' and (not np.isfinite(values).all() or
            np.any(values != np.floor(values)) or np.any(np.abs(values) >= 2**53)):
        raise ValueError(f'inexact finder integer array {name}')
    if np.any(values < 0) or np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f'invalid finder integer range {name}')
    return values.astype(np.int64)


def output_tables(out, pulses, events):
    ev = _integer_array(out['cl_event'], 'cl_event')
    c = pd.DataFrame(dict(ev=ev, cid=np.arange(len(ev), dtype=np.int64),
                          sectortype=_integer_array(out['cl_type'], 'cl_type')))
    if (ev >= events).any():
        raise ValueError('finder cluster outside census')
    for name in PROPERTIES[:6]:
        c[name] = np.asarray(out['cl_' + name])
    pos = np.asarray(out['cl_pos'])
    if pos.shape != (len(c), 3):
        raise ValueError('invalid finder position array')
    c[['x', 'y', 'z']] = pos
    m = pd.DataFrame({name: _integer_array(out[src], src) for name, src in
                      [('cid', 'mem_cl'), ('target', 'mem_idx'), ('ev', 'mem_event')]})
    if (m.cid >= len(c)).any() or not np.array_equal(m.ev, ev[m.cid.to_numpy()]):
        raise ValueError('finder membership event/cluster disagreement')
    R._unique(m, ['cid', 'target'], 'finder member edge')
    check = m.merge(pulses[['ev', 'idx']].rename(columns={'idx': 'target'}).assign(exists=True),
                    on=['ev', 'target'], how='left', validate='many_to_one')
    if check.exists.isna().any():
        raise ValueError('finder member outside input pulses')
    return c, m[['cid', 'target']]


def replay(census, pulses, G, status, J, cfg, **finder_options):
    """One batched full-finder call, with only the finder's overflow fallback permitted."""
    start = perf_counter()
    p = _indexed(pulses, census).sort_values(['ev', 'idx']).reset_index(drop=True)
    R._integers(p, ['idx', 'bar', 'sipm', 'seq'])
    R._unique(p, ['ev', 'idx'], 'pulse index')
    if ((p.bar >= len(G.sector)).any() or (p.sipm >= 4).any() or
            not np.isfinite(p[['time', 'pe']].to_numpy()).all()):
        raise ValueError('invalid persisted pulse input')
    timers = {}
    columns = [p[n].to_numpy() for n in ('ev', 'idx', 'bar', 'sipm', 'time', 'pe', 'seq')]
    launched = perf_counter()
    handle = J.find_async(*columns, G, status, cfg, with_pos=True, timers=timers, **finder_options)
    out = J.finish(handle, fallback=True)
    finished = perf_counter()
    tables = output_tables(out, p, len(census))
    fallback = _integer_array(out['fallback_events'], 'fallback_events')
    if (fallback >= len(census)).any() or len(np.unique(fallback)) != len(fallback):
        raise ValueError('invalid fallback event census')
    metrics = dict(events=len(census), pulse_rows=len(p), empty_pulse_events=len(census) - p.ev.nunique(),
        finder_seconds=finished - launched, total_seconds=perf_counter() - start,
        fallback_events=len(fallback), fallback_pulses=int(p.ev.isin(fallback).sum()),
        fallback_policy='candidate coinc.py only for JAX tier overflow',
        timers={k: float(v) for k, v in timers.items()})
    return tables, metrics


def replay_pair(ref, hybrid, expected, G, status, J, cfg):
    start = perf_counter()
    a, la, ma = R._read(ref)
    b, lb, mb = R._read(hybrid)
    census = event_index(a['events'])
    other = event_index(b['events'])
    if len(census) != expected or not census.equals(other):
        raise ValueError('full retained event census differs')
    pulse_identity = R._join(a['pulses'][R.KEY + R.PULSE_ID], b['pulses'][R.KEY + R.PULSE_ID], R.KEY)
    if ((pulse_identity._merge != 'both').any() or
            any((pulse_identity[n + '_a1'] != pulse_identity[n + '_hybrid']).any() for n in R.PULSE_ID)):
        raise ValueError('retained native pulse identities differ')
    if (R._join(la['pulses'], lb['pulses'], R.KEY + ['target'])._merge != 'both').any():
        raise ValueError('retained waveform membership differs')
    retained = dict(a1=retained_tables(a, la, census), hybrid=retained_tables(b, lb, census))
    results = {}
    for label, tables in [('a1_persisted_pulses', a), ('hybrid_persisted_pulses', b)]:
        runs, controls = {}, {}
        for enabled in (False, True):
            control = 'enabled' if enabled else 'disabled'
            runs[control], metrics = replay(census, tables['pulses'], G, status, J,
                                            replace(cfg, add_opposite_side_pulses=enabled))
            controls[control] = dict(runtime=metrics, comparisons={
                name: compare_tables(value, runs[control], len(census)) for name, value in retained.items()})
        results[label] = dict(controls=controls,
            enabled_vs_disabled=compare_tables(runs['disabled'], runs['enabled'], len(census)))
    return dict(events=len(census), tree_cycles=dict(a1=ma, hybrid=mb),
        retained_a1_vs_hybrid=compare_tables(retained['a1'], retained['hybrid'], len(census)),
        replay=results, seconds=perf_counter() - start)


def attachment_summary(results):
    summary = {}
    for arm in ('a1_persisted_pulses', 'hybrid_persisted_pulses'):
        comparisons = [r['replay'][arm]['enabled_vs_disabled'] for r in results.values()]
        sectors = sorted({s for c in comparisons for s in c['sectors']}, key=int)
        fields = ('added_membership_links', 'removed_membership_links', 'membership_mismatch_events',
                  'cluster_count_mismatch_events')
        summary[arm] = {name: sum(c[name] for c in comparisons) for name in fields}
        summary[arm]['sectors'] = {s: {name: sum(c['sectors'].get(s, {}).get(name, 0)
            for c in comparisons) for name in fields} for s in sectors}
    return summary


def _inventory(roots):
    return {str(p.resolve()): digest(p) for root in roots for p in sorted(root.rglob('*.py'))}


def _one(watched, suffix):
    paths = [p for p in watched if p.endswith(suffix)]
    if len(paths) != 1:
        raise ValueError(f'expected one frozen binding for {suffix}: {paths}')
    return str(Path(paths[0]).resolve())


def _output_guard(out, attribution, watched, code_roots):
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if out.suffix.lower() != '.json':
        raise ValueError('output must be a fresh JSON file')
    resolved = out.resolve()
    roots = [Path(attribution).resolve(), *code_roots]
    for path in watched:
        p = Path(path).resolve()
        if p.suffix == '.py':
            roots.extend(parent.parent for parent in p.parents if parent.name in ('crvfastsim', 'valhist', 'mu2efastsim'))
    if resolved in {Path(p).resolve() for p in watched} or any(resolved.is_relative_to(root) for root in roots):
        raise ValueError('output must be outside attribution, inputs, and code trees')


def _check_imports(candidate, core, watched):
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        package = name.split('.')[0]
        if package not in ('crvfastsim', 'valhist', 'mu2efastsim') or not getattr(module, '__file__', None):
            continue
        path = Path(module.__file__).resolve()
        root = core if package == 'mu2efastsim' else candidate
        if not path.is_relative_to(root / package) or str(path) not in watched:
            raise ValueError(f'unbound or wrong-tree import: {name}: {path}')
        loaded[name] = str(path)
    return loaded


def _runtime(report, watched, frozen_bindings, candidate, core, backend):
    sys.path[:0] = [str(candidate), str(core)]
    sys.dont_write_bytecode = True
    importlib.import_module('mu2efastsim')
    C = importlib.import_module('crvfastsim.reco.coinc')
    J = importlib.import_module('crvfastsim.reco.coinc_jax')
    _check_imports(candidate, core, watched)
    geometry_path = _one(frozen_bindings, '/crvfastsim/geometry.py')
    coinc_path = _one(frozen_bindings, '/crvfastsim/reco/coinc.py')
    geometry = B.load_module(geometry_path, '_membership_replay_frozen_geometry')
    frozen = B.load_module(coinc_path, '_membership_replay_frozen_coinc')
    counters = _one(frozen_bindings, '/crv_counters_extracted_v04.txt')
    calibration = str(Path(report['conditions']['calibration']).resolve())
    if calibration not in {str(Path(p).resolve()) for p in frozen_bindings}:
        raise ValueError('calibration is not frozen by attribution')
    geom = geometry.from_offline_config(counters)
    G = frozen.sector_geom(geom, geometry.parse_simpleconfig(Path(counters).read_text()))
    data = np.loadtxt(calibration, delimiter=',', comments='#', ndmin=2)
    if data.shape[1] < 6 or not np.isfinite(data[:, [0, 5]]).all():
        raise ValueError('invalid calibration channel/status columns')
    channels, statuses = data[:, 0].astype(np.int64), data[:, 5].astype(np.int64)
    if (not np.array_equal(channels, data[:, 0]) or not np.array_equal(statuses, data[:, 5]) or
            np.any(statuses < 0) or not np.array_equal(np.sort(channels), np.arange(4 * geom.nbars))):
        raise ValueError('incomplete or invalid calibration channels/status')
    status = np.empty(4 * geom.nbars, dtype=np.int64)
    status[channels] = statuses
    cfg = C.FinderConfig()
    if cfg.add_opposite_side_pulses is not True:
        raise ValueError('candidate default attachment must be enabled')
    device = J.jax.devices(backend)[0]
    provenance = dict(backend=backend, device=str(device), jax_version=J.jax.__version__,
        numpy_version=np.__version__, pandas_version=pd.__version__, finder_config=asdict(cfg),
        frozen_imports=dict(geometry=geometry_path, coinc=coinc_path, core=str(core)),
        counters=counters, calibration=calibration, tiers=J.TIERS)
    return G, status, J, cfg, device, provenance


def create(attribution, out, backend='cpu', core=None):
    """Verify frozen inputs and candidate source inventory twice; exclusively publish NEW.json."""
    start = perf_counter()
    out = Path(out).absolute()
    candidate = Path(__file__).resolve().parents[1]
    core = Path(core).resolve() if core else candidate.parent / 'core-thread-b-dqm-20261025'
    _output_guard(out, attribution, {}, [candidate, core])
    report, pairs, watched = R._validate(attribution)
    _output_guard(out, attribution, watched, [candidate, core])
    frozen_bindings = dict(watched)
    roots = [candidate / 'crvfastsim', candidate / 'valhist', core / 'mu2efastsim']
    if any(not root.is_dir() for root in roots):
        raise ValueError('candidate/frozen core package root missing')
    sources = _inventory(roots)
    for path, sha in sources.items():
        R._bind(watched, path, sha)
    R._check(watched)
    G, status, J, cfg, device, provenance = _runtime(report, watched, frozen_bindings, candidate, core, backend)
    _check_imports(candidate, core, watched)
    results = {}
    with J.jax.default_device(device):
        for shard, ref, hybrid, expected in pairs:
            results[str(shard)] = dict(ref=ref, hybrid=hybrid,
                **replay_pair(ref, hybrid, expected, G, status, J, cfg))
            print(f'shard {shard}: full retained-input finder replay complete', flush=True)
    loaded = _check_imports(candidate, core, watched)
    if _inventory(roots) != sources:
        raise ValueError('candidate/core Python inventory changed during replay')
    R._check(watched)
    result = dict(schema_version=1, diagnostic='retained-input full candidate JAX finder',
        attribution=str(Path(attribution).resolve()), shards=report['shards'], per_shard=results,
        disabled_hybrid_memberships_close=all(r['replay']['hybrid_persisted_pulses']['controls']['disabled']
            ['comparisons']['hybrid']['membership_equal'] for r in results.values()),
        attachment_effect=attachment_summary(results),
        runtime_seconds=perf_counter() - start, provenance=provenance, imported_python=loaded,
        input_sha256=watched, frozen_attribution_sha256=frozen_bindings,
        analysis_source_sha256=sources, hashes_verified_before_and_after=True,
        limitations=[
            'Persisted pulses only: no waveform refit, response replay, transport, or retained input modification.',
            'Full candidate finder on both pulse arms, with attachment enabled and disabled; not seed-only attachment.',
            'Exact event/sector/native member sets define cluster identity; duplicate identities withhold numerical pairing.',
            'Added/removed membership links count event/sector/pulse occupancy; regrouping is captured by unmatched clusters.',
            'All census events, including empty events, enter mismatch counts; no packed event keys or truth matching.',
            'Numerical deltas require exact member sets; persisted float serialization and finder arithmetic can differ.',
            'Only tier-overflow events use candidate CPU fallback; no full scalar retained-sample equivalence replay.',
            'Conditional retained-sample diagnostic, not hardcoded closure acceptance or downstream systematic assignment.',
            'Historical retained-attribution counters binding limitation remains.'])
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    _output_guard(out, attribution, watched, [candidate, core])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        stream.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attribution', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--backend', default='cpu', choices=('cpu', 'gpu', 'tpu'))
    parser.add_argument('--core', help='explicit frozen core checkout (default: sibling core-thread-b-dqm-20261025)')
    args = parser.parse_args(argv)
    try:
        create(args.attribution, args.out, args.backend, args.core)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.error(str(exc))
    print(f'Completed read-only full finder replay: {args.out}')


if __name__ == '__main__':
    main()
