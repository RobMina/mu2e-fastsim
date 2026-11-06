"""Explicit once-only frozen Run1A held-out comparison using existing stock outputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import tscampaign, tsrawreference
from .tscompare import _keys, _load, _validate, compare, plot

INDICES = tuple(range(100, 200))
SEED = 20261105
SETTINGS = dict(step_mm=2.5, material_step_mm=.25, loss_fraction=.05,
                max_path_mm=100000., method='rk4')


def validate(reference, development, census):
    """Check complete counted exposure, disjoint event identities and every retained source."""
    r, d = _load(reference), _load(development)
    _validate(r, True)
    units = np.column_stack((np.full(len(INDICES), 1430), 920000 + np.asarray(INDICES)))
    if (census['cohort'] != 'heldout' or census['indices'] != list(INDICES)
            or not census['checks_passed'] or census['finished_art_events'] != 5000 * len(INDICES)):
        raise ValueError('all counted held-out shards required; no partial or failed census')
    for key, width in (('event_id', 3), ('parent_shard_ids', 2)):
        if r[key].ndim != 2 or r[key].shape[1] != width or r[key].dtype.kind not in 'iu':
            raise ValueError('invalid held-out ' + key)
        if len(np.unique(_keys(r[key]))) != len(r[key]):
            raise ValueError('duplicate held-out ' + key)
    events, counts = np.unique(r['event_id'][:, :2], axis=0, return_counts=True)
    order = np.argsort(_keys(r['parent_shard_ids']))
    if (not np.array_equal(events, units) or not np.all(counts == 5000)
            or not np.array_equal(r['parent_shard_ids'][order], units)
            or not np.array_equal(r['parent_shard_event_counts'][order], counts)
            or len(r['event_id']) != census['finished_art_events']):
        raise ValueError('held-out reference and finished-event census disagree')
    if (np.any(r['id'][:, 0] != 1430) or np.any(~np.isin(r['shard'], units[:, 1]))
            or not np.array_equal(r['id'][:, 1], r['shard'])
            or not np.all(np.isin(_keys(r['id'][:, :3]), _keys(r['event_id'])))):
        raise ValueError('source identities outside held-out events')
    if np.intersect1d(_keys(r['event_id']), _keys(d['event_id'])).size:
        raise ValueError('held-out/development event overlap')
    if np.intersect1d(_keys(r['id']), _keys(d['id'])).size:
        raise ValueError('held-out/development source overlap')
    if (np.any(r['endpoint_ancestry_incomplete'])
            or np.any(r['stock_multiplicity'] != r['stock_reached'])
            or np.any(r['endpoint_source'] < 0)):
        raise ValueError('unresolved, multiple or unlinked DS records; report without dropping rows')
    if (len(r['endpoint_id']) != census['ds_mu_pi_rows']
            or len(r['id']) + len(r['nested_id']) != census['coll1_mu_pi_forward_tracks']):
        raise ValueError('reference records disagree with full held-out census')
    return r


def prepare(extraction, development, gdml, status_path, out):
    """Merge all100 explicitly named shards and check geometry/status, without transport."""
    extraction, out = Path(extraction), Path(out)
    if out.exists():
        raise ValueError('fresh prepared-reference directory required')
    census = tscampaign.combine([extraction / f'census-{i:03d}.json' for i in INDICES], cohort='heldout')
    r, report = tsrawreference.combine([extraction / f'reference-{i:03d}.npz' for i in INDICES],
                                       root_cohort=True, cohort='heldout')
    validate(r, development, census)
    from .tsgeometry import Geometry
    geometry = Geometry(gdml)
    coverage = dict(source_outside=int(np.sum(~geometry.inside(r['x']))),
                    endpoint_outside=int(np.sum(~geometry.inside(r['endpoint_x']))))
    if any(coverage.values()):
        raise ValueError('held-out states outside frozen geometry; report without changing geometry')
    status = tscampaign.status_histograms(status_path)
    if (status['events'] != census['finished_art_events'] or status['nonzero_status_events']
            or status['overflow_events'] or status['step_limit_events']):
        raise ValueError('stock status/count check needs review; do not discard held-out events')
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'reference.npz').open('xb') as handle:
        np.savez_compressed(handle, **r)
    report.update(geometry_check=coverage, census=census, status=status,
                  development=str(development), heldout_disjoint=True, gdml=str(gdml), status_input=str(status_path))
    with (out / 'reference.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('cohort', 'source_tracks', 'recorded_source_tracks', 'events',
        'shards', 'stock_reached', 'nested_tracks_represented_by_ancestor', 'upstream_ancestry_incomplete_sources')}), flush=True)


def run(reference, development, gdml, field_config, offline_root, maps, out):
    """Single fixed all-charge batch; do not select settings from held-out results."""
    reference, out = Path(reference), Path(out)
    if out.exists():
        raise ValueError('fresh held-out run directory required; never rerun a completed comparison')
    report = json.loads(reference.with_suffix('.json').read_text())
    r = validate(reference, development, report['census'])
    if Path(report['gdml']).resolve() != Path(gdml).resolve():
        raise ValueError('run geometry differs from prepared frozen reference')
    from . import tsfield, tsgeometry, tstransport
    field = tsfield.from_config(field_config, offline_root, maps)
    geometry = tsgeometry.Geometry(gdml)
    out.mkdir(parents=True, exist_ok=False)
    paths = dict(reference=str(reference.resolve()), development=str(Path(development).resolve()),
                 gdml=str(Path(gdml).resolve()), field_config=str(Path(field_config).resolve()),
                 offline_root=str(Path(offline_root).resolve()), maps=str(Path(maps).resolve()))
    with (out / 'run.json').open('x') as handle:
        json.dump(dict(cohort='heldout', inputs=paths, seed=SEED, settings=SETTINGS), handle, indent=2)
    f = tstransport.transport({k: r[k].copy() for k in ('x', 'p', 't', 'pdg')}, field, geometry,
                             seed=SEED, **SETTINGS)
    f.update({k: r[k] for k in ('id', 'shard', 'event_id', 'parent_shard_ids', 'parent_shard_event_counts')})
    with (out / 'fast.npz').open('xb') as handle:
        np.savez_compressed(handle, **f)
    result = compare(r, f, geometry_label='Run1A')
    result.update(label='held-out', scope='Run1A frozen-baseline held-out TS comparison; no matched TS-only stock timing')
    with (out / 'numbers.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    plot(result, out / 'ts.png', label='held-out')
    print(json.dumps(dict(ninput=len(r['id']), events=len(r['event_id']), seconds=float(f['seconds']),
        fates=result['fast_fate_counts'], charges={q: result['charges'][q]['headline'] for q in ('-1', '1')})), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    for key in ('development', 'gdml', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    for key in ('extraction', 'status', 'reference', 'field-config', 'offline-root', 'maps'):
        parser.add_argument('--' + key, type=Path)
    a = parser.parse_args(argv)
    if a.action == 'prepare':
        if a.extraction is None or a.status is None:
            parser.error('prepare requires extraction and status paths')
        prepare(a.extraction, a.development, a.gdml, a.status, a.out)
    else:
        if any(getattr(a, key) is None for key in ('reference', 'field_config', 'offline_root', 'maps')):
            parser.error('run requires prepared reference and all frozen field paths')
        run(a.reference, a.development, a.gdml, a.field_config, a.offline_root, a.maps, a.out)


if __name__ == '__main__':
    main()
