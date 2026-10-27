"""Verify compiled Cat censuses and locate generator/packed-identity denominator losses."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import uproot

from .catsteps_v2 import convert, PoolV2, IDS
from .replay import sha256
from .source_provenance import sam_summary


def census_stats(events, gen, steps=None):
    """Keep full event tuples and expose every legacy packed-key collision."""
    if events.duplicated(IDS).any():
        raise ValueError('duplicate native event IDs')
    for rows, field in ((gen, 'ngen'), (steps, 'nsteps')):
        if rows is None:
            if not (events.nsteps == -1).all():
                raise ValueError('missing steps without disabled-step sentinel')
            continue
        if not pd.MultiIndex.from_frame(rows[IDS]).isin(pd.MultiIndex.from_frame(events[IDS])).all():
            raise ValueError('row event absent from census')
        counts = rows.groupby(IDS).size().reindex(pd.MultiIndex.from_frame(events[IDS]), fill_value=0).to_numpy()
        if not np.array_equal(counts, events[field]):
            raise ValueError(f'{field} census/row mismatch')
    gids = gen[IDS].drop_duplicates()
    packed = (gids.subrun.to_numpy(dtype=np.int64) << 20) | gids.event.to_numpy(dtype=np.int64)
    collision = gids.assign(packed=packed)
    collision = collision[collision.duplicated('packed', keep=False)].sort_values(['packed', *IDS])
    zero = events.loc[events.ngen == 0, IDS]
    return {'events': len(events), 'generator_rows': len(gen), 'generator_events': len(gids),
            'empty_generator_events': len(zero), 'empty_generator_ids': zero.to_dict(orient='list'),
            'step_rows': len(steps) if steps is not None else None,
            'stepped_events': int((events.nsteps > 0).sum()),
            'legacy_packed_generator_events': len(np.unique(packed)),
            'packed_collisions_lost': len(gids) - len(np.unique(packed)),
            'collision_ids': collision.to_dict(orient='list'),
            'native_ids_sha256': hashlib.sha256(events[IDS].sort_values(IDS).to_numpy(dtype='<i8').tobytes()).hexdigest()}


def run(probe, out, sam=None, export=False):
    probe, out = Path(probe), Path(out)
    if out.exists():
        raise ValueError('choose new report')
    path = probe / 'dump.root'
    with uproot.open(path) as root:
        events = pd.DataFrame(root['dump/events'].arrays(library='np'))
        gen = pd.DataFrame(root['dump/gen'].arrays(IDS, library='np'))
        steps = pd.DataFrame(root['dump/steps'].arrays(IDS, library='np')) if 'dump/steps' in root else None
    result = census_stats(events, gen, steps)
    match = re.search(r'TrigReport Events total = (\d+)', (probe / 'job.log').read_text())
    if not match or int(match.group(1)) != len(events):
        raise ValueError('completed event count disagrees with job log')
    result.update(dump_sha256=sha256(path), log_sha256=sha256(probe / 'job.log'), code_sha256=sha256(__file__))
    if sam:
        meta = sam_summary(sam)
        result['catalog'] = meta
        result['full_source_count_matches'] = meta['event_count'] == len(events)
    if export:
        if not sam or not result['full_source_count_matches']:
            raise ValueError('v2 export requires full-source catalog census')
        code = Path(__file__).resolve().parents[2]
        source = (probe / 'source.txt').read_text().strip()
        extraction = {'id': str(probe.resolve()), 'command': (probe / 'command.txt').read_text().strip(),
                      'software_version': 'user-built CrvFastSim satellite; exact source/binary hashes retained in probe',
                      'fcl_sha256': sha256(probe / 'effective.fcl'),
                      'dumper_sha256': sha256(code / 'art/CrvFastSim/src/CrvStepTableDumper_module.cc'),
                      'write_steps': True, 'source_count': 1, 'event_selection': 'all'}
        target = out.with_suffix('.parquet')
        metadata = convert(path, 'NeutralsFlashCat', target,
                           source_art={'identity': 'sam:mu2e:' + meta['file_id'], 'uri': source, 'sha256': meta['source_sha256']},
                           extraction=extraction)
        pool = PoolV2([target])
        result['v2'] = {'frames': pool.frames, 'stepped': pool.stepped, 'steps': len(pool.steps),
                        'metadata': metadata, 'path': str(target), 'sha256': sha256(target)}
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'v2'}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sam')
    parser.add_argument('--export-v2', action='store_true')
    args = parser.parse_args(argv)
    run(args.probe, args.out, args.sam, args.export_v2)


if __name__ == '__main__':
    main()
