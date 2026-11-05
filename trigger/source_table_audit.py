"""Reconcile exported CE births with original art identities and inspect GDML straw endpoint coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

import event_keys
import integrity


def audit(sources, tables, gdml, expected, origin):
    from trkfastsim import geometry
    from trkfastsim.validation.fieldce import starts
    from mu2efastsim import tables as table_io
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError('need explicit finite production DetectorSystem origin')
    source_keys = []
    for path in sources:
        with uproot.open(path) as root:
            source_keys.append(event_keys.read(root['Events']))
    source_keys = np.concatenate(source_keys)
    births = starts(tables)
    birth_keys = births[table_io.KEY].to_numpy(dtype=np.uint64)
    ordered = lambda keys: keys[np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))]
    if (len(source_keys) != expected or len(np.unique(source_keys, axis=0)) != expected
            or not np.array_equal(ordered(source_keys), ordered(birth_keys))):
        raise ValueError('exported primary births do not cover every unique original source event')
    steps = table_io.read_many(tables, 'sgsteps')
    geom = geometry.from_gdml(gdml)
    indices = steps.upanel.to_numpy(dtype=np.int64) * geometry.NSTRAWS + steps.straw.to_numpy(dtype=np.int64)
    if (np.any(indices < 0) or np.any(indices >= geom.nstraws)
            or not np.array_equal(geom.sid[indices], steps.sid.to_numpy())):
        raise ValueError('GDML StrawId mapping differs from exported production steps')
    endpoints = {}
    for end in ('0', '1'):
        position = geometry.to_mu2e(steps[[f'x{end}', f'y{end}', f'z{end}']].to_numpy(), origin=origin)
        delta = position - geom.mid[indices]
        along = np.einsum('ij,ij->i', delta, geom.dir[indices])
        radius = np.linalg.norm(delta - along[:, None] * geom.dir[indices], axis=1)
        endpoints[end] = {'max_radius_excess_mm': float(np.max(radius - geom.radius)),
            'outside_radius_plus_1um': int(np.count_nonzero(radius > geom.radius + .001)),
            'outside_half_length_plus_1um': int(np.count_nonzero(np.abs(along) > geom.halflen[indices] + .001))}
    return {'events': expected, 'source_birth_identities_verified': True, 'steps': len(steps),
        'gdml_straw_ids_verified': True, 'detector_origin_mm': origin.tolist(), 'endpoints': endpoints,
        'all_endpoints_within_1um': all(row['outside_radius_plus_1um'] == 0 and row['outside_half_length_plus_1um'] == 0 for row in endpoints.values()),
        'inputs': integrity.files([*sources, *tables, gdml]),
        'scope': 'source truth export and straw-location diagnostic; not full GDML/material/fast transport certification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', type=Path, required=True)
    parser.add_argument('--tables', nargs='+', type=Path, required=True)
    parser.add_argument('--gdml', type=Path, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    parser.add_argument('--origin', nargs=3, type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse source table receipt overwrite')
    report = audit(args.sources, list(map(str, args.tables)), str(args.gdml), args.expected_events, args.origin)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"events={report['events']} exact exported birth coverage; steps={report['steps']} GDML IDs agree")


if __name__ == '__main__':
    main()
