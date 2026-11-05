"""Exact RPC birth-overlap census; diagnostic only, no approximate matching or admitted exposure."""
import argparse
import json

import numpy as np


def overlap(parent, reference):
    dtype = np.dtype([(k, '<f4') for k in ('x', 'y', 'z')])
    xyz = np.column_stack([parent[k] for k in ('xstop', 'ystop', 'zstop')]).astype('<f4')
    query = np.column_stack([reference[k] for k in ('x', 'y', 'z')]).astype('<f4')
    keys = np.ascontiguousarray(xyz).view(dtype).ravel()
    q = np.ascontiguousarray(query).view(dtype).ravel()
    keys = np.sort(keys)
    counts = np.searchsorted(keys, q, 'right') - np.searchsorted(keys, q, 'left')
    return dict(parent_stops=len(xyz), reference_rows=len(query), exact_unique=int((counts == 1).sum()),
                unmatched=int((counts == 0).sum()), ambiguous=int((counts > 1).sum()),
                parent_ranges=np.column_stack([xyz.min(axis=0), xyz.max(axis=0)]).tolist(),
                birth_ranges=np.column_stack([query.min(axis=0), query.max(axis=0)]).tolist(),
                exact_component_matches={k: int(np.isin(query[:, j], xyz[:, j]).sum()) for j, k in enumerate(('x', 'y', 'z'))})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', action='append', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    report = dict(mode='exact-birth-diagnostic-no-exposure', comparisons={})
    with np.load(a.reference) as ref:
        for i in np.unique(ref['ifile']):
            subset = {k: ref[k][ref['ifile'] == i] for k in ('x', 'y', 'z')}
            report['comparisons'][str(i)] = {}
            for path in a.parent:
                with np.load(path) as parent:
                    report['comparisons'][str(i)][path] = overlap(parent, subset)
    with open(a.out, 'w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
