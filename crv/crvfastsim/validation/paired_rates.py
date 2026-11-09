"""Paired event-level cluster/digi/pulse rates on complete, matching CRV dump event populations."""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from .pilot import digest, tree_inventory


def event_keys(a):
    return np.stack([a[k].astype(np.int64) for k in ('run', 'subrun', 'event')], axis=1)


def records(keys):
    return np.ascontiguousarray(keys).view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()


def paired_ratio(a, b):
    """Delta-method paired ratio SE from independent event units, including zero-activity events."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired arrays must be finite, same-sized one-dimensional arrays with at least two events')
    n = len(a)
    total_a, total_b = float(a.sum()), float(b.sum())
    result = dict(events=n, total_ref=total_a, total_cand=total_b, mean_ref=float(a.mean()),
                  mean_cand=float(b.mean()), difference=total_b - total_a,
                  difference_se=float(np.sqrt(n * np.var(b - a, ddof=1))))
    if total_a <= 0:
        return dict(**result, relative_difference=None, relative_se=None, status='undefined_zero_reference')
    ratio = total_b / total_a
    return dict(**result, relative_difference=ratio - 1,
                relative_se=float(np.sqrt(n * np.var(b - ratio * a, ddof=1)) / total_a), status='descriptive')


def read(path):
    evmeta = tree_inventory(path, 'crvdump/events')
    if evmeta['rows'] != evmeta['unique_events']:
        raise ValueError(f'duplicate event rows: {path}')
    with uproot.open(path) as f:
        ev = f['crvdump/events'].arrays(['run', 'subrun', 'event', 'ndigi', 'npulse', 'ncluster'], library='np')
        keys = event_keys(ev)
        order = np.lexsort(keys.T[::-1])
        keys = keys[order]
        values = {k: np.asarray(ev[k][order], float) for k in ('ndigi', 'npulse', 'ncluster')}
        cl = f['crvdump/clusters'].arrays(['run', 'subrun', 'event', 'sectortype'], library='np')
        u, inv = np.unique(records(keys), return_inverse=True)
        pos = np.searchsorted(u, records(event_keys(cl)))
        if np.any(pos >= len(u)) or not np.array_equal(u[pos], records(event_keys(cl))):
            raise ValueError('cluster event not in event census')
        rows = np.argsort(inv)[pos]
        counts = np.bincount(rows, minlength=len(keys))
        if not np.array_equal(counts, values['ncluster']):
            raise ValueError('cluster rows disagree with per-event cluster census')
        for sector in np.unique(cl['sectortype']):
            values[f'clusters_sector_{sector}'] = np.bincount(rows[cl['sectortype'] == sector], minlength=len(keys))
    return keys, values, evmeta


def run(ref, cand, out):
    if Path(out).exists():
        raise FileExistsError(out)
    kr, vr, ir = read(ref)
    kc, vc, ic = read(cand)
    if not np.array_equal(kr, kc):
        raise ValueError('paired analysis requires identical persisted event populations, including empty events')
    zero = np.zeros(len(kr))
    metrics = {k: paired_ratio(vr.get(k, zero), vc.get(k, zero)) for k in sorted(vr.keys() | vc.keys())}
    result = dict(ref=ref, cand=cand, ref_sha256=digest(ref), cand_sha256=digest(cand),
                  ref_events=ir, cand_events=ic, metrics=metrics,
                  uncertainty='Event-paired delta-method SE; assumes independent generated events; conditional on fixed library and conditions. Not finite-library/model uncertainty or a significance-calibrated acceptance test.')
    Path(out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(metrics, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ref')
    p.add_argument('cand')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    run(a.ref, a.cand, a.out)


if __name__ == '__main__':
    main()
