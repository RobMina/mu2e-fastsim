"""Conditional event-paired cluster shapes from complete crvdump event censuses.

Explicit edges only: these are dump observables, not claims of DQM histogram closure.
See docs/valhist-event-shapes.md for the estimator and its limitations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
from scipy import sparse
import uproot

from .pilot import digest


# Validated against art/CrvFastSim/src/CrvRecoTableDumper_module.cc.
OBSERVABLES = {
    'pe': ('pe', 'CrvCoincidenceCluster::GetPEs(); stored float'),
    'npulses': ('pulses', 'size of CrvCoincidenceCluster::GetCrvRecoPulses()'),
    'start': ('start', 'CrvCoincidenceCluster::GetStartTime()'),
    'duration': ('start,end', 'GetEndTime() - GetStartTime()'),
    'x': ('x', 'CrvCoincidenceCluster::GetAvgHitPos().x(); stored float'),
    'y': ('y', 'CrvCoincidenceCluster::GetAvgHitPos().y(); stored float'),
    'z': ('z', 'CrvCoincidenceCluster::GetAvgHitPos().z(); stored float'),
    'sectortype': ('sectortype', 'CrvCoincidenceCluster::GetCrvSectorType()'),
}
IDS = ('run', 'subrun', 'event')
KEY_DTYPE = np.dtype([(k, '<i8') for k in IDS])
MAX_BINS = 256  # Dense covariance is bins x bins, never events x bins.
MAX_REPLICAS = 10000
QUANTILES = np.array([0.025, 0.5, 0.9, 0.95, 0.975])
LIMITATIONS = (
    'Conditional independent-generated-event diagnostics on paired persisted censuses. '
    'Completeness relative to the generator must be established externally; equally missing events '
    'cannot be detected from two dumps. Fixed secondary library and conditions: no finite-library, '
    'calibration, transport or model uncertainty. No p-value, physics acceptance, simultaneous '
    'coverage or cross-observable covariance. Multiplier quantiles are not certified confidence intervals.'
)


def validate_edges(features):
    """No default geometry, tolerant edge matching, rebinning or DQM inference."""
    if not isinstance(features, dict) or not features:
        raise ValueError('edges JSON must be a nonempty object: observable -> exact edges')
    result = {}
    for name, values in features.items():
        if name not in OBSERVABLES:
            raise ValueError(f'unsupported observable {name!r}; choose {sorted(OBSERVABLES)}')
        edges = np.asarray(values, dtype=np.float64)
        if (edges.ndim != 1 or not 2 <= len(edges) <= MAX_BINS + 1
                or not np.isfinite(edges).all() or not np.all(edges[1:] > edges[:-1])):
            raise ValueError(f'{name}: require 1..{MAX_BINS} bins with finite strictly increasing edges')
        result[name] = edges
    return result


def _integer(values, name):
    a = np.asarray(values)
    if (a.ndim != 1 or a.dtype.kind not in 'iu' or np.any(a < 0)
            or np.any(a > np.iinfo(np.int64).max)):
        raise ValueError(f'{name}: expected nonnegative int64-compatible integer column')
    return a.astype(np.int64, copy=False)


def _keys(arrays):
    keys = np.empty(len(arrays['event']), dtype=KEY_DTYPE)
    for name in IDS:
        keys[name] = _integer(arrays[name], name)
    return keys


def _highest(root, name):
    cycles = [int(k.rsplit(';', 1)[1]) for k in root.keys(recursive=True, cycle=True)
              if k.rsplit(';', 1)[0] == name]
    if not cycles:
        raise ValueError(f'missing {name}')
    key = f'{name};{max(cycles)}'
    tree = root[key]
    if tree.classname != 'TTree':
        raise ValueError(f'{key}: highest cycle is not a TTree')
    return tree, dict(key=key, cycles=sorted(cycles), rows=int(tree.num_entries))


def _rows(keys, arrays):
    cluster_keys = _keys(arrays)
    rows = np.searchsorted(keys, cluster_keys)
    if np.any(rows >= len(keys)):
        raise ValueError('cluster event not in full event census')
    if not np.array_equal(keys[rows], cluster_keys):
        raise ValueError('cluster event not in full event census')
    return rows


def _inspect(path, max_events, max_rows):
    """Validate the actual highest cycles, full census, and every collection index."""
    source_hash = digest(path)
    with uproot.open(path) as root:
        ev, emeta = _highest(root, 'crvdump/events')
        cl, cmeta = _highest(root, 'crvdump/clusters')
        if ev.num_entries > max_events or cl.num_entries > max_rows:
            raise ValueError('input exceeds event/cluster-row memory limits')
        arrays = ev.arrays([*IDS, 'ncluster'], library='np')
        keys = _keys(arrays)
        order = np.argsort(keys)
        keys = keys[order]
        if len(keys) > 1 and np.any(keys[1:] == keys[:-1]):
            raise ValueError(f'duplicate event rows: {path}')
        counts = _integer(arrays['ncluster'], 'ncluster')[order]
        if np.any(counts > cl.num_entries) or counts.sum() != cl.num_entries:
            raise ValueError('cluster rows disagree with per-event cluster census')
        arrays = cl.arrays([*IDS, 'idx'], library='np')
        rows = _rows(keys, arrays)
        if not np.array_equal(np.bincount(rows, minlength=len(keys)), counts):
            raise ValueError('cluster rows disagree with per-event cluster census')
        idx = _integer(arrays['idx'], 'cluster idx')
        if np.any(idx >= counts[rows]):
            raise ValueError('cluster idx outside per-event collection')
        offsets = np.cumsum(counts) - counts
        if len(np.unique(offsets[rows] + idx)) != len(rows):
            raise ValueError('duplicate cluster collection indices')
    if digest(path) != source_hash:
        raise ValueError('input changed during census inspection')
    meta = dict(path=str(Path(path).resolve()), sha256=source_hash, events=emeta, clusters=cmeta,
                sorted_event_keys_sha256=hashlib.sha256(keys.tobytes()).hexdigest())
    return dict(keys=keys, counts=counts, meta=meta)


def _matrix(arm, name, edges):
    """One observable in one file; sparse event/bin counts, zeros retained."""
    branches = OBSERVABLES[name][0].split(',')
    with uproot.open(arm['meta']['path']) as root:
        # Explicit selected cycle, not uproot's implicit cycle lookup.
        tree = root[arm['meta']['clusters']['key']]
        arrays = tree.arrays([*IDS, *branches], library='ak')
        rows = _rows(arm['keys'], arrays)
        if name == 'npulses':
            values = ak.to_numpy(ak.num(arrays['pulses'], axis=1)).astype(float)
        elif name == 'duration':
            with np.errstate(over='ignore', invalid='ignore'):
                values = ak.to_numpy(arrays['end']) - ak.to_numpy(arrays['start'])
        else:
            values = ak.to_numpy(arrays[branches[0]])
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f'{name}: nonfinite or nonscalar cluster observable')
        # ROOT convention: x == upper edge belongs to overflow.
        bins = np.searchsorted(edges, values, side='right')
        matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, bins)),
                                   shape=(len(arm['keys']), len(edges) + 1))
    return matrix


def rate(a, b):
    """Cluster fills per generated event, not an exposure-corrected physical rate."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    ta, tb = float(a.sum()), float(b.sum())
    se = float(np.sqrt(n * np.var(b - a, ddof=1)))
    result = dict(total_ref=ta, total_cand=tb, total_difference=tb - ta,
                  total_difference_se=se, mean_ref=ta / n, mean_cand=tb / n,
                  mean_difference=(tb - ta) / n, mean_difference_se=se / n,
                  relative_difference=None, relative_se=None)
    if ta > 0:
        ratio = tb / ta
        result.update(relative_difference=ratio - 1,
                      relative_se=float(np.sqrt(n * np.var(b - ratio * a, ddof=1)) / ta))
    return result


def shape(a, b, replicas=499, seed=12345):
    """Paired normalized-bin covariance and centered Gaussian multiplier TV.

    U_i = (B_i - q*b_i)/sum(b) - (A_i - p*a_i)/sum(a).
    Cov(q-p) = N/(N-1) * sum_i (U_i-Ubar)'(U_i-Ubar).
    U is sparse plus rank two; it is NEVER materialized as an N x bins array.
    """
    a, b = sparse.csr_matrix(a, dtype=float), sparse.csr_matrix(b, dtype=float)
    if (a.shape != b.shape or a.shape[0] < 2 or a.shape[1] > MAX_BINS + 2
            or not np.isfinite(a.data).all() or not np.isfinite(b.data).all()
            or np.any(a.data < 0) or np.any(b.data < 0)):
        raise ValueError('require matching finite nonnegative event/bin matrices with >=2 events')
    if not isinstance(replicas, (int, np.integer)) or not 2 <= replicas <= MAX_REPLICAS:
        raise ValueError(f'replicas must be in 2..{MAX_REPLICAS}')
    n = a.shape[0]
    ac, bc = np.asarray(a.sum(axis=0)).ravel(), np.asarray(b.sum(axis=0)).ravel()
    result = dict(counts_ref=ac.tolist(), counts_cand=bc.tolist())
    ta, tb = float(ac.sum()), float(bc.sum())
    if not np.isfinite([ta, tb]).all():
        raise ValueError('nonfinite regional totals')
    if ta == 0 or tb == 0:
        return dict(**result, status='undefined_empty_regional_total', empty_ref=ta == 0,
                    empty_cand=tb == 0, covariance=None, tv=None,
                    conditional_null_tv_quantiles=None, linearized_tv_uncertainty_quantiles=None)
    p, q = ac / ta, bc / tb
    delta = q - p
    totals_a, totals_b = np.asarray(a.sum(axis=1)).ravel(), np.asarray(b.sum(axis=1)).ravel()
    d = b / tb - a / ta
    v = np.column_stack((totals_a / ta, -totals_b / tb))
    pq = np.stack((p, q))
    cross = (d.T @ v) @ pq
    gram = (d.T @ d).toarray() + cross + cross.T + pq.T @ (v.T @ v) @ pq
    usum = np.asarray(d.sum(axis=0)).ravel() + v.sum(axis=0) @ pq
    covariance = (n / (n - 1)) * (gram - np.outer(usum, usum) / n)
    covariance = (covariance + covariance.T) / 2
    rng = np.random.default_rng(seed)
    fluctuations = np.empty((replicas, len(p)), dtype=float)
    scale = np.sqrt(n / (n - 1))
    for replica in range(replicas):
        weights = rng.standard_normal(n)
        weights -= weights.mean()
        fluctuations[replica] = scale * (d.T @ weights + (weights @ v) @ pq)
    null_tv = np.abs(fluctuations).sum(axis=1) / 2
    uncertain_tv = np.abs(fluctuations + delta).sum(axis=1) / 2
    return dict(**result, status='conditional_diagnostics', fraction_ref=p.tolist(),
                fraction_cand=q.tolist(), bin_difference=delta.tolist(),
                covariance=covariance.tolist(), bin_difference_se=np.sqrt(np.maximum(0, np.diag(covariance))).tolist(),
                tv=float(np.abs(delta).sum() / 2), quantile_levels=QUANTILES.tolist(),
                conditional_null_tv_quantiles=np.quantile(null_tv, QUANTILES).tolist(),
                linearized_tv_uncertainty_quantiles=np.quantile(uncertain_tv, QUANTILES).tolist(),
                linearized_bin_difference_quantiles=np.quantile(fluctuations + delta, QUANTILES, axis=0).tolist())


def _pairs(pairs):
    if not isinstance(pairs, list) or not pairs:
        raise ValueError('pairs must be a nonempty JSON list of {"ref": path, "cand": path}')
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {'ref', 'cand'}:
            raise ValueError('each pair must have exactly ref and cand keys')
        if any(not isinstance(pair[k], (str, Path)) or not str(pair[k]) for k in ('ref', 'cand')):
            raise ValueError('pair paths must be nonempty filenames')
    return pairs


def run(pairs, features, out, *, replicas=499, seed=12345,
        max_events=1000000, max_cluster_rows=5000000):
    """Read-only paired shards, strict JSON output created exclusively at a new path."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing existing output: {out}')
    features = validate_edges(features)
    pairs = _pairs(pairs)
    if not 2 <= replicas <= MAX_REPLICAS or seed < 0 or max_events < 2 or max_cluster_rows < 0:
        raise ValueError('invalid replicas, seed or memory limits')
    inspected, events, rows_ref, rows_cand = [], 0, 0, 0
    for pair in pairs:
        ref = _inspect(pair['ref'], max_events - events, max_cluster_rows - rows_ref)
        cand = _inspect(pair['cand'], max_events - events, max_cluster_rows - rows_cand)
        if not np.array_equal(ref['keys'], cand['keys']):
            raise ValueError('paired files require identical full event keys, including zero-cluster events')
        inspected.append((ref, cand))
        events += len(ref['keys'])
        rows_ref += ref['meta']['clusters']['rows']
        rows_cand += cand['meta']['clusters']['rows']
    if events < 2:
        raise ValueError('require at least two generated events')
    all_keys = np.concatenate([r['keys'] for r, _ in inspected])
    if len(np.unique(all_keys)) != events:
        raise ValueError('duplicate/reused event keys across shard pairs')
    key_digest = hashlib.sha256(np.sort(all_keys).tobytes()).hexdigest()
    del all_keys
    result = dict(schema_version=1, events=events, sorted_event_keys_sha256=key_digest,
                  pairs=[dict(ref=r['meta'], cand=c['meta']) for r, c in inspected],
                  limits=dict(max_events=max_events, max_cluster_rows_per_arm=max_cluster_rows,
                              max_visible_bins=MAX_BINS),
                  multiplier=dict(kind='centered Gaussian paired-event influence multiplier',
                                  replicas=replicas, seed=seed,
                                  null='zero normalized bin difference; plug-in observed paired influence covariance',
                                  uncertainty='TV(delta + multiplier fluctuation); linearized, not percentile event bootstrap'),
                  limitations=LIMITATIONS, observables={})
    result['total_cluster_rate'] = rate(np.concatenate([r['counts'] for r, _ in inspected]),
                                        np.concatenate([c['counts'] for _, c in inspected]))
    # At most one observable's sparse event/bin contributions are resident at a time.
    for name, edges in features.items():
        a = sparse.vstack([_matrix(r, name, edges) for r, _ in inspected], format='csr')
        b = sparse.vstack([_matrix(c, name, edges) for _, c in inspected], format='csr')
        inclusive = shape(a, b, replicas, seed)
        visible = shape(a[:, 1:-1], b[:, 1:-1], replicas, seed)
        visible_rate = rate(np.asarray(a[:, 1:-1].sum(axis=1)).ravel(),
                            np.asarray(b[:, 1:-1].sum(axis=1)).ravel())
        flow_rate = rate(np.asarray(a[:, [0, a.shape[1] - 1]].sum(axis=1)).ravel(),
                         np.asarray(b[:, [0, b.shape[1] - 1]].sum(axis=1)).ravel())
        result['observables'][name] = dict(definition=OBSERVABLES[name][1], edges=edges.tolist(),
                                          bin_order='underflow, visible bins in edge order, overflow',
                                          edge_source='explicit caller edges; no booked histogram closure claim',
                                          inclusive=inclusive, visible=visible,
                                          visible_rate=visible_rate, flow_rate=flow_rate)
        del a, b
    for pair in inspected:
        for arm in pair:
            if digest(arm['meta']['path']) != arm['meta']['sha256']:
                raise ValueError('input changed during observable analysis')
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with out.open('x') as stream:
        stream.write(text)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', metavar='ROOT', help='exactly REF CAND, or use --pairs')
    parser.add_argument('--pairs', help='JSON list of explicit ref/cand shard pairs (paths relative to current directory)')
    parser.add_argument('--edges', required=True, help='JSON object: observable -> exact finite edges')
    parser.add_argument('--out', required=True, help='new JSON output; never overwrite')
    parser.add_argument('--replicas', type=int, default=499)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--max-events', type=int, default=1000000)
    parser.add_argument('--max-cluster-rows', type=int, default=5000000)
    args = parser.parse_args(argv)
    try:
        if args.pairs:
            if args.files:
                raise ValueError('use --pairs OR two files, not both')
            pairs = json.loads(Path(args.pairs).read_text())
        else:
            if len(args.files) != 2:
                raise ValueError('supply two files REF CAND or --pairs JSON')
            pairs = [dict(ref=args.files[0], cand=args.files[1])]
        result = run(pairs, json.loads(Path(args.edges).read_text()), args.out,
                     replicas=args.replicas, seed=args.seed,
                     max_events=args.max_events, max_cluster_rows=args.max_cluster_rows)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        parser.error(str(exc))
    print(f'Wrote {args.out}: {result["events"]} paired events; conditional diagnostics only')


if __name__ == '__main__':
    main()
