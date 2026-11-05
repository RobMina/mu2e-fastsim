"""Audit development exposure of DIO source ancestry without inventing independent parents."""
import argparse
import json
from pathlib import Path

import numpy as np

from .target_parent_audit import checksum


def integer_rows(values):
    a = np.asarray(values)
    if a.ndim != 1 or not np.isfinite(a).all() or np.any(a != np.floor(a)):
        raise ValueError('finite integer parent indices required')
    return a.astype(np.int64)


def audit(parent_path, draws, original_catalog=None):
    with np.load(parent_path, allow_pickle=False) as saved:
        parent = {k: saved[k] for k in ('run', 'subrun', 'evt', 'trk')}
    n = len(parent['run'])
    if n < 1 or any(v.shape != (n,) for v in parent.values()):
        raise ValueError('nonempty parent identity vectors required')
    identity = np.column_stack([integer_rows(parent[k]) for k in parent])
    if np.unique(identity, axis=0).shape[0] != n:
        raise ValueError('duplicate parent identities')
    clusters, inverse = np.unique(identity[:, :2], axis=0, return_inverse=True)
    used = np.zeros(n, bool)
    receipts = []
    for source, stop_path in draws:
        with np.load(source, allow_pickle=False) as draw:
            idx = integer_rows(draw['stop_idx'])
        with np.load(stop_path, allow_pickle=False) as stops:
            rows = np.column_stack([integer_rows(stops[k]) for k in parent])
            if idx.size < 1 or np.any(idx < 0) or np.any(idx >= rows.shape[0]):
                raise ValueError('source stop indices outside declared stop population')
            born = rows[idx]
        dtype = np.dtype([(k, '<i8') for k in parent])
        keys = np.ascontiguousarray(identity, dtype='<i8').view(dtype).ravel()
        query = np.ascontiguousarray(born, dtype='<i8').view(dtype).ravel()
        order = np.argsort(keys)
        left = np.searchsorted(keys[order], query, side='left')
        right = np.searchsorted(keys[order], query, side='right')
        if np.any(right - left != 1):
            raise ValueError('draw identity does not exactly join the declared full parent')
        touched = order[left]
        used[touched] = True
        receipts.append(dict(source_sha256=checksum(source), stops_sha256=checksum(stop_path),
            source_rows=int(idx.size), unique_stop_rows=int(np.unique(touched).size),
            touched_run_subruns=int(np.unique(inverse[touched]).size)))
    touched_count = np.bincount(inverse[used], minlength=len(clusters))
    row_count = np.bincount(inverse, minlength=len(clusters))
    untouched = clusters[touched_count == 0]
    report = dict(mode='development-source-parent-coverage-audit', development_only=True,
        parent_sha256=checksum(parent_path), parent_stop_rows=n, run_subrun_clusters=len(clusters),
        development_draws=receipts, unique_used_stop_rows=int(used.sum()),
        used_run_subrun_clusters=int((touched_count > 0).sum()),
        unused_run_subrun_clusters=int(len(untouched)), unused_run_subrun_keys=untouched.tolist(),
        enough_unused_labels_for_24_cluster_design=bool(len(untouched) >= 24),
        smallest_cluster_stop_rows=int(row_count.min()), largest_cluster_stop_rows=int(row_count.max()),
        parent_independence_established=False, generated_yield_validated=False,
        filter_implication_certified=False,
        caveat='run/subrun coverage is a necessary exclusion audit, not proof of ancestry independence; unlisted development draws can only reduce unused support')
    if original_catalog:
        files = json.loads(Path(original_catalog).read_text())
        mapping = []
        for record in files:
            meta = record['metadata']
            if meta['rs.first_run'] != meta['rs.last_run'] or meta['rs.first_subrun'] != meta['rs.last_subrun']:
                raise ValueError('original source file spans multiple run/subrun labels')
            mapping.append((meta['rs.first_run'], meta['rs.first_subrun']))
        mapping = np.asarray(mapping, dtype=np.int64)
        if len(mapping) != np.unique(mapping, axis=0).shape[0]:
            raise ValueError('original catalogue does not identify one file per run/subrun')
        report.update(original_catalog_sha256=checksum(original_catalog), original_catalog_files=len(files),
            parent_labels_present_in_catalog=bool(np.isin(
                np.ascontiguousarray(clusters).view('V16').ravel(),
                np.ascontiguousarray(mapping).view('V16').ravel()).all()))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', required=True)
    ap.add_argument('--draw', nargs=2, action='append', required=True, metavar=('SOURCE', 'STOPS'))
    ap.add_argument('--original-catalog')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise FileExistsError('refusing to overwrite coverage audit')
    report = audit(a.parent, a.draw, a.original_catalog)
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with open(a.out, 'x') as stream:
        stream.write(text)
    print(text)


if __name__ == '__main__':
    main()
