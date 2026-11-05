"""Reconcile every inclusive primary with exact parent-window exposure, without transport."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import geometry as G
from .target_parent_audit import checksum, physical_event_order, production_exposure
from .target_rpc_parent_audit import COLUMNS, exact_window, reduce_parent


def verify(parent, reference, generated_events, code):
    """Require exact source support and complete emitted-electron count for this inclusive window."""
    if int(reference['creation_code']) != code or not np.array_equal(reference['gencount'], [generated_events]):
        raise ValueError('inclusive process/event exposure mismatch')
    ng = int(len(reference['x']))
    event = np.asarray(reference['evt'])
    if not np.isfinite(event).all() or np.any(event != np.floor(event)):
        raise ValueError('finite integer inclusive event identities required')
    if not np.array_equal(np.unique(event), np.arange(1, generated_events + 1)):
        raise ValueError('inclusive primaries must cover every declared event')
    if code == 166:
        record = production_exposure(parent, reference, generated_events)['files'][0]
    elif code == 179:
        subset = {k: reference[k] for k in ('x', 'y', 'z', 'time', 'evt', 'ifile')}
        record = exact_window(parent, subset, generated_events)
    else:
        raise ValueError('DIO/internal-RPC inclusive source only')
    if record['generated_electrons'] != ng:
        raise ValueError('inclusive emitted-electron count disagrees with exact parent-window exposure')
    offset = record['inferred_first_parent_event_index']
    events = (offset + np.arange(generated_events)) % len(parent['event_multiplicity'])
    expected = np.asarray(parent['event_multiplicity'])[events]
    actual = np.bincount(event.astype(int) - 1, minlength=generated_events)
    if not np.array_equal(expected, actual):
        raise ValueError('inclusive per-event electron multiplicity disagrees with exact parent window')
    return dict(mode='inclusive-exact-parent-window-census', generated_events=generated_events,
                generated_electrons=ng, creation_code=code, exact_parent_exposure_reconciled=True,
                window=record, filter_implication_certified=False, generated_yield_validated=False,
                caveat='source support and exposure only; not transport, joint-generator or filter certification')


def main(argv=None):
    import uproot
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--generated-events', required=True, type=int)
    ap.add_argument('--code', type=int, required=True, choices=(166, 179))
    ap.add_argument('--parent', help='ordered DIO target_parent_audit NPZ')
    ap.add_argument('--parent-root', help='RPC raw stopped-pion tree')
    ap.add_argument('--parent-art', help='RPC parent art metadata order')
    ap.add_argument('--parent-events', type=int)
    ap.add_argument('--fileindex-byteswap', action='store_true')
    ap.add_argument('--offline-root')
    ap.add_argument('--geom', default='Offline/Mu2eG4/geom/geom_run1_a.txt')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise FileExistsError('refusing to replace inclusive parent receipt')
    if a.code == 166:
        if not a.parent:
            ap.error('DIO requires ordered parent NPZ')
        with np.load(a.parent, allow_pickle=False) as saved:
            parent = {k: saved[k] for k in saved.files if k != 'metadata'}
        provenance = dict(parent_sha256=checksum(a.parent))
    else:
        if not all((a.parent_root, a.parent_art, a.parent_events, a.offline_root)):
            ap.error('RPC requires parent root/art/events/geometry')
        with uproot.open(a.parent_root) as root:
            parent = reduce_parent(root['cetree/ntpart'].arrays(list(COLUMNS), library='np'),
                                   G.from_offline(a.geom, a.offline_root), a.parent_events)
        parent, ordering = physical_event_order(parent, a.parent_art, a.fileindex_byteswap)
        provenance = dict(parent_tree_sha256=checksum(a.parent_root), parent_art_sha256=checksum(a.parent_art),
                          physical_event_order=ordering)
    with np.load(a.reference, allow_pickle=False) as reference:
        report = verify(parent, reference, a.generated_events, a.code)
    report.update(reference_sha256=checksum(a.reference), **provenance)
    Path(a.out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
