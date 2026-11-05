"""Audit inclusive target-born electrons against an independently keyed stock event ledger."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import cechain as C
from . import cecompare as CC
from .target_filter_probe import missed_proxy
from .target_parent_audit import checksum

IDENTITY = ('run', 'subrun', 'evt', 'trk')
START = ('x', 'y', 'z', 'px', 'py', 'pz', 'time')
VD = ('isvd', 'xvd', 'yvd', 'zvd', 'pxvd', 'pyvd', 'pzvd', 'tvd')


def integer_keys(values, width):
    """Reject truncation and nonfinite identities before any integer join."""
    a = np.asarray(values)
    if (a.ndim != 2 or a.shape[1] != width or not np.isfinite(a).all()
            or np.any(a < 0) or np.any(a != np.floor(a)) or np.any(a >= 2 ** 63)):
        raise ValueError('finite nonnegative integer identities required')
    return a.astype('<i8')


def keys(a):
    a = np.ascontiguousarray(a, dtype='<i8')
    return a.view(np.dtype([(str(i), '<i8') for i in range(a.shape[1])])).ravel()


def event_indices(identity, event_identity):
    """Require a complete exact integer event-key join, never a row-order association."""
    ev = integer_keys(event_identity, 3)
    if np.unique(ev, axis=0).shape[0] != ev.shape[0]:
        raise ValueError('duplicate event ledger identities')
    rows = integer_keys(identity, 4)
    order = np.argsort(keys(ev))
    pos = np.searchsorted(keys(ev)[order], keys(rows[:, :3]))
    if np.any(pos == order.size):
        raise ValueError('primary event absent from ledger')
    index = order[pos]
    if not np.array_equal(ev[index], rows[:, :3]):
        raise ValueError('primary event absent from ledger')
    return index


def read_primaries(path, code, tree='cetree/ntpart'):
    """Stream raw scalar starts and only the first VD13 state; retain pair daughters."""
    import uproot
    names = IDENTITY + START + VD + ('pdg', 'code', 'codestop')
    parts = {k: [] for k in names}
    with uproot.open(path) as root:
        source = root[tree]
        cycles = {k: int(root[k].num_entries) for k in root.keys() if k.split(';')[0] == tree}
        for a in source.iterate(list(names), step_size='8 MB', library='np'):
            sel = (a['code'] == code) & np.isin(a['pdg'], [11, -11])
            for k in names:
                v = a[k][sel]
                if k in VD:
                    v = np.stack(v)[:, 12] if v.size else np.empty(0)
                parts[k].append(v.copy())
    out = {k: np.concatenate(v) if v else np.empty(0) for k, v in parts.items()}
    identity = integer_keys(np.column_stack([out[k] for k in IDENTITY]), 4)
    if np.unique(identity, axis=0).shape[0] != identity.shape[0]:
        raise ValueError('duplicate inclusive primary identities')
    return identity, out, cycles


def census(identity, columns, ledger, code):
    """Keep source multiplicity and nonzero statuses explicit in a primary/event census."""
    ev = integer_keys(ledger['identities'], 3)
    index = event_indices(identity, ev)
    n = ev.shape[0]
    status = np.asarray(ledger['status'])
    passed = np.asarray(ledger['accepted'])
    if status.shape != (n,) or passed.shape != (n,) or not np.isin(passed, [False, True]).all():
        raise ValueError('complete status and PrimaryPath decisions required')
    if not np.isfinite(status).all() or np.any(status < 0) or np.any(status != np.floor(status)):
        raise ValueError('invalid G4 statuses')
    el = np.asarray(columns['pdg']) == 11
    po = np.asarray(columns['pdg']) == -11
    ne, np_ = np.bincount(index[el], minlength=n), np.bincount(index[po], minlength=n)
    if code == 166 and (np.any(ne < 1) or np.any(np_)):
        raise ValueError('DIO control requires every audited parent event to emit negative electrons only')
    if code == 179 and (np.any(ne != 1) or np.any(np_ != 1)):
        raise ValueError('one-pion RPC control requires exactly one electron and positron per event')
    if code not in (166, 179):
        raise ValueError('audited DIO or internal-RPC creation code required')
    if np.any(passed & (status > 9)):
        raise ValueError('PrimaryPath cannot pass a G4 status rejected by the released status filter')
    return index, dict(generated_events=int(n), generated_electrons=int(el.sum()),
                      generated_positrons=int(po.sum()), passed_events=int(passed.sum()),
                      electron_multiplicity={str(int(k)): int(v) for k, v in zip(*np.unique(ne, return_counts=True))},
                      status_counts={str(int(k)): int(v) for k, v in zip(*np.unique(status, return_counts=True))},
                      all_status_zero=bool(np.all(status == 0)),
                      caveat='counts are this inclusive pilot only; status<=9 acceptance is not status-zero certification')


def compare(path, ledger_path, geom, field, code):
    with np.load(ledger_path, allow_pickle=False) as saved:
        if 'generation_source_sha256' not in saved:
            raise ValueError('independent generation census reconciliation required for inclusive interpretation')
        generation_sha = str(saved['generation_source_sha256'].item())
        if len(generation_sha) != 64 or any(c not in '0123456789abcdef' for c in generation_sha):
            raise ValueError('invalid generation census hash')
        ledger = {k: saved[k] for k in ('identities', 'status', 'accepted')}
    identity, a, cycles = read_primaries(path, code)
    index, report = census(identity, a, ledger, code)
    a['g4_status'] = ledger['status'][index]
    a['primary_path_accepted'] = ledger['accepted'][index]
    electron = a['pdg'] == 11
    hit = electron & a['isvd'].astype(bool)
    x = np.column_stack([a[k][hit] for k in ('xvd', 'yvd', 'zvd')]).astype(float)
    pv = np.column_stack([a[k][hit] for k in ('pxvd', 'pyvd', 'pzvd')]).astype(float)
    p = np.linalg.norm(pv, axis=1)
    if not np.isfinite(x).all() or not np.isfinite(pv).all() or np.any(p <= 0):
        raise ValueError('finite positive-momentum primary VD states required')
    rmax, _ = C.helix_rmax(geom, x, pv / p[:, None], p, field(x)[:, 2])
    if not np.isfinite(rmax).all():
        raise ValueError('nonfinite inclusive helix proxy')
    n = ledger['identities'].shape[0]
    accepted = ledger['accepted'].astype(bool)
    status = ledger['status']
    cuts = {}
    for rc in CC.RMAX_CUTS:
        selected = np.zeros(identity.shape[0], bool)
        selected[hit] = (p > CC.P_MIN) & (rmax > rc)
        events = np.bincount(index[selected], minlength=n) > 0
        missed = selected & ~accepted[index]
        event_diagnostic = missed_proxy(events & (status == 0), accepted)
        event_diagnostic['proxy_events'] = event_diagnostic.pop('proxy_electrons')
        cuts[str(rc)] = dict(selected_electrons=int(selected.sum()), selected_events=int(events.sum()),
            selected_status_zero_electrons=int((selected & (status[index] == 0)).sum()),
            missed_status_zero_electrons=int((missed & (status[index] == 0)).sum()),
            selected_nonzero_status_electrons=int((selected & (status[index] != 0)).sum()),
            path_rejected_electrons=int(missed.sum()),
            missed_identities=identity[missed].tolist(),
            status_zero_event_miss_diagnostic=event_diagnostic,
            binomial_caveat='event-level iid-binomial descriptive limit on this pilot only; fixed sequential parent support and library uncertainty not covered')
    report.update(mode='inclusive-matching-background-filter-diagnostic', process='dio' if code == 166 else 'rpc-internal',
        primary_vd13_electrons=int(hit.sum()), tree_cycles=cycles, cuts=cuts,
        reference_sha256=checksum(path), ledger_sha256=checksum(ledger_path),
        generation_census_reconciled=True, generation_source_sha256=generation_sha,
        filter_implication_certified=False, generated_yield_validated=False,
        distribution_validated=False, source_library_uncertainty_quantified=False)
    return report, identity, a


def save_reference(out, identity, columns, code, generated_events):
    """One inclusive file's actual primaries, including zeros at VD, for same-start transport."""
    el = columns['pdg'] == 11
    reference = {k: columns[k][el] for k in START + ('codestop',)}
    reference.update({k: identity[el, i] for i, k in enumerate(IDENTITY)})
    reference.update({k + '_front': columns[k][el] for k in VD})
    reference.update(ifile=np.zeros(int(el.sum()), dtype=np.int64), gencount=np.array([generated_events]),
                     rows=np.array([el.sum()]), creation_code=code)
    for k in ('g4_status', 'primary_path_accepted'):
        if k in columns:
            reference[k] = columns[k][el]
    np.savez_compressed(out, **reference)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--ledger', required=True)
    ap.add_argument('--code', type=int, required=True, choices=(166, 179))
    ap.add_argument('--offline-root', required=True)
    ap.add_argument('--geom', default='Offline/Mu2eG4/geom/geom_run1_a.txt')
    ap.add_argument('--maps', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-out', required=True)
    a = ap.parse_args(argv)
    if Path(a.out).exists() or Path(a.reference_out).exists():
        raise FileExistsError('refusing to replace inclusive diagnostic outputs')
    report, identity, columns = compare(a.root, a.ledger, C.from_offline(a.geom, a.offline_root), C.field_maps(a.maps), a.code)
    save_reference(a.reference_out, identity, columns, a.code, report['generated_events'])
    report['reduced_reference_sha256'] = checksum(a.reference_out)
    Path(a.out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
