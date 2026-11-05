"""Measure VD-proxy misses of the production detector-step filter on an existing inclusive control."""
import argparse
import json

import numpy as np

from . import cechain as C
from . import cecompare as CC


def missed_proxy(selected, passed, alpha=.05):
    """One-sided exact Clopper-Pearson upper bound; zero misses is not logical implication."""
    from scipy.stats import beta
    selected, passed = np.asarray(selected, bool), np.asarray(passed, bool)
    if selected.shape != passed.shape or selected.ndim != 1:
        raise ValueError("equal-length proxy and filter masks required")
    n, missed = int(selected.sum()), int((selected & ~passed).sum())
    upper = float(beta.ppf(1 - alpha, missed + 1, n - missed)) if n > missed else (1. if n else None)
    return dict(proxy_electrons=n, missed_by_filter=missed, miss_fraction=missed / n if n else None,
                miss_fraction_upper_95=upper, method="one-sided exact binomial upper bound on this inclusive control only")


def read(tree, code):
    names = ("run", "subrun", "evt", "trk", "code", "pdg", "isvd", "xvd", "yvd", "zvd", "pxvd", "pyvd", "pzvd")
    parts = {k: [] for k in names}
    for a in tree.iterate(list(names), step_size='8 MB', library='np'):
        keep = (a['code'] == code) & (a['pdg'] == 11)
        for k in names:
            values = np.stack(a[k])[:, 12] if k in ('isvd', 'xvd', 'yvd', 'zvd', 'pxvd', 'pyvd', 'pzvd') else a[k]
            parts[k].append(values[keep].copy())
    reduced = {k: np.concatenate(v) for k, v in parts.items()}
    identity = np.column_stack([reduced[k] for k in ('run', 'subrun', 'evt', 'trk')]).astype('<i8')
    if np.unique(identity, axis=0).shape[0] != identity.shape[0]:
        raise ValueError("duplicate primary identities in control tree")
    return identity, reduced


def compare(path, geom, field, code=167, expected_events=2000, passed_path=None):
    import uproot
    with uproot.open(path) as root:
        identity, a = read(root['all/ntpart'], code)
        cycles = {k: int(root[k].num_entries) for k in root.keys() if k.split(';')[0] == 'all/ntpart'}
    with uproot.open(passed_path or path) as root:
        kept, _ = read(root['passed/ntpart'], code)
        cycles.update({k: int(root[k].num_entries) for k in root.keys() if k.split(';')[0] == 'passed/ntpart'})
    if np.unique(identity[:, :3], axis=0).shape[0] != expected_events or identity.shape[0] != expected_events:
        raise ValueError("control must contain one primary electron per externally audited event")
    dtype = np.dtype([(k, '<i8') for k in ('run', 'subrun', 'evt', 'trk')])
    all_keys = np.ascontiguousarray(identity).view(dtype).ravel()
    kept_keys = np.ascontiguousarray(kept).view(dtype).ravel()
    if not np.isin(kept_keys, all_keys).all():
        raise ValueError("filtered control identities absent from inclusive arm")
    passed = np.isin(all_keys, kept_keys)
    hit = a['isvd'].astype(bool)
    x = np.column_stack([a[k] for k in ('xvd', 'yvd', 'zvd')]).astype(float)
    pv = np.column_stack([a[k] for k in ('pxvd', 'pyvd', 'pzvd')]).astype(float)
    p = np.linalg.norm(pv, axis=1)
    rmax, _ = C.helix_rmax(geom, x[hit], pv[hit] / p[hit, None], p[hit], field(x[hit])[:, 2])
    cuts = {}
    for rc in CC.RMAX_CUTS:
        sel = np.zeros(hit.size, bool)
        sel[hit] = (p[hit] > CC.P_MIN) & (rmax > rc)
        cuts[str(rc)] = missed_proxy(sel, passed)
        cuts[str(rc)]['missed_identities'] = identity[sel & ~passed].tolist()
    return dict(mode='inclusive-CE-control-filter-probe-not-DIO-RPC-certification', events=expected_events,
                passed_events=int(passed.sum()), primary_vd13=int(hit.sum()), tree_cycles=cycles, cuts=cuts,
                filter_implication_certified=False, generated_yield_validated=False,
                caveat="bounded CE control, different release/source energy and population; measures or falsifies implication here, never certifies DIO/RPC completeness")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--passed-root', help='separate passed-arm ReadVirtualDetector tree')
    ap.add_argument('--offline-root', required=True)
    ap.add_argument('--geom', default='Offline/Mu2eG4/geom/geom_run1_a.txt')
    ap.add_argument('--maps', nargs='+', required=True)
    ap.add_argument('--expected-events', type=int, default=2000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    report = compare(a.root, C.from_offline(a.geom, a.offline_root), C.field_maps(a.maps),
                     expected_events=a.expected_events, passed_path=a.passed_root)
    with open(a.out, 'w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
