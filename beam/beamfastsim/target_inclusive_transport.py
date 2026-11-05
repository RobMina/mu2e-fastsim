"""Same-start fast/G4 diagnostic on every primary of an audited inclusive control."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from . import cechain as C
from . import cecompare as CC
from .target_compare import compare as paired_compare, clustered_mean
from .target_inclusive_compare import event_indices
from .target_parent_audit import checksum


def compare(reference, fast_pattern, census_path, geom, field):
    census = json.loads(Path(census_path).read_text())
    if (census.get('mode') != 'inclusive-matching-background-filter-diagnostic'
            or not census.get('generation_census_reconciled', False)
            or census.get('reduced_reference_sha256') != checksum(reference)):
        raise ValueError('hash-bound inclusive primary census required')
    with np.load(reference, allow_pickle=False) as r:
        n = len(r['x'])
        code = int(r['creation_code'])
        if code not in (166, 179) or n != census['generated_electrons']:
            raise ValueError('inclusive primary count/code mismatch')
        rows = np.column_stack([r[k] for k in ('run', 'subrun', 'evt', 'trk')])
        events, inverse = np.unique(rows[:, :3], axis=0, return_inverse=True)
        if events.shape[0] != census['generated_events']:
            raise ValueError('all inclusive generated events must contribute to primary census')
        event_indices(rows, events)
        if ('g4_status' not in r or r['g4_status'].shape != (n,)
                or not np.array_equal(np.unique(r['g4_status']), [0])):
            raise ValueError('nonzero G4 statuses retained; separate validity study required before transport comparison')
        decisions = r['primary_path_accepted']
        if decisions.shape != (n,) or not np.isin(decisions, [0, 1]).all():
            raise ValueError('one explicit pass/fail decision per primary required')
        event_count = np.bincount(inverse)
        pass_count = np.bincount(inverse, weights=decisions)
        if np.any((pass_count != 0) & (pass_count != event_count)):
            raise ValueError('inconsistent PrimaryPath decision within event')
        passed = decisions.astype(bool)
    fast = CC.fast_plane(glob.glob(fast_pattern))
    idx = np.asarray(fast['idx'])
    if not np.array_equal(np.sort(idx), np.arange(n)):
        raise ValueError('fast rows must cover all inclusive primary starts exactly once')
    report = paired_compare(reference, fast_pattern, geom, field)
    report.update(mode='inclusive-same-start-electron-marginal-diagnostic',
        generated_events=int(census['generated_events']), generated_electrons=n,
        reference_sha256=checksum(reference), census_sha256=checksum(census_path),
        filter_implication_certified=False, generated_yield_validated=False,
        caveat='inclusive source removes retained-output conditioning here, but pilot precision and joint/core/tail tolerances do not certify transport or pair response')
    g4 = CC.g4_plane(reference)
    go, fo = CC.observables(geom, field, g4), CC.observables(geom, field, fast)
    for rc in CC.RMAX_CUTS:
        gs, fs = np.zeros(n, bool), np.zeros(n, bool)
        gs[g4['ok']] = (go['p'] > CC.P_MIN) & (go['rmax'] > rc)
        fs[fast['ok']] = (fo['p'] > CC.P_MIN) & (fo['rmax'] > rc)
        counts = report['cuts'][str(rc)]['kept_yield']
        counts['normalization'] = 'per actual generated primary electron, not per event or kept output'
        ratio = counts['ratio']
        event_error = clustered_mean(fs.astype(float) - ratio * gs[idx], inverse[idx]) if ratio is not None else None
        counts['row_iid_diagnostic_errors'] = {k: counts.pop(k) for k in ('diff_err', 'ratio_err')}
        counts['diff_err'] = clustered_mean(fs.astype(float) - gs[idx], inverse[idx])['err']
        counts['ratio_err'] = (event_error['err'] / gs.mean()
            if event_error is not None and event_error['err'] is not None and gs.any() else None)
        counts['event_clustered_ratio_err'] = counts['ratio_err']
        counts['error_method'] = 'event-cluster influence on paired electron difference and ratio'
        counts['event_cluster_caveat'] = 'within-pilot event clusters only; shared sequential parent library and between-file uncertainty not covered; row_iid_diagnostic_errors are not primary errors'
        report['cuts'][str(rc)]['inclusive_yield'] = report['cuts'][str(rc)].pop('kept_yield')
        report['cuts'][str(rc)]['retained_population_replay'] = dict(
            starts=int(passed.sum()), g4_selected=int((gs & passed).sum()),
            fast_selected=int((fs & passed[idx]).sum()),
            interpretation='diagnostic replay on the same stock-retained subset, never an independent closure test')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--fast', required=True)
    ap.add_argument('--census', required=True)
    ap.add_argument('--offline-root', required=True)
    ap.add_argument('--geom', default='Offline/Mu2eG4/geom/geom_run1_a.txt')
    ap.add_argument('--maps', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise FileExistsError('refusing to replace inclusive transport diagnostic')
    report = compare(a.ref, a.fast, a.census, C.from_offline(a.geom, a.offline_root), C.field_maps(a.maps))
    Path(a.out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cuts'}, indent=2, allow_nan=False))
    for rc, cut in report['cuts'].items():
        print(json.dumps(dict(rmax_cut=rc, inclusive_yield=cut['inclusive_yield']), allow_nan=False))


if __name__ == '__main__':
    main()
