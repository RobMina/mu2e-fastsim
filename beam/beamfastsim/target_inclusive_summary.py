"""Pool completed inclusive pilot censuses without extending their physics scope."""
import argparse
import json
from pathlib import Path

import numpy as np

from .target_filter_probe import missed_proxy
from .target_parent_audit import checksum


def summarize(paths, fast_paths=()):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    if not reports or len({r['process'] for r in reports}) != 1:
        raise ValueError('one nonempty inclusive process family required')
    if (any(r.get('mode') != 'inclusive-matching-background-filter-diagnostic'
            or not r.get('generation_census_reconciled') for r in reports)
            or len({r['reference_sha256'] for r in reports}) != len(reports)):
        raise ValueError('distinct independently reconciled inclusive references required')
    if any(not r['all_status_zero'] for r in reports):
        raise ValueError('nonzero status pilot retained; requires explicit separate pooling treatment')
    events = sum(r['generated_events'] for r in reports)
    electrons = sum(r['generated_electrons'] for r in reports)
    out = dict(mode='inclusive-pilot-pooled-diagnostic', process=reports[0]['process'],
               generated_events=events, generated_electrons=electrons,
               generated_positrons=sum(r['generated_positrons'] for r in reports),
               primary_vd13_electrons=sum(r['primary_vd13_electrons'] for r in reports),
               passed_events=sum(r['passed_events'] for r in reports), files=len(reports),
               census_sha256=[checksum(p) for p in paths], cuts={},
               generated_yield_validated=False, filter_implication_certified=False,
               caveat='capped matching-background pilot only; not an equivalence test, shape/tail validation, source-library uncertainty estimate, full pair or external conversion certification')
    fast = [json.loads(Path(p).read_text()) for p in fast_paths]
    if fast:
        if (len(fast) != len(reports) or any(r.get('mode') != 'inclusive-same-start-electron-marginal-diagnostic' for r in fast)
                or [r['generated_electrons'] for r in fast] != [r['generated_electrons'] for r in reports]
                or [r['census_sha256'] for r in fast] != [checksum(p) for p in paths]):
            raise ValueError('fast diagnostics must match every inclusive census in order')
        out['fast_compare_sha256'] = [checksum(p) for p in fast_paths]
    for rc in reports[0]['cuts']:
        selected = sum(r['cuts'][rc]['selected_electrons'] for r in reports)
        rejected = sum(r['cuts'][rc]['path_rejected_electrons'] for r in reports)
        proxy_events = sum(r['cuts'][rc]['status_zero_event_miss_diagnostic']['proxy_events'] for r in reports)
        missed_events = sum(r['cuts'][rc]['status_zero_event_miss_diagnostic']['missed_by_filter'] for r in reports)
        stats = missed_proxy(np.ones(proxy_events, bool), np.arange(proxy_events) >= missed_events)
        stats['proxy_events'] = stats.pop('proxy_electrons')
        cut = dict(g4_selected=selected, path_rejected_electrons=rejected,
                   g4_per_generated_electron=selected / electrons,
                   g4_per_generated_event=selected / events, event_miss_diagnostic=stats,
                   upper_limit_caveat='descriptive event-iid binomial limit for this pilot only; shared sequential parent support/library effects not covered')
        if fast:
            if any(r['cuts'][rc]['migration']['both'] + r['cuts'][rc]['migration']['base_only'] != reports[i]['cuts'][rc]['selected_electrons']
                   for i, r in enumerate(fast)):
                raise ValueError('fast comparison G4 selections disagree with inclusive census')
            g = np.array([r['cuts'][rc]['selected_electrons'] for r in reports], float)
            f = np.array([r['cuts'][rc]['migration']['both'] + r['cuts'][rc]['migration']['cand_only'] for r in fast], float)
            ratio = f.sum() / g.sum() if g.sum() else None
            influence = f - ratio * g if ratio is not None else None
            n = len(reports)
            err = (np.sqrt(n / (n - 1) * np.sum(influence ** 2)) / g.sum()
                   if n > 1 and influence is not None else None)
            event_error = None
            if len(fast) == 1:
                event_error = fast[0]['cuts'][rc]['inclusive_yield']['event_clustered_ratio_err']
            elif reports[0]['process'] == 'rpc-internal' and events == electrons and ratio is not None:
                both = sum(r['cuts'][rc]['migration']['both'] for r in fast)
                variance_sum = f.sum() + ratio ** 2 * g.sum() - 2 * ratio * both
                event_error = np.sqrt(events / (events - 1) * max(0., variance_sum)) / g.sum()
            cut.update(fast_selected=int(f.sum()), fast_per_generated_electron=float(f.sum() / electrons),
                       fast_over_g4=float(ratio) if ratio is not None else None,
                       paired_file_cluster_ratio_err=float(err) if err is not None else None,
                       paired_event_ratio_err=float(event_error) if event_error is not None else None,
                       event_error_method='paired event influence; RPC one electron per capture, DIO single-file event-cluster receipt; shared parent library not covered',
                       cluster_method='paired per-file influence; null with one file; few clusters can give unstable or spuriously small errors, not a substitute for within-pilot event uncertainty or parent-library uncertainty')
        out['cuts'][rc] = cut
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--census', action='append', required=True)
    ap.add_argument('--fast', action='append', default=[])
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise FileExistsError('refusing to replace pooled pilot summary')
    report = summarize(a.census, a.fast)
    Path(a.out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
