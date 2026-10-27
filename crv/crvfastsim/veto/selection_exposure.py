"""Audit retained selected-track exposure; MC PBI is a diagnostic, not a measured correction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from .replay import sha256

IDS = ['run', 'subrun', 'event']


def tree_schema(path):
    """Inspect only event-tree schemas, without loading track or art product collections."""
    with uproot.open(path, timeout=60) as root:
        trees = {}
        for key, classname in root.classnames(recursive=True, cycle=False).items():
            if classname in ('TTree', 'ROOT::RNTuple'):
                tree = root[key]
                trees[key] = {'entries': tree.num_entries,
                              'exposure_branches': [k for k in tree.keys() if any(
                                  x in k.lower() for x in ('evt', 'proton', 'pbtime', 'intensity'))]}
        return {'path': path, 'uuid': str(root.file.uuid), 'trees': trees}


def correlate(x, y):
    """Return a descriptive Pearson correlation, or null for a degenerate marginal."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def joined_diagnostic(selected, exposure):
    """Join by full event identity and retain rejected-selection events in exposure denominators."""
    if selected.duplicated(IDS).any() or exposure.duplicated(IDS).any():
        raise ValueError('ambiguous event identity; require source-file identity before combining duplicates')
    if not np.isfinite(exposure[['nprotons', 'pbtime']].to_numpy()).all() or np.any(exposure.nprotons < 0):
        raise ValueError('invalid exposure values')
    matched = selected.merge(exposure, on=IDS, how='inner', validate='one_to_one')
    if not len(matched):
        raise ValueError('no selected events match the supplied exposure')
    t = ak.from_arrow(pa.array(matched.t_mid))
    n = ak.to_numpy(ak.num(t))
    flat = ak.to_numpy(ak.flatten(t))
    pbi = np.repeat(matched.nprotons.to_numpy(), n)
    pb = np.repeat(matched.pbtime.to_numpy(), n)
    if not len(flat) or not np.isfinite(flat).all():
        raise ValueError('invalid selected-track times')
    edges = np.unique(np.quantile(exposure.nprotons, np.linspace(0, 1, 6)))
    rows = []
    for i in range(max(0, len(edges) - 1)):
        lo, hi = edges[i:i + 2]
        allmask = (exposure.nprotons >= lo) & ((exposure.nprotons <= hi) if i == len(edges) - 2 else (exposure.nprotons < hi))
        smask = (matched.nprotons >= lo) & ((matched.nprotons <= hi) if i == len(edges) - 2 else (matched.nprotons < hi))
        trackmask = (pbi >= lo) & ((pbi <= hi) if i == len(edges) - 2 else (pbi < hi))
        rows.append({'pbi_lo': float(lo), 'pbi_hi': float(hi), 'upper_inclusive': i == len(edges) - 2,
                     'ntuple_events': int(allmask.sum()), 'selected_events': int(smask.sum()),
                     'selected_tracks': int(trackmask.sum()),
                     'selected_fraction_of_ntuple_events': float(smask.sum() / allmask.sum()) if allmask.sum() else None,
                     'track_time_mean_ns': float(flat[trackmask].mean()) if trackmask.any() else None})
    return {'ntuple_events': len(exposure), 'selected_events_matched': len(matched), 'selected_tracks_matched': len(flat),
            'selected_events_not_in_supplied_ntuples': len(selected) - len(matched),
            'pbi_mean_ntuple_events': float(exposure.nprotons.mean()),
            'pbi_mean_selected_events': float(matched.nprotons.mean()), 'pbi_mean_selected_tracks': float(pbi.mean()),
            'corr_track_time_mc_pbi': correlate(flat, pbi), 'corr_track_time_mc_pbtime': correlate(flat, pb),
            'corr_mc_pbi_mc_pbtime_ntuple': correlate(exposure.nprotons, exposure.pbtime),
            'pbi_quantile_strata': rows,
            'scope': 'MC-truth exposure diagnostic conditional on retained ntuple events; not generated-CE efficiency, a correction model, or a factorisation certificate.'}


def run(selection, ntuples, out, tree=None, pmin=85., pmax=200.):
    out = Path(out)
    if out.exists():
        raise ValueError('choose a new report')
    schema = pq.read_schema(selection)
    report = {'selection': {'path': selection, 'sha256': sha256(selection), 'fields': schema.names,
                            'has_pbi': any('proton' in n or 'intensity' in n for n in schema.names)},
              'ntuples': [tree_schema(p) for p in ntuples], 'code_sha256': sha256(__file__)}
    if tree:
        parts = []
        for path in ntuples:
            with uproot.open(path, timeout=60) as root:
                tr = root[tree]
                keys = tr.keys()
                columns = {}
                for name in IDS:
                    candidates = [k for k in keys if k in (name, 'evtinfo.' + name, 'evtinfo/' + name)]
                    if len(candidates) != 1:
                        raise ValueError(f'ambiguous/missing {name} branch')
                    columns[name] = tr[candidates[0]].array(library='np')
                for name in ('nprotons', 'pbtime'):
                    candidates = [k for k in keys if k in ('evtinfomc.' + name, 'evtinfomc/' + name)]
                    if len(candidates) != 1:
                        raise ValueError(f'ambiguous/missing MC {name} branch')
                    columns[name] = tr[candidates[0]].array(library='np')
                for name in ('nprotons', 'pbtime'):
                    candidates = [k for k in keys if k in ('evtinfo.' + name, 'evtinfo/' + name)]
                    if len(candidates) == 1:
                        columns['measured_' + name] = tr[candidates[0]].array(library='np')
                parts.append(pd.DataFrame(columns))
        exposure = pd.concat(parts, ignore_index=True)
        ledger = out.with_suffix('.parquet')
        if ledger.exists():
            raise ValueError('choose a new exposure ledger')
        selected = ak.from_parquet(selection, columns=IDS + ['t_mid', 'p_mid'])
        mask = (selected.p_mid > pmin) & (selected.p_mid < pmax)
        selected = ak.with_field(selected, selected.t_mid[mask], 't_mid')
        selected = selected[ak.num(selected.t_mid) > 0]
        data = pd.DataFrame({k: ak.to_numpy(selected[k]) for k in IDS})
        data['t_mid'] = ak.to_arrow_table(selected[['t_mid']], extensionarray=False).to_pandas()['t_mid']
        report['momentum_window'] = {'pmin': pmin, 'pmax': pmax, 'strict': True,
                                     'selected_events': len(selected), 'selected_tracks': int(ak.sum(ak.num(selected.t_mid)))}
        report['diagnostic'] = joined_diagnostic(data, exposure)
        for name in ('nprotons', 'pbtime'):
            measured = 'measured_' + name
            if measured in exposure:
                report.setdefault('measured_field_checks', {})[name] = {
                    'unique_values': int(exposure[measured].nunique()),
                    'min': float(exposure[measured].min()), 'max': float(exposure[measured].max()),
                    'equals_mc': bool(np.array_equal(exposure[measured], exposure[name]))}
        out.parent.mkdir(parents=True, exist_ok=True)
        exposure.to_parquet(ledger, index=False)
        report['exposure_ledger'] = {'path': str(ledger), 'sha256': sha256(ledger)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--ntuples', nargs='*', default=[])
    parser.add_argument('--ntuple-list', help='known paths, no namespace discovery')
    parser.add_argument('--tree', help='explicit tree to analyze after reviewing its schema')
    parser.add_argument('--out', required=True)
    parser.add_argument('--pmin', type=float, default=85.)
    parser.add_argument('--pmax', type=float, default=200.)
    args = parser.parse_args(argv)
    paths = args.ntuples + (Path(args.ntuple_list).read_text().splitlines() if args.ntuple_list else [])
    run(args.selection, paths, args.out, args.tree, args.pmin, args.pmax)


if __name__ == '__main__':
    main()
