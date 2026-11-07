"""Compare two saved numerical-setting outputs using paired independent shards."""
import argparse
import json
from pathlib import Path

import numpy as np

from .tscompare import BINS, MUONS, _charge, _hist, _keys, _load, _paired, _shards, compare
from .tsdiagnose import FATES, _join

EXPOSURE = ('event_id', 'parent_shard_ids', 'parent_shard_event_counts')


def _difference(b, c, n):
    p = _paired(b, c, n)
    return {k.replace('stock', 'baseline').replace('fast', 'candidate'): v for k, v in p.items()}


def _exposure(table):
    """Validate the event census, parent inventory and source-event membership."""
    for key in EXPOSURE:
        if key not in table:
            raise ValueError('missing exposure: ' + key)
    for key, width in (('event_id', 3), ('parent_shard_ids', 2)):
        a = table[key]
        if (a.ndim != 2 or a.shape[1] != width or a.dtype.kind not in 'iu'
                or np.any(a > np.iinfo(np.int64).max) or len(np.unique(_keys(a))) != len(a)):
            raise ValueError('invalid/duplicate exposure: ' + key)
    _shards(table)
    events, parents = _keys(table['event_id']), _keys(table['parent_shard_ids'])
    event_units = _keys(table['event_id'][:, :2])
    if not np.all(np.isin(event_units, parents)):
        raise ValueError('event exposure absent from parent inventory')
    order = np.argsort(parents)
    groups = np.searchsorted(parents[order], event_units)
    counts = np.bincount(groups, minlength=len(parents))
    if not np.array_equal(counts, table['parent_shard_event_counts'][order]):
        raise ValueError('event exposure and parent counts disagree')
    if not np.all(np.isin(_keys(table['id'][:, :3]), events)):
        raise ValueError('source event IDs absent from exposure')
    return (np.sort(events), parents[order], counts)


def summarize(reference, baseline, candidate):
    """Exact-ID populations, paired-shard setting differences and stock comparisons."""
    r = _load(reference)
    b, c = _join(r, _load(baseline)), _join(r, _load(candidate))
    exposure = _exposure(r)
    for table in (b, c):
        if not any(key in table for key in EXPOSURE):
            continue
        if any(not np.array_equal(a, v) for a, v in zip(exposure, _exposure(table))):
            raise ValueError('saved output exposure differs from reference')
    row_units, units = _shards(r)
    group = np.searchsorted(_keys(units), _keys(row_units))
    k, q = len(units), _charge(r['pdg'])
    count = lambda mask: np.bincount(group[mask], minlength=k)
    out = dict(ninput=len(q), parent_events=len(r['event_id']), independent_shards=units.tolist(),
               setting_difference_definition='candidate minus baseline; ratio candidate/baseline',
               baseline_vs_stock=dict(charges=compare(r, b)['charges']),
               candidate_vs_stock=dict(charges=compare(r, c)['charges']), charges={})
    for charge in (-1, 1):
        mask = q == charge
        n = count(mask)
        bs, cs = mask & (b['fate'] == 1), mask & (c['fate'] == 1)
        result = dict(ninput=int(mask.sum()), ninput_shard=n.tolist(), ds=_difference(count(bs), count(cs), n),
                      fates={arm: dict(zip(FATES, np.bincount(f['fate'][mask], minlength=len(FATES)).tolist()))
                             for arm, f in (('baseline', b), ('candidate', c))},
                      ds_species={arm: {str(pdg): int(np.sum(mask & (f['fate'] == 1) & (f['pdg'] == pdg)))
                                        for pdg in (-211, -13, 13, 211)} for arm, f in (('baseline', b), ('candidate', c))},
                      capped_ids={arm: r['id'][mask & (f['fate'] == 6)].tolist() for arm, f in (('baseline', b), ('candidate', c))},
                      fate_transition_labels=list(FATES),
                      fate_transition=np.bincount(len(FATES) * b['fate'][mask].astype(int) + c['fate'][mask].astype(int),
                          minlength=len(FATES)**2).reshape(len(FATES), len(FATES)).tolist(), muons={})
        for name, pdg in MUONS.items():
            bm, cm = bs & (b['pdg'] == pdg), cs & (c['pdg'] == pdg)
            m = _difference(count(bm), count(cm), n)
            m.update(both=int((bm & cm).sum()), baseline_only=int((bm & ~cm).sum()),
                     candidate_only=int((~bm & cm).sum()), histograms={})
            for variable, edges in BINS.items():
                bv, cv = ((np.linalg.norm(b['p'], axis=1), np.linalg.norm(c['p'], axis=1)) if variable == 'momentum'
                          else (b['t'] - r['t'], c['t'] - r['t']))
                bh, bf = _hist(bv, bm, group, k, edges)
                ch, cf = _hist(cv, cm, group, k, edges)
                h = _difference(bh, ch, n)
                pull = np.asarray(h['pull'], float)
                used = (bh.sum(0) + ch.sum(0)) > 0
                chi2 = float(np.sum(pull[used]**2)) if np.all(np.isfinite(pull[used])) else None
                h.update(edges=edges.tolist(), baseline_shard=bh.tolist(), candidate_shard=ch.tolist(),
                         baseline_flow=bf, candidate_flow=cf, chi2_diagonal=chi2, chi2_bins=int(used.sum()))
                m['histograms'][variable] = h
            result['muons'][name] = m
        out['charges'][str(charge)] = result
    return out


def plot_settings(result, path):
    """Three-arm spectra and paired candidate-minus-baseline pulls, with unchanged bins."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(2, 2, hspace=.45, wspace=.3)
    for row, (q, name) in enumerate((('-1', 'mu-'), ('1', 'mu+'))):
        charge = result['charges'][q]
        m = charge['muons'][name]
        stock = result['baseline_vs_stock']['charges'][q]['muons'][name]
        for col, variable in enumerate(BINS):
            sub = grid[row, col].subgridspec(2, 1, height_ratios=(3, 1), hspace=.06)
            ax = fig.add_subplot(sub[0])
            pa = fig.add_subplot(sub[1], sharex=ax)
            h = m['histograms'][variable]
            edges = np.asarray(h['edges'])
            for values, label, color in ((stock['histograms'][variable]['stock'], 'stock', '#0072B2'),
                    (h['baseline'], 'baseline', '#333333'), (h['candidate'], 'candidate', '#D55E00')):
                ax.stairs(np.asarray(values) / max(charge['ninput'], 1), edges, label=label, color=color)
            pa.plot((edges[1:] + edges[:-1]) / 2, np.asarray(h['pull'], float), color='#333333')
            bad = np.asarray(h['zero_variance_nonzero_difference'], bool)
            if bad.any():
                pa.scatter(((edges[1:] + edges[:-1]) / 2)[bad], np.sign(np.asarray(h['difference'])[bad]) * 5,
                           marker='x', color='#D55E00', label='undefined SE')
                pa.legend(fontsize=7)
            pa.axhline(0, color='gray', linewidth=.7)
            pa.set_ylabel('setting\npull')
            pa.set_xlabel('Momentum [MeV/c]' if variable == 'momentum' else 'Elapsed time [ns]')
            ax.set_title(f"{name}: stock {stock['stock']}, baseline {m['baseline']}, candidate {m['candidate']}; N={charge['ninput']}")
            ax.set_ylabel('DS muons / source-charge input / bin')
            ax.tick_params(labelbottom=False)
            ax.legend(fontsize=8)
    fig.suptitle('Saved-output numerical-setting comparison', fontsize=14)
    fig.text(.5, .02, 'Paired independent-shard setting pulls; spectral chi2 descriptive only.\n'
             'Flow counts retained in JSON; undefined zero-SE pulls marked x.',
             ha='center', fontsize=9)
    fig.subplots_adjust(top=.91, bottom=.14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('reference', 'baseline', 'candidate', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error('fresh output directory required')
    result = summarize(args.reference, args.baseline, args.candidate)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / 'summary.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    plot_settings(result, args.out / 'settings.png')
    print(json.dumps(dict(ninput=result['ninput'], parent_events=result['parent_events'],
                         baseline={q: result['baseline_vs_stock']['charges'][q]['headline'] for q in ('-1', '1')},
                         candidate={q: result['candidate_vs_stock']['charges'][q]['headline'] for q in ('-1', '1')})), flush=True)


if __name__ == '__main__':
    main()
