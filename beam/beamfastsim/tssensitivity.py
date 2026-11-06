"""One joint-tightening all-charge Run1A development check; reuse the saved baseline."""
import argparse
import json
from pathlib import Path

import numpy as np

from .tscompare import BINS, MUONS, _charge, _hist, _keys, _load, _paired, _shards, compare, plot
from .tsdiagnose import FATES, _join
from .tsstudy import cohort

SEED = 20261105
BASELINE = (2.5, .25, .05)
TIGHTER = (1.25, .125, .025)
EXPECTED = 25078
EXPECTED_EVENTS = 500000
EXPOSURE = ('event_id', 'parent_shard_ids', 'parent_shard_event_counts')
WARNINGS = [
    'One prespecified joint tightening, not a convergence sequence or a seed ensemble.',
    'Changing steps changes RNG consumption; identical seed labels are not common-random histories.',
    'Paired independent-shard errors include outcome variability; one realization per setting cannot isolate numerical bias.',
    'Field/material/loss caps change jointly; no attribution to one control.',
    'Spectral diagonal chi2 includes yield and ignores bin covariance; descriptive, not a calibrated significance.',
    'No yield-based setting selection, held-out access or default promotion.',
]


def _difference(b, c, n):
    p = _paired(b, c, n)
    return {k.replace('stock', 'baseline').replace('fast', 'tighter'): v for k, v in p.items()}


def _exposure(reference, fast):
    for key in EXPOSURE:
        if key in fast and not np.array_equal(reference[key], fast[key]):
            raise ValueError('saved fast exposure differs from reference: ' + key)


def baseline_record(path, arrays):
    """Check the original transport log; material caps are the documented baseline defaults."""
    text = Path(path).with_suffix('.log').read_text()
    start = text.index('{')
    record, _ = json.JSONDecoder().raw_decode(text[start:])
    fates = dict(zip(FATES, np.bincount(arrays['fate'], minlength=len(FATES)).tolist()))
    if (record['seed'] != SEED or record['step_mm'] != BASELINE[0]
            or record['n'] != len(arrays['id']) or record['seconds'] != float(arrays['seconds'])
            or record['fates'] != fates):
        raise ValueError('baseline log differs from prescribed seed/field step or saved output')
    return dict(path=str(Path(path).with_suffix('.log').resolve()),
                material_caps_source='published baseline command used transport defaults 0.25mm/5%; not embedded in NPZ')


def saved_candidate(path, reference, baseline):
    """Read the companion run record before assigning this diagnostic's settings labels."""
    record = json.loads(Path(path).with_name('run.json').read_text())
    if (Path(record['candidate']).resolve() != Path(path).resolve()
            or Path(record['baseline']).resolve() != Path(baseline).resolve()
            or Path(record['reference']).resolve() != Path(reference).resolve()
            or record['seed'] != SEED or record['baseline_settings'] != list(BASELINE)
            or record['tighter_settings'] != list(TIGHTER)):
        raise ValueError('saved candidate run record differs from prescribed comparison')
    return _load(path)


def summarize(reference, baseline, tighter):
    """Exact-ID populations, paired-shard setting differences and stock comparisons."""
    r = _load(reference)
    cohort(r)
    if len(r['id']) != EXPECTED or len(r['event_id']) != EXPECTED_EVENTS:
        raise ValueError('complete all-charge development reference and event census required')
    b, c = _join(r, _load(baseline)), _join(r, _load(tighter))
    _exposure(r, b)
    _exposure(r, c)
    row_units, units = _shards(r)
    group = np.searchsorted(_keys(units), _keys(row_units))
    k, q = len(units), _charge(r['pdg'])
    count = lambda mask: np.bincount(group[mask], minlength=k)
    out = dict(ninput=len(q), parent_events=len(r['event_id']), independent_shards=units.tolist(),
               baseline_settings=list(BASELINE), tighter_settings=list(TIGHTER), seed=SEED,
               warnings=WARNINGS, setting_difference_definition='tighter minus baseline; ratio tighter/baseline',
               baseline_vs_stock=compare(r, b, geometry_label='Run1A'),
               tighter_vs_stock=compare(r, c, geometry_label='Run1A'), charges={})
    for charge in (-1, 1):
        mask = q == charge
        n = count(mask)
        bs, cs = mask & (b['fate'] == 1), mask & (c['fate'] == 1)
        result = dict(ninput=int(mask.sum()), ninput_shard=n.tolist(), ds=_difference(count(bs), count(cs), n),
                      fates={arm: dict(zip(FATES, np.bincount(f['fate'][mask], minlength=len(FATES)).tolist()))
                             for arm, f in (('baseline', b), ('tighter', c))},
                      ds_species={arm: {str(pdg): int(np.sum(mask & (f['fate'] == 1) & (f['pdg'] == pdg)))
                                        for pdg in (-211, -13, 13, 211)} for arm, f in (('baseline', b), ('tighter', c))},
                      capped_ids={arm: r['id'][mask & (f['fate'] == 6)].tolist() for arm, f in (('baseline', b), ('tighter', c))},
                      fate_transition_labels=list(FATES),
                      fate_transition=np.bincount(len(FATES) * b['fate'][mask].astype(int) + c['fate'][mask],
                          minlength=len(FATES)**2).reshape(len(FATES), len(FATES)).tolist(), muons={})
        for name, pdg in MUONS.items():
            bm, cm = bs & (b['pdg'] == pdg), cs & (c['pdg'] == pdg)
            m = _difference(count(bm), count(cm), n)
            m.update(both=int((bm & cm).sum()), baseline_only=int((bm & ~cm).sum()),
                     tighter_only=int((~bm & cm).sum()), histograms={})
            for variable, edges in BINS.items():
                bv, cv = ((np.linalg.norm(b['p'], axis=1), np.linalg.norm(c['p'], axis=1)) if variable == 'momentum'
                          else (b['t'] - r['t'], c['t'] - r['t']))
                bh, bf = _hist(bv, bm, group, k, edges)
                ch, cf = _hist(cv, cm, group, k, edges)
                h = _difference(bh, ch, n)
                pull = np.asarray(h['pull'], float)
                used = (bh.sum(0) + ch.sum(0)) > 0
                chi2 = float(np.sum(pull[used]**2)) if np.all(np.isfinite(pull[used])) else None
                h.update(edges=edges.tolist(), baseline_shard=bh.tolist(), tighter_shard=ch.tolist(),
                         baseline_flow=bf, tighter_flow=cf, chi2_diagonal=chi2, chi2_bins=int(used.sum()))
                m['histograms'][variable] = h
            result['muons'][name] = m
        out['charges'][str(charge)] = result
    return out


def plot_settings(result, path):
    """Three-arm spectra and paired tighter-minus-baseline pulls, with unchanged bins."""
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
                    (h['baseline'], 'baseline', '#333333'), (h['tighter'], 'joint tighter', '#D55E00')):
                ax.stairs(np.asarray(values) / charge['ninput'], edges, label=label, color=color)
            pa.plot((edges[1:] + edges[:-1]) / 2, np.asarray(h['pull'], float), color='#333333')
            bad = np.asarray(h['zero_variance_nonzero_difference'], bool)
            if bad.any():
                pa.scatter(((edges[1:] + edges[:-1]) / 2)[bad], np.sign(np.asarray(h['difference'])[bad]) * 5,
                           marker='x', color='#D55E00', label='undefined SE')
                pa.legend(fontsize=7)
            pa.axhline(0, color='gray', linewidth=.7)
            pa.set_ylabel('setting\npull')
            pa.set_xlabel('Momentum [MeV/c]' if variable == 'momentum' else 'Elapsed time [ns]')
            ax.set_title(f"{name}: stock {stock['stock']}, baseline {m['baseline']}, tighter {m['tighter']}; N={charge['ninput']}")
            ax.set_ylabel('DS muons / source-charge input / bin')
            ax.tick_params(labelbottom=False)
            ax.legend(fontsize=8)
    fig.suptitle('Run1A DEVELOPMENT: one joint-tightening sensitivity check', fontsize=14)
    fig.text(.5, .02, 'Paired independent-shard setting pulls; spectral chi2 descriptive only.\n'
             'Different RNG consumption; not common-random trajectories or proof of convergence. Flow counts retained in JSON.',
             ha='center', fontsize=9)
    fig.subplots_adjust(top=.91, bottom=.14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('reference', 'baseline', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--candidate', type=Path, help='summarize an existing tighter NPZ; no transport')
    for key in ('gdml', 'field-config', 'offline-root', 'maps'):
        parser.add_argument('--' + key, type=Path)
    args = parser.parse_args(argv)
    inputs = ('gdml', 'field_config', 'offline_root', 'maps')
    if not args.candidate and any(getattr(args, key) is None for key in inputs):
        parser.error('a new tighter run requires all geometry/field paths')
    if args.candidate and any(getattr(args, key) is not None for key in inputs):
        parser.error('saved summary accepts no geometry/field arguments')
    if args.out.exists():
        parser.error('fresh output directory required')
    r, b = _load(args.reference), _load(args.baseline)
    cohort(r)
    if len(r['id']) != EXPECTED or len(r['event_id']) != EXPECTED_EVENTS or not np.array_equal(r['id'], b['id']):
        raise ValueError('complete reference/exposure and baseline in original input order required')
    _join(r, b)
    _exposure(r, b)
    baseline_info = baseline_record(args.baseline, b)
    c = saved_candidate(args.candidate, args.reference, args.baseline) if args.candidate else None
    if c is not None:
        _exposure(r, c)
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = dict(reference=str(args.reference.resolve()), baseline=str(args.baseline.resolve()),
                      candidate=str(args.candidate.resolve()) if args.candidate else str((args.out / 'tighter.npz').resolve()),
                      inputs={key: str(getattr(args, key).resolve()) for key in inputs if getattr(args, key) is not None},
                      baseline_settings=list(BASELINE), tighter_settings=list(TIGHTER), seed=SEED,
                      baseline_record=baseline_info, warnings=WARNINGS)
    with (args.out / 'run.json').open('x') as handle:
        json.dump(provenance, handle, indent=2, allow_nan=False)
    if c is None:
        from . import tsfield, tsgeometry, tstransport
        field = tsfield.from_config(args.field_config, args.offline_root, args.maps)
        geometry = tsgeometry.Geometry(args.gdml)
        step, material, loss = TIGHTER
        c = tstransport.transport({key: r[key].copy() for key in ('x', 'p', 't', 'pdg')}, field, geometry,
                                 seed=SEED, step_mm=step, material_step_mm=material, loss_fraction=loss,
                                 max_path_mm=100000., method='rk4')
        c.update({key: r[key] for key in ('id', 'shard', 'event_id', 'parent_shard_ids', 'parent_shard_event_counts')})
        with (args.out / 'tighter.npz').open('xb') as handle:
            np.savez_compressed(handle, **c)
    result = summarize(r, b, c)
    result['provenance'] = provenance
    with (args.out / 'summary.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    plot(result['tighter_vs_stock'], args.out / 'tighter-vs-stock.png')
    plot_settings(result, args.out / 'settings.png')
    print(json.dumps(dict(ninput=result['ninput'], parent_events=result['parent_events'],
                         baseline_seconds=float(b['seconds']), tighter_seconds=float(c['seconds']),
                         baseline={q: result['baseline_vs_stock']['charges'][q]['headline'] for q in ('-1', '1')},
                         tighter={q: result['tighter_vs_stock']['charges'][q]['headline'] for q in ('-1', '1')})), flush=True)


if __name__ == '__main__':
    main()
