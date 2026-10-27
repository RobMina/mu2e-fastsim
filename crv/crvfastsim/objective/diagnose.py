"""Read-only conditional PE and module timing diagnosis of saved paired response accumulators."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from . import bars, crvresponse as R, validation as V


LIMITS = [
    'All summaries are conditional on saved in-range histogram entries; underflow/overflow were not saved.',
    'Quantiles interpolate within bins and moments use bin centers; these are not unbinned estimates.',
    'PE-along uses measured CRV-line position and path-corrected pulse PE, not truth position.',
    'PE-path uses measured line path factor and raw pulse PE; neither PE histogram resolves modules.',
    'Module timing uses selected-cluster minus EX time in [-40,40] ns; objective dt uses [-25,25] ns.',
    'Cluster matching already requires |dt|<20 ns; no conclusion about rejected clusters is possible.',
    'Standardization is diagnostic reweighting of saved distributions, not a response or timing correction.',
    'No joint timing-versus-position/PE or event-level pairing is saved; causal attribution is unavailable.',
    'Common random seeds do not guarantee photon/background draws stay aligned when response knobs change.',
    'No paired-difference significance or noise-model fit is estimated; the saved objective is not changed.',
]


def histogram(value, shape, label):
    h = np.asarray(value, dtype=float)
    if h.shape != shape or not np.all(np.isfinite(h)) or np.any(h < 0):
        raise ValueError(f'invalid histogram {label}: expected {shape}, got {h.shape}')
    return h


def quantile(h, edges, q):
    total = float(h.sum())
    if total == 0:
        return None
    c = np.cumsum(h)
    i = min(int(np.searchsorted(c, q * total, side='left')), len(h) - 1)
    prev = c[i - 1] if i else 0.0
    return float(edges[i] + (edges[i + 1] - edges[i]) * (q * total - prev) / h[i])


def moments(h, edges):
    h, edges = np.asarray(h, float), np.asarray(edges, float)
    n = float(h.sum())
    if not n:
        return dict(entries=0.0, mean=None, rms=None, q16=None, q50=None, q84=None)
    centers = (edges[1:] + edges[:-1]) / 2
    mean = float(h @ centers / n)
    var = float(h @ ((centers - mean) ** 2) / n)
    return dict(entries=n, mean=mean, rms=float(np.sqrt(max(var, 0.0))),
                **{f'q{int(q * 100)}': quantile(h, edges, q) for q in (0.16, 0.5, 0.84)})


def normalized(h):
    return h / h.sum() if h.sum() else np.zeros_like(h)


def distribution(h, edges, ref):
    stats = moments(h, edges)
    stats['probability'] = normalized(h).tolist()
    stats['tv_to_data'] = float(np.abs(normalized(h) - normalized(ref)).sum() / 2) if h.sum() and ref.sum() else None
    return stats


def conditional(states, edges, min_entries):
    """Compare conditional shapes at common support using data's fixed cell fractions."""
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError('invalid conditional histogram edges')
    if set(states) != {'data', 'nominal', 'best'} or type(min_entries) is not int or min_entries <= 0:
        raise ValueError('conditional comparison requires data/nominal/best and positive integer min_entries')
    shape = np.asarray(states['data']).shape
    if len(shape) < 2 or shape[-1] != len(edges) - 1:
        raise ValueError('invalid conditional histogram shape')
    arrays = {k: histogram(h, shape, k).reshape(-1, len(edges) - 1) for k, h in states.items()}
    counts = {k: h.sum(axis=1) for k, h in arrays.items()}
    common = np.logical_and.reduce([n >= min_entries for n in counts.values()])
    ref = arrays['data']
    cells = []
    for i in range(len(ref)):
        cells.append(dict(index=i, common_support=bool(common[i]), samples={
            k: dict(distribution(h[i], edges, ref[i]),
                    population_fraction=float(counts[k][i] / counts[k].sum()) if counts[k].sum() else None)
            for k, h in arrays.items()}))
    weights = counts['data'] * common
    weights = normalized(weights)
    standardized = {}
    if common.any():
        mixtures = {k: np.sum(np.divide(h, counts[k][:, None], out=np.zeros_like(h),
                                       where=counts[k][:, None] > 0) * weights[:, None], axis=0)
                    for k, h in arrays.items()}
        for k, h in mixtures.items():
            standardized[k] = distribution(h, edges, mixtures['data'])
            standardized[k].pop('entries')
    support_marginal, decomposition = {}, {}
    if common.any():
        selected = {k: h[common] for k, h in arrays.items()}
        support_marginal = {k: distribution(h.sum(axis=0), edges, selected['data'].sum(axis=0))
                            for k, h in selected.items()}
        centers = (np.asarray(edges)[1:] + np.asarray(edges)[:-1]) / 2
        means = {k: h @ centers / h.sum(axis=1) for k, h in selected.items()}
        fractions = {k: normalized(h.sum(axis=1)) for k, h in selected.items()}
        for old, new in (('nominal', 'best'), ('data', 'nominal'), ('data', 'best')):
            w0, w1, m0, m1 = fractions[old], fractions[new], means[old], means[new]
            within = float(((w0 + w1) / 2 * (m1 - m0)).sum())
            population = float(((m0 + m1) / 2 * (w1 - w0)).sum())
            decomposition[f'{new}_minus_{old}'] = dict(
                total=float(w1 @ m1 - w0 @ m0), within_cell=within, cell_population=population)
    return dict(support_marginal=support_marginal, mean_decomposition=decomposition,
                min_entries=min_entries, edges=np.asarray(edges).tolist(), cells=cells,
                common_cells=int(common.sum()), cell_count=len(common),
                common_support_fraction={k: float(n[common].sum() / n.sum()) if n.sum() else None
                                         for k, n in counts.items()},
                data_cell_weights=weights.tolist(), standardized=standardized,
                marginal={k: distribution(h.sum(axis=0), edges, ref.sum(axis=0)) for k, h in arrays.items()})


def validate_state(state):
    for name, axes in R.SHAPES.items():
        for slot in R.SLOTS_OF.get(name, R.SLOTS):
            histogram(state['h'][f'{name}_{slot}'], R._shape(name), f'{name}_{slot}')
    histogram(state['dtp'], (R.EX_BODIES, R.N_BODY, len(R.DTP_EDGES) - 1), 'dtp')
    for key in ('events', 'lines_z', 'lines_x', 'lines_xz'):
        if type(state[key]) is not int or state[key] < 0:
            raise ValueError(f'invalid state count {key}')
    if state['events'] == 0:
        raise ValueError('empty state exposure')


def score_bins(h, ref, n0):
    """Decompose the saved deterministic bias-corrected term without uncertainty resampling."""
    h, ref = np.asarray(h, float), np.asarray(ref, float)
    n, nr = h.sum(), ref.sum()
    ndf = int(np.count_nonzero(h + ref) - 1)
    if n <= 0 or nr <= 0 or ndf <= 0:
        return None
    p, q = h / n, ref / nr
    numerator = (p - q) ** 2 - h / n ** 2 - ref / nr ** 2
    terms = np.divide(numerator, p + q, out=np.zeros_like(h), where=(p + q) > 0) * n0 / ndf
    return dict(ndf=ndf, score=float(1 + terms.sum()), contribution=terms.tolist())


def supplement_timing(reference, calibration_path, directory):
    """Add only absent timing diagnostics after exact equality of every saved scoring observable."""
    calibration = json.loads(Path(calibration_path).read_text())
    names = calibration['data_files']
    if not names or len(set(names)) != len(names) or any(Path(n).name != n for n in names):
        raise ValueError('timing supplement requires unique plain filenames from the fixed calibration')
    records, states = [], []
    for name in names:
        path = Path(directory) / name
        sha = V.digest(path)
        state = json.loads(path.read_text())
        validate_state(state)
        if V.digest(path) != sha:
            raise ValueError('supplement changed during read')
        records.append(dict(path=str(path), sha256=sha))
        states.append(state)
    merged = R.merge_states(states)
    keys = ('events', 'files', 'exposure_ns', 'lines_z', 'lines_x', 'lines_xz', 'h', 'oot_n')
    for key in keys:
        if merged[key] != reference[key]:
            raise ValueError(f'timing supplement differs from frozen scoring reference: {key}')
    import copy
    updated = copy.deepcopy(reference)
    for key in ('dtp', 'ftp'):
        if np.any(reference[key]) and reference[key] != merged[key]:
            raise ValueError(f'cannot replace nonempty timing reference: {key}')
        updated[key] = merged[key]
    if not np.any(updated['dtp']):
        raise ValueError('timing supplement has no module-pair entries')
    for row in records:
        if V.digest(row['path']) != row['sha256']:
            raise ValueError('supplement changed during merge')
    return updated, dict(files=records, matched_scoring_fields=list(keys),
                         calibration=dict(path=str(calibration_path), sha256=V.digest(calibration_path)),
                         replaced_diagnostic_fields=['dtp', 'ftp'],
                         caution='Exact accumulated-observable equality, not a new event-identity audit; no scoring reference changed.')


def summarize_block(states, min_entries, n0=R.N0):
    for state in states.values():
        validate_state(state)
    out = dict(exposure={k: {field: s[field] for field in ('events', 'lines_z', 'lines_x', 'lines_xz')}
                         for k, s in states.items()}, sectors={})
    body_groups = {'EX': list(range(4)), 'T1': [4, 5], 'T2': [6, 7], 'TAG': list(range(8, 16))}
    for slot in R.SLOTS:
        scalar = {}
        for name, axes in R.SHAPES.items():
            if len(axes) != 1 or slot not in R.SLOTS_OF.get(name, R.SLOTS):
                continue
            h = {k: np.asarray(s['h'][f'{name}_{slot}'], float) for k, s in states.items()}
            scalar[name] = dict(edges=R.E[axes[0]].tolist(), samples={
                k: distribution(a, R.E[axes[0]], h['data']) for k, a in h.items()})
            terms = {k: score_bins(h[k], h['data'], n0) for k in ('nominal', 'best')}
            if all(terms.values()):
                delta = np.asarray(terms['best']['contribution']) - np.asarray(terms['nominal']['contribution'])
                scalar[name]['score_decomposition'] = dict(n0=n0, samples=terms,
                    best_minus_nominal=float(delta.sum()), bin_best_minus_nominal=delta.tolist())
        along = conditional({k: s['h'][f'pe_along_{slot}'] for k, s in states.items()}, R.E['pe'], min_entries)
        for row in along['cells']:
            side, i = divmod(row['index'], len(R.E['along']) - 1)
            row.update(side=side, along_low=float(R.E['along'][i]), along_high=float(R.E['along'][i + 1]))
        path = conditional({k: s['h'][f'pe_path_{slot}'] for k, s in states.items()}, R.E['pe'], min_entries)
        for row in path['cells']:
            i = row['index']
            row.update(path_low=float(R.E['path'][i]), path_high=float(R.E['path'][i + 1]))
        entry = dict(scalar=scalar, pe_along=along, pe_path=path)
        if slot != 'EX':
            bodies = body_groups[slot]
            dtp = conditional({k: np.asarray(s['dtp'])[:, bodies, :] for k, s in states.items()},
                              R.DTP_EDGES, min_entries)
            for row in dtp['cells']:
                e, m = divmod(row['index'], len(bodies))
                row.update(ex_module=bars.BODY_NAMES[e], module=bars.BODY_NAMES[bodies[m]])
            entry['module_timing'] = dtp
        out['sectors'][slot] = entry
    return out


def code_identity():
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    paths = [Path(__file__), Path(__file__).with_name('diagnose_plots.py'), Path(V.__file__), Path(R.__file__)]
    return dict(path=str(root), head=head, dirty=bool(status),
                source_sha256={str(p.relative_to(root)): V.digest(p) for p in paths},
                versions={k: importlib.metadata.version(k) for k in ('numpy', 'awkward', 'matplotlib')})


def diagnose(directory, out, min_entries=100, plots=True, timing_dir=None):
    if type(min_entries) is not int or min_entries <= 0:
        raise ValueError('min_entries must be a positive integer')
    dest = Path(out)
    if dest.exists():
        raise FileExistsError(dest)
    root = Path(directory)
    with tempfile.TemporaryDirectory(prefix='crv-diagnose-') as tmp:
        pairing = V.paired(root, Path(tmp) / 'paired.json')
    points = {f'{b}_{a}': {k: json.loads((root / f'{b}_{a}' / f'{k}.json').read_text())
                          for k in ('r3', 'comparison', 'provenance')}
              for b in ('a', 'b') for a in ('nominal', 'best')}
    refspec = points['a_nominal']['provenance']['manifest']['assets']['reference']
    if V.digest(refspec['path']) != refspec['sha256']:
        raise ValueError('data reference hash mismatch')
    reference = json.loads(Path(refspec['path']).read_text())
    for point in points.values():
        p = point['provenance']
        if p['manifest']['assets']['reference'] != refspec:
            raise ValueError('data reference identity mismatch')
    fixed = {k: points['a_nominal']['provenance'][k] for k in ('code', 'runtime_assets', 'python', 'versions')}
    for name, point in points.items():
        p = point['provenance']
        if any(p[k] != v for k, v in fixed.items()):
            raise ValueError(f'{name}: cross-block runtime identity mismatch')
        if point['r3']['knobs'] != points['a_' + name.split('_', 1)[1]]['r3']['knobs']:
            raise ValueError(f'{name}: cross-block knobs mismatch')
    supplement = None
    if timing_dir is not None:
        timespec = points['a_nominal']['provenance']['manifest']['assets']['time_shift']
        if V.digest(timespec['path']) != timespec['sha256']:
            raise ValueError('fixed time calibration hash mismatch')
        reference, supplement = supplement_timing(reference, timespec['path'], timing_dir)
        if V.digest(timespec['path']) != timespec['sha256']:
            raise ValueError('fixed time calibration changed during diagnosis')
    report = dict(schema=1, kind='saved-response-diagnosis', limits=LIMITS, timing_supplement=supplement,
                  diagnostic_code=code_identity(), reference=refspec, frozen=points['a_nominal']['provenance'],
                  paired_receipt=pairing, blocks={})
    for block in ('a', 'b'):
        states = dict(data=reference, **{arm: points[f'{block}_{arm}']['r3']['response_state']
                                       for arm in ('nominal', 'best')})
        scores = {arm: points[f'{block}_{arm}']['comparison'] for arm in ('nominal', 'best')}
        report['blocks'][block] = summarize_block(states, min_entries, scores['nominal'].get('n0', R.N0))
        for slot, sector in report['blocks'][block]['sectors'].items():
            for name, scalar in sector['scalar'].items():
                key = f'{name}_{slot}'
                if key not in scores['nominal']['terms'] or 'score_decomposition' not in scalar:
                    continue
                for arm in ('nominal', 'best'):
                    value = scalar['score_decomposition']['samples'][arm]['score']
                    if not np.isclose(value, scores[arm]['terms'][key]['chi2ndf'], rtol=1e-10, atol=1e-10):
                        raise ValueError(f'bin decomposition does not reproduce saved score: {block}/{arm}/{key}')
    for name, point in points.items():
        for key in ('r3', 'comparison', 'provenance'):
            expected = pairing['inputs'][name]['files'][key]['sha256']
            if V.digest(root / name / f'{key}.json') != expected:
                raise ValueError('saved point changed during diagnosis')
    if V.digest(refspec['path']) != refspec['sha256']:
        raise ValueError('data reference changed during diagnosis')
    dest.mkdir(parents=True, exist_ok=False)
    with (dest / 'diagnosis.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    if plots:
        from .diagnose_plots import plot_report
        plot_report(report, dest)
    return report


def print_summary(report):
    for block, data in report['blocks'].items():
        print(f'=== block {block}: T1 in-range summaries (binned, no significance)')
        s = data['sectors']['T1']
        for name in ('pe_layer', 'tspread', 'dt_cluster'):
            row = s['scalar'][name]['samples']
            print(name, 'data/nominal/best q50:', *(round(row[k]['q50'], 4) for k in ('data', 'nominal', 'best')))
        for name in ('pe_along', 'pe_path', 'module_timing'):
            item = s[name]
            print(name, 'common cells', item['common_cells'], '/', item['cell_count'],
                  'coverage', item['common_support_fraction'])
            print('  support-matched marginal/standardized TV nominal,best:',
                  [(item['support_marginal'].get(k, {}).get('tv_to_data'),
                    item['standardized'].get(k, {}).get('tv_to_data')) for k in ('nominal', 'best')])
            print('  mean shift decomposition:', item['mean_decomposition'].get('best_minus_nominal'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True, help='four frozen CPU replica directories')
    p.add_argument('--out', required=True, help='new diagnostic directory; never overwrite')
    p.add_argument('--min-entries', type=int, default=100, help='per sample/cell common-support threshold')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--timing-dir', help='supplement absent data timing from calibrated file list after exact scoring-state equality')
    a = p.parse_args()
    report = diagnose(a.directory, a.out, a.min_entries, not a.no_plots, a.timing_dir)
    print_summary(report)
    print(f'wrote {a.out}/diagnosis.json')


if __name__ == '__main__':
    main()
