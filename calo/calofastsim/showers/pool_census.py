"""Exact frozen K8 eligibility census; no random draw, reconstruction or model change."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import placement_diagnostic as pd
from . import production_compare as pc
from . import residual_diagnostic as rd
from .compare import paired_mean
from .library import Binning, _radius_pool, cells, load
from .radius_trial import NEIGHBORS, SEED, RECO_SEED, digest, fingerprint


def donor_outcomes(library):
    """Whole-event both-disk Birks outcomes, including zero-step donors."""
    n = len(library['entry_energy'])
    offsets = library['offsets']
    if offsets.shape != (n + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError('invalid library offsets')
    energy = library['step_energy_birks']
    if len(energy) != offsets[-1] or np.any(~np.isfinite(energy) | (energy < 0)):
        raise ValueError('invalid library Birks energies')
    total = np.bincount(np.repeat(np.arange(n), np.diff(offsets)), weights=energy, minlength=n)
    return dict(energy=total, empty=np.diff(offsets) == 0, weak=total <= 10)


def enumerate_pools(library, entries, saved_draw):
    """Reproduce same-disk cell fallback and radius eligibility without invoking the sampler."""
    binning = Binning(*(library['bins_' + k] for k in ('energy', 'cosine', 'radius')))
    query, outside = cells(entries['energy'], entries['cosine'], entries['radius'], binning)
    donor_cell, donor_outside = cells(*(library['entry_' + k] for k in ('energy', 'cosine', 'radius')), binning)
    pd.exact('library_cells', donor_cell, library['entry_cell'])
    pd.exact('library_outside', donor_outside, library['entry_outside_bins'])
    labels = np.column_stack((library['entry_disk'], donor_cell))
    occupied, inverse = np.unique(labels, axis=0, return_inverse=True)
    count = np.bincount(inverse, minlength=len(occupied))
    n = len(query)
    eligible = np.full((n, NEIGHBORS), -1, np.int64)
    chosen = np.full(n, -1, np.int64)
    size, full_size, exact_size = (np.zeros(n, np.int64) for _ in range(3))
    fallback = outside.copy()
    for disk in np.unique(entries['disk']):
        where = np.flatnonzero(entries['disk'] == disk)
        pool = np.flatnonzero(occupied[:, 0] == disk)
        if not len(pool):
            fallback[where] = True
            continue
        distance = np.sum((query[where, None, :] - occupied[pool, 1:][None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distance, axis=1)
        chosen[where] = pool[nearest]
        full_size[where] = count[chosen[where]]
        exact = distance[np.arange(len(where)), nearest] == 0
        exact_size[where] = np.where(exact, full_size[where], 0)
        fallback[where] |= ~exact
        for cell in np.unique(chosen[where]):
            target = where[chosen[where] == cell]
            rows = np.flatnonzero(inverse == cell)
            k = min(NEIGHBORS, len(rows))
            size[target] = k
            eligible[target, :k] = rows if k == len(rows) else _radius_pool(rows, library['entry_radius'], entries['radius'][target], k)
    supported = chosen >= 0
    for name, value in dict(supported=supported, fallback=fallback, radius_pool_size=size, bin_pool_size=full_size).items():
        pd.exact('pool_' + name, value, saved_draw[name])
    donor = saved_draw['donor']
    if (donor.shape != (n,) or not np.array_equal(donor >= 0, supported) or
            np.any(~np.any(eligible[supported] == donor[supported, None], axis=1))):
        raise ValueError('saved donor is outside eligible pool')
    return dict(eligible=eligible, query_cell=np.column_stack((entries['disk'], query)), chosen_cell=chosen,
                occupied=occupied, library_cell=inverse, counts=count, size=size, full_size=full_size,
                exact_size=exact_size, supported=supported, fallback=fallback, outside=outside,
                substituted=supported & (exact_size == 0))


def coverage(library, entries, pools):
    """Radius and azimuth-rotated direction coverage of the actual K8 pools, not a fitted metric."""
    valid = pools['eligible'] >= 0
    query, slot = np.nonzero(valid)
    donor = pools['eligible'][query, slot]
    features = {name: np.full(valid.shape, np.nan) for name in ('radius_delta', 'angle_deg')}
    features['radius_delta'][query, slot] = library['entry_radius'][donor] - entries['radius'][query]
    ds = np.searchsorted(library['geometry_disk_disk'], entries['disk'][query])
    origin = np.column_stack([library['geometry_disk_origin_' + a] for a in 'xy'])[ds]
    target_xy = entries['position'][query, :2] - origin
    source_xy = library['entry_position'][donor, :2] - origin
    phi = np.arctan2(target_xy[:, 1], target_xy[:, 0]) - np.arctan2(source_xy[:, 1], source_xy[:, 0])
    direction = library['entry_direction'][donor].copy()
    x, y = direction[:, 0].copy(), direction[:, 1].copy()
    direction[:, 0], direction[:, 1] = np.cos(phi) * x - np.sin(phi) * y, np.sin(phi) * x + np.cos(phi) * y
    dot = np.sum(direction * entries['direction'][query], axis=1)
    features['angle_deg'][query, slot] = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))
    return features


def probabilities(values, pools):
    """Outcome means on each supported pool; unsupported stays NaN rather than becoming empty."""
    valid = pools['eligible'] >= 0
    gathered = np.where(valid, values[np.maximum(pools['eligible'], 0)], 0)
    return np.divide(gathered.sum(axis=1), pools['size'], out=np.full(len(valid), np.nan), where=pools['supported'])


def outcome_summary(probability, observed, original, rows):
    """Conditional independent-uniform draw spread, separate from source-block comparisons."""
    expected = float(probability.sum())
    variance = float(np.sum(probability * (1 - probability)))
    observed_count = int(observed.sum())
    return dict(queries=len(rows), original=int(original.sum()), saved_draw=observed_count,
                expected_count=expected, expected_fraction=float(probability.mean()) if len(rows) else None,
                conditional_draw_sd=float(np.sqrt(variance)),
                draw_minus_expected=observed_count - expected,
                draw_standardized_difference=(observed_count - expected) / np.sqrt(variance) if variance > 0 else None,
                expected_minus_original=paired_mean(probability, original, rows // 100),
                probability=pd.summarize(probability),
                deterministic_queries=int(((probability == 0) | (probability == 1)).sum()),
                zero_probability_queries=int((probability == 0).sum()),
                unit_probability_queries=int((probability == 1).sum()))


def region_summary(mask, library, entries, pools, outcomes, prob, original, saved_draw, cover):
    supported = mask & pools['supported']
    ids = pools['eligible'][supported]
    valid = ids >= 0
    weights = np.broadcast_to(1 / pools['size'][supported, None], ids.shape)
    exposure = np.bincount(ids[valid], weights=weights[valid], minlength=len(outcomes['energy']))
    with np.errstate(divide='ignore'):
        log_unused = np.bincount(ids[valid], weights=np.log1p(-weights[valid]), minlength=len(exposure))
    expected_unique = float(np.sum(-np.expm1(log_unused)))
    union = exposure > 0
    selected = np.bincount(saved_draw['donor'][supported], minlength=len(exposure))
    rows = entries['event_row'][supported]
    result = dict(queries=int(mask.sum()), supported=int(supported.sum()), unsupported=int((mask & ~pools['supported']).sum()),
        fallback=int((mask & pools['fallback']).sum()), exact_cell_empty=int((mask & (pools['exact_size'] == 0)).sum()),
        outside_bins=int((mask & pools['outside']).sum()), substituted=int((mask & pools['substituted']).sum()),
        expected_unique_drawn=expected_unique,
        unique_eligible_donors=int(union.sum()), eligible_empty=int((union & outcomes['empty']).sum()),
        eligible_weak=int((union & outcomes['weak']).sum()), unique_drawn_donors=int((selected > 0).sum()),
        maximum_draw_reuse=int(selected.max(initial=0)), maximum_expected_exposure=float(exposure.max(initial=0)),
        exposure_concentration=float(np.sum(exposure ** 2) / exposure.sum() ** 2) if exposure.sum() else None,
        cell_occupancy=pd.summarize(pools['full_size'][supported]), exact_cell_occupancy=pd.summarize(pools['exact_size'][mask]),
        radius_pool_size=pd.summarize(pools['size'][supported]), outcomes={}, coverage={})
    for kind in ('empty', 'weak'):
        observed = outcomes[kind][saved_draw['donor'][supported]]
        result['outcomes'][kind] = outcome_summary(prob[kind][supported], observed, original[kind][supported], rows)
        cell_mean = np.bincount(pools['library_cell'], weights=outcomes[kind], minlength=len(pools['counts'])) / pools['counts']
        result['outcomes'][kind]['full_cell_expected_count'] = float(cell_mean[pools['chosen_cell'][supported]].sum())
    result['outcomes']['energy'] = dict(expected_minus_original=paired_mean(prob['energy'][supported], original['energy'][supported], rows // 100),
        saved_minus_expected=paired_mean(outcomes['energy'][saved_draw['donor'][supported]], prob['energy'][supported], rows // 100))
    for name, value in cover.items():
        absolute = np.abs(value[supported])
        best = np.min(np.where(valid, absolute, np.inf), axis=1)
        worst = np.max(np.where(valid, absolute, -np.inf), axis=1)
        mean = np.sum(np.where(valid, absolute, 0), axis=1) / pools['size'][supported]
        result['coverage'][name] = dict(best=pd.summarize(best), worst=pd.summarize(worst), mean=pd.summarize(mean))
    result['exposure_by_outcome'] = {kind: float(exposure[outcomes[kind]].sum()) for kind in ('empty', 'weak')}
    return result, exposure, selected


def incident_coordinates(library, entries):
    """Disk-local radial/tangential direction and azimuth for coverage diagnostics only."""
    disk_ids = library['geometry_disk_disk']
    slot = np.searchsorted(disk_ids, entries['disk'])
    valid = (slot < len(disk_ids)) & (disk_ids[np.minimum(slot, len(disk_ids) - 1)] == entries['disk'])
    origin = np.full((len(slot), 2), np.nan)
    origin[valid] = np.column_stack([library['geometry_disk_origin_' + a] for a in 'xy'])[slot[valid]]
    xy = entries['position'][:, :2] - origin
    norm = np.linalg.norm(xy, axis=1)
    radial = np.divide(xy, norm[:, None], out=np.tile([1., 0.], (len(norm), 1)), where=norm[:, None] > 0)
    radial[~valid] = np.nan
    u = entries['direction']
    return dict(energy=entries['energy'], cosine=entries['cosine'], radius=entries['radius'],
        azimuth_deg=np.rad2deg(np.arctan2(xy[:, 1], xy[:, 0])),
        direction_radial=np.sum(radial * u[:, :2], axis=1),
        direction_tangent=radial[:, 0] * u[:, 1] - radial[:, 1] * u[:, 0])


def cell_census(library, entries, pools, outcomes, original):
    """Counts for every occupied query or library cell; no response-based acceptance cut."""
    labels = np.unique(np.vstack((pools['query_cell'], pools['occupied'])), axis=0)
    query_features = incident_coordinates(library, entries)
    donor_features = incident_coordinates(library, {k.removeprefix('entry_'): v for k, v in library.items() if k.startswith('entry_')})
    records = []
    for label in labels:
        query = np.all(pools['query_cell'] == label, axis=1)
        donor = np.all(np.column_stack((library['entry_disk'], library['entry_cell'])) == label, axis=1)
        record = dict(cell=label.tolist(), queries=int(query.sum()), donors=int(donor.sum()),
            original_empty=int((query & original['empty']).sum()), original_weak=int((query & original['weak']).sum()),
            donor_empty=int((donor & outcomes['empty']).sum()), donor_weak=int((donor & outcomes['weak']).sum()),
            fallback_queries=int((query & pools['fallback']).sum()),
            query_incident={k: pd.summarize(v[query]) for k, v in query_features.items()},
            donor_incident={k: pd.summarize(v[donor]) for k, v in donor_features.items()},
            weak_donor_incident={k: pd.summarize(v[donor & outcomes['weak']]) for k, v in donor_features.items()},
            strong_donor_incident={k: pd.summarize(v[donor & ~outcomes['weak']]) for k, v in donor_features.items()})
        records.append(record)
    return records


def diagnose(library, entries, draw, original, saved_clusters):
    outcomes = donor_outcomes(library)
    pools = enumerate_pools(library, entries, draw)
    cover = coverage(library, entries, pools)
    prob = {kind: probabilities(value, pools) for kind, value in outcomes.items()}
    n = len(entries['energy'])
    original_outcomes = dict(energy=pd.energy_sum(original, n), empty=np.bincount(original['entry'], minlength=n) == 0)
    original_outcomes['weak'] = original_outcomes['energy'] <= 10
    supported = pools['supported']
    pd.exact('saved_draw_energy', pd.energy_sum(draw, n)[supported], outcomes['energy'][draw['donor'][supported]])
    pd.exact('saved_draw_empty', (np.diff(draw['offsets']) == 0)[supported], outcomes['empty'][draw['donor'][supported]])
    masks = pd.strata(entries)
    for disk in (0, 1):
        mask = (entries['disk'] == disk) & (entries['radius'] < 450) & (entries['energy'] >= 95)
        masks[f'inner_highE_disk{disk}'] = mask
        for label, lo, hi in (('coslow', 0, .7), ('coshigh', .7, 1.000001)):
            masks[f'inner_highE_disk{disk}_{label}'] = mask & (entries['cosine'] >= lo) & (entries['cosine'] < hi)
    masks.update(fallback=pools['fallback'], exact_no_fallback=~pools['fallback'])
    regions, exposures, draws = {}, {}, {}
    for name, mask in masks.items():
        summary, exposure, selected = region_summary(mask, library, entries, pools, outcomes, prob, original_outcomes, draw, cover)
        rows = entries['event_row'][mask]
        summary['saved_selected'] = {arm: int((saved_clusters[arm + '_energy'][rows] > 10).sum())
                                    for arm in ('stockref', 'control', 'geometric', 'crystal')}
        regions[name], exposures[name], draws[name] = summary, exposure, selected
    census = cell_census(library, entries, pools, outcomes, original_outcomes)
    queried = [r for r in census if r['queries']]
    targets = {str(target): dict(queried_cells=len(queried), existing_in_queried_cells=sum(r['donors'] for r in queried),
        cells_below_target=sum(r['donors'] < target for r in queried),
        additional_donors=sum(max(target - r['donors'], 0) for r in queried),
        binomial_worst_case_se=.5 / np.sqrt(target)) for target in (100, 200, 400)}
    arrays = {k: v for k, v in pools.items()}
    arrays.update(event_row=entries['event_row'], **{'prob_' + k: v for k, v in prob.items()},
                  **{'pool_' + k: v for k, v in cover.items()})
    donors = {'outcome_' + k: v for k, v in outcomes.items()}
    donors.update({k: v for k, v in library.items() if k.startswith('entry_')})
    donors.update({'exposure_' + k: v for k, v in exposures.items()})
    donors.update({'draw_count_' + k: v for k, v in draws.items()})
    result = dict(neighbors=NEIGHBORS, library_donors=len(outcomes['energy']), library_empty=int(outcomes['empty'].sum()),
        library_weak=int(outcomes['weak'].sum()), regions=regions, cells=census, coverage_targets=targets,
        caveat='Exact fixed-library eligibility census, no new random draw or reconstruction. Empty/weak refer to '
        'whole-event both-disk Birks CSS before placement/timing, not leading-cluster selection. Unsupported is '
        'counted separately; outcome comparisons use supported queries. Conditional draw SD=sqrt(sum p(1-p)) '
        'assumes independent uniform choices conditional on fixed queries/library; overlapping pools do not make '
        'independent library observations. It excludes training uncertainty, original-sample uncertainty and '
        'population mismatch. Source-row//100 errors on expected-minus-original are a separate conditional '
        'diagnostic. No p-value/promotion; exposure concentration is not independent sample size. Coverage targets '
        'are planning scenarios for per-cell marginal Bernoulli precision, not a guarantee for local K8 pools, '
        'a required generated-event count, or a physics validation criterion.')
    return result, arrays, donors


def plot_report(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = ('all', 'radius_0_450', 'inner_highE_disk1', 'inner_highE_disk1_coshigh')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(names))
    for ax, outcome in zip(axes, ('empty', 'weak')):
        stats = [result['regions'][k]['outcomes'][outcome] for k in names]
        count = np.array([v['queries'] for v in stats], float)
        for field, offset, label in (('original', -.22, 'Original CSS'), ('saved_draw', 0, 'Saved K8 draw'),
                                      ('expected_count', .22, 'Exact K8 pool expectation')):
            y = np.divide([v[field] for v in stats], count, out=np.full(len(count), np.nan), where=count > 0)
            ax.bar(x + offset, y, .22, label=label)
            if field == 'expected_count':
                err = np.divide([v['conditional_draw_sd'] for v in stats], count,
                                out=np.full(len(count), np.nan), where=count > 0)
                ax.errorbar(x + offset, y, yerr=err, color='black', fmt='none', capsize=3)
        ax.set_xticks(x, ['All', 'r < 450', 'Disk1, inner\nE >= 95', 'Same\ncos >= .7'])
        ax.set_ylabel(outcome + ' CSS fraction')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    fig.suptitle('Frozen K8 pools: error bars are draw spread only, NOT library uncertainty')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('tables', 'library', 'trial', 'baseline', 'residual', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('output must be new or empty')
    receipt = json.loads((args.residual / 'numbers.json').read_text())
    paths = dict(tables=args.tables, library=args.library, clusters=args.baseline / 'clusters.npz')
    paths.update({name: args.trial / name for name in ('preparation.json', 'draw.npz', 'clusters.npz', 'css_arrays.npz', 'numbers.json')})
    for name, path in paths.items():
        if digest(path) != receipt['inputs'][name]['sha256']:
            raise ValueError('residual input hash mismatch: ' + name)
    manifest = json.loads(paths['preparation.json'].read_text())
    if (manifest['neighbors'], manifest['seed'], manifest['reco_seed']) != (NEIGHBORS, SEED, RECO_SEED):
        raise ValueError('not the fixed K8 trial')
    root = Path(__file__).resolve().parents[1]
    if digest(root / 'showers/library.py') != manifest['calo_source_sha256']['showers/library.py']:
        raise ValueError('sampler source changed since K8 trial')
    library, metadata = load(args.library)
    draw, saved = rd.archive(paths['draw.npz']), rd.archive(paths['clusters.npz'])
    prepared = rd.prepare(args.tables, library, metadata, saved, draw, manifest, rd.archive(paths['clusters']))
    checked, _ = rd.features(library, prepared, rd.archive(paths['css_arrays.npz']))
    residual_arrays = rd.archive(args.residual / 'arrays.npz')
    for name, value in checked.items():
        if not np.array_equal(value, residual_arrays[name], equal_nan=True):
            raise ValueError('residual array mismatch: ' + name)
    result, arrays, donors = diagnose(library, prepared['entries'], draw, prepared['original'], saved)
    result.update(census=prepared['census'], independence=prepared['independence'])
    paths.update(residual_report=args.residual / 'numbers.json', residual_arrays=args.residual / 'arrays.npz')
    result['inputs'] = {k: dict(path=str(v), sha256=digest(v)) for k, v in paths.items()}
    result['source_sha256'] = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}
    result['draw_fingerprint'] = fingerprint(draw)
    args.out.mkdir(parents=True, exist_ok=True)
    pc._json(args.out / 'numbers.json', result)
    pc._json(args.out / 'summary.json', {k: result[k] for k in ('library_donors', 'library_empty', 'library_weak', 'regions', 'coverage_targets', 'caveat')})
    np.savez_compressed(args.out / 'pools.npz', **arrays)
    np.savez_compressed(args.out / 'donors.npz', **donors)
    plot_report(result, args.out / 'pool_expectations.png')
    print(json.dumps(dict(queries=len(prepared['entries']['energy']), library_donors=result['library_donors'],
                         output=str(args.out)), allow_nan=False))


if __name__ == '__main__':
    main()
