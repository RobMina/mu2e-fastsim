"""Saved K8 residual associations; no sampling, reconstruction or replacement model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import uproot

from ..artsteps import onspill_timing
from ..readout import ReadoutConfig
from ..response import fold_step_time
from . import placement_diagnostic as pd
from . import production_compare as pc
from .compare import paired_mean, paired_ratio
from .crystal_placement import place_crystals
from .export import EVENT, columns, keys, read
from .library import load
from .radius_trial import ARMS, FIELDS, NEIGHBORS, SEED, RECO_SEED, digest, fingerprint
from .response import place
from .validity_report import observables, compare_observables


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k] for k in source.files}


def prepare(tables, library, metadata, saved, draw, manifest, baseline):
    """Validate saved identities and payloads, then reproduce only deterministic placements."""
    if (manifest['neighbors'], manifest['seed'], manifest['reco_seed']) != (NEIGHBORS, SEED, RECO_SEED):
        raise ValueError('not the fixed K8 trial')
    if fingerprint(draw) != manifest['audit']['fingerprints']['candidate_draw']:
        raise ValueError('saved draw fingerprint mismatch')
    entries, original, geometry, census, _ = read(tables, input_mode='primary')
    with uproot.open(tables) as source:
        full_ids = keys(columns(source, 'events', EVENT), EVENT)
        for prefix, table in (('crystal', 'crystals'), ('disk', 'disks'), ('vd', 'virtualdetectors')):
            extra = [k.removeprefix('geometry_' + prefix + '_') for k in library
                     if k.startswith('geometry_' + prefix + '_') and k.removeprefix('geometry_') not in geometry]
            geometry.update({prefix + '_' + k: v for k, v in columns(source, table, extra).items()} if extra else {})
    for name, value in dict(full_ids=full_ids, event_row=np.arange(len(full_ids)),
                            query_event_rows=entries['event_row']).items():
        pd.exact(name, value, saved[name])
    for prefix, values in (('entry_', entries), ('geometry_', geometry)):
        for name, value in values.items():
            pd.exact(prefix + name, value, saved[prefix + name])
    pc.check_geometry(library, geometry)
    independence = pc.check_independence(tables, library, metadata, full_ids)
    timing = onspill_timing(full_ids['event'], **pc.TIMING)
    for name, value in timing.items():
        pd.exact('timing_' + name, value, saved['timing_' + name])
    for name, value in baseline.items():
        target = 'baseline_' + name if name.startswith(('geometric_', 'crystal_', 'sampling_')) else name
        if not np.array_equal(value, saved[target], equal_nan=value.dtype.kind in 'fc'):
            raise ValueError('saved baseline mismatch: ' + name)
    n = len(entries['energy'])
    for name in ('donor', 'supported', 'fallback', 'offsets', 'radius_pool_size', 'bin_pool_size'):
        pd.exact('sampling_' + name, draw[name], saved['sampling_' + name])
    lengths = np.diff(draw['offsets'])
    if (draw['offsets'].shape != (n + 1,) or draw['offsets'][0] != 0 or np.any(lengths < 0) or
            not np.array_equal(draw['entry'], np.repeat(np.arange(n), lengths))):
        raise ValueError('invalid saved draw offsets')
    supported, donor = draw['supported'], draw['donor']
    if (supported.shape != (n,) or donor.shape != (n,) or not np.array_equal(supported, donor >= 0) or
            np.any(donor[supported] >= len(library['entry_energy'])) or np.any(lengths[~supported] != 0)):
        raise ValueError('invalid donor support')
    for arm in ('stockref', 'control', *ARMS):
        energy = saved[arm + '_energy']
        if energy.shape != (len(full_ids),) or np.any(~np.isfinite(energy) | (energy < 0)):
            raise ValueError('invalid saved energy: ' + arm)
        for field in FIELDS[1:]:
            value = saved[arm + '_' + field]
            if value.shape != energy.shape or not np.all(np.isfinite(value[energy > 0])):
                raise ValueError('invalid positive-cluster field: ' + arm + '_' + field)
    original_placed = place(library, original)
    if not np.all(original_placed['mapped']):
        raise ValueError('original CSS must map')
    pd.exact('original_crystals', original['crystal'], original_placed['crystal'])
    placed = dict(geometric=place(library, draw), crystal=place_crystals(library, entries, draw))
    for arm, steps in placed.items():
        if fingerprint(steps) != manifest['audit']['fingerprints']['candidate_placements'][arm]:
            raise ValueError('saved placement fingerprint mismatch: ' + arm)
    return dict(entries=entries, original=original, placed=placed, draw=draw, timing=timing,
                geometry=geometry, census=census, independence=independence)


def gate(time, source_rows, timing):
    """Exact pre-readout lower gate; no artificial upper cut or second PBT subtraction."""
    cfg = ReadoutConfig()
    get = {k: timing[k][source_rows] for k in ('pbtime', 'mbtime', 'event_length', 'onspill')}
    folded, accepted = fold_step_time(time, **get, time_from_protons_to_dr_marker=timing['marker'][source_rows],
        digitization_start=cfg.digitization_start, digitization_end=cfg.digitization_end,
        digitization_buffer=cfg.digitization_buffer)
    pb = np.asarray(get['pbtime'], np.float32)
    marker = np.asarray(timing['marker'][source_rows], np.float32)
    threshold = ((np.float32(cfg.digitization_start) - pb) - marker) - np.float32(cfg.digitization_buffer)
    return folded, accepted, folded - threshold


def time_exposure(steps, entries, timing, keep):
    """Late CSS energy is nominal window exposure, not actual lost photon or digi energy."""
    rows = entries['event_row'][steps['entry']]
    folded, _, _ = gate(steps['time'], rows, timing)
    cfg = ReadoutConfig()
    end = (np.float32(cfg.digitization_end) - np.asarray(timing['pbtime'][rows], np.float32)) - np.asarray(timing['marker'][rows], np.float32)
    delay = steps['time'] - entries['time'][steps['entry']]
    n = len(entries['energy'])
    return dict(end_exposed=pd.energy_sum(steps, n, keep & (folded >= end)),
                delay_ge20=pd.energy_sum(steps, n, keep & (delay >= 20)),
                delay_negative=pd.energy_sum(steps, n, keep & (delay < 0)))


def occupancy(steps, n, keep=None):
    """Positive-Birks occupied crystals and concentration; both disks, not cluster multiplicity."""
    keep = steps['energy_birks'] > 0 if keep is None else keep & (steps['energy_birks'] > 0)
    pair, inverse = np.unique(np.column_stack((steps['entry'][keep], steps['crystal'][keep])), axis=0, return_inverse=True)
    energy = np.bincount(inverse, weights=steps['energy_birks'][keep], minlength=len(pair))
    rows = pair[:, 0].astype(np.int64)
    count = np.bincount(rows, minlength=n)
    total = np.bincount(rows, weights=energy, minlength=n)
    maximum = np.zeros(n)
    np.maximum.at(maximum, rows, energy)
    return dict(crystals=count, max_fraction=np.divide(maximum, total, out=np.zeros(n), where=total > 0))


def features(library, prepared, saved_css):
    entries, original, draw, timing = (prepared[k] for k in ('entries', 'original', 'draw', 'timing'))
    n, rows = len(entries['energy']), entries['event_row']
    arrays = dict(event_row=rows, **pd.donor_features(library, entries, draw))
    arrays.update(original=pd.energy_sum(original, n), donor=pd.energy_sum(draw, n), donor_steps=np.diff(draw['offsets']),
                  original_steps=np.bincount(original['entry'], minlength=n))
    for name, steps in (('original', original), ('donor', draw)):
        _, accepted, _ = gate(steps['time'], rows[steps['entry']], timing)
        arrays[name + '_time_accepted'] = pd.energy_sum(steps, n, accepted)
        arrays.update({name + '_' + k: v for k, v in time_exposure(steps, entries, timing, accepted).items()})
        arrays.update({name + '_' + k: v for k, v in occupancy(steps, n).items()})
    folded, accepted, margin = gate(entries['time'], rows, timing)
    arrays.update(entry_folded_time=folded, entry_time_accepted=accepted, entry_gate_margin=margin)
    reuse = np.bincount(draw['donor'][draw['supported']], minlength=len(library['entry_energy']))
    arrays['donor_reuse'] = np.zeros(n, np.int64)
    arrays['donor_reuse'][draw['supported']] = reuse[draw['donor'][draw['supported']]]
    closure = {}
    for arm, steps in prepared['placed'].items():
        _, accepted, _ = gate(steps['time'], rows[steps['entry']], timing)
        mapped = pd.energy_sum(steps, n, steps['mapped'])
        lost = pd.energy_sum(steps, n, ~steps['mapped'])
        timed = pd.energy_sum(steps, n, steps['mapped'] & accepted)
        rejected = pd.energy_sum(steps, n, steps['mapped'] & ~accepted)
        residual = arrays['donor'] - lost - rejected - timed
        if not np.allclose(residual, 0, atol=1e-10, rtol=0):
            raise ValueError('timing/spatial partition mismatch')
        closure[arm] = float(np.max(np.abs(residual), initial=0))
        arrays.update({arm + '_mapped': mapped, arm + '_lost': lost,
                       arm + '_time_accepted': timed, arm + '_time_rejected': rejected})
        arrays.update({arm + '_' + k: v for k, v in time_exposure(steps, entries, timing, steps['mapped'] & accepted).items()})
        arrays.update({arm + '_' + k: v for k, v in occupancy(steps, n, steps['mapped'] & accepted).items()})
    for name in ('event_row', 'original', 'donor', 'radius_delta', 'energy_delta', 'cosine_delta', 'angle_deg',
                 'geometric_mapped', 'geometric_lost', 'crystal_mapped', 'crystal_lost'):
        if not np.array_equal(arrays[name], saved_css[name], equal_nan=True):
            raise ValueError('saved CSS mismatch: ' + name)
    return arrays, closure


def strata(entries, draw, arrays):
    """Fixed descriptive bins and limited cross-strata; no fitted threshold or model selection."""
    masks = pd.strata(entries)
    for name, value, edges in (
        ('energy', entries['energy'], (0, 50, 80, 95, 110, np.inf)),
        ('cosine', entries['cosine'], (0, .4, .7, 1.000001)),
        ('angle', arrays['angle_deg'], (0, 10, 30, np.inf)),
        ('margin', arrays['entry_gate_margin'], (-np.inf, 0, 50, np.inf)),
        ('occupancy', arrays['donor_crystals'], (0, 1, 2, 5, np.inf)),
        ('pool', draw['bin_pool_size'], (0, 1, 9, 33, np.inf)),
        ('reuse', arrays['donor_reuse'], (0, 1, 2, 5, np.inf))):
        masks.update({f'{name}_{lo}_{hi}': (value >= lo) & (value < hi) for lo, hi in zip(edges[:-1], edges[1:])})
    masks.update(unsupported=~draw['supported'], fallback=draw['fallback'], no_fallback=~draw['fallback'],
        donor_empty=draw['supported'] & (arrays['donor_steps'] == 0),
        donor_zero_nonempty=draw['supported'] & (arrays['donor_steps'] > 0) & (arrays['donor'] == 0),
        donor_positive_le10=(arrays['donor'] > 0) & (arrays['donor'] <= 10), donor_gt10=arrays['donor'] > 10,
        radius_pool_lt8=draw['supported'] & (draw['radius_pool_size'] < 8), radius_pool_8=draw['radius_pool_size'] == 8)
    return masks


def summarize(mask, entries, draw, arrays, saved, obs, shapes=False):
    rows, n = entries['event_row'][mask], int(mask.sum())
    group, zero = rows // 100, np.zeros(n)
    mean = lambda x: paired_mean(np.asarray(x)[mask], zero, group)
    donor = draw['donor'][mask & draw['supported']]
    counts = np.bincount(donor)
    result = dict(entries=n, source_blocks=int(len(np.unique(group))),
        donor_concentration=float(np.sum(counts ** 2) / len(donor) ** 2) if len(donor) else None,
        stock_selected=int((obs['stockref']['energy'][mask] > 10).sum()),
        supported=int(draw['supported'][mask].sum()), fallback=int(draw['fallback'][mask].sum()),
        empty=int((draw['supported'][mask] & (arrays['donor_steps'][mask] == 0)).sum()),
        unique_donors=int((counts > 0).sum()), max_donor_reuse=int(counts.max(initial=0)),
        features={k: pd.summarize(v[mask]) for k, v in arrays.items() if k in
                  ('angle_deg', 'energy_delta', 'radius_delta', 'entry_gate_margin', 'donor_reuse', 'donor_crystals')},
        css=dict(donor_minus_original=paired_mean(arrays['donor'][mask], arrays['original'][mask], group),
                 original_time_rejected=mean(arrays['original'] - arrays['original_time_accepted']),
                 donor_time_rejected=mean(arrays['donor'] - arrays['donor_time_accepted']),
                 original_end_exposed=mean(arrays['original_end_exposed']),
                 donor_end_exposed=mean(arrays['donor_end_exposed']),
                 original_delay_ge20=mean(arrays['original_delay_ge20']), donor_delay_ge20=mean(arrays['donor_delay_ge20'])), arms={})
    supported_mask = mask & draw['supported']
    result['css']['occupancy_comparison_queries'] = int(supported_mask.sum())
    result['css']['occupancy_comparison'] = {
        label: dict(original=int(source[supported_mask].sum()), donor=int(candidate[supported_mask].sum()),
                    donor_only=int((candidate[supported_mask] & ~source[supported_mask]).sum()),
                    original_only=int((source[supported_mask] & ~candidate[supported_mask]).sum()),
                    difference=paired_mean(candidate[supported_mask], source[supported_mask],
                                           entries['event_row'][supported_mask] // 100))
        for label, source, candidate in (
            ('empty', arrays['original_steps'] == 0, draw['supported'] & (arrays['donor_steps'] == 0)),
            ('zero_energy', arrays['original'] == 0, arrays['donor'] == 0),
            ('energy_le10', arrays['original'] <= 10, arrays['donor'] <= 10),
            ('crystals_lt5', arrays['original_crystals'] < 5, arrays['donor_crystals'] < 5))}
    stock = obs['stockref']['energy'][mask] > 10
    control = obs['control']['energy'][mask] > 10
    for arm in ('control', *ARMS):
        fast = obs[arm]['energy'][mask] > 10
        miss, gain, both = stock & ~fast, ~stock & fast, stock & fast
        item = dict(selected=int(fast.sum()), misses=int(miss.sum()), gains=int(gain.sum()),
            net_deficit=int(miss.sum() - gain.sum()), control_misses=int((control & ~fast).sum()),
            ratio=paired_ratio(fast, stock, group), selection_difference=paired_mean(fast, stock, group),
            energy=paired_mean(obs[arm]['energy'][mask], obs['stockref']['energy'][mask], group),
            jointly_selected=int(both.sum()),
            time=paired_mean(obs[arm]['time'][mask][both], obs['stockref']['time'][mask][both], group[both]))
        if shapes:
            item['shapes'] = compare_observables({k: v[mask] for k, v in obs[arm].items()},
                                               {k: v[mask] for k, v in obs['stockref'].items()}, rows)
        if arm in ARMS:
            total, mapped, timed, rejected = (arrays[k][mask] for k in
                ('donor', arm + '_mapped', arm + '_time_accepted', arm + '_time_rejected'))
            categories = dict(unsupported=~draw['supported'][mask],
                zero_donor=draw['supported'][mask] & (total == 0),
                all_spatial_loss=(total > 0) & (mapped == 0),
                all_time_rejected=(mapped > 0) & (timed == 0),
                partial_time_rejected=(timed > 0) & (rejected > 1e-12),
                positive_no_time_rejection=(timed > 0) & (rejected <= 1e-12))
            if not np.all(np.sum(list(categories.values()), axis=0) == 1):
                raise ValueError('miss category partition mismatch')
            item.update(miss_partition={k: int((miss & v).sum()) for k, v in categories.items()},
                gain_partition={k: int((gain & v).sum()) for k, v in categories.items()},
                css=dict(mapped=mean(arrays[arm + '_mapped']), spatial_loss=mean(arrays[arm + '_lost']),
                    time_rejected=mean(arrays[arm + '_time_rejected']), time_accepted=mean(arrays[arm + '_time_accepted']),
                    end_exposed=mean(arrays[arm + '_end_exposed']), delay_ge20=mean(arrays[arm + '_delay_ge20']),
                    accepted_minus_original=paired_mean(timed, arrays['original_time_accepted'][mask], group),
                    accepted_gt10=int((timed > 10).sum()),
                    timing_downcross=int(((mapped > 10) & (timed <= 10)).sum()),
                    timing_downcross_misses=int((miss & (mapped > 10) & (timed <= 10)).sum()),
                    mean_occupied_crystals=mean(arrays[arm + '_crystals'])))
        result['arms'][arm] = item
    return result


def diagnose(library, prepared, saved, saved_css):
    entries, draw = prepared['entries'], prepared['draw']
    arrays, closure = features(library, prepared, saved_css)
    rows = entries['event_row']
    origin = np.column_stack([prepared['geometry']['disk_origin_' + a] for a in 'xy'])
    axes = np.stack([np.column_stack([prepared['geometry']['disk_axis_' + a + '_' + b] for b in 'xyz'])
                     for a in 'xyz'], axis=2)
    if not np.allclose(origin, origin[:1]) or not np.allclose(axes, np.eye(3)):
        raise ValueError('incident diagnostics require aligned common-origin disks')
    obs = {arm: observables({k: saved[arm + '_' + k][rows] for k in FIELDS}, entries,
                           arrays['entry_folded_time'], origin[0]) for arm in ('stockref', 'control', *ARMS)}
    masks = strata(entries, draw, arrays)
    regions = dict(all=np.ones(len(rows), bool), inner=entries['radius'] < 450,
        disk0=entries['disk'] == 0, disk1=entries['disk'] == 1,
        inner_highE=(entries['radius'] < 450) & (entries['energy'] >= 95))
    regions.update({f'inner_highE_disk{disk}': regions['inner_highE'] & (entries['disk'] == disk) for disk in (0, 1)})
    result = dict(census=prepared['census'], independence=prepared['independence'], closure=closure,
        readout_config=asdict(ReadoutConfig()), regions={},
        caveat='Saved fixed K8 associations only, no sampling/reconstruction/counterfactual. Both-disk Birks sums and '
        'pre-readout timing gates are not leading-cluster selection. No upper fold-time cut, photon propagation or '
        'waveform acceptance model. Truth incident variables are diagnostics, not a correction model. '
        'Energy/truth E/p use all queries including zeros; spatial/time use jointly selected pairs. '
        'Paired 100-source-row block errors omit donor-training/draw uncertainty and new stock-seed spread. '
        'Legacy parent independence not established; reused donors are not independent library observations.')
    for region, base in regions.items():
        chosen = masks if region in ('all', 'inner', 'disk0', 'disk1') else {
            k: v for k, v in masks.items() if k == 'all' or k.startswith(('angle_', 'cosine_', 'donor_'))}
        result['regions'][region] = {label: summarize(base & mask, entries, draw, arrays, saved, obs,
            shapes=label == 'all' or label.startswith(('energy_', 'angle_'))) for label, mask in chosen.items()}
    for arm in ('stockref', 'control', *ARMS):
        arrays[arm + '_selected'] = obs[arm]['energy'] > 10
    return result, arrays


def plot_report(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feature, title in zip(axes.flat, ('energy_', 'angle_', 'occupancy_', 'margin_'),
                                ('Incident energy [MeV]', 'Rotated donor angle [deg]',
                                 'Positive-Birks donor crystals', 'Incident gate margin [ns]')):
        records = result['regions']['inner']
        names = [k for k in records if k.startswith(feature)]
        x = np.arange(len(names))
        for arm, offset in (('control', -.16), ('geometric', 0), ('crystal', .16)):
            values = [records[k]['arms'][arm]['ratio'] for k in names]
            y = [np.nan if v['ratio'] is None else v['ratio'] for v in values]
            err = [np.nan if v['error'] is None else v['error'] for v in values]
            ax.errorbar(x + offset, y, yerr=err, marker='o', ls='none', label=arm, capsize=2)
        ax.axhline(1, color='gray', lw=1)
        ax.set_xticks(x, ['[' + k.removeprefix(feature).replace('_', ', ') + ')\nN=' + str(records[k]['entries']) +
                         ', stock=' + str(records[k]['stock_selected']) for k in names], fontsize=8)
        ax.set_title(title)
        ax.set_ylabel('Saved selected / stock selected')
        ax.legend(fontsize=8)
    fig.suptitle('K8 residuals: r < 450 mm, both disks; descriptive strata, not causal effects')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('tables', 'library', 'trial', 'baseline', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('output must be new or empty; do not overwrite saved analysis')
    manifest = json.loads((args.trial / 'preparation.json').read_text())
    inputs = dict(tables=args.tables, library=args.library, clusters=args.baseline / 'clusters.npz')
    for name, path in inputs.items():
        if digest(path) != manifest['inputs'][name]['sha256']:
            raise ValueError('trial input hash mismatch: ' + name)
    root = Path(__file__).resolve().parents[1]
    for name in ('response.py', 'readout.py', 'showers/response.py', 'showers/crystal_placement.py'):
        if digest(root / name) != manifest['calo_source_sha256'][name]:
            raise ValueError('trial deterministic source mismatch: ' + name)
    inputs.update({name: args.trial / name for name in ('preparation.json', 'draw.npz', 'clusters.npz', 'css_arrays.npz', 'numbers.json')})
    report = json.loads((args.trial / 'numbers.json').read_text())
    if report['manifest'] != manifest:
        raise ValueError('trial manifests differ')
    library, metadata = load(args.library)
    saved, draw = archive(args.trial / 'clusters.npz'), archive(args.trial / 'draw.npz')
    prepared = prepare(args.tables, library, metadata, saved, draw, manifest, archive(inputs['clusters']))
    result, arrays = diagnose(library, prepared, saved, archive(args.trial / 'css_arrays.npz'))
    for arm in ARMS:
        actual = result['regions']['all']['all']['arms'][arm]
        previous = report['candidate'][arm]['paired']
        if actual['selected'] != previous['fast_selected'] or actual['energy'] != previous['energy_mean']:
            raise ValueError('saved trial reconstructed totals differ: ' + arm)
    result['inputs'] = {k: dict(path=str(v), sha256=digest(v)) for k, v in inputs.items()}
    result['source_sha256'] = {str(v.relative_to(root)): digest(v) for v in sorted(root.rglob('*.py'))}
    args.out.mkdir(parents=True, exist_ok=True)
    pc._json(args.out / 'numbers.json', result)
    pc._json(args.out / 'summary.json', {k: v['all'] for k, v in result['regions'].items()})
    np.savez_compressed(args.out / 'arrays.npz', **arrays)
    plot_report(result, args.out / 'inner_residuals.png')
    print(json.dumps(dict(queries=len(arrays['event_row']), closure=result['closure'], output=str(args.out)), allow_nan=False))


if __name__ == '__main__':
    main()
