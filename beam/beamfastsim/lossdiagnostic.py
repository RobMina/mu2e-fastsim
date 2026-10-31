"""Read-only reduced-reference audit and precision planning; never runs transport or Geant4.

Reduced arrays cannot replace missing ROOT/configuration/receipt bundles. Current
hashes and matching summary reductions establish only the stated recovery tier.
Mixture components and path correlations are diagnostics, not causal attribution.
CDF planning uses the existing pointwise 99% two-sample DKW convention.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import g4tables, physics_tables as pt
from .foilcompare import contrast, moments
from .reference_audit import _identity


def precision_plan(band=.02, alpha=.01, margin=.01, fast_samples=1000000):
    """Size each independent sample, including both DKW radii and distance headroom."""
    if (not 0 < alpha < 1 or not 0 < margin < band < 1
            or isinstance(fast_samples, bool) or int(fast_samples) != fast_samples or fast_samples < 2):
        raise ValueError('valid band, confidence, distance margin and sample count required')
    coefficient = math.log(4 / alpha) / 2
    radius_fast = math.sqrt(coefficient / fast_samples)
    allowance = band - margin - radius_fast
    return dict(band=band, confidence=1-alpha, distance_headroom=margin,
                fast_samples=int(fast_samples), fast_radius=radius_fast,
                minimum_reference_infinite_fast_zero_distance=math.ceil(coefficient / band**2),
                minimum_each_equal_samples_zero_distance=math.ceil(4 * coefficient / band**2),
                minimum_each_equal_samples_with_headroom=math.ceil(4 * coefficient / (band-margin)**2),
                minimum_reference_with_fast_and_headroom=(math.ceil(coefficient / allowance**2)
                                                         if allowance > 0 else None),
                current_reference_5000_radius=math.sqrt(coefficient / 5000),
                scope='Precision feasibility only, not a predicted PASS, power calculation, runtime or authorization; pointwise not simultaneous')


def variance_precision(stats, relative_budget=.025):
    """Fourth-moment pilot planning proxy; no asymptotic precision guarantee."""
    if not 0 < relative_budget < 1:
        raise ValueError('positive subunit relative variance budget required')
    variance = stats['variance']
    if variance <= 0:
        return dict(reference_samples=None, scope='Zero pilot variance gives no reliable tail precision estimate')
    required = math.ceil(stats['n'] * (3*stats['variance_se']/(relative_budget*variance))**2)
    return dict(reference_samples=max(2, required), relative_3se_budget=relative_budget,
                scope='Pilot fourth-moment asymptotic scaling proxy only; not tail/power certification or runtime')


def mixture(energy, threshold=.1):
    """All-entry population variance = positive-component within + between masses."""
    x = np.asarray(energy, float)
    if x.ndim != 1 or len(x) < 2 or np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError('finite nonnegative independent energies required')
    y = np.where(x < threshold, 0., x)
    live = y > 0
    q, mean, second = live.mean(), y.mean(), np.mean(y*y)
    within = second - mean*mean/q if q else 0.
    between = mean*mean*(1/q-1) if q else 0.
    result = dict(n=len(x), above_fraction=float(q), raw=moments(x), transformed=moments(y),
                  within_population_variance=float(within), between_population_variance=float(between),
                  population_variance=float(y.var()), decomposition_residual=float(within+between-y.var()),
                  conditional_above_mean=float(y[live].mean()) if q else None,
                  conditional_above_variance=float(y[live].var()) if q else None,
                  threshold_MeV=threshold, uncertainty='Pointwise influence-function SE, not simultaneous; conditional component is outcome-selected')
    if q:
        qi, mi = live-q, y-mean
        wi = y*y-second - 2*mean/q*mi + mean*mean/q**2*qi
        bi = 2*mean*(1/q-1)*mi - mean*mean/q**2*qi
        result.update(within_se=float(wi.std(ddof=1)/np.sqrt(len(x))),
                      between_se=float(bi.std(ddof=1)/np.sqrt(len(x))))
    else:
        result.update(within_se=0., between_se=0.)
    return result


def residual_variance_bound(variance, floor):
    """Upper change of population variance for a one-sided [0,floor] continuation."""
    if not np.isfinite(variance) or variance < 0 or not np.isfinite(floor) or floor < 0:
        raise ValueError('nonnegative finite variance and residual bound required')
    return floor * math.sqrt(variance) + floor**2/4


def _checked(record, path):
    actual = _identity(path)
    if any(actual[k] != record[k] for k in ('sha256', 'bytes')):
        raise ValueError(f'hash/size mismatch: {path}')
    return actual


def reduced_audit(data, summary):
    """Check all retained identities/outcomes against historical census, no event loop."""
    case = summary['case']
    n = case['events']
    required = ('event', 'track', 'entry_MeV', 'exit_MeV', 'exit', 'physical_stop', 'other')
    if any(k not in data for k in required) or any(v.shape != (n,) for v in data.values()):
        raise ValueError(f'reduced array schema/census mismatch: missing={set(required)-data.keys()}, shapes={ {k: v.shape for k, v in data.items()} }, expected={n}')
    if not np.array_equal(data['event'], np.arange(1, n+1)):
        raise ValueError('reduced event coverage mismatch')
    if np.any(~np.isfinite(data['track'])) or np.any(data['track'] != np.floor(data['track'])):
        raise ValueError('invalid track identities')
    for key in ('exit', 'physical_stop', 'other'):
        if not np.all(np.isin(data[key], [False, True])):
            raise ValueError('invalid outcome flag')
    e = data['exit_MeV']
    ex, stop = data['exit'].astype(bool), data['physical_stop'].astype(bool)
    if (np.any(~np.isfinite(e)) or np.any(e < 0) or np.any(data['other']) or np.any(ex == stop)
            or np.any(ex & (e <= 0)) or np.any(stop & (e != 0))):
        raise ValueError('unresolved/inconsistent reduced outcomes')
    for key, value, tolerance in (('entry_MeV', case['energy_MeV'], 1e-5), ('entry_cos', case['cosine'], 1e-6)):
        if key not in data and key == 'entry_cos':
            continue
        if not np.allclose(data[key], value, rtol=0, atol=tolerance):
            raise ValueError('reduced cell differs from summary')
    measured = dict(primaries=n, exits=int(ex.sum()), physical_stops=int(stop.sum()),
                    threshold_stops=int(np.sum(e < .1)), subthreshold_exits=int(np.sum(ex & (e < .1))), other=0)
    if any(summary[k] != value for k, value in measured.items()):
        raise ValueError('reduced census differs from summary')
    if not np.isclose(e.mean(), summary['mean_exit_MeV'], rtol=1e-12, atol=1e-14):
        raise ValueError('reduced mean differs from summary')
    if not np.isclose(e.var(ddof=1), summary['var_exit_MeV2'], rtol=1e-12, atol=1e-14):
        raise ValueError('reduced variance differs from summary')
    return measured


def _correlation(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze(root, package_dir, residual_floor=.0001):
    root, package_dir = Path(root), Path(package_dir)
    final = root / 'foil_threadG_final_report_20261026_v2'
    report_path, manifest_path = final / 'summary.json', final / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    checked = [_checked(manifest['report'], report_path), _identity(manifest_path)]
    historical = json.loads(report_path.read_text())
    if historical.get('complete') is not True:
        raise ValueError('completed historical summary required')
    result = dict(schema_version=1, scope=__doc__, reference_configuration_qualification='UNVERIFIED',
                  physics_closure=False, default_changed=False, references={}, comparisons={},
                  missing_reference_artifacts=[], precision=precision_plan(), residual_floor_MeV=residual_floor)
    covered = {Path(p['path']).name: p for p in manifest['outputs']}
    seen_hashes, seen_seeds = set(), set()
    arrays = {}
    for tag, summary in historical['g4'].items():
        name = f'g4_{tag}.npz'
        candidates = [final / name, root / 'foil_threadG_compare_20261026' / name]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            result['missing_reference_artifacts'].append(name)
            continue
        record = _checked(covered[name], path) if name in covered else _identity(path)
        seed = summary['case']['seed']
        if record['sha256'] in seen_hashes or seed in seen_seeds:
            raise ValueError('duplicate independent sample hash/seed')
        seen_hashes.add(record['sha256'])
        seen_seeds.add(seed)
        with np.load(path, allow_pickle=False) as saved:
            g = {k: saved[k] for k in saved.files}
        census = reduced_audit(g, summary)
        checked.append(record)
        arrays[tag] = g
        stats = mixture(g['exit_MeV'])
        refs = dict(identity=record, historical_array_hash_verified=name in covered,
                    census_matches_historical_summary=True, census=census, case=summary['case'], mixture=stats,
                    variance_precision_proxy=variance_precision(stats['transformed']),
                    residual_variance_bound_MeV2=residual_variance_bound(stats['population_variance'], residual_floor),
                    bound_scope='If only a one-sided residual in [0,floor] is unknown; no claim that the missing fast arrays were verified',
                    trees_recorded=summary.get('trees'), trees_reaudited=False,
                    configuration_qualification='UNVERIFIED: original effective config, macro, log and receipt absent',
                    angular_distribution='UNAVAILABLE: reduced arrays have no exit direction',
                    entry_cosine_verified='entry_cos' in g,
                    entry_cosine_source='reduced array' if 'entry_cos' in g else 'historical case only, NOT measured',
                    path_scope='Chord sum is not true G4 path; correlations are conditional diagnostics')
        if 'chord_sum_mm' in g:
            ratio = g['chord_sum_mm'] / (summary['case']['thickness_mm']/g.get('entry_cos', summary['case']['cosine']))
            refs['exit_chord_ratio_quantiles'] = (np.quantile(ratio[g['exit'].astype(bool)], [0,.5,.9,1]).tolist()
                                                  if np.any(g['exit']) else [])
            refs['above_energy_chord_correlation'] = _correlation(g['exit_MeV'][g['exit_MeV'] >= .1], ratio[g['exit_MeV'] >= .1])
        result['references'][tag] = refs
    for cell in ('nom_edge', 'moved_edge', 'oblique_edge', 'moved_5'):
        for candidate, baseline in (('full', 'loss'), ('full', 'loss_fine'), ('loss_finest', 'loss_finer'), ('loss_gridfine', 'loss_fine')):
            a, b = f'{cell}_{candidate}', f'{cell}_{baseline}'
            if a not in arrays or b not in arrays:
                continue
            r = contrast(arrays[a]['exit_MeV'], arrays[b]['exit_MeV'], historical['g4'][a]['case']['energy_MeV'])
            r['scope'] = ('Configured scattering toggle; signatures still UNVERIFIED' if baseline == 'loss'
                          else 'Combined settings/refinement contrast, NOT a pure scattering attribution')
            result['comparisons'][f'{a}_minus_{baseline}'] = r
    receipts = historical.get('receipts', [])
    result['configuration_inventory'] = dict(original_receipts_recorded=len(receipts), original_receipts_available=0,
        original_receipt_identities=receipts, original_macro_log_config_available=False,
        verdict='UNVERIFIED: saved case dictionaries cannot certify material/cut/process signatures')
    slab, target = [g4tables.load(package_dir / n, expected_material=m) for n, m in
                    [('g4_al_mu.npz', 'G4_Al'), ('g4_target_al_mu.npz', 'StoppingTarget_Al')]]
    for n in ('g4_al_mu.npz', 'g4_target_al_mu.npz'):
        checked.append(_identity(package_dir / n))
    table_records = slab['meta'].get('provenance', {}).get('files', {})
    table_audit = {}
    for charge in ('mu-', 'mu+'):
        tables, info, paths = pt.load_target(root / 'foil_threadG_smoke_20261026' / 'tables', material='G4_Al', particle=charge)
        for p in paths:
            records = [v for v in historical['material'][charge]['table_identities'] if Path(v['path']).name == p.name and 'foil_threadG_smoke_' in v['path']]
            if len(records) != 1:
                raise ValueError('missing historical table identity')
            checked.append(_checked(records[0], p))
            if p.name in table_records:
                _checked(table_records[p.name], p)
        grid = np.geomspace(.1, 40., 2001)
        one = dict(couple=info, all_consumed_loss_tables_hash_verified=True, export_bundle_reaudited=False,
                   packaged_source_qualification=slab['meta']['qualification'],
                   packaged_source_hash_coverage=bool(table_records), slab_target_dedx_ratio={})
        for energy in (.1, .2, 1., 2.02, 5.):
            ratio = np.exp(np.interp(np.log(energy), np.log(slab['energy']), np.log(slab[f'dedx_nr_{charge}']))
                           - np.interp(np.log(energy), np.log(target['energy']), np.log(target[f'dedx_nr_{charge}'])))
            one['slab_target_dedx_ratio'][str(energy)] = float(ratio)
        mean = np.exp(np.interp(np.log(grid), np.log(tables['Range']['energy']), np.log(tables['Range']['values'])))
        dedx = pt.interpolator(tables['DEDX'])(grid)
        # Bohr selection with a frozen local power is a diagnostic leg-length screen.
        from mu2efastsim.eloss import MMU, kinematics, _BOHR_MIN
        momentum = np.sqrt(grid*(grid+2*MMU))
        top = kinematics(momentum)[4]
        cut = info['energy_cuts_MeV']['e-']
        overlap = (cut < top) & (top <= 2*cut)
        leg = np.minimum(.01, mean - np.exp(np.interp(np.log(.95*grid),
                            np.log(tables['Range']['energy']), np.log(tables['Range']['values']))))
        outgoing = np.exp(np.interp(np.log(np.maximum(mean-leg, tables['Range']['values'][0])),
                                   np.log(tables['Range']['values']), np.log(tables['Range']['energy'])))
        max_soft = float(np.max((grid-outgoing)[overlap])) if np.any(overlap) else None
        one['overlapping_hard_bohr_screen'] = dict(
            default_leg_screen_max_soft_mean_MeV=max_soft,
            unsupported_minimum_soft_mean_MeV=float(_BOHR_MIN*cut),
            default_leg_screen_reaches_unsupported=bool(max_soft is not None and max_soft >= _BOHR_MIN*cut),
            energy_interval_MeV=[float(grid[overlap].min()), float(grid[overlap].max())] if np.any(overlap) else None,
            minimum_frozen_power_leg_mm=float(np.min(_BOHR_MIN*cut/dedx[overlap])) if np.any(overlap) else None,
            scope='Unsupported when actual restricted leg mean >=10*cut; not a count of missing response histories',
            native_range_mm_interval=[float(mean[overlap].min()), float(mean[overlap].max())] if np.any(overlap) else None)
        table_audit[charge] = one
    result['tables'] = table_audit
    result['target_material_muplus'] = 'NOT VALIDATED: different material-name-dependent low-energy tables; no surviving mu+ slab/target response'
    result['shared_muplus_lambda1_for_muminus'] = 'UNVERIFIED: packaged scattering table has one charge source'
    result['fast_edge_failure_diagnosis'] = 'BLOCKED: original split response arrays deleted; cannot decompose/recompute the reported fast/G4 variance ratios'
    result['first_foil_deficit'] = 'OPEN: saved full G4 alone cannot replace deleted paired fast baseline or establish a loss-model cause'
    result['inputs'] = checked
    for record in checked:
        _checked(record, record['path'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovered-root', required=True)
    parser.add_argument('--package-dir', default=str(Path(__file__).with_name('data')))
    parser.add_argument('--residual-floor', type=float, default=.0001)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    result = analyze(args.recovered_root, args.package_dir, args.residual_floor)
    result['sources'] = [_identity(__file__)]
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(references=len(result['references']), signature=result['reference_configuration_qualification'],
                          precision=result['precision']), indent=2))
    return result


if __name__ == '__main__':
    main()
