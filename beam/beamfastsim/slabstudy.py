"""Small opt-in slab-diffusion diagnostic; never launches Geant4 or production transport.

Example (from the repository root)::
  PYTHONPATH=../beam-core:beam /path/to/.venv/bin/python3 -m beamfastsim.slabstudy \
    --cell moved_edge --pdg 13 --samples 2000 --arms no_scatter diffuse \
    --seed 71 --slab-tables /path/to/audited/export --out /path/to/NEW

Tables must exactly match the shipped G4_Al slab NPZ and an audited export.
CSDA means plus finite-cut Urban fluctuations are a diagnostic hybrid, not a
restricted-loss or hard-secondary model. The mu+-derived lambda1 is assumed
for both charges. Infinite planes, no field/decay/secondary transport; no G4
physics parity or full-chain acceptance is claimed. Numerical refinement uses
common row/replica/seed identities, not Brownian-bridge coupled trajectories.

FLOOR/CAP/NUMERICAL are NOT physical stops or exits. RAWZERO is a sampled loss
clamp, NOT a certified physical stop. [rawzero, rawzero+unresolved] is reported
only as a diagnostic physical-stop identification convention, not a certified
physical bound. Threshold bounds use monotone energy debit for non-numerical
unresolved residuals below 0.1 MeV; numerical endpoints remain unknown. No
ordinary exit-energy CDF is computed with unresolved endpoints. The explicit
threshold transform can be identified when every allowed continuation is below
threshold; this does not identify physical stopping or ordinary exit energy. All sample
counts and identification bounds retain the original denominator. Statistical
intervals are pointwise diagnostics, not simultaneous acceptance gates.

Outputs are fresh-only: summary.json, inputs.npz, complete Response/ledger
NPZs, and identity-paired difference NPZs. Runtime includes first compilation.
Source/input hashes are checked again before marking the study complete.
--responses re-reports saved arrays without rerunning transport; historical
source hashes remain in that study, distinct from current reporting sources.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.stats import beta

from . import g4tables, slabdiffusion as sd, slabendpoints as se, slabrefs
from .foilcampaign import TOLERANCES
from .foilscan import SCANS
from .reference_audit import _identity

ARMS = ('no_scatter', 'diffuse', 'half_step', 'half_angle', 'half_energy', 'refined', 'lower_floor')
THRESHOLD = .1
SCOPE = __doc__


def verify(records):
    for record in records:
        actual = _identity(record['path'])
        if any(actual[k] != record[k] for k in ('sha256', 'bytes')):
            raise ValueError(f"source/input identity changed: {record['path']}")


def _stats(x):
    x = np.asarray(x, float).ravel()
    finite = x[np.isfinite(x)]
    return dict(n=len(x), finite=len(finite), mean=float(finite.mean()) if len(finite) else None,
                se=float(finite.std(ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else None,
                quantiles=np.quantile(finite, [0, .01, .5, .99, 1]).tolist() if len(finite) else None)


def _ci(k, n, alpha=.01):
    return [0. if k == 0 else float(beta.ppf(alpha / 2, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-alpha / 2, k+1, n-k))]


def _bound(lo, hi, n):
    return dict(count_bounds=[int(lo), int(hi)], fraction_bounds=[lo/n, hi/n],
                interval_99pct=[_ci(lo, n)[0], _ci(hi, n)[1]])


def classifications(response):
    """Per-identity lower/upper indicators, never silently relabel unresolved."""
    s, e = np.asarray(response.status), np.asarray(response.raw_energy)
    if not s.size or np.any(~np.isin(s, list(sd.STATUS_NAMES))):
        raise ValueError('empty response or unknown/active endpoint status')
    ex = (s == sd.TRANSMITTED) | (s == sd.REFLECTED)
    raw = s == sd.RAWZERO
    unresolved = np.isin(s, [sd.UNRESOLVED_FLOOR, sd.UNRESOLVED_CAP, sd.UNRESOLVED_NUMERICAL])
    if np.any(ex & (~np.isfinite(e) | (e <= 0))) or np.any(raw & (e != 0)):
        raise ValueError('inconsistent resolved energy/status')
    bounded_low = unresolved & (s != sd.UNRESOLVED_NUMERICAL) & np.isfinite(e) & (e >= 0) & (e < THRESHOLD)
    threshold = raw | (ex & (e < THRESHOLD)) | bounded_low
    sub = ex & (e < THRESHOLD)
    above = ex & (e >= THRESHOLD)
    return dict(threshold_stop=(threshold, threshold | unresolved),
                diagnostic_physical_stop=(raw, raw | unresolved),
                subthreshold_exit=(sub, sub | unresolved),
                above_threshold_exit=(above, above | (unresolved & ~bounded_low)),
                transmitted=(s == sd.TRANSMITTED, (s == sd.TRANSMITTED) | unresolved),
                reflected=(s == sd.REFLECTED, (s == sd.REFLECTED) | unresolved))


def summarize(response):
    classes = classifications(response)
    s = np.asarray(response.status)
    counts = {name: int(np.sum(s == code)) for code, name in sd.STATUS_NAMES.items()}
    unresolved = sum(counts[sd.STATUS_NAMES[k]] for k in (sd.UNRESOLVED_FLOOR, sd.UNRESOLVED_CAP, sd.UNRESOLVED_NUMERICAL))
    return dict(n=s.size, endpoint_counts=counts, unresolved=unresolved,
                certified_physical_stop_count=None,
                classes={k: _bound(int(lo.sum()), int(hi.sum()), s.size) for k, (lo, hi) in classes.items()},
                endpoint_residual_energy_MeV=_stats(response.raw_energy), path_mm=_stats(response.path),
                steps=_stats(response.steps),
                ledger={k: _stats(v) for k, v in response.ledger.items()},
                certified_physical_endpoint_distribution=False,
                resolved_algorithmic_distribution=unresolved == 0,
                positive_exit_distribution_scope='Selection conditional; not the all-primary physical endpoint distribution',
                physical_stop_scope='Diagnostic rawzero identification convention only; not certified physical stopping')


def flatten(response):
    result = {k: np.asarray(getattr(response, k)) for k in sd.Response._fields if k != 'ledger'}
    result.update({'ledger__' + k: np.asarray(v) for k, v in response.ledger.items()})
    return result


def paired(candidate, base):
    """Same-shape, caller-verified identities; differences are candidate minus base."""
    a, b = flatten(candidate), flatten(base)
    if a.keys() != b.keys() or any(a[k].shape != b[k].shape for k in a):
        raise ValueError('paired response shape/schema mismatch')
    diffs = {'delta__' + k: a[k].astype(float) - b[k].astype(float) for k in a if k != 'status'}
    diffs.update(candidate_status=a['status'], base_status=b['status'])
    ca, cb = classifications(candidate), classifications(base)
    bounds = {}
    for k in ca:
        lo = ca[k][0].astype(int) - cb[k][1].astype(int)
        hi = ca[k][1].astype(int) - cb[k][0].astype(int)
        diffs['delta_lower__' + k], diffs['delta_upper__' + k] = lo, hi
        bounds[k] = dict(mean_bounds=[float(lo.mean()), float(hi.mean())],
                         lower_indicator=_stats(lo), upper_indicator=_stats(hi))
    codes = list(sd.STATUS_NAMES)
    transitions = [[int(np.sum((base.status == i) & (candidate.status == j))) for j in codes] for i in codes]
    both = np.isin(a['status'], [sd.TRANSMITTED, sd.REFLECTED]) & np.isin(b['status'], [sd.TRANSMITTED, sd.REFLECTED])
    return diffs, dict(n=a['status'].size, class_differences=bounds, status_codes=codes,
                       transitions_base_rows_candidate_columns=transitions,
                       endpoint_residual_difference=_stats(diffs['delta__raw_energy']),
                       path_difference=_stats(diffs['delta__path']),
                       both_exit_energy_difference_selection_conditional=_stats(diffs['delta__raw_energy'][both]),
                       scope='Common keyed identities, not common physical trajectories; endpoint residual is not exit energy')


def reference_summary(g, energy, cosine, thickness):
    e = np.asarray(g['exit_MeV'])
    ex, stop = np.asarray(g['exit'], bool), np.asarray(g['physical_stop'], bool)
    if (not len(e) or np.any(g['other']) or np.any(ex == stop) or np.any(~np.isfinite(e))
            or np.any(e < 0) or np.any(ex & (e <= 0)) or np.any(stop & (e != 0))
            or not np.allclose(g['entry_MeV'], energy, rtol=0, atol=1e-6)
            or not np.allclose(g['entry_cos'], cosine, rtol=0, atol=1e-6)):
        raise ValueError('G4 reference is unresolved, inconsistent, or not the selected monoenergetic cell')
    z = np.asarray(g['exit_z_mm'])
    axial = np.isfinite(z) & (np.abs(np.abs(z) - thickness / 2) <= 2e-6)
    if np.any(ex & ~axial):
        raise ValueError('non-axial or nonfinite G4 exit: finite reference box is not an infinite-plane slab')
    back = ex & (z < 0)
    counts = dict(threshold_stop=int(np.sum(e < THRESHOLD)), physical_stop=int(stop.sum()),
                  subthreshold_exit=int(np.sum(ex & (e < THRESHOLD))),
                  above_threshold_exit=int(np.sum(ex & (e >= THRESHOLD))),
                  transmitted=int(np.sum(ex & ~back)), reflected=int(back.sum()))
    return dict(n=len(e), endpoint_counts=counts,
                fraction_intervals={k: _bound(v, v, len(e)) for k, v in counts.items()})


def compare_reference(response, g, ref):
    n, m = response.status.size, ref['n']
    result = {}
    for name, (lo, hi) in classifications(response).items():
        key = 'physical_stop' if name == 'diagnostic_physical_stop' else name
        k, low, high = ref['endpoint_counts'][key], int(lo.sum()), int(hi.sum())
        rlo, rhi = _ci(k, m, .005)
        result[name] = dict(delta_fraction_bounds=[low/n-k/m, high/n-k/m],
                            interval_99pct=[_ci(low, n, .005)[0]-rhi, _ci(high, n, .005)[1]-rlo])
    distribution = None
    intervals = se.endpoint_intervals(response, float(g['entry_MeV'][0]), THRESHOLD)
    identified = np.array_equal(intervals['threshold_lower'], intervals['threshold_upper'])
    if identified and min(n, m) > 1:
        from .foilcompare import contrast
        distribution = contrast(intervals['threshold_lower'].ravel(), g['exit_MeV'],
                                float(g['entry_MeV'][0]))
    return dict(classes=result, threshold_aligned_distribution=distribution,
                threshold_distribution_identified=identified,
                scope='Independent G4 samples; no paired histories. Threshold transform only, conditional on monotone continuation and algorithmic rawzero convention; not ordinary exit-energy or physical-stop closure. Null distribution means unidentified transformed endpoints or too few samples.')


def arm_settings(names, floor, lower_floor, domain_floor, domain_ceiling, **base):
    tolerance = 16 * np.finfo(float).eps * domain_floor
    if abs(floor - domain_floor) <= tolerance:
        floor = domain_floor
    if lower_floor is not None and abs(lower_floor - domain_floor) <= tolerance:
        lower_floor = domain_floor
    if not domain_floor <= floor < domain_ceiling:
        raise ValueError('floor outside common table domain')
    if 'lower_floor' in names and (lower_floor is None or not domain_floor <= lower_floor < floor):
        raise ValueError('lower_floor requires --lower-floor within table domain and below --floor')
    result = {}
    for name in names:
        kw = dict(base, floor=floor, scatter=name != 'no_scatter')
        for arm, parameter in [('half_step', 'max_step'), ('half_angle', 'angular_fraction'), ('half_energy', 'energy_fraction')]:
            if name in (arm, 'refined'):
                kw[parameter] /= 2
        if name == 'lower_floor':
            kw['floor'] = lower_floor
        result[name] = kw
    return result


def _save(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return _identity(path)


def saved_responses(directory, args, settings):
    """Reuse immutable responses, retaining historical provenance without rerunning transport."""
    root = Path(directory)
    manifest = root / 'summary.json'
    old = json.loads(manifest.read_text())
    if not old.get('complete'):
        raise ValueError('saved response study is incomplete')
    for name in ('cell', 'pdg', 'samples', 'seed'):
        if old['args'][name] != getattr(args, name):
            raise ValueError(f'saved response identity differs: {name}')
    records = [_identity(manifest)] + old['inputs'] + old['outputs']
    verify(records)
    responses = {}
    for arm, kw in settings.items():
        if arm not in old['arms'] or old['arms'][arm]['params'] != kw:
            raise ValueError(f'saved response settings differ: {arm}')
        item = old['arms'][arm]['output']
        verify([item])
        with np.load(item['path'], allow_pickle=False) as f:
            if (not np.array_equal(f['row_ids'], [0]) or not np.array_equal(f['replica_ids'], np.arange(args.samples))
                    or int(f['seed']) != args.seed):
                raise ValueError('saved response stream identities differ')
            ledger = {k.removeprefix('ledger__'): f[k] for k in f.files if k.startswith('ledger__')}
            responses[arm] = sd.Response(*(f[k] for k in sd.Response._fields if k != 'ledger'), ledger)
        if responses[arm].status.shape != (args.samples, 1):
            raise ValueError('saved response population shape differs')
    return responses, records


def run(a):
    from mu2efastsim import eloss, eloss_jax, materials
    import jax
    import jaxlib
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f'refusing existing output directory: {out}')
    if a.samples < 1 or a.samples > 2**32 or not 0 <= a.seed < 2**32 or a.batch < 1:
        raise ValueError('positive samples/batch and uint32 seed/replica identities required')
    if len(set(a.arms)) != len(a.arms):
        raise ValueError('duplicate arms')
    thickness, energy, emax, cosine, cmax = SCANS[a.cell]
    if energy != emax or cosine != cmax:
        raise ValueError('monoenergetic, fixed-angle SCANS cell required')
    started = time.monotonic()
    source_names = ('slabstudy.py', 'slabdiffusion.py', 'slabendpoints.py', 'slabrefs.py',
                    'g4tables.py', 'physics_tables.py', 'foilscan.py', 'foilreport.py', 'foilclosure.py',
                    'foilcampaign.py', 'foilcompare.py', 'reference_audit.py', 'slicestudy.py')
    sources = [_identity(Path(__file__).with_name(n)) for n in source_names]
    core_sources = [_identity(m.__file__) for m in (eloss, eloss_jax, materials)]
    table_identity = _identity(g4tables.SLAB)
    check = g4tables.compare(a.slab_tables, g4tables.SLAB, 'G4_Al', require_audited_export=True)
    if not check['all_arrays_equal'] or not check['material_equal']:
        raise ValueError('shipped slab NPZ differs from audited G4_Al export')
    tables = g4tables.load(g4tables.SLAB, expected_material='G4_Al')
    charge = 'mu-' if a.pdg == 13 else 'mu+'
    tb = sd.make_tables(tables['energy'], tables['csda_range_' + charge], tables['msc_energy'], tables['msc_lambda1'])
    # Match the kernel's exp(log(node)) domain exactly, including roundoff.
    lower = float(np.exp(max(np.asarray(tb.loge)[0], np.asarray(tb.lam_loge)[0])))
    upper = float(np.exp(min(np.asarray(tb.loge)[-1], np.asarray(tb.lam_loge)[-1])))
    floor = lower if a.floor is None else a.floor
    tcut = float(check['fresh']['couples'][charge]['energy_cuts_MeV']['e-'])
    settings = arm_settings(a.arms, floor, a.lower_floor, lower, upper, max_step=a.max_step,
                            angular_fraction=a.angular_fraction, energy_fraction=a.energy_fraction,
                            max_steps=a.max_steps, loss='sample', tcut=tcut)
    if not (np.isfinite(tcut) and tcut > 0 and 0 < a.energy_fraction < 1 and a.max_steps >= 0
            and np.isfinite(a.max_step) and a.max_step > 0 and np.isfinite(a.angular_fraction) and a.angular_fraction > 0
            and energy <= upper):
        raise ValueError('invalid numerical settings or energy outside table ceiling')
    provenance = check['fresh']['provenance']
    inputs = [table_identity] + list(provenance['files'].values()) + list(provenance['audit']['artifacts'].values())
    refs, reference_reports = {}, {}
    if a.campaign:
        refs, records = slabrefs.load_references(a.campaign, a.cell, a.pdg, thickness, energy, cosine)
        inputs.extend(records)
        reference_reports = {mode: reference_summary(g, energy, cosine, thickness) for mode, g in refs.items()}
    reused = {}
    if a.responses:
        reused, records = saved_responses(a.responses, a, settings)
        inputs.extend(records)
    verify(sources + core_sources + inputs)
    out.mkdir(parents=True, exist_ok=False)
    entry = np.array([energy])
    direction = np.array([[np.sqrt(1-cosine**2), 0., cosine]])
    row_ids, replica_ids = np.array([0], np.uint32), np.arange(a.samples, dtype=np.uint32)
    identities = dict(row_ids=row_ids, replica_ids=replica_ids, seed=np.array(a.seed, np.uint32))
    outputs = [_save(out / 'inputs.npz', dict(energy=entry, direction=direction, thickness=np.array(thickness), **identities))]
    report = dict(schema_version=2, complete=False, scope=SCOPE, args=vars(a), threshold_MeV=THRESHOLD,
                  tolerances=TOLERANCES,
                  sample_identity='row_id=0, replica_id=0..samples-1; same seed/identities in every arm; batch-independent kernel',
                  sources=sources, core_sources=core_sources, inputs=inputs, outputs=outputs,
                  table_check=check, table_domain_MeV=[lower, upper], material=asdict(materials.ALUMINIUM),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, jax=jax.__version__,
                                   jaxlib=jaxlib.__version__, backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64)),
                  g4=reference_reports, arms={}, paired={})
    def checkpoint():
        (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    checkpoint()
    responses = {}
    for arm, kw in settings.items():
        begin = time.monotonic()
        response = reused[arm] if a.responses else sd.respond(
            entry, direction, thickness, tb, material=materials.ALUMINIUM,
            replicas=a.samples, row_ids=row_ids, seed=a.seed, batch=a.batch, **kw)
        runtime = time.monotonic() - begin
        saved = _save(out / (arm + '.npz'), dict(flatten(response), **identities))
        outputs.append(saved)
        summary = summarize(response)
        summary['endpoint_identification'] = se.summarize_endpoints(response, entry, thickness, tb, THRESHOLD)
        report['arms'][arm] = dict(params=kw, response_reused=bool(a.responses), runtime_s_including_compilation=runtime, summary=summary,
                                   output=saved, vs_g4={m: compare_reference(response, g, reference_reports[m]) for m, g in refs.items()})
        responses[arm] = response
        checkpoint()
        print(json.dumps(dict(arm=arm, runtime_s=runtime, endpoint_counts=summary['endpoint_counts'])), flush=True)
    base = 'diffuse' if 'diffuse' in responses else a.arms[0]
    for arm, response in responses.items():
        if arm == base:
            continue
        diffs, summary = paired(response, responses[base])
        tag = arm + '_minus_' + base
        saved = _save(out / ('paired_' + tag + '.npz'), dict(diffs, **identities))
        outputs.append(saved)
        report['paired'][tag] = dict(summary, output=saved)
    verify(sources + core_sources + inputs + outputs)
    report.update(complete=True, wall_s=time.monotonic()-started)
    checkpoint()
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cells = [k for k, v in SCANS.items() if v[1] == v[2] and v[3] == v[4]]
    ap.add_argument('--cell', required=True, choices=cells)
    ap.add_argument('--pdg', required=True, type=int, choices=[13, -13])
    ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--arms', nargs='+', choices=ARMS, default=['no_scatter', 'diffuse'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True, help='new directory; never overwritten')
    ap.add_argument('--slab-tables', required=True, help='audited G4_Al export root or tables/')
    ap.add_argument('--campaign', nargs='+', help='optional existing foilscan/foilcampaign references (read-only)')
    ap.add_argument('--responses', help='existing completed slabstudy to re-report; never rerun its response kernel')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--max-step', type=float, default=.01, help='mm')
    ap.add_argument('--angular-fraction', type=float, default=.02)
    ap.add_argument('--energy-fraction', type=float, default=.05)
    ap.add_argument('--max-steps', type=int, default=10000)
    ap.add_argument('--floor', type=float, help='MeV, default common table lower edge; unresolved, NOT stop')
    ap.add_argument('--lower-floor', type=float, help='MeV for lower_floor arm; must remain inside table domain')
    return run(ap.parse_args(argv))


if __name__ == '__main__':
    main()
