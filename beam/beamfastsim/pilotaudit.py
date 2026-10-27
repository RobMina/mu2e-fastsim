"""Re-audit a saved convergence pilot without transport or historical source certification.

Run with ``python -m beamfastsim.pilotaudit --pilot DIR --g4 ORIGINAL --out NEW_JSON``.
Paired errors are source-event-cluster standard errors, conditional on the saved inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mu2efastsim import eloss
from . import convergence, g4ref, reference_audit

T_STOP_MEV = 0.1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label):
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label}: expected integer")
    return value


def _verify_identity(actual, recorded, label):
    # Paths may legitimately change when copying a saved pilot, but content may not.
    _require(actual['sha256'] == recorded.get('sha256'), f"{label}: SHA256 mismatch")
    _require(actual['bytes'] == recorded.get('bytes'), f"{label}: byte count mismatch")


def _match(saved, current, label):
    """Check every historical field, allowing fields added by newer summarize versions."""
    if isinstance(saved, dict):
        _require(isinstance(current, dict), f"{label}: summary type mismatch")
        for key, value in saved.items():
            _require(key in current, f"{label}.{key}: unknown saved summary field")
            _match(value, current[key], f"{label}.{key}")
    elif isinstance(saved, list):
        _require(isinstance(current, list) and len(saved) == len(current), f"{label}: summary length mismatch")
        for i, (a, b) in enumerate(zip(saved, current)):
            _match(a, b, f"{label}[{i}]")
    elif isinstance(current, int) and not isinstance(current, bool):
        _require(isinstance(saved, int) and not isinstance(saved, bool) and saved == current,
                 f"{label}: saved summary mismatch ({saved!r} != {current!r})")
    elif isinstance(saved, (int, float)) and not isinstance(saved, bool):
        _require(isinstance(current, (int, float)) and
                 np.isclose(saved, current, rtol=1e-12, atol=1e-14, equal_nan=False),
                 f"{label}: saved summary mismatch ({saved!r} != {current!r})")
    else:
        _require(saved == current, f"{label}: saved summary mismatch")


def _summary_check(saved, current, name):
    required = {'n', 'stop', 'g4_stop', 'fast_minus_g4', 'first_foil', 'g4_first_foil',
                'fates', 'foil_profile', 'stop_flight_ns_mean', 'mu_minus', 'mu_plus',
                'minus_nominal', 'first_minus_nominal', 'both_stop', 'flight_time_delta_ns_given_both', 'meta'}
    _require(required <= saved.keys(), f"{name}: missing essential saved summary fields: {required - saved.keys()}")
    for key in ('stop', 'g4_stop', 'fast_minus_g4', 'first_foil', 'g4_first_foil',
                'minus_nominal', 'first_minus_nominal', 'flight_time_delta_ns_given_both'):
        _require({'mean', 'stderr', 'events'} <= saved[key].keys(), f"{name}.{key}: incomplete cluster statistic")
    for key in ('mu_minus', 'mu_plus'):
        _require({'muons', 'stops_fast', 'stops_g4'} <= saved[key].keys(), f"{name}.{key}: incomplete charge counts")
    _match(saved, current, name)
    return sorted(current.keys() - saved.keys())


def _validate_fast(f, n, name):
    required = {'x', 'p', 't', 'fate', 'foil', 'nfoil', 'meta'}
    _require(required <= f.keys(), f"{name}: missing output arrays: {required - f.keys()}")
    for key, value in f.items():
        if key != 'meta':
            _require(value.ndim >= 1 and value.shape[0] == n,
                     f"{name}.{key}: every per-row array must have exactly n={n} rows")
    _require(f['x'].shape == (n, 3), f"{name}.x: expected (n,3)")
    for key in ('p', 't', 'fate', 'foil', 'nfoil'):
        _require(f[key].shape == (n,), f"{name}.{key}: expected (n,)")
    for key in ('fate', 'foil', 'nfoil'):
        _require(np.all(np.isfinite(f[key])) and np.all(f[key] == np.floor(f[key])),
                 f"{name}.{key}: nonfinite/nonintegral labels")
    _require(np.all(np.isin(f['fate'], np.arange(1, 10))), f"{name}: invalid final fate (expected 1..9)")
    stop = f['fate'] == 1
    _require(np.all(f['foil'][stop] >= 0) and np.all(f['foil'][~stop] == -1),
             f"{name}: foil must be nonnegative exactly for fate=1, otherwise -1")
    _require(np.all(f['nfoil'] >= 0), f"{name}: negative foil crossing count")
    _require(np.all(np.isfinite(f['x'])) and np.all(np.isfinite(f['t'])), f"{name}: nonfinite position/time")
    _require(f['meta'].ndim == 0, f"{name}: metadata must be scalar JSON")
    meta = json.loads(str(f['meta']))
    _require(meta.get('n') == n, f"{name}: metadata n mismatch")
    # Bad momentum is not silently discarded: it belongs in the inconsistent stop category.
    return meta


def _contrasts(g, f, baseline, sel):
    ids = np.column_stack([g[k][sel] for k in ('run', 'subrun', 'evt')])
    stop = (f['fate'] == 1)[sel]
    first = stop & (f['foil'][sel] == 0)
    gs = g['foil'][sel] >= 0
    gf = g['foil'][sel] == 0
    bs = (baseline['fate'] == 1)[sel]
    bf = bs & (baseline['foil'][sel] == 0)
    p = np.asarray(f['p'][sel], dtype=float)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        # Algebraically equivalent to sqrt(p*p+m*m)-m, stable at very small p.
        kinetic = p * (p / (np.hypot(p, eloss.MMU) + eloss.MMU))
    raw_zero = stop & (p == 0)
    threshold = stop & np.isfinite(p) & (p > 0) & (kinetic > 0) & (kinetic < T_STOP_MEV)
    inconsistent = stop & ~(raw_zero | threshold)
    cm = lambda a: convergence.cluster_mean(a, ids)
    delta = lambda a, b: cm(a.astype(float) - b.astype(float))
    return dict(muons=int(sel.sum()), stop=cm(stop), g4_stop=cm(gs), first_foil=cm(first),
                g4_first_foil=cm(gf), fast_minus_g4=delta(stop, gs), first_foil_minus_g4=delta(first, gf),
                minus_nominal=delta(stop, bs), first_minus_nominal=delta(first, bf),
                stop_counts=dict(fast=int(stop.sum()), g4=int(gs.sum()), both=int((stop & gs).sum())),
                first_foil_counts=dict(fast=int(first.sum()), g4=int(gf.sum()), both=int((first & gf).sum())),
                fate1_split=dict(total=int(stop.sum()), raw_p_zero=int(raw_zero.sum()),
                                 positive_T_below_0p1_MeV=int(threshold.sum()), inconsistent=int(inconsistent.sum())),
                raw_p_zero_minus_g4=delta(raw_zero, gs),
                raw_p_zero_first_foil_minus_g4=delta(raw_zero & (f['foil'][sel] == 0), gf))


def audit(pilot, original):
    """Validate immutable saved artifacts, then report current paired comparisons.

    No transport module is imported and no geometry, source or binary provenance is
    inferred retroactively. The original reference row offsets remain authoritative.
    """
    pilot = Path(pilot)
    paths = dict(original_input=Path(original), selected_input=pilot / 'input_block.npz',
                 manifest=pilot / 'manifest.json', summary=pilot / 'summary.json')
    artifacts = {key: reference_audit._identity(path) for key, path in paths.items()}
    manifest = json.loads(paths['manifest'].read_text())
    summary = json.loads(paths['summary'].read_text())
    _verify_identity(artifacts['original_input'], manifest['input'], 'original input')
    _verify_identity(artifacts['selected_input'], manifest['selected_input'], 'selected input')
    n = _integer(manifest['n'], 'manifest n')
    start = _integer(manifest['block']['row_start'], 'block row_start')
    original_g = g4ref.load(original)
    g, block = convergence.select_block(original_g, start, n)
    _require(block == manifest['block'], 'original block identity differs from manifest')
    selected = g4ref.load(paths['selected_input'])
    _require(g.keys() == selected.keys(), 'selected input columns differ from original block')
    for key in g:
        _require(np.array_equal(g[key], selected[key], equal_nan=True),
                 f'selected input column {key}: not exactly equal to original block in row order')
    _require({'run', 'subrun', 'evt', 'foil', 'q', 'time'} <= g.keys(), 'missing reference comparison columns')
    reference_audit._ids(g, ('run', 'subrun', 'evt'))
    _require(np.all(np.isin(g['q'], [-1, 1])), 'reference charge must be +/-1')
    _require(np.all(np.isfinite(g['foil'])) and np.all(g['foil'] == np.floor(g['foil'])) and
             np.all(g['foil'] >= -1), 'invalid cached G4 foil labels')
    _require(np.all(np.isfinite(g['time'])), 'nonfinite reference time')
    arms, seeds = manifest['arms'], manifest['seeds']
    _require(arms and arms[0] == 'nominal' and len(set(arms)) == len(arms) and
             seeds and len(set(seeds)) == len(seeds), 'invalid manifest arms/seeds')
    _require(all(isinstance(a, str) and a.replace('_', '').isalnum() for a in arms), 'unsafe arm name')
    for seed in seeds:
        _integer(seed, 'seed')
    names = {f'{arm}_seed{seed}' for arm in arms for seed in seeds}
    runs = summary['runs']
    _require(bool(runs) and runs.keys() <= names, 'empty or unrecognized saved runs')
    files = {p.stem for p in pilot.glob('*.npz') if p.name != 'input_block.npz'}
    _require(files == set(runs), 'saved arm files and summary runs differ')
    unstarted = summary.get('unstarted', [])
    _require(len(set(unstarted)) == len(unstarted) and set(unstarted) == names - runs.keys(),
             'unstarted arms do not account for missing runs')
    output = {}
    artifacts['output_arms'] = {}
    for seed in seeds:
        base_name = f'nominal_seed{seed}'
        if not any(f'{arm}_seed{seed}' in runs for arm in arms):
            continue
        _require(base_name in runs, f'seed {seed}: saved nominal arm is required')
        baseline = None
        for arm in arms:
            name = f'{arm}_seed{seed}'
            if name not in runs:
                continue
            path = pilot / f'{name}.npz'
            artifacts['output_arms'][name] = reference_audit._identity(path)
            f = g4ref.load(path)
            meta = _validate_fast(f, n, name)
            if arm == 'nominal':
                baseline = f
            current = dict(convergence.summarize(g, f, baseline), meta=meta)
            added = _summary_check(runs[name], current, name)
            output[name] = dict(summary_verified=True, historical_summary_missing_added_keys=added,
                                recomputed_summary=current, all=_contrasts(g, f, baseline, np.ones(n, bool)),
                                mu_minus=_contrasts(g, f, baseline, g['q'] < 0),
                                mu_plus=_contrasts(g, f, baseline, g['q'] > 0))
    # Hash current comparison sources, not purported historical transport producers.
    source_paths = [Path(m.__file__) for m in (convergence, g4ref, reference_audit, eloss)] + [Path(__file__)]
    return dict(schema='beam-saved-pilot-audit-v1', verified=True, n=n, block=block,
                recorded_input_identity=manifest['input'], recorded_selected_input_identity=manifest['selected_input'],
                artifacts=artifacts, current_comparison_sources=[reference_audit._identity(p) for p in source_paths],
                source_certification='Current comparison sources only; manifest producer claims are retained, '
                                     'not retroactively certified. Artifact hashes identify bytes, not historical binaries.',
                statistics='Muon-weighted paired means and source (run,subrun,evt) event-cluster standard errors; '
                           'charge subsets retain source-event clusters. Empty subsets return null means/errors. '
                           'Seeds repeat response on the same source sample and are not independent input samples.',
                definitions=dict(fast_stop='fate == 1: fast threshold stop, not necessarily physical rest',
                                 g4_stop='cached foil >= 0: existing G4 physical-stop criterion, not relabelled',
                                 first_foil='fast fate == 1 and foil == 0; G4 cached foil == 0',
                                 threshold_MeV=T_STOP_MEV, muon_mass_MeV=eloss.MMU,
                                 inconsistent='fate=1 outside raw p=0 and finite p>0 with 0<T<0.1 MeV',
                                 foil_validation='Integral nonnegative foil for fate=1, -1 otherwise; no historical geometry upper-bound certification'),
                interpretation='Fast threshold versus physical-stop definition mismatch is explicit. Raw p=0 is '
                               'a saved-momentum diagnostic, not proof of matched physics. Pilot only; no closure claim.',
                saved_complete=summary.get('complete'), unstarted=unstarted, runs=output)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot', required=True)
    ap.add_argument('--g4', required=True, help='original full reference, not input_block.npz')
    ap.add_argument('--out', required=True, help='new JSON file; existing files are rejected')
    args = ap.parse_args(argv)
    try:
        _require(not Path(args.out).exists(), 'output already exists; a new JSON path is required')
        report = audit(args.pilot, args.g4)
        text = json.dumps(report, indent=2, allow_nan=False)
        with open(args.out, 'x') as stream:
            stream.write(text + '\n')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        ap.error(str(exc))
    print(f'Verified saved pilot ({report["n"]} original rows, {len(report["runs"])} arms) -> {args.out}; no closure claim')
    return report


if __name__ == '__main__':
    main()
