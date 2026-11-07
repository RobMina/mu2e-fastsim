"""One capped-direction draw and exactly two CPU calls on authenticated exposed file0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import candidate_trial as ct, candidate_support as auth, source_overlap_report
from . import production_compare as pc, support_audit as sa
from .radius_trial import ARMS, CONDITIONS, NEIGHBORS, RECO_SEED, SEED, digest, fingerprint

PINS = dict(ct.PINS,
    production='dee6f99faa4882a3ac98ded6a9a7d160aa011e1d739aab78235c443adc4eaf5a',
    direction_support='3fa365bfa6d455cdec4d7cfc0a251afbc01d669a0e3da2ec9d0b8d44580754f4')
POLICY = dict(neighbors=8, angle_degrees=10., radius_mm=25., both_direction_signs=True,
              zero_sign='nonnegative', insufficient='original_K8', orientation='disk')
SCOPE = ('One fixed capped-direction candidate draw, exactly two CPU reconstruction calls, saved references only. '
         'Direction-compatible nearest8 within25mm and10deg only with at least8 donors; otherwise originalK8. '
         'Same cell/fallback, both disks, zeros and weak responses retained; no outcome veto or rescale. '
         'Development only, not confirmation, inclusive acceptance, current-head compatibility or throughput. '
         'Stored-row block errors omit library/model uncertainty; 165 development/pilot overlaps remain.')


def equal(name, actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc'):
        raise ValueError('direction trial mismatch: ' + name)


def references(inputs, args):
    receipt = auth.pinned_json(inputs, args.production / 'summary.json', PINS['production'])
    if receipt.get('completed') is not True or receipt.get('sampler_calls') != 1 or receipt.get('reconstruction_calls') != 2:
        raise ValueError('production reference is not completed')
    saved = {name: auth.pinned_npz(inputs, args.production / name, receipt['output_sha256'][name])
             for name in ('clusters.npz', 'diagnostics.npz', 'draw.npz')}
    manifest = auth.pinned_json(inputs, args.production / 'preparation.json', receipt['output_sha256']['preparation.json'])
    support = auth.pinned_json(inputs, args.direction_support / 'summary.json', PINS['direction_support'])
    census = auth.pinned_npz(inputs, args.direction_support / 'arrays.npz', support['output_sha256']['arrays.npz'])
    return saved, manifest, census


def validate_policy(pools, original, census, entries, production):
    """New support must equal the predeclared capped subset of the frozen diagnostic witness."""
    used = census['compatible_within25_count'] >= NEIGHBORS
    expected = original['eligible'].copy()
    expected[used] = census['compatible_eligible'][used]
    for key in ('query_cell', 'chosen_cell', 'occupied', 'library_cell', 'counts', 'size', 'full_size',
                'exact_size', 'supported', 'fallback', 'outside', 'substituted'):
        equal('pool ' + key, pools[key], original[key])
    equal('eligible', pools['eligible'], expected)
    equal('direction_used', pools['direction_used'], used)
    equal('direction_fallback', pools['direction_fallback'], original['supported'] & ~used)
    equal('direction_count', pools['direction_count'], census['compatible_within25_count'])
    equal('census query rows', census['event_row'], entries['event_row'])
    equal('census full IDs', census['full_ids'], production['full_ids'])
    if used.sum() != 2284 or pools['direction_fallback'].sum() != 112:
        raise ValueError('not the fixed 2284/112 policy population')


def validate_production(prepared, baseline, production):
    ct.validate_clusters(production, len(prepared['full_ids']))
    for name, value in baseline.items():
        if not name.startswith(('geometric_', 'crystal_', 'sampling_')):
            equal('production ' + name, production[name], value)


def authenticate(args):
    from . import direction_sampler

    state = ct.authenticate(args)
    saved, manifest, census = references(state['inputs'], args)
    prepared, baseline = state['prepared'], state['baseline']
    production = saved['clusters.npz']
    validate_production(prepared, baseline, production)
    if manifest['runtime'] != state['prior']['runtime'] or manifest['timing'] != pc.TIMING:
        raise ValueError('production runtime or timing differs')
    if (manifest['neighbors'], manifest['seed'], manifest['reco_seed']) != (NEIGHBORS, SEED, RECO_SEED):
        raise ValueError('production fixed seeds or neighbors differ')
    for key in ('full_ids', 'query_event_rows', 'event_row'):
        equal('production diagnostics ' + key, production[key], saved['diagnostics.npz'][key])
    pools = direction_sampler.enumerate_pools(state['library'], prepared['entries'])
    validate_policy(pools, state['pools'], census, prepared['entries'], production)
    updated = dict(state['pools'], **pools)
    for name in ('empty', 'weak', 'energy'):
        updated['prob_' + name] = sa._probabilities(updated['outcome_' + name], updated)
    state.update(pools=updated, production=production, production_diagnostics=saved['diagnostics.npz'],
                 production_draw=saved['draw.npz'], production_manifest=manifest)
    state['inputs'].unchanged()
    return state


def validate_draw(state, draw):
    ct.validate_draw(state['library'], state['prepared']['entries'], draw, state['pools'])
    for name in ('direction_used', 'direction_fallback', 'direction_count'):
        equal(name, draw[name], state['pools'][name])
    fallback = draw['direction_fallback']
    equal('originalK8 fallback donor', draw['donor'][fallback], state['production_draw']['donor'][fallback])


def publish(out, state, output, report_out=None):
    from . import direction_report

    target = out if report_out is None else report_out
    names = ('numbers.json', 'diagnostics.npz', 'conditional.json', 'conditional_arrays.npz', 'summary.json')
    if target == out and any((target / name).exists() for name in names):
        raise ValueError('report output exists; use new saved-only report destination')
    if target != out and target.exists():
        raise ValueError('recovery report destination must be new')
    result, arrays, conditional, conditional_arrays = direction_report.report(state['library'], state['prepared'],
        state['baseline'], state['production'], output, state['pools'], state['production_diagnostics'])
    state['inputs'].unchanged()
    if target != out:
        target.mkdir(parents=True, exist_ok=False)
    pc._json(target / 'numbers.json', result)
    ct.save_npz(target / 'diagnostics.npz', arrays)
    pc._json(target / 'conditional.json', conditional)
    ct.save_npz(target / 'conditional_arrays.npz', conditional_arrays)
    if hasattr(direction_report, 'plot'):
        direction_report.plot(conditional, target)
    artifacts = {p.name: digest(p) for p in sorted(target.iterdir()) if p.is_file() and p.name != 'summary.json'}
    pc._json(target / 'summary.json', dict(completed=True, sampler_calls=1, reconstruction_calls=2,
        policy=POLICY, pins=PINS, input_sha256=state['inputs'].sha256, output_sha256=artifacts, scope=SCOPE))
    return result


def saved_report(args):
    """Pure saved-input recovery: never sample, place, read ROOT or reconstruct."""
    if not args.execution_sha256:
        raise ValueError('saved reporting requires an external execution SHA256')
    target = args.report_out or args.out
    if (target != args.out and target.exists()) or (target == args.out and any(
            (target / name).exists() for name in ('numbers.json', 'diagnostics.npz', 'conditional.json', 'conditional_arrays.npz', 'summary.json'))):
        raise ValueError('use a new report-out destination')
    inputs = source_overlap_report.Inputs()
    execution = auth.pinned_json(inputs, args.out / 'execution.json', args.execution_sha256)
    if execution['draws'] != 1 or execution['calls'] != 2 or [r['arm'] for r in execution['runs']] != list(ARMS):
        raise ValueError('not a completed two-call execution')
    for name, sha in execution['output_sha256'].items():
        if Path(name).name != name:
            raise ValueError('unsafe execution artifact name')
        ct.checked_file(inputs, args.out / name, sha)
    manifest = inputs.json(args.out / 'preparation.json')
    if manifest['pins'] != PINS or manifest['policy'] != POLICY or (manifest['seed'], manifest['reco_seed']) != (SEED, RECO_SEED):
        raise ValueError('saved fixed policy or settings differ')
    for path, sha in manifest['input_sha256'].items():
        if '/calo/calofastsim/' not in path:
            ct.checked_file(inputs, path, sha)
    candidate = auth.pinned_json(inputs, args.candidate / 'summary.json', PINS['candidate'])
    packed = auth.pinned_npz(inputs, args.candidate / 'library.npz', candidate['output_sha256']['library.npz'])
    packed.pop('metadata')
    saved, production_manifest, census = references(inputs, args)
    support = auth.pinned_json(inputs, args.support / 'summary.json', PINS['support'])
    original_pools = auth.pinned_npz(inputs, args.support / 'candidate_pools.npz', support['output_sha256']['candidate_pools.npz'])
    baseline = auth.pinned_npz(inputs, args.baseline / 'clusters.npz', manifest['input_sha256'][str((args.baseline / 'clusters.npz').absolute())])
    output = inputs.npz(args.out / 'clusters.npz')
    prepared = {key: {k[len(prefix):]: v for k, v in output.items() if k.startswith(prefix)}
                for key, prefix in (('entries', 'entry_'), ('geometry', 'geometry_'), ('timing', 'timing_'))}
    prepared.update(full_ids=output['full_ids'], original=inputs.npz(args.out / 'original.npz'),
        draw=inputs.npz(args.out / 'draw.npz'),
        placed={arm: inputs.npz(args.out / (arm + '_placement.npz')) for arm in ARMS})
    state = dict(inputs=inputs, library=packed, prepared=prepared, baseline=baseline,
                 production=saved['clusters.npz'], production_diagnostics=saved['diagnostics.npz'],
                 production_draw=saved['draw.npz'], pools=inputs.npz(args.out / 'pools.npz'))
    validate_production(prepared, baseline, state['production'])
    validate_production(prepared, baseline, output)
    ct.validate_clusters(output, len(prepared['full_ids']))
    validate_policy(state['pools'], original_pools, census, prepared['entries'], state['production'])
    validate_draw(state, prepared['draw'])
    if fingerprint(prepared['draw']) != manifest['fingerprints']['draw'] or any(
            fingerprint(value) != manifest['fingerprints']['placements'][arm] for arm, value in prepared['placed'].items()):
        raise ValueError('saved physics fingerprints differ')
    root = Path(__file__).resolve().parents[1]
    for source in sorted(root.rglob('*.py')):
        inputs.read(source)
    inputs.unchanged()
    return publish(args.out, state, output, args.report_out)


def run(args):
    if args.report_only:
        return saved_report(args)
    if not args.validate_only and args.out.exists():
        raise ValueError('trial output exists; no sampling or reconstruction replay permitted')
    from . import direction_sampler, direction_report, reconstruct

    state = authenticate(args)
    value = ct.runtime(state['prior'])
    if not callable(direction_report.report) or not callable(reconstruct.steps_for_reco):
        raise RuntimeError('report/reconstruction dependencies unavailable')
    if args.validate_only:
        print(json.dumps(dict(validated=True, sampled=False, reconstructed=False, runtime=value,
            direction_used=int(state['pools']['direction_used'].sum()), direction_fallback=int(state['pools']['direction_fallback'].sum()))), flush=True)
        return state
    from .. import stockio
    from ..cluster import Geometry

    resources = (stockio.load_sim_crystal(args.reco_inputs / CONDITIONS[0]),
                 stockio.load_pulse(args.reco_inputs / CONDITIONS[1]),
                 stockio.load_noise(args.reco_inputs / CONDITIONS[2]),
                 stockio.load_propagation(args.reco_inputs / CONDITIONS[3]), Geometry.load(args.tables))
    state['inputs'].unchanged()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(pins=PINS, policy=POLICY, seed=SEED, reco_seed=RECO_SEED, runtime=value, scope=SCOPE,
        input_sha256=state['inputs'].sha256, calo_source_sha256=state['sources'], timing=pc.TIMING)
    pc._json(args.out / 'preparation.json', manifest)
    prepared = state['prepared']
    ct.save_npz(args.out / 'original.npz', prepared['original'])
    ct.save_npz(args.out / 'pools.npz', state['pools'])
    rng = np.random.default_rng(SEED)
    manifest['draw_rng_before'] = rng.bit_generator.state
    pc._json(args.out / 'preparation.json', manifest)
    pc._json(args.out / 'draw_started.json', dict(seed=SEED, retry_authorized=False))
    try:
        draw = direction_sampler.sample(state['library'], prepared['entries'], rng)
    finally:
        manifest['draw_rng_after'] = rng.bit_generator.state
        pc._json(args.out / 'preparation.json', manifest)
    ct.save_npz(args.out / 'draw.npz', draw)
    if manifest['draw_rng_after'] != state['production_manifest']['draw_rng_after']:
        raise ValueError('uniform draw schedule differs from productionK8')
    validate_draw(state, draw)
    prepared.update(draw=draw, placed=dict(geometric=ct.place(state['library'], draw),
        crystal=ct.place_crystals(state['library'], prepared['entries'], draw)))
    for arm, steps in prepared['placed'].items():
        ct.save_npz(args.out / (arm + '_placement.npz'), steps)
    manifest['fingerprints'] = dict(draw=fingerprint(draw), placements={k: fingerprint(v) for k, v in prepared['placed'].items()})
    pc._json(args.out / 'preparation.json', manifest)
    state['inputs'].unchanged()
    output, runs = ct.run_calls(args.out, prepared, state['baseline'], resources)
    pc._json(args.out / 'execution.json', dict(draws=1, calls=2, runs=runs,
        output_sha256={p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()}))
    return publish(args.out, state, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'support', 'direction-support', 'production', 'tables', 'baseline', 'k8', 'reco-inputs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--report-only', action='store_true')
    parser.add_argument('--execution-sha256')
    parser.add_argument('--report-out', type=Path)
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
