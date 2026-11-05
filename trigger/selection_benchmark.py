"""Synchronized CPU batch microbenchmark of two unchanged measured BO selections.

Not reconstruction, full-trigger timing, C++ speedup, GPU throughput or SU cost.
Saved-file CLI retains the unpromoted feature_saved validation report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
from time import perf_counter_ns

import jax
import numpy as np

import feature_saved as saved
import feature_status
import feature_table
import feature_validate
import integrity
import jax_feature_validate as arrays
import jax_kalseed as selection
import path_state
import replay

PATHS = tuple(selection.PATH_FILTERS)


def _sync(outputs):
    # Every leaf, not just the first path/event scalar. Also used for device_put.
    for leaf in jax.tree.leaves(outputs):
        leaf.block_until_ready()
    return outputs


def _timed(call, *args):
    start = perf_counter_ns()
    result = _sync(call(*args))
    return result, perf_counter_ns() - start


def _combined(features, reach, contracts):
    result = {}
    for path in PATHS:
        tracks, events = selection.evaluate(features, contracts[path])
        result[path] = (tracks, events, reach[path] & events)
    return result


def _contracts(contracts):
    arrays._mapping(contracts, 'contracts')
    for path, d0, dip in zip(PATHS, (200., 600.), (100., 1000.)):
        contract = contracts[path]
        arrays._contract(contract, path)
        expected = [dict(fitparticle=pdg, fitdirection='downstream', requireCaloCluster=False,
            maxChi2DOF=20., maxD0=d0, maxMomErr=10., maxMomentum=500., maxTanDip=dip,
            minD0=-d0, minFitCons=-1., minMomentum=mom, minNStrawHits=15, minTanDip=0.,
            minNStereo=0, minNPlanes=0, seedFitFlag=['SeedOK']) for pdg, mom in ((11, 80.), (-11, 70.))]
        if contract['cuts'] != expected or contract['minNTrks'] != 1 or contract['noFilter']:
            raise ValueError('benchmark requires unchanged original thresholds')
    if len({contracts[p]['config_sha256'] for p in PATHS}) != 1:
        raise ValueError('both contracts must bind the same full configuration SHA')


def _digest(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        value = np.asarray(leaf)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _summary(outputs, reach):
    return {p: dict(track_pass=int(np.count_nonzero(outputs[p][0])),
        reached_track_pass=int(np.count_nonzero(np.asarray(outputs[p][0]) & reach[p][:, None])),
        event_pass=int(np.count_nonzero(outputs[p][1])),
        composed_pass=int(np.count_nonzero(outputs[p][2])),
        reached=int(np.count_nonzero(reach[p]))) for p in PATHS}


def benchmark(features, contracts, reach, *, repeats=30, warmup=3):
    """Time all padded input events together; p95 is an observed BATCH percentile.

    API validates declared contracts and eager/JIT equality, not artifact provenance.
    CLI additionally requires saved-file measured-array gates and actual endpoints.
    """
    if type(repeats) is not int or not 1 <= repeats <= 10000:
        raise ValueError('repeats must be an integer in 1..10000')
    if type(warmup) is not int or not 1 <= warmup <= 100:
        raise ValueError('warmup must be an integer in 1..100')
    selection.validate_features(features)
    n, width = features['valid_track'].shape
    if n == 0:
        raise ValueError('cannot benchmark zero input events')
    _contracts(contracts)
    arrays._mapping(reach, 'reach')
    reach = {p: arrays._vector(reach[p], n, p + ' reach', boolean=True) for p in PATHS}
    if jax.default_backend() != 'cpu' or any(d.platform != 'cpu' for d in jax.devices()):
        raise ValueError('this report requires the CPU backend')
    before = _digest((features, reach, json.dumps(contracts, sort_keys=True).encode()))
    # Backend initialization remains; discard validation/eager in-memory caches.
    jax.clear_caches()
    (device_features, device_reach), transfer = _timed(jax.device_put, (features, reach))
    kernel = jax.jit(lambda f, r: _combined(f, r, contracts))
    previous = jax.config.jax_enable_compilation_cache
    jax.config.update('jax_enable_compilation_cache', False)
    try:
        start = perf_counter_ns()
        lowered = kernel.lower(device_features, device_reach)
        lower_ns = perf_counter_ns() - start
        start = perf_counter_ns()
        executable = lowered.compile()
        compile_ns = perf_counter_ns() - start
    finally:
        jax.config.update('jax_enable_compilation_cache', previous)
    outputs, first_ns = _timed(executable, device_features, device_reach)
    eager = _sync(_combined(device_features, device_reach, contracts))
    for actual, expected in zip(jax.tree.leaves(outputs), jax.tree.leaves(eager)):
        if not np.array_equal(actual, expected):
            raise ValueError('combined eager/JIT outputs differ')
    counts, checksum = _summary(outputs, reach), _digest(outputs)
    for _ in range(warmup):
        _sync(executable(device_features, device_reach))
    samples = [_timed(executable, device_features, device_reach)[1] for _ in range(repeats)]
    if before != _digest((features, reach, json.dumps(contracts, sort_keys=True).encode())):
        raise ValueError('benchmark inputs changed')
    stats = dict(median=float(np.median(samples)), min=float(np.min(samples)),
                 max=float(np.max(samples)), p95=float(np.percentile(samples, 95)))
    return dict(schema='bo-selection-cpu-batch-v1', events=n, padded_tracks=width, shape=[n, width],
        tracks=int(np.count_nonzero(features['valid_track'])),
        zero_track_events=int(np.count_nonzero(~np.any(features['valid_track'], axis=1))),
        contracts=contracts, paths=counts, eager_jit_equal=True, output_sha256=checksum,
        input_sha256=before, repeats=repeats, warmup=warmup, batch_samples_ns=samples,
        batch_latency_ns=stats, amortized_us_per_all_input_event={k: v / n / 1000 for k, v in stats.items()},
        device_put_ns=transfer, lower_ns=lower_ns, compile_ns=compile_ns, first_execution_ns=first_ns,
        runtime=dict(jax_version=jax.__version__, backend=jax.default_backend(),
            devices=[str(d) for d in jax.devices()], x64_enabled=bool(jax.config.x64_enabled),
            platform=platform.platform(), machine=platform.machine(), host=socket.gethostname(),
            thread_environment={k: v for k, v in os.environ.items() if
                k.startswith(('OMP_', 'MKL_', 'OPENBLAS_', 'TF_NUM_', 'JAX_', 'XLA_'))}),
        timing_scope='two original final selections on all padded events; actual fast reach AND event pass; '
            'explicit block_until_ready on EVERY output leaf in first/warmup/repeated executions; '
            'device_put all features/reach synchronized separately; no per-event loops; '
            'p95 is observed batch latency, not per-event tail; amortized values divide by ALL events',
        compilation_scope='jax.clear_caches after validation, before lowering; persistent compilation cache '
            'disabled for lower/compile; backend initialization retained, not cold process startup',
        exclusions='loading, padding, validation, reconstruction and full trigger; no C++ ratio, speedup, GPU or SU claim',
        measured_parity_established=False, runtime_admitted=False)


def _load(args, validation):
    manifest = saved.quiet._load(args.prepared / 'manifest.json')
    config_text = args.config.read_text()
    result = replay.parse_expanded((args.prepared / 'replay.fcl').read_text())
    schedules = saved.quiet._schedule(config_text, result, manifest)
    keys = saved._saved_keys(args.baseline_art, validation['events'])
    words, _ = saved.quiet._read_art(args.art, keys, manifest, schedules)
    data = feature_table.read(args.table, expected_keys=keys)
    contracts, sequences = feature_validate.selection_contracts(config_text, result)
    features, *_ = arrays._padded_features(data['features'], data['track_index'],
        len(keys), data['product_present'])
    states, indices, _ = path_state.decode(words[manifest['replay_process']])
    names = [name for _, name, _ in schedules[1]]
    reach = {p: arrays._reach(states[:, names.index(p)], indices[:, names.index(p)], sequences[p], p)[0]
             for p in PATHS}
    outputs = _sync(_combined(features, reach, contracts))
    for p in PATHS:
        selected_counts, selected = arrays._ragged(data['accepted_keys'][p], len(keys), p)
        expected = np.zeros_like(features['valid_track'])
        expected[np.repeat(np.arange(len(keys)), selected_counts), selected.astype(int)] = True
        if (not np.array_equal(np.asarray(outputs[p][0])[reach[p]], expected[reach[p]]) or
                not np.array_equal(outputs[p][2], states[:, names.index(p)] == 1) or
                int(np.count_nonzero(outputs[p][2])) != validation['selection']['paths'][p]['actual_pass']):
            raise ValueError('loaded benchmark outputs differ from established saved comparison')
    return features, contracts, reach, _digest(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('art', 'table', 'prepared', 'config', 'baseline-art', 'baseline-comparison',
                 'source-directory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse selection benchmark overwrite')
    jax.config.update('jax_enable_x64', True)
    paths = [args.config, args.table, args.art, args.prepared / 'manifest.json',
        args.prepared / 'replay.fcl', args.baseline_art, args.baseline_comparison,
        *[Path(module.__file__) for module in (saved, feature_status, feature_table, feature_validate,
            integrity, arrays, selection, path_state, replay)], Path(__file__),
        *[args.source_directory / name for name in feature_status.SOURCE_SHA256]]
    before = integrity.files(paths)
    sources = feature_status._sources(args.source_directory)
    start = perf_counter_ns()
    validation = saved.validate(args.art, args.table, args.prepared, config=args.config,
        baseline_art=args.baseline_art, baseline_comparison=args.baseline_comparison,
        source_directory=args.source_directory)
    report = dict(validation=validation, validation_ns=perf_counter_ns() - start, benchmark=None)
    passed = validation['measured_arrays_equal'] is True and validation['nonvacuous_track_comparison'] is True
    if passed:
        start = perf_counter_ns()
        features, contracts, reach, target_digest = _load(args, validation)
        load_ns = perf_counter_ns() - start
        report['benchmark'] = benchmark(features, contracts, reach, repeats=args.repeats, warmup=args.warmup)
        if report['benchmark']['output_sha256'] != target_digest:
            raise ValueError('timed combined outputs differ from actual saved targets')
        report.update(load_padding_target_check_ns=load_ns, counts=validation['counts'],
                      actual_target_output_sha256=target_digest, actual_target_outputs_equal=True)
    after = integrity.files(paths)
    if before != after or sources != feature_status._sources(args.source_directory):
        raise ValueError('saved benchmark inputs/source pins changed')
    report.update(evidence_before=before, evidence_after=after, released_sources=sources)
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"CPU selection batch: {validation['events']} events; measured gate={passed}; no speedup claim")
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
