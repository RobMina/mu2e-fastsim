"""Reproducible, synthetic-only reconstruction kernel diagnostics across JAX backends.

Run as ``python3 -m crvfastsim.reco.pvc_profile --out DIR``. Compile, resident
execution and transfers are timed separately; exact bytes, not tolerance gates,
are compared when ``--against`` names another run. No reconstruction defaults
are changed. Shapes are explicit so comparisons cannot silently change padding.
"""
import argparse
import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def _timed(fn):
    import jax
    t = time.perf_counter()
    value = fn()
    jax.block_until_ready(value)
    return value, time.perf_counter() - t


def _differences(a, b):
    out = {}
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        same_shape = x.shape == y.shape and x.dtype == y.dtype
        exact = same_shape and x.tobytes() == y.tobytes()
        rec = dict(shape_equal=same_shape, bit_identical=exact)
        if same_shape and np.issubdtype(x.dtype, np.number):
            finite = np.isfinite(x) & np.isfinite(y)
            rec.update(unequal=int(np.count_nonzero(x != y)),
                       max_abs=float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else None)
        out[key] = rec
    return out


def _selected_differences(a, b):
    """Compare reported pulses and accepted cluster roots, excluding padded slots."""
    result = {}
    for prefix, selector in (('pulse__', 'found'), ('coinc__', 'root')):
        key = prefix + selector
        if key not in a or key not in b:
            continue
        ma, mb = np.asarray(a[key]), np.asarray(b[key])
        if not np.array_equal(ma, mb):
            result[prefix] = dict(selection_equal=False)
            continue
        ca = {k: np.asarray(v)[ma] for k, v in a.items()
              if k.startswith(prefix) and np.asarray(v).shape[:ma.ndim] == ma.shape}
        cb = {k: np.asarray(b[k])[mb] for k in ca}
        result[prefix] = dict(selection_equal=True, selected=int(ma.sum()), columns=_differences(ca, cb))
    return result


def _node_summary(directory):
    """Report simultaneous-worker bottleneck rates without treating waveforms as events."""
    result = {}
    for backend in ('cpu', 'oneapi'):
        root = directory / f'node_{backend}'
        files = sorted(root.glob('tile*/profile.json'))
        kind = 'pulse'
        if not files:
            files = sorted(root.glob('tile*/bench_chain_stats.json'))
            kind = 'chain'
        if not files:
            result[backend] = dict(workers=0, complete=False)
            continue
        records = [json.loads(f.read_text()) for f in files]
        if kind == 'pulse':
            medians = [float(np.median(r['measurements'][0]['resident_s'])) for r in records]
            population = sum(r['metadata']['rows'] for r in records)
            unit = 'waveforms_per_s'
            compile_s = [r['measurements'][0]['compile_s'] for r in records]
        else:
            medians = [float(np.median(r['pass_wall_s'])) for r in records]
            population = sum(r['events'] for r in records)
            unit = 'synthetic_events_per_s'
            compile_s = [r['first_call_s'] for r in records]
        result[backend] = dict(workers=len(records), complete=len(records) == 12, kind=kind,
                               median_worker_s=medians, compile_or_cold_s=compile_s,
                               bottleneck_rate={unit: population / max(medians)},
                               files=[str(f) for f in files])
    return result


def _trace_summary(directory):
    """Summarize Chrome trace duration events without assuming device tracing exists."""
    from collections import defaultdict
    totals, counts = defaultdict(float), defaultdict(int)
    files = sorted(directory.rglob('*.trace.json.gz'))
    for path in files:
        with gzip.open(path, 'rt') as stream:
            trace = json.load(stream)
        for event in trace.get('traceEvents', []):
            if event.get('ph') != 'X':
                continue
            name = event.get('name', '<unnamed>')
            totals[name] += event.get('dur', 0)
            counts[name] += 1
    return dict(files=[str(p) for p in files],
                duration_events=sorted((dict(name=k, calls=counts[k], total_us=v) for k, v in totals.items()),
                                       key=lambda x: -x['total_us'])[:80])


def _measure(name, fn, args, outdir, repeat, trace=False):
    import jax
    record = dict(name=name, input_shapes=[list(x.shape) for x in args])
    t = time.perf_counter()
    executable = fn.lower(*args).compile()
    record['compile_s'] = time.perf_counter() - t
    try:
        record['cost_analysis'] = executable.cost_analysis()
        record['memory_analysis'] = str(executable.memory_analysis())
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        record['analysis_error'] = str(exc)
    text = executable.as_text()
    (outdir / f'{name}.hlo.txt').write_text(text or '')
    value, record['first_execution_s'] = _timed(lambda: executable(*args))
    first_host = jax.device_get(value)
    times = []
    for _ in range(repeat):
        value, dt = _timed(lambda: executable(*args))
        times.append(dt)
    record['resident_s'] = times
    t = time.perf_counter()
    host = jax.device_get(value)
    record['device_to_host_s'] = time.perf_counter() - t
    before = first_host if isinstance(first_host, dict) else {'value': first_host}
    after = host if isinstance(host, dict) else {'value': host}
    record['repeat_identity'] = _differences(before, after)
    if trace:
        try:
            with jax.profiler.trace(str(outdir / f'trace_{name}')):
                _timed(lambda: executable(*args))
        except (ImportError, RuntimeError) as exc:
            record['trace_error'] = str(exc)
    print(json.dumps(record), flush=True)
    return record, host


def run(args):
    import jax
    import jax.numpy as jnp
    from . import pulsefit, pulsefit_jax as PJ, synth
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    records, arrays = [], {}
    meta = dict(jax=jax.__version__, jaxlib=__import__('jaxlib').__version__,
                devices=[str(d) for d in jax.devices()], backend=jax.default_backend(),
                rows=args.rows, seed=args.seed, repeat=args.repeat, rounds=args.rounds)
    (outdir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    cfg = dataclasses.replace(pulsefit.RecoConfig(), max_pulses=args.rounds)
    if args.part in ('all', 'pulse'):
        raw = tuple(x[:args.rows] for x in synth.waveforms((args.rows + 19) // 20, seed=args.seed, L=12))
        device, transfer = _timed(lambda: jax.device_put(raw))
        fn = jax.jit(lambda *xs: PJ.reconstruct_dense(*xs, cfg=cfg))
        rec, host = _measure('pulse', fn, device, outdir, args.repeat, args.trace)
        rec['host_to_device_s'] = transfer
        records.append(rec)
        arrays.update({f'pulse__{k}': v for k, v in host.items()})
    if args.part in ('all', 'linalg'):
        rng = np.random.default_rng(args.seed)
        J = rng.normal(size=(args.rows, 16, 3))
        M = np.einsum('kpi,kpj->kij', J, J) + np.eye(3) * 1e-3
        v = rng.normal(size=(args.rows, 3))
        dm, dv = jax.device_put((M, v))
        cases = {'solve': (jax.jit(lambda m, v: jnp.linalg.solve(m, v[..., None])[..., 0]), (dm, dv)),
                 'eigvalsh': (jax.jit(jnp.linalg.eigvalsh), (dm,))}
        for name, (fn, xs) in cases.items():
            rec, host = _measure(name, fn, xs, outdir, args.repeat, args.trace)
            records.append(rec)
            arrays[name] = host
    if args.part in ('all', 'coinc'):
        from . import cli, coinc_jax as CJ
        G = cli._coinc_geom()
        P = synth.coinc_events(args.events, G, seed=args.seed)
        status = np.zeros(4 * len(G.sector), np.int64)
        params, tables = CJ.type_params(G), CJ.hit_tables(G, status)
        cols = [P[k] for k in ('ev', 'idx', 'bar', 'sipm', 'time', 'pe', 'seq')]
        tt = sorted(params)[0]
        tp = params[tt]
        arr, _, _ = CJ.pack(*cols, G, tables, tt, H=args.hits)
        n = min(len(arr['valid']), args.groups)
        arr = {k: np.pad(v[:n], [(0, args.groups - n)] + [(0, 0)] * (v.ndim - 1)) for k, v in arr.items()}
        xs = jax.device_put(tuple(arr[k] for k in ('valid', 'idx', 'side', 'bar', 'sipm', 'first', 'layer', 'counter', 'x', 'y', 't', 'pe', 'pos', 'cut')) + (tp.tp, tp.cu))
        fn = jax.jit(lambda *a: CJ.find_dense(*a, ld=tp.ld, ns=tp.ns, uniform=tp.uniform, K=args.slots))
        rec, host = _measure(f'coinc_H{args.hits}_K{args.slots}', fn, xs, outdir, args.repeat, args.trace)
        rec.update(real_groups=n, hits=args.hits, slots=args.slots)
        records.append(rec)
        arrays.update({f'coinc__{k}': v for k, v in host.items()})
    np.savez(outdir / 'arrays.npz', **arrays)
    result = dict(metadata=meta, measurements=records,
                  hashes={k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()})
    if args.against:
        with np.load(Path(args.against) / 'arrays.npz') as ref:
            result['comparison'] = _differences(arrays, ref)
            result['selected_comparison'] = _selected_differences(arrays, ref)
    (outdir / 'profile.json').write_text(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summarize-trace', help='summarize an existing profile tree and exit')
    ap.add_argument('--compare-arrays', help='compare saved arrays with --against without rerunning')
    ap.add_argument('--summarize-node', help='summarize saved simultaneous-worker profiles and exit')
    ap.add_argument('--out')
    ap.add_argument('--against')
    ap.add_argument('--part', choices=('all', 'pulse', 'linalg', 'coinc'), default='all')
    ap.add_argument('--rows', type=int, default=4096)
    ap.add_argument('--events', type=int, default=1000)
    ap.add_argument('--groups', type=int, default=128)
    ap.add_argument('--hits', type=int, default=16)
    ap.add_argument('--slots', type=int, default=6)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--trace', action='store_true')
    args = ap.parse_args()
    if args.summarize_trace:
        print(json.dumps(_trace_summary(Path(args.summarize_trace)), indent=2))
        return
    if args.summarize_node:
        print(json.dumps(_node_summary(Path(args.summarize_node)), indent=2))
        return
    if args.compare_arrays:
        if not args.against:
            ap.error('--compare-arrays requires --against')
        with np.load(Path(args.compare_arrays) / 'arrays.npz') as a, np.load(Path(args.against) / 'arrays.npz') as b:
            print(json.dumps(dict(all_slots=_differences(a, b), selected=_selected_differences(a, b)), indent=2))
        return
    if not args.out:
        ap.error('--out is required for a profile run')
    if min(args.rows, args.events, args.groups, args.hits, args.slots, args.rounds, args.repeat) < 1:
        ap.error('shapes, repeat and rounds must be positive')
    run(args)


if __name__ == '__main__':
    main()
