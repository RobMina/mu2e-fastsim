"""Synthetic CPU batch timing seams; not measured physics/performance evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import selection_benchmark as bench
from test_trigger_jax_feature_validate import sample
from test_trigger_feature_saved import local, measured, fast, saved, dump

PATHS = bench.PATHS


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', previous)


@pytest.fixture
def case():
    data = sample((2, 0, 1, 0))
    features, *_ = bench.arrays._padded_features(data['features'], data['track_index'], 4,
                                               data['product_present'])
    features['momentum'][0, 1] = 80.  # strict threshold
    features['d0'][2, 0] = 300.       # nominal only
    reach = {PATHS[0]: np.array([True, True, True, False]),
             PATHS[1]: np.array([False, True, True, False])}
    return features, data['contracts'], reach


def test_numeric_combined_parity_and_all_event_denominator(case):
    features, contracts, reach = case
    before = deepcopy(case)
    outputs = bench._sync(bench._combined(features, reach, contracts))
    np.testing.assert_array_equal(outputs[PATHS[0]][0], [[1, 0], [0, 0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(outputs[PATHS[0]][2], [1, 0, 0, 0])
    np.testing.assert_array_equal(outputs[PATHS[1]][2], [0, 0, 1, 0])
    result = bench.benchmark(*case, repeats=3, warmup=1)
    assert result['events'] == 4 and result['tracks'] == 3
    assert result['zero_track_events'] == 2
    assert result['output_sha256'] == bench._digest(outputs)
    assert result['eager_jit_equal']
    assert result['paths'][PATHS[1]]['track_pass'] == 2
    assert result['paths'][PATHS[1]]['composed_pass'] == 1
    assert len(result['batch_samples_ns']) == 3
    for name, ns in result['batch_latency_ns'].items():
        assert result['amortized_us_per_all_input_event'][name] == ns / 4000
    assert result['batch_latency_ns']['p95'] == np.percentile(result['batch_samples_ns'], 95)
    assert result['runtime']['backend'] == 'cpu' and result['runtime']['x64_enabled']
    assert result['compile_ns'] > 0 and result['device_put_ns'] > 0
    assert json.loads(json.dumps(result, allow_nan=False))['events'] == 4
    assert bench._digest(case) == bench._digest(before)


@pytest.mark.parametrize('kwargs', [dict(repeats=0), dict(repeats=10001), dict(repeats=True),
    dict(repeats=1.5), dict(warmup=0), dict(warmup=101), dict(warmup=True)])
def test_invalid_repetitions(case, kwargs):
    with pytest.raises(ValueError, match='repeats|warmup'):
        bench.benchmark(*case, **kwargs)


@pytest.mark.parametrize('mutation', ['shape', 'dtype', 'reach-dtype', 'reach-shape', 'empty',
                                      'source', 'config', 'threshold', 'missing-path'])
def test_invalid_inputs(case, mutation):
    f, c, r = case
    if mutation == 'shape': f['pz'] = f['pz'][:1]
    elif mutation == 'dtype': f['pz'] = f['pz'].astype(np.float32)
    elif mutation == 'reach-dtype': r[PATHS[0]] = np.ones(4, np.uint8)
    elif mutation == 'reach-shape': r[PATHS[0]] = np.ones(3, bool)
    elif mutation == 'empty': f = {k: v[:0] for k, v in f.items()}
    elif mutation == 'source': c[PATHS[0]]['source']['sha256'] = 'a' * 64
    elif mutation == 'config': c[PATHS[0]]['config_sha256'] = 'a' * 64
    elif mutation == 'threshold': c[PATHS[0]]['cuts'][0]['minMomentum'] = 81.
    elif mutation == 'missing-path': c.pop(PATHS[0])
    with pytest.raises(ValueError):
        bench.benchmark(f, c, r, repeats=1, warmup=1)


def test_x64_and_cpu_required(case, monkeypatch):
    jax.config.update('jax_enable_x64', False)
    with pytest.raises(ValueError, match='x64'):
        bench.benchmark(*case)
    jax.config.update('jax_enable_x64', True)
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    with pytest.raises(ValueError, match='CPU'):
        bench.benchmark(*case)


def test_every_output_synchronized_inside_each_clock_interval(case, monkeypatch):
    events = []
    arrays = bench._combined(case[0], case[2], case[1])

    class Leaf:
        def __init__(self, value, name):
            self.value, self.name = np.asarray(value), name
        def __array__(self, dtype=None, copy=None):
            return np.asarray(self.value, dtype=dtype)
        def block_until_ready(self):
            events.append(self.name)

    outputs = {p: tuple(Leaf(v, (p, i)) for i, v in enumerate(arrays[p])) for p in PATHS}
    def execute(*args):
        events.append('execute')
        return outputs
    def lower(*args):
        events.append('lower')
        return SimpleNamespace(compile=compile_kernel)
    def compile_kernel():
        events.append('compile')
        assert not jax.config.jax_enable_compilation_cache
        return execute
    def clock():
        events.append('clock')
        return len(events) * 100
    clear = Mock(side_effect=lambda: events.append('clear'))
    monkeypatch.setattr(jax, 'clear_caches', clear)
    monkeypatch.setattr(jax, 'jit', lambda fn: SimpleNamespace(lower=lower))
    monkeypatch.setattr(bench, '_combined', execute)
    monkeypatch.setattr(bench, 'perf_counter_ns', clock)
    result = bench.benchmark(*case, repeats=4, warmup=3)
    leaves = [(p, i) for p in sorted(PATHS) for i in range(3)]
    assert events.index('clear') < events.index('lower') < events.index('compile')
    # First, eager, three warmups, four repeats each visit all six leaves.
    assert events.count('execute') == 9
    for name in leaves:
        assert events.count(name) == 9
    for i, event in enumerate(events):
        if event == 'execute':
            assert events[i + 1:i + 7] == leaves
    # First execution and repeated batches have both clocks around execution+sync.
    synchronized = ['clock', 'execute', *leaves, 'clock']
    assert sum(events[i:i + 9] == synchronized for i in range(len(events))) == 5
    assert result['batch_samples_ns'] == [800] * 4


@pytest.fixture
def cli(tmp_path, monkeypatch, case):
    args = ['selection_benchmark.py']
    paths = {}
    for name in ('art', 'table', 'prepared', 'config', 'baseline-art', 'baseline-comparison',
                 'source-directory', 'output'):
        path = tmp_path / name
        args += ['--' + name, str(path)]
        paths[name] = path
        if name in ('prepared', 'source-directory'):
            path.mkdir()
        elif name != 'output':
            path.write_bytes(b'unchanged synthetic input')
    for name in ('manifest.json', 'replay.fcl'):
        (paths['prepared'] / name).write_bytes(b'synthetic prepared input')
    for name in bench.feature_status.SOURCE_SHA256:
        (paths['source-directory'] / name).write_bytes(b'synthetic source seam')
    validation = dict(events=4, measured_arrays_equal=True, nonvacuous_track_comparison=True,
                      counts={'events': 4}, selection={'synthetic': 'full report retained'})
    validate = Mock(return_value=validation)
    monkeypatch.setattr(bench.saved, 'validate', validate)
    monkeypatch.setattr(bench.feature_status, '_sources', Mock(return_value={'seven': 'source seam'}))
    target = bench._digest(bench._combined(case[0], case[2], case[1]))
    loader = Mock(return_value=(*case, target))
    monkeypatch.setattr(bench, '_load', loader)
    monkeypatch.setattr(sys, 'argv', args + ['--repeats', '2', '--warmup', '1'])
    return SimpleNamespace(paths=paths, validation=validation, validate=validate, loader=loader)


def test_cli_fresh_report_immutable_inputs_and_refuse_overwrite(cli):
    bench.main()
    report = json.loads(cli.paths['output'].read_text())
    assert report['validation'] == cli.validation
    assert report['evidence_before'] == report['evidence_after']
    assert len(report['evidence_before']) == 24
    assert report['actual_target_outputs_equal']
    assert report['actual_target_output_sha256'] == report['benchmark']['output_sha256']
    assert report['benchmark']['events'] == 4
    before = cli.paths['output'].read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        bench.main()
    assert cli.paths['output'].read_bytes() == before
    assert cli.validate.call_count == 1


@pytest.mark.parametrize('gate', ['measured_arrays_equal', 'nonvacuous_track_comparison'])
def test_cli_preserves_mismatch_report_without_timing(cli, gate):
    cli.validation[gate] = False
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 1
    report = json.loads(cli.paths['output'].read_text())
    assert report['validation'] == cli.validation and report['benchmark'] is None
    cli.loader.assert_not_called()


def test_cli_changed_input_refuses_report(cli):
    def changed(*args):
        cli.paths['table'].write_bytes(b'changed')
        return cli.loader.return_value
    cli.loader.side_effect = changed
    with pytest.raises(ValueError, match='changed'):
        bench.main()
    assert not cli.paths['output'].exists()


def test_load_actual_fast_reach_and_targets_from_saved_endpoints(local):
    validation = bench.saved.validate(local.art, local.table, local.prepared,
        config=local.config, baseline_art=local.baseline_art,
        baseline_comparison=local.comparison_path, source_directory=local.source_directory)
    args = SimpleNamespace(art=local.art, table=local.table, prepared=local.prepared,
                           config=local.config, baseline_art=local.baseline_art)
    features, contracts, reach, digest = bench._load(args, validation)
    assert features['valid_track'].shape == (6, 1)
    np.testing.assert_array_equal(reach[PATHS[0]], [0, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(reach[PATHS[1]], [1, 1, 0, 1, 1, 0])
    result = bench.benchmark(features, contracts, reach, repeats=1, warmup=1)
    assert result['output_sha256'] == digest
    for path in PATHS:
        assert result['paths'][path]['composed_pass'] == validation['selection']['paths'][path]['actual_pass']
    validation['selection']['paths'][PATHS[0]]['actual_pass'] += 1
    with pytest.raises(ValueError, match='established saved comparison'):
        bench._load(args, validation)
