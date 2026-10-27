"""Synthetic saved-accumulator diagnosis: no response execution or real inputs."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from crvfastsim.objective import crvresponse as R
from crvfastsim.objective import diagnose as D
from crvfastsim.objective import validation as V


def synthetic_state(scale=1):
    state = R.new_state()
    state.update(events=100, exposure_ns=100000.0, lines_z=80, lines_x=70, lines_xz=60)
    for name, axes in R.SHAPES.items():
        for slot in R.SLOTS_OF.get(name, R.SLOTS):
            if len(axes) == 1:
                state['h'][f'{name}_{slot}'][1] = 12 * scale
    for slot in R.SLOTS:
        state['h'][f'pe_along_{slot}'][0][0][2] = 12 * scale
        state['h'][f'pe_along_{slot}'][1][-1][4] = 8 * scale
        state['h'][f'pe_path_{slot}'][0][2] = 12 * scale
        state['h'][f'pe_path_{slot}'][-1][4] = 8 * scale
    for body in range(4, 16):
        state['dtp'][0][body][80 + body - 4] = (body + 2) * scale
    state['dtp'][3][15][72] = 9 * scale
    return state


def test_histogram_accepts_nonnegative_counts_without_mutation():
    value = [[0, 2, 3], [4, 0, 1]]
    before = copy.deepcopy(value)
    actual = D.histogram(value, (2, 3), 'example')
    assert actual.dtype.kind == 'f'
    np.testing.assert_array_equal(actual, value)
    assert value == before


@pytest.mark.parametrize('value', [
    [1, 2], [[1, 2, 3]], [[1, -1], [0, 2]],
    [[1, float('nan')], [0, 2]], [[1, float('inf')], [0, 2]],
    [[1, float('-inf')], [0, 2]], [[1], [2, 3]], [['bad', 1], [2, 3]],
])
def test_histogram_rejects_malformed_or_nonfinite_counts(value):
    with pytest.raises(ValueError):
        D.histogram(value, (2, 2), 'example')


def test_empty_quantiles_and_moments():
    h, edges = np.zeros(3), np.arange(4)
    for q in (0.16, 0.5, 0.84):
        assert D.quantile(h, edges, q) is None
    assert D.moments(h, edges) == dict(entries=0.0, mean=None, rms=None,
                                     q16=None, q50=None, q84=None)


def test_fixed_bin_moments_and_interpolated_quantiles():
    h = np.array([0, 2, 0, 6])
    edges = np.array([-2, 0, 2, 4, 6])
    expected = dict(entries=8, mean=4, rms=np.sqrt(3), q16=1.28,
                    q50=14 / 3, q84=4 + 2 * 4.72 / 6)
    assert D.moments(h, edges) == pytest.approx(expected)
    for q, key in ((0.16, 'q16'), (0.5, 'q50'), (0.84, 'q84')):
        assert D.quantile(h, edges, q) == pytest.approx(expected[key])
    # A populated bin has zero bin-center RMS, but interpolated quantiles span it.
    assert D.moments([0, 4, 0], [0, 2, 4, 6]) == pytest.approx(
        dict(entries=4, mean=3, rms=0, q16=2.32, q50=3, q84=3.68))


def test_conditional_common_support_uses_fixed_data_weights_and_empty_cells():
    states = dict(data=[[2, 0], [0, 6], [9, 0], [0, 0], [0, 5]],
                  nominal=[[4, 0], [0, 2], [8, 0], [1, 0], [0, 0]],
                  best=[[3, 0], [0, 3], [1, 0], [0, 0], [1, 0]])
    before = copy.deepcopy(states)
    out = D.conditional(states, np.array([0, 1, 2]), min_entries=2)
    assert out['common_cells'] == 2
    assert out['cell_count'] == 5
    assert [c['common_support'] for c in out['cells']] == [True, True, False, False, False]
    assert out['data_cell_weights'] == pytest.approx([0.25, 0.75, 0, 0, 0])
    assert out['common_support_fraction'] == pytest.approx(
        dict(data=8 / 22, nominal=6 / 15, best=6 / 8))
    for sample in out['standardized'].values():
        assert sample['probability'] == pytest.approx([0.25, 0.75])
        assert sample['mean'] == pytest.approx(1.25)
        assert sample['tv_to_data'] == pytest.approx(0)
    empty = out['cells'][3]['samples']['data']
    assert empty['entries'] == 0
    assert empty['mean'] is None
    assert empty['tv_to_data'] is None
    assert empty['probability'] == [0, 0]
    assert empty['population_fraction'] == 0
    assert out['cells'][2]['samples']['best']['entries'] == 1
    assert states == before
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize('states', [
    dict(data=[[0, 0], [0, 0]], nominal=[[0, 0], [0, 0]], best=[[0, 0], [0, 0]]),
    dict(data=[[2, 0], [0, 2]], nominal=[[0, 0], [0, 0]], best=[[2, 0], [0, 2]]),
    dict(data=[[2, 0], [0, 0]], nominal=[[0, 0], [0, 2]], best=[[2, 0], [0, 2]]),
])
def test_conditional_without_common_support_is_explicit_and_finite(states):
    out = D.conditional(states, np.array([0, 1, 2]), min_entries=1)
    assert out['common_cells'] == 0
    assert out['standardized'] == {}
    assert out['data_cell_weights'] == [0, 0]
    for name, h in states.items():
        assert out['common_support_fraction'][name] == (0 if np.sum(h) else None)
        if not np.sum(h):
            assert out['marginal'][name]['mean'] is None
            assert out['marginal'][name]['tv_to_data'] is None
            assert all(c['samples'][name]['population_fraction'] is None for c in out['cells'])
    json.dumps(out, allow_nan=False)


def test_standardization_removes_population_mix_not_conditional_shapes():
    out = D.conditional(dict(data=[[90, 0], [0, 10]], nominal=[[10, 0], [0, 90]],
                             best=[[50, 0], [0, 50]]), np.array([0, 1, 2]), 1)
    assert out['marginal']['nominal']['tv_to_data'] == pytest.approx(0.8)
    assert out['marginal']['best']['tv_to_data'] == pytest.approx(0.4)
    for row in out['cells']:
        assert all(s['tv_to_data'] == 0 for s in row['samples'].values())
    for sample in out['standardized'].values():
        assert sample['probability'] == pytest.approx([0.9, 0.1])
        assert sample['tv_to_data'] == pytest.approx(0)
        assert sample['mean'] == pytest.approx(0.6)


def test_deterministic_score_bins_reproduce_objective_term():
    h, ref = np.array([0, 10, 20, 10, 0]), np.array([3, 2, 1, 1, 0])
    out = D.score_bins(h, ref, 10000)
    expected = R.shape_term(h, ref, 10000)
    assert out['ndf'] == expected['ndf']
    assert out['score'] == pytest.approx(expected['chi2ndf'])
    assert out['score'] == pytest.approx(1 + sum(out['contribution']))
    assert D.score_bins(np.zeros(5), ref, 10000) is None


def test_mean_decomposition_preserves_total_and_separates_mixture():
    out = D.conditional(dict(data=[[90, 0], [0, 10]], nominal=[[10, 0], [0, 90]],
                             best=[[50, 0], [0, 50]]), np.array([0, 1, 2]), 1)
    row = out['mean_decomposition']['best_minus_nominal']
    assert row == pytest.approx(dict(total=-0.4, within_cell=0, cell_population=-0.4))
    out = D.conditional(dict(data=[[80, 0, 0], [0, 20, 0]], nominal=[[10, 0, 0], [0, 90, 0]],
                             best=[[0, 10, 0], [0, 0, 90]]), np.arange(4), 1)
    row = out['mean_decomposition']['best_minus_nominal']
    assert row == pytest.approx(dict(total=1, within_cell=1, cell_population=0))
    for row in out['mean_decomposition'].values():
        assert row['total'] == pytest.approx(row['within_cell'] + row['cell_population'])


@pytest.mark.parametrize('patch', [
    dict(data=[[1, -2], [0, 4]]), dict(data=[[1, float('nan')], [0, 4]]),
    dict(best=[[1, 2]]), dict(extra=[[1, 2], [0, 4]]),
])
def test_conditional_rejects_bad_sample(patch):
    samples = dict(data=[[1, 2], [0, 4]], nominal=[[1, 2], [0, 4]], best=[[1, 2], [0, 4]])
    samples.update(patch)
    with pytest.raises(ValueError):
        D.conditional(samples, np.arange(3), 1)


def test_plot_outputs_synthetic_all_sectors(tmp_path):
    from crvfastsim.objective.diagnose_plots import plot_report
    report = dict(blocks={b: D.summarize_block({k: synthetic_state() for k in ('data', 'nominal', 'best')}, 1)
                          for b in ('a', 'b')})
    plot_report(report, tmp_path)
    pngs = list(tmp_path.glob('*.png'))
    assert len(pngs) == 10
    assert all(p.stat().st_size > 1000 for p in pngs)


def test_standardization_retains_actual_conditional_shift():
    out = D.conditional(dict(data=[[80, 0, 0], [0, 20, 0]],
                             nominal=[[10, 0, 0], [0, 90, 0]],
                             best=[[0, 10, 0], [0, 0, 90]]), np.arange(4), 1)
    standard = out['standardized']
    assert standard['nominal']['tv_to_data'] == pytest.approx(0)
    assert standard['best']['probability'] == pytest.approx([0, 0.8, 0.2])
    assert standard['best']['tv_to_data'] == pytest.approx(0.8)
    assert standard['best']['mean'] - standard['data']['mean'] == pytest.approx(1)
    assert all(c['samples']['best']['tv_to_data'] == 1 for c in out['cells'])


@pytest.mark.parametrize('target', ['pe_along_T1', 'dtp'])
@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), float('-inf'), 'shape'])
def test_validate_state_rejects_invalid_histograms(target, bad):
    state = synthetic_state()
    container = state if target == 'dtp' else state['h']
    if bad == 'shape':
        container[target] = [0]
    else:
        container[target][0][0][0] = bad
    with pytest.raises(ValueError, match='invalid histogram'):
        D.validate_state(state)


@pytest.mark.parametrize('field', ['events', 'lines_z', 'lines_x', 'lines_xz'])
@pytest.mark.parametrize('bad', [-1, 1.5, True, float('nan')])
def test_validate_state_rejects_invalid_exposure_counts(field, bad):
    state = synthetic_state()
    state[field] = bad
    with pytest.raises(ValueError, match='invalid state count'):
        D.validate_state(state)


def test_validate_state_rejects_empty_exposure():
    with pytest.raises(ValueError, match='empty state exposure'):
        D.validate_state(R.new_state())


def test_summarize_block_all_sectors_and_module_labels_without_mutation():
    states = {name: synthetic_state(scale) for name, scale in [('data', 1), ('nominal', 2), ('best', 3)]}
    before = copy.deepcopy(states)
    out = D.summarize_block(states, min_entries=1)
    assert set(out['sectors']) == {'EX', 'T1', 'T2', 'TAG'}
    for exposure in out['exposure'].values():
        assert exposure == dict(events=100, lines_z=80, lines_x=70, lines_xz=60)
    modules = {'T1': ['T10', 'T11'], 'T2': ['T20', 'T21'],
               'TAG': [f'M{i}' for i in range(1, 9)]}
    body_offset = {'T1': 4, 'T2': 6, 'TAG': 8}
    for slot, sector in out['sectors'].items():
        expected_scalar = {'tspread', 'nlayers', 'pe_layer', 'oot_pe'}
        if slot != 'EX':
            expected_scalar.add('dt_cluster')
        assert set(sector['scalar']) == expected_scalar
        for scalar in sector['scalar'].values():
            for name, scale in [('data', 1), ('nominal', 2), ('best', 3)]:
                assert scalar['samples'][name]['entries'] == 12 * scale
                assert scalar['samples'][name]['tv_to_data'] == pytest.approx(0)
        along, path = sector['pe_along'], sector['pe_path']
        assert along['cell_count'] == 20
        assert path['cell_count'] == 8
        assert (along['cells'][0]['side'], along['cells'][-1]['side']) == (0, 1)
        assert along['cells'][0]['along_low'] == -1
        assert along['cells'][-1]['along_high'] == 1
        for row in along['cells']:
            side, i = divmod(row['index'], len(R.E['along']) - 1)
            assert row['side'] == side
            assert (row['along_low'], row['along_high']) == tuple(R.E['along'][i:i + 2])
        for row in path['cells']:
            i = row['index']
            assert (row['path_low'], row['path_high']) == tuple(R.E['path'][i:i + 2])
        for conditional in (along, path):
            assert conditional['common_cells'] == 2
            assert conditional['marginal']['data']['mean'] == pytest.approx(16.5)
            assert conditional['standardized']['best']['tv_to_data'] == pytest.approx(0)
        if slot == 'EX':
            assert 'module_timing' not in sector
            continue
        timing = sector['module_timing']
        assert timing['cell_count'] == 4 * len(modules[slot])
        assert [(c['ex_module'], c['module']) for c in timing['cells']] == [
            (f'EX{e}', module) for e in range(4) for module in modules[slot]]
        assert timing['edges'] == R.DTP_EDGES.tolist()
        for i, module in enumerate(modules[slot]):
            cell = timing['cells'][i]
            body = body_offset[slot] + i
            assert cell['module'] == module
            assert cell['samples']['data']['entries'] == body + 2
            assert cell['samples']['data']['mean'] == pytest.approx(0.25 + 0.5 * (body - 4))
        if slot == 'TAG':
            last = timing['cells'][-1]
            assert (last['ex_module'], last['module']) == ('EX3', 'M8')
            assert last['samples']['data']['entries'] == 9
            assert last['samples']['data']['mean'] == pytest.approx(-3.75)
    assert states == before
    json.dumps(out, allow_nan=False)


@pytest.fixture
def saved_replicas(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('diagnosis attempted simulation, input inspection, or rescoring')

    for name in ('run', 'verify', 'inspect_inputs', 'revision', 'runtime_assets'):
        monkeypatch.setattr(V, name, forbidden)
    monkeypatch.setattr(R, 'objective', forbidden)
    monkeypatch.setattr(D, 'code_identity', lambda: {'head': 'synthetic', 'dirty': False})

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False))
        return path

    reference = save(tmp_path / 'reference.json', synthetic_state())
    refspec = dict(path=str(reference), sha256=V.digest(reference))
    root = tmp_path / 'replicas'
    for block, seed in [('a', 17), ('b', 23)]:
        for arm in ('nominal', 'best'):
            knobs = {} if arm == 'nominal' else {'photon_yield_scale': 1.1}
            point = root / f'{block}_{arm}'
            save(point / 'r3.json', dict(seed=seed, knobs=knobs, response_state=synthetic_state()))
            save(point / 'comparison.json', dict(objective=2 if arm == 'nominal' else 1,
                                                terms={'synthetic': {'chi2ndf': 3}}))
            save(point / 'provenance.json', dict(
                manifest={'assets': {'reference': refspec}}, code={'crv': 'frozen-crv', 'core': 'frozen-core'},
                runtime_assets={'lut': {'sha256': 'synthetic-lut'}}, python='synthetic',
                versions={'numpy': 'synthetic'}, seed=seed, knobs=knobs))
    receipt = dict(passed=True, inputs={
        point.name: {'files': {p.stem: {'sha256': V.digest(p)} for p in point.glob('*.json')}}
        for point in root.iterdir()})

    def paired(directory, out):
        assert Path(directory) == root
        save(Path(out), receipt)
        return copy.deepcopy(receipt)

    paired_mock = Mock(side_effect=paired)
    monkeypatch.setattr(V, 'paired', paired_mock)
    return root, reference, receipt, paired_mock


def test_diagnose_saved_outputs_preserves_inputs_and_objectives(saved_replicas, tmp_path):
    root, reference, receipt, paired_mock = saved_replicas
    snapshots = {p: p.read_bytes() for p in [reference, *root.glob('*/*.json')]}
    out = tmp_path / 'diagnosis'
    report = D.diagnose(root, out, min_entries=1, plots=False)
    paired_mock.assert_called_once()
    assert report['kind'] == 'saved-response-diagnosis'
    assert report['paired_receipt'] == receipt
    assert set(report['blocks']) == {'a', 'b'}
    assert report['blocks']['a'] == report['blocks']['b']
    assert set(report['blocks']['a']['sectors']) == {'EX', 'T1', 'T2', 'TAG'}
    assert json.loads((out / 'diagnosis.json').read_text()) == report
    assert {p: p.read_bytes() for p in snapshots} == snapshots
    assert {p.name for p in out.iterdir()} == {'diagnosis.json'}


def supplement_fixture(tmp_path):
    source = synthetic_state()
    source['files'] = 1
    path = tmp_path / 'part.json'
    path.write_text(json.dumps(source))
    calibration = tmp_path / 'timecal.json'
    calibration.write_text(json.dumps(dict(data_files=['part.json'])))
    ref = copy.deepcopy(source)
    ref['dtp'] = R.new_state()['dtp']
    ref['ftp'] = R.new_state()['ftp']
    return source, ref, calibration


def test_supplement_requires_exact_scoring_equality_and_preserves_reference(tmp_path):
    source, ref, calibration = supplement_fixture(tmp_path)
    before = copy.deepcopy(ref)
    updated, receipt = D.supplement_timing(ref, calibration, tmp_path)
    assert updated['dtp'] == source['dtp']
    assert ref == before
    assert receipt['matched_scoring_fields'] == [
        'events', 'files', 'exposure_ns', 'lines_z', 'lines_x', 'lines_xz', 'h', 'oot_n']
    assert receipt['files'][0]['sha256'] == V.digest(tmp_path / 'part.json')


@pytest.mark.parametrize('field', ['events', 'lines_z', 'exposure_ns', 'h', 'oot_n'])
def test_supplement_rejects_mismatched_scoring_reference(tmp_path, field):
    _, ref, calibration = supplement_fixture(tmp_path)
    if field == 'h':
        ref['h']['pe_layer_T1'][1] += 1
    elif field == 'oot_n':
        ref['oot_n']['T1'] += 1
    else:
        ref[field] += 1
    with pytest.raises(ValueError, match='differs from frozen scoring reference'):
        D.supplement_timing(ref, calibration, tmp_path)


@pytest.mark.parametrize('names', [[], ['part.json', 'part.json'], ['../part.json'], ['/part.json']])
def test_supplement_rejects_invalid_calibration_list(tmp_path, names):
    _, ref, calibration = supplement_fixture(tmp_path)
    calibration.write_text(json.dumps(dict(data_files=names)))
    with pytest.raises(ValueError, match='unique plain filenames'):
        D.supplement_timing(ref, calibration, tmp_path)


def test_supplement_will_not_replace_nonempty_different_timing(tmp_path):
    _, ref, calibration = supplement_fixture(tmp_path)
    ref['dtp'][0][4][0] = 1
    with pytest.raises(ValueError, match='cannot replace nonempty'):
        D.supplement_timing(ref, calibration, tmp_path)


def test_diagnose_rejects_reference_hash_mismatch(saved_replicas, tmp_path):
    root, reference, _, _ = saved_replicas
    reference.write_text(reference.read_text() + '\n')
    out = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='reference hash mismatch'):
        D.diagnose(root, out, min_entries=1, plots=False)
    assert not out.exists()


@pytest.mark.parametrize('existing_kind', ['file', 'directory'])
def test_diagnose_refuses_overwrite_before_reading_inputs(saved_replicas, tmp_path, existing_kind):
    root, _, _, paired_mock = saved_replicas
    out = tmp_path / 'existing'
    if existing_kind == 'directory':
        out.mkdir()
        sentinel = out / 'diagnosis.json'
    else:
        sentinel = out
    sentinel.write_text('do not overwrite')
    with pytest.raises(FileExistsError):
        D.diagnose(root, out, min_entries=1, plots=False)
    assert sentinel.read_text() == 'do not overwrite'
    paired_mock.assert_not_called()


@pytest.mark.parametrize('minimum', [0, -1, 1.5, True])
def test_diagnose_rejects_invalid_threshold_without_outputs(tmp_path, monkeypatch, minimum):
    paired_mock = Mock(side_effect=AssertionError('must reject before reading inputs'))
    monkeypatch.setattr(V, 'paired', paired_mock)
    out = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='positive integer'):
        D.diagnose(tmp_path / 'missing', out, min_entries=minimum, plots=False)
    paired_mock.assert_not_called()
    assert not out.exists()
