"""Independent two-level setting contrasts and shared-arm covariance, no sampling."""
import numpy as np
import pytest

from beamfastsim import edgefactorialreport as er
from beamfastsim.foilcompare import moments


def samples():
    base = np.array([0., .12, .24, .42, .75, 1.])
    return [base, base * 1.1, base * 1.2, base * 1.3]


def test_independent_main_effects_and_interaction_algebra():
    values = samples()
    report = er.factorial(values)
    for metric in ('mean', 'variance'):
        arm = np.array([moments(v)[metric] for v in values])
        for name, weights in er.WEIGHTS.items():
            assert report['estimates'][metric][name]['delta'] == pytest.approx(np.dot(weights, arm))
    result = report['estimates']['mean']
    assert result['step_main_average']['delta'] == pytest.approx(.2 * values[0].mean())
    assert result['bins_main_average']['delta'] == pytest.approx(.1 * values[0].mean())
    assert result['step_bin_interaction']['delta'] == pytest.approx(0., abs=1e-15)


def test_shared_arm_covariance_is_not_independence_of_contrasts():
    values = samples()
    report = er.factorial(values)
    weights = np.array(list(er.WEIGHTS.values()))
    se2 = np.array([moments(v)['variance_se'] ** 2 for v in values])
    expected = (weights * se2) @ weights.T
    np.testing.assert_allclose(report['contrast_covariance']['variance'], expected)
    names = list(er.WEIGHTS)
    a, b = names.index('step_at_bins7'), names.index('bins_at_stock_step')
    assert expected[a, b] == pytest.approx(se2[0])
    assert expected[a, b] > 0
    for i, name in enumerate(names):
        assert report['estimates']['variance'][name]['se'] == pytest.approx(np.sqrt(expected[i, i]))


def test_threshold_transform_retains_every_primary():
    raw = np.array([0., .01, .0999999, .1, .2, 1.])
    y = er.threshold(raw)
    np.testing.assert_array_equal(y, [0., 0., 0., .1, .2, 1.])
    report = er.factorial([y] * 4)
    assert report['arms'][0]['threshold_fraction'] == .5
    assert report['estimates']['threshold_fraction']['step_bin_interaction']['delta'] == 0.


def test_interaction_se_uses_all_four_independent_arms():
    values = samples()
    report = er.factorial(values)
    expected = np.sqrt(sum(moments(v)['mean_se'] ** 2 for v in values))
    assert report['estimates']['mean']['step_bin_interaction']['se'] == pytest.approx(expected)


def test_refuses_missing_arms_or_invalid_vectors():
    with pytest.raises(ValueError, match='four'):
        er.factorial(samples()[:3])
    with pytest.raises(ValueError, match='finite'):
        er.factorial([np.array([0., np.nan])] * 4)


def pipeline(tmp_path, monkeypatch, *, dense_equal=True, runtime_change=False):
    import json
    root, reused = tmp_path / 'newrefs', tmp_path / 'reused'
    root.mkdir()
    reused.mkdir()
    ref = dict(event=np.arange(1, 5001), exit_MeV=np.tile([0., .2, .3, .4], 1250))
    seeds = iter(range(1, 20))
    for cell in er.edgeg4.CELLS:
        for mode in er.ORDER:
            directory = (reused if mode == 'full_refined' else root) / (cell + '_' + mode)
            directory.mkdir()
            fraction, final_range, bins = er.ef.MODES.get(mode, (.005, .00005, 168))
            case = dict(events=5000, seed=next(seeds), bins=bins, step_fraction=fraction,
                        final_range=final_range, lateral_displacement=True)
            receipt = dict(case=case, sources=[], artifacts=[], tables=[], output={})
            (directory / 'receipt.json').write_text(json.dumps(receipt))
            for name in ('expanded.fcl', 'preinit.mac'):
                (directory / name).write_text('fixture')
            (directory / 'runtime.sha256').write_text('changed' if runtime_change and mode == 'full_refined' else 'identical')
    monkeypatch.setattr(er.edgeg4, 'checked', lambda item: item)
    monkeypatch.setattr(er.edgeg4, 'signature', lambda *a: {})
    monkeypatch.setattr(er.ef, 'macro_signature', lambda *a: {})
    monkeypatch.setattr(er.ef, 'native_signature', lambda *a: {})
    monkeypatch.setattr(er.slabrefs, 'load_references', lambda *a: (dict.fromkeys(set(er.ORDER) | set(er.edgeg4.MODES), ref), []))
    monkeypatch.setattr(er.foilreport, 'audit', lambda *a: (dict(primaries=5000, other=0), ref, {}))
    monkeypatch.setattr(er.g4tables, 'compare', lambda directory, *a, **k:
        dict(all_arrays_equal=dense_equal and ('bins168' in directory.name or directory.name.endswith('full_refined'))))
    vector = dict(energy=np.array([.01, .1, 1., 3., 10.]), values=np.ones(5))
    monkeypatch.setattr(er.pt, 'load_target', lambda *a, **k: (dict(Range=vector), {}, []))
    historical = dict(inputs=[], references={})
    for cell in er.edgeg4.CELLS:
        path = tmp_path / (cell + '_stock.npz')
        np.savez(path, exit_MeV=ref['exit_MeV'])
        historical['references'][cell + '_full'] = dict(identity=dict(path=str(path)))
    monkeypatch.setattr(er.ld, 'analyze', lambda *a: historical)
    return root, reused


def test_complete_report_reuses_baseline_and_accepts_changed_coarse_grid(tmp_path, monkeypatch):
    import json
    root, reused = pipeline(tmp_path, monkeypatch)
    out = tmp_path / 'report'
    result = er.run(root, reused, [], 'recovered', tmp_path / 'package.npz', out)
    assert result['complete'] and not result['physics_closure']
    assert len(result['references']) == 8
    assert len(result['factorial']) == 2
    assert 'UNVERIFIED' in result['historical']['moved_edge']['qualification']
    assert not result['references']['moved_edge_stock_step_bins7']['table_comparison']['all_arrays_equal']
    assert result['references']['moved_edge_stock_step_bins168']['table_comparison']['all_arrays_equal']
    assert json.loads((out / 'summary.json').read_text())['complete']
    with pytest.raises(FileExistsError):
        er.run(root, reused, [], 'recovered', tmp_path / 'package.npz', out)


def test_report_refuses_changed_dense_grid(tmp_path, monkeypatch):
    root, reused = pipeline(tmp_path, monkeypatch, dense_equal=False)
    with pytest.raises(ValueError, match='dense tables'):
        er.run(root, reused, [], 'recovered', tmp_path / 'package.npz', tmp_path / 'report')


def test_report_refuses_unmatched_runtime(tmp_path, monkeypatch):
    root, reused = pipeline(tmp_path, monkeypatch, runtime_change=True)
    with pytest.raises(ValueError, match='runtime/binary'):
        er.run(root, reused, [], 'recovered', tmp_path / 'package.npz', tmp_path / 'report')


def test_report_uses_completed_fast_arrays_without_sampler(tmp_path, monkeypatch):
    root, reused = pipeline(tmp_path, monkeypatch)
    values = np.tile([0., .2, .3, .4], 2000)
    report = dict(args=dict(cell='moved_edge'), reference_configuration_qualification='UNVERIFIED historical binding',
                  arms=dict(diffuse=dict(output={})))
    monkeypatch.setattr(er.le, 'read_study', lambda *a: (report, dict(diffuse={}), {}))
    monkeypatch.setattr(er.le, 'threshold_array', lambda *a: values)
    result = er.run(root, reused, ['completed_fast'], 'recovered', tmp_path / 'package.npz', tmp_path / 'report')
    entry = result['fast']['completed_fast_diffuse']
    assert 'UNVERIFIED' in entry['qualification']
    assert set(entry['references']) == set(er.ORDER)
    assert entry['references']['full_refined']['candidate']['n'] == 8000


def test_shape_contrast_caps_all_entries_and_preserves_exact_mixture():
    a = np.array([0., .08, .12, .3, .6, 1.2])
    b = np.array([0., .02, .15, .25, .5, .8])
    result = er.shape_contrast(a, b)
    delta = er.threshold(a).var() - er.threshold(b).var()
    assert result['variance_explanation']['population_variance_delta'] == pytest.approx(delta)
    for cap, item in result['capped_variance_diagnostics'].items():
        expected = np.minimum(er.threshold(a), float(cap)).var(ddof=1) - np.minimum(er.threshold(b), float(cap)).var(ddof=1)
        assert item['delta'] == pytest.approx(expected)
        assert item['candidate']['n'] == item['reference']['n'] == 6
        assert item['se'] >= 0.
