"""Reduced-data recovery bounds and sizing tests; no transport or Geant4 runs."""
import json
import numpy as np
import pytest

from beamfastsim import lossdiagnostic as ld


def test_mixture_identity_and_se():
    x = np.array([0., .02, .09, .1, .3, .8, 1.])
    result = ld.mixture(x)
    y = np.where(x < .1, 0., x)
    assert result['above_fraction'] == 4/7
    assert result['within_population_variance'] + result['between_population_variance'] == pytest.approx(y.var())
    assert result['transformed']['variance'] == pytest.approx(y.var(ddof=1))
    assert result['within_se'] > 0 and result['between_se'] > 0
    assert result['raw']['n'] == len(x)


@pytest.mark.parametrize('x', [[0., 0.], [.01, .09], [.1, .2], [.2, .2]])
def test_mixture_degenerate_and_boundary(x):
    result = ld.mixture(x)
    assert result['decomposition_residual'] == pytest.approx(0., abs=1e-15)
    assert result['above_fraction'] == np.mean(np.asarray(x) >= .1)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('x', [[1.], [0., np.nan], [0., -1.], [[1., 2.]]])
def test_invalid_energies(x):
    with pytest.raises(ValueError):
        ld.mixture(x)


def test_precision_both_samples_and_headroom():
    r = ld.precision_plan()
    assert r['minimum_reference_infinite_fast_zero_distance'] == 7490
    assert r['minimum_each_equal_samples_zero_distance'] == 29958
    assert r['minimum_each_equal_samples_with_headroom'] == 119830
    assert r['current_reference_5000_radius'] > .02
    n = r['minimum_reference_with_fast_and_headroom']
    coefficient = np.log(400.)/2
    assert np.sqrt(coefficient/n)+r['fast_radius']+.01 <= .02
    assert np.sqrt(coefficient/(n-1))+r['fast_radius']+.01 > .02
    assert ld.precision_plan(fast_samples=5000)['minimum_reference_with_fast_and_headroom'] is None


@pytest.mark.parametrize('kwargs', [dict(band=0), dict(alpha=1), dict(margin=.02),
                                  dict(fast_samples=1), dict(fast_samples=True), dict(fast_samples=2.5)])
def test_bad_precision(kwargs):
    with pytest.raises(ValueError):
        ld.precision_plan(**kwargs)


def test_one_sided_floor_bound():
    x = np.array([0., .2, 0., .8])
    floor = .0001
    perturbation = np.array([floor, 0., floor/2, 0.])
    assert abs((x+perturbation).var()-x.var()) <= ld.residual_variance_bound(x.var(), floor)
    assert ld.residual_variance_bound(0., 0.) == 0.
    with pytest.raises(ValueError):
        ld.residual_variance_bound(1., -1.)


def reference():
    data = dict(event=np.arange(1, 5), track=np.ones(4), entry_MeV=np.full(4, 2.02),
                entry_cos=np.ones(4), exit_MeV=np.array([0., .05, .2, .4]),
                exit=np.array([False, True, True, True]), physical_stop=np.array([True, False, False, False]),
                other=np.zeros(4, bool))
    summary = dict(case=dict(events=4, energy_MeV=2.02, cosine=1.), primaries=4, exits=3, physical_stops=1,
                   threshold_stops=2, subthreshold_exits=1, other=0,
                   mean_exit_MeV=float(data['exit_MeV'].mean()), var_exit_MeV2=float(data['exit_MeV'].var(ddof=1)))
    return data, summary


@pytest.mark.parametrize('legacy', [False, True])
def test_reduced_census(legacy):
    data, summary = reference()
    if legacy:
        del data['entry_cos']
    result = ld.reduced_audit(data, summary)
    assert result['physical_stops'] == 1 and result['subthreshold_exits'] == 1


@pytest.mark.parametrize('change', ['event', 'entry', 'shape', 'flag', 'stop', 'nan', 'track', 'mean', 'var', 'count'])
def test_reduced_audit_fails_closed(change):
    data, summary = reference()
    if change == 'event':
        data['event'][0] = 2
    elif change == 'entry':
        data['entry_MeV'][0] = 1.
    elif change == 'shape':
        data['extra'] = np.ones(3)
    elif change == 'flag':
        data['exit'] = np.array([0, 1, 1, 2])
    elif change == 'stop':
        data['exit_MeV'][0] = .01
    elif change == 'nan':
        data['exit_MeV'][2] = np.nan
    elif change == 'track':
        data['track'][1] = 1.5
    elif change == 'mean':
        summary['mean_exit_MeV'] += .01
    elif change == 'var':
        summary['var_exit_MeV2'] += .01
    else:
        summary['threshold_stops'] += 1
    with pytest.raises(ValueError):
        ld.reduced_audit(data, summary)


def test_hash_size_tampering(tmp_path):
    path = tmp_path / 'input'
    path.write_text('original')
    record = ld._identity(path)
    assert ld._checked(record, path) == record
    path.write_text('modified')
    with pytest.raises(ValueError, match='hash/size'):
        ld._checked(record, path)


def test_cli_fresh_only(tmp_path, monkeypatch):
    called = []
    def analyze(*args):
        called.append(args)
        return dict(references={}, reference_configuration_qualification='UNVERIFIED', precision=ld.precision_plan())
    monkeypatch.setattr(ld, 'analyze', analyze)
    out = tmp_path / 'new.json'
    args = ['--recovered-root', 'saved', '--out', str(out)]
    result = ld.main(args)
    assert len(called) == 1 and json.loads(out.read_text()) == result
    with pytest.raises(FileExistsError):
        ld.main(args)
    assert len(called) == 1


def test_correlation_constant():
    assert ld._correlation(np.ones(3), np.arange(3)) is None
    assert ld._correlation(np.arange(3), np.arange(3)) == pytest.approx(1.)


@pytest.fixture
def recovery(tmp_path):
    root = tmp_path / 'recovered'
    final = root / 'foil_threadG_final_report_20261026_v2'
    final.mkdir(parents=True)
    export = root / 'foil_threadG_smoke_20261026' / 'tables'
    export.mkdir(parents=True)
    (export / 'couple.dat').write_text('COUPLE-V3.0\n1\n0\nG4_Al\nDefaultRegionForTheWorld\n.01\n.01\n.01\n0\n')
    (export / 'material.dat').write_text('MATERIAL-V3.0\n1\nG4_Al 2.699\n')
    (export / 'cut.dat').write_text('CUT-V3.0\n1\n.01 .99\n.01 34.10575\n.01 33.72249\n0 0\n')
    grid = np.geomspace(.0001, 1000., 30)
    for charge in ('mu-', 'mu+'):
        for name in ld.pt.TABLES:
            values = grid/2 if name in ('Range', 'CSDARange') else np.full(len(grid), 2.)
            path = export / f'{name}.muIoni.{charge}.asc'
            path.write_text('1\n2\n' + f'{grid[0]} {grid[-1]} {len(grid)}\n{len(grid)}\n' +
                            '\n'.join(f'{x:.17g} {y:.17g}' for x, y in zip(grid, values))+'\n')
    packages = tmp_path / 'packages'
    packages.mkdir()
    meta = dict(material='G4_Al', couple=dict(material='G4_Al'))
    arrays = dict(energy=grid, msc_energy=grid, msc_lambda1=np.ones_like(grid),
                  **{f'{name}_{charge}': grid/2 if name == 'csda_range' else np.full(len(grid), 2.)
                     for charge in ('mu-', 'mu+') for name in ('csda_range', 'dedx_nr')})
    np.savez(packages / 'g4_al_mu.npz', **arrays, meta=json.dumps(meta))
    meta = dict(material='StoppingTarget_Al', couple=dict(material='StoppingTarget_Al'))
    np.savez(packages / 'g4_target_al_mu.npz', **arrays, meta=json.dumps(meta))
    data, summary = reference()
    del data['entry_cos']
    summary['case'].update(thickness_mm=.2, pdg=13, seed=101)
    a = final / 'g4_moved_edge_full.npz'
    np.savez(a, **data)
    second = json.loads(json.dumps(summary))
    second['case']['seed'] = 102
    b = final / 'g4_moved_edge_loss.npz'
    np.savez(b, **dict(data, track=data['track']+1))
    historical = dict(complete=True, g4=dict(moved_edge_full=summary, moved_edge_loss=second),
                      receipts=[dict(path='/unavailable/receipt.json', bytes=1, sha256='0'*64)],
                      material={charge: dict(table_identities=[ld._identity(p) for p in
                                [export/n for n in ('couple.dat', 'cut.dat', 'material.dat')] +
                                [export/f'{n}.muIoni.{charge}.asc' for n in ld.pt.TABLES]]) for charge in ('mu-', 'mu+')})
    report = final / 'summary.json'
    report.write_text(json.dumps(historical))
    manifest = dict(report=ld._identity(report), outputs=[ld._identity(a)])
    (final / 'manifest.json').write_text(json.dumps(manifest))
    return root, packages, final, historical


def test_recovery_pipeline_preserves_evidence_tiers(recovery):
    root, packages, _, _ = recovery
    result = ld.analyze(root, packages)
    json.dumps(result, allow_nan=False)
    assert len(result['references']) == 2 and not result['physics_closure']
    assert result['reference_configuration_qualification'] == 'UNVERIFIED'
    full, loss = [result['references'][f'moved_edge_{name}'] for name in ('full', 'loss')]
    assert full['historical_array_hash_verified'] and not loss['historical_array_hash_verified']
    assert not full['entry_cosine_verified'] and not full['trees_reaudited']
    assert result['tables']['mu-']['all_consumed_loss_tables_hash_verified']
    assert not result['tables']['mu-']['packaged_source_hash_coverage']
    assert not result['tables']['mu-']['overlapping_hard_bohr_screen']['default_leg_screen_reaches_unsupported']
    assert result['configuration_inventory']['original_receipts_available'] == 0
    assert result['comparisons']['moved_edge_full_minus_loss']['cdf']['distance'] == 0.


@pytest.mark.parametrize('kind', ['summary', 'array', 'table', 'seed'])
def test_recovery_tampering_rejected(recovery, kind):
    root, packages, final, historical = recovery
    if kind == 'summary':
        (final / 'summary.json').write_text('{}')
    elif kind == 'array':
        (final / 'g4_moved_edge_full.npz').write_text('tampered')
    elif kind == 'table':
        (root / 'foil_threadG_smoke_20261026/tables/Range.muIoni.mu-.asc').write_text('tampered')
    else:
        historical['g4']['moved_edge_loss']['case']['seed'] = 101
        report = final / 'summary.json'
        report.write_text(json.dumps(historical))
        manifest = json.loads((final / 'manifest.json').read_text())
        manifest['report'] = ld._identity(report)
        (final / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ld.analyze(root, packages)


def test_variance_precision_is_pilot_proxy():
    stats = dict(n=5000, variance=.0045, variance_se=.00011)
    result = ld.variance_precision(stats)
    assert result['reference_samples'] == int(np.ceil(5000*(3*.00011/(.025*.0045))**2))
    assert ld.variance_precision(dict(stats, variance=0))['reference_samples'] is None
    with pytest.raises(ValueError):
        ld.variance_precision(stats, 0.)
