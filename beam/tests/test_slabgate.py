"""Synthetic saved-report tests; no transport or reference physics runs."""
import json
from copy import deepcopy

import pytest

from beamfastsim import slabgate as sg


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / 'study'
    root.mkdir()
    for name in ('input.dat', 'output.dat', 'old_source.py'):
        (root / name).write_text(name)
    identity = lambda name: sg._identity(root / name)
    bound = dict(fraction_bounds=[.1, .1], interval_99pct=[.08, .12])
    distribution = dict(candidate=dict(variance=1.2), reference=dict(variance=1.),
                        mean_exit=dict(delta=.001, se=.0001, tolerance=.002, state='PASS'),
                        variance=dict(delta=.2, se=.01, tolerance=.05, state='FAIL'),
                        cdf=dict(distance=.03, interval=[0., .06], radius_99pct_dkw=.03, state='INDETERMINATE'))
    comparison = dict(threshold_aligned_distribution=distribution, threshold_distribution_identified=True,
                      classes=dict(threshold_stop=dict(delta_fraction_bounds=[0., 0.], interval_99pct=[-.01, .01])))
    modes = [f'{family}{suffix}' for family in ('loss', 'full') for suffix in ('', '_fine', '_finer', '_finest')]
    report = dict(complete=True, args=dict(cell='edge', pdg=-13, loss_model='split'), material=dict(name='G4_Al'),
                  inputs=[identity('input.dat')], outputs=[identity('output.dat')],
                  sources=[identity('old_source.py')], core_sources=[identity('old_source.py')],
                  response_generation_provenance=dict(sources=[identity('old_source.py')], seed=72),
                  reference_configuration_qualification='UNVERIFIED: configuration not matched',
                  threshold_MeV=.1, tolerances=dict(stop_abs=.02), environment=dict(python='historical'),
                  split_mean_audit=dict(range_interpolation_audit=dict(max_difference=.01),
                                        max_total_mean_relative_difference=.03, energy_MeV=[1, 2]),
                  g4={m: dict(n=100, endpoint_counts=dict(physical_stop=5, threshold_stop=10),
                              fraction_intervals=dict(threshold_stop=bound)) for m in modes}, arms={})
    for arm in ('no_scatter', 'diffuse', 'refined'):
        report['arms'][arm] = dict(summary=dict(n=100, unresolved=2,
                                              endpoint_counts=dict(UNRESOLVED_FLOOR=2, RAWZERO=10),
                                              classes=dict(threshold_stop=bound)),
                                   vs_g4={m: deepcopy(comparison) for m in modes})
    def save():
        (root / 'summary.json').write_text(json.dumps(report))
        return root
    def forbidden(*args, **kwargs):
        pytest.fail('report-only path must not run physics')
    monkeypatch.setattr(sg.slabstudy, 'run', forbidden)
    monkeypatch.setattr(sg.slabstudy.sd, 'respond', forbidden)
    save()
    return root, report, save


def test_report_provenance_diagnostics_and_cli(study, tmp_path, capsys):
    root, original, _ = study
    (root / 'old_source.py').unlink()
    out = tmp_path / 'new.json'
    result = sg.main(['--study', str(root), str(root), '--out', str(out)])
    assert json.loads(out.read_text()) == result
    assert len(result['rows']) == 6
    row = result['rows'][0]
    assert row['reference'] == 'loss_finest'
    assert result['rows'][1]['reference'] == 'full_finest'
    assert row['mean']['delta'] == .001 and row['mean']['state'] == 'PASS'
    assert row['variance'] == dict(fast=1.2, reference=1., delta=.2, se=.01, tolerance=.05,
                                   ratio=1.2, relative=.2, state='FAIL')
    assert row['cdf']['interval'] == [0., .06]
    assert row['threshold']['state'] == 'PASS'
    assert row['unresolved']['fast'] == 2
    assert row['endpoint_counts']['fast']['RAWZERO'] == 10
    assert row['audits']['range'] == dict(max_difference=.01)
    assert 'energy_MeV' not in row['audits']['mean']
    provenance = result['studies'][0]
    assert provenance['original_report'] == sg._identity(root / 'summary.json')
    assert provenance['inputs'] == original['inputs'] and provenance['outputs'] == original['outputs']
    assert provenance['historical_source_provenance']['sources'] == original['sources']
    assert provenance['historical_source_provenance']['response_generation_provenance']['seed'] == 72
    assert provenance['historical_sources_verified'] is False
    assert result['global_pass'] is None and not result['physical_stop_closure']
    text = capsys.readouterr().out
    assert 'cell\tPDG\tarm\tref' in text and 'loss_finest' in text and 'full_finest' in text
    assert 'not physical-stop' in text
    with pytest.raises(FileExistsError):
        sg.main(['--study', str(root), '--out', str(out)])


@pytest.mark.parametrize('suffix', ['_finest', '_finer', '_fine', ''])
def test_reference_priority(study, suffix):
    root, report, save = study
    order = ['_finest', '_finer', '_fine', '']
    for family in ('loss', 'full'):
        for higher in order[:order.index(suffix)]:
            del report['g4'][family + higher]
    save()
    rows = sg.summarize([root])['rows']
    assert [r['reference'] for r in rows] == ['loss' + suffix, 'full' + suffix, 'full' + suffix]


@pytest.mark.parametrize('remaining,missing_arm', [('loss', 'diffuse'), ('full', 'no_scatter')])
def test_no_cross_family_closure(study, remaining, missing_arm):
    root, report, save = study
    report['g4'] = {k: v for k, v in report['g4'].items() if k.startswith(remaining)}
    save()
    row = next(r for r in sg.summarize([root])['rows'] if r['arm'] == missing_arm)
    assert row['reference'] is None and row['reference_state'] == 'UNVERIFIED'
    for key in ('mean', 'variance', 'cdf', 'threshold'):
        assert row[key]['state'] == 'UNVERIFIED'


@pytest.mark.parametrize('name', ['input.dat', 'output.dat'])
def test_manifest_tampering_rejected(study, tmp_path, name):
    root, _, _ = study
    (root / name).write_text('tampered')
    out = tmp_path / 'rejected.json'
    with pytest.raises(ValueError, match='identity changed'):
        sg.main(['--study', str(root), '--out', str(out)])
    assert not out.exists()


def test_incomplete_and_missing_manifest_rejected(study):
    root, report, save = study
    report['complete'] = False
    save()
    with pytest.raises(ValueError, match='completed'):
        sg.summarize([root])
    report['complete'] = True
    del report['outputs']
    save()
    with pytest.raises(KeyError):
        sg.summarize([root])


def test_historical_unidentified_and_missing_comparison(study):
    root, report, save = study
    del report['reference_configuration_qualification']
    del report['args']['loss_model']
    compare = report['arms']['no_scatter']['vs_g4']['loss_finest']
    compare['threshold_aligned_distribution'] = None
    compare['threshold_distribution_identified'] = False
    del report['arms']['diffuse']['vs_g4']['full_finest']
    save()
    result = sg.summarize([root])
    row = result['rows'][0]
    assert row['reference_configuration_qualification'] == 'UNVERIFIED'
    assert row['loss_model'] == 'hybrid' and row['mean']['state'] == 'UNVERIFIED'
    assert row['threshold']['state'] == 'PASS' and row['unresolved']['fast'] == 2
    assert result['rows'][1]['reference_state'] == 'UNVERIFIED'
    assert result['rows'][1]['reference'] == 'full_finest'
    assert result['studies'][0]['reference_configuration_qualification'] == 'UNVERIFIED'


@pytest.mark.parametrize('interval,state', [([.03, .05], 'FAIL'), ([-.05, -.03], 'FAIL'),
                                           ([-.01, .03], 'INDETERMINATE'), ([-.02, .02], 'PASS')])
def test_threshold_interval_states(study, interval, state):
    root, report, save = study
    comparison = report['arms']['no_scatter']['vs_g4']['loss_finest']
    comparison['classes']['threshold_stop']['interval_99pct'] = interval
    comparison['threshold_aligned_distribution']['reference']['variance'] = 0.
    save()
    row = sg.summarize([root])['rows'][0]
    assert row['threshold']['state'] == state
    assert row['variance']['ratio'] is None and row['variance']['relative'] is None


def test_missing_referenced_file_rejected(study):
    root, _, _ = study
    (root / 'output.dat').unlink()
    with pytest.raises(FileNotFoundError):
        sg.summarize([root])
