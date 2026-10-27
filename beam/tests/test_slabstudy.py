"""Diagnostic runner contracts; synthetic tables/mocks, no G4 or production jobs."""
import json
from pathlib import Path

import numpy as np
import pytest

from beamfastsim import slabstudy as ss, slabdiffusion as sd, slabrefs


def response(status, energy):
    s, e = np.asarray(status, np.int32).reshape(-1, 1), np.asarray(energy, float).reshape(-1, 1)
    shape = s.shape
    return sd.Response(e, s, np.ones(shape), np.zeros(shape + (3,)),
                       np.broadcast_to([0., 0., 1.], shape + (3,)).copy(), np.ones(shape, np.int32),
                       dict(applied_loss=2.02-e, requested_loss=2.02-e, clipped_loss=np.zeros(shape),
                            mean_loss=np.ones(shape), angular_time=np.zeros(shape),
                            face_reached=np.isin(s, [sd.TRANSMITTED, sd.REFLECTED]),
                            floor_reached=s == sd.UNRESOLVED_FLOOR, rawzero=s == sd.RAWZERO,
                            cap=s == sd.UNRESOLVED_CAP, numerical=s == sd.UNRESOLVED_NUMERICAL))


def test_endpoint_census_and_identification_bounds():
    r = response([sd.TRANSMITTED, sd.REFLECTED, sd.RAWZERO, sd.UNRESOLVED_FLOOR,
                  sd.UNRESOLVED_CAP, sd.UNRESOLVED_NUMERICAL], [1, .05, 0, .02, .5, .01])
    s = ss.summarize(r)
    assert s['n'] == 6 and sum(s['endpoint_counts'].values()) == 6
    assert s['unresolved'] == 3 and s['certified_physical_stop_count'] is None
    assert not s['resolved_algorithmic_distribution']
    assert s['classes']['diagnostic_physical_stop']['count_bounds'] == [1, 4]
    assert s['classes']['threshold_stop']['count_bounds'] == [3, 5]
    assert s['classes']['subthreshold_exit']['count_bounds'] == [1, 4]
    assert s['classes']['above_threshold_exit']['count_bounds'] == [1, 3]
    with pytest.raises(ValueError, match='status'):
        ss.summarize(response([sd.ACTIVE], [1]))


def g4():
    return dict(entry_MeV=np.full(3, 2.02), entry_cos=np.ones(3),
                exit_MeV=np.array([1., .05, 0]), exit=np.array([True, True, False]),
                physical_stop=np.array([False, False, True]), other=np.zeros(3, bool),
                exit_z_mm=np.array([.1, -.1, 0.]))


def test_unresolved_never_enters_cdf(monkeypatch):
    from beamfastsim import foilcompare
    monkeypatch.setattr(foilcompare, 'contrast', lambda *a, **kw: pytest.fail('unresolved ordinary CDF'))
    r = response([sd.UNRESOLVED_FLOOR, sd.UNRESOLVED_CAP, sd.UNRESOLVED_NUMERICAL], [.01, .02, .01])
    ref = ss.reference_summary(g4(), 2.02, 1., .2)
    result = ss.compare_reference(r, g4(), ref)
    assert result['threshold_aligned_distribution'] is None
    # floor and cap below threshold are bounded by monotone debit; numerical is not.
    assert ss.summarize(r)['classes']['threshold_stop']['count_bounds'] == [2, 3]
    assert result['classes']['diagnostic_physical_stop']['delta_fraction_bounds'] == pytest.approx([-1/3, 2/3])


def test_paired_retains_all_identities_and_unresolved():
    a = response([sd.TRANSMITTED, sd.UNRESOLVED_CAP], [1., .4])
    b = response([sd.UNRESOLVED_FLOOR, sd.RAWZERO], [.02, 0])
    diffs, summary = ss.paired(a, b)
    assert summary['n'] == 2
    assert np.sum(summary['transitions_base_rows_candidate_columns']) == 2
    np.testing.assert_array_equal(diffs['delta_lower__threshold_stop'].ravel(), [-1, -1])
    np.testing.assert_array_equal(diffs['delta_upper__threshold_stop'].ravel(), [-1, 0])
    assert summary['both_exit_energy_difference_selection_conditional']['n'] == 0
    assert set(ss.flatten(a)) - {'status'} == {k.removeprefix('delta__') for k in diffs if k.startswith('delta__')}


def test_settings_and_supported_floors():
    base = dict(max_step=.01, angular_fraction=.02, energy_fraction=.05)
    settings = ss.arm_settings(ss.ARMS, .02, .01, .001, 100., **base)
    assert not settings['no_scatter']['scatter']
    assert settings['half_step']['max_step'] == .005
    assert settings['half_angle']['angular_fraction'] == .01
    assert settings['half_energy']['energy_fraction'] == .025
    for k, v in base.items():
        assert settings['refined'][k] == v/2
    assert settings['lower_floor']['floor'] == .01
    for floor in (None, .0001, .02, .1):
        with pytest.raises(ValueError, match='lower_floor'):
            ss.arm_settings(['lower_floor'], .02, floor, .001, 100., **base)


@pytest.fixture
def mocked_tables(tmp_path, monkeypatch):
    table = tmp_path / 'slab.npz'
    table.write_bytes(b'synthetic mocked NPZ')
    exported = tmp_path / 'export.dat'
    exported.write_text('synthetic export identity')
    monkeypatch.setattr(ss.g4tables, 'SLAB', table)
    nodes = np.array([.001, 1., 10.])
    arrays = dict(energy=nodes, msc_energy=nodes, msc_lambda1=np.ones(3),
                  **{'csda_range_mu-': nodes, 'csda_range_mu+': nodes * 2})
    monkeypatch.setattr(ss.g4tables, 'load', lambda *a, **kw: arrays)
    check = dict(all_arrays_equal=True, material_equal=True,
                 fresh=dict(couples={q: dict(energy_cuts_MeV={'e-': .7}) for q in ('mu-', 'mu+')},
                            provenance=dict(files={'export': ss._identity(exported)}, audit=dict(artifacts={}))))
    def compare(directory, existing, material, *, require_audited_export):
        assert require_audited_export and material == 'G4_Al' and existing == table
        return check
    monkeypatch.setattr(ss.g4tables, 'compare', compare)
    return check, exported


def cli(tmp_path, *extra):
    return ['--cell', 'moved_edge', '--pdg', '13', '--samples', '3', '--seed', '71',
            '--slab-tables', str(tmp_path), '--out', str(tmp_path / 'output'), *extra]


def test_cli_preserves_response_ledger_pairing_and_identity(tmp_path, monkeypatch, mocked_tables):
    calls = []
    def respond(e, u, thickness, tb, **kw):
        calls.append(kw)
        assert e.tolist() == [2.02] and thickness == .2
        assert kw['tcut'] == .7 and kw['loss'] == 'sample'
        np.testing.assert_allclose(np.exp(tb.logr), [.001, 1., 10.])
        return response([sd.TRANSMITTED, sd.RAWZERO, sd.UNRESOLVED_CAP], [1., 0., .3])
    monkeypatch.setattr(sd, 'respond', respond)
    result = ss.main(cli(tmp_path))
    assert result['complete'] and len(calls) == 2
    assert calls[0]['seed'] == calls[1]['seed'] == 71
    assert not calls[0]['scatter'] and calls[1]['scatter']
    root = tmp_path / 'output'
    saved = json.loads((root / 'summary.json').read_text())
    assert saved['complete'] and saved['sources'] and saved['core_sources'] and saved['inputs']
    for arm in ('no_scatter', 'diffuse'):
        with np.load(root / (arm + '.npz'), allow_pickle=False) as f:
            assert set(ss.flatten(response([1], [1.]))) <= set(f.files)
            np.testing.assert_array_equal(f['replica_ids'], [0, 1, 2])
            np.testing.assert_array_equal(f['status'].ravel(), [1, 4, 5])
    assert 'no_scatter_minus_diffuse' in saved['paired']
    ss.verify(saved['sources'] + saved['core_sources'] + saved['inputs'] + saved['outputs'])
    with pytest.raises(FileExistsError):
        ss.main(cli(tmp_path))


def test_changed_inputs_leave_incomplete(tmp_path, monkeypatch, mocked_tables):
    _, exported = mocked_tables
    def respond(*args, **kw):
        exported.write_text('modified during run')
        return response([sd.TRANSMITTED] * 3, [1.] * 3)
    monkeypatch.setattr(sd, 'respond', respond)
    with pytest.raises(ValueError, match='identity changed'):
        ss.main(cli(tmp_path))
    assert not json.loads((tmp_path / 'output' / 'summary.json').read_text())['complete']


def test_table_mismatch_rejected_before_output(tmp_path, mocked_tables):
    mocked_tables[0]['all_arrays_equal'] = False
    with pytest.raises(ValueError, match='differs'):
        ss.main(cli(tmp_path))
    assert not (tmp_path / 'output').exists()


def test_existing_campaign_uses_loader_and_never_scores_unresolved_cdf(tmp_path, monkeypatch, mocked_tables):
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    (campaign / 'manifest.json').write_text(json.dumps(dict(complete=True, runs={
        'sample': dict(scan='moved_edge', pdg=13, thickness_mm=.2)})))
    def load_runs(roots, cell, pdg, thickness, energy, cosine):
        assert roots == [str(campaign)]
        return {'full_finest': g4()}, []
    monkeypatch.setattr(slabrefs, 'load_references', load_runs)
    monkeypatch.setattr(sd, 'respond', lambda *args, **kw: response([sd.UNRESOLVED_CAP] * 3, [.2] * 3))
    result = ss.main(cli(tmp_path, '--campaign', str(campaign)))
    assert result['g4']['full_finest']['n'] == 3
    assert result['arms']['diffuse']['vs_g4']['full_finest']['threshold_aligned_distribution'] is None


def test_tiny_kernel_batch_independence_and_charge_tables(tmp_path, mocked_tables):
    # Real kernel, synthetic audited-table mock. Zero cap is cheap and explicitly unresolved.
    args = cli(tmp_path, '--arms', 'diffuse', '--max-steps', '0', '--batch', '1', '--pdg', '-13')
    first = ss.main(args)
    args[args.index('--out') + 1] = str(tmp_path / 'second')
    args[args.index('--batch') + 1] = '3'
    second = ss.main(args)
    with np.load(tmp_path / 'output' / 'diffuse.npz') as a, np.load(tmp_path / 'second' / 'diffuse.npz') as b:
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k])
    assert first['arms']['diffuse']['summary']['unresolved'] == 3
    assert second['arms']['diffuse']['summary']['classes']['diagnostic_physical_stop']['count_bounds'] == [0, 3]



def test_fully_resolved_threshold_distribution_is_explicitly_allowed():
    r = response([sd.TRANSMITTED, sd.REFLECTED, sd.RAWZERO], [1., .05, 0.])
    result = ss.compare_reference(r, g4(), ss.reference_summary(g4(), 2.02, 1., .2))
    assert result['threshold_aligned_distribution']['threshold_aligned']
    assert result['threshold_aligned_distribution']['cdf']['distance'] == 0
    assert result['classes']['threshold_stop']['delta_fraction_bounds'] == [0., 0.]


@pytest.mark.parametrize('z', [.0, .05, np.nan, np.inf])
def test_nonaxial_reference_exit_rejected(z):
    g = g4()
    g['exit_z_mm'][0] = z
    with pytest.raises(ValueError, match='non-axial'):
        ss.reference_summary(g, 2.02, 1., .2)


def test_floor_identifies_only_threshold_transformed_distribution():
    r = response([sd.TRANSMITTED, sd.UNRESOLVED_FLOOR, sd.UNRESOLVED_CAP], [1., .0001, .02])
    result = ss.compare_reference(r, g4(), ss.reference_summary(g4(), 2.02, 1., .2))
    assert result['threshold_distribution_identified']
    assert result['threshold_aligned_distribution']['cdf']['distance'] == 0
    assert not ss.summarize(r)['certified_physical_endpoint_distribution']
    assert not ss.summarize(r)['resolved_algorithmic_distribution']


def test_saved_response_reuse_and_identity_guard(tmp_path, monkeypatch, mocked_tables):
    monkeypatch.setattr(sd, 'respond', lambda *a, **kw: response([sd.TRANSMITTED] * 3, [1.] * 3))
    original = ss.main(cli(tmp_path))
    monkeypatch.setattr(sd, 'respond', lambda *a, **kw: pytest.fail('saved response rerun'))
    args = cli(tmp_path, '--responses', str(tmp_path / 'output'))
    args[args.index('--out') + 1] = str(tmp_path / 're_report')
    report = ss.main(args)
    assert report['complete'] and all(a['response_reused'] for a in report['arms'].values())
    assert report['tolerances'] == ss.TOLERANCES and 'scipy' in report['environment']
    assert any(Path(x['path']).name == 'foilcampaign.py' for x in report['sources'])
    for arm in original['arms']:
        with np.load(original['arms'][arm]['output']['path']) as a, np.load(report['arms'][arm]['output']['path']) as b:
            for k in a.files:
                np.testing.assert_array_equal(a[k], b[k])
    args[args.index('--out') + 1] = str(tmp_path / 'bad_identity')
    args[args.index('--seed') + 1] = '72'
    with pytest.raises(ValueError, match='identity differs'):
        ss.main(args)
    assert not (tmp_path / 'bad_identity').exists()


def test_split_cli_restricted_charge_binding_and_reuse(tmp_path, monkeypatch, mocked_tables):
    charge_calls = []
    scale = [2.]
    def native(directory, *, material, particle):
        charge_calls.append(particle)
        assert material == 'G4_Al'
        e = np.geomspace(.001, 100., 101)
        return ({'Range': dict(energy=e, values=scale[0]*e),
                 'DEDX': dict(energy=e, values=np.full_like(e, 1/scale[0])),
                 'DEDXnr': dict(energy=e, values=np.ones_like(e))},
                dict(energy_cuts_MeV={'e-': .7}), [])
    monkeypatch.setattr(ss.pt, 'load_target', native)
    def respond(e, u, thickness, tb, **kw):
        assert kw['loss'] == 'split' and kw['range_kind'] == 'restricted'
        np.testing.assert_allclose(np.exp(tb.logr), 2*np.exp(tb.loge))
        return response([sd.TRANSMITTED] * 3, [1.] * 3)
    monkeypatch.setattr(sd, 'respond', respond)
    args = cli(tmp_path, '--loss-model', 'split', '--pdg', '-13')
    original = ss.main(args)
    assert charge_calls == ['mu+']
    assert original['transport_binding']['charge'] == 'mu+'
    assert original['split_mean_audit']['range_interpolation_audit']['max_midpoint_power_relative_difference'] < 1e-12
    monkeypatch.setattr(sd, 'respond', lambda *a, **kw: pytest.fail('saved response rerun'))
    args += ['--responses', str(tmp_path / 'output')]
    args[args.index('--out') + 1] = str(tmp_path / 're_report')
    report = ss.main(args)
    assert report['response_generation_provenance'] == original['response_generation_provenance']
    scale[0] = 2.1
    args[args.index('--out') + 1] = str(tmp_path / 'bad_range')
    with pytest.raises(ValueError, match='transport table/material binding'):
        ss.main(args)
    assert not (tmp_path / 'bad_range').exists()
