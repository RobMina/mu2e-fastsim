"""Recorded decrement/chord budgets and independent step-setting diagnostics."""
import json

import numpy as np
import pytest

from beamfastsim import edgestepgeometry as sg, foilreport as fr
from test_edgeangles import Chunks
from test_foilclosure import fixture


def tables(low=.01, high=3.):
    grid = np.linspace(low, high, 9)
    return dict(DEDX=dict(energy=grid, values=np.full(9, 2.)),
                Range=dict(energy=grid, values=grid/2))


@pytest.mark.parametrize('chunk', [1, 2, 3, 100])
def test_recorded_decrements_and_constant_chord_budget(chunk):
    p, rows, case = fixture()
    _, ref = fr.census(p, rows, case)
    result = sg.decrement_steps(Chunks(rows), ref, tables(), step_size=chunk)
    np.testing.assert_array_equal(result['pairs'].sum(axis=1), [2., 0.])
    np.testing.assert_allclose(result['chord_mm'].sum(axis=1), [.2, 0.])
    np.testing.assert_allclose(result['energy_decrement'].sum(axis=1), [1., 0.])
    np.testing.assert_allclose(result['left_mean_budget'].sum(axis=1), [.4, 0.])
    np.testing.assert_allclose(result['trapezoid_mean_budget'].sum(axis=1), [.4, 0.])
    np.testing.assert_allclose(result['native_range_decrement'].sum(axis=1), [.5, 0.])
    np.testing.assert_array_equal(result['large_fraction_pairs'].sum(axis=1), [1., 0.])
    np.testing.assert_allclose(result['pre_energy'].sum(axis=1), [3., 0.])


def test_decrement_census_and_domain_refused():
    p, rows, case = fixture()
    _, ref = fr.census(p, rows, case)
    with pytest.raises(ValueError, match='outside'):
        sg.decrement_steps(Chunks(rows), ref, tables(1.5), step_size=3)
    changed = dict(ref, exit=np.array([False, False]))
    with pytest.raises(ValueError, match='census'):
        sg.decrement_steps(Chunks(rows), changed, tables(), step_size=3)
    rows['evt'][0] = 3
    with pytest.raises(ValueError, match='population'):
        sg.decrement_steps(Chunks(rows), ref, tables(), step_size=3)


def test_second_entry_after_first_exit_not_counted():
    p, rows, case = fixture()
    _, ref = fr.census(p, rows, case)
    rows = {key:np.insert(value, 4, value[2]) for key, value in rows.items()}
    rows['pz'][4] = 1.
    result = sg.decrement_steps(Chunks(rows), ref, tables(), step_size=2)
    assert result['pairs'].sum() == 2
    assert result['energy_decrement'].sum() == pytest.approx(1.)


def test_clustered_decrement_summary_and_outcome_cohort():
    p, rows, case = fixture()
    _, ref = fr.census(p, rows, case)
    result = sg.decrement_steps(Chunks(rows), ref, tables(), step_size=2)
    repeated = {key:np.repeat(value, 3, axis=0) for key, value in result.items()}
    ref = {key:np.repeat(value, 3) for key, value in ref.items()}
    summary = sg.summarize_decrements(repeated, ref)
    total = summary['cohorts']['all_primaries']['all_pre_energies']
    selected = summary['cohorts']['forward_above_exits']['all_pre_energies']
    assert total['pair_count'] == 6 and total['primary_count'] == 6
    assert selected['primary_count'] == 3
    assert total['fractional_pre_post_change']['value'] == pytest.approx(1/3)
    assert total['total_decrement_over_left_restricted_chord_budget']['value'] == pytest.approx(2.5)
    assert total['reconstructed_range_decrement_over_chord']['value'] == pytest.approx(2.5)
    assert total['fraction_pairs_with_drop_over_10pct']['value'] == .5
    json.dumps(summary, allow_nan=False)


def test_forward_angle_selection_is_not_all_exit_or_all_primary():
    data = dict(exit=np.array([1, 1, 0, 1], bool), exit_z_mm=np.array([1., -1., 0., 1.]),
                exit_MeV=np.array([.2, .3, 0., .05]))
    directions = np.array([[.6, 0., .8], [0., 0., -1.], [np.nan]*3, [1., 0., 0.]])
    energy, angle = sg.forward_sample(data, directions, 1.)
    np.testing.assert_array_equal(energy, [.2])
    assert angle[0] == pytest.approx(np.arccos(.8))


def test_negative_drop_is_recorded_not_clipped():
    p, rows, case = fixture()
    rows['pz'][3] = rows['pz'][1]
    _, ref = fr.census(p, rows, case)
    result = sg.decrement_steps(Chunks(rows), ref, tables(), step_size=1)
    assert result['negative_drop_pairs'].sum() == 1
    assert result['energy_decrement'].sum() == pytest.approx(0., abs=1e-13)


def test_secondary_only_chunks_do_not_break_primary_adjacent_state():
    p, rows, case = fixture()
    _, ref = fr.census(p, rows, case)
    changed = {key:np.insert(value, 2, value[1]) for key, value in rows.items()}
    changed['trk'][2], changed['pdg'][2] = 2, 11
    result = sg.decrement_steps(Chunks(changed), ref, tables(), step_size=1)
    assert result['pairs'].sum() == 2
    assert result['energy_decrement'].sum() == pytest.approx(1.)


def pipeline(tmp_path, monkeypatch, *, changed_pairs=False):
    from pathlib import Path
    root, reused, prior = tmp_path/'factorial', tmp_path/'reused', tmp_path/'prior'
    root.mkdir()
    reused.mkdir()
    prior.mkdir()
    previous = dict(complete=True, references={})
    n = 5000
    energy = np.tile([0., .15, .25, .4, .8], 1000)
    ref = dict(event=np.arange(1, n+1), track=np.ones(n), exit_MeV=energy,
        exit=energy > 0, exit_z_mm=np.where(energy > 0, .1, 0.), chord_sum_mm=np.full(n, .2),
        hard_MeV=np.zeros(n), hard_count=np.zeros(n), foil_steps=np.full(n, 2))
    for cell in sg.edgeg4.CELLS:
        for mode in sg.ef.ORDER:
            tag = cell+'_'+mode
            directory = (reused if mode == 'full_refined' else root)/tag
            directory.mkdir()
            output = directory/'foil.root'
            output.write_text('fixture')
            receipt = dict(sources=[], tables=[], output=sg._identity(output))
            path = directory/'receipt.json'
            path.write_text(json.dumps(receipt))
            previous['references'][tag] = dict(receipt=sg._identity(path))
    (prior/'summary.json').write_text(json.dumps(previous))
    monkeypatch.setattr(sg.slabrefs, 'load_references', lambda *a: (dict.fromkeys(sg.ef.ORDER, ref), []))
    monkeypatch.setattr(sg.pt, 'load_target', lambda *a, **k: (tables(), {}, []))
    monkeypatch.setattr(sg.g4tables, '_extract', lambda *a, **k:
        dict(msc_energy=np.array([.1, 3.]), msc_lambda1=np.ones(2)))
    class File:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(sg.uproot, 'open', lambda *a, **k: File())
    monkeypatch.setattr(sg.fr, 'highest', lambda *a, **k: (None, dict(key='steps;3', entries=10000)))
    direction = np.tile([0., 0., 1.], (n, 1))
    angles = dict(event=ref['event'], track=ref['track'], direction=direction)
    monkeypatch.setattr(sg.edgeangles, 'first_exit', lambda *a: angles)
    features = dict(adjacent_chord_sum_mm=ref['chord_sum_mm'])
    monkeypatch.setattr(sg.es, 'step_features', lambda *a: features)
    pairs = np.zeros((n, len(sg.DECREMENT_EDGES)-1))
    monkeypatch.setattr(sg.em, 'reduce_steps', lambda *a: dict(pairs=pairs))
    monkeypatch.setattr(sg, 'decrement_steps', lambda *a, **k:
        dict(pairs=pairs+1 if changed_pairs else pairs))
    monkeypatch.setattr(sg.em, 'summarize', lambda *a: {})
    monkeypatch.setattr(sg, 'summarize_decrements', lambda *a: {})
    def summarize(data, angles, features, thickness, cosine):
        return dict(tails=sg.es.tails(energy, ref['exit'], ref['exit']),
                    exit_statistics=sg.es.exit_statistics(energy, ref['exit'], ref['exit'], direction, np.ones(n), cosine),
                    features={c:dict(metrics=dict(chord=sg.es.estimate(ref['chord_sum_mm'])))
                              for c in ('all', 'forward_above')})
    monkeypatch.setattr(sg.es, 'summarize_g4', summarize)
    report = dict(args=dict(cell='moved_edge'), arms=dict(diffuse=dict(output=sg._identity(prior/'summary.json'))))
    arrays = dict(status=np.where(ref['exit'], sg.sd.TRANSMITTED, sg.sd.UNRESOLVED_FLOOR),
                  raw_energy=energy, direction=direction, path=np.full(n, .2))
    monkeypatch.setattr(sg.le, 'read_study', lambda *a: (report, dict(diffuse=arrays), sg._identity(prior/'summary.json')))
    monkeypatch.setattr(sg.le, 'threshold_array', lambda *a: energy)
    return root, reused, prior


def test_complete_saved_pipeline_and_fast_comparisons(tmp_path, monkeypatch):
    root, reused, prior = pipeline(tmp_path, monkeypatch)
    out = tmp_path/'results'
    result = sg.run(root, reused, prior, ['completed_fast'], out)
    assert result['complete'] and not result['new_transport'] and not result['physics_closure']
    assert len(result['g4']) == 8 and len(result['outputs']) == 32
    assert set(result['pairs']['moved_edge']) == set(sg.ef.PAIRS)
    assert set(result['fast_comparisons']['completed_fast_diffuse']) == set(sg.ef.ORDER)
    assert result['pairs']['moved_edge']['step_at_bins7']['standardized_forward_angle']['available']
    assert json.loads((out/'summary.json').read_text())['complete']
    with pytest.raises(FileExistsError):
        sg.run(root, reused, prior, [], out)


def test_pipeline_refuses_different_angular_and_decrement_pairs(tmp_path, monkeypatch):
    root, reused, prior = pipeline(tmp_path, monkeypatch, changed_pairs=True)
    with pytest.raises(ValueError, match='retained-pair census'):
        sg.run(root, reused, prior, [], tmp_path/'results')


def test_pipeline_refuses_incomplete_prior_before_outputs(tmp_path, monkeypatch):
    root, reused, prior = pipeline(tmp_path, monkeypatch)
    (prior/'summary.json').write_text(json.dumps(dict(complete=False)))
    out = tmp_path/'results'
    with pytest.raises(ValueError, match='complete independent'):
        sg.run(root, reused, prior, [], out)
    assert not out.exists()


def test_range_averaged_inverse_lambda_constant_and_linear_analytic():
    grid = np.array([.1, 3.])
    constant = sg.range_averaged_inverse_lambda(tables(), grid, np.full(2, 2.))
    np.testing.assert_allclose(constant(np.array([2., 1.]), np.array([1., 1.])), [.5, .5])
    inverse_linear = sg.range_averaged_inverse_lambda(tables(), grid, 1/grid)
    value = inverse_linear(np.array([2., 1., .2]), np.array([1., 1., .15]))
    np.testing.assert_allclose(value, [1.5, 1., .175], rtol=1e-7)
    with pytest.raises(ValueError, match='domain'):
        constant(np.array([4.]), np.array([1.]))


def test_range_average_rejects_nonmonotone_range_and_bad_lambda():
    broken = tables()
    broken['Range']['values'] = broken['Range']['values'][::-1]
    with pytest.raises(ValueError, match='monotone'):
        sg.range_averaged_inverse_lambda(broken, [.1, 3.], [1., 1.])
    with pytest.raises(ValueError, match='ordered lambda'):
        sg.range_averaged_inverse_lambda(tables(), [3., .1], [1., 1.])


def test_decrement_angular_budget_is_same_recorded_pair_not_new_trajectory():
    p, rows, case = fixture()
    rows['px'][2], rows['pz'][2] = rows['pz'][2], 0.
    _, ref = fr.census(p, rows, case)
    result = sg.decrement_steps(Chunks(rows), ref, tables(), msc_energy=np.array([.1, 3.]),
                               lambda1=np.full(2, 2.), step_size=1)
    np.testing.assert_allclose(result['sum_one_minus_cos'].sum(axis=1), [2., 0.])
    for key in ('budget_left', 'budget_mid_energy', 'budget_range_average'):
        np.testing.assert_allclose(result[key].sum(axis=1), [.1, 0.])
    with pytest.raises(ValueError, match='both lambda'):
        sg.decrement_steps(Chunks(rows), ref, tables(), msc_energy=np.array([.1, 3.]), step_size=1)
