"""Synthetic control diagnosis; no response runs, ROOT inputs, or objective scoring."""
import copy
import json
from pathlib import Path

import pytest

from crvfastsim.kit_task import TERMS
from crvfastsim.objective import control_diagnosis as C
from crvfastsim.objective import crvresponse as R
from crvfastsim.objective import diagnose as D
from crvfastsim.objective import validation as V


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return path


def mutate(path, change):
    value = json.loads(path.read_text())
    change(value)
    save(path, value)


def state():
    out = R.new_state()
    out.update(events=10, files=1, exposure_ns=10000, lines_z=8, lines_x=7, lines_xz=6)
    for name, axes in R.SHAPES.items():
        for slot in R.SLOTS_OF.get(name, R.SLOTS):
            if len(axes) == 1:
                out['h'][f'{name}_{slot}'][1] = 200
    for slot in R.SLOTS:
        out['h'][f'pe_along_{slot}'][0][0][2] = 200
        out['h'][f'pe_path_{slot}'][0][2] = 200
    for body in range(4, 16):
        out['dtp'][0][body][80] = 200
    return out


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('attempted original input access, simulation, calibration or objective scoring')

    for name in ('run', 'verify', 'inspect_inputs', 'keys', 'revision', 'runtime_assets'):
        monkeypatch.setattr(V, name, forbidden)
    monkeypatch.setattr(R, 'objective', forbidden)
    monkeypatch.setattr(D, 'code_identity', lambda: dict(head='synthetic', dirty=False))
    reference = save(tmp_path / 'reference.json', state())
    calibration = save(tmp_path / 'timecal.json', {'data_files': ['part.json']})
    assets = {k: dict(path=f'/unavailable/{k}', sha256=k)
              for k in ('calib', 'deviation', 'background')}
    assets.update(reference=dict(path=str(reference), sha256=V.digest(reference)),
                  time_shift=dict(path=str(calibration), sha256=V.digest(calibration)))
    replicas = tmp_path / 'replicas'
    roots = {b: tmp_path / f'controls_{b}' for b in ('a', 'b')}
    for block, seed in [('a', 17), ('b', 23)]:
        manifest = dict(schema=1, events=10, events_with_steps=8, assets=assets, excluded=[],
                        inputs=[dict(path=f'/unavailable/{block}.root', sha256=block,
                                     events=10, events_with_steps=8)])
        provenance = dict(manifest=manifest, manifest_sha256=block, seed=seed, knobs={},
                          code={'crvfastsim': {'sha': 'crv'}, 'mu2efastsim': {'sha': 'core'}},
                          runtime_assets={'lut': {'path': '/unavailable/lut', 'sha256': 'lut'}},
                          python='synthetic', versions={'numpy': 'synthetic'}, jax=False,
                          calibration=True, future_field={'must': 'match too'})
        result = dict(events_with_steps=8, events_without_steps=2, target=8, seed=seed, knobs={},
                      passes=[dict(events=8, wall_s=1)], timers={}, response_state=state(),
                      cache_files=[f'/unavailable/{block}.root'], cache_events=8,
                      jax=False, noise=True, v05=True, batch_events=100,
                      background='/unavailable/background', time_shift=str(calibration),
                      resolved_config={'timing': {'window': [0, 1000]}})
        score = dict(objective=2, n_terms=32, excluded=[], required_terms=list(TERMS), n0=R.N0,
                     sigma=0.5, sigma_ref=0.6,
                     terms={k: dict(chi2ndf=2, sigma=0.5) for k in TERMS})
        for arm, knobs in [('nominal', {}), ('best', {'atten_T1': 0.4, 'speed_EX': 0.98,
                                                    'photon_yield_scale': 1.1, 'speed_T2': 1.02,
                                                    'atten_T2': 1.1})]:
            p, r, s = copy.deepcopy((provenance, result, score))
            p['knobs'] = r['knobs'] = knobs
            if arm == 'best':
                s['objective'] = 1
                for term in s['terms'].values():
                    term['chi2ndf'] = 1
            for key, value in [('provenance', p), ('r3', r), ('comparison', s)]:
                save(replicas / f'{block}_{arm}' / f'{key}.json', value)
        roots[block].mkdir()
        (roots[block] / 'complete.txt').write_text('complete\n')
        for suffix, knobs in C.CONTROLS.items():
            p, r, s = copy.deepcopy((provenance, result, score))
            p['knobs'] = r['knobs'] = knobs
            s['objective'] = 1.75
            for term in s['terms'].values():
                term['chi2ndf'] = 1.75
            for slot in ('T1', 'T2', 'TAG'):
                r['response_state']['h'][f'dt_cluster_{slot}'][1:3] = [0, 200]
            for body in range(4, 16):
                r['response_state']['dtp'][0][body][80:82] = [0, 200]
            for key, value in [('provenance', p), ('r3', r), ('comparison', s)]:
                save(roots[block] / f'{block}_{suffix}' / f'{key}.json', value)
    baseline_dir = tmp_path / 'baseline'
    D.diagnose(replicas, baseline_dir, plots=False)
    # Control diagnosis must use the stored paths, not rediscover/revalidate ROOT inputs.
    monkeypatch.setattr(V, 'paired', forbidden)
    return dict(baseline_report=baseline_dir / 'diagnosis.json', controls_a=roots['a'],
                controls_b=roots['b'], out=tmp_path / 'output')


def test_control_plot_outputs_and_no_overwrite(frozen, tmp_path):
    from crvfastsim.objective.diagnose_plots import plot_controls
    report = C.diagnose(**frozen)
    dest = tmp_path / 'plots'
    plot_controls(report, dest)
    assert len(list(dest.glob('*.png'))) == 4
    assert all(p.stat().st_size > 1000 for p in dest.glob('*.png'))
    with pytest.raises(FileExistsError):
        plot_controls(report, dest)


def test_matching_controls_full_best_labels_saved_deltas_and_immutability(frozen, tmp_path, capsys):
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report = C.diagnose(**frozen)
    assert report['kind'] == 'saved-control-response-diagnosis'
    assert report['min_entries'] == 100
    assert report['diagnostic_code']['source_sha256']['crvfastsim/objective/control_diagnosis.py'] == V.digest(C.__file__)
    for block in ('a', 'b'):
        original = report['blocks'][block]['full_best']
        assert original['saved_score_deltas']['objective']['candidate_minus_nominal'] == -1
        assert original['sample_role']['best'] == f'{block}_best: full-best (not a control)'
        for name, summary in report['blocks'][block]['controls'].items():
            assert summary['sample_role']['best'] == f'{block}_{name}: single-knob control (not full-best)'
            score = summary['saved_score_deltas']
            assert score['objective'] == dict(nominal=2, candidate=1.75, candidate_minus_nominal=-0.25)
            assert set(score['terms']) == set(TERMS)
            assert all(t['candidate_minus_nominal'] == -0.25 for t in score['terms'].values())
            assert summary['sectors']['T1']['pe_along']['common_cells'] == 1
            timing = summary['sectors']['T1']['module_timing']
            assert {c['module'] for c in timing['cells']} == {'T10', 'T11'}
            assert {c['ex_module'] for c in timing['cells']} == {'EX0', 'EX1', 'EX2', 'EX3'}
            for shifts in summary['timing_mean_shifts_ns'].values():
                assert shifts['dt_cluster'] == pytest.approx(1)
                assert shifts['module_timing'] == pytest.approx(
                    dict(marginal=0.5, support_marginal=0.5, standardized=0.5))
    assert {p: p.read_bytes() for p in before} == before
    receipts = {Path(r['path']): r['sha256'] for r in report['input_receipts']['all_files']}
    assert all(receipts[p] == V.digest(p) for p in before if p.name != 'timecal.json')
    assert json.loads((frozen['out'] / 'control_diagnosis.json').read_text()) == report
    assert [p.name for p in frozen['out'].iterdir()] == ['control_diagnosis.json']
    C.print_summary(report)
    printed = capsys.readouterr().out
    for text in ('single-knob control (not full-best)', 'T1 pe_layer', 'T1 tspread', 'T1 dt_cluster',
                 'T2 mean shift ns', 'TAG mean shift ns', 'not a linear decomposition'):
        assert text in printed


@pytest.mark.parametrize('change', [
    lambda p: p.update(python='different'),
    lambda p: p['versions'].update(numpy='different'),
    lambda p: p['code']['crvfastsim'].update(sha='different'),
    lambda p: p['runtime_assets']['lut'].update(sha256='different'),
    lambda p: p.update(manifest_sha256='different'),
    lambda p: p['manifest']['inputs'][0].update(sha256='different'),
    lambda p: p['manifest']['assets']['time_shift'].update(sha256='different'),
    lambda p: p.update(future_field={'must': 'not match'}),
    lambda p: p.pop('future_field'),
    lambda p: p.update(extra='not allowed'),
])
def test_provenance_must_match_every_field_except_knobs(frozen, change):
    mutate(frozen['controls_a'] / 'a_t1atten/provenance.json', change)
    with pytest.raises(ValueError, match='provenance differs'):
        C.diagnose(**frozen)
    assert not frozen['out'].exists()


@pytest.mark.parametrize('knobs', [{}, {'atten_T1': 0.4367}, {'speed_EX': 0.9840676938698677},
                                  {'atten_T1': 0.4367132104009329, 'speed_EX': 0.9840676938698677},
                                  {'atten_T1': True}])
def test_wrong_control_knobs(frozen, knobs):
    for key in ('r3', 'provenance'):
        mutate(frozen['controls_a'] / f'a_t1atten/{key}.json', lambda p: p.update(knobs=knobs))
    with pytest.raises(ValueError, match='wrong single-knob'):
        C.diagnose(**frozen)
    assert not frozen['out'].exists()


@pytest.mark.parametrize('key,value', [('noise', False), ('jax', True), ('v05', False),
                                      ('batch_events', 101), ('background', None),
                                      ('time_shift', 'new'), ('device', 'unexpected')])
def test_fixed_result_flags_match_nominal(frozen, key, value):
    mutate(frozen['controls_b'] / 'b_exspeed/r3.json', lambda r: r.update({key: value}))
    with pytest.raises(ValueError, match='fixed result flag'):
        C.diagnose(**frozen)


@pytest.mark.parametrize('change', [
    lambda r: r.update(target=9),
    lambda r: r.update(events_with_steps=7),
    lambda r: r['response_state'].update(events=11),
    lambda r: r.update(passes=[{'events': 4}, {'events': 4}]),
    lambda r: r['timers'].update(response_overflow_batches=1),
    lambda r: r['response_state'].update(exposure_ns=11000),
    lambda r: r['response_state']['h']['pe_layer_T1'].__setitem__(0, -1),
])
def test_point_validation_and_histogram_validation_are_not_bypassed(frozen, change):
    mutate(frozen['controls_a'] / 'a_t1atten/r3.json', change)
    with pytest.raises(ValueError):
        C.diagnose(**frozen)
    assert not frozen['out'].exists()


@pytest.mark.parametrize('change', [lambda s: s.update(n_terms=31),
                                   lambda s: s['terms'].pop(TERMS[0]),
                                   lambda s: s.update(n0=9999)])
def test_saved_scores_must_be_complete_and_fixed_normalization(frozen, change):
    mutate(frozen['controls_b'] / 'b_exspeed/comparison.json', change)
    with pytest.raises(ValueError):
        C.diagnose(**frozen)


@pytest.mark.parametrize('content', ['running', 'incomplete', '', 'complete extra', None])
def test_complete_marker_required(frozen, content):
    marker = frozen['controls_b'] / 'complete.txt'
    if content is None:
        marker.unlink()
    else:
        marker.write_text(content)
    with pytest.raises((ValueError, FileNotFoundError)):
        C.diagnose(**frozen)
    assert not frozen['out'].exists()


@pytest.mark.parametrize('target', ['reference.json', 'replicas/a_nominal/r3.json',
                                    'replicas/b_best/comparison.json', 'replicas/a_best/provenance.json'])
def test_baseline_and_reference_hashes_are_checked(frozen, tmp_path, target):
    path = tmp_path / target
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        C.diagnose(**frozen)
    assert not frozen['out'].exists()


@pytest.mark.parametrize('target', ['controls_a/a_t1atten/r3.json', 'controls_b/b_exspeed/provenance.json',
                                    'controls_a/a_exspeed/comparison.json', 'controls_b/complete.txt',
                                    'replicas/b_best/r3.json', 'reference.json', 'baseline/diagnosis.json'])
def test_detects_inputs_changed_during_summaries(frozen, tmp_path, monkeypatch, target):
    summarize = D.summarize_block
    path = tmp_path / target

    def changed(*args, **kwargs):
        result = summarize(*args, **kwargs)
        path.write_text(path.read_text() + '\n')
        return result

    monkeypatch.setattr(D, 'summarize_block', changed)
    with pytest.raises(ValueError, match='hash mismatch|changed during diagnosis'):
        C.diagnose(**frozen)
    assert not frozen['out'].exists()


@pytest.mark.parametrize('kind', ['file', 'directory', 'dangling_symlink'])
def test_output_never_overwrites_and_rejects_before_inputs(tmp_path, kind):
    out = tmp_path / 'out'
    if kind == 'directory':
        out.mkdir()
        sentinel = out / 'control_diagnosis.json'
    elif kind == 'dangling_symlink':
        out.symlink_to(tmp_path / 'missing')
        sentinel = None
    else:
        sentinel = out
    if sentinel:
        sentinel.write_text('keep me')
    with pytest.raises(FileExistsError):
        C.diagnose('unreadable', 'unreadable', 'unreadable', out)
    if sentinel:
        assert sentinel.read_text() == 'keep me'


@pytest.mark.parametrize('minimum', [0, -1, 1.5, True])
def test_invalid_support_threshold(tmp_path, minimum):
    with pytest.raises(ValueError, match='positive integer'):
        C.diagnose('unreadable', 'unreadable', 'unreadable', tmp_path / 'out', minimum)


def add_supplement(frozen, tmp_path):
    part = save(tmp_path / 'timing/part.json', state())
    # Existing reference timing is identical: supplement still verifies all scoring fields.
    baseline = json.loads(frozen['baseline_report'].read_text())
    _, receipt = D.supplement_timing(state(), tmp_path / 'timecal.json', part.parent)
    baseline['timing_supplement'] = receipt
    save(frozen['baseline_report'], baseline)
    return part


def test_supplement_reuses_only_pinned_files_and_calibration(frozen, tmp_path):
    part = add_supplement(frozen, tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report = C.diagnose(**frozen)
    assert report['timing_supplement']['files'][0]['sha256'] == V.digest(part)
    assert report['timing_supplement']['matched_scoring_fields'] == [
        'events', 'files', 'exposure_ns', 'lines_z', 'lines_x', 'lines_xz', 'h', 'oot_n']
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('target', ['timing/part.json', 'timecal.json'])
def test_supplement_hashes_checked_before_use(frozen, tmp_path, target):
    add_supplement(frozen, tmp_path)
    path = tmp_path / target
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        C.diagnose(**frozen)
    assert not frozen['out'].exists()


def test_supplement_no_unreceipted_file_reads(frozen, tmp_path):
    add_supplement(frozen, tmp_path)
    mutate(frozen['baseline_report'], lambda b: b['timing_supplement']['files'][0].update(
        path=str(tmp_path / 'timing/unreceipted.json')))
    with pytest.raises(ValueError, match='file list differs'):
        C.diagnose(**frozen)


def test_cli(frozen, monkeypatch, capsys):
    argv = ['control_diagnosis']
    for key, value in frozen.items():
        argv.extend(['--' + key.replace('_', '-'), str(value)])
    monkeypatch.setattr('sys.argv', argv + ['--min-support', '101'])
    C.main()
    report = json.loads((frozen['out'] / 'control_diagnosis.json').read_text())
    assert report['min_entries'] == 101
    assert 'wrote ' in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        C.main()
    assert exc.value.code == 2
