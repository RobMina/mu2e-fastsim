"""Saved-output continuation: no simulation, scoring, census or asset reads."""
import copy
import json

import pytest

from crvfastsim.kit_task import TERMS
from crvfastsim.objective import validation as V


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def result(seed=1, knobs=None):
    return dict(events_with_steps=8, events_without_steps=2, target=8, seed=seed,
                passes=[dict(events=8, wall_s=1)], timers={}, knobs=knobs or {},
                response_state=dict(events=10, exposure_ns=10000, h={'example': [1, 2]}))


@pytest.fixture
def replicas(tmp_path, monkeypatch):
    # Postprocessing must never recompute inputs, score, or execute response.
    def forbidden(*args, **kwargs):
        raise AssertionError('postprocessing attempted execution')
    for name in ('run', 'verify', 'inspect_inputs', 'revision', 'runtime_assets'):
        monkeypatch.setattr(V, name, forbidden)
    monkeypatch.setattr(V.R, 'objective', forbidden)
    for block, seed in [('a', 17), ('b', 23)]:
        manifest = dict(schema=1, events_with_steps=8, events=10,
                        inputs=[dict(path=f'/unavailable/{block}.root', sha256=block,
                                     events_with_steps=8, events=10)], excluded=[],
                        assets={k: dict(path=f'/unavailable/{k}', sha256=k)
                                for k in ('calib', 'deviation', 'reference', 'background', 'time_shift')})
        for point in ('nominal', 'best'):
            knobs = {} if point == 'nominal' else {'photon_yield_scale': 1.1}
            provenance = dict(manifest=manifest, manifest_sha256=block, seed=seed, knobs=knobs,
                              calibration=True, runtime_assets={'lut': {'sha256': 'lut'}},
                              code={'crvfastsim': {'sha': 'crv'}, 'mu2efastsim': {'sha': 'core'}})
            score = dict(objective=2 if point == 'nominal' else 1, n_terms=32, excluded=[],
                         required_terms=list(TERMS), sigma=0.5, sigma_ref=0.6,
                         terms={k: dict(chi2ndf=2 if point == 'nominal' else 1, sigma=0.5)
                                for k in TERMS})
            root = tmp_path / f'{block}_{point}'
            for filename, data in [('r3', result(seed, knobs)), ('provenance', provenance),
                                   ('comparison', score)]:
                save(root / f'{filename}.json', data)
    return tmp_path


def mutate(root, point, filename, change):
    path = root / point / f'{filename}.json'
    data = json.loads(path.read_text())
    change(data)
    save(path, data)


def test_paired_cli(replicas, monkeypatch, capsys):
    out = replicas / 'report.json'
    monkeypatch.setattr('sys.argv', ['validation', 'paired', '--directory', str(replicas), '--out', str(out)])
    V.main()
    report = json.loads(out.read_text())
    assert report['passed']
    assert 'PASS' in capsys.readouterr().out
    assert 'not a significance' in report['caution']
    for block in report['blocks'].values():
        assert block['objective']['best_minus_nominal'] == -1
        assert set(block['terms']) == set(TERMS)
        assert all(term['best_minus_nominal'] == -1 for term in block['terms'].values())
        assert 'sigma' not in block['objective']
    with pytest.raises(FileExistsError):
        V.paired(replicas, out)


@pytest.mark.parametrize('filename,change', [
    ('r3', lambda d: d.update(events_with_steps=7)),
    ('r3', lambda d: d.update(target=9)),
    ('r3', lambda d: d['response_state'].update(events=11)),
    ('r3', lambda d: d.update(events_without_steps=3)),
    ('r3', lambda d: d['response_state'].update(exposure_ns=11000)),
    ('r3', lambda d: d['response_state'].update(exposure_ns=float('nan'))),
    ('r3', lambda d: d.update(resolved_config={'timing': {'window': [0, 2000]}})),
    ('r3', lambda d: d.update(passes=[{'events': 4}, {'events': 4}])),
    ('r3', lambda d: d['passes'][0].update(events=7)),
    ('r3', lambda d: d['timers'].update(response_overflow_batches=1)),
    ('r3', lambda d: d.update(seed=9)),
    ('r3', lambda d: d.update(knobs={})),
    ('provenance', lambda d: d['manifest'].update(events=11)),
    ('provenance', lambda d: d['manifest']['inputs'][0].update(path='/different.root')),
    ('provenance', lambda d: d['manifest']['inputs'][0].update(sha256='changed')),
    ('provenance', lambda d: d.update(manifest_sha256='changed')),
    ('provenance', lambda d: d['manifest']['assets']['reference'].update(sha256='changed')),
    ('provenance', lambda d: d['runtime_assets'].update(lut={'sha256': 'changed'})),
    ('provenance', lambda d: d['code']['crvfastsim'].update(sha='changed')),
    ('provenance', lambda d: d.update(calibration=False)),
    ('comparison', lambda d: d.update(objective=float('nan'))),
    ('comparison', lambda d: d.update(n_terms=31)),
    ('comparison', lambda d: d.update(excluded=[TERMS[0]])),
    ('comparison', lambda d: d['terms'].pop(TERMS[0])),
    ('comparison', lambda d: d['terms'].update(unexpected=d['terms'].pop(TERMS[0]))),
    ('comparison', lambda d: d['terms'][TERMS[0]].update(chi2ndf=float('inf'))),
    ('comparison', lambda d: d['terms'][TERMS[0]].update(sigma=float('nan'))),
])
def test_invalid_point(replicas, filename, change):
    mutate(replicas, 'a_best', filename, change)
    out = replicas / 'invalid.json'
    with pytest.raises(ValueError):
        V.paired(replicas, out)
    assert not out.exists()


@pytest.mark.parametrize('mode', ['missing_point', 'missing_file', 'same_seed', 'pair_seed', 'overlap'])
def test_invalid_blocks(replicas, mode):
    if mode.startswith('missing'):
        files = list((replicas / 'b_best').glob('*.json'))
        for path in files if mode == 'missing_point' else files[:1]:
            path.unlink()
        with pytest.raises(FileNotFoundError):
            V.paired(replicas, replicas / 'out.json')
        return
    for point in (['b_best'] if mode == 'pair_seed' else ['b_nominal', 'b_best']):
        if mode in ('same_seed', 'pair_seed'):
            for filename in ('r3', 'provenance'):
                mutate(replicas, point, filename, lambda d: d.update(seed=17))
        else:
            mutate(replicas, point, 'provenance',
                   lambda d: d['manifest']['inputs'][0].update(path='/other/location/a.root'))
    with pytest.raises(ValueError, match={'same_seed': 'distinct seeds', 'pair_seed': 'seed mismatch',
                                        'overlap': 'disjoint'}[mode]):
        V.paired(replicas, replicas / 'out.json')


def test_replay_cli_ignores_timing_and_device(tmp_path, monkeypatch, capsys):
    ref = result()
    cand = copy.deepcopy(ref)
    cand.update(device='different device', wall_s=200, jax=True)
    cand['passes'][0]['wall_s'] = 100
    cand['timers']['response_s'] = 100
    r, c = save(tmp_path / 'ref.json', ref), save(tmp_path / 'cand.json', cand)
    out = tmp_path / 'replay.json'
    monkeypatch.setattr('sys.argv', ['validation', 'compare-replay', '--reference', str(r),
                                    '--candidate', str(c), '--out', str(out)])
    V.main()
    assert json.loads(out.read_text())['response_state_equal']
    assert 'PASS' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        V.compare_replay(r, c, out)


@pytest.mark.parametrize('change', [
    lambda d: d.update(events_with_steps=7),
    lambda d: d.update(events_with_steps=7, target=7, events_without_steps=3, passes=[{'events': 7}]),
    lambda d: d.update(target=9),
    lambda d: d.update(seed=2),
    lambda d: d.update(knobs={'pe_stretch': 1.1}),
    lambda d: d.update(passes=[{'events': 4}, {'events': 4}]),
    lambda d: d['passes'][0].update(events=7),
    lambda d: d['timers'].update(response_overflow_batches=1),
    lambda d: d['response_state']['h'].update(example=[1, 2.0000000001]),
    lambda d: d['response_state'].update(extra=1),
])
def test_replay_mismatch_cli_nonzero(tmp_path, monkeypatch, change):
    ref, cand = result(), result()
    change(cand)
    r, c = save(tmp_path / 'ref.json', ref), save(tmp_path / 'cand.json', cand)
    out = tmp_path / 'replay.json'
    monkeypatch.setattr('sys.argv', ['validation', 'compare-replay', '--reference', str(r),
                                    '--candidate', str(c), '--out', str(out)])
    with pytest.raises(SystemExit) as exc:
        V.main()
    assert exc.value.code != 0
    assert not out.exists()


def test_paired_failure_cli_nonzero(replicas, monkeypatch):
    (replicas / 'a_best' / 'r3.json').unlink()
    out = replicas / 'failed.json'
    monkeypatch.setattr('sys.argv', ['validation', 'paired', '--directory', str(replicas), '--out', str(out)])
    with pytest.raises(SystemExit) as exc:
        V.main()
    assert exc.value.code != 0
    assert not out.exists()
