"""Fail-closed scoring and R3 response guards using synthetic inputs only."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from crvfastsim import chain, kit_task as KT
from crvfastsim.objective import crvresponse as R
from crvfastsim.reco import r3


def state():
    s = R.new_state()
    s.update(events=1000, exposure_ns=1e8)
    for k, h in s['h'].items():
        a = np.asarray(h)
        a.flat[:1 if k in KT.NO_TERM else 2] = 1000
        s['h'][k] = a.tolist()
    s['oot_n'] = dict.fromkeys(R.SLOTS, 1000)
    return s


def result(s):
    return dict(response_state=s, events_with_steps=900, events_without_steps=100, target=900, timers={})


def test_complete_contract_preserves_score():
    s = state()
    old, new = R.objective(s, s), KT.score_result(result(s), s)
    assert len(KT.TERMS) == new['n_terms'] == 32
    for k in ('objective', 'sigma', 'sigma_ref', 'objective_raw', 'objective_all', 'terms'):
        assert new[k] == old[k]
    assert new['required_terms'] == list(KT.TERMS)


@pytest.mark.parametrize('arm', ['candidate', 'reference'])
@pytest.mark.parametrize('failure', ['missing', 'empty', 'nan', 'negative', 'shape'])
def test_bad_histogram_cannot_change_score_denominator(arm, failure):
    s, ref = state(), state()
    target = s if arm == 'candidate' else ref
    k = 'pe_along_T1'
    a = np.asarray(target['h'][k])
    if failure == 'missing':
        del target['h'][k]
    else:
        if failure == 'empty':
            a[:] = 0
        elif failure == 'shape':
            a = a.ravel()
        else:
            a.flat[0] = np.nan if failure == 'nan' else -1
        target['h'][k] = a.tolist()
    with pytest.raises(ValueError, match='objective'):
        KT.score_result(result(s), ref)


def test_p3_exclusion_is_explicit_and_does_not_hide_other_terms():
    s, ref = state(), state()
    out = KT.score_result(result(s), ref, exclude=['dt_cluster_T2'])
    assert out['n_terms'] == 31
    assert out['excluded'] == ['dt_cluster_T2']
    assert np.isfinite(out['objective_all'])
    del s['h']['dt_cluster_T2']
    out = KT.score_result(result(s), ref, exclude=['dt_cluster_T2'])
    assert out['n_terms'] == 31 and np.isnan(out['objective_all'])
    del s['h']['tdiff_T2']
    with pytest.raises(ValueError, match='tdiff_T2'):
        KT.score_result(result(s), ref, exclude=['dt_cluster_T2'])


def test_unexpected_term_and_invalid_exclusion_refused():
    s = state()
    s['h']['nlayers_T1'][1] = 500
    with pytest.raises(ValueError, match='unexpected=.*nlayers_T1'):
        KT.score_result(result(s), s)
    for exclude in (KT.TERMS, ['typo']):
        with pytest.raises(ValueError):
            R.objective(state(), state(), required=KT.TERMS, exclude=exclude)


@pytest.mark.parametrize('field,value', [('exposure_ns', 0), ('exposure_ns', np.inf), ('oot_n', None)])
def test_invalid_rate_inputs_refused(field, value):
    s, ref = state(), state()
    if field == 'oot_n':
        s[field]['EX'] = -1
    else:
        s[field] = value
    with pytest.raises(ValueError, match='objective'):
        KT.score_result(result(s), ref)


def test_nonfinite_term_uncertainty_refused(monkeypatch):
    original = R.shape_term
    def bad(*args, **kwargs):
        term = original(*args, **kwargs)
        if term is not None:
            term['sigma'] = np.nan
        return term
    monkeypatch.setattr(R, 'shape_term', bad)
    with pytest.raises(ValueError, match='nonfinite'):
        KT.score_result(result(state()), state())


@pytest.mark.parametrize('change', [dict(events_with_steps=800), dict(events_without_steps=None),
                                     dict(timers={'response_overflow_batches': 1})])
def test_invalid_r3_result_refused_before_objective(monkeypatch, change):
    j = result(state())
    j.update(change)
    monkeypatch.setattr(R, 'objective', lambda *a, **kw: pytest.fail('scored an invalid result'))
    with pytest.raises(ValueError, match='crv_response'):
        KT.score_result(j, state())


@pytest.mark.parametrize('jax', [False, True])
@pytest.mark.parametrize('setting', ['thermal_rate', 'trap0_prob', 'trap1_prob'])
def test_unsupported_processes_refused_before_response(jax, setting):
    cfg = chain.ChainConfig(jax=jax)
    cfg = replace(cfg, sipm=replace(cfg.sipm, **{setting: 0.01}))
    with pytest.raises(ValueError, match=setting):
        r3.run_batch(None, 0, 1, None, cfg, None, None, None, None, None, None, {})


@pytest.mark.parametrize('window', [(0, 200000), (-100, 100000), (100, 100000)])
def test_unsupported_window_refused_before_response(window):
    with pytest.raises(ValueError, match='window'):
        r3.validate_response(replace(chain.ChainConfig(), window=window))


def test_exposure_window_mismatch_refused():
    with pytest.raises(ValueError, match='exposure'):
        r3.run_batch(None, 0, 1, None, chain.ChainConfig(), None, None, None, None, None, None, {}, window_ns=1)


@pytest.mark.parametrize('source', ['photons', 'sipm'])
def test_residual_gpu_overflow_fails_before_host_consumption(monkeypatch, source):
    jax = pytest.importorskip('jax')
    import pandas as pd
    from crvfastsim.response import device
    c = dict(ek=np.array([1]), events=np.array([1]), steps=pd.DataFrame({'x': [0]}), pdg=np.array([13]))
    monkeypatch.setattr(device, 'response_batched', lambda *a, **kw:
                        [dict(overflow=True, **{'overflow_' + source: True})])
    monkeypatch.setattr(device, 'to_host_batches', lambda *a: pytest.fail('consumed truncated response'))
    timers = {}
    cond = chain.Conditions(*(np.zeros(4) for _ in range(8)))
    with pytest.raises(RuntimeError, match='overflow'):
        r3.run_batch(c, 0, 1, cond, chain.ChainConfig(jax=True), None, None, None, None,
                     np.random.default_rng(1), jax.random.PRNGKey(1), timers)
    assert timers['response_overflow_batches'] == 1


@pytest.mark.parametrize('valid', [True, False])
def test_rendered_scoring_block_writes_metrics_only_for_valid_point(tmp_path, valid):
    import json
    import subprocess
    import sys
    s = state()
    j = result(s)
    j['wall_s'] = 1.0
    if not valid:
        del s['h']['pe_along_T1']
    (tmp_path / 'r3.json').write_text(json.dumps(j))
    ref = tmp_path / 'reference.json'
    ref.write_text(json.dumps(state()))
    code = KT.BODY_TAIL.split("<<'EOF'\n", 1)[1].rsplit('\nEOF', 1)[0]
    proc = subprocess.run([sys.executable, '-', str(tmp_path), '', str(ref)], input=code,
                          capture_output=True, text=True)
    assert (proc.returncode == 0) == valid, proc.stderr
    assert (tmp_path / 'metrics.json').exists() == valid
    if valid:
        ob = json.loads((tmp_path / 'objective.json').read_text())['objective']
        assert ob['required_terms'] == list(KT.TERMS)
    else:
        assert 'pe_along_T1' in proc.stderr


def test_diagnostic_empty_objective_remains_reportable(tmp_path):
    s = R.new_state()
    out = R.report(s, str(tmp_path), deepcopy(s))
    assert out['objective']['n_terms'] == 0
    assert np.isnan(out['objective']['objective'])
