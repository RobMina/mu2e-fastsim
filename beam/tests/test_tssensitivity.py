"""Saved-output regressions using synthetic identities, exposures and outcomes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from beamfastsim import tssensitivity as ts


def tables():
    n = 8
    shard = np.array([4, 4, 4, 8, 8, 8, 12, 12])
    ids = np.column_stack((np.full(n, 27), shard, np.arange(n) + 100, np.arange(n) + 2**53 + 1))
    reached = np.array([True, True, False, True, True, False, False, False])
    p = np.tile([3., 4., 12.], (n, 1))
    r = dict(id=ids, shard=shard, x=np.zeros((n, 3)), p=p, t=np.arange(n, dtype=float),
             pdg=np.array([-13, 211, -13, 13, -211, 13, 211, 13]), stock_reached=reached,
             stock_x=np.where(reached[:, None], 1., np.nan) * np.ones((n, 3)),
             stock_p=np.where(reached[:, None], p, np.nan), stock_t=np.where(reached, np.arange(n) + 10., np.nan),
             stock_pdg=np.array([-13, -13, 0, 13, 13, 0, 0, 0]),
             event_id=np.vstack((ids[:, :3], [[27, 4, 200], [42, 4, 201], [42, 4, 202]])),
             parent_shard_ids=np.array([[27, 4], [27, 8], [27, 12], [42, 4], [42, 99]]),
             parent_shard_event_counts=np.array([4, 3, 2, 2, 0]))
    b = dict(id=ids.copy(), shard=shard.copy(), pdg=np.array([-13, -13, -13, 13, 13, 13, -13, 13]),
             x=r['x'] + 1, p=p.copy(), t=r['t'] + 20, fate=np.array([1, 4, 1, 1, 2, 2, 4, 2]),
             path=np.ones(n), material_path=np.ones(n), seconds=np.asarray(1.))
    b.update({key: r[key].copy() for key in ts.EXPOSURE})
    c = {key: value.copy() for key, value in b.items()}
    c['fate'] = np.array([4, 1, 1, 1, 1, 2, 4, 2])
    return r, b, c


def test_paired_population_differences_and_original_charge_denominators():
    r, b, c = tables()
    result = ts.summarize(r, b, c)
    assert result['ninput'] == 8
    assert result['parent_events'] == 11
    assert result['independent_shards'] == [[27, 4], [27, 8], [27, 12], [42, 4], [42, 99]]
    minus = result['charges']['-1']['muons']['mu-']
    assert (minus['baseline'], minus['candidate'], minus['both'], minus['candidate_only']) == (1, 2, 1, 1)
    assert minus['difference'] == .25
    assert minus['ratio'] == 2
    assert minus['difference_se'] == pytest.approx(np.sqrt(1.25 * (.25**2 + .25**2)) / 4)
    plus = result['charges']['1']['muons']['mu+']
    assert (plus['baseline'], plus['candidate'], plus['both'], plus['baseline_only'], plus['candidate_only']) == (2, 2, 1, 1, 1)
    assert result['charges']['1']['ninput_shard'] == [3, 0, 1, 0, 0]
    assert result['charges']['-1']['ninput_shard'] == [0, 3, 1, 0, 0]
    assert np.sum(result['charges']['1']['fate_transition']) == 4
    for charge, name in (('-1', 'mu-'), ('1', 'mu+')):
        m = result['charges'][charge]['muons'][name]
        for h in m['histograms'].values():
            for arm in ('baseline', 'candidate'):
                assert sum(h[arm]) + sum(h[arm + '_flow']) == m[arm]
                assert not np.any(np.asarray(h[arm + '_shard'])[2:])
    json.dumps(result, allow_nan=False)


def test_exact_join_permutation_and_flow_bins():
    r, b, c = tables()
    c['t'][3] = r['t'][3] - 1
    c['t'][4] = r['t'][4] + 1000
    c['p'][4] = [0, 0, 300]
    b['p'][3] = [0, 0, 200]
    b['t'][3] = r['t'][3] + 500
    reference = ts.summarize(r, b, c)
    for table in (b, c):
        for key, value in table.items():
            if key not in ts.EXPOSURE and value.ndim:
                table[key] = value[::-1]
        table['event_id'] = table['event_id'][::-1]
        table['parent_shard_ids'] = table['parent_shard_ids'][::-1]
        table['parent_shard_event_counts'] = table['parent_shard_event_counts'][::-1]
    assert ts.summarize(r, b, c) == reference
    minus = reference['charges']['-1']['muons']['mu-']
    assert minus['histograms']['momentum']['candidate_flow'] == [0, 1]
    assert minus['histograms']['elapsed_time']['candidate_flow'] == [1, 1]
    for h in minus['histograms'].values():
        assert h['baseline_flow'] == [0, 0]
        assert h['baseline'][-1] == 1


def test_species_fates_and_capped_ids():
    r, b, c = tables()
    c['pdg'][4] = -211
    c['pdg'][1] = 211
    c['fate'][7] = 6
    result = ts.summarize(r, b, c)
    negative = result['charges']['-1']
    assert negative['ds']['candidate'] == 2
    assert negative['muons']['mu-']['candidate'] == 1
    assert negative['ds_species']['candidate']['-211'] == 1
    assert result['charges']['1']['ds_species']['candidate']['211'] == 1
    assert negative['fates']['candidate']['path_limit'] == 1
    assert negative['capped_ids']['candidate'] == [r['id'][7].tolist()]
    assert negative['fate_transition'][2][6] == 1


@pytest.mark.parametrize('arm', [1, 2])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint64])
def test_unsigned_fates_match_signed_summaries(arm, dtype):
    inputs = tables()
    expected = ts.summarize(*inputs)
    inputs[arm]['fate'] = inputs[arm]['fate'].astype(dtype)
    assert ts.summarize(*inputs) == expected


def test_identical_outputs_have_zero_difference():
    r, b, _ = tables()
    result = ts.summarize(r, b, b)
    for charge in ('-1', '1'):
        for m in result['charges'][charge]['muons'].values():
            assert m['difference'] == m['difference_se'] == 0
            for h in m['histograms'].values():
                assert h['chi2_diagonal'] == 0


def test_zero_se_nonzero_difference_is_undefined():
    r, b, c = tables()
    b['fate'][:] = 4
    c['fate'][:] = 1
    result = ts.summarize(r, b, c)
    for charge, name in (('-1', 'mu-'), ('1', 'mu+')):
        m = result['charges'][charge]['muons'][name]
        assert m['difference'] == 1
        assert m['difference_se'] == 0
        assert m['pull'] is None
        assert m['zero_variance_nonzero_difference']
        assert m['ratio'] is None
        for h in m['histograms'].values():
            assert h['chi2_diagonal'] is None
            assert h['chi2_bins'] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('arm', [0, 1, 2])
@pytest.mark.parametrize('key', ts.EXPOSURE)
def test_missing_exposure_rejected(arm, key):
    inputs = tables()
    del inputs[arm][key]
    with pytest.raises(ValueError, match='missing exposure'):
        ts.summarize(*inputs)


@pytest.mark.parametrize('arm', [0, 1, 2])
@pytest.mark.parametrize('change', ['counts', 'duplicate_event', 'duplicate_parent', 'missing_source_event', 'unknown_parent', 'float_events'])
def test_invalid_exposure_rejected(arm, change):
    inputs = tables()
    table = inputs[arm]
    if change == 'counts':
        table['parent_shard_event_counts'][0] += 1
    elif change == 'duplicate_event':
        table['event_id'][-1] = table['event_id'][-2]
    elif change == 'duplicate_parent':
        table['parent_shard_ids'][-1] = table['parent_shard_ids'][-2]
    elif change == 'missing_source_event':
        table['event_id'][0, 2] += 1000
    elif change == 'unknown_parent':
        table['event_id'][-1, 1] += 1000
    else:
        table['event_id'] = table['event_id'].astype(float)
    with pytest.raises(ValueError):
        ts.summarize(*inputs)


@pytest.mark.parametrize('arm', [1, 2])
@pytest.mark.parametrize('key', ['event_id', 'parent_shard_ids'])
def test_different_valid_exposure_rejected(arm, key):
    inputs = tables()
    inputs[arm][key][-1, -1] += 1000
    with pytest.raises(ValueError, match='exposure differs'):
        ts.summarize(*inputs)


@pytest.mark.parametrize('arm', [1, 2])
@pytest.mark.parametrize('change', ['missing_row', 'duplicate', 'track', 'shard'])
def test_exact_identity_rejected(arm, change):
    inputs = tables()
    table = inputs[arm]
    if change == 'missing_row':
        for key, value in table.items():
            if key not in ts.EXPOSURE and value.ndim:
                table[key] = value[:-1]
    elif change == 'duplicate':
        table['id'][0] = table['id'][1]
    elif change == 'track':
        table['id'][0, 3] += 1
    else:
        table['shard'][0] += 1
    with pytest.raises(ValueError):
        ts.summarize(*inputs)


def test_complete_reference_supplies_exposure_for_metadata_free_outputs():
    r, b, c = tables()
    expected = ts.summarize(r, b, c)
    for table in (b, c):
        for key in ts.EXPOSURE:
            del table[key]
    assert ts.summarize(r, b, c) == expected
    c['event_id'] = r['event_id']
    with pytest.raises(ValueError, match='missing exposure'):
        ts.summarize(r, b, c)


def test_single_shard_errors_are_undefined():
    inputs = tables()
    for table in inputs:
        table['id'][:, :2] = [71, 6]
        table['shard'][:] = 6
        table['event_id'][:, :2] = [71, 6]
        table['parent_shard_ids'] = np.array([[71, 6]])
        table['parent_shard_event_counts'] = np.array([len(table['event_id'])])
    result = ts.summarize(*inputs)
    assert result['independent_shards'] == [[71, 6]]
    assert result['charges']['-1']['muons']['mu-']['difference_se'] is None
    assert result['charges']['-1']['muons']['mu-']['ratio_se'] is None
    json.dumps(result, allow_nan=False)


def test_empty_charge_plot(tmp_path):
    r, b, c = tables()
    r['pdg'][:] = 13
    result = ts.summarize(r, b, c)
    assert result['charges']['1']['ds']['difference'] is None
    ts.plot_settings(result, tmp_path / 'settings.png')
    assert (tmp_path / 'settings.png').read_bytes().startswith(b'\x89PNG')


@pytest.mark.slow   # subprocess re-run
def test_saved_only_cli_no_sidecars_or_transport(tmp_path):
    paths = [tmp_path / (name + '.npz') for name in ('reference', 'baseline', 'candidate')]
    for path, table in zip(paths, tables()):
        np.savez(path, **table)
    before = [path.read_bytes() for path in paths]
    out = tmp_path / 'out'
    args = [value for name, path in zip(('reference', 'baseline', 'candidate', 'out'), [*paths, out])
            for value in ('--' + name, str(path))]
    script = '''
import importlib.abc
import runpy
import sys
class BlockTransport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.', 1)[-1] in ('tsfield', 'tsgeometry', 'tstransport', 'tsstudy'):
            raise AssertionError('saved-only CLI imported ' + fullname)
sys.meta_path.insert(0, BlockTransport())
runpy.run_module('beamfastsim.tssensitivity', run_name='__main__')
'''
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / 'matplotlib'),
               PYTHONPATH=str(Path(ts.__file__).resolve().parents[1]) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    command = [sys.executable, '-c', script, *args]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)['parent_events'] == 11
    assert {p.name for p in out.iterdir()} == {'summary.json', 'settings.png'}
    assert (out / 'settings.png').read_bytes().startswith(b'\x89PNG')
    result = json.loads((out / 'summary.json').read_text())
    assert result == ts.summarize(*paths)
    assert not any(word in json.dumps(result) for word in ('warnings', 'provenance', 'seed', 'settings', str(tmp_path)))
    summary = (out / 'summary.json').read_bytes()
    assert subprocess.run(command, env=env, capture_output=True, timeout=60).returncode != 0
    assert (out / 'summary.json').read_bytes() == summary
    assert [path.read_bytes() for path in paths] == before
    assert not list(tmp_path.glob('*.log')) and not list(tmp_path.rglob('run.json'))


@pytest.mark.parametrize('extra', [['--gdml', 'geometry'], ['--field-config', 'field'], ['--offline-root', 'offline'], ['--maps', 'maps']])
def test_cli_rejects_field_inputs(tmp_path, extra):
    out = tmp_path / 'out'
    with pytest.raises(SystemExit):
        ts.main(['--reference', 'reference.npz', '--baseline', 'baseline.npz', '--candidate', 'candidate.npz',
                 '--out', str(out), *extra])
    assert not out.exists()


def test_cli_requires_candidate_and_validates_before_writing(tmp_path):
    out = tmp_path / 'out'
    args = ['--reference', str(tmp_path / 'reference.npz'), '--baseline', str(tmp_path / 'baseline.npz'), '--out', str(out)]
    with pytest.raises(SystemExit):
        ts.main(args)
    r, b, c = tables()
    del c['event_id']
    for name, table in zip(('reference', 'baseline', 'candidate'), (r, b, c)):
        np.savez(tmp_path / (name + '.npz'), **table)
    with pytest.raises(ValueError, match='missing exposure'):
        ts.main([*args, '--candidate', str(tmp_path / 'candidate.npz')])
    assert not out.exists()
