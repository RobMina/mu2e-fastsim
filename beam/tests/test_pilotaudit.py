"""Local saved-pilot integrity and source-event paired diagnostics; no transport."""
import json

import numpy as np
import pytest

from beamfastsim import convergence, g4ref, pilotaudit
from beamfastsim.reference_audit import _identity


def _json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def pilot(tmp_path):
    root = tmp_path / 'pilot'
    root.mkdir()
    original = tmp_path / 'g4ref.npz'
    g = dict(run=np.ones(10), subrun=np.zeros(10), evt=np.repeat(np.arange(10, 15), 2),
             trk=np.tile([1, 2], 5), q=-np.ones(10), time=np.zeros(10),
             foil=np.array([-1, -1, -1, -1, -1, -1, 0, 0, -1, -1]),
             extra=np.arange(10) + 0.5)
    g4ref.save(g, original)
    selected, block = convergence.select_block(g, 2, 6)
    g4ref.save(selected, root / 'input_block.npz')
    _json(root / 'manifest.json', dict(input=_identity(original),
          selected_input=_identity(root / 'input_block.npz'), block=block, n=6,
          seeds=[17], arms=['nominal', 'halfstep']))
    baseline = dict(x=np.zeros((6, 3)), p=np.array([0., 1., 10., np.nan, 20., 20.]),
                    t=np.arange(6, dtype=float), fate=np.array([1, 1, 1, 1, 6, 9]),
                    foil=np.array([0, 0, 1, 0, -1, -1]), nfoil=np.ones(6, int),
                    meta=np.array(json.dumps(dict(n=6, params={}))))
    runs = {}
    for arm in ('nominal', 'halfstep'):
        f = {k: v.copy() for k, v in baseline.items()}
        if arm == 'halfstep':
            f['fate'][2] = 6
            f['foil'][2] = -1
        g4ref.save(f, root / f'{arm}_seed17.npz')
        runs[f'{arm}_seed17'] = dict(convergence.summarize(selected, f, baseline),
                                    meta=json.loads(str(f['meta'])))
    _json(root / 'summary.json', dict(complete=True, unstarted=[], runs=runs))
    return root, original


def test_original_nonzero_block_threshold_split_and_cluster_errors(pilot):
    root, original = pilot
    report = pilotaudit.audit(root, original)
    assert report['block'] == dict(row_start=2, row_stop_exclusive=8, events=3,
                                   first_event=[1, 0, 11], last_event=[1, 0, 13])
    a = report['runs']['nominal_seed17']
    assert a['all']['fate1_split'] == dict(total=4, raw_p_zero=1,
                                        positive_T_below_0p1_MeV=1, inconsistent=2)
    assert a['all']['fast_minus_g4'] == dict(mean=pytest.approx(1 / 3), stderr=pytest.approx(2 / 3), events=3)
    assert a['mu_minus']['fast_minus_g4'] == a['all']['fast_minus_g4']
    assert a['mu_plus']['fast_minus_g4'] == dict(mean=None, stderr=None, events=0)
    assert a['mu_plus']['first_foil_minus_g4'] == dict(mean=None, stderr=None, events=0)
    assert a['mu_plus']['muons'] == 0
    assert a['all']['first_foil_minus_g4']['stderr'] == pytest.approx(np.sqrt(13) / 6)
    assert report['runs']['halfstep_seed17']['all']['minus_nominal']['mean'] == pytest.approx(-1 / 6)
    assert len(report['artifacts']['output_arms']) == 2
    assert all(len(s['sha256']) == 64 for s in report['current_comparison_sources'])
    assert report['artifacts']['original_input']['sha256'] == _identity(original)['sha256']
    assert 'no closure claim' in report['interpretation']
    assert 'not retroactively certified' in report['source_certification']
    json.dumps(report, allow_nan=False)


def test_selected_input_tampering_rejected(pilot):
    root, original = pilot
    path = root / 'input_block.npz'
    g = g4ref.load(path)
    g['extra'][0] += 1
    g4ref.save(g, path)
    with pytest.raises(ValueError, match='selected input: SHA256 mismatch'):
        pilotaudit.audit(root, original)


def test_original_input_tampering_rejected(pilot):
    root, original = pilot
    g = g4ref.load(original)
    g['extra'][-1] += 1
    g4ref.save(g, original)
    with pytest.raises(ValueError, match='original input: SHA256 mismatch'):
        pilotaudit.audit(root, original)


@pytest.mark.parametrize('change', ['reorder', 'extra_column'])
def test_exact_block_equality_even_with_updated_selected_hash(pilot, change):
    root, original = pilot
    path = root / 'input_block.npz'
    g = g4ref.load(path)
    if change == 'reorder':
        g = {k: v[::-1] for k, v in g.items()}
    else:
        g['extra'][0] += 1
    g4ref.save(g, path)
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['selected_input'] = _identity(path)
    _json(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='not exactly equal'):
        pilotaudit.audit(root, original)


@pytest.mark.parametrize('key', ['fate', 'p', 'x', 't', 'foil', 'nfoil', 'new_array'])
def test_every_arm_array_requires_exact_count(pilot, key):
    root, original = pilot
    path = root / 'halfstep_seed17.npz'
    f = g4ref.load(path)
    f[key] = np.zeros(5) if key == 'new_array' else f[key][:-1]
    g4ref.save(f, path)
    with pytest.raises(ValueError, match='exactly n=6 rows'):
        pilotaudit.audit(root, original)


@pytest.mark.parametrize('key,value', [('fate', 0), ('fate', 10), ('foil', -1), ('nfoil', -1)])
def test_invalid_fates_and_foils(pilot, key, value):
    root, original = pilot
    path = root / 'nominal_seed17.npz'
    f = g4ref.load(path)
    f[key][0] = value
    g4ref.save(f, path)
    with pytest.raises(ValueError):
        pilotaudit.audit(root, original)


@pytest.mark.parametrize('field', ['stop', 'first_foil', 'fast_minus_g4', 'minus_nominal', 'first_minus_nominal'])
def test_saved_essential_summary_mismatch(pilot, field):
    root, original = pilot
    path = root / 'summary.json'
    summary = json.loads(path.read_text())
    summary['runs']['halfstep_seed17'][field]['mean'] += 0.01
    _json(path, summary)
    with pytest.raises(ValueError, match='saved summary mismatch'):
        pilotaudit.audit(root, original)


def test_historical_summary_without_added_keys(pilot):
    root, original = pilot
    path = root / 'summary.json'
    summary = json.loads(path.read_text())
    for saved in summary['runs'].values():
        for key in ('first_foil_minus_g4', 'first_foil_counts'):
            saved.pop(key)
        for charge in ('mu_minus', 'mu_plus'):
            saved[charge].pop('first_foil_fast')
            saved[charge].pop('first_foil_g4')
    _json(path, summary)
    report = pilotaudit.audit(root, original)
    assert report['runs']['nominal_seed17']['historical_summary_missing_added_keys'] == [
        'first_foil_counts', 'first_foil_minus_g4']
    summary['runs']['nominal_seed17'].pop('minus_nominal')
    _json(path, summary)
    with pytest.raises(ValueError, match='missing essential'):
        pilotaudit.audit(root, original)


def test_charge_cluster_contrasts_not_row_errors(pilot):
    root, original = pilot
    g = g4ref.load(original)
    # Both charges in all three original source events.
    g['q'][::2] = 1
    g4ref.save(g, original)
    selected, _ = convergence.select_block(g, 2, 6)
    g4ref.save(selected, root / 'input_block.npz')
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest.update(input=_identity(original), selected_input=_identity(root / 'input_block.npz'))
    _json(root / 'manifest.json', manifest)
    summary = json.loads((root / 'summary.json').read_text())
    baseline = g4ref.load(root / 'nominal_seed17.npz')
    for name in summary['runs']:
        f = g4ref.load(root / f'{name}.npz')
        summary['runs'][name] = dict(convergence.summarize(selected, f, baseline), meta=json.loads(str(f['meta'])))
    _json(root / 'summary.json', summary)
    report = pilotaudit.audit(root, original)
    for charge in ('mu_minus', 'mu_plus'):
        a = report['runs']['nominal_seed17'][charge]
        assert a['muons'] == 3
        assert a['fast_minus_g4']['stderr'] == pytest.approx(2 / 3)
        assert a['first_foil_minus_g4']['events'] == 3
    summary['runs']['nominal_seed17']['mu_minus']['stops_fast'] += 1
    _json(root / 'summary.json', summary)
    with pytest.raises(ValueError, match='saved summary mismatch'):
        pilotaudit.audit(root, original)


def test_cli_new_output_only_and_preserves_artifacts(pilot, tmp_path):
    root, original = pilot
    before = {p: p.read_bytes() for p in root.iterdir()}
    output = tmp_path / 'audit.json'
    argv = ['--pilot', str(root), '--g4', str(original), '--out', str(output)]
    pilotaudit.main(argv)
    assert json.loads(output.read_text())['verified']
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(SystemExit):
        pilotaudit.main(argv)


def test_block_boundary_split_rejected(pilot):
    root, original = pilot
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['block']['row_start'] = 1
    _json(path, manifest)
    with pytest.raises(ValueError, match='splits an event'):
        pilotaudit.audit(root, original)


def test_manifest_count_mismatch_rejected(pilot):
    root, original = pilot
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['n'] = 4
    _json(path, manifest)
    with pytest.raises(ValueError, match='original block identity'):
        pilotaudit.audit(root, original)


def test_unlisted_arm_cannot_escape_verification(pilot):
    root, original = pilot
    (root / 'unlisted_seed17.npz').write_bytes((root / 'nominal_seed17.npz').read_bytes())
    with pytest.raises(ValueError, match='arm files and summary runs differ'):
        pilotaudit.audit(root, original)


def test_cli_invalid_input_writes_no_report(pilot, tmp_path):
    root, original = pilot
    path = root / 'halfstep_seed17.npz'
    f = g4ref.load(path)
    f['p'] = f['p'][:-1]
    g4ref.save(f, path)
    output = tmp_path / 'invalid.json'
    with pytest.raises(SystemExit):
        pilotaudit.main(['--pilot', str(root), '--g4', str(original), '--out', str(output)])
    assert not output.exists()
