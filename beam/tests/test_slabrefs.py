"""Selected G4 references are identity-bound, unpaired, and read-only."""
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from beamfastsim import slabrefs
from beamfastsim.reference_audit import _identity


@pytest.fixture
def make_campaign(tmp_path, monkeypatch):
    audits = {}
    audit = Mock(side_effect=lambda directory: audits[str(Path(directory))])
    monkeypatch.setattr(slabrefs.foilreport, 'audit', audit)

    def make(name='campaign', modern=False, pdg=-13, seed=71, mode='full', dropped=False, n=3):
        root = tmp_path / name
        tag = f'moved_edge_{mode}' + ('_mup' if modern else '')
        directory = root / tag
        directory.mkdir(parents=True)
        case = dict(pdg=pdg, thickness_mm=.2, energy_MeV=2.02, cosine=1., events=n, seed=seed,
                    no_msc=mode.startswith('loss'), energy_max_MeV=2.02, cosine_min=1.)
        artifact, output = directory / 'job.fcl', directory / 'foil.root'
        artifact.write_text(f'config {seed}')
        output.write_text(f'ROOT identity {seed}')
        receipt = dict(complete=True, case=case, artifacts=[_identity(artifact)], output=_identity(output))
        receipt['artifacts'][0]['path'] = '/moved/worker/job.fcl'
        receipt['output']['path'] = '/moved/worker/foil.root'
        (directory / 'receipt.json').write_text(json.dumps(receipt))
        ex = np.arange(n) != 0
        g = dict(event=np.arange(1, n + 1), entry_MeV=np.full(n, 2.02), entry_cos=np.ones(n),
                 exit_MeV=np.where(ex, .5 + seed * 1e-4, 0.), exit=ex,
                 physical_stop=~ex, other=np.zeros(n, bool), exit_z_mm=np.where(ex, .1, 0.))
        census = dict(primaries=n, exits=n - 1, physical_stops=1, threshold_stops=1, other=0, back_exits=0)
        run = dict(receipt=_identity(directory / 'receipt.json'), census=census)
        manifest = dict(complete=True, cases=[['moved_edge', 2.02, .2, 1.]], modes={mode: []})
        if modern:
            np.savez(directory / 'scan.npz', **g)
            run.update(scan='moved_edge', mode=mode, pdg=pdg, thickness_mm=.2,
                       scan_npz=_identity(directory / 'scan.npz'), seed_offset=0)
            manifest.pop('cases')
            manifest['scans'] = {'moved_edge': [.2, 2.02, 2.02, 1., 1.]}
        manifest['runs'] = {tag: run}
        (root / 'manifest.json').write_text(json.dumps(manifest))
        audits[str(directory)] = (census, g, receipt)
        if dropped:
            output.unlink()
        return root, directory, manifest, receipt, g

    make.audit = audit
    return make


def load(roots, pdg=-13):
    return slabrefs.load_references(roots, 'moved_edge', pdg, .2, 2.02, 1.)


def save_manifest(root, manifest):
    (root / 'manifest.json').write_text(json.dumps(manifest))


def save_receipt(root, directory, manifest, receipt):
    (directory / 'receipt.json').write_text(json.dumps(receipt))
    manifest['runs'][directory.name]['receipt'] = _identity(directory / 'receipt.json')
    save_manifest(root, manifest)


@pytest.mark.parametrize('modern', [False, True])
def test_formats_preserve_full_label_and_local_identities(make_campaign, modern):
    root, directory, _, _, g = make_campaign(modern=modern)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    refs, records = load(str(root))
    assert set(refs) == {'full'}
    for key in g:
        np.testing.assert_array_equal(refs['full'][key], g[key])
    assert {'manifest', 'receipt', 'artifact', 'output'} <= {r['role'] for r in records}
    for record in records:
        assert Path(record['path']).is_file()
        assert record['sha256'] == _identity(record['path'])['sha256']
    assert next(r for r in records if r['role'] == 'output')['path'] == str(directory / 'foil.root')
    assert make_campaign.audit.call_count == int(not modern)
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_legacy_reads_actual_receipt_charge_not_label(make_campaign):
    root, _, _, _, _ = make_campaign(pdg=13)
    with pytest.raises(ValueError, match='no matching'):
        load(root)
    refs, _ = load(root, pdg=13)
    assert len(refs['full']['exit_MeV']) == 3


def test_multiple_roots_pool_all_rows_without_pairing(make_campaign):
    a, *_ = make_campaign('a', seed=71, n=3)
    b, *_ = make_campaign('b', modern=True, seed=72, n=5)
    c, *_ = make_campaign('c', seed=73, mode='loss_finer', n=4)
    refs, records = load([a, b, c])
    assert set(refs) == {'full', 'loss_finer'}
    assert len(refs['full']['exit_MeV']) == 8
    np.testing.assert_array_equal(refs['full']['event'], [1, 2, 3, 1, 2, 3, 4, 5])
    assert len(refs['loss_finer']['exit_MeV']) == 4
    assert sum(r['role'] == 'manifest' for r in records) == 3


def test_modern_dropped_root_keeps_recorded_output(make_campaign):
    root, directory, _, receipt, _ = make_campaign(modern=True, dropped=True)
    refs, records = load(root)
    assert len(refs['full']['exit_MeV']) == 3
    output = next(r for r in records if r['role'] == 'receipt')['recorded_output']
    assert output['path'] == str(directory / 'foil.root')
    assert output['sha256'] == receipt['output']['sha256']
    assert output['available'] is False and output['verified'] is False
    assert all(Path(r['path']).is_file() for r in records)
    make_campaign.audit.assert_not_called()


def test_selected_only_modern_does_not_open_other_runs(make_campaign):
    root, _, manifest, _, _ = make_campaign(modern=True)
    manifest['runs']['unavailable'] = dict(scan='nom_edge', pdg=-13)
    manifest['runs']['other_charge'] = dict(scan='moved_edge', pdg=13)
    save_manifest(root, manifest)
    assert 'full' in load(root)[0]


@pytest.mark.parametrize('modern', [False, True])
@pytest.mark.parametrize('key,value', [('thickness_mm', .4), ('energy_MeV', 3.), ('cosine', .8),
                                      ('energy_max_MeV', 3.), ('cosine_min', .8)])
def test_wrong_receipt_cell_rejected(make_campaign, modern, key, value):
    root, directory, manifest, receipt, _ = make_campaign(modern=modern)
    receipt['case'][key] = value
    save_receipt(root, directory, manifest, receipt)
    with pytest.raises(ValueError, match='receipt'):
        load(root)
    make_campaign.audit.assert_not_called()


def test_modern_wrong_receipt_charge_rejected(make_campaign):
    root, directory, manifest, receipt, _ = make_campaign(modern=True)
    receipt['case']['pdg'] = 13
    save_receipt(root, directory, manifest, receipt)
    with pytest.raises(ValueError, match='pdg'):
        load(root)


@pytest.mark.parametrize('modern', [False, True])
def test_wrong_manifest_geometry_rejected(make_campaign, modern):
    root, _, manifest, _, _ = make_campaign(modern=modern)
    if modern:
        next(iter(manifest['runs'].values()))['thickness_mm'] = .4
    else:
        manifest['cases'][0][2] = .4
    save_manifest(root, manifest)
    with pytest.raises(ValueError, match='manifest'):
        load(root)


@pytest.mark.parametrize('field', ['receipt', 'scan_npz'])
def test_missing_hash_fails(make_campaign, field):
    root, _, manifest, _, _ = make_campaign(modern=True)
    del next(iter(manifest['runs'].values()))[field]['sha256']
    save_manifest(root, manifest)
    with pytest.raises(ValueError, match='missing hash'):
        load(root)


@pytest.mark.parametrize('field', ['artifacts', 'output'])
def test_missing_artifact_hash_fails(make_campaign, field):
    root, directory, manifest, receipt, _ = make_campaign(modern=True, dropped=True)
    item = receipt[field][0] if field == 'artifacts' else receipt[field]
    del item['sha256']
    save_receipt(root, directory, manifest, receipt)
    with pytest.raises(ValueError, match='missing hash'):
        load(root)


@pytest.mark.parametrize('filename', ['receipt.json', 'scan.npz', 'job.fcl', 'foil.root'])
def test_changed_saved_input_rejected(make_campaign, filename):
    root, directory, _, _, _ = make_campaign(modern=True)
    with (directory / filename).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed hash'):
        load(root)


@pytest.mark.parametrize('modern', [False, True])
def test_duplicate_directory_rejected(make_campaign, modern):
    root, *_ = make_campaign(modern=modern)
    with pytest.raises(ValueError, match='duplicate'):
        load([root, root])


def test_duplicate_seed_across_formats_rejected(make_campaign):
    a, *_ = make_campaign('a', seed=71)
    b, *_ = make_campaign('b', modern=True, seed=71, mode='full_finest')
    with pytest.raises(ValueError, match='duplicate'):
        load([a, b])


def test_duplicate_output_identity_despite_different_seed_rejected(make_campaign):
    a, adir, *_ = make_campaign('a', seed=71)
    b, bdir, manifest, receipt, _ = make_campaign('b', modern=True, seed=72)
    (bdir / 'foil.root').write_bytes((adir / 'foil.root').read_bytes())
    receipt['output'] = _identity(bdir / 'foil.root')
    save_receipt(b, bdir, manifest, receipt)
    with pytest.raises(ValueError, match='duplicate'):
        load([a, b])


@pytest.mark.parametrize('modern', [False, True])
def test_incomplete_manifest_rejected(make_campaign, modern):
    root, _, manifest, _, _ = make_campaign(modern=modern)
    manifest['complete'] = False
    save_manifest(root, manifest)
    with pytest.raises(ValueError, match='incomplete'):
        load(root)


def test_incomplete_receipt_rejected(make_campaign):
    root, directory, manifest, receipt, _ = make_campaign(modern=True)
    receipt['complete'] = False
    save_receipt(root, directory, manifest, receipt)
    with pytest.raises(ValueError, match='incomplete'):
        load(root)


@pytest.mark.parametrize('problem', ['other', 'entry', 'census', 'event', 'length'])
def test_saved_scan_still_requires_consistent_audited_census(make_campaign, problem):
    root, directory, manifest, _, g = make_campaign(modern=True)
    if problem == 'other':
        g['other'][1] = True
    elif problem == 'entry':
        g['entry_MeV'][1] = 3.
    elif problem == 'event':
        g['event'][1] = 1
    elif problem == 'length':
        g['exit_MeV'] = g['exit_MeV'][:1]
    else:
        manifest['runs'][directory.name]['census']['primaries'] = 9
    np.savez(directory / 'scan.npz', **g)
    manifest['runs'][directory.name]['scan_npz'] = _identity(directory / 'scan.npz')
    save_manifest(root, manifest)
    with pytest.raises(ValueError):
        load(root)


def test_legacy_audit_failure_not_bypassed(make_campaign):
    root, *_ = make_campaign()
    make_campaign.audit.side_effect = ValueError('highest-cycle census failure')
    with pytest.raises(ValueError, match='highest-cycle'):
        load(root)


def test_legacy_requires_root(make_campaign):
    root, *_ = make_campaign(dropped=True)
    with pytest.raises(ValueError, match='missing G4 ROOT'):
        load(root)


def test_no_match_fails(make_campaign):
    root, *_ = make_campaign(modern=True)
    with pytest.raises(ValueError, match='no matching'):
        slabrefs.load_references([root], 'nom_edge', -13, .1056, 1.4, 1.)


@pytest.mark.parametrize('modern', [False, True])
@pytest.mark.parametrize('key', ['seed', 'pdg'])
def test_missing_receipt_identity_rejected(make_campaign, modern, key):
    root, directory, manifest, receipt, _ = make_campaign(modern=modern)
    del receipt['case'][key]
    save_receipt(root, directory, manifest, receipt)
    with pytest.raises(ValueError, match=key):
        load(root)


def test_altered_legacy_receipt_rejected_before_audit(make_campaign):
    root, directory, _, receipt, _ = make_campaign()
    receipt['case']['pdg'] = 13
    (directory / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='changed hash'):
        load(root)
    make_campaign.audit.assert_not_called()


def test_duplicate_scan_with_distinct_receipt_and_root_rejected(make_campaign):
    a, adir, *_ = make_campaign('a', modern=True, seed=71)
    b, bdir, manifest, _, _ = make_campaign('b', modern=True, seed=72)
    (bdir / 'scan.npz').write_bytes((adir / 'scan.npz').read_bytes())
    manifest['runs'][bdir.name]['scan_npz'] = _identity(bdir / 'scan.npz')
    save_manifest(b, manifest)
    with pytest.raises(ValueError, match='duplicate'):
        load([a, b])
