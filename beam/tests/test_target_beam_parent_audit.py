"""Synthetic selected-beam content checks; no generation or production data."""
import json

import numpy as np
import pytest

from beamfastsim import target_beam_parent_audit as A


def columns():
    return dict(run=np.ones(4, int), subrun=np.full(4, 2), evt=np.array([1, 1, 2, 2]),
        trk=np.array([1, 2, 1, 2]), pdg=np.array([13, -13, 13, 13]), codestop=np.array([54, 54, 1, 54]),
        pstop=np.array([10., 20., 30., 40.]), xstop=np.zeros(4), ystop=np.ones(4), zstop=np.full(4, 5.),
        tstop=np.arange(4.) + 350.)


def test_exact_accepted_set_support_and_selector_multiplicity():
    c = columns()
    r = A.selected_beam(c, [[1, 2, 2], [1, 2, 1]])
    assert r['selected_events'] == 2 and r['selected_muon_rows'] == 4
    assert r['eligible_muons'] == 3 and r['eligible_mu_minus'] == 2 and r['eligible_mu_plus'] == 1
    assert r['event_multiplicity'] == {'1': 1, '2': 1}
    assert r['eligible_momentum_range'] == [10., 40.]
    assert r['exact_accepted_event_keys_verified'] and not r['source_event_ancestry_audited']
    assert not r['compressed_volume_contract_verified'] and not r['POT_normalization_inferred']
    order = np.array([2, 0, 3, 1])
    assert A.selected_beam({k: v[order] for k, v in c.items()}, [[1, 2, 1], [1, 2, 2]]) == r


@pytest.mark.parametrize('key,value', [
    ('evt', np.array([1, 1, 3, 3])), ('trk', np.ones(4, int)), ('codestop', np.zeros(4, int)),
    ('codestop', np.array([54, 54, 1, 1])), ('pdg', np.array([13, -13, 11, 13])),
    ('xstop', np.array([0., 1., np.nan, 3.])), ('pstop', np.array([-1., 1., 2., 3.])),
    ('run', np.array([1., 1.5, 1., 1.])), ('run', np.full(4, 2 ** 24, dtype=np.float32))])
def test_wrong_subset_invalid_states_or_empty_event_support_fail(key, value):
    c = columns()
    c[key] = value
    with pytest.raises(ValueError):
        A.selected_beam(c, [[1, 2, 1], [1, 2, 2]])


@pytest.mark.parametrize('accepted', [[], [[1, 2, 1]], [[1, 2, 1], [1, 2, 1]], [[1, 2, 1, 9]]])
def test_invalid_accepted_set(accepted):
    with pytest.raises(ValueError):
        A.selected_beam(columns(), accepted)


def test_cli_highest_cycle_hashes_and_exclusive_output(tmp_path):
    uproot = pytest.importorskip('uproot')
    root_path, admission, output = (tmp_path / k for k in ('selected.root', 'admission.json', 'beam.json'))
    with uproot.recreate(root_path) as root:
        root.mktree('cetree/ntpart', {k: v.dtype for k, v in columns().items()}).extend(columns())
        root.mktree('cetree/ntpart', {k: v.dtype for k, v in columns().items()}).extend(columns())
    admission.write_text(json.dumps(dict(process='R1LibraryPOT', all_event_status_zero=True,
        generation_census_reconciled=True, accepted_events=2, accepted_event_keys=[[1, 2, 1], [1, 2, 2]])))
    argv = ['--root', str(root_path), '--admission', str(admission), '--out', str(output)]
    A.main(argv)
    report = json.loads(output.read_text())
    assert report['selected_tree_sha256'] == A.checksum(root_path)
    assert report['admission_sha256'] == A.checksum(admission)
    assert len(report['tree_cycles']) == 2 and report['eligible_muons'] == 3
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        A.main(argv)
    assert output.read_bytes() == before


def test_cli_rejects_unreconciled_admission_without_output(tmp_path):
    admission, output = tmp_path / 'admission.json', tmp_path / 'out.json'
    admission.write_text(json.dumps(dict(process='R1LibraryPOT', all_event_status_zero=True)))
    with pytest.raises(ValueError, match='reconciled POT'):
        A.main(['--root', str(tmp_path / 'missing.root'), '--admission', str(admission), '--out', str(output)])
    assert not output.exists()
