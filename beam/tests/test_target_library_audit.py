"""Synthetic fresh-library contracts; no transport, network or site data."""
import json

import numpy as np
import pytest

from beamfastsim import target_library_audit as A


PROCESS = "R1LibraryStops"


def event(evt, *, status=0, tracks=7, overflow=0, steps=0, accepted=True, trigger=True):
    text = (f"PrintModule Run/Subrun/Event 1430 6000 {evt}\n"
            f"ProductPrint mu2e::StatusG4_g4run__{PROCESS}\n"
            f"status: {status} nG4Tracks: {tracks} overflowSimP: {overflow} nStepLimit: {steps}\n")
    if trigger:
        text += (f"ProductPrint art::TriggerResults_TriggerResults__{PROCESS}\n"
                 f"{'pass' if accepted else 'fail'} 0 PrimaryPath\n")
    return text


def logs(tmp_path, original=None, saved=None):
    gen, read = tmp_path / "generation.log", tmp_path / "read.log"
    gen.write_text(original if original is not None else event(1, trigger=False) + event(2, trigger=False))
    read.write_text(saved if saved is not None else event(2, accepted=False) + event(1))
    return gen, read


def columns():
    return dict(run=np.full(4, 1430, np.float32), subrun=np.full(4, 6000, np.float32),
                evt=np.array([7, 7, 7, 9], np.float32), trk=np.array([1, 2, 3, 1], np.float32),
                pdg=np.array([13, -13, 13, 13], np.float32), codestop=np.array([32, 0, 32, 32], np.float32),
                pstop=np.zeros(4), xstop=np.zeros(4), ystop=np.zeros(4), zstop=np.zeros(4), tstop=np.ones(4))


def test_shuffled_reconciliation_exact_accepted_subset(tmp_path):
    gen, read = logs(tmp_path)
    ledger, report = A.reconcile(gen, read, PROCESS, 2)
    assert ledger['accepted'].tolist() == [0, 1]
    assert report['accepted_event_keys'] == [[1430, 6000, 1]]
    assert report['generation_census_reconciled'] and report['all_event_status_zero']
    assert report['fresh_POT_events'] is None and report['beam_draw_events'] == 2
    assert not report['filter_implication_certified'] and not report['generated_yield_validated']


@pytest.mark.parametrize('change', [dict(status=1), dict(status=10), dict(tracks=8), dict(overflow=1), dict(steps=1)])
def test_reconciliation_checks_every_counter(tmp_path, change):
    gen, read = logs(tmp_path, saved=event(2, accepted=False) + event(1, **change))
    with pytest.raises(ValueError, match='independent generation'):
        A.reconcile(gen, read, PROCESS, 2)


@pytest.mark.parametrize('change', [dict(status=1), dict(status=10), dict(overflow=1), dict(steps=1)])
def test_matching_nonzero_census_blocks_even_rejected_events(tmp_path, change):
    gen, read = logs(tmp_path, event(1, trigger=False, **change), event(1, accepted=False, **change))
    assert A.read_log(read, 1, PROCESS)['status'].tolist() == [change.get('status', 0)]
    with pytest.raises(ValueError, match='all-event'):
        A.reconcile(gen, read, PROCESS, 1)


@pytest.mark.parametrize('saved', [event(1, status=9), event(1) + event(1), event(1, trigger=False), event(3)])
def test_unknown_duplicate_missing_or_wrong_identity(tmp_path, saved):
    expected = 2 if saved.count('Run/Subrun/Event') == 2 else 1
    original = event(1, trigger=False) + (event(2, trigger=False) if expected == 2 else '')
    gen, read = logs(tmp_path, original, saved)
    with pytest.raises(ValueError):
        A.reconcile(gen, read, PROCESS, expected)


def test_no_zero_event_or_empty_library_admission(tmp_path):
    gen, read = logs(tmp_path)
    for expected in (0, -1, 20001, True):
        with pytest.raises(ValueError):
            A.reconcile(gen, read, PROCESS, expected)
    gen, read = logs(tmp_path, event(1, trigger=False), event(1, accepted=False))
    with pytest.raises(ValueError, match='no automatic refill'):
        A.reconcile(gen, read, PROCESS, 1)
    gen, read = logs(tmp_path)
    with pytest.raises(ValueError, match='insufficient'):
        A.reconcile(gen, read, PROCESS, 2, minimum_stop_events=2)


def test_selected_parent_exact_join_and_multiplicity(monkeypatch):
    monkeypatch.setattr(A.G, 'foil_index', lambda geom, x, y, z: np.zeros(len(x), int))
    accepted = np.array([[1430, 6000, 9], [1430, 6000, 7]])
    out, report = A.selected_parent(columns(), None, accepted)
    assert out['event_multiplicity'].tolist() == [2, 1]
    assert out['trk'].tolist() == [1, 3, 1] and out['at_rest'].all()
    assert report['eligible_muons'] == 3
    for ids in (accepted[:1], np.array([[1430, 6000, 8], [1430, 6000, 9]])):
        with pytest.raises(ValueError, match='exact accepted'):
            A.selected_parent(columns(), None, ids)
    for field, value in [('evt', 7.5), ('run', 2**24), ('trk', 1.5)]:
        c = columns()
        c[field][0] = value
        with pytest.raises(ValueError, match='unsafe'):
            A.selected_parent(c, None, accepted)
    c = columns()
    c['trk'][1] = 1
    with pytest.raises(ValueError, match='duplicate'):
        A.selected_parent(c, None, accepted)
    c = columns()
    c['codestop'][:] = 0
    with pytest.raises(ValueError, match='no captured'):
        A.selected_parent(c, None, accepted)
    c = columns()
    c['codestop'][-1] = 0
    with pytest.raises(ValueError, match='without eligible'):
        A.selected_parent(c, None, accepted)
    monkeypatch.setattr(A.G, 'foil_index', lambda geom, x, y, z: np.full(len(x), -1))
    with pytest.raises(ValueError, match='outside staged target foils'):
        A.selected_parent(columns(), None, accepted)


def test_cli_exclusive_strict_json_npz_and_highest_art_cycle(tmp_path, monkeypatch, capsys):
    uproot = pytest.importorskip('uproot')
    gen, read = logs(tmp_path, event(7, trigger=False) + event(8, trigger=False) + event(9, trigger=False),
                     event(9) + event(8, accepted=False) + event(7))
    root_path, art = tmp_path / 'read.root', tmp_path / 'selected.art'
    with uproot.recreate(root_path) as root:
        root[TREE := A.TREE] = columns()
    with uproot.recreate(art) as root:
        root.mktree('Events', {'dummy': 'int32'}).extend({'dummy': np.zeros(1, np.int32)})
        root.mktree('Events', {'dummy': 'int32'}).extend({'dummy': np.zeros(2, np.int32)})
    assert A.art_entries(art) == 2
    monkeypatch.setattr(A.G, 'from_offline', lambda *args: None)
    monkeypatch.setattr(A.G, 'foil_index', lambda geom, x, y, z: np.zeros(len(x), int))
    monkeypatch.setattr(A, 'physical_event_order', lambda parent, path, swap: (parent, {'explicit_byteswap': swap}))
    out, parent = tmp_path / 'receipt.json', tmp_path / 'parent.npz'
    args = ['--generation-log', str(gen), '--read-log', str(read), '--process', PROCESS,
            '--expected-events', '3', '--out', str(out), '--parent-root', str(root_path),
            '--parent-art', str(art), '--offline-root', str(tmp_path), '--parent-out', str(parent),
            '--beam-parent-events', '1', '--fileindex-byteswap']
    A.main(args)
    report = json.loads(out.read_text(), parse_constant=lambda token: pytest.fail(token))
    assert report == json.loads(capsys.readouterr().out)
    assert report['library_muons'] == 3 and report['selected_parent_verified']
    assert report['parent_tree'] == TREE and report['beam_parent_events'] == 1
    assert report['parent_art_sha256'] == A.checksum(art)
    assert report['parent_tree_sha256'] == A.checksum(root_path)
    assert report['generation_source_sha256'] == A.checksum(gen)
    assert report['read_source_sha256'] == A.checksum(read)
    with np.load(parent, allow_pickle=False) as saved:
        assert json.loads(saved['metadata'].item()) == report
        assert saved['event_multiplicity'].tolist() == [2, 1]
        assert all(saved[k].dtype.kind != 'O' for k in saved.files)
    for existing, other in ((out, parent), (parent, out)):
        if not existing.exists():
            existing.write_bytes(b'preserve existing output')
        other.unlink(missing_ok=True)
        before = existing.read_bytes()
        with pytest.raises(FileExistsError):
            A.main(args)
        assert existing.read_bytes() == before and not other.exists()
        existing.unlink()


def test_cli_bad_art_count_and_invalid_input_leave_no_receipt(tmp_path, monkeypatch):
    gen, read = logs(tmp_path)
    out = tmp_path / 'receipt.json'
    args = ['--generation-log', str(gen), '--read-log', str(read), '--process', PROCESS,
            '--expected-events', '2', '--out', str(out)]
    monkeypatch.setattr(A, 'art_entries', lambda path: 2)
    with pytest.raises(ValueError, match='Events count'):
        A.main(args + ['--parent-art', str(tmp_path / 'bad.art')])
    assert not out.exists()
    read.write_text(event(2, accepted=False) + event(1, tracks=99))
    with pytest.raises(ValueError, match='independent generation'):
        A.main(args)
    assert not out.exists()


def test_pot_receipt_is_fresh_event_census_not_a_draw_normalization(tmp_path, capsys):
    gen, read = logs(tmp_path)
    for path in (gen, read):
        path.write_text(path.read_text().replace(PROCESS, 'R1LibraryPOT'))
    out = tmp_path / 'pot.json'
    A.main(['--generation-log', str(gen), '--read-log', str(read), '--process', 'R1LibraryPOT',
            '--expected-events', '2', '--out', str(out)])
    report = json.loads(out.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report['fresh_POT_events'] == 2 and report['beam_draw_events'] is None
    assert not report['POT_normalization_inferred'] and not report['selected_parent_verified']
