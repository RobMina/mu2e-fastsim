"""Exact crossing selection on synthetic ntuples; no campaign or reference sidecars."""
import numpy as np
import pytest
import uproot

from beamfastsim import tscrossings as ts

TARGET = np.array([7, 42, 9, 3])


def rows(times, sid=None):
    n = len(times)
    a = {k: np.full(n, v, dtype='f4') for k, v in zip(ts.IDS, TARGET)}
    a.update({k: np.full(n, v, dtype='f4') for k, v in dict(pdg=-13, code=14,
               x=10, y=1, z=-20, px=3, py=4, pz=50, ke=10).items()})
    a['time'] = np.asarray(times, dtype='f4')
    a['sid'] = np.asarray([1] * n if sid is None else sid, dtype='f4')
    return a


def fixture(path):
    vd = rows([20, 10, 15, 15, 9], [2, 1, 91, 91, 1])
    vd['pz'][3] = -50
    vd['evt'][4] += 1
    with uproot.recreate(path) as f:
        f.mktree('inspect/ntvd', rows([999]))
        f.mktree('inspect/ntvd', vd)
        f.mktree('inspect/nttvd', rows([31, 30], [0, 0]))
        f.mktree('inspect/ntpart', {'must_not_read': np.array([42])})
    return path


def test_cli_exact_ids_highest_cycle_all_crossings(tmp_path, monkeypatch):
    path = fixture(tmp_path / 'stock.root')
    out = tmp_path / 'fresh' / 'crossings.npz'
    original = ts.read_table
    calls = []

    def read(path, tree):
        assert tree in ('ntvd', 'nttvd')
        calls.append(tree)
        return original(path, tree)

    monkeypatch.setattr(ts, 'read_table', read)
    args = ['--ntuple', str(path), '--id', *map(str, TARGET), '--out', str(out)]
    ts.main(args)
    assert calls == ['ntvd', 'nttvd']
    with np.load(out, allow_pickle=False) as a:
        np.testing.assert_array_equal(a['vd_time'], [10, 15, 15, 20])
        np.testing.assert_array_equal(a['vd_entry'], [1, 2, 3, 0])
        np.testing.assert_array_equal(a['vd_sid'], [1, 91, 91, 2])
        np.testing.assert_array_equal(a['ds_time'], [30, 31])
        assert a['vd_pz'][2] < 0
        assert a['vd_momentum_mev_c'].dtype == np.dtype('f8')
    assert not out.with_suffix('.json').exists()
    before = out.read_bytes()
    with pytest.raises(FileExistsError, match='overwrite'):
        ts.main(args)
    assert out.read_bytes() == before
    assert calls == ['ntvd', 'nttvd']


def test_multiple_files_target_order_and_empty_matches(tmp_path):
    first = fixture(tmp_path / 'a.root')
    second = fixture(tmp_path / 'b.root')
    absent = TARGET + 100
    a = ts.extract([first, second], [absent, TARGET])
    np.testing.assert_array_equal(a['target_id'], [absent, TARGET])
    np.testing.assert_array_equal(a['vd_target'], np.ones(8, dtype=int))
    np.testing.assert_array_equal(a['ds_time'], [30, 30, 31, 31])
    np.testing.assert_array_equal(a['ds_file_index'], [0, 1, 0, 1])
    empty = ts.extract([first], [absent])
    assert empty['vd_x'].shape == (0,)
    assert empty['ds_time'].shape == (0,)
    assert empty['vd_momentum_mev_c'].dtype == np.dtype('f8')


@pytest.mark.parametrize('bad', [(), [TARGET, TARGET], [1, 2, 3, 4], [[1, 2, 3]], [[1, 2, 3, .5]]])
def test_reject_duplicate_empty_or_inexact_ids(bad):
    with pytest.raises(ValueError):
        ts.targets_checked(bad)


def test_arbitrary_species_cohorts_and_integer_precision():
    targets = np.array([[2, 300, 4, 5], [9, 2**53 + 1, 2, 8], [5, 600, 4, 12]], dtype=np.int64)
    a = rows([3, 1, 2])
    a.update(dict(zip(ts.IDS, targets.T)))
    a['pdg'] = np.array([-211, 13, 211])
    selected = ts.select_exact(a, targets[::-1])
    np.testing.assert_array_equal(selected['pdg'], [211, 13, -211])
    np.testing.assert_array_equal(selected['target'], [0, 1, 2])
    a = rows([1])
    a['evt'][0] = 2**24
    with pytest.raises(ValueError, match='precision'):
        ts.select_exact(a, [TARGET])


def test_no_spatial_or_time_fallback(tmp_path):
    path = fixture(tmp_path / 'stock.root')
    a = ts.extract([path], [TARGET + [0, 0, 0, 1]])
    assert a['vd_time'].size == a['ds_time'].size == 0


def test_nonfinite_selected_time_rejected():
    with pytest.raises(ValueError, match='nonfinite crossing time'):
        ts.select_exact(rows([np.nan]), [TARGET])


def test_incompatible_schema_rejected(tmp_path):
    first = fixture(tmp_path / 'a.root')
    second = fixture(tmp_path / 'b.root')
    with uproot.update(second) as f:
        f.mktree('inspect/ntvd', {**rows([1]), 'extra': np.ones(1)})
    with pytest.raises(ValueError, match='incompatible scalar schemas'):
        ts.extract([first, second], [TARGET])


def test_required_inputs_and_ids(tmp_path):
    with pytest.raises(SystemExit):
        ts.main(['--ntuple', 'absent.root', '--out', str(tmp_path / 'out.npz')])
    with pytest.raises(ValueError, match='at least one'):
        ts.extract([], [TARGET])
