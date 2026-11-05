"""Unit tests: DIO generator from muon stops (beamfastsim.diospec) on synthetic tables and stop lists."""
import os

import numpy as np
import pytest

from beamfastsim import diospec as D


def _table(lo=1.0, hi=104.97, step=0.01):
    """Synthetic DIO-like table: left edges lo..hi in 0.01 MeV rows, steeply falling to the endpoint."""
    x = np.round(np.arange(lo, hi + step / 2, step), 2)
    y = np.clip(105.0 - x, 0.0, None) ** 5 + 1e-30
    return x, y


def test_restriction_follows_binned_spectrum_cut():
    x, y = _table()
    s = D.Spectrum(x, y, elow=95.0, ehi=1000.0)
    # rows are left edges: the first kept row is 95.00 (its bin starts at elow), the last is the table end
    assert s.pdf.size == 998
    assert s.xmin == pytest.approx(95.0, abs=1e-9)
    assert s.xmax == pytest.approx(104.98, abs=1e-9)
    assert s.pdf[0] == y[np.argmin(np.abs(x - 95.0))]
    # an ehi inside the table ends it at the first bin reaching above ehi
    s2 = D.Spectrum(x, y, elow=95.0, ehi=100.0)
    assert s2.xmax == pytest.approx(100.0, abs=1e-9)
    s3 = D.Spectrum(x, y)
    assert s3.xmin == pytest.approx(1.0, abs=1e-9) and s3.pdf.size == x.size


def test_sampled_spectrum_matches_table_chi2_and_respects_the_cut():
    x, y = _table()
    s = D.Spectrum(x, y, elow=95.0)
    rng = np.random.default_rng(7)
    E = s.sample(rng, 400_000)
    assert E.min() >= 95.0 and E.max() <= s.xmax
    # group the 998 table bins by 10 (0.1 MeV) and keep groups with >= 20 expected
    edges = s.edges
    h = np.histogram(E, edges)[0]
    expct = s.probabilities() * E.size
    g = 10
    hg = np.add.reduceat(h, np.arange(0, h.size, g))
    eg = np.add.reduceat(expct, np.arange(0, expct.size, g))
    m = eg >= 20
    chi2 = (((hg - eg) ** 2) / eg)[m].sum()
    ndf = m.sum() - 1
    assert ndf > 50
    assert chi2 / ndf < 1.3
    # uniform inside a bin: the position within the first (most populated) bin is flat
    f = (E[E < edges[1]] - edges[0]) / (edges[1] - edges[0])
    assert abs(f.mean() - 0.5) < 0.02


def test_offline_table_restriction_if_available():
    root = os.environ.get("MU2E_OFFLINE_ROOT")
    path = root and os.path.join(root, D.TABLE)
    if not path or not os.path.exists(path):
        pytest.skip("MU2E_OFFLINE_ROOT with the DIO table not set")
    s = D.Spectrum.from_file(path, 95.0, 1000.0)
    assert s.xmin == pytest.approx(95.0, abs=1e-6) and s.xmax == pytest.approx(104.98, abs=1e-6)


def test_momentum_and_direction():
    E = np.array([95.0, 104.9])
    assert np.allclose(D.momentum(E) ** 2 + D.ME ** 2, E ** 2)
    u = D.isotropic(np.random.default_rng(1), 200_000)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    assert abs(u[:, 2].mean()) < 0.006 and abs(np.mean(u[:, 2] ** 2) - 1 / 3) < 0.005


def _stops(nev=1000, seed=3):
    """Synthetic stop table: one stop per event, every 10th event has a second stop."""
    rng = np.random.default_rng(seed)
    iev = np.sort(np.concatenate([np.arange(nev), np.arange(0, nev, 10)]))
    n = iev.size
    return dict(iev=iev, x=rng.normal(-3904.0, 30.0, n).astype(np.float32).astype(float),
                y=rng.normal(0.0, 30.0, n).astype(np.float32).astype(float),
                z=rng.uniform(5470.0, 6270.0, n).astype(np.float32).astype(float),
                t=rng.uniform(100.0, 600.0, n), nev=nev)


def test_window_wraps_and_counts_every_stop():
    st = _stops()
    rows = D.window(st, 990, 20)
    ev = st["iev"][rows]
    assert list(dict.fromkeys(ev.tolist())) == list(range(990, 1000)) + list(range(0, 10))
    assert rows.size == 20 + 2          # events 990 and 0 have two stops
    assert ev[0] == 990 and ev[-1] == 9


def test_recover_window_from_kept_electrons():
    st = _stops()
    start, nev = 970, 50
    rows = D.window(st, start, nev)
    evt = (st["iev"][rows] - start) % st["nev"] + 1          # art event numbers 1..nev
    keep = np.random.default_rng(5).random(rows.size) < 0.5  # the production filter keeps about half
    s0, nm, cons = D.recover_window(st, st["x"][rows][keep].astype(np.float32), st["y"][rows][keep].astype(np.float32),
                                    st["z"][rows][keep].astype(np.float32), evt[keep])
    assert s0 == start and nm == keep.sum() and cons == 1.0


def test_generate_starts_at_the_stop_and_later_in_time():
    st = _stops()
    x, y = _table()
    s = D.Spectrum(x, y, elow=95.0)
    rows = D.window(st, 0, 100)
    g = D.generate(np.random.default_rng(2), s, st, rows)
    assert g["p"].size == rows.size
    assert np.array_equal(g["x"][:, 2], st["z"][rows])
    dt = g["t"] - st["t"][rows]
    assert (dt > 0).all()
    big = D.generate(np.random.default_rng(3), s, st, np.zeros(200_000, int))
    assert abs((big["t"] - st["t"][0]).mean() / D.TAU_MU_AL - 1) < 0.01


def test_starts_cli_writes_cechain_ref_format(tmp_path):
    st = _stops()
    np.savez_compressed(tmp_path / "stops.npz", **st)
    x, y = _table()
    np.savetxt(tmp_path / "t.tbl", np.stack([x, y], axis=1), fmt="%.2f\t%.17g")
    out = tmp_path / "starts.npz"
    D.main(["starts", "--stops", str(tmp_path / "stops.npz"), "--table", str(tmp_path / "t.tbl"), "--events", "40",
            "--window-starts", "0", "500", "--out", str(out)])
    r = np.load(out)
    p = np.sqrt(r["px"] ** 2 + r["py"] ** 2 + r["pz"] ** 2)
    assert np.allclose(p, D.momentum(r["E"]))
    assert list(r["shard_generated"]) == [44, 44] and r["x"].size == 88
    assert r["E"].min() >= 95.0


def test_entry_order_reorders_events_to_tree_entries():
    st = _stops(nev=6)
    st["run"] = np.full(st["iev"].size, 1430)
    st["subrun"] = np.zeros(st["iev"].size, int)
    st["evt"] = 100 + st["iev"]                       # read (sorted) order: events 100..105
    entries = np.array([103, 100, 105, 101, 104, 102])  # tree-entry order of the same events
    out = D.entry_order(st, np.full(6, 1430), np.zeros(6, int), entries)
    assert list(out["iev"][st["iev"] == 0]) == [1, 1]    # event 100 (two stops) is entry 1
    assert out["iev"][st["evt"] == 103][0] == 0 and out["nev"] == 6
    assert np.array_equal(out["iev_read"], st["iev"])
    with pytest.raises(ValueError):
        D.entry_order(st, np.full(6, 1430), np.zeros(6, int), entries + 1000)


def test_starts_cli_uses_each_file_gencount_and_fails_cleanly(tmp_path):
    st = _stops()
    np.savez_compressed(tmp_path / "stops.npz", **st)
    x, y = _table()
    np.savetxt(tmp_path / "t.tbl", np.stack([x, y], axis=1), fmt="%.2f\t%.17g")
    rows0, rows1 = D.window(st, 0, 30), D.window(st, 600, 50)
    ref = dict(ifile=np.r_[np.zeros(rows0.size, int), np.ones(rows1.size, int)], gencount=np.array([30, 50]),
               x=st["x"][np.r_[rows0, rows1]], y=st["y"][np.r_[rows0, rows1]], z=st["z"][np.r_[rows0, rows1]],
               evt=np.r_[st["iev"][rows0] + 1, st["iev"][rows1] - 600 + 1])
    np.savez_compressed(tmp_path / "ref.npz", **ref)
    out = tmp_path / "starts.npz"
    D.main(["starts", "--stops", str(tmp_path / "stops.npz"), "--table", str(tmp_path / "t.tbl"),
            "--ref", str(tmp_path / "ref.npz"), "--out", str(out)])
    r = np.load(out)
    assert list(r["shard_start"]) == [0, 600] and list(r["shard_events"]) == [30, 50]
    assert list(r["shard_generated"]) == [rows0.size, rows1.size]
    bad = dict(ref, x=ref["x"] + 1.0)
    np.savez_compressed(tmp_path / "bad.npz", **bad)
    with pytest.raises(SystemExit):
        D.main(["starts", "--stops", str(tmp_path / "stops.npz"), "--table", str(tmp_path / "t.tbl"),
                "--ref", str(tmp_path / "bad.npz"), "--out", str(tmp_path / "x.npz")])
