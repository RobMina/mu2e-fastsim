"""Synthetic generator contracts; no production tables or histogram fixtures."""
from types import SimpleNamespace

import numpy as np
import pytest

from beamfastsim import cechain as C
from beamfastsim import target_sources as S


def _stops(pdg=13):
    return dict(pdg=np.array([pdg, pdg, -13]), at_rest=np.array([True, True, True]), foil=np.array([0, 1, -1]),
                xstop=np.array([1., 2., 3.]), ystop=np.array([4., 5., 6.]), zstop=np.array([7., 8., 9.]),
                tstop=np.array([10., 20., 30.]), run=np.array([1, 1, 1]),
                subrun=np.array([2, 2, 2]), evt=np.array([0, 1, 2]))


def test_offline_table_bins_are_left_edges_and_only_whole_bins_are_kept(tmp_path):
    f = tmp_path / "spectrum.tbl"
    np.savetxt(f, np.c_[np.arange(1., 7.), np.arange(1., 7.)])
    s = S.BinnedSpectrum.table(f, 2., 5.)
    assert np.array_equal(s.edges, [2, 3, 4, 5])
    assert np.array_equal(s.pdf, [2, 3, 4])
    centred = S.BinnedSpectrum.table(f, 2., 5., bin_center=True)
    assert np.array_equal(centred.edges, [2.5, 3.5, 4.5])
    assert np.array_equal(centred.pdf, [3, 4])
    a = s.sample(np.random.default_rng(1), 100_000)
    counts = np.histogram(a, s.edges)[0]
    assert np.all(np.abs(counts - counts.sum() * s.pdf / s.pdf.sum()) < 5 * np.sqrt(counts))
    assert np.allclose([a[(a >= lo) & (a < hi)].mean() for lo, hi in zip(s.edges[:-1], s.edges[1:])],
                       (s.edges[:-1] + s.edges[1:]) / 2, atol=0.01)


@pytest.mark.parametrize("edges,pdf", [([0, 1], [-1]), ([0, 1], [0]), ([0, 1, 3], [1, 1]),
                                       ([0, np.nan], [1]), ([0, 1], [np.inf])])
def test_invalid_binned_spectra_fail(edges, pdf):
    with pytest.raises(ValueError):
        S.BinnedSpectrum(edges, pdf)


def test_table_rejects_nonuniform_input_and_empty_restriction(tmp_path):
    f = tmp_path / "bad.tbl"
    np.savetxt(f, [[1, 1], [2, 1], [4, 1]])
    with pytest.raises(ValueError, match="uniform"):
        S.BinnedSpectrum.table(f, 1, 5)
    np.savetxt(f, [[1, 1], [2, 1], [3, 1]])
    with pytest.raises(ValueError, match="two whole bins"):
        S.BinnedSpectrum.table(f, 5, 6)


def test_dio_keeps_stop_identity_mass_shell_and_exponential_time():
    stops = _stops()
    spec = S.BinnedSpectrum([95, 100, 105], [3, 1])
    a = S.from_stops(np.random.default_rng(2), stops, 100_000, "dio", spec)
    b = S.from_stops(np.random.default_rng(2), stops, 100_000, "dio", spec)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    idx = a["stop_idx"]
    assert set(idx) == {0, 1}
    assert np.array_equal(a["x"], stops["xstop"][idx])
    delay = a["time"] - stops["tstop"][idx]
    assert (delay >= 0).all()
    assert delay.mean() == pytest.approx(C.TAU_MU_AL, rel=0.015)
    assert np.hypot(a["p"], S.ME).min() >= 95
    assert np.hypot(a["p"], S.ME).max() < 105
    assert np.allclose(np.linalg.norm(a["u"], axis=1), 1)
    assert abs(a["u"][:, 2].mean()) < 0.01
    assert np.mean(a["u"][:, 2] ** 2) == pytest.approx(1 / 3, abs=0.005)


def test_rpc_photons_use_pion_stop_time_and_are_not_electron_starts():
    a = S.from_stops(np.random.default_rng(3), _stops(-211), 1000, "rpc-external", S.rpc_spectrum())
    assert np.array_equal(a["time"], _stops(-211)["tstop"][a["stop_idx"]])
    assert (a["pdg"] == 22).all()
    assert a["p"].min() >= 50 and a["p"].max() < 134.6
    with pytest.raises(ValueError, match="photon conversion"):
        S.electron_starts(a)
    with pytest.raises(ValueError, match="pion stops"):
        S.from_stops(np.random.default_rng(3), _stops(13), 10, "rpc-internal", S.rpc_spectrum())


def test_internal_pair_conserves_energy_and_invariant_mass():
    E = np.linspace(50, 134, 8000)
    a = S.internal_pair(np.random.default_rng(4), E)
    ee = np.hypot(a["p"], S.ME)
    ep = np.hypot(a["positron_p"], S.ME)
    assert np.allclose(ee + ep, E, atol=1e-10)
    p3 = a["p"][:, None] * a["u"] + a["positron_p"][:, None] * a["positron_u"]
    m2 = E ** 2 - np.sum(p3 ** 2, axis=1)
    assert np.allclose(m2, a["pair_mass"] ** 2, rtol=1e-8, atol=1e-8)
    eta = np.sqrt(1 - (2 * S.ME / a["pair_mass"]) ** 2)
    assert (np.abs(a["asymmetry"]) <= eta).all()
    assert abs(a["asymmetry"].mean()) < 0.02
    assert np.allclose(np.linalg.norm(a["positron_u"], axis=1), 1)


def test_kwj_support_and_envelope():
    E = np.array([50., 100., 134.])[:, None, None]
    x = np.linspace(2 * S.ME, 134., 500)[None, :, None]
    y = np.linspace(-1., 1., 101)[None, None, :]
    pdf = S.kwj_density(E, x, y)
    assert np.isfinite(pdf).all() and (pdf >= 0).all()
    assert (pdf <= S.kwj_density(E, 2 * S.ME, 0) * (1 + 1e-12)).all()
    assert S.kwj_density(100., 101., 0.) == 0
    assert S.kwj_density(100., S.ME, 0.) == 0
    assert S.kwj_density(100., 2 * S.ME, 0.1) == 0


def test_source_adapter_chunks_without_resampling_and_does_not_change_ce_defaults(tmp_path):
    source = S.from_stops(np.random.default_rng(5), _stops(), 50, "dio", S.BinnedSpectrum([95, 100], [1]))
    f = tmp_path / "source.npz"
    np.savez(f, **source)
    args = SimpleNamespace(source=str(f), ref=None, chunk=1, nchunks=3)
    x, u, p, t, i = C._starts(args, np.random.default_rng(6))
    assert np.array_equal(i, np.arange(50)[1::3])
    assert np.array_equal(x[:, 0], source["x"][i])
    assert np.allclose(p, source["p"][i]) and np.array_equal(t, source["time"][i])
    assert np.allclose(np.linalg.norm(u, axis=1), 1)
    old = C.from_stops(np.random.default_rng(7), np.array([[1., 2., 3.]]), np.array([4.]), 20)
    assert np.array_equal(old[2], np.full(20, C.P_CE_AL))


def test_weighted_sources_are_not_silently_used_as_unweighted_electrons():
    source = S.from_stops(np.random.default_rng(5), _stops(), 10, "dio", S.BinnedSpectrum([95, 100], [1]))
    source["weight"] = np.full(10, 0.5)
    with pytest.raises(ValueError, match="weighted-yield"):
        S.electron_starts(source)


@pytest.mark.parametrize("E", [[0.], [2 * S.ME], [np.nan], [np.inf]])
def test_invalid_pair_energy(E):
    with pytest.raises(ValueError):
        S.internal_pair(np.random.default_rng(1), E)


def test_beam_point_stops_are_muon_stops_and_weights_fail_closed():
    beam = dict(x=np.array([[1., 2., 3.], [4., 5., 6.]]), t=np.array([1., 2.]), q=np.array([-1., 1.]),
                foil=np.array([0, 1]), muons=np.array(100))
    spec = S.BinnedSpectrum([95, 100], [1])
    a = S.from_stops(np.random.default_rng(1), beam, 10, "dio", spec)
    assert np.all(a["stop_idx"] == 0) and np.all(a["x"] == 1)
    with pytest.raises(ValueError, match="pion stops"):
        S.from_stops(np.random.default_rng(1), beam, 10, "rpc-external", spec)
    beam["weight"] = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="weighted stop"):
        S.from_stops(np.random.default_rng(1), beam, 10, "dio", spec)


def test_offline_load_table_drops_last_row_when_file_has_no_trailing_whitespace(tmp_path):
    f = tmp_path / "table.tbl"
    f.write_text("1 1\n2 2\n3 3\n4 4")
    s = S.BinnedSpectrum.table(f, 1, 10)
    assert np.allclose(s.pdf, [1, 2, 3])
    assert np.allclose(s.edges, [1, 2, 3, 4])
    f.write_text(f.read_text() + "\n")
    s = S.BinnedSpectrum.table(f, 1, 10)
    assert np.allclose(s.pdf, [1, 2, 3, 4])
    assert np.allclose(s.edges, [1, 2, 3, 4, 5])


def test_source_cli_and_transport_keep_metadata_and_stop_identity(tmp_path, monkeypatch):
    import json
    import sys
    stops = tmp_path / "stops.npz"
    table = tmp_path / "spectrum.tbl"
    source = tmp_path / "source.npz"
    result = tmp_path / "plane.npz"
    np.savez(stops, **_stops())
    table.write_text("95 3\n96 2\n97 1\n")
    S.main(["--process", "dio", "--stops", str(stops), "--spectrum-table", str(table),
            "--elow", "95", "--ehi", "98", "--n", "12", "--out", str(source)])
    with np.load(source) as s:
        meta = json.loads(str(s["metadata"]))
        assert meta["elow"] == 95 and meta["ehi"] == 98
        assert len(meta["stops_sha256"]) == 64 and len(meta["spectrum_sha256"]) == 64
        expected_stop = s["stop_idx"].copy()
    monkeypatch.setattr(C, "from_offline", lambda *a, **k: None)
    monkeypatch.setattr(C, "field_maps", lambda *a, **k: None)
    def transport(rng, geom, field, x, u, p, t, **kw):
        n = p.size
        return C.Result(np.full(n, C.PLANE), x, u, np.hypot(p, S.ME) - S.ME, t,
                        np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
    monkeypatch.setattr(C, "transport", transport)
    monkeypatch.setattr(sys, "argv", ["cechain", "--source", str(source), "--offline-root", "unused",
                                     "--maps", "unused.header", "--out", str(result)])
    C.main()
    with np.load(result) as r:
        assert np.array_equal(r["stop_idx"], expected_stop)
        assert len(str(r["source_sha256"])) == 64
        assert str(r["source_metadata"]) == json.dumps(meta)
    with pytest.raises(SystemExit):
        S.main(["--process", "rpc-internal", "--stops", str(stops), "--elow", "50", "--ehi", "139.5",
                "--n", "12", "--out", str(source)])


def test_rpc_resamples_event_then_pion_not_flattened_stop_rows():
    n = 11
    stops = dict(pdg=np.full(n, -211), at_rest=np.ones(n, bool), foil=np.zeros(n, int),
                 xstop=np.arange(n), ystop=np.zeros(n), zstop=np.zeros(n), tstop=np.full(n, 350.),
                 run=np.ones(n, int), subrun=np.zeros(n, int), evt=np.r_[0, np.ones(10, int)])
    a = S.from_stops(np.random.default_rng(1), stops, 20000, "rpc-external", S.rpc_spectrum())
    assert np.mean(a["stop_idx"] == 0) == pytest.approx(0.5, abs=0.02)
    del stops["evt"]
    with pytest.raises(ValueError, match="run/subrun/evt"):
        S.from_stops(np.random.default_rng(1), stops, 10, "rpc-external", S.rpc_spectrum())


def test_malformed_stop_columns_fail_before_sampling():
    stops = _stops()
    spec = S.BinnedSpectrum([95, 100], [1])
    stops["tstop"] = np.ones(2)
    with pytest.raises(ValueError, match="equal-length"):
        S.from_stops(np.random.default_rng(1), stops, 10, "dio", spec)
    stops = _stops()
    stops["at_rest"] = np.ones(3)
    with pytest.raises(ValueError, match="boolean"):
        S.from_stops(np.random.default_rng(1), stops, 10, "dio", spec)


def test_rpc_window_occurrences_preserve_repeated_event_exposure():
    stops = dict(pdg=np.full(3, -211), at_rest=np.ones(3, bool), foil=np.zeros(3, int),
                 xstop=np.arange(3), ystop=np.zeros(3), zstop=np.zeros(3), tstop=np.full(3, 350.),
                 run=np.ones(3, int), subrun=np.zeros(3, int), evt=np.array([1, 1, 2]),
                 rpc_event_occurrence=np.arange(3))
    a = S.from_stops(np.random.default_rng(1), stops, 20000, "rpc-external", S.rpc_spectrum())
    assert np.mean(a["stop_idx"] < 2) == pytest.approx(2 / 3, abs=0.02)
    stops["rpc_event_occurrence"] = np.array([0., 0.5, 2.])
    with pytest.raises(ValueError, match="event occurrence"):
        S.from_stops(np.random.default_rng(1), stops, 10, "rpc-external", S.rpc_spectrum())
