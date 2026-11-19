"""Track-tagged EX labels and the censored response likelihood."""

import json
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot
from scipy.special import ndtr

from crvfastsim.geometry import CRVGeometry
from crvfastsim.objective import ex_observations as obs
from crvfastsim.objective import ex_response as response
from crvfastsim.reco import vr4eff


def test_export_tracks_and_exact_sensor_labels(tmp_path, monkeypatch):
    geom = CRVGeometry(
        ["EX", "T1"], np.array([0, 0, 1]), np.zeros(3, int), np.array([2, 2, 0]), np.arange(3),
        np.array([[-3904, 100, 24171], [-3904, 100, 24231], [-3904, 100, 24171]]),
        np.tile([500, 10, 26], (3, 1)), np.tile([1, 2, 0], (3, 1)),
    )
    tk = dict(x=np.array([200, -100, 0, 0]), y=np.array([80, 120, 100, 100]), z=np.array([0, 0, 21, 0]),
              dx=np.array([0, 0, 0, 1]), dy=np.array([2, 1, 1, 0]), dz=np.array([0, 0, 0.8, 0]),
              time=np.full(4, 100.))
    line_module = SimpleNamespace(DEFAULT_CUTS={}, lines=lambda *args, **kwargs: tk)
    monkeypatch.setitem(sys.modules, "crvextrap", SimpleNamespace(trackline=line_module))
    f = tmp_path / "events.root"
    pulse_fields = {"barId": [[0, 0, 0, 0, 0, 1], [0, 0, 0], [], []],
                    "SiPMId": [[0, 1, 1, 2, 3, 1], [0, 0, 1], [], []],
                    "sequenceIndex": [[0, 0, 1, 0, 0, 0], [0, 0, 0], [], []],
                    "PEs": [[8., 10., 10., 15., 15., 10.], [8., 9., 7.], [], []],
                    "time": [[109., 125., 107., 109., 107., 107.], [106., 107., 108.], [], []]}
    fields = {f"evtinfo/{k}": np.array(v) for k, v in
              dict(run=[19]*4, subrun=[2]*4, event=[10, 11, 12, 13]).items()}
    fields.update({f"crvpulses/crvpulses.{k}": ak.Array(v) for k, v in pulse_fields.items()})
    with uproot.recreate(f) as root:
        root.mktree("EventNtuple/ntuple", fields)
    rows = vr4eff.track_pulse_rows(f, per_sensor=True)

    legacy_file = tmp_path / "legacy.npz"
    vr4eff.prep(f, legacy_file)
    with np.load(legacy_file) as legacy:
        assert set(legacy.files) == set(rows) - {"run", "p_sipm"}
        np.testing.assert_array_equal(legacy["p_ev"], [0]*6 + [1]*3)
        for key in legacy.files:
            np.testing.assert_array_equal(legacy[key], rows[key])

    present, status = np.ones(12, bool), np.zeros(12, int)
    present[2], status[3] = False, 2
    settings = dict(run=19, margin_mm=2., threshold_pe=8., time_window_ns=2., pe_per_mm=1.,
                    attenuation_mm=2000., speed_mm_ns=200., delay_ns=5.)
    r = obs.observations(geom, present, status, rows, **settings)
    np.testing.assert_array_equal(r["event"], [10]*4 + [11]*4)
    np.testing.assert_array_equal(r["layer"], [2]*8)
    np.testing.assert_array_equal(r["channel"], [0, 1, 2, 3]*2)
    np.testing.assert_allclose(r["track_time_ns"], np.repeat([100 + 20/299.792458, 100 - 20/299.792458], 4))
    np.testing.assert_allclose(r["path_mm"], 20.)
    np.testing.assert_allclose(r["distance_mm"], [700, 300, 700, 300, 400, 600, 400, 600])
    np.testing.assert_allclose(r["nominal_log_pe"], np.log(20) - r["distance_mm"]/2000)
    np.testing.assert_allclose(r["nominal_time_ns"], r["track_time_ns"] + 5 + r["distance_mm"]/200)
    np.testing.assert_array_equal(r["ambiguous"], [False]*4 + [True, False, False, False])
    np.testing.assert_array_equal(r["present"], [True, True, False, True]*2)
    np.testing.assert_array_equal(r["status"], [0, 0, 0, 2]*2)
    np.testing.assert_array_equal(r["eligible"], [True, True, False, False, False, True, False, False])
    np.testing.assert_array_equal(r["seen"], [True] + [False]*7)
    np.testing.assert_array_equal(r["pe"], [8.] + [np.nan]*7)
    assert r["time_ns"][0] == 109.
    with pytest.raises(ValueError, match="only the specified run"):
        obs.observations(geom, present, status, rows, **dict(settings, run=20))


@pytest.fixture
def crossings():
    return dict(channel=np.array([2, 5, 2, 5, 2, 5, 9]), eligible=np.array([True]*6 + [False]),
                seen=np.array([True, True, False, False, True, True, False]),
                distance_mm=np.array([110, 447, 309, 500, 894, 973, np.nan]),
                nominal_log_pe=np.log([11, 14, 11, 18, 19, 20, np.nan]),
                nominal_time_ns=np.array([104., 106., 109., 110., 110., 110., np.nan]),
                threshold_pe=np.array([8., 8., 8., 10., 12., 12., np.nan]),
                time_half_width_ns=np.array([3., 12., 1.8, 7., 4., 4., np.nan]),
                pe=np.array([10., 20., np.nan, np.nan, 19., 18., np.nan]),
                time_ns=np.array([103., 111., np.nan, np.nan, 112., 114., np.nan]))


def test_marked_likelihood_and_gradient(crossings):
    _, x = response._design(crossings, None, 0.31, 1.8, np.float64)
    par = np.array([0.3, -0.5, 0.1, 0.7, -0.12, -0.31])
    terms, grad = response._nll(par, x)
    a, lo, hi, mt = response._model(par, x)
    _, _, thresh, _, seen, zpe, zt = x
    prob = ndtr(-a) * (ndtr(hi) - ndtr(lo))
    np.testing.assert_allclose(terms[~seen], -np.log1p(-prob[~seen]), rtol=1e-12)
    np.testing.assert_allclose(terms[seen],
                               (0.5 * ((zpe + a - thresh)**2 + (zt - mt)**2) + np.log(2*np.pi))[seen])
    step = np.eye(len(par)) * 1e-5
    fd = [(response._nll(par + dx, x)[0].sum() - response._nll(par - dx, x)[0].sum()) / 2e-5 for dx in step]
    np.testing.assert_allclose(grad, fd, rtol=1e-7, atol=1e-8)

    # A rare miss must remain finite even when the pass probability rounds to one.
    rare = (np.array([0]), np.array([1.]), np.array([-40.]), np.array([39.]), np.array([False]),
            np.array([0.]), np.array([0.]))
    loss, grad = response._nll(np.zeros(4), rare)
    assert 700 < loss[0] < 800 and np.isfinite(grad).all()


def test_fit_rejects_unsupported_response_and_scores_eligible_only(crossings):
    widths = dict(sigma_log_pe=0.31, sigma_time_ns=1.8)
    fit = response.fit(crossings, **widths)
    assert fit["channel"] == [2, 5]
    assert fit["eligible"] == 6 and fit["seen"] == 4 and fit["candidate_only"]
    assert np.mean(fit["log_gain"]) == pytest.approx(0.)
    assert np.mean(fit["time_offset_ns"]) == pytest.approx(0.)
    for dtype in (np.float32, np.float64):
        prediction = response.predict(fit, crossings, dtype=dtype)
        for result in prediction.values():
            assert result.shape == (6,) and result.dtype == dtype and np.isfinite(result).all()

    with pytest.raises(ValueError, match="needs a recorded pulse"):
        response.fit(dict(crossings, seen=crossings["seen"] & (crossings["channel"] == 2)), **widths)
    with pytest.raises(ValueError, match="within-channel"):
        response.fit(dict(crossings, distance_mm=crossings["channel"]*100), **widths)
    with pytest.raises(ValueError, match="frozen selection"):
        response.fit(dict(crossings, threshold_pe=np.full(7, 1000.)), **widths)
    with pytest.raises(ValueError, match="without a fitted response"):
        response.predict(dict(fit, channel=[1, 2]), crossings)


def test_candidate_cli_holdout(tmp_path, crossings):
    paths = [tmp_path / f"rows{run}.npz" for run in (1, 2)]
    for run, path in enumerate(paths, 1):
        np.savez(path, **crossings, run=np.full(7, run))
    out = tmp_path / "candidate.json"
    args = [str(paths[0]), str(out), "--sigma-log-pe", "0.31", "--sigma-time-ns", "1.8", "--holdout"]
    with pytest.raises(ValueError, match="separate runs"):
        response.main(args + [str(paths[0])])
    assert not out.exists()
    response.main(args + [str(paths[1])])
    saved = json.loads(out.read_text())
    assert saved["fit_runs"] == [1] and saved["holdout"]["runs"] == [2]
    assert len(saved["observations_sha256"]) == 64
