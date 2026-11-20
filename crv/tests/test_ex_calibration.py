"""Association-tagged EX crossings and the censored response likelihood."""

import json

import awkward as ak
import numpy as np
import pytest
import uproot
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from crvfastsim.geometry import CRVGeometry
from crvfastsim.objective import ex_observations as obs
from crvfastsim.objective import ex_response as response
from crvfastsim.objective import ex_synthetic as synthetic


# ---------------------------------------------------------------------------------------------------------------
# Export from the track-CRV association

def _quality(*, nactive, **overrides):
    return dict(dict(status=1, goodfit=1, fitalg=3, nactive=nactive, nplanes=12, ndof=16, chisq=19., fitcon=0.25),
                **overrides)


def _split(name, events):
    """A split EventNtuple record: fields named '<branch>.<leaf>'."""
    arr = ak.Array(events)
    return ak.zip({f"{name}.{k}": arr[k] for k in arr.fields}, depth_limit=1)


def _xing(pos, mom, *, sector=0, inbounds=True, layer=2, bar=0, dbar=127, upred=None, tpred=100.):
    def vector(v):
        return {"fCoordinates": dict(zip(("fX", "fY", "fZ"), map(float, v)))}

    probe_bar, probe_dbar = [-1] * 4, [127] * 4
    probe_upred, probe_tpred = [0.] * 4, [0.] * 4
    probe_bar[layer], probe_dbar[layer] = bar, dbar
    probe_upred[layer] = float(pos[0] if upred is None else upred)
    probe_tpred[layer] = tpred
    return dict(sid=-1, sector=sector, pos=vector(pos), dir=vector(np.asarray(mom) / np.linalg.norm(mom)),
                time=tpred, sigu=5., sigv=3., inbounds=inbounds, dedgeu=100., dedgev=100., best=-1, nlayers=4,
                probe_bar=probe_bar, probe_dbar=probe_dbar, probe_upred=probe_upred, probe_tpred=probe_tpred,
                probe_pe=[0.] * 4)


def _geometry():
    # Two EX bars (layer 2) of half lengths 500 (x, along), 10 (y, thickness), 26 (z, width); one T1 bar.
    return CRVGeometry(
        ["EX", "T1"], np.array([0, 0, 1]), np.zeros(3, int), np.array([2, 2, 0]), np.arange(3),
        np.array([[-3904, 100, 24171], [-3904, 100, 24231], [-3904, 100, 24171]]),
        np.tile([500, 10, 26], (3, 1)), np.tile([1, 2, 0], (3, 1)),
    )


def _write_ntuple(path, *, clustered_only=False):
    q = _quality(nactive=25)
    up = [0, 1, 0]   # detector-frame direction, along the EX thickness (y)
    # Event: (tracks, association crossings per track, matched/probe per track)
    events = [
        ([q], [[_xing([200, 100, 0], up), _xing([0, 2000, 0], up, sector=1, layer=0, bar=2)]]),   # 0: seen + silent
        ([q], [[_xing([-100, 100, 0], up)]]),                          # 1: ambiguous sipm 0, sub-threshold sipm 1
        ([q], [[_xing([0, 100, 21], [0, 1, 0.8])]]),                   # 2: leaves through a side face
        ([q], [[_xing([0, 100, 0], [1, 0, 0])]]),                      # 3: parallel to the broad faces
        ([q], [[_xing([0, 100, 0], up)]]),                             # 4: no pulses: four silent crossings
        ([q], [[_xing([0, 100, 0], up, dbar=1)]]),                     # 5: probe pulse on the neighbouring bar
        ([_quality(nactive=19)], [[_xing([0, 100, 0], up)]]),          # 6: no quality track
        ([q], [[_xing([0, 100, 0], up, inbounds=False)]]),             # 7: a projection only
        ([_quality(nactive=21), _quality(nactive=30)],                 # 8: the second track ranks first
         [[_xing([300, 100, 0], up)], [_xing([-300, 100, 0], up, tpred=90.)]]),
    ]
    trk = _split("trk", [e[0] for e in events])
    crv = _split("trkcrv", [[dict(matched=True, probe=3, nxings=len(x), timeoffset=0.) for x in e[1]]
                            for e in events])
    xings = ak.Array([e[1] for e in events])
    pulses = {"barId": [[0, 0, 0, 0, 0, 1, 2], [0, 0, 0], [], [], [], [], [], [], [0]],
              "SiPMId": [[0, 1, 1, 2, 3, 1, 0], [0, 0, 1], [], [], [], [], [], [], [0]],
              "sequenceIndex": [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0], [], [], [], [], [], [], [0]],
              "PEs": [[8., 10., 10., 15., 15., 10., 30.], [8., 9., 7.], [], [], [], [], [], [], [12.]],
              "time": [[109., 125., 107., 109., 107., 107., 100.], [106., 107., 108.], [], [], [], [], [], [],
                       [96.5]],
              "crvHitIndex": [[0, 0, -1, 0, 0, -1, 1], [0, 0, -1], [], [], [], [], [], [], [0]]}
    if clustered_only:
        pulses["crvHitIndex"] = [[max(h, 0) for h in ev] for ev in pulses["crvHitIndex"]]
    p = ak.Array({f"crvpulses.{k}": ak.Array(v) for k, v in pulses.items()})
    n = len(events)
    evt = ak.Array(dict(run=[19] * n, subrun=[2] * n, event=list(range(10, 10 + n))))
    with uproot.recreate(path) as root:
        root.mkrntuple("EventNtuple/ntuple", ak.zip(dict(evtinfo=evt, trk=trk, trkcrv=crv, trkcrvxings=xings,
                                                         crvpulses=p), depth_limit=1))


SETTINGS = dict(run=19, margin_mm=2., threshold_pe=8., time_window_ns=2., pe_per_mm=1., attenuation_mm=2000.,
                speed_mm_ns=200., delay_ns=5.)


def test_association_export_tracks_and_exact_sensor_labels(tmp_path):
    f = tmp_path / "events.root"
    _write_ntuple(f)
    rows = obs.association_rows(f)
    np.testing.assert_array_equal(rows["trk_selected"], [True] * 6 + [False] + [True] * 2)
    np.testing.assert_array_equal(rows["x_ev"], [0, 0, 1, 2, 3, 4, 5, 7, 8])

    geom = _geometry()
    present, status = np.ones(12, bool), np.zeros(12, int)
    present[2], status[3] = False, 2
    r = obs.observations(geom, present, status, rows, **SETTINGS)
    # Rows: events 0, 1, 4, 5, 8, four SiPMs each; the T1 crossing, the side exit, the parallel line and the
    # projection give none.
    np.testing.assert_array_equal(r["event"], np.repeat([10, 11, 14, 15, 18], 4))
    np.testing.assert_array_equal(r["channel"], [0, 1, 2, 3] * 5)
    np.testing.assert_array_equal(r["layer"], [2] * 20)
    np.testing.assert_allclose(r["path_mm"], 20.)
    np.testing.assert_allclose(r["distance_mm"], [700, 300, 700, 300, 400, 600, 400, 600] + [500] * 8
                               + [200, 800, 200, 800])
    np.testing.assert_allclose(r["track_time_ns"], [100.] * 16 + [90.] * 4)
    np.testing.assert_allclose(r["nominal_log_pe"], np.log(20) - r["distance_mm"] / 2000)
    np.testing.assert_allclose(r["nominal_time_ns"], r["track_time_ns"] + 5 + r["distance_mm"] / 200)
    np.testing.assert_allclose(r["du_line_mm"], 0., atol=1e-9)
    np.testing.assert_array_equal(r["neighbour"], [False] * 12 + [True] * 4 + [False] * 4)
    np.testing.assert_array_equal(r["ambiguous"], [False] * 4 + [True] + [False] * 15)
    eligible = np.array([True, True, False, False] * 5)
    eligible[4] = False         # ambiguous
    eligible[12:16] = False     # neighbour
    np.testing.assert_array_equal(r["eligible"], eligible)
    seen = np.zeros(20, bool)
    seen[[0, 16]] = True
    np.testing.assert_array_equal(r["seen"], seen)
    np.testing.assert_array_equal(r["pe"][seen], [8., 12.])
    np.testing.assert_array_equal(r["time_ns"][seen], [109., 96.5])
    assert str(r["pulse_match"]) == obs.PULSE_MATCH

    kept = obs.observations(geom, present, status, rows, **SETTINGS, exclude_neighbour=False)
    np.testing.assert_array_equal(kept["eligible"][12:16], [True, True, False, False])
    with pytest.raises(ValueError, match="only the specified run"):
        obs.observations(geom, present, status, rows, **dict(SETTINGS, run=20))


def test_association_export_refuses_clustered_only_and_missing_branches(tmp_path):
    f = tmp_path / "clustered.root"
    _write_ntuple(f, clustered_only=True)
    rows = obs.association_rows(f)
    present, status = np.ones(12, bool), np.zeros(12, int)
    with pytest.raises(ValueError, match="clustered pulses only"):
        obs.observations(_geometry(), present, status, rows, **SETTINGS)
    assert obs.observations(_geometry(), present, status, rows, **SETTINGS, allow_clustered_only=True)["seen"].any()

    bare = tmp_path / "bare.root"
    with uproot.recreate(bare) as root:
        root.mkrntuple("EventNtuple/ntuple", ak.zip(dict(
            evtinfo=ak.Array(dict(run=[1], subrun=[0], event=[0])), trk=_split("trk", [[_quality(nactive=25)]])),
            depth_limit=1))
    with pytest.raises(ValueError, match="association branches"):
        obs.association_rows(bare)


# ---------------------------------------------------------------------------------------------------------------
# Likelihood, gradient and Hessian

@pytest.fixture
def crossings():
    return dict(channel=np.array([2, 5, 2, 5, 2, 5, 9]), eligible=np.array([True] * 6 + [False]),
                seen=np.array([True, True, False, False, True, True, False]),
                distance_mm=np.array([110, 447, 309, 500, 894, 973, np.nan]),
                nominal_log_pe=np.log([11, 14, 11, 18, 19, 20, np.nan]),
                nominal_time_ns=np.array([104., 106., 109., 110., 110., 110., np.nan]),
                threshold_pe=np.array([8., 8., 8., 10., 12., 12., np.nan]),
                time_half_width_ns=np.array([3., 12., 1.8, 7., 4., 4., np.nan]),
                pe=np.array([10., 20., np.nan, np.nan, 19., 18., np.nan]),
                time_ns=np.array([103., 111., np.nan, np.nan, 112., 114., np.nan]))


def _theta(nchan, walk, rng, excess=0.):
    return np.r_[0.3, -0.5, np.log(0.31), np.log(1.8), walk, excess, rng.normal(0, 0.2, nchan),
                 rng.normal(0, 1, nchan)]


def test_marked_likelihood_matches_closed_form_without_walk(crossings):
    _, d = response._design(crossings, None)
    theta = _theta(2, 0., np.random.default_rng(1))
    terms, _ = response._nll(theta, d, 2, 2.5)
    eta = response._eta(theta, d, 2)
    mu_q, mu_t, sq, st = eta[:, 0], eta[:, 1], np.exp(eta[:, 2]), np.exp(eta[:, 3])
    assert eta.shape == (6, response.NG)
    a = (d["lthr"] - mu_q) / sq
    prob = ndtr(-a) * (ndtr((d["hi"] - mu_t) / st) - ndtr((d["lo"] - mu_t) / st))
    s = d["seen"]
    np.testing.assert_allclose(terms[~s], -np.log1p(-prob[~s]), rtol=1e-12)
    z, r = (d["lpe"] - mu_q) / sq, (d["tm"] - mu_t) / st
    np.testing.assert_allclose(terms[s], (0.5 * (z**2 + r**2) + np.log(sq * st) + np.log(2 * np.pi))[s], rtol=1e-12)


def test_miss_probability_with_walk_matches_numerical_integral():
    a = np.array([-2.0, 0.3, 1.5, 4.0])
    c = np.array([1.0, -0.5, 2.5, 0.2])
    lo, hi = np.full(4, -4.), np.full(4, 4.)
    lsq, lst, walk = np.full(4, np.log(0.4)), np.full(4, np.log(2.0)), np.array([1.5, -1.5, 2.0, -2.5])
    log_miss = response._miss_logp(a, c, lsq, lst, walk, lo, hi)
    for i in range(4):
        def passes(z):
            m = c[i] - walk[i] * 0.4 * z
            return norm.pdf(z) * (norm.cdf((hi[i] - m) / 2) - norm.cdf((lo[i] - m) / 2))
        miss = 1 - quad(passes, a[i], np.inf, epsabs=1e-14, epsrel=1e-12)[0]
        assert np.log(miss) == pytest.approx(log_miss[i], rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("walk,excess", [(0., 0.), (0.9, 0.), (-0.7, 1.3)])
def test_analytic_gradient_matches_finite_differences(crossings, walk, excess):
    _, d = response._design(crossings, None)
    rng = np.random.default_rng(3)
    for point in range(3):
        theta = _theta(2, walk, rng, excess)
        _, grad = response._nll(theta, d, 2, 2.5)
        f = lambda t: response._nll(t, d, 2, 2.5)[0].sum()
        fd = np.array([(f(theta + 1e-6 * e) - f(theta - 1e-6 * e)) / 2e-6 for e in np.eye(len(theta))])
        np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-7)

    # A rare miss stays finite when the pass probability rounds to one.
    rare = dict(index=np.array([0]), x=np.array([1.]), q0=np.array([0.]), t0=np.array([0.]),
                lthr=np.array([-40 * 0.31]), lo=np.array([-39 * 1.8]), hi=np.array([39 * 1.8]),
                seen=np.array([False]), lpe=np.array([0.]), tm=np.array([0.]))
    theta = np.r_[0., 0., np.log(0.31), np.log(1.8), walk, excess, 0., 0.]
    loss, grad = response._nll(theta, rare, 1, 0.)
    assert np.isfinite(loss).all() and np.isfinite(grad).all()
    if not excess:
        assert 700 < loss[0] < 800


def test_assembled_hessian_matches_finite_differences(crossings):
    _, d = response._design(crossings, None)
    theta = _theta(2, 0.7, np.random.default_rng(5), 0.8)
    free = np.ones(response.NG, bool)
    H = response._hessian(theta, d, 2, 2.5, free)
    g = lambda t: response._nll(t, d, 2, 2.5)[1]
    fd = np.column_stack([(g(theta + 1e-5 * e) - g(theta - 1e-5 * e)) / 2e-5 for e in np.eye(len(theta))])
    np.testing.assert_allclose(H, 0.5 * (fd + fd.T), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------------------------------------------
# Fits, closure and uncertainties

def _fixture(seed=27, n=1200, model="lognormal", **truth_args):
    t = synthetic.truth(np.random.default_rng(seed), 8, **truth_args)
    rows, noise = synthetic.draw(seed + 1, 1, n, t, model=model)
    return t, rows


def test_censored_closure_with_fitted_widths_walk_and_pulls():
    truth, rows = _fixture(walk=0.8, excess_noise=1.0, sigma_log_pe=0.25)
    cal = response.fit(rows)
    assert cal["hessian_positive_definite"] and cal["fitted"] == dict(
        sigma_log_pe=True, sigma_time_ns=True, walk_ns_per_log_pe=True, excess_noise=True)
    pulls = synthetic.pulls(cal, truth)
    assert set(pulls) == {"delta_attenuation_per_mm", "delta_slowness_ns_per_mm", "sigma_log_pe", "sigma_time_ns",
                          "walk_ns_per_log_pe", "excess_noise", "log_scale", "time_zero_ns"}
    assert max(abs(v) for v in pulls.values()) < 3.5, pulls
    assert np.mean(cal["log_gain"]) == pytest.approx(0, abs=1e-12)
    assert np.mean(cal["time_offset_ns"]) == pytest.approx(0, abs=1e-12)
    gain_pull = (np.asarray(cal["log_gain"]) - truth["log_gain"]) / np.asarray(cal["errors"]["log_gain"])
    assert np.all(np.abs(gain_pull) < 4)
    cov = np.asarray(cal["covariance_global"]["matrix"])
    np.testing.assert_allclose(cov, cov.T, atol=1e-18)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    np.testing.assert_allclose(np.sqrt(np.diag(cov))[:2], [cal["errors"]["delta_attenuation_per_mm"],
                                                           cal["errors"]["delta_slowness_ns_per_mm"]])
    # The full covariance agrees with the reported errors; the gauge makes the channel terms sum to zero.
    full = response.covariance(cal, rows)
    nchan = len(cal["channel"])
    g = slice(response.NG + 1, response.NG + 1 + nchan)
    np.testing.assert_allclose(np.sqrt(np.diag(full))[g], cal["errors"]["log_gain"], rtol=1e-6)
    np.testing.assert_allclose(full[g, g].sum(axis=0), 0, atol=1e-12)

    no_walk = response.fit(rows, walk=False)
    assert no_walk["walk_ns_per_log_pe"] == 0 and not no_walk["fitted"]["walk_ns_per_log_pe"]


def test_fixed_widths_bias_the_attenuation_and_fitted_widths_do_not():
    # Validation 2026-11-19 (60 toys): fit widths 0.8x / 1.2x true gave attenuation pulls -3.5 / +2.6.
    truth, rows = _fixture(n=4800)
    fitted = response.fit(rows, walk=False)
    low = response.fit(rows, sigma_log_pe=0.8 * truth["sigma_log_pe"], sigma_time_ns=truth["sigma_time_ns"],
                       fit_widths=False, walk=False)
    high = response.fit(rows, sigma_log_pe=1.2 * truth["sigma_log_pe"], sigma_time_ns=truth["sigma_time_ns"],
                        fit_widths=False, walk=False)
    assert low["excess_noise"] == 0 and not low["fitted"]["excess_noise"]
    pull = lambda cal: synthetic.pulls(cal, truth)["delta_attenuation_per_mm"]
    assert abs(pull(fitted)) < 2.5
    assert pull(low) < -3 and pull(high) > 2


def test_poisson_lognormal_mismatch_closure():
    # Poisson PE counts on a log-normal yield (smear 0.25), the threshold on the count: the fitted log-normal widths
    # absorb most of the mismatch that biased the fixed-width fit (pull -4.1 in the 2026-11-19 validation).
    truth, rows = _fixture(n=4800, model="poisson")
    fitted = response.fit(rows, walk=False)
    fixed = response.fit(rows, sigma_log_pe=truth["sigma_log_pe"], sigma_time_ns=truth["sigma_time_ns"],
                         fit_widths=False, walk=False)
    err = lambda cal: abs(synthetic.errors(cal, truth)["delta_attenuation_per_mm"])
    assert abs(synthetic.pulls(fitted, truth)["delta_attenuation_per_mm"]) < 3
    assert err(fitted) < err(fixed)


def test_fit_rejects_unsupported_response_and_scores_eligible_only(crossings):
    fit = response.fit(crossings, sigma_log_pe=0.31, sigma_time_ns=1.8, fit_widths=False, walk=False)
    assert fit["channel"] == [2, 5]
    assert fit["eligible"] == 6 and fit["seen"] == 4 and fit["candidate_only"]
    assert np.mean(fit["log_gain"]) == pytest.approx(0.)
    assert np.mean(fit["time_offset_ns"]) == pytest.approx(0.)
    for dtype in (np.float32, np.float64):
        prediction = response.predict(fit, crossings, dtype=dtype)
        for key in ("probability", "log_pe", "time_ns"):
            result = prediction[key]
            assert result.shape == (6,) and result.dtype == dtype and np.isfinite(result).all()
    p32, p64 = (response.predict(fit, crossings, dtype=t)["probability"] for t in (np.float32, np.float64))
    np.testing.assert_allclose(p32, p64, atol=1e-5)

    with pytest.raises(ValueError, match="needs a recorded pulse"):
        response.fit(dict(crossings, seen=crossings["seen"] & (crossings["channel"] == 2)))
    with pytest.raises(ValueError, match="within-channel"):
        response.fit(dict(crossings, distance_mm=crossings["channel"] * 100.))
    with pytest.raises(ValueError, match="frozen selection"):
        response.fit(dict(crossings, threshold_pe=np.full(7, 1000.)))
    with pytest.raises(ValueError, match="fixed widths"):
        response.fit(crossings, fit_widths=False)
    with pytest.raises(ValueError, match="without a fitted response"):
        response.predict(dict(fit, channel=[1, 2]), crossings)


# ---------------------------------------------------------------------------------------------------------------
# Positive-only closed form (formerly PR 308's charge_time_calib)

def test_weighted_profile_matches_dense_solve_with_missing_channel_ids():
    rng = np.random.default_rng(42)
    channel = np.repeat([2, 6, 11], [10, 17, 26])
    index = np.searchsorted([2, 6, 11], channel)
    d = rng.uniform(0, 1000, len(channel)) + 100 * index
    base_pe = rng.uniform(30, 70, len(channel))
    base_time = rng.normal(100, 3, len(channel))
    log_pe = 0.02 + np.array([-0.1, 0.02, 0.08])[index] - 0.0002 * d + rng.normal(0, 0.02, len(channel))
    dt = 1.1 + np.array([-0.2, 0.0, 0.2])[index] + 0.0003 * d + rng.normal(0, 0.2, len(channel))
    wc, wt = rng.uniform(0.5, 2, (2, len(channel)))
    got = response.profile_fit(channel, d, base_pe * np.exp(log_pe), base_pe, base_time + dt, base_time,
                               charge_weight=wc, time_weight=wt)
    design = np.column_stack((d, np.eye(3)[index]))
    for y, w, slope, g, local in (
        (log_pe, wc, -got["delta_attenuation_per_mm"], got["log_scale"], got["log_gain"]),
        (dt, wt, got["delta_slowness_ns_per_mm"], got["time_zero_ns"], got["time_offset_ns"]),
    ):
        coef = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
        np.testing.assert_allclose(np.r_[slope, g + np.asarray(local)], coef, atol=1e-12)
        assert abs(sum(local)) < 1e-12
    assert got["channel"] == [2, 6, 11]
    assert 0 < got["errors"]["delta_attenuation_per_mm"] < 1e-4


def test_positive_only_arm_is_the_profile_fit_with_weights():
    truth, rows = _fixture(n=200)
    s = rows["seen"]
    w = np.where(s, 1 + rows["distance_mm"] / 1000, 1.)
    cal = response.fit(dict(rows, w=w), include_missing=False, charge_weight="w")
    ref = response.profile_fit(rows["channel"][s], rows["distance_mm"][s], rows["pe"][s],
                               np.exp(rows["nominal_log_pe"][s]), rows["time_ns"][s], rows["nominal_time_ns"][s],
                               charge_weight=w[s])
    for key in ("delta_attenuation_per_mm", "delta_slowness_ns_per_mm", "log_scale", "time_zero_ns"):
        assert cal[key] == pytest.approx(ref[key], rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(cal["log_gain"], ref["log_gain"], atol=1e-12)
    assert cal["weighted"] == dict(charge=True, time=False)


def test_reject_confounded_distance_and_nonpositive_charge():
    ch = [1, 1, 2, 2]
    with pytest.raises(ValueError, match="within-channel"):
        response.profile_fit(ch, [10, 10, 30, 30], [1] * 4, [1] * 4, [2] * 4, [2] * 4)
    with pytest.raises(ValueError, match="pe.*positive"):
        response.profile_fit(ch, [10, 20, 30, 40], [0, 1, 1, 1], [1] * 4, [2] * 4, [2] * 4)


def test_candidate_cli_holdout_and_covariance(tmp_path, crossings):
    paths = [tmp_path / f"rows{run}.npz" for run in (1, 2)]
    for run, path in enumerate(paths, 1):
        np.savez(path, **crossings, run=np.full(7, run))
    out = tmp_path / "candidate.json"
    args = [str(paths[0]), str(out), "--sigma-log-pe", "0.31", "--sigma-time-ns", "1.8", "--fix-widths",
            "--no-walk", "--holdout"]
    with pytest.raises(ValueError, match="separate runs"):
        response.main(args + [str(paths[0])])
    assert not out.exists()
    cov = tmp_path / "cov.npz"
    response.main(args + [str(paths[1]), "--covariance-out", str(cov)])
    saved = json.loads(out.read_text())
    assert saved["fit_runs"] == [1] and saved["holdout"]["runs"] == [2]
    assert len(saved["observations_sha256"]) == 64
    assert saved["schema"] == response.SCHEMA and saved["errors"]["delta_attenuation_per_mm"] > 0
    with np.load(cov) as c:
        assert c["covariance"].shape == (response.NG + 2 + 2 * 2,) * 2 and len(c["names"]) == response.NG + 6
