"""Unit tests: fast stand-in of the approximate CE sensitivity (beamfastsim.cesob)."""
import math

import numpy as np
import pytest

from beamfastsim import cesob as S

AXIS = -3904.0
CALO = {"mu2e.solenoidOffset": 3904.0, "calorimeter.caloMotherZ0": 11842.0, "calorimeter.caloMotherZ1": 13220.0,
        "calorimeter.vdThickness": 0.02, "calorimeter.crystalZLength": 200.0, "calorimeter.crystalCapZLength": 2.0,
        "calorimeter.FPCarbonZLength": 3.0, "calorimeter.FPFoamZLength": 21.75, "calorimeter.pipeRadius": 4.75,
        "calorimeter.FPCoolPipeRadius": 16.85, "calorimeter.BPHoleZLength": 15.0, "calorimeter.FEEZLength": 58.0,
        "calorimeter.FEEBoxThickness": 0.7, "calorimeter.BPPipeZOffset": 43.0, "calorimeter.BPPipeRadiusHigh": 13.45,
        "calorimeter.FEBToDiskZOffset": 1.98, "calorimeter.diskZMotherShift": [0.0, 700.0],
        "calorimeter.diskCrystalRIn": 374.0, "calorimeter.diskCrystalROut": 660.0}


def test_calo_crystal_faces_follow_the_maker_arithmetic():
    c = S.calo_from_config(CALO)
    assert c.axis_x == AXIS
    assert c.z_front == pytest.approx((11882.875, 12582.875))
    assert c.z_back == pytest.approx((12082.875, 12782.875))
    assert (c.r_in, c.r_out) == (374.0, 660.0)


def _helix_starts(rc, R, n=1, cos=0.7, z=8539.9):
    """e- in +1 T (Bz): start on a helix of radius R about the centre at distance rc from the axis."""
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    st = math.sqrt(1 - cos * cos)
    # in +Bz an e- turns counterclockwise seen from +z: centre to the left of the transverse direction
    c = np.stack([rc * np.ones(n), np.zeros(n)], 1)
    pos = c + R * np.stack([np.cos(ph), np.sin(ph)], 1)
    ut = np.stack([-np.sin(ph), np.cos(ph)], 1)
    x = np.stack([AXIS + pos[:, 0], pos[:, 1], np.full(n, z)], 1)
    u = np.concatenate([st * ut, np.full((n, 1), cos)], 1)
    return x, u, R / st


def test_calo_accept_in_a_uniform_field_hits_only_helices_that_reach_the_crystal_annulus():
    c = S.calo_from_config(CALO)
    T = 103.0
    p = math.sqrt(T * (T + 2 * 0.51099895))
    st = math.sqrt(1 - 0.7 ** 2)
    R = st * p / (0.299792458 * 1.0)                       # helix radius in 1 T
    for rc, want in ((0.0, False), (520.0 - R, True)):     # r_max = R (inside the hole) or 520 mm
        x, u, _ = _helix_starts(rc, R, n=8)
        hit = S.calo_accept([0.0, 0.0, 1.0], c, x, u, np.full(8, T))
        assert hit.all() == want and hit.any() == want


def test_gaussian_kernel_and_density_kernel_keep_their_totals():
    m, first = S.gaussian_kernel(0.2, 0.2)
    assert first == -5 and m.sum() == pytest.approx(1.0) and np.allclose(m, m[::-1])
    vals = np.zeros(500)
    vals[480:495] = 1.0
    mass, lo = S.density_kernel(vals, -100.0, 0.2, 0.01)
    assert mass.sum() == pytest.approx(vals.sum() * 0.2)
    assert lo * 0.01 == pytest.approx(-4.09)            # linear ramp from the empty bin centre below


def test_smear_drops_content_off_the_axis_and_regrid_sums_by_centre():
    v = np.zeros(10)
    v[0] = 1.0
    out = S.smear(v, (np.array([0.5, 0.5]), -1))
    assert out.sum() == pytest.approx(0.5) and out[0] == 0.5
    r = S.regrid(np.ones(20), 0.0, 0.01, 2, 0.0, 0.1)
    assert r.tolist() == [10.0, 10.0]


def test_scan_box_finds_the_window_of_a_peak_over_flat_background():
    centers = 50.0 + (np.arange(300) + 0.5) * 0.2
    sig = np.where(np.abs(centers - 100.0) < 0.5, 10.0, 0.0)
    m = S.scan_box(sig, np.zeros(300), np.full(300, 1.0), centers)
    assert m["signal_rate"] == pytest.approx(40.0)          # centres 99.7 ... 100.3
    assert m["signal_box_low_mev"] == pytest.approx(99.7) and m["signal_box_high_mev"] == pytest.approx(100.3)
    assert m["sensitivity"] == pytest.approx(40.0 / math.sqrt(4.0))


def test_sensitivity_scales_with_the_signal_efficiency_and_reports_the_factors():
    rng = np.random.default_rng(3)
    e = S.E_START - rng.exponential(0.5, 20000)
    h = S.histograms(e, S.E_START, np.ones(e.size, bool))
    assert h["entries"] == 20000 and h["front"].sum() == 20000
    dio = np.zeros(S.DIO_BINS[0])
    dio[:9000] = 1.0 / 90.0
    a = S.sob_metrics(h, 10000, 20000, 0.08, dio)
    b = S.sob_metrics(h, 10000, 20000, 0.04, dio)
    assert a["ce_acceptance"] == 0.5 and a["stops_per_pot"] == pytest.approx(0.08 * S.UPSTREAM_EFF)
    assert a["box_eff_per_stop"] == pytest.approx(a["box_frac_selected"] * 0.5)
    assert a["sensitivity"] > b["sensitivity"] > 0
    assert a["signal_box_high_mev"] <= 105.2 and a["signal_box_low_mev"] >= 100.0


def test_histograms_skip_events_without_an_entry_energy():
    e = np.array([104.0, np.nan, 103.0])
    h = S.histograms(e, S.E_START, np.array([True, True, False]))
    assert h["entries"] == 1 and h["resp"].sum() == 1
