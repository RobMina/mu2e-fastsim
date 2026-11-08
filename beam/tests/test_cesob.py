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
        "calorimeter.diskCrystalRIn": 374.0, "calorimeter.diskCrystalROut": 660.0,
        "calorimeter.crystalXYLength": 34.0, "calorimeter.wrapperThickness": 0.15,
        "calorimeter.diskInAlRingRIn": 336.0, "calorimeter.diskInAlRingZLength": 14.0,
        "calorimeter.diskInCFRingRIn": 356.0, "calorimeter.diskInCFRingROut": 360.0}


def test_calo_crystal_faces_follow_the_maker_arithmetic():
    c = S.calo_from_config(CALO)
    assert c.axis_x == AXIS
    assert c.z_front == pytest.approx((11882.875, 12582.875))
    assert c.z_back == pytest.approx((12082.875, 12782.875))
    assert (c.r_in, c.r_out) == (374.0, 660.0)


def test_crystal_grid_has_the_offline_674_crystals_per_disk():
    c = S.calo_from_config(CALO)
    assert c.grid.sum() == 674 and c.cell == pytest.approx(34.3)
    # Offline crystalPos.txt rows: (-188.65, 377.3) and (120.05, 377.3) are crystals, (0, 377.3) is not
    assert S.in_crystal(c, np.array([-188.65, 120.05, 0.0]), np.full(3, 377.3)).tolist() == [True, True, False]
    assert not S.in_crystal(c, np.array([-188.65 + 17.1]), np.array([377.3]))[0]       # in the wrapper


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


def test_calo_accept_in_a_uniform_field_hits_only_helices_that_reach_the_crystals():
    c = S.calo_from_config(CALO)
    T = 103.0
    p = math.sqrt(T * (T + 2 * 0.51099895))
    st = math.sqrt(1 - 0.7 ** 2)
    R = st * p / (0.299792458 * 1.0)                       # helix radius in 1 T
    for rc, want in ((0.0, False), (520.0 - R, True)):     # r_max = R (inside the hole) or 520 mm
        x, u, _ = _helix_starts(rc, R, n=8)
        for annulus in (False, True):
            hit = S.calo_accept([0.0, 0.0, 1.0], c, x, u, np.full(8, T), annulus=annulus)
            assert hit.all() == want and hit.any() == want


def test_calo_wall_geometry_from_the_config():
    c = S.calo_from_config(CALO)
    w = S.calo_wall_from_config(CALO, c)
    assert w.case_lo == pytest.approx((11880.875, 12580.875)) and w.case_hi == c.z_back
    z = np.array([11881.0, 11894.8, 11895.0, 11981.875, 12075.0, 12100.0])
    assert w.in_al(z).tolist() == [True, True, False, True, True, False]
    assert w.in_case(z).tolist() == [True] * 5 + [False]


def test_calo_wall_is_inert_inside_the_bore_and_stops_through_a_thick_wall():
    c = S.calo_from_config(CALO)
    w = S.calo_wall_from_config(CALO, c)
    T = 103.0
    p = math.sqrt(T * (T + 2 * 0.51099895))
    cos = 0.15
    R = math.sqrt(1 - cos ** 2) * p / (0.299792458 * 1.0)
    x, u, _ = _helix_starts(0.0, R, n=8)                   # r_max = R < 336: never meets the wall
    info = {}
    hit = S.calo_accept([0.0, 0.0, 1.0], c, x, u, np.full(8, T), wall=w, info=info)
    assert not hit.any() and info["al"] == info["cf"] == 0
    x, u = _wall_starts(R, cos)
    thick = S.CaloWall(w.case_lo, w.case_hi, 200.0, 356.0, 360.0, 200.0)   # 156 mm of Al along each whole case
    info2 = {}
    hit2 = S.calo_accept([0.0, 0.0, 1.0], c, x, u, np.full(8, T), wall=thick, rng=np.random.default_rng(3),
                         info=info2)
    assert info2["al"] == 8 and not hit2.any()


def _wall_starts(R, cos):
    # 8 e- born in the bore inside disk 0's case (r = 162 mm, moving outward to r_max = 520 mm within the case)
    x, u, _ = _helix_starts(520.0 - R, R, n=2, cos=cos, z=11895.0)
    return np.repeat(x[1:], 8, 0), np.repeat(u[1:], 8, 0)


@pytest.mark.slow   # opt-in calo_model='wall'; the bore and thick-wall cases above run per PR
def test_calo_wall_stock_rings_are_crossed_by_electrons_that_reach_the_crystals_in_vacuum():
    c = S.calo_from_config(CALO)
    w = S.calo_wall_from_config(CALO, c)
    T = 103.0
    p = math.sqrt(T * (T + 2 * 0.51099895))
    cos = 0.15
    R = math.sqrt(1 - cos ** 2) * p / (0.299792458 * 1.0)
    x, u = _wall_starts(R, cos)
    vac = S.calo_accept([0.0, 0.0, 1.0], c, x, u, np.full(8, T))
    info = {}
    S.calo_accept([0.0, 0.0, 1.0], c, x, u, np.full(8, T), wall=w, rng=np.random.default_rng(3), info=info)
    assert vac.all() and info["cf"] == 8


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
