"""beamfastsim.ptgt_deck: a generated G4beamline target group as a device Target, and the rotation convention."""
import numpy as np
import pytest

from beamfastsim import ptgt_chain as C
from beamfastsim import ptgt_deck as K

DECK = """# a generated file (two cores, an end ring, fins)
material WL10 density=18.75 W,0.99 La,0.0085 O,0.0015
tubs core_long outerRadius=3.150000 length=5.000000 material=WL10 color=$Tungsten
tubs end_ring innerRadius=15.000000 outerRadius=17.000000 length=8.000000 material=WL10 color=$Tungsten
box fin_long length=5.000000 height=10.489000 width=1. material=WL10 color=$Tungsten
box fin_top length=216.000000 height=2.510000 width=1. material=WL10 color=$Tungsten
group pTarget material=Vacuum radius=17.0 length=232.0
\tplace core_long rename=+_core_0 z=107.500000
\tplace core_long rename=+_core_1 z=-2.500000
\tplace end_ring rename=+_upstream z=112.000000
\tplace fin_long rename=+_fin_0_0 z=107.500000 x=0.000000 y=8.394500 rotation=Z0
\tplace fin_long rename=+_fin_0_1 z=107.500000 x=8.394500 y=0.000000 rotation=Z90
\tplace fin_top rename=+_fintop_3 z=0 x=-14.895000 y=0.000000 rotation=Z270
endgroup
tubs rod0up innerRadius=0 outerRadius=1.0 length=155.5 material=WL10
place rod0up x=45.3 y=89.7 z=82.4 rotation=X68.5,Y-29.0
"""


def test_group_to_target(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text(DECK)
    t = K.target(str(f), ["G4_W", "WL10"])
    assert t.n_cone == 3 and len(t.bx) == 3 and set(t.mat.tolist()) == {1}
    assert np.allclose(t.zr, [[105.0, 110.0], [-5.0, 0.0], [108.0, 116.0]]) and np.allclose(t.r_in[:, 0], [0, 0, 15])
    assert np.allclose(t.bx[0], [-0.5, 0.5, 8.3945 - 5.2445, 8.3945 + 5.2445, 105.0, 110.0])
    assert np.allclose(t.bx[1], [8.3945 - 5.2445, 8.3945 + 5.2445, -0.5, 0.5, 105.0, 110.0])         # Z90: width <-> height
    assert np.allclose(t.bx[2], [-14.895 - 1.255, -14.895 + 1.255, -0.5, 0.5, -108.0, 108.0])         # Z270
    assert list(t.locate(np.array([[0, 0, 107.0], [0, 0, 100.0], [0, 10.0, 107.0], [0, 16.0, 112.0]]))) == [0, -1, 3, 2]
    with pytest.raises(ValueError, match="not in the bank"):
        K.target(str(f), ["G4_W"])


@pytest.mark.parametrize("bad", ["box fin_long length=5 height=10 width=1 material=WL10\ngroup pTarget\nplace fin_long z=0 rotation=Z45\nendgroup",
                                 "tubs c outerRadius=3 length=5 initialPhi=0 finalPhi=90 material=WL10\ngroup pTarget\nplace c z=0\nendgroup",
                                 "tubs c outerRadius=$R length=5 material=WL10\ngroup pTarget\nplace c z=0\nendgroup",
                                 "tubs c outerRadius=3 length=5 material=WL10\ngroup pTarget\nplace c z=0 rotation=X10\nendgroup"])
def test_refuses_what_it_cannot_represent(tmp_path, bad):
    f = tmp_path / "b.txt"
    f.write_text(bad + "\n")
    with pytest.raises(ValueError):
        K.target(str(f), ["WL10"])


def test_rotation_matches_the_placement_convention():
    p = C.placement(tilt_x_deg=0.5)
    assert np.allclose(K.rotation("X0.5,Y14"), p["rot"])
    assert np.allclose(K.rotation("Y14"), C.placement()["rot"])
    rz = K.rotation("Z-30,Y14")
    assert np.allclose(rz[:, 2], C.placement()["rot"][:, 2])          # a z rotation first keeps the target axis


def test_ral_target_matches_the_deck_arithmetic():
    """The deck's own formulas (Proton_Target_RAL.txt): hub polycones, ball and spoke end positions, rod lengths."""
    p = C.placement()
    t = K.ral_target(0, **p)
    assert (t.n_cone, t.n_sphere, len(t.cy)) == (5, 6, 6) and t.z_beam == 80.0
    assert t.z_hi == pytest.approx(80 + 4 + 16 * np.cos(np.pi / 4)) and t.z_lo == -t.z_hi
    assert np.allclose(t.zr[1], [72, 84]) and np.allclose(t.r_out[2], [4.15, 4.15 + 16 * np.sin(np.pi / 4)])
    ball = t.sp[0]
    yoff = 4.15 + 9 * np.sin(np.pi / 4) - 2.25 * np.cos(np.pi / 4)
    assert np.allclose(ball[:2], [-np.sin(np.deg2rad(105)) * yoff, np.cos(np.deg2rad(105)) * yoff])
    deg, th = np.pi / 180, 14 * np.pi / 180
    for i, (sgn, att) in enumerate([(1, 105), (1, 225), (1, 345), (-1, 15), (-1, 135), (-1, 255)]):
        ang = att * deg
        zpre = sgn * (84 + 9 * np.cos(np.pi / 4) - 0.1 * np.sin(np.pi / 4))
        ypre = 4.15 + 9 * np.sin(np.pi / 4) + 0.1 * np.cos(np.pi / 4)
        xpr, ypr = -np.sin(ang) * ypre, np.cos(ang) * ypre
        hub = np.array([np.cos(th) * xpr + np.sin(th) * zpre, ypr, -np.sin(th) * xpr + np.cos(th) * zpre])
        yring = 176 - 35 * np.sin(65 * deg)
        ring = np.array([-np.sin(ang) * yring, np.cos(ang) * yring, -sgn * (9.525 + 35 * np.cos(65 * deg))])
        a, b = p["rot"] @ t.cy[i, 0:3], p["rot"] @ t.cy[i, 3:6]      # the deck's (target-centred) frame
        assert np.linalg.norm(b - a) == pytest.approx(np.linalg.norm(ring - hub) - 1.0)    # rodlength
        assert np.allclose(0.5 * (a + b), 0.5 * (ring + hub))                               # rodcenter
        e, g = (b - a) / np.linalg.norm(b - a), (ring - hub) / np.linalg.norm(ring - hub)
        assert np.allclose(e, g, atol=1e-12)                                                # along hub -> ring


STICKMAN = """// a Stickman block (the v1.0 values; 5 plates)
int targetPS_numberOfPlates = 5;
vector<string> targetPS_plateMaterial = { "Inconel718", "Inconel718", "Inconel718", "Inconel718", "Inconel718" };
vector<double> targetPS_rOut = { 3.15, 3.15, 3.15, 3.15, 3.15 }; // core radius
int targetPS_nStickmanFins = 3;
vector<double> targetPS_plateFinAngles = {285.,165.,45.};
double targetPS_plateFinOuterRadius    = 18.075;
double targetPS_plateFinWidth          = 2.0;
double targetPS_plateCenterToLugCenter = 19.6;
double targetPS_plateLugInnerRadius    = 1.525;
double targetPS_plateLugOuterRadius    = 3.0;
vector<double> targetPS_plateThickness = { 5.0, 5.0, 5.0, 5.0, 5.0 };
vector<double> targetPS_plateLugThickness = { 6.0, 6.0, 6.0, 6.0, 6.0 };
bool targetPS_addFilletToPlateCore = true;
double targetPS_spacerHalfLength  = 1.5;
double targetPS_supportRingLength          = 8.1;
double targetPS_supportRingInnerRadius     = 15.0;
double targetPS_supportRingOuterRadius     = 17.0;
string targetPS_supportRingMaterial        = "Inconel718";
"""


def test_stickman_target(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text(STICKMAN)
    cfg = K.read_simpleconfig(str(f))
    assert cfg["targetPS_numberOfPlates"] == 5 and cfg["targetPS_plateFinAngles"] == [285.0, 165.0, 45.0]
    assert cfg["targetPS_addFilletToPlateCore"] is True and cfg["targetPS_supportRingMaterial"] == "Inconel718"
    t = K.stickman_target(cfg, mat=1)
    half = 8.1 + 3.0 + 15.0                                          # ring + 2 spacer halves + 5 x 6 / 2
    assert (t.z_lo, t.z_hi) == (-half, half) and t.n_cone == 7 and t.n_box == 21 and len(t.cy) == 3
    assert set(t.mat.tolist()) == {1}
    z0 = -half + 11.1                                                 # first plate's -z face
    a45 = np.deg2rad(45.0)
    pts = np.array([[0, 0, z0 + 2.5], [0, 0, z0 + 5.5], [10 * np.cos(a45), 10 * np.sin(a45), z0 + 2.5],
                    [10 * np.cos(a45), 10 * np.sin(a45), z0 + 5.5], [19.6 * np.cos(a45), 19.6 * np.sin(a45), 0.0],
                    [16.0, 0, -half + 4.0], [16.0, 0, 0.0], [10.0, 0, z0 + 2.5]])
    seg = t.locate(pts)
    assert seg[0] == 0 and seg[1] == -1 and seg[2] >= 7 and seg[3] == -1          # core; gap; fin; gap beside the fin
    assert seg[4] >= 7 + 21 and seg[5] == 5 and seg[6] == -1 and seg[7] == -1     # lug cylinder; ring; no ring mid-way
    e, n = D_slice(t)
    assert n == 7 and np.allclose(e[:8], [-half, z0, z0 + 6, z0 + 12, z0 + 18, z0 + 24, z0 + 30, half])
    r, th, lug = K.stickman_6d((3.15, 3.15, 3.15), (5.0, 5.0, 5.0), n=5)
    assert np.allclose(r, 3.15) and np.allclose(th, 5.0) and np.allclose(lug, [5.0, 5.75, 5.75, 5.75, 5.0])
    r, th, _ = K.stickman_6d((2.0, 3.0, 4.5), (3.0, 8.0, 4.0), n=5)
    assert np.allclose([r[0], r[2], r[4], th[0], th[2], th[4]], [2.0, 3.0, 4.5, 3.0, 8.0, 4.0])
    t6 = K.stickman_target(cfg, mat=1, r_out=r, thickness=th, lug=K.stickman_6d((2.0, 3.0, 4.5), (3.0, 8.0, 4.0), n=5)[2])
    assert t6.z_hi == pytest.approx(8.1 + 3.0 + 0.5 * (3.0 + th[1] + th[2] + th[3] + 0.75 * 3 + 4.0))
    with pytest.raises(ValueError):
        K.stickman_target(cfg, thickness=np.full(5, 7.0))            # lug 6 < thickness 7


def D_slice(t):
    from beamfastsim import ptgt_device as D
    return D.slice_edges(t)
