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
