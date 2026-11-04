"""Gun lists from the Geant4 start table."""
import numpy as np

from beamfastsim import g4ref


def test_gunfile(tmp_path):
    g = dict(x=np.array([-3904.0, -3900.5]), y=np.array([1.0, -2.0]), z=np.array([4179.0, 4179.0]),
             px=np.array([1.0, 2.0]), py=np.array([0.5, -0.5]), pz=np.array([40.0, 30.0]),
             time=np.array([100.0, 120.0]), pdg=np.array([13, -13]))
    p = tmp_path / "gun.txt"
    assert g4ref.write_gunfile(g, p, charge=1) == 2
    rows = [line.split() for line in p.read_text().splitlines()]
    assert all(len(r) == 12 and r[7] == "-13" for r in rows) and float(rows[1][3]) == 2.0
    g4ref.write_gunfile(g, p)
    assert [line.split()[7] for line in p.read_text().splitlines()] == ["13", "-13"]


