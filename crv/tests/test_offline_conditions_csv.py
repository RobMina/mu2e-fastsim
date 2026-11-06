"""Offline CRV text conditions -> runchain CSVs (scripts/offline_conditions_csv.py)."""
import importlib.util
from pathlib import Path

import numpy as np

from crvfastsim import chain

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("offline_conditions_csv", ROOT / "scripts/offline_conditions_csv.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_convert_and_load(tmp_path):
    (tmp_path / "calib.txt").write_text("TABLE CRVTime\n#channel,timeOffset\n0,1.5\n1,-0.5\nTABLE CRVSiPM\n"
                                        "#channel,pedestal,calibPulseHeight,calibPulseArea\n0,100.0,11.4,394.6\n"
                                        "1,101.0,11.0,390.0\n")
    (tmp_path / "status.txt").write_text("TABLE CRVBadChan\n#channel,status\n1,2\n")
    (tmp_path / "py.txt").write_text("TABLE CRVPhoton\n#channel,photonYieldDeviation\n0,0.01\n1,-0.02\n")
    (tmp_path / "ord.txt").write_text("Channel\tROC\tFEB\tFEBchannel\n0\t1\t1\t0\n1\t1\t3\t0\n")
    m = _mod()
    c, d = m.convert(*(str(tmp_path / f) for f in ("calib.txt", "status.txt", "py.txt", "ord.txt")))
    np.savetxt(tmp_path / "c.csv", c, delimiter=",")
    np.savetxt(tmp_path / "d.csv", d, delimiter=",")
    cond = chain.Conditions.from_csv(str(tmp_path / "c.csv"), str(tmp_path / "d.csv"))
    assert cond.pedestal.tolist() == [100.0, 101.0] and cond.pulsearea.tolist() == [394.6, 390.0]
    assert cond.pulseheight.tolist() == [11.4, 11.0] and cond.timeoffset.tolist() == [1.5, -0.5]
    assert cond.status.tolist() == [0, 2] and cond.feb.tolist() == [1, 3]
    assert cond.deviation.tolist() == [0.01, -0.02]
