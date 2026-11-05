"""Run 1A V-series paired statistics and the runchain --save-reco layout."""
import numpy as np
import pytest

from crvfastsim.analyses import runchain
from crvfastsim.validation import run1a_vseries as V


def test_paired_ratio_identical_and_shifted():
    r = np.array([100., 120., 90., 110.])
    same = V.paired_ratio(r, r)
    assert same["ratio"] == 1.0 and same["err"] == 0.0
    s = V.paired_ratio(1.1 * r, r)
    assert s["ratio"] == pytest.approx(1.1) and s["err"] == pytest.approx(0.0, abs=1e-12)
    n = V.paired_ratio(r + np.array([3., -3., 3., -3.]), r)
    assert n["ratio"] == pytest.approx(1.0) and n["err"] > 0


def test_paired_shape_pulls_and_chi2():
    rng = np.random.default_rng(1)
    R = rng.poisson(50, size=(10, 6)).astype(float)
    F = R + rng.normal(0, 3, size=R.shape)
    res = V.paired_shape(F, R)
    assert res["ndf"] == 6 and np.isfinite(res["chi2_ndf"])
    big = V.paired_shape(R + 10 + rng.normal(0, 1, size=R.shape), R)
    assert big["chi2_ndf"] > 100 and min(big["pull"]) > 5
    zero = V.paired_shape(R, R)
    assert zero["ndf"] == 0 and np.isnan(zero["chi2_ndf"])


def test_save_reco_round_trip(tmp_path):
    out = [{"cl_event": np.array([5, 5, 7]), "cl_type": np.array([1, 1, 1]), "cl_pe": np.array([10., 20., 30.]),
            "cl_start": np.zeros(3), "cl_end": np.ones(3), "cl_avgtime": np.ones(3),
            "mem_cl": np.array([0, 0, 1, 2]), "mem_idx": np.array([0, 1, 2, 0]),
            "pulses": dict(ev=np.array([5, 5, 5, 7]), idx=np.array([0, 1, 2, 0]), bar=np.array([0, 1, 2, 3]),
                           sipm=np.zeros(4, int), pe=np.array([5., 6., 7., 8.]), time=np.zeros(4))}]
    p = tmp_path / "r.npz"
    runchain._save_reco(p, out, 100, "run1a_v01")
    cl, mem, pulses = V._reco_frames_fast(p)
    assert cl.pe.tolist() == [10., 20., 30.] and mem.ev.tolist() == [5, 5, 5, 7]
    m = mem.merge(pulses, on=["ev", "idx"])
    assert m.bar.tolist() == [0, 1, 2, 3]
