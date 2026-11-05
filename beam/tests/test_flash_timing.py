"""Stored float precision is not a certificate for microbunch phase."""
import numpy as np
import pytest
from beamfastsim.flash_timing import timing_summary


def _columns(t):
    t = np.asarray(t)
    return dict(time=t, run=np.ones(t.size), subrun=np.zeros(t.size), evt=np.arange(t.size),
                trk=np.ones(t.size), pdg=np.full(t.size, 11), code=np.full(t.size, 17))


def test_precision_tail_is_reported_without_time_cut_or_fold():
    t = np.array([52.0, 1700.0, 1e9, 1.03314901434368e16], np.float32)
    out = timing_summary(_columns(t))
    assert out["rows"] == 4
    assert out["phase_unresolved_rows"] == 2
    assert out["rows_above_threshold"] == [3, 2, 1, 1, 1]
    assert out["largest_rows"]["time_ns"][0] == float(t[-1])
    assert out["largest_rows"]["spacing_ns"][0] == 1073741824.0
    assert out["storage_dtype"] == "float32"
    assert np.array_equal(t, np.array([52.0, 1700.0, 1e9, 1.03314901434368e16], np.float32))


def test_empty_and_double_precision_times():
    assert timing_summary(_columns(np.array([], np.float32)))["time_quantiles_ns"] is None
    assert timing_summary(_columns(np.array([1e9], np.float64)))["phase_unresolved_rows"] == 0


@pytest.mark.parametrize("times", [[np.nan], [np.inf]])
def test_invalid_times_reject(times):
    with pytest.raises(ValueError, match="finite"):
        timing_summary(_columns(np.array(times)))


def test_invalid_identity_and_resolution_reject():
    a = _columns(np.array([0.0]))
    a["evt"][0] = 2**24
    with pytest.raises(ValueError, match="exact"):
        timing_summary(a)
    with pytest.raises(ValueError, match="resolution"):
        timing_summary(_columns(np.array([0.0])), resolution_ns=0)
