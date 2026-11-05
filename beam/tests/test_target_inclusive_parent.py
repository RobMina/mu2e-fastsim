"""Every emitted primary must agree with exact parent exposure, not merely a total count."""
import numpy as np
import pytest

from beamfastsim.target_inclusive_parent import verify


def test_inclusive_dio_counts_complete_multi_stop_window():
    parent = dict(xstop=np.array([10., 20., 21.]), ystop=np.zeros(3), zstop=np.zeros(3),
                  event_index=np.array([0, 1, 1]), event_multiplicity=np.array([1, 2]))
    ref = dict(creation_code=166, gencount=np.array([2]), x=np.array([20., 21., 10.]), y=np.zeros(3),
               z=np.zeros(3), evt=np.array([1, 1, 2]), ifile=np.zeros(3, int))
    report = verify(parent, ref, 2, 166)
    assert report['exact_parent_exposure_reconciled'] and report['generated_electrons'] == 3
    assert report['window']['inferred_first_parent_event_index'] == 1
    shortened = {k: v[:2] if isinstance(v, np.ndarray) and v.shape == (3,) else v for k, v in ref.items()}
    with pytest.raises(ValueError, match='every declared event'):
        verify(parent, shortened, 2, 166)


def test_inclusive_rpc_repeated_stop_phase_and_time():
    parent = dict(xstop=np.array([10., 20., 10.]), ystop=np.zeros(3), zstop=np.zeros(3),
                  tstop=np.array([350., 360., 350.]), event_index=np.arange(3), event_multiplicity=np.ones(3, int))
    ref = dict(creation_code=179, gencount=np.array([4]), x=np.array([20., 10., 10., 20.]),
               y=np.zeros(4), z=np.zeros(4), time=np.array([360., 350., 350., 360.]),
               evt=np.arange(1, 5), ifile=np.zeros(4, int))
    report = verify(parent, ref, 4, 179)
    assert report['generated_electrons'] == 4
    ref['time'][1] = 351.
    with pytest.raises(ValueError, match='birth times'):
        verify(parent, ref, 4, 179)
