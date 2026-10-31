"""Outcome-independent scan selection and paired-bin statistics, no physics runs."""
import numpy as np
import pytest
from beamfastsim import lossscan as ls


def test_selection_independent_deterministic_without_replacement():
    events=np.arange(1,1001)
    a=ls.selection(events,100,17)
    np.testing.assert_array_equal(a,ls.selection(events,100,17))
    assert len(np.unique(a))==100 and np.all(np.diff(a)>0)
    assert not np.array_equal(a,ls.selection(events,100,18))
    for bad in (1,1001):
        with pytest.raises(ValueError):
            ls.selection(events,bad,1)


def test_paired_bins_keep_all_entries_and_empty_bins():
    e=np.array([1.2,1.2,1.2,2.5])
    c=np.array([.6,.6,.6,.9])
    result=ls.compare_bins(e,c,[0.,.2,0.,.2],[0.,0.,.2,.2])
    assert sum(r['n'] for r in result)==4
    first=result[0]
    assert first['n']==3
    assert first['paired_threshold_delta']['mean']==0.
    assert first['paired_threshold_delta']['stderr']==pytest.approx(1/np.sqrt(3))
    assert any(r['n']==0 and r['paired_threshold_delta']['mean'] is None for r in result)


def test_float32_incidence_promoted_before_normalization():
    cosine=np.linspace(.5,1.,1001,dtype=np.float32)
    vectors=ls.entry_directions(cosine)
    np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1.,atol=2e-16,rtol=0)
    for value in (np.nan,.49,1.01):
        with pytest.raises(ValueError):
            ls.entry_directions([value])
