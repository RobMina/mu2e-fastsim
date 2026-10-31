"""Exact variance bookkeeping and numerical-statistic contracts, no transport."""
import numpy as np
import pytest

from beamfastsim import lossexplain as le
from test_slabstudy import response


@pytest.mark.parametrize('a,b', [([0.,0.,.2,.4],[0.,.1,.2,.3]), ([.2,.2,.2],[0.,.1,.3]),
                                ([0.,.1],[0.,.3]), ([.2,.4],[.2,.4])])
def test_exact_symmetric_attribution(a,b):
    r=le.explain(a,b)
    assert r['population_variance_delta']==pytest.approx(np.var(a)-np.var(b),abs=1e-15)
    back=le.explain(b,a)
    for name in ('occupancy','conditional_mean_squared','conditional_variance'):
        assert r[name]['delta']==pytest.approx(-back[name]['delta'],abs=1e-15)
        assert r[name]['se']>=0


def test_paired_variance_uses_covariance():
    a=np.array([0.,.1,.2,.4])
    r=le.paired_variance(a,a)
    assert r['population_variance_delta']==0. and r['variance_se']==0.
    with pytest.raises(ValueError):
        le.paired_variance(a,a[:-1])


def test_parameter_influences_finite():
    p,c=le.conditional_parameters([0.,.1,.2,.4])
    assert p[0]==.75 and p[1]==pytest.approx(.7/3)
    assert np.linalg.eigvalsh(c).min()>-1e-14
    with pytest.raises(ValueError):
        le.conditional_parameters([0.,0.])


def test_threshold_only_response_retains_unresolved():
    from beamfastsim import slabdiffusion as sd
    r=response([sd.UNRESOLVED_FLOOR,sd.TRANSMITTED],[.0001,.3])
    data=le.ss.flatten(r)
    np.testing.assert_equal(le.threshold_array(data,2.02),[0.,.3])
    r=response([sd.UNRESOLVED_NUMERICAL,sd.TRANSMITTED],[.0001,.3])
    with pytest.raises(ValueError,match='unidentified'):
        le.threshold_array(le.ss.flatten(r),2.02)


def test_loss_covariance_budget_closes_all_entries():
    from beamfastsim import slabdiffusion as sd
    r=response([sd.UNRESOLVED_FLOOR,sd.TRANSMITTED],[.0001,.3])
    data=le.ss.flatten(r)
    hard=np.array([[0.],[.04]])
    data['ledger__hard_applied']=hard
    data['ledger__soft_applied']=2.02-r.raw_energy-hard
    result=le.ledger_variance(data,2.02)
    assert result['soft_applied_variance']+result['hard_applied_variance']+result['twice_soft_hard_covariance']==pytest.approx(result['raw_residual_variance'])
    data['ledger__soft_applied'][0]+=.01
    with pytest.raises(ValueError,match='close'):
        le.ledger_variance(data,2.02)


def test_hard_cohorts_are_outcome_selected_not_closure():
    x=np.array([0.,.1,.2,.3])
    result=le.hard_cohorts(x,np.array([0,0,1,1]),x,np.array([1,1,0,0]))
    assert result['no_hard']['fast_n']==2
    assert result['no_hard']['fast_mean']==.05
    assert 'not a closure' in result['no_hard']['scope']
    assert le.hard_cohorts(x,np.ones(4),x,np.ones(4))['no_hard'] is None
