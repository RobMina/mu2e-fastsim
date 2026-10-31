"""Lower-tier new-response runner contracts; synthetic references, never Geant4."""
import json

import numpy as np
import pytest

from beamfastsim import lossfollowup as lf, slabdiffusion as sd
from test_lossdiagnostic import recovery  # noqa: F401
from test_slabstudy import response


def test_components_independent_difference():
    a,b=[lf.ld.mixture(x) for x in ([0.,.2,.4],[0.,.1,.3])]
    result=lf.component_contrast(a,b)
    total=sum(result[k]['delta'] for k in ('within_population_variance','between_population_variance'))
    assert total==pytest.approx(result['population_variance_delta'])
    assert result['within_population_variance']['se']>0


def test_endpoints_never_fabricate_ordinary_closure():
    tb=sd.make_tables([.0001,1.,10.],[.0001,1.,10.],[.0001,1.,10.],[1.,1.,1.])
    r=response([sd.TRANSMITTED,sd.UNRESOLVED_FLOOR,sd.RAWZERO],[.3,.0001,0.])
    result=lf.response_diagnostics(r,2.02,.2,tb)
    assert result['threshold_distribution_identified']
    assert result['mixture']['n']==3 and result['mixture']['above_fraction']==pytest.approx(1/3)
    assert result['certified_physical_stop_count'] is None
    r=response([sd.TRANSMITTED,sd.UNRESOLVED_NUMERICAL],[.3,.0001])
    result=lf.response_diagnostics(r,2.02,.2,tb)
    assert not result['threshold_distribution_identified'] and result['mixture'] is None


def test_runner_records_lower_tier_and_new_arrays(recovery,tmp_path,monkeypatch):
    root,packages,_,_=recovery
    check=dict(all_arrays_equal=True,material_equal=True,fresh=dict(provenance=dict(files={})))
    monkeypatch.setattr(lf.g4tables,'compare',lambda *a,**kw:check)
    calls=[]
    def respond(*a,**kw):
        calls.append(kw)
        return response([sd.TRANSMITTED,sd.UNRESOLVED_FLOOR,sd.RAWZERO,sd.TRANSMITTED],[.3,.0001,0.,.2])
    monkeypatch.setattr(sd,'respond',respond)
    out=tmp_path/'new'
    argv=['--recovered-root',str(root),'--package-dir',str(packages),'--out',str(out),
          '--cell','moved_edge','--samples','4','--arms','no_scatter','diffuse']
    result=lf.main(argv)
    assert result['complete'] and result['current_execution'] and not result['historical_response_reproduced']
    assert result['reference_configuration_qualification'].startswith('UNVERIFIED')
    assert not result['physical_closure'] and len(result['outputs'])==3
    assert calls[0]['range_kind']=='restricted' and calls[0]['loss']=='split'
    assert not calls[0]['scatter'] and calls[1]['scatter']
    assert result['arms']['diffuse']['vs_g4']['full']['components']['population_variance_delta'] is not None
    lf.ss.verify(result['inputs']+result['sources']+result['outputs'])
    with pytest.raises(FileExistsError):
        lf.main(argv)
    assert len(calls)==2


def test_different_table_numeric_identity_fails_before_transport(recovery,tmp_path,monkeypatch):
    root,packages,_,_=recovery
    monkeypatch.setattr(lf.g4tables,'compare',lambda *a,**kw:dict(all_arrays_equal=False))
    monkeypatch.setattr(sd,'respond',lambda *a,**kw:pytest.fail('must not run'))
    with pytest.raises(ValueError,match='differ'):
        lf.main(['--recovered-root',str(root),'--package-dir',str(packages),'--out',str(tmp_path/'new'),'--cell','moved_edge'])
