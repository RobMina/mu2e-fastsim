"""Expanded-config parsing and fail-closed archived bundle audit; no art or G4."""
import json
from pathlib import Path

import pytest

from beamfastsim import lossarchive as la
from beamfastsim.reference_audit import _identity
from test_lossdiagnostic import recovery  # noqa: F401


def test_expanded_blocks_and_scalars():
    text='physics: { producers: { g4run: { physics: { physicsListName: "ShieldingM" minRangeCut: 1e-2 } note: "brace } in string" } } }'
    phys=la.block(la.block(la.block(la.block(text,'physics'),'producers'),'g4run'),'physics')
    assert la.scalar(phys,'minRangeCut')==.01
    assert la.scalar(phys,'physicsListName')=='ShieldingM'
    for bad in ('x: {', 'x: {} x: {}'):
        with pytest.raises(ValueError):
            la.block(bad,'x')
    with pytest.raises(ValueError):
        la.scalar('x: 1 x: 2','x')


def archived(recovery):
    recovered,_,_,_=recovery
    root=recovered/'scan'
    run=root/'tag'
    run.mkdir(parents=True)
    case=dict(events=4,energy_MeV=2.02,cosine=1.,pdg=13)
    import numpy as np
    data=dict(event=np.arange(1,5),exit_MeV=np.array([0.,.05,.2,.4]),exit=np.array([False,True,True,True]),
              physical_stop=np.array([True,False,False,False]),other=np.zeros(4,bool),exit_z_mm=np.array([0.,.1,.1,.1]))
    np.savez(run/'scan.npz',**data)
    (run/'expanded.fcl').write_text('physics: { producers: { g4run: { g4Macro: "preinit.mac" physics: { physicsListName: "ShieldingM" minRangeCut: .01 protonProductionCut: 0 setMuHadLateralDisplacement: true useDensityEffectInIonizationLossCalc: false } } } }')
    (run/'foil_geom.txt').write_text('string box.wallMaterialName = "G4_Al";')
    (run/'preinit.mac').write_text('/control/verbose 2\n')
    (run/'job.log').write_text('Geant4 version Name: synthetic\nphysicsListDecider invoked with ShieldingM\n( Box, 2, G4_Al )\nTrigReport Events total = 4\nArt has completed and will exit with status 0\n')
    receipt=dict(complete=True,exit_code=0,case=case,release='synthetic',output=dict(path='/missing/foil.root',bytes=1,sha256='0'*64),
                 artifacts=[_identity(run/n) for n in ('expanded.fcl','foil_geom.txt','preinit.mac','job.log')])
    (run/'receipt.json').write_text(json.dumps(receipt))
    manifest=dict(complete=True,runs=dict(tag=dict(mode='full',receipt=_identity(run/'receipt.json'),scan_npz=_identity(run/'scan.npz'),
                  census=dict(primaries=4,exits=3,physical_stops=1,threshold_stops=2,back_exits=0))))
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root,recovered/'foil_threadG_smoke_20261026'


def test_archived_signature_scope_and_byte_checks(recovery):
    root,tables=archived(recovery)
    result=la.scan_audit(root,tables)
    assert not result['physics_closure']
    scan=result['scans']['tag']
    assert scan['material_range_cut_process_signature']=='MATCHED_RECORDED_CONFIGURATION'
    assert not scan['historical_root_reaudited']
    assert scan['energy_cut_export_for_this_run'].startswith('UNAVAILABLE')
    (root/'tag/preinit.mac').write_text('modified')
    with pytest.raises(ValueError,match='hash/size'):
        la.scan_audit(root,tables)


def test_full_baseline_qualifies_alignment_and_counts(tmp_path):
    import numpy as np
    from beamfastsim import g4ref
    root=tmp_path/'g4'
    root.mkdir()
    g=dict(run=np.ones(4),subrun=np.zeros(4),evt=np.arange(1,5),foil=np.array([0,0,-1,1]),
           q=np.array([-1.,-1.,1.,1.]),time=np.zeros(4),px=np.zeros(4),py=np.zeros(4),pz=np.array([15.,25.,35.,45.]))
    g4ref.save(g,root/'g4ref.npz')
    (root/'reference_audit_20261025_threadG.json').write_text(json.dumps(dict(table=_identity(root/'g4ref.npz'))))
    f=dict(x=np.zeros((4,3)),p=np.array([0.,20.,20.,0.]),t=np.zeros(4),fate=np.array([1,6,6,1]),
           foil=np.array([0,-1,-1,1]),nfoil=np.ones(4,int),meta=np.array(json.dumps(dict(n=4))))
    fast=tmp_path/'fast.npz'
    g4ref.save(f,fast)
    metrics=tmp_path/'metrics.json'
    metrics.write_text(json.dumps(dict(n_muons=4,foil_profile_g4=[2,1],foil_profile_fast=[1,1])))
    result=la.full_baseline(root,fast,metrics)
    assert result['historical_fast_producer_binding'].startswith('UNVERIFIED')
    assert result['groups']['all']['all']['first_foil_counts']['fast']==1
    assert result['groups']['all']['incident_momentum_diagnostics'][1]['first_g4']==1
    assert result['groups']['all']['all']['first_foil_minus_g4']['mean']==-.25
    metrics.write_text(json.dumps(dict(n_muons=4,foil_profile_g4=[3,1],foil_profile_fast=[1,1])))
    with pytest.raises(ValueError,match='profile'):
        la.full_baseline(root,fast,metrics)


@pytest.mark.parametrize('change',['material','cut','physics','macro','exit','count'])
def test_semantic_signature_rejected_even_after_hash_refresh(recovery,change):
    root,tables=archived(recovery)
    directory=root/'tag'
    if change in ('cut','physics'):
        p=directory/'expanded.fcl'
        p.write_text(p.read_text().replace('.01','.02') if change=='cut' else p.read_text().replace('ShieldingM','Other'))
    elif change=='material':
        (directory/'foil_geom.txt').write_text('string box.wallMaterialName = "Other";')
    elif change=='macro':
        (directory/'preinit.mac').write_text('/control/verbose 2\n/process/inactivate msc mu-\n')
    elif change=='exit':
        p=directory/'job.log'
        p.write_text(p.read_text().replace('status 0','status 1'))
    else:
        p=directory/'job.log'
        p.write_text(p.read_text().replace('total = 4','total = 3'))
    receipt=json.loads((directory/'receipt.json').read_text())
    receipt['artifacts']=[_identity(directory/Path(item['path']).name) for item in receipt['artifacts']]
    (directory/'receipt.json').write_text(json.dumps(receipt))
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['runs']['tag']['receipt']=_identity(directory/'receipt.json')
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        la.scan_audit(root,tables)


def test_unhashed_consumed_config_refused(recovery):
    root,tables=archived(recovery)
    directory=root/'tag'
    receipt=json.loads((directory/'receipt.json').read_text())
    receipt['artifacts']=[p for p in receipt['artifacts'] if Path(p['path']).name!='expanded.fcl']
    (directory/'receipt.json').write_text(json.dumps(receipt))
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['runs']['tag']['receipt']=_identity(directory/'receipt.json')
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='must hash'):
        la.scan_audit(root,tables)
