"""Audit recovered original grid artifacts; no transport, G4, or source re-certification.

A complete archived scan retains hash-bound runtime config/log/macros and a
producer-audited reduced array. Its deleted ROOT cannot be re-audited. The scan
is not a replacement for missing monoenergetic reference signatures. A recovered
pilot is checked with pilotaudit against the original full G4 table identity.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from . import foilclosure, lossdiagnostic as ld, pilotaudit, simpleconfig
from .reference_audit import _identity


def block(text, key):
    """Read an exact expanded-FHiCL table by name, respecting quoted braces."""
    candidates={m.start():m for m in re.finditer(r'(?<![\w.])'+re.escape(key)+r'\s*:\s*\{',text)}
    matches=[]
    depth,quoted,escaped=0,False,False
    for i,char in enumerate(text):
        if not quoted and depth==0 and i in candidates:
            matches.append(candidates[i])
        if escaped:
            escaped=False
        elif char=='\\' and quoted:
            escaped=True
        elif char=='"':
            quoted=not quoted
        elif not quoted:
            depth += (char=='{')-(char=='}')
    if len(matches)!=1:
        raise ValueError(f'unique expanded table required: {key}')
    start=matches[0].end()
    depth,quoted,escaped=1,False,False
    for i in range(start,len(text)):
        char=text[i]
        if escaped:
            escaped=False
            continue
        if char=='\\' and quoted:
            escaped=True
        elif char=='"':
            quoted=not quoted
        elif not quoted:
            depth += (char=='{')-(char=='}')
            if depth==0:
                return text[start:i]
    raise ValueError('unclosed expanded table')


def scalar(text,key):
    pattern=r'(?<![\w.])'+re.escape(key)+r'\s*:\s*("(?:[^"\\]|\\.)*"|[^\s{}\[\],]+)'
    found=re.findall(pattern,text)
    if len(found)!=1:
        raise ValueError(f'unique expanded scalar required: {key}')
    value=found[0]
    if value.startswith('"') or value in ('true','false','null'):
        return json.loads(value)
    try:
        return float(value)
    except ValueError:
        return value


def scan_audit(scan_root, tables_root):
    scan_root=Path(scan_root)
    manifest_path=scan_root/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('complete') is not True:
        raise ValueError('complete archived scan required')
    inputs=[_identity(manifest_path)]
    results={}
    for tag,run in manifest['runs'].items():
        root=scan_root/tag
        inputs.append(ld._checked(run['receipt'],root/'receipt.json'))
        receipt=json.loads((root/'receipt.json').read_text())
        if receipt.get('complete') is not True or receipt.get('exit_code')!=0:
            raise ValueError('complete successful archived receipt required')
        names=[Path(item['path']).name for item in receipt['artifacts']]
        if len(names)!=len(set(names)) or not {'expanded.fcl','job.log','preinit.mac','foil_geom.txt'}<=set(names):
            raise ValueError('receipt must hash each consumed configuration artifact')
        for item in receipt['artifacts']:
            inputs.append(ld._checked(item,root/Path(item['path']).name))
        scan_path=root/'scan.npz'
        inputs.append(ld._checked(run['scan_npz'],scan_path))
        text=(root/'job.log').read_text()
        if 'Art has completed and will exit with status 0' not in text or foilclosure.macro_errors(text):
            raise ValueError('archived runtime log not successful')
        n=receipt['case']['events']
        if not re.search(r'TrigReport Events total\s*=\s*'+str(n)+r'\b',text):
            raise ValueError('archived log event count mismatch')
        with np.load(scan_path,allow_pickle=False) as saved:
            g={k:saved[k] for k in saved.files}
        if any(v.shape!=(n,) for v in g.values()) or not np.array_equal(g['event'],np.arange(1,n+1)):
            raise ValueError('archived array census mismatch')
        ex,stop=g['exit'].astype(bool),g['physical_stop'].astype(bool)
        if np.any(g['other']) or np.any(ex==stop) or np.any(~np.isfinite(g['exit_MeV'])):
            raise ValueError('archived unresolved outcomes')
        count=dict(primaries=n,exits=int(ex.sum()),physical_stops=int(stop.sum()),
                   threshold_stops=int(np.sum(g['exit_MeV']<.1)),back_exits=int(np.sum(ex&(g['exit_z_mm']<0))))
        if any(count[k]!=v for k,v in run['census'].items() if k in count):
            raise ValueError('archived stored census mismatch')
        cfg=(root/'expanded.fcl').read_text()
        physics=block(block(block(cfg,'physics'),'producers'),'g4run')
        table=block(physics,'physics')
        signature={k:scalar(table,k) for k in ('physicsListName','minRangeCut','protonProductionCut',
                        'setMuHadLateralDisplacement','noDecay','useDensityEffectInIonizationLossCalc') if k!='noDecay'}
        signature['g4Macro']=scalar(physics,'g4Macro')
        macro=(root/'preinit.mac').read_text()
        commands=[s.strip() for s in macro.splitlines() if s.strip() and not s.strip().startswith('#')]
        expected=['/control/verbose 2']
        if run['mode']=='full' and commands!=expected:
            raise ValueError('stock full archive contains unexpected macro controls')
        geometry=simpleconfig.read(root/'foil_geom.txt',root)
        _,couple,paths=ld.pt.load_target(Path(tables_root)/'tables',material='G4_Al')
        inputs.extend(_identity(p) for p in paths)
        if (signature['physicsListName']!='ShieldingM' or 'physicsListDecider invoked with ShieldingM' not in text
                or geometry['box.wallMaterialName']!='G4_Al' or '( Box, 2, G4_Al )' not in text
                or signature['minRangeCut']!=couple['range_cuts_mm']['e-']
                or signature['protonProductionCut']!=couple['range_cuts_mm']['proton']):
            raise ValueError('recorded process/material/range-cut signature mismatch')
        results[tag]=dict(census=count,receipt_case=receipt['case'],release=receipt['release'],
            observed_geant4=re.search(r'Geant4 version Name:\s*([^\n]+)',text).group(1),
            effective_signature=signature,macro_commands=commands,geometry=geometry,
            table_couple=couple,material_range_cut_process_signature='MATCHED_RECORDED_CONFIGURATION',
            energy_cut_export_for_this_run='UNAVAILABLE: no run-specific export',
            historical_root_reaudited=False,original_root_identity=receipt['output'],
            reference_scope='Original archived scan only; no transfer of qualification to missing monoenergetic cells',
            gun_incidence_arrays_present='entry_cos' in g,exit_direction_arrays_present=False)
    for item in inputs:
        ld._checked(item,item['path'])
    return dict(complete=True,scope=__doc__,scans=results,inputs=inputs,physics_closure=False)


def full_baseline(g4_directory, fast_path, metrics_path):
    """Audit a historical full-chain baseline; preserve missing producer identity caveat."""
    from . import convergence, g4ref
    root=Path(g4_directory)
    audit_path=root/'reference_audit_20261025_threadG.json'
    audit=json.loads(audit_path.read_text())
    records=[_identity(audit_path),ld._checked(audit['table'],root/'g4ref.npz'),_identity(fast_path),_identity(metrics_path)]
    g,f=g4ref.load(root/'g4ref.npz'),g4ref.load(fast_path)
    meta=pilotaudit._validate_fast(f,len(g['evt']),'historical full baseline')
    saved=json.loads(Path(metrics_path).read_text())
    ids=np.column_stack([g[k] for k in ('run','subrun','evt')])
    profile_g=np.bincount(g['foil'][g['foil']>=0]).tolist()
    profile_f=np.bincount(f['foil'][f['fate']==1],minlength=len(profile_g)).tolist()
    if (profile_g!=saved['foil_profile_g4'] or profile_f!=saved['foil_profile_fast']
            or saved['n_muons']!=len(g['evt']) or meta['n']!=len(g['evt'])):
        raise ValueError('historical full baseline count/profile mismatch')
    missed=(g['foil']==0)&~((f['fate']==1)&(f['foil']==0))
    gained=(g['foil']!=0)&(f['fate']==1)&(f['foil']==0)
    migration=dict(g4_first_missed_fast=int(missed.sum()),fast_first_not_g4=int(gained.sum()),
        missed_fast_later_foil=int(np.sum(missed&(f['fate']==1)&(f['foil']>0))),
        missed_fast_fates={str(int(k)):int(v) for k,v in zip(*np.unique(f['fate'][missed],return_counts=True))},
        missed_incident_radius_mm=np.quantile(np.hypot(g['x'][missed]+3904.,g['y'][missed]),[0,.5,.9,1]).tolist() if 'x' in g and np.any(missed) else None)
    result=dict(n=len(g['evt']),events=len(np.unique(ids,axis=0)),inputs=records,migration=migration,
                original_g4_table_hash_verified=True,
                historical_fast_producer_binding='UNVERIFIED: fast metadata has no original G4 input hash or saved row IDs',
                physics_closure=False,groups={})
    for label,select in [('all',np.ones(len(g['evt']),bool)),('mu_minus',g['q']<0),('mu_plus',g['q']>0)]:
        stats=pilotaudit._contrasts(g,f,f,select)
        p=np.sqrt(sum(np.asarray(g[k],float)**2 for k in ('px','py','pz')))
        c=np.asarray(g['pz'],float)/p
        bins=[]
        for lo,hi in [(0.,20.),(20.,30.),(30.,40.),(40.,60.),(60.,10000.)]:
            use=select&(p>=lo)&(p<hi)
            first_fast=(f['fate']==1)&(f['foil']==0)
            first_g4=g['foil']==0
            bins.append(dict(incident_p_MeV=[lo,hi],muons=int(use.sum()),
                first_fast=int(np.sum(use&first_fast)),first_g4=int(np.sum(use&first_g4)),
                conditional_deficit=convergence.cluster_mean((first_fast.astype(float)-first_g4)[use],ids[use]),
                incidence_cosine_mean=float(c[use].mean()) if use.any() else None))
        result['groups'][label]=dict(all=stats,incident_momentum_diagnostics=bins)
    for item in records:
        ld._checked(item,item['path'])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',required=True)
    parser.add_argument('--table-root',required=True)
    parser.add_argument('--g4',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--full-fast')
    parser.add_argument('--full-metrics')
    args=parser.parse_args(argv)
    if bool(args.full_fast)!=bool(args.full_metrics):
        raise ValueError('full fast and historical metrics required together')
    out=Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    archive=Path(args.archive_root)
    result=dict(scope=__doc__,scan=scan_audit(archive/'foilscan_t1056_mum',args.table_root),
                moved_pilot=pilotaudit.audit(archive/'arms_pilot_moved',args.g4),sources=[_identity(__file__)])
    if args.full_fast:
        result['full_baseline']=full_baseline(Path(args.g4).parent,args.full_fast,args.full_metrics)
    with out.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Original archived scan signatures checked; moved pilot re-audited; no closure claim')
    return result


if __name__=='__main__':
    main()
