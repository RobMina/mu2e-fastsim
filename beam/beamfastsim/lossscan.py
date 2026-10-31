"""New split/diffusion response on actual entries of a recovered original stock-G4 scan.

The seed-selected subset is independent of outcomes. Every selected event is
retained; no replica truncation or physics tuning. Selection uncertainty is
paired by entry identity, conditional on the archived scan. Threshold and
physical endpoints remain distinct. No G4 true-path or exit-angular closure is
possible from these reduced arrays. Configuration qualification is limited to
recorded effective config/log/receipt checks in lossarchive.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from mu2efastsim import materials
from . import g4tables, lossarchive as la, lossfollowup as lf, physics_tables as pt
from . import slabdiffusion as sd, slabstudy as ss, slabendpoints as se
from .convergence import cluster_mean
from .reference_audit import _identity


def selection(events, samples, seed):
    events=np.asarray(events)
    if events.ndim!=1 or samples<2 or samples>len(events) or len(np.unique(events))!=len(events):
        raise ValueError('unique independent events and valid subset size required')
    return np.sort(np.random.default_rng(seed).choice(len(events),samples,replace=False))


def entry_directions(cosine):
    """Promote float ROOT reductions before vector arithmetic; validate gun domain."""
    cosine=np.asarray(cosine,float)
    if np.any(~np.isfinite(cosine)) or np.any((cosine<.5-1e-6)|(cosine>1+1e-6)):
        raise ValueError('measured incidence outside archived gun range')
    cosine=np.clip(cosine,-1.,1.)
    return np.column_stack([np.sqrt(1-cosine*cosine),np.zeros(len(cosine)),cosine])


def compare_bins(entry,cosine,fast,g4):
    entry,cosine,fast,g4=[np.asarray(v).ravel() for v in (entry,cosine,fast,g4)]
    result=[]
    for clo,chi in ((.5,.625),(.625,.75),(.75,.875),(.875,1.000001)):
        for elo,ehi in ((1.,1.4),(1.4,1.8),(1.8,2.2),(2.2,2.600001)):
            keep=(cosine>=clo)&(cosine<chi)&(entry>=elo)&(entry<ehi)
            delta=(fast[keep]<.1).astype(float)-(g4[keep]<.1)
            result.append(dict(energy_MeV=[elo,ehi],cosine=[clo,chi],n=int(keep.sum()),
                threshold_fast=float(np.mean(fast[keep]<.1)) if keep.any() else None,
                threshold_g4=float(np.mean(g4[keep]<.1)) if keep.any() else None,
                paired_threshold_delta=cluster_mean(delta,np.arange(len(delta))[:,None])))
    return result


def run(args):
    from . import lossdiagnostic as ld
    started=time.monotonic()
    out=Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    scan_root=Path(args.archive_root)/'foilscan_t1056_mum'
    audit=la.scan_audit(scan_root,args.table_root)
    reduced=ld.analyze(args.recovered_root,args.package_dir)
    package=Path(args.package_dir)/'g4_al_mu.npz'
    comparison=g4tables.compare(args.table_root,package,'G4_Al',require_audited_export=False)
    if not comparison['all_arrays_equal']:
        raise ValueError('slab package numeric mismatch')
    directory=scan_root/'t1056_full_mum'
    with np.load(directory/'scan.npz',allow_pickle=False) as saved:
        g={k:saved[k] for k in saved.files}
    idx=selection(g['event'],args.samples,args.selection_seed)
    g={k:v[idx] for k,v in g.items()}
    tb_package=g4tables.load(package,expected_material='G4_Al')
    tables,info,_=pt.load_target(Path(args.table_root)/'tables',material='G4_Al')
    tb=sd.make_tables(tables['Range']['energy'],tables['Range']['values'],tb_package['msc_energy'],tb_package['msc_lambda1'])
    energy,cosine=np.asarray(g['entry_MeV'],float),np.asarray(g['entry_cos'],float)
    directions=entry_directions(cosine)
    consumed=audit['inputs']+reduced['inputs']
    sources=[_identity(m.__file__) for m in (la,ld,lf,sd,ss,se,pt,g4tables)]+[_identity(__file__)]
    out.mkdir(parents=True,exist_ok=False)
    inputs_record=ss._save(out/'selected_inputs.npz',dict(g,original_row=idx))
    report=dict(complete=False,scope=__doc__,args=vars(args),archive_audit=audit,inputs=consumed,
                sources=sources,outputs=[inputs_record],selection=dict(n=len(idx),seed=args.selection_seed,scope='Uniform subset without replacement, selected before fast response; every selected event retained'),
                physics_closure=False)
    checkpoint=lambda:(out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    checkpoint()
    response=sd.respond(energy,directions,.1056,tb,material=materials.ALUMINIUM,tcut=info['energy_cuts_MeV']['e-'],
                        loss='split',range_kind='restricted',seed=args.seed,row_ids=g['event'].astype(np.uint32),
                        max_steps=20000)
    outputs=ss._save(out/'response.npz',ss.flatten(response))
    report['outputs'].append(outputs)
    intervals=se.endpoint_intervals(response,energy)
    if not np.array_equal(intervals['threshold_lower'],intervals['threshold_upper']):
        raise ValueError('scan threshold endpoints not identified; arrays saved, report incomplete')
    fast=intervals['threshold_lower'].ravel()
    delta=(fast<.1).astype(float)-(g['exit_MeV']<.1)
    report.update(summary=ss.summarize(response),bins=compare_bins(energy,cosine,fast,g['exit_MeV']),
                  all_paired_threshold=cluster_mean(delta,g['event'][:,None]),
                  selected_g4_physical_stops=int(np.sum(g['physical_stop'])),selected_g4_threshold=int(np.sum(g['exit_MeV']<.1)))
    ss.verify(sources+consumed+report['outputs'])
    report.update(complete=True,wall_s=time.monotonic()-started)
    checkpoint()
    print(json.dumps(dict(all=report['all_paired_threshold'],endpoints=report['summary']['endpoint_counts'],wall_s=report['wall_s'])),flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',required=True)
    parser.add_argument('--table-root',required=True)
    parser.add_argument('--recovered-root',required=True)
    parser.add_argument('--package-dir',default=str(Path(__file__).with_name('data')))
    parser.add_argument('--out',required=True)
    parser.add_argument('--samples',type=int,default=8000)
    parser.add_argument('--selection-seed',type=int,default=20261030)
    parser.add_argument('--seed',type=int,default=195)
    return run(parser.parse_args(argv))


if __name__=='__main__':
    main()
