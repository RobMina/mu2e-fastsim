"""Explain saved threshold variance failures and numerical sensitivity without rerunning physics.

All component contrasts are diagnostics on independent samples, not causal
attribution. The threshold transform keeps every history. Conditional moments
select the above-threshold outcome. Missing G4 directions/true paths prevent
angular/path closure; no adjustment or fitted model is proposed here.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import lossdiagnostic as ld, slabstudy as ss
from .reference_audit import _identity


def conditional_parameters(x):
    y=np.asarray(x,float).ravel()
    positive=y>=.1
    q=float(positive.mean())
    if q==0:
        raise ValueError('positive component required for conditional decomposition')
    mean=float(y[positive].mean())
    var=float(y[positive].var())
    influence=np.column_stack([positive-q,positive*(y-mean)/q,
                               positive*((y-mean)**2-var)/q])
    return np.array([q,mean,var]),np.cov(influence,rowvar=False,ddof=1)/len(y)


def contributions(a,b):
    """Exact symmetric product decomposition of V=q*s2+q*(1-q)*m2."""
    q,m,s=a
    r,n,t=b
    return np.array([(q-r)*((s+t)/2+(m*m+n*n)/2*(1-q-r)),
                     (m*m-n*n)*(q*(1-q)+r*(1-r))/2,
                     (s-t)*(q+r)/2])


def explain(candidate,reference):
    a,ca=conditional_parameters(candidate)
    b,cb=conditional_parameters(reference)
    delta=contributions(a,b)
    jac=np.empty((3,6))
    values=np.r_[a,b]
    for k in range(6):
        h=1e-5*max(abs(values[k]),1e-3)
        low,high=values.copy(),values.copy()
        low[k]-=h
        high[k]+=h
        jac[:,k]=(contributions(high[:3],high[3:])-contributions(low[:3],low[3:]))/(2*h)
    covariance=jac[:,:3]@ca@jac[:,:3].T+jac[:,3:]@cb@jac[:,3:].T
    labels=('occupancy','conditional_mean_squared','conditional_variance')
    result={name:dict(delta=float(delta[i]),se=float(np.sqrt(max(0,covariance[i,i])))) for i,name in enumerate(labels)}
    result.update(population_variance_delta=float(delta.sum()),component_covariance=covariance.tolist(),
                  parameters=dict(candidate=dict(zip(('q','mean','variance'),a.tolist())),
                                  reference=dict(zip(('q','mean','variance'),b.tolist()))),
                  scope='Exact symmetric algebraic decomposition; delta-method pointwise SE; components correlated, outcome-conditioned, not causal mechanisms')
    return result


def hard_cohorts(fast, fast_count, reference, reference_count):
    """Outcome-selected hard-emission controls; not independent closure tests."""
    result={}
    for name,fmask,gmask in [('no_hard',fast_count==0,reference_count==0),
                             ('has_hard',fast_count>0,reference_count>0)]:
        if min(int(fmask.sum()),int(gmask.sum()))<2:
            result[name]=None
            continue
        a,b=np.asarray(fast)[fmask],np.asarray(reference)[gmask]
        result[name]=dict(fast_n=len(a),reference_n=len(b),
            fast_variance=float(a.var(ddof=1)),reference_variance=float(b.var(ddof=1)),
            fast_mean=float(a.mean()),reference_mean=float(b.mean()),
            scope='Each sample conditioned on its own hard-emission outcome; collider/selection diagnostic, not a closure gate')
    return result


def ledger_variance(arrays, entry):
    """All-entry applied-loss covariance budget; endpoints retained, no causality claim."""
    keys=['ledger__soft_applied','ledger__hard_applied']
    if any(k not in arrays for k in keys):
        return None
    soft,hard=[np.asarray(arrays[k],float).ravel() for k in keys]
    residual=np.asarray(arrays['raw_energy'],float).ravel()
    if not np.allclose(soft+hard+residual,entry,rtol=0,atol=1e-10):
        raise ValueError('saved energy ledger does not close')
    covariance=np.cov(np.stack([soft,hard]),ddof=0)
    transformed=threshold_array(arrays,entry)
    return dict(soft_applied_variance=float(covariance[0,0]),hard_applied_variance=float(covariance[1,1]),
                twice_soft_hard_covariance=float(2*covariance[0,1]),
                total_applied_variance=float((soft+hard).var()),raw_residual_variance=float(residual.var()),
                threshold_minus_raw_variance=float(transformed.var()-residual.var()),
                mean_hard_applied=float(hard.mean()),hard_applied_max=float(hard.max()),
                positive_hard_fraction=float((hard>0).mean()),
                scope='Stopped/path-selected applied channels; covariance includes variable stopping/path, not primitive microscopic fluctuation variances or causal attribution')


def paired_variance(candidate,baseline):
    a,b=np.asarray(candidate,float).ravel(),np.asarray(baseline,float).ravel()
    if a.shape!=b.shape or len(a)<2:
        raise ValueError('same complete sample identities required')
    da,db=a-a.mean(),b-b.mean()
    influence=(da*da-a.var())-(db*db-b.var())
    d=(a<.1).astype(float)-(b<.1).astype(float)
    return dict(population_variance_delta=float(a.var()-b.var()),
                variance_se=float(influence.std(ddof=1)/np.sqrt(len(a))),
                threshold_fraction_delta=float(d.mean()),threshold_se=float(d.std(ddof=1)/np.sqrt(len(a))),
                scope='Keyed-identity paired response diagnostic, not Brownian-bridge coupled physical histories')


def read_study(directory):
    directory=Path(directory)
    path=directory/'summary.json'
    report=json.loads(path.read_text())
    if report.get('complete') is not True:
        raise ValueError('finished new-response study required')
    ss.verify(report['inputs']+report['outputs'])
    arrays={}
    for name,item in report['arms'].items():
        ld._checked(item['output'],directory/(name+'.npz'))
        with np.load(directory/(name+'.npz'),allow_pickle=False) as saved:
            arrays[name]={k:saved[k] for k in saved.files}
    return report,arrays,_identity(path)


def threshold_array(arrays,entry):
    from . import slabdiffusion as sd, slabendpoints as se
    response=sd.Response(*(arrays[k] for k in sd.Response._fields if k!='ledger'),
                         {k.removeprefix('ledger__'):v for k,v in arrays.items() if k.startswith('ledger__')})
    interval=se.endpoint_intervals(response,entry)
    if not np.array_equal(interval['threshold_lower'],interval['threshold_upper']):
        raise ValueError('unidentified threshold distribution; no variance explanation')
    return interval['threshold_lower'].ravel()


def summarize(directories):
    result=dict(scope=__doc__,studies=[],independent_comparisons={},paired_refinements={},physics_closure=False)
    baselines={}
    all_studies=[]
    for directory in directories:
        report,arrays,identity=read_study(directory)
        result['studies'].append(identity)
        cell=report['args']['cell']
        refs=report['recovered_reference_audit']['references']
        for arm,data in arrays.items():
            if arm=='diffuse':
                baselines[(cell,report['args']['pdg'],report['args']['seed'],report['args']['samples'],report['args']['loss_model'])]=(report,data)
            if report['args']['pdg']==13:
                mode='loss_finest' if arm=='no_scatter' else 'full'
                tag=cell+'_'+mode
                if tag not in refs:
                    continue
                record=refs[tag]['identity']
                ld._checked(record,record['path'])
                with np.load(record['path'],allow_pickle=False) as saved:
                    raw_reference=np.asarray(saved['exit_MeV'],float)
                    reference_count=np.asarray(saved['hard_count'])
                    reference=np.where(raw_reference<.1,0.,raw_reference)
                entry=refs[tag]['case']['energy_MeV']
                candidate=threshold_array(data,entry)
                summary=explain(candidate,reference)
                summary.update(hard_emission_cohorts=hard_cohorts(candidate,np.asarray(data['ledger__hard_count']).ravel(),reference,reference_count),
                               raw_vs_threshold_diagnostic=dict(
                    raw_residual_variance=float(np.var(data['raw_energy'])),
                    reference_raw_variance=float(raw_reference.var()),
                    reference_threshold_variance=float(reference.var()),
                    raw_residual_minus_reference_variance=float(np.var(data['raw_energy'])-raw_reference.var()),
                    raw_endpoint_scope='Fast raw residuals include unresolved floor, not ordinary physical endpoint distribution'),
                               loss_covariance_budget=ledger_variance(data,entry),
                               distribution=report['arms'][arm]['vs_g4'][mode]['distribution'],
                               endpoints=report['arms'][arm]['summary']['endpoint_counts'],
                               ledger=report['arms'][arm]['summary']['ledger'])
                result['independent_comparisons'][str(Path(directory).name)+'/'+arm]=summary
        all_studies.append((directory,report,arrays))
    for directory,report,arrays in all_studies:
        key=tuple(report['args'][k] for k in ('cell','pdg','seed','samples','loss_model'))
        if key not in baselines:
            continue
        base,base_arrays=baselines[key]
        for name,data in arrays.items():
            if name in ('no_scatter','diffuse'):
                continue
            for field in ('seed','samples','loss_model'):
                if report['args'][field]!=base['args'][field]:
                    raise ValueError('unpaired response settings')
            for field in ('row_ids','replica_ids','seed'):
                if not np.array_equal(data[field],base_arrays[field]):
                    raise ValueError('unpaired response identities')
            tag=next(k for k in base['recovered_reference_audit']['references'] if k.startswith(key[0]+'_'))
            energy=base['recovered_reference_audit']['references'][tag]['case']['energy_MeV']
            result['paired_refinements'][str(Path(directory).name)+'/'+name]=paired_variance(threshold_array(data,energy),threshold_array(base_arrays,energy))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',nargs='+',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    out=Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    result=summarize(args.study)
    result['sources']=[_identity(__file__)]
    with out.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    main()
