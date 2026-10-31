"""New fast-response diagnostics against explicitly lower-tier recovered G4 reductions.

This is NOT slabstudy's audited-reference path and never upgrades its qualification.
Original G4 config/log/receipt/ROOT evidence is missing. Reduced references and table
bytes are checked by lossdiagnostic; numerical table equality is not configuration
or historical-binary certification. No G4, production transport or default change.
Each execution is new and retains all rows, identities, response arrays and ledgers.
Threshold-transform comparisons do not certify physical endpoints. Conditional
path/energy diagnostics and moment decompositions are not causal physics closure.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import jax
import numpy as np
import scipy

from mu2efastsim import eloss, eloss_jax, eloss_split_jax, materials
from . import g4tables, lossdiagnostic as ld, physics_tables as pt
from . import slabdiffusion as sd, slabendpoints as se, slabstudy as ss
from .foilcampaign import CASES
from .foilcompare import contrast
from .reference_audit import _identity


def component_contrast(candidate, reference):
    """Independent-sample component differences; not paired stochastic histories."""
    result = {}
    for key, se_key in (('within_population_variance', 'within_se'),
                        ('between_population_variance', 'between_se')):
        result[key] = dict(delta=candidate[key]-reference[key],
                           se=float(np.hypot(candidate[se_key], reference[se_key])))
    result['population_variance_delta'] = candidate['population_variance']-reference['population_variance']
    result['above_fraction_delta'] = candidate['above_fraction']-reference['above_fraction']
    result['scope'] = 'Independent-sample pointwise SE; components correlated; no sole-cause attribution'
    return result


def response_diagnostics(response, energy, thickness, tb):
    """Retain unresolved endpoints and return identified threshold-only statistics."""
    result = ss.summarize(response)
    result['endpoint_identification'] = se.summarize_endpoints(response, np.array([energy]), thickness, tb)
    intervals = se.endpoint_intervals(response, energy)
    identified = np.array_equal(intervals['threshold_lower'], intervals['threshold_upper'])
    result['threshold_distribution_identified'] = identified
    result['mixture'] = ld.mixture(intervals['threshold_lower'].ravel()) if identified else None
    keep = np.isin(response.status, [sd.TRANSMITTED, sd.REFLECTED])
    above = keep & (response.raw_energy >= .1)
    direction = np.asarray(response.direction)
    result['resolved_exit_direction_z'] = ss._stats(direction[..., 2][keep])
    result['above_energy_path_correlation'] = ld._correlation(response.raw_energy[above], response.path[above])
    result['exit_path_over_incident_projected'] = ss._stats(response.path[keep]/thickness)
    result['raw_residual_variance_bound_MeV2'] = (
        ld.residual_variance_bound(float(np.var(response.raw_energy)),
                                  float(np.max(intervals['raw_upper']-intervals['raw_lower']))) if identified else None)
    return result


def run(args):
    """Generate only requested fast arms, with lower-tier provenance explicit."""
    started = time.monotonic()
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if (args.samples < 2 or args.pdg not in (13, -13) or args.seed < 0 or not args.arms
            or 'lower_floor' in args.arms or len(set(args.arms)) != len(args.arms)):
        raise ValueError('independent sample count, charge, seed and unique arms required')
    audit = ld.analyze(args.recovered_root, args.package_dir)
    root = Path(args.recovered_root)
    comparison = g4tables.compare(root/'foil_threadG_smoke_20261026', Path(args.package_dir)/'g4_al_mu.npz',
                                  'G4_Al', require_audited_export=False)
    if not comparison['all_arrays_equal'] or not comparison['material_equal']:
        raise ValueError('recovered table arrays differ from supplied slab package')
    charge = 'mu-' if args.pdg == 13 else 'mu+'
    native, info, paths = pt.load_target(root/'foil_threadG_smoke_20261026/tables', material='G4_Al', particle=charge)
    tables = g4tables.load(Path(args.package_dir)/'g4_al_mu.npz', expected_material='G4_Al')
    vector = native['Range'] if args.loss_model == 'split' else dict(energy=tables['energy'], values=tables['csda_range_'+charge])
    tb = sd.make_tables(vector['energy'], vector['values'], tables['msc_energy'], tables['msc_lambda1'])
    cell = next(c for c in CASES if c[0] == args.cell)
    _, energy, thickness, cosine = cell
    lower = float(np.exp(max(np.asarray(tb.loge)[0], np.asarray(tb.lam_loge)[0])))
    upper = float(np.exp(min(np.asarray(tb.loge)[-1], np.asarray(tb.lam_loge)[-1])))
    floor = lower if args.floor is None else args.floor
    settings = ss.arm_settings(args.arms, floor, None, lower, upper, max_step=args.max_step,
                               angular_fraction=args.angular_fraction, energy_fraction=args.energy_fraction,
                               max_steps=args.max_steps, loss='split' if args.loss_model == 'split' else 'sample',
                               tcut=float(info['energy_cuts_MeV']['e-']))
    if args.loss_model == 'split':
        for item in settings.values():
            item['range_kind'] = 'restricted'
    sources = [_identity(m.__file__) for m in (ld, ss, sd, se, pt, g4tables, eloss, eloss_jax, eloss_split_jax, materials)]
    sources.append(_identity(__file__))
    consumed = audit['inputs'] + list(comparison['fresh']['provenance']['files'].values())
    ss.verify(sources+consumed)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, complete=False, scope=__doc__, args=vars(args),
                  reference_configuration_qualification='UNVERIFIED: original configs/macros/logs/receipts/ROOT absent',
                  current_execution=True, historical_response_reproduced=False, physical_closure=False,
                  recovered_reference_audit=audit, table_numeric_comparison=comparison,
                  table_couple=info, material=asdict(materials.ALUMINIUM),
                  tables_sha256={name: hashlib.sha256(np.asarray(v, dtype='<f8').tobytes()).hexdigest()
                                 for name,v in zip(sd.Tables._fields,tb)},
                  sources=sources, inputs=consumed, outputs=[], arms={}, paired={},
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                   jax=jax.__version__, backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64)))
    checkpoint = lambda: (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    checkpoint()
    refs = {}
    if args.pdg == 13:
        for tag, item in audit['references'].items():
            if tag.startswith(args.cell+'_'):
                with np.load(item['identity']['path'], allow_pickle=False) as data:
                    refs[tag[len(args.cell)+1:]] = {k:data[k] for k in data.files}
    responses = {}
    for name, kw in settings.items():
        begin = time.monotonic()
        response = sd.respond(np.array([energy]), np.array([[np.sqrt(1-cosine**2),0.,cosine]]), thickness,
                              tb, material=materials.ALUMINIUM, replicas=args.samples, seed=args.seed, **kw)
        runtime = time.monotonic()-begin
        arrays = dict(ss.flatten(response), row_ids=np.array([0],np.uint32),
                      replica_ids=np.arange(args.samples,dtype=np.uint32), seed=np.array(args.seed,np.uint32))
        record = ss._save(out/(name+'.npz'), arrays)
        report['outputs'].append(record)
        summary = response_diagnostics(response, energy, thickness, tb)
        summary['exit_path_over_incident_projected'] = ss._stats(response.path[np.isin(response.status,[sd.TRANSMITTED,sd.REFLECTED])]/(thickness/cosine))
        item = dict(params=kw, output=record, runtime_s_including_compile=runtime, summary=summary, vs_g4={})
        if summary['mixture'] is not None:
            y = se.endpoint_intervals(response,energy)['threshold_lower'].ravel()
            for mode,g in refs.items():
                item['vs_g4'][mode] = dict(distribution=contrast(y,g['exit_MeV'],energy),
                    components=component_contrast(summary['mixture'],audit['references'][args.cell+'_'+mode]['mixture']))
        report['arms'][name] = item
        responses[name] = response
        checkpoint()
        print(json.dumps(dict(arm=name, runtime_s=runtime, endpoints=summary['endpoint_counts'],
                              mixture=summary['mixture'])),flush=True)
    base = 'diffuse' if 'diffuse' in responses else args.arms[0]
    for name,response in responses.items():
        if name == base:
            continue
        arrays, summary = ss.paired(response,responses[base])
        record = ss._save(out/('paired_'+name+'_minus_'+base+'.npz'),arrays)
        report['outputs'].append(record)
        report['paired'][name+'_minus_'+base] = dict(summary=summary,output=record)
    ss.verify(sources+consumed+report['outputs'])
    report.update(complete=True,wall_s=time.monotonic()-started)
    checkpoint()
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovered-root',required=True)
    parser.add_argument('--package-dir',default=str(Path(__file__).with_name('data')))
    parser.add_argument('--out',required=True)
    parser.add_argument('--cell',choices=[c[0] for c in CASES],required=True)
    parser.add_argument('--pdg',type=int,choices=[13,-13],default=13)
    parser.add_argument('--samples',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=92)
    parser.add_argument('--arms',nargs='+',choices=ss.ARMS,default=['no_scatter','diffuse','refined'])
    parser.add_argument('--loss-model',choices=['split','hybrid'],default='split')
    parser.add_argument('--max-step',type=float,default=.01)
    parser.add_argument('--angular-fraction',type=float,default=.02)
    parser.add_argument('--energy-fraction',type=float,default=.05)
    parser.add_argument('--max-steps',type=int,default=20000)
    parser.add_argument('--floor',type=float)
    return run(parser.parse_args(argv))


if __name__=='__main__':
    main()
