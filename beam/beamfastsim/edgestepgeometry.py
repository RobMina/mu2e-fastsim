"""Recorded angular, chord and energy-decrement signatures of the step/bin controls.

Only completed references and fast arrays are read. Each exit-conditioned arm
selects its own outcomes. Chord lengths and spline table reconstructions are
not Geant4 true paths or interpolation parity. Local post-energy >=0.1-MeV
selection is explicit; total primary decrements include stochastic/hard losses.
No correction, transport, new G4 or exclusive causal attribution is performed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot
from scipy.integrate import cumulative_trapezoid

from . import edgeg4, edgeangles, edgefactorialreport as ef, edgeshape as es, edgemoments as em
from . import foilreport as fr, g4tables, physics_tables as pt, slabrefs, slabdiffusion as sd, lossexplain as le
from .reference_audit import _identity


DECREMENT_EDGES = em.EDGES


def range_averaged_inverse_lambda(tables, msc_energy, lambda1):
    """Diagnostic integral of inverse lambda along reconstructed restricted range."""
    msc_energy, lambda1 = np.asarray(msc_energy, float), np.asarray(lambda1, float)
    if (msc_energy.ndim != 1 or lambda1.shape != msc_energy.shape or len(msc_energy) < 2
            or np.any(~np.isfinite(msc_energy+lambda1)) or np.any(msc_energy <= 0)
            or np.any(lambda1 <= 0) or np.any(np.diff(msc_energy) <= 0)):
        raise ValueError('positive ordered lambda grid required')
    vector = tables['Range']
    lo, hi = max(.1, msc_energy[0], vector['energy'][0]), min(3., msc_energy[-1], vector['energy'][-1])
    if hi <= lo:
        raise ValueError('overlapping range/lambda diagnostic domain required')
    grid = np.geomspace(lo, hi, 65537)
    ranges = pt.interpolate(vector, grid)
    if np.any(np.diff(ranges) <= 0):
        raise ValueError('monotone reconstructed restricted range required')
    inverse = np.exp(-np.interp(np.log(grid), np.log(msc_energy), np.log(lambda1)))
    primitive = cumulative_trapezoid(inverse, ranges, initial=0.)
    def evaluate(pre, post):
        pre, post = np.asarray(pre, float), np.asarray(post, float)
        if pre.shape != post.shape or np.any(~np.isfinite(pre+post)) or np.any(np.minimum(pre, post) < lo) or np.any(np.maximum(pre, post) > hi):
            raise ValueError('pair outside range-averaged lambda domain')
        rp, rr = pt.interpolate(vector, pre), pt.interpolate(vector, post)
        delta = rp-rr
        numerator = np.interp(pre, grid, primitive)-np.interp(post, grid, primitive)
        value = np.exp(-np.interp(np.log((pre+post)/2), np.log(msc_energy), np.log(lambda1)))
        np.divide(numerator, delta, out=value, where=np.abs(delta) > 1e-10*np.maximum(np.abs(rp), np.abs(rr)))
        return value
    return evaluate


def decrement_steps(tree, ref, tables, *, msc_energy=None, lambda1=None, step_size=fr.CHUNK_SIZE):
    """Cluster total primary energy changes versus recorded-chord stopping budgets."""
    n, bins = len(ref['event']), len(DECREMENT_EDGES)-1
    names = ('pairs', 'chord_mm', 'pre_energy', 'energy_decrement', 'left_mean_budget',
             'trapezoid_mean_budget', 'native_range_decrement', 'large_fraction_pairs', 'negative_drop_pairs')
    if (msc_energy is None) != (lambda1 is None):
        raise ValueError('both lambda grid and values required')
    angular = msc_energy is not None
    if angular:
        names += ('sum_one_minus_cos', 'budget_left', 'budget_mid_energy', 'budget_range_average')
        range_inverse = range_averaged_inverse_lambda(tables, msc_energy, lambda1)
    out = {key:np.zeros((n, bins)) for key in names}
    power = pt.interpolator(tables['DEDX'])
    native_range = pt.interpolator(tables['Range'])
    first, end = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    fields = 'run evt trk pdg sid px py pz x y z'.split()
    previous, offset = None, 0
    for chunk in tree.iterate(fields, step_size=step_size, library='np'):
        event = fr.integer(chunk['evt'], 'decrement event')-1
        if np.any(event < 0) or np.any(event >= n) or np.any(chunk['run'] != 1):
            raise ValueError('decrement event outside reference population')
        match = fr.integer(chunk['trk'], 'decrement track') == ref['track'][event]
        ev = event[match]
        if not len(ev):
            continue
        rows = {key:chunk[key][match] for key in fields}
        if (np.any(rows['pdg'] != 13) or np.any(np.diff(ev) < 0)
                or (previous is not None and ev[0] < previous['event'][-1])):
            raise ValueError('decrement primary identity/order mismatch')
        at = offset+np.arange(len(ev))
        inside = rows['sid'] == 2
        selected, loc = np.unique(ev[inside], return_index=True)
        ix = np.flatnonzero(inside)[loc]
        new = first[selected] < 0
        first[selected[new]] = at[ix[new]]
        outside = (~inside) & (first[ev] >= 0) & (at > first[ev])
        selected, loc = np.unique(ev[outside], return_index=True)
        ix = np.flatnonzero(outside)[loc]
        new = end[selected] < 0
        end[selected[new]] = at[ix[new]]
        p = np.column_stack([np.asarray(rows[key], float) for key in ('px', 'py', 'pz')])
        xyz = np.column_stack([np.asarray(rows[key], float) for key in ('x', 'y', 'z')])
        e, a, sid = ev, at, rows['sid']
        if previous is not None:
            p, xyz = np.vstack((previous['p'], p)), np.vstack((previous['xyz'], xyz))
            e, a, sid = np.r_[previous['event'], e], np.r_[previous['at'], a], np.r_[previous['sid'], sid]
        if np.any(~np.isfinite(p)) or np.any(~np.isfinite(xyz)):
            raise ValueError('finite decrement states required')
        norm = np.linalg.norm(p, axis=1)
        energy = fr.kinetic(norm)
        use = (e[:-1] == e[1:]) & (sid[:-1] == 2) & ((end[e[:-1]] < 0) | (a[:-1] < end[e[:-1]]))
        use &= (energy[:-1] >= .1) & (energy[1:] >= .1)
        ix = np.flatnonzero(use)
        pre, post = energy[ix], energy[ix+1]
        chord = np.linalg.norm(xyz[ix+1]-xyz[ix], axis=1)
        left, right = power(pre), power(post)
        delta = pre-post
        ib = np.searchsorted(DECREMENT_EDGES, pre, side='right')-1
        if np.any(ib < 0) or np.any(ib >= bins):
            raise ValueError('declared decrement energy bins do not cover pair')
        flat = e[ix]*bins+ib
        values = dict(pairs=np.ones(len(ix)), chord_mm=chord, pre_energy=pre, energy_decrement=delta,
            left_mean_budget=chord*left, trapezoid_mean_budget=chord*(left+right)/2,
            native_range_decrement=native_range(pre)-native_range(post),
            large_fraction_pairs=(delta > .1*pre).astype(float), negative_drop_pairs=(delta < -1e-6).astype(float))
        if angular:
            cosine = np.clip(np.sum(p[ix]*p[ix+1], axis=1)/(norm[ix]*norm[ix+1]), -1, 1)
            inverse = lambda energy: np.exp(-np.interp(np.log(energy), np.log(msc_energy), np.log(lambda1)))
            values.update(sum_one_minus_cos=1-cosine, budget_left=chord*inverse(pre),
                budget_mid_energy=chord*inverse((pre+post)/2), budget_range_average=chord*range_inverse(pre, post))
        for key, value in values.items():
            out[key] += np.bincount(flat, weights=value, minlength=n*bins).reshape(n, bins)
        previous = dict(p=p[-1:], xyz=xyz[-1:], event=e[-1:], at=a[-1:], sid=sid[-1:])
        offset += len(ev)
    if np.any(first < 0) or not np.array_equal(end >= 0, ref['exit'].astype(bool)):
        raise ValueError('decrement first-exit census mismatch')
    return out


def summarize_decrements(arrays, ref):
    cohorts = {'all_primaries':np.ones(len(ref['event']), bool),
               'forward_above_exits':ref['exit'].astype(bool) & (ref['exit_z_mm'] >= 0) & (ref['exit_MeV'] >= .1)}
    result = {}
    for cohort, mask in cohorts.items():
        result[cohort] = {}
        for i in range(-1, len(DECREMENT_EDGES)-1):
            rows = {key:(value.sum(axis=1) if i < 0 else value[:, i])[mask] for key, value in arrays.items()}
            name = 'all_pre_energies' if i < 0 else f'{DECREMENT_EDGES[i]}_to_{DECREMENT_EDGES[i+1]}'
            pairs, loss = rows['pairs'], rows['energy_decrement']
            result[cohort][name] = dict(primary_count=int(mask.sum()), pair_count=int(pairs.sum()),
                fractional_pre_post_change=em.ratio(loss, rows['pre_energy']),
                fraction_pairs_with_drop_over_10pct=em.ratio(rows['large_fraction_pairs'], pairs),
                total_decrement_over_left_restricted_chord_budget=em.ratio(loss, rows['left_mean_budget']),
                total_decrement_over_trapezoid_restricted_chord_budget=em.ratio(loss, rows['trapezoid_mean_budget']),
                reconstructed_range_decrement_over_chord=em.ratio(rows['native_range_decrement'], rows['chord_mm']),
                negative_drop_pair_count=int(rows['negative_drop_pairs'].sum()),
                mean_observed_pair_energy_decrement_MeV=es.estimate(loss),
                mean_chord_left_restricted_budget_MeV=es.estimate(rows['left_mean_budget']))
            if 'budget_range_average' in rows:
                result[cohort][name]['angular_budget_sensitivity'] = {
                    label:em.ratio(rows['sum_one_minus_cos'], rows[key]) for label, key in
                    (('left_pre_energy', 'budget_left'), ('mid_pre_post_energy', 'budget_mid_energy'),
                     ('range_averaged_inverse_lambda', 'budget_range_average'))}
    return dict(cohorts=result, scope='Total kinetic decrement includes stochastic/hard loss; >=0.1-MeV pre/post selection, primary-cluster SE; chord/spline proxies, not a mean-law calibration')


def forward_sample(data, directions, cosine):
    mask = data['exit'].astype(bool) & (data['exit_z_mm'] >= 0) & (data['exit_MeV'] >= .1)
    entry = np.array([np.sqrt(1-cosine*cosine), 0., cosine])
    angle = np.arccos(np.clip(np.asarray(directions)[mask] @ entry, -1, 1))
    return data['exit_MeV'][mask], angle


def compare(a, b, candidate_angles, reference_angles):
    result = es.compare(a, b)
    result['standardized_forward_angle'] = es.standardize_angle(*candidate_angles, *reference_angles)
    result['features'] = {}
    for cohort in ('all', 'forward_above'):
        result['features'][cohort] = {name:es.difference(value, b['features'][cohort]['metrics'][name])
            for name, value in a['features'][cohort]['metrics'].items()
            if value is not None and b['features'][cohort]['metrics'][name] is not None}
    return result


def run(root, reused, prior, studies, out):
    root, reused, prior, out = map(Path, (root, reused, prior, out))
    if out.exists():
        raise FileExistsError(out)
    previous = json.loads((prior/'summary.json').read_text())
    if previous.get('complete') is not True:
        raise ValueError('complete independent factorial report required')
    records = [_identity(prior/'summary.json')]
    sources = [_identity(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    result = dict(complete=False, scope=__doc__, physics_closure=False, new_transport=False,
                  g4={}, fast={}, pairs={}, fast_comparisons={}, inputs=records, sources=sources)
    out.mkdir(parents=True, exist_ok=False)
    for cell, (energy, thickness, cosine) in edgeg4.CELLS.items():
        refs, identities = slabrefs.load_references([root, reused], cell, 13, thickness, energy, cosine)
        records.extend(identities)
        angle_samples = {}
        for mode in ef.ORDER:
            tag = cell+'_'+mode
            data = refs[mode]
            directory = (reused if mode == 'full_refined' else root)/tag
            old = previous['references'][tag]['receipt']
            edgeg4.checked(old)
            receipt = json.loads((directory/'receipt.json').read_text())
            for item in receipt['sources']+receipt['tables']+[receipt['output']]:
                records.append(edgeg4.checked(item))
            if not Path(receipt['output']['path']).is_file() or len(data['event']) != 5000:
                raise ValueError('original ROOT and full bounded census required')
            tables, couple, _ = pt.load_target(directory/'tables', material='G4_Al')
            package = g4tables._extract(directory, 'G4_Al', require_audited_export=True)
            with uproot.open(directory/'foil.root', array_cache=None) as file:
                steps, tree = fr.highest(file, 'readback/ntstepper', arrays=False)
                angles = edgeangles.first_exit(steps, data)
                features = es.step_features(steps, data)
                angular_moments = em.reduce_steps(steps, data, package['msc_energy'], package['msc_lambda1'])
                decrements = decrement_steps(steps, data, tables, msc_energy=package['msc_energy'], lambda1=package['msc_lambda1'])
            if not np.allclose(features['adjacent_chord_sum_mm'], data['chord_sum_mm'], rtol=1e-6, atol=2e-6):
                raise ValueError('geometry chord sum differs from audited reference')
            if not np.array_equal(angular_moments['pairs'], decrements['pairs']):
                raise ValueError('angular/decrement retained-pair census mismatch')
            for suffix, arrays in (('angles', angles), ('features', features), ('moments', angular_moments), ('decrements', decrements)):
                np.savez_compressed(out/(tag+'_'+suffix+'.npz'), **arrays)
            summary = es.summarize_g4(data, angles, features, thickness, cosine)
            summary.update(tree=tree, angular_moments=em.summarize(angular_moments, data),
                           decrements=summarize_decrements(decrements, data), couple=couple,
                           hard_electron_mean=es.estimate(data['hard_MeV']),
                           mean_hard_electron_count=es.estimate(data['hard_count']))
            result['g4'][tag] = summary
            angle_samples[mode] = forward_sample(data, angles['direction'], cosine)
        result['pairs'][cell] = {name:compare(result['g4'][cell+'_'+ef.ORDER[a]], result['g4'][cell+'_'+ef.ORDER[b]],
                    angle_samples[ef.ORDER[a]], angle_samples[ef.ORDER[b]]) for name, (a, b) in ef.PAIRS.items()}
        for study in studies:
            report, arms, identity = le.read_study(study)
            if report['args']['cell'] != cell:
                continue
            records.append(identity)
            arrays = arms['diffuse']
            records.append(report['arms']['diffuse']['output'])
            y = le.threshold_array(arrays, energy)
            status = np.asarray(arrays['status']).ravel()
            exits, forward = np.isin(status, [sd.TRANSMITTED, sd.REFLECTED]), status == sd.TRANSMITTED
            raw = np.asarray(arrays['raw_energy']).ravel()
            directions = np.asarray(arrays['direction']).reshape(-1, 3)
            summary = dict(tails=es.tails(y, exits, forward), exit_statistics=es.exit_statistics(raw, exits, forward,
                directions, np.asarray(arrays['path']).ravel()/(thickness/cosine), cosine))
            tag = Path(study).name+'_diffuse'
            result['fast'][tag] = summary
            fast_angles = forward_sample(dict(exit=exits, exit_z_mm=np.where(forward, 1., -1.), exit_MeV=raw), directions, cosine)
            result['fast_comparisons'][tag] = {mode:dict(es.compare(summary, result['g4'][cell+'_'+mode]),
                standardized_forward_angle=es.standardize_angle(*fast_angles, *angle_samples[mode])) for mode in ef.ORDER}
    for item in records+sources:
        edgeg4.checked(item)
    result['outputs'] = [_identity(p) for p in sorted(out.glob('*.npz'))]
    result['complete'] = True
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True, g4_arms=len(result['g4']), fast_arms=len(result['fast']))), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--reused-root', required=True)
    parser.add_argument('--prior', required=True)
    parser.add_argument('--study', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    run(args.root, args.reused_root, args.prior, args.study, args.out)


if __name__ == '__main__':
    main()
