"""Approximate local angular-increment/chord-budget audit of retained G4 steps.

Recorded adjacent pre-step chords are not true G4 path lengths. Lambda is the
exported mu+-sourced transport table, assumed shared by charge; it does not
certify a Gaussian law. First-exit and post-energy selection remain diagnostic.
All uncertainties cluster increments by primary event, never treat steps as iid.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from . import edgeg4, foilreport as fr, slabrefs, g4tables, edgeshape
from .edgeshape import estimate, probability
from .reference_audit import _identity

EDGES = np.array([.1, .2, .5, 1., 1.5, 3.])


def reduce_steps(steps, ref, msc_energy, lambda1, *, step_size=fr.CHUNK_SIZE):
    n, bins = len(ref['event']), len(EDGES)-1
    sentinel = np.iinfo(np.int64).max
    first, end = np.full(n, sentinel), np.full(n, sentinel)
    names = ('pairs', 'chord_mm', 'budget_left', 'budget_trapezoid', 'sum_one_minus_cos',
             'sum_one_minus_cos_squared', 'gaussian_small_q_second', 'sum_theta2', 'sum_theta4',
             'gaussian_theta4', 'large_jumps_0p3', 'large_jumps_0p6', 'budget_large_jumps_0p3', 'budget_large_jumps_0p6',
             'moment_large_jumps_0p3', 'moment_large_jumps_0p6', 'small_q_pairs', 'small_q_budget',
             'small_q_one_minus_cos', 'small_q_theta4', 'small_q_gaussian_theta4', 'small_q_large_jumps_0p6',
             'small_q_gaussian_large_jumps_0p6', 'q_above_0p001_pairs', 'pre_post_energy_drop', 'pair_pre_energy')
    out = {k:np.zeros((n, bins)) for k in names}
    loge, loglam = np.log(msc_energy), np.log(lambda1)
    fields = 'run evt trk pdg sid px py pz x y z'.split()
    previous, offset = None, 0
    for rows in steps.iterate(fields, step_size=step_size, library='np'):
        ev = fr.integer(rows['evt'], 'moment event')-1
        if np.any(ev < 0) or np.any(ev >= n) or np.any(rows['run'] != 1):
            raise ValueError('moment event outside reference population')
        matched = fr.integer(rows['trk'], 'moment track') == ref['track'][ev]
        ev = ev[matched]
        if not len(ev):
            continue
        s = {k:rows[k][matched] for k in fields}
        if np.any(s['pdg'] != 13) or np.any(np.diff(ev) < 0) or (previous is not None and ev[0] < previous['event'][-1]):
            raise ValueError('moment primary identity/order mismatch')
        at = offset+np.arange(len(ev))
        np.minimum.at(first, ev[s['sid'] == 2], at[s['sid'] == 2])
        outside = (s['sid'] != 2) & (at > first[ev])
        np.minimum.at(end, ev[outside], at[outside])
        p = np.column_stack([np.asarray(s[k], float) for k in ('px', 'py', 'pz')])
        xyz = np.column_stack([np.asarray(s[k], float) for k in ('x', 'y', 'z')])
        e, a, sid = ev, at, s['sid']
        if previous is not None:
            p, xyz = np.vstack((previous['p'], p)), np.vstack((previous['xyz'], xyz))
            e, a = np.r_[previous['event'], e], np.r_[previous['at'], a]
            sid = np.r_[previous['sid'], sid]
        norm = np.linalg.norm(p, axis=1)
        energy = fr.kinetic(norm)
        pair = (e[:-1] == e[1:]) & (sid[:-1] == 2) & (a[:-1] < end[e[:-1]]) & (energy[:-1] >= .1) & (energy[1:] >= .1)
        index = np.flatnonzero(pair)
        pre, post = energy[index], energy[index+1]
        if np.any(pre < msc_energy[0]) or np.any(pre > msc_energy[-1]) or np.any(post < msc_energy[0]) or np.any(post > msc_energy[-1]):
            raise ValueError('angular budget outside exported lambda domain')
        if np.any(~np.isfinite(p)) or np.any(~np.isfinite(xyz)):
            raise ValueError('nonfinite primary angular-budget states')
        chord = np.linalg.norm(xyz[index+1]-xyz[index], axis=1)
        left = np.exp(-np.interp(np.log(pre), loge, loglam))
        right = np.exp(-np.interp(np.log(post), loge, loglam))
        q = chord*left
        qtrap = chord*(left+right)/2
        cosine = np.clip(np.sum(p[index]*p[index+1], axis=1)/(norm[index]*norm[index+1]), -1, 1)
        angle, y = np.arccos(cosine), 1-cosine
        bin_index = np.searchsorted(EDGES, pre, side='right')-1
        if np.any(bin_index < 0) or np.any(bin_index >= bins):
            raise ValueError('predeclared angular energy bins do not cover retained pair')
        flat = e[index]*bins+bin_index
        small = q <= .001
        gaussian_tail = np.zeros(len(q))
        np.divide(-.6**2, 2*q, out=gaussian_tail, where=q > 0)
        gaussian_tail = np.where(q > 0, np.exp(gaussian_tail), 0.)
        values = dict(pairs=np.ones(len(index)), chord_mm=chord, budget_left=q, budget_trapezoid=qtrap,
                      sum_one_minus_cos=y, sum_one_minus_cos_squared=y*y, gaussian_small_q_second=2*q*q,
                      sum_theta2=angle*angle, sum_theta4=angle**4, gaussian_theta4=8*q*q,
                      large_jumps_0p3=(angle > .3).astype(float), large_jumps_0p6=(angle > .6).astype(float),
                      budget_large_jumps_0p3=q*(angle > .3), budget_large_jumps_0p6=q*(angle > .6),
                      moment_large_jumps_0p3=y*(angle > .3), moment_large_jumps_0p6=y*(angle > .6),
                      small_q_pairs=small.astype(float), small_q_budget=q*small, small_q_one_minus_cos=y*small,
                      small_q_theta4=angle**4*small, small_q_gaussian_theta4=8*q*q*small,
                      small_q_large_jumps_0p6=((angle > .6) & small).astype(float),
                      small_q_gaussian_large_jumps_0p6=gaussian_tail*small, q_above_0p001_pairs=(~small).astype(float),
                      pre_post_energy_drop=pre-post, pair_pre_energy=pre)
        for key, value in values.items():
            out[key] += np.bincount(flat, weights=value, minlength=n*bins).reshape(n, bins)
        previous = dict(p=p[-1:], xyz=xyz[-1:], event=e[-1:], at=a[-1:], sid=sid[-1:])
        offset += len(ev)
    if np.any(first == sentinel) or not np.array_equal(end != sentinel, ref['exit'].astype(bool)):
        raise ValueError('angular-budget first-exit census mismatch')
    return out


def ratio(numerator, denominator):
    x, y = np.asarray(numerator, float), np.asarray(denominator, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x+y)):
        raise ValueError('finite per-primary numerator/denominator required')
    if y.mean() <= 0:
        return None
    value = x.mean()/y.mean()
    influence = (x-value*y)/y.mean()
    return dict(value=float(value), se=float(influence.std(ddof=1)/np.sqrt(len(x))), n=len(x),
                numerator_sum=float(x.sum()), denominator_sum=float(y.sum()),
                scope='Per-primary cluster influence SE for ratio of sums; chord/pre-energy approximation, not calibration')


def summarize(arrays, ref):
    result = {}
    masks = {'all_entries':np.ones(len(ref['event']), bool),
             'forward_above_exits':ref['exit'].astype(bool) & (ref['exit_z_mm'] >= 0) & (ref['exit_MeV'] >= .1)}
    for name, mask in masks.items():
        result[name] = {}
        for i in range(-1, len(EDGES)-1):
            values = {key:(v.sum(axis=1) if i < 0 else v[:, i])[mask] for key, v in arrays.items()}
            label = 'all_pre_energies' if i < 0 else f'{EDGES[i]}_to_{EDGES[i+1]}'
            y, q = values['sum_one_minus_cos'], values['budget_left']
            result[name][label] = dict(primary_count=int(mask.sum()), pair_count=int(values['pairs'].sum()),
                transport_ratio_left=ratio(y, q), transport_ratio_trapezoid=ratio(y, values['budget_trapezoid']),
                theta4_over_small_q_gaussian=ratio(values['sum_theta4'], values['gaussian_theta4']),
                one_minus_cos2_over_small_q_gaussian=ratio(values['sum_one_minus_cos_squared'], values['gaussian_small_q_second']),
                large_jump_event_probability=probability(values['large_jumps_0p6'] > 0),
                fraction_transport_moment_from_large_jumps=ratio(values['moment_large_jumps_0p6'], y),
                mean_transport_budget=estimate(q), mean_one_minus_cos=estimate(y),
                small_q_pair_count=int(values['small_q_pairs'].sum()),
                small_q_transport_ratio=ratio(values['small_q_one_minus_cos'], values['small_q_budget']),
                small_q_theta4_ratio=ratio(values['small_q_theta4'], values['small_q_gaussian_theta4']),
                small_q_large_jump_count=int(values['small_q_large_jumps_0p6'].sum()),
                small_q_expected_gaussian_large_jump_count=float(values['small_q_gaussian_large_jumps_0p6'].sum()),
                small_q_primary_large_jump_probability=probability(values['small_q_large_jumps_0p6'] > 0),
                fractional_energy_change=ratio(values['pre_post_energy_drop'], values['pair_pre_energy']))
    return result


def run(root, out):
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError(out)
    sources = [_identity(p) for p in (Path(__file__), Path(fr.__file__), Path(slabrefs.__file__), Path(g4tables.__file__),
               Path(edgeg4.__file__), Path(edgeshape.__file__))]
    records, result = [], dict(complete=False, scope=__doc__, physics_closure=False, runs={}, sources=sources)
    out.mkdir(parents=True, exist_ok=False)
    for cell, (energy, thickness, cosine) in edgeg4.CELLS.items():
        refs, identities = slabrefs.load_references(root, cell, 13, thickness, energy, cosine)
        records.extend(identities)
        for mode, ref in refs.items():
            tag = cell+'_'+mode
            receipt = json.loads((root/tag/'receipt.json').read_text())
            for item in receipt['tables']:
                records.append(edgeg4.checked(item))
            tables = g4tables._extract(root/tag, 'G4_Al', require_audited_export=True)
            with uproot.open(root/tag/'foil.root', array_cache=None) as file:
                steps, tree = fr.highest(file, 'readback/ntstepper', arrays=False)
                arrays = reduce_steps(steps, ref, tables['msc_energy'], tables['msc_lambda1'])
            np.savez_compressed(out/(tag+'_moments.npz'), **arrays)
            result['runs'][tag] = dict(tree=tree, statistics=summarize(arrays, ref),
                lambda_source='Exported LambdaMod0.msc.mu+.asc, energy**2/table; charge equivalence assumed, not verified')
    for item in records+sources:
        edgeg4.checked(item)
    result.update(complete=True, inputs=records, outputs=[_identity(p) for p in sorted(out.glob('*.npz'))])
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True, runs=len(result['runs']))), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    run(args.root, args.out)


if __name__ == '__main__':
    main()
