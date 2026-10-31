"""Coarse chord-block angular moments from retained primary readback only.

Fixed 0.001/0.005/0.01-mm chord bins aggregate fine G4 increments toward the
fast leg scale without changing a trajectory. Actual summed chord/lambda
budgets are retained; block boundaries are not interpolated. No truth path or
fast-trajectory parity is claimed. Every error clusters blocks by primary.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from . import edgeg4, slabrefs, foilreport as fr, g4tables, edgemoments, edgeshape, reference_audit
from .edgemoments import ratio
from .edgeshape import estimate, probability
from .reference_audit import _identity

LENGTHS = (.001, .005, .01)


def primary_rows(tree, ref, *, step_size=fr.CHUNK_SIZE):
    fields = 'run evt trk pdg sid px py pz x y z'.split()
    columns = {key:[] for key in fields}
    for chunk in tree.iterate(fields, step_size=step_size, library='np'):
        ev = fr.integer(chunk['evt'], 'block event')-1
        if np.any(ev < 0) or np.any(ev >= len(ref['event'])) or np.any(chunk['run'] != 1):
            raise ValueError('block event outside reference population')
        match = fr.integer(chunk['trk'], 'block track') == ref['track'][ev]
        for key in columns:
            columns[key].append(chunk[key][match])
    result = {k:np.concatenate(v) for k, v in columns.items()}
    if np.any(result['pdg'] != 13) or np.any(np.diff(result['evt']) < 0):
        raise ValueError('primary block identity/order mismatch')
    return result


def blocks(rows, ref, msc_energy, lambda1, length):
    if not np.isfinite(length) or length <= 0:
        raise ValueError('positive finite block length required')
    n = len(ref['event'])
    event = fr.integer(rows['evt'], 'block event')-1
    if np.any(event < 0) or np.any(event >= n) or np.any(np.diff(event) < 0):
        raise ValueError('ordered complete block events required')
    at = np.arange(len(event))
    first, end = np.full(n, len(at)), np.full(n, len(at))
    np.minimum.at(first, event[rows['sid'] == 2], at[rows['sid'] == 2])
    outside = (rows['sid'] != 2) & (at > first[event])
    np.minimum.at(end, event[outside], at[outside])
    if np.any(first == len(at)) or not np.array_equal(end < len(at), ref['exit'].astype(bool)):
        raise ValueError('block first-exit census mismatch')
    p = np.column_stack([np.asarray(rows[k], float) for k in ('px', 'py', 'pz')])
    xyz = np.column_stack([np.asarray(rows[k], float) for k in ('x', 'y', 'z')])
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(xyz)):
        raise ValueError('finite primary block states required')
    norm = np.linalg.norm(p, axis=1)
    energy = fr.kinetic(norm)
    use = (event[:-1] == event[1:]) & (rows['sid'][:-1] == 2) & (at[:-1] < end[event[:-1]])
    use &= (energy[:-1] >= .1) & (energy[1:] >= .1)
    ix = np.flatnonzero(use)
    if not len(ix):
        raise ValueError('above-threshold adjacent block pairs required')
    ev = event[ix]
    dl = np.linalg.norm(xyz[ix+1]-xyz[ix], axis=1)
    inverse = np.exp(-np.interp(np.log(energy[ix]), np.log(msc_energy), np.log(lambda1)))
    if np.any(energy[ix] < msc_energy[0]) or np.any(energy[ix] > msc_energy[-1]):
        raise ValueError('block budget outside lambda domain')
    prefix = np.r_[0., np.cumsum(dl)]
    start = np.r_[0, np.flatnonzero(np.diff(ev))+1]
    offsets = np.zeros(n)
    offsets[ev[start]] = prefix[start]
    local = prefix[:-1]-offsets[ev]
    bin_id = np.floor(np.maximum(local, 0.)/length).astype(np.int64)
    groups = np.r_[0, np.flatnonzero((np.diff(ev) != 0) | (np.diff(bin_id) != 0))+1]
    ends = np.r_[groups[1:]-1, len(ix)-1]
    count = len(groups)
    group_index = np.repeat(np.arange(count), ends-groups+1)
    q = np.bincount(group_index, weights=dl*inverse, minlength=count)
    chord = np.bincount(group_index, weights=dl, minlength=count)
    i, j = ix[groups], ix[ends]+1
    cos = np.clip(np.sum(p[i]*p[j], axis=1)/(norm[i]*norm[j]), -1, 1)
    theta = np.arccos(cos)
    return dict(event=ev[groups], pre_energy=energy[i], post_energy=energy[j], chord_mm=chord,
                budget=q, cosine=cos, theta=theta, fine_pairs=ends-groups+1)


def summarize(data, ref):
    result = dict(blocks=len(data['event']), chord_quantiles_mm=np.quantile(data['chord_mm'], [0, .1, .5, .9, 1]).tolist(),
                  budget_quantiles=np.quantile(data['budget'], [0, .1, .5, .9, 1]).tolist(),
                  fine_pairs_quantiles=np.quantile(data['fine_pairs'], [0, .1, .5, .9, 1]).tolist(), cohorts={})
    n = len(ref['event'])
    forward = ref['exit'].astype(bool) & (ref['exit_z_mm'] >= 0) & (ref['exit_MeV'] >= .1)
    for name, select_primary in [('all_primaries', np.ones(n, bool)), ('forward_above_exits', forward)]:
        use = select_primary[data['event']]
        ev = data['event'][use]
        q, theta, cosine = data['budget'][use], data['theta'][use], data['cosine'][use]
        hist = lambda x: np.bincount(ev, weights=x, minlength=n)[select_primary]
        y = hist(1-cosine)
        budget = hist(q)
        theta4, gauss4 = hist(theta**4), hist(8*q*q)
        jumps, jump_moment = hist((theta > .6).astype(float)), hist((1-cosine)*(theta > .6))
        result['cohorts'][name] = dict(n=int(select_primary.sum()), blocks=int(use.sum()),
            transport_ratio=ratio(y, budget), theta4_over_gaussian=ratio(theta4, gauss4),
            jump_over_0p6_primary_probability=probability(jumps > 0),
            transport_moment_fraction_from_jump_over_0p6=ratio(jump_moment, y),
            mean_budget=estimate(budget), mean_transport_moment=estimate(y))
    return result


def run(root, out):
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError(out)
    sources = [_identity(p) for p in (Path(__file__), Path(fr.__file__), Path(slabrefs.__file__), Path(g4tables.__file__),
               Path(edgeg4.__file__), Path(edgemoments.__file__), Path(edgeshape.__file__), Path(reference_audit.__file__))]
    records = []
    result = dict(complete=False, scope=__doc__, physics_closure=False, runs={}, block_lengths_mm=LENGTHS)
    out.mkdir(parents=True, exist_ok=False)
    for cell, (energy, thickness, cosine) in edgeg4.CELLS.items():
        refs, identities = slabrefs.load_references(root, cell, 13, thickness, energy, cosine)
        records.extend(identities)
        for mode, ref in refs.items():
            tag = cell+'_'+mode
            receipt = json.loads((root/tag/'receipt.json').read_text())
            for item in receipt['tables']:
                records.append(edgeg4.checked(item))
            table = g4tables._extract(root/tag, 'G4_Al', require_audited_export=True)
            with uproot.open(root/tag/'foil.root', array_cache=None) as file:
                steps, tree = fr.highest(file, 'readback/ntstepper', arrays=False)
                rows = primary_rows(steps, ref)
            result['runs'][tag] = dict(tree=tree, lengths={})
            for length in LENGTHS:
                data = blocks(rows, ref, table['msc_energy'], table['msc_lambda1'], length)
                np.savez_compressed(out/(tag+'_'+str(length)+'_blocks.npz'), **data)
                result['runs'][tag]['lengths'][str(length)] = summarize(data, ref)
    for item in sources+records:
        edgeg4.checked(item)
    result.update(complete=True, sources=sources, inputs=records, outputs=[_identity(p) for p in sorted(out.glob('*.npz'))])
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
