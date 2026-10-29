"""Validate persisted dumper policy, pointer identities and legacy payload equality after a user build."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import uproot

from .. import genassoc as G
from .census_check import census_stats
from .replay import sha256

POLICIES = {'retained_primary': 2, 'staged_cat': 3}


def validate_policy(events, gen, parts, steps, policy):
    """Validate every pointer by full event tuple, ProductID and key, without kinematic matching."""
    if policy not in POLICIES or not len(events):
        raise ValueError('known policy and nonempty census required')
    for name in (*G.EVENT_COLUMNS, 'ngen', 'nsteps', 'association_policy'):
        if name not in events:
            raise ValueError('missing event policy/census column: ' + name)
    if not events.association_policy.eq(policy).all() or not events.schema_version.eq(POLICIES[policy]).all():
        raise ValueError('persisted association policy/schema mismatch')
    if not events.associations_checked.eq(1).all() or not events.full_generator_stream.eq(0).all():
        raise ValueError('step policy requires checked associations, not a full generator stream')
    for name in G.KEY + ['gen_product_id', 'sim_product_id']:
        G._integer(events, [name], 'events', ('u', 4))
    for name in ('n_gen', 'n_particles', 'n_steps', 'ngen', 'nsteps'):
        G._integer(events, [name], 'events')
        G._require(events[name] >= 0, 'negative event census')
    G._unique(events, G.KEY, 'events')
    G._require(events.gen_product_id > 0, 'null generator product')
    G._require(events.sim_product_id > 0, 'null SimParticle product')
    G._require(events.ngen == events.n_gen, 'generator census aliases disagree')
    G._require(events.nsteps == events.n_steps, 'step census aliases disagree')
    for name, frame in (('gen', gen), ('particles', parts), ('steps', steps)):
        rows = G._census(frame, events, name)
        columns = {'gen': ['gen_product_id'], 'particles': ['sim_product_id', 'gen_product_id', 'parent_product_id'],
                   'steps': ['sp_product_id']}[name]
        G._integer(frame, columns, name, ('u', 4))
        observed = frame[{'gen': 'gen_product_id', 'particles': 'sim_product_id', 'steps': 'sp_product_id'}[name]]
        expected = events['gen_product_id' if name == 'gen' else 'sim_product_id'].to_numpy()[rows]
        G._require(observed == expected, name + ': ProductID differs from census')
    G._integer(gen, ['idx', 'pdg'], 'gen')
    G._integer(parts, ['key', 'parent', 'genidx', 'isprim', 'pdg'], 'particles')
    G._integer(steps, ['sp'], 'steps')
    G._unique(gen, G.KEY + ['idx'], 'gen')
    G._unique(parts, G.KEY + ['key'], 'particles')
    G._require(gen.idx >= 0, 'negative generator index')
    grows = G._lookup(gen, events, G.KEY, G.KEY, 'gen events')
    G._require(gen.idx < events.n_gen.to_numpy()[grows], 'generator index out of range')
    G._require(parts.key >= 0, 'negative particle key')
    G._require(parts.isprim.isin([0, 1]), 'invalid primary flag')
    G._require(np.isfinite(gen[['e', 't']].to_numpy()), 'nonfinite generator energy/time')
    G._lookup(steps, parts, G.KEY + ['sp_product_id', 'sp'], G.KEY + ['sim_product_id', 'key'], 'step pointer')
    for key, product in (('parent', 'parent_product_id'), ('genidx', 'gen_product_id')):
        G._require(parts[key] >= -1, 'invalid pointer key')
        G._require((parts[key] == -1) == (parts[product] == 0), 'inconsistent null pointer')
    parents = parts.parent >= 0
    G._lookup(parts.loc[parents], parts, G.KEY + ['parent_product_id', 'parent'],
              G.KEY + ['sim_product_id', 'key'], 'parent pointer')
    linked = parts.genidx >= 0
    rows = G._lookup(parts.loc[linked], gen, G.KEY + ['gen_product_id', 'genidx'],
                     G.KEY + ['gen_product_id', 'idx'], 'generator pointer')
    mismatch = parts.loc[linked, 'pdg'].to_numpy() != gen.pdg.to_numpy()[rows]
    primary = parts.loc[linked, 'isprim'].to_numpy() == 1
    if policy == 'retained_primary':
        G._require(~(primary & mismatch), 'retained-primary signed PDG mismatch')
    return {'policy': policy, 'schema_version': POLICIES[policy], 'pointer_ProductIDs_keys_checked': True,
            'linked_primary_pdg_mismatches': int((primary & mismatch).sum()),
            **census_stats(events, gen, steps)}


def read_policy(path, policy):
    with uproot.open(path) as root:
        frames = {name: pd.DataFrame(root['dump/' + name].arrays(columns, library='np')) for name, columns in
                  (('events', G.EVENT_COLUMNS + ['ngen', 'nsteps', 'association_policy']),
                   ('gen', G.GEN_COLUMNS), ('particles', G.PART_COLUMNS), ('steps', G.STEP_COLUMNS))}
    result = validate_policy(frames['events'], frames['gen'], frames['particles'], frames['steps'], policy)
    if policy == 'retained_primary':
        result['cosmic_donor_validation'] = G.validate_donor(path)
    else:
        try:
            G.validate_donor(path)
        except ValueError as error:
            if 'unsupported association schema' not in str(error):
                raise
            result['cosmic_donor_rejected'] = str(error)
        else:
            raise ValueError('staged Cat was accepted by cosmic donor gate')
    return result


def compare_payload(candidate, reference, prefix=False):
    """Compare every reference branch exactly in stored order, with equal NaNs; additions are listed."""
    result = {}
    with uproot.open(candidate) as new, uproot.open(reference) as old:
        for name in ('events', 'gen', 'particles', 'steps', 'subruns'):
            key = 'dump/' + name
            if key not in old or key not in new:
                raise ValueError('missing comparison tree: ' + key)
            a, b = new[key], old[key]
            if not prefix and a.num_entries != b.num_entries:
                raise ValueError('different payload row count: ' + name)
            if a.num_entries > b.num_entries:
                raise ValueError('candidate longer than reference: ' + name)
            missing = set(b.keys()) - set(a.keys())
            if missing:
                raise ValueError(f'missing reference branches in {name}: {sorted(missing)}')
            differences = []
            for branch in b.keys():
                av = a[branch].array(library='np')
                bv = b[branch].array(entry_stop=a.num_entries, library='np')
                equal = np.array_equal(av, bv, equal_nan=True) if av.dtype.kind in 'fc' else np.array_equal(av, bv)
                if av.dtype != bv.dtype or not equal:
                    differences.append(branch)
            if differences:
                raise ValueError(f'changed reference payload {name}: {differences}')
            result[name] = {'candidate_rows': a.num_entries, 'reference_rows': b.num_entries,
                            'reference_branches_equal': len(b.keys()), 'added_branches': sorted(set(a.keys()) - set(b.keys()))}
    return result


def run(probe, out, policy, reference=None, prefix=False, events=None):
    probe, out = Path(probe), Path(out)
    if out.exists():
        raise ValueError('choose new report')
    path, log = probe / 'dump.root', probe / 'job.log'
    inputs = [path, log] + ([Path(reference)] if reference else [])
    hashes = {str(p): sha256(p) for p in inputs}
    text = log.read_text()
    if 'Art has completed and will exit with status 0.' not in text:
        raise ValueError('no successful art completion')
    result = read_policy(path, policy)
    match = re.findall(r'TrigReport Events total = (\d+)', text)
    if len(match) != 1 or int(match[0]) != result['events'] or (events is not None and events != result['events']):
        raise ValueError('completed event count differs')
    if reference:
        result['reference_comparison'] = compare_payload(path, reference, prefix)
        result['reference_comparison_scope'] = 'stored prefix only' if prefix else 'complete input payload'
    if hashes != {str(p): sha256(p) for p in inputs}:
        raise ValueError('input changed during validation')
    result.update(inputs_sha256=hashes, code_sha256=sha256(__file__))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return result


def check_negative(probe, guard, completed):
    """Expected rejection requires a specific guard, exact retained prefix and recorded exit one."""
    probe = Path(probe)
    text = (probe / 'job.log').read_text()
    if (probe / 'exit_status.txt').read_text().strip() != '1' or 'Art has completed and will exit with status 1.' not in text:
        raise ValueError('negative control did not exit one')
    if guard == 'product':
        match = re.search(r'references GenParticle \((\d+), (\d+)\); selected \S+ ProductID (\d+) size (\d+)', text)
        if not match or match[1] == match[3]:
            raise ValueError('missing unequal generator ProductID rejection')
    elif guard == 'primary':
        if 'primary/gen signed PDG mismatch' not in text:
            raise ValueError('missing retained-primary rejection')
    else:
        raise ValueError('unknown negative guard')
    with uproot.open(probe / 'dump.root') as root:
        counts = {name: root['dump/' + name].num_entries for name in ('events', 'steps', 'particles', 'gen')}
    if counts['events'] != completed or (completed == 0 and any(counts.values())):
        raise ValueError('negative control wrote unexpected payload')
    return {'guard': guard, 'exit_status': 1, 'completed_event_prefix': completed, 'entries_highest_cycle': counts,
            'inputs_sha256': {str(probe / p): sha256(probe / p) for p in ('dump.root', 'job.log', 'exit_status.txt')}}


def check_compact(first, second):
    """Validate each real pool separately; never falsify extraction metadata to demonstrate aliasing."""
    from .catsteps_v2 import PoolV2
    pools = [PoolV2([p]) for p in (first, second)]
    meta = [p.manifest[0]['metadata'] for p in pools]
    for field in ('frames', 'stepped_frames', 'steps', 'census_sha256', 'content_sha256'):
        if meta[0][field] != meta[1][field]:
            raise ValueError('compact aliases differ: ' + field)
    try:
        PoolV2([first, second])
    except ValueError as error:
        rejection = str(error)
        if not ('pool extraction metadata disagreement' in rejection or 'duplicate content_sha256' in rejection):
            raise
    else:
        raise ValueError('alias pool was not rejected')
    return {'input_sha256': {str(p): sha256(p) for p in (first, second)},
            'identical_compact_content': {k: meta[0][k] for k in ('frames', 'stepped_frames', 'steps', 'census_sha256', 'content_sha256')},
            'joint_pool_rejection': rejection,
            'scope': 'Separately validated aliases; differing extraction contracts are preserved, never patched to pass pooling.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe')
    parser.add_argument('--policy', choices=tuple(POLICIES))
    parser.add_argument('--reference')
    parser.add_argument('--prefix', action='store_true')
    parser.add_argument('--events', type=int)
    parser.add_argument('--negative-guard', choices=('product', 'primary'))
    parser.add_argument('--compact', nargs=2, help='two independently validated Cat-v2 aliases')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.compact or args.negative_guard:
        if args.compact and (args.probe or args.negative_guard) or args.negative_guard and (not args.probe or args.events is None):
            parser.error('choose compact pair or negative guard with probe and expected completed events')
        out = Path(args.out)
        if out.exists():
            raise ValueError('choose new report')
        result = check_compact(*args.compact) if args.compact else check_negative(args.probe, args.negative_guard, args.events)
        result['code_sha256'] = sha256(__file__)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2))
    else:
        if not args.probe or not args.policy:
            parser.error('positive check requires probe and policy')
        run(args.probe, args.out, args.policy, args.reference, args.prefix, args.events)


if __name__ == '__main__':
    main()
