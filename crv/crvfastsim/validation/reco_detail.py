"""Read-only, hash-bound pulse and coincidence diagnostics on retained A1/hybrid dumps."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from . import event_shapes
from .pilot import digest

IDS = list(event_shapes.IDS)
KEY = IDS + ['idx']
PULSE_ID = ['bar', 'sipm', 'seq']
PULSE_REQUIRED = PULSE_ID + ['flags', 'pe', 'time']
PULSE_OPTIONAL = ['peph', 'height', 'beta', 'chi2', 'ped', 'letime', 'pedfromdb']
CLUSTER_REQUIRED = ['sectortype', 'poscalc', 'pe', 'x', 'y', 'z']
CLUSTER_OPTIONAL = ['start', 'end', 'avgtime', 'slope', 'sidehits0', 'sidehits1',
                    'sidetime0', 'sidetime1', 'sidepe0', 'sidepe1']
QUANTILES = [0, .025, .5, .9, .95, .975, 1]
PE_ATOL = 2 * float(np.finfo(np.float32).eps)
PE_RTOL = 2 * float(np.finfo(np.float32).eps)


def _records(frame):
    return json.loads(frame.to_json(orient='records'))


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return dict(count=len(values), nonfinite=int((~np.isfinite(values)).sum()),
                min=float(finite.min()) if len(finite) else None,
                max=float(finite.max()) if len(finite) else None,
                quantiles=dict(zip(map(str, QUANTILES), np.quantile(finite, QUANTILES).tolist()))
                if len(finite) else {})


def _equal(a, b):
    return np.asarray(a) == np.asarray(b)


def numerical(a, b, *, pe=False):
    """Keep exact disagreement separate from an explicit float32 PE serialization envelope."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    exact = a == b
    delta = b[finite] - a[finite]
    tolerance = PE_ATOL + PE_RTOL * np.maximum(np.abs(a[finite]), np.abs(b[finite])) if pe else 0.
    outside = np.abs(delta) > tolerance
    return dict(rows=len(a), exact_mismatches=int((~exact).sum()),
                nonfinite_pairs=int((~finite).sum()), finite_mismatches=int((delta != 0).sum()),
                beyond_tolerance=int(outside.sum()), atol=PE_ATOL if pe else 0.,
                rtol=PE_RTOL if pe else 0., delta=_stats(delta),
                mismatch_delta=_stats(delta[delta != 0]),
                beyond_tolerance_delta=_stats(delta[outside]))


def _fields(frame, names):
    return {name: numerical(frame[name + '_a1'], frame[name + '_hybrid'],
                            pe=name in ('pe', 'peph', 'sidepe0', 'sidepe1')) for name in names}


def _integers(frame, names, *, nonnegative=True):
    for name in names:
        a = frame[name].to_numpy()
        if a.dtype.kind not in 'iu' or np.any(a > np.iinfo(np.int64).max) or (nonnegative and np.any(a < 0)):
            raise ValueError(f'invalid integer key/field {name}')
        frame[name] = a.astype(np.int64)


def _unique(frame, columns, what):
    if frame.duplicated(columns).any():
        raise ValueError(f'duplicate {what}')


def _join(a, b, columns):
    return a.merge(b, on=columns, how='outer', suffixes=('_a1', '_hybrid'),
                   indicator=True, validate='one_to_one')


def _subset(frame, events):
    return frame.merge(events[IDS].drop_duplicates(), on=IDS, how='inner', validate='many_to_one')


def _without(frame, events):
    return frame.merge(events[IDS].drop_duplicates().assign(_bad=True), on=IDS,
                       how='left', validate='many_to_one').loc[lambda x: x._bad.isna()].drop(columns='_bad')


def _read(path):
    tables, metadata, jagged = {}, {}, {}
    with uproot.open(path) as root:
        for name, required, optional, jag in (
            ('events', ['ndigi', 'npulse', 'ncluster'], [], None),
            ('digis', ['idx', 'bar', 'sipm'], [], None),
            ('pulses', ['idx'] + PULSE_REQUIRED, PULSE_OPTIONAL, 'wfidx'),
            ('clusters', ['idx'] + CLUSTER_REQUIRED, CLUSTER_OPTIONAL, 'pulses')):
            tree, metadata[name] = event_shapes._highest(root, 'crvdump/' + name)
            fields = IDS + required + [n for n in optional if n in tree.keys()]
            arrays = tree.arrays(fields + ([jag] if jag else []), library='ak')
            frame = pd.DataFrame({n: ak.to_numpy(arrays[n]) for n in fields})
            _integers(frame, IDS + (['idx'] if name != 'events' else required))
            _unique(frame, IDS if name == 'events' else KEY, name + ' keys')
            if name == 'pulses':
                _integers(frame, PULSE_ID + ['flags'])
            if name == 'digis':
                _integers(frame, ['bar', 'sipm'])
            if name == 'clusters':
                _integers(frame, ['sectortype', 'poscalc'], nonnegative=False)
            if jag:
                lengths = ak.to_numpy(ak.num(arrays[jag], axis=1))
                rows = np.repeat(np.arange(len(frame)), lengths)
                links = frame.iloc[rows][KEY].reset_index(drop=True)
                links['target'] = ak.to_numpy(ak.flatten(arrays[jag], axis=1))
                _integers(links, ['target'])
                _unique(links, KEY + ['target'], name + ' membership')
                jagged[name] = links
                frame['nlinks'] = lengths
            tables[name] = frame.sort_values(IDS if name == 'events' else KEY).reset_index(drop=True)
    events = tables['events']
    for name, count in [('digis', 'ndigi'), ('pulses', 'npulse'), ('clusters', 'ncluster')]:
        frame = tables[name]
        joined = frame.merge(events[IDS + [count]], on=IDS, how='left', validate='many_to_one')
        if joined[count].isna().any() or (joined['idx'] >= joined[count]).any():
            raise ValueError(f'{name}: key outside full event census/collection')
        observed = frame.groupby(IDS).size().rename('observed').reset_index()
        census = events.merge(observed, on=IDS, how='left').fillna({'observed': 0})
        if not np.array_equal(census[count], census.observed):
            raise ValueError(f'{name}: per-event counts disagree')
    for name, target in [('pulses', 'digis'), ('clusters', 'pulses')]:
        links = jagged[name]
        targets = tables[target][KEY].rename(columns={'idx': 'target'})
        check = links.merge(targets.assign(_exists=True), on=IDS + ['target'], how='left', validate='many_to_one')
        if check._exists.isna().any():
            raise ValueError(f'{name}: invalid membership target')
        if name == 'pulses':
            channels = links.merge(tables['pulses'][KEY + ['bar', 'sipm']], on=KEY)
            channels = channels.merge(tables['digis'][KEY + ['bar', 'sipm']].rename(columns={'idx': 'target'}),
                                      on=IDS + ['target'], suffixes=('_p', '_d'))
            if ((channels.bar_p != channels.bar_d) | (channels.sipm_p != channels.sipm_d)).any():
                raise ValueError('pulse waveform channel differs from referenced digi')
            if (tables['pulses'].nlinks == 0).any():
                raise ValueError('pulse has empty waveform membership')
    return tables, jagged, metadata


def _membership(a, b, clusters, pulses_a, pulses_b):
    merged = _join(a, b, KEY + ['target'])
    merged = merged.merge(clusters[KEY + ['sectortype_a1']], on=KEY, validate='many_to_one')
    changed = merged[merged._merge != 'both']
    removed = changed[changed._merge == 'left_only'].drop(columns='_merge')
    added = changed[changed._merge == 'right_only'].drop(columns='_merge')
    def describe(links, pulses):
        data = links.merge(pulses.rename(columns={'idx': 'target'}), on=IDS + ['target'], validate='many_to_one')
        return dict(links=len(data), pe=_stats(data.pe), time=_stats(data.time),
                    sipm_counts=_records(data.groupby('sipm').size().rename('count').reset_index()))
    changed_keys = changed[KEY].drop_duplicates()
    return dict(common_links=int((merged._merge == 'both').sum()), removed_links=len(removed),
                added_links=len(added), changed_clusters=len(changed_keys),
                removed=_records(removed), added=_records(added),
                removed_pulses=describe(removed, pulses_a), added_pulses=describe(added, pulses_b)), changed_keys


def compare_pair(ref, hybrid):
    """Compare one shard; no packed keys, nearest-neighbour or truth matching."""
    a, la, ma = _read(ref)
    b, lb, mb = _read(hybrid)
    ev = _join(a['events'], b['events'], IDS)
    if (ev._merge != 'both').any():
        raise ValueError('full event identities differ')
    pulses = _join(a['pulses'], b['pulses'], KEY)
    common = pulses._merge == 'both'
    identity_bad = ~common
    identity_counts = {}
    for name in PULSE_ID:
        bad = common & ~_equal(pulses[name + '_a1'], pulses[name + '_hybrid'])
        identity_counts[name] = int(bad.sum())
        identity_bad |= bad
    wf = _join(la['pulses'], lb['pulses'], KEY + ['target'])
    wf_bad = wf[wf._merge != 'both']
    bad_events = pd.concat([pulses.loc[identity_bad, IDS], wf_bad[IDS]], ignore_index=True).drop_duplicates()
    good_events = _without(ev, bad_events)
    aligned = _subset(pulses[common], good_events)
    pulse_names = [n for n in PULSE_REQUIRED + PULSE_OPTIONAL if n in a['pulses'] and n in b['pulses']]
    unequal = np.zeros(len(aligned), dtype=bool)
    for name in pulse_names:
        left, right = aligned[name + '_a1'], aligned[name + '_hybrid']
        unequal |= ~_equal(left, right) | ~np.isfinite(left) | ~np.isfinite(right)
    identical_events = _without(good_events, aligned.loc[unequal, IDS])
    clusters = _join(a['clusters'], b['clusters'], KEY)
    cluster_bad = pd.concat([ev.loc[ev.ncluster_a1 != ev.ncluster_hybrid, IDS],
        clusters.loc[(clusters._merge != 'both') |
                     (clusters.sectortype_a1 != clusters.sectortype_hybrid), IDS]], ignore_index=True).drop_duplicates()
    cluster_events = _without(ev, cluster_bad)
    paired = _subset(clusters, cluster_events)
    membership_events = _subset(cluster_events, good_events)
    member_clusters = _subset(paired, membership_events)
    eligible_la = _subset(la['clusters'], membership_events)
    eligible_lb = _subset(lb['clusters'], membership_events)
    membership, changed_keys = _membership(eligible_la, eligible_lb, member_clusters, a['pulses'], b['pulses'])
    changed_identical = _subset(changed_keys, identical_events)
    cfields = [n for n in CLUSTER_REQUIRED + CLUSTER_OPTIONAL + ['nlinks']
               if n in a['clusters'] and n in b['clusters'] and n != 'sectortype']
    classified = member_clusters.merge(changed_keys.assign(membership_changed=True), on=KEY,
                                      how='left', validate='one_to_one')
    classified['membership_changed'] = classified.membership_changed.eq(True)
    classified = classified.merge(identical_events[IDS].assign(identical_pulses=True), on=IDS,
                                  how='left', validate='many_to_one')
    classified['identical_pulses'] = classified.identical_pulses.eq(True)
    classification = []
    for same_pulses in (False, True):
        for changed_membership in (False, True):
            subset = classified[(classified.identical_pulses == same_pulses) &
                                (classified.membership_changed == changed_membership)]
            classification.append(dict(identical_pulses=same_pulses, membership_changed=changed_membership,
                clusters=len(subset), events=len(subset[IDS].drop_duplicates()), fields=_fields(subset, cfields)))
    sectors = {}
    for sector in np.union1d(a['clusters'].sectortype, b['clusters'].sectortype):
        ca = a['clusters'][a['clusters'].sectortype == sector]
        cb = b['clusters'][b['clusters'].sectortype == sector]
        cp = paired[paired.sectortype_a1 == sector]
        mp = member_clusters[member_clusters.sectortype_a1 == sector]
        member_keys = pd.concat([eligible_la.merge(mp[KEY], on=KEY), eligible_lb.merge(mp[KEY], on=KEY)])
        pulse_keys = member_keys[IDS + ['target']].rename(columns={'target': 'idx'}).drop_duplicates()
        sp = aligned.merge(pulse_keys, on=KEY, validate='one_to_one')
        sectors[str(int(sector))] = dict(a1_clusters=len(ca), hybrid_clusters=len(cb),
            cluster_count_delta=len(cb) - len(ca), paired_clusters=len(cp),
            poscalc_a1=_records(ca.groupby('poscalc').size().rename('count').reset_index()),
            poscalc_hybrid=_records(cb.groupby('poscalc').size().rename('count').reset_index()),
            poscalc_transitions=_records(cp.groupby(['poscalc_a1', 'poscalc_hybrid']).size().rename('count').reset_index()),
            fields=_fields(cp, cfields), pulse_fields=_fields(sp, pulse_names),
            removed_links=int((pd.DataFrame(membership['removed']).get('sectortype_a1', pd.Series(dtype=float)) == sector).sum()),
            added_links=int((pd.DataFrame(membership['added']).get('sectortype_a1', pd.Series(dtype=float)) == sector).sum()))
    return dict(events=len(ev), metadata=dict(a1=ma, hybrid=mb),
        counts={name: numerical(ev[name + '_a1'], ev[name + '_hybrid']) for name in ('ndigi', 'npulse', 'ncluster')},
        totals={arm: {name: int(tables['events'][name].sum()) for name in ('ndigi', 'npulse', 'ncluster')}
                for arm, tables in [('a1', a), ('hybrid', b)]},
        pulse_identity=dict(channel_sequence_mismatches=identity_counts,
            missing_a1_keys=int((pulses._merge == 'right_only').sum()),
            missing_hybrid_keys=int((pulses._merge == 'left_only').sum()),
            waveform_membership_difference_rows=len(wf_bad),
            withheld_events=_records(bad_events), eligible_events=len(good_events), eligible_pulses=len(aligned)),
        pulse_fields=_fields(aligned, pulse_names),
        optional_fields=dict(a1=[n for n in PULSE_OPTIONAL if n in a['pulses']],
                             hybrid=[n for n in PULSE_OPTIONAL if n in b['pulses']]),
        cluster_alignment=dict(label='collection-index diagnostic, not truth matching',
            withheld_events=_records(cluster_bad), eligible_events=len(cluster_events), paired_clusters=len(paired)),
        membership=dict(eligible_events=len(membership_events), **membership), sectors=sectors,
        cluster_strata=classification,
        identical_pulses=dict(definition='exact equality of all common persisted pulse scalar fields and waveform membership',
            common_fields=pulse_names, events=len(identical_events),
            different_membership_clusters=len(changed_identical),
            different_membership_events=len(changed_identical[IDS].drop_duplicates()),
            conclusion='different membership with identical persisted pulses' if len(changed_identical) else
                       'no different membership demonstrated among eligible identical-pulse events'))


def _bind(watched, path, sha):
    path = str(Path(path).absolute())
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f'invalid SHA256 binding: {path}')
    if path in watched and watched[path] != sha:
        raise ValueError(f'conflicting SHA256 binding: {path}')
    watched[path] = sha


def _check(watched):
    for path, sha in watched.items():
        if digest(path) != sha:
            raise ValueError(f'input hash changed: {path}')


def _validate(directory):
    directory = Path(directory).resolve()
    report_path, completion_path = directory / 'attribution.json', directory / 'completion.json'
    watched = {str(completion_path): digest(completion_path)}
    completion = json.loads(completion_path.read_text())
    report = json.loads(report_path.read_text())
    if (completion.get('completed') is not True or completion.get('changed_inputs') != [] or
            report.get('hashes_verified_before_and_after') is not True or report.get('schema_version') != 1):
        raise ValueError('attribution is not completed and hash-verified')
    if not report.get('input_sha256') or completion.get('input_sha256') != report['input_sha256']:
        raise ValueError('completion/report input bindings disagree')
    outputs = completion.get('output_sha256', {})
    if str(report_path) not in outputs or str(directory / 'manifest.json') not in outputs:
        raise ValueError('completion does not bind attribution and manifest')
    for bindings in (report['input_sha256'], outputs, report.get('analysis_source_sha256', {})):
        for path, sha in bindings.items():
            _bind(watched, path, sha)
    shards = report['shards']
    if not shards or any(type(s) is not int for s in shards) or len(set(shards)) != len(shards):
        raise ValueError('invalid attribution shard keys')
    if set(report['hybrids']) != set(map(str, shards)):
        raise ValueError('hybrid shard inventory differs')
    pairs = []
    for shard in shards:
        commands = [c for c in report['commands'] if c['shard'] == shard]
        if len(commands) != 1:
            raise ValueError('missing/duplicate reconstruction command')
        argv = commands[0]['argv']
        hybrid = report['hybrids'][str(shard)]
        if len(argv) < 6 or argv[1:4] != ['-m', 'valhist.fasttables', 'reco'] or argv[5] != hybrid['path']:
            raise ValueError('invalid retained reconstruction command')
        ref = argv[4]
        if ref not in report['input_sha256']:
            raise ValueError('reference is not bound by attribution input_sha256')
        if outputs.get(hybrid['path']) != hybrid['sha256']:
            raise ValueError('hybrid is not bound by completion')
        _bind(watched, ref, report['input_sha256'][ref])
        pairs.append((shard, ref, hybrid['path'], hybrid['events']['rows']))
    _check(watched)
    return report, pairs, watched


def create(attribution, out):
    """Validate every accepted binding before/after reading and exclusively create NEW.json."""
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    report, pairs, watched = _validate(attribution)
    if str(out) in watched or out.resolve().is_relative_to(Path(attribution).resolve()):
        raise ValueError('output must be outside attribution directory and inputs')
    sources = {str(Path(__file__).resolve()): digest(__file__)}
    for module in (event_shapes,):
        sources[str(Path(module.__file__).resolve())] = digest(module.__file__)
    for path, sha in sources.items():
        _bind(watched, path, sha)
    results = {}
    for shard, ref, hybrid, expected in pairs:
        result = compare_pair(ref, hybrid)
        if result['events'] != expected:
            raise ValueError('finished event entries differ from accepted attribution')
        results[str(shard)] = dict(ref=ref, hybrid=hybrid, **result)
    _check(watched)
    result = dict(schema_version=1, attribution=str(Path(attribution).resolve()),
        shards=report['shards'], per_shard=results, input_sha256=watched,
        analysis_source_sha256=sources, hashes_verified_before_and_after=True,
        limitations=[
            'Collection-index diagnostics only; not truth matching. No time/position heuristic matching.',
            'Direct pulse numbers and memberships are withheld for entire events failing channel/sequence/waveform identity.',
            'Cluster index pairing requires equal event cluster counts and identical sector order.',
            'PE tolerance is a float32 serialization envelope, not physics acceptance; exact mismatches remain reported.',
            'Identical persisted pulse fields do not establish identical internal pre-serialization values or algorithms.',
            'Removed/added rows are membership edges, not necessarily lost/created pulses. All sector types are included.',
            'Conditional retained-sample diagnostics; no refit, response replay or causal percentage.'])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def create_pairs(pairs, out):
    """Unbound compare_pair over explicit (reference, candidate) dumps; exclusively create NEW.json."""
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    results = [dict(ref=str(Path(ref).resolve()), cand=str(Path(cand).resolve()), **compare_pair(ref, cand))
               for ref, cand in pairs]
    result = dict(schema_version=1, diagnostic='explicit dump pairs (a1 = reference, hybrid = candidate)',
                  pairs=results, input_sha256={str(Path(p).resolve()): digest(p) for pair in pairs for p in pair})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--attribution')
    group.add_argument('--pair', nargs=2, action='append', metavar=('REF', 'CAND'),
                       help='compare two CrvRecoTableDumper-layout dumps directly (repeatable; no attribution binding)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        create(args.attribution, args.out) if args.attribution else create_pairs(args.pair, args.out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f'Completed read-only reconstruction detail: {args.out}')


if __name__ == '__main__':
    main()
