"""Join retained P4 pulse-fit fields to frozen measured identities; no response or reconstruction replay."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import uproot

from . import joint_compare as J, joint_data as D, joint_timing as T, joint_pairs as P
from . import joint_shape_stats as S, joint_shape_calib as C, validation as V

FIELDS = ('barId', 'SiPMId', 'sequenceIndex', 'crvHitIndex', 'time', 'PEs',
          'pulseBeta', 'pulseFitChi2', 'pulseHeight', 'PEsPulseHeight')
EXTRAS = ('beta', 'chi2', 'pulse_height', 'height_pe', 'entry_index')


def event_rows(keys):
    a = np.asarray(keys)
    J.require(a.ndim == 2 and a.shape[1] == 3 and a.dtype.kind in 'iu' and
              np.all(a >= 0) and np.all(a <= np.iinfo(np.int64).max), 'invalid full event identities')
    return np.ascontiguousarray(a, dtype='<i8').view([(k, '<i8') for k in ('run', 'subrun', 'event')]).reshape(-1)


def join_fields(columns, event_keys, pulses, *, source_index, entry_start):
    """Pulse ordinal refers only to this checksum-bound ROOT vector, never another sample or arm."""
    root_keys = event_rows(event_keys)
    J._unique(root_keys, 'ROOT event')
    captured = J._keys(columns, 'event', ())
    slots = columns['event_slot']
    for slot in range(4):
        J.require(np.array_equal(np.sort(captured[slots == slot]), np.sort(root_keys)),
                  'captured exposure identities differ from the ROOT entry range')
    J.require(np.all(columns['event_source_index'] == source_index) and
              np.all(columns['hit_source_index'] == source_index), 'capture source index mismatch')
    hits = J._keys(columns, 'hit', ())
    J._unique(J._keys(columns, 'hit', ('pulse_index',)), 'saved pulse')
    order = np.argsort(root_keys)
    pos = np.searchsorted(root_keys[order], hits)
    J.require(np.all(pos < len(root_keys)), 'orphan saved hit event')
    idx = order[pos]
    J.require(np.array_equal(root_keys[idx], hits), 'orphan saved hit event')
    counts = ak.to_numpy(ak.num(pulses['barId']))
    J.require(len(counts) == len(root_keys), 'pulse/event length mismatch')
    for values in pulses.values():
        J.require(np.array_equal(counts, ak.to_numpy(ak.num(values))), 'inconsistent pulse field lengths')
    ordinal = columns['hit_pulse_index']
    J.require(ordinal.dtype.kind in 'iu' and np.all(ordinal >= 0) and np.all(ordinal < counts[idx]),
              'saved pulse ordinal out of range')
    flat_index = np.r_[0, np.cumsum(counts)][idx] + ordinal
    values = {name: ak.to_numpy(ak.flatten(array))[flat_index] for name, array in pulses.items()}
    for name, saved in (('barId', 'bar_id'), ('SiPMId', 'sipm_id'), ('sequenceIndex', 'sequence_index'),
                        ('crvHitIndex', 'chosen_cluster'), ('time', 'time'), ('PEs', 'pe_raw')):
        J.require(np.array_equal(values[name], columns['hit_' + saved], equal_nan=True),
                  f'same-product pulse identity/value mismatch: {name}')
    J.require(np.array_equal(np.maximum(values['PEs'], 0), columns['hit_pe'], equal_nan=True),
              'clipped analysis PE differs from source PE')
    out = dict(columns)
    out.update(hit_beta=values['pulseBeta'].astype(float), hit_chi2=values['pulseFitChi2'].astype(float),
               hit_pulse_height=values['pulseHeight'].astype(float), hit_height_pe=values['PEsPulseHeight'].astype(float),
               hit_entry_index=entry_start+idx)
    return out, dict(events=len(root_keys), hits=len(hits), exact_identity_and_values=True,
                     nonfinite_time=int((~np.isfinite(values['time'])).sum()),
                     nonfinite_pe=int((~np.isfinite(values['PEs'])).sum()))


def _subset_pair_rows(rows, mask):
    return {k: v[mask] for k, v in rows.items()}


def data_reports(paths, receipts):
    """Read prior blocks sequentially; retain only data checks, support and immutable input receipts."""
    result = []
    first_data = None
    for path in paths:
        old = J._json(path, receipts)
        J.require(old.get('comparison') == 'joint-pairs' and old.get('passed') is True, 'invalid previous pair report')
        data = old['samples']['data']
        if first_data is not None:
            J.require(data == first_data, 'previous A/B data reductions differ')
        else:
            first_data = data
        result.append(dict(path=str(path), data=old['data'], map=old['map'],
                           support={mode: old['standardization'][mode] for mode in ('band', 'exact')},
                           files={f['path']: f['sha256'] for f in old['files']}))
        del old
    J.require(bool(result), 'need completed pair reports')
    return result, first_data


def summarize(h, rows, reports, events, *, partial=False):
    output = {}
    for report in reports:
        blocks = {}
        for mode in ('band', 'exact'):
            blocks[mode] = {}
            for name in T.G.BODY_NAMES:
                prior = report['support'][mode][name]['all_pe']
                row = S.fixed_support(rows, prior['keys'], prior['data_pair_weights'], mode, events)
                row['historical_data'] = prior['samples']['data']
                if row['supported'] and not partial:
                    J.require(row['pairs'] <= prior['samples']['data']['common_pairs'], 'shape pairs exceed frozen support')
                    row['shape_pair_coverage'] = row['pairs']/prior['samples']['data']['common_pairs']
                    if row['shape_pair_coverage'] == 1:
                        J.require(np.isclose(row['means']['dt2'], prior['samples']['data']['data_weights']['second_moment'],
                                             rtol=1e-10, atol=1e-8), 'fixed support timing moment mismatch')
                blocks[mode][name] = row
        output[report['path']] = blocks
    by_body = {}
    for body, name in enumerate(T.G.BODY_NAMES):
        selected = h['body'] == body
        by_body[name] = dict(pulses=int(selected.sum()), events=len(np.unique(h['entry_index'][selected])),
                            fields={key: J.stats(h[key][selected]) for key in ('beta', 'chi2', 'pulse_height', 'height_pe')},
                            chi2_negative=int(np.count_nonzero(h['chi2'][selected] < 0)),
                            beta_equal_19=int(np.count_nonzero(h['beta'][selected] == 19.)))
    return dict(fixed_support=output, selected_pulses=by_body)


def run(data_dir, previous, out, *, max_shards=None):
    destination = Path(out)
    J.require(not destination.exists(), 'output already exists')
    J.require(max_shards is None or isinstance(max_shards, int) and max_shards > 0, 'invalid shard limit')
    receipts = {}
    for module in (__file__, S.__file__, C.__file__, J.__file__, D.__file__, T.__file__, P.__file__, V.__file__,
                   J.joint.__file__, T.G.__file__, T.R.__file__, T.R.CL.__file__):
        J._receipt(module, receipts)
    reports, old_data = data_reports(previous, receipts)
    root = Path(data_dir).resolve()
    prov = J._json(root / 'provenance.json', receipts)
    manifest = J._json(root / 'summary.json', receipts)
    J._data_policy(prov)
    J.require(prov['complete'] is True and manifest['complete'] is True and
              manifest['provenance']['sha256'] == V.digest(root / 'provenance.json'), 'incomplete capture provenance')
    J.require(len(prov['inputs']) == 1, 'this bounded diagnostic requires exactly one original data file')
    source = prov['inputs'][0]
    J.require(source['source_index'] == 0 and source['entries_finished'] == source['entries_available'],
              'data capture is not a complete single-source file')
    records = manifest['raw_shards']
    expected_npz = {str(root / r['path']): r['sha256'] for r in records}
    J.require(len(expected_npz) == len(records), 'duplicate shard paths')
    for report in reports:
        J.require(report['data']['provenance'] == prov, 'prior report has different data provenance')
        J.require(report['data']['events'] == prov['event_key_count'], 'prior report data event count differs')
        J.require({p: s for p, s in report['files'].items() if p in expected_npz} == expected_npz,
                  'data capture set/hash differs from previous pair report')
        for name in ('provenance.json', 'summary.json'):
            p = str(root / name)
            J.require(report['files'].get(p) == receipts[p]['sha256'], 'previous data metadata hash mismatch')
    asset = reports[0]['map']['asset']
    J.require(all(r['map'] == reports[0]['map'] for r in reports), 'frozen projected map differs across reports')
    mapping = T.channel_map(asset['path'], receipts, asset['sha256'])
    input_path = source['path']
    before = D.digest(input_path)
    J.require(before == source['sha256'], 'ROOT file differs from original capture')
    started = time.monotonic()
    prepared, audits, all_keys, flows = [], [], [], []
    previous_stop = 0
    records_to_read = records if max_shards is None else records[:max_shards]
    with open(input_path, 'rb') as stream, uproot.open(stream) as f:
        tree, cycle = D._highest_tree(f)
        J.require(cycle == source['tree_cycle'] and tree.num_entries == source['entries_available'],
                  'ROOT tree cycle/entry census differs from capture')
        for record in records_to_read:
            a, b = record['entry_start'], record['entry_stop']
            J.require(a == previous_stop and b > a and record['source_index'] == 0, 'noncontiguous capture ranges')
            previous_stop = b
            path = root / record['path']
            columns, summary = J._npz(path, receipts)
            J.require(receipts[str(path)]['sha256'] == record['sha256'] and summary == record['summary'],
                      'data shard hash/summary mismatch')
            J._audit(columns, summary)
            keys = D.read_event_keys(tree, source['event_branches'], a, b)
            arrays = D._arrays(tree, ['crvpulses.' + name for name in FIELDS], a, b)
            p = {name: arrays['crvpulses.' + name] for name in FIELDS}
            joined, audit = join_fields(columns, keys, p, source_index=0, entry_start=a)
            h, flow = T.prepare(joined, mapping, extra_fields=EXTRAS)
            h['group'] = 4*h['entry_index'] + h['slot']
            prepared.append(h)
            all_keys.append(keys)
            audits.append(dict(path=str(path), entry_start=a, entry_stop=b, **audit))
            flows.append(flow)
            print(f'joined entries {a}:{b}, {audit["hits"]} pulses, {len(h["group"])} timing members', flush=True)
            del columns, joined, arrays, p, h
    keys = np.concatenate(all_keys)
    D._unique_keys(keys)
    full = previous_stop == tree.num_entries
    event_sha = hashlib.sha256(keys.tobytes()).hexdigest()
    if full:
        J.require(event_sha == source['event_key_sha256'] and len(keys) == prov['event_key_count'],
                  'full original event key census/hash mismatch')
    h = {name: np.concatenate([piece[name] for piece in prepared]) for name in prepared[0]}
    del prepared
    legacy_rows, pair_flow = P.pair_rows(h)
    summed_flow = {k: sum(f[k] for f in flows) for k in flows[0]}
    if full:
        J.require(summed_flow == old_data['selected_flow'], 'historical timing flow mismatch')
        for name in T.G.BODY_NAMES:
            a, b = pair_flow[name], old_data['pair_flow'][name]
            J.require({k: v for k, v in a.items() if k != 'all_second_moment'} ==
                      {k: v for k, v in b.items() if k != 'all_second_moment'}, 'historical pair flow mismatch')
            am, bm = a['all_second_moment'], b['all_second_moment']
            J.require(am is None and bm is None or am is not None and bm is not None and
                      np.isclose(am, bm, rtol=1e-10, atol=1e-8), 'historical all-pair moment mismatch')
    legacy = {mode: S.compare_cells(P.cell_table(legacy_rows, mode), old_data['strata'][mode], partial=not full)
              for mode in ('band', 'exact')}
    shape_rows, shape_flow = S.pair_rows(h)
    result = summarize(h, shape_rows, reports, len(keys), partial=not full)
    result['model_implied_calibration'] = C.audit(h, np.loadtxt(asset['path'], delimiter=',', comments='#', ndmin=2))
    result.update(schema=1, comparison='joint-retained-pulse-shapes', completed=True, full_file=full,
                  new_response_capture=False, response_changed=False, events=len(keys), event_key_sha256=event_sha,
                  source=dict(path=input_path, sha256=before, tree_cycle=cycle,
                              hash_scope='whole ROOT file before and after analysis'),
                  audits=audits, original_population_replay=legacy, selected_flow=summed_flow, pair_flow=pair_flow,
                  shape_flow=shape_flow, derived_le_factor=S.ALPHA,
                  cautions=['Within-product pulse index plus full identity/value equality, not cross-arm pulse pairing.',
                    'Peak and beta covariance are fitted-observable associations, not an upstream cause or independent resolution.',
                    'Derived LE is t-0.985*beta algebra, not a newly measured threshold crossing or commissioned estimator.',
                    'Area/height PE ratio divided by beta depends on calibration, not independent shape evidence.',
                    'Stored chi2 is an implementation-dependent reduced fit metric; negatives remain explicit, not chi2 probabilities.',
                    'Frozen support and weights exclude support-learning and target-weight uncertainty from conditional event SE.',
                    'P4 is a scoring-reference subset; no holdout calibration or temporal independence certification.',
                    'Historical cabling, conditions and implementation bytes remain uncertified.'])
    J.require(D.digest(input_path) == before, 'ROOT file changed during analysis')
    for record in receipts.values():
        J.require(V.digest(record['path']) == record['sha256'], f'file changed: {record["path"]}')
    result['files'] = list(receipts.values())
    result['elapsed_seconds'] = time.monotonic()-started
    with destination.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--previous', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-shards', type=int)
    args = p.parse_args(argv)
    result = run(args.data, args.previous, args.out, max_shards=args.max_shards)
    print(json.dumps({k: result[k] for k in ('completed', 'full_file', 'events', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
