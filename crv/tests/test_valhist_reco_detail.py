"""Synthetic retained-dump diagnostics; never launch reconstruction or a remote process."""
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from valhist import reco_detail as R
from valhist.pilot import digest


def dump(path, *, change=False, seq=False, wf=False, sector=False, count=False,
         invalid=None, pe_shift=0., time_shift=0., empty=False, reorder=False, high_cycle=False):
    ev = np.array([1, 1, 1 << 25], dtype=np.int64)
    runs = np.array([1400, 1401, 1400], dtype=np.int64)
    def ids(rows):
        return dict(run=runs[rows], subrun=np.full(len(rows), 11, np.int64), event=ev[rows])
    p_rows = np.array([], dtype=int) if empty else np.array([0, 0, 1, 1])
    c_rows = np.array([], dtype=int) if empty else np.array([0, 1])
    if count:
        c_rows = c_rows[:1]
    npulse, nc = len(p_rows), len(c_rows)
    idx = np.tile([0, 1], npulse // 2).astype(np.int64)
    if invalid == 'duplicate':
        idx[1] = 0
    if invalid == 'negative':
        idx[0] = -1
    if invalid == 'float':
        idx = idx.astype(float)
    pulse_event = ids(p_rows)
    if invalid == 'orphan':
        pulse_event['event'][0] = 999
    waveform = np.tile([1, 0] if wf else [0, 1], npulse // 2).astype(np.int64)
    if invalid == 'wf':
        waveform[0] = 9
    targets = np.ones(nc, np.int64) if change else np.zeros(nc, np.int64)
    if invalid == 'member':
        targets[0] = 9
    tables = {
        'events': dict(**ids(np.arange(3)), ndigi=np.bincount(p_rows, minlength=3),
                       npulse=np.bincount(p_rows, minlength=3), ncluster=np.bincount(c_rows, minlength=3)),
        'digis': dict(**ids(p_rows), idx=np.tile([0, 1], npulse // 2).astype(np.int64),
                      bar=np.full(npulse, 12, np.int64), sipm=np.zeros(npulse, np.int64)),
        'pulses': dict(**pulse_event, idx=idx, bar=np.full(npulse, 12, np.int64),
                       sipm=np.zeros(npulse, np.int64), seq=np.full(npulse, int(seq), np.int64),
                       flags=np.zeros(npulse, np.int64), pe=np.full(npulse, 25 + pe_shift, np.float32),
                       time=np.full(npulse, 100 + time_shift, np.float64),
                       wfidx=ak.unflatten(waveform, np.ones(npulse, np.int64))),
        'clusters': dict(**ids(c_rows), idx=np.zeros(nc, np.int64),
                         sectortype=np.array([2, 7][:nc], np.int64) + int(sector),
                         poscalc=np.full(nc, int(not change), np.int64),
                         pe=np.full(nc, 25 + pe_shift, np.float32), x=np.full(nc, int(change), np.float32),
                         y=np.zeros(nc, np.float32), z=np.zeros(nc, np.float32),
                         pulses=ak.unflatten(targets, np.ones(nc, np.int64)))
    }
    if invalid == 'census':
        tables['events']['npulse'][0] += 1
    if invalid == 'duplicate_event':
        tables['events']['run'][1] = 1400
    if invalid == 'nonfinite':
        tables['pulses']['time'][0] = np.nan
    if reorder:
        tables = {name: {k: v[::-1] for k, v in table.items()} for name, table in tables.items()}
    with uproot.recreate(path) as root:
        for name, table in tables.items():
            root.mktree('crvdump/' + name, table)
        if high_cycle:
            root.mktree('crvdump/events', tables['events'])


@pytest.fixture
def pair(tmp_path):
    a, b = tmp_path / 'a1.root', tmp_path / 'hybrid.root'
    dump(a)
    dump(b, change=True)
    return a, b


def test_identical_pulses_different_membership_all_sectors_and_full_ids(pair):
    got = R.compare_pair(*pair)
    assert got['events'] == 3
    assert got['totals']['a1']['ncluster'] == got['totals']['hybrid']['ncluster'] == 2
    assert got['pulse_fields']['pe']['exact_mismatches'] == 0
    assert got['identical_pulses']['different_membership_clusters'] == 2
    assert got['membership']['removed_links'] == got['membership']['added_links'] == 2
    assert {r['run'] for r in got['membership']['removed']} == {1400, 1401}
    assert {r['target'] for r in got['membership']['removed']} == {0}
    assert {r['target'] for r in got['membership']['added']} == {1}
    assert set(got['sectors']) == {'2', '7'}
    assert got['sectors']['2']['poscalc_transitions'] == [dict(poscalc_a1=1, poscalc_hybrid=0, count=1)]
    assert got['counts']['ncluster']['exact_mismatches'] == 0


@pytest.mark.parametrize('mode', ['seq', 'wf'])
def test_identity_mismatch_withholds_entire_event_numerics(pair, mode):
    dump(pair[1], change=True, **{mode: True}, pe_shift=10)
    got = R.compare_pair(*pair)
    assert len(got['pulse_identity']['withheld_events']) == 2
    assert got['pulse_fields']['pe']['rows'] == 0
    assert got['membership']['removed_links'] == 0
    assert got['membership']['eligible_events'] == 1
    assert got['cluster_alignment']['paired_clusters'] == 2


@pytest.mark.parametrize('mode', ['sector', 'count'])
def test_cluster_alignment_gate(pair, mode):
    dump(pair[1], change=True, **{mode: True})
    got = R.compare_pair(*pair)
    expected = 0 if mode == 'sector' else 1
    assert got['cluster_alignment']['paired_clusters'] == expected
    assert got['membership']['removed_links'] == expected
    assert got['pulse_fields']['pe']['rows'] == 4


@pytest.mark.parametrize('invalid', ['duplicate', 'negative', 'float', 'orphan', 'wf', 'member', 'census', 'duplicate_event'])
def test_invalid_keys_and_membership_rejected(pair, invalid):
    dump(pair[1], invalid=invalid)
    with pytest.raises(ValueError):
        R.compare_pair(*pair)


def test_empty_collections_and_highest_cycle(pair):
    dump(pair[0], empty=True, high_cycle=True)
    dump(pair[1], empty=True)
    got = R.compare_pair(*pair)
    assert got['events'] == 3 and got['sectors'] == {}
    assert got['metadata']['a1']['events']['key'].endswith(';2')


def test_row_order_not_identity_and_explicit_pe_tolerance(pair):
    dump(pair[1], reorder=True, pe_shift=np.spacing(np.float32(25)), time_shift=.001)
    got = R.compare_pair(*pair)
    assert not got['pulse_identity']['withheld_events']
    pe = got['pulse_fields']['pe']
    assert pe['exact_mismatches'] == 4 and pe['beyond_tolerance'] == 0
    assert pe['mismatch_delta']['min'] > 0
    assert got['pulse_fields']['time']['beyond_tolerance'] == 4
    assert got['identical_pulses']['different_membership_events'] == 0
    assert got['pulse_fields']['time']['mismatch_delta']['quantiles']['0.5'] > 0


def test_nonfinite_values_are_explicit_not_json_nan(pair):
    dump(pair[1], invalid='nonfinite')
    got = R.compare_pair(*pair)
    assert got['pulse_fields']['time']['nonfinite_pairs'] == 1
    json.dumps(got, allow_nan=False)


@pytest.fixture
def report(tmp_path, pair):
    directory = tmp_path / 'attribution'
    directory.mkdir()
    helper = tmp_path / 'frozen.py'
    helper.write_text('# frozen\n')
    manifest, log = directory / 'manifest.json', directory / 'reco_11.log'
    manifest.write_text('{}')
    log.write_text('finished\n')
    a, b = map(str, pair)
    inputs = {a: digest(a), str(helper): digest(helper)}
    record = dict(schema_version=1, shards=[11], input_sha256=inputs,
                  analysis_source_sha256={str(helper): digest(helper)},
                  hashes_verified_before_and_after=True,
                  commands=[dict(shard=11, argv=['python3', '-m', 'valhist.fasttables', 'reco', a, b])],
                  hybrids={'11': dict(path=b, sha256=digest(b), events=dict(rows=3))})
    rp = directory / 'attribution.json'
    rp.write_text(json.dumps(record))
    outputs = {str(p): digest(p) for p in [rp, manifest, log, pair[1]]}
    cp = directory / 'completion.json'
    cp.write_text(json.dumps(dict(completed=True, changed_inputs=[], input_sha256=inputs, output_sha256=outputs)))
    return directory, tmp_path / 'detail.json', helper, log


def test_create_and_no_overwrite(report):
    directory, out, _, _ = report
    got = R.create(directory, out)
    assert got['hashes_verified_before_and_after'] and out.exists()
    assert got['per_shard']['11']['events'] == 3
    with pytest.raises(FileExistsError):
        R.create(directory, out)


@pytest.mark.parametrize('target', ['helper', 'log', 'a1', 'hybrid', 'manifest', 'attribution'])
def test_all_bound_hashes_checked_before_output(report, pair, target):
    directory, out, helper, log = report
    p = {'helper': helper, 'log': log, 'a1': pair[0], 'hybrid': pair[1],
         'manifest': directory / 'manifest.json', 'attribution': directory / 'attribution.json'}[target]
    if target == 'attribution':
        p.write_text(p.read_text() + ' ')
    else:
        with p.open('ab') as stream:
            stream.write(b'drift')
    with pytest.raises(ValueError, match='hash changed'):
        R.create(directory, out)
    assert not out.exists()


def test_mid_read_drift_never_publishes(report, monkeypatch):
    directory, out, helper, _ = report
    compare = R.compare_pair
    def drift(*args):
        result = compare(*args)
        helper.write_text('# drift\n')
        return result
    monkeypatch.setattr(R, 'compare_pair', drift)
    with pytest.raises(ValueError, match='hash changed'):
        R.create(directory, out)
    assert not out.exists()


def test_reference_cannot_be_unbound_even_if_completion_is_rewritten(report, pair):
    directory, out, _, _ = report
    rp, cp = directory / 'attribution.json', directory / 'completion.json'
    record, completion = json.loads(rp.read_text()), json.loads(cp.read_text())
    del record['input_sha256'][str(pair[0])]
    completion['input_sha256'] = record['input_sha256']
    rp.write_text(json.dumps(record))
    completion['output_sha256'][str(rp)] = digest(rp)
    cp.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match='reference is not bound'):
        R.create(directory, out)


def test_incomplete_report_rejected(report):
    directory, out, _, _ = report
    cp = directory / 'completion.json'
    record = json.loads(cp.read_text())
    record['completed'] = False
    cp.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='not completed'):
        R.create(directory, out)


def test_missing_pulse_keys_withhold_not_nearest_neighbour(pair):
    dump(pair[1], empty=True)
    got = R.compare_pair(*pair)
    assert got['pulse_identity']['missing_hybrid_keys'] == 4
    assert got['pulse_fields']['pe']['rows'] == 0
    assert got['cluster_alignment']['paired_clusters'] == 0


def test_channel_identity_change_with_consistent_waveform(pair):
    with uproot.open(pair[1]) as root:
        pulse = root['crvdump/pulses'].arrays(library='ak')
        digi = root['crvdump/digis'].arrays(library='ak')
    with uproot.update(pair[1]) as root:
        for name, arr in [('pulses', pulse), ('digis', digi)]:
            fields = [n for n in arr.fields if n != 'nwfidx']
            data = {n: arr[n] for n in fields}
            data['bar'] = arr.bar + 1
            root.mktree('crvdump/' + name, data)
    got = R.compare_pair(*pair)
    assert got['pulse_identity']['channel_sequence_mismatches']['bar'] == 4
    assert got['pulse_fields']['pe']['rows'] == 0


def test_membership_stratified_cluster_values(pair):
    got = R.compare_pair(*pair)
    groups = got['cluster_strata']
    assert len(groups) == 4
    group = next(g for g in groups if g['identical_pulses'] and g['membership_changed'])
    assert group['clusters'] == 2 and group['fields']['x']['exact_mismatches'] == 2


def test_full_event_identity_difference_is_not_inner_joined_away(pair):
    with uproot.open(pair[1]) as root:
        arrays = root['crvdump/events'].arrays(library='np')
    arrays['event'][-1] += 1
    with uproot.update(pair[1]) as root:
        root.mktree('crvdump/events', arrays)
    with pytest.raises(ValueError, match='full event identities differ'):
        R.compare_pair(*pair)


def test_tolerance_does_not_hide_large_pe_shift(pair):
    dump(pair[1], pe_shift=.01)
    got = R.compare_pair(*pair)
    assert got['pulse_fields']['pe']['beyond_tolerance'] == 4
    assert got['pulse_fields']['pe']['beyond_tolerance_delta']['min'] > 0
