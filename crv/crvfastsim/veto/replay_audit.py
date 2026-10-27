"""Read-only current-finder, paired-frame and conditional replica diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .apply import gate
from .card import VetoCard
from .compareframes import dump_frames
from .deadtime import frame_coverage, read_coincs
from .frameaudit import track_histogram
from .replay import sha256
from .replay_metrics import paired_stats, validate_frames, verify_metadata
from .replay_recover import IDS, prepare_pulses
from valhist.membership_replay import compare_tables

BLOCKS = ('block0', 'block1', 'block2_fullinventory')


def read_json(path):
    return json.loads(Path(path).read_text())


def bind(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'changed input: {path}')


def weights(reference, selection):
    config = dict(reference['card'])
    expected = config.pop('sha')
    card = VetoCard.from_dict(config)
    if card.sha() != expected:
        raise ValueError('changed card')
    ev, _ = read_coincs([str(selection)])
    t, p = ak.to_numpy(ak.flatten(ev.t_mid)), ak.to_numpy(ak.flatten(ev.p_mid))
    sel = np.ones(len(t), bool)
    if reference.get('pmin') is not None:
        sel &= p > reference['pmin']
    if reference.get('pmax') is not None:
        sel &= p < reference['pmax']
    if sel.sum() != reference['selected']:
        raise ValueError('selected track count changed')
    return card, *track_histogram(t[sel], reference['F']['grid_ns'])


def tables(clusters, events, pulse_channels=None):
    """Native membership for identical pulse draws, channel sets otherwise (never truth)."""
    ids = pd.DataFrame({k: ak.to_numpy(events[k]) for k in IDS})
    if ids.duplicated().any():
        raise ValueError('duplicate event')
    ids['ev'] = np.arange(len(ids))
    names = [*IDS, 'idx', 'sectortype', 'pe', 'start', 'end', 'avgtime']
    c = pd.DataFrame({k: ak.to_numpy(clusters[k]) for k in names})
    c = c.merge(ids, on=list(IDS), validate='many_to_one')
    if len(c) != len(clusters):
        raise ValueError('cluster outside census')
    c['cid'] = np.arange(len(c))
    n = ak.to_numpy(ak.num(clusters.pulses))
    target = ak.to_numpy(ak.flatten(clusters.pulses))
    m = pd.DataFrame({'cid': np.repeat(c.cid.to_numpy(), n), 'target': target})
    if pulse_channels is not None:
        m['ev'] = np.repeat(c.ev.to_numpy(), n)
        m = m.merge(pulse_channels.rename(columns={'idx': 'target'}), on=['ev', 'target'],
                    how='left', validate='many_to_one')
        if m.chan.isna().any():
            raise ValueError('member outside pulse collection')
        m = m[['cid', 'chan']].rename(columns={'chan': 'target'}).drop_duplicates()
    return c[['ev', 'cid', 'sectortype', 'pe', 'start', 'end', 'avgtime']], m


def equal_pulses(first, second):
    """Require the full ordered retained pulse collection, not nearest-time matching."""
    return set(ak.fields(first)) == set(ak.fields(second)) and len(first) == len(second) and all(
        np.array_equal(ak.to_numpy(first[k]), ak.to_numpy(second[k])) for k in ak.fields(first))


def replica_stats(first, second, stock, groups):
    """Average replicas per original frame; never count reused stock twice."""
    a, b, s = (np.asarray(x, float) for x in (first, second, stock))
    if a.shape != b.shape or a.shape != s.shape or a.ndim != 1:
        raise ValueError('replica coverage vectors must align')
    paired_stats(a, s)
    paired_stats(b, s)
    if np.asarray(groups).shape != a.shape:
        raise ValueError('one block label per unique frame required')
    result = paired_stats((a + b) / 2, s)
    result['unique_frames'] = len(a)
    result['fast_draws_per_frame'] = 2
    result['replica_contrast'] = paired_stats(b, a)
    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    n = np.bincount(inv)
    sums = np.bincount(inv, weights=(a + b) / 2 - s)
    g = len(n)
    result['delta_err_blocks'] = float(np.sqrt(g / (g - 1) * np.sum(
        (sums - n * result['delta']) ** 2)) / len(a)) if g > 1 else None
    result['scope'] = 'Two conditional fast draws on the same frames/library and one stock draw; not finite-library uncertainty.'
    return result


def run(base, candidate, replica, out):
    base, candidate, replica, out = map(Path, (base, candidate, replica, out))
    if out.exists():
        raise ValueError('choose new output directory')
    paired = read_json(candidate / 'paired.json')
    selection, reference_path = Path(paired['selection']), Path(paired['reference'])
    bind(selection, paired['selection_sha256'])
    bind(reference_path, paired['reference_sha256'])
    card, hist, edges = weights(read_json(reference_path), selection)
    rows, ledgers, first, second, stocks, groups = [], [], [], [], [], []
    for i, name in enumerate(BLOCKS):
        path = candidate / name
        check, meta = read_json(path / 'check.json'), read_json(path / 'fast.json')
        verify_metadata(check, meta, sha256(path / 'fast.json'))
        bind(path / 'fast.parquet', check['fast_sha256'])
        bind(check['stock'], check['stock_sha256'])
        bind(meta['pulse_file'], meta['pulse_sha256'])
        bind(base / name / 'mixed.root', check['input_sha256'])
        fast, stock = ak.from_parquet(path / 'fast.parquet'), dump_frames([check['stock']])
        validate_frames(fast, stock, check, meta)
        with uproot.open(base / name / 'mixed.root') as f:
            ev = f['dump/events'].arrays(library='ak')
        recovery = read_json(path / 'recovery/recovery.json')
        for source in recovery['sources'].values():
            bind(source['path'], source['sha256'])
        bind(path / 'recovery/recovered.parquet', recovery['output_sha256']['recovered.parquet'])
        clusters = ak.from_parquet(path / 'recovery/recovered.parquet')
        pulse = ak.from_parquet(meta['pulse_file'])
        oldmeta = read_json(base / name / 'fast.json')
        bind(oldmeta['pulse_file'], oldmeta['pulse_sha256'])
        exact = equal_pulses(pulse, ak.from_parquet(oldmeta['pulse_file']))
        if not exact:
            raise ValueError('current and retained fast pulse draws differ; native membership not comparable')
        old = read_json(base / f'reco_only_20261025/fast{i}/check_final.json')
        for source in old['bindings'].values():
            bind(source['path'], source['sha256'])
        with uproot.open(old['bindings']['reco']['path']) as f:
            control = f['crvdump/clusters'].arrays(library='ak')
        membership = compare_tables(tables(control, ev), tables(clusters, ev), len(ev))
        if not membership['membership_equal']:
            raise ValueError('current finder differs from retained same-pulse stock finder')
        control_frames = dump_frames([old['bindings']['reco']['path']])
        fc, sc = (frame_coverage(x.coincs, card, hist, edges) for x in (fast, stock))
        cc = frame_coverage(control_frames.coincs, card, hist, edges)
        replica_path = replica / name
        rcheck, rmeta = read_json(replica_path / 'check.json'), read_json(replica_path / 'fast.json')
        verify_metadata(rcheck, rmeta, sha256(replica_path / 'fast.json'))
        bind(replica_path / 'fast.parquet', rcheck['fast_sha256'])
        if (rcheck['input_sha256'] != check['input_sha256'] or rcheck['stock_sha256'] != check['stock_sha256']
                or rcheck['response_identity_sha256'] != check['response_identity_sha256']
                or rmeta['seed'] == meta['seed']):
            raise ValueError('replica requires same inputs/response and different seed')
        rf = ak.from_parquet(replica_path / 'fast.parquet')
        validate_frames(rf, stock, rcheck, rmeta)
        rc = frame_coverage(rf.coincs, card, hist, edges)
        batch = prepare_pulses(pulse, ev)
        pc = pd.DataFrame({'ev': np.repeat(np.arange(len(ev)), np.diff(batch.offsets)),
                           'idx': batch.pulses['idx'], 'chan': batch.pulses['chan']})
        with uproot.open(check['stock']) as f:
            stc = f['crvdump/clusters'].arrays(library='ak')
            stp = f['crvdump/pulses'].arrays(['run', 'subrun', 'event', 'idx', 'bar', 'sipm'], library='pd')
        census = pd.DataFrame({k: ak.to_numpy(ev[k]) for k in IDS}).assign(ev=np.arange(len(ev)))
        stp = stp.merge(census, on=list(IDS), validate='many_to_one')
        stp['chan'] = stp.bar * 4 + stp.sipm
        observed = compare_tables(tables(stc, ev, stp[['ev', 'idx', 'chan']]), tables(clusters, ev, pc), len(ev))
        columns = {k: ak.to_numpy(ev[k]) for k in IDS}
        columns.update(block=np.full(len(ev), i), fast_coverage=fc, stock_coverage=sc,
                       control_coverage=cc, replica_coverage=rc, fast_gate=gate(fast.coincs, card),
                       stock_gate=gate(stock.coincs, card), fast_n=ak.to_numpy(ak.num(fast.coincs.time)),
                       stock_n=ak.to_numpy(ak.num(stock.coincs.time)))
        ledgers.append(ak.Array(columns))
        rows.append({'block': name, 'ordered_pulses_exact': exact, 'same_pulse_stock_finder': membership,
                     'coverage_exact_to_control': bool(np.array_equal(fc, cc)),
                     'observed_channel_sets': observed,
                     'observed_scope': 'Channel sets across independent response draws; no pulse or truth identity.',
                     'fast_gate_fail': int((~columns['fast_gate']).sum()),
                     'stock_gate_fail': int((~columns['stock_gate']).sum()),
                     'recovery_sha256': sha256(path / 'recovery/recovery.json'),
                     'control_check_sha256': sha256(base / f'reco_only_20261025/fast{i}/check_final.json')})
        first.append(fc); second.append(rc); stocks.append(sc); groups.append(np.full(len(ev), i))
    report = {'blocks': rows, 'replicas': replica_stats(*(np.concatenate(x) for x in (first, second, stocks, groups))),
              'candidate_metrics_sha256': sha256(candidate / 'paired.json'),
              'replica_metrics_sha256': sha256(replica / 'paired.json'), 'code_sha256': sha256(__file__)}
    out.mkdir(parents=True)
    ak.to_parquet(ak.concatenate(ledgers), out / 'frames.parquet')
    report['frames_sha256'] = sha256(out / 'frames.parquet')
    with (out / 'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report['replicas'], indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'candidate', 'replica', 'out'):
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args(argv)
    run(args.base, args.candidate, args.replica, args.out)


if __name__ == '__main__':
    main()
