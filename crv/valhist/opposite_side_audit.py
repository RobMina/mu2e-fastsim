"""Replay only stock opposite-side member attachment on retained fast cluster seeds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import membership_boundary as B
from . import reco_detail as R
from .pilot import digest


def prepare_pulses(pulses, G, status, C):
    p = pulses.copy()
    cfg = C.FinderConfig()
    bar = p.bar.to_numpy()
    sector = G.sector[bar]
    p['sectortype'] = G.sector_type[sector]
    p['side'] = p.sipm % 2
    p['layer'] = G.layer[bar]
    p['counter'] = G.counter_in_type[bar]
    cuts = [cfg.cuts[n] for n in G.sector_names]
    p['threshold'] = np.asarray([c.pe_threshold for c in cuts])[sector]
    p['adjacent_dt'] = np.asarray([c.max_dt_adjacent for c in cuts])[sector]
    other = bar * 4 + p.sipm.to_numpy() + np.where(p.sipm.to_numpy() < 2, 2, -2)
    compensate = (status[other] & sum(1 << bit for bit in cfg.compensate_status_bits)) != 0
    p['pe'] = p.pe.to_numpy().astype(float) * np.where(compensate, 2., 1.)
    p['attach_dt'] = 0.
    for st, limits in C._type_limits(G, cfg).items():
        sel = p.sectortype == st
        ss = sector[sel]
        side = p.loc[sel, 'side'].to_numpy()
        correction = np.where(side == 0, G.pos_side0[ss] - limits['p0'], limits['p1'] - G.pos_side1[ss]) / cfg.fiber_signal_speed
        p.loc[sel, 'time'] = p.loc[sel, 'time'].to_numpy() + correction
        p.loc[sel, 'attach_dt'] = limits['max_dt'] + 2 * limits['half'] / cfg.fiber_signal_speed
    return p.reset_index(drop=True)


def filter_hits(p):
    """Vectorized same/adjacent counter sums with the stock inclusive time cut."""
    p = p[p.seq == 0].reset_index(drop=True)
    group = R.IDS + ['sectortype', 'layer', 'side', 'counter']
    total = np.zeros((len(p), 3))
    q = p.assign(row=np.arange(len(p)))
    for col, offset in enumerate((-1, 0, 1)):
        right = q[group + ['time', 'pe']].copy()
        right['counter'] -= offset
        pairs = q[group + ['row', 'time', 'adjacent_dt']].merge(right, on=group, suffixes=('', '_other'))
        pairs = pairs[(pairs.time - pairs.time_other).abs() <= pairs.adjacent_dt]
        np.add.at(total[:, col], pairs.row.to_numpy(), pairs.pe.to_numpy())
    keep = (total[:, 1] + total[:, 0] >= p.threshold) | (total[:, 1] + total[:, 2] >= p.threshold)
    return q.loc[keep].drop(columns='row')


def attach(filtered, all_pulses, links, clusters):
    """Return selected edges, preserving stock minimum-dt then collection-order ties."""
    pulse_key = R.IDS + ['target']
    p = all_pulses.rename(columns={'idx': 'target'})
    anchors = links.merge(p, on=pulse_key, validate='many_to_one')
    occupied = anchors[R.KEY + ['bar', 'side']].drop_duplicates().assign(occupied=True)
    anchors = anchors[anchors.seq == 0]
    used = links[pulse_key].drop_duplicates().assign(used=True)
    candidates = filtered.assign(candidate_order=np.arange(len(filtered))).rename(columns={'idx': 'target'}).merge(
        used, on=pulse_key, how='left', validate='one_to_one')
    candidates = candidates[candidates.used.isna() & (candidates.seq == 0)].drop(columns='used')
    join = R.IDS + ['sectortype', 'bar']
    pair = candidates.merge(anchors[join + ['idx', 'side', 'time']], on=join, suffixes=('', '_anchor'))
    pair = pair[pair.side != pair.side_anchor].copy()
    pair = pair.merge(occupied, on=R.KEY + ['bar', 'side'], how='left', validate='many_to_one')
    pair = pair[pair.occupied.isna()].copy()
    pair['dt'] = (pair.time - pair.time_anchor).abs()
    pair = pair[pair.dt < pair.attach_dt]
    best = pair.sort_values(pulse_key + ['dt', 'idx'], kind='stable').drop_duplicates(pulse_key)
    best = best.sort_values(R.KEY + ['bar', 'sipm', 'dt', 'candidate_order'], kind='stable').drop_duplicates(R.KEY + ['bar', 'sipm'])
    return best[R.KEY + ['target', 'sectortype', 'dt', 'attach_dt']].reset_index(drop=True)


def compare(ref, hybrid, G, status, C):
    a, la, _ = R._read(ref)
    b, lb, _ = R._read(hybrid)
    ev = R._join(a['events'], b['events'], R.IDS)
    if (ev._merge != 'both').any() or (ev.ncluster_a1 != ev.ncluster_hybrid).any():
        raise ValueError('event or cluster counts differ')
    pulses = R._join(a['pulses'], b['pulses'], R.KEY)
    if (pulses._merge != 'both').any() or any((pulses[n + '_a1'] != pulses[n + '_hybrid']).any() for n in R.PULSE_ID):
        raise ValueError('pulse identity mismatch')
    if (R._join(la['pulses'], lb['pulses'], R.KEY + ['target'])._merge != 'both').any():
        raise ValueError('waveform identity mismatch')
    clusters = R._join(a['clusters'], b['clusters'], R.KEY)
    if (clusters._merge != 'both').any() or (clusters.sectortype_a1 != clusters.sectortype_hybrid).any():
        raise ValueError('cluster order mismatch')
    original = R._join(la['clusters'], lb['clusters'], R.KEY + ['target'])
    if (original._merge == 'right_only').any():
        raise ValueError('hybrid has new members')
    removed = original[original._merge == 'left_only'][R.KEY + ['target']]
    results = {}
    for label, raw in (('a1_persisted_pulses', a['pulses']), ('hybrid_persisted_pulses', b['pulses'])):
        p = prepare_pulses(raw, G, status, C)
        selected = attach(filter_hits(p), p, lb['clusters'], b['clusters'])
        expected = R._join(removed, selected[R.KEY + ['target']], R.KEY + ['target'])
        union = pd.concat([lb['clusters'], selected[R.KEY + ['target']]], ignore_index=True)
        residual = R._join(la['clusters'], union, R.KEY + ['target'])
        sector = selected.groupby('sectortype').agg(links=('target', 'size'), min_dt=('dt', 'min'), max_dt=('dt', 'max'), window=('attach_dt', 'first')).reset_index()
        results[label] = dict(selected=len(selected), recovered=int((expected._merge == 'both').sum()),
                              missing=int((expected._merge == 'left_only').sum()), extra=int((expected._merge == 'right_only').sum()),
                              union_residual_links=int((residual._merge != 'both').sum()), sectors=R._records(sector),
                              selected_edges=R._records(selected))
    changed_keys = removed[R.KEY].drop_duplicates()
    changed = clusters.merge(changed_keys, on=R.KEY, validate='one_to_one')
    transitions = changed.groupby(['sectortype_a1', 'poscalc_a1', 'poscalc_hybrid']).size().rename('clusters').reset_index()
    return dict(events=len(ev), clusters=len(clusters), pulse_rows=len(pulses), removed_links=len(removed),
                changed_clusters=len(changed_keys), observed_position_transitions=R._records(transitions), replay=results)


def create(attribution, out, stock_source, stock_fcl):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    report, pairs, watched = R._validate(attribution)
    if out.resolve().is_relative_to(Path(attribution).resolve()):
        raise ValueError('output overlaps retained attribution')
    def one(suffix):
        found = [p for p in watched if p.endswith(suffix)]
        if len(found) != 1:
            raise ValueError(f'nonunique bound {suffix}: {found}')
        return found[0]
    geom_path = one('/crvfastsim/geometry.py')
    coinc_path = one('/crvfastsim/reco/coinc.py')
    counters = one('/crv_counters_extracted_v04.txt')
    calib = report['conditions']['calibration']
    for path in (__file__, R.__file__, B.__file__, R.event_shapes.__file__, stock_source, stock_fcl):
        R._bind(watched, path, digest(path))
    if str(out) in watched:
        raise ValueError('output overlaps input')
    R._check(watched)
    geometry = B.load_module(geom_path, '_thread_b_opposite_geometry')
    C = B.load_module(coinc_path, '_thread_b_opposite_coinc')
    geom = geometry.from_offline_config(counters)
    G = C.sector_geom(geom, geometry.parse_simpleconfig(Path(counters).read_text()))
    calibration = np.loadtxt(calib, delimiter=',', comments='#')
    status = np.zeros(4 * geom.nbars, np.int64)
    ch = calibration[:, 0].astype(int)
    if len(ch) != len(status) or not np.array_equal(np.sort(ch), np.arange(len(status))):
        raise ValueError('incomplete calibration channels')
    status[ch] = calibration[:, 5].astype(np.int64)
    results = {}
    for shard, ref, hybrid, expected in pairs:
        got = compare(ref, hybrid, G, status, C)
        if got['events'] != expected:
            raise ValueError('event count mismatch')
        results[str(shard)] = got
        print(f'shard {shard}: ' + json.dumps({k: {n: r[n] for n in ("selected", "recovered", "missing", "extra", "union_residual_links")} for k, r in got['replay'].items()}), flush=True)
    R._check(watched)
    result = dict(schema_version=1, input_sha256=watched, hashes_verified_before_and_after=True,
                  per_shard=results, limitations=[
                      'Source-derived attachment-only replay on retained fast cluster seeds; no art or full finder rerun.',
                      'A1 and hybrid persisted pulse inputs are separate controls; PE internal doubles are not recovered.',
                      'Position transitions are observed in original paired clusters, not recomputed with stock side-PE rounding.',
                      'No physics source change, reference replacement, acceptance relaxation, or downstream systematic assignment.',
                      'Historical geometry counters binding limitation remains.'])
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('attribution', 'out', 'stock-source', 'stock-fcl'):
        p.add_argument('--' + arg, required=True)
    a = p.parse_args(argv)
    create(a.attribution, a.out, a.stock_source, a.stock_fcl)


if __name__ == '__main__':
    main()
