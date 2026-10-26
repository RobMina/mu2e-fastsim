"""Hash-bound, vectorized coincidence-combination audit on retained cluster members."""
import argparse
from dataclasses import replace
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from . import reco_detail as R
from .pilot import digest


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stock_order_geometry(geom, c):
    """Reproduce the grouped last += in CosmicRayShieldMaker::makeSingleSector."""
    center = np.empty_like(geom.center)
    for s, name in enumerate(geom.sector_names):
        sel = geom.sector == s
        layer_dir = np.asarray(c['crs.layerDirection' + name])
        gap_dir = np.asarray(c['crs.gapDirection' + name])
        off_dir = np.asarray(c['crs.offsetDirection' + name])
        offsets = [np.zeros(3)]
        for layer in range(1, int(c['crs.nLayers'])):
            v = offsets[-1] + c['crs.scintillatorBarThickness'] * layer_dir
            if not c.get('crs.countersOnly' + name, False):
                v = v + c['crs.gapBetweenLayers'][layer - 1] * layer_dir
            offsets.append(v + c['crs.layerOffset'] * off_dir)
        width = c['crs.scintillatorBarWidth']
        vs = (width + c['crs.gapSmall']) * gap_dir
        vl = (width + c['crs.gapLarge']) * gap_dir
        vm = (width + c['crs.gapBetweenModules']) * gap_dir
        nc = int(c['crs.nCountersPerModule' + name])
        m, k = geom.module[sel, None], geom.counter[sel, None]
        p = np.asarray(c['crs.firstCounter' + name]) + np.asarray(offsets)[geom.layer[sel]]
        p = p + m * ((nc // 2 - 1) * vl + (nc // 2) * vs)
        p = p + m * vm
        center[sel] = p + ((k // 2) * vl + ((k + 1) // 2) * vs)
    return replace(geom, center=center)


def combination_mask(x, y, t, cuts):
    """Use native strict rejections, without a tolerance or rounded coordinates."""
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = np.diff(x, axis=1) / np.diff(y, axis=1)
    time_margin = cuts.max_dt - np.ptp(t, axis=1)
    slope_margin = np.minimum(slopes - cuts.min_slope, cuts.max_slope - slopes).min(axis=1)
    diff_margin = cuts.max_slope_diff - np.ptp(slopes, axis=1)
    ok = (time_margin >= 0) & (slope_margin >= 0) & (diff_margin >= 0)
    if not np.isfinite(slopes).all():
        raise ValueError('nonfinite combination slopes')
    return ok, time_margin, slope_margin, diff_margin


def audit_pair(ref, hybrid, geom, stock, c, C):
    a, la, ma = R._read(ref)
    b, lb, mb = R._read(hybrid)
    ev = R._join(a['events'], b['events'], R.IDS)
    if (ev._merge != 'both').any() or (ev.ncluster_a1 != ev.ncluster_hybrid).any():
        raise ValueError('event/cluster census differs')
    pulses = R._join(a['pulses'], b['pulses'], R.KEY)
    if (pulses._merge != 'both').any() or any((pulses[n + '_a1'] != pulses[n + '_hybrid']).any() for n in R.PULSE_ID):
        raise ValueError('pulse identities differ')
    wf = R._join(la['pulses'], lb['pulses'], R.KEY + ['target'])
    if (wf._merge != 'both').any():
        raise ValueError('waveform identity differs')
    clusters = R._join(a['clusters'], b['clusters'], R.KEY)
    if (clusters._merge != 'both').any() or (clusters.sectortype_a1 != clusters.sectortype_hybrid).any():
        raise ValueError('cluster sector order differs')
    edges = R._join(la['clusters'], lb['clusters'], R.KEY + ['target'])
    if (edges._merge == 'right_only').any():
        raise ValueError('candidate has added members; this audit requires subset membership')
    members = la['clusters'].merge(a['clusters'][R.KEY + ['sectortype']], on=R.KEY, validate='many_to_one')
    members = members.merge(a['pulses'].rename(columns={'idx': 'target'}), on=R.IDS + ['target'], validate='many_to_one')
    members = members.merge(b['pulses'][R.KEY + ['time']].rename(columns={'idx': 'target', 'time': 'hybrid_time'}),
                            on=R.IDS + ['target'], validate='many_to_one')
    members = members.merge(edges[R.KEY + ['target', '_merge']], on=R.KEY + ['target'], validate='one_to_one')
    members['removed'] = members._merge == 'left_only'
    members['edge'] = np.arange(len(members), dtype=np.int64)
    members['side'] = members.sipm % 2
    bar = members.bar.to_numpy()
    G, GS = C.sector_geom(geom, c), C.sector_geom(stock, c)
    cfg = C.FinderConfig()
    members['layer'] = geom.layer[bar]
    members['sector'] = geom.sector[bar]
    if (members.seq != 0).any():
        raise ValueError('secondary member found')
    if not np.isfinite(members[['time', 'hybrid_time', 'pe']].to_numpy()).all():
        raise ValueError('nonfinite member pulse')
    covered = {name: np.zeros(len(members), dtype=bool) for name in ('fast_a1time', 'stock_a1time', 'fast_hybridtime', 'stock_hybridtime')}
    best = {name: np.full(len(members), -np.inf) for name in ('time_margin', 'slope_margin', 'slope_diff_margin', 'joint_margin')}
    result = {}
    for st in sorted(members.sectortype.unique()):
        data = members[members.sectortype == st].copy()
        secs = np.unique(data.sector)
        cuts = [cfg.cuts[G.sector_names[s]] for s in secs]
        if any(cut != cuts[0] for cut in cuts):
            raise ValueError('mixed sector cuts not supported')
        cut = cuts[0]
        n = cut.coincidence_layers
        if n not in (3, 4):
            raise ValueError('unexpected layer requirement')
        groups = R.KEY + ['side']
        counts = dict(combinations=0, fast_pass=0, stock_pass=0, rescued=0, lost=0,
                      time_sensitive_fast=0, time_sensitive_stock=0)
        examples = []
        for layers in combinations(range(4), n):
            parts = [data.loc[data.layer == l, groups + ['edge', 'bar', 'time', 'hybrid_time']].rename(
                columns={v: f'{v}_{i}' for v in ('edge', 'bar', 'time', 'hybrid_time')}) for i, l in enumerate(layers)]
            combo = parts[0]
            for part in parts[1:]:
                sizes = combo.groupby(groups).size().to_frame('left').join(part.groupby(groups).size().rename('right'), how='inner')
                if (sizes.left * sizes.right).sum() > 2_000_000:
                    raise ValueError('combination cap exceeded')
                combo = combo.merge(part, on=groups, how='inner')
            if not len(combo):
                continue
            bars = combo[[f'bar_{i}' for i in range(n)]].to_numpy()
            ids = combo[[f'edge_{i}' for i in range(n)]].to_numpy()
            time = combo[[f'time_{i}' for i in range(n)]].to_numpy()
            hybrid_time = combo[[f'hybrid_time_{i}' for i in range(n)]].to_numpy()
            lim = C._type_limits(G, cfg)[int(st)]
            secs = G.sector[bars]
            side = combo.side.to_numpy()[:, None]
            correction = np.where(side == 0, G.pos_side0[secs] - lim['p0'], lim['p1'] - G.pos_side1[secs]) / cfg.fiber_signal_speed
            masks = {}
            margins = {}
            for label, geometry, times in (('fast_a1time', G, time), ('stock_a1time', GS, time),
                                           ('fast_hybridtime', G, hybrid_time), ('stock_hybridtime', GS, hybrid_time)):
                mask, tm, sm, dm = combination_mask(geometry.x[bars], geometry.y[bars], times + correction, cut)
                masks[label] = mask
                margins[label] = (tm, sm, dm)
                covered[label][ids[mask].ravel()] = True
            tm, sm, dm = margins['fast_a1time']
            for field, vals in zip(best, (tm, sm, dm, np.minimum.reduce([tm, sm, dm]))):
                np.maximum.at(best[field], ids.ravel(), np.repeat(vals, n))
            f, s = masks['fast_a1time'], masks['stock_a1time']
            rescue, lost = ~f & s, f & ~s
            counts['combinations'] += len(combo)
            counts['fast_pass'] += int(f.sum())
            counts['stock_pass'] += int(s.sum())
            counts['rescued'] += int(rescue.sum())
            counts['lost'] += int(lost.sum())
            counts['time_sensitive_fast'] += int((f != masks['fast_hybridtime']).sum())
            counts['time_sensitive_stock'] += int((s != masks['stock_hybridtime']).sum())
            ex = combo.loc[rescue | lost].head(max(0, 8 - len(examples))).copy()
            ii = np.flatnonzero(rescue | lost)[:len(ex)]
            for label in ('fast_a1time', 'stock_a1time'):
                for field, vals in zip(('time_margin', 'slope_margin', 'slope_diff_margin'), margins[label]):
                    ex[label + '_' + field] = vals[ii]
            examples.extend(R._records(ex))
        result[str(st)] = dict(counts, examples=examples)
    for label, mask in covered.items():
        members[label] = mask
    summary = members.groupby(['sectortype', 'removed'] + list(covered), dropna=False).size().rename('links').reset_index()
    removed_diagnostics = {}
    for st in sorted(members.sectortype.unique()):
        sel = (members.sectortype == st) & members.removed
        removed_diagnostics[str(st)] = {field: R._stats(vals[sel]) for field, vals in best.items()}
        examples = members.loc[sel, R.KEY + ['target', 'bar', 'layer', 'side', 'time', 'pe']].head(12).copy()
        for field, vals in best.items():
            examples[field] = np.where(np.isfinite(vals[examples.index]), vals[examples.index], np.nan)
        removed_diagnostics[str(st)]['examples'] = R._records(examples)
    delta = stock.center - geom.center
    return dict(removed_diagnostics=removed_diagnostics, events=len(ev), clusters=len(clusters), member_links=len(members), removed_links=int(members.removed.sum()),
                combinations=result, coverage=R._records(summary), tree_cycles=dict(a1=ma, hybrid=mb),
                geometry=dict(changed_bars=int(np.any(delta != 0, axis=1).sum()), max_abs_mm=float(np.abs(delta).max()),
                              by_sector={name: dict(changed_bars=int(np.any(delta[geom.sector == s] != 0, axis=1).sum()),
                                                   max_abs_mm=float(np.abs(delta[geom.sector == s]).max()))
                                         for s, name in enumerate(geom.sector_names)}))


def create(attribution, out, stock_source):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    report, pairs, watched = R._validate(attribution)
    if out.resolve().is_relative_to(Path(attribution).resolve()):
        raise ValueError('output must be outside retained attribution')
    paths = list(watched)
    def one(suffix):
        found = [p for p in paths if p.endswith(suffix)]
        if len(found) != 1:
            raise ValueError(f'expected one bound {suffix}: {found}')
        return found[0]
    geometry_path = one('/crvfastsim/geometry.py')
    coinc_path = one('/crvfastsim/reco/coinc.py')
    counters = one('/crv_counters_extracted_v04.txt')
    sources = [__file__, R.__file__, R.event_shapes.__file__, stock_source]
    for path in sources:
        R._bind(watched, path, digest(path))
    if str(out) in watched:
        raise ValueError('output overlaps input')
    R._check(watched)
    geometry = load_module(geometry_path, '_thread_b_frozen_geometry')
    C = load_module(coinc_path, '_thread_b_frozen_coinc')
    c = geometry.parse_simpleconfig(Path(counters).read_text())
    geom = geometry.from_offline_config(counters)
    stock = stock_order_geometry(geom, c)
    results = {}
    for shard, ref, hybrid, expected in pairs:
        got = audit_pair(ref, hybrid, geom, stock, c, C)
        if got['events'] != expected:
            raise ValueError('event census mismatch')
        results[str(shard)] = got
        print(f'shard {shard}: {got["removed_links"]} removed links', flush=True)
    R._check(watched)
    result = dict(schema_version=1, input_sha256=watched, hashes_verified_before_and_after=True,
                  per_shard=results, limitations=[
                      'Combination audit on retained stock cluster members; not a full finder replay or corrected production geometry.',
                      'No PE filtering or greedy reclustering replay; membership support is necessary evidence, not full output equivalence.',
                      'Stock-order arithmetic is a source-derived diagnostic, not measured internal C++ coordinates.',
                      'No tolerance changes, pulse refits, response retuning, truth conditioning, or downstream uncertainty assignment.',
                      'Historical counters-hash limitation of retained attribution remains.'])
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attribution', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--stock-source', required=True)
    a = p.parse_args(argv)
    create(a.attribution, a.out, a.stock_source)


if __name__ == '__main__':
    main()
