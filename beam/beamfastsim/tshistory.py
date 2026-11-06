"""Summarize saved selected-lane fast traces alongside sparse exact-ID stock crossings."""
import argparse
import json
from pathlib import Path

import numpy as np

from .tscrossings import TARGETS
from .tsreference import lookup
from .tstrace import LIMITS, SEEDS
from .tstransport import FATES


def _plain(columns):
    return {k: v.tolist() if isinstance(v, (np.ndarray, np.generic)) else v for k, v in columns.items()}


def material_episodes(a, row):
    """Group consecutive same-material calls, retaining chronological sampler order."""
    take = np.flatnonzero(a['material_row'] == row)
    take = take[np.lexsort((a['material_order'][take], a['material_step'][take]))]
    if not len(take):
        return dict(count=0)
    s, name = a['material_step'][take], a['material_name'][take]
    begin = np.r_[0, np.flatnonzero((name[1:] != name[:-1]) | (s[1:] > s[:-1] + 1)) + 1]
    end = np.r_[begin[1:] - 1, len(take) - 1]
    step_rows = np.flatnonzero(a['step_row'] == row)
    step_numbers = a['step_step'][step_rows]
    si = step_rows[np.searchsorted(step_numbers, s[begin])]
    ei = step_rows[np.searchsorted(step_numbers, s[end])]
    used = a['material_requested_mm'][take] * a['material_fraction'][take]
    return _plain(dict(count=len(begin), material=name[begin], first_step=s[begin], last_step=s[end],
                       calls=end - begin + 1, used_mm=np.add.reduceat(used, begin),
                       requested_mm=np.add.reduceat(a['material_requested_mm'][take], begin),
                       stopped=a['material_stopped'][take[end]],
                       p_before=np.linalg.norm(a['material_p_before'][take[begin]], axis=1),
                       p_after=np.linalg.norm(a['material_p_after'][take[end]], axis=1),
                       step_x0=a['step_x0'][si], step_x1=a['step_x1'][ei],
                       t0=a['step_t0'][si], t1=a['step_t1'][ei],
                       path0=a['step_path0'][si], path1=a['step_path1'][ei]))


def x_plane_samples(a, row, planes):
    """All directional fast chord crossings of stock sample x planes, not VD hit matching."""
    take = np.flatnonzero(a['step_row'] == row)
    x0, x1 = a['step_x0'][take], a['step_x1'][take]
    planes = np.asarray(planes)
    delta = x1[:, 0] - x0[:, 0]
    fraction = np.divide(planes[None, :] - x0[:, :1], delta[:, None],
                         out=np.full((len(take), len(planes)), np.nan), where=delta[:, None] != 0)
    hit, plane = np.nonzero((fraction > 0) & (fraction <= 1))
    order = np.lexsort((plane, fraction[hit, plane], hit))
    hit, plane = hit[order], plane[order]
    f, selected = fraction[hit, plane], take[hit]
    x = x0[hit] + f[:, None] * (x1[hit] - x0[hit])
    return _plain(dict(plane=plane, step=a['step_step'][selected], direction=np.sign(delta[hit]).astype(int),
                       fraction=f, x=x, t=a['step_t0'][selected] + f * (a['step_t1'][selected] - a['step_t0'][selected]),
                       p_before=np.linalg.norm(a['step_p0'][selected], axis=1),
                       p_after=np.linalg.norm(a['step_p1'][selected], axis=1)))


def summarize(stock_path, paths):
    with np.load(stock_path, allow_pickle=False) as f:
        stock = {k: f[k] for k in f.files}
    targets = np.asarray(TARGETS)
    if not np.array_equal(stock['target_id'], targets):
        raise ValueError('stock target order/identities differ')
    out = dict(stock=str(stock_path), targets=[], runs=[], limits=LIMITS + [
        'Stock contains only VD/DS pre-step samples, not continuous material histories.',
        'Fast x-plane intersections are chord interpolation at recorded VD3..6 x coordinates, not validated VD hits.',
        'Plane intersections keep repeats and both directions; no nearest-position or time matching.',
        'Crossing momentum is bracketed by fast step endpoints, not interpolated through a material kick.',
        'Episode positions/times bracket accepted steps; they are not exact material entry/exit coordinates.',
        'No stock material thickness can be inferred from the sparse crossing momentum differences.',
    ], units=dict(position='mm', time='ns', momentum='MeV/c', material_path='mm'))
    for i, target in enumerate(targets):
        vd = np.flatnonzero(stock['vd_target'] == i)
        ds = np.flatnonzero(stock['ds_target'] == i)
        if len(ds) != 1:
            raise ValueError('this comparison requires one stock DS record per target')
        if any(np.sum(stock['vd_sid'][vd] == sid) != 1 for sid in (3, 4, 5, 6)):
            raise ValueError('plane comparison requires unique stock VD3..6 records')
        gate = np.array([vd[np.flatnonzero(stock['vd_sid'][vd] == sid)[0]] for sid in (3, 4, 5, 6)])
        p = stock['vd_momentum_mev_c'][gate]
        out['targets'].append(dict(id=target.tolist(), source_p=float(np.linalg.norm(stock['reference_p'][i])),
            stock_ds_p=float(stock['ds_momentum_mev_c'][ds[0]]),
            stock_vd=_plain({k: stock['vd_' + k][vd] for k in ('sid', 'time', 'x', 'y', 'z', 'momentum_mev_c')}),
            stock_ds=_plain({k: stock['ds_' + k][ds] for k in ('time', 'x', 'y', 'z', 'momentum_mev_c')}),
            x_plane_labels=[3, 4, 5, 6], x_planes=stock['vd_x'][gate].tolist(),
            stock_vd4_to_vd5_dp=float(p[1] - p[2])))
    seen = set()
    for path in paths:
        path = Path(path)
        info = json.loads(path.with_suffix('.json').read_text())
        if not info['exact_replay'] or info['seed'] not in SEEDS or info['seed'] in seen:
            raise ValueError('unique selected exact-replay traces required')
        seen.add(info['seed'])
        with np.load(path, allow_pickle=False) as f:
            a = {k: f[k] for k in f.files}
        index = lookup(a['target_id'], targets)
        if np.any(index < 0):
            raise ValueError('missing trace target')
        run = dict(seed=info['seed'], path=str(path), exact_replay=True, targets=[])
        for i, j in enumerate(index):
            row = a['target_row'][j]
            episodes = material_episodes(a, row)
            used = np.sum(episodes.get('used_mm', []))
            if not np.isclose(used, a['terminal_material_path'][j], rtol=1e-12, atol=1e-12):
                raise ValueError('trace material accounting differs from terminal output')
            crossings = x_plane_samples(a, row, out['targets'][i]['x_planes'])
            vd = np.flatnonzero(stock['vd_target'] == i)
            gate = np.array([vd[np.flatnonzero(stock['vd_sid'][vd] == sid)[0]] for sid in (3, 4, 5, 6)])
            matches = gate[np.asarray(crossings['plane'], dtype=int)]
            stock_x = np.column_stack([stock['vd_' + k][matches] for k in ('x', 'y', 'z')])
            delta = np.asarray(crossings['x']).reshape(-1, 3) - stock_x
            crossings.update(_plain(dict(stock_sid=stock['vd_sid'][matches], fast_minus_stock_x=delta,
                distance_mm=np.linalg.norm(delta, axis=1),
                fast_minus_stock_t=np.asarray(crossings['t']) - stock['vd_time'][matches])))
            run['targets'].append(dict(id=targets[i].tolist(), fate=FATES[a['terminal_fate'][j]],
                terminal=_plain({k: a['terminal_' + k][j] for k in ('x', 'p', 't', 'path', 'material_path', 'steps')}),
                terminal_pmag=float(np.linalg.norm(a['terminal_p'][j])),
                episodes=episodes, plane_crossings=crossings))
        out['runs'].append(run)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock', required=True, type=Path)
    parser.add_argument('--traces', nargs='+', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    report = summarize(args.stock, args.traces)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    for run in report['runs']:
        for t in run['targets']:
            e = t['episodes']
            print(json.dumps(dict(seed=run['seed'], id=t['id'], fate=t['fate'],
                                  material=e.get('material', []), used_mm=e.get('used_mm', []),
                                  p_before=e.get('p_before', []), p_after=e.get('p_after', []))))


if __name__ == '__main__':
    main()
