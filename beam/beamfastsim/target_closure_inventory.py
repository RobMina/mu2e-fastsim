"""Development-only precision inventory of saved inclusive same-start pilot arrays."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from . import cechain as C
from . import cecompare as CC
from .target_inclusive_compare import integer_keys, event_indices
from .target_parent_audit import checksum

# Fixed development-coarse boundaries, with flow cells; never confirmatory bins.
COARSE_EDGES = dict(p=(60., 80., 95., 100., 105.), cos=(-1., -.5, 0., .5, 1.),
                    r=(0., 200., 400., 600., 800.), rmax=(450., 500., 550., 700., 1000.),
                    t=(0., 350., 700., 1000., 1700.))
JOINTS = (('p', 'rmax'), ('p', 'cos'), ('r', 't'))
TAILS = (.25, .5, 1., 2.)
MIN_SUPPORT = 20
METRICS = ('p', 'cos', 'r', 'rmax', 't', 'tof')
MASS = .51099895


def cdf_sup(a, b):
    """Own-selection empirical CDF sup distance, including ties exactly."""
    a, b = np.sort(a), np.sort(b)
    if not a.size or not b.size:
        return None
    points = np.sort(np.concatenate((a, b)))
    return float(np.max(np.abs(np.searchsorted(a, points, side='right') / a.size
                              - np.searchsorted(b, points, side='right') / b.size)))


def joint_tv(a, b, sa, sb, pair):
    """Full emitted-electron measure; cell zero means nonreach/out of selection."""
    x, y = pair
    ex, ey = COARSE_EDGES[x], COARSE_EDGES[y]
    width, size = len(ey) + 1, 1 + (len(ex) + 1) * (len(ey) + 1)
    def counts(o, selected):
        cells = np.zeros(selected.size, dtype=int)
        cells[selected] = (1 + np.searchsorted(ex, o[x][selected], side='right') * width
                           + np.searchsorted(ey, o[y][selected], side='right'))
        return np.bincount(cells, minlength=size)
    ha, hb = counts(a, sa), counts(b, sb)
    return dict(tv=float(np.abs(ha - hb).sum() / (2 * sa.size)),
                g4_counts=ha.tolist(), fast_counts=hb.tolist(), outside_cell=0,
                grid_shape=[len(ex) + 1, width], sparse_threshold=5,
                g4_zero_cells=int((ha == 0).sum()), fast_zero_cells=int((hb == 0).sum()),
                sparse_cells=int(((ha < 5) | (hb < 5)).sum()),
                insufficient_cell_support=bool(((ha < 5) | (hb < 5)).any()))


def event_se(influence, groups):
    """Finite-cluster sandwich SE of already normalized row influences."""
    _, inv = np.unique(groups, return_inverse=True)
    m = int(inv.max()) + 1
    h = np.bincount(inv, weights=influence, minlength=m)
    return float(np.sqrt(m / (m - 1) * np.dot(h, h))) if m > 1 else None


def conditional_difference(sa, sb, ta, tb, groups):
    """Paired conditional-probability difference with both random denominators."""
    na, nb = int(sa.sum()), int(sb.sum())
    pa, pb = (float(ta.sum() / na) if na else None), (float(tb.sum() / nb) if nb else None)
    if not na or not nb:
        return dict(g4=pa, fast=pb, diff=None, event_se=None)
    influence = (tb - pb * sb) / nb - (ta - pa * sa) / na
    return dict(g4=pa, fast=pb, diff=pb - pa, event_se=event_se(influence, groups))


def _projection(se, scale, critical, supported=True):
    projected = se * scale if se is not None and supported else None
    return dict(event_se=se, projected_event_se=projected,
                projected_ci_halfwidth=critical * projected if projected is not None else None)


def _saved(reference, fast_pattern, census_path):
    census = json.loads(Path(census_path).read_text())
    if (census.get('mode') != 'inclusive-matching-background-filter-diagnostic'
            or census.get('generation_census_reconciled') is not True
            or census.get('reduced_reference_sha256') != checksum(reference)):
        raise ValueError('hash-bound inclusive primary census required')
    sha = census.get('generation_source_sha256', '')
    if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('independent generation census hash required')
    with np.load(reference, allow_pickle=False) as saved:
        r = {k: saved[k] for k in saved.files}
    n = r['x'].size
    vectors = ('x', 'y', 'z', 'px', 'py', 'pz', 'time', 'run', 'subrun', 'evt', 'trk',
               'g4_status', 'primary_path_accepted', 'isvd_front', 'xvd_front', 'yvd_front',
               'zvd_front', 'pxvd_front', 'pyvd_front', 'pzvd_front', 'tvd_front')
    if n < 1 or any(r[k].shape != (n,) for k in vectors):
        raise ValueError('reference event/state shapes must be one scalar per emitted electron')
    if r['creation_code'].size != 1 or r['creation_code'].item() not in (166, 179):
        raise ValueError('inclusive DIO/internal RPC creation code required')
    if r['gencount'].shape != (1,) or r['gencount'].dtype.kind not in 'iu':
        raise ValueError('one explicit inclusive generated-event count required')
    rows = integer_keys(np.column_stack([r[k] for k in ('run', 'subrun', 'evt', 'trk')]), 4)
    events, groups = np.unique(rows[:, :3], axis=0, return_inverse=True)
    event_indices(rows, events)
    m = len(events)
    if (np.unique(rows, axis=0).shape[0] != n or n != census['generated_electrons']
            or m != census['generated_events'] or r['gencount'].sum() != m):
        raise ValueError('inclusive identity/count mismatch')
    if not np.all(r['g4_status'] == 0):
        raise ValueError('nonzero G4 status requires separate validity study')
    decisions = r['primary_path_accepted']
    if not np.isin(decisions, [0, 1]).all() or not np.isin(r['isvd_front'], [0, 1]).all():
        raise ValueError('explicit binary decisions and plane flags required')
    passed = np.bincount(groups, weights=decisions, minlength=m)
    if np.any((passed != 0) & (passed != np.bincount(groups, minlength=m))):
        raise ValueError('inconsistent PrimaryPath decision within event')
    paths = sorted(glob.glob(str(fast_pattern)))
    if not paths:
        raise ValueError('no saved fast plane files')
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            part = {k: saved[k] for k in ('idx', 'fate', 'x', 'u', 'T', 't', 't0', 'x0', 'u0', 'p0')}
        length = part['idx'].size
        if any(v.shape != ((length, 3) if k in ('x', 'u', 'x0', 'u0') else (length,))
               for k, v in part.items()):
            raise ValueError('fast state shapes invalid')
        parts.append(part)
    f = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    idx = f['idx']
    if idx.dtype.kind not in 'iu' or not np.array_equal(np.sort(idx), np.arange(n)):
        raise ValueError('fast rows must cover all inclusive primary starts exactly once')
    if (not np.isin(f['fate'], np.arange(len(C.FATES))).all()
            or not np.isfinite(f['T']).all() or np.any(f['T'] < 0)):
        raise ValueError('invalid fast fate/kinetic energy')
    birth = np.column_stack([r[k] for k in ('px', 'py', 'pz')])
    if (any(not np.isfinite(r[k]).all() for k in ('x', 'y', 'z'))
            or any(not np.isfinite(f[k]).all() for k in ('x0', 'u0', 'p0', 't0'))
            or np.any(f['p0'] <= 0) or not np.allclose(np.linalg.norm(f['u0'], axis=1), 1., rtol=0, atol=1e-8)):
        raise ValueError('finite positive reference starts and unit directions required')
    if not (np.allclose(f['x0'], np.column_stack([r[k][idx] for k in ('x', 'y', 'z')]), rtol=0, atol=1e-8)
            and np.allclose(f['p0'][:, None] * f['u0'], birth[idx], rtol=1e-10, atol=1e-8)
            and np.allclose(f['t0'], r['time'][idx], rtol=0, atol=1e-8)
            and np.isfinite(birth).all() and np.isfinite(r['time']).all()):
        raise ValueError('fast transport did not use finite reference starts')
    g = dict(x=np.column_stack([r[k] for k in ('xvd_front', 'yvd_front', 'zvd_front')]),
             t=r['tvd_front'], t0=r['time'], ok=r['isvd_front'].astype(bool))
    pv = np.column_stack([r[k] for k in ('pxvd_front', 'pyvd_front', 'pzvd_front')])
    g['p'] = np.linalg.norm(pv, axis=1)
    g['u'] = pv / np.maximum(g['p'], 1e-12)[:, None]
    f.update(p=np.sqrt(f['T'] * (f['T'] + 2 * MASS)), ok=f['fate'] == C.PLANE)
    for state in (g, f):
        ok = state['ok']
        if any(not np.isfinite(state[k][ok]).all() for k in ('x', 'u', 'p', 't', 't0')) or np.any(state['p'][ok] <= 0):
            raise ValueError('nonfinite or nonpositive reached-plane state')
    return g, f, groups[idx], np.hypot(np.linalg.norm(birth[idx], axis=1), MASS), census, paths


def inventory(reference, fast_pattern, census_path, geom, field, planned_events, critical_value=4.):
    if (not np.isfinite(planned_events) or planned_events <= 0 or int(planned_events) != planned_events
            or not np.isfinite(critical_value) or critical_value <= 0):
        raise ValueError('positive integer planned-events and finite positive critical-value required')
    g, f, groups, born, census, paths = _saved(reference, fast_pattern, census_path)
    n, m = len(groups), int(census['generated_events'])
    scale = float(np.sqrt(m / planned_events))
    obs = []
    for state in (g, f):
        o = CC.observables(geom, field, state)
        full = {}
        for k in METRICS:
            if not np.isfinite(o[k]).all():
                raise ValueError('nonfinite plane observables')
            full[k] = np.full(n, np.nan)
            full[k][state['ok']] = o[k]
        obs.append(full)
    a, b = {k: v[f['idx']] for k, v in obs[0].items()}, obs[1]
    la, lb = born - np.hypot(a['p'], MASS), born - np.hypot(b['p'], MASS)
    report = dict(mode='development-only-saved-inclusive-precision-inventory', development_only=True,
        generated_events=m, generated_electrons=n, planned_events=int(planned_events), critical_value=critical_value,
        reference_sha256=checksum(reference), census_sha256=checksum(census_path),
        generation_source_sha256=census['generation_source_sha256'],
        fast_files=[dict(path=p, sha256=checksum(p)) for p in paths],
        filter_implication_certified=False, generated_yield_validated=False, distribution_validated=False,
        source_library_uncertainty_quantified=False, certificate_established=False, power_established=False,
        confirmatory_binning_established=False, normal_decision_established=False,
        uncertainty_scope='event-statistical planning projections only; parent/library and between-file uncertainty not included',
        energy_loss_definition='truth diagnostic Eborn - Evd in each arm own selection; never a correction input',
        projection_assumption='same event population/multiplicity; SE times sqrt(Npilot_events/Nplanned_events)',
        coarse_edges=COARSE_EDGES, joint_cell_order='outside first, then x-major flow-inclusive grid', cuts={})
    for rc in CC.RMAX_CUTS:
        sa, sb = (a['p'] > CC.P_MIN) & (a['rmax'] > rc), (b['p'] > CC.P_MIN) & (b['rmax'] > rc)
        ones = np.ones(n, bool)
        yields = conditional_difference(ones, ones, sa, sb, groups)
        ng, nf = int(sa.sum()), int(sb.sum())
        ratio = nf / ng if ng else None
        ratio_se = event_se((sb - ratio * sa) / ng, groups) if ng else None
        discordant_events = int(np.unique(groups[sa ^ sb]).size)
        yield_support = min(ng, nf, n - ng, n - nf, discordant_events) >= MIN_SUPPORT
        yields.update(g4_count=ng, fast_count=nf, ratio=ratio, generated_electrons=n, generated_events=m,
                      discordant_events=discordant_events, insufficient_yield_support=not yield_support,
                      min_support=MIN_SUPPORT,
                      **_projection(yields['event_se'], scale, critical_value, yield_support))
        yields['ratio_precision'] = _projection(ratio_se, scale, critical_value, yield_support)
        cut = dict(yield_per_emitted_electron=yields, tails={}, cdf={}, joints={})
        if rc == 450.:
            qa = np.quantile(la[sa], [.16, .5, .84]).tolist() if ng else None
            qb = np.quantile(lb[sb], [.16, .5, .84]).tolist() if nf else None
            cut['loss_core'] = dict(quantile_levels=[.16, .5, .84], g4=qa, fast=qb,
                median_diff=qb[1] - qa[1] if qa and qb else None,
                halfwidth68_diff=((qb[2] - qb[0]) - (qa[2] - qa[0])) / 2 if qa and qb else None)
        for threshold in TAILS:
            ta, tb = sa & (la > threshold), sb & (lb > threshold)
            tail = conditional_difference(sa, sb, ta, tb, groups)
            ka, kb = int(ta.sum()), int(tb.sum())
            ea, eb = np.unique(groups[ta]).size, np.unique(groups[tb]).size
            discord = ta ^ tb
            ed = np.unique(groups[discord]).size
            support = min(ka, kb, ng - ka, nf - kb, ea, eb, ed) >= MIN_SUPPORT
            se = tail['event_se'] if support else None
            tail.update(g4_count=ka, fast_count=kb, g4_selected=ng, fast_selected=nf,
                both_count=int((ta & tb).sum()), g4_only_count=int((ta & ~tb).sum()),
                fast_only_count=int((tb & ~ta).sum()), discordant_count=int(discord.sum()),
                g4_tail_events=int(ea), fast_tail_events=int(eb), discordant_events=int(ed),
                insufficient_tail_support=not support, normal_decision_established=False,
                count_requirements=dict(min_each_tail_and_complement=MIN_SUPPORT,
                    min_each_tail_events=MIN_SUPPORT, min_discordant_events=MIN_SUPPORT),
                planned_events_for_observed_support=(int(np.ceil(m * MIN_SUPPORT /
                    min(ka, kb, ng - ka, nf - kb, ea, eb, ed))) if min(ka, kb, ng - ka, nf - kb, ea, eb, ed) else None),
                **_projection(se, scale, critical_value))
            cut['tails'][str(threshold)] = tail
        for k in METRICS:
            cut['cdf'][k] = dict(g4_selected=ng, fast_selected=nf, empirical_sup=cdf_sup(a[k][sa], b[k][sb]))
        for pair in JOINTS:
            cut['joints'][','.join(pair)] = joint_tv(a, b, sa, sb, pair)
        report['cuts'][str(rc)] = cut
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('ref', 'fast', 'census', 'offline-root', 'out'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--maps', nargs='+', required=True)
    ap.add_argument('--planned-events', type=int, required=True)
    ap.add_argument('--critical-value', type=float, default=4.)
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise FileExistsError('refusing to replace development inventory')
    geom = C.from_offline('Offline/Mu2eG4/geom/geom_run1_a.txt', a.offline_root)
    report = inventory(a.ref, a.fast, a.census, geom, C.field_maps(a.maps), a.planned_events, a.critical_value)
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with open(a.out, 'x') as stream:
        stream.write(text)


if __name__ == '__main__':
    main()
