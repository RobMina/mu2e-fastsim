"""Saved edge-reference bulk/tail, first-exit and adjacent-step diagnostics.

No transport or altered acceptance gate. Fixed energy caps/bins and topology
partitions are diagnostic estimands, not fitted corrections. Exit angles and
chord correlations condition on each arm's own outcome; G4 chords are not true
path lengths. A process ending a step does not identify every interaction in it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom
import uproot

from . import edgeg4, foilreport as fr, slabrefs, lossexplain as le, slabdiffusion as sd
from .reference_audit import _identity

ENERGY_EDGES = np.array([0., .1, .2, .3, .5, .75, 1., np.inf])
CAPS = (.3, .5, .75, 1.)
ANGLE_EDGES = np.array([0., .05, .1, .2, .3, .4, .6, .8, 1.2, np.pi])


def vector(x):
    x = np.asarray(x, float)
    if x.ndim != 1 or len(x) < 2 or np.any(~np.isfinite(x)):
        raise ValueError('at least two finite independent samples required')
    return x


def estimate(x):
    x = vector(x)
    return dict(n=len(x), mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))))


def probability(mask):
    mask = np.asarray(mask, bool)
    n, k = len(mask), int(mask.sum())
    if not n:
        raise ValueError('nonempty probability population required')
    return dict(n=n, count=k, fraction=k/n,
                interval_99pct=[0. if not k else float(beta.ppf(.005, k, n-k+1)),
                                1. if k == n else float(beta.ppf(.995, k+1, n-k))])


def variance_parts(y, masks, labels):
    """Exact partition of population variance, including estimated-mean covariance."""
    y = vector(y)
    masks = np.asarray(masks, bool)
    if masks.shape != (len(y), len(labels)) or not np.all(masks.sum(axis=1) == 1):
        raise ValueError('exclusive exhaustive complete variance partition required')
    centered = y - y.mean()
    values = masks * centered[:, None] ** 2
    means = values.mean(axis=0)
    leverage = (masks * centered[:, None]).mean(axis=0)
    influence = values - means - 2 * centered[:, None] * leverage
    covariance = np.atleast_2d(np.cov(influence, rowvar=False, ddof=1)) / len(y)
    return dict(population_variance=float(y.var()), components={name:dict(
        count=int(masks[:, i].sum()), contribution=float(means[i]),
        se=float(np.sqrt(max(0., covariance[i, i])))) for i, name in enumerate(labels)},
        covariance=covariance.tolist(), closure=float(means.sum()-y.var()),
        scope='All-entry centered-variance partition with correlated influence-function SE, not causal attribution')


def tails(y, exits=None, forward=None):
    y = vector(y)
    if np.any(y < 0):
        raise ValueError('nonnegative threshold-transform energies required')
    bins = np.searchsorted(ENERGY_EDGES, y, side='right') - 1
    labels = ['zero_or_below_0.1', '0.1_to_0.2', '0.2_to_0.3', '0.3_to_0.5',
              '0.5_to_0.75', '0.75_to_1', '1_and_above']
    masks = bins[:, None] == np.arange(len(labels))
    result = dict(energy_variance_parts=variance_parts(y, masks, labels), caps={},
                  above_0p75=probability(y >= .75), above_1=probability(y >= 1.), maximum=float(y.max()))
    v, mu = y.var(), y.mean()
    for cap in CAPS:
        z = np.minimum(y, cap)
        influence = ((y-mu)**2-v) - ((z-z.mean())**2-z.var())
        result['caps'][str(cap)] = dict(cap_MeV=cap, affected=probability(y > cap),
            capped_population_variance=float(z.var()), capped_variance_se=float(
                np.std((z-z.mean())**2, ddof=1) / np.sqrt(len(y))),
            removed_population_variance=float(v-z.var()),
            removed_variance_se=float(influence.std(ddof=1) / np.sqrt(len(y))),
            scope='Winsorized all-entry diagnostic, not an exit distribution or acceptance gate')
    if exits is not None:
        exits, forward = np.asarray(exits, bool), np.asarray(forward, bool)
        result['topology'] = variance_parts(y, np.column_stack([~exits, exits & forward, exits & ~forward]),
                                           ['nonexit', 'forward_exit', 'back_exit'])
        result['back_exit_probability'] = probability(exits & ~forward)
    return result


def correlation(x, y):
    x, y = vector(x), vector(y)
    if x.shape != y.shape:
        raise ValueError('paired within-sample correlation vectors required')
    dx, dy = x-x.mean(), y-y.mean()
    sx, sy = np.sqrt(np.mean(dx*dx)), np.sqrt(np.mean(dy*dy))
    if min(sx, sy) == 0:
        return None
    zx, zy = dx/sx, dy/sy
    rho = float(np.mean(zx*zy))
    influence = zx*zy - .5*rho*(zx*zx+zy*zy)
    return dict(n=len(x), value=rho, se=float(influence.std(ddof=1)/np.sqrt(len(x))))


def quantile_interval(values, q=.5, alpha=.01):
    """Distribution-free order-statistic interval with explicit infinite endpoints."""
    values = np.sort(vector(values))
    lower, upper = int(binom.ppf(alpha/2, len(values), q)), int(binom.ppf(1-alpha/2, len(values), q))+1
    return dict(q=q, estimate=float(np.quantile(values, q)), lower=None if lower < 1 else float(values[lower-1]),
                upper=None if upper > len(values) else float(values[upper-1]),
                scope='Pointwise exact-binomial order-statistic interval; null denotes unbounded endpoint')


def standardize_angle(candidate_energy, candidate_angle, reference_energy, reference_angle):
    """Symmetric composition/shape decomposition on fixed forward-exit energy bins."""
    ce, ca, re, ra = map(vector, (candidate_energy, candidate_angle, reference_energy, reference_angle))
    if ce.shape != ca.shape or re.shape != ra.shape or np.any(ce < .1) or np.any(re < .1):
        raise ValueError('paired above-threshold forward exits required')
    edges = np.array([.1, .2, .3, .5, np.inf])
    labels = ['0.1_to_0.2', '0.2_to_0.3', '0.3_to_0.5', '0.5_and_above']
    arms = []
    for energy, angle in ((ce, ca), (re, ra)):
        bins = np.searchsorted(edges, energy, side='right')-1
        mask = bins[:, None] == np.arange(4)
        count = mask.sum(axis=0)
        if np.any(count < 2):
            return dict(available=False, reason='at least two exits required in every declared energy bin; no adaptive merging',
                        bin_counts=count.tolist())
        weight, mean = count/len(energy), (mask*angle[:, None]).sum(axis=0)/count
        wi = mask-weight
        mi = mask*(angle[:, None]-mean)/weight
        arms.append((weight, mean, wi, mi))
    p, m, ip, im = arms[0]
    q, n, iq, inn = arms[1]
    composition, shape = .5*np.sum((p-q)*(m+n)), .5*np.sum((p+q)*(m-n))
    ci = np.column_stack([.5*(ip@(m+n)+im@(p-q)), .5*(ip@(m-n)+im@(p+q))])
    ri = np.column_stack([.5*(-iq@(m+n)+inn@(p-q)), .5*(iq@(m-n)-inn@(p+q))])
    covariance = np.cov(ci, rowvar=False, ddof=1)/len(ce)+np.cov(ri, rowvar=False, ddof=1)/len(re)
    return dict(available=True, bins=labels, candidate_weights=p.tolist(), reference_weights=q.tolist(),
        candidate_bin_mean_rad=m.tolist(), reference_bin_mean_rad=n.tolist(),
        total_mean_angle_delta=float(ca.mean()-ra.mean()), composition=dict(delta=float(composition), se=float(np.sqrt(covariance[0, 0]))),
        conditional_shape=dict(delta=float(shape), se=float(np.sqrt(covariance[1, 1]))), covariance=covariance.tolist(),
        closure=float(composition+shape-(ca.mean()-ra.mean())),
        scope='Symmetric algebraic split of outcome-selected mean angle; coarse bins leave within-bin energy composition; not causal')


def exit_statistics(energy, exits, forward, direction, path_proxy, cosine):
    energy = vector(energy)
    exits, forward = np.asarray(exits, bool), np.asarray(forward, bool)
    direction = np.asarray(direction, float).reshape(-1, 3)
    path_proxy = np.asarray(path_proxy, float).ravel()
    if (direction.shape != (len(energy), 3) or exits.shape != energy.shape
            or forward.shape != energy.shape or path_proxy.shape != energy.shape):
        raise ValueError('complete energy/topology/direction/path shape required')
    entry = np.array([np.sqrt(1-cosine*cosine), 0., cosine])
    costheta = direction @ entry
    angle = np.arccos(np.clip(costheta, -1, 1))
    masks = {'all_exits': exits, 'above_exits': exits & (energy >= .1),
             'above_forward_exits': exits & forward & (energy >= .1)}
    for low, high in zip(ENERGY_EDGES[1:-1], ENERGY_EDGES[2:]):
        masks[f'forward_energy_{low}_{high}'] = exits & forward & (energy >= low) & (energy < high)
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        item = dict(n=count, probability_among_all_entries=probability(mask))
        if count >= 2:
            d, a, e, p = direction[mask], angle[mask], energy[mask], path_proxy[mask]
            if (np.any(~np.isfinite(d)) or not np.allclose(np.linalg.norm(d, axis=1), 1., atol=1e-6, rtol=0)
                    or np.any(~np.isfinite(p))):
                raise ValueError('invalid resolved exit vectors/path')
            item.update(mean_deflection_cosine=estimate(costheta[mask]), mean_deflection_rad=estimate(a),
                        mean_geometric_path_ratio=estimate(p), mean_energy=estimate(e),
                        mean_theta2=estimate(a*a), mean_theta4=estimate(a**4),
                        mean_legendre2=estimate((3*costheta[mask]**2-1)/2),
                        median_deflection_interval_99pct=quantile_interval(a),
                        angle_energy_correlation=correlation(a, e), energy_path_correlation=correlation(e, p),
                        angle_path_correlation=correlation(a, p),
                        deflection_quantiles=np.quantile(a, [.1, .5, .9, .99, 1]).tolist(),
                        deflection_over_0p6=probability(a > .6))
        result[name] = item
    above = exits & (energy >= .1)
    hist, _, _ = np.histogram2d(energy[above], angle[above], bins=(ENERGY_EDGES[1:], ANGLE_EDGES))
    result['joint_counts'] = hist.astype(int).tolist()
    result['scope'] = 'Outcome-conditioned pointwise estimates; no all-primary angle closure; G4 chord vs fast polygon path are different proxies'
    return result


def step_features(steps, ref, *, step_size=fr.CHUNK_SIZE):
    """Per-primary adjacent pre-step jumps, stopping at the same first exit as fr.audit."""
    n = len(ref['event'])
    sentinel = np.iinfo(np.int64).max
    first, end = np.full(n, sentinel), np.full(n, sentinel)
    fields = 'run evt trk pdg sid code x y z px py pz'.split()
    out = dict(foil_steps=np.zeros(n, np.int64), msc_ended_steps=np.zeros(n, np.int64),
               coulomb_ended_steps=np.zeros(n, np.int64), jumps_over_0p1=np.zeros(n, np.int64),
               jumps_over_0p3=np.zeros(n, np.int64), jumps_over_0p6=np.zeros(n, np.int64),
               max_jump_rad=np.zeros(n), sum_one_minus_cos=np.zeros(n),
               energy_at_max_jump_MeV=np.zeros(n), code_at_max_jump=np.zeros(n, np.int64),
               adjacent_chord_sum_mm=np.zeros(n), transverse_chord_sum_mm=np.zeros(n),
               transverse_chord_square_mm2=np.zeros(n), max_transverse_chord_mm=np.zeros(n))
    offset, previous = 0, None
    for rows in steps.iterate(fields, step_size=step_size, library='np'):
        ev = fr.integer(rows['evt'], 'feature event') - 1
        if np.any(ev < 0) or np.any(ev >= n) or np.any(rows['run'] != 1):
            raise ValueError('feature event outside primary population')
        matched = fr.integer(rows['trk'], 'feature track') == ref['track'][ev]
        ev = ev[matched]
        s = {k: rows[k][matched] for k in fields}
        if not len(ev):
            continue
        if np.any(s['pdg'] != 13) or np.any(np.diff(ev) < 0) or (previous is not None and ev[0] < previous['event'][-1]):
            raise ValueError('primary feature identity/order mismatch')
        at = offset + np.arange(len(ev))
        inside = s['sid'] == 2
        np.minimum.at(first, ev[inside], at[inside])
        outside = np.flatnonzero((s['sid'] != 2) & (at > first[ev]))
        np.minimum.at(end, ev[outside], at[outside])
        keep = inside & (at < end[ev])
        np.add.at(out['foil_steps'], ev[keep], 1)
        np.add.at(out['msc_ended_steps'], ev[keep & (s['code'] == 29)], 1)
        np.add.at(out['coulomb_ended_steps'], ev[keep & (s['code'] == 59)], 1)
        p = np.column_stack([np.asarray(s[k], float) for k in ('px', 'py', 'pz')])
        if np.any(~np.isfinite(p)):
            raise ValueError('nonfinite primary feature momentum')
        xyz = np.column_stack([np.asarray(s[k], float) for k in ('x', 'y', 'z')])
        e, a, sid, code = ev, at, s['sid'], s['code']
        if previous is not None:
            p = np.vstack((previous['p'], p))
            xyz = np.vstack((previous['xyz'], xyz))
            e = np.concatenate((previous['event'], e))
            a = np.concatenate((previous['at'], a))
            sid = np.concatenate((previous['sid'], sid))
            code = np.concatenate((previous['code'], code))
        norm = np.linalg.norm(p, axis=1)
        geometric = (e[:-1] == e[1:]) & (sid[:-1] == 2) & (a[:-1] < end[e[:-1]]) & (norm[:-1] > 0)
        gi = np.flatnonzero(geometric)
        chord = xyz[gi+1]-xyz[gi]
        unit = p[gi]/norm[gi, None]
        along = np.sum(chord*unit, axis=1)
        transverse = np.linalg.norm(chord-along[:, None]*unit, axis=1)
        np.add.at(out['adjacent_chord_sum_mm'], e[gi], np.linalg.norm(chord, axis=1))
        np.add.at(out['transverse_chord_sum_mm'], e[gi], transverse)
        np.add.at(out['transverse_chord_square_mm2'], e[gi], transverse*transverse)
        np.maximum.at(out['max_transverse_chord_mm'], e[gi], transverse)
        index = np.flatnonzero(geometric & (norm[1:] > 0))
        cosine = np.sum(p[index]*p[index+1], axis=1) / (norm[index]*norm[index+1])
        cosine = np.clip(cosine, -1, 1)
        angle = np.arccos(cosine)
        new_maximum = out['max_jump_rad'].copy()
        np.maximum.at(new_maximum, e[index], angle)
        winners = (angle > out['max_jump_rad'][e[index]]) & (angle == new_maximum[e[index]])
        winning_events, locations = np.unique(e[index][winners], return_index=True)
        wi = index[winners][locations]
        out['energy_at_max_jump_MeV'][winning_events] = fr.kinetic(norm[wi])
        out['code_at_max_jump'][winning_events] = code[wi].astype(np.int64)
        out['max_jump_rad'] = new_maximum
        np.add.at(out['sum_one_minus_cos'], e[index], 1-cosine)
        for threshold, key in ((.1, 'jumps_over_0p1'), (.3, 'jumps_over_0p3'), (.6, 'jumps_over_0p6')):
            np.add.at(out[key], e[index][angle > threshold], 1)
        previous = dict(p=p[-1:], xyz=xyz[-1:], event=e[-1:], at=a[-1:], sid=sid[-1:], code=code[-1:])
        offset += len(ev)
    if (np.any(first == sentinel) or not np.array_equal(end != sentinel, ref['exit'].astype(bool))
            or not np.array_equal(out['foil_steps'], ref['foil_steps'])):
        raise ValueError('step-feature/strict-reference census mismatch')
    return out


def difference(a, b):
    return dict(delta=float(a['mean']-b['mean']), se=float(np.hypot(a['se'], b['se'])),
                scope='Independent-sample pointwise SE; no new acceptance gate')


def compare(a, b):
    result = dict(variance_components={}, capped_variances={}, angular={})
    for name, x in a['tails']['energy_variance_parts']['components'].items():
        y = b['tails']['energy_variance_parts']['components'][name]
        result['variance_components'][name] = dict(delta=x['contribution']-y['contribution'], se=float(np.hypot(x['se'], y['se'])))
    for cap in CAPS:
        x, y = a['tails']['caps'][str(cap)], b['tails']['caps'][str(cap)]
        result['capped_variances'][str(cap)] = dict(delta=x['capped_population_variance']-y['capped_population_variance'],
            se=float(np.hypot(x['capped_variance_se'], y['capped_variance_se'])), scope='Diagnostic capped estimand, not an acceptance gate')
    for cohort, x in a['exit_statistics'].items():
        if not isinstance(x, dict) or 'mean_deflection_cosine' not in x:
            continue
        y = b['exit_statistics'][cohort]
        if 'mean_deflection_cosine' not in y:
            continue
        result['angular'][cohort] = {key:difference(x[key], y[key]) for key in
            ('mean_deflection_cosine', 'mean_deflection_rad', 'mean_energy', 'mean_geometric_path_ratio',
             'mean_theta2', 'mean_theta4', 'mean_legendre2')}
    return result


def summarize_g4(data, angles, features, thickness, cosine):
    energy = np.where(data['exit_MeV'] < .1, 0., data['exit_MeV'])
    exits, forward = data['exit'].astype(bool), data['exit_z_mm'] >= 0
    result = dict(tails=tails(energy, exits, forward), exit_statistics=exit_statistics(
        data['exit_MeV'], exits, forward, angles['direction'], data['chord_sum_mm']/(thickness/cosine), cosine),
        features={}, rare_rows={})
    for name, mask in [('all', np.ones(len(energy), bool)), ('forward_above', exits & forward & (energy >= .1)),
                       ('back', exits & ~forward), ('above_0p75', energy >= .75), ('above_1', energy >= 1.)]:
        selected = {k:v[mask] for k, v in features.items()}
        result['features'][name] = dict(n=int(mask.sum()), metrics={k:estimate(v) if len(v) >= 2 else None for k, v in selected.items()},
            any_large_jump=probability(features['jumps_over_0p6'][mask] > 0) if mask.any() else None)
    selected = (energy >= .75) | (exits & ~forward)
    result['rare_rows'] = {k:np.asarray(v)[selected].tolist() for k, v in dict(
        event=data['event'], track=data['track'], exit_MeV=data['exit_MeV'], exit_z_mm=data['exit_z_mm'],
        chord_ratio=data['chord_sum_mm']/(thickness/cosine), hard_count=data['hard_count'],
        direction_z=angles['direction'][:, 2], **features).items()}
    for name in result['features']:
        mask = np.ones(len(energy), bool) if name == 'all' else ((exits & forward & (energy >= .1)) if name == 'forward_above'
            else ((exits & ~forward) if name == 'back' else energy >= (.75 if name == 'above_0p75' else 1.)))
        large = mask & (features['max_jump_rad'] > .6)
        result['features'][name]['energy_at_large_max_jump'] = estimate(features['energy_at_max_jump_MeV'][large]) if large.sum() >= 2 else None
        codes, counts = np.unique(features['code_at_max_jump'][large], return_counts=True)
        result['features'][name]['large_max_jump_step_end_codes'] = dict(zip(codes.astype(str), counts.astype(int).tolist()))
    result['scope'] = 'Adjacent pre-step deflections only, not G4 true-path or exclusive single-scatter identification'
    return result


def run(root, prior, studies, out):
    root, prior, out = Path(root), Path(prior), Path(out)
    if out.exists():
        raise FileExistsError(out)
    previous = json.loads((prior/'summary.json').read_text())
    if previous.get('complete') is not True:
        raise ValueError('completed angle reduction required')
    records = [_identity(prior/'summary.json')]
    for item in previous['inputs'] + previous['outputs']:
        records.append(edgeg4.checked(item))
    covered = {Path(item['path']).name:item for item in previous['outputs']}
    sources = [_identity(p) for p in (Path(__file__), Path(le.__file__), Path(fr.__file__), Path(slabrefs.__file__),
               Path(edgeg4.__file__), Path(sd.__file__), Path(le.ld.__file__), Path(le.ss.__file__))]
    out.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, scope=__doc__, new_transport=False, physics_closure=False,
                  diagnostic_energy_edges_MeV=ENERGY_EDGES[:-1].tolist()+['inf'],
                  diagnostic_caps_MeV=list(CAPS), g4={}, historical={}, fast={}, comparisons={}, inputs=records, sources=sources)
    angle_samples = {}
    for cell, (energy, thickness, cosine) in edgeg4.CELLS.items():
        refs, identities = slabrefs.load_references(root, cell, 13, thickness, energy, cosine)
        records.extend(identities)
        for mode, data in refs.items():
            tag = cell+'_'+mode
            name = tag+'_angles.npz'
            edgeg4.checked(covered[name])
            with np.load(prior/name, allow_pickle=False) as saved:
                angles = {k:saved[k] for k in saved.files}
            if not np.array_equal(angles['event'], data['event']) or not np.array_equal(angles['track'], data['track']):
                raise ValueError('angle/reference identity mismatch')
            with uproot.open(root/tag/'foil.root', array_cache=None) as file:
                steps, tree = fr.highest(file, 'readback/ntstepper', arrays=False)
                features = step_features(steps, data)
            np.savez_compressed(out/(tag+'_features.npz'), **features)
            result['g4'][tag] = summarize_g4(data, angles, features, thickness, cosine)
            result['g4'][tag]['tree'] = tree
            keep = data['exit'].astype(bool) & (data['exit_z_mm'] >= 0) & (data['exit_MeV'] >= .1)
            entry_direction = np.array([np.sqrt(1-cosine*cosine), 0., cosine])
            angle_samples[tag] = (data['exit_MeV'][keep], np.arccos(np.clip(angles['direction'][keep] @ entry_direction, -1, 1)))
            result['g4'][tag]['adjacent_chord_reduction_matches_audited_sum'] = bool(np.allclose(
                features['adjacent_chord_sum_mm'], data['chord_sum_mm'], rtol=1e-6, atol=2e-6))
            if not result['g4'][tag]['adjacent_chord_reduction_matches_audited_sum']:
                raise ValueError('adjacent chord reduction disagrees with audited sum')
        candidates = [p for p in previous['inputs'] if Path(p['path']).name == 'g4_'+cell+'_full.npz']
        if len(candidates) != 1:
            raise ValueError('unique hash-bound historical reference required')
        old_identity = candidates[0]
        records.append(edgeg4.checked(old_identity))
        with np.load(old_identity['path'], allow_pickle=False) as saved:
            old_energy = np.asarray(saved['exit_MeV'], float)
        result['historical'][cell] = dict(tails=tails(np.where(old_energy < .1, 0., old_energy)),
            reference_configuration='UNVERIFIED historical stock; reduced array hash/census checked by prior report')
        a, b = result['g4'][cell+'_full_refined']['tails'], result['historical'][cell]['tails']
        result['comparisons'][cell+'_refined_minus_historical_caps'] = {
            str(cap):dict(delta=a['caps'][str(cap)]['capped_population_variance']-b['caps'][str(cap)]['capped_population_variance'],
                se=float(np.hypot(a['caps'][str(cap)]['capped_variance_se'], b['caps'][str(cap)]['capped_variance_se']))) for cap in CAPS}
    for directory in studies:
        report, arms, identity = le.read_study(directory)
        cell = report['args']['cell']
        energy, thickness, cosine = edgeg4.CELLS[cell]
        records.append(identity)
        for arm, arrays in arms.items():
            if arm != 'diffuse':
                continue
            records.append(report['arms'][arm]['output'])
            y = le.threshold_array(arrays, energy)
            exits = np.isin(np.asarray(arrays['status']).ravel(), [sd.TRANSMITTED, sd.REFLECTED])
            forward = np.asarray(arrays['status']).ravel() == sd.TRANSMITTED
            tag = Path(directory).name+'_'+arm
            result['fast'][tag] = dict(tails=tails(y, exits, forward), exit_statistics=exit_statistics(
                np.asarray(arrays['raw_energy']).ravel(), exits, forward, arrays['direction'],
                np.asarray(arrays['path']).ravel()/(thickness/cosine), cosine),
                scope='Positive-floor residuals not physical stops; only identified threshold transform is used')
            keep = exits & forward & (np.asarray(arrays['raw_energy']).ravel() >= .1)
            direction = np.asarray(arrays['direction']).reshape(-1, 3)
            entry_direction = np.array([np.sqrt(1-cosine*cosine), 0., cosine])
            angle_samples[tag] = (np.asarray(arrays['raw_energy']).ravel()[keep], np.arccos(np.clip(direction[keep] @ entry_direction, -1, 1)))
            for mode in edgeg4.MODES:
                result['comparisons'][tag+'_minus_'+mode] = compare(result['fast'][tag], result['g4'][cell+'_'+mode])
                result['comparisons'][tag+'_minus_'+mode]['standardized_forward_angle'] = standardize_angle(
                    *angle_samples[tag], *angle_samples[cell+'_'+mode])
    for cell in edgeg4.CELLS:
        result['comparisons'][cell+'_no_lateral_minus_refined'] = compare(
            result['g4'][cell+'_full_refined_no_lateral'], result['g4'][cell+'_full_refined'])
    for item in records+sources:
        edgeg4.checked(item)
    result['outputs'] = [_identity(p) for p in sorted(out.glob('*.npz'))]
    result['complete'] = True
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True, g4=len(result['g4']), fast=len(result['fast']))), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--prior', required=True)
    parser.add_argument('--study', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    run(args.root, args.prior, args.study, args.out)


if __name__ == '__main__':
    main()
