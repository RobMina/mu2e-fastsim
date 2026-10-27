"""Same-product shape/pair sufficient statistics; no fitted correction or pulse cross product."""
import numpy as np

from . import joint_compare as J, joint_pairs as P, joint_timing as T

ALPHA = .985
MOMENTS = ('dt', 'dt2', 'dbeta', 'dbeta2', 'dt_dbeta', 'le2', 'beta_low', 'beta_high',
           'height_pe_low', 'height_pe_high', 'ratio_over_beta_low', 'ratio_over_beta_high',
           'chi2_low', 'chi2_high', 'either_chi2_negative', 'either_beta_19')


def _sensor(base, h, valid, pebin):
    index = (base[valid]*2 + h['sipm_id'][valid]//2)*5 + pebin[valid]
    keys, inv = np.unique(index, return_inverse=True)
    n = np.bincount(inv, minlength=len(keys))
    avg = lambda v: T._mean(np.bincount(inv, weights=v, minlength=len(keys)), n)
    t, beta = h['time_residual'][valid], h['beta'][valid]
    mt, mb = avg(t), avg(beta)
    ct, cb = t-mt[inv], beta-mb[inv]
    result = dict(n=n, t=mt, beta=mb, tt=avg(ct*ct), bb=avg(cb*cb), tb=avg(ct*cb),
                  le_variance=avg((ct-ALPHA*cb)**2))
    for name, value in dict(height_pe=h['height_pe'], ratio_over_beta=h['pe']/h['height_pe']/h['beta'],
                            chi2=h['chi2'], negative=h['chi2'] < 0, beta19=h['beta'] == 19.).items():
        result[name] = avg(value[valid])
    return keys, result


def pair_rows(h):
    """Keep original group N; shape-invalid pulses only remove explicitly counted pair contributions."""
    base, first, inv = np.unique(np.column_stack((h['group'], h['bar_id'], h['side'])),
                                 axis=0, return_index=True, return_inverse=True)
    for name in ('body', 'along', 'path', 'group_count', 'layer', 'distance'):
        J.require(np.allclose(h[name], h[name][first][inv], rtol=0, atol=1e-10, equal_nan=True),
                  f'same-bar/end pulses disagree in {name}')
    pe, peok = J._bins(h['pe'], J.PE)
    shape_valid = np.ones(len(peok), bool)
    for name in ('beta', 'height_pe', 'pulse_height', 'chi2'):
        shape_valid &= np.isfinite(h[name])
    shape_valid &= (h['beta'] > 0) & (h['height_pe'] > 0) & (h['pulse_height'] >= 0)
    valid = peok & shape_valid
    with np.errstate(divide='ignore', invalid='ignore'):
        ids, stats = _sensor(inv, h, valid, pe)
    along, aok = J._bins(h['along'][first], J.ALONG)
    path, pok = J._bins(h['path'][first], J.PATH)
    for name, edges in (('along', J.ALONG), ('path', J.PATH)):
        bins, ok = J._bins(h[name], edges)
        J.require(np.array_equal(bins, bins[first][inv]) and np.array_equal(ok, ok[first][inv]),
                  f'same-bar/end {name} bin or validity mismatch')
    pieces = []
    for p0 in range(5):
        for p1 in range(5):
            il, ir = np.flatnonzero(ids % 10 == p0), np.flatnonzero(ids % 10 == p1+5)
            b, left, right = np.intersect1d(ids[il]//10, ids[ir]//10, return_indices=True)
            ok = aok[b] & pok[b]
            b, left, right = b[ok], il[left[ok]], ir[right[ok]]
            l, r = ({k: v[i] for k, v in stats.items()} for i in (left, right))
            dt, db = l['t']-r['t'], l['beta']-r['beta']
            tt, bb = l['tt']+r['tt']+dt*dt, l['bb']+r['bb']+db*db
            tb = l['tb']+r['tb']+dt*db
            values = dict(dt=dt, dt2=tt, dbeta=db, dbeta2=bb, dt_dbeta=tb,
                          le2=l['le_variance']+r['le_variance']+(dt-ALPHA*db)**2,
                          beta_low=l['beta'], beta_high=r['beta'], height_pe_low=l['height_pe'],
                          height_pe_high=r['height_pe'], ratio_over_beta_low=l['ratio_over_beta'],
                          ratio_over_beta_high=r['ratio_over_beta'], chi2_low=l['chi2'], chi2_high=r['chi2'],
                          either_chi2_negative=1-(1-l['negative'])*(1-r['negative']),
                          either_beta_19=1-(1-l['beta19'])*(1-r['beta19']))
            keys = np.column_stack((h['body'][first][b], h['side'][first][b], along[b], path[b],
                                    np.full(len(b), p0, int), np.full(len(b), p1, int),
                                    h['group_count'][first][b], h['layer'][first][b]))
            pieces.append(dict(keys=keys, group=h['group'][first][b], count=l['n']*r['n'], **values))
    rows = {k: np.concatenate([piece[k] for piece in pieces]) for k in pieces[0]}
    return rows, dict(pulses=len(valid), shape_invalid_pulses=int((~shape_valid).sum()),
                      pe_inrange_shape_invalid_pulses=int((peok & ~shape_valid).sum()))


def mode_keys(rows, mode):
    J.require(mode in ('exact', 'band'), 'shape support must be exact or band')
    keys = rows['keys'][:, :7].copy()
    if mode == 'band':
        keys[:, 6] = np.searchsorted(T.MULT, keys[:, 6], side='right')-1
    return keys


def structured(keys):
    values = np.asarray(keys)
    J.require(values.ndim == 2 and values.dtype.kind in 'iu', 'cell keys must be an integer matrix')
    return np.ascontiguousarray(values, dtype='<i8').view(
        [(f'k{i}', '<i8') for i in range(values.shape[1])]).reshape(-1)


def fixed_support(rows, keys, weights, mode, events):
    """Conditional full-event-cluster SE with fixed external support/target fractions."""
    if not keys:
        return dict(supported=False, reason='no previously supported cells', cells=0)
    requested = structured(np.asarray(keys, dtype=np.int64))
    J._unique(requested, 'requested support cell')
    weights = np.asarray(weights, float)
    J.require(weights.shape == requested.shape and np.all(np.isfinite(weights)) and np.all(weights > 0) and
              np.isclose(weights.sum(), 1., rtol=0, atol=1e-12), 'invalid frozen support weights')
    order = np.argsort(requested)
    all_keys = structured(mode_keys(rows, mode))
    pos = np.searchsorted(requested[order], all_keys)
    inside = pos < len(requested)
    inside[inside] &= requested[order][pos[inside]] == all_keys[inside]
    cell = order[pos[inside]]
    count = rows['count'][inside]
    event = rows['group'][inside]//4
    J.require(events > 1 and np.all((event >= 0) & (event < events)), 'invalid independent event universe')
    nc = np.bincount(cell, weights=count, minlength=len(requested))
    if np.any(nc == 0):
        return dict(supported=False, reason='shape-valid subset has empty frozen cells', cells=len(keys),
                    empty_cells=int((nc == 0).sum()), pairs=int(nc.sum()))
    means = {}
    influence = {}
    for field in MOMENTS:
        value = rows[field][inside]
        J.require(np.all(np.isfinite(value)), f'nonfinite shape moment {field}')
        mc = np.bincount(cell, weights=count*value, minlength=len(requested))/nc
        theta = float(np.dot(weights, mc))
        means[field] = theta
        psi = np.bincount(event, weights=weights[cell]/nc[cell]*count*(value-mc[cell]), minlength=events)
        J.require(abs(psi.sum()) < 1e-8*max(1, abs(theta)), 'influence function is not centered')
        influence[field] = psi
    se = lambda psi: float(np.sqrt(events/(events-1)*np.dot(psi, psi)))
    contrast = means['le2']-means['dt2']
    algebra = -2*ALPHA*means['dt_dbeta']+ALPHA**2*means['dbeta2']
    J.require(np.isclose(contrast, algebra, atol=1e-8, rtol=1e-10), 'LE covariance identity failed')
    return dict(supported=True, cells=len(keys), pairs=int(nc.sum()), contributing_events=len(np.unique(event)),
                event_universe=events, means=means,
                conditional_event_se={k: se(v) for k, v in influence.items()},
                le_minus_peak=contrast, le_minus_peak_conditional_event_se=se(influence['le2']-influence['dt2']),
                algebra=dict(beta_term=ALPHA**2*means['dbeta2'], covariance_term=-2*ALPHA*means['dt_dbeta'],
                             residual=contrast-algebra),
                uncertainty_scope='Independent full events; frozen support/target weights; no support/weight-estimation or serial-correlation uncertainty.')


def compare_cells(actual, expected, *, partial=False):
    """Reproduce historical pair counts/groups and all scalar moments before interpreting shapes."""
    aa, ee = ({tuple(row['key']): row for row in values} for values in (actual, expected))
    J.require(set(aa) <= set(ee) if partial else set(aa) == set(ee), 'historical pair cell set mismatch')
    max_difference = 0.
    for key, a in aa.items():
        e = ee[key]
        for field in ('count', 'groups'):
            J.require(a[field] <= e[field] if partial else a[field] == e[field], f'pair cell {field} mismatch')
        if partial:
            continue
        for field in (*P.FIELDS, 'variance', 'group_mean_second_moment'):
            delta = abs(a[field]-e[field])
            max_difference = max(max_difference, delta)
            J.require(np.isclose(a[field], e[field], rtol=1e-10, atol=1e-8), f'historical {field} mismatch')
    return dict(cells=len(aa), historical_cells=len(ee), full_census_and_moments_equal=not partial,
                max_absolute_moment_difference=max_difference if not partial else None)
