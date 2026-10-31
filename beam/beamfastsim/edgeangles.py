"""Add identity-bound first-exit direction/position from existing pre-step readback.

First vacuum pre-step after the foil is the same exit convention as foilreport.
Stopped histories have NaN directions, never a fabricated zero-angle. Chord sums
remain geometric; the readback does not contain Geant4 true step lengths.
"""
import numpy as np

from . import foilreport as fr


def first_exit(steps, reference, *, step_size=fr.CHUNK_SIZE):
    n = len(reference['event'])
    if not np.array_equal(reference['event'], np.arange(1, n + 1)):
        raise ValueError('complete ordered reference event census required')
    sentinel = np.iinfo(np.int64).max
    first, end = np.full(n, sentinel), np.full(n, sentinel)
    momentum, position = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    fields = 'run evt trk pdg sid x y z px py pz'.split()
    offset = 0
    for rows in steps.iterate(fields, step_size=step_size, library='np'):
        ev = fr.integer(rows['evt'], 'direction event') - 1
        if np.any(ev < 0) or np.any(ev >= n) or np.any(rows['run'] != 1):
            raise ValueError('direction event outside primary population')
        matched = fr.integer(rows['trk'], 'direction track') == reference['track'][ev]
        s = {k: rows[k][matched] for k in fields}
        ev = ev[matched]
        if np.any(s['pdg'] != 13):
            raise ValueError('primary direction charge mismatch')
        at = offset + np.arange(len(ev))
        inside = s['sid'] == 2
        np.minimum.at(first, ev[inside], at[inside])
        outside = np.flatnonzero((s['sid'] != 2) & (at > first[ev]))
        unique, ix = np.unique(ev[outside], return_index=True)
        new = end[unique] == sentinel
        unique, ix = unique[new], outside[ix[new]]
        end[unique] = at[ix]
        momentum[unique] = np.column_stack([s[k][ix] for k in ('px', 'py', 'pz')])
        position[unique] = np.column_stack([s[k][ix] for k in ('x', 'y', 'z')])
        offset += len(ev)
    if np.any(first == sentinel) or not np.array_equal(end != sentinel, reference['exit'].astype(bool)):
        raise ValueError('direction/reference first-exit census mismatch')
    exits = reference['exit'].astype(bool)
    norm = np.linalg.norm(momentum[exits], axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0) or np.any(~np.isfinite(position[exits])):
        raise ValueError('invalid first-exit momentum/position')
    direction = momentum.copy()
    direction[exits] /= norm[:, None]
    if not np.allclose(position[exits, 2], reference['exit_z_mm'][exits], rtol=0, atol=0):
        raise ValueError('direction/reference exit-position mismatch')
    return dict(event=reference['event'], track=reference['track'], resolved_exit=exits,
                direction=direction, position_mm=position)


def summary(energy, exit_mask, direction, cosine):
    energy = np.asarray(energy, float).ravel()
    direction = np.asarray(direction, float).reshape(-1, 3)
    exit_mask = np.asarray(exit_mask, bool).ravel()
    if direction.shape != (len(energy), 3) or exit_mask.shape != energy.shape:
        raise ValueError('same complete direction/energy population required')
    entry = np.array([np.sqrt(1 - cosine ** 2), 0., cosine])
    angle = np.arccos(np.clip(direction @ entry, -1, 1))
    result = {}
    for name, mask in (('all_exits', exit_mask), ('above_threshold_exits', exit_mask & (energy >= .1))):
        result[name] = dict(n=int(mask.sum()), z_quantiles=None, deflection_rad_quantiles=None,
                            mean_deflection_cosine=None, energy_angle_correlation=None)
        if not mask.any():
            continue
        d, a = direction[mask], angle[mask]
        if np.any(~np.isfinite(d)) or not np.allclose(np.linalg.norm(d, axis=1), 1, rtol=0, atol=1e-6):
            raise ValueError('invalid resolved exit direction')
        result[name].update(z_quantiles=np.quantile(d[:, 2], [0, .1, .5, .9, 1]).tolist(),
                            deflection_rad_quantiles=np.quantile(a, [0, .1, .5, .9, 1]).tolist(),
                            mean_deflection_cosine=float(np.mean(d @ entry)))
        if len(a) > 1 and np.std(a) > 0 and np.std(energy[mask]) > 0:
            result[name]['energy_angle_correlation'] = float(np.corrcoef(energy[mask], a)[0, 1])
    result['scope'] = 'Outcome-conditioned first-exit angular diagnostics, not causal attribution or all-primary angle closure'
    return result
