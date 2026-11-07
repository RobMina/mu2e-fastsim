"""Deterministic signed-direction support witnesses, not a sampler or reconstruction trial."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import candidate_support as auth, pool_census, source_overlap_report, support_audit as sa
from . import production_compare as pc
from .compare import paired_mean
from .source_events import file_digest

CANDIDATE_SHA = "ed177ed1018e3bf515160c965d435b2bd4bd22b35b8d7bf6ed0db02c39a5d1f7"
SUPPORT_SHA = "e9e9206104804c46f211e12093001756c359a01438ccdbc27fab872645a4936e"
TRIAL_SHA = "dee6f99faa4882a3ac98ded6a9a7d160aa011e1d739aab78235c443adc4eaf5a"
ANGLE_DEG = 10.
RADIUS_MM = 25.
K = 8
CAVEAT = ("Deterministic support assessment only; no draw, placement or reconstruction. "
          "Both signed local direction components must agree (zero nonnegative), then angle<=10deg. "
          "Nearest up-to8 compatible donors in the existing chosen cell are a support witness, not a deployed policy. "
          "The witness may widen the original K8 radius envelope; 25mm is a diagnostic locality boundary, not applied as a cut. "
          "No compatible donors means explicit direction fallback to original K8, not an empty response. "
          "Whole-event both-disk zeros/weak responses retained without outcome filtering. "
          "Conditional uniform expectations/exposures are not effective sample size, reconstruction closure, "
          "inclusive acceptance or training/model uncertainty. Incident state is a simulation input, not an output correction.")


def local_direction(library, entries):
    """Existing disk-azimuth convention, with signed radial/tangent components and axial direction."""
    sa._geometry(library)
    entries = sa._entries(entries)
    c = pool_census.incident_coordinates(library, entries)
    result = np.column_stack((c['direction_radial'], c['direction_tangent'], entries['direction'][:, 2]))
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite local direction')
    return result


def compatible(query, donor):
    """Pairwise nested sign and angular conditions; inputs end in three local components."""
    sign = np.all((query[..., :2] >= 0) == (donor[..., :2] >= 0), axis=-1)
    angle = np.rad2deg(np.arccos(np.clip(np.sum(query * donor, axis=-1), -1., 1.)))
    return sign, sign & (angle <= ANGLE_DEG), angle


def nearest(rows, distance, mask):
    """Nearest compatible rows, then library order, with -1 padding; no outcome inputs."""
    rank = np.lexsort((np.broadcast_to(rows, distance.shape), np.where(mask, distance, np.inf)), axis=1)
    k = min(K, len(rows))
    order = rank[:, :k]
    good = np.take_along_axis(mask, order, axis=1)
    selected = np.where(good, rows[order], np.iinfo(np.int64).max)
    selected.sort(axis=1)
    result = np.full((len(distance), K), -1, np.int64)
    result[:, :k] = np.where(selected == np.iinfo(np.int64).max, -1, selected)
    return result


def assess(library, entries, pools):
    """Cell-vectorized support census; the baseline pools must be authenticated by the caller."""
    q = local_direction(library, entries)
    d = local_direction(library, {k[6:]: v for k, v in library.items() if k.startswith('entry_')})
    n = len(q)
    origin = np.column_stack([library['geometry_disk_origin_' + a] for a in 'xy'])
    out = dict(query_direction=q, donor_direction=d, event_row=np.asarray(entries['event_row']).copy(),
               axis_query=np.all(entries['position'][:, :2] == origin[entries['disk']], axis=1),
               axis_donor=np.all(library['entry_position'][:, :2] == origin[library['entry_disk']], axis=1))
    for kind in ('sign', 'compatible'):
        for key in ('count', 'within25_count', 'baseline_count'):
            out[f'{kind}_{key}'] = np.zeros(n, np.int64)
        for key in ('nearest_radius', 'last_radius'):
            out[f'{kind}_{key}'] = np.full(n, np.nan)
    witness = np.full((n, K), -1, np.int64)
    for cell in np.unique(pools['chosen_cell'][pools['supported']]):
        target = np.flatnonzero(pools['chosen_cell'] == cell)
        rows = np.flatnonzero(pools['library_cell'] == cell)
        radius = np.abs(entries['radius'][target, None] - library['entry_radius'][rows][None, :])
        sign, good, _ = compatible(q[target, None, :], d[None, rows, :])
        for kind, mask in (('sign', sign), ('compatible', good)):
            out[kind + '_count'][target] = mask.sum(axis=1)
            out[kind + '_within25_count'][target] = (mask & (radius <= RADIUS_MM)).sum(axis=1)
            ids = nearest(rows, radius, mask)
            valid = ids >= 0
            dist = np.abs(library['entry_radius'][np.maximum(ids, 0)] - entries['radius'][target, None])
            out[kind + '_nearest_radius'][target] = np.where(valid, dist, np.inf).min(axis=1)
            out[kind + '_last_radius'][target] = np.where(valid, dist, -np.inf).max(axis=1)
            if kind == 'compatible':
                witness[target] = ids
    for kind in ('sign', 'compatible'):
        missing = out[kind + '_count'] == 0
        out[kind + '_nearest_radius'][missing] = np.nan
        out[kind + '_last_radius'][missing] = np.nan
    base = pools['eligible']
    valid = base >= 0
    lookup = np.vstack((d, np.zeros((1, 3))))
    sign, good, _ = compatible(q[:, None, :], lookup[np.maximum(base, 0)])
    out['sign_baseline_count'] = (sign & valid).sum(axis=1)
    out['compatible_baseline_count'] = (good & valid).sum(axis=1)
    fallback = pools['supported'] & (out['compatible_count'] == 0)
    out.update(compatible_eligible=witness.copy(), direction_fallback=fallback)
    witness[fallback] = base[fallback]
    out.update(witness_eligible=witness, witness_size=(witness >= 0).sum(axis=1),
               baseline_eligible=base.copy(), baseline_size=pools['size'].copy(),
               original_fallback=pools['fallback'].copy(), supported=pools['supported'].copy(),
               query_cell=pools['query_cell'].copy())
    for name, ids in (('baseline', base), ('witness', witness)):
        valid = ids >= 0
        donor = np.maximum(ids, 0)
        _, _, angle = compatible(q[:, None, :], lookup[donor])
        fields = {key + '_delta': np.r_[library['entry_' + key], 0.][donor] - entries[key][:, None]
                  for key in ('radius', 'energy', 'cosine')}
        fields.update(angle_deg=angle, radial_delta=lookup[donor, 0] - q[:, None, 0],
                      tangent_delta=lookup[donor, 1] - q[:, None, 1])
        for key, value in fields.items():
            out[name + '_' + key] = np.where(valid, value, np.nan)
    return out


def distribution(value):
    """Finite-value summary with an explicit unavailable count."""
    value = np.asarray(value)
    finite = np.isfinite(value)
    x = value[finite]
    result = dict(n=int(x.size), unavailable=int(value.size - x.size))
    result.update({k: float(v) for k, v in zip(('min', 'p50', 'p90', 'p95', 'max'),
                  np.quantile(x, [0, .5, .9, .95, 1]))} if x.size else
                  {k: None for k in ('min', 'p50', 'p90', 'p95', 'max')})
    result['mean'] = float(x.mean()) if x.size else None
    return result


def regions(entries, pools, arrays):
    n = len(entries['energy'])
    result = dict(all=np.ones(n, bool), fallback=pools['fallback'])
    for disk in (0, 1):
        result[f'disk{disk}'] = entries['disk'] == disk
    for name, edges in (('radius', (0, 450, 550, np.inf)), ('energy', (0, 50, 80, 95, np.inf)),
                        ('cosine', (0, .4, .7, np.inf))):
        for lo, hi in zip(edges[:-1], edges[1:]):
            result[f'{name}_{lo}_{hi}'] = (entries[name] >= lo) & (entries[name] < hi)
    for i, name in enumerate(('radial', 'tangent')):
        result[name + '_negative'] = arrays['query_direction'][:, i] < 0
        result[name + '_nonnegative'] = arrays['query_direction'][:, i] >= 0
    high = (entries['disk'] == 1) & (entries['radius'] < 450) & (entries['energy'] >= 95)
    result.update(highEinnerdisk1=high, problem109=high & (entries['cosine'] >= .7))
    cell = np.ravel_multi_index(pools['query_cell'].T, (2, 4, 3, 4))
    result.update({f'cell_{i:02d}': cell == i for i in range(96)})
    return result


def report(library, entries, pools, arrays):
    """Exact fixed-library uniform expectations; no sampling and no outcome-based support changes."""
    outcomes = pool_census.donor_outcomes(library)
    for name in ('baseline', 'witness'):
        ids = arrays[name + '_eligible']
        p = dict(eligible=ids, size=arrays[name + '_size'], supported=pools['supported'])
        for key, value in outcomes.items():
            arrays[name + '_prob_' + key] = sa._probabilities(value, p)
    compatible_pool = dict(eligible=arrays['compatible_eligible'],
        size=(arrays['compatible_eligible'] >= 0).sum(axis=1), supported=arrays['compatible_count'] > 0)
    for key, value in outcomes.items():
        arrays['compatible_prob_' + key] = sa._probabilities(value, compatible_pool)
    result = dict(scope=CAVEAT, angle_degrees=ANGLE_DEG, diagnostic_radius_mm=RADIUS_MM, k=K,
                  donors=len(outcomes['energy']), empty_donors=int(outcomes['empty'].sum()),
                  weak_donors=int(outcomes['weak'].sum()), axis_donors=int(arrays['axis_donor'].sum()), regions={})
    for region, mask in regions(entries, pools, arrays).items():
        supported = mask & pools['supported']
        item = dict(queries=int(mask.sum()), supported=int(supported.sum()), unsupported=int((mask & ~pools['supported']).sum()),
                    axis_queries=int((mask & arrays['axis_query']).sum()),
                    original_fallback=int((mask & pools['fallback']).sum()),
                    direction_fallback=int((mask & arrays['direction_fallback']).sum()))
        base_max = np.max(np.where(arrays['baseline_eligible'][supported] >= 0,
            np.abs(arrays['baseline_radius_delta'][supported]), -np.inf), axis=1)
        witness_max = np.max(np.where(arrays['witness_eligible'][supported] >= 0,
            np.abs(arrays['witness_radius_delta'][supported]), -np.inf), axis=1)
        item['radius_expansion_mm'] = distribution(witness_max - base_max)
        item['compatible_only_outcomes'] = {key: distribution(arrays['compatible_prob_' + key][supported])
                                             for key in ('empty', 'weak', 'energy')}
        for kind in ('sign', 'compatible'):
            count = arrays[kind + '_count'][supported]
            item[kind] = dict(zero=int((count == 0).sum()), fewer_than8=int((count < K).sum()),
                full_count=distribution(count), within25_count=distribution(arrays[kind + '_within25_count'][supported]),
                within25_zero=int((arrays[kind + '_within25_count'][supported] == 0).sum()),
                within25_atleast8=int((arrays[kind + '_within25_count'][supported] >= K).sum()),
                baseline_count=distribution(arrays[kind + '_baseline_count'][supported]),
                baseline_zero=int((arrays[kind + '_baseline_count'][supported] == 0).sum()),
                baseline_atleast8=int((arrays[kind + '_baseline_count'][supported] >= K).sum()),
                nearest_radius=distribution(arrays[kind + '_nearest_radius'][supported]),
                last_radius=distribution(arrays[kind + '_last_radius'][supported]),
                eighth_radius=distribution(np.where(count >= K, arrays[kind + '_last_radius'][supported], np.nan)))
        for name in ('baseline', 'witness'):
            ids, size = arrays[name + '_eligible'][supported], arrays[name + '_size'][supported]
            valid = ids >= 0
            weights = np.broadcast_to(1. / size[:, None], ids.shape)
            exposure = np.bincount(ids[valid], weights=weights[valid], minlength=len(outcomes['energy']))
            used = exposure > 0
            section = dict(size=distribution(size), eligible_union=int(used.sum()),
                eligible_empty=int((used & outcomes['empty']).sum()), eligible_weak=int((used & outcomes['weak']).sum()),
                max_expected_exposure=float(exposure.max(initial=0)), outcomes={})
            for key in ('empty', 'weak', 'energy'):
                prob = arrays[name + '_prob_' + key][supported]
                section['outcomes'][key] = dict(expected_sum=float(prob.sum()), probability=distribution(prob))
                if key != 'energy':
                    section['outcomes'][key]['conditional_count_sd'] = float(np.sqrt(np.sum(prob * (1 - prob))))
                if 'original_' + key in pools:
                    section['outcomes'][key]['original_sum'] = float(pools['original_' + key][supported].sum())
                    section['outcomes'][key]['expected_minus_original'] = paired_mean(
                        prob, pools['original_' + key][supported], entries['event_row'][supported] // 100)
            for key in ('radius_delta', 'angle_deg', 'radial_delta', 'tangent_delta', 'energy_delta', 'cosine_delta'):
                values = arrays[name + '_' + key][supported]
                values = np.abs(values) if key != 'angle_deg' else values
                section[key + '_pool_mean_abs'] = distribution(np.divide(np.nansum(values, axis=1), size))
                section[key + '_pool_max_abs'] = distribution(np.max(np.where(valid, values, -np.inf), axis=1))
            item[name] = section
        result['regions'][region] = item
    return result


def run(args):
    if args.out.exists():
        raise ValueError('new output directory required')
    inputs = source_overlap_report.Inputs()
    candidate = auth.pinned_json(inputs, args.candidate / 'summary.json', CANDIDATE_SHA)
    support = auth.pinned_json(inputs, args.support / 'summary.json', SUPPORT_SHA)
    trial = auth.pinned_json(inputs, args.trial / 'summary.json', TRIAL_SHA)
    packed = auth.pinned_npz(inputs, args.candidate / 'library.npz', candidate['output_sha256']['library.npz'])
    packed.pop('metadata')
    pools = auth.pinned_npz(inputs, args.support / 'candidate_pools.npz', support['output_sha256']['candidate_pools.npz'])
    saved = auth.pinned_npz(inputs, args.trial / 'clusters.npz', trial['output_sha256']['clusters.npz'])
    entries = {k[6:]: v for k, v in saved.items() if k.startswith('entry_')}
    entries = sa._entries(entries)
    if not np.array_equal(entries['event_row'], saved['query_event_rows']) or not np.array_equal(entries['event_row'], pools['event_row']):
        raise ValueError('query row alignment differs')
    if np.any(entries['event_row'] >= len(saved['full_ids'])):
        raise ValueError('query row outside full stored census')
    current = sa.enumerate_pools(packed, entries)
    for key, value in current.items():
        if not np.array_equal(value, pools[key]):
            raise ValueError('baseline pool differs: ' + key)
    arrays = assess(packed, entries, pools)
    result = report(packed, entries, pools, arrays)
    for key in ('empty', 'weak', 'energy'):
        if not np.array_equal(arrays['baseline_prob_' + key], pools['prob_' + key], equal_nan=True):
            raise ValueError('baseline expectation differs: ' + key)
    arrays['full_ids'] = saved['full_ids']
    for module in (sa, pool_census):
        inputs.read(module.__file__)
    inputs.read(__file__)
    inputs.unchanged()
    result.update(input_sha256=inputs.sha256, sampling_performed=False, reconstruction_performed=False)
    args.out.mkdir(parents=True)
    pc._json(args.out / 'numbers.json', result)
    np.savez_compressed(args.out / 'arrays.npz', **arrays)
    pc._json(args.out / 'summary.json', dict(scope=CAVEAT, input_sha256=inputs.sha256,
        output_sha256={name: file_digest(args.out / name) for name in ('numbers.json', 'arrays.npz')},
        all=result['regions']['all'], sampling_performed=False, reconstruction_performed=False))
    print('Deterministic direction-support assessment complete; no sampling, placement or reconstruction.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'support', 'trial', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
