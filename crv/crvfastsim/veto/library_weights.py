"""Read-only source-occurrence weighting and parent-overlap audit for retained replay manifests."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import re

import numpy as np

from .mixframes import MIXERS
from .replay import sha256
from .source_provenance import source_name


def _digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('invalid SHA256 identity')
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'invalid {label}')
    return value


def parent_groups(provenance):
    """Require a complete disjoint-or-identical parent census; partial overlap needs another model."""
    sources = provenance['sources']
    if not sources:
        raise ValueError('empty source provenance')
    groups, payloads = {}, {}
    for name, record in sources.items():
        sam = record.get('sam')
        if not isinstance(sam, dict) or sam.get('file_name') != name:
            raise ValueError('missing or mismatched catalog identity')
        _digest(sam['source_sha256'])
        _digest(sam['metadata_sha256'])
        group, payload = _digest(sam['parents_sha256']), _digest(record['table_sha256'])
        _integer(sam['parent_count'], 'parent count', 1)
        _integer(sam['event_count'], 'catalog events', 1)
        _integer(record['table_frames'], 'table frames', 1)
        _integer(record['step_events'], 'step events')
        if record['step_events'] > record['table_frames']:
            raise ValueError('stepped events exceed denominator')
        groups.setdefault(group, []).append(name)
        payloads.setdefault(payload, set()).add(group)
    if any(len(g) != 1 for g in payloads.values()):
        raise ValueError('identical payload has different parent groups')
    for names in groups.values():
        signatures = {(sources[n]['table_sha256'], sources[n]['table_frames'], sources[n]['step_events'],
                       sources[n]['sam']['event_count'], sources[n]['sam']['parent_count']) for n in names}
        if len(signatures) != 1:
            raise ValueError('identical parents have unequal payload/counts; cannot treat as aliases')
    expected = set(itertools.combinations(sorted(sources), 2))
    seen = set()
    for overlap in provenance['parent_overlaps']:
        pair = tuple(sorted((overlap['first'], overlap['second'])))
        if pair not in expected or pair in seen:
            raise ValueError('unknown or duplicate parent-overlap pair')
        seen.add(pair)
        a, b = (sources[n]['sam'] for n in pair)
        same = a['parents_sha256'] == b['parents_sha256']
        shared = _integer(overlap['shared_parents'], 'shared parent count')
        if type(overlap['identical_parent_sets']) is not bool or overlap['identical_parent_sets'] != same:
            raise ValueError('inconsistent parent equality')
        if shared != (a['parent_count'] if same else 0):
            raise ValueError('partial or inconsistent parent overlap; grouped audit unsupported')
    if seen != expected:
        raise ValueError('incomplete pairwise parent census')
    return {key: sorted(value) for key, value in sorted(groups.items())}


def weight_summary(weights):
    """Weight concentration only; its inverse is not a count of independent showers."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('invalid group weights')
    if not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError('group weights must sum to one')
    return {'weights': weights.tolist(), 'sum_squared_weights': float(weights @ weights),
            'inverse_concentration': float(1. / (weights @ weights))}


def audit(manifests, provenance, species='NeutralsFlashCat'):
    """Quantify the historical sampling law, without changing any source or assigning physical weights."""
    if not manifests or species not in MIXERS:
        raise ValueError('nonempty manifests and a configured species required')
    groups = parent_groups(provenance)
    names = list(groups)
    group_index = {source: i for i, g in enumerate(names) for source in groups[g]}
    sources = provenance['sources']
    block_rows, matrix, exposure, seen_sources, input_hashes = [], [], [], set(), set()
    eff, factor = MIXERS[species]
    for block, manifest in manifests.items():
        if manifest.get('schema') != 'crv-mixed-replay-v1' or 'exposure_contract' in manifest:
            raise ValueError('this retained-law audit requires legacy replay-v1')
        digest = _digest(manifest['output_sha256'])
        if digest in input_hashes:
            raise ValueError('same replay input repeated')
        input_hashes.add(digest)
        events = _integer(manifest['events'], 'mixture events', 1)
        intensity = manifest['mean_intensity']
        if not np.isfinite(intensity) or intensity <= 0 or list(manifest['mixers'][species]) != [eff, factor]:
            raise ValueError('invalid intensity or mixer efficiency')
        counts = np.zeros(len(names), dtype=np.int64)
        frames = np.zeros(len(names), dtype=np.int64)
        catalog = np.zeros(len(names), dtype=np.int64)
        stepped = np.zeros(len(names), dtype=np.int64)
        selected = manifest['sources'][species]
        if not selected:
            raise ValueError('empty source subset')
        block_sources = set()
        for item in selected:
            source = source_name(item['path'])
            if source not in sources or source in block_sources:
                raise ValueError('unbound or repeated source locator')
            block_sources.add(source)
            seen_sources.add(source)
            known, metadata = sources[source], item['metadata']
            if (metadata['dataset'] != species or item['sha256'] != known['table_sha256'] or
                    _integer(metadata['frames'], 'manifest frames', 1) != known['table_frames'] or
                    _integer(metadata['stepped_frames'], 'manifest stepped frames') != known['step_events']):
                raise ValueError('manifest differs from retained source provenance')
            i = group_index[source]
            counts[i] += 1
            frames[i] += metadata['frames']
            catalog[i] += known['sam']['event_count']
            stepped[i] += metadata['stepped_frames']
        active = counts > 0
        unique_frames = frames[active] // counts[active]
        unique_steps = stepped[active] // counts[active]
        current = frames.astype(float) / frames.sum()
        alternate = np.zeros(len(names))
        alternate[active] = unique_frames / unique_frames.sum()
        expected_draws = float(events * intensity * eff * factor)
        tv = float(np.abs(current - alternate).sum() / 2)
        block_rows.append({'block': block, 'input_sha256': digest, 'events': events,
            'occurrences': len(selected), 'groups': int(active.sum()), 'occurrences_per_group': counts.tolist(),
            'table_frames_occurrences': int(frames.sum()), 'catalog_events_occurrences': int(catalog.sum()),
            'catalog_minus_table_occurrences': int(catalog.sum() - frames.sum()),
            'unique_payload_table_frames': int(unique_frames.sum()),
            'current': weight_summary(current), 'one_copy_per_group_diagnostic': weight_summary(alternate),
            'group_weight_total_variation_diagnostic': tv,
            'stepped_probability_current': float(stepped.sum() / frames.sum()),
            'stepped_probability_one_copy_diagnostic': float(unique_steps.sum() / unique_frames.sum()),
            'stepped_probability_catalog_denominator_only_diagnostic': float(stepped.sum() / catalog.sum()),
            'expected_source_draws_given_saved_mean_PBI': expected_draws})
        matrix.append(current)
        exposure.append(expected_draws)
    if seen_sources != set(sources):
        raise ValueError('provenance contains sources absent from supplied manifests')
    matrix, exposure = np.asarray(matrix), np.asarray(exposure)
    gram = matrix @ matrix.T
    norms = np.sqrt(np.diag(gram))
    pooled = np.average(matrix, axis=0, weights=exposure)
    return {'schema': 'crv-library-weight-audit-v1', 'species': species,
        'groups': [{'parents_sha256': g, 'sources': groups[g], 'table_sha256': sources[groups[g][0]]['table_sha256']}
                   for g in names], 'blocks': block_rows,
        'expected_source_draw_weighted_group_concentration': weight_summary(pooled),
        'block_group_weight_gram': gram.tolist(), 'block_group_weight_cosine': (gram / np.outer(norms, norms)).tolist(),
        'finite_library_uncertainty': None, 'physical_reuse_weighting': 'unresolved; historical occurrences unchanged',
        'scope': 'Deterministic accounting of retained manifests/catalog summaries, not a new simulation or source revalidation. '
                 'One-copy and catalog-only alternatives are diagnostics, not corrections to F. Gram/cosine and inverse '
                 'concentration describe library weights, not a measured response covariance or effective independent sample size. '
                 'Disjoint immediate parents do not certify independence of upstream ancestry. No F error or promotion is inferred.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', nargs='+', required=True)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--species', default='NeutralsFlashCat', choices=tuple(MIXERS))
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError('choose new report')
    paths = [Path(p) for p in args.manifests]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('repeated manifest path')
    inputs = [*paths, Path(args.provenance)]
    hashes = {str(p): sha256(p) for p in inputs}
    result = audit({str(p): json.loads(p.read_text()) for p in paths},
                   json.loads(Path(args.provenance).read_text()), args.species)
    if hashes != {str(p): sha256(p) for p in inputs}:
        raise ValueError('input changed during audit')
    result.update(inputs_sha256=hashes, code_sha256=sha256(__file__))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('blocks', 'expected_source_draw_weighted_group_concentration')}, indent=2))


if __name__ == '__main__':
    main()
