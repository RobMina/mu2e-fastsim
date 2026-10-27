"""Manifest-driven library x conditional draw x held-out validation-shard diagnostics."""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import uproot

from .campaign import logical_physics
from .library_uncertainty import build_identity, components
from .library_variants import SCHEMA as PLAN_SCHEMA, _event_inventory, _variants
from .paired_rates import read
from .pilot import digest


SCHEMA = 'valhist.library_ensemble.v1'
HISTOGRAMS = ('PEc', 'NPc', 'X_extracted', 'Y_extracted', 'Z_extracted')
ARMS = ('a1', 'e2e')
LIMITATIONS = [
    'Diagnostic nested method of moments, not calibrated confidence coverage or a DQM acceptance test.',
    'Libraries are whole-donor-file bootstrap realizations from one common plan, not independent donor datasets.',
    'Independent evaluation shards remain a validation axis; they are never counted as additional libraries.',
    'Conditional on these exact evaluation events, fixed generator denominator, transport, stock baseSeed and conditions.',
    'Distinct declared tier1b/fast-response seeds do not empirically prove random-stream independence; stock streams remain fixed.',
    'Donor variation includes refitted quantile edges; denominator uncertainty and validation-population uncertainty are not resampled.',
    'Disjointness covers persisted full event keys only, not absent donor events or independence of production conditions.',
    'Negative noise-subtracted variances and indefinite covariance matrices are retained, not evidence of negative physical variance.',
    'No extrapolation beyond the listed shards; no cross-bin covariance, global significance, or calibrated finite-library standard error.',
    'Physics hashes are matched historical provenance, not a re-execution or retroactive certification of missing inputs.',
]


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def _path(value, parent):
    if not isinstance(value, str) or not value:
        raise ValueError('expected a nonempty path string')
    return (parent / value).resolve(strict=True)


def _bind(path, bindings, expected=None):
    path = Path(path)
    key = str(path)
    if key not in bindings:
        bindings[key] = digest(path)
    if expected is not None and bindings[key] != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    return bindings[key]


def _json(path, bindings, sealed=False):
    sha = _bind(path, bindings)
    if sealed:
        sidecar = path.with_suffix(path.suffix + '.sha256')
        _bind(sidecar, bindings)
        if sidecar.read_text().strip() != sha:
            raise ValueError(f'metadata seal mismatch: {path}')
    return json.loads(path.read_text())


def validate_manifest(document, parent):
    """Resolve explicit paths and require a complete balanced Cartesian design."""
    allowed = {'schema', 'run', 'cells', 'generator_denominator_files', 'generator_denominator_metadata'}
    if not isinstance(document, dict) or set(document) - allowed or document.get('schema') != SCHEMA:
        raise ValueError('unsupported ensemble manifest schema or fields')
    run_number = _integer(document.get('run', 1400), 'run', 0)
    cells = document.get('cells')
    if not isinstance(cells, list) or not cells:
        raise ValueError('cells must be a nonempty list')
    required = {'index', 'draw', 'shard', 'directory', 'library_metadata', 'seed', 'expected_generated_events'}
    normalized, used, seeds, directories = [], set(), set(), set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != required:
            raise ValueError('cell fields must be exactly ' + ', '.join(sorted(required)))
        row = dict(cell)
        for key in ('index', 'draw', 'shard', 'expected_generated_events'):
            _integer(row[key], key)
        _integer(row['seed'], 'seed', 0)
        key = tuple(row[k] for k in ('index', 'draw', 'shard'))
        if key in used:
            raise ValueError('duplicate library/draw/shard cell')
        if row['seed'] in seeds:
            raise ValueError('independent cells require globally distinct declared seeds')
        row['directory'] = _path(row['directory'], parent)
        row['library_metadata'] = _path(row['library_metadata'], parent)
        if row['directory'] in directories:
            raise ValueError('reused replay directory')
        used.add(key)
        seeds.add(row['seed'])
        directories.add(row['directory'])
        normalized.append(row)
    axes = {k: sorted({r[k] for r in normalized}) for k in ('index', 'draw', 'shard')}
    if any(len(v) < 2 for v in axes.values()):
        raise ValueError('full report requires >=2 libraries, >=2 draws and >=2 distinct shards')
    if used != set(itertools.product(*axes.values())):
        raise ValueError('require balanced library x draw x shard cells; missing cells are not dropped')
    for index in axes['index']:
        if len({r['library_metadata'] for r in normalized if r['index'] == index}) != 1:
            raise ValueError('each library realization must be shared globally across draws and shards')
    for shard in axes['shard']:
        if len({r['expected_generated_events'] for r in normalized if r['shard'] == shard}) != 1:
            raise ValueError('expected exposure must match within each shard')
    return sorted(normalized, key=lambda r: (r['index'], r['draw'], r['shard'])), axes, run_number


def covariance(values):
    """Same-observable cross-shard nested covariance; output axes are (..., shard, shard)."""
    x = np.asarray(values, float)
    if x.ndim < 3 or min(x.shape[:3]) < 2 or not np.isfinite(x).all():
        raise ValueError('need finite (>=2 libraries, >=2 draws, >=2 shards, ...) values')
    libraries, draws = x.shape[:2]
    means = x.mean(axis=1)
    residual = x - means[:, None]
    centered = means - means.mean(axis=0)
    within = np.einsum('lds...,ldt...->...st', residual, residual) / (libraries * (draws - 1))
    between = np.einsum('ls...,lt...->...st', centered, centered) / (libraries - 1)
    return dict(conditional_draw_covariance=within.tolist(), library_mean_covariance=between.tolist(),
                library_covariance_noise_subtracted=(between - within / draws).tolist())


def _nullable(values):
    values = np.asarray(values)
    return np.where(np.isfinite(values), values, None).tolist()


def _estimate(values):
    if not np.isfinite(values).all():
        return dict(status='undefined_zero_histogram_total', components=None)
    return dict(status='defined', components=components(values))


def _statistics(per_shard, pooled, shards):
    result = dict(values_per_shard=_nullable(per_shard), values_pooled=_nullable(pooled),
                  pooled=_estimate(pooled),
                  per_shard={str(s): _estimate(per_shard[:, :, j]) for j, s in enumerate(shards)})
    result['cross_shard'] = (dict(status='defined', **covariance(per_shard)) if np.isfinite(per_shard).all()
                             else dict(status='undefined_zero_histogram_total', covariance=None))
    return result


def summarize(raw, exposure, shards):
    """Pool counts and generated exposures before any rates or inclusive normalization."""
    exposure = np.asarray(exposure, float)
    if exposure.shape != (len(shards),) or not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise ValueError('need one positive exposure per shard')
    vectors = {}
    for arm in ARMS:
        count = np.asarray(raw[arm]['cluster_count'], float)
        if count.ndim != 3 or count.shape[2] != len(shards) or min(count.shape[:2]) < 2 or len(shards) < 2:
            raise ValueError('invalid balanced count array')
        if not np.isfinite(count).all() or np.any(count < 0):
            raise ValueError('invalid cluster counts')
        vectors[arm] = {'cluster_rate': (count / exposure, count.sum(axis=2) / exposure.sum())}
        for name in HISTOGRAMS:
            hist = np.asarray(raw[arm][name], float)
            if hist.ndim != 4 or hist.shape[:3] != count.shape or not np.isfinite(hist).all() or np.any(hist < 0):
                raise ValueError('invalid inclusive histogram counts')
            pooled = hist.sum(axis=2)
            with np.errstate(divide='ignore', invalid='ignore'):
                vectors[arm][name + '_inclusive_fraction'] = (hist / hist.sum(axis=-1, keepdims=True),
                                                              pooled / pooled.sum(axis=-1, keepdims=True))
    if np.shape(raw['a1']['cluster_count']) != np.shape(raw['e2e']['cluster_count']):
        raise ValueError('arm shapes differ')
    vectors['e2e_minus_a1'] = {name: tuple(b - a for a, b in zip(pair, vectors['e2e'][name]))
                               for name, pair in vectors['a1'].items()}
    return {arm: {name: _statistics(*pair, shards) for name, pair in metrics.items()}
            for arm, metrics in vectors.items()}


def _library(path, index, bindings):
    info = _json(path, bindings, sealed=True)
    provenance = info['provenance']
    plan_path = path.parent / 'plan.json'
    plan = _json(plan_path, bindings, sealed=True)
    _bind(plan_path, bindings, provenance['plan_sha256'])
    variant = f'bootstrap_{index:04d}'
    if plan['schema'] != PLAN_SCHEMA or provenance['schema'] != PLAN_SCHEMA or provenance['variant'] != variant:
        raise ValueError('require matching planned bootstrap index and provenance schema')
    n = len(plan['donors'])
    generated = _integer(plan['generated_per_file'], 'generated_per_file')
    if n < 2 or plan['nominal_generated'] != n * generated:
        raise ValueError('invalid full-size donor exposure')
    selections = _variants(n, generated, plan['bootstrap'], plan['seed'])
    if plan['variants'] != selections or variant not in selections:
        raise ValueError('plan selections disagree with deterministic bootstrap specification')
    selection = selections[variant]
    inputs = [plan['donors'][i]['path'] for i in selection['donor_indices']]
    if (provenance['selection'] != selection or info['inputs'] != inputs or info['files'] != n or
            info['generated_scaled_to'] != plan['nominal_generated'] or
            info['gen'] != [r['path'] for r in plan['gen']] or
            provenance['build_code_sha256'] != plan['build_code_sha256']):
        raise ValueError('library selection, denominator, exposure or build identity differs from plan')
    _bind(path.with_suffix('.root'), bindings, provenance['library_sha256'])
    return info, plan


def _inventory(path, tree, bindings, expected=None):
    _bind(path, bindings, expected)
    return _event_inventory(path, tree)


def _disjoint(left, right, label):
    if len(np.unique(np.concatenate([left, right]), axis=0)) != len(left) + len(right):
        raise ValueError(label + ' event-key overlap')


def _audit_plan(plan, bindings):
    banks, evidence = {}, {}
    for name, tree in [('donors', 'dump/particles'), ('gen', 'dump/gen')]:
        populations = []
        if not plan[name]:
            raise ValueError('empty source bank: ' + name)
        for record in plan[name]:
            keys, inventory = _inventory(Path(record['path']), tree, bindings, record['sha256'])
            if inventory != record['events']:
                raise ValueError('persisted source inventory differs from plan')
            populations.append(keys)
        combined = np.concatenate(populations)
        banks[name] = np.unique(combined, axis=0)
        if len(combined) != len(banks[name]):
            raise ValueError('cross-file source event-key overlap: ' + name)
        evidence[name] = dict(files=plan[name], unique_events=len(combined))
    _disjoint(banks['donors'], banks['gen'], 'donor/denominator')
    return banks, evidence


def _extra_denominators(document, parent, bindings):
    paths, metadata_evidence = [], None
    if 'generator_denominator_files' in document:
        entries = document['generator_denominator_files']
        if not isinstance(entries, list) or not entries:
            raise ValueError('explicit generator_denominator_files must be nonempty')
        paths.extend(_path(p, parent) for p in entries)
    if 'generator_denominator_metadata' in document:
        metadata_path = _path(document['generator_denominator_metadata'], parent)
        metadata = _json(metadata_path, bindings)
        if not isinstance(metadata.get('gen'), list) or not metadata['gen']:
            raise ValueError('explicit denominator metadata requires a nonempty gen list')
        paths.extend(_path(p, metadata_path.parent) for p in metadata['gen'])
        metadata_evidence = dict(path=str(metadata_path), sha256=bindings[str(metadata_path)])
    return sorted(set(paths)), metadata_evidence


def _cell(row, run_number, library, bindings, transport, bookings):
    directory, shard = row['directory'], row['shard']
    manifest = _json(directory / 'manifest.json', bindings)
    complete = _json(directory / 'completion.json', bindings)
    expected = row['expected_generated_events']
    if (complete.get('completed') is not True or complete.get('changed_inputs') != [] or
            complete.get('expected_generated_events') != expected or
            manifest['generated']['unique_events'] != expected or manifest['seed'] != row['seed']):
        raise ValueError('incomplete, drifted, wrong exposure or wrong seed replay')
    if manifest['files'][manifest['library']] != library['provenance']['library_sha256']:
        raise ValueError('replay does not use the shared library realization')
    _bind(Path(manifest['library']), bindings, library['provenance']['library_sha256'])
    sha = manifest['files'][manifest['fastc']]
    if shard not in transport:
        keys, inventory = _inventory(Path(manifest['fastc']), 'dump/gen', bindings, sha)
        if len(keys) != expected or not np.all(keys[:, :2] == [run_number, shard]):
            raise ValueError('wrong actual transport event population')
        transport[shard] = (sha, keys, inventory)
    else:
        _bind(Path(manifest['fastc']), bindings, sha)
    transport_sha, keys, inventory = transport[shard]
    fields = ('rows', 'unique_events', 'event_keys_sha256', 'cycles', 'selected_cycle')
    if sha != transport_sha or any(manifest['generated'].get(k) != inventory[k] for k in fields):
        raise ValueError('transport identity or generated inventory differs within shard')
    values = {}
    for arm in ARMS:
        path = directory / arm / 'valhist.root'
        saved = complete['arms'][arm]
        _bind(path, bindings, saved['sha256'])
        if saved['completed'] is not True or Path(saved['path']).resolve() != path:
            raise ValueError('incomplete arm or output path mismatch')
        persisted, audited = _event_inventory(path, 'crvdump/events')
        actual, rates, census = read(path)
        recorded = saved['trees']['events']
        for ev in (census, recorded, audited):
            if (ev['rows'] != expected or ev['unique_events'] != expected or
                    ev['event_keys_sha256'] != inventory['event_keys_sha256']):
                raise ValueError('persisted event census differs from exact transport population')
        if not np.array_equal(actual, keys) or not np.array_equal(persisted, keys):
            raise ValueError('persisted event identities differ from transport')
        counts = np.asarray(rates['ncluster'])
        if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError('invalid cluster census')
        values[arm] = dict(cluster_count=float(counts.sum()))
        with uproot.open(path) as root:
            for name in HISTOGRAMS:
                hist = root['crvRecoDQM/' + name]
                if not hist.classname.startswith('TH1') or len(hist.axes) != 1:
                    raise ValueError('require one-dimensional count histogram')
                edges = np.asarray(hist.axis().edges(), float)
                booking = dict(classname=hist.classname, edges=edges.tolist())
                if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
                    raise ValueError('invalid histogram edges')
                if name in bookings and bookings[name] != booking:
                    raise ValueError('histogram booking differs across arms/cells')
                bookings[name] = booking
                h = np.asarray(hist.values(flow=True), float)
                if not np.isfinite(h).all() or np.any(h < 0) or np.any(h != np.floor(h)):
                    raise ValueError('invalid count histogram')
                if h.sum() != counts.sum():
                    raise ValueError('inclusive histogram total disagrees with cluster census')
                values[arm][name] = h
    evidence = {k: str(v) if isinstance(v, Path) else v for k, v in row.items()}
    evidence.update(manifest_sha256=bindings[str(directory / 'manifest.json')],
                    completion_sha256=bindings[str(directory / 'completion.json')],
                    library_metadata_sha256=bindings[str(row['library_metadata'])],
                    event_keys_sha256=inventory['event_keys_sha256'], transport_sha256=sha,
                    output_sha256={a: complete['arms'][a]['sha256'] for a in ARMS})
    return values, logical_physics(manifest), evidence


def run(manifest, out):
    out = Path(out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    manifest = Path(manifest).resolve(strict=True)
    bindings = {}
    sources = {str(p): digest(p) for p in (Path(__file__).resolve(), Path(__file__).with_name('library_uncertainty.py'),
               Path(__file__).with_name('campaign.py'), Path(__file__).with_name('library_variants.py'),
               Path(__file__).with_name('paired_rates.py'), Path(__file__).with_name('pilot.py'))}
    document = _json(manifest, bindings)
    rows, axes, run_number = validate_manifest(document, manifest.parent)
    libraries, common_build, plan = {}, None, None
    for index in axes['index']:
        row = next(r for r in rows if r['index'] == index)
        info, current_plan = _library(row['library_metadata'], index, bindings)
        identity = build_identity(info)
        if common_build is not None and identity != common_build:
            raise ValueError('libraries require common plan, denominator, build code and geometry')
        libraries[index], common_build, plan = info, identity, current_plan
    if len({v['provenance']['library_sha256'] for v in libraries.values()}) != len(libraries):
        raise ValueError('duplicate built library realizations are not independent libraries')
    banks, audit = _audit_plan(plan, bindings)
    extras, metadata_evidence = _extra_denominators(document, manifest.parent, bindings)
    extra_keys, extra_evidence = [], []
    for path in extras:
        keys, inventory = _inventory(path, 'dump/gen', bindings)
        _disjoint(banks['donors'], keys, 'donor/explicit denominator')
        extra_keys.append(keys)
        extra_evidence.append(dict(path=str(path), sha256=bindings[str(path)], events=inventory))
    raw = {a: {name: [] for name in ('cluster_count',) + HISTOGRAMS} for a in ARMS}
    transport, bookings, provenance, physics = {}, {}, [], None
    for row in rows:
        values, identity, evidence = _cell(row, run_number, libraries[row['index']], bindings, transport, bookings)
        if physics is not None and identity != physics:
            raise ValueError('replays use different conditions or physics inputs')
        physics = identity
        provenance.append(evidence)
        for arm, metrics in values.items():
            for name, value in metrics.items():
                raw[arm][name].append(value)
    validation = np.concatenate([transport[s][1] for s in axes['shard']])
    if len(np.unique(validation, axis=0)) != len(validation):
        raise ValueError('validation shards have overlapping event keys')
    for label, keys in banks.items():
        _disjoint(validation, keys, 'validation/' + label)
    for keys in extra_keys:
        _disjoint(validation, keys, 'validation/explicit denominator')
    shape = tuple(len(axes[k]) for k in ('index', 'draw', 'shard'))
    raw = {a: {k: np.asarray(v).reshape(shape + np.asarray(v).shape[1:]) for k, v in metrics.items()}
           for a, metrics in raw.items()}
    exposure = [len(transport[s][1]) for s in axes['shard']]
    result = dict(schema=SCHEMA, scope='Multi-shard conditional donor-file bootstrap variance diagnostic',
                  axes=axes, array_order=['index', 'draw', 'shard', 'histogram_bin_if_present'],
                  generated_events_per_shard=exposure, pooled_generated_events=sum(exposure),
                  raw_counts={a: {k: v.tolist() for k, v in metrics.items()} for a, metrics in raw.items()},
                  pooled_counts={a: {k: v.sum(axis=2).tolist() for k, v in metrics.items()}
                                 for a, metrics in raw.items()},
                  histogram_bookings=bookings, metrics=summarize(raw, exposure, axes['shard']),
                  provenance=provenance, physics=physics, build_identity=common_build,
                  source_audit=dict(**audit, validation_unique_events=len(validation),
                                    validation_donor_overlap=0, validation_denominator_overlap=0,
                                    explicit_denominator_files=extra_evidence,
                                    explicit_denominator_metadata=metadata_evidence),
                  input_sha256=bindings, analysis_source_sha256=sources, limitations=LIMITATIONS)
    for filename, sha in {**bindings, **sources}.items():
        if digest(filename) != sha:
            raise ValueError('input changed during report: ' + filename)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with out.open('x') as stream:
        stream.write(text)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='explicit balanced JSON cells; paths relative to this JSON')
    parser.add_argument('--out', required=True, help='new JSON file; existing parent directory required')
    args = parser.parse_args(argv)
    try:
        report = run(args.manifest, args.out)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'library_ensemble: {error}\n')
    print(json.dumps(dict(axes=report['axes'], pooled_generated_events=report['pooled_generated_events'])))


if __name__ == '__main__':
    main()
