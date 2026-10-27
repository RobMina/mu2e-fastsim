"""Audit saved mixed-event PBI, source-pool coverage and generator-based denominators."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import ndtr
import uproot

from .mixframes import MIXERS, PBI, PBTIMES
from .replay import sha256

IDS = ('run', 'subrun', 'event')


def pbi_moment(order, *, beam_conditioned=False, extended_mean=PBI['extended_mean'],
               sdf=PBI['sdf'], cut_max=PBI['cut_max']):
    """Analytic truncated lognormal moment; beam conditioning shifts log-mean by sigma²."""
    if order < 0 or not 0 < sdf < 1 or not extended_mean > 0 or not cut_max > 0:
        raise ValueError('invalid PBI parameters')
    variance = -np.log(sdf)
    mu = np.log(extended_mean) - variance / 2 + (variance if beam_conditioned else 0.)
    sigma = np.sqrt(variance)
    z = (np.log(cut_max) - mu) / sigma
    return float(np.exp(order * mu + order * order * variance / 2) * ndtr(z - order * sigma) / ndtr(z))


def count_identities(columns):
    """Count full event tuples and detect collisions in the historical packed denominator."""
    values = np.column_stack([np.asarray(columns[k], dtype=np.int64) for k in IDS])
    unique = np.unique(values, axis=0)
    packed = (values[:, 1] << 20) | values[:, 2]
    return {'rows': len(values), 'full_event_ids': len(unique), 'historical_packed_ids': len(np.unique(packed)),
            'packed_collision': len(unique) != len(np.unique(packed)),
            'event_outside_20bit': int(np.sum((values[:, 2] < 0) | (values[:, 2] >= 2**20)))}


def audit_manifest(manifest, intensity, pbtime):
    """Assess realized mixtures, without treating source filenames as independent showers."""
    x = np.asarray(intensity, float)
    pb = np.asarray(pbtime, float)
    if not len(x) or x.shape != pb.shape or not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError('invalid saved exposure arrays')
    if manifest['events'] != len(x) or manifest['pbi'] != PBI or not np.isin(pb, PBTIMES).all():
        raise ValueError('saved exposure contract changed')
    if not np.isclose(x.mean(), manifest['mean_intensity'], rtol=0, atol=1e-8):
        raise ValueError('saved mean intensity mismatch')
    mu = pbi_moment(1)
    sd = np.sqrt(pbi_moment(2) - mu * mu)
    pools = {}
    for species, (eff, factor) in MIXERS.items():
        if list(manifest['mixers'][species]) != [eff, factor]:
            raise ValueError('mixer efficiency contract changed')
        source = manifest['sources'][species]
        meta = [s['metadata'] for s in source]
        if any(m['dataset'] != species or type(m['frames']) is not int or m['frames'] <= 0
               or not 0 <= m['stepped_frames'] <= m['frames'] for m in meta):
            raise ValueError('invalid source denominator')
        frame_counts = np.asarray([m['frames'] for m in meta])
        stepped = np.asarray([m['stepped_frames'] for m in meta])
        rates = stepped / frame_counts
        observed = manifest['mean_frames_per_event'][species]
        expected = float(x.mean() * eff * factor)
        pools[species] = {'files': len(source), 'frames': int(frame_counts.sum()),
            'stepped_frames': int(stepped.sum()), 'steps': int(sum(m['steps'] for m in meta)),
            'step_occupancy': float(stepped.sum() / frame_counts.sum()),
            'file_occupancy_min_max': [float(rates.min()), float(rates.max())],
            'drawn_mean_frames': observed, 'expected_mean_given_saved_PBI': expected,
            'conditional_poisson_mean_se': float(np.sqrt(expected / len(x))),
            'conditional_poisson_z': float((observed - expected) / np.sqrt(expected / len(x))),
            'table_sha256': [s['sha256'] for s in source],
            'table_basenames': [Path(s['path']).name for s in source],
            'denominator_provenance': 'unique generator-containing events, not an independent all-event census'}
    return {'events': len(x), 'intensity_mean': float(x.mean()), 'intensity_mean_se': float(x.std(ddof=1) / np.sqrt(len(x))),
            'intensity_mean_z_unweighted': float((x.mean() - mu) / (sd / np.sqrt(len(x)))),
            'rounded_PBI_max_difference': float(np.max(np.abs(x - np.rint(x)))),
            'pbtime_counts': {str(t): int(np.sum(pb == t)) for t in PBTIMES}, 'pools': pools}


def raw_census(path):
    """Persisted highest-cycle generator/step counts; logs are corroboration only."""
    with uproot.open(path) as root:
        gen = root['dump/gen'].arrays(list(IDS), library='np')
        steps = root['dump/steps'].arrays(list(IDS), library='np')
        keys = root.keys()
    g, s = count_identities(gen), count_identities(steps)
    gv = np.unique(np.column_stack([gen[k] for k in IDS]), axis=0)
    sv = np.unique(np.column_stack([steps[k] for k in IDS]), axis=0)
    union = np.unique(np.concatenate([gv, sv]), axis=0)
    log = path.with_suffix('.log')
    match = re.search(r'TrigReport Events total = (\d+)', log.read_text()) if log.exists() else None
    processed = int(match.group(1)) if match else None
    return {'path': str(path), 'sha256': sha256(path), 'gen': g, 'steps': s,
            'step_events_absent_from_gen': len(union) - len(gv),
            'processed_events_log': processed, 'log_matches_gen_ids': processed == len(gv) if processed is not None else None,
            'log_sha256': sha256(log) if log.exists() else None,
            'has_all_event_tree': any(k.startswith('dump/events;') for k in keys),
            'scope': 'Generator IDs counted from persisted rows; absence of empty-gen events is not established by that tree alone.'}


def efficiency_audit(path):
    """Compare explicitly supplied DB payload values, including stock's float parsing."""
    with Path(path).open() as stream:
        values = {r['tag']: float(r['eff']) for r in csv.DictReader(stream)}
    tags = {'EleBeamFlashCat': ('EleBeamCat', 'EleBeamFlashCat'),
            'MuBeamFlashCat': ('MuBeamCat', 'MuBeamFlashCat'),
            'MuStopPileupCat': ('MuBeamCat', 'MuminusStopsCat', 'MuStopPileupCat'),
            'NeutralsFlashCat': ('NeutralsCat', 'NeutralsFlashCat')}
    result = {}
    for species, names in tags.items():
        raw = np.asarray([values[n] for n in names], float)
        fast = MIXERS[species][0]
        stock = float(np.prod(raw.astype(np.float32).astype(float)))
        result[species] = {'tags': names, 'database_product': float(np.prod(raw)), 'fast_product': fast,
                           'decimal_values_match': bool(np.prod(raw) == fast),
                           'stock_float_parse_product': stock, 'fast_over_stock_minus_one': fast / stock - 1}
    return {'path': str(path), 'sha256': sha256(path), 'species': result,
            'scope': 'CID payload only; purpose/version/IOV linkage must be retained separately.'}


def duplicate_payloads(manifests):
    """Content-addressed overlap includes aliases within a subset, not just across blocks."""
    result = {}
    for species in MIXERS:
        records = {}
        for block, manifest in enumerate(manifests):
            for source in manifest['sources'][species]:
                records.setdefault(source['sha256'], []).append({'block': block, 'basename': Path(source['path']).name})
        result[species] = {'table_occurrences': sum(len(x) for x in records.values()),
                           'unique_payloads': len(records),
                           'duplicate_groups': {k: v for k, v in records.items() if len(v) > 1}}
    return result


def run(base, probes, out, efficiencies=None):
    base, probes, out = map(Path, (base, probes, out))
    if out.exists():
        raise ValueError('choose new report')
    rows, sources, manifests = [], [], []
    for block in ('block0', 'block1', 'block2_fullinventory'):
        path = base / block / 'mixed.root'
        manifest_path = path.with_suffix('.json')
        manifest = json.loads(manifest_path.read_text())
        if sha256(path) != manifest['output_sha256']:
            raise ValueError('changed mixture input')
        with uproot.open(path) as root:
            events = root['dump/events'].arrays(['intensity', 'pbtime'], library='np')
        row = audit_manifest(manifest, events['intensity'], events['pbtime'])
        row.update(block=block, input_sha256=manifest['output_sha256'], manifest_sha256=sha256(manifest_path))
        rows.append(row)
        manifests.append(manifest)
        sources.append({k: set(s['sha256'] for s in manifest['sources'][k]) for k in MIXERS})
    overlaps = {k: [len(sources[a][k] & sources[b][k]) for a, b in ((0, 1), (0, 2), (1, 2))] for k in MIXERS}
    raw = [raw_census(p) for p in sorted(probes.glob('*.root')) if p.name.startswith(('full_', 'probe_'))]
    report = {'blocks': rows, 'source_hash_overlap_01_02_12': overlaps, 'raw_census': raw,
              'source_content': duplicate_payloads(manifests),
              'efficiencies': efficiency_audit(efficiencies) if efficiencies else None,
              'PBI': {'unweighted_truncated_mean': pbi_moment(1),
                      'beam_conditioned_truncated_mean': pbi_moment(1, beam_conditioned=True),
                      'mean_ratio': pbi_moment(1, beam_conditioned=True) / pbi_moment(1)},
              'finite_library_uncertainty': None,
              'not_certified': ['cross-file shower independence', 'all-inventory empty-gen denominator coverage',
                                'selected-track/PBI correlation',
                                'new broader-subset response/stock closure'],
              'code_sha256': sha256(__file__)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'PBI': report['PBI'], 'overlaps': overlaps,
                      'raw': [{k: r[k] for k in ('path', 'gen', 'processed_events_log', 'log_matches_gen_ids')} for r in raw]}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'probes', 'out'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--efficiencies', help='saved QE sim.efficiencies2 CSV for the explicitly reviewed CID')
    args = ap.parse_args(argv)
    run(args.base, args.probes, args.out, args.efficiencies)


if __name__ == '__main__':
    main()
