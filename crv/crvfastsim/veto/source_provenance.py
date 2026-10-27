"""Trace retained Cat table aliases to extraction lists/logs and source-art event metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import uproot

from .catsteps import read_meta
from .replay import sha256

IDS = ['run', 'subrun', 'event']


def source_name(table):
    """Recover the source-art basename, without inferring its event IDs from its name."""
    name = Path(table).name
    if not name.startswith('steps.dts.') or not name.endswith('.parquet'):
        raise ValueError('not a Cat table basename')
    return name[len('steps.'):-len('.parquet')] + '.art'


def extraction_index(path):
    """Require unique, explicit source paths in the recorded extraction list."""
    result = {}
    for line, text in enumerate(Path(path).read_text().splitlines(), 1):
        if not text.strip():
            continue
        species, source = text.split()
        name = Path(source).name
        if name in result:
            raise ValueError(f'duplicate source basename in extraction list: {name}')
        result[name] = {'species': species, 'path': source, 'line': line}
    return result


def identity_summary(frame):
    """Summarize actual stored IDs; file subruns need not equal contained event subruns."""
    values = frame[IDS].drop_duplicates().sort_values(IDS).to_numpy(dtype='<i8')
    return {'unique_events': len(values),
            'id_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
            'run_values': np.unique(values[:, 0]).tolist(),
            'subrun_count': int(len(np.unique(values[:, 1]))),
            'subrun_min_max': [int(values[:, 1].min()), int(values[:, 1].max())] if len(values) else None,
            'event_min_max': [int(values[:, 2].min()), int(values[:, 2].max())] if len(values) else None}


def log_matches(logdir, names):
    """Record matching conversion/copy/error lines and bind each retained log once."""
    result = {n: [] for n in names}
    for path in sorted(Path(logdir).rglob('*')):
        if not path.is_file() or path.suffix not in ('.out', '.err'):
            continue
        matches = []
        for line, text in enumerate(path.read_text(errors='replace').splitlines(), 1):
            for name in names:
                if name.removesuffix('.art') in text:
                    matches.append((name, line, text))
        if matches:
            digest = sha256(path)
            for name, line, text in matches:
                result[name].append({'log': str(path), 'sha256': digest, 'line': line, 'text': text})
    return result


def art_header(path):
    """Read ROOT metadata and event auxiliary IDs, never art physics product collections."""
    with uproot.open(path, timeout=60) as root:
        tree = root['Events']
        result = {'path': path, 'events': tree.num_entries,
                  'auxiliary_branches': [k for k in tree.keys() if 'EventAuxiliary' in k]}
        candidates = [k for k in tree.keys() if 'EventAuxiliary' in k and re.search(r'(event_|subRun_|run_)$', k)]
        result['candidate_ids'] = candidates
        result['file_uuid'] = str(root.file.uuid)
        if not candidates and 'EventAuxiliary' in tree:
            result['event_identity_status'] = 'Unsplit art EventAuxiliary requires an art census; no uproot object-streaming fallback.'
        if candidates:
            arrays = tree.arrays(candidates, library='np')
            result['auxiliary_id_columns'] = {k: {'unique': np.unique(v).tolist() if len(np.unique(v)) <= 100 else None,
                'min': int(v.min()) if len(v) else None, 'max': int(v.max()) if len(v) else None,
                'sha256': hashlib.sha256(np.asarray(v).tobytes()).hexdigest()} for k, v in arrays.items()}
        return result


def sam_summary(path):
    """Bind saved catalog parentage; catalog content hashes do not replace event censuses."""
    text = Path(path).read_text()
    def field(name):
        match = re.search(r'^\s*' + re.escape(name) + r':\s*(.+)$', text, re.M)
        return match.group(1) if match else None
    parents = sorted(re.findall(r'dts\.mu2e\.NeutralsFlash\.MDC2025ac\.[0-9_]+\.art', text))
    if not parents or len(set(parents)) != len(parents):
        raise ValueError('missing or duplicate parent declarations')
    return {'metadata_sha256': sha256(path), 'file_name': field('File Name'),
            'file_id': field('File Id'), 'source_sha256': field('dh.sha256'),
            'event_count': int(field('Event Count')), 'parent_count': len(parents),
            'parents_sha256': hashlib.sha256(('\n'.join(parents) + '\n').encode()).hexdigest()}


def summarize_prior(prior, sam_dir, out):
    """Compact saved evidence without rereading source inputs or scanning storage namespaces."""
    report = json.loads(Path(prior).read_text())
    summaries = {s['file_name']: s for path in sorted(Path(sam_dir).glob('sam_*.txt'))
                 for s in [sam_summary(path)]}
    rows = {}
    for name, record in report['sources'].items():
        ids = record['step_ids']
        rows[name] = {'table_sha256': record['table_sha256'], 'table_frames': record['metadata']['frames'],
                     'step_event_ids_sha256': ids['id_sha256'], 'step_events': ids['unique_events'],
                     'source_art': record['source_art'], 'sam': summaries.get(name),
                     'log_matches': len(record['extraction_log_matches'])}
        if name in summaries:
            rows[name]['catalog_events_minus_table_frames'] = summaries[name]['event_count'] - record['metadata']['frames']
    parent_sets = {}
    for path in sorted(Path(sam_dir).glob('sam_*.txt')):
        name = sam_summary(path)['file_name']
        parent_sets[name] = set(re.findall(r'dts\.mu2e\.NeutralsFlash\.MDC2025ac\.[0-9_]+\.art', path.read_text()))
    overlaps = [{'first': a, 'second': b, 'shared_parents': len(parent_sets[a] & parent_sets[b]),
                 'identical_parent_sets': parent_sets[a] == parent_sets[b]} for a in sorted(parent_sets)
                for b in sorted(parent_sets) if a < b]
    result = {'prior_sha256': sha256(prior), 'code_sha256': sha256(__file__), 'sources': rows,
              'parent_overlaps': overlaps,
              'scope': 'Repeated parent sets establish upstream overlap, not its intended production weighting; denominator differences need explicit full-event identities.'}
    with Path(out).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


def run(base, extraction_list, logs, out, art_headers=False):
    base, out = Path(base), Path(out)
    if out.exists():
        raise ValueError('choose a new report')
    index = extraction_index(extraction_list)
    sources = {}
    manifests = {}
    for block in ('block0', 'block1', 'block2_fullinventory'):
        manifest_path = base / block / 'mixed.json'
        manifests[block] = sha256(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        declared = {Path(s['path']).name: s for s in manifest['sources']['NeutralsFlashCat']}
        for line in (base / block / 'subset.txt').read_text().splitlines():
            species, path = line.split()
            if species != 'NeutralsFlashCat':
                continue
            name = source_name(path)
            if name not in index or index[name]['species'] != species:
                raise ValueError(f'missing extraction binding: {name}')
            if name not in sources:
                digest = sha256(path)
                if digest != declared[Path(path).name]['sha256']:
                    raise ValueError('retained table payload changed')
                frame = pd.read_parquet(path, columns=IDS)
                sources[name] = {'table_path': path, 'table_sha256': digest, 'metadata': read_meta(path),
                    'source_art': index[name], 'step_ids': identity_summary(frame), 'blocks': []}
            sources[name]['blocks'].append(block)
    matches = log_matches(logs, sources)
    for name, record in sources.items():
        record['extraction_log_matches'] = matches[name]
        if art_headers:
            record['source_art_header'] = art_header(record['source_art']['path'])
    report = {'schema': 'crv-source-provenance-audit-v1', 'manifests': manifests,
              'extraction_list': {'path': str(extraction_list), 'sha256': sha256(extraction_list)},
              'sources': sources, 'code_sha256': sha256(__file__),
              'scope': 'No deduplication or renormalization; table equality alone does not identify the production cause.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({name: {'hash': s['table_sha256'], 'source': s['source_art'], 'ids': s['step_ids'],
                            'log_matches': len(s['extraction_log_matches']), 'art': s.get('source_art_header')}
                      for name, s in sources.items()}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'extraction-list', 'logs'):
        parser.add_argument('--' + name)
    parser.add_argument('--out', required=True)
    parser.add_argument('--prior', help='extend a retained report without rereading source parquets')
    parser.add_argument('--art-names', nargs='*', help='explicit subset of source basenames for header inspection')
    parser.add_argument('--art-headers', action='store_true')
    parser.add_argument('--sam-dir', help='summarize prior report with saved exact-file SAM metadata')
    args = parser.parse_args(argv)
    if args.sam_dir:
        if not args.prior:
            parser.error('sam-dir requires prior')
        summarize_prior(args.prior, args.sam_dir, args.out)
    elif args.prior:
        report = json.loads(Path(args.prior).read_text())
        report['parent_report_sha256'] = sha256(args.prior)
        report['code_sha256'] = sha256(__file__)
        if Path(args.out).exists():
            raise ValueError('choose new report')
        for name in args.art_names or report['sources']:
            record = report['sources'][name]
            record['source_art_header'] = art_header(record['source_art']['path'])
            print(json.dumps({name: record['source_art_header']}, indent=2), flush=True)
        with Path(args.out).open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    else:
        if not all((args.base, args.extraction_list, args.logs)):
            parser.error('supply base, extraction-list and logs, or prior')
        run(args.base, args.extraction_list, args.logs, args.out, args.art_headers)


if __name__ == '__main__':
    main()
