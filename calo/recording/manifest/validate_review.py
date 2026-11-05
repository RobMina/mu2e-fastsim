#!/usr/bin/env python3
"""External preflight and completed-job audit; no downstream response input.

ROOT reading requires uproot. Unit/source tests do not require it. An expected
EventID list must come from independent source enumeration or an approved
EmptyEvent run/subrun/first-event/count specification, never a response table.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_snapshot(root=HERE / 'snapshot', manifest=HERE / 'source_manifest.json'):
    pins = json.loads(Path(manifest).read_text())
    for rel, expected in pins['files'].items():
        if sha256(Path(root) / rel) != expected:
            raise ValueError(f'snapshot mismatch: {rel}')
    for rel, expected in pins.get('proposal_files', {}).items():
        if sha256(Path(manifest).parent / rel) != expected:
            raise ValueError(f'proposal mismatch: {rel}')


def normalized_fcl_sha256(path):
    # Lex only: preserve all bytes except two actual unquoted key/string values.
    # Expanded FHiCL uses # comments; also handle C/C++ comments defensively.
    text = Path(path).read_bytes().decode('utf-8')
    token = re.compile(r'(?P<space>\s+)|(?P<comment>\#[^\n]*|//[^\n]*|/\*.*?\*/)|'
                       r'(?P<string>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|'
                       r'(?P<word>[A-Za-z_][A-Za-z_0-9.]*)|(?P<other>.)', re.S)
    tokens = [m for m in token.finditer(text) if m.lastgroup not in ('space', 'comment')]
    if any((m.lastgroup == 'other' and m.group() in ('\"', "'"))
           or (m.lastgroup == 'other' and text.startswith('/*', m.start()))
           for m in tokens):
        raise ValueError('unterminated quoted string or comment in expanded FCL')
    spans = []
    for i, match in enumerate(tokens):
        if match.group() != 'expandedFclSha256' or match.lastgroup != 'word':
            continue
        if (i + 2 >= len(tokens) or tokens[i + 1].group() != ':'
                or tokens[i + 2].lastgroup != 'string'
                or not tokens[i + 2].group().startswith('"')):
            raise ValueError('expandedFclSha256 must have a quoted string value')
        spans.append(tokens[i + 2].span())
    if len(spans) != 2:
        raise ValueError('expected exactly two expandedFclSha256 values in expanded FCL')
    for start, end in reversed(spans):
        text = text[:start] + '"<NORMALIZED_FCL_SHA256>"' + text[end:]
    return hashlib.sha256(text.encode()).hexdigest()


def verify_claims(claim, artifact, snapshot_manifest, expanded_fcl):
    """Hash actual supplied bytes. Does NOT authenticate who supplied the paths."""
    expected = {
        'source_sha256': sha256(artifact),
        'source_snapshot_sha256': sha256(snapshot_manifest),
        'expanded_fcl_sha256': normalized_fcl_sha256(expanded_fcl),
    }
    for key, digest in expected.items():
        if claim[key] != digest:
            raise ValueError(f'{key} external hash mismatch')


ID_FIELDS = ('run', 'subrun', 'event')
STATE_FIELDS = ('time', 'position_x', 'position_y', 'position_z',
                'momentum_px', 'momentum_py', 'momentum_pz', 'momentum_e')


def expected_event_ids(text):
    """Strict JSON integer triples; reject bool/float/string and uint32 overflow."""
    integer = r'(?:0|[1-9][0-9]*)'
    triple = rf'\[\s*{integer}\s*,\s*{integer}\s*,\s*{integer}\s*\]'
    if not re.fullmatch(rf'\s*\[\s*(?:{triple}(?:\s*,\s*{triple})*)?\s*\]\s*', text):
        raise ValueError('expected EventIDs must be JSON integer triples')

    def bounded(value):
        if len(value) > 10 or int(value) > np.iinfo(np.uint32).max:
            raise ValueError('expected EventID outside uint32 range')
        return int(value)

    return np.asarray(json.loads(text, parse_int=bounded), dtype=np.uint32).reshape(-1, 3)


def validate_columns(inputs, events, products, roots, expected_ids, mode='record_all'):
    """Vectorized completed-job audit in native collection order; never repair gaps."""
    if mode not in ('record_all', 'require_one'):
        raise ValueError('unknown multiplicity policy')
    n = len(expected_ids)
    for table in (inputs, events, products):
        if not np.array_equal(table['ordinal'], np.arange(n, dtype=np.uint64)):
            raise ValueError('missing, repeated, or reordered event ordinal')
        for col, name in enumerate(ID_FIELDS):
            if not np.array_equal(table[name], expected_ids[:, col]):
                raise ValueError('EventID sequence differs from independent expectation')
    if not np.all(events['present']) or not np.all(products['present']):
        raise ValueError('missing designated product is not empty generation')
    counts = events['n_roots']
    nr = len(roots['ordinal'])
    if counts.dtype.kind not in 'iu' or np.any(counts < 0) or np.any(counts > nr):
        raise ValueError('invalid root count')
    if not np.array_equal(products['size'], counts) or np.any(products['kind'] != 2):
        raise ValueError('collection count/type mismatch')
    if np.any(products['type'] != 'mu2e::StageParticleCollection') or np.any(products['branch_name'] == ''):
        raise ValueError('missing StageParticle registry branch')
    if not np.array_equal(events['product_id'], products['product_id']) or np.any(products['product_id'] == 0):
        raise ValueError('product identity mismatch')
    if not np.array_equal(events['single_root_supported'], counts == 1):
        raise ValueError('single-root support flag mismatch')
    if mode == 'require_one' and np.any(counts != 1):
        raise ValueError('strict single-root violation')
    if np.sum(counts, dtype=np.uint64) != nr:
        raise ValueError('missing or extra root rows')
    counts = counts.astype(np.intp, copy=False)
    owners = np.repeat(np.arange(n, dtype=np.uint64), counts)
    starts = np.cumsum(counts) - counts
    indices = np.arange(nr) - np.repeat(starts, counts)
    if (not np.array_equal(roots['ordinal'], owners)
            or not np.array_equal(roots['index'], indices)
            or not np.array_equal(roots['product_id'], np.repeat(products['product_id'], counts))
            or np.any(roots['kind'] != 2)):
        raise ValueError('root identity/order mismatch')
    for col, name in enumerate(ID_FIELDS):
        if not np.array_equal(roots[name], np.repeat(expected_ids[:, col], counts)):
            raise ValueError('root EventID mismatch')
    for name in STATE_FIELDS:
        values = roots[name]
        if values.dtype.kind != 'f' or values.dtype.itemsize != 8:
            raise ValueError('native state branch must be double')
        if not np.all(np.isfinite(values)):
            raise ValueError('nonfinite native state')


def validate_rows(inputs, events, products, roots, expected_ids, mode='record_all'):
    """Compatibility for small list-of-dict synthetic fixtures only, not ROOT IO."""
    def columns(rows, fields):
        return {key: np.asarray([row[key] for row in rows], dtype=dtype)
                for key, dtype in fields.items()}
    ids = dict.fromkeys(('ordinal',) + ID_FIELDS, np.uint64)
    inp = columns(inputs, ids)
    evt = columns(events, dict(ids, present=bool, n_roots=np.int64,
                              product_id=np.uint32, single_root_supported=bool))
    prod = columns(products, dict(ids, present=bool, size=np.int64, product_id=np.uint32,
                                 kind=np.int32, type=str, branch_name=str))
    root = columns(roots, dict(ids, index=np.uint64, product_id=np.uint32, kind=np.int32,
                              **dict.fromkeys(STATE_FIELDS, np.float64)))
    validate_columns(inp, evt, prod, root, expected_event_ids(json.dumps(expected_ids)), mode)


def read_columns(tree):
    arrays = tree.arrays(library='np')
    # TTree returns a mapping; structured arrays are accepted without row conversion.
    columns = arrays if isinstance(arrays, dict) else {k: arrays[k] for k in arrays.dtype.names}
    if any(v.ndim != 1 or len(v) != tree.num_entries for v in columns.values()):
        raise ValueError('manifest branches must be scalar columns')
    return columns


def read_metadata(tree):
    if tree.num_entries != 1:
        raise ValueError('expected one metadata row per module')
    return {key: values[0] for key, values in read_columns(tree).items()}


def audit(args):
    import uproot  # optional until an actual art/ROOT integration exists
    check_snapshot(manifest=args.snapshot_manifest)
    expected = expected_event_ids(Path(args.expected_events).read_text())
    with uproot.open(args.manifest) as f:
        im = read_metadata(f['inputEventManifest/metadata'])
        rm = read_metadata(f['incidentManifest/metadata'])
        for meta in (im, rm):
            if meta['schema_version'] != 1 or not meta['source_artifact']:
                raise ValueError('invalid schema or empty source artifact ID')
            verify_claims(meta, args.artifact, args.snapshot_manifest, args.expanded_fcl)
        for key in ('source_artifact', 'source_sha256', 'source_snapshot_sha256', 'expanded_fcl_sha256'):
            if im[key] != rm[key]:
                raise ValueError('module metadata disagreement')
        if rm['incident_type'] != 'mu2e::StageParticleCollection':
            raise ValueError('wrong collection type')
        if rm['coordinate_frame'] != 'Mu2e' or rm['position_unit'] != 'mm' or rm['momentum_unit'] != 'MeV' or rm['time_unit'] != 'ns':
            raise ValueError('wrong native state frame/units')
        products = read_columns(f['incidentManifest/products'])
        if np.any(products['input_tag'] != rm['incident_tag']):
            raise ValueError('registry tag differs from configured tag')
        validate_columns(read_columns(f['inputEventManifest/inputEvents']),
                         read_columns(f['incidentManifest/events']), products,
                         read_columns(f['incidentManifest/roots']), expected, rm['root_mode'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('check-snapshot')
    h = sub.add_parser('normalized-fcl-hash'); h.add_argument('path')
    a = sub.add_parser('audit')
    for name in ('manifest', 'artifact', 'expanded-fcl', 'expected-events'):
        a.add_argument('--' + name, required=True)
    a.add_argument('--snapshot-manifest', default=str(HERE / 'source_manifest.json'))
    args = p.parse_args()
    if args.command == 'check-snapshot': check_snapshot()
    elif args.command == 'normalized-fcl-hash': print(normalized_fcl_sha256(args.path))
    else: audit(args)


if __name__ == '__main__':
    main()
