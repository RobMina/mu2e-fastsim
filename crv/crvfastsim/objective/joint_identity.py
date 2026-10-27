"""Audit captured full event keys against immutable generated input censuses, without response replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import tables
from . import validation as V


def key_rows(columns):
    arrays = [np.asarray(columns[k]) for k in tables.KEY]
    if not all(a.ndim == 1 and a.dtype.kind in 'iu' and np.all(a >= 0) and
               np.all(a <= np.iinfo(np.int64).max) for a in arrays):
        raise ValueError('invalid full integer event identities')
    out = np.empty(len(arrays[0]), dtype=[(k, '<i8') for k in tables.KEY])
    for k, a in zip(tables.KEY, arrays):
        out[k] = a
    return out


def canonical_keys(rows):
    return np.unique(key_rows({k: rows[k] for k in tables.KEY}))


def audit_capture(captured, source_index, slots, inputs):
    """Require exactly one event row per slot/source/input generated identity."""
    reports = []
    if not np.array_equal(np.unique(source_index), np.arange(len(inputs))):
        raise ValueError('capture source indices differ from manifest inputs')
    for index, expected in enumerate(inputs):
        expected = canonical_keys(expected)
        selected = source_index == index
        if not np.array_equal(np.unique(slots[selected]), np.arange(4)):
            raise ValueError('capture source missing sector slots')
        for slot in range(4):
            actual = captured[selected & (slots == slot)]
            if len(actual) != len(expected) or not np.array_equal(np.sort(actual), expected):
                raise ValueError(f'capture/input event identities differ: source {index}, slot {slot}')
        reports.append(dict(source_index=index, events=len(expected),
                            canonical_event_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
    return reports


def audit(directory, out):
    root = Path(directory)
    receipts, result = {}, {}
    manifests = []
    for arm in ('nominal', 'atten', 'yield', 'joint'):
        p = root / arm / 'provenance.json'
        receipts[str(p)] = V.digest(p)
        prov = json.loads(p.read_text())
        if prov['status'] != 'complete' or prov['integrity_errors']:
            raise ValueError('incomplete or failed arm')
        manifests.append(prov['manifest'])
    if any(m != manifests[0] for m in manifests):
        raise ValueError('four-arm input manifests differ')
    manifest = manifests[0]
    V.verify(manifest)
    expected = [V.keys(row['path'], 'gen') for row in manifest['inputs']]
    for row, keys in zip(manifest['inputs'], expected):
        if len(keys) != row['events']:
            raise ValueError('generated identities differ from manifest census')
    all_expected = np.concatenate(expected)
    if len(np.unique(all_expected)) != len(all_expected):
        raise ValueError('overlapping source generated identities')
    for arm in ('nominal', 'atten', 'yield', 'joint'):
        path = root / arm / 'joint.npz'
        receipts[str(path)] = V.digest(path)
        prov = json.loads((root / arm / 'provenance.json').read_text())
        if receipts[str(path)] != prov['joint_files']['joint.npz']['sha256']:
            raise ValueError('capture receipt hash mismatch')
        with np.load(path, allow_pickle=False) as archive:
            keys = key_rows({k: archive['event_' + k] for k in tables.KEY})
            reports = audit_capture(keys, archive['event_source_index'], archive['event_slot'], expected)
        result[arm] = dict(events=sum(r['events'] for r in reports), sources=reports,
                           matches_generated_input_keys=True)
    V.verify(manifest)
    if any(V.digest(path) != sha for path, sha in receipts.items()):
        raise ValueError('saved files changed during identity audit')
    return V._report(out, dict(schema=1, passed=True, comparison='capture-to-generated-identities', arms=result,
                              files=receipts, inputs=manifest['inputs'],
                              key_encoding='sorted unique structured little-endian int64 run/subrun/event',
                              source_code={str(Path(__file__)): V.digest(__file__),
                                           str(Path(V.__file__)): V.digest(V.__file__)},
                              caution='Exact generated exposure identity only; no pulse/draw identity claim.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = audit(args.directory, args.out)
    print(json.dumps(dict(passed=report['passed'], counts={k: v['events'] for k, v in report['arms'].items()})))


if __name__ == '__main__':
    main()
