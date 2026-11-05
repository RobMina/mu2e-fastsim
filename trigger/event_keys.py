"""Decode event identities from the observed fixed art EventAuxiliary basket layout only."""
from __future__ import annotations

import numpy as np

from serialized import entry_bytes


HEADERS = ((0, bytes.fromhex('40000019000a40000013000a10')),
           (29, bytes.fromhex('4000001a000c40000010000a40000006000a')),
           (59, bytes.fromhex('4000000a000a')))


def decode(values):
    import awkward as ak

    if not ak.all(ak.num(values, axis=1) == 78):
        raise ValueError('unsupported EventAuxiliary serialized size')
    raw = ak.to_numpy(values)
    for offset, expected in HEADERS:
        if not np.all(raw[:, offset:offset + len(expected)] == np.frombuffer(expected, dtype=np.uint8)):
            raise ValueError('unsupported EventAuxiliary nested framing/version')
    payload = np.ascontiguousarray(raw[:, 47:59])
    keys = payload.view('>u4').reshape(-1, 3).astype(np.uint64)
    if np.any(keys == 0) or np.unique(keys, axis=0).shape[0] != len(keys):
        raise ValueError('zero or duplicate ROOT event identities')
    return keys


def read(tree):
    branch = tree['EventAuxiliary']
    if branch.typename != 'art::EventAuxiliary':
        raise ValueError('not an art EventAuxiliary branch')
    keys = decode(entry_bytes(branch))
    if len(keys) != tree.num_entries:
        raise ValueError('EventAuxiliary entries differ from Events tree count')
    return keys
