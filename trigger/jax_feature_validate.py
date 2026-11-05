"""Vectorized synthetic/measured-array selection checks, not an artifact admission gate.

Callers supply canonical path-keyed mappings for info_present, accepted_keys,
states, indices, sequences and contracts. States/indices must be the ACTUAL
TriggerReplay endpoints, not Digitize/stock reach. Contracts must be unchanged
compile_selection results from the exact full embedded configuration. This pure
API checks their declared binding, but cannot authenticate configuration bytes,
product IDs, exporter provenance or producer-only downstream module roles.

No file I/O or CLI is provided. Array agreement is not measured parity: this
module has no measured export/provenance admission and always reports that fact.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

import awkward as ak
import jax
import numpy as np

if __package__:
    from . import jax_kalseed as selection
else:
    import jax_kalseed as selection

PATHS = tuple(selection.PATH_FILTERS)


def _mapping(value, name):
    if not isinstance(value, Mapping) or set(value) != set(PATHS):
        raise ValueError(f'{name} must have exactly the two canonical path names')


def _vector(value, n, name, boolean=False):
    value = np.asarray(value)
    if value.shape != (n,):
        raise ValueError(f'{name} must have shape ({n},)')
    if boolean:
        if value.dtype != np.dtype(bool):
            raise ValueError(f'{name} must have bool dtype')
    elif value.dtype.kind not in 'iu' or np.any(value < 0):
        raise ValueError(f'{name} must be a nonnegative integer vector')
    return value


def _ragged(value, n, name, empty_dtype=np.int64):
    try:
        value = ak.Array(value)
        if value.ndim != 2 or len(value) != n:
            raise ValueError('expected events x variable-length values')
        counts = ak.to_numpy(ak.num(value, axis=1), allow_missing=False)
        flattened = ak.flatten(value, axis=1)
        flat = ak.to_numpy(flattened, allow_missing=False)
        if isinstance(flattened.layout, ak.contents.EmptyArray):
            flat = np.empty(0, dtype=empty_dtype)
        if flat.ndim != 1 or counts.dtype.kind not in 'iu':
            raise ValueError('expected scalar leaves and nonmissing lists')
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: malformed or missing ragged values') from exc
    return counts.astype(np.int64), flat


def _integer(flat, name, signed=False):
    if flat.dtype.kind not in ('i' if signed else 'iu'):
        raise ValueError(f'{name} must have integer dtype' + (' (signed)' if signed else ''))
    if not signed and np.any(flat < 0):
        raise ValueError(f'{name} must be nonnegative')


def _padded_features(features, track_index, n, product_present):
    if not isinstance(features, Mapping) or set(features) != set(selection.FEATURES):
        raise ValueError('features must contain exactly the jax_kalseed fields')
    counts, tracks = _ragged(track_index, n, 'track_index')
    _integer(tracks, 'track_index')
    if np.any((~product_present) & (counts != 0)):
        raise ValueError('absent collection must have zero tracks')
    local = ak.to_numpy(ak.flatten(ak.local_index(ak.Array(track_index), axis=1)), allow_missing=False)
    if not np.array_equal(tracks, local):
        raise ValueError('track_index must be original contiguous 0..count-1 indices')
    width = int(np.max(counts, initial=0))
    rows = np.repeat(np.arange(n), counts)
    cols = local.astype(np.int64)
    padded = {}
    for name in selection.FEATURES:
        empty_dtype = (np.uint8 if name in selection.BOOL_FEATURES else
                       np.float64 if name in selection.FLOAT_FEATURES else np.int64)
        feature_counts, flat = _ragged(features[name], n, name, empty_dtype)
        if not np.array_equal(feature_counts, counts):
            raise ValueError(f'{name} is not aligned with track_index')
        if name in selection.BOOL_FEATURES:
            if (flat.dtype not in (np.dtype(bool), np.dtype('uint8')) or
                              np.any((flat != 0) & (flat != 1))):
                raise ValueError(f'{name} must be bool or strict uint8 0/1')
            dtype = bool
            if name == 'valid_track' and not np.all(flat == 1):
                raise ValueError('exported tracks must all have valid_track=1; padding is derived')
        elif name in selection.FLOAT_FEATURES:
            if flat.dtype != np.dtype('float64'):
                raise ValueError(f'{name} must have float64 dtype')
            dtype = np.float64
        else:
            _integer(flat, name, signed=name == 'fit_particle')
            dtype = flat.dtype if flat.size else np.int64
        padded[name] = np.zeros((n, width), dtype=dtype)
        padded[name][rows, cols] = flat
    selection.validate_features(padded)
    return padded, counts, rows, cols


def _contract(contract, path):
    if not isinstance(contract, Mapping):
        raise ValueError('contract must come from compile_selection')
    source = contract.get('source', {})
    if (contract.get('schema') != 'jax-kalseed-selection-prototype-v1' or
            contract.get('path') != path or contract.get('filter') != selection.PATH_FILTERS[path] or
            contract.get('collections') != ['TrigKSFDe'] or contract.get('process') != 'Digitize' or
            source.get('sha256') != selection.SOURCE_SHA256 or source.get('url') != selection.SOURCE_URL or
            source.get('release') != 'v13_41_01' or contract.get('noInfo') is not False or
            contract.get('momSurface') is not None or
            re.fullmatch('[0-9a-f]{64}', str(contract.get('config_sha256'))) is None):
        raise ValueError('contract source/path/full-config SHA binding is invalid')
    if (type(contract.get('minNTrks')) is not int or not 0 <= contract['minNTrks'] < 2**32 or
            type(contract.get('noFilter')) is not bool or not contract.get('cuts')):
        raise ValueError('invalid compiled selection contract')


def _reach(state, index, sequence, path):
    if not isinstance(sequence, (list, tuple)) or not all(isinstance(s, str) for s in sequence):
        raise ValueError('sequence must be an ordered module-name sequence')
    label = selection.PATH_FILTERS[path]
    if len(set(sequence)) != len(sequence) or sequence.count(label) != 1 or sequence.count('TrigKSFDe') != 1:
        raise ValueError('sequence must contain unique modules and the exact shared fit/filter')
    k = sequence.index(label) + 1
    if sequence.index('TrigKSFDe') >= k - 1:
        raise ValueError('shared KalSeed fit must precede the selection filter')
    if np.any((state != 1) & (state != 2)):
        raise ValueError('Ready/Exception endpoints are unsupported; do not collapse to Fail')
    if np.any((index < 1) | (index > len(sequence))) or np.any((state == 1) & (index != len(sequence))):
        raise ValueError('invalid TriggerReplay endpoint: expected one-based executed-worker count')
    if np.any((state == 2) & (index > k)):
        raise ValueError('unsupported downstream failure after KalSeedFilter')
    reached = (state == 1) | ((state == 2) & (index >= k))
    cpp_pass = (state == 1) | (index > k)
    return reached, cpp_pass, k


def validate_arrays(keys, features, track_index, info_present, accepted_keys,
                    states, indices, sequences, contracts, product_present):
    """Compare eager AND JIT reached-track identities, filter returns and path bits.

    Ragged features are the exact kernel fields (including exporter uint8 bools).
    Extra exporter fields, raw_status, t0 and provenance belong to a future I/O
    gate; no guessed SeedOK bit decoding is performed. Empty/missing collections
    are distinct. Nonfinite feature doubles retain released C++ semantics.
    Zero mismatches establishes only agreement of these supplied arrays; suffix
    producer roles and same-product accepted-key identity require external proof.
    """
    keys = np.asarray(keys)
    if keys.ndim != 2 or keys.shape[1] != 3 or keys.dtype.kind not in 'iu' or np.any(keys <= 0):
        raise ValueError('keys must be positive integer (N,3) event identities')
    n = len(keys)
    if len(np.unique(keys, axis=0)) != n:
        raise ValueError('event keys must be unique')
    for name, mapping in (('info_present', info_present), ('accepted_keys', accepted_keys),
                          ('states', states), ('indices', indices), ('sequences', sequences),
                          ('contracts', contracts)):
        _mapping(mapping, name)
    product_present = _vector(product_present, n, 'product_present', boolean=True)
    padded, counts, rows, cols = _padded_features(features, track_index, n, product_present)
    result = {}
    full_sha = None
    all_equal = True
    for path in PATHS:
        contract = contracts[path]
        _contract(contract, path)
        if full_sha is not None and full_sha != contract['config_sha256']:
            raise ValueError('both contracts must bind the same full configuration SHA')
        full_sha = contract['config_sha256']
        state = _vector(states[path], n, f'{path} state')
        index = _vector(indices[path], n, f'{path} index')
        reached, cpp_pass, k = _reach(state, index, sequences[path], path)
        info = _vector(info_present[path], n, f'{path} info_present', boolean=True)
        if not np.array_equal(info, reached):
            raise ValueError(f'{path}: info presence differs from actual TriggerReplay filter reach')
        if np.any(reached & ~product_present):
            raise ValueError(f'{path}: reached selection requires a present collection, including empty')
        selected_counts, selected = _ragged(accepted_keys[path], n, f'{path} accepted_keys')
        _integer(selected, 'accepted_keys')
        if np.any((~info) & (selected_counts != 0)):
            raise ValueError('absent TriggerInfo must have zero accepted keys')
        if np.any(selected >= np.repeat(counts, selected_counts)):
            raise ValueError('accepted key outside original track collection range')
        offsets = np.cumsum(counts) - counts
        flat_indices = np.repeat(offsets, selected_counts) + selected.astype(np.int64)
        multiplicity = np.bincount(flat_indices, minlength=int(np.sum(counts)))
        if np.any(multiplicity > 1):
            raise ValueError('duplicate accepted key within event')
        expected = np.zeros(padded['valid_track'].shape, dtype=bool)
        expected[rows, cols] = multiplicity.astype(bool)
        reached_features = {name: value[reached] for name, value in padded.items()}
        eager_tracks, eager_events = selection.evaluate(reached_features, contract)
        jit_tracks, jit_events = jax.jit(lambda f: selection.evaluate(f, contract))(reached_features)
        et, ee = np.asarray(eager_tracks), np.asarray(eager_events)
        jt, je = np.asarray(jit_tracks), np.asarray(jit_events)
        target = expected[reached]
        cpp = cpp_pass[reached]
        final = np.zeros(n, dtype=bool)
        final[reached] = ee
        jit_final = np.zeros(n, dtype=bool)
        jit_final[reached] = je
        actual = state == 1
        row = {
            'reached': int(np.count_nonzero(reached)), 'unreached': int(np.count_nonzero(~reached)),
            'empty_reached': int(np.count_nonzero(reached & (counts == 0))),
            'tracks_compared': int(np.sum(counts[reached])),
            'track_false_positives': int(np.count_nonzero(et & ~target)),
            'track_false_negatives': int(np.count_nonzero(~et & target)),
            'track_identity_mismatches': int(np.count_nonzero(np.any(et != target, axis=1))),
            'track_count_mismatches': int(np.count_nonzero(np.sum(et, axis=1) != selected_counts[reached])),
            'filter_false_positives': int(np.count_nonzero(ee & ~cpp)),
            'filter_false_negatives': int(np.count_nonzero(~ee & cpp)),
            'full_path_false_positives': int(np.count_nonzero(final & ~actual)),
            'full_path_false_negatives': int(np.count_nonzero(~final & actual)),
            'full_path_mismatches': int(np.count_nonzero(final != actual)),
            'jit_track_mismatches': int(np.count_nonzero(jt != target)),
            'jit_filter_mismatches': int(np.count_nonzero(je != cpp)),
            'jit_full_path_mismatches': int(np.count_nonzero(jit_final != actual)),
            'eager_jit_track_mismatches': int(np.count_nonzero(et != jt)),
            'eager_jit_event_mismatches': int(np.count_nonzero(ee != je)),
            'actual_pass': int(np.count_nonzero(actual)),
            'actual_fail': int(np.count_nonzero(state == 2)),
            'filter_index_one_based': k,
        }
        equal = not any(value for name, value in row.items() if
                        name.endswith('mismatches') or name.endswith('positives') or name.endswith('negatives'))
        row['arrays_equal'] = equal
        all_equal &= equal
        result[path] = row
    return {
        'events': n, 'paths': result, 'arrays_equal': bool(all_equal),
        'jax_runtime': {'version': jax.__version__, 'backend': jax.default_backend(),
                        'x64_enabled': bool(jax.config.x64_enabled)},
        'config_sha256': full_sha, 'measured_metadata_verified': False, 'measured_parity_established': False,
        'comparison_process': 'TriggerReplay', 'track_identity': 'original collection index; same-product Ptr identity checked by exporter, not authenticated here',
        'composition': 'actual preceding TriggerReplay reach AND JAX filter return; no stock reach copied',
        'downstream_assumption': 'only MergeTriggerInfo producer(s) follow selection; roles require external config admission',
        'scope': 'pure array agreement only; no artifact/source-order/baseline equality admission',
    }
