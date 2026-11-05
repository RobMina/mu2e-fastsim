"""Read the measured r7SelectionFeatures exporter, not a parity/admission gate.

Only the schema-v1 fundamental/std::vector TTree contract is supported. No truth,
stock-track substitute, cut emulation, path reach inference or JAX execution is
performed. Schema strings are declarations, not loaded-binary attestation.
Raw status bits cannot be authenticated from this file: the exporter does not
include the released TrkFitFlag name/bit definitions. In particular this reader
never guesses a SeedOK bit or derives supported_fit from an unbound checkout.
"""
from __future__ import annotations

import re

import awkward as ak
import numpy as np
import uproot

if __package__:
    from . import jax_kalseed as selection
else:
    import jax_kalseed as selection

LABEL = 'r7SelectionFeatures'
PATHS = tuple(selection.PATH_FILTERS)
SLOTS = dict(zip(PATHS, ('D0200', 'nominal')))
SCHEMA = {
    'offline_release': 'v13_41_01',
    'offline_commit': 'b132ab6567ac021851353e05b7a1feb45bfb7c14',
    'filter_sha256': '51b6dbfe6f5dcd115afdfb8b9ada7aa187b8610f717cdbbc1527e7b573fabd39',
    'required_backing': '/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ba',
    'precision': 'chi2_dof=float(chisquared)/int(nDOF), then double; fit_consistency=float then double; segment kinematics=double',
    'kalSeeds': 'TrigKSFDe::TriggerReplay',
    'selectionInfo_D0200': 'TprTrkDe80m70pD0200KSFilter::TriggerReplay',
    'selectionInfo_nominal': 'TprTrkDe80m70pKSFilter::TriggerReplay',
}
SCALARS = {'run': 'uint32', 'subrun': 'uint32', 'event': 'uint32',
           'source_ordinal': 'uint64', 'product_present': 'bool',
           'product_id': 'string', 'info_present_D0200': 'bool',
           'info_present_nominal': 'bool'}
VECTORS = {
    'track_index': 'uint64', 'raw_status': 'uint32',
    **{name: 'uint8' for name in selection.BOOL_FEATURES},
    'fit_particle': 'int32', 'active_hits': 'uint32',
    **{name: 'float64' for name in selection.FLOAT_FEATURES},
    't0_segment_index': 'int64', 't0': 'float64',
    'accepted_track_keys_D0200': 'uint64', 'accepted_track_keys_nominal': 'uint64',
}
# Persistent ROOT spellings, not inferred from data (including all-empty data).
TYPES = {
    'uint8': {'uint8_t', 'unsignedchar', 'UChar_t'},
    'uint32': {'uint32_t', 'unsignedint', 'UInt_t'},
    'uint64': {'uint64_t', 'unsignedlong', 'unsignedlonglong', 'ULong64_t'},
    'int32': {'int32_t', 'int', 'Int_t'},
    'int64': {'int64_t', 'long', 'longlong', 'Long64_t'},
    'float64': {'double', 'Double_t'}, 'bool': {'bool', 'Bool_t'},
    'string': {'string', 'basic_string<char>', 'char*'},
}


def _typename(value):
    return re.sub(r'\s+', '', value.replace('std::', ''))


def _tree(directory, name):
    # Events can autosave more often than schema; choose each highest cycle,
    # never fall back to an older valid cycle. Equal cycle numbers are not proof
    # of coherence and are not required by TFileService.
    cycles = [int(key.rsplit(';', 1)[1]) for key in directory.keys(recursive=False, cycle=True)
              if key.rsplit(';', 1)[0] == name]
    if not cycles:
        raise ValueError(f'missing {LABEL}/{name} TTree')
    cycle = max(cycles)
    tree = directory[f'{name};{cycle}']
    if tree.classname != 'TTree':
        raise ValueError(f'{name} must be a TTree')
    return tree, cycle


def _branches(tree, scalars, vectors):
    expected = set(scalars) | set(vectors)
    if set(tree.keys(recursive=False)) != expected:
        raise ValueError(f'{tree.name}: exact branch set required')
    for name, dtype in {**scalars, **vectors}.items():
        typename = _typename(tree[name].typename)
        if name in vectors:
            match = re.fullmatch(r'vector<([^<>]+)>', typename)
            if match is None:
                raise ValueError(f'{name}: persistent std::vector<{dtype}> required')
            typename = match[1]
        if typename not in TYPES[dtype]:
            raise ValueError(f'{name}: persistent {dtype} dtype required')
    return tree.arrays(library='ak', how=dict)


def _vector(value, n, name, dtype):
    try:
        value = ak.Array(value)
        # Reject options even if no item is missing: allow_missing=False alone
        # does not reject a list option with an all-valid mask.
        if value.ndim != 1 or len(value) != n or value.layout.is_option:
            raise ValueError('expected nonoptional scalar rows')
        if dtype == 'string':
            if value.layout.parameter('__array__') != 'string':
                raise ValueError('expected nonoptional strings')
            return value
        array = ak.to_numpy(value, allow_missing=False)
        if array.shape != (n,) or array.dtype != np.dtype(dtype):
            raise ValueError(f'expected {dtype}')
        return array
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: malformed scalar or wrong {dtype} dtype') from exc


def _ragged(value, n, name, dtype):
    """Strict variant of jax_feature_validate._ragged; never invent leaf types."""
    try:
        value = ak.Array(value)
        if (value.ndim != 2 or len(value) != n or
                not isinstance(value.layout, (ak.contents.ListOffsetArray,
                                              ak.contents.ListArray,
                                              ak.contents.RegularArray)) or
                not isinstance(value.layout.content, ak.contents.NumpyArray)):
            raise ValueError('expected nonoptional lists with typed scalar leaves')
        counts = ak.to_numpy(ak.num(value, axis=1), allow_missing=False)
        flat = ak.to_numpy(ak.flatten(value, axis=1), allow_missing=False)
        if flat.ndim != 1 or flat.dtype != np.dtype(dtype):
            raise ValueError(f'expected persistent {dtype} leaves, also when empty')
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: malformed or missing ragged values / wrong {dtype} dtype') from exc
    return counts.astype(np.int64), flat


def _expected(value):
    value = np.asarray(value)
    if (value.ndim != 2 or value.shape[1] != 3 or value.dtype.kind not in 'iu' or
            np.any(value <= 0) or np.any(value > np.iinfo(np.uint32).max)):
        raise ValueError('expected_keys must be positive uint32-range integer (N,3) identities')
    if len(np.unique(value, axis=0)) != len(value):
        raise ValueError('expected_keys must be unique')
    return value


def read(path, *, expected_keys):
    """Return ragged measured features after strict table validation.

    expected_keys must include every source event in order, including missing and
    zero-track responses; a typed shape-(0,3) array represents a zero-event file.
    All arrays retain their exporter dtypes (byte booleans included). Metadata
    explicitly separates validated declarations/structure from authentication.
    """
    expected = _expected(expected_keys)
    n = len(expected)
    with uproot.open(path) as root:
        try:
            directory = root[LABEL]
        except KeyError as exc:
            raise ValueError(f'missing analyzer label {LABEL}') from exc
        events, event_cycle = _tree(directory, 'events')
        schema, schema_cycle = _tree(directory, 'schema')
        if schema.num_entries != 1:
            raise ValueError('schema must have exactly one row')
        declaration = _branches(schema, {'schema_version': 'uint32',
                                         **{name: 'string' for name in SCHEMA}}, {})
        if _vector(declaration['schema_version'], 1, 'schema_version', 'uint32')[0] != 1:
            raise ValueError('schema_version must be 1')
        for name, wanted in SCHEMA.items():
            actual = _vector(declaration[name], 1, name, 'string')
            if not bool(ak.all(actual == wanted)):
                raise ValueError(f'schema declaration differs: {name}')
        if events.num_entries != n:
            raise ValueError('event count differs from expected_keys (including zero responses)')
        data = _branches(events, SCALARS, VECTORS)

    scalars = {name: _vector(data[name], n, name, dtype) for name, dtype in SCALARS.items()}
    keys = np.column_stack((scalars['run'], scalars['subrun'], scalars['event']))
    if np.any(keys == 0) or len(np.unique(keys, axis=0)) != n:
        raise ValueError('event keys must be positive and unique')
    if not np.array_equal(keys, expected):
        raise ValueError('event IDs/order differ from expected_keys')
    if not np.array_equal(scalars['source_ordinal'], np.arange(1, n + 1, dtype=np.uint64)):
        raise ValueError('source_ordinal must be contiguous 1..N')
    present = scalars['product_present']
    nonempty_id = ak.to_numpy(ak.num(scalars['product_id'], axis=1) != 0)
    if not np.array_equal(present, nonempty_id):
        raise ValueError('product_id must be nonempty iff product_present')

    ragged = {name: _ragged(data[name], n, name, dtype) for name, dtype in VECTORS.items()}
    counts, tracks = ragged['track_index']
    local = ak.to_numpy(ak.flatten(ak.local_index(data['track_index'], axis=1)), allow_missing=False)
    if not np.array_equal(tracks, local):
        raise ValueError('track_index must be contiguous original collection indices')
    if np.any((~present) & (counts != 0)):
        raise ValueError('absent collection must have zero tracks')
    for name in VECTORS:
        if not name.startswith('accepted_track_keys_') and not np.array_equal(ragged[name][0], counts):
            raise ValueError(f'{name}: per-track alignment differs from track_index')
    for name in selection.BOOL_FEATURES:
        flat = ragged[name][1]
        if np.any((flat != 0) & (flat != 1)):
            raise ValueError(f'{name}: byte boolean must be strict 0/1')
    if np.any(ragged['valid_track'][1] != 1):
        raise ValueError('valid_track must be 1 for every exported track')

    seed = ragged['status_has_SeedOK'][1].astype(bool)
    segment = ragged['has_t0_segment'][1].astype(bool)
    supported = ragged['supported_fit'][1].astype(bool)
    index = ragged['t0_segment_index'][1]
    t0 = ragged['t0'][1]
    if np.any(segment & ~seed):
        raise ValueError('t0 segment requires SeedOK status gate')
    if np.any((segment & (index < 0)) | (~segment & (index != -1))):
        raise ValueError('t0_segment_index inconsistent with has_t0_segment')
    if np.any(~segment & ~np.isnan(t0)):
        raise ValueError('unavailable t0 must be NaN')
    available = seed & segment & supported
    for name in selection.FLOAT_FEATURES:
        if np.any(~available & ~np.isnan(ragged[name][1])):
            raise ValueError(f'{name}: unavailable derived features must be NaN')
    # Available arithmetic may legitimately produce NaN/Inf. No finite veto,
    # pz/pt approximation, or recomputation of released C++ expressions is used.
    info, accepted = {}, {}
    offsets = np.cumsum(counts) - counts
    for canonical, slot in SLOTS.items():
        presence = scalars[f'info_present_{slot}']
        name = f'accepted_track_keys_{slot}'
        selected_counts, selected = ragged[name]
        if np.any(presence & ~present):
            raise ValueError(f'{name}: present TriggerInfo requires present shared collection')
        if np.any((~presence) & (selected_counts != 0)):
            raise ValueError(f'{name}: absent TriggerInfo must have zero accepted keys')
        if np.any(selected >= np.repeat(counts, selected_counts)):
            raise ValueError(f'{name}: accepted key outside collection range')
        locations = np.repeat(offsets, selected_counts) + selected.astype(np.int64)
        if np.any(np.bincount(locations, minlength=int(np.sum(counts))) > 1):
            raise ValueError(f'{name}: duplicate accepted key within event')
        info[canonical], accepted[canonical] = presence, data[name]

    return {
        'keys': keys, 'features': {name: data[name] for name in selection.FEATURES},
        'track_index': data['track_index'], 'product_present': present,
        'info_present': info, 'accepted_keys': accepted,
        'metadata': {
            'schema_version': 1, **SCHEMA, 'analyzer_label': LABEL,
            'highest_cycles': {'events': event_cycle, 'schema': schema_cycle},
            'events': n, 'schema_declarations_verified': True,
            'raw_status_authenticated': False, 'supported_fit_authenticated': False,
            'status_authentication': 'cannot authenticate SeedOK or supported_fit bits: exact released TrkFitFlag name/bit definitions are not bound by schema v1',
            'runtime_admitted': False, 'measured_parity_established': False,
            'source_ordinal': scalars['source_ordinal'], 'product_id': data['product_id'],
            'raw_status': data['raw_status'], 't0_segment_index': data['t0_segment_index'],
            't0': data['t0'],
            'scope': 'measured table structure and declared binding only; no binary/source attestation, replay reach, baseline decision equality or measured selection parity; same-product Ptr identity is asserted by exporter, not independently authenticated',
        },
    }
