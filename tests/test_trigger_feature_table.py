"""Local ROOT/table validation tests, not art/runtime or measured parity evidence.

uproot can write jagged leaf-count TTrees, but cannot write std::vector TTree
branches (and PyROOT is not a dependency). The narrow _VectorTree facade below
only replaces that persistent branch spelling and hides writer-only count leaves.
All scalar/string branches, jagged buffers, cycles and I/O are real uproot ROOT
fixtures. Separate tests reject the unadapted leaf-count/RNTuple alternatives.
"""
from pathlib import Path
import ast
import sys

import awkward as ak
import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import feature_table as table

REAL_OPEN = uproot.open

def sample(counts=(2, 0, 1, 0)):
    counts = np.asarray(counts, np.int64)
    n, total = len(counts), int(np.sum(counts))
    keys = np.column_stack((np.ones(n, np.uint32), np.ones(n, np.uint32),
                            np.arange(1, n + 1, dtype=np.uint32)))
    events = {
        'run': keys[:, 0].copy(), 'subrun': keys[:, 1].copy(), 'event': keys[:, 2].copy(),
        'source_ordinal': np.arange(1, n + 1, dtype=np.uint64),
        'product_present': np.ones(n, bool),
        'product_id': ak.Array(np.full(n, '1:4')),
        'info_present_D0200': np.ones(n, bool),
        'info_present_nominal': np.ones(n, bool),
    }
    for name, dtype in table.VECTORS.items():
        value = 1
        if name in ('track_index', 'accepted_track_keys_D0200', 'accepted_track_keys_nominal'):
            flat = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        else:
            if name == 'fit_particle':
                value = 11
            elif name == 'active_hits':
                value = 15
            elif name == 't0_segment_index':
                value = 0
            elif name == 'raw_status':
                value = 0xFFFFFFFF  # explicitly not interpreted as a guessed bit map
            flat = np.full(total, value, dtype=dtype)
        events[name] = ak.unflatten(np.asarray(flat, dtype=dtype), counts)
    schema = {'schema_version': np.array([1], np.uint32),
              **{name: ak.Array([value]) for name, value in table.SCHEMA.items()}}
    return keys, events, schema


def _types(data):
    # Only fixed fields, never event/track iteration.
    return {name: (ak.type(value).content if isinstance(value, ak.Array) else value.dtype)
            for name, value in data.items()}


def write_root(path, events, schema, *, older=None, latest=None, latest_schema=None,
               event_name='r7SelectionFeatures/events', schema_name='r7SelectionFeatures/schema'):
    events = {name: ak.from_regular(value, axis=1) if name in table.VECTORS else value
              for name, value in events.items()}
    with uproot.recreate(path) as root:
        root.mktree(event_name, _types(events))
        if len(events['run']):
            root[event_name].extend(events)
        root.mktree(schema_name, _types(schema))
        if len(schema['schema_version']):
            root[schema_name].extend(schema)
        if older is not None:
            # The original table is the older cycle, latest may be malformed.
            root.mktree(event_name, _types(latest))
            if len(latest['run']):
                root[event_name].extend(latest)
        if latest_schema is not None:
            root.mktree(schema_name, _types(latest_schema))
            if len(latest_schema['schema_version']):
                root[schema_name].extend(latest_schema)


class _Branch:
    def __init__(self, branch, typename):
        self.typename = typename


class _VectorTree:
    def __init__(self, tree, overrides):
        self.tree, self.overrides = tree, overrides
        self.name, self.classname, self.num_entries = tree.name, tree.classname, tree.num_entries

    def keys(self, **kwargs):
        return [name for name in self.tree.keys(**kwargs)
                if not (name.startswith('n') and name[1:] in table.VECTORS)]

    def __getitem__(self, name):
        branch = self.tree[name]
        typename = branch.typename
        if name in table.VECTORS:
            typename = 'std::vector<' + typename.removesuffix('[]') + '>'
        return _Branch(branch, self.overrides.get(name, typename))

    def arrays(self, **kwargs):
        data = self.tree.arrays(**kwargs)
        return {name: data[name] for name in self.keys()}


class _Directory:
    def __init__(self, directory, overrides):
        self.directory, self.overrides = directory, overrides

    def keys(self, **kwargs):
        return self.directory.keys(**kwargs)

    def __getitem__(self, name):
        value = self.directory[name]
        return _VectorTree(value, self.overrides) if value.classname == 'TTree' else value


class _Root:
    def __init__(self, path, overrides):
        self.root, self.overrides = REAL_OPEN(path), overrides

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.root.close()

    def __getitem__(self, name):
        return _Directory(self.root[name], self.overrides)


@pytest.fixture
def reader(monkeypatch, tmp_path):
    def read(events=None, schema=None, keys=None, *, overrides=None, **write_options):
        default_keys, default_events, default_schema = sample()
        events = default_events if events is None else events
        schema = default_schema if schema is None else schema
        keys = default_keys if keys is None else keys
        path = tmp_path / 'features.root'
        write_root(path, events, schema, **write_options)
        monkeypatch.setattr(table.uproot, 'open', lambda path: _Root(path, overrides or {}))
        return table.read(path, expected_keys=keys)
    return read


def test_exact_interface_fields_dtypes_and_scope(reader):
    result = reader()
    assert set(result) == {'keys', 'features', 'track_index', 'product_present',
                           'info_present', 'accepted_keys', 'metadata'}
    assert tuple(result['features']) == table.selection.FEATURES
    assert set(result['info_present']) == set(result['accepted_keys']) == set(table.PATHS)
    assert result['keys'].dtype == np.dtype('uint32')
    for name in table.selection.FEATURES:
        assert ak.to_numpy(ak.flatten(result['features'][name])).dtype == np.dtype(table.VECTORS[name])
    metadata = result['metadata']
    assert metadata['highest_cycles'] == {'events': 1, 'schema': 1}
    assert metadata['schema_declarations_verified'] is True
    assert metadata['runtime_admitted'] is False
    assert metadata['measured_parity_established'] is False
    assert metadata['raw_status_authenticated'] is False
    assert metadata['supported_fit_authenticated'] is False
    assert 'cannot authenticate' in metadata['status_authentication']


@pytest.mark.parametrize('counts', [(0, 0), ()])
def test_all_empty_preserves_declared_leaf_types(reader, counts):
    keys, events, schema = sample(counts)
    result = reader(events, schema, keys)
    assert result['keys'].shape == (len(counts), 3)
    for name in table.selection.FEATURES:
        assert ak.to_numpy(ak.flatten(result['features'][name])).dtype == np.dtype(table.VECTORS[name])


def test_missing_collection_distinct_from_present_empty(reader):
    keys, events, schema = sample((1, 0, 0))
    events['product_present'][2] = False
    events['product_id'] = ak.Array(['1:4', '1:4', ''])
    events['info_present_D0200'][2] = events['info_present_nominal'][2] = False
    result = reader(events, schema, keys)
    np.testing.assert_array_equal(result['product_present'], [True, True, False])
    np.testing.assert_array_equal(ak.num(result['track_index']), [1, 0, 0])


@pytest.mark.parametrize('field', tuple(table.SCHEMA) + ('schema_version',))
def test_exact_schema_declarations(reader, field):
    keys, events, schema = sample()
    schema[field] = np.array([2], np.uint32) if field == 'schema_version' else ak.Array(['not exact'])
    with pytest.raises(ValueError, match='schema_version|schema declaration'):
        reader(events, schema, keys)


@pytest.mark.parametrize('rows', [0, 2])
def test_schema_one_row(reader, rows):
    keys, events, schema = sample()
    schema = {name: ak.concatenate([value, value])[:rows] for name, value in schema.items()}
    with pytest.raises(ValueError, match='exactly one row'):
        reader(events, schema, keys)


@pytest.mark.parametrize('change', ['extra_event', 'missing_event', 'extra_schema', 'missing_schema'])
def test_exact_branch_sets(reader, change):
    keys, events, schema = sample()
    if change == 'extra_event':
        events['proxy_truth'] = np.ones(len(keys))
    elif change == 'missing_event':
        del events['raw_status']
    elif change == 'extra_schema':
        schema['optional'] = np.ones(1)
    else:
        del schema['precision']
    with pytest.raises(ValueError, match='exact branch set'):
        reader(events, schema, keys)


@pytest.mark.parametrize('name,bad', [
    ('run', 'int32_t'), ('source_ordinal', 'uint32_t'),
    ('product_present', 'uint8_t'), ('product_id', 'uint32_t'),
    ('info_present_nominal', 'uint8_t'), ('track_index', 'std::vector<int64_t>'),
    ('raw_status', 'std::vector<uint64_t>'), ('valid_track', 'std::vector<bool>'),
    ('fit_particle', 'std::vector<unsigned int>'), ('active_hits', 'std::vector<int>'),
    ('pz', 'std::vector<float>'), ('t0', 'std::vector<float>'),
    ('t0_segment_index', 'std::vector<uint64_t>'),
    ('accepted_track_keys_D0200', 'uint64_t[]'), ('schema_version', 'int32_t'),
])
def test_persistent_dtype_contract_even_empty(reader, name, bad):
    keys, events, schema = sample((0, 0))
    with pytest.raises(ValueError, match='persistent'):
        reader(events, schema, keys, overrides={name: bad})


def test_float32_content_not_widened_by_typename(reader):
    keys, events, schema = sample()
    events['pz'] = ak.values_astype(events['pz'], np.float32)
    with pytest.raises(ValueError, match='pz.*float64'):
        reader(events, schema, keys, overrides={'pz': 'std::vector<double>'})


def test_highest_cycles_only_no_fallback(reader):
    keys, events, schema = sample()
    shorter = {name: values[:2] for name, values in events.items()}
    with pytest.raises(ValueError, match='event count'):
        reader(events, schema, keys, older=True, latest=shorter)
    result = reader(shorter, schema, keys, older=True, latest=events)
    assert result['metadata']['highest_cycles'] == {'events': 2, 'schema': 1}
    bad_schema = {**schema, 'schema_version': np.array([2], np.uint32)}
    with pytest.raises(ValueError, match='schema_version'):
        reader(events, schema, keys, latest_schema=bad_schema)


@pytest.mark.parametrize('mutation', ['count', 'order', 'duplicate', 'zero', 'ordinal'])
def test_exact_event_population_order_identity_and_ordinals(reader, mutation):
    keys, events, schema = sample()
    if mutation == 'count':
        events = {name: value[:3] for name, value in events.items()}
    elif mutation == 'order':
        events = {name: value[::-1] for name, value in events.items()}
    elif mutation == 'duplicate':
        events['event'][1] = events['event'][0]
    elif mutation == 'zero':
        events['run'][0] = 0
    else:
        events['source_ordinal'][1] = 3
    with pytest.raises(ValueError, match='event count|IDs/order|positive and unique|source_ordinal'):
        reader(events, schema, keys)


@pytest.mark.parametrize('keys', [np.ones((2, 3), int), np.zeros((1, 3), int),
                                  np.ones((1, 3), float), np.array([], int),
                                  np.full((1, 3), 2**32, np.uint64)])
def test_invalid_expected_keys_rejected_without_io(keys):
    with pytest.raises(ValueError, match='expected_keys'):
        table.read('/never/open.root', expected_keys=keys)


@pytest.mark.parametrize('mutation', ['empty_id', 'absent_with_id', 'absent_tracks', 'info_no_product'])
def test_product_and_info_presence_identity(reader, mutation):
    keys, events, schema = sample()
    if mutation == 'empty_id':
        events['product_id'] = ak.Array(['', '1:4', '1:4', '1:4'])
    elif mutation == 'absent_with_id':
        events['product_present'][1] = False
    else:
        index = 0 if mutation == 'absent_tracks' else 1
        events['product_present'][index] = False
        events['product_id'] = ak.Array(np.where(events['product_present'], '1:4', ''))
    with pytest.raises(ValueError, match='product_id|absent collection|present TriggerInfo'):
        reader(events, schema, keys)


@pytest.mark.parametrize('field', [*table.selection.FEATURES, 'raw_status', 't0_segment_index', 't0'])
def test_full_per_track_alignment(reader, field):
    keys, events, schema = sample()
    events[field] = ak.unflatten(np.ones(2, dtype=table.VECTORS[field]), [1, 0, 1, 0])
    with pytest.raises(ValueError, match='per-track alignment'):
        reader(events, schema, keys)


@pytest.mark.parametrize('name', table.selection.BOOL_FEATURES)
def test_byte_boolean_strict_values(reader, name):
    keys, events, schema = sample()
    events[name] = ak.unflatten(np.array([2, 1, 1], np.uint8), [2, 0, 1, 0])
    with pytest.raises(ValueError, match='strict 0/1'):
        reader(events, schema, keys)


def test_valid_track_not_padding(reader):
    keys, events, schema = sample()
    events['valid_track'] = ak.zeros_like(events['valid_track'])
    with pytest.raises(ValueError, match='valid_track must be 1'):
        reader(events, schema, keys)


def test_original_track_indices(reader):
    keys, events, schema = sample()
    events['track_index'] = events['track_index'] + np.uint64(1)
    with pytest.raises(ValueError, match='contiguous original'):
        reader(events, schema, keys)


@pytest.mark.parametrize('slot', ['D0200', 'nominal'])
@pytest.mark.parametrize('mutation', ['range', 'duplicate', 'absent'])
def test_accepted_keys_identity_and_presence(reader, slot, mutation):
    keys, events, schema = sample()
    name = f'accepted_track_keys_{slot}'
    if mutation == 'range':
        events[name] = events[name] + np.uint64(1)
    elif mutation == 'duplicate':
        events[name] = ak.unflatten(np.array([0, 0, 0], np.uint64), [2, 0, 1, 0])
    else:
        events[f'info_present_{slot}'][0] = False
    with pytest.raises(ValueError, match='outside collection|duplicate accepted|absent TriggerInfo'):
        reader(events, schema, keys)


def gated_sample():
    keys, events, schema = sample((4,))
    # No SeedOK; no segment; unsupported with segment; supported pathological.
    events['status_has_SeedOK'] = ak.Array(np.array([[0, 1, 1, 1]], np.uint8))
    events['has_t0_segment'] = ak.Array(np.array([[0, 0, 1, 1]], np.uint8))
    events['supported_fit'] = ak.Array(np.array([[1, 1, 0, 1]], np.uint8))
    events['t0_segment_index'] = ak.Array(np.array([[-1, -1, 0, 2]], np.int64))
    events['t0'] = ak.Array(np.array([[np.nan, np.nan, 10., np.nan]], np.float64))
    for name in table.selection.FLOAT_FEATURES:
        events[name] = ak.Array(np.array([[np.nan, np.nan, np.nan, np.inf]], np.float64))
    for slot in ('D0200', 'nominal'):
        events[f'accepted_track_keys_{slot}'] = ak.unflatten(np.array([3], np.uint64), [1])
    return keys, events, schema


def test_gated_nan_diagnostics_and_available_nonfinite_legal(reader):
    keys, events, schema = gated_sample()
    result = reader(events, schema, keys)
    assert np.isinf(ak.to_numpy(ak.flatten(result['features']['pz']))[-1])
    assert result['metadata']['raw_status_authenticated'] is False


@pytest.mark.parametrize('mutation', ['seed_gate', 'missing_index', 'unexpected_index', 'unexpected_t0',
                                      *table.selection.FLOAT_FEATURES])
def test_status_segment_derived_diagnostic_consistency(reader, mutation):
    keys, events, schema = gated_sample()
    if mutation == 'seed_gate':
        events['has_t0_segment'] = ak.Array(np.array([[1, 0, 1, 1]], np.uint8))
    elif mutation == 'missing_index':
        events['t0_segment_index'] = ak.Array(np.array([[-1, -1, -1, 2]], np.int64))
    elif mutation == 'unexpected_index':
        events['t0_segment_index'] = ak.Array(np.array([[0, -1, 0, 2]], np.int64))
    elif mutation == 'unexpected_t0':
        events['t0'] = ak.Array(np.ones((1, 4), np.float64))
    else:
        events[mutation] = ak.Array(np.ones((1, 4), np.float64))
    with pytest.raises(ValueError, match='requires SeedOK|inconsistent|unavailable'):
        reader(events, schema, keys)


@pytest.mark.parametrize('bad', [ak.Array([None, []]), ak.Array([[None], []]),
                                ak.Array([[], []]), ak.Array([[1.0], []])])
def test_ragged_missing_unknown_or_float32_not_silently_retyped(bad):
    with pytest.raises(ValueError, match='malformed or missing'):
        table._ragged(bad, 2, 'track_index', 'uint64')


def test_optional_all_valid_lists_and_leaves_rejected():
    good = ak.unflatten(np.array([0], np.uint64), [1, 0])
    lists = ak.mask(good, [True, True])
    leaves = ak.mask(good, good == 0)
    with pytest.raises(ValueError):
        table._ragged(lists, 2, 'track_index', 'uint64')
    with pytest.raises(ValueError):
        table._ragged(leaves, 2, 'track_index', 'uint64')


def test_unadapted_real_uproot_leaf_count_ttree_rejected(tmp_path):
    path = tmp_path / 'leafcounts.root'
    keys, events, schema = sample()
    write_root(path, events, schema)
    with pytest.raises(ValueError, match='exact branch set'):
        table.read(path, expected_keys=keys)


def test_wrong_label_and_rntuple_rejected(tmp_path):
    keys, events, schema = sample()
    path = tmp_path / 'wrong.root'
    write_root(path, events, schema, event_name='wrong/events', schema_name='wrong/schema')
    with pytest.raises(ValueError, match='missing analyzer label'):
        table.read(path, expected_keys=keys)
    with uproot.recreate(path) as root:
        root['r7SelectionFeatures/events'] = events
        root['r7SelectionFeatures/schema'] = schema
    with pytest.raises(ValueError, match='must be a TTree'):
        table.read(path, expected_keys=keys)



def test_paths_independent_and_stored_acceptance_order_preserved(reader):
    keys, events, schema = sample()
    events['info_present_D0200'][0] = False
    events['accepted_track_keys_D0200'] = ak.unflatten(np.array([0], np.uint64), [0, 0, 1, 0])
    events['accepted_track_keys_nominal'] = ak.unflatten(np.array([1, 0, 0], np.uint64), [2, 0, 1, 0])
    result = reader(events, schema, keys)
    a, b = table.PATHS
    assert not result['info_present'][a][0] and result['info_present'][b][0]
    np.testing.assert_array_equal(ak.to_numpy(ak.flatten(result['accepted_keys'][b])), [1, 0, 0])


def test_missing_tree_never_substitutes_other_label(reader):
    with pytest.raises(ValueError, match='missing r7SelectionFeatures/events'):
        reader(event_name='other/events')
    with pytest.raises(ValueError, match='missing r7SelectionFeatures/schema'):
        reader(schema_name='other/schema')


@pytest.mark.parametrize('dtype', ['uint64', 'float64'])
def test_typed_empty_ragged_accepted_but_untyped_not_repaired(dtype):
    values = ak.unflatten(np.empty(0, dtype=dtype), [0, 0])
    counts, flat = table._ragged(values, 2, 'empty', dtype)
    np.testing.assert_array_equal(counts, [0, 0])
    assert flat.dtype == np.dtype(dtype)
    with pytest.raises(ValueError):
        table._ragged(ak.Array([[], []]), 2, 'empty', dtype)


def test_scalar_optional_even_all_valid_not_accepted():
    values = ak.mask(ak.Array(np.array([True, False])), [True, True])
    with pytest.raises(ValueError, match='malformed scalar'):
        table._vector(values, 2, 'product_present', 'bool')


def test_reader_has_no_data_loops_or_to_list():
    source = Path(table.__file__).read_text()
    assert 'ak.to_list' not in source
    parsed = ast.parse(source)
    # Explicit loops only over fixed schema fields, vector fields or two paths.
    loops = [node for node in ast.walk(parsed) if isinstance(node, (ast.For, ast.While))]
    assert not any(isinstance(node, ast.While) for node in loops)
    assert len(loops) == 6
