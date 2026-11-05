"""Inspect/compare serialized value-only digi/timing bytes via ROOT baskets, without physics deserialization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


TOKENS = ('StrawDigis_makeSD_', 'StrawDigiADCWaveforms_makeSD_', 'CaloDigis_CaloDigiMaker_',
          'EventWindowMarker_EWMProducer_', 'ProtonBunchTime_EWMProducer_', 'ProtonBunchTimeMC_EWMProducer_')


def entry_bytes(branch, limit=None):
    parts = []
    covered = 0
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('serialized entry limit must be positive')
    for basket_num in range(branch.num_baskets):
        start, stop = branch.basket_entry_start_stop(basket_num)
        if start != covered or stop < start:
            raise ValueError('noncontiguous basket entry coverage')
        if limit is not None and start >= limit:
            break
        basket = branch.basket(basket_num)
        if basket.num_entries != stop - start:
            raise ValueError('basket entry count differs from branch coverage')
        covered = stop
        offsets = basket.byte_offsets
        if offsets is None:
            if len(basket.data) % basket.num_entries:
                raise ValueError('basket has neither offsets nor fixed-size entries')
            offsets = np.arange(basket.num_entries + 1) * (len(basket.data) // basket.num_entries)
        offsets = np.asarray(offsets, dtype=np.int64)
        if offsets.size != basket.num_entries + 1 or offsets[0] != 0 or offsets[-1] != len(basket.data) or np.any(np.diff(offsets) < 0):
            raise ValueError('malformed basket entry boundaries')
        values = ak.unflatten(np.asarray(basket.data, dtype=np.uint8), np.diff(offsets))
        parts.append(values if limit is None else values[:max(0, limit - start)])
    if not parts:
        raise ValueError('no serialized entries')
    result = ak.concatenate(parts)
    expected = branch.num_entries if limit is None else min(limit, branch.num_entries)
    if len(result) != expected or (limit is None and covered != branch.num_entries):
        raise ValueError('serialized basket coverage differs from branch entry count')
    return result


def inventory(path, limit=1):
    with uproot.open(path) as root:
        tree = root['Events']
        branches = []
        for name, branch in tree.items(recursive=True):
            if name == 'EventAuxiliary' or (name.endswith('.obj') and any(token in name for token in TOKENS)):
                values = entry_bytes(branch, limit)
                first = ak.to_numpy(values[0])
                branches.append({'name': name, 'typename': branch.typename, 'entries': branch.num_entries,
                                 'first_entry_hex': first.tobytes().hex(), 'first_entry_bytes': len(first)})
        classes = {}
        for name, versions in root.file.streamers.items():
            if any(token in name for token in ('StrawDigi', 'CaloDigi', 'EventWindowMarker', 'ProtonBunchTime', 'EventAuxiliary', 'EventID', 'SubRunID', 'RunID', 'Hash<2>')):
                classes[name] = {str(version): [{'name': item.name, 'typename': item.typename} for item in streamer.elements]
                                 for version, streamer in versions.items()}
        return {'events': tree.num_entries, 'branches': branches, 'streamers': classes,
                'scope': 'raw serialized value-only product probe, no field decoding or equality assertion'}


def serialized_equal(a, b):
    if len(a) != len(b):
        raise ValueError('unequal entry counts in serialized comparison')
    lengths_a, lengths_b = ak.to_numpy(ak.num(a, axis=1)), ak.to_numpy(ak.num(b, axis=1))
    result = lengths_a == lengths_b
    selected = np.flatnonzero(result)
    result[selected] = ak.to_numpy(ak.all(a[selected] == b[selected], axis=1))
    return result


def product_data(paths):
    import event_keys

    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    keys, products, types = [], {token: [] for token in TOKENS}, {}
    for path in paths:
        with uproot.open(path) as root:
            tree = root['Events']
            keys.append(event_keys.read(tree))
            for token in TOKENS:
                parents = [branch for name, branch in tree.items(recursive=False) if token in name]
                if len(parents) != 1:
                    raise ValueError(f'need exactly one product matching {token}')
                parent = parents[0]
                if not np.all(parent[parent.name[:-1] + '.present'].array(library='np')):
                    raise ValueError(f'absent product: {token}')
                branch = parent[parent.name[:-1] + '.obj']
                if token in types and branch.typename != types[token]:
                    raise ValueError('different serialized product types across input files')
                types[token] = branch.typename
                values = entry_bytes(branch)
                if len(values) != tree.num_entries:
                    raise ValueError('product entries differ from Events tree')
                products[token].append(values)
    joined_keys = np.concatenate(keys)
    if np.unique(joined_keys, axis=0).shape[0] != len(joined_keys):
        raise ValueError('duplicate event identities across product files')
    return joined_keys, {token: ak.concatenate(values) for token, values in products.items()}, types


def compare_products(reference, candidate, expected_overlap):
    import event_keys

    if isinstance(candidate, (list, tuple)):
        return compare_product_sets(product_data(reference), product_data(candidate), expected_overlap)
    if expected_overlap <= 0:
        raise ValueError('expected overlap must be positive')
    with uproot.open(reference) as root_a, uproot.open(candidate) as root_b:
        a, b = root_a['Events'], root_b['Events']
        keys_a, keys_b = event_keys.read(a), event_keys.read(b)
        dtype = np.dtype([('run', '<u8'), ('subrun', '<u8'), ('event', '<u8')])
        ids_a, ids_b = np.ascontiguousarray(keys_a).view(dtype).ravel(), np.ascontiguousarray(keys_b).view(dtype).ravel()
        _, index_a, index_b = np.intersect1d(ids_a, ids_b, return_indices=True)
        if len(index_a) != expected_overlap:
            raise ValueError(f'event-key overlap {len(index_a)} != expected {expected_overlap}')
        rows = {}
        for token in TOKENS:
            parents = [branch for name, branch in a.items(recursive=False) if token in name]
            if len(parents) != 1:
                raise ValueError(f'need exactly one original product matching {token}')
            parent_a = parents[0]
            if parent_a.name not in b:
                raise ValueError(f'missing candidate product {parent_a.name}')
            parent_b = b[parent_a.name]
            present_name = parent_a.name[:-1] + '.present'
            if not np.all(parent_a[present_name].array(library='np')[index_a]) or not np.all(parent_b[present_name].array(library='np')[index_b]):
                raise ValueError(f'absent product in overlap: {token}')
            name = parent_a.name[:-1] + '.obj'
            branch_a, branch_b = parent_a[name], parent_b[name]
            if branch_a.typename != branch_b.typename:
                raise ValueError('different serialized product types')
            raw_a, raw_b = entry_bytes(branch_a)[index_a], entry_bytes(branch_b)[index_b]
            equal = serialized_equal(raw_a, raw_b)
            examples = keys_a[index_a[~equal]][:10].tolist()
            rows[token] = {'typename': branch_a.typename, 'events': len(equal), 'mismatches': int(np.count_nonzero(~equal)),
                           'mismatch_examples': examples,
                           'reference_sha256': hashlib.sha256(ak.to_numpy(ak.flatten(raw_a)).tobytes()).hexdigest(),
                           'candidate_sha256': hashlib.sha256(ak.to_numpy(ak.flatten(raw_b)).tobytes()).hexdigest()}
        return {'reference_events': len(keys_a), 'candidate_events': len(keys_b), 'overlap_events': len(index_a),
                'products': rows, 'equal': all(row['mismatches'] == 0 for row in rows.values()),
                'comparison': 'exact raw serialized value-only product bytes, including framing and all persisted fields',
                'root_event_ids_verified': True,
                'caveats': ['byte inequality may reflect serialization differences; equality is stronger than field equality',
                            'no detector physics fields are interpreted or corrected; value-only product types only',
                            'overlap equality alone does not validate omitted source events or full trigger history']}


def compare_product_sets(reference, candidate, expected_overlap):
    keys_a, products_a, types_a = reference
    keys_b, products_b, types_b = candidate
    if types_a != types_b or expected_overlap <= 0:
        raise ValueError('different serialized types or nonpositive expected overlap')
    dtype = np.dtype([('run', '<u8'), ('subrun', '<u8'), ('event', '<u8')])
    _, index_a, index_b = np.intersect1d(np.ascontiguousarray(keys_a).view(dtype).ravel(),
        np.ascontiguousarray(keys_b).view(dtype).ravel(), return_indices=True)
    if len(index_a) != expected_overlap:
        raise ValueError('event-key overlap differs from expected')
    rows = {}
    for token in TOKENS:
        raw_a, raw_b = products_a[token][index_a], products_b[token][index_b]
        equal = serialized_equal(raw_a, raw_b)
        rows[token] = {'typename': types_a[token], 'events': len(equal), 'mismatches': int(np.count_nonzero(~equal)),
            'mismatch_examples': keys_a[index_a[~equal]][:10].tolist(),
            'reference_sha256': hashlib.sha256(ak.to_numpy(ak.flatten(raw_a)).tobytes()).hexdigest(),
            'candidate_sha256': hashlib.sha256(ak.to_numpy(ak.flatten(raw_b)).tobytes()).hexdigest()}
    return {'reference_events': len(keys_a), 'candidate_events': len(keys_b), 'overlap_events': len(index_a),
            'products': rows, 'equal': all(row['mismatches'] == 0 for row in rows.values()),
            'root_event_ids_verified': True, 'comparison': 'exact raw serialized value-only bytes by event identity',
            'caveats': ['byte inequality can reflect framing rather than physics; no payload fields interpreted',
                        'selected overlap alone does not certify complete source/trigger history']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('art', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--expected-overlap', type=int)
    args = parser.parse_args()
    if args.candidate:
        if args.expected_overlap is None:
            raise ValueError('comparison needs --expected-overlap')
        report = compare_products(args.art, args.candidate, args.expected_overlap)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f"overlap={report['overlap_events']} equal={report['equal']}")
        for name, row in report['products'].items():
            print(f"{name} mismatches={row['mismatches']}")
        raise SystemExit(0 if report['equal'] else 1)
    report = inventory(args.art)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"events={report['events']} probed branches={len(report['branches'])}")


if __name__ == '__main__':
    main()
