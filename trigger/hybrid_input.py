"""Audit a same-event fast-digi handoff against unchanged stock products without detector-field decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

import event_keys
import integrity
import path_state
import replay
import schedule_registry
import serialized


STOCK = [('mu2e::StrawDigis', 'makeSD'), ('mu2e::StrawDigiADCWaveforms', 'makeSD'),
         ('mu2e::CaloDigis', 'CaloDigiMaker'), ('mu2e::EventWindowMarker', 'EWMProducer'),
         ('mu2e::ProtonBunchTime', 'EWMProducer'), ('mu2e::ProtonBunchTimeMC', 'EWMProducer')]
TAG = re.compile(r'[A-Za-z_][A-Za-z_0-9]*:[A-Za-z_0-9]*:[A-Za-z_][A-Za-z_0-9]*\Z')


def product(tree, friendly_type, tag):
    if not TAG.fullmatch(tag):
        raise ValueError('product tag must be explicit label:instance:process')
    name = friendly_type + '_' + tag.replace(':', '_')
    parent = tree[name + '.']
    present = parent[name + '.present'].array(library='np')
    if len(present) != tree.num_entries or not np.all(present):
        raise ValueError(f'missing product entries: {name}')
    branch = parent[name + '.obj']
    values = serialized.entry_bytes(branch)
    if len(values) != tree.num_entries or np.any(ak.to_numpy(ak.num(values, axis=1)) == 0):
        raise ValueError(f'empty serialized payload or incorrect event count: {name}')
    return values, branch.typename


def audit(reference, candidate, config_text, expected, straw_tag=None, calo_tag=None, original_process='Digitize'):
    reference = [Path(path) for path in reference]
    candidate = [Path(path) for path in candidate]
    if type(expected) is not int or expected <= 0 or not reference or len(reference) != len(candidate):
        raise ValueError('need positive event count and corresponding original-parent-boundary file lists')
    if not straw_tag and not calo_tag:
        raise ValueError('declare at least one substituted detector')
    for tag in (straw_tag, calo_tag):
        if tag is not None and (not TAG.fullmatch(tag) or tag in
                (f'makeSD::{original_process}', f'CaloDigiMaker::{original_process}')):
            raise ValueError('replacement tag must be explicit and distinct from stock')
    config = replay.production_config(config_text, original_process)
    paths = replay.paths(config)
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    all_keys, rows = [], []
    for source, target in zip(reference, candidate):
        with uproot.open(source, custom_classes=classes) as root_a, uproot.open(target, custom_classes=classes) as root_b:
            a, b = root_a['Events'], root_b['Events']
            if a.classname != 'TTree' or b.classname != 'TTree':
                raise ValueError('handoff requires ordinary art Events TTrees')
            keys_a, keys_b = event_keys.read(a), event_keys.read(b)
            order_a = np.lexsort((keys_a[:, 2], keys_a[:, 1], keys_a[:, 0]))
            order_b = np.lexsort((keys_b[:, 2], keys_b[:, 1], keys_b[:, 0]))
            if not np.array_equal(keys_a[order_a], keys_b[order_b]):
                raise ValueError('handoff changes parent-boundary event identities or drops zero-output events')
            all_keys.append(keys_a[order_a])
            checks = {}
            for friendly, label in STOCK:
                tag = f'{label}::{original_process}'
                stock_name = f'{friendly}_{label}__{original_process}.'
                shadows = [name for name in b.keys(recursive=False)
                           if name.startswith(f'{friendly}_{label}__') and name != stock_name]
                if shadows:
                    raise ValueError(f'candidate shadows an unqualified original stock product: {friendly}/{label}')
                raw_a, type_a = product(a, friendly, tag)
                raw_b, type_b = product(b, friendly, tag)
                if type_a != type_b or not np.all(serialized.serialized_equal(raw_a[order_a], raw_b[order_b])):
                    raise ValueError(f'original stock product changed through injection: {friendly}/{tag}')
                checks[f'{friendly}/{tag}'] = {'events': len(keys_a), 'serialized_equal': True, 'typename': type_a}
            injected = {}
            for tag, friendly in [(straw_tag, 'mu2e::StrawDigis'), (straw_tag, 'mu2e::StrawDigiADCWaveforms'),
                                  (calo_tag, 'mu2e::CaloDigis')]:
                if tag is not None:
                    _, typename = product(b, friendly, tag)
                    stock_label = 'CaloDigiMaker' if friendly == 'mu2e::CaloDigis' else 'makeSD'
                    if typename != checks[f'{friendly}/{stock_label}::{original_process}']['typename']:
                        raise ValueError(f'injected product type differs from stock collection: {friendly}/{tag}')
                    injected[f'{friendly}/{tag}'] = {'present_all_events': True, 'typename': typename}
            path_state.check_schema(root_a)
            path_state.check_schema(root_b)
            words_a, id_a = path_state.read_status(a, original_process, len(paths))
            words_b, id_b = path_state.read_status(b, original_process, len(paths))
            if id_a != id_b or not np.array_equal(words_a[order_a], words_b[order_b]):
                raise ValueError('original reference TriggerResults changed through injection')
            registry = schedule_registry.verify(root_b, id_b, paths)
            rows.append({'events': len(keys_a), 'stock_products': checks, 'injected_products': injected,
                         'original_statuses_preserved': True, 'schedule_registry': registry})
    keys = np.concatenate(all_keys)
    if len(keys) != expected or np.unique(keys, axis=0).shape[0] != expected:
        raise ValueError('handoff count differs from complete source or duplicates events across files')
    return {'events': expected, 'files': rows, 'reference_files': integrity.files(reference),
            'candidate_files': integrity.files(candidate), 'config_sha256': hashlib.sha256(config_text.encode()).hexdigest(),
            'straw_digi_tag': straw_tag, 'calo_digi_tag': calo_tag, 'same_event_handoff_metadata_passed': True,
            'source_traversal': 'corresponding parent files in supplied order; default numeric event traversal within each',
            'scope': 'handoff metadata and retained stock bytes only; no fast physics or trigger validation',
            'prerequisites_not_certified': ['owner receipt must establish complete source population and matching geometry/conditions/runtime',
                'injected digi field semantics and ADC alignment need owner/runtime contract validation',
                'fast arm must run every scheduled path with its own filter-invocation history']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, nargs='+', required=True)
    parser.add_argument('--candidate', type=Path, nargs='+', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    parser.add_argument('--straw-digi-tag')
    parser.add_argument('--calo-digi-tag')
    parser.add_argument('--original-process', default='Digitize')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse handoff report overwrite')
    report = audit(args.reference, args.candidate, args.config.read_text(), args.expected_events,
                   args.straw_digi_tag, args.calo_digi_tag, args.original_process)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"events={report['events']} same-event stock-byte preservation and injected presence verified; not physics validation")


if __name__ == '__main__':
    main()
