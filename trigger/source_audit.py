"""Inventory authentic preselection parents and random-state prerequisites without launching simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import replay


def audit(text, process='Digitize'):
    config = replay.production_config(text, process)
    source = replay.plain(config['source'])
    files = source.get('fileNames')
    if source.get('module_type') != 'RootInput' or not isinstance(files, list) or not files:
        raise ValueError('need an embedded RootInput source with explicit parent files')
    if not all(isinstance(url, str) and url for url in files) or len(set(files)) != len(files):
        raise ValueError('empty, duplicate or nonliteral source files')
    parents = []
    for ordinal, url in enumerate(files):
        name = Path(urlsplit(url).path).name
        if not re.fullmatch(r'[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+\.art', name):
            raise ValueError('unsupported art parent filename')
        parents.append({'configured_ordinal': ordinal, 'url': url, 'did': f'mu2e:{name}'})
    physics = config['physics']
    modules = replay.plain({**physics.get('producers', {}), **physics.get('filters', {})})
    sequence = replay.plain(physics['DigitizePath'])
    if not isinstance(sequence, list) or not all(label in modules for label in sequence):
        raise ValueError('undefined digitizer module')
    services = replay.plain(config['services'])
    if 'SeedService' not in services or 'RandomNumberGenerator' not in services:
        raise ValueError('missing original random-engine services')
    return {'config_sha256': hashlib.sha256(text.encode()).hexdigest(), 'process': process,
        'parents': parents, 'configured_parent_order_not_certified_runtime_order': True,
        'source': source, 'seed_service': services['SeedService'],
        'random_number_generator': services['RandomNumberGenerator'],
        'digitize_modules': [{'label': label, 'module_type': modules[label]['module_type']} for label in sequence],
        'all_configured_module_labels': sorted({*modules, *physics.get('analyzers', {}), *config.get('outputs', {})}),
        'stock_output_selection': replay.plain(config['outputs']['Output']['SelectEvents']),
        'can_recover_from_selected_digis': False, 'full_regeneration_prepared': False,
        'required_checks': [
            'actual complete source order and event IDs including all rejected events',
            'matching released binaries, conditions, geometry, external data and random engine implementation',
            'unchanged module set/construction order and SeedService engine registration for autoIncrement',
            'original random state from job start or authentic checkpoint; a selected-file offset is insufficient',
            'original stored-event straw/calo/timing payload equality before attributing exact trigger closure',
            'unfiltered digi output preserves all source events and original TriggerResults for second-pass stock replay',
            'stock and each fast arm evolve their own invocation history on the complete preselection population'],
        'scope': 'read-only prerequisites, not authorization, generated FCL, or production identity certification'}


def catalog_parents(report, client):
    result = []
    for parent in report['parents']:
        record = client.get_file(did=parent['did'], with_metadata=True)
        if record is None:
            raise ValueError(f"parent missing from catalog: {parent['did']}")
        if record.get('retired') or record.get('name') != parent['did'].split(':', 1)[1]:
            raise ValueError('retired or inconsistent parent catalog record')
        if type(record.get('size')) is not int or record['size'] <= 0:
            raise ValueError('missing or invalid parent size')
        result.append(record)
    counts = [record.get('metadata', {}).get('rse.nevent') for record in result]
    generated = [record.get('metadata', {}).get('gen.count') for record in result]
    return {'files': result, 'total_bytes': sum(record['size'] for record in result),
            'catalog_stored_parent_events': sum(counts) if all(type(n) is int and n >= 0 for n in counts) else None,
            'catalog_generated_parent_count': sum(generated) if all(type(n) is int and n >= 0 for n in generated) else None,
            'scope': 'catalog metadata counts only; no finished-entry, location, availability, payload or transfer validation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--process', default='Digitize')
    parser.add_argument('--catalog', action='store_true', help='read-only metacat get_file calls after ops setup')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.config.read_text(), args.process)
    if args.catalog:
        from metacat.webapi import MetaCatClient
        report['catalog'] = catalog_parents(report, MetaCatClient())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"parents={len(report['parents'])} digitize_modules={len(report['digitize_modules'])}; no simulation launched")
    if args.catalog:
        print(f"catalog_total_bytes={report['catalog']['total_bytes']}")


if __name__ == '__main__':
    main()
