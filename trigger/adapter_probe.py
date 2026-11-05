"""Prepare a bounded unchanged-StrawDigi round trip inside original production events."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import replay


def prepare(config_text, source, table, process='R7Adapter'):
    if process in ('Digitize', 'TriggerReplay') or not process.isidentifier():
        raise ValueError('need a distinct adapter process')
    original = replay.production_config(config_text, 'Digitize')
    services = deepcopy(original['services'])
    services['TFileService'] = {'fileName': 'stock-digis.root'}
    dump = {'process_name': 'R7DigiDump', 'source': {'module_type': 'RootInput', 'fileNames': [str(source)]},
        'services': services, 'physics': {'analyzers': {'trkdigi': {'module_type': 'StrawDigiTableDumper',
            'StrawDigiTag': 'makeSD::Digitize'}}, 'Dump': ['trkdigi'], 'end_paths': ['Dump']}}
    inject = deepcopy(dump)
    inject['process_name'] = process
    inject['services']['TFileService']['fileName'] = 'roundtrip-digis.root'
    inject['physics'].update(producers={'tableSD': {'module_type': 'StrawDigisFromTable', 'tableFile': str(table)}},
        Inject=['tableSD'], trigger_paths=['Inject'], Dump=['trkdigi', 'Combined'])
    inject['physics']['analyzers']['trkdigi'].update(StrawDigiTag=f'tableSD::{process}', compareTo='makeSD::Digitize')
    inject['outputs'] = {'Combined': {'module_type': 'RootOutput', 'fileName': 'combined.art',
        'outputCommands': ['keep *']}}
    return dump, inject, {'stock_tag': 'makeSD::Digitize', 'injected_tag': f'tableSD::{process}',
        'config_sha256': hashlib.sha256(config_text.encode()).hexdigest(),
        'scope': 'unchanged stock-digi adapter validation only; bounded population, no fast physics or trigger closure'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dump, inject, receipt = prepare(args.config.read_text(), args.source, args.table)
    if args.output.exists():
        raise ValueError('refuse prepared adapter overwrite')
    args.output.mkdir(parents=True)
    for name, config in [('dump', dump), ('inject', inject)]:
        text = '\n'.join(f'{key}: {replay.render(value)}' for key, value in config.items()) + '\n'
        (args.output / f'{name}.fcl').write_text(text)
        receipt[f'{name}_fcl_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    (args.output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('prepared exact unchanged-digi adapter probe; no job launched')


if __name__ == '__main__':
    main()
