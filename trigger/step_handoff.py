"""Prepare primary-only tracker-step digitization inside unchanged stock production events."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import integrity
import replay


def preflight(prepared, config_text, table, source, admission=None):
    receipt = json.loads((prepared / 'manifest.json').read_text())
    text = (prepared / 'job.fcl').read_text()
    if hashlib.sha256(text.encode()).hexdigest() != receipt['fcl_sha256']:
        raise ValueError('step handoff FCL changed')
    if hashlib.sha256(config_text.encode()).hexdigest() != receipt['config_sha256']:
        raise ValueError('step handoff production configuration changed')
    result, _ = prepare(config_text, receipt['input_files'], receipt['table_file'], receipt['digitizer_seed'])
    expected_text = '\n'.join(f'{key}: {replay.render(value)}' for key, value in result.items()) + '\n'
    if text != expected_text or [Path(path).resolve() for path in receipt['input_files']] != [Path(source).resolve()]:
        raise ValueError('step handoff source, seed, or producer configuration differs')
    if Path(receipt['table_file']).resolve() != Path(table).resolve():
        raise ValueError('step handoff does not use the audited fast table')
    source_bytes, table_bytes = integrity.digest(source), integrity.digest(table)
    if admission is not None:
        rows = {str(Path(row['path']).resolve()): dict(row, path=str(Path(row['path']).resolve()))
                for row in admission['inputs']}
        if rows.get(source_bytes['path']) != source_bytes or rows.get(table_bytes['path']) != table_bytes:
            raise ValueError('source or fast table bytes differ from step admission')
    return {'fcl_sha256': receipt['fcl_sha256'], 'source': source_bytes, 'table': table_bytes,
        'digitizer_seed': receipt['digitizer_seed']}


def prepare(config_text, inputs, table=None, seed=20261104):
    if not inputs or len(set(map(str, inputs))) != len(inputs):
        raise ValueError('need nonempty unique ordered input files')
    if type(seed) is not int or not 1 <= seed <= 900000000:
        raise ValueError('need explicit positive bounded digitizer seed')
    original = replay.production_config(config_text, 'Digitize')
    services = deepcopy(original['services'])
    source = {'module_type': 'RootInput', 'fileNames': list(map(str, inputs))}
    if table is None:
        services['TFileService'] = {'fileName': 'source-tables.root'}
        result = {'process_name': 'R7StepDump', 'source': source, 'services': services,
            'physics': {'analyzers': {'dump': {'module_type': 'StrawGasStepTableDumper',
                'genParticles': 'compressDetStepMCs', 'writeTrajectories': replay.Atom('false')}},
                'Dump': ['dump'], 'end_paths': ['Dump']}}
        scope = 'read-only original dts truth export; not fast transport or trigger validation'
    else:
        make_sd = deepcopy(original['physics']['producers']['makeSD'])
        if replay.plain(make_sd['module_type']) != 'StrawDigisFromStrawGasSteps':
            raise ValueError('unsupported original tracker digitizer')
        make_sd.pop('module_label', None)
        make_sd['StrawGasStepModules'] = ['R7FastSteps::R7TrackerInject']
        make_sd['EventWindowMarker'] = 'EWMProducer::Digitize'
        make_sd['ProtonBunchTimeMC'] = 'EWMProducer::Digitize'
        services['SeedService']['baseSeed'] = replay.Atom(str(seed))
        services['TFileService'] = {'fileName': 'fast-digis.root'}
        producers = {'R7FastSteps': {'module_type': 'StrawGasStepsFromTable', 'trkTableFile': str(table),
            'crvTableFile': '', 'genParticles': 'compressDetStepMCs', 'trkKeyOffset': replay.Atom('0'),
            'primaryKey': replay.Atom('1'), 'makeTrajectories': replay.Atom('false')}, 'R7FastSD': make_sd}
        result = {'process_name': 'R7TrackerInject', 'source': source, 'services': services,
            'physics': {'producers': producers,
                'analyzers': {'trkdigi': {'module_type': 'StrawDigiTableDumper', 'StrawDigiTag': 'R7FastSD::R7TrackerInject'}},
                'Inject': ['R7FastSteps', 'R7FastSD'], 'trigger_paths': ['Inject'],
                'Save': ['trkdigi', 'Combined'], 'end_paths': ['Save']},
            'outputs': {'Combined': {'module_type': 'RootOutput', 'fileName': 'unfiltered_%#.art',
                'fileProperties': {'maxInputFiles': replay.Atom('1'), 'granularity': 'InputFile'},
                'outputCommands': ['keep *']}}}
        scope = 'primary-only fast-step plus stock-digitizer diagnostic; no secondary/return or generated efficiency claim'
    receipt = {'config_sha256': hashlib.sha256(config_text.encode()).hexdigest(), 'input_files': source['fileNames'],
        'table_file': str(table) if table is not None else None, 'digitizer_seed': seed if table is not None else None,
        'stock_context_preserved_by_distinct_labels': True, 'physics_filters_added': False, 'scope': scope,
        'warning': 'step importer emits empty calo/VD under R7FastSteps only; stock calo/timing remain original products'}
    return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--table', type=Path)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, help='verify existing prepared injection and write execution binding')
    parser.add_argument('--admission', type=Path, help='bind executed bytes to independently audited step table')
    args = parser.parse_args()
    if args.preflight is not None:
        if args.output.exists() or len(args.inputs) != 1 or args.table is None:
            raise ValueError('need new receipt, one source, and audited table for injection preflight')
        if args.admission is None:
            raise ValueError('injection preflight requires a step admission receipt')
        receipt = preflight(args.preflight, args.config.read_text(), args.table, args.inputs[0], json.loads(args.admission.read_text()))
        args.output.write_text(json.dumps(receipt, indent=2) + '\n')
        return
    result, receipt = prepare(args.config.read_text(), args.inputs, args.table, args.seed)
    if args.output.exists():
        raise ValueError('refuse handoff preparation overwrite')
    args.output.mkdir(parents=True)
    text = '\n'.join(f'{key}: {replay.render(value)}' for key, value in result.items()) + '\n'
    (args.output / 'job.fcl').write_text(text)
    receipt['fcl_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    (args.output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('prepared detector-step handoff stage; no job launched')


if __name__ == '__main__':
    main()
