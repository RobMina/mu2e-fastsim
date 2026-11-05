"""Separate full-stock SGS response-control preparation; primary-only campaign gates stay unchanged."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

import event_keys
import stage_flow
import hybrid_history
import injection_campaign
import integrity
import plugin_runtime
import replay
import response_presence
import response_table
import split_source_tables
import step_handoff

KIND = 'full-stock-imported-SGS-response-control-v1'
SCOPE = ('full original exported SGS through existing importer and separately seeded stock digitization; '
         'C-S combines RNG/reset/importer/state/runtime differences, not electronics-only; F-C not pure secondaries')
IMPORTER = '98eb5ef4620e8561e802e93012f604176ffe71b46fdc2f275c9aef52a0be39e5'
DUMPER = '070247a18f58919cef3d184125420ece10f36ae15f0da5653aa4276f317dc0a3'


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def render(value):
    return '\n'.join(f'{key}: {replay.render(item)}' for key, item in value.items()) + '\n'


def keys(path):
    with uproot.open(path) as root:
        tree = root['Events']
        if tree.classname != 'TTree':
            raise ValueError('response control requires original art Events TTree')
        result = event_keys.read(tree)
    return result[np.lexsort((result[:, 2], result[:, 1], result[:, 0]))]


def same_files(a, b, label):
    hybrid_history._same_files(a, b, label)


def intended(config, source, table, seed):
    result, receipt = step_handoff.prepare(config, [str(source)], str(table), seed)
    receipt.update(admission_kind=KIND, scope=SCOPE, primary_only_admission_used=False,
                   particle_state_lossless=False, electronics_only=False)
    text = render(result)
    receipt['fcl_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    return result, receipt, text


def compare_fast_configuration(fast, control):
    """Only source/table paths differ; producer construction, services and schedules remain identical."""
    a, b = deepcopy(fast), deepcopy(control)
    for value in (a, b):
        value['source']['fileNames'] = ['BOUND-PARENT.art']
        value['physics']['producers']['R7FastSteps']['trkTableFile'] = 'BOUND-TABLE.root'
    if injection_campaign.semantic(a) != injection_campaign.semantic(b):
        raise ValueError('control changes F producer construction, electronics/services, labels, or schedule')


def inspect_plugins(fast_plan, fast_completed):
    runtime_dir = Path(fast_plan['parents'][0]['injection']) / 'runtime'
    directory = runtime_dir / 'plugins'
    receipt = load(runtime_dir / 'runtime.json')
    plugins = sorted(directory.glob('libtrkfastsim_*_module.so'))
    actual = integrity.files(plugins)
    expected = {Path(row['path']).name: (row['bytes'], row['sha256']) for row in receipt['plugins']}
    observed = {Path(row['path']).name: (row['bytes'], row['sha256']) for row in actual}
    if observed != expected:
        raise ValueError('frozen F plugin bundle changed')
    for name, digest in [('StrawGasStepsFromTable', IMPORTER), ('StrawGasStepTableDumper', DUMPER)]:
        if observed.get(f'libtrkfastsim_{name}_module.so', (None, None))[1] != digest:
            raise ValueError('response control requires the actual completed F importer/dumper bytes')
    runner = runtime_dir / 'offline_plugins.sh'
    if integrity.digest(runner)['sha256'] != receipt['adapted_runner_sha256']:
        raise ValueError('completed F runtime runner changed')
    evidence = {row['sha256'] for row in fast_completed['injection_evidence']}
    if any(row['sha256'] not in evidence for row in actual + integrity.files([runner, runtime_dir / 'runtime.json'])):
        raise ValueError('runtime plugin/runner/receipt bytes not bound to completed F')
    return {'plugins': actual, 'runner': integrity.digest(runner), 'receipt': integrity.digest(runtime_dir / 'runtime.json')}


def create(fast_plan_path, tables, table_receipt, output, fast_prepared, fast_art, fast_comparison):
    output = Path(output).resolve()
    tables = [Path(path).resolve() for path in tables]
    if output.exists() or output.is_symlink():
        raise ValueError('refuse response-control preparation reuse')
    fast_plan = load(fast_plan_path)
    completed = injection_campaign.completed(fast_plan)
    if not completed['all_planned_parents_verified'] or len(fast_plan['parents']) != 10 or fast_plan['events'] != 28044:
        raise ValueError('control requires the completed ten-parent development F reference')
    sources = [Path(row['source']['path']).resolve() for row in fast_plan['parents']]
    original_receipt = load(table_receipt)
    for table in tables:
        actual = integrity.digest(table)
        matching = [row for row in original_receipt['inputs'] if row['sha256'] == actual['sha256'] and row['bytes'] == actual['bytes']]
        if len(matching) != 1:
            raise ValueError('original source-table receipt does not bind exact table bytes')
        same_files(matching, [actual], 'original stock table')
    if (original_receipt.get('source_birth_identities_verified') is not True
            or original_receipt.get('events') != fast_plan['events']):
        raise ValueError('original source-table receipt lacks complete development history')
    for source in sources:
        actual = integrity.digest(source)
        matching = [row for row in original_receipt['inputs'] if row['sha256'] == actual['sha256'] and row['bytes'] == actual['bytes']]
        same_files(matching, [actual], 'original stock context')
    bound = inspect_plugins(fast_plan, completed)
    denominator = np.concatenate([keys(path) for path in sources])
    if len(denominator) != fast_plan['events'] or len(np.unique(denominator, axis=0)) != len(denominator):
        raise ValueError('original source denominator changed')
    flow = stage_flow.analyze(fast_art, Path(fast_plan['config']['path']), fast_prepared, fast_comparison)
    if flow['events'] != fast_plan['events'] or len(flow['paths']) != 52:
        raise ValueError('saved full-history decision admission differs')
    with uproot.open(fast_art) as root:
        if not np.array_equal(event_keys.read(root['Events']), denominator):
            raise ValueError('saved decision-art identities/order differ from original parent history')
    table_admission = response_table.audit(tables, denominator)
    output.mkdir(parents=True)
    save(output / 'table-admission.json', table_admission)
    counts = [row['events'] for row in fast_plan['parents']]
    history = load(Path(fast_prepared) / 'manifest.json')['production_history_receipt']
    export_sources = [Path(row['path']) for row in history['parent_census']['original_parent_files']]
    export_presence = response_presence.admit(export_sources, counts, tag='compressDetStepMCs::Primary')
    presence = response_presence.admit(sources, counts, tag='compressDigiMCs::Digitize')
    for ledger in (presence, export_presence):
        if not ledger['all_present'] or not np.array_equal(np.asarray(ledger['event_keys_numeric']), denominator):
            raise ValueError('source SGS presence ledger differs from ordered original denominator')
    save(output / 'source-presence.json', presence)
    save(output / 'export-source-presence.json', export_presence)
    split = split_source_tables.split(sources, tables, output / 'tables')
    rows = []
    for ordinal, (source, fast_row) in enumerate(zip(sources, fast_plan['parents']), 1):
        table = output / 'tables' / f'parent_{ordinal}.root'
        admission = response_table.audit([table], keys(source))
        prepared = output / 'parents' / f'parent_{ordinal}'
        prepared.mkdir(parents=True)
        save(prepared / 'table-admission.json', admission)
        result, manifest, text = intended(Path(fast_plan['config']['path']).read_text(), source, table, fast_plan['digitizer_seed'])
        compare_fast_configuration(replay.parse_expanded((Path(fast_row['prepared']) / 'job.fcl').read_text()), result)
        (prepared / 'job.fcl').write_text(text)
        save(prepared / 'manifest.json', manifest)
        rows.append({'ordinal': ordinal, 'events': fast_row['events'], 'source': integrity.digest(source),
                     'table': integrity.digest(table), 'prepared': str(prepared),
                     'table_admission': integrity.digest(prepared / 'table-admission.json'),
                     'fast_prepared': fast_row['prepared']})
    evidence = integrity.files([fast_plan_path, table_receipt, output / 'table-admission.json', output / 'source-presence.json',
                               output / 'tables/split.json', output / 'export-source-presence.json', fast_art, fast_comparison,
                               Path(fast_prepared) / 'manifest.json', Path(fast_prepared) / 'replay.fcl'])
    evidence += completed['injection_evidence']
    plan = {'admission_kind': KIND, 'scope': SCOPE, 'events': fast_plan['events'], 'parents': rows,
            'runtime': fast_plan['runtime'], 'runtime_binding': bound, 'config': fast_plan['config'],
            'digitizer_seed': fast_plan['digitizer_seed'], 'digitizer_seed_policy': fast_plan['digitizer_seed_policy'],
            'source_presence': presence, 'original_tables': integrity.files(tables),
            'evidence': evidence, 'fast_plan': integrity.digest(fast_plan_path),
            'primary_only_admission_used': False, 'electronics_only': False, 'particle_state_lossless': False,
            'source_simparticle_product_ids_authenticated': False}
    save(output / 'plan.json', plan)
    return plan


def preflight(plan_path, ordinal):
    plan = load(plan_path)
    if (plan.get('admission_kind') != KIND or plan.get('primary_only_admission_used') is not False
            or plan.get('electronics_only') is not False or plan.get('particle_state_lossless') is not False
            or plan.get('source_simparticle_product_ids_authenticated') is not False):
        raise ValueError('need separate full-stock control admission, without strong provenance promotion')
    if type(ordinal) is not int or not 1 <= ordinal <= len(plan['parents']):
        raise ValueError('invalid original parent ordinal')
    if ([row['ordinal'] for row in plan['parents']] != list(range(1, 11))
            or sum(row['events'] for row in plan['parents']) != plan['events']):
        raise ValueError('control parent order or denominator changed')
    same_files([plan['config']], integrity.files(plan['config']['path']), 'control config')
    same_files(plan['original_tables'], integrity.files([row['path'] for row in plan['original_tables']]), 'original tables')
    same_files(plan['evidence'], integrity.files([row['path'] for row in plan['evidence']]), 'control evidence')
    same_files([plan['fast_plan']], integrity.files(plan['fast_plan']['path']), 'completed F plan')
    fast_plan = load(plan['fast_plan']['path'])
    for field in ('runtime', 'digitizer_seed', 'digitizer_seed_policy', 'config', 'events'):
        if plan[field] != fast_plan[field]:
            raise ValueError('control runtime or reset policy differs from hash-bound F plan')
    for control_row, fast_row in zip(plan['parents'], fast_plan['parents']):
        if ((control_row['ordinal'], control_row['events']) != (fast_row['ordinal'], fast_row['events'])
                or Path(control_row['fast_prepared']).resolve() != Path(fast_row['prepared']).resolve()):
            raise ValueError('control parent mapping differs from completed F')
        same_files([control_row['source']], [fast_row['source']], 'completed F parent mapping')
    expected_runtime = Path(fast_plan['parents'][0]['injection']) / 'runtime'
    bound = plan['runtime_binding']
    if Path(bound['runner']['path']).resolve() != (expected_runtime / 'offline_plugins.sh').resolve():
        raise ValueError('control runner not from completed F')
    for entries in [bound['plugins'], [bound['runner']], [bound['receipt']]]:
        same_files(entries, integrity.files([row['path'] for row in entries]), 'completed F runtime')
    row = plan['parents'][ordinal - 1]
    for name in ('source', 'table', 'table_admission'):
        same_files([row[name]], integrity.files(row[name]['path']), f'control {name}')
    source, table, prepared = Path(row['source']['path']), Path(row['table']['path']), Path(row['prepared'])
    config = Path(plan['config']['path']).read_text()
    stored_manifest = load(prepared / 'manifest.json')
    input_paths = stored_manifest['input_files']
    if len(input_paths) != 1:
        raise ValueError('control preparation must use exactly one original parent')
    same_files(integrity.files(input_paths), [row['source']], 'prepared control source alias')
    same_files(integrity.files(stored_manifest['table_file']), [row['table']], 'prepared table alias')
    result, manifest, text = intended(config, input_paths[0], stored_manifest['table_file'], plan['digitizer_seed'])
    if stored_manifest != manifest or (prepared / 'job.fcl').read_text() != text:
        raise ValueError('control source/table/seed or configuration differs')
    compare_fast_configuration(replay.parse_expanded((Path(row['fast_prepared']) / 'job.fcl').read_text()), result)
    actual = response_table.audit([table], keys(source))
    if actual != load(row['table_admission']['path']):
        raise ValueError('full-stock table admission changed')
    return plan, row


def dump_prepare(config, candidate, output):
    if output.exists():
        raise ValueError('refuse response re-export preparation reuse')
    result, receipt = step_handoff.prepare(config, [str(candidate)])
    result['physics']['analyzers']['dump'].update(strawGasSteps='R7FastSteps::R7TrackerInject',
        genParticles='compressDigiMCs::Digitize', simParticles='R7FastSteps::R7TrackerInject', surfaceSteps='R7FastSteps::R7TrackerInject',
        vdSteps='R7FastSteps:virtualdetector:R7TrackerInject')
    text = render(result)
    output.mkdir(parents=True)
    (output / 'job.fcl').write_text(text)
    save(output / 'manifest.json', dict(receipt, fcl_sha256=hashlib.sha256(text.encode()).hexdigest(),
        scope='read-only imported SGS numerical roundtrip; no digitization or trigger rerun'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create_cmd = commands.add_parser('create')
    create_cmd.add_argument('--fast-plan', type=Path, required=True)
    create_cmd.add_argument('--tables', type=Path, nargs='+', required=True)
    create_cmd.add_argument('--table-receipt', type=Path, required=True)
    create_cmd.add_argument('--output', type=Path, required=True)
    for name in ('fast-prepared', 'fast-art', 'fast-comparison'):
        create_cmd.add_argument('--' + name, type=Path, required=True)
    check = commands.add_parser('preflight')
    check.add_argument('--plan', type=Path, required=True)
    check.add_argument('--ordinal', type=int, required=True)
    check.add_argument('--output', type=Path, required=True)
    dump = commands.add_parser('prepare-dump')
    for name in ('config', 'candidate', 'output'):
        dump.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'create':
        result = create(args.fast_plan, args.tables, args.table_receipt, args.output,
                        args.fast_prepared, args.fast_art, args.fast_comparison)
        print(f"admitted full-stock control: {result['events']} events; not electronics-only")
    elif args.command == 'preflight':
        if args.output.exists():
            raise ValueError('refuse control execution receipt overwrite')
        plan, row = preflight(args.plan, args.ordinal)
        save(args.output, {'plan': integrity.digest(args.plan), 'parent': row, 'runtime': plan['runtime'],
                           'scope': SCOPE, 'admission_kind': KIND})
    else:
        dump_prepare(args.config.read_text(), args.candidate, args.output)


if __name__ == '__main__':
    main()
