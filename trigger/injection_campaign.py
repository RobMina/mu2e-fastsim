"""Plan parent-isolated digitization and bind completed injections before one full-history trigger replay."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import shlex

import integrity
import replay
import step_handoff


def load(path):
    return json.loads(Path(path).read_text())


def canonical(rows):
    return [dict(row, path=str(Path(row['path']).resolve())) for row in rows]


def semantic(value):
    if isinstance(value, replay.Atom):
        try:
            return Decimal(value.text)
        except InvalidOperation:
            return value.text
    if isinstance(value, dict):
        return {key: semantic(item) for key, item in value.items()}
    if isinstance(value, list):
        return [semantic(item) for item in value]
    return value


def create(sources, counts, transports, injections, preparations, config, seed, runtime):
    if (not sources or len(sources) != len(counts) or len(sources) != len(transports)
            or len(sources) != len(injections) or len(sources) != len(preparations)
            or len(set(map(str, sources))) != len(sources) or any(count <= 0 for count in counts)):
        raise ValueError('need corresponding unique source/transport/injection/preparation rows and positive counts')
    step_handoff.prepare(config.read_text(), [str(sources[0])], 'schema-only.root', seed)
    prepared_config, _ = step_handoff.prepare(config.read_text(), [str(sources[0])], 'schema-only.root', seed)
    return {'events': sum(counts), 'config': integrity.digest(config), 'digitizer_seed': seed,
        'electronics_services': {name: replay.plain(prepared_config['services'].get(name))
            for name in ('SeedService', 'RandomNumberGenerator')},
        'truth_input_policy': 'inherited unqualified compressDetStepMCs; no same-label producer added, original context retained',
        'digitizer_seed_policy': 'reset_each_parent; correlated streams, not original production draws or independent replicas',
        'runtime': runtime, 'parents': [dict(ordinal=i + 1, events=count, source=integrity.digest(source),
            transport=str(transport), injection=str(injection), prepared=str(prepared))
            for i, (source, count, transport, injection, prepared) in enumerate(
                zip(sources, counts, transports, injections, preparations))],
        'scope': 'primary-only tracker fast steps plus stock digitization and stock calo/timing; no generated exposure'}


def check_plan(plan):
    rows = plan['parents']
    if ([row['ordinal'] for row in rows] != list(range(1, len(rows) + 1))
            or sum(row['events'] for row in rows) != plan['events']
            or len({row['source']['path'] for row in rows}) != len(rows)
            or len({row['injection'] for row in rows}) != len(rows)):
        raise ValueError('campaign parent order, count, or boundaries differ')
    if integrity.digest(plan['config']['path']) != plan['config']:
        raise ValueError('campaign production configuration changed')
    if not plan['digitizer_seed_policy'].startswith('reset_each_parent;'):
        raise ValueError('unsupported campaign electronics seed policy')
    return rows


def prepare_parent(plan, ordinal, env_output):
    rows = check_plan(plan)
    if not 2 <= ordinal <= len(rows) or env_output.exists():
        raise ValueError('only new remaining-parent tasks can be prepared; first parent is reused')
    row = rows[ordinal - 1]
    source, transport = Path(row['source']['path']), Path(row['transport'])
    if integrity.digest(source) != row['source']:
        raise ValueError('campaign stock parent bytes changed')
    admission = load(transport / 'step-admission.json')
    if admission['events'] != row['events'] or not admission['source_birth_keys_complete']:
        raise ValueError('transport does not cover the planned parent')
    output, prepared = Path(row['injection']), Path(row['prepared'])
    if output.exists() or prepared.exists():
        raise ValueError('refuse injection output or preparation reuse')
    result, receipt = step_handoff.prepare(Path(plan['config']['path']).read_text(),
        [str(source)], str(transport / 'transport/fasttrk_field.root'), plan['digitizer_seed'])
    text = '\n'.join(f'{key}: {replay.render(value)}' for key, value in result.items()) + '\n'
    receipt['fcl_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    prepared.mkdir(parents=True)
    (prepared / 'job.fcl').write_text(text)
    (prepared / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    env = dict(TRIGGER_SOURCE=str(source), TRIGGER_TRANSPORT=str(transport), TRIGGER_OUTPUT=str(output),
        TRIGGER_PREPARED=str(prepared), TRIGGER_CONFIG=plan['config']['path'], OFFLINE_EVENTS=str(row['events']))
    env.update(plan['runtime'])
    env_output.write_text(''.join(f'export {key}={shlex.quote(str(value))}\n' for key, value in env.items()))
    return row


def completed(plan, ordinals=None):
    rows = check_plan(plan)
    if ordinals is not None:
        rows = [row for row in rows if row['ordinal'] in ordinals]
        if len(rows) != len(ordinals):
            raise ValueError('requested parent not present in campaign')
    references, candidates, receipts, evidence = [], [], [], []
    plugin_sets = []
    for row in rows:
        source = Path(row['source']['path'])
        directory, prepared, transport = map(Path, (row['injection'], row['prepared'], row['transport']))
        candidate = directory / 'unfiltered_1.art'
        outputs = sorted(path.name for path in directory.glob('unfiltered_*.art'))
        if outputs != ['unfiltered_1.art'] or (directory / 'art.exit').read_text().strip() != '0':
            raise ValueError('injection incomplete or parent boundaries split unexpectedly')
        names = ['admission.json', 'event-count.json', 'execution-binding.json', 'step-admission.json',
                 'art.exit', 'runtime/runtime.json', 'runtime/offline_plugins.sh', 'inject.expanded.fcl', 'inject.log']
        files = [directory / name for name in names] + [prepared / 'manifest.json', prepared / 'job.fcl']
        admission, binding, step, runtime, manifest = map(load, [directory / 'admission.json',
            directory / 'execution-binding.json', directory / 'step-admission.json',
            directory / 'runtime/runtime.json', prepared / 'manifest.json'])
        if (not admission['same_event_handoff_metadata_passed'] or admission['events'] != row['events']
                or canonical(admission['reference_files']) != integrity.files([source])
                or canonical(admission['candidate_files']) != integrity.files([candidate])
                or admission['config_sha256'] != plan['config']['sha256']
                or admission['straw_digi_tag'] != 'R7FastSD::R7TrackerInject' or admission['calo_digi_tag'] is not None):
            raise ValueError('injection admission does not bind the planned stock/fast bytes')
        table = transport / 'transport/fasttrk_field.root'
        expected = step_handoff.preflight(prepared, Path(plan['config']['path']).read_text(), table, source, step)
        binding = dict(binding, source=canonical([binding['source']])[0], table=canonical([binding['table']])[0])
        if (binding != expected or manifest['digitizer_seed'] != plan['digitizer_seed']
                or step['events'] != row['events'] or not step['source_birth_keys_complete']):
            raise ValueError('injection execution, source coverage, or reset seed differs')
        plugins = [dict(path=Path(item['path']).name, bytes=item['bytes'], sha256=item['sha256'])
                   for item in runtime['plugins']]
        plugin_sets.append(plugins)
        for item in plugins:
            file = directory / 'runtime/plugins' / item['path']
            actual = integrity.digest(file)
            if actual['sha256'] != item['sha256'] or actual['bytes'] != item['bytes']:
                raise ValueError('frozen injection plugin bytes changed')
            files.append(file)
        actual_runner = integrity.digest(directory / 'runtime/offline_plugins.sh')
        if actual_runner['sha256'] != runtime['adapted_runner_sha256']:
            raise ValueError('injection runtime runner changed')
        log = (directory / 'inject.log').read_text()
        if ('Art has completed and will exit with status 0.' not in log
                or f"image={plan['runtime']['OFFLINE_IMAGE']}" not in log
                or f"MUSE_WORK_DIR={plan['runtime']['TRIGGER_AREA']} " not in log):
            raise ValueError('injection log lacks successful pinned production runtime')
        expanded = replay.parse_expanded((directory / 'inject.expanded.fcl').read_text())
        intended = replay.parse_expanded((prepared / 'job.fcl').read_text())
        for service in ('SeedService', 'RandomNumberGenerator', 'GeometryService', 'DbService'):
            if semantic(expanded['services'][service]) != semantic(intended['services'][service]):
                raise ValueError(f'executed injection service {service} differs from intended configuration')
        for key in ('process_name', 'source', 'physics', 'outputs'):
            if semantic(expanded[key]) != semantic(intended[key]):
                raise ValueError(f'executed injection {key} differs from intended configuration')
        count = load(directory / 'event-count.json')
        if not count['equal'] or count['events'] != row['events']:
            raise ValueError('injection finished entry count differs')
        step_inputs = canonical(step['inputs'])
        if integrity.files([item['path'] for item in step_inputs]) != step_inputs:
            raise ValueError('audited transport states, handoff, or fast table changed')
        evidence.extend(integrity.files(files))
        evidence.extend(step_inputs)
        receipts.append({'ordinal': row['ordinal'], 'events': row['events'], 'source': integrity.digest(source),
                         'candidate': integrity.digest(candidate), 'execution': binding,
                         'zero_step_events': step['zero_step_events']})
        references.append(str(source))
        candidates.append(str(candidate))
    if any(plugins != plugin_sets[0] for plugins in plugin_sets[1:]):
        raise ValueError('injection parents use different adapter plugin sets')
    return {'events': sum(row['events'] for row in rows), 'all_planned_parents_verified': ordinals is None,
        'reference': references, 'candidate': candidates,
        'parents': receipts, 'injection_evidence': evidence, 'digitizer_seed': plan['digitizer_seed'],
        'digitizer_seed_policy': plan['digitizer_seed_policy'], 'runtime': plan['runtime'],
        'scope': plan['scope']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create_cmd = sub.add_parser('create')
    for name in ('sources', 'counts', 'config', 'runtime', 'first-transport', 'transport-root',
                 'first-injection', 'injection-root', 'first-prepared', 'prepared-root', 'output'):
        create_cmd.add_argument('--' + name, type=Path, required=True)
    create_cmd.add_argument('--seed', type=int, required=True)
    prepare_cmd = sub.add_parser('prepare-parent')
    prepare_cmd.add_argument('--plan', type=Path, required=True)
    prepare_cmd.add_argument('--ordinal', type=int, required=True)
    prepare_cmd.add_argument('--env-output', type=Path, required=True)
    check_cmd = sub.add_parser('check-completed')
    check_cmd.add_argument('--plan', type=Path, required=True)
    check_cmd.add_argument('--ordinals', type=int, nargs='+')
    check_cmd.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'check-completed':
        if args.output.exists():
            raise ValueError('refuse campaign verification receipt overwrite')
        report = completed(load(args.plan), args.ordinals)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f"verified {report['events']} injection events; all planned parents={report['all_planned_parents_verified']}")
    elif args.command == 'create':
        if args.output.exists():
            raise ValueError('refuse campaign plan overwrite')
        sources = list(map(Path, args.sources.read_text().splitlines()))
        counts = list(map(int, args.counts.read_text().splitlines()))
        paths = lambda first, root: [first] + [root / f'parent_{i}' for i in range(2, len(sources) + 1)]
        report = create(sources, counts, paths(args.first_transport, args.transport_root),
            paths(args.first_injection, args.injection_root), paths(args.first_prepared, args.prepared_root),
            args.config, args.seed, load(args.runtime))
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    else:
        prepare_parent(load(args.plan), args.ordinal, args.env_output)


if __name__ == '__main__':
    main()
