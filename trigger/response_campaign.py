"""Admit all full-stock control parents before one continuous all-52 trigger process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hybrid_history
import hybrid_input
import hybrid_replay
import integrity
import response_control as control
import response_validate


def prepare(plan_path, output_root, fast_prepared, output):
    if output.exists() or output.is_symlink():
        raise ValueError('refuse full-stock control trigger preparation overwrite')
    plan = control.load(plan_path)
    rows = plan['parents']
    if len(rows) != 10 or [row['ordinal'] for row in rows] != list(range(1, 11)):
        raise ValueError('need every original full-stock parent in numeric ordinal order')
    config = Path(plan['config']['path']).read_text()
    reports, evidence, sources, candidates = [], [], [], []
    for row in rows:
        directory = output_root / f"parent_{row['ordinal']}"
        current = response_validate.validate(plan_path, row['ordinal'], directory)
        saved = control.load(directory / 'completed.json')
        if current != saved:
            raise ValueError('control parent postflight no longer equals completed receipt')
        reports.append(current)
        sources.append(row['source']['path'])
        candidates.append(str(directory / 'unfiltered_1.art'))
        evidence += integrity.files([directory / 'completed.json'])
        evidence += current['inputs_after'] + current['roundtrip']['inputs_after'] + current['outputs']
    original = control.load(fast_prepared / 'manifest.json')
    if original.get('full_production_history_certified') is not True:
        raise ValueError('saved F must bind a complete certified original stored history')
    history = original['production_history_receipt']
    hybrid_history.verify(history, config, sources, plan['events'])
    admission = hybrid_input.audit(sources, candidates, config, plan['events'], 'R7FastSD::R7TrackerInject')
    text, manifest = hybrid_replay.prepare(config, sources, candidates, admission, plan['events'], history)
    if len(manifest['paths']) != 52:
        raise ValueError('full-stock response control requires every scheduled physics path')
    manifest['response_control'] = {'admission_kind': control.KIND, 'scope': control.SCOPE,
        'plan': integrity.digest(plan_path), 'parents': reports, 'electronics_only': False,
        'particle_state_lossless': False, 'source_simparticle_product_ids_authenticated': False,
        'trigger_counter_policy': 'one process, own continuous all-parent filter invocation history'}
    manifest['injection_campaign'] = {'runtime': plan['runtime'], 'scope': control.SCOPE,
        'events': plan['events'], 'all_planned_parents_verified': True}
    manifest['injection_campaign_evidence'] = evidence + integrity.files([plan_path,
        fast_prepared / 'manifest.json', fast_prepared / 'replay.fcl'])
    output.mkdir(parents=True)
    control.save(output / 'admission.json', admission)
    (output / 'replay.fcl').write_text(text)
    control.save(output / 'manifest.json', manifest)
    hybrid_replay.preflight(output, config)
    return manifest


def validate(art, prepared, config, log):
    report = hybrid_replay.validate(art, prepared, config.read_text(), log.read_text())
    manifest, _ = hybrid_replay.preflight(prepared, config.read_text())
    if manifest.get('response_control', {}).get('admission_kind') != control.KIND:
        raise ValueError('not a separate full-stock response control')
    report['scope'] = manifest['population'] + '; ' + control.SCOPE + '; no physics closure or independent uncertainty'
    report['response_control'] = manifest['response_control']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('prepare')
    for name in ('plan', 'output-root', 'fast-prepared', 'output'):
        create.add_argument('--' + name, type=Path, required=True)
    check = commands.add_parser('validate')
    for name in ('art', 'prepared', 'config', 'log', 'output'):
        check.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        report = prepare(args.plan, args.output_root, args.fast_prepared, args.output)
        print(f"prepared ONE full-stock SGS control trigger: {report['expected_events']} events, all52 paths")
    else:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError('refuse response-control trigger comparison overwrite')
        report = validate(args.art, args.prepared, args.config, args.log)
        control.save(args.output, report)
        print(f"measured {report['events']} control events; migrations are not an equality gate")


if __name__ == '__main__':
    main()
