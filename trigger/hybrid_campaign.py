"""Prepare one all-path trigger process from completed parent-isolated fast injections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hybrid_history
import hybrid_input
import hybrid_replay
import injection_campaign
import integrity
import production_history_log


def prepare(plan_path, regeneration, stock_directory, comparison, output, production_log):
    if output.exists():
        raise ValueError('refuse full-history hybrid preparation overwrite')
    plan = injection_campaign.load(plan_path)
    completed = injection_campaign.completed(plan)
    config = Path(plan['config']['path']).read_text()
    log_history = production_history_log.verify(config, regeneration, production_log)
    history = hybrid_history.certify(config, regeneration, stock_directory, comparison, plan['events'])
    admission = hybrid_input.audit(completed['reference'], completed['candidate'], config,
        plan['events'], 'R7FastSD::R7TrackerInject')
    text, manifest = hybrid_replay.prepare(config, completed['reference'], completed['candidate'],
        admission, plan['events'], history)
    if len(manifest['paths']) != 52:
        raise ValueError('full-history Run 1A diagnostic requires all52 scheduled physics paths')
    manifest['injection_campaign'] = completed
    manifest['production_log_history'] = log_history
    manifest['injection_campaign_evidence'] = integrity.files([plan_path, production_log]) + completed['injection_evidence']
    output.mkdir(parents=True)
    (output / 'admission.json').write_text(json.dumps(admission, indent=2) + '\n')
    (output / 'replay.fcl').write_text(text)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    hybrid_replay.preflight(output, config)
    print(f"prepared one52path trigger process over {plan['events']} events/{len(plan['parents'])} parents")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'regeneration', 'stock-directory', 'comparison', 'output', 'production-log'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.plan, args.regeneration, args.stock_directory, args.comparison, args.output, args.production_log)


if __name__ == '__main__':
    main()
