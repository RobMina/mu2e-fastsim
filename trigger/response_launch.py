"""Write a verified full-stock parent launch environment; never execute art or build."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex

import integrity
import response_control as control


def prepare(plan_path, ordinal, output):
    output = Path(output)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError('need an existing empty fresh control output directory')
    plan, row = control.preflight(plan_path, ordinal)
    binding = {'plan': integrity.digest(plan_path), 'parent': row, 'runtime': plan['runtime'],
               'scope': control.SCOPE, 'admission_kind': control.KIND}
    control.save(output / 'execution-binding.json', binding)
    runtime = plan['runtime']
    runner = Path(plan['runtime_binding']['runner']['path'])
    plugins = runner.parent / 'plugins'
    selection = {'runner': str(runner.resolve()), 'plugin_directory': str(plugins.resolve())}
    control.save(output / 'runtime-selection.json', selection)
    (output / 'roundtrip').mkdir()
    control.save(output / 'roundtrip/runtime-selection.json', selection)
    values = {'OFFLINE_EVENTS': row['events'], 'OFFLINE_IMAGE': runtime['OFFLINE_IMAGE'],
              'CVMFSEXEC_DIST': runtime['CVMFSEXEC_DIST'], 'TRIGGER_AREA': runtime['TRIGGER_AREA'],
              'CONTROL_RUNNER': str(runner), 'CONTROL_PLUGINS': str(plugins),
              'CONTROL_PREPARED': row['prepared'], 'CONTROL_CONFIG': plan['config']['path']}
    with (output / 'launch.env').open('x') as stream:
        stream.write(''.join(f'export {name}={shlex.quote(str(value))}\n' for name, value in values.items()))
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--ordinal', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.plan, args.ordinal, args.output)


if __name__ == '__main__':
    main()
