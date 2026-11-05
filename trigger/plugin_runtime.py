"""Prepare a private released-runtime runner exposing existing hash-bound adapter plugins; never build."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import integrity


ANCHOR = '  export FHICL_FILE_PATH="$(dirname "$FCL"):${OFFLINE_FCL_PATH:+$OFFLINE_FCL_PATH:}$FHICL_FILE_PATH"'
ENV = 'OFFLINE_SEARCH_PATH OFFLINE_OUTPUT_FILE OFFLINE_EVENTS OFFLINE_FCL_PATH OFFLINE_WARM_REPEAT;'


def prepare(runner, directory, output, capture_geometry=False):
    runner, directory, output = Path(runner), Path(directory), Path(output)
    text = runner.read_text()
    if text.count(ANCHOR) != 1 or text.count(ENV) != 1:
        raise ValueError('unsupported Offline runner anchors')
    libraries = sorted(directory.glob('libtrkfastsim_*_module.so'))
    if not libraries or output.exists():
        raise ValueError('need existing plugin libraries and a new runtime directory')
    extra = '\n'.join(['  [[ -d ${OFFLINE_PLUGIN_DIR:-} ]] || fail "missing existing adapter plugin directory"',
        '  export LD_LIBRARY_PATH="$OFFLINE_PLUGIN_DIR:${LD_LIBRARY_PATH:-}"',
        '  export CET_PLUGIN_PATH="$OFFLINE_PLUGIN_DIR:${CET_PLUGIN_PATH:-}"'])
    adapted = text.replace(ANCHOR, ANCHOR + '\n' + extra).replace(ENV, ENV[:-1] + ' OFFLINE_PLUGIN_DIR;')
    if capture_geometry:
        anchor = '  mu2e --version'
        if adapted.count(anchor) != 1:
            raise ValueError('unsupported version anchor for geometry capture')
        capture = '\n'.join(['  IFS=: read -r -a geometry_roots <<< "$MU2E_SEARCH_PATH"',
            '  geometry_saved=0', '  for geometry_root in "${geometry_roots[@]}"; do',
            '    if [[ -f $geometry_root/Offline/Mu2eG4/geom/geom_run1_a.txt ]]; then',
            '      mkdir -p "$OFFLINE_WORK_DIR/runtime_geometry/Offline/Mu2eG4"',
            '      cp -a "$geometry_root/Offline/Mu2eG4/geom" "$OFFLINE_WORK_DIR/runtime_geometry/Offline/Mu2eG4/"',
            '      printf "%s\\n" "$geometry_root" > "$OFFLINE_WORK_DIR/geometry-source.txt"',
            '      geometry_saved=1; break', '    fi', '  done',
            '  [[ $geometry_saved == 1 ]] || fail "production geometry source not found"'])
        adapted = adapted.replace(anchor, anchor + '\n' + capture)
    receipt = {'runner': integrity.digest(runner), 'plugins': integrity.files(libraries),
        'adapted_runner_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
        'scope': 'existing plugins with released runtime; binary compatibility still requires an actual round trip'}
    output.mkdir(parents=True)
    bundle = output / 'plugins'
    bundle.mkdir()
    for library, expected in zip(libraries, receipt['plugins']):
        target = bundle / library.name
        shutil.copy2(library, target)
        if integrity.digest(target)['sha256'] != expected['sha256']:
            raise ValueError('plugin changed while freezing private bundle')
        target.chmod(0o500)
    checksums = '\n'.join(f"{row['sha256']}  {row['path'].rsplit('/', 1)[-1]}" for row in receipt['plugins']) + '\n'
    (bundle / 'SHA256SUMS').write_text(checksums)
    check = '  ( cd "$OFFLINE_PLUGIN_DIR" && sha256sum -c SHA256SUMS ) || fail "adapter plugin bundle changed"'
    adapted = adapted.replace('  export CET_PLUGIN_PATH=', check + '\n  export CET_PLUGIN_PATH=')
    receipt['adapted_runner_sha256'] = hashlib.sha256(adapted.encode()).hexdigest()
    (output / 'offline_plugins.sh').write_text(adapted)
    (output / 'runtime.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runner', type=Path)
    parser.add_argument('--plugins', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-geometry', action='store_true')
    args = parser.parse_args()
    report = prepare(args.runner, args.plugins, args.output, args.capture_geometry)
    print(f"bound {len(report['plugins'])} existing plugin libraries; not binary compatibility certification")


if __name__ == '__main__':
    main()
