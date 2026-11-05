"""No-build receipts and a private, single-plugin released-runtime runner.

Capture on the Linux build host in the already configured p109 environment.
A user completion log is evidence to hash, not proof of compilation provenance.
Preparation does not run art or establish feature/parity validity. Verification
requires the original source area, report and stock runner to remain available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


BACKING = '/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ba'
RELEASE = 'v13_41_01'
STUB = 'al9-prof-e29-p109'
LIBRARY = 'libr7trigger_KalSeedSelectionFeatures_module.so'
LIBRARY_RELATIVE = f'build/{STUB}/R7Trigger/lib/{LIBRARY}'
SOURCE_HASHES = {
    'src/KalSeedSelectionFeatures_module.cc': 'f9bdf6ea4fc2a4b7742f6fb84e2ed41d764ad8d29fb58478f34db22045873582',
    'src/SConscript': '0c06bc3f00a692291d052b4d7206f71640113e4ee7c3fe88db5c32e2f0c376e0',
    'CMakeLists.txt': '7b6a8b8c7b41ef1dc75bfcba55fd7f0bc0f925b6f7dc44eaa4ab5c81302f61d5',
    '.muse': '3bb8d3125409dfcbf2839e17cec3433c6556694a7a19eab94d40bea23912efa4',
}
REGISTRATION = ('make_module', 'moduleType', 'get_source_location')
REQUIRED_NEEDED = ('libmu2e_RecoDataProducts.so', 'libmu2e_GeneralUtilities.so',
                   'libart_Utilities.so', 'libcetlib.so')
ANCHOR = '  export FHICL_FILE_PATH="$(dirname "$FCL"):${OFFLINE_FCL_PATH:+$OFFLINE_FCL_PATH:}$FHICL_FILE_PATH"'
ENV = 'OFFLINE_SEARCH_PATH OFFLINE_OUTPUT_FILE OFFLINE_EVENTS OFFLINE_FCL_PATH OFFLINE_WARM_REPEAT;'
SCOPE = ('Byte-bound read-only admission, not reproducible-build attestation, loaded-library '
         'tracing, successful art execution, feature validation or measured parity.')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pin(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('explicit expected library SHA256 required')
    return value


def _path(path):
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError('explicit filesystem path required, not a boolean or trust flag')
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f'refuse symlink path: {path}')
    return path


def _file(path):
    path = _path(path)
    if not path.is_file():
        raise ValueError(f'missing regular file: {path}')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': _sha(path.read_bytes())}


def _json(path):
    return json.loads(Path(path).read_text())


def _command(*args):
    try:
        result = subprocess.run(list(args), capture_output=True, text=True, check=False,
                                timeout=120, env={**os.environ, 'LC_ALL': 'C'})
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f'inspection unavailable: {args[0]}') from error
    text = result.stdout + result.stderr
    if result.returncode:
        raise ValueError(f'{args[0]} inspection failed: {text}')
    return text


def _allowed_resolution(name, path):
    offline = f'/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/{RELEASE}/build/{STUB}/Offline/lib/'
    if '$' in path or not path.startswith('/') or Path(path).name != name:
        return False
    normalized = os.path.normpath(path)
    if name.startswith('libmu2e_'):
        return normalized == offline + name
    if normalized.startswith('/cvmfs/'):
        view = ('/cvmfs/mu2e.opensciencegrid.org/spackages/241207/spack/var/spack/'
                f'environments/muse-{STUB}/.spack-env/view')
        candidates = [Path(view) / subdir / name for subdir in ('lib', 'lib64', 'lib/root', 'tbb/2022.0/lib')]
        if normalized in map(str, candidates):
            return True
        return any(candidate.is_file() and os.path.samefile(candidate, normalized) for candidate in candidates)
    system = re.fullmatch(r'(?:lib(?:c|m|dl|rt|pthread|resolv|util|gcc_s|stdc\+\+|nsl|crypt|crypto|ssl|freetype|harfbuzz|glib-2\.0|graphite2|brotlidec|brotlicommon|z)\.so(?:\.[0-9]+)*|ld-linux-x86-64\.so\.2)', name)
    return bool(system and normalized.startswith(('/lib/', '/lib64/', '/usr/lib/', '/usr/lib64/')))


def _resolutions(text, needed):
    if re.search(r'not found|undefined symbol|symbol lookup error|not a dynamic executable|statically linked', text, re.I):
        raise ValueError('ldd -r missing library or unresolved symbol')
    paths = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'\s*(\S+)\s+=>\s+(/\S+)\s+\(0x[0-9a-fA-F]+\)\s*', line)
        if match:
            name, path = match.groups()
        else:
            direct = re.fullmatch(r'\s*(/\S+)\s+\(0x[0-9a-fA-F]+\)\s*', line)
            if direct:
                path = direct[1]
                name = Path(path).name
            elif re.fullmatch(r'\s*linux-vdso\.so\.1\s+\(0x[0-9a-fA-F]+\)\s*', line):
                continue
            else:
                raise ValueError(f'unrecognized ldd evidence: {line}')
        if name in paths or not _allowed_resolution(name, path):
            raise ValueError(f'wrong or uncheckable Offline/external resolution: {name} => {path}')
        paths[name] = path
    if not set(needed) <= paths.keys():
        raise ValueError('ldd evidence lacks ELF NEEDED dependencies')
    return paths


def inspect_build(area, *, expected_library_sha256, user_build_report):
    """Hash exact sources/library and inspect ELF, relocations and registration.

    user_build_report is a regular text file containing the user's successful
    SCons completion line, not a bool, a success JSON field or a build launcher.
    Strict resolution policy rejects uncheckable non-p109 external paths too.
    """
    pin = _pin(expected_library_sha256)
    area = _path(area)
    if not area.is_dir():
        raise ValueError('missing build area')
    backing = area / 'backing'
    if not backing.is_symlink() or os.readlink(backing) != BACKING:
        raise ValueError('need exact explicit released backing symlink')
    repos = {p.name for p in area.iterdir() if p.is_dir() and not p.is_symlink()
             and ((p / 'src/SConscript').exists() or (p / '.muse').exists() or p.name in ('Offline', 'KinKal'))}
    if repos != {'R7Trigger'} or any((area / name).exists() or (area / name).is_symlink()
                                     for name in ('Offline', 'KinKal')):
        raise ValueError('build area must contain only the R7Trigger local repository')
    root = _path(area / 'R7Trigger')
    source_dir = _path(root / 'src')
    if (not source_dir.is_dir() or
            {p.name for p in source_dir.iterdir()} != {'SConscript', 'KalSeedSelectionFeatures_module.cc'}):
        raise ValueError('unexpected satellite source names')
    sources = []
    for name, sha in SOURCE_HASHES.items():
        row = _file(root / name)
        if row['sha256'] != sha:
            raise ValueError(f'source/build hash differs: {name}')
        sources.append(row)
    report = _file(user_build_report)
    report_text = Path(report['path']).read_text()
    logged = re.search(r'^scons: done building targets\.\s*$', report_text, re.M)
    reported = ('User report: The build succeeded that time' in report_text and
                re.search(r'\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} to \d{2}:\d{2}:\d{2}', report_text))
    if not (logged or reported):
        raise ValueError('need textual user successful build report plus completion record; boolean trust is not evidence')
    if re.search(r'scons: \*\*\*|building terminated because of errors', report_text, re.I):
        raise ValueError('ambiguous or failed user build report')
    completion = _file(area / f'build/{STUB}/.musebuild') if reported else None
    if reported and Path(completion['path']).read_text().strip() not in report_text:
        raise ValueError('reported completion record differs from actual .musebuild')
    library = _file(area / LIBRARY_RELATIVE)
    if library['sha256'] != pin:
        raise ValueError('explicit library SHA256 differs')
    library_time = Path(library['path']).stat().st_mtime_ns
    if any(Path(row['path']).stat().st_mtime_ns >= library_time for row in sources):
        raise ValueError('library must be newer than every source/build input')
    header = _command('readelf', '-h', library['path'])
    if not all(re.search(pattern, header) for pattern in
               (r'Class:\s+ELF64', r'Data:.*little endian', r'Type:\s+DYN', r'Machine:.*X86-64')):
        raise ValueError('need ELF64 x86-64 shared object')
    dynamic = _command('readelf', '-d', library['path'])
    needed = sorted(set(re.findall(r'\(NEEDED\).*\[([^\]]+)\]', dynamic)))
    if not set(REQUIRED_NEEDED) <= set(needed):
        raise ValueError('missing required direct ELF dependencies')
    for entry in re.findall(r'\((?:RPATH|RUNPATH)\).*\[([^\]]*)\]', dynamic):
        for path in entry.split(':'):
            if not _allowed_search(path):
                raise ValueError(f'uncheckable or wrong ELF search path: {path}')
    relocations = _command('ldd', '-r', library['path'])
    paths = _resolutions(relocations, needed)
    symbols = _command('nm', '-D', '--defined-only', library['path'])
    defined = set(re.findall(r'^\s*[0-9a-fA-F]+\s+[TW]\s+(\S+)\s*$', symbols, re.M))
    if not set(REGISTRATION) <= defined:
        raise ValueError('missing art module registration symbols')
    if not backing.is_symlink() or os.readlink(backing) != BACKING:
        raise ValueError('backing changed during inspection')
    if library != _file(library['path']) or sources != [_file(row['path']) for row in sources] or report != _file(report['path']):
        raise ValueError('build evidence changed during inspection')
    return {'version': 1, 'area': str(area), 'required_backing': BACKING, 'muse_stub': STUB,
            'production_profile': 'prof', 'offline_release': RELEASE, 'source_files': sources,
            'library': library, 'library_mtime_ns': library_time, 'user_build_report': report,
            'completion_record': completion,
            'needed': needed, 'resolved_library_paths': paths,
            'inspection': {'readelf_header': header, 'readelf_dynamic': dynamic,
                           'ldd_r': re.sub(r'\(0x[0-9a-fA-F]+\)', '(ADDRESS)', relocations), 'nm_defined': symbols},
            'scope': SCOPE, 'runtime_ready': False}


def _allowed_search(path):
    if path in ('${ORIGIN}/../../R7Trigger/lib', '$ORIGIN/../../R7Trigger/lib',
                f'{BACKING}/build/{STUB}/Production/lib', f'{BACKING}/build/{STUB}/mu2e-trig-config/lib',
                '/cvmfs/mu2e.opensciencegrid.org/spackages/241207/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.5.0/gcc-13.3.0-33ycq6cwlyc3h4v2kizlvbaljeizsl3d/lib64'):
        return True
    if path == f'/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/{RELEASE}/build/{STUB}/Offline/lib':
        return True
    if (path.startswith('/cvmfs/mu2e.opensciencegrid.org/') and
            f'/environments/muse-{STUB}/.spack-env/view/' in path):
        return '$' not in path and '..' not in Path(path).parts
    return False


def verify_build(receipt, *, expected_library_sha256):
    """Reinspect original evidence; no saved inspection boolean is trusted."""
    receipt = _json(receipt) if isinstance(receipt, (str, Path)) else receipt
    observed = inspect_build(receipt['area'], expected_library_sha256=expected_library_sha256,
                             user_build_report=receipt['user_build_report']['path'])
    if observed != receipt:
        raise ValueError('build receipt differs from current independent inspection')
    return observed


def _source_set_sha(receipt):
    rows = {name: row['sha256'] for name, row in zip(SOURCE_HASHES, receipt['source_files'])}
    return _sha(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode())


def _worker_checks(receipt):
    sha = receipt['library']['sha256']
    source_sha = receipt['source_files'][0]['sha256']
    source_set_sha = _source_set_sha(receipt)
    checks = rf'''  [[ $AREA == {shlex.quote(BACKING)} ]] || fail "feature runner requires exact released area"
  [[ $MUSE_WORK_DIR == {shlex.quote(BACKING)} ]] || fail "wrong feature runtime MUSE_WORK_DIR"
  [[ $MUSE_STUB == {STUB} && $MUSE_ENVSET == p109 ]] || fail "wrong feature runtime stub/envset"
  [[ -n ${{OFFLINE_FEATURE_BUNDLE:-}} && -d $OFFLINE_FEATURE_BUNDLE ]] || fail "missing feature bundle"
  feature_lib="$OFFLINE_FEATURE_BUNDLE/plugins/{LIBRARY}"
  [[ $(readlink -f "$OFFLINE_FEATURE_BUNDLE") == "$OFFLINE_FEATURE_BUNDLE" ]] || fail "redirected feature bundle root"
  [[ $(find "$OFFLINE_FEATURE_BUNDLE/plugins" -mindepth 1 -maxdepth 1 | wc -l) -eq 1 ]] || fail "need one feature plugin"
  [[ ! -L $OFFLINE_FEATURE_BUNDLE && ! -L $OFFLINE_FEATURE_BUNDLE/plugins && ! -L $feature_lib && -f $feature_lib ]] || fail "redirected feature bundle"
  [[ $(sha256sum "$feature_lib" | cut -d ' ' -f1) == {sha} ]] || fail "feature library SHA differs"
  [[ $(sha256sum "$0" | cut -d ' ' -f1) == "$OFFLINE_FEATURE_RUNNER_SHA256" ]] || fail "feature runner SHA differs"
  export LD_LIBRARY_PATH="$OFFLINE_FEATURE_BUNDLE/plugins:${{LD_LIBRARY_PATH:-}}"
  export CET_PLUGIN_PATH="$OFFLINE_FEATURE_BUNDLE/plugins:${{CET_PLUGIN_PATH:-}}"
  feature_ldd=$(ldd -r "$feature_lib" 2>&1) || fail "worker feature ldd -r failed: $feature_ldd"
  for feature_needed in libmu2e_RecoDataProducts.so libmu2e_GeneralUtilities.so libart_Utilities.so libcetlib.so; do
    [[ $feature_ldd == *"$feature_needed => /"* ]] || fail "missing worker dependency: $feature_needed"
  done
  printf '%s\n' "$feature_ldd"
  [[ ! $feature_ldd =~ 'not found' && ! $feature_ldd =~ 'undefined symbol' && ! $feature_ldd =~ 'symbol lookup error' ]] || fail "worker feature relocations unresolved"
  while IFS= read -r feature_line; do
    [[ -n $feature_line ]] || continue
    [[ $feature_line != *'$'* ]] || fail "redirected worker dependency"
    feature_name=; feature_path=
    if [[ $feature_line =~ ^[[:space:]]*([^[:space:]]+)[[:space:]]+\=\>[[:space:]]+(/[^[:space:]]+)[[:space:]]+\(0x[[:xdigit:]]+\) ]]; then
      feature_name=${{BASH_REMATCH[1]}}; feature_path=${{BASH_REMATCH[2]}}
    elif [[ $feature_line =~ ^[[:space:]]*(/[^[:space:]]+)[[:space:]]+\(0x[[:xdigit:]]+\) ]]; then
      feature_path=${{BASH_REMATCH[1]}}; feature_name=${{feature_path##*/}}
    elif [[ $feature_line =~ ^[[:space:]]*linux-vdso.so.1[[:space:]]+\(0x[[:xdigit:]]+\) ]]; then
      continue
    else fail "unrecognized worker ldd evidence: $feature_line"; fi
    feature_path=$(readlink -m "$feature_path")
    case "$feature_name" in
      libmu2e_*) [[ $feature_path == /cvmfs/mu2e.opensciencegrid.org/Musings/Offline/{RELEASE}/build/{STUB}/Offline/lib/"$feature_name" ]] || fail "wrong worker Offline" ;;
      *) case "$feature_path" in
        /cvmfs/mu2e.opensciencegrid.org/*)
          feature_bound=0
          for feature_subdir in lib lib64 lib/root tbb/2022.0/lib; do
            feature_candidate="/cvmfs/mu2e.opensciencegrid.org/spackages/241207/spack/var/spack/environments/muse-{STUB}/.spack-env/view/$feature_subdir/$feature_name"
            if [[ $feature_path == "$feature_candidate" || $feature_path -ef $feature_candidate ]]; then feature_bound=1; fi
          done
          [[ $feature_bound == 1 ]] || fail "worker external is not an exact p109-view library: $feature_path" ;;
        /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*)
          [[ $feature_name =~ ^(lib(c|m|dl|rt|pthread|resolv|util|gcc_s|stdc\+\+|nsl|crypt|crypto|ssl|freetype|harfbuzz|glib-2\.0|graphite2|brotlidec|brotlicommon|z)\.so(\.[0-9]+)*|ld-linux-x86-64\.so\.2)$ ]] || fail "uncheckable worker system dependency" ;;
        *) fail "wrong or uncheckable worker external: $feature_path" ;;
      esac ;;
    esac
  done <<< "$feature_ldd"
  printf 'feature_runtime library_sha256=%s source_sha256=%s source_set_sha256=%s stub=%s backing=%s area=%s runner_sha256=%s relocations=checked\n' {sha} {source_sha} {source_set_sha} "$MUSE_STUB" {shlex.quote(BACKING)} "$AREA" "$OFFLINE_FEATURE_RUNNER_SHA256"'''
    return checks


def _adapt(text, receipt):
    if text.count(ANCHOR) != 1 or text.count(ENV) != 1:
        raise ValueError('unsupported stock runner anchors')
    outer = 'SELF=$(readlink -f "$0")'
    if text.count(outer) != 1 or text.count('  mu2e --version') != 1:
        raise ValueError('unsupported stock runner structure')
    checks = _worker_checks(receipt)
    text = text.replace(ANCHOR, ANCHOR + '\n' + checks)
    text = text.replace(ENV, ENV[:-1] + ' OFFLINE_FEATURE_BUNDLE OFFLINE_FEATURE_RUNNER_SHA256;')
    outer_checks = '''[[ $AREA == ''' + shlex.quote(BACKING) + ''' ]] || fail "feature runner requires exact released area"
OFFLINE_FEATURE_BUNDLE=$(dirname "$SELF")
OFFLINE_FEATURE_RUNNER_SHA256=$(sha256sum "$SELF" | cut -d ' ' -f1)'''
    return text.replace(outer, outer + '\n' + outer_checks)


def _manifest(receipt, stock, adapted):
    return {'version': 1, 'build_receipt': receipt, 'stock_runner': stock,
            'adapted_runner_sha256': _sha(adapted.encode()),
            'runtime_area': BACKING, 'library_relative': 'plugins/' + LIBRARY,
            'scope': SCOPE, 'runtime_ready': False}


def prepare(runner, output, *, source_module, build_receipt, expected_library_sha256):
    """Freeze one verified library and an adapted copy; never modify stock runner.

    The whole output directory is self-contained for worker use and can be moved
    as a unit. The source/report/runner paths in runtime.json remain build-host
    evidence and must still exist for independent Python verification.
    """
    output = _path(output)
    if output.exists():
        raise ValueError('refuse runtime bundle overwrite')
    receipt = verify_build(build_receipt, expected_library_sha256=expected_library_sha256)
    if _file(source_module) != receipt['source_files'][0]:
        raise ValueError('source module is not the verified build input')
    stock = _file(runner)
    adapted = _adapt(Path(stock['path']).read_text(), receipt)
    manifest = _manifest(receipt, stock, adapted)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plugins').mkdir()
    target = output / 'plugins' / LIBRARY
    with target.open('xb') as destination, Path(receipt['library']['path']).open('rb') as source:
        shutil.copyfileobj(source, destination)
    if _file(target)['sha256'] != expected_library_sha256:
        raise ValueError('library changed while freezing bundle')
    target.chmod(0o500)
    (output / 'offline_features.sh').write_text(adapted)
    (output / 'offline_features.sh').chmod(0o500)
    (output / 'runtime.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(output, expected_library_sha256=expected_library_sha256)
    return manifest


def verify(bundle, *, expected_library_sha256):
    """Recheck frozen bytes and rederive runner from independently checked build."""
    bundle = _path(bundle)
    if {p.name for p in bundle.iterdir()} != {'plugins', 'offline_features.sh', 'runtime.json'}:
        raise ValueError('unexpected runtime bundle members')
    _file(bundle / 'runtime.json')
    manifest = _json(bundle / 'runtime.json')
    receipt = verify_build(manifest['build_receipt'], expected_library_sha256=expected_library_sha256)
    stock = _file(manifest['stock_runner']['path'])
    adapted = _adapt(Path(stock['path']).read_text(), receipt)
    if (manifest != _manifest(receipt, stock, adapted) or
            _file(bundle / 'offline_features.sh')['sha256'] != _sha(adapted.encode())):
        raise ValueError('runner/manifest binding differs')
    plugins = _path(bundle / 'plugins')
    if {p.name for p in plugins.iterdir()} != {LIBRARY}:
        raise ValueError('need exactly one frozen plugin')
    if _file(plugins / LIBRARY)['sha256'] != _pin(expected_library_sha256):
        raise ValueError('frozen library SHA differs')
    return manifest


def verify_deployment(bundle, *, expected_library_sha256, expected_manifest_sha256):
    """Verify a moved frozen bundle against an independently pinned manifest SHA."""
    bundle = _path(Path(bundle).resolve(strict=True))
    if {p.name for p in bundle.iterdir()} != {'plugins', 'offline_features.sh', 'runtime.json'}:
        raise ValueError('unexpected runtime bundle members')
    if _file(bundle / 'runtime.json')['sha256'] != _pin(expected_manifest_sha256):
        raise ValueError('deployment manifest SHA differs')
    manifest = _json(bundle / 'runtime.json')
    receipt = manifest['build_receipt']
    pin = _pin(expected_library_sha256)
    if (receipt['library']['sha256'] != pin or manifest.get('runtime_ready') is not False
            or manifest['runtime_area'] != BACKING or receipt['required_backing'] != BACKING
            or receipt['muse_stub'] != STUB or receipt['production_profile'] != 'prof'
            or receipt['offline_release'] != RELEASE
            or [row['sha256'] for row in receipt['source_files']] != list(SOURCE_HASHES.values())):
        raise ValueError('deployed build/source/runtime declaration differs')
    plugins = _path(bundle / 'plugins')
    if {p.name for p in plugins.iterdir()} != {LIBRARY}:
        raise ValueError('need exactly one deployed plugin')
    if (_file(plugins / LIBRARY)['sha256'] != pin or manifest['library_relative'] != 'plugins/' + LIBRARY
            or _file(bundle / 'offline_features.sh')['sha256'] != manifest['adapted_runner_sha256']):
        raise ValueError('deployed library or runner differs')
    return manifest


def observe_worker(bundle, log, *, expected_library_sha256, expected_manifest_sha256):
    """Check pinned deployment and exact worker receipt, not successful art or parity."""
    manifest = verify_deployment(bundle, expected_library_sha256=expected_library_sha256,
                                 expected_manifest_sha256=expected_manifest_sha256)
    expected = (f'feature_runtime library_sha256={expected_library_sha256} '
                f'source_sha256={SOURCE_HASHES["src/KalSeedSelectionFeatures_module.cc"]} '
                f'source_set_sha256={_source_set_sha(manifest["build_receipt"])} '
                f'stub={STUB} backing={BACKING} area={BACKING} '
                f'runner_sha256={manifest["adapted_runner_sha256"]} relocations=checked')
    if [line for line in log.splitlines() if line.startswith('feature_runtime ')] != [expected]:
        raise ValueError('missing, duplicate or wrong worker receipt')
    return {'worker_receipt': expected, 'deployment_manifest_sha256': expected_manifest_sha256,
            'library_sha256': expected_library_sha256, 'runtime_ready': False, 'scope': SCOPE}


def validate_run(bundle, replay_log, expanded_fcl, *, expected_library_sha256, expected_expanded_sha256):
    """Check emitted worker receipt + exact expanded FCL; not art/parity success.

    The expected expanded hash must be admitted independently, not copied from
    the candidate run. Logs are observations, not cryptographic execution proof.
    """
    manifest = verify(bundle, expected_library_sha256=expected_library_sha256)
    expected = (f'feature_runtime library_sha256={expected_library_sha256} '
                f'source_sha256={SOURCE_HASHES["src/KalSeedSelectionFeatures_module.cc"]} '
                f'source_set_sha256={_source_set_sha(manifest["build_receipt"])} '
                f'stub={STUB} backing={BACKING} area={BACKING} '
                f'runner_sha256={manifest["adapted_runner_sha256"]} relocations=checked')
    log = Path(_file(replay_log)['path']).read_text()
    receipts = [line for line in log.splitlines() if line.startswith('feature_runtime ')]
    if receipts != [expected]:
        raise ValueError('missing, duplicate or wrong worker receipt')
    expanded = _file(expanded_fcl)
    if expanded['sha256'] != _pin(expected_expanded_sha256):
        raise ValueError('expanded FCL SHA differs')
    return {'worker_receipt': expected, 'expanded_fcl': expanded, 'replay_log': _file(replay_log),
            'scope': SCOPE, 'runtime_ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('capture')
    capture.add_argument('--area', type=Path, required=True)
    capture.add_argument('--user-build-report', type=Path, required=True)
    capture.add_argument('--output', type=Path, required=True)
    create = commands.add_parser('prepare')
    for name in ('runner', 'source-module', 'build-receipt', 'output'):
        create.add_argument('--' + name, type=Path, required=True)
    check = commands.add_parser('verify')
    check.add_argument('--bundle', type=Path, required=True)
    deployed = commands.add_parser('verify-deployment')
    deployed.add_argument('--bundle', type=Path, required=True)
    deployed.add_argument('--expected-manifest-sha256', required=True)
    for command in (capture, create, check, deployed):
        command.add_argument('--expected-library-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'capture':
        output = _path(args.output)
        if output.exists():
            raise ValueError('refuse receipt overwrite')
        result = inspect_build(args.area, expected_library_sha256=args.expected_library_sha256,
                               user_build_report=args.user_build_report)
        with output.open('x') as destination:
            destination.write(json.dumps(result, indent=2) + '\n')
    elif args.command == 'prepare':
        prepare(args.runner, args.output, source_module=args.source_module,
                build_receipt=args.build_receipt, expected_library_sha256=args.expected_library_sha256)
    elif args.command == 'verify':
        verify(args.bundle, expected_library_sha256=args.expected_library_sha256)
    else:
        verify_deployment(args.bundle, expected_library_sha256=args.expected_library_sha256,
                          expected_manifest_sha256=args.expected_manifest_sha256)
    print(SCOPE)


if __name__ == '__main__':
    main()
