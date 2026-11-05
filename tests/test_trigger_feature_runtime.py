"""No compiler, muse, art or remote calls: mock ELF inspection and fake bundles."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import feature_runtime as runtime

REAL_RUN = subprocess.run
PROJECT = Path(__file__).resolve().parents[1]
SOURCES = PROJECT / 'trigger/art/R7Trigger'
RUNNER = PROJECT / 'container/offline/offline_uva.sh'
VERIFIED_LIBRARY_SHA = '203dce0661049cf9ee6e8fcad5d5fa1af764c667ec705c617e18c8dede617562'
OFFLINE = f'/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/{runtime.RELEASE}/build/{runtime.STUB}/Offline/lib/'
VIEW = f'/cvmfs/mu2e.opensciencegrid.org/spackages/241207/spack/var/spack/environments/muse-{runtime.STUB}/.spack-env/view/lib/'


@pytest.fixture
def built(tmp_path, monkeypatch):
    tmp = tmp_path.resolve()
    area = tmp / 'area'
    area.mkdir()
    shutil.copytree(SOURCES, area / 'R7Trigger')
    (area / 'backing').symlink_to(runtime.BACKING)
    library = area / runtime.LIBRARY_RELATIVE
    library.parent.mkdir(parents=True)
    library.write_bytes(b'fake ELF for subprocess-mocked tests only')
    newest = max((area / 'R7Trigger' / name).stat().st_mtime_ns for name in runtime.SOURCE_HASHES)
    os.utime(library, ns=(newest + 1000000000, newest + 1000000000))
    report = tmp / 'user-build.txt'
    report.write_text('User reports full build finished.\nscons: done building targets.\n')
    paths = {name: (OFFLINE if name.startswith('libmu2e_') else VIEW) + name
             for name in runtime.REQUIRED_NEEDED}
    paths['libCore.so.6.38'] = VIEW + 'root/libCore.so.6.38'
    paths['libc.so.6'] = '/lib64/libc.so.6'
    evidence = {
        'header': 'Class: ELF64\nData: 2\'s complement, little endian\nType: DYN (Shared object file)\nMachine: Advanced Micro Devices X86-64\n',
        'dynamic': ''.join(f' 0x001 (NEEDED) Shared library: [{name}]\n' for name in runtime.REQUIRED_NEEDED),
        'ldd': ''.join(f'\t{name} => {path} (0x1234)\n' for name, path in paths.items()) + '\t/lib64/ld-linux-x86-64.so.2 (0x4321)\n\tlinux-vdso.so.1 (0x9999)\n',
        'nm': ''.join(f'0000001234 T {name}\n' for name in runtime.REGISTRATION),
    }
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        assert args[-1] == str(library)
        key = {('readelf', '-h'): 'header', ('readelf', '-d'): 'dynamic',
               ('ldd', '-r'): 'ldd', ('nm', '-D'): 'nm'}[tuple(args[:2])]
        return SimpleNamespace(returncode=0, stdout=evidence[key], stderr='')
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    return SimpleNamespace(tmp=tmp, area=area, library=library, report=report,
        pin=hashlib.sha256(library.read_bytes()).hexdigest(), evidence=evidence, calls=calls,
        source=area / 'R7Trigger/src/KalSeedSelectionFeatures_module.cc')


def inspect(built, **kwargs):
    return runtime.inspect_build(built.area, **(dict(expected_library_sha256=built.pin,
                                                   user_build_report=built.report) | kwargs))


def freeze(built):
    receipt = inspect(built)
    output = built.tmp / 'bundle'
    manifest = runtime.prepare(RUNNER, output, source_module=built.source,
                              build_receipt=receipt, expected_library_sha256=built.pin)
    return output, manifest


def test_current_source_pins_and_successful_readonly_inspection(built):
    assert {name: hashlib.sha256((SOURCES / name).read_bytes()).hexdigest()
            for name in runtime.SOURCE_HASHES} == runtime.SOURCE_HASHES
    receipt = inspect(built)
    assert receipt['library']['sha256'] == built.pin
    assert len(receipt['source_files']) == 4
    assert receipt['muse_stub'] == 'al9-prof-e29-p109'
    assert receipt['required_backing'] == runtime.BACKING
    assert receipt['production_profile'] == 'prof'
    assert not receipt['runtime_ready']
    assert 'ADDRESS' in receipt['inspection']['ldd_r']
    assert [args[:2] for args, _ in built.calls] == [['readelf', '-h'], ['readelf', '-d'], ['ldd', '-r'], ['nm', '-D']]
    assert all(kwargs['env']['LC_ALL'] == 'C' and kwargs['timeout'] == 120
               and kwargs.get('shell', False) is False for _, kwargs in built.calls)
    runtime.verify_build(receipt, expected_library_sha256=built.pin)
    built.evidence['ldd'] = built.evidence['ldd'].replace('0x1234', '0xabcde')
    runtime.verify_build(receipt, expected_library_sha256=built.pin)


@pytest.mark.parametrize('pin', [None, True, '', '0' * 64, '203dce', VERIFIED_LIBRARY_SHA])
def test_library_sha_must_be_explicit_and_equal_not_defaulted(built, pin):
    with pytest.raises(ValueError):
        inspect(built, expected_library_sha256=pin)


@pytest.mark.parametrize('report', [True, False, None])
def test_boolean_build_proof_is_never_accepted(built, report):
    with pytest.raises(ValueError, match='path'):
        inspect(built, user_build_report=report)


@pytest.mark.parametrize('text', ['success', '{"user_reported_build_succeeded": true}',
    'scons: *** failure\nscons: done building targets.\n'])
def test_report_is_hashed_completion_evidence_not_success_flag(built, text):
    built.report.write_text(text)
    with pytest.raises(ValueError):
        inspect(built)


@pytest.mark.parametrize('change', ['source', 'scons', 'cmake', 'muse', 'extra-source',
                                   'owner-repo', 'offline', 'backing', 'relative-backing',
                                   'stale', 'missing-src', 'symlink-library', 'symlink-source'])
def test_source_layout_backing_and_freshness_gates(built, change):
    root = built.area / 'R7Trigger'
    if change in ('source', 'scons', 'cmake', 'muse'):
        name = {'source': 'src/KalSeedSelectionFeatures_module.cc', 'scons': 'src/SConscript',
                'cmake': 'CMakeLists.txt', 'muse': '.muse'}[change]
        path = root / name
        path.write_bytes(path.read_bytes() + b'\n')
    elif change == 'extra-source':
        (root / 'src/Other_module.cc').write_text('extra')
    elif change == 'owner-repo':
        (built.area / 'owner/src').mkdir(parents=True)
        (built.area / 'owner/src/SConscript').write_text('')
    elif change == 'offline':
        (built.area / 'Offline').symlink_to(built.area / 'R7Trigger')
    elif change in ('backing', 'relative-backing'):
        (built.area / 'backing').unlink()
        (built.area / 'backing').symlink_to(runtime.BACKING + '/.' if change == 'relative-backing'
                                           else runtime.BACKING.replace('ba', 'bb'))
    elif change == 'missing-src':
        shutil.rmtree(root / 'src')
    elif change == 'stale':
        os.utime(built.library, ns=(1, 1))
    else:
        path = built.library if change == 'symlink-library' else built.source
        copy = built.tmp / 'redirect'
        path.rename(copy)
        path.symlink_to(copy)
    with pytest.raises(ValueError):
        inspect(built)


@pytest.mark.parametrize('kind', ['missing', 'unresolved', 'wrong-offline', 'wrong-stub',
    'uncheckable', 'empty', 'unknown-line', 'not-elf', 'debug-rpath', 'missing-direct', 'missing-registration'])
def test_inspection_rejects_linkage_and_registration_gaps(built, kind):
    e = built.evidence
    if kind == 'missing':
        e['ldd'] = e['ldd'].replace(VIEW + 'libcetlib.so (0x1234)', 'not found')
    elif kind == 'unresolved':
        e['ldd'] += 'undefined symbol: _Zbad (library.so)\n'
    elif kind == 'wrong-offline':
        e['ldd'] = e['ldd'].replace('v13_41_01', 'v13_40_00')
    elif kind == 'wrong-stub':
        e['ldd'] = e['ldd'].replace('p109', 'p108')
    elif kind == 'uncheckable':
        e['ldd'] = e['ldd'].replace(VIEW, '/private/spack/hash/lib/')
    elif kind == 'empty':
        e['ldd'] = ''
    elif kind == 'unknown-line':
        e['ldd'] += 'suspicious output\n'
    elif kind == 'not-elf':
        e['header'] = 'not an ELF'
    elif kind == 'debug-rpath':
        e['dynamic'] += '0x123 (RUNPATH) Library runpath: [/unadmitted/debug/lib]\n'
    elif kind == 'missing-direct':
        e['dynamic'] = e['dynamic'].replace('(NEEDED) Shared library: [libcetlib.so]', '(OTHER) [libcetlib.so]')
    else:
        e['nm'] = e['nm'].replace(' T make_module', ' U make_module')
    with pytest.raises(ValueError):
        inspect(built)


def test_tools_unavailable_or_failed_never_silently_skip(built, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('readelf missing')
    monkeypatch.setattr(runtime.subprocess, 'run', missing)
    with pytest.raises(ValueError, match='unavailable'):
        inspect(built)
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout='', stderr='failure'))
    with pytest.raises(ValueError, match='failed'):
        inspect(built)


def test_receipt_changes_require_reinspection(built):
    receipt = inspect(built)
    receipt['library']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='receipt'):
        runtime.verify_build(receipt, expected_library_sha256=built.pin)


def test_bundle_is_single_frozen_library_and_stock_runner_is_unchanged(built, monkeypatch):
    before = RUNNER.read_bytes()
    output, manifest = freeze(built)
    assert RUNNER.read_bytes() == before
    assert sorted(p.name for p in output.iterdir()) == ['offline_features.sh', 'plugins', 'runtime.json']
    assert [p.name for p in (output / 'plugins').iterdir()] == [runtime.LIBRARY]
    assert (output / 'plugins' / runtime.LIBRARY).read_bytes() == built.library.read_bytes()
    text = (output / 'offline_features.sh').read_text()
    assert text.index('muse setup') < text.index('export CET_PLUGIN_PATH') < text.index('ldd -r') < text.index('  mu2e --version')
    assert 'OFFLINE_WARM_REPEAT OFFLINE_FEATURE_BUNDLE OFFLINE_FEATURE_RUNNER_SHA256;' in text
    assert manifest['adapted_runner_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert runtime.BACKING in text
    assert not manifest['runtime_ready']
    monkeypatch.undo()
    result = subprocess.run(['bash', '-n', str(output / 'offline_features.sh')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', ['library', 'runner', 'manifest', 'ready-flag', 'extra-plugin', 'symlink', 'report'])
def test_frozen_bundle_tampering_fails(built, change):
    output, manifest = freeze(built)
    if change == 'library':
        path = output / 'plugins' / runtime.LIBRARY
        path.chmod(0o600)
        path.write_bytes(b'changed')
    elif change == 'runner':
        path = output / 'offline_features.sh'
        path.chmod(0o600)
        path.write_text(path.read_text() + '\n')
    elif change == 'manifest':
        manifest['runtime_area'] += '/other'
        (output / 'runtime.json').write_text(json.dumps(manifest))
    elif change == 'ready-flag':
        manifest['runtime_ready'] = True
        (output / 'runtime.json').write_text(json.dumps(manifest))
    elif change == 'extra-plugin':
        (output / 'plugins/extra.so').write_bytes(b'other')
    elif change == 'report':
        built.report.write_text(built.report.read_text() + 'changed')
    else:
        path = output / 'plugins' / runtime.LIBRARY
        path.unlink()
        path.symlink_to(built.library)
    with pytest.raises(ValueError):
        runtime.verify(output, expected_library_sha256=built.pin)


def test_textual_user_report_and_actual_completion_record(built):
    marker = built.area / f'build/{runtime.STUB}/.musebuild'
    marker.write_text('11/04/26 18:24:43 to 18:24:45\n')
    built.report.write_text('User report: The build succeeded that time\n' + marker.read_text())
    result = inspect(built)
    assert result['completion_record']['sha256'] == hashlib.sha256(marker.read_bytes()).hexdigest()
    marker.write_text('11/04/26 18:24:43 to 18:25:45\n')
    with pytest.raises(ValueError, match='completion record'):
        inspect(built)


def test_deployed_bundle_does_not_need_build_host_but_needs_independent_manifest_pin(built):
    output, manifest = freeze(built)
    pin = hashlib.sha256((output / 'runtime.json').read_bytes()).hexdigest()
    shutil.rmtree(built.area)
    assert runtime.verify_deployment(output, expected_library_sha256=built.pin,
                                     expected_manifest_sha256=pin) == manifest
    with pytest.raises(ValueError, match='manifest SHA'):
        runtime.verify_deployment(output, expected_library_sha256=built.pin,
                                  expected_manifest_sha256='0' * 64)
    plugin = output / 'plugins' / runtime.LIBRARY
    plugin.chmod(0o600)
    plugin.write_text('changed')
    with pytest.raises(ValueError, match='library or runner'):
        runtime.verify_deployment(output, expected_library_sha256=built.pin,
                                  expected_manifest_sha256=pin)


def test_real_link_path_aliases_need_exact_p109_view_identity(monkeypatch):
    offline_alias = OFFLINE + '../../Offline/lib/libmu2e_DataProducts.so'
    assert runtime._allowed_resolution('libmu2e_DataProducts.so', offline_alias)
    resolved = '/cvmfs/mu2e.opensciencegrid.org/spackages/package-hash/lib/libcetlib.so'
    monkeypatch.setattr(Path, 'is_file', lambda path: str(path) == VIEW + 'libcetlib.so')
    monkeypatch.setattr(runtime.os.path, 'samefile', lambda left, right:
                        str(left) == VIEW + 'libcetlib.so' and right == resolved)
    assert runtime._allowed_resolution('libcetlib.so', resolved)
    assert not runtime._allowed_resolution('libcetlib.so', resolved.replace('package-hash', 'other'))
    assert runtime._allowed_resolution('libfreetype.so.6', '/lib64/libfreetype.so.6')
    assert not runtime._allowed_resolution('libunknown.so.1', '/lib64/libunknown.so.1')


def test_overwrite_and_wrong_source_are_rejected(built):
    output, manifest = freeze(built)
    with pytest.raises(ValueError, match='overwrite'):
        runtime.prepare(RUNNER, output, source_module=built.source,
                        build_receipt=manifest['build_receipt'], expected_library_sha256=built.pin)
    other = built.tmp / 'other.cc'
    other.write_bytes(built.source.read_bytes())
    with pytest.raises(ValueError, match='source module'):
        runtime.prepare(RUNNER, built.tmp / 'other-bundle', source_module=other,
                        build_receipt=manifest['build_receipt'], expected_library_sha256=built.pin)


def test_postrun_requires_exact_worker_receipt_and_independent_expanded_hash(built):
    output, manifest = freeze(built)
    expanded = built.tmp / 'replay.expanded.fcl'
    expanded.write_text('independently admitted expanded FCL')
    log = built.tmp / 'replay.log'
    line = (f'feature_runtime library_sha256={built.pin} source_sha256={runtime.SOURCE_HASHES["src/KalSeedSelectionFeatures_module.cc"]} '
            f'source_set_sha256={runtime._source_set_sha(manifest["build_receipt"])} '
            f'stub={runtime.STUB} backing={runtime.BACKING} area={runtime.BACKING} '
            f'runner_sha256={manifest["adapted_runner_sha256"]} relocations=checked')
    log.write_text(line + '\n')
    options = dict(expected_library_sha256=built.pin,
                   expected_expanded_sha256=hashlib.sha256(expanded.read_bytes()).hexdigest())
    assert not runtime.validate_run(output, log, expanded, **options)['runtime_ready']
    for bad in ('', line + '\n' + line, line.replace('p109', 'p108')):
        log.write_text(bad)
        with pytest.raises(ValueError, match='receipt'):
            runtime.validate_run(output, log, expanded, **options)
    log.write_text(line + '\n')
    expanded.write_text('changed')
    with pytest.raises(ValueError, match='expanded FCL'):
        runtime.validate_run(output, log, expanded, **options)


def test_cli_capture_prepare_verify_and_no_overwrite(built, monkeypatch, capsys):
    receipt = built.tmp / 'build.json'
    common = ['--expected-library-sha256', built.pin]
    capture = ['feature_runtime', 'capture', '--area', str(built.area), '--user-build-report', str(built.report),
               '--output', str(receipt), *common]
    monkeypatch.setattr(sys, 'argv', capture)
    runtime.main()
    assert receipt.is_file()
    with pytest.raises(ValueError, match='overwrite'):
        runtime.main()
    output = built.tmp / 'cli-bundle'
    monkeypatch.setattr(sys, 'argv', ['feature_runtime', 'prepare', '--runner', str(RUNNER),
        '--source-module', str(built.source), '--build-receipt', str(receipt), '--output', str(output), *common])
    runtime.main()
    monkeypatch.setattr(sys, 'argv', ['feature_runtime', 'verify', '--bundle', str(output), *common])
    runtime.main()
    assert 'not reproducible-build' in capsys.readouterr().out


@pytest.mark.parametrize('change', ['none', 'missing', 'unresolved', 'wrong-offline', 'wrong-external',
                                    'stub', 'area', 'sha', 'extra-plugin'])
def test_actual_bash_worker_gate_without_art_or_compiler(built, change):
    output, manifest = freeze(built)
    ldd = built.evidence['ldd']
    stub, area, pin = runtime.STUB, runtime.BACKING, built.pin
    if change == 'missing':
        ldd = ldd.replace(VIEW + 'libcetlib.so (0x1234)', 'not found')
    elif change == 'unresolved':
        ldd += 'undefined symbol: broken\n'
    elif change == 'wrong-offline':
        ldd = ldd.replace('v13_41_01', 'v13_40_00')
    elif change == 'wrong-external':
        ldd = ldd.replace(VIEW, '/unadmitted/private/lib/')
    elif change == 'stub':
        stub = 'al9-debug-e29-p109'
    elif change == 'area':
        area = str(built.area)
    elif change == 'sha':
        pin = '0' * 64
    elif change == 'extra-plugin':
        (output / 'plugins/extra.so').write_text('other')
    script = built.tmp / 'gate.sh'
    script.write_text('set -eo pipefail\nfail() { echo "gate: $*" >&2; exit 2; }\n'
        + f'AREA={area}\nMUSE_WORK_DIR={runtime.BACKING}\nMUSE_STUB={stub}\nMUSE_ENVSET=p109\n'
        + f'OFFLINE_FEATURE_BUNDLE={output}\nOFFLINE_FEATURE_RUNNER_SHA256=runnerpin\n'
        + 'sha256sum() { case "$1" in *module.so) echo "' + pin + '  $1";; *) echo "runnerpin  $1";; esac; }\n'
        + 'readlink() { if [[ $1 == -m ]]; then echo "$2"; else command readlink "$@"; fi; }\n'
        + "ldd() { cat <<'LDD'\n" + ldd + 'LDD\n}\n'
        + runtime._worker_checks(manifest['build_receipt']) + '\n'
        + '[[ $CET_PLUGIN_PATH == "$OFFLINE_FEATURE_BUNDLE/plugins:"* ]]\n'
        + '[[ $LD_LIBRARY_PATH == "$OFFLINE_FEATURE_BUNDLE/plugins:"* ]]\n')
    result = REAL_RUN(['bash', str(script)], capture_output=True, text=True)
    if change == 'none':
        assert result.returncode == 0, result.stderr
        assert 'feature_runtime library_sha256=' in result.stdout
        assert 'source_set_sha256=' in result.stdout
    else:
        assert result.returncode != 0, result.stdout
        assert 'gate:' in result.stderr
