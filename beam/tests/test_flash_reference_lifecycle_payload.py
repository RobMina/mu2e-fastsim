"""Portable input/schema/static checks only: NEVER fork or run a Linux payload.

These tests deliberately provide no containment or descendant-lifecycle evidence.
No host directories, subprocesses, timers, signals or security policy are changed.
"""

import ast
import copy
import inspect
import json

import pytest

from beamfastsim import flash_reference_lifecycle_payload as payload


def security_text(**changes):
    fields = {"NoNewPrivs": "1", "CapEff": "0000000000000000",
              "CapPrm": "0000000000000000", "CapInh": "0000000000000000",
              "CapAmb": "0000000000000000", "Seccomp": "2"}
    fields.update(changes)
    return "Name:\tpython\n" + "\n".join(key + ":\t" + value for key, value in fields.items())


def manifest_arguments(mode="normal", nonce="external_nonce-123"):
    def process(pid, ppid, sid, pgid):
        return {"pid": pid, "ppid": ppid, "sid": sid, "pgid": pgid,
                "pid_namespace": "pid:[200]"}
    return {"mode": mode, "nonce": nonce, "outer_pid_namespace": "pid:[100]",
            "security": payload.check_security_status(security_text()),
            "primary_mode_properties": {"sigterm_ignored": mode in ("linger", "closed-stdio"),
                                        "stdio_devnull_checked": mode == "closed-stdio"},
            "processes": {"parent": process(5, 1, 5, 5),
                          "session_child": process(6, 5, 6, 6),
                          "orphan": process(7, 1, 6, 6)}}


@pytest.mark.parametrize("mode", payload.MODES)
def test_modes_and_maximum_bounded_arguments(mode):
    payload.validate_inputs(mode, "n" * 64, "pid:[18446744073709551615]")


@pytest.mark.parametrize("mode,nonce,namespace", [
    ("unknown", "n", "pid:[1]"), (None, "n", "pid:[1]"),
    ("normal", "", "pid:[1]"), ("normal", "x" * 65, "pid:[1]"),
    ("normal", "../../escape", "pid:[1]"), ("normal", "a\nb", "pid:[1]"),
    ("normal", "a\x00b", "pid:[1]"), ("normal", "é", "pid:[1]"),
    ("normal", b"n", "pid:[1]"), ("normal", "n", "pid:[0]"),
    ("normal", "n", "pid:[01]"), ("normal", "n", "pid:[" + "1" * 21 + "]"),
    ("normal", "n", "/proc/1/ns/pid"), ("normal", "n", "pid:[1]\n"),
    ("normal", "n", None),
])
def test_rejects_invalid_inputs_without_probing(mode, nonce, namespace):
    with pytest.raises(ValueError):
        payload.validate_inputs(mode, nonce, namespace)


@pytest.mark.parametrize("argv", [[], ["--mode", "normal"],
    ["--mode", "normal", "--nonce", "n", "--outer-pid-namespace", "pid:[1]"],
    ["--enable-lifecycle-payload", "--mode", "unknown", "--nonce", "n",
     "--outer-pid-namespace", "pid:[1]"],
    ["--enable-lifecycle-payload", "--mode", "normal", "--nonce", "n",
     "--outer-pid-namespace", "pid:[1]", "--expiry", "300"],
])
def test_cli_requires_explicit_complete_opt_in_and_has_no_expiry_override(argv):
    # All invocations fail during argument parsing; run() is never entered.
    with pytest.raises(SystemExit) as exc:
        payload.main(argv)
    assert exc.value.code == 2


def test_cli_invalid_nonce_refused_before_linux_entry():
    with pytest.raises(SystemExit) as exc:
        payload.main(["--enable-lifecycle-payload", "--mode", "normal", "--nonce", "../bad",
                      "--outer-pid-namespace", "pid:[1]"])
    assert exc.value.code == 125


@pytest.mark.parametrize("field,value", [("NoNewPrivs", "0"), ("CapEff", "1"),
    ("CapPrm", "1"), ("CapInh", "1"), ("CapAmb", "1"),
    ("Seccomp", "0"), ("Seccomp", "1"), ("Seccomp", "garbage")])
def test_security_snapshot_fails_closed(field, value):
    with pytest.raises(ValueError):
        payload.check_security_status(security_text(**{field: value}))


def test_missing_duplicate_or_oversized_security_status_refused():
    for text in ("", security_text() + "\nSeccomp: 2", "x" * 16385,
                 security_text().replace("NoNewPrivs:\t1", "NotNNP:\t1")):
        with pytest.raises(ValueError):
            payload.check_security_status(text)


@pytest.mark.parametrize("mode", payload.MODES)
def test_ready_schema_is_bounded_and_not_a_certificate(mode):
    arguments = manifest_arguments(mode, "x" * 64)
    record = payload.make_ready_manifest(**arguments)
    encoded = payload.encode_record(record)
    assert len(encoded) <= 4096
    assert json.loads(encoded) == record
    assert record["schema_version"] == 1
    assert record["nonce"] == "x" * 64
    assert record["mode"] == mode
    assert record["pid"] == record["processes"]["orphan"]["pid"]
    assert record["ppid"] == 1
    assert record["pid_namespace"] != record["outer_pid_namespace"]
    assert record["emergency_expiry_seconds"] == 30
    assert record["heartbeat_max_bytes"] == 4096
    assert record["orphan_ignores_sigterm"] is True
    assert record["intermediate_reaped"] is True
    assert record["primary_mode_properties"] == arguments["primary_mode_properties"]
    assert record["verified"] is False
    assert record["transport_events"] == 0


@pytest.mark.parametrize("mutation", ["same_namespace", "mixed_namespace", "not_orphan",
                                     "not_session", "missing_role", "extra_process_field",
                                     "negative_pid", "boolean_pid", "wrong_security"])
def test_ready_schema_rejects_incomplete_or_inconsistent_snapshots(mutation):
    arguments = copy.deepcopy(manifest_arguments())
    processes = arguments["processes"]
    if mutation == "same_namespace":
        arguments["outer_pid_namespace"] = "pid:[200]"
    elif mutation == "mixed_namespace":
        processes["parent"]["pid_namespace"] = "pid:[300]"
    elif mutation == "not_orphan":
        processes["orphan"]["ppid"] = 6
    elif mutation == "not_session":
        processes["session_child"]["sid"] = 5
    elif mutation == "missing_role":
        del processes["parent"]
    elif mutation == "extra_process_field":
        processes["orphan"]["unexpected"] = "value"
    elif mutation == "negative_pid":
        processes["orphan"]["pid"] = -1
    elif mutation == "boolean_pid":
        processes["orphan"]["pid"] = True
    elif mutation == "wrong_security":
        arguments["security"]["NoNewPrivs"] = 0
    with pytest.raises(ValueError):
        payload.make_ready_manifest(**arguments)


def test_heartbeat_total_statically_bounded_even_at_maximum_integer_width():
    lines = [payload.encode_record({"sequence": sequence, "pid": 2**31 - 1,
                                    "monotonic_ns": 2**64 - 1})
             for sequence in range(payload.MAX_HEARTBEATS)]
    assert len(b"".join(lines)) <= payload.MAX_FILE_BYTES <= 4096
    assert payload.MAX_HEARTBEATS == 30
    with pytest.raises(ValueError):
        payload.encode_record({"unbounded": "x" * 4096})
    with pytest.raises(ValueError):
        payload.encode_record({"invalid": float("nan")})


def test_static_lifecycle_contract_without_running_any_linux_operation():
    source = inspect.getsource(payload)
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    fork_calls = [node for node in calls if isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name) and node.func.value.id == "os"
                  and node.func.attr == "fork"]
    assert len(fork_calls) == 2
    assert payload.MODES == ("normal", "nonzero", "crash", "linger", "closed-stdio")
    assert payload.EXPIRY_SECONDS == 30
    assert payload.READY_TIMEOUT_SECONDS < payload.EXPIRY_SECONDS
    assert payload.WORK == "/mnt/work"
    assert payload.READY_NAME == "lifecycle-ready.json"
    assert payload.HEARTBEAT_NAME == "lifecycle-heartbeat.jsonl"
    for contract in ('os.setsid()', 'signal.SIGTERM, signal.SIG_IGN',
                     'signal.SIGABRT, signal.SIG_DFL', 'os.kill(os.getpid(), signal.SIGABRT)',
                     'resource.setrlimit(resource.RLIMIT_CORE, (0, 0))',
                     'signal.setitimer(signal.ITIMER_REAL, remaining)',
                     'os.waitpid(child, os.WNOHANG)', 'select.select([read_fd]',
                     'os.O_EXCL | os.O_NOFOLLOW', '_write_bounded(ready_fd, encode_record(processes), 0)',
                     'time.monotonic()', 'return 7 if mode == "nonzero" else 0',
                     'os.dup2(fd, destination, inheritable=True)'):
        assert contract in source
    assert source.count('_arm_expiry(deadline)') == 4  # definition, parent and both forks
    for forbidden in ("subprocess", "ctypes", "os.exec", "os.system", "unshare(",
                      "setns(", "prctl(", "generallauncher", "import resource\nresource"):
        assert forbidden not in source
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert [node.module for node in imports if isinstance(node, ast.ImportFrom)] == ["pathlib"]
    # Import has definitions/constants only; no top-level Linux execution.
    assert not any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   for node in tree.body)


def test_static_readiness_order_and_inheritance():
    run = inspect.getsource(payload.run)
    assert (run.index('validate_inputs(') < run.index('_validate_linux()')
            < run.index('_arm_expiry(deadline)') < run.index('_check_environment('))
    first_fork = run.index('os.fork()')
    assert run.index('signal.SIGTERM, signal.SIG_IGN') < first_fork
    assert run.index('_close_stdio()') < run.index('_check_primary_mode_properties(mode)') < first_fork
    assert run.count('_close_stdio()') == 1
    assert run.index('_read_ready(') < run.index('os.waitpid(')
    assert run.index('raise RuntimeError("intermediate lifecycle child failed")') < run.index('make_ready_manifest(')
    assert run.index('raise RuntimeError("bounded intermediate reap failed")') < run.index('make_ready_manifest(')
    assert run.index('make_ready_manifest(') < run.index('_open_output(directory_fd, READY_NAME)')
    orphan = inspect.getsource(payload._orphan)
    assert orphan.index('while os.getppid()') < orphan.index('_snapshot()')
    assert orphan.index('HEARTBEAT_NAME') < orphan.index('_write_bounded(ready_fd')
    assert '_close_stdio()' not in orphan
    assert '_open_output(directory_fd, READY_NAME)' not in orphan
    assert 'outer watchdog' in payload.__doc__
    assert 'NOT containment' in payload.__doc__


@pytest.mark.parametrize('mode', payload.MODES)
@pytest.mark.parametrize('field', ['sigterm_ignored', 'stdio_devnull_checked'])
def test_manifest_refuses_unchecked_primary_properties(mode, field):
    arguments = manifest_arguments(mode)
    arguments['primary_mode_properties'][field] = not arguments['primary_mode_properties'][field]
    with pytest.raises(ValueError, match='primary mode properties'):
        payload.make_ready_manifest(**arguments)


def test_manifest_rejects_boolean_impostors():
    arguments = manifest_arguments('closed-stdio')
    arguments['primary_mode_properties']['sigterm_ignored'] = 1
    with pytest.raises(ValueError, match='primary mode properties'):
        payload.make_ready_manifest(**arguments)


@pytest.mark.parametrize('mode', payload.MODES)
def test_primary_properties_checked_with_only_mocked_probes(monkeypatch, mode):
    from types import SimpleNamespace
    calls = []
    expected = manifest_arguments(mode)['primary_mode_properties']
    disposition = payload.signal.SIG_IGN if expected['sigterm_ignored'] else payload.signal.SIG_DFL
    monkeypatch.setattr(payload, 'signal', SimpleNamespace(
        SIGTERM=15, SIG_IGN=1, SIG_DFL=0, getsignal=lambda _: disposition))
    def stat_null(path):
        calls.append(path)
        return SimpleNamespace(st_mode=0o20666, st_rdev=123)
    def fstat(fd):
        calls.append(fd)
        return SimpleNamespace(st_mode=0o20666, st_rdev=123)
    monkeypatch.setattr(payload, 'os', SimpleNamespace(stat=stat_null, fstat=fstat))
    assert payload._check_primary_mode_properties(mode) == expected
    assert calls == (['/dev/null', 0, 1, 2] if mode == 'closed-stdio' else [])


@pytest.mark.parametrize('fault', ['term', 'regular_null', 'regular_stdio', 'wrong_device'])
def test_primary_properties_fail_closed_with_mocked_probes(monkeypatch, fault):
    from types import SimpleNamespace
    monkeypatch.setattr(payload, 'signal', SimpleNamespace(
        SIGTERM=15, SIG_IGN=1, SIG_DFL=0,
        getsignal=lambda _: 0 if fault == 'term' else 1))
    monkeypatch.setattr(payload, 'os', SimpleNamespace(
        stat=lambda _: SimpleNamespace(st_mode=0o100600 if fault == 'regular_null' else 0o20666,
                                       st_rdev=123),
        fstat=lambda _: SimpleNamespace(st_mode=0o100600 if fault == 'regular_stdio' else 0o20666,
                                        st_rdev=124 if fault == 'wrong_device' else 123)))
    with pytest.raises(RuntimeError):
        payload._check_primary_mode_properties('closed-stdio')


@pytest.mark.parametrize('chunks,readable,accepted', [
    ([b'{"ok":', b'true}\n', b''], True, True),
    ([b''], True, False), ([b'{}', b''], True, False),
    ([b'x' * 4097], True, False), ([b'no-json\n', b''], True, False),
    ([], False, False),
])
def test_private_snapshot_pipe_is_bounded_and_fails_closed(monkeypatch, chunks, readable, accepted):
    from types import SimpleNamespace
    iterator = iter(chunks)
    monkeypatch.setattr(payload, 'os', SimpleNamespace(read=lambda *_: next(iterator)))
    monkeypatch.setattr(payload, 'select', SimpleNamespace(
        select=lambda *_: ([12] if readable else [], [], [])))
    monkeypatch.setattr(payload, 'time', SimpleNamespace(monotonic=lambda: 1))
    if accepted:
        assert payload._read_ready(12, 2) == {'ok': True}
    else:
        with pytest.raises((RuntimeError, ValueError)):
            payload._read_ready(12, 2)


@pytest.fixture
def mocked_primary(monkeypatch):
    """All runtime effects replaced; even resource and os are private fakes."""
    from types import SimpleNamespace
    import sys
    events = []
    snapshots = manifest_arguments()['processes']
    def event(name, result=None):
        def call(*args, **kwargs):
            events.append((name, args))
            return result
        return call
    fake_os = SimpleNamespace(
        umask=event('umask'), open=event('directory', 10), pipe=event('pipe', (11, 12)),
        fork=event('fork', 6), close=event('close'), waitpid=event('reap', (6, 0)),
        O_RDONLY=0, O_DIRECTORY=0, O_NOFOLLOW=0, O_CLOEXEC=0, WNOHANG=1)
    monkeypatch.setattr(payload, 'os', fake_os)
    monkeypatch.setattr(payload, 'signal', SimpleNamespace(
        SIGTERM=15, SIG_IGN=1, SIG_DFL=0, signal=event('signal')))
    monkeypatch.setattr(payload, 'time', SimpleNamespace(monotonic=lambda: 1))
    monkeypatch.setitem(sys.modules, 'resource', SimpleNamespace(RLIMIT_CORE=0, setrlimit=event('core')))
    monkeypatch.setattr(payload, '_validate_linux', event('linux'))
    monkeypatch.setattr(payload, '_arm_expiry', event('arm'))
    monkeypatch.setattr(payload, '_check_environment', event('environment', manifest_arguments()['security']))
    monkeypatch.setattr(payload, '_close_stdio', event('stdio'))
    def check(mode):
        events.append(('properties', (mode,)))
        return manifest_arguments(mode)['primary_mode_properties']
    monkeypatch.setattr(payload, '_check_primary_mode_properties', check)
    monkeypatch.setattr(payload, '_snapshot', event('snapshot', snapshots['parent']))
    monkeypatch.setattr(payload, '_read_ready', event('read', snapshots))
    monkeypatch.setattr(payload, '_open_output', event('publish', 13))
    monkeypatch.setattr(payload, '_write_bounded', event('write'))
    monkeypatch.setattr(payload, '_sleep_until', event('sleep'))
    return events, fake_os


@pytest.mark.parametrize('mode', ['normal', 'nonzero', 'linger', 'closed-stdio'])
def test_mocked_primary_publishes_only_after_modes_snapshot_and_reap(mocked_primary, mode):
    events, _ = mocked_primary
    result = payload.run(mode=mode, nonce='n', outer_pid_namespace='pid:[100]')
    names = [name for name, _ in events]
    assert names.index('linux') < names.index('arm') < names.index('environment')
    assert names.index('properties') < names.index('fork') < names.index('read') < names.index('reap') < names.index('publish')
    if mode == 'closed-stdio':
        assert names.index('stdio') < names.index('properties')
    else:
        assert 'stdio' not in names
    ignored = ('signal', (15, 1))
    if mode in ('linger', 'closed-stdio'):
        assert events.index(ignored) < names.index('fork')
    else:
        assert ignored not in events
    data = next(args[1] for name, args in events if name == 'write')
    record = json.loads(data)
    assert record['intermediate_reaped'] is True
    assert record['primary_mode_properties'] == manifest_arguments(mode)['primary_mode_properties']
    assert result == (124 if mode in ('linger', 'closed-stdio') else 7 if mode == 'nonzero' else 0)


@pytest.mark.parametrize('failure', ['read', 'reap', 'properties'])
def test_mocked_primary_never_publishes_on_pre_ready_failure(monkeypatch, mocked_primary, failure):
    events, fake_os = mocked_primary
    def fail(*_):
        raise RuntimeError('injected failure')
    if failure == 'reap':
        fake_os.waitpid = lambda *_: (6, 256)
    else:
        monkeypatch.setattr(payload, '_read_ready' if failure == 'read' else '_check_primary_mode_properties', fail)
    with pytest.raises(RuntimeError):
        payload.run(mode='closed-stdio', nonce='n', outer_pid_namespace='pid:[100]')
    assert 'publish' not in [name for name, _ in events]


def test_invalid_inputs_and_nonlinux_refused_before_timer_or_probes(monkeypatch):
    from types import SimpleNamespace
    def forbidden(*_):
        pytest.fail('unexpected runtime effect')
    monkeypatch.setattr(payload, '_arm_expiry', forbidden)
    monkeypatch.setattr(payload, '_check_environment', forbidden)
    monkeypatch.setattr(payload, 'time', SimpleNamespace(monotonic=forbidden))
    monkeypatch.setattr(payload, 'sys', SimpleNamespace(platform='not-linux'))
    with pytest.raises(ValueError):
        payload.run(mode='normal', nonce='../bad', outer_pid_namespace='pid:[100]')
    with pytest.raises(RuntimeError, match='requires Linux'):
        payload.run(mode='normal', nonce='n', outer_pid_namespace='pid:[100]')
