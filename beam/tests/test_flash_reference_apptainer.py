"""Portable ONE-Apptainer construction/refusal tests, not isolation evidence.

No Apptainer, CVMFS, mounts, namespaces, seccomp or payload is ever launched.
"""

from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from beamfastsim import flash_reference_apptainer as apptainer


# Model the pinned site file without reading or modifying the host installation.
_SITE_BYTES = (b"# trusted site configuration fixture\n"
               b"allow setuid = no\n"
               b"# mount slave = no\n"
               b"mount slave = yes\n"
               b"mount hostfs = no\n")
_PRIVATE_BYTES = _SITE_BYTES.replace(b"mount slave = yes\n", b"mount slave = no\n")


@pytest.fixture
def site_config(node, monkeypatch):
    path = node / "site-apptainer.conf"
    path.write_bytes(_SITE_BYTES)
    monkeypatch.setattr(apptainer, "_SITE_CONFIG", str(path))
    monkeypatch.setattr(apptainer, "_SITE_CONFIG_SHA256", hashlib.sha256(_SITE_BYTES).hexdigest())
    return path


@pytest.fixture
def node(tmp_path, monkeypatch):
    # macOS /tmp is a symlink: use a resolved, real directory as the node root.
    root = tmp_path.resolve()
    monkeypatch.setattr(apptainer, "_LOCAL_ROOTS", (str(root),))
    return root


@pytest.fixture
def layout(node, site_config):
    for root, children in (
        ("runtime", ("home", "tmp", "cache", "config", "session", "empty")),
        ("work", ("tmp", "home", "shm")),
        ("code", ()), ("area", ()),
    ):
        path = node / root
        path.mkdir(mode=0o700)
        for child in children:
            (path / child).mkdir(mode=0o700)
    for name in ("flash_reference_apptainer.py", "flash_reference_sandbox.py"):
        (node / "code" / name).write_text("# trusted fixture, never executed\n")
    executable = node / "apptainer"
    executable.write_text("not a real executable\n")
    return dict(apptainer=str(executable), runtime=str(node / "runtime"),
                work=str(node / "work"), code=str(node / "code"),
                area=str(node / "area"), command=["/usr/bin/true"])


def _values(argv, option):
    return [argv[index + 1] for index, value in enumerate(argv) if value == option]


def test_exact_single_pinned_container_and_explicit_bind_inventory(layout, node):
    input_file = node / "events.art"
    input_file.write_bytes(b"input")
    plan = apptainer.build_command(**layout, input_file=input_file, file_size_limit=99)
    argv = plan.argv
    image = ("/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/"
             "bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40")
    assert apptainer.PINNED_IMAGE == image
    assert argv.count(image) == 1
    assert argv[:20] == (
        layout["apptainer"], "--config", layout["runtime"] + "/config/apptainer.conf",
        "exec", "--userns", "--cleanenv", "--containall",
        "--no-home", "--no-mount", "bind-paths,hostfs,cwd,sys,devpts",
        "--drop-caps", "ALL", "--no-privs", "--no-eval", "--disable-cache",
        "--workdir", layout["runtime"] + "/session", "--cwd", "/mnt/work", "--bind",
    )
    outer = argv[:argv.index(image)]
    assert _values(outer, "--config") == [layout["runtime"] + "/config/apptainer.conf"]
    assert _values(outer, "--no-mount") == ["bind-paths,hostfs,cwd,sys,devpts"]
    assert _values(outer, "--bind") == [
        "/cvmfs:/cvmfs:ro",
        layout["code"] + ":/opt/r5/checks:ro",
        layout["area"] + ":/opt/r5/area:ro",
        layout["work"] + ":/mnt/work:rw",
        layout["work"] + "/tmp:/tmp:rw",
        layout["work"] + "/tmp:/var/tmp:rw",
        layout["work"] + "/shm:/dev/shm:rw",
        layout["runtime"] + "/empty:/run:ro",
        layout["runtime"] + "/empty:/var/run:ro",
        layout["runtime"] + "/empty:/sys:ro",
        str(input_file) + ":/opt/r5/input.art:ro",
    ]
    # Exact argv also excludes broad host mounts, writable runtime/cache, and
    # any nested bubblewrap/second-container entry around the trusted handoff.
    env = [
        "TMPDIR=/mnt/work/tmp", "TMP=/mnt/work/tmp",
        "TEMP=/mnt/work/tmp", "SQLITE_TMPDIR=/mnt/work/tmp",
        "XDG_CACHE_HOME=/mnt/work/tmp", "XDG_RUNTIME_DIR=/mnt/work/tmp",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    assert _values(outer, "--env") == env
    expected = list(argv[:19])
    for bind in _values(outer, "--bind"):
        expected.extend(("--bind", bind))
    for value in env:
        expected.extend(("--env", value))
    expected.extend((image, "/usr/bin/python3", "-I", "-B",
                     "/opt/r5/checks/flash_reference_apptainer.py", "--inside",
                     "--file-size-limit", "99", "--", "/usr/bin/true"))
    assert argv == tuple(expected)
    assert not any("bwrap" in arg or "bubblewrap" in arg for arg in argv)
    assert plan.configuration == _PRIVATE_BYTES
    assert plan.verified is False
    with pytest.raises(FrozenInstanceError):
        plan.configuration = b"unsafe override"
    with pytest.raises(FrozenInstanceError):
        plan.verified = True
    with pytest.raises(TypeError):
        apptainer.ApptainerCommand(argv=argv, environment={}, verified=True)


def test_configuration_is_returned_without_builder_writes(layout, site_config):
    config = Path(layout["runtime"]) / "config" / "apptainer.conf"
    plan = apptainer.build_command(**layout)
    assert isinstance(plan.configuration, bytes)
    assert plan.configuration == _PRIVATE_BYTES
    assert not config.exists()
    assert site_config.read_bytes() == _SITE_BYTES
    for root in ("runtime", "work"):
        assert all(not list(child.iterdir()) for child in Path(layout[root]).iterdir())
    # Construction is repeatable: only the caller may install the returned bytes.
    assert apptainer.build_command(**layout).configuration == plan.configuration
    config.write_bytes(plan.configuration)
    config.chmod(0o400)
    assert config.read_bytes() == _PRIVATE_BYTES
    assert stat.S_IMODE(config.stat().st_mode) == 0o400
    assert site_config.read_bytes() == _SITE_BYTES
    # A written configuration is no longer a fresh layout for a second plan.
    with pytest.raises(ValueError, match="empty"):
        apptainer.build_command(**layout)


def test_private_mount_configuration_changes_only_the_single_active_line(site_config):
    before = site_config.read_bytes()
    after = apptainer.private_mount_configuration(before)
    assert after == _PRIVATE_BYTES
    assert [(old, new) for old, new in zip(before.splitlines(keepends=True),
                                          after.splitlines(keepends=True)) if old != new] == [
        (b"mount slave = yes\n", b"mount slave = no\n")]
    assert site_config.read_bytes() == before


def test_wrong_expected_site_hash_refused(layout, monkeypatch):
    monkeypatch.setattr(apptainer, "_SITE_CONFIG_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="config identity changed"):
        apptainer.private_mount_configuration(_SITE_BYTES)
    with pytest.raises(ValueError, match="config identity changed"):
        apptainer.build_command(**layout)
    assert not list((Path(layout["runtime"]) / "config").iterdir())


@pytest.mark.parametrize("changed", [
    _SITE_BYTES + b"# otherwise harmless change\n",
    _PRIVATE_BYTES,
])
def test_changed_site_bytes_refused_without_repinning(layout, site_config, changed):
    site_config.write_bytes(changed)
    with pytest.raises(ValueError, match="config identity changed"):
        apptainer.private_mount_configuration(changed)
    with pytest.raises(ValueError, match="config identity changed"):
        apptainer.build_command(**layout)
    assert site_config.read_bytes() == changed
    assert not list((Path(layout["runtime"]) / "config").iterdir())


@pytest.mark.parametrize("value", [None, "mount slave = yes\n", bytearray(_SITE_BYTES)])
def test_nonbytes_site_configuration_refused(site_config, value):
    with pytest.raises(ValueError, match="config identity changed"):
        apptainer.private_mount_configuration(value)


@pytest.mark.parametrize("directive", [
    b"mount slave = yes\nmount slave = yes\n",
    b"mount slave = yes\nmount slave = no\n",
    b"",
    b"# mount slave = yes\n",
    b"mount slave = no\n",
    b"mount slave = maybe\n",
    b" mount slave = yes\n",
    b"mount slave=yes\n",
    b"mount slave = yes # comment\n",
    b"mount slave = yes\r\n",
    b"mount slave = yes",
], ids=["duplicate", "conflicting-duplicate", "missing", "comment-only", "already-no",
        "unexpected-value", "indented", "unexpected-spacing", "inline-comment", "crlf",
        "missing-newline"])
def test_pinned_but_unexpected_mount_slave_directive_refused(site_config, monkeypatch, directive):
    # Re-pin only for semantic tests so identity rejection cannot mask parsing.
    candidate = b"# modeled site config\nallow setuid = no\n" + directive
    monkeypatch.setattr(apptainer, "_SITE_CONFIG_SHA256", hashlib.sha256(candidate).hexdigest())
    with pytest.raises(ValueError, match="exactly one pinned mount slave directive"):
        apptainer.private_mount_configuration(candidate)


def test_optional_input_absent_and_payload_flags_cannot_be_container_options(layout):
    command = ("/usr/bin/printf", "--bind", "/:/host:rw", "--env", "HOME=/bad", "")
    plan = apptainer.build_command(**dict(layout, command=command))
    assert plan.argv[-len(command):] == command
    assert plan.argv[-len(command) - 1] == "--"
    outer = plan.argv[:plan.argv.index(apptainer.PINNED_IMAGE)]
    assert len(_values(outer, "--bind")) == 10
    assert not any("/opt/r5/input.art" in arg for arg in outer)
    assert plan.argv[plan.argv.index("--file-size-limit") + 1] == str(1024**3)
    assert plan.verified is False


def test_launcher_environment_is_fresh_exact_allowlist(layout, monkeypatch):
    poison = {
        "APPTAINER_BIND": "/:/host:rw", "APPTAINER_BINDPATH": "/secret",
        "APPTAINERENV_HOME": "/bad", "APPTAINER_TMPDIR": "/bad",
        "APPTAINER_CACHEDIR": "/bad", "APPTAINER_CONFIGDIR": "/bad",
        "SINGULARITY_BIND": "/:/host", "SINGULARITYENV_LD_PRELOAD": "/evil.so",
        "LD_PRELOAD": "/evil.so", "LD_LIBRARY_PATH": "/evil",
        "PYTHONPATH": "/evil", "BASH_ENV": "/evil", "SECRET_TOKEN": "secret",
        "PATH": "/evil", "HOME": "/bad", "TMPDIR": "/bad",
    }
    for key, value in poison.items():
        monkeypatch.setenv(key, value)
    runtime = layout["runtime"]
    expected = {
        "PATH": str(Path(layout["apptainer"]).parent) + ":/usr/bin:/bin",
        "HOME": runtime + "/home", "LANG": "C", "LC_ALL": "C",
        "TMPDIR": runtime + "/tmp", "APPTAINER_TMPDIR": runtime + "/tmp",
        "APPTAINER_CACHEDIR": runtime + "/cache",
        "APPTAINER_CONFIGDIR": runtime + "/config",
    }
    plan = apptainer.build_command(**layout)
    assert plan.environment == expected
    assert apptainer.launcher_environment(runtime, layout["apptainer"]) == expected
    assert all(os.environ[key] == value for key, value in poison.items())
    plan.environment["SECRET_TOKEN"] = "modified"
    assert apptainer.build_command(**layout).environment == expected


@pytest.mark.parametrize("field", ["apptainer", "runtime", "work", "code", "area", "input_file"])
@pytest.mark.parametrize("bad", ["relative", "", b"/bytes", "/", "/tmp/../escape",
                                  "/tmp/./escape", "/tmp//escape", "/tmp/trailing/",
                                  "/tmp/a:b", "/tmp/a,b", "/tmp/a\0b",
                                  "/tmp/a\nb", "/tmp/a\rb"])
def test_noncanonical_sources_fail_closed(layout, field, bad):
    with pytest.raises((ValueError, TypeError)):
        apptainer.build_command(**dict(layout, **{field: bad}))


@pytest.mark.parametrize("field", ["apptainer", "runtime", "work", "code", "area", "input_file"])
@pytest.mark.parametrize("ancestor", [False, True])
def test_symlink_sources_and_ancestors_refused(layout, node, field, ancestor):
    source = Path(layout.get(field, str(node / "events.art")))
    if field == "input_file":
        source.write_bytes(b"input")
    alias = node / "alias"
    if ancestor:
        alias.symlink_to(node, target_is_directory=True)
        value = alias / source.name
    else:
        alias.symlink_to(source, target_is_directory=source.is_dir())
        value = alias
    with pytest.raises(ValueError, match="symlinks"):
        apptainer.build_command(**dict(layout, **{field: value}))


@pytest.mark.parametrize("field", ["apptainer", "runtime", "work", "code", "area", "input_file"])
def test_wrong_file_type_and_missing_source_refused(layout, node, field):
    directory_expected = field in ("runtime", "work", "code", "area")
    wrong = node / "wrong"
    if directory_expected:
        wrong.write_text("not a directory")
    else:
        wrong.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="file type"):
        apptainer.build_command(**dict(layout, **{field: wrong}))
    with pytest.raises(OSError):
        apptainer.build_command(**dict(layout, **{field: node / "missing"}))


@pytest.mark.parametrize("field", ["runtime", "work"])
@pytest.mark.parametrize("mode", [0o755, 0o770])
def test_private_roots_require_exact_mode(layout, field, mode):
    os.chmod(layout[field], mode)
    with pytest.raises(ValueError, match="0700"):
        apptainer.build_command(**layout)


def test_private_directory_requires_current_owner(layout, monkeypatch):
    monkeypatch.setattr(apptainer.os, "geteuid", lambda: os.stat(layout["runtime"]).st_uid + 1)
    with pytest.raises(ValueError, match="uid"):
        apptainer.build_command(**layout)


@pytest.mark.parametrize("field", ["runtime", "work"])
def test_node_root_itself_and_prefix_lookalike_are_not_local(layout, node, monkeypatch, field):
    # Existing private directories still require a slash-delimited root prefix.
    monkeypatch.setattr(apptainer, "_LOCAL_ROOTS", (layout[field],))
    with pytest.raises(ValueError, match="node-local"):
        apptainer.build_command(**layout)
    monkeypatch.setattr(apptainer, "_LOCAL_ROOTS", (str(node / field[:-1]),))
    with pytest.raises(ValueError, match="node-local"):
        apptainer.build_command(**layout)


@pytest.mark.parametrize("left,right", [
    ("runtime", "work"), ("runtime", "code"), ("runtime", "area"),
    ("work", "code"), ("work", "area"), ("code", "area"),
])
@pytest.mark.parametrize("relation", ["equal", "left_contains", "right_contains"])
def test_directory_bind_overlap_refused(layout, left, right, relation):
    changed = dict(layout)
    if relation == "equal":
        changed[right] = changed[left]
    else:
        parent, child = (left, right) if relation == "left_contains" else (right, left)
        nested = Path(changed[parent]) / "nested"
        nested.mkdir(mode=0o700)
        changed[child] = str(nested)
    with pytest.raises(ValueError, match="overlap"):
        apptainer.build_command(**changed)


@pytest.mark.parametrize("root", ["runtime", "work"])
@pytest.mark.parametrize("fault", ["missing", "extra", "nonempty", "mode", "symlink", "file"])
def test_fresh_private_layout_is_exact(layout, node, root, fault):
    parent = Path(layout[root])
    child = parent / "home"
    if fault == "missing":
        child.rmdir()
    elif fault == "extra":
        (parent / "extra").mkdir(mode=0o700)
    elif fault == "nonempty":
        (child / "stale").write_text("old data")
    elif fault == "mode":
        child.chmod(0o755)
    elif fault == "symlink":
        child.rmdir()
        child.symlink_to(parent / "tmp", target_is_directory=True)
    else:
        child.rmdir()
        child.write_text("not a directory")
    with pytest.raises(ValueError):
        apptainer.build_command(**layout)


@pytest.mark.parametrize("name", ["flash_reference_apptainer.py", "flash_reference_sandbox.py"])
@pytest.mark.parametrize("fault", ["missing", "directory", "symlink"])
def test_required_trusted_modules_must_be_regular_files(layout, node, name, fault):
    path = Path(layout["code"]) / name
    path.unlink()
    if fault == "directory":
        path.mkdir()
    elif fault == "symlink":
        path.symlink_to(node / "apptainer")
    with pytest.raises((ValueError, OSError)):
        apptainer.build_command(**layout)


@pytest.mark.parametrize("root", ["runtime", "work", "code", "area"])
def test_input_cannot_be_inside_any_directory_bind(layout, root):
    # code already has a regular file; other roots need a file in a child to
    # preserve the exact top-level layout until the overlap check is reached.
    if root in ("runtime", "work"):
        path = Path(layout[root]) / "home" / "input.art"
        path.write_bytes(b"input")
        # Fresh-layout rejection is already sufficient for these writable roots.
        with pytest.raises(ValueError, match="empty"):
            apptainer.build_command(**layout, input_file=path)
    else:
        path = Path(layout[root]) / "input.art"
        path.write_bytes(b"input")
        with pytest.raises(ValueError, match="input must be separate"):
            apptainer.build_command(**layout, input_file=path)


@pytest.mark.parametrize("command", [[], (), "true", None, [""], ["relative"], ["--bind"],
                                     ["../escape"], ["/bin/../bin/true"], ["/"],
                                     [b"/bin/true"], ["/bin/true", b"bytes"],
                                     ["/bin/true", None], ["/bin/true", "bad\0arg"]])
def test_invalid_payload_argv_refused(layout, command):
    with pytest.raises(ValueError):
        apptainer.build_command(**dict(layout, command=command))


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "123", None, 2**63, 2**64])
def test_invalid_file_limit_refused(layout, value):
    with pytest.raises(ValueError, match="positive signed 64-bit"):
        apptainer.build_command(**layout, file_size_limit=value)


@pytest.mark.parametrize("value", [1, 2**63 - 1])
def test_file_limit_boundaries_preserved(layout, value):
    plan = apptainer.build_command(**layout, file_size_limit=value)
    assert _values(plan.argv, "--file-size-limit") == [str(value)]
    assert plan.verified is False


@pytest.mark.parametrize("option", ["options", "binds", "image", "verified", "no_mount"])
def test_no_arbitrary_container_policy_overrides(layout, option):
    with pytest.raises(TypeError):
        apptainer.build_command(**layout, **{option: "unsafe override"})


@pytest.fixture
def safe_handoff(monkeypatch):
    """Model layout only; trip before even importing the security implementation."""
    def forbidden(*args, **kwargs):
        pytest.fail("refused handoff must not reach security setup or exec")

    monkeypatch.setattr(importlib.util, "spec_from_file_location", forbidden)
    monkeypatch.setattr(apptainer.os, "execvpe", forbidden)
    monkeypatch.setattr(apptainer.os, "close", forbidden)
    monkeypatch.setattr(apptainer.sys, "platform", "linux")
    monkeypatch.setattr(apptainer.os, "getcwd", lambda: "/mnt/work")
    monkeypatch.setenv("HOME", "/apptainer-selected-home")
    monkeypatch.setenv("TMPDIR", "/mnt/work/tmp")
    monkeypatch.setattr(apptainer.os, "lstat", lambda value: SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o700, st_uid=os.geteuid()))
    # Avoid resolving /opt on the host: this is only a modeled guard input.
    monkeypatch.setattr(apptainer, "Path", lambda value: SimpleNamespace(
        resolve=lambda: "/opt/r5/checks/flash_reference_apptainer.py"))


@pytest.mark.parametrize("guard", ["platform", "cwd", "tmp", "module"])
def test_inside_refuses_each_noncontainer_layout_guard(safe_handoff, monkeypatch, guard):
    if guard == "platform":
        monkeypatch.setattr(apptainer.sys, "platform", "darwin")
    elif guard == "cwd":
        monkeypatch.setattr(apptainer.os, "getcwd", lambda: "/")
    elif guard == "tmp":
        monkeypatch.setenv("TMPDIR", "/tmp")
    else:
        monkeypatch.setattr(apptainer, "Path", lambda value: SimpleNamespace(
            resolve=lambda: "/host/flash_reference_apptainer.py"))
    with pytest.raises(RuntimeError, match="fixed container layout"):
        apptainer._inside(["/usr/bin/true"], 42)


@pytest.mark.parametrize("limit", [True, False, 0, -1, None, "1", 1.5, 2**63])
def test_inside_invalid_limits_refused_before_security(safe_handoff, limit):
    with pytest.raises(ValueError, match="invalid payload or file limit"):
        apptainer._inside(["/usr/bin/true"], limit)


@pytest.mark.parametrize("command", [[], ["relative"], ["/bin/../bin/true"], ["/bin/a\0b"]])
def test_inside_invalid_command_refused_before_security(safe_handoff, command):
    with pytest.raises(ValueError):
        apptainer._inside(command, 42)


def test_cli_inside_refusal_is_explicit_failure(safe_handoff, monkeypatch, capsys):
    monkeypatch.setattr(apptainer.sys, "platform", "darwin")
    with pytest.raises(SystemExit) as exc:
        apptainer.main(["--inside", "--file-size-limit", "42", "--", "/usr/bin/true"])
    assert exc.value.code == 125
    assert "Apptainer handoff refused: internal handoff requires the fixed container layout" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["--file-size-limit", "1", "--", "/bin/true"],
                                  ["--inside"], ["--inside", "--file-size-limit", "bad"]])
def test_cli_requires_internal_mode_and_integer_limit(safe_handoff, argv):
    with pytest.raises(SystemExit) as exc:
        apptainer.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("mode,uid", [(stat.S_IFLNK | 0o700, None),
                                      (stat.S_IFREG | 0o700, None),
                                      (stat.S_IFDIR | 0o755, None),
                                      (stat.S_IFDIR | 0o700, -1)])
def test_inside_private_home_must_be_real_owned_private_directory(
        safe_handoff, monkeypatch, mode, uid):
    monkeypatch.setattr(apptainer.os, "lstat", lambda value: SimpleNamespace(
        st_mode=mode, st_uid=os.geteuid() if uid is None else uid))
    with pytest.raises(RuntimeError, match="private home directory"):
        apptainer._inside(["/usr/bin/true"], 42)


def test_home_is_set_by_trusted_handoff_not_rejected_container_env_option(safe_handoff, monkeypatch):
    observed = []
    def stop_before_security(*args, **kwargs):
        observed.append(os.environ["HOME"])
        raise RuntimeError("stop before any security operations")
    monkeypatch.setattr(importlib.util, "spec_from_file_location", stop_before_security)
    with pytest.raises(RuntimeError, match="stop before any security"):
        apptainer._inside(["/usr/bin/true"], 42)
    assert observed == ["/mnt/work/home"]


def test_missing_private_home_fails_closed(safe_handoff, monkeypatch, capsys):
    def missing(value):
        raise FileNotFoundError("missing private home")
    monkeypatch.setattr(apptainer.os, "lstat", missing)
    with pytest.raises(SystemExit) as exc:
        apptainer.main(["--inside", "--file-size-limit", "42", "--", "/usr/bin/true"])
    assert exc.value.code == 125
    assert "missing private home" in capsys.readouterr().err
