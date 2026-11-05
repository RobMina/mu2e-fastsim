"""Post-termination inventory tests; no execution/confinement/certificate claims."""

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import socket

import pytest

from beamfastsim import flash_reference_inventory as inventory


NAMES = {
    inventory.OutputLayout.REFERENCE: ("raw_1.art", "export.root", "job.fcl"),
    inventory.OutputLayout.NOG4_FIXTURE: (
        "flash_reference_fixture_1.art", "export.root", "job.fcl"),
}


@pytest.fixture
def work(tmp_path):
    # macOS pytest temp roots can have a symlink ancestor (/var -> /private/var).
    path = tmp_path.resolve() / "work"
    path.mkdir()
    for index, name in enumerate(NAMES[inventory.OutputLayout.REFERENCE]):
        (path / name).write_bytes(b"x" * index)
    return path


def rejected(work):
    with pytest.raises(inventory.InventoryError):
        inventory.validate_inventory(work)


@pytest.mark.parametrize("layout", list(inventory.OutputLayout))
def test_exact_fixed_layouts_and_metadata(work, layout):
    if layout is inventory.OutputLayout.NOG4_FIXTURE:
        (work / "raw_1.art").rename(work / "flash_reference_fixture_1.art")
    result = inventory.validate_inventory(work, layout)
    assert result.layout is layout
    assert result.work == str(work)
    assert result.verified is False
    assert result.total_bytes == 3
    assert result.empty_directories == ()
    assert tuple(item.name for item in result.files) == NAMES[layout]
    for item in result.files:
        info = (work / item.name).stat()
        assert item.byte_count == info.st_size
        assert item.identity == inventory.StatIdentity(
            info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_uid, info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    assert result.work_identity.inode == work.stat().st_ino
    with pytest.raises(FrozenInstanceError):
        result.verified = True
    with pytest.raises(FrozenInstanceError):
        result.files[0].name = "other"
    with pytest.raises(FrozenInstanceError):
        result.files[0].identity.size = 999
    assert isinstance(result.files, tuple)


def test_layout_matches_supervisor_sources():
    from beamfastsim import flash_reference_guarded_run as guarded
    for layout in inventory.OutputLayout:
        assert NAMES[layout] == tuple(source for source, _ in guarded._OUTPUTS[
            guarded.OutputLayout(layout.value)])


def test_all_three_zero_byte_files_are_structurally_allowed(work):
    for name in NAMES[inventory.OutputLayout.REFERENCE]:
        (work / name).write_bytes(b"")
    assert inventory.validate_inventory(work).total_bytes == 0


@pytest.mark.parametrize("name", NAMES[inventory.OutputLayout.REFERENCE])
def test_each_file_required(work, name):
    (work / name).unlink()
    rejected(work)


@pytest.mark.parametrize("name", [
    "raw_2.art", "raw_0.art", "raw_10.art", "export_1.root", "output.root",
    "flash_reference_fixture_1.art", "RootOutput-unique.tmp", ".hidden",
    "job.fcl.bak", "timing.txt", "merged.log", "raw_1.art.tmp.12345",
])
def test_no_extra_files_silently_omitted(work, name):
    (work / name).touch()
    rejected(work)
    assert (work / name).exists()  # No cleanup on failure.


@pytest.mark.parametrize("name", ["unexpected", ".hidden", "nested"])
def test_no_extra_directories_even_empty(work, name):
    (work / name).mkdir()
    rejected(work)


def test_many_extra_files_fail_closed(work):
    for index in range(200):
        (work / f"raw_{index + 2}.art").touch()
    rejected(work)
    assert len(list(work.iterdir())) == 203


@pytest.mark.parametrize("dirs", [(), ("tmp",), ("home", "shm"), ("tmp", "home", "shm")])
def test_only_declared_empty_scratch_directories(work, dirs):
    for name in dirs:
        (work / name).mkdir()
    before = {p.name: p.lstat() for p in work.iterdir()}
    result = inventory.validate_inventory(work)
    assert result.empty_directories == tuple(name for name in ("tmp", "home", "shm") if name in dirs)
    for path in work.iterdir():
        after = path.lstat()
        assert (after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
            before[path.name].st_size, before[path.name].st_mtime_ns, before[path.name].st_ctime_ns)


@pytest.mark.parametrize("scratch", ["tmp", "home", "shm"])
@pytest.mark.parametrize("kind", ["file", "hidden", "directory", "symlink", "fifo"])
def test_scratch_contents_are_never_ignored(work, scratch, kind):
    directory = work / scratch
    directory.mkdir()
    child = directory / (".sqlite-journal" if kind == "hidden" else "residual")
    if kind == "directory":
        child.mkdir()
        (child / "nested").touch()
    elif kind == "symlink":
        child.symlink_to(directory / "missing")
    elif kind == "fifo":
        os.mkfifo(child)
    else:
        child.touch()
    rejected(work)
    assert os.path.lexists(child)


@pytest.mark.parametrize("name", ["raw_1.art", "export.root", "job.fcl", "tmp", "home", "shm"])
@pytest.mark.parametrize("kind", ["symlink", "broken_symlink", "fifo", "wrong_type", "socket"])
def test_reject_broken_or_unsafe_expected_objects(work, name, kind, monkeypatch):
    path = work / name
    if path.exists():
        path.unlink()
    sock = None
    try:
        if kind == "symlink":
            path.symlink_to(work / "job.fcl" if name != "job.fcl" else work / "export.root")
        elif kind == "broken_symlink":
            path.symlink_to(work / "missing")
        elif kind == "fifo":
            os.mkfifo(path)
        elif kind == "socket":
            if not hasattr(socket, "AF_UNIX"):
                pytest.skip("Unix sockets unavailable")
            sock = socket.socket(socket.AF_UNIX)
            # AF_UNIX has a small sockaddr limit, especially with macOS temp roots.
            # Relative binding avoids a platform-dependent path-length failure.
            monkeypatch.chdir(work)
            sock.bind(name)
        elif name in ("tmp", "home", "shm"):
            path.touch()
        else:
            path.mkdir()
        rejected(work)
    finally:
        if sock is not None:
            sock.close()


@pytest.mark.parametrize("name", NAMES[inventory.OutputLayout.REFERENCE])
def test_hardlinks_rejected_even_when_other_link_outside_work(work, name):
    outside = work.parent / "other-link"
    os.link(work / name, outside)
    rejected(work)
    assert outside.stat().st_nlink == 2


def test_symlink_work_and_symlink_ancestor_rejected(work):
    alias = work.parent / "alias"
    alias.symlink_to(work, target_is_directory=True)
    rejected(alias)
    ancestor = work.parent / "ancestor-alias"
    ancestor.symlink_to(work.parent, target_is_directory=True)
    rejected(ancestor / work.name)


@pytest.mark.parametrize("bad", ["relative", "", b"/bytes", None, "/null\0path"])
def test_invalid_paths(bad):
    rejected(bad)


def test_missing_directory_regular_work_and_dot_components_rejected(work):
    rejected(work / "absent")
    rejected(work / "job.fcl")
    rejected(str(work) + "/../work")
    rejected(str(work) + "/.")


@pytest.mark.parametrize("layout", ["reference", "nog4_fixture", None, {}, ("raw_1.art",)])
def test_arbitrary_layouts_rejected(work, layout):
    with pytest.raises(inventory.InventoryError, match="fixed OutputLayout"):
        inventory.validate_inventory(work, layout)


def test_other_fixed_layout_not_interchangeable(work):
    with pytest.raises(inventory.InventoryError):
        inventory.validate_inventory(work, inventory.OutputLayout.NOG4_FIXTURE)


def test_metadata_measurement_does_not_read_file_contents(work, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not read contents")
    monkeypatch.setattr(os, "read", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert inventory.validate_inventory(work).total_bytes == 3


def test_observed_mutation_rejected(work, monkeypatch):
    original = inventory._names
    calls = 0

    def mutate(fd, allowed):
        nonlocal calls
        calls += 1
        if calls == 2:
            (work / "export.root").write_bytes(b"changed")
        return original(fd, allowed)

    monkeypatch.setattr(inventory, "_names", mutate)
    with pytest.raises(inventory.InventoryError, match="changed"):
        inventory.validate_inventory(work)


def test_descriptors_closed_on_success_and_failure(work, monkeypatch):
    original_open, original_close = os.open, os.close
    opened = set()

    def tracked_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.add(fd)
        return fd

    def tracked_close(fd):
        original_close(fd)
        opened.remove(fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)
    inventory.validate_inventory(work)
    assert not opened
    (work / "export.root").unlink()
    rejected(work)
    assert not opened


def test_filesystem_errors_are_fail_closed(work, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("denied")
    monkeypatch.setattr(os, "scandir", denied)
    with pytest.raises(inventory.InventoryError, match="cannot inventory"):
        inventory.validate_inventory(work)
