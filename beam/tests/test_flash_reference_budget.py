"""Adversarial tests for the saved-content primitive, not payload confinement."""

from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
import threading

import pytest

from beamfastsim import flash_reference_budget as budget


VERIFIED = {"confinement_verified": True, "payload_terminated": True}


def make_writer(tmp_path, **kwargs):
    kwargs.setdefault("byte_limit", 2048)
    kwargs.setdefault("status_reserve", 128)
    return budget.SerializedBudget(tmp_path, budget.SLOT_NAMES[0], **kwargs)


def saved_on_disk(writer, tmp_path):
    return sum(path.stat().st_size for path in (tmp_path / writer.slot).iterdir())


def assert_charged(writer, tmp_path):
    assert writer.saved_bytes == saved_on_disk(writer, tmp_path)
    assert writer.saved_bytes <= writer.byte_limit


def payload(tmp_path, data=b"payload"):
    path = tmp_path / "source"
    path.write_bytes(data)
    return path


def test_exact_four_fixed_slots_and_caps(tmp_path):
    assert budget.SLOT_NAMES == (
        "bpzup21_first10", "bpzup12_first10",
        "bpzup21_remaining990", "bpzup12_remaining990",
    )
    assert budget.SLOT_BUDGET_BYTES == 5 * 1024**3
    assert budget.TOTAL_BUDGET_BYTES == 20 * 1024**3
    assert budget.MAX_LOG_BYTES == 1024**2
    for slot in budget.SLOT_NAMES:
        with budget.SerializedBudget(tmp_path, slot, byte_limit=512, status_reserve=128) as writer:
            writer.start()
            writer.finish()
            assert_charged(writer, tmp_path)
    for slot in ("fifth", "../bpzup21_first10", "bpzup21_first10/extra", "/tmp/slot"):
        with pytest.raises(ValueError):
            budget.SerializedBudget(tmp_path, slot)
    assert set(path.name for path in tmp_path.iterdir()) == set(budget.SLOT_NAMES)


@pytest.mark.parametrize("kwargs", [
    {"byte_limit": budget.SLOT_BUDGET_BYTES + 1},
    {"log_limit": budget.MAX_LOG_BYTES + 1}, {"log_limit": -1},
    {"status_reserve": 127}, {"byte_limit": 128, "status_reserve": 128},
    {"byte_limit": True}, {"log_limit": 1.5},
])
def test_invalid_limits_do_not_consume_slot(tmp_path, kwargs):
    with pytest.raises(ValueError):
        make_writer(tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())


def test_exclusive_slot_no_resume_after_close_or_removing_receipt(tmp_path):
    writer = make_writer(tmp_path)
    receipt = tmp_path / writer.slot / "reservation.json"
    assert json.loads(receipt.read_bytes())["byte_limit"] == 2048
    writer.close()
    receipt.unlink()
    with pytest.raises(FileExistsError):
        make_writer(tmp_path)
    assert (tmp_path / writer.slot).is_dir()


def test_concurrent_reservation_only_one_wins(tmp_path):
    barrier = threading.Barrier(2)

    def reserve():
        barrier.wait()
        try:
            writer = make_writer(tmp_path)
        except FileExistsError:
            return False
        writer.close()
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: reserve(), range(2))) == [False, True]


def test_start_is_required_and_consumed_once(tmp_path):
    with make_writer(tmp_path) as writer:
        with pytest.raises(budget.SavedOutputError, match="successful exclusive start"):
            writer.publish(payload(tmp_path), "output.root", **VERIFIED)
        writer.start()
        original = (tmp_path / writer.slot / "start.json").read_bytes()
        with pytest.raises(budget.SavedOutputError, match="already attempted"):
            writer.start()
        assert (tmp_path / writer.slot / "start.json").read_bytes() == original
        assert_charged(writer, tmp_path)


def test_start_failure_consumes_attempt_without_cleanup(tmp_path, monkeypatch):
    with make_writer(tmp_path) as writer:
        real_write = os.write
        calls = 0

        def fail_after_prefix(fd, data):
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_write(fd, data[:3])
            raise OSError(errno.ENOSPC, "injected")

        with monkeypatch.context() as patch:
            patch.setattr(budget.os, "write", fail_after_prefix)
            with pytest.raises(budget.SavedOutputError) as error:
                writer.start()
        assert error.value.file_bytes == 3
        assert error.value.saved_bytes == writer.saved_bytes
        assert (tmp_path / writer.slot / "start.json").stat().st_size == 3
        with pytest.raises(budget.SavedOutputError, match="already attempted"):
            writer.start()
        with pytest.raises(budget.SavedOutputError):
            writer.stream_log([b"no"])
        writer.finish("failed")
        assert_charged(writer, tmp_path)


def test_too_small_initial_budget_consumes_directory(tmp_path):
    with pytest.raises(budget.SavedOutputError) as error:
        make_writer(tmp_path, byte_limit=129)
    assert error.value.saved_bytes == 1
    assert (tmp_path / budget.SLOT_NAMES[0] / "reservation.json").stat().st_size == 1
    with pytest.raises(FileExistsError):
        make_writer(tmp_path)


@pytest.mark.parametrize("flags", [{}, {"confinement_verified": True},
                                  {"payload_terminated": True},
                                  {"confinement_verified": 1, "payload_terminated": True}])
def test_publish_requires_independent_attestations(tmp_path, flags):
    with make_writer(tmp_path) as writer:
        writer.start()
        with pytest.raises(budget.SavedOutputError) as error:
            writer.publish(payload(tmp_path), "output.root", **flags)
        assert error.value.reason == "prerequisite"
        assert not (tmp_path / writer.slot / "output.root").exists()


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "nested/file", "anything.root",
                                  "status.json", "reservation.json", "start.json",
                                  "manifest.json", "merged.log"])
def test_publish_fixed_payload_allowlist(tmp_path, name):
    with make_writer(tmp_path) as writer:
        writer.start()
        before = writer.saved_bytes
        with pytest.raises(budget.SavedOutputError) as error:
            writer.publish(payload(tmp_path), name, **VERIFIED)
        assert error.value.reason == "name"
        assert writer.saved_bytes == before


def test_success_and_status_include_every_saved_byte(tmp_path):
    with make_writer(tmp_path) as writer:
        writer.start()
        source = payload(tmp_path)
        for name in sorted(budget.PAYLOAD_FILENAMES):
            receipt = writer.publish(source, name, **VERIFIED)
            assert receipt.saved_bytes == len(b"payload")
            assert_charged(writer, tmp_path)
        manifest = writer.write_manifest()
        assert manifest.saved_bytes > 0
        writer.finish()
        status = json.loads((tmp_path / writer.slot / "status.json").read_bytes())
        assert status == {"state": "complete", "saved_bytes": writer.saved_bytes}
        assert_charged(writer, tmp_path)
        assert set(path.name for path in (tmp_path / writer.slot).iterdir()) <= budget.FIXED_FILENAMES
        for operation in (writer.start, writer.write_manifest, lambda: writer.stream_log([b"x"]), writer.finish):
            with pytest.raises(budget.SavedOutputError):
                operation()


def test_large_source_reads_only_remaining_and_retains_prefix(tmp_path, monkeypatch):
    with make_writer(tmp_path, byte_limit=512) as writer:
        writer.start()
        allowance = writer.remaining_bytes
        source = payload(tmp_path, b"x" * 10000)
        real_read = os.read
        sizes = []

        def checked_read(fd, count):
            sizes.append(count)
            assert count <= writer.remaining_bytes
            return real_read(fd, count)

        monkeypatch.setattr(budget.os, "read", checked_read)
        with pytest.raises(budget.SavedOutputError) as error:
            writer.publish(source, "output.root", **VERIFIED)
        assert error.value.reason == "budget"
        assert error.value.file_bytes == allowance
        assert error.value.saved_bytes == writer.saved_bytes
        assert sum(sizes) == allowance
        assert (tmp_path / writer.slot / "output.root").read_bytes() == b"x" * allowance
        assert writer.remaining_bytes == 0
        writer.finish("failed")
        assert_charged(writer, tmp_path)


def test_short_writes_charge_actual_and_io_failure_prefix(tmp_path, monkeypatch):
    with make_writer(tmp_path) as writer:
        writer.start()
        source = payload(tmp_path, b"abcdefghij")
        real_write = os.write
        calls = 0

        def short_then_error(fd, data):
            nonlocal calls
            calls += 1
            if calls <= 2:
                return real_write(fd, data[:2])
            raise OSError(errno.EIO, "injected")

        before = writer.saved_bytes
        with monkeypatch.context() as patch:
            patch.setattr(budget.os, "write", short_then_error)
            with pytest.raises(budget.SavedOutputError) as error:
                writer.publish(source, "output.root", **VERIFIED)
        assert error.value.file_bytes == 4
        assert error.value.saved_bytes == before + 4
        assert (tmp_path / writer.slot / "output.root").read_bytes() == b"abcd"
        with pytest.raises(budget.SavedOutputError):
            writer.publish(source, "output.root", **VERIFIED)
        writer.finish("failed")
        assert_charged(writer, tmp_path)


def test_short_successful_writes_and_zero_write(tmp_path, monkeypatch):
    with make_writer(tmp_path) as writer:
        writer.start()
        source = payload(tmp_path, b"abcdefghij")
        real_write = os.write
        with monkeypatch.context() as patch:
            patch.setattr(budget.os, "write", lambda fd, data: real_write(fd, data[:2]))
            assert writer.publish(source, "output.root", **VERIFIED).saved_bytes == 10
        with monkeypatch.context() as patch:
            patch.setattr(budget.os, "write", lambda fd, data: 0)
            with pytest.raises(budget.SavedOutputError) as error:
                writer.publish(source, "output.art", **VERIFIED)
        assert error.value.file_bytes == 0
        assert_charged(writer, tmp_path)


def test_log_flood_is_drained_bounded_and_cumulative(tmp_path):
    with make_writer(tmp_path, log_limit=31) as writer:
        writer.start()
        consumed = 0

        def flood():
            nonlocal consumed
            for _ in range(10000):
                consumed += 1
                yield b"a" * 1024

        first = writer.stream_log(flood())
        second = writer.stream_log([b"more", b"data"])
        assert consumed == 10000
        assert first.saved_bytes == 31
        assert first.discarded_bytes == 10000 * 1024 - 31
        assert second.saved_bytes == 0 and second.discarded_bytes == 8
        assert (tmp_path / writer.slot / "merged.log").read_bytes() == b"a" * 31
        writer.finish()
        assert_charged(writer, tmp_path)


def test_log_cannot_eat_status_reserve(tmp_path):
    with make_writer(tmp_path, byte_limit=512) as writer:
        writer.start()
        available = writer.remaining_bytes
        receipt = writer.stream_log([b"x" * 5000])
        assert receipt.saved_bytes == available
        assert receipt.discarded_bytes == 5000 - available
        assert writer.remaining_bytes == 0
        writer.finish()
        status = json.loads((tmp_path / writer.slot / "status.json").read_bytes())
        assert status["saved_bytes"] == writer.saved_bytes
        assert_charged(writer, tmp_path)


def test_log_partial_write_error_reports_charge(tmp_path, monkeypatch):
    with make_writer(tmp_path) as writer:
        writer.start()
        real_write = os.write
        calls = 0

        def partial(fd, data):
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_write(fd, data[:5])
            raise OSError(errno.ENOSPC, "injected")

        with monkeypatch.context() as patch:
            patch.setattr(budget.os, "write", partial)
            with pytest.raises(budget.SavedOutputError) as error:
                writer.stream_log([b"x" * 100])
        assert error.value.file_bytes == 5
        assert error.value.saved_bytes == writer.saved_bytes
        writer.finish("failed")
        assert_charged(writer, tmp_path)


def test_manifest_exhaustion_is_charged_and_status_survives(tmp_path):
    with make_writer(tmp_path, byte_limit=512) as writer:
        writer.start()
        writer.stream_log([b"x" * (writer.remaining_bytes - 7)])
        with pytest.raises(budget.SavedOutputError) as error:
            writer.write_manifest()
        assert error.value.reason == "budget"
        assert error.value.file_bytes == 7
        assert (tmp_path / writer.slot / "manifest.json").stat().st_size == 7
        writer.finish("failed")
        assert_charged(writer, tmp_path)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory", "parent_symlink", "dotdot"])
def test_unsafe_sources_refused_without_destination(tmp_path, kind):
    source = payload(tmp_path)
    bad = tmp_path / "unsafe"
    if kind == "symlink":
        bad.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, bad)
    elif kind == "fifo":
        os.mkfifo(bad)
    elif kind == "directory":
        bad.mkdir()
    elif kind == "parent_symlink":
        directory = tmp_path / "real"
        directory.mkdir()
        (directory / "file").write_bytes(b"data")
        bad.symlink_to(directory, target_is_directory=True)
        bad = bad / "file"
    else:
        bad = str(tmp_path) + "/unused/../source"
    with make_writer(tmp_path) as writer:
        writer.start()
        before = writer.saved_bytes
        with pytest.raises(budget.SavedOutputError):
            writer.publish(bad, "output.root", **VERIFIED)
        assert writer.saved_bytes == before
        assert not (tmp_path / writer.slot / "output.root").exists()


@pytest.mark.parametrize("kind", ["regular", "symlink", "fifo", "hardlink", "directory"])
def test_existing_destination_never_overwritten(tmp_path, kind):
    with make_writer(tmp_path) as writer:
        writer.start()
        source = payload(tmp_path, b"unchanged")
        destination = tmp_path / writer.slot / "output.root"
        if kind == "regular":
            destination.write_bytes(b"existing")
        elif kind == "symlink":
            destination.symlink_to(source)
        elif kind == "fifo":
            os.mkfifo(destination)
        elif kind == "hardlink":
            os.link(source, destination)
        else:
            destination.mkdir()
        before = writer.saved_bytes
        inode = destination.lstat().st_ino
        with pytest.raises(budget.SavedOutputError):
            writer.publish(source, "output.root", **VERIFIED)
        assert writer.saved_bytes == before
        assert destination.lstat().st_ino == inode
        assert source.read_bytes() == b"unchanged"


def test_existing_start_and_status_are_not_overwritten_or_retried(tmp_path):
    with make_writer(tmp_path) as writer:
        start = tmp_path / writer.slot / "start.json"
        start.write_bytes(b"existing")
        with pytest.raises(budget.SavedOutputError):
            writer.start()
        start.unlink()
        with pytest.raises(budget.SavedOutputError, match="already attempted"):
            writer.start()
        status = tmp_path / writer.slot / "status.json"
        status.write_bytes(b"existing status")
        with pytest.raises(budget.SavedOutputError):
            writer.finish("failed")
        assert status.read_bytes() == b"existing status"
        status.unlink()
        with pytest.raises(budget.SavedOutputError):
            writer.finish("failed")


def test_root_symlink_and_dotdot_refused(tmp_path):
    root = tmp_path / "real"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError):
        make_writer(alias)
    with pytest.raises(ValueError):
        make_writer(str(tmp_path) + "/real/../real")
    assert not list(root.iterdir())


def test_concurrent_calls_share_one_serialized_log_limit(tmp_path):
    with make_writer(tmp_path, log_limit=47) as writer:
        writer.start()
        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(lambda _: writer.stream_log([b"x" * 30]), range(20)))
        assert sum(item.saved_bytes for item in receipts) == 47
        assert sum(item.discarded_bytes for item in receipts) == 600 - 47
        writer.finish()
        assert_charged(writer, tmp_path)


def test_closed_writer_never_reopens(tmp_path):
    writer = make_writer(tmp_path)
    writer.close()
    writer.close()
    with pytest.raises(budget.SavedOutputError):
        writer.start()
    with pytest.raises(FileExistsError):
        make_writer(tmp_path)


def test_failed_final_status_charges_partial_bytes_and_never_retries(tmp_path, monkeypatch):
    with make_writer(tmp_path, byte_limit=512) as writer:
        writer.start()
        writer.stream_log([b"x" * 10000])
        before = writer.saved_bytes
        real_write = os.write
        calls = 0

        def partial_status(fd, data):
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_write(fd, data[:9])
            raise OSError(errno.ENOSPC, "injected")

        with monkeypatch.context() as patch:
            patch.setattr(budget.os, "write", partial_status)
            with pytest.raises(budget.SavedOutputError) as error:
                writer.finish("failed")
        assert error.value.file_bytes == 9
        assert error.value.saved_bytes == before + 9
        assert writer.saved_bytes == before + 9
        assert_charged(writer, tmp_path)
        with pytest.raises(budget.SavedOutputError):
            writer.finish("failed")
        assert (tmp_path / writer.slot / "status.json").stat().st_size == 9


@pytest.mark.parametrize("mutation", ["grow", "shrink"])
def test_changed_source_is_reported_with_all_saved_bytes(tmp_path, monkeypatch, mutation):
    with make_writer(tmp_path) as writer:
        writer.start()
        source = payload(tmp_path, b"abcdefghij")
        real_read = os.read
        mutated = False

        def read_then_mutate(fd, count):
            nonlocal mutated
            data = real_read(fd, min(count, 3))
            if not mutated:
                mutated = True
                with source.open("r+b") as stream:
                    stream.truncate(100 if mutation == "grow" else 3)
            return data

        monkeypatch.setattr(budget.os, "read", read_then_mutate)
        with pytest.raises(budget.SavedOutputError) as error:
            writer.publish(source, "output.root", **VERIFIED)
        assert error.value.reason == "source_changed"
        assert error.value.file_bytes == (10 if mutation == "grow" else 3)
        assert error.value.saved_bytes == writer.saved_bytes
        writer.finish("failed")
        assert_charged(writer, tmp_path)


def test_destination_open_is_exclusive_nofollow_and_not_append(tmp_path, monkeypatch):
    with make_writer(tmp_path) as writer:
        writer.start()
        source = payload(tmp_path)
        real_open = os.open
        opens = []

        def checked_open(path, flags, *args, **kwargs):
            if path == "output.root" and flags & os.O_WRONLY:
                opens.append(flags)
                assert flags & os.O_EXCL
                assert flags & os.O_CREAT
                assert flags & os.O_NOFOLLOW
                assert not flags & os.O_APPEND
                assert not flags & os.O_TRUNC
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(budget.os, "open", checked_open)
        writer.publish(source, "output.root", **VERIFIED)
        assert len(opens) == 1
        assert_charged(writer, tmp_path)


def test_empty_payload_is_safe_when_data_budget_is_full(tmp_path):
    with make_writer(tmp_path, byte_limit=512) as writer:
        writer.start()
        writer.stream_log([b"x" * 10000])
        assert writer.remaining_bytes == 0
        source = payload(tmp_path, b"")
        assert writer.publish(source, "output.root", **VERIFIED).saved_bytes == 0
        writer.finish()
        assert_charged(writer, tmp_path)
