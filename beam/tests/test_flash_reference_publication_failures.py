"""Local supervisor/inventory/writer fault probes, NOT confinement evidence.

All saved slots live under pytest's temporary directory. The accepting callback
is an explicitly unit-only fake; neither it nor a successful metadata inventory
certifies a runtime boundary, descendant lifecycle, or valid art/ROOT contents.
"""

import errno
import json
import os
import sys

import pytest

from beamfastsim import flash_reference_budget as budget
from beamfastsim import flash_reference_guarded_run as guarded
from beamfastsim import flash_reference_inventory as inventory


LIMIT = 2048
RESERVE = 128
RECEIPT = "unit-only fake receipt: NO confinement or termination evidence"
OUTPUTS = (("raw_1.art", "output.art"), ("export.root", "output.root"),
           ("job.fcl", "config.fcl"))
PAYLOAD = {"raw_1.art": b"art:" + b"a" * 37,
           "export.root": b"root:" + b"r" * 53,
           "job.fcl": b"config:" + b"c" * 29}


def make_tree(tmp_path):
    # Inventory deliberately refuses symlink ancestors (/var on macOS).
    base = tmp_path.resolve()
    root, work = base / "saved", base / "work"
    root.mkdir()
    work.mkdir(mode=0o700)
    for name, data in PAYLOAD.items():
        (work / name).write_bytes(data)
    return root, work


@pytest.fixture
def integration(monkeypatch):
    """Observe, but do not replace, the real inventory and bounded publisher."""
    events = []
    real_inventory = guarded.validate_inventory
    real_publish = budget.SerializedBudget.publish

    def unit_only_fake_verify(receipt, argv, work, returncode):
        assert receipt == RECEIPT
        assert argv[:2] == (sys.executable, "-c")
        assert work.is_dir()
        assert returncode == 0
        events.append("unit-only-fake-verify")
        return True  # UNIT ONLY: intentionally NOT an external security check.

    def census(work, layout):
        assert events[-1] == "unit-only-fake-verify"
        events.append("inventory-enter")
        measured = real_inventory(work, layout)
        assert measured.verified is False
        events.append("inventory-accepted")
        return measured

    def publish(self, source, name, **kwargs):
        assert "inventory-accepted" in events
        events.append("publish:" + name)
        return real_publish(self, source, name, **kwargs)

    monkeypatch.setattr(guarded, "validate_inventory", census)
    monkeypatch.setattr(budget.SerializedBudget, "publish", publish)

    def run(root, work, *, slot=budget.SLOT_NAMES[0], script="pass", **kwargs):
        kwargs.setdefault("byte_limit", LIMIT)
        kwargs.setdefault("log_limit", 31)
        return guarded.execute_and_publish(
            root, slot, [sys.executable, "-c", script], work=work,
            parent_timeout=5, status_reserve=RESERVE,
            independently_verified_confinement_receipt=RECEIPT,
            verify_callback=unit_only_fake_verify, **kwargs)

    return run, events


def snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def assert_charged(root, report, limit=LIMIT, *, status="failed"):
    saved = snapshot(root / report.slot)
    assert set(saved) <= budget.FIXED_FILENAMES
    assert sum(map(len, saved.values())) == report.saved_bytes <= limit
    assert report.fully_enforced is False
    assert report.wrapper_reaped is True
    assert report.returncode == 0
    assert report.publication_verified_by_caller is True
    if status is not None:
        assert json.loads(saved["status.json"]) == {
            "state": status, "saved_bytes": report.saved_bytes}
    return saved


def assert_permanent(run, root, work, report, **kwargs):
    before = snapshot(root / report.slot)
    with pytest.raises(FileExistsError):
        run(root, work, slot=report.slot, **kwargs)
    assert snapshot(root / report.slot) == before


def inject_partial_writes(monkeypatch, names, prefix=7):
    """Make an actual short os.write, then EIO, only inside named writes.

    Lifecycle markers/logs remain real and unfaulted. Restricting the patch to
    _write's dynamic scope avoids mistaking a reused fd for the faulted file.
    """
    real_write = budget.os.write
    real_bounded_write = budget.SerializedBudget._write
    calls = {name: 0 for name in names}

    def bounded_write(self, fd, name, data, **kwargs):
        if name not in calls:
            return real_bounded_write(self, fd, name, data, **kwargs)

        def short_then_error(target_fd, chunk):
            assert target_fd == fd
            calls[name] += 1
            if calls[name] == 1:
                assert len(chunk) > prefix
                return real_write(target_fd, chunk[:prefix])
            raise OSError(errno.EIO, "unit-only partial write: " + name)

        with monkeypatch.context() as patch:
            patch.setattr(budget.os, "write", short_then_error)
            return real_bounded_write(self, fd, name, data, **kwargs)

    monkeypatch.setattr(budget.SerializedBudget, "_write", bounded_write)
    return calls


def test_four_independent_permanent_slots_share_no_allowance(tmp_path, integration):
    run, events = integration
    root, first_work = make_tree(tmp_path)
    caps = (768, 1024, 1536, 2048)
    assert len(budget.SLOT_NAMES) == 4
    assert max(caps) <= budget.SLOT_BUDGET_BYTES == 5 * 1024**3
    retained, reports, works = {}, [], []
    for index, (slot, cap) in enumerate(zip(budget.SLOT_NAMES, caps)):
        work = first_work if index == 0 else tmp_path.resolve() / f"work-{index}"
        if index:
            work.mkdir()
        for name, data in PAYLOAD.items():
            (work / name).write_bytes(data)
        if index < 2:
            (work / OUTPUTS[index][0]).write_bytes(bytes([65 + index]) * 8192)
            with pytest.raises(guarded.GuardedRunError) as error:
                run(root, work, slot=slot, byte_limit=cap)
            report = error.value.report
            assert error.value.__cause__.reason == "budget"
            saved = assert_charged(root, report, cap)
            assert report.state == "failed"
        else:
            report = run(root, work, slot=slot, byte_limit=cap)
            saved = assert_charged(root, report, cap, status="complete")
            assert report.state == "complete"
        assert json.loads(saved["reservation.json"])["byte_limit"] == cap
        for previous_slot, contents in retained.items():
            assert snapshot(root / previous_slot) == contents
        retained[slot] = saved
        reports.append(report)
        works.append(work)

    assert {path.name for path in root.iterdir()} == set(budget.SLOT_NAMES)
    assert events.count("inventory-accepted") == 4
    assert sum(report.saved_bytes for report in reports) <= sum(caps)
    before_events = list(events)
    for report, work, cap in zip(reports, works, caps):
        assert_permanent(run, root, work, report, byte_limit=cap)
    assert events == before_events  # Refuse before another launch/callback.


@pytest.mark.parametrize("stage", range(3), ids=[pair[1] for pair in OUTPUTS])
def test_overflow_keeps_exact_charged_prefix_and_no_later_outputs(
        tmp_path, integration, stage):
    run, events = integration
    root, work = make_tree(tmp_path)
    source, destination = OUTPUTS[stage]
    data = bytes(range(256)) * 32
    (work / source).write_bytes(data)
    with pytest.raises(guarded.GuardedRunError) as error:
        run(root, work, byte_limit=768)
    report, cause = error.value.report, error.value.__cause__
    saved = assert_charged(root, report, 768)
    assert report.state == "failed"
    assert isinstance(cause, budget.SavedOutputError)
    assert cause.reason == "budget"
    before_prefix = sum(len(value) for name, value in saved.items()
                        if name not in (destination, "status.json"))
    allowance = 768 - RESERVE - before_prefix
    assert 0 < allowance < len(data)
    assert saved[destination] == data[:allowance]
    assert cause.file_bytes == allowance
    assert cause.saved_bytes == 768 - RESERVE
    assert report.saved_bytes == cause.saved_bytes + len(saved["status.json"])
    assert report.published == tuple(name for _, name in OUTPUTS[:stage])
    assert not {name for _, name in OUTPUTS[stage + 1:]} & saved.keys()
    assert "manifest.json" not in saved
    assert events[:3] == ["unit-only-fake-verify", "inventory-enter", "inventory-accepted"]
    assert_permanent(run, root, work, report, byte_limit=768)
    assert (work / source).read_bytes() == data


@pytest.mark.parametrize("stage", range(3), ids=[pair[1] for pair in OUTPUTS])
def test_publication_short_write_charges_actual_bytes_not_requested(
        tmp_path, monkeypatch, integration, stage):
    run, _ = integration
    root, work = make_tree(tmp_path)
    source, destination = OUTPUTS[stage]
    before_work = snapshot(work)
    calls = inject_partial_writes(monkeypatch, [destination])
    with pytest.raises(guarded.GuardedRunError) as error:
        run(root, work)
    report, cause = error.value.report, error.value.__cause__
    saved = assert_charged(root, report)
    assert calls == {destination: 2}
    assert isinstance(cause, budget.SavedOutputError)
    assert cause.reason == "io"
    assert cause.file_bytes == 7
    assert saved[destination] == PAYLOAD[source][:7]
    assert report.saved_bytes == cause.saved_bytes + len(saved["status.json"])
    assert report.published == tuple(name for _, name in OUTPUTS[:stage])
    assert not {name for _, name in OUTPUTS[stage + 1:]} & saved.keys()
    assert "manifest.json" not in saved
    assert snapshot(work) == before_work
    assert_permanent(run, root, work, report)
    assert calls == {destination: 2}  # No recovery publisher or write retry.


@pytest.mark.parametrize("targets", [
    ("manifest.json",), ("status.json",), ("manifest.json", "status.json"),
], ids=["manifest", "status", "manifest-and-status"])
def test_manifest_and_final_status_faults_retain_all_charges(
        tmp_path, monkeypatch, integration, targets):
    run, events = integration
    root, work = make_tree(tmp_path)
    calls = inject_partial_writes(monkeypatch, targets)
    with pytest.raises(guarded.GuardedRunError) as error:
        run(root, work)
    report, cause = error.value.report, error.value.__cause__
    saved = assert_charged(root, report, status=None if "status.json" in targets else "failed")
    assert report.state == "failed"
    assert isinstance(cause, budget.SavedOutputError)
    assert cause.reason == "io" and cause.file_bytes == 7
    assert report.published == tuple(name for _, name in OUTPUTS)
    assert calls == dict.fromkeys(targets, 2)
    assert events[-3:] == ["publish:" + name for _, name in OUTPUTS]
    for source, destination in OUTPUTS:
        assert saved[destination] == PAYLOAD[source]
    for target in targets:
        assert len(saved[target]) == 7
        with pytest.raises(json.JSONDecodeError):
            json.loads(saved[target])
    if "status.json" in targets:
        assert "final status failed" in report.failure
    if targets == ("status.json",):
        manifest = json.loads(saved["manifest.json"])
        assert manifest["saved_bytes_before_manifest"] == sum(
            len(value) for name, value in saved.items()
            if name not in ("manifest.json", "status.json"))
        assert report.saved_bytes == cause.saved_bytes
    else:
        assert report.saved_bytes == cause.saved_bytes + len(saved["status.json"])
    assert_permanent(run, root, work, report)
    assert calls == dict.fromkeys(targets, 2)


@pytest.mark.parametrize("case", [
    "raw_2.art", "raw_10.art", "RootOutput-unique.tmp", "raw_1.art.tmp.12345",
    ".RootOutput.tmp", "numbered-directory", "temp-symlink", "temp-hardlink",
    "temp-fifo", "scratch-residual", "fixed-symlink", "fixed-hardlink",
])
def test_unsafe_rootoutput_objects_refuse_before_any_source_open(
        tmp_path, monkeypatch, integration, case):
    run, events = integration
    root, work = make_tree(tmp_path)
    if case == "numbered-directory":
        (work / "raw_2.art").mkdir()
    elif case == "temp-symlink":
        (work / "RootOutput-unique.tmp").symlink_to(work / "raw_1.art")
    elif case == "temp-hardlink":
        os.link(work / "raw_1.art", work / "RootOutput-unique.tmp")
    elif case == "temp-fifo":
        os.mkfifo(work / "RootOutput-unique.tmp")
    elif case == "scratch-residual":
        (work / "tmp").mkdir()
        (work / "tmp" / "RootOutput-unique.tmp").write_bytes(b"unfinished")
    elif case == "fixed-symlink":
        (work / "raw_1.art").unlink()
        (work / "raw_1.art").symlink_to(work / "export.root")
    elif case == "fixed-hardlink":
        os.link(work / "raw_1.art", tmp_path.resolve() / "outside-hardlink")
    else:
        (work / case).write_bytes(b"unclosed or numbered output")
    before = {str(path.relative_to(work)): path.lstat() for path in work.rglob("*")}

    def forbidden_source_open(*args, **kwargs):
        pytest.fail("strict inventory must refuse before opening any publication source")

    monkeypatch.setattr(budget, "_open_source", forbidden_source_open)
    with pytest.raises(guarded.GuardedRunError) as error:
        run(root, work)
    report = error.value.report
    assert isinstance(error.value.__cause__, inventory.InventoryError)
    assert events == ["unit-only-fake-verify", "inventory-enter"]
    saved = assert_charged(root, report)
    assert report.state == "failed" and report.published == ()
    assert not (budget.PAYLOAD_FILENAMES | {"manifest.json"}) & saved.keys()
    assert {str(path.relative_to(work)): path.lstat() for path in work.rglob("*")} == before
    assert_permanent(run, root, work, report)


@pytest.mark.parametrize("log_limit", [0, 37, budget.MAX_LOG_BYTES], ids=["zero", "small", "one-MiB"])
def test_merged_log_flood_drains_both_pipes_then_publishes(
        tmp_path, integration, log_limit):
    run, events = integration
    root, work = make_tree(tmp_path)
    # Finite descendant-free writes finish after the discarded-output flood drains.
    script = """
import os
from pathlib import Path
for index in range(48):
    data = memoryview((b'o' if index % 2 == 0 else b'e') * 65536)
    while data:
        written = os.write(1 if index % 2 == 0 else 2, data)
        data = data[written:]
Path('job.fcl').write_bytes(b'finished after both log streams')
"""
    cap = budget.MAX_LOG_BYTES + 8192
    report = run(root, work, script=script, byte_limit=cap, log_limit=log_limit)
    saved = assert_charged(root, report, cap, status="complete")
    assert report.state == "complete"
    assert report.published == tuple(name for _, name in OUTPUTS)
    expected = (b"o" * 65536 + b"e" * 65536) * 24
    assert saved["merged.log"] == expected[:log_limit]
    assert report.discarded_log_bytes == len(expected) - log_limit
    assert len(saved["merged.log"]) <= budget.MAX_LOG_BYTES
    assert saved["config.fcl"] == b"finished after both log streams"
    assert events[:3] == ["unit-only-fake-verify", "inventory-enter", "inventory-accepted"]
    assert_permanent(run, root, work, report, byte_limit=cap, log_limit=log_limit)


def inject_completed_write_fsync_failure(monkeypatch, method, target):
    # These injected faults probe local accounting, not runtime durability.
    # Scope the fault to one operation and discover its fd from completed writes.
    original = getattr(budget.SerializedBudget, method)
    real_write = budget.SerializedBudget._write
    real_fsync = budget.os.fsync
    faults = []

    def operation(self, *args, **kwargs):
        name = args[1] if method == "publish" else args[0]
        if name != target:
            return original(self, *args, **kwargs)
        completed_fd = None

        def write(writer, fd, name, data, **write_kwargs):
            nonlocal completed_fd
            result = real_write(writer, fd, name, data, **write_kwargs)
            if writer is self and name == target:
                completed_fd = fd
            return result

        def fsync(fd):
            if fd == completed_fd:
                assert os.fstat(fd).st_size == self._files[target] > 0
                faults.append(self._files[target])
                raise OSError(errno.EIO, "unit-only completed-write fsync: " + target)
            return real_fsync(fd)

        with monkeypatch.context() as patch:
            patch.setattr(budget.SerializedBudget, "_write", write)
            patch.setattr(budget.os, "fsync", fsync)
            return original(self, *args, **kwargs)

    monkeypatch.setattr(budget.SerializedBudget, method, operation)
    return faults


@pytest.mark.parametrize("target", [pair[1] for pair in OUTPUTS] + [
    "manifest.json", "status.json",
])
def test_completed_file_write_then_fsync_failure_is_charged_not_success(
        tmp_path, monkeypatch, integration, target):
    run, _ = integration
    root, work = make_tree(tmp_path)
    cap = 1024
    method = "publish" if target in budget.PAYLOAD_FILENAMES else "_write_blob"
    faults = inject_completed_write_fsync_failure(monkeypatch, method, target)
    with pytest.raises(guarded.GuardedRunError) as error:
        run(root, work, byte_limit=cap)
    report, cause = error.value.report, error.value.__cause__
    saved = assert_charged(root, report, cap, status=None if target == "status.json" else "failed")
    assert report.state == "failed"
    assert isinstance(cause, budget.SavedOutputError) and cause.reason == "io"
    assert faults == [len(saved[target])] == [cause.file_bytes]
    assert report.saved_bytes == cause.saved_bytes + (
        0 if target == "status.json" else len(saved["status.json"]))
    if target in budget.PAYLOAD_FILENAMES:
        stage = [pair[1] for pair in OUTPUTS].index(target)
        assert saved[target] == PAYLOAD[OUTPUTS[stage][0]]
        assert report.published == tuple(name for _, name in OUTPUTS[:stage])
        assert not {name for _, name in OUTPUTS[stage + 1:]} & saved.keys()
        assert "manifest.json" not in saved
    else:
        assert report.published == tuple(name for _, name in OUTPUTS)
        assert json.loads(saved["manifest.json"])["slot"] == report.slot
        if target == "status.json":
            assert json.loads(saved[target]) == {
                "state": "complete", "saved_bytes": report.saved_bytes}
            assert "final status failed" in report.failure
    assert_permanent(run, root, work, report, byte_limit=cap)
    assert faults == [len(saved[target])]


@pytest.mark.parametrize("method", ["stream_log", "finish"], ids=["first-flush", "finish-flush"])
def test_merged_log_fsync_failure_retains_bytes_and_consumes_slot(
        tmp_path, monkeypatch, integration, method):
    run, events = integration
    root, work = make_tree(tmp_path)
    cap = 1024
    original = getattr(budget.SerializedBudget, method)
    real_fsync = budget.os.fsync
    faults = []

    def operation(self, *args, **kwargs):
        def fsync(fd):
            if fd == self._log_fd:
                assert os.fstat(fd).st_size == len(b"merged-log-probe")
                faults.append(self._files["merged.log"])
                raise OSError(errno.EIO, "unit-only merged-log fsync: " + method)
            return real_fsync(fd)

        with monkeypatch.context() as patch:
            patch.setattr(budget.os, "fsync", fsync)
            return original(self, *args, **kwargs)

    monkeypatch.setattr(budget.SerializedBudget, method, operation)
    with pytest.raises(guarded.GuardedRunError) as error:
        run(root, work, byte_limit=cap, script="import os; os.write(1, b'merged-log-probe')")
    report, cause = error.value.report, error.value.__cause__
    saved = snapshot(root / report.slot)
    assert set(saved) <= budget.FIXED_FILENAMES
    assert sum(map(len, saved.values())) == report.saved_bytes <= cap
    assert report.state == "failed" and report.fully_enforced is False
    assert report.wrapper_reaped is True and report.returncode is not None
    assert isinstance(cause, budget.SavedOutputError) and cause.reason == "io"
    assert faults == [len(saved["merged.log"])] == [cause.file_bytes]
    assert saved["merged.log"] == b"merged-log-probe"
    assert report.discarded_log_bytes == 0
    if method == "stream_log":
        assert events == [] and report.publication_verified_by_caller is False
        assert report.published == ()
        assert not (budget.PAYLOAD_FILENAMES | {"manifest.json"}) & saved.keys()
        assert json.loads(saved["status.json"]) == {
            "state": "failed", "saved_bytes": report.saved_bytes}
        assert report.saved_bytes == cause.saved_bytes + len(saved["status.json"])
    else:
        assert report.publication_verified_by_caller is True
        assert report.published == tuple(name for _, name in OUTPUTS)
        assert "manifest.json" in saved and "status.json" not in saved
        assert report.saved_bytes == cause.saved_bytes
        assert "final status failed" in report.failure
    assert_permanent(run, root, work, report, byte_limit=cap)
    assert faults == [len(saved["merged.log"])]


def test_slot_directory_fsync_failure_retains_empty_payload_and_charge(
        tmp_path, monkeypatch, integration):
    run, _ = integration
    root, work = make_tree(tmp_path)
    cap = 1024
    real_create = budget.SerializedBudget._create
    real_fsync = budget.os.fsync
    faults = []

    def create(self, name):
        if name != "output.art":
            return real_create(self, name)

        def fsync(fd):
            if fd == self._dir_fd:
                faults.append(name)
                raise OSError(errno.EIO, "unit-only slot-directory fsync")
            return real_fsync(fd)

        with monkeypatch.context() as patch:
            patch.setattr(budget.os, "fsync", fsync)
            return real_create(self, name)

    monkeypatch.setattr(budget.SerializedBudget, "_create", create)
    with pytest.raises(guarded.GuardedRunError) as error:
        run(root, work, byte_limit=cap)
    report, cause = error.value.report, error.value.__cause__
    saved = assert_charged(root, report, cap)
    assert report.state == "failed" and report.published == ()
    assert isinstance(cause, budget.SavedOutputError) and cause.reason == "io"
    assert cause.file_bytes == 0 and saved["output.art"] == b""
    assert not {"output.root", "config.fcl", "manifest.json"} & saved.keys()
    assert report.saved_bytes == cause.saved_bytes + len(saved["status.json"])
    assert_permanent(run, root, work, report, byte_limit=cap)
    assert faults == ["output.art"]


def test_root_directory_fsync_failure_consumes_empty_slot_before_report(
        tmp_path, monkeypatch, integration):
    run, events = integration
    root, work = make_tree(tmp_path)
    cap = 1024
    root_info = root.stat()
    real_fsync = budget.os.fsync
    faults = []

    def fsync(fd):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (root_info.st_dev, root_info.st_ino):
            faults.append("root")
            raise OSError(errno.EIO, "unit-only root-directory fsync")
        return real_fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(budget.os, "fsync", fsync)
        with pytest.raises(OSError, match="unit-only root-directory fsync") as error:
            run(root, work, byte_limit=cap)
    assert error.value.errno == errno.EIO and not hasattr(error.value, "report")
    assert events == [] and faults == ["root"]
    slot = root / budget.SLOT_NAMES[0]
    assert snapshot(slot) == {}
    assert sum(path.stat().st_size for path in slot.iterdir()) == 0 <= cap
    with pytest.raises(FileExistsError):
        run(root, work, byte_limit=cap)
    assert snapshot(slot) == {} and events == []
