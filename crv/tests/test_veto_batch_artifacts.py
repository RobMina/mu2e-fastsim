"""Synthetic-only tests of bounded artifact publication and fail-closed batch status."""
import hashlib
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crvfastsim.veto import batch_artifacts as B


def context():
    return {"registry_sha256": "a" * 64,
            "batch_identity": {"plan_sha256": "b" * 64, "batch": 7, "frame_first": 2, "frame_stop": 5},
            "arms": ["nominal", "comparison"], "seeds": [0, 2**64 - 1]}


def transaction(path, names=("output.json",), cap=100000, provenance=None):
    return B.BatchArtifacts(path, max_bytes=cap, context=context() if provenance is None else provenance,
                            expected_artifacts=names)


def failed(path):
    assert json.loads((path / B.INCOMPLETE).read_text())["state"] == "incomplete"
    assert json.loads((path / B.FAILED).read_text())["state"] == "failed"
    assert not (path / B.COMPLETE).exists()


def test_json_parquet_success_and_frozen_provenance(tmp_path):
    path = tmp_path / "success"
    provenance = context()
    tx = transaction(path, ("output.json", "frames.parquet"), provenance=provenance)
    provenance["seeds"].append(12)
    frame = pd.DataFrame({"frame_id": [2, 3, 4], "count": [0, 2, 1]})
    with tx:
        assert (path / B.INCOMPLETE).exists()
        assert not (path / B.COMPLETE).exists()
        tx.write_json("output.json", {"frames": 3, "truncated": False})
        tx.write_parquet("frames.parquet", frame, compression="NONE")
        assert not (path / B.COMPLETE).exists()
    receipt = json.loads((path / B.COMPLETE).read_text())
    assert receipt["context"] == context()
    assert receipt["context_sha256"] == hashlib.sha256(B._canonical(context())).hexdigest()
    assert receipt["execution_admitted"] is False
    assert receipt["physics_truncated"] is False and receipt["denominator_renormalized"] is False
    assert receipt["state"] == "complete"
    pd.testing.assert_frame_equal(pd.read_parquet(path / "frames.parquet"), frame)
    assert json.loads((path / "output.json").read_text()) == {"frames": 3, "truncated": False}
    for name, record in receipt["artifacts"].items():
        data = (path / name).read_bytes()
        assert record == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert receipt["artifact_bytes"] == sum((path / name).stat().st_size for name in receipt["artifacts"])
    assert (path / B.PENDING).read_bytes() == (path / B.COMPLETE).read_bytes()
    assert tx.persisted_path_bytes == sum(p.stat().st_size for p in path.iterdir()) <= 100000
    assert not (path / B.FAILED).exists()


@pytest.mark.parametrize("count", [0, 1, 16])
def test_proposed_write_refused_before_any_bytes_and_caught_failure_latches(tmp_path, count):
    path = tmp_path / "cap"
    tx = transaction(path, ("data.bin",), cap=4000)
    with pytest.raises(B.TransactionFailed):
        with tx:
            available = tx._budget.limit - tx._budget.reserve - tx.persisted_path_bytes
            with tx.open_artifact("data.bin") as stream:
                stream.write(b"x" * (available - count))
                size = (path / "data.bin").stat().st_size
                with pytest.raises(B.ByteLimitExceeded):
                    stream.write(b"y" * (count + 1))
                assert (path / "data.bin").stat().st_size == size
    failed(path)
    assert sum(p.stat().st_size for p in path.iterdir()) <= 4000


def test_exact_stream_cap_and_failure_reserve(tmp_path):
    path = tmp_path / "exact"
    tx = transaction(path, ("data.bin",), cap=4000)
    with pytest.raises(B.ByteLimitExceeded):
        with tx:
            available = tx._budget.limit - tx._budget.reserve - tx.persisted_path_bytes
            with tx.open_artifact("data.bin") as stream:
                assert stream.write(b"x" * available) == available
                assert stream.write(b"") == 0
    failed(path)
    assert sum(p.stat().st_size for p in path.iterdir()) == 4000


def test_complete_receipt_cap_boundary(tmp_path):
    reference = transaction(tmp_path / "reference", cap=9999)
    with reference:
        reference.write_json("output.json", {"n": 1})
    reserve = len(reference._failure_payload)
    boundary = reference.persisted_path_bytes + reserve
    assert len(str(boundary)) == len(str(9999))
    exact = transaction(tmp_path / "exact", cap=boundary)
    with exact:
        exact.write_json("output.json", {"n": 1})
    assert exact.persisted_path_bytes + reserve == boundary
    path = tmp_path / "short"
    with pytest.raises(B.ByteLimitExceeded):
        with transaction(path, cap=boundary - 1) as tx:
            tx.write_json("output.json", {"n": 1})
    failed(path)
    assert not (path / B.PENDING).exists()
    assert sum(p.stat().st_size for p in path.iterdir()) <= boundary - 1


def test_parquet_footer_is_bounded_before_write(tmp_path, monkeypatch):
    table = pa.table({"frame_id": [1, 2, 3], "value": [0.0, 0.5, 1.0]})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="NONE")
    encoded_size = sink.tell()
    calls = []
    original = B._Writer.write

    def observe(self, data):
        before = self.size
        try:
            return original(self, data)
        finally:
            calls.append((bytes(data), before, self.size))

    monkeypatch.setattr(B._Writer, "write", observe)
    path = tmp_path / "footer"
    tx = transaction(path, ("frames.parquet", "padding.bin"), cap=5000)
    with pytest.raises(B.ByteLimitExceeded):
        with tx:
            remaining = tx._budget.limit - tx._budget.reserve - tx.persisted_path_bytes
            padding = remaining - encoded_size + 1
            assert padding >= 0
            with tx.open_artifact("padding.bin") as stream:
                stream.write(b"x" * padding)
            tx.write_parquet("frames.parquet", table, compression="NONE")
    failed(path)
    refused = [(data, before, after) for data, before, after in calls if data and before == after]
    assert refused and refused[-1][0] == b"PAR1"
    assert sum(p.stat().st_size for p in path.iterdir()) <= 5000


def test_injected_encoder_failure_and_no_recovery(tmp_path, monkeypatch):
    def broken(table, stream, **options):
        stream.write(b"PAR1partial")
        raise RuntimeError("injected encoder failure")

    monkeypatch.setattr(pq, "write_table", broken)
    path = tmp_path / "encoder"
    with pytest.raises(B.TransactionFailed):
        with transaction(path, ("frames.parquet",)) as tx:
            with pytest.raises(RuntimeError, match="injected"):
                tx.write_parquet("frames.parquet", pa.table({"x": [1]}))
    failed(path)
    assert (path / "frames.parquet").read_bytes() == b"PAR1partial"


@pytest.mark.parametrize("change", ["same_size", "append", "replace", "symlink", "status", "extra"])
def test_precommit_modification_rejected(tmp_path, change):
    path = tmp_path / "altered"
    with pytest.raises((B.TransactionFailed, OSError)):
        with transaction(path) as tx:
            tx.write_json("output.json", {"value": 123})
            target = path / "output.json"
            if change == "same_size":
                target.write_bytes(target.read_bytes().replace(b"123", b"456"))
            elif change == "append":
                with target.open("ab") as handle:
                    handle.write(b" ")
            elif change == "replace":
                other = tmp_path / "replacement"
                other.write_bytes(target.read_bytes())
                os.replace(other, target)
            elif change == "symlink":
                other = tmp_path / "other"
                other.write_bytes(target.read_bytes())
                target.unlink()
                target.symlink_to(other)
            elif change == "status":
                status = path / B.INCOMPLETE
                status.write_bytes(status.read_bytes().replace(b"incomplete", b"xxxxxxxxxx"))
            else:
                (path / "unexpected.bin").write_bytes(b"x")
    assert (path / B.FAILED).exists() and not (path / B.COMPLETE).exists()


@pytest.mark.parametrize("complete", [False, True])
def test_existing_directory_never_reopened_or_changed(tmp_path, complete):
    path = tmp_path / "existing"
    if complete:
        with transaction(path) as tx:
            tx.write_json("output.json", {"ok": True})
    else:
        path.mkdir()
        (path / "unrelated.bin").write_bytes(b"old")
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(FileExistsError):
        with transaction(path):
            pytest.fail("must not enter existing directory")
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


def test_existing_symlink_directory_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    path = tmp_path / "link"
    path.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        with transaction(path):
            pass
    assert not list(target.iterdir())


@pytest.mark.parametrize("name", ["../x.json", "/x.json", "sub/x.json", "sub\\x.json", ".x.json", "x..json",
                                  "x.json ", "x\x00.json", "X.json", "é.json", "con.json", "com1.json",
                                  B.INCOMPLETE, B.FAILED, B.COMPLETE, B.PENDING, "a" * 97 + ".json", "noextension"])
def test_invalid_names_rejected_before_directory_creation(tmp_path, name):
    path = tmp_path / "invalid"
    with pytest.raises(ValueError):
        transaction(path, (name,))
    assert not path.exists()


@pytest.mark.parametrize("field,value", [("registry_sha256", "bad"), ("seeds", [True]), ("seeds", [-1]),
                                         ("seeds", [2**64]), ("seeds", [1, 1]), ("arms", []),
                                         ("arms", ["same", "same"]), ("batch_identity", {}),
                                         ("batch_identity", {"bad": float("nan")}),
                                         ("batch_identity", {1: "nonstring"})])
def test_provenance_validation(tmp_path, field, value):
    provenance = context()
    provenance[field] = value
    with pytest.raises((ValueError, TypeError)):
        transaction(tmp_path / "invalid", provenance=provenance)
    assert not (tmp_path / "invalid").exists()


def test_insufficient_status_budget_rejected_without_directory(tmp_path):
    with pytest.raises(B.ByteLimitExceeded):
        transaction(tmp_path / "tiny", cap=1)
    assert not (tmp_path / "tiny").exists()


def test_no_overwrite_missing_and_unexpected_artifacts(tmp_path):
    for kind in ("duplicate", "missing", "unexpected"):
        path = tmp_path / kind
        with pytest.raises(B.TransactionFailed):
            with transaction(path) as tx:
                if kind != "missing":
                    tx.write_json("output.json", {"original": True})
                    tx.write_json("output.json" if kind == "duplicate" else "other.json", {"replacement": True})
        failed(path)
        if kind != "missing":
            assert json.loads((path / "output.json").read_text()) == {"original": True}


def test_receipt_published_only_after_complete_verified_staging(tmp_path, monkeypatch):
    path = tmp_path / "atomic"
    original_link, original_verify = B.os.link, B.BatchArtifacts._verify
    verified = []

    def verify(self, name, record):
        original_verify(self, name, record)
        verified.append(name)

    def publish(source, destination, **options):
        assert source == B.PENDING and destination == B.COMPLETE
        assert not (path / B.COMPLETE).exists()
        assert verified == [B.INCOMPLETE, "output.json", B.PENDING]
        assert json.loads((path / B.PENDING).read_text())["state"] == "complete"
        original_link(source, destination, **options)
        assert (path / B.COMPLETE).read_bytes() == (path / B.PENDING).read_bytes()

    monkeypatch.setattr(B.BatchArtifacts, "_verify", verify)
    monkeypatch.setattr(B.os, "link", publish)
    with transaction(path) as tx:
        tx.write_json("output.json", {"ok": True})


def test_publication_failure_leaves_only_incomplete_staging(tmp_path, monkeypatch):
    path = tmp_path / "publish_failure"

    def fail_link(*args, **kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(B.os, "link", fail_link)
    with pytest.raises(OSError, match="publication"):
        with transaction(path) as tx:
            tx.write_json("output.json", {"ok": True})
    failed(path)
    assert (path / B.PENDING).exists()


def test_fsync_failure_prevents_receipt(tmp_path, monkeypatch):
    path = tmp_path / "sync_failure"
    original = B.os.fsync
    with pytest.raises(OSError, match="injected fsync"):
        with transaction(path) as tx:
            tx.write_json("output.json", {"ok": True})

            def fail_once(fd):
                monkeypatch.setattr(B.os, "fsync", original)
                raise OSError("injected fsync failure")

            monkeypatch.setattr(B.os, "fsync", fail_once)
    failed(path)
    assert not (path / B.PENDING).exists()


def test_body_exception_preserved_no_physics_success(tmp_path):
    path = tmp_path / "body_failure"
    with pytest.raises(RuntimeError, match="physics guard"):
        with transaction(path) as tx:
            tx.write_json("output.json", {"partial_diagnostic": True})
            raise RuntimeError("physics guard")
    failed(path)
    status = json.loads((path / B.FAILED).read_text())
    assert status["execution_admitted"] is False
    assert status["physics_truncated"] is False
    assert status["denominator_renormalized"] is False


def test_closed_encoder_sink_and_transaction_reuse_rejected(tmp_path):
    path = tmp_path / "closed"
    tx = transaction(path, ("data.bin",))
    with pytest.raises(B.TransactionFailed, match="closed"):
        with tx:
            with tx.open_artifact("data.bin") as stream:
                stream.write(b"partial")
                stream.close()
    failed(path)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(B.TransactionFailed, match="reused"):
        with tx:
            pass
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


def test_caught_close_failure_latches(tmp_path, monkeypatch):
    path = tmp_path / "close_failure"
    original = B._Writer.close
    with pytest.raises(B.TransactionFailed):
        with transaction(path) as tx:
            def fail_once(self):
                original(self)
                monkeypatch.setattr(B._Writer, "close", original)
                raise OSError("injected close failure")

            monkeypatch.setattr(B._Writer, "close", fail_once)
            with pytest.raises(OSError, match="close failure"):
                tx.write_json("output.json", {"n": 1})
    failed(path)



def inject_stream_errors(monkeypatch, target, close_error, primary=None, operation="write"):
    original_open, original_write = B.BatchArtifacts._open, B.os.write

    def open_stream(self, name):
        stream = original_open(self, name)
        if name == target:
            original_close = stream.close

            def close():
                if not stream.closed:
                    original_close()
                    raise close_error

            monkeypatch.setattr(stream, "close", close)
            if primary is not None:
                if operation == "write":
                    def write(fd, data):
                        if fd == stream._fd:
                            monkeypatch.setattr(B.os, "write", original_write)
                            raise primary
                        return original_write(fd, data)

                    monkeypatch.setattr(B.os, "write", write)
                else:
                    def finish():
                        raise primary

                    monkeypatch.setattr(stream, "finish", finish)
        return stream

    monkeypatch.setattr(B.BatchArtifacts, "_open", open_stream)


@pytest.mark.parametrize("operation", ["encoder", "body", "write"])
@pytest.mark.parametrize("caught", [False, True])
def test_artifact_primary_error_survives_close_failure(tmp_path, monkeypatch, operation, caught):
    path = tmp_path / "double_failure"
    primary = RuntimeError("injected primary " + operation)
    close_error = OSError("injected artifact close failure")
    inject_stream_errors(monkeypatch, "frames.parquet", close_error,
                         primary if operation == "write" else None)

    def encode(table, stream, **options):
        stream.write(b"PAR1partial")
        raise primary

    monkeypatch.setattr(pq, "write_table", encode)
    with pytest.raises(B.TransactionFailed if caught else RuntimeError) as outer:
        with transaction(path, ("frames.parquet",)) as tx:
            with pytest.raises(RuntimeError) as raised:
                if operation == "body":
                    with tx.open_artifact("frames.parquet") as stream:
                        stream.write(b"partial")
                        raise primary
                else:
                    tx.write_parquet("frames.parquet", pa.table({"x": [1]}))
            assert raised.value is primary
            assert any(repr(close_error) in note for note in primary.__notes__)
            assert tx._poisoned and not tx._active
            if not caught:
                raise primary
    if not caught:
        assert outer.value is primary
    failed(path)


@pytest.mark.parametrize("target", [B.INCOMPLETE, B.PENDING, B.FAILED])
@pytest.mark.parametrize("operation", ["write", "finish"])
def test_control_primary_error_survives_close_failure(tmp_path, monkeypatch, target, operation):
    path = tmp_path / "control_double_failure"
    primary = OSError("injected control " + operation + " failure")
    close_error = OSError("injected control close failure")
    body_error = RuntimeError("injected body failure")
    inject_stream_errors(monkeypatch, target, close_error, primary, operation)
    tx = transaction(path)
    with pytest.raises(RuntimeError if target == B.FAILED else OSError) as raised:
        with tx:
            if target == B.FAILED:
                raise body_error
            tx.write_json("output.json", {"n": 1})
    assert raised.value is (body_error if target == B.FAILED else primary)
    assert any(repr(close_error) in note for note in primary.__notes__)
    if target == B.FAILED:
        assert any(repr(primary) in note for note in body_error.__notes__)
        assert any(repr(close_error) in note for note in body_error.__notes__)
    else:
        assert json.loads((path / B.FAILED).read_text())["state"] == "failed"
    assert tx._poisoned and tx._state == "failed" and tx._dir_fd is None
    assert not (path / B.COMPLETE).exists()
    with pytest.raises(B.TransactionFailed):
        with tx:
            pass


@pytest.mark.parametrize("target", [B.INCOMPLETE, B.PENDING])
def test_control_close_failure_without_primary_fails_closed(tmp_path, monkeypatch, target):
    path = tmp_path / "control_close_failure"
    close_error = OSError("injected control close failure")
    inject_stream_errors(monkeypatch, target, close_error)
    with pytest.raises(OSError) as raised:
        with transaction(path) as tx:
            tx.write_json("output.json", {"n": 1})
    assert raised.value is close_error
    failed(path)


def test_partial_system_writes_are_counted_and_hashed(tmp_path, monkeypatch):
    original = B.os.write

    def short_write(fd, data):
        return original(fd, data[:7])

    monkeypatch.setattr(B.os, "write", short_write)
    path = tmp_path / "short_writes"
    with transaction(path) as tx:
        tx.write_json("output.json", {"payload": "x" * 80})
    receipt = json.loads((path / B.COMPLETE).read_text())
    data = (path / "output.json").read_bytes()
    assert receipt["artifacts"]["output.json"] == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert tx.persisted_path_bytes == sum(p.stat().st_size for p in path.iterdir())


def test_failure_marker_io_failure_preserves_original_error(tmp_path, monkeypatch):
    path = tmp_path / "failure_marker"
    original = B.BatchArtifacts._control

    def fail_marker(self, name, payload):
        if name == B.FAILED:
            raise OSError("injected marker IO failure")
        return original(self, name, payload)

    monkeypatch.setattr(B.BatchArtifacts, "_control", fail_marker)
    with pytest.raises(RuntimeError, match="body error"):
        with transaction(path):
            raise RuntimeError("body error")
    assert json.loads((path / B.INCOMPLETE).read_text())["state"] == "incomplete"
    assert not (path / B.COMPLETE).exists()


def test_atomic_publication_does_not_overwrite_racing_name(tmp_path, monkeypatch):
    path = tmp_path / "collision"
    original = B.os.link

    def collide(source, destination, **options):
        (path / B.COMPLETE).write_bytes(b"external sentinel")
        original(source, destination, **options)

    monkeypatch.setattr(B.os, "link", collide)
    with pytest.raises(FileExistsError):
        with transaction(path) as tx:
            tx.write_json("output.json", {"n": 1})
    assert (path / B.COMPLETE).read_bytes() == b"external sentinel"
    assert json.loads((path / B.FAILED).read_text())["state"] == "failed"
