"""Fresh-process timing receipts do not substitute reported event counts for output census."""
import json
import sys

import pytest

from mixing.process_timer import main, run


def test_success_receipt_covers_subprocess_and_is_not_overwritten(tmp_path):
    path = tmp_path / "time.json"
    command = [sys.executable, "-c", "pass"]
    assert run(command, path) == 0
    report = json.loads(path.read_text())
    assert report["exit_status"] == 0
    assert len(path.read_text().strip().splitlines()) == 1
    assert report["wall_s"] > 0 and report["user_s"] >= 0 and report["system_s"] >= 0
    with pytest.raises(FileExistsError):
        run(command, path)


def test_failure_receipt_and_cli_return_nonzero(tmp_path):
    path = tmp_path / "time.json"
    assert main(["--receipt", str(path), "--", sys.executable, "-c", "raise SystemExit(7)"]) == 7
    assert json.loads(path.read_text())["exit_status"] == 7


def test_missing_command_rejected_before_receipt(tmp_path):
    path = tmp_path / "time.json"
    with pytest.raises(ValueError, match="command is required"):
        run([], path)
    assert not path.exists()


@pytest.mark.parametrize("platform,rss", [("darwin", 16384), ("linux", 16)])
def test_rss_units_are_kib_on_both_platforms(tmp_path, monkeypatch, platform, rss):
    from types import SimpleNamespace
    from mixing import process_timer
    monkeypatch.setattr(process_timer.sys, "platform", platform)
    monkeypatch.setattr(process_timer.resource, "getrusage", lambda _: SimpleNamespace(ru_utime=0., ru_stime=0., ru_maxrss=rss))
    monkeypatch.setattr(process_timer.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    path = tmp_path / "time.json"
    assert run(["synthetic"], path) == 0
    assert json.loads(path.read_text())["max_rss_kb"] == 16
