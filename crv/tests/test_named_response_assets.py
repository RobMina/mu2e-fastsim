"""Synthetic, resolver-level tests; no payload parsing, network or event loops."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from crvfastsim import geometry, runconfig
from crvfastsim.analyses import named_response_assets as assets
from crvfastsim.reco.waveform import WaveformConfig
from crvfastsim.response import lookup, sipm


@pytest.fixture
def payloads(tmp_path, monkeypatch):
    root = tmp_path / "source" / "Offline"
    optical = tmp_path / "optical"
    optical.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(geometry, "OFFLINE_ROOTS", ())
    monkeypatch.setenv("MU2E_SEARCH_PATH", str(root.parent))
    monkeypatch.setenv("CRVFS_LOOKUP_DIR", str(optical))
    monkeypatch.delenv("CRVFS_NO_FETCH", raising=False)
    for name in assets.RUN_CONFIGS:
        rc = runconfig.get(name)
        for configured in (rc.counters, WaveformConfig().single_pe_file):
            path = root / configured.removeprefix("Offline/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic {configured}\n".encode())
        for table in {s.table for s in rc.sectors.values()}:
            (optical / f"LookupTable_{table}").write_bytes(f"synthetic {table}\n".encode())
    # Deliberately not a ROOT file: the guard must not invoke a payload reader.
    (optical / "photonMap.root").write_bytes(b"opaque, not ROOT\x00\xff")
    (optical / "LookupTable_unused").write_bytes(b"not configured")
    return root, optical, work


def expected_paths(name, root, optical):
    rc = runconfig.get(name)
    return {
        "counters": root / rc.counters.removeprefix("Offline/"),
        "waveform": root / WaveformConfig().single_pe_file.removeprefix("Offline/"),
        "photonMap.root": optical / "photonMap.root",
        **{f"LookupTable_{s.table}": optical / f"LookupTable_{s.table}" for s in rc.sectors.values()},
    }


@pytest.mark.parametrize("name", assets.RUN_CONFIGS)
def test_correct_explicit_root_and_all_named_assets(payloads, monkeypatch, name):
    root, optical, _ = payloads
    calls = []
    resolver = geometry.find_counters

    def resolve(configured):
        calls.append(configured)
        return resolver(configured)

    def no_parsing(*args, **kwargs):
        pytest.fail("inventory must hash bytes, not parse payloads")

    monkeypatch.setattr(geometry, "find_counters", resolve)
    monkeypatch.setattr(lookup, "load", no_parsing)
    monkeypatch.setattr(sipm, "photon_map", no_parsing)
    result = assets.inventory(name, root)
    assert result["run_config"] == name
    assert result["scope"] == "diagnostic_asset_identity_only"
    assert result["schema_version"] == 1
    expected = expected_paths(name, root, optical)
    assert set(result["assets"]) == set(expected)
    for key, path in expected.items():
        data = path.read_bytes()
        assert result["assets"][key] == {
            "path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
        }
    assert calls == [runconfig.get(name).counters, WaveformConfig().single_pe_file] * 2
    assert "CRVFS_NO_FETCH" not in os.environ


@pytest.mark.parametrize("key", ["counters", "waveform"])
@pytest.mark.parametrize("same_bytes", [True, False])
def test_cwd_shadow_is_rejected_even_with_identical_bytes(payloads, key, same_bytes):
    root, optical, work = payloads
    expected = expected_paths("run2_v10", root, optical)[key]
    shadow = work / "Offline" / expected.relative_to(root)
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_bytes(expected.read_bytes() if same_bytes else b"wrong root and bytes")
    with pytest.raises(ValueError, match=rf"{key}: resolver selected .*expected explicit Offline asset"):
        assets.inventory("run2_v10", root)


@pytest.mark.parametrize("key", ["counters", "waveform"])
def test_search_path_shadow_is_rejected(payloads, monkeypatch, key):
    root, optical, work = payloads
    expected = expected_paths("run2_v10", root, optical)[key]
    shadow = work / "other" / "Offline" / expected.relative_to(root)
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_bytes(expected.read_bytes())
    monkeypatch.setenv("MU2E_SEARCH_PATH", f"{work / 'other'}:{root.parent}")
    with pytest.raises(ValueError, match=f"{key}: resolver selected"):
        assets.inventory("run2_v10", root)


def test_offline_roots_are_used_by_actual_resolver(payloads, monkeypatch):
    root, _, _ = payloads
    monkeypatch.setenv("MU2E_SEARCH_PATH", "")
    monkeypatch.setattr(geometry, "OFFLINE_ROOTS", (str(root.parent),))
    assert assets.inventory("run1a_v01", root)["run_config"] == "run1a_v01"


@pytest.mark.parametrize("key", ["counters", "waveform", "LookupTable_6000_0", "photonMap.root"])
@pytest.mark.parametrize("directory", [False, True])
def test_missing_or_nonregular_asset_refused(payloads, key, directory):
    root, optical, _ = payloads
    path = expected_paths("run2_v10", root, optical)[key]
    path.unlink()
    if directory:
        path.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        assets.inventory("run2_v10", root)


def test_non_offline_basename_is_not_silently_used(payloads, monkeypatch):
    root, _, _ = payloads
    explicit = root.with_name("payloads")
    root.rename(explicit)
    monkeypatch.setenv("MU2E_SEARCH_PATH", f"{explicit}:{explicit.parent}")
    called = []

    def fetch_disabled(relative):
        assert os.environ["CRVFS_NO_FETCH"] == "1"
        called.append(relative)
        return None

    monkeypatch.setattr(geometry, "_fetch_public", fetch_disabled)
    with pytest.raises(FileNotFoundError, match="not found"):
        assets.inventory("run2_v10", explicit)
    assert called
    assert "CRVFS_NO_FETCH" not in os.environ


def test_non_offline_basename_works_if_actual_resolver_reaches_it(payloads):
    root, _, _ = payloads
    explicit = root.with_name("payloads")
    root.rename(explicit)
    root.symlink_to(explicit, target_is_directory=True)
    result = assets.inventory("run2_v10", explicit)
    for key in ("counters", "waveform"):
        assert Path(result["assets"][key]["path"]).is_relative_to(explicit.resolve())


@pytest.mark.parametrize("key", ["counters", "waveform", "LookupTable_6000_0", "photonMap.root"])
def test_second_inventory_reports_changed_bytes(payloads, key):
    root, optical, _ = payloads
    before = assets.inventory("run2_v10", root)
    path = expected_paths("run2_v10", root, optical)[key]
    # Equal-length changes must still change the JSON (not only stat size).
    path.write_bytes(path.read_bytes()[::-1])
    after = assets.inventory("run2_v10", root)
    assert before["assets"][key]["size"] == after["assets"][key]["size"]
    assert before["assets"][key]["sha256"] != after["assets"][key]["sha256"]
    assert json.dumps(before, sort_keys=True) != json.dumps(after, sort_keys=True)


def test_final_rehash_detects_mutation_during_inventory(payloads, monkeypatch):
    root, optical, _ = payloads
    target = expected_paths("run2_v10", root, optical)["waveform"].resolve()
    record = assets._record
    changed = False

    def mutate_after_hash(path):
        nonlocal changed
        result = record(path)
        if path == target and not changed:
            changed = True
            path.write_bytes(path.read_bytes()[::-1])
        return result

    monkeypatch.setattr(assets, "_record", mutate_after_hash)
    with pytest.raises(ValueError, match="changed while taking inventory"):
        assets.inventory("run2_v10", root)
    assert changed


def test_final_resolution_detects_same_byte_symlink_retarget(payloads, monkeypatch):
    root, optical, _ = payloads
    target = optical / "photonMap.root"
    first, second = optical / "first", optical / "second"
    first.write_bytes(target.read_bytes())
    second.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(first)
    record = assets._record

    def retarget_after_hash(path):
        result = record(path)
        if path == first.resolve():
            target.unlink()
            target.symlink_to(second)
        return result

    monkeypatch.setattr(assets, "_record", retarget_after_hash)
    with pytest.raises(ValueError, match="changed while taking inventory"):
        assets.inventory("run2_v10", root)


@pytest.mark.parametrize("previous", [None, "", "already-disabled"])
@pytest.mark.parametrize("fail", [False, True])
def test_no_fetch_environment_restored(payloads, monkeypatch, previous, fail):
    root, optical, _ = payloads
    if previous is not None:
        monkeypatch.setenv("CRVFS_NO_FETCH", previous)
    if fail:
        (optical / "photonMap.root").unlink()
        with pytest.raises(FileNotFoundError):
            assets.inventory("run2_v10", root)
    else:
        assets.inventory("run2_v10", root)
    assert os.environ.get("CRVFS_NO_FETCH") == previous


def test_stable_sorted_json_cli_and_no_files_written(payloads):
    root, _, work = payloads
    project = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(project), env.get("PYTHONPATH", "")])
    command = [sys.executable, "-m", "crvfastsim.analyses.named_response_assets",
               "--run-config", "run2_v10", "--offline", str(root)]
    before = sorted(str(p) for p in root.parent.parent.rglob("*"))
    runs = [subprocess.run(command, cwd=work, env=env, capture_output=True, text=True) for _ in range(2)]
    for result in runs:
        assert result.returncode == 0, result.stderr
        assert not result.stderr
    assert runs[0].stdout == runs[1].stdout
    parsed = json.loads(runs[0].stdout)
    assert runs[0].stdout == json.dumps(parsed, sort_keys=True, indent=2) + "\n"
    assert parsed == assets.inventory("run2_v10", root)
    assert before == sorted(str(p) for p in root.parent.parent.rglob("*"))


def test_cli_failure_has_no_inventory_stdout(payloads, capsys):
    root, optical, _ = payloads
    (optical / "photonMap.root").unlink()
    with pytest.raises(SystemExit) as exc:
        assets.main(["--run-config", "run2_v10", "--offline", str(root)])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "photonMap.root" in captured.err


def test_unsupported_config_refused(payloads):
    with pytest.raises(ValueError, match="unsupported"):
        assets.inventory("extracted_v04", payloads[0])
