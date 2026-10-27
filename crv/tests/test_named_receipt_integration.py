"""Prospective receipt comparison is conditional integrity evidence, never full V2 certification."""
import copy
import json

import pytest

from crvfastsim import cli, response_receipt as RR
from crvfastsim.analyses import v2compare as V
from test_named_response import named, populated


def test_v2_receipt_flags():
    args = cli.build_parser().parse_args(["v2compare", "fast.root", "--compare", "stock.root",
        "--run-config", "run1a_v01", "--candidate-receipt", "receipt.json", "--conditions-manifest", "manifest.json"])
    assert args.candidate_receipt == "receipt.json" and args.conditions_manifest == "manifest.json"


@pytest.mark.parametrize("extra", [{}, {"run_config": "run1a_v01"},
    {"run_config": "run1a_v01", "conditions_manifest": "m", "emit": "out.json"},
    {"run_config": "run1a_v01", "conditions_manifest": "m", "generated": 10}])
def test_receipt_options_fail_early(extra):
    with pytest.raises(ValueError, match="receipt audit"):
        V.run(["fast"], "out", compare=["stock"], candidate_receipt="receipt", **extra)


def test_receipt_binds_comparison_file_but_not_full_v2(named, monkeypatch, tmp_path):
    rc = named("run1a_v01")
    reference = populated(rc)
    fast, stock, wrong = (tmp_path / p for p in ("fast.root", "stock.root", "wrong.root"))
    fast.write_bytes(b"fast")
    stock.write_bytes(b"stock")
    wrong.write_bytes(b"other")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"metadata": {"stage": "reco"}, "inputs": {"input": RR._record(stock)}}))
    calls = []
    def validate(*args, **kwargs):
        calls.append(args)
        return {"status": "bound_fast_output_and_assets", "named_v2_pass": False}
    monkeypatch.setattr(RR, "validate_receipt", validate)
    monkeypatch.setattr(RR, "event_identity", lambda p: {"count": 10000, "sha256": "same"})
    monkeypatch.setattr(V, "_fill_arm", lambda arm, *args: arm.update(copy.deepcopy(reference["ref"])))
    monkeypatch.setattr(V, "report", lambda state, *args: V.summarise(state))
    result = V.run([str(fast)], tmp_path / "report", compare=[str(stock)], run_config=rc,
                   candidate_receipt=receipt, conditions_manifest="manifest")
    assert len(calls) == 2 and result["metrics_pass"]
    assert not result["pass"] and result["status"] == "unsupported"
    assert result["input_provenance_status"].startswith("partial_")
    with pytest.raises(ValueError, match="recorded stock"):
        V.run([str(fast)], tmp_path, compare=[str(wrong)], run_config=rc,
              candidate_receipt=receipt, conditions_manifest="manifest")
    monkeypatch.setattr(RR, "event_identity", lambda p: {"path": str(p)})
    with pytest.raises(ValueError, match="populations differ"):
        V.run([str(fast)], tmp_path, compare=[str(stock)], run_config=rc,
              candidate_receipt=receipt, conditions_manifest="manifest")
    state = V._new_state(rc)
    audited = copy.deepcopy(state)
    audited["producer_receipt_audit"] = {}
    with pytest.raises(ValueError, match="cannot be merged"):
        V._merge_into(state, audited)
    assert state == V._new_state(rc)


def test_v2_freezes_glob_before_validation(named, monkeypatch, tmp_path):
    rc = named("run1a_v01")
    fast, stock = tmp_path / "fast1.root", tmp_path / "stock.root"
    fast.write_bytes(b"fast")
    stock.write_bytes(b"stock")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"metadata": {"stage": "reco"}, "inputs": {"input": RR._record(stock)}}))
    def validate(*args, **kwargs):
        (tmp_path / "fast2.root").write_bytes(b"must not enter comparison")
        return {}
    monkeypatch.setattr(RR, "validate_receipt", validate)
    monkeypatch.setattr(RR, "event_identity", lambda p: {"count": 1})
    seen = []
    monkeypatch.setattr(V, "_fill_arm", lambda arm, paths, *args: seen.append(paths))
    monkeypatch.setattr(V, "report", lambda state, *args: state)
    V.run([str(tmp_path / "fast*.root")], tmp_path, compare=[str(stock)], run_config=rc,
          candidate_receipt=receipt, conditions_manifest="manifest")
    assert seen == [[str(fast.resolve())], [str(stock.resolve())]]


def test_cli_uses_snapshot_target_after_alias_changes(monkeypatch, tmp_path):
    import dataclasses
    import sys
    from crvfastsim import runconfig
    from valhist import fasttables as FT
    original, replacement, manifest = (tmp_path / name for name in ("original", "replacement", "manifest"))
    original.write_bytes(b"old")
    replacement.write_bytes(b"new")
    manifest.write_text("{}")
    alias = tmp_path / "input"
    alias.symlink_to(original)
    snapshot = RR.snapshot_inputs
    def capture(paths):
        result = snapshot(paths)
        alias.unlink()
        alias.symlink_to(replacement)
        return result
    monkeypatch.setattr(RR, "snapshot_inputs", capture)
    cfg = dataclasses.replace(runconfig.get("run1a_v01").chain_config(), window=(0., 100000.))
    monkeypatch.setattr(FT, "configure", lambda **kwargs: (None, None, None, cfg, {}))
    seen = []
    def digis(path):
        seen.append(path)
        raise RuntimeError("stop after verifying producer path")
    monkeypatch.setattr(FT, "stock_digis", digis)
    monkeypatch.setattr(sys, "argv", ["fasttables", "reco", str(alias), str(tmp_path / "out.root"),
        "--run-config", "run1a_v01", "--conditions-manifest", str(manifest), "--write-receipt"])
    with pytest.raises(RuntimeError, match="verifying producer"):
        FT.main()
    assert seen == [str(original.resolve())]
