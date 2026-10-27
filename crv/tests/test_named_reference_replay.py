"""Completed-campaign replay rejects incomplete, changed and inconsistent evidence."""
from copy import deepcopy
from pathlib import Path

import pytest

from crvfastsim.analyses import named_reference_replay as R
from test_named_reference_campaign import certificates, retained, mocked_driver, dump


@pytest.fixture
def completed(mocked_driver, monkeypatch):
    args, calls, frozen = mocked_driver
    for module, key in ((R.V, "diagnostic_sha256"), (R.B, "v1_sha256"), (R.C, "certifier_sha256")):
        frozen[key] = R.C.fingerprint(module.__file__)["sha256"]
    frozen["path_sources"] = {R.P.__name__: R.C.fingerprint(R.P.__file__)["sha256"]}
    for cfg in R.N.CONFIGS:
        geometry = []
        for name in ("counters", "world"):
            path = args["outdir"].parent / f"{cfg}_{name}.txt"
            path.write_text(f"{cfg} {name}")
            geometry.append(R.C.fingerprint(path))
        frozen["configs"][cfg]["geometry"] = geometry
        for root, filename, kind in ((args["retained_audits"], "audit50k.json", "audit"),
                                     (args["retained_paths"], "paths50k.json", "paths")):
            path = root / cfg / filename
            value = R.N.read_json(path)
            value["geometry"] = geometry
            if kind == "audit":
                value["diagnostic_sha256"] = frozen["diagnostic_sha256"]
            else:
                value["source_sha256"] = frozen["path_sources"]
            dump(path, value)
    for module in (R.V, R.P):
        original = module.merge

        def merge(inputs, output, original=original):
            value = original(inputs, output)
            value["source_audits"] = [R.C.fingerprint(p) for p in inputs]
            dump(Path(output), value)
            return value

        monkeypatch.setattr(module, "merge", merge)
    R.N.run_campaign(**args)
    calls.clear()
    return args, calls


def test_complete_replay_validation_does_not_generate(completed):
    args, calls = completed
    summary, _, shards, files, _ = R.validate_saved(args["outdir"], args["certificates"])
    assert summary["execution_complete"] and not summary["v1_pass"]
    assert len(shards) == 40 and all(len(v["emits"]) == 20 for v in files.values())
    assert not calls


@pytest.mark.parametrize("change,message", [
    (lambda r: r.update(execution_complete=False), "execution incomplete"),
    (lambda r: r.update(requested_configs=["run1a"]), "both configurations"),
    (lambda r: r["configs"]["run1a"].update(generated_events=999999), "count mismatch"),
    (lambda r: r.update(v1_pass=True), "overall V1 verdict"),
])
def test_incomplete_or_inconsistent_summary(completed, change, message):
    args, _ = completed
    path = args["outdir"] / "summary.json"
    value = R.N.read_json(path)
    change(value)
    dump(path, value)
    with pytest.raises(ValueError, match=message):
        R.validate_saved(args["outdir"], args["certificates"])


def test_changed_emit_fails_before_output(completed):
    args, _ = completed
    (args["outdir"] / "run2" / "120" / "v1_emit.json").write_text("modified")
    output = args["outdir"].parent / "replay"
    with pytest.raises(ValueError, match="artifact changed"):
        R.replay(args["outdir"], args["certificates"], output)
    assert not output.exists()


def test_duplicate_historical_certificate_shard(completed):
    args, _ = completed
    path = args["outdir"] / "manifest.json"
    value = R.N.read_json(path)
    value["shards"][1] = deepcopy(value["shards"][0])
    dump(path, value)
    with pytest.raises(ValueError, match="duplicate or extra"):
        R.validate_saved(args["outdir"], args["certificates"])


def test_changed_source_snapshot(completed):
    args, _ = completed
    path = args["outdir"] / "run1a" / "102" / "source_after.json"
    value = R.N.read_json(path)
    value["code"]["crv"]["source_sha256"]["transport.py"] = "changed"
    dump(path, value)
    with pytest.raises(ValueError, match="frozen code changed"):
        R.validate_saved(args["outdir"], args["certificates"])


def test_replay_exact_and_no_overwrite(completed, monkeypatch):
    args, calls = completed
    monkeypatch.setattr(R, "markdown", lambda summary: "synthetic table\n")
    output = args["outdir"].parent / "replay"
    report = R.replay(args["outdir"], args["certificates"], output)
    assert report["status"] == "PASS" and not report["v1_pass"]
    assert all(r["v1_exact"] and r["paired_exact"] for r in report["results"].values())
    assert not calls
    saved = (output / "verification.json").read_bytes()
    with pytest.raises(ValueError, match="existing output"):
        R.replay(args["outdir"], args["certificates"], output)
    assert (output / "verification.json").read_bytes() == saved


def test_replay_difference_keeps_partial_without_success(completed, monkeypatch):
    args, calls = completed
    original = R.B.run

    def different(*pos, **kwargs):
        value = original(*pos, **kwargs)
        value["pass"] = not value["pass"]
        return value

    monkeypatch.setattr(R.B, "run", different)
    output = args["outdir"].parent / "replay"
    with pytest.raises(ValueError, match="V1 replay differs"):
        R.replay(args["outdir"], args["certificates"], output)
    assert output.exists() and not (output / "verification.json").exists() and not calls


def test_markdown_preserves_missing_uncertainty():
    assert R.percent(None) == "unavailable"
    assert R.paired({"relative": .01, "se": None}, "relative", "se") == "+1.000"
    assert R.paired({"relative": .01, "se": .002}, "relative", "se") == "+1.000 ± 0.200"
