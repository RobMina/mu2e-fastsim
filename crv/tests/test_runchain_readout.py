"""Readout-policy admission, forwarding and merge tests with synthetic pipeline assets."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crvfastsim import cli, named_conditions, readout
from test_e2e_bench import B, R, pipeline


def ordinary(state, **kw):
    opts = {k: state[k] for k in ("inputs", "outdir", "library", "calib", "gdml")}
    return R.run(**{**opts, **kw})


def test_cli_readout_is_explicit_and_off_by_default():
    parser = cli.build_parser()
    assert not parser.parse_args(["runchain", "gen.root"]).geometry_readout
    assert parser.parse_args(["runchain", "gen.root", "--geometry-readout"]).geometry_readout


@pytest.mark.parametrize("change", [
    {}, {"run_config": "run1a_v01"}, {"run_config": "run2_v10"},
    {"conditions_manifest": "manifest.json"},
    {"run_config": "run1b_v01", "conditions_manifest": "manifest.json"},
    {"run_config": "extracted_v04", "conditions_manifest": "manifest.json"},
    {"run_config": "run2_v10", "conditions_manifest": "manifest.json", "counters": "override.txt"},
])
def test_readout_admission_precedes_resolution(monkeypatch, change):
    monkeypatch.setattr(R, "_resolve", lambda *a, **kw: pytest.fail("resolved before admission"))
    with pytest.raises(ValueError, match="requires Run1A/Run2"):
        R.run([], "unused", geometry_readout=True, **change)


@pytest.mark.parametrize("entry", ["prepare", "run_prepared", "run_benchmark", "prepare_benchmark"])
def test_readout_reuse_rejected_before_resolution(monkeypatch, entry):
    monkeypatch.setattr(R, "_resolve", lambda *a, **kw: pytest.fail("resolved before admission"))
    kw = {"geometry_readout": True}
    if "benchmark" in entry:
        kw["benchmark"] = object()
    if entry == "run_prepared":
        kw["prepared"] = object()
    with pytest.raises(ValueError, match="not supported with prepared contexts or benchmarks"):
        if entry.startswith("prepare"):
            R.prepare(**kw)
        else:
            R.run([], "unused", **kw)


def test_benchmark_entry_rejects_readout_before_output(pipeline):
    kw = {k: pipeline[k] for k in ("inputs", "outdir", "library", "gdml")}
    with pytest.raises(ValueError, match="unsupported benchmark options"):
        B.run(**kw, geometry_readout=True)
    assert not Path(pipeline["outdir"]).exists()


def test_disabled_does_not_inspect_readout_or_add_policy(pipeline, monkeypatch, tmp_path):
    monkeypatch.setattr(readout, "apply", lambda *a: pytest.fail("default applied mask"))
    emit = tmp_path / "legacy.json"
    result = ordinary(pipeline, emit=str(emit), geometry_readout=False)
    assert "geometry_readout" not in result
    assert "geometry_readout" not in json.loads(emit.read_text())
    assert "geometry_readout" not in R.merge([str(emit)], tmp_path / "merge")


@pytest.mark.parametrize("name", ["run1a_v01", "run2_v10"])
@pytest.mark.parametrize("prefetch", [False, True])
def test_named_mask_forwarded_to_all_stages_without_mutation(pipeline, monkeypatch, tmp_path, name, prefetch):
    names = list(R.runconfig.get(name).sectors)
    geom = SimpleNamespace(nbars=len(names), sector_names=names, sector=np.arange(len(names)))
    ccfg = {f"crs.sipmsAtSide{side}{n}": not (side == 1 and n == "R4")
            for n in names for side in (0, 1)}
    cond = R.runconfig.nominal_conditions(geom)
    cond.status[0] = 4
    before = copy.deepcopy(cond)
    evidence = {"schema": "synthetic", "hash": name}
    monkeypatch.setattr(R.runconfig.RunConfig, "geometry", lambda self: geom)
    monkeypatch.setattr(R.runconfig.RunConfig, "counters_cfg", lambda self: ccfg)
    monkeypatch.setattr(R.runconfig.RunConfig, "tagger_bar", lambda *a: None)
    monkeypatch.setattr(R.runconfig, "sector_type_names", lambda *a: {1: names[0]})
    monkeypatch.setattr(named_conditions, "load_conditions", lambda *a: (cond, evidence))
    stages = []
    mask = readout.sensor_mask(geom, ccfg)

    def observe(fn, index, label):
        def wrapped(*args, **kw):
            used = args[index]
            assert used is not cond
            np.testing.assert_array_equal(used.readout_mask, mask)
            np.testing.assert_array_equal(used.status, before.status)
            stages.append(label)
            return fn(*args, **kw)
        return wrapped

    for function, index in (("charges", 3), ("digis", 1), ("reco_waveforms", 1), ("clusters", 1)):
        monkeypatch.setattr(R.chain, function, observe(getattr(R.chain, function), index, function))
    second = tmp_path / "second.root"
    second.write_bytes(b"synthetic")
    emit = tmp_path / "masked.json"
    result = ordinary(pipeline, inputs=[*pipeline["inputs"], str(second)], emit=str(emit), calib=None,
                      run_config=name, conditions_manifest="manifest.json", geometry_readout=True,
                      no_below_world=True, prefetch=prefetch)
    shard = json.loads(emit.read_text())
    stats = json.loads((Path(pipeline["outdir"]) / "runchain_stats.json").read_text())
    expected = readout.apply(cond, geom, ccfg)[1]
    assert expected["absent"] == (2 if name == "run2_v10" else 0)
    assert result["geometry_readout"] == shard["geometry_readout"] == stats["geometry_readout"] == expected
    assert result["conditions"] == shard["conditions"] == evidence
    assert stages == ["charges", "digis", "reco_waveforms", "clusters"] * 2
    assert cond.readout_mask is None
    for key, original in vars(before).items():
        if isinstance(original, np.ndarray):
            np.testing.assert_array_equal(getattr(cond, key), original)
    merged = R.merge([str(emit)], tmp_path / "merged")
    assert merged["geometry_readout"] == expected


@pytest.mark.parametrize("change", ["missing", "null", "policy", "hash", "count", "negative", "float", "bool",
                                    "extra", "short_hash", "uppercase", "fractional_bar"])
def test_merge_readout_guard_runs_before_accumulation(tmp_path, change):
    _, window = R.resolve_runchain_window("run2_v10")
    policy = {"policy": "opt_in_counters_sipmsAtSide", "present": 6, "absent": 2, "mask_sha256": "a" * 64}
    first = {"run_config": "run2_v10", "window": [0, 100000], "time_window": window,
             "geometry_readout": policy}
    second = copy.deepcopy(first)
    if change == "missing":
        del second["geometry_readout"]
    elif change == "null":
        second["geometry_readout"] = None
    else:
        key, value = {"policy": ("policy", "default"), "hash": ("mask_sha256", "b" * 64),
                      "count": ("present", 10), "negative": ("present", -2), "float": ("absent", 2.0),
                      "bool": ("absent", True), "extra": ("unbound", "value"),
                      "short_hash": ("mask_sha256", "a"), "uppercase": ("mask_sha256", "A" * 64),
                      "fractional_bar": ("present", 7)}[change]
        second["geometry_readout"][key] = value
    paths = []
    for i, record in enumerate((first, second)):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(record))
        paths.append(str(path))
    with pytest.raises(ValueError, match="geometry_readout"):
        R.merge(paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_apply_reuses_fasttable_digest_and_validates_status_shape():
    geom = SimpleNamespace(nbars=1, sector_names=["R4"], sector=np.array([0]))
    ccfg = {"crs.sipmsAtSide0R4": True, "crs.sipmsAtSide1R4": False}
    cond = R.runconfig.nominal_conditions(geom)
    masked, evidence = readout.apply(cond, geom, ccfg)
    assert evidence["mask_sha256"] == named_conditions._digest([True, False, True, False])
    assert cond.readout_mask is None and masked.status is cond.status
    cond.status = np.zeros(5, int)
    with pytest.raises(ValueError, match="exactly one entry"):
        readout.apply(cond, geom, ccfg)
