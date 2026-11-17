"""benchmarks/fastsim_score: statistics, flag rule, yields, both-dtype requirement, ROOT conversion, Codabench bundle."""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.codabench import build_bundle  # noqa: E402
from benchmarks.fastsim_score import score as S  # noqa: E402
from benchmarks.fastsim_score import synthetic  # noqa: E402


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    return synthetic.write(tmp_path_factory.mktemp("fsbench"))


def test_chi2_sf_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for k in (1, 2, 5, 17, 60, 300):
        for x in (0.01, 0.5, k * 0.7, k, k * 1.6, k * 4 + 10):
            assert S.chi2_sf(x, k) == pytest.approx(stats.chi2.sf(x, k), rel=1e-9, abs=1e-300)


def test_same_events_statistic_by_hand():
    h1 = {"edges": [0, 1, 2, 3], "counts": [10, 0, 4]}
    h2 = {"edges": [0, 1, 2, 3], "counts": [6, 0, 4]}
    st = S.statistic(h1, h2)
    assert st["ndf"] == 2 and st["chi2"] == pytest.approx(16 / 16) and st["kind"] == "same_events"


def test_shape_mode_ignores_normalisation():
    h1 = {"edges": [0, 1, 2], "counts": [100, 50]}
    h2 = {"edges": [0, 1, 2], "counts": [200, 100]}
    assert S.statistic(h1, h2, "shape")["chi2"] == pytest.approx(0.0)
    assert S.statistic(h1, h2, "same_events")["chi2ndf"] > 2


def test_paired_shards_statistic():
    rng = np.random.default_rng(1)
    base = rng.poisson(500, size=(10, 6)).astype(float)
    a = {"edges": list(range(7)), "shards": {f"s{i}": base[i].tolist() for i in range(10)}}
    b = {"edges": list(range(7)), "shards": {f"s{i}": (base[i] + rng.normal(0, 5, 6)).tolist() for i in range(10)}}
    st = S.statistic(a, b)
    assert st["kind"] == "paired_10_shards" and st["ndf"] == 6 and st["p"] > 1e-3


def test_binning_mismatch_is_an_error():
    with pytest.raises(S.ScoreError, match="binning"):
        S.statistic({"edges": [0, 1, 2], "counts": [1, 1]}, {"edges": [0, 1, 3], "counts": [1, 1]})


def test_good_shifted_slow(example):
    ref = example / "reference"
    good = S.score(example / "submissions" / "good", ref)
    shifted = S.score(example / "submissions" / "shifted", ref)
    slow = S.score(example / "submissions" / "slow", ref)
    assert good["validity_score"] == pytest.approx(1.0)
    assert shifted["validity_score"] < 0.5
    assert slow["validity_score"] == pytest.approx(1.0) and slow["throughput_geomean"] < good["throughput_geomean"]
    r = good["workflows"]["WA"]["float32"]
    assert r["throughput_multiplier"] == pytest.approx(5000 * 1.6 / 12.5) and r["throughput_verified"] is False
    assert "su_per_event" not in r


def test_su_only_when_private(example):
    res = S.score(example / "submissions" / "good", example / "reference", private=True)
    r = res["workflows"]["WB"]["float64"]
    assert r["su_per_event"] == pytest.approx(3.0 / (3600 * 5000))


def test_both_dtypes_required(example, tmp_path):
    sub = tmp_path / "sub"
    shutil.copytree(example / "submissions" / "good", sub)
    shutil.rmtree(sub / "WB" / "float32")
    with pytest.raises(S.ScoreError, match="both dtypes"):
        S.score(sub, example / "reference")


def test_missing_histogram_and_dtype_mismatch(example, tmp_path):
    sub = tmp_path / "sub"
    shutil.copytree(example / "submissions" / "good", sub)
    h = json.loads((sub / "WA" / "float64" / "hists.json").read_text())
    del h["hists"]["time"]
    (sub / "WA" / "float64" / "hists.json").write_text(json.dumps(h))
    with pytest.raises(S.ScoreError, match="time missing"):
        S.score(sub, example / "reference")
    shutil.copytree(example / "submissions" / "good" / "WA" / "float64", sub / "WA" / "float64", dirs_exist_ok=True)
    b = json.loads((sub / "WA" / "float64" / "bench.json").read_text())
    b["dtype"] = "float32"
    (sub / "WA" / "float64" / "bench.json").write_text(json.dumps(b))
    with pytest.raises(S.ScoreError, match="dtype"):
        S.score(sub, example / "reference")


def test_cli_exit_codes(example, tmp_path):
    assert S.main([str(example / "submissions" / "good"), str(example / "reference"),
                   "--out", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text())["n_workflows"] == 2
    assert S.main([str(tmp_path / "nothing"), str(example / "reference")]) == 1


def test_from_root_roundtrip(tmp_path):
    uproot = pytest.importorskip("uproot")
    from benchmarks.fastsim_score import from_root
    edges = np.linspace(0, 10, 11)
    for i in range(3):
        with uproot.recreate(tmp_path / f"shard_{i:02d}.root") as f:
            f["Validation/CrvCoinc/N"] = (np.arange(10.0) + i, edges)
            f["Other/X"] = (np.ones(10), edges)
    out = from_root.read(sorted(tmp_path.glob("shard_*.root")), "^Validation", {"coinc": "Validation/CrvCoinc/N"})
    assert list(out["hists"]) == ["Validation/CrvCoinc/N"]
    assert out["hists"]["Validation/CrvCoinc/N"]["shards"]["shard_01"] == pytest.approx(list(np.arange(10.0) + 1))
    assert out["yields"]["coinc"]["shards"]["shard_02"] == pytest.approx(float((np.arange(10.0) + 2).sum()))


def test_codabench_bundle_scores_in_a_fake_worker(tmp_path):
    z = build_bundle.build(tmp_path / "build")
    b = tmp_path / "build" / "bundle"
    assert z.is_file() and (b / "competition.yaml").is_file()
    keys = {line.split("key: ")[1].split(",")[0] for line in (b / "competition.yaml").read_text().splitlines()
            if "key: " in line and "sorting" in line}
    for label, sub, want in (("good", tmp_path / "build" / "example" / "submissions" / "good", 0),
                             ("wrapped", None, 0)):
        root = tmp_path / f"worker_{label}"
        shutil.copytree(b / "scoring_program", root / "program")
        shutil.copytree(b / "reference_data", root / "input" / "ref")
        dest = root / "input" / "res" / ("sub" if sub is None else "")
        shutil.copytree(tmp_path / "build" / "example" / "submissions" / "good", dest)
        p = subprocess.run([sys.executable, str(root / "program" / "scoring.py")], capture_output=True, text=True,
                           env={**os.environ, "CODABENCH_ROOT": str(root)}, timeout=120)
        assert p.returncode == want, p.stderr
        scores = json.loads((root / "output" / "scores.json").read_text())
        assert set(scores) == keys and scores["validity_score"] == pytest.approx(1.0)
    with zipfile.ZipFile(z) as zf:
        assert "scoring_program/score.py" in zf.namelist()
