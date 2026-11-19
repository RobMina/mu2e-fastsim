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


def test_values_kind_by_hand():
    a = {"edges": [0, 1, 2, 3], "values": [1.0, 2.0, 3.0], "errors": [0.1, 0.0, 0.2]}
    b = {"edges": [0, 1, 2, 3], "values": [1.1, 2.0, 3.0], "errors": [0.1, 0.0, 0.0]}
    st = S.statistic(a, b)
    assert st["kind"] == "values" and st["ndf"] == 2 and st["chi2"] == pytest.approx(0.01 / 0.02)


def test_efficiency_kind_by_hand():
    a = {"edges": [0, 1, 2], "passed": [90, 0], "total": [100, 50]}
    b = {"edges": [0, 1, 2], "passed": [80, 0], "total": [100, 0]}
    st = S.statistic(a, b)
    assert st["kind"] == "efficiency" and st["ndf"] == 1
    assert st["chi2"] == pytest.approx(0.1 ** 2 / (0.9 * 0.1 / 100 + 0.8 * 0.2 / 100))
    with pytest.raises(S.ScoreError, match="passed <= total"):
        S.statistic(a, {"edges": [0, 1, 2], "passed": [101, 0], "total": [100, 50]})


def test_kinds_must_match():
    with pytest.raises(S.ScoreError, match="reference is counts"):
        S.statistic({"edges": [0, 1], "counts": [3]}, {"edges": [0, 1], "values": [3], "errors": [1]})


def test_measured_yield():
    y = S.yield_ratio({"value": 0.0020, "error": 0.00002}, {"value": 0.00196, "error": 0.00004}, 0.0)
    assert y["ratio"] == pytest.approx(0.98) and y["pass"]
    assert not S.yield_ratio({"value": 1.0, "error": 0.001}, {"value": 0.9, "error": 0.001}, 0.01)["pass"]
    with pytest.raises(S.ScoreError, match="measured value"):
        S.yield_ratio({"value": 1.0, "error": 0.1}, 10, 0.0)


def test_rate_unit_and_jobs_ratio(example, tmp_path):
    ref = tmp_path / "ref"
    shutil.copytree(example / "reference", ref)
    cfg = json.loads((ref / "WA" / "reference.json").read_text())
    (ref / "WA" / "reference.json").write_text(json.dumps({**cfg, "rate_unit": "tables/s", "stock_jobs_per_unit": 3000}))
    sub = tmp_path / "sub"
    shutil.copytree(example / "submissions" / "good", sub)
    for dt in ("float32", "float64"):
        b = json.loads((sub / "WA" / dt / "bench.json").read_text())
        (sub / "WA" / dt / "bench.json").write_text(json.dumps({**b, "rate_unit": "tables/s", "jobs_per_unit": 4}))
    r = S.score(sub, ref)["workflows"]["WA"]["float64"]
    assert r["rate_unit"] == "tables/s" and r["jobs_ratio"] == pytest.approx(750)
    b = json.loads((sub / "WA" / "float64" / "bench.json").read_text())
    (sub / "WA" / "float64" / "bench.json").write_text(json.dumps({**b, "rate_unit": "events/s"}))
    with pytest.raises(S.ScoreError, match="rate is in"):
        S.score(sub, ref)


def test_allow_missing_lists_arms(example, tmp_path):
    sub = tmp_path / "sub"
    shutil.copytree(example / "submissions" / "good", sub)
    shutil.rmtree(sub / "WB" / "float32")
    res = S.score(sub, example / "reference", allow_missing=True)
    assert "both dtypes" in res["missing"]["WB"]["float32"] and res["n_arms_scored"] == 3
    assert "missing" not in S.score(example / "submissions" / "good", example / "reference")


def test_validity_only_task(example, tmp_path):
    ref = tmp_path / "ref"
    shutil.copytree(example / "reference", ref)
    cfg = json.loads((ref / "WB" / "reference.json").read_text())
    (ref / "WB" / "reference.json").write_text(json.dumps({**cfg, "stock_rate_per_slot": None}))
    sub = tmp_path / "sub"
    shutil.copytree(example / "submissions" / "good", sub)
    for dt in ("float32", "float64"):
        (sub / "WB" / dt / "bench.json").unlink()
    res = S.score(sub, ref)
    r = res["workflows"]["WB"]["float32"]
    assert r["throughput_scored"] is False and r["throughput_multiplier"] is None and r["validity"] == pytest.approx(1)
    assert res["throughput_geomean"] == pytest.approx(5000 * 1.6 ** 0.5 / 12.5)


def test_paired_shape_ignores_the_yield():
    rng = np.random.default_rng(3)
    base = rng.poisson(800, size=(12, 8)).astype(float)
    a = {"edges": list(range(9)), "shards": {f"s{i}": base[i].tolist() for i in range(12)}}
    b = {"edges": list(range(9)), "shards": {f"s{i}": (0.96 * base[i] + rng.normal(0, 4, 8)).tolist() for i in range(12)}}
    st = S.statistic(a, b, "shape")
    assert st["kind"] == "paired_shape_12_shards" and st["ndf"] == 7 and st["p"] > 1e-3
    assert S.statistic(a, b, "same_events")["chi2ndf"] > 10
    empty = {"edges": list(range(9)), "shards": {f"s{i}": [0.0] * 8 for i in range(12)}}
    assert S.statistic(a, empty, "shape")["kind"] == "one_empty"


def test_task_table_lists_every_task():
    from benchmarks.fastsim_score import tasks
    t = tasks.table()
    for k in ("W1", "W8", "W9", "CRVLY", "PTGT", "B1", "B2", "B3", "B4"):
        assert f"| {k} |" in t
    assert all(tasks.TASKS[k]["status"] == "defined" for k in ("CRVLY", "B1", "B2", "B3", "B4"))


# ---------------------------------------------------------------- B1-B4 extensions

def test_auc_ties_and_order():
    assert S.auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)
    assert S.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert S.auc(np.array([0.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(0.25)


def test_classifier_score_separates_shifted_arms():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=(4000, 3)), rng.normal(size=(4000, 3))
    same = S.classifier_score(a, b)
    assert same["cls_score"] > 0.85 and same["n_per_arm"] == 4000
    wide = b.copy()
    wide[:, 1] *= 1.6  # same mean, wider: only the quadratic features see it
    assert S.classifier_score(a, wide)["cls_score"] < 0.5
    assert S.classifier_score(a, b, seed=1)["auc"] != same["auc"]
    with pytest.raises(S.ScoreError, match="same columns"):
        S.classifier_score(a, b[:, :2])


def _with_cfg(example, tmp_path, w, **extra):
    ref = tmp_path / "ref"
    shutil.copytree(example / "reference", ref)
    cfg = json.loads((ref / w / "reference.json").read_text())
    (ref / w / "reference.json").write_text(json.dumps({**cfg, **extra}))
    sub = tmp_path / "sub"
    shutil.copytree(example / "submissions" / "good", sub)
    return ref, sub


def test_classifier_in_score(example, tmp_path):
    ref, sub = _with_cfg(example, tmp_path, "WA", classifier={"file": "features.npz", "seed": 3})
    rng = np.random.default_rng(9)
    for d in (ref / "WA" / "stock", ref / "WA" / "floor", sub / "WA" / "float32", sub / "WA" / "float64"):
        np.savez(d / "features.npz", X=rng.normal(size=(500, 4)))
    r = S.score(sub, ref)["workflows"]["WA"]["float64"]
    assert r["classifier"]["cls_score"] > 0.6 and "floor_cls_score" in r["classifier"]
    assert r["validity"] == pytest.approx(1.0)  # reported beside validity, not in it
    assert "wa_f64_cls_score" in S.leaderboard_row(S.score(sub, ref))
    (sub / "WA" / "float32" / "features.npz").unlink()
    with pytest.raises(S.ScoreError, match="features.npz"):
        S.score(sub, ref)


def test_yield_max_se_fails_an_underpowered_yield(example, tmp_path):
    ref, sub = _with_cfg(example, tmp_path, "WA", yield_max_se={"selected": 1e-4})
    r = S.score(sub, ref)["workflows"]["WA"]["float32"]
    y = r["yields"][0]
    assert y["underpowered"] and y["pass"] is False and r["validity"] == 0.0


def test_scaling_curve_and_crossover(example, tmp_path):
    ref, sub = _with_cfg(example, tmp_path, "WA", scaling_batches=[10, 1000, 100000])
    for dt in ("float32", "float64"):
        b = json.loads((sub / "WA" / dt / "bench.json").read_text())
        b.pop("warm_rate")
        b["scaling"] = [{"batch": 10, "rate": 5.0}, {"batch": 1000, "rate": 50.0}, {"batch": 100000, "rate": 500.0}]
        (sub / "WA" / dt / "bench.json").write_text(json.dumps(b))
    r = S.score(sub, ref)["workflows"]["WA"]["float32"]
    assert [c["multiplier"] for c in r["scaling"]] == pytest.approx([0.4, 4.0, 40.0])
    assert r["crossover_batch"] == 1000 and r["throughput_multiplier"] == pytest.approx(40.0)
    b = json.loads((sub / "WA" / "float32" / "bench.json").read_text())
    b["scaling"] = b["scaling"][:2]
    (sub / "WA" / "float32" / "bench.json").write_text(json.dumps(b))
    with pytest.raises(S.ScoreError, match="lacks batch"):
        S.score(sub, ref)


def _tracks(rng, n=2000, k=5):
    truth = rng.normal(size=(n, k))
    sigma = np.full((n, k), 0.2)
    params = truth + sigma * rng.normal(size=(n, k))
    return {"id": np.arange(n) + 7, "params": params, "sigma": sigma, "truth": truth, "good": np.ones(n, bool)}


def test_track_agreement_by_hand():
    rng = np.random.default_rng(2)
    st = _tracks(rng)
    close = {"id": st["id"][::-1], "params": st["params"][::-1] + 0.001, "sigma": st["sigma"][::-1]}
    r = S.track_agreement(st, close, 0.1, min_fraction=0.99)
    assert r["agreement"] == pytest.approx(1.0) and r["pass"] and r["good_fit"]["ratio"] == pytest.approx(1.0)
    assert abs(r["pulls"]["stock"]["width"][0] - 1) < 0.1 and abs(r["pulls"]["fast"]["width"][0] - 1) < 0.1
    far = {"id": st["id"][:1000], "params": st["params"][:1000] + 0.2, "good": np.r_[np.ones(900), np.zeros(100)]}
    r = S.track_agreement(st, far, 0.1, min_fraction=0.99)
    assert r["agreement"] == 0.0 and r["n_matched"] == 1000 and not r["pass"]
    assert r["good_fit"]["ratio"] == pytest.approx(0.45)
    with pytest.raises(S.ScoreError, match="unique"):
        S.track_agreement(st, {"id": np.zeros(3), "params": np.zeros((3, 5))})


def test_tracks_only_task(tmp_path):
    rng = np.random.default_rng(4)
    st = _tracks(rng, 500)
    ref, sub = tmp_path / "ref", tmp_path / "sub"
    (ref / "B3line" / "stock").mkdir(parents=True)
    np.savez(ref / "B3line" / "stock" / "tracks.npz", **st)
    (ref / "B3line" / "reference.json").write_text(json.dumps(
        {"stock_rate_per_slot": 200.0, "rate_unit": "tracks/s", "yield_tolerance": 0.0, "scaling_batches": [10, 1000],
         "track_agreement": {"file": "tracks.npz", "threshold_sigma": 0.1, "min_fraction": 0.95}}))
    for dt in ("float32", "float64"):
        d = sub / "B3line" / dt
        d.mkdir(parents=True)
        np.savez(d / "tracks.npz", id=st["id"], params=st["params"] + 0.005, good=st["good"])
        (d / "bench.json").write_text(json.dumps({"dtype": dt, "rate_unit": "tracks/s", "device": "toy-gpu",
                                                  "scaling": [{"batch": 10, "rate": 100.0},
                                                              {"batch": 1000, "rate": 4000.0}]}))
    r = S.score(sub, ref)["workflows"]["B3line"]["float64"]
    assert r["validity"] == pytest.approx(1.0) and r["n_hists"] == 0
    assert {y["yield"] for y in r["yields"]} == {"good_fit_ratio", "track_agreement"}
    assert r["crossover_batch"] == 1000 and r["throughput_multiplier"] == pytest.approx(20.0)


def test_groups_portability_and_capture_budget(example, tmp_path):
    ref, sub = tmp_path / "ref", tmp_path / "sub"
    layouts = {"B2dev": ("dev", "good"), "B2pa": ("place", "good"), "B2pb": ("place", "shifted"),
               "B2r41": ("design", "good")}
    for w, (part, src) in layouts.items():
        shutil.copytree(example / "reference" / "WA", ref / w)
        cfg = json.loads((ref / w / "reference.json").read_text())
        extra = {"group": "B2", "part": part, **({"capture_budget": 200000} if part == "design" else {})}
        (ref / w / "reference.json").write_text(json.dumps({**cfg, **extra}))
        shutil.copytree(example / "submissions" / src / "WA", sub / w)
        for dt in ("float32", "float64"):
            b = json.loads((sub / w / dt / "bench.json").read_text())
            (sub / w / dt / "bench.json").write_text(json.dumps({**b, "capture_entries": 150000}))
    res = S.score(sub, ref)
    g = res["groups"]["B2"]["float32"]
    assert g["dev_validity"] == pytest.approx(1.0) and g["n_place"] == 2 and g["n_design"] == 1
    pb = res["workflows"]["B2pb"]["float32"]["validity"]
    assert g["portability_ratio"] == pytest.approx((1.0 + pb) / 2) and g["portability_ratio"] < 1
    assert g["adaptation_score"] == pytest.approx(1.0)
    assert "b2_f32_portability" in S.leaderboard_row(res)
    b = json.loads((sub / "B2r41" / "float64" / "bench.json").read_text())
    (sub / "B2r41" / "float64" / "bench.json").write_text(json.dumps({**b, "capture_entries": 250000}))
    with pytest.raises(S.ScoreError, match="capture budget exceeded"):
        S.score(sub, ref)
