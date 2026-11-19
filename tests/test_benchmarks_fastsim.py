"""benchmarks/fastsim_score: statistics, flag rule, yields, both-dtype requirement, ROOT conversion, Codabench bundle,
and B1 (CALOG4): CE peak fit, power rule, classifier score, calorimeter arm definitions."""
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
from benchmarks.fastsim_score import calo  # noqa: E402
from benchmarks.fastsim_score import classifier as C  # noqa: E402
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
    assert json.loads((tmp_path / "s.json").read_text())["n_workflows"] == 3
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
    assert "both dtypes" in res["missing"]["WB"]["float32"] and res["n_arms_scored"] == 5
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
    mults = [a["throughput_multiplier"] for by in res["workflows"].values() for a in by.values()
             if a["throughput_multiplier"] is not None]
    assert len(mults) == 4 and res["throughput_geomean"] == pytest.approx(float(np.exp(np.mean(np.log(mults)))))


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
    for k in ("W1", "W8", "W9", "CRVLY", "PTGT", "CALOG4"):
        assert f"| {k} |" in t


# ---------------------------------------------------------------- B1 (CALOG4)

def test_auc_by_hand():
    assert C.auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)
    assert C.auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == pytest.approx(0.0)
    assert C.auc(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == pytest.approx(0.5)
    assert C.auc(np.array([2.0]), np.array([2.0])) == pytest.approx(0.5)      # ties count half


def test_classifier_same_and_separated():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(4000, 3)), rng.normal(size=(4000, 3))
    same = C.classifier_score(a, b, seed=1)
    assert abs(same["auc"] - 0.5) < 0.04 and same["cls_score"] > 0.9 and same["n_terms"] == 3 + 6
    assert C.classifier_score(a, b, seed=1) == same                                 # fixed seed: deterministic
    wide = rng.normal(size=(4000, 3)) * np.array([1.0, 1.6, 1.0])                  # a width difference: quadratic terms
    assert C.classifier_score(a, wide, seed=1)["auc"] > 0.6
    far = C.classifier_score(a, a + 5.0, seed=1)
    assert far["auc"] > 0.999 and far["cls_score"] == pytest.approx(0.0, abs=2e-3)
    with pytest.raises(ValueError, match=">= 50"):
        C.classifier_score(a[:20], b[:20])


def test_gauss_core_matches_scipy_curve_fit():
    opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(8)
    e = np.r_[rng.normal(87.6, 2.4, 20000), 87.6 - rng.exponential(8.0, 5000)]
    edges = calo.PEAK_EDGES
    h, _ = np.histogram(e, edges)
    got = S.gauss_core(edges, h, 75.0, 3.0)
    # CAL1's loop with scipy, on the same bins
    c = 0.5 * (edges[1:] + edges[:-1])
    mu, sig = float(np.median(e[e > 75.0])), 3.0
    above = c > 75.0
    cum = np.cumsum(h[above])
    mu = float(c[above][np.searchsorted(cum, 0.5 * cum[-1])])
    f = lambda x, a, m, s: a * np.exp(-0.5 * ((x - m) / s) ** 2)  # noqa: E731
    for _ in range(6):
        w = (c > mu - 1.5 * sig) & (c < mu + 2.0 * sig) & (h > 0)
        p, cov = opt.curve_fit(f, c[w], h[w], p0=[h[w].max(), mu, sig], sigma=np.sqrt(h[w]), absolute_sigma=True)
        mu, sig = float(p[1]), float(abs(p[2]))
    assert got["peak"] == pytest.approx(mu, abs=1e-6) and got["width"] == pytest.approx(sig, abs=1e-6)
    assert got["peak_error"] == pytest.approx(float(np.sqrt(cov[1, 1])), rel=1e-4)
    assert abs(got["peak"] - 87.6) < 0.5          # the low tail pulls the core fit down a little


def test_peak_ratio_bootstrap_and_fit_errors():
    rng = np.random.default_rng(9)
    edges = calo.PEAK_EDGES
    mk = lambda m: {f"s{i}": np.histogram(rng.normal(m, 2.4, 3000), edges)[0].tolist() for i in range(20)}  # noqa
    ref, same, low = ({"edges": edges.tolist(), "shards": mk(m)} for m in (87.6, 87.6, 87.0))
    r = S.peak_ratio(ref, same, 0.002, "p", {"start_above": 75.0, "start_width": 3.0})
    assert r["se_method"] == "bootstrap_200_of_20_shards" and 0.0001 < r["se"] < 0.001 and r["pass"]
    assert abs(r["ratio"] - 1) < 4 * r["se"]
    lo = S.peak_ratio(ref, low, 0.002, "p", {"start_above": 75.0, "start_width": 3.0})
    assert lo["ratio"] == pytest.approx(87.0 / 87.6, abs=0.001) and not lo["pass"]
    one = {"edges": edges.tolist(), "counts": np.sum(list(ref["shards"].values()), axis=0).tolist()}
    two = {"edges": edges.tolist(), "counts": np.sum(list(same["shards"].values()), axis=0).tolist()}
    fe = S.peak_ratio(one, two, 0.002, "p", {"start_above": 75.0, "start_width": 3.0})
    assert fe["se_method"] == "fit_errors" and fe["se"] == pytest.approx(r["se"], rel=0.5)
    # the bootstrap agrees with the spread of the ratio over independent samples of the same size
    reps = [S.peak_ratio({"edges": edges.tolist(), "shards": mk(87.6)}, {"edges": edges.tolist(), "shards": mk(87.6)},
                         0.002, "p", {"start_above": 75.0, "start_width": 3.0, "n_boot": 20})["ratio"] for _ in range(30)]
    assert float(np.std(reps, ddof=1)) == pytest.approx(r["se"], rel=0.5)


def test_shape_from_hits_by_hand():
    out = calo.shape_from_hits([0, 0, 1, -1], [3.0, 1.0, 5.0, 9.0], [0.0, 4.0, 10.0, 0.0], [0.0, 0.0, 2.0, 0.0], 2)
    assert out["seed_energy"].tolist() == [3.0, 5.0]
    # cluster 0: centre x = 1, mxx = (3 * 1 + 1 * 9) / 4 = 3; cluster 1: one hit, zero moments
    assert out["mxx"] == pytest.approx([3.0, 0.0]) and out["myy"] == pytest.approx([0.0, 0.0])
    assert out["mxy"] == pytest.approx([0.0, 0.0])


def test_calo_arm_definitions():
    cl = {"event": np.array([0, 0, 1, 3]), "energy": np.array([88.0, 4.0, 60.0, 12.0]),
          "time": np.array([700.0, 701.0, 900.0, 1200.0]), "x": np.array([400.0, 420.0, -500.0, 0.0]),
          "y": np.array([0.0, 0.0, 0.0, 450.0]), "size": np.array([6, 1, 4, 2]),
          "seed_energy": np.array([44.0, 4.0, 30.0, 6.0]), "mxx": np.ones(4), "myy": np.ones(4), "mxy": np.zeros(4)}
    hits = {"event": np.array([0, 0, 1, 2]), "energy": np.array([44.0, 0.7, 30.0, 2.2])}
    h, f = calo.arm(cl, hits, n_events=4, block=2)
    assert set(h["hists"]) == set(calo.HIST_EDGES) and list(h["hists"]["cluster_energy"]["shards"]) == ["b00000",
                                                                                                         "b00001"]
    assert sum(h["hists"]["cluster_energy"]["shards"]["b00000"]) == 3                 # every cluster
    assert sum(h["hists"]["cluster_time"]["shards"]["b00000"]) == 2                   # above 10 MeV only
    assert h["hists"]["clusters_per_event"]["shards"]["b00001"][:2] == [1.0, 1.0]     # events 2 (none), 3 (one)
    assert h["yields"]["events_cluster_gt10"]["shards"] == {"b00000": 2, "b00001": 1}
    assert sum(h["peaks"]["ce_peak"]["shards"]["b00000"]) == 2 and sum(h["peaks"]["ce_peak"]["shards"]["b00001"]) == 0
    assert f["seed_frac"] == pytest.approx([0.5, 1.0, 0.5, 0.5]) and set(f) == set(calo.FEATURES) | {"event"}
    with pytest.raises(ValueError, match="outside the population"):
        calo.arm(dict(cl, event=np.array([0, 0, 1, 9])), hits, n_events=4)


def test_calo_population_pairs_by_event_id(monkeypatch):
    files = {"a": (np.array([[1, 1, 5], [1, 1, 6]]),
                   {"event": np.array([0, 1]), "energy": np.array([50.0, 60.0])},
                   {"event": np.array([0, 1, 1]), "energy": np.array([1.0, 2.0, 3.0])}),
             "b": (np.array([[1, 2, 7]]), {"event": np.array([0]), "energy": np.array([70.0])},
                   {"event": np.array([0]), "energy": np.array([4.0])})}
    monkeypatch.setattr(calo, "read_ntuple", lambda f: files[f])
    ids, cl, ht = calo.population(["a", "b"], np.array([[1, 2, 7], [1, 1, 5]]))
    assert ids.tolist() == [[1, 2, 7], [1, 1, 5]]
    assert cl["event"].tolist() == [0, 1] and cl["energy"].tolist() == [70.0, 50.0]
    assert sorted(zip(ht["event"].tolist(), ht["energy"].tolist())) == [(0, 4.0), (1, 1.0)]
    with pytest.raises(ValueError, match="not in the ntuples"):
        calo.population(["a"], np.array([[9, 9, 9]]))


def test_calog4_good_and_shifted(example):
    ref = example / "reference"
    good = S.score(example / "submissions" / "good", ref)
    shifted = S.score(example / "submissions" / "shifted", ref)
    for dt in S.DTYPES:
        g = good["workflows"]["CALOG4"][dt]
        assert g["validity"] == pytest.approx(1.0) and g["power_ok"] is True and g["n_hists"] == 8
        pk = next(y for y in g["yields"] if y["yield"] == "ce_peak")
        assert pk["kind"] == "peak" and pk["se"] <= calo.MAX_SE and pk["se_method"].startswith("bootstrap")
        assert g["cls_score"] > 0.85 and g["classifier"]["n_features"] == len(calo.FEATURES)
        sh = shifted["workflows"]["CALOG4"][dt]
        assert sh["validity"] < 0.5 and sh["cls_score"] < g["cls_score"] - 0.2
        assert not next(y for y in sh["yields"] if y["yield"] == "ce_peak")["pass"]
    assert good["cls_score"] == pytest.approx(np.mean([good["workflows"]["CALOG4"][d]["cls_score"] for d in S.DTYPES]))
    assert good["classifier_floor"]["CALOG4"]["cls_score"] > 0.85
    row = S.leaderboard_row(good)
    assert {"cls_score", "calog4_f32_cls", "calog4_f64_power"} <= set(row) and row["calog4_f64_power"] == 1.0


def test_power_rule_fails_a_small_sample(example, tmp_path):
    ref = tmp_path / "ref"
    shutil.copytree(example / "reference", ref)
    cfg = json.loads((ref / "CALOG4" / "reference.json").read_text())
    (ref / "CALOG4" / "reference.json").write_text(json.dumps({**cfg, "max_se": {"ce_peak": 1e-6}}))
    r = S.score(example / "submissions" / "good", ref)["workflows"]["CALOG4"]["float64"]
    pk = next(y for y in r["yields"] if y["yield"] == "ce_peak")
    assert pk["pass_tolerance"] is True and pk["power_ok"] is False and pk["pass"] is False
    assert r["power_ok"] is False and r["validity"] == pytest.approx(0.5)
    (ref / "CALOG4" / "reference.json").write_text(json.dumps({**cfg, "max_se": {"no_such": 0.1}}))
    with pytest.raises(S.ScoreError, match="no_such"):
        S.score(example / "submissions" / "good", ref)


def test_features_required_and_checked(example, tmp_path):
    sub = tmp_path / "sub"
    shutil.copytree(example / "submissions" / "good", sub)
    (sub / "CALOG4" / "float32" / "features.npz").unlink()
    with pytest.raises(S.ScoreError, match="features.npz"):
        S.score(sub, example / "reference")
    z = dict(np.load(example / "submissions" / "good" / "CALOG4" / "float32" / "features.npz"))
    z["mxx"][0] = np.nan
    np.savez(sub / "CALOG4" / "float32" / "features.npz", **z)
    with pytest.raises(S.ScoreError, match="finite"):
        S.score(sub, example / "reference")
    del z["mxx"]
    np.savez(sub / "CALOG4" / "float32" / "features.npz", **z)
    with pytest.raises(S.ScoreError, match="mxx"):
        S.score(sub, example / "reference")


def test_mu2eg4_host_throughput_and_memory(example, tmp_path):
    sub = tmp_path / "sub"
    shutil.copytree(example / "submissions" / "good", sub)
    for dt in S.DTYPES:
        b = json.loads((sub / "CALOG4" / dt / "bench.json").read_text())
        (sub / "CALOG4" / dt / "bench.json").write_text(json.dumps(
            {**b, "host": "mu2eg4", "warm_rate": 50.0, "baseline_rate": 30.0, "memory_mb_per_thread": 310,
             "edge_rule": "D_edge 40 mm"}))
    r = S.score(sub, example / "reference")["workflows"]["CALOG4"]["float32"]
    assert r["throughput_multiplier"] == pytest.approx(50 / 30) and r["throughput_basis"] == "same node, model off"
    assert r["memory_ok"] is False and r["edge_rule"] == "D_edge 40 mm"
    b = json.loads((sub / "CALOG4" / "float32" / "bench.json").read_text())
    del b["baseline_rate"]
    (sub / "CALOG4" / "float32" / "bench.json").write_text(json.dumps(b))
    with pytest.raises(S.ScoreError, match="baseline_rate"):
        S.score(sub, example / "reference")
