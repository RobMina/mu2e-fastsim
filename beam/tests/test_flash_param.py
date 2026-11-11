"""Synthetic checks of the parametric flash estimate (no stock results are used)."""
import json

import numpy as np
import pytest

from beamfastsim import flash_param as P

DTYPES = [np.float32, np.float64]


def _stack(n=5, rin=0.0, rout=75.0, t=0.1, z0=6271.0, dz=22.0):
    z = z0 + (np.arange(n) - n // 2) * dz
    return P.Stack(z, np.full(n, rin), np.full(n, rout), np.full(n, t))


@pytest.mark.parametrize("dtype", DTYPES)
def test_feature_analytic_annulus_and_dtype(dtype):
    s = _stack(rin=10.0, rout=60.0, t=0.2)
    expect = 5 * 0.2 * (np.exp(-100 / 800) - np.exp(-3600 / 800))
    got = P.feature(s, 20.0, dtype=dtype)
    assert got.dtype == dtype
    assert got == pytest.approx(expect, rel=1e-6 if dtype is np.float32 else 1e-12)


@pytest.mark.parametrize("dtype", DTYPES)
def test_z_weight_counts_downstream_material_more(dtype):
    up, down = _stack(n=1, z0=5800.0), _stack(n=1, z0=6800.0)
    ratio = P.feature(down, 30.0, 1000.0, dtype=dtype) / P.feature(up, 30.0, 1000.0, dtype=dtype)
    assert ratio == pytest.approx(np.e, rel=1e-6)
    assert P.feature(down, 30.0, None, dtype=dtype) == P.feature(up, 30.0, None, dtype=dtype)


def test_hole_larger_than_beam_removes_the_material():
    assert P.feature(_stack(rin=200.0, rout=240.0), 20.0) < 1e-20
    assert P.feature(_stack(rin=0.0, rout=500.0, t=0.1), 20.0) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [dict(sigma=0.0), dict(sigma=20.0, lam=-1.0)])
def test_invalid_shape_parameters_raise(bad):
    with pytest.raises(ValueError):
        P.feature(_stack(), **bad)


def _designs(rng, n):
    return [_stack(n=7, rin=rng.uniform(0, 60), rout=rng.uniform(70, 200), t=rng.uniform(0.02, 0.4),
                   z0=rng.uniform(5800, 6700)) for _ in range(n)]


@pytest.mark.parametrize("dtype", DTYPES)
def test_calibration_recovers_synthetic_constants(dtype):
    rng = np.random.default_rng(3)
    stacks = _designs(rng, 40)
    truth = P.FlashParam(2.5e-6, 1.4e-6, 30.0, 1000.0)
    y = truth(stacks)
    p = P.calibrate(stacks, y, dtype=dtype)
    assert (p.sigma, p.lam) == (30.0, 1000.0)
    assert p.c0 == pytest.approx(truth.c0, rel=1e-4) and p.c1 == pytest.approx(truth.c1, rel=1e-4)
    oof = P.kfold(stacks, y, k=5, dtype=dtype)
    m = P.metrics(oof, y)
    assert m["spearman"] > 0.999 and m["rel_rms"] < 1e-4


def test_float32_and_float64_agree_on_noisy_ranking():
    rng = np.random.default_rng(4)
    stacks = _designs(rng, 30)
    y = P.FlashParam(2.5e-6, 1.4e-6, 30.0, 1000.0)(stacks) * rng.lognormal(0, 0.05, 30)
    p32, p64 = (P.calibrate(stacks, y, dtype=d) for d in DTYPES)
    assert (p32.sigma, p32.lam) == (p64.sigma, p64.lam)
    assert np.allclose(P.predict(stacks, p32, np.float32), P.predict(stacks, p64), rtol=1e-5)


def test_calibrate_rejects_bad_targets():
    stacks = _designs(np.random.default_rng(5), 3)
    with pytest.raises(ValueError):
        P.calibrate(stacks, [1e-6, -1e-6, 2e-6])
    with pytest.raises(ValueError):
        P.calibrate(stacks[:2], [1e-6, 2e-6])


def test_spearman_handles_ties_and_constant_input():
    assert P.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert P.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert P.spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.894427191)
    assert np.isnan(P.spearman([1, 1, 1], [1, 2, 3]))


def test_metrics_budget_agreement():
    m = P.metrics([1.0, 3.0, 5.0], [1.5, 1.8, 6.0], budget=2.0)
    assert m["budget_agreement"] == pytest.approx(2 / 3)
    assert m["n"] == 3


def _overlay(path, radii, holes, half):
    path.write_text("\n".join([
        f"vector<double> stoppingTarget.radii = {{{','.join(map(str, radii))}}};",
        f"vector<double> stoppingTarget.holeRadii = {{{','.join(map(str, holes))}}};",
        f"vector<double> stoppingTarget.halfThicknesses = {{{','.join(map(str, half))}}};",
        "double stoppingTarget.deltaZ = 22.222222;", "double stoppingTarget.z0InMu2e = 5871.0;",
        "bool hasTarget = true;"]))


@pytest.mark.parametrize("flag", [[], ["--float32"]])
def test_cli_calibrates_and_predicts_from_overlays(tmp_path, flag):
    rng = np.random.default_rng(6)
    truth = P.FlashParam(2.5e-6, 1.4e-6, 30.0, 1000.0)
    lines = ["name\tgeom\tflash"]
    for i in range(8):
        r = rng.uniform(80, 200)
        h, t = rng.uniform(0, 60), rng.uniform(0.02, 0.3)
        _overlay(tmp_path / f"d{i}.txt", [r] * 5, [h] * 5, [t] * 5)
        s = P.stack_from_offline(f"d{i}.txt", str(tmp_path))
        assert np.allclose(s.thickness, 2 * t) and s.z[2] == pytest.approx(5871.0)
        lines.append(f"d{i}\td{i}.txt\t{truth([s])[0]:.9e}")
    (tmp_path / "t.tsv").write_text("\n".join(lines))
    out = tmp_path / "o.json"
    P.main(["--table", str(tmp_path / "t.tsv"), "--offline-root", str(tmp_path), "--kfold", "4", "--out", str(out), *flag])
    d = json.loads(out.read_text())
    assert d["dtype"] == ("float32" if flag else "float64")
    assert d["in_sample"]["rel_rms"] < 1e-4 and d["kfold"]["spearman"] > 0.99
    (tmp_path / "p.json").write_text(json.dumps(d["params"]))
    P.main(["--table", str(tmp_path / "t.tsv"), "--offline-root", str(tmp_path), "--params", str(tmp_path / "p.json"),
            "--out", str(tmp_path / "o2.json"), *flag])
    d2 = json.loads((tmp_path / "o2.json").read_text())
    assert d2["fixed_params"]["spearman"] == pytest.approx(1.0)
