"""Exposure laws, integer Poisson inputs, historical RNG regression and fail-closed export."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.veto.exposure import CE_CONDITIONED, LEGACY, PBI, ExposureContract, llrint_pbi
from crvfastsim.veto.mixframes import MIXERS, PBTIMES, draw_intensity, mix
from crvfastsim.veto.replay import CE_EVENT_TYPES, EVENT_TYPES, export, replay_tables, write_replay


def historical_draw(rng, n, extended_mean=PBI["extended_mean"], sdf=PBI["sdf"], cut_max=PBI["cut_max"]):
    s = math.sqrt(-math.log(sdf))
    mu = math.log(extended_mean) - 0.5 * s * s
    out = np.empty(0)
    while len(out) < n:
        x = rng.lognormal(mu, s, 2 * (n - len(out)) + 16)
        out = np.r_[out, x[x < cut_max]]
    return out[:n]


def historical_mix(pools, n, rng, scale=1.0):
    intensity = historical_draw(rng, n) * scale
    pb = rng.choice(PBTIMES, n)
    rows, means, nsp = [], {}, 0
    for name, pool in pools.items():
        eff, fac = MIXERS[name]
        nf = rng.poisson(intensity * eff * fac)
        means[name] = float(nf.mean())
        if pool.stepped == 0 or nf.sum() == 0:
            continue
        k = rng.binomial(nf, pool.stepped / pool.frames)
        ev = np.repeat(np.arange(n), k)
        hit = rng.integers(0, pool.stepped, len(ev))
        if not len(hit):
            continue
        cnt = pool.count[hit]
        idx = np.repeat(pool.start[hit] - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
        d = pool.steps.iloc[idx].copy()
        d["event"] = (np.repeat(ev, cnt) + 1).astype(np.int64)
        d["subrun"], d["run"] = 0, 1430
        d["sp"] = np.where(d.sp.to_numpy() >= 0, np.repeat(nsp + np.arange(len(hit)) * 100000, cnt) + d.sp.to_numpy(), -1)
        nsp += len(hit) * 100000
        rows.append(d)
    steps = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return steps, np.arange(1, n + 1, dtype=np.int64), -pb, {
        "mean_frames_per_event": means, "mean_intensity": float(intensity.mean())}, intensity


def synthetic_pools(stepped=True):
    return {name: SimpleNamespace(frames=7, stepped=2 if stepped else 0,
                                 start=np.array([0, 2]), count=np.array([2, 1]),
                                 steps=pd.DataFrame({"run": [1, 1, 2], "subrun": [2, 2, 3], "event": [3, 3, 4],
                                                     "sp": [1, -1, 2], "pdg": [13, 13, 11], "bar": [1, 2, 3]}))
            for name in MIXERS}


@pytest.mark.parametrize("seed", [0, 17, 9876])
@pytest.mark.parametrize("scale", [1.0, 0.37])
def test_legacy_mixture_and_rng_are_exact(seed, scale):
    pools = synthetic_pools()
    old_rng, new_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    old = historical_mix(pools, 23, old_rng, scale)
    new = mix(pools, 23, new_rng, scale, return_events=True)
    pd.testing.assert_frame_equal(old[0], new[0])
    np.testing.assert_array_equal(old[1], new[1])
    np.testing.assert_array_equal(old[2], new[2])
    assert old[3] == new[3]
    np.testing.assert_array_equal(old[4], new[4].intensity)
    assert list(new[4]) == list(EVENT_TYPES)
    np.testing.assert_array_equal(old_rng.random(20), new_rng.random(20))


@pytest.mark.parametrize("n", [0, 1, 1000])
def test_draw_default_exact_even_with_rejections(n):
    old_rng, new_rng = np.random.default_rng(11), np.random.default_rng(11)
    params = {"extended_mean": 10.0, "sdf": 0.4, "cut_max": 7.0}
    np.testing.assert_array_equal(draw_intensity(new_rng, n, **params), historical_draw(old_rng, n, **params))
    np.testing.assert_array_equal(old_rng.random(5), new_rng.random(5))


def test_shifted_distribution_against_truncated_analytic_moments():
    contract = ExposureContract(CE_CONDITIONED, extended_mean=12.0, sdf=0.25, cut_max=40.0)
    base = ExposureContract(LEGACY, extended_mean=12.0, sdf=0.25, cut_max=40.0)
    assert contract.logmean == base.logmean + base.sigma**2
    x = contract.draw_raw(np.random.default_rng(51), 200000)
    mu, s, cut = contract.logmean, contract.sigma, contract.cut_max
    phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
    moment = lambda k: math.exp(k * mu + 0.5 * k*k*s*s) * phi((math.log(cut)-mu-k*s*s)/s) / phi((math.log(cut)-mu)/s)
    mean, variance = moment(1), moment(2) - moment(1)**2
    assert np.all((x > 0) & (x < cut))
    assert abs(x.mean() - mean) < 6 * math.sqrt(variance / len(x))
    assert abs(x.var() - variance) < 0.02 * variance
    assert contract.record()["density"].startswith("x*f_LN")


def test_rounding_is_ties_to_even_and_not_truncation():
    np.testing.assert_array_equal(llrint_pbi([0, 0.5, 1.5, 2.5, 3.5, 4.6]), [0, 0, 2, 2, 4, 5])
    np.testing.assert_array_equal(llrint_pbi([np.nextafter(0.5, 0), np.nextafter(0.5, 1)]), [0, 1])


@pytest.mark.parametrize("raw", [[-1], [np.nan], [np.inf], [float(2**63)]])
def test_invalid_rounding(raw):
    with pytest.raises(ValueError):
        llrint_pbi(raw)


def test_ce_rounds_before_poisson_and_retains_counts_without_extra_rng():
    class RecordingRng:
        def __init__(self):
            self.rng = np.random.default_rng(5)
            self.means = []

        def lognormal(self, mean, sigma, n):
            assert mean == ExposureContract(CE_CONDITIONED).logmean
            return np.resize([1.5, 2.5, 3.5, 4.6], n)

        def choice(self, *args):
            return self.rng.choice(*args)

        def poisson(self, means):
            self.means.append(means.copy())
            return self.rng.poisson(means)

    rng = RecordingRng()
    result = mix(synthetic_pools(False), 4, rng, 2.0, return_events=True, exposure_mode=CE_CONDITIONED)
    ev = result[-1]
    np.testing.assert_array_equal(ev.intensity_raw, [3.0, 5.0, 7.0, 9.2])
    np.testing.assert_array_equal(ev.intensity_rounded, [3, 5, 7, 9])
    for (name, (eff, fac)), means in zip(MIXERS.items(), rng.means):
        np.testing.assert_array_equal(means, ev.intensity_rounded.to_numpy() * eff * fac)
        assert ev[f"n_frames_{name}"].dtype == np.int64
    assert result[3]["exposure_contract"]["parameters"]["intensity_scale"] == 2.0
    a, b = np.random.default_rng(8), np.random.default_rng(8)
    short = mix(synthetic_pools(), 10, a, exposure_mode=CE_CONDITIONED)
    full = mix(synthetic_pools(), 10, b, exposure_mode=CE_CONDITIONED, return_events=True)
    pd.testing.assert_frame_equal(short[0], full[0])
    assert short[3] == full[3]
    np.testing.assert_array_equal(a.random(5), b.random(5))


@pytest.mark.parametrize("kwargs", [
    {"mode": "CE"}, {"extended_mean": 0}, {"extended_mean": -1}, {"extended_mean": np.nan},
    {"sdf": 0}, {"sdf": -1}, {"sdf": 1.1}, {"sdf": np.inf},
    {"cut_max": 0}, {"cut_max": -1}, {"cut_max": np.inf},
    {"intensity_scale": 0}, {"intensity_scale": -1}, {"intensity_scale": np.nan},
    {"sdf": 1, "extended_mean": 10, "cut_max": 5},
    {"extended_mean": 1e20, "cut_max": 1, "sdf": 0.99},
    {"mode": CE_CONDITIONED, "cut_max": float(2**63)},
])
def test_bad_parameters_fail_without_drawing(kwargs):
    with pytest.raises(ValueError):
        ExposureContract(**kwargs)


@pytest.mark.parametrize("n", [-1, 0.5, True])
def test_invalid_sample_size(n):
    with pytest.raises(ValueError):
        draw_intensity(np.random.default_rng(1), n)


def test_degenerate_distribution_and_json_contract():
    c = ExposureContract(CE_CONDITIONED, extended_mean=8, sdf=1, cut_max=10)
    np.testing.assert_array_equal(c.draw_raw(np.random.default_rng(1), 3), np.full(3, math.exp(math.log(8))))
    assert json.loads(json.dumps(c.record(), allow_nan=False))["mode"] == CE_CONDITIONED


def test_ce_replay_event_types_validation_and_empty_events(tmp_path):
    result = mix(synthetic_pools(False), 12, np.random.default_rng(3), return_events=True, exposure_mode=CE_CONDITIONED)
    events = result[-1]
    tables = replay_tables(pd.DataFrame(), events, exposure_mode=CE_CONDITIONED)
    assert set(tables["events"]) == set(CE_EVENT_TYPES)
    path = tmp_path / "synthetic.root"
    assert write_replay(path, tables)["events"] == 12
    with uproot.open(path) as f:
        assert f["dump/events/intensity_raw"].typename == "double"
        assert f["dump/events/intensity_rounded"].typename == "int64_t"
        for name in MIXERS:
            assert f[f"dump/events/n_frames_{name}"].typename == "int64_t"
    assert set(replay_tables(pd.DataFrame(), events)["events"]) == set(EVENT_TYPES)
    bad = events.copy()
    bad.loc[0, "intensity_rounded"] += 1
    with pytest.raises(ValueError, match="disagree"):
        replay_tables(pd.DataFrame(), bad, exposure_mode=CE_CONDITIONED)
    for value in [-1, 0.5, float(2**63)]:
        bad = events.copy()
        col = f"n_frames_{next(iter(MIXERS))}"
        bad[col] = bad[col].astype(float)
        bad.loc[0, col] = value
        with pytest.raises(ValueError, match="integer exposure"):
            replay_tables(pd.DataFrame(), bad, exposure_mode=CE_CONDITIONED)


def legacy_tables(folder):
    import pyarrow as pa
    import pyarrow.parquet as pq
    for name in MIXERS:
        out = folder / name
        out.mkdir(parents=True)
        df = pd.DataFrame({"subrun": pd.Series(dtype="int64"), "event": pd.Series(dtype="int64")})
        meta = {"dataset": name, "frames": 10, "stepped_frames": 0, "steps": 0}
        table = pa.Table.from_pandas(df).replace_schema_metadata({b"crvfastsim": json.dumps(meta).encode()})
        pq.write_table(table, out / "steps.parquet")


def test_export_fails_closed_for_legacy_ce_but_legacy_schema_stays(tmp_path):
    folder, output = tmp_path / "tables", tmp_path / "ce.root"
    legacy_tables(folder)
    with pytest.raises(ValueError, match="crv-cat-v2"):
        export(folder, 10, 5, output, exposure_mode=CE_CONDITIONED)
    assert not output.exists() and not output.with_suffix(".json").exists()
    manifest = export(folder, 10, 5, tmp_path / "legacy.root")
    assert manifest["schema"] == "crv-mixed-replay-v1"
    assert "legacy_unconditional_v1" in manifest["scope"]
    assert "exposure_contract" not in manifest
    with uproot.open(tmp_path / "legacy.root") as f:
        assert set(f["dump/events"].keys()) == set(EVENT_TYPES)


@pytest.mark.parametrize("with_steps", [False, True])
def test_ce_export_with_synthetic_validated_v2_library(tmp_path, with_steps):
    from crvfastsim.veto.catsteps_v2 import F32, convert
    folder = tmp_path / "v2"
    for i, name in enumerate(MIXERS):
        directory = folder / name
        directory.mkdir(parents=True)
        source = tmp_path / f"synthetic-{i}.root"
        ids = {"run": np.array([i+1], np.uint32), "subrun": np.array([0], np.uint32), "event": np.array([1], np.uint32)}
        empty_ids = {k: v[:0] for k, v in ids.items()}
        with uproot.recreate(source) as f:
            f["dump/events"] = ids | {"ngen": np.array([0], np.int32), "nsteps": np.array([int(with_steps)], np.int32)}
            f["dump/gen"] = empty_ids | {"idx": np.array([], np.int32), "pdg": np.array([], np.int32)}
            particle_ids = ids if with_steps else empty_ids
            f["dump/particles"] = particle_ids | {"key": np.ones(int(with_steps), np.int64),
                                                  "pdg": np.full(int(with_steps), 13, np.int32)}
            f["dump/steps"] = particle_ids | {k: np.ones(int(with_steps), np.int64) for k in ("sp", "bar")} | {
                k: np.zeros(int(with_steps), np.float64) for k in [*F32, "t0", "t1"]}
        convert(source, name, directory / "steps.parquet",
                source_art={"identity": f"synthetic-{i}", "uri": f"synthetic://source/{i}", "sha256": f"{i+1:064x}"},
                extraction={"id": f"test-{i}", "command": "synthetic fixture", "software_version": "test",
                            "fcl_sha256": "a"*64, "dumper_sha256": "b"*64, "write_steps": True,
                            "event_selection": "all", "source_count": 1})
    out = tmp_path / "ce.root"
    report = export(folder, 20, 15, out, exposure_mode=CE_CONDITIONED)
    assert report["schema"] == "crv-mixed-replay-v2"
    assert report["exposure_contract"]["mode"] == CE_CONDITIONED
    assert report["library_contract"]["schema"] == "crv-cat-v2"
    assert {"exposure.py", "catsteps_v2.py"}.issubset(report["code_sha256"])
    with uproot.open(out) as f:
        ev = pd.DataFrame(f["dump/events"].arrays(library="np"))
        total_frames = ev[[f"n_frames_{name}" for name in MIXERS]].to_numpy().sum()
        assert f["dump/steps"].num_entries == (total_frames if with_steps else 0)
        np.testing.assert_array_equal(ev.intensity_rounded, np.rint(ev.intensity_raw).astype(np.int64))
        for name in MIXERS:
            assert ev[f"n_frames_{name}"].mean() == report["mean_frames_per_event"][name]
    assert json.loads(out.with_suffix(".json").read_text())["exposure_contract"] == report["exposure_contract"]


def test_ce_export_rejects_schema_tag_without_census_and_provenance(tmp_path):
    import pyarrow.parquet as pq
    folder = tmp_path / "pretend-v2"
    legacy_tables(folder)
    for name in MIXERS:
        path = folder / name / "steps.parquet"
        table = pq.read_table(path)
        meta = json.loads(table.schema.metadata[b"crvfastsim"])
        meta["schema"] = "crv-cat-v2"
        pq.write_table(table.replace_schema_metadata({b"crvfastsim": json.dumps(meta).encode()}), path)
    out = tmp_path / "rejected.root"
    with pytest.raises(ValueError, match="source_art|provenance"):
        export(folder, 5, 6, out, exposure_mode=CE_CONDITIONED)
    assert not out.exists() and not out.with_suffix(".json").exists()
