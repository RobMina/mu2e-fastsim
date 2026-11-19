"""Generate CPU-only EX calibration fixtures and arms for the common fastsim scorer."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from crvfastsim.objective import ex_response as response

WORKFLOW = "CRV-EX-RESPONSE"
BOUNDARY = "fixed EX track/channel crossings -> accepted charge and time"
SIGMA = dict(sigma_log_pe=0.36, sigma_time_ns=2.0)
CASES = {"dev": (27, 0.00024, 0.0015), "final": (51, -0.00017, -0.0011)}


def _draw(seed, run, n, truth):
    rng = np.random.default_rng(seed)
    nc = len(truth["log_gain"])
    chan = np.tile(np.arange(nc), n)
    distance = rng.uniform(100, 1700, len(chan))
    path = rng.uniform(18, 30, len(chan))
    q0 = np.log(0.65 * path) - distance / 2200
    t0 = 120 + distance / 160
    threshold = np.full(len(chan), 9.)
    window = np.full(len(chan), 4.)
    gain = np.asarray(truth["log_gain"])[chan]
    dt = np.asarray(truth["time_offset_ns"])[chan]
    q_noise = rng.normal(0, SIGMA["sigma_log_pe"], len(chan))
    t_noise = rng.normal(0, SIGMA["sigma_time_ns"], len(chan))
    q = q0 + truth["log_scale"] + gain - distance * truth["delta_attenuation_per_mm"] + q_noise
    t = t0 + truth["time_zero_ns"] + dt + distance * truth["delta_slowness_ns_per_mm"] + t_noise
    pe = np.exp(q)
    seen = (pe >= threshold) & (np.abs(t - t0) <= window)
    return dict(run=np.full(len(chan), run), subrun=np.repeat(np.arange(n) % 8, nc),
                event=np.repeat(np.arange(n), nc), channel=chan,
                nominal_log_pe=q0, nominal_time_ns=t0, distance_mm=distance, eligible=np.ones(len(chan), bool),
                threshold_pe=threshold, time_half_width_ns=window, seen=seen, pe=np.where(seen, pe, np.nan),
                time_ns=np.where(seen, t, np.nan)), (q_noise, t_noise)


def _arm(rows):
    use = rows["seen"]
    block = rows["subrun"]
    blocks = np.unique(block)

    def counts(values, edges):
        return dict(edges=edges.tolist(), shards={str(s): np.histogram(values[use & (block == s)], edges)[0].tolist()
                                                  for s in blocks})

    edges = np.linspace(100, 1700, 9)
    hists = dict(log_pe=counts(np.log(rows["pe"]), np.linspace(np.log(9), np.log(100), 21)),
                 time_residual_ns=counts(rows["time_ns"] - rows["nominal_time_ns"], np.linspace(-4, 4, 17)),
                 hit_efficiency=dict(edges=edges.tolist(), passed=np.histogram(rows["distance_mm"][use], edges)[0].tolist(),
                                     total=np.histogram(rows["distance_mm"], edges)[0].tolist()))
    return dict(hists=hists, yields={"seen": {"shards": {str(s): int(use[block == s].sum()) for s in blocks}}})


def _replay(cal, rows, noise, dtype):
    pred = response.predict(cal, rows, dtype=dtype)
    pe = np.exp(pred["log_pe"] + noise[0].astype(dtype))
    tm = pred["time_ns"] + noise[1].astype(dtype)
    threshold, nominal, window = (rows[k].astype(dtype) for k in ("threshold_pe", "nominal_time_ns", "time_half_width_ns"))
    seen = (pe >= threshold) & (np.abs(tm - nominal) <= window)
    return dict(rows, pe=np.where(seen, pe, np.nan), time_ns=np.where(seen, tm, np.nan), seen=seen)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(out, phase="dev", n=1200):
    """Write synthetic fit/holdout tables, reference tree and three submission trees."""
    out = Path(out)
    seed, attenuation, slowness = CASES[phase]
    rng = np.random.default_rng(seed)
    gain, dt = rng.normal(0, 0.09, 8), rng.normal(0, 0.4, 8)
    truth = dict(channel=list(range(8)), log_scale=-0.16, log_gain=(gain - gain.mean()).tolist(),
                 time_zero_ns=0.7, time_offset_ns=(dt - dt.mean()).tolist(),
                 delta_attenuation_per_mm=attenuation, delta_slowness_ns_per_mm=slowness, **SIGMA)
    train, _ = _draw(seed + 1, 1, n, truth)
    test, noise = _draw(seed + 2, 2, n, truth)
    ref = out / "reference" / WORKFLOW
    _write_json(ref / "reference.json", dict(stock_rate_per_slot=None, mode="shape", boundary=BOUNDARY,
                                            yield_tolerance=0.03, min_count=5, synthetic=True, phase=phase))
    _write_json(ref / "stock" / "hists.json", _arm(test))
    np.savez_compressed(out / "train.npz", **train)
    np.savez_compressed(out / "holdout.npz", **test)
    _write_json(out / "truth.json", truth)
    result = {}
    for arm in ("nominal", "positive_only", "censored"):
        start = time.perf_counter()
        if arm == "nominal":
            cal = dict(truth, log_scale=0., log_gain=[0.] * len(gain), time_zero_ns=0., time_offset_ns=[0.] * len(dt),
                       delta_attenuation_per_mm=0., delta_slowness_ns_per_mm=0.)
        else:
            cal = response.fit(train, **SIGMA, include_missing=arm == "censored")
        fit_s = time.perf_counter() - start
        scores = response.diagnostics(cal, test)
        for field in ("delta_attenuation_per_mm", "delta_slowness_ns_per_mm", "log_scale", "time_zero_ns"):
            scores[field + "_abs_error"] = abs(cal[field] - truth[field])
        for field in ("log_gain", "time_offset_ns"):
            scores[field + "_rmse"] = float(np.sqrt(np.mean((np.asarray(cal[field]) - truth[field])**2)))
        _write_json(out / "calibration" / f"{arm}.json", cal)
        result[arm] = scores
        for dtype in ("float32", "float64"):
            path = out / "submissions" / arm / WORKFLOW / dtype
            start = time.perf_counter()
            replay = _replay(cal, test, noise, np.dtype(dtype))
            wall = time.perf_counter() - start
            _write_json(path / "hists.json", _arm(replay))
            _write_json(path / "bench.json", dict(device="CPU", dtype=dtype, boundary=BOUNDARY, fit_dtype="float64",
                                                 fit_s=fit_s, replay_s=wall, crossings=len(replay["seen"])))
    _write_json(out / "calibration_scores.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--phase", choices=CASES, default="dev")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.out, args.phase), indent=2))


if __name__ == "__main__":
    main()
