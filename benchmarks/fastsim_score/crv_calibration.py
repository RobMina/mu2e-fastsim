"""Generate CPU-only EX calibration fixtures and arms for the common fastsim scorer."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from crvfastsim.objective import ex_response as response
from crvfastsim.objective import ex_synthetic as synthetic

WORKFLOW = "CRV-EX-RESPONSE"
BOUNDARY = "fixed EX track/channel crossings -> accepted charge and time"
SIGMA = dict(sigma_log_pe=0.36, sigma_time_ns=2.0)
CASES = {"dev": (27, 0.00024, 0.0015), "final": (51, -0.00017, -0.0011)}


def _draw(seed, run, n, truth):
    return synthetic.draw(seed, run, n, truth)


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
    """Replay the held-out crossings with the same standardized noise draws, at the calibration's widths."""
    pred = response.predict(cal, rows, dtype=dtype)
    zq, zt = (z.astype(dtype) for z in noise)
    lpe = pred["log_pe"] + pred["sigma_log_pe"] * zq
    pe = np.exp(lpe)
    tm = pred["time_ns"] - pred["walk_ns_per_log_pe"] * (lpe - pred["log_pe"]) + dtype.type(cal["sigma_time_ns"]) * zt
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
    truth = synthetic.truth(rng, 8, attenuation=attenuation, slowness=slowness, **SIGMA)
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
            nc = len(truth["channel"])
            cal = dict(truth, log_scale=0., log_gain=[0.] * nc, time_zero_ns=0., time_offset_ns=[0.] * nc,
                       delta_attenuation_per_mm=0., delta_slowness_ns_per_mm=0.)
        else:
            cal = response.fit(train, include_missing=arm == "censored")
        fit_s = time.perf_counter() - start
        scores = response.diagnostics(cal, test)
        for field in ("delta_attenuation_per_mm", "delta_slowness_ns_per_mm", "log_scale", "time_zero_ns"):
            scores[field + "_abs_error"] = abs(cal[field] - truth[field])
        for field in ("log_gain", "time_offset_ns"):
            scores[field + "_rmse"] = float(np.sqrt(np.mean((np.asarray(cal[field]) - truth[field])**2)))
        if arm == "censored":
            scores["pulls"] = synthetic.pulls(cal, truth)
            scores["errors"] = {k: v for k, v in cal["errors"].items() if isinstance(v, float)}
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
