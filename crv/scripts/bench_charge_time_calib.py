"""Compare profiled and dense weighted channel fits on synthetic CRV calibration data."""
import argparse
import json
import time

import numpy as np

from crvfastsim.objective.charge_time_calib import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--per-channel", type=int, default=100)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    nchan = args.channels
    idx = np.repeat(np.arange(nchan), args.per_channel)
    d = rng.uniform(100, 1600, len(idx))
    gains = rng.normal(0, .1, nchan)
    offsets = rng.normal(0, .5, nchan)
    gains -= gains.mean()
    offsets -= offsets.mean()
    atten, slow = 2.e-4, 5.e-4
    x = .1 + gains[idx] - atten * d + rng.normal(0, .03, len(d))
    t = .7 + offsets[idx] + slow * d + rng.normal(0, .1, len(d))
    pe0 = rng.uniform(50, 80, len(d))
    t0 = rng.uniform(80, 120, len(d))
    pe, tm = pe0 * np.exp(x), t0 + t
    train = np.arange(len(d)) % 5 != 0
    test = ~train

    def profiled():
        return fit(idx[train], d[train], pe[train], pe0[train], tm[train], t0[train])

    def dense():
        design = np.eye(nchan)[idx[train]]
        design = np.column_stack((d[train], design))
        target = np.column_stack((np.log(pe[train]) - np.log(pe0[train]), tm[train] - t0[train]))
        return np.linalg.lstsq(design, target, rcond=None)[0]

    runs = {"profiled": profiled, "dense": dense}
    output = {k: fn() for k, fn in runs.items()}
    ms = {k: [] for k in runs}
    for _ in range(9):
        for k, fn in runs.items():
            start = time.perf_counter()
            fn()
            ms[k].append(1000 * (time.perf_counter() - start))
    p = output["profiled"]
    q_int = p["log_scale"] + np.asarray(p["log_gain"])
    t_int = p["time_zero_ns"] + np.asarray(p["time_offset_ns"])
    q_slope, t_slope = -p["delta_attenuation_per_mm"], p["delta_slowness_ns_per_mm"]
    diff = np.column_stack((np.r_[q_slope, q_int], np.r_[t_slope, t_int])) - output["dense"]
    eq = x[test] - (q_int[idx[test]] + q_slope * d[test])
    et = t[test] - (t_int[idx[test]] + t_slope * d[test])
    print(json.dumps(dict(channels=nchan, train=int(train.sum()), test=int(test.sum()),
                          median_ms={k: float(np.median(v)) for k, v in ms.items()},
                          dense_max_diff=float(np.abs(diff).max()), attenuation_error_pct=100 * (-q_slope / atten - 1),
                          slowness_error_pct=100 * (t_slope / slow - 1), heldout_rms_log_pe=float(np.sqrt((eq**2).mean())),
                          heldout_rms_ns=float(np.sqrt((et**2).mean()))), indent=2))


if __name__ == "__main__":
    main()
