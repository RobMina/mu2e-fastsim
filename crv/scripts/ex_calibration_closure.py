"""Multi-seed closure of the censored EX response fit: pulls, model mismatch, width and walk sensitivity.

Synthetic, public seeds; prints summary statistics only (mean and spread of the pulls over the toys). Each toy has
8 channels x ``--events`` crossings (default 1200, i.e. 9,600 crossings, as the benchmark fixtures).

    python crv/scripts/ex_calibration_closure.py --toys 60 --out closure.json
"""

import argparse
import json
import time

import numpy as np

from crvfastsim.objective import ex_response as response
from crvfastsim.objective import ex_synthetic as synthetic

SQ, ST = 0.36, 2.0
PE_DEP = dict(sigma_log_pe=0.25, excess_noise=1.0)     # photostatistics-like truth: wider at low PE
CASES = {
    # name: (generator model, truth options, fit options)
    "model_const_fitted": ("lognormal", dict(), dict(walk=False)),
    "model_pedep_fitted": ("lognormal", PE_DEP, dict(walk=False)),
    "model_pedep_fitted_walk": ("lognormal", dict(PE_DEP, walk=0.8), dict()),
    "walk_truth_fit_without_walk": ("lognormal", dict(PE_DEP, walk=0.8), dict(walk=False)),
    "model_const_fixed_0.8x": ("lognormal", dict(), dict(sigma_log_pe=0.8 * SQ, sigma_time_ns=ST, fit_widths=False,
                                                         walk=False)),
    "model_const_fixed_1.0x": ("lognormal", dict(), dict(sigma_log_pe=SQ, sigma_time_ns=ST, fit_widths=False,
                                                         walk=False)),
    "model_const_fixed_1.2x": ("lognormal", dict(), dict(sigma_log_pe=1.2 * SQ, sigma_time_ns=ST, fit_widths=False,
                                                         walk=False)),
    "model_pedep_fit_const_width": ("lognormal", PE_DEP, dict(excess_noise=0., walk=False)),
    "mismatch_fixed": ("poisson", dict(), dict(sigma_log_pe=SQ, sigma_time_ns=ST, fit_widths=False, walk=False)),
    "mismatch_fit_const_width": ("poisson", dict(), dict(excess_noise=0., walk=False)),
    "mismatch_fitted": ("poisson", dict(), dict(walk=False)),
    "mismatch_fitted_walk": ("poisson", dict(walk=0.8), dict()),
    "mismatch_fitted_threshold_8.5": ("poisson", dict(), dict(walk=False, threshold_pe=8.5)),
    "mismatch_spe_fixed": ("poisson_spe", dict(), dict(sigma_log_pe=SQ, sigma_time_ns=ST, fit_widths=False,
                                                       walk=False)),
    "mismatch_spe_fitted": ("poisson_spe", dict(), dict(walk=False)),
    "positive_only": ("lognormal", dict(), dict(include_missing=False)),
}
KEYS = ("delta_attenuation_per_mm", "delta_slowness_ns_per_mm", "log_scale", "time_zero_ns", "sigma_log_pe",
        "sigma_time_ns", "walk_ns_per_log_pe", "excess_noise")


def stats(values):
    v = np.asarray(values, float)
    if not len(v):
        return None
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0., n=len(v),
                err_mean=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.)


def run_case(name, toys, events):
    model, topts, opts = CASES[name]
    pulls = {k: [] for k in KEYS}
    errs = {k: [] for k in ("delta_attenuation_per_mm", "delta_slowness_ns_per_mm", "log_gain_rmse",
                            "time_offset_ns_rmse", "sigma_log_pe", "sigma_time_ns", "walk_ns_per_log_pe",
                            "excess_noise")}
    cover = {k: [] for k in ("delta_attenuation_per_mm", "delta_slowness_ns_per_mm")}
    fails, times = 0, []
    for s in range(toys):
        truth = synthetic.truth(np.random.default_rng(5000 + s), 8, **dict(dict(sigma_log_pe=SQ, sigma_time_ns=ST),
                                                                           **topts),
                                attenuation=np.random.default_rng(7000 + s).uniform(-3e-4, 3e-4),
                                slowness=np.random.default_rng(8000 + s).uniform(-1.5e-3, 1.5e-3))
        rows, _ = synthetic.draw(6000 + s, 1, events, truth, model=model)
        fit_opts = dict(opts)
        if "threshold_pe" in fit_opts:
            # Integer counts: the fit's threshold half a count below the cut (continuity correction).
            rows = dict(rows, threshold_pe=np.full(len(rows["channel"]), fit_opts.pop("threshold_pe")))
        start = time.perf_counter()
        try:
            cal = response.fit(rows, **fit_opts)
        except (RuntimeError, ValueError):
            fails += 1
            continue
        times.append(time.perf_counter() - start)
        dev = synthetic.errors(cal, truth)
        for k in errs:
            errs[k].append(dev[k])
        for k, v in synthetic.pulls(cal, truth).items():
            pulls[k].append(v)
        for k in cover:
            e = (cal.get("errors") or {}).get(k)
            if e:
                cover[k].append(abs(dev[k]) < e)
    return dict(model=model, truth=topts, fit={k: v for k, v in opts.items()}, toys=toys, crossings=8 * events,
                fit_failures=fails, fit_time_s=stats(times),
                pulls={k: stats(v) for k, v in pulls.items() if v},
                errors={k: stats(v) for k, v in errs.items()},
                coverage_1sigma={k: float(np.mean(v)) for k, v in cover.items() if v})


def precision_check(events):
    truth = synthetic.truth(np.random.default_rng(11), 8, walk=0.8, **PE_DEP)
    rows, _ = synthetic.draw(77, 1, events, truth)
    cal = response.fit(rows)
    p64, p32 = response.predict(cal, rows, np.float64), response.predict(cal, rows, np.float32)
    return {k: float(np.max(np.abs(p64[k] - p32[k].astype(np.float64)))) for k in ("probability", "log_pe", "time_ns")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--toys", type=int, default=60)
    parser.add_argument("--events", type=int, default=1200)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    out = {name: run_case(name, args.toys, args.events) for name in args.cases}
    out["predict_float32_vs_float64_max_abs"] = precision_check(args.events)
    text = json.dumps(out, indent=1)
    if args.out:
        with open(args.out, "w") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
