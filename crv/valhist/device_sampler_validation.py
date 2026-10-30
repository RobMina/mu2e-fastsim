"""GPU physics validation and synchronized warm timing of the execution twin.

Run in the existing core/crvfastsim JAX environment, for example::

    python -m valhist.device_sampler_validation --out /path/to/data/sampler.json

The fourteen points, input seed 7, NumPy seed 0, JAX key 0, full sample
shape, quantiles and statistical thresholds reproduce core/docs/gpu_port.md
and core/tests/test_eloss_jax.py without importing pytest. Both JAX samplers
are checked against NumPy, and their same-key outputs are compared at
rtol=2e-12, atol=2e-13 MeV. No distributional draws are chunked. Default
N=4,000,000 is the original validation size; smaller --n is diagnostic only
and cannot certify the sampler. Statistical tolerances never scale with --n.

Each sampler is compiled/warmed separately before measuring its synchronized
calls, with keys, constants and inputs already on the GPU. Host transfers,
NumPy physics, statistics and compilation are outside warm timings. Per-point
speedups are reported separately: the acceptance throughput boundary is complete
transport, not every possible homogeneous sampler regime. This is a standalone
sampler certificate, not a near-stop transport-state certificate.
Only scalar metrics and the eight quantiles are emitted, never sample arrays.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


N_DEFAULT = 4_000_000
QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
RTOL = 2e-12
ATOL = 2e-13


def points(E, M):
    """Return the original fourteen material, momentum, length and cut points."""
    gas = M.Material("StrawGas", 0.001696, 158.62097145526,
                     {"Ar": 0.784057035890315, "O": 0.15700963928136, "C": 0.0589333248283256})

    def tk_step(mat, p):
        tk = (p * p + E.MMU ** 2) ** 0.5 - E.MMU
        return 0.05 * tk / float(E.dedx_restricted(mat, p, 1e9))

    return [
        ("gas 990eV 1mm 3GeV", gas, 3000., 1., 990e-6),
        ("gas 990eV 5mm 3GeV", gas, 3000., 5., 990e-6),
        ("gas 990eV 10mm 300MeV", gas, 300., 10., 990e-6),
        ("gas 990eV 1-10mm 0.2-10GeV", gas, (200., 1e4), (1., 10.), 990e-6),
        ("PS 87.2keV 20mm 3GeV", M.POLYSTYRENE, 3000., 20., .0872),
        ("PS 87.2keV 20mm 300MeV", M.POLYSTYRENE, 300., 20., .0872),
        ("concrete 1e9 100mm 200MeV (gauss)", M.CONCRETE_MARS, 200., 100., 1e9),
        ("concrete 1e9 500mm 1GeV (urban, mean/cut 2.5)", M.CONCRETE_MARS, 1000., 500., 1e9),
        ("concrete 1e9 50mm 3GeV (urban)", M.CONCRETE_MARS, 3000., 50., 1e9),
        ("concrete 1e9 5%tk 100MeV (range-out)", M.CONCRETE_MARS, 100., tk_step(M.CONCRETE_MARS, 100.), 1e9),
        ("concrete 1e9 5%tk 40MeV (range-out)", M.CONCRETE_MARS, 40., tk_step(M.CONCRETE_MARS, 40.), 1e9),
        ("PS 87.2keV 5%tk 60MeV (range-out)", M.POLYSTYRENE, 60., tk_step(M.POLYSTYRENE, 60.), .0872),
        ("steel 1e9 2mm 50MeV (gauss)", M.MILD_STEEL, 50., 2., 1e9),
        ("steel 1e9 0.1um 1.85MeV (gamma)", M.MILD_STEEL, 1.85, 1e-4, 1e9),
    ]


def inputs(np, p, length, n):
    """Reproduce the reference input generator, independently of sampler RNGs."""
    rng = np.random.default_rng(7)
    pp = np.exp(rng.uniform(np.log(p[0]), np.log(p[1]), n)) if isinstance(p, tuple) else np.full(n, p)
    ll = rng.uniform(length[0], length[1], n) if isinstance(length, tuple) else np.full(n, length)
    return pp, ll


def branches(np, E, mat, p, length, tcut):
    """Return NumPy Gauss/Gamma/Urban branch fractions."""
    _, beta2, _, gam, tmax = E.kinematics(p)
    cut = np.minimum(tcut, tmax)
    mean = E.dedx_restricted(mat, p, tcut) * length
    bohr = (mean >= E._BOHR_MIN * cut) & (tmax <= 2 * cut)
    sig = np.sqrt(np.clip((tmax / beta2 - .5 * cut) * E.TWOPI_MC2_RCL2 * length
                          * mat.electron_density_per_mm3, 0, None))
    sn = mean / np.where(sig > 0, sig, 1)
    return [float((bohr & (sn >= 2)).mean()), float((bohr & (sn < 2)).mean()), float((~bohr).mean())]


def distribution(np, stats, actual, ref):
    """Apply the original NumPy distribution criteria, failing closed on unusable metrics."""
    finite = bool(actual.shape == ref.shape and actual.dtype == np.float64
                  and np.isfinite(actual).all() and (actual >= 0).all()
                  and np.isfinite(ref).all() and (ref >= 0).all())
    result = {"checks": {"finite_nonnegative_float64": finite}}
    if not finite or not (actual > 0).any() or not (ref > 0).any() or ref.mean() <= 0:
        result["passed"] = False
        result["error"] = "Invalid, empty-positive or zero-mean distribution"
        return result
    p0r, p0a = float((ref == 0).mean()), float((actual == 0).mean())
    sigma = float(np.sqrt(2 * max(p0r, 1 / ref.size) * (1 - p0r) / ref.size))
    qr, qa = np.quantile(ref[ref > 0], QS), np.quantile(actual[actual > 0], QS)
    qrel = qa / qr - 1
    mean_rel = float(actual.mean() / ref.mean() - 1)
    rb = stats.ratio_band(actual, ref, q=(.05, .95), nbins=20, min_ref=400)
    band = rb["max_abs_dev_minus_2sigma"]
    result.update(mean_ref_mev=float(ref.mean()), mean_actual_mev=float(actual.mean()),
                  mean_rel=mean_rel, p0_ref=p0r, p0_actual=p0a, p0_sigma=sigma,
                  p0_abs_difference=abs(p0a - p0r), quantiles_ref_mev=qr.tolist(),
                  quantiles_actual_mev=qa.tolist(), quantiles_rel=qrel.tolist(),
                  band=rb["max_abs_dev"], band_minus_2sigma=band, band_bins_used=rb["nbins_used"])
    result["checks"].update(mean=bool(np.isfinite(mean_rel) and abs(mean_rel) < .005),
                            p0=bool(abs(p0a - p0r) <= 5 * sigma),
                            quantiles=bool(np.isfinite(qrel).all() and (np.abs(qrel) < .02).all()),
                            ratio_band=bool(band is not None and np.isfinite(band) and band <= .01))
    result["passed"] = all(result["checks"].values())
    return result


def paired(np, fast, core):
    """Report bounded-size same-key diagnostics with an explicit near-zero relative floor."""
    valid = bool(fast.shape == core.shape and fast.dtype == core.dtype == np.float64
                 and np.isfinite(fast).all() and np.isfinite(core).all()
                 and (fast >= 0).all() and (core >= 0).all())
    if not valid:
        return {"passed": False, "error": "Invalid paired outputs"}
    delta = np.abs(fast - core)
    relative = delta / np.maximum(np.abs(core), ATOL)
    scaled = delta / (ATOL + RTOL * np.abs(core))
    return {"passed": bool((scaled <= 1).all()), "rtol": RTOL, "atol_mev": ATOL,
            "max_abs_mev": float(delta.max()), "max_relative": float(relative.max()),
            "relative_denominator_floor_mev": ATOL,
            "max_tolerance_fraction": float(scaled.max()),
            "outside_tolerance": int((scaled > 1).sum()),
            "zero_mismatches": int(((fast == 0) != (core == 0)).sum())}


def timed(np, jax, fn, args, repeat, n):
    """Compile and warm this sampler separately, then time synchronized device-resident calls."""
    start = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    first = time.perf_counter() - start
    del out
    out = fn(*args)
    jax.block_until_ready(out)
    del out
    values = []
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        values.append(time.perf_counter() - start)
        del out
    out = fn(*args)
    jax.block_until_ready(out)
    host = np.asarray(jax.device_get(out)).copy()
    del out
    median = float(np.median(values))
    return {"compile_and_first_s": first, "warm_s": values, "median_warm_s": median,
            "steps_per_s": n / median, "warmup_calls": 2}, host


def dedx_checks(np, jax, E, EJ, fast, M, gas, device):
    """Preserve the original exact dE/dx grid over eight materials and three cuts."""
    p = np.geomspace(1., 1e5, 2001)
    pj = jax.device_put(p, device)
    rows = []
    for mat in (gas, M.POLYSTYRENE, M.CONCRETE_MARS, M.ALUMINIUM, M.COPPER, M.TUNGSTEN, M.MILD_STEEL, M.AIR):
        for tc in (990e-6, .0872, 1e9):
            ref = E.dedx_restricted(mat, p, tc)
            row = {"material": mat.name, "tcut_mev": tc}
            for name, module in (("core", EJ), ("fast", fast)):
                actual = np.asarray(module.dedx_restricted(mat, pj, tc))
                row[name] = {"passed": bool(np.isfinite(actual).all() and np.allclose(actual, ref, rtol=1e-12, atol=0)),
                             "max_abs_mev_per_mm": float(np.abs(actual - ref).max())}
            row["passed"] = row["core"]["passed"] and row["fast"]["passed"]
            rows.append(row)
    return rows


def run(args, result):
    """Run full-shape GPU validation with no silent CPU fallback."""
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    from mu2efastsim import eloss as E, eloss_jax as EJ, materials as M, stats
    from crvfastsim import eloss_transport_jax as fast

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("GPU backend required; CPU timing is not a GPU certificate")
    device = devices[0]
    result.update(device=str(device), device_kind=device.device_kind, jax_version=jax.__version__,
                  numpy_version=np.__version__, python_version=platform.python_version(),
                  core_source=str(Path(EJ.__file__).resolve()), fast_source=str(Path(fast.__file__).resolve()),
                  x64=bool(jax.config.jax_enable_x64), prng_impl=str(jax.config.jax_default_prng_impl),
                  preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
    pts = points(E, M)
    with jax.default_device(device):
        result["dedx"] = dedx_checks(np, jax, E, EJ, fast, M, pts[0][1], device)
        core_fn = jax.jit(EJ.sample_loss)
        fast_fn = jax.jit(fast.sample_loss)
        for label, mat, p, length, tc in pts:
            row = {"label": label, "passed": False}
            result["points"].append(row)
            try:
                pp, ll = inputs(np, p, length, args.n)
                row["branches_gauss_gamma_urban"] = branches(np, E, mat, pp, ll, tc)
                ref = E.sample_loss(np.random.default_rng(0), mat, pp, ll, tc)
                resident = jax.device_put((jax.random.key(0), EJ.consts(mat), pp, ll,
                                          np.float64(tc), np.float64(E.MMU)), device)
                jax.block_until_ready(resident)
                row["core_timing"], core = timed(np, jax, core_fn, resident, args.repeat, args.n)
                row["fast_timing"], actual = timed(np, jax, fast_fn, resident, args.repeat, args.n)
                row["core_vs_numpy"] = distribution(np, stats, core, ref)
                row["fast_vs_numpy"] = distribution(np, stats, actual, ref)
                row["same_key_fast_vs_core"] = paired(np, actual, core)
                speedup = row["core_timing"]["median_warm_s"] / row["fast_timing"]["median_warm_s"]
                row["median_warm_speedup"] = speedup
                row["accelerated"] = bool(np.isfinite(speedup) and speedup > 1)
                row["passed"] = all(row[k]["passed"] for k in
                                    ("core_vs_numpy", "fast_vs_numpy", "same_key_fast_vs_core"))
                del actual, core, ref, pp, ll, resident
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                print(f"{label}: {'PASS' if row['passed'] else 'FAIL'}", file=sys.stderr, flush=True)
                emit(args.out, result)
        coverage = np.asarray([r["branches_gauss_gamma_urban"] for r in result["points"]])
        result["branch_coverage_passed"] = bool((coverage > .99).any(axis=0).all())
    result["completed"] = True
    result["passed"] = bool(args.n >= N_DEFAULT and len(result["points"]) == 14
                            and result["branch_coverage_passed"]
                            and all(r["passed"] for r in result["dedx"])
                            and all(r["passed"] for r in result["points"]))


def json_safe(value):
    """Represent nonfinite diagnostic scalars as JSON null, never nonstandard NaN tokens."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def emit(path, result):
    """Write strict JSON; stdout is reserved for the final complete report."""
    text = json.dumps(json_safe(result), indent=2, allow_nan=False)
    if path is not None:
        path.write_text(text + "\n")
    return text


def main():
    """Parse the standalone CLI and return nonzero for every incomplete or failed validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", type=Path, help="JSON report path in the data area; JSON is also printed to stdout")
    args = parser.parse_args()
    if args.n <= 0 or args.repeat <= 0:
        parser.error("--n and --repeat must be positive")
    result = {"passed": False, "completed": False, "n": args.n, "repeat": args.repeat,
              "minimum_certification_n": N_DEFAULT, "full_shape_distribution_sampling": True,
              "seeds": {"inputs": 7, "numpy": 0, "jax": 0}, "quantiles": list(QS),
              "criteria": {"mean_abs_rel_lt": .005, "p0_binomial_sigma_le": 5,
                           "nonzero_quantile_abs_rel_lt": .02, "band_minus_2sigma_le": .01,
                           "same_key_rtol": RTOL, "same_key_atol_mev": ATOL},
              "timing_note": "Per-point sampler timing diagnostic; throughput gate is complete transport.", "points": []}
    try:
        if args.out is not None and not args.out.parent.is_dir():
            raise FileNotFoundError(f"Output parent must already exist: {args.out.parent}")
        run(args, result)
    except Exception as exc:
        result["passed"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
    try:
        print(emit(args.out, result), flush=True)
    except Exception as exc:
        result["passed"] = False
        result["report_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(json_safe(result), indent=2, allow_nan=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
