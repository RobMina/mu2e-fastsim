"""Audit a real ROOT propagation table and optional public stock lookup observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import uproot

from calofastsim.propagation import build_cdf, propagation_time, validate_table


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_axis(axis):
    """Reconstruct ROOT TAxis regular-bin arithmetic; not observed ROOT calls."""
    n = int(axis.member("fNbins"))
    low, high = float(axis.member("fXmin")), float(axis.member("fXmax"))
    edges = np.asarray(axis.member("fXbins"), dtype=np.float64)
    if n <= 0 or not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("invalid ROOT axis")
    if len(edges):
        if len(edges) != n + 1 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("invalid variable ROOT axis")
        centers = edges[:-1] + 0.5 * np.diff(edges)
        width = edges[1] - edges[0]
    else:
        width = (high - low) / n
        centers = low + (np.arange(n, dtype=np.float64) + 0.5) * width
    widths = np.diff(edges) if len(edges) else np.full(n, width)
    return np.float32(width), centers.astype(np.float32), {
        "bins": n, "low": low, "high": high, "variable": bool(len(edges)),
        "bin_width_min": float(widths.min()), "bin_width_max": float(widths.max()),
    }


def read_table(path, histogram):
    with uproot.open(path) as source:
        hist = source[histogram]
        if hist.classname != "TH2F":
            raise ValueError("stock requires TH2F, not another histogram storage class")
        contents = np.asarray(hist.values(flow=False), dtype=np.float32)
        flow = np.asarray(hist.values(flow=True), dtype=np.float32)
        dz, _, depth_axis = root_axis(hist.axis(0))
        _, times, time_axis = root_axis(hist.axis(1))
    cdf, times, dz = validate_table(build_cdf(contents), times, dz)
    if contents.shape != (depth_axis["bins"], time_axis["bins"]):
        raise ValueError("ROOT matrix/axis shape mismatch")
    return contents, cdf, times, dz, {
        "depth_axis": depth_axis, "time_axis": time_axis,
        "nonzero_flow_bins_ignored_by_stock": int(np.count_nonzero(flow) - np.count_nonzero(contents)),
    }


def source_cdf(contents):
    """Independent source-order double-expression then float-storage recurrence."""
    counts = np.asarray(contents, dtype=np.float32)
    total = np.full(len(counts), np.float32(1e-6))
    columns = []
    for column in counts.T:
        total = (total.astype(np.float64) + column.astype(np.float64)).astype(np.float32)
        columns.append(total.copy())
    return np.column_stack(columns) / total[:, None]


def probes(cdf, dz):
    """All CDF thresholds/neighbors and all depth boundaries/neighbors, arraywise."""
    ndepth, ntime = cdf.shape
    centers = (np.arange(ndepth, dtype=np.float32) + np.float32(.5)) * dz
    threshold = cdf.ravel()
    draws = np.clip(np.column_stack((np.nextafter(threshold, np.float32(0)), threshold,
                                    np.nextafter(threshold, np.float32(1)))).ravel(), 0, 1)
    depth = np.repeat(centers, ntime * 3)
    rows = np.repeat(np.arange(ndepth), ntime * 3)
    boundary = np.arange(ndepth + 1, dtype=np.float32) * dz
    edge_depth = np.column_stack((np.nextafter(boundary, np.float32(-np.inf)), boundary,
                                 np.nextafter(boundary, np.float32(np.inf)))).ravel()
    edge_depth = np.r_[np.float32(-100) * dz, edge_depth, np.float32(ndepth + 100) * dz]
    edge_u = np.array([0, np.nextafter(np.float32(0), np.float32(1)), .25, .5, .75, 1], dtype=np.float32)
    edges = np.repeat(edge_depth, len(edge_u))
    edge_rows = np.clip(np.trunc(edges / dz), 0, ndepth - 1).astype(np.int64)
    return np.r_[depth, edges], np.r_[draws, np.tile(edge_u, len(edge_depth))], np.r_[rows, edge_rows]


def expected_bins(cdf, rows, draws):
    result = np.empty(len(rows), dtype=np.int64)
    for row in range(len(cdf)):
        selected = rows == row
        result[selected] = np.minimum(np.searchsorted(cdf[row], draws[selected], side="left"), cdf.shape[1] - 1)
    return result


def compare(contents, cdf, times, dz, *, backend="numpy", jit=False):
    reference = source_cdf(contents)
    depth, draws, rows = probes(reference, dz)
    expected = times[expected_bins(reference, rows, draws)]
    fn = propagation_time
    runtime = {"python": platform.python_version(), "numpy": np.__version__}
    if backend == "jax":
        import jax
        from calofastsim.propagation_jax import propagation_time as fn
        runtime.update(jax=jax.__version__, devices=[str(device) for device in jax.devices()],
                       x64=bool(jax.config.x64_enabled))
        if jit:
            fn = jax.jit(fn)
    elif jit:
        raise ValueError("JIT requires JAX")
    observed = np.asarray(fn(depth, draws, cdf, times, dz))
    mismatches = int(np.count_nonzero(observed.view(np.uint32) != expected.view(np.uint32)))
    return {
        "source_reconstructed_cdf_exact": bool(np.array_equal(cdf, reference)),
        "cdf_values": int(cdf.size), "probe_count": len(depth),
        "supplied_uniform_lookup_exact": mismatches == 0, "lookup_bit_mismatches": mismatches,
        "backend": backend, "mode": "jit" if jit else "eager", "runtime": runtime,
        "depth_range_mm": [float(depth.min()), float(depth.max())],
        "time_centers_range_ns": [float(times.min()), float(times.max())],
        "zero_content_depth_slices": int(np.count_nonzero(~np.any(contents, axis=1))),
        "cdf_repeated_adjacent_values": int(np.count_nonzero(np.diff(cdf, axis=1) == 0)),
        "dz_float32_mm": float(dz),
        "stock_cache_directly_observed": False, "stock_sampling_executed": False,
        "clhep_rng_validated": False, "stochastic_response_validated": False,
        "scope": "actual ROOT histogram; source-reconstructed TAxis/CDF and supplied float32 uniforms only",
    }


def compare_stock(path, contents, cdf, times, dz, *, expected_sha256, expected_file,
                  histogram="h_zposTime", backend="numpy", jit=False):
    """Check pinned stock outputs, observed ROOT metadata and exact probe coverage."""
    if sha256(path) != expected_sha256:
        raise ValueError("stock control SHA256 mismatch")
    with uproot.open(path) as source:
        metadata = source["propagation/metadata"].arrays(["nz", "nt", "dz", "refractive_index", "file", "histogram"], library="np")
        bins = source["propagation/bins"].arrays(["iz", "it", "content", "time", "probe_threshold"], library="np")
        observed = source["propagation/probes"].arrays(library="np")
    if list(metadata["file"]) != [expected_file] or list(metadata["histogram"]) != [histogram]:
        raise ValueError("observed stock file/histogram metadata mismatch")
    if (len(metadata["nz"]) != 1 or metadata["nz"][0] != len(cdf) or
            metadata["nt"][0] != len(times) or metadata["dz"][0] != dz or
            not np.isfinite(metadata["refractive_index"][0]) or metadata["refractive_index"][0] <= 0):
        raise ValueError("stock metadata differs from histogram")
    if not (np.array_equal(bins["iz"], np.repeat(np.arange(len(cdf)), len(times))) and
            np.array_equal(bins["it"], np.tile(np.arange(len(times)), len(cdf)))):
        raise ValueError("stock ROOT bin registry/order mismatch")
    for name, expected in (("content", contents.ravel()), ("time", np.tile(times, len(cdf))),
                           ("probe_threshold", cdf.ravel())):
        if not np.array_equal(bins[name].view(np.uint32), expected.view(np.uint32)):
            raise ValueError(f"stock ROOT observation/reconstructed probe {name} mismatch")
    depth, draws, _ = probes(cdf, dz)
    for name, expected in (("depth", depth), ("uniform", draws)):
        if not np.array_equal(observed[name].view(np.uint32), expected.view(np.uint32)):
            raise ValueError(f"stock {name} probe coverage mismatch")
    n = len(depth)
    if not (np.array_equal(observed["draws_before"], np.arange(n)) and
            np.array_equal(observed["draws_after"], np.arange(1, n + 1))):
        raise ValueError("stock calls did not consume exactly one supplied draw per probe")
    fn = propagation_time
    if backend == "jax":
        import jax
        from calofastsim.propagation_jax import propagation_time as fn
        if jit:
            fn = jax.jit(fn)
    prediction = np.asarray(fn(depth, draws, cdf, times, dz))
    mismatch = int(np.count_nonzero(prediction.view(np.uint32) != observed["time"].view(np.uint32)))
    return {"stock_control_sha256": sha256(path), "stock_control_sha256_verified": True,
            "stock_resolved_file": expected_file, "stock_histogram": histogram, "stock_lookup_probe_count": n,
            "stock_lookup_bit_mismatches": mismatch, "stock_public_lookup_exact": mismatch == 0,
            "stock_sampling_executed": True, "stock_engine": "prescribed CLHEP NonRandomEngine",
            "stock_cache_directly_observed": False, "clhep_rng_validated": False,
            "stochastic_response_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--histogram", default="h_zposTime")
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--stock-source", type=Path, required=True)
    parser.add_argument("--stock-source-sha256", required=True)
    parser.add_argument("--stock-control", type=Path)
    parser.add_argument("--stock-control-sha256")
    parser.add_argument("--stock-resolved-file")
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--jax-jit", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stock_arguments = (args.stock_control, args.stock_control_sha256, args.stock_resolved_file)
    if any(item is not None for item in stock_arguments) and not all(item is not None for item in stock_arguments):
        parser.error("--stock-control, --stock-control-sha256 and --stock-resolved-file are required together")
    if args.out.exists():
        raise FileExistsError(args.out)
    if sha256(args.input) != args.input_sha256 or sha256(args.stock_source) != args.stock_source_sha256:
        raise ValueError("input/source SHA256 mismatch")
    contents, cdf, times, dz, axes = read_table(args.input, args.histogram)
    result = compare(contents, cdf, times, dz, backend=args.backend, jit=args.jax_jit)
    result.update(axes, input_sha256=sha256(args.input), stock_source_sha256=sha256(args.stock_source),
                  checker_sha256=sha256(__file__), histogram=args.histogram,
                  numpy_backend_sha256=sha256(Path(__file__).parents[1] / "calofastsim/propagation.py"))
    if args.backend == "jax":
        result["jax_backend_sha256"] = sha256(Path(__file__).parents[1] / "calofastsim/propagation_jax.py")
    if args.stock_control is not None:
        result.update(compare_stock(args.stock_control, contents, cdf, times, dz,
                                    expected_sha256=args.stock_control_sha256, expected_file=args.stock_resolved_file,
                                    histogram=args.histogram, backend=args.backend, jit=args.jax_jit))
        result["scope"] = "unmodified stock public build-plus-lookup with prescribed uniforms; not production RNG/response"
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if (not result["source_reconstructed_cdf_exact"] or not result["supplied_uniform_lookup_exact"] or
            not result.get("stock_public_lookup_exact", True)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
