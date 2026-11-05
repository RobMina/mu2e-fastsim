"""Measure the frozen CE shower route and cesob proxy on identical saved primaries."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np

from .compare import apply_response, paired_hist, paired_mean, paired_ratio
from .handoff import from_cechain
from .library import load
from .response import cluster_sum


def evaluate(result, data, field, library, backend, seed, timing_options):
    """Frozen library + deterministic response; all generated events remain in denominators."""
    from beamfastsim import cechain

    begin = time.perf_counter()
    handoff = from_cechain(result, field, library, np.random.default_rng(seed))
    handoff_seconds = time.perf_counter() - begin
    query, steps = handoff["entries"], handoff["steps"]
    n = len(result.fate)
    condition = {k.removeprefix("condition_"): v[query["event_row"]] for k, v in data.items() if k.startswith("condition_")}
    begin = time.perf_counter()
    cluster = apply_response(backend, steps, query, condition, data["lru"], **timing_options)
    response_seconds = time.perf_counter() - begin
    visible = cluster_sum(steps, len(query["energy"]), accepted=steps["mapped"])["energy"]
    out = {k: np.full((n,) + v.shape[1:], np.nan) for k, v in cluster.items()}
    out["energy"][:] = 0
    for k, v in cluster.items():
        out[k][query["event_row"]] = v
    out["visible"] = np.zeros(n)
    out["visible"][query["event_row"]] = visible
    denominator = float(steps["energy_birks"].sum())
    info = dict(tracker_entries=int(np.sum(result.fate == cechain.PLANE)), disk_front_entries=len(query["energy"]),
                fallback_entries=int(steps["fallback"].sum()), unsupported_entries=int((~steps["supported"]).sum()),
                css_rows=len(steps["entry"]), unmapped_css=int((~steps["mapped"]).sum()),
                unmapped_energy_fraction=float(steps["energy_birks"][~steps["mapped"]].sum() / denominator) if denominator else None,
                handoff_seconds=handoff_seconds, response_sum_seconds=response_seconds)
    return out, info


def compare(arms, event_rows):
    group = event_rows // 100
    names = ("fast_visible", "fast_response", "proxy_wall", "proxy_vacuum", "stock_visible", "stock_response")
    masks = {k: arms[k] > 10 if "proxy" not in k else arms[k].astype(bool) for k in names}
    result = {"counts": {k: int(v.sum()) for k, v in masks.items()}, "events": len(event_rows)}
    for a, b in (("fast_response", "stock_response"), ("fast_visible", "stock_visible"),
                 ("proxy_wall", "stock_visible"), ("proxy_vacuum", "stock_visible"),
                 ("fast_visible", "proxy_wall")):
        difference = paired_mean(masks[a], masks[b], group)
        ratio = paired_ratio(masks[a], masks[b], group)
        result[a + "_vs_" + b] = dict(**difference, ratio=ratio["ratio"], ratio_error=ratio["error"],
                                      fast_only=int(np.sum(masks[a] & ~masks[b])), stock_only=int(np.sum(~masks[a] & masks[b])))
    result["energy_mean"] = paired_mean(arms["fast_response"], arms["stock_response"], group)
    result["energy_shape"] = paired_hist(arms["fast_response"], arms["stock_response"], group, np.arange(0., 111., 5.))
    return result


def main():
    from beamfastsim import cechain, cesob, simpleconfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--maps", required=True, nargs="+")
    parser.add_argument("--response-source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", choices=("development", "heldout", "all"), required=True)
    parser.add_argument("--backend", choices=("numpy", "device"), required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit", type=int, help="bounded development smoke only")
    parser.add_argument("--lanes", type=int, default=1024)
    parser.add_argument("--stock-events-per-second", type=float, default=23.9)
    parser.add_argument("--digitization-start", type=float, required=True)
    parser.add_argument("--digitization-end", type=float, required=True)
    parser.add_argument("--digitization-buffer", type=float, required=True)
    parser.add_argument("--umbrella-commit", required=True)
    parser.add_argument("--core-commit", required=True)
    args = parser.parse_args()
    if args.repeat < 1 or (args.limit is not None and (args.arm != "development" or args.limit <= 0)):
        parser.error("positive repeats; limit is permitted only for development smoke")
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    library, metadata = load(args.library)
    with np.load(args.inputs, allow_pickle=False) as source:
        data = {k: source[k] for k in source.files}
    rows = np.arange(len(data["p0"]))
    selected = rows < metadata["split_row"] if args.arm == "development" else rows >= metadata["split_row"]
    if args.arm == "all":
        selected[:] = True
    rows = rows[selected][:args.limit]
    if len(rows) == 0:
        parser.error("empty selected source event range")
    data = {k: v if k == "lru" else v[rows] for k, v in data.items()}
    geom = cechain.from_offline(str(args.geometry), str(args.geometry.parent))
    config = simpleconfig.read(str(args.geometry), str(args.geometry.parent))
    calo = cesob.calo_from_config(config)
    wall = cesob.calo_wall_from_config(config, calo)
    field = cechain.field_maps(args.maps)
    spec = importlib.util.spec_from_file_location("saved_response_backend", args.response_source)
    response = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(response)
    versions = {"numpy": np.__version__, "python": platform.python_version(), "machine": platform.machine(), "platform": platform.platform()}
    if args.backend == "device":
        import jax
        from beamfastsim import cechain_jax
        jax.config.update("jax_enable_x64", True)
        if jax.default_backend() != args.device:
            raise RuntimeError(f"requested {args.device}, got {jax.devices()}")
        device_field = cechain_jax.device_maps(args.maps, np.float64)
        versions.update(jax=jax.__version__, devices=[str(d) for d in jax.devices()])
    load_seconds = time.perf_counter() - started
    options = dict(start=args.digitization_start, end=args.digitization_end, buffer=args.digitization_buffer)
    measurements = []
    arrays = None
    for repeat in range(args.repeat):
        begin = time.perf_counter()
        if args.backend == "device":
            result = cechain_jax.transport(jax.random.PRNGKey(args.seed), geom, device_field,
                                          data["x0"], data["u0"], data["p0"], data["t0"], dtype=np.float64,
                                          lanes=args.lanes, shrink=32, unroll=args.device == "gpu")
            jax.effects_barrier()
        else:
            result = cechain.transport(np.random.default_rng(args.seed), geom, field,
                                       data["x0"], data["u0"], data["p0"], data["t0"], backend="numpy")
        transport_seconds = time.perf_counter() - begin
        fast, info = evaluate(result, data, field, library, response, args.seed + 1, options)
        info.update(repeat=repeat, transport_seconds=transport_seconds,
                    chain_seconds=transport_seconds + info["handoff_seconds"] + info["response_sum_seconds"])
        measurements.append(info)
        if repeat == 0:
            arrays = {"fast_response": fast["energy"], "fast_visible": fast["visible"], "fast_time": fast["time"],
                      "fast_position": fast["position"], "stock_visible": data["stock_visible"], "stock_response": data["stock_response"],
                      "fate": result.fate, "tracker_x": result.x, "tracker_u": result.u, "tracker_T": result.T, "tracker_t": result.t}
            np.savez_compressed(args.out / "chain.npz", event_row=rows, **arrays)
        else:
            if not np.array_equal(arrays["fast_response"], fast["energy"]):
                raise RuntimeError("same-seed timing repeat changed the result")
        print(json.dumps(info), flush=True)
        (args.out / "timing.json").write_text(json.dumps(dict(load_seconds=load_seconds, measurements=measurements), indent=2) + "\n")
    plane = result.fate == cechain.PLANE
    for label, material in (("wall", wall), ("vacuum", None)):
        begin = time.perf_counter()
        accepted = np.zeros(len(rows), bool)
        if plane.any():
            accepted[plane] = cesob.calo_accept(field, calo, result.x[plane], result.u[plane], result.T[plane],
                                                wall=material, rng=np.random.default_rng(args.seed + 2), backend="numpy")
        arrays["proxy_" + label] = accepted
        measurements[0]["proxy_" + label + "_seconds"] = time.perf_counter() - begin
    report = compare(arrays, rows)
    warm = measurements[1:] if len(measurements) > 1 else []
    median = float(np.median([m["chain_seconds"] for m in warm])) if warm else None
    rate = len(rows) / median if median else None
    report.update(arm=args.arm, backend=args.backend, seed=args.seed, source_split=metadata["split"],
                  error_method="paired 100-source-event block jackknife; library training uncertainty excluded",
                  model="CE primary transport + vacuum helix to disk front + frozen sampled CSS + deterministic response cluster sum",
                  timing=dict(load_seconds=load_seconds, measurements=measurements, warm_chain_seconds=median,
                              warm_generated_per_second=rate, stock_generated_per_second=args.stock_events_per_second,
                              stock_grid_slot_equivalent=rate / args.stock_events_per_second if rate else None,
                              boundary_note="fast primary-only route to cluster sum; stock rate is full generated-CE simulation, not matched reconstruction",
                              process_elapsed_seconds=time.perf_counter() - started),
                  versions=versions, provenance={k: str(v) for k, v in vars(args).items()})
    (args.out / "numbers.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "chain.npz", event_row=rows, **arrays)
    print(json.dumps({k: v for k, v in report.items() if k != "energy_shape"}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
