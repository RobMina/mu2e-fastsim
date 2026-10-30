"""Stage-0 benchmark of the existing runchain: cached CRY tables to pulses and coincidences."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

from .. import chain, geometry, tables, tier1b_assets
from . import runchain

BOUNDARY = "cached CRY gen tables -> transport -> tier1b -> response -> digis -> pulses + coincidences"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def integer(a, name):
    a = np.asarray(a)
    if a.size and (a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a > np.iinfo(np.int64).max)):
        raise ValueError(f"{name}: expected nonnegative int64-compatible integers")
    return a.astype(np.int64)


def rows(a):
    a = np.ascontiguousarray(a, dtype=np.int64)
    return a.view(np.dtype([(f"k{i}", "<i8") for i in range(a.shape[1])])).reshape(-1)


def event_keys(gen):
    """Unique full event keys; reject identities that the existing chain cannot represent safely."""
    if not set(tables.KEY).issubset(gen.columns) or not len(gen):
        raise ValueError("empty gen table or missing full run/subrun/event keys")
    keys = np.column_stack([integer(gen[k].to_numpy(), k) for k in tables.KEY])
    keys = np.unique(keys, axis=0)
    if np.any(keys[:, 2] >= 2 ** 20):
        raise ValueError("event exceeds the chain's 20-bit event field")
    if np.any(keys[:, 1] > (np.iinfo(np.int64).max // 1_000_000 >> 20) - 1):
        raise ValueError("subrun overflows the chain's packed channel/FEB labels")
    packed = (keys[:, 1] << 20) | keys[:, 2]
    if len(np.unique(packed)) != len(keys):
        raise ValueError("run collision in the chain's packed subrun/event key")
    return keys, packed


def fingerprint(arrays):
    """Hash named array dtype, shape and bytes, independent of NPZ ZIP metadata."""
    h = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        if a.dtype.hasobject:
            raise ValueError(f"object array is not a reproducible table: {name}")
        h.update(json.dumps([name, a.dtype.str, a.shape], separators=(",", ":")).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def require_complete(values):
    for name, value in values.items():
        if isinstance(value, dict):
            require_complete(value)
        elif any(word in name.lower() for word in ("truncat", "overflow", "dropped")) and np.any(value):
            raise ValueError(f"incomplete output: {name}")


def detail_arrays(keys, outs):
    """Validate raw detailed outputs and flatten only their dictionary nesting, not their values."""
    keys = integer(keys, "generated keys")
    if keys.ndim != 2 or keys.shape[1] != 3 or not len(keys):
        raise ValueError("empty or malformed selected generated universe")
    packed = (keys[:, 1] << 20) | keys[:, 2]
    order = np.argsort(packed)
    universe = packed[order]
    if len(np.unique(universe)) != len(keys):
        raise ValueError("ambiguous selected generated universe")
    arrays = {"generated_keys": keys, "generated_packed": packed, "chunk_count": np.asarray(len(outs))}
    counts = dict(clusters=0, members=0, pulses=0, fallback_events=0)
    sectors, pulse_ids = {}, []

    def full(ev, name):
        ev = integer(ev, name)
        if ev.ndim != 1 or not np.all(np.isin(ev, universe)):
            raise ValueError(f"{name}: output outside generated universe")
        return keys[order[np.searchsorted(universe, ev)]]

    for i, out in enumerate(outs):
        require_complete(out)
        required = {"cl_event", "cl_type", "cl_pos", "cl_pe", "cl_start", "cl_end", "cl_slope", "cl_avgtime",
                    "cl_poscalc", "mem_cl", "mem_idx", "mem_event", "pulses"}
        if not required.issubset(out):
            raise ValueError(f"missing detailed output columns: {sorted(required - out.keys())}")
        p = out["pulses"]
        if not {"ev", "idx", "bar", "sipm", "seq", "pe", "time"}.issubset(p):
            raise ValueError("missing detailed pulse columns")
        ce, me = integer(out["cl_event"], "cluster event"), integer(out["mem_event"], "member event")
        pe, pi = integer(p["ev"], "pulse event"), integer(p["idx"], "pulse idx")
        mc, mi = integer(out["mem_cl"], "member cluster"), integer(out["mem_idx"], "member pulse")
        for name, value in out.items():
            if name.startswith("cl_") and len(value) != len(ce):
                raise ValueError(f"truncated cluster column: {name}")
            if name.startswith("mem_") and len(value) != len(me):
                raise ValueError(f"truncated member column: {name}")
        if any(len(v) != len(pe) for v in p.values()):
            raise ValueError("truncated pulse column")
        ids = rows(np.column_stack([pe, pi]))
        if len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate pulse identity")
        ordered = np.lexsort((pi, pe))
        events = pe[ordered]
        expected_idx = np.arange(len(events)) - np.searchsorted(events, events)
        if not np.array_equal(pi[ordered], expected_idx):
            raise ValueError("truncated/noncontiguous pulse indices")
        pulse_ids.append(ids)
        if np.any(mc >= len(ce)) or not np.array_equal(ce[mc], me):
            raise ValueError("dangling member cluster reference or wrong event")
        if not np.all(np.isin(rows(np.column_stack([me, mi])), ids)):
            raise ValueError("dangling member pulse reference")
        edges = rows(np.column_stack([me, mc, mi]))
        if len(np.unique(edges)) != len(edges):
            raise ValueError("duplicate membership edge")
        if np.asarray(out["cl_pos"]).shape != (len(ce), 3):
            raise ValueError("truncated cluster position")
        prefix = f"chunk_{i:04d}__"
        arrays[prefix + "cluster_keys"] = full(ce, "cluster")
        arrays[prefix + "member_keys"] = full(me, "member")
        arrays[prefix + "pulse_keys"] = full(pe, "pulse")
        for name, value in out.items():
            if name == "pulses":
                arrays.update({prefix + "pulses__" + k: np.asarray(v) for k, v in value.items()})
            else:
                arrays[prefix + name] = np.asarray(value)
        if "fallback_events" in out:
            full(out["fallback_events"], "fallback")
        counts["clusters"] += len(ce)
        counts["members"] += len(me)
        counts["pulses"] += len(pe)
        counts["fallback_events"] += len(out.get("fallback_events", []))
        for sector, n in zip(*np.unique(integer(out["cl_type"], "sector type"), return_counts=True)):
            sectors[str(sector)] = sectors.get(str(sector), 0) + int(n)
    if pulse_ids:
        ids = np.concatenate(pulse_ids)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate pulse identity across reconstruction chunks")
    for kind, column in (("clusters", "cl_event"), ("pulses", None)):
        labels = np.unique(np.concatenate([o[column] if column else o["pulses"]["ev"] for o in outs])) if outs else np.zeros(0, np.int64)
        arrays[f"events_with_{kind}_keys"] = full(labels, kind)
        counts[f"events_with_{kind}"] = len(labels)
    return arrays, counts, sectors


@contextmanager
def local_assets_only():
    previous = os.environ.get("CRVFS_NO_FETCH")
    os.environ["CRVFS_NO_FETCH"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CRVFS_NO_FETCH", None)
        else:
            os.environ["CRVFS_NO_FETCH"] = previous


def source_paths():
    import mu2efastsim
    import crvfastsim
    return sorted({str(p.resolve()) for module in (mu2efastsim, crvfastsim)
                   for root in module.__path__ for p in Path(root).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".json")})


def software_hardware():
    versions = {}
    for name in ("numpy", "pandas", "scipy", "awkward", "uproot", "jax", "jaxlib", "jax-oneapi-plugin"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"platform": platform.platform(), "hostname": platform.node(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(), "python": sys.version,
            "executable": sys.executable, "versions": versions,
            "environment": {k: v for k, v in os.environ.items() if k.startswith(("CRVFS_", "CRVFASTSIM_", "JAX_", "XLA_"))
                            or k in ("CUDA_VISIBLE_DEVICES", "ZE_AFFINITY_MASK", "ONEAPI_DEVICE_SELECTOR",
                                     "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONPATH", "MU2E_SEARCH_PATH")}}


def accelerator(requested):
    if not requested:
        return nullcontext(), {"response": "numpy CPU", "reco": "numpy CPU"}
    import jax
    import jax.numpy as jnp
    probe = jnp.zeros(1).block_until_ready()
    devices = list(probe.devices())
    if not devices or any(d.platform not in ("gpu", "cuda", "rocm", "oneapi") for d in devices):
        raise ValueError(f"GPU requested but actual default array placement is {devices}")
    return jax.default_device(devices[0]), {"response": str(devices[0]), "reco": str(devices[0]),
                                          "devices": [{"id": d.id, "platform": d.platform,
                                                       "kind": d.device_kind, "description": str(d),
                                                       "platform_version": getattr(d.client, "platform_version", None)} for d in jax.devices()],
                                          "jax_enable_x64": bool(jax.config.jax_enable_x64)}


class Recorder:
    """Optional runchain callback; owns no physics and writes each input before releasing its detail."""

    def __init__(self, outdir, seed, provenance, reuse_assets=False):
        self.outdir, self.seed, self.provenance = Path(outdir), seed, provenance
        self.files, self.full_keys, self.full_packed = [], {}, []
        self.setup_hash_s = 0.0
        self.reuse_assets = reuse_assets

    def setup(self, rc, cfg, gdml, tier1b, calib, deviation):
        from ..reco.waveform import WaveformConfig
        from ..response import lookup
        assets = [gdml, *tier1b.paths(), rc.counters_path(), geometry.find_counters((cfg.waveform or WaveformConfig()).single_pe_file),
                  str(Path(lookup.lookup_dir()) / "photonMap.root")]
        assets += [str(Path(lookup.lookup_dir()) / f"LookupTable_{name}") for name in sorted({s.table for s in cfg.response.sectors.values()})]
        assets += [p for p in (calib, deviation) if p and p != "nominal"]
        a = time.perf_counter()
        paths = sorted({str(Path(p).resolve()) for p in assets})
        if self.reuse_assets:
            if paths != sorted(self.provenance.get("assets_sha256", {})):
                raise ValueError("prepared asset paths changed across repeats")
        else:
            resolved = {p: sha256(p) for p in paths}
            if "assets_sha256" in self.provenance and resolved != self.provenance["assets_sha256"]:
                raise ValueError("asset provenance changed across repeats")
            self.provenance["assets_sha256"] = resolved
        for field, value in (("effective_run_config", rc), ("effective_chain_config", cfg), ("effective_tier1b", tier1b)):
            current = dataclasses.asdict(value)
            if field in self.provenance and current != self.provenance[field]:
                raise ValueError("effective configuration changed across repeats")
            self.provenance[field] = current
        self.setup_hash_s += time.perf_counter() - a

    def check_input(self, i, path, gen):
        keys, packed = event_keys(gen)
        if self.full_packed and np.any(np.isin(packed, np.concatenate(self.full_packed))):
            raise ValueError("duplicate generated event or run/subrun/event packing collision across inputs")
        self.full_keys[i] = keys
        self.full_packed.append(packed)

    def output(self, i, path, keys, outs, *, st, pa, charges, digis, waveforms, timers):
        a = time.perf_counter()
        require_complete(timers)
        arrays, counts, sectors = detail_arrays(keys, outs)
        packed = arrays["generated_packed"]
        for name, data, reference in (("charges", charges, "chan"), ("digis", digis, "chan"),
                                      ("waveforms", waveforms, "ev")):
            require_complete(data)
            row_columns = {k: v for k, v in data.items() if not (name == "digis" and k.startswith("cluster_"))}
            if any(len(v) != len(data[reference]) for v in row_columns.values()):
                raise ValueError(f"truncated {name} columns")
            if name == "digis" and "cluster_chan" in data:
                if any(len(v) != len(data["cluster_chan"]) for k, v in data.items() if k.startswith("cluster_")):
                    raise ValueError("truncated digi source-cluster columns")
        arrays["input_generated_keys"] = self.full_keys.pop(i)
        if not np.all(np.isin(rows(keys), rows(arrays["input_generated_keys"]))):
            raise ValueError("selected generated universe outside full input universe")
        arrays["rng_seed_and_input_index"] = np.asarray([self.seed, i], np.int64)
        for name, frame in (("steps", st), ("particles", pa)):
            fk = np.unique(integer(frame[tables.KEY].to_numpy(), name), axis=0)
            if not np.all(np.isin(rows(fk), rows(keys))):
                raise ValueError(f"{name} outside generated universe")
            arrays[f"events_with_{name}_keys"] = fk
            counts[name] = len(frame)
            counts[f"events_with_{name}"] = len(fk)
        for name, ev in (("charges", integer(charges["chan"], "charge labels") // chain.CHAN_MULT),
                         ("digis", integer(digis["chan"], "digi labels") // chain.CHAN_MULT),
                         ("waveforms", integer(waveforms["ev"], "waveform events"))):
            if not np.all(np.isin(ev, packed)):
                raise ValueError(f"{name} outside generated universe")
            arrays[f"events_with_{name}_keys"] = keys[np.isin(packed, np.unique(ev))]
            counts[name] = len(ev)
            counts[f"events_with_{name}"] = len(np.unique(ev))
        counts.update(generated_events=len(keys), input_generated_events=len(arrays["input_generated_keys"]),
                      photons=int(timers["photons"]) if "photons" in timers else None)
        for name in ("pulses", "charges", "digis"):
            if name in timers and counts[name] != timers[name]:
                raise ValueError(f"truncated {name}: table/timer count mismatch")
        arrays.update({"count__" + name: np.asarray(value, np.int64) for name, value in counts.items() if value is not None})
        fp = fingerprint(arrays)
        name = f"input_{i:04d}.npz"
        with (self.outdir / name).open("xb") as stream:
            np.savez(stream, **arrays)
        self.files.append({"input": str(path), "input_index": i, "seed": self.seed, "npz": name,
                           "fingerprint": fp, "npz_sha256": sha256(self.outdir / name), "counts": counts,
                           "cluster_type_counts": sectors, "phase_wall_s": timers,
                           "detail_validate_fingerprint_output_s": time.perf_counter() - a})


def run(inputs, outdir=None, repeat=2, seed=1, reuse_context=False, **kwargs):
    """Run one cold-labelled pass and repeat-1 same-process warm passes; publish only after exact closure."""
    if not outdir:
        raise ValueError("e2e-bench requires an explicit new --outdir")
    if repeat < 1 or seed < 0 or (kwargs.get("max_events") is not None and kwargs["max_events"] < 1):
        raise ValueError("repeat/max-events must be positive and seed nonnegative")
    if kwargs.get("compare") or kwargs.get("emit"):
        raise ValueError("stage-0 does not compare stock acceptance or write external --emit products")
    kwargs.pop("compare", None)
    kwargs.pop("emit", None)
    allowed = {"library", "calib", "deviation", "gdml", "jax", "max_events", "counters", "reco_cpu", "msc_tail",
               "tail_k", "sliced_msc", "first_muon_only", "no_absorbers", "run_config", "window", "fused_response",
               "prefetch", "unbanded_world", "no_below_world", "tier1b_set", "tier1b_dir", "transport_backend", "legacy_transport", "tier1b_backend"}
    if set(kwargs) - allowed:
        raise ValueError(f"unsupported benchmark options: {sorted(set(kwargs) - allowed)}")
    if not kwargs.get("calib"):
        raise ValueError("e2e-bench requires --calib")
    if reuse_context and kwargs.get("fused_response"):
        raise ValueError("prepared context does not support memory-dependent fused response batching")
    files = [str(Path(p).resolve()) for p in tables.expand(inputs)]
    if not files or len(set(files)) != len(files) or any(not Path(p).is_file() or not Path(p).stat().st_size for p in files):
        raise ValueError("empty, missing, or repeated input file")
    destination = Path(outdir)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing existing output: {destination}")
    invocation_start = time.perf_counter()
    context, device = accelerator(bool(kwargs.get("jax")))
    accelerator_preflight_s = time.perf_counter() - invocation_start
    a = time.perf_counter()
    if kwargs.get("reco_cpu") and kwargs.get("jax"):
        device["reco"] = "explicit JAX CPU (mixed-device run)"
    provenance = {"input_sha256": {p: sha256(p) for p in files},
                  "source_sha256": {p: sha256(p) for p in source_paths()},
                  "argument_asset_paths": {k: str(Path(kwargs[k]).resolve()) for k in
                                           ("library", "tier1b_dir", "calib", "deviation", "gdml", "counters")
                                           if kwargs.get(k) and kwargs[k] != "nominal"}}
    provenance_preflight_s = time.perf_counter() - a
    destination.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "status": "complete", "boundary": BOUNDARY, "cry_generation_timed": False,
              "cold_definition": "first runchain call in this process; no OS cache flush or persistent JIT cache reset",
              "timing_policy": "pass elapsed includes setup, input IO, detail validation/fingerprints/output and stats; phase timers overlap, never sum for rates",
              "hardware_software": software_hardware(), "device": device, "provenance": provenance,
              "accelerator_preflight_s": accelerator_preflight_s, "provenance_preflight_s": provenance_preflight_s,
              "arguments": {"inputs": files, "seed": seed, "repeat": repeat, **kwargs}, "passes": [],
              "context_policy": "prepared_once" if reuse_context else "reload_each_pass"}
    with context, local_assets_only():
        prepared = None
        if reuse_context:
            options = {k: v for k, v in kwargs.items() if k in
                       {"library", "calib", "deviation", "gdml", "jax", "counters", "no_absorbers",
                        "run_config", "window", "fused_response", "unbanded_world", "no_below_world",
                        "below_margin", "tier1b_set", "tier1b_dir"}}
            setup_recorder = Recorder(destination, seed, provenance)
            a = time.perf_counter()
            prepared = runchain.prepare(benchmark=setup_recorder, **options)
            report["preparation_elapsed_s"] = time.perf_counter() - a
            report["preparation_timers"] = prepared.preparation_timers
            report["preparation_asset_hash_s"] = setup_recorder.setup_hash_s
            report["context_fingerprint"] = prepared.fingerprint
            report["timing_policy"] = ("pass elapsed includes input IO, pre/post context audit, detail validation/output and stats; "
                                       "one-time preparation measured separately; immutable assets hashed at preparation and final verification")
            report["cold_definition"] = "first runchain execution after measured context preparation; no execution warmup or cache flush"
        transport_context = {} if reuse_context and kwargs.get("transport_backend") == "jax" else None
        tier1b_context = {} if reuse_context and kwargs.get("tier1b_backend") == "jax" else None
        passes_start = time.perf_counter()
        for index in range(repeat):
            folder = destination / f"pass_{index:03d}"
            folder.mkdir()
            recorder = Recorder(folder, seed, provenance, reuse_assets=reuse_context)
            a = time.perf_counter()
            device_kw = {"transport_context": transport_context} if transport_context is not None else {}
            if tier1b_context is not None:
                device_kw["tier1b_context"] = tier1b_context
            result = runchain.run(files, str(folder), seed=seed, benchmark=recorder, prepared=prepared,
                                  **device_kw, **kwargs)
            elapsed = time.perf_counter() - a
            if kwargs.get("jax"):
                _, actual_device = accelerator(True)
                if actual_device["response"] != device["response"]:
                    raise ValueError("accelerator placement changed during benchmark")
                device["jax_enable_x64_after_chain"] = actual_device["jax_enable_x64"]
            if len(recorder.files) != len(files):
                raise ValueError("missing per-input output")
            generated = sum(f["counts"]["generated_events"] for f in recorder.files)
            if generated <= 0 or generated != result["generated_events"]:
                raise ValueError("generated-event denominator mismatch")
            for f in recorder.files:
                for sector in result.get("sector_type_names", {}):
                    f["cluster_type_counts"].setdefault(str(sector), 0)
            fps = [f["fingerprint"] for f in recorder.files]
            if index and fps != [f["fingerprint"] for f in report["passes"][0]["files"]]:
                raise ValueError("warm output fingerprint differs from cold pass")
            totals = {name: sum(f["counts"][name] for f in recorder.files)
                      for name in ("events_with_steps", "pulses", "clusters", "members", "digis")}
            rates = {name + "_per_s": count / elapsed for name, count in totals.items()}
            report["passes"].append({"counts": totals, "rates": rates,
                                     "index": index, "stage": "cold" if index == 0 else "warm",
                                     "elapsed_s": elapsed, "generated_events": generated,
                                     "generated_events_per_s": generated / elapsed,
                                     "generated_events_per_hour": 3600 * generated / elapsed,
                                     "setup_hash_s": recorder.setup_hash_s, "files": recorder.files,
                                     "runchain": result})
        report["passes_elapsed_s"] = time.perf_counter() - passes_start
        report["context_audits_passed"] = True if prepared is not None else None
    a = time.perf_counter()
    for category in ("input_sha256", "assets_sha256", "source_sha256"):
        for path, expected in provenance.get(category, {}).items():
            if sha256(path) != expected:
                raise ValueError(f"provenance changed during benchmark: {path}")
    for item in report["passes"]:
        for record in item["files"]:
            path = destination / f"pass_{item['index']:03d}" / record["npz"]
            if sha256(path) != record["npz_sha256"]:
                raise ValueError(f"retained output changed during benchmark: {path}")
    report["final_provenance_verify_s"] = time.perf_counter() - a
    report["invocation_elapsed_before_report_s"] = time.perf_counter() - invocation_start
    report["invocation_generated_events_per_s"] = sum(p["generated_events"] for p in report["passes"]) / report["invocation_elapsed_before_report_s"]
    report["repeat_fingerprints_identical"] = True if repeat > 1 else None
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with (destination / "benchmark.json").open("x") as stream:
        stream.write(payload)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--outdir", required=True)
    p.add_argument("--repeat", type=int, default=2, help="total passes, cold first then same-process warm")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--reuse-context", action="store_true", help="opt-in measured one-time preparation with audited context reuse")
    p.add_argument("--calib", required=True)
    p.add_argument("--tier1b", dest="tier1b_set", choices=sorted(tier1b_assets.SETS),
                   help="tier-1b asset set (default: the run configuration's; crvfastsim.tier1b_assets)")
    for name in ("library", "tier1b-dir", "deviation", "gdml", "counters", "run-config", "window", "emit"):
        p.add_argument("--" + name)
    p.add_argument("--compare", nargs="+")
    p.add_argument("--max-events", type=int)
    p.add_argument("--transport-backend", choices=["numpy", "jax"], default="numpy")
    p.add_argument("--tier1b-backend", choices=["numpy", "jax"], default="numpy")
    p.add_argument("--tail-k", type=float)
    for name in ("jax", "reco-cpu", "msc-tail", "sliced-msc", "first-muon-only", "no-absorbers",
                 "no-below-world", "unbanded-world", "prefetch", "fused-response", "legacy-transport"):
        p.add_argument("--" + name, action="store_true")
    args = vars(p.parse_args(argv))
    try:
        run(**args)
    except (ValueError, OSError) as exc:
        p.exit(1, f"e2e-bench: {exc}\n")


if __name__ == "__main__":
    main()
