"""Resident JAX source/response backend, with one terminal output transfer per batch.

A kernel factory is mandatory; existing host-staged physics drivers are not usable
as kernels. Synthetic kernels require explicit opt-in and never advertise physical
tracker/CRV support. By default ROOT serialization and final-output compaction stay on
the host; the opt-in occupancy output reduces the detector tables on the device instead.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import time
from types import SimpleNamespace

import awkward as ak
import numpy as np
import uproot

from . import device_occupancy
from .device_source import Layout, pack_source, sample_overlay
from .fast_detectors import save_sampling, write_output
from .frames import IDS
from .stream_source import StreamSource


@dataclass(frozen=True)
class Request:
    count: int
    ordinal_start: int
    batch_index: int


class Source(StreamSource):
    """Host setup only; requests contain no host overlay or detector payload."""

    def next_batch(self, count):
        if type(count) is not int or not 1 <= count <= self.population:
            raise ValueError("count must satisfy 1 <= count <= population")
        return Request(count, self._ordinal, self._batch_index)

    def commit(self, request):
        if (request.ordinal_start, request.batch_index) != (self._ordinal, self._batch_index):
            raise ValueError("stale device request")
        self._ordinal += request.count
        self._batch_index += 1


def _audit_program(program):
    """Reject host callbacks, including pure callbacks nested in compiled control flow."""
    if isinstance(program, (tuple, list)):
        for child in program:
            _audit_program(child)
    elif isinstance(program, dict):
        for child in program.values():
            _audit_program(child)
    elif hasattr(program, "eqns"):
        if program.effects:
            raise ValueError("device program has effects/host callbacks")
        for equation in program.eqns:
            name = equation.primitive.name.lower()
            if any(token in name for token in ("callback", "infeed", "outfeed")):
                raise ValueError(f"device program contains host operation: {name}")
            _audit_program(equation.params)
    elif hasattr(program, "jaxpr"):
        _audit_program(program.jaxpr)


class Backend:
    source_factory = Source

    def __init__(self, args, source):
        from .stream_benchmark import factory
        import jax
        import jax.numpy as jnp

        if not isinstance(source, Source):
            raise ValueError("device backend requires its resident Source")
        if not jax.config.x64_enabled:
            raise ValueError("device backend requires JAX_ENABLE_X64=1 for lossless IDs and truth keys")
        platform = getattr(args, "device_platform", "gpu")
        if platform not in ("gpu", "cpu"):
            raise ValueError("device platform must be gpu or cpu")
        devices = jax.devices(platform)
        if not devices:
            raise RuntimeError(f"no {platform} device available")
        self.device, self.jax, self.source = devices[0], jax, source
        spec = getattr(args, "device_kernel", None)
        if not spec:
            raise ValueError("device-kernel is required; host-staged detector drivers are not device kernels")
        capacities = json.loads(Path(args.device_capacities).read_text())
        if not {"max_donors", "rows"} <= set(capacities) or set(capacities) - {"max_donors", "rows", "response"}:
            raise ValueError("device capacities must contain max_donors and rows, with optional response")
        self._capacities = capacities
        self.layout = Layout(args.batch_events, capacities["max_donors"], capacities["rows"], args.max_selected_rows)
        packed, law = pack_source(source, self.layout)
        self.pools = jax.device_put(packed, self.device)
        self.state = jax.device_put({"first": np.array(True), "cursors": np.zeros(4, np.int64)}, self.device)
        model = factory(spec)(args, source, self.layout)
        if model.kind not in ("synthetic", "physics"):
            raise ValueError("kernel kind must be synthetic or physics")
        if model.kind == "synthetic" and not getattr(args, "synthetic", False):
            raise ValueError("synthetic kernel requires --synthetic; no physics throughput may be inferred")
        if model.kind == "physics" and getattr(args, "synthetic", False):
            raise ValueError("--synthetic cannot relabel a physics kernel")
        self.assets = jax.device_put(model.assets, self.device)
        self._seeds = jax.device_put(np.array([args.sampling_seed, args.seed, args.tracker_seed], np.uint64), self.device)
        self.output_mode = getattr(args, "device_output", None) or "root"
        if self.output_mode not in ("root", "occupancy"):
            raise ValueError("device output must be root or occupancy")
        if self.output_mode == "occupancy":
            if not {"tracker_digis", "crv_reco"} <= set(model.support):
                raise ValueError("occupancy output needs a model with tracker digis and CRV reco")
            geometry = getattr(args, "occupancy_geometry", None)
            if not geometry:
                raise ValueError("occupancy output requires --occupancy-geometry")
            sector_map, self._sector_names, self._occupancy_geometry = device_occupancy.load_sector_map(geometry)
            self.assets = {"model": self.assets, "sector_map": jax.device_put(sector_map, self.device)}

        def domain(seed, batch_index):
            key = jax.random.PRNGKey(jnp.uint32(seed))
            key = jax.random.fold_in(key, jnp.uint32(seed >> jnp.uint64(32)))
            key = jax.random.fold_in(key, jnp.uint32(batch_index))
            return jax.random.fold_in(key, jnp.uint32(batch_index >> jnp.uint64(32)))

        def execute(pools, state, assets, seeds, ordinal, batch_index, size):
            mixed, receipt, next_state, errors = sample_overlay(pools, state, domain(seeds[0], batch_index),
                                                               ordinal, size, layout=self.layout, law=law)
            output, response_errors = model.apply(mixed, assets, domain(seeds[1], batch_index), domain(seeds[2], batch_index))
            errors = {**errors, **{"response/" + name: value for name, value in response_errors.items()}}
            usage = output.pop("usage", {}) if isinstance(output, dict) else {}
            usage = {**{"rows/" + name: value for name, value in receipt["actual_rows"].items()},
                     "max_donors": jnp.max(receipt["multiplicities"]), "selected_rows": receipt["selected_rows_total"],
                     **{"response/" + name: value for name, value in usage.items()}}
            result = {"output": output, "receipt": receipt, "errors": errors, "usage": usage,
                      "keys": mixed["keys"], "event_mask": mixed["event_mask"]}
            return result, next_state

        if self.output_mode == "occupancy":
            full, nsectors = execute, len(self._sector_names)

            def execute(pools, state, assets, seeds, ordinal, batch_index, size):
                result, next_state = full(pools, state, assets["model"], seeds, ordinal, batch_index, size)
                occupancy, occupancy_errors = device_occupancy.reduce(result["output"], assets["sector_map"],
                                                                      nsectors, self.layout.batch_events)
                result = {**result, "output": occupancy, "receipt": device_occupancy.compact_receipt(result["receipt"]),
                          "errors": {**result["errors"], **occupancy_errors}}
                return result, next_state

        self._arguments = (self.pools, self.state, self.assets, self._seeds)
        probe = tuple(jax.device_put(np.array(value, dtype=dtype), self.device)
                      for value, dtype in ((0, np.uint64), (0, np.uint64), (args.batch_events, np.int64)))
        _audit_program(jax.make_jaxpr(execute)(*self._arguments, *probe))
        self.execute = jax.jit(execute)
        self.metadata = {
            "name": "mixing.device_backend:Backend", "support": list(model.support),
            "residency": "device-resident sampling, overlay and response; one terminal device_get for output/receipt; host ROOT serialization",
            "devices": [str(self.device)], "platform": platform, "model_kind": model.kind,
            "model": model.metadata, "sampling_receipt_owner": "backend", "host_callbacks": "rejected by JAXPR audit",
            "scope": "synthetic integration only; not physical detector throughput" if model.kind == "synthetic" else model.metadata["scope"],
            "rng": "JAX uint64 seed/batch domains folded as low/high uint32; distinct from NumPy and stock",
            "capacity_policy": "fixed padded buffers, masks and explicit overflow rejection; no truncation acceptance or CPU fallback",
            "output_event_tables": [], "reconstruction": "not implemented",
        }
        if self.output_mode == "occupancy":
            self.metadata.update(
                output_format="occupancy",
                residency="device-resident sampling, overlay, response and occupancy reduction; one terminal device_get "
                          "of per-event plane/sector counts and a compact receipt; no detector tables persisted",
                occupancy={"tracker": "straw digis per plane (36)", "crv": "reconstructed pulses per sector",
                           "sector_names": self._sector_names, **self._occupancy_geometry})
        source.metadata.update(residency="source snapshots uploaded once; next_batch emits scalar requests only",
                               rng=self.metadata["rng"], device_capacities=capacities)
        self.sync()

    def sync(self):
        self.jax.block_until_ready((self.pools, self.state, self.assets, self._seeds))

    def run(self, request, outdir):
        """Dispatch one compiled graph, then download only its terminal products/receipt."""
        path = Path(outdir) / "fast.root"
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
        if path.exists():
            raise FileExistsError(path)
        if (request.ordinal_start, request.batch_index) != (self.source._ordinal, self.source._batch_index):
            raise ValueError("stale device request")
        if not 1 <= request.count <= self.layout.batch_events:
            raise ValueError("request exceeds device batch capacity")
        if request.ordinal_start + request.count > np.iinfo(np.int64).max:
            raise ValueError("realization ordinal exceeds int64")
        started = time.perf_counter()
        scalars = tuple(self.jax.device_put(np.array(value, dtype=dtype), self.device) for value, dtype in
                        ((request.ordinal_start, np.uint64), (request.batch_index, np.uint64), (request.count, np.int64)))
        with self.jax.transfer_guard("disallow"):
            result, state = self.execute(self.pools, self.state, self.assets, self._seeds, *scalars)
            self.jax.block_until_ready((result, state))
        response_s = time.perf_counter() - started
        started = time.perf_counter()
        host = self.jax.device_get(result)
        usage = {name: int(value) for name, value in host.pop("usage").items()}
        failed = [name for name, value in host["errors"].items() if np.any(value)]
        if failed:
            raise OverflowError("device batch rejected: " + ", ".join(_overflow_detail(name, usage, self._capacities)
                                                                     for name in failed))
        keys = host["keys"][:request.count]
        if not np.array_equal(host["event_mask"], np.arange(self.layout.batch_events) < request.count):
            raise ValueError("device event mask differs from requested census")
        receipt = host["receipt"]
        for field in ("signal_indices", "signal_event_ids", "intensity", "multiplicities"):
            receipt[field] = receipt[field][:request.count]
        for selection in receipt["selections"].values():
            for field in ("counts", "donors", "mask", "source_event_ids"):
                if field in selection:
                    selection[field] = selection[field][:request.count]
        receipt.update(batch_index=request.batch_index, ordinal_start=request.ordinal_start,
                       version=2, rng=self.metadata["rng"], model_kind=self.metadata["model_kind"])
        if self.output_mode == "occupancy":
            receipt["donors"] = "omitted: sequential donors follow from counts and initial_cursor"
            save_sampling(path.parent, receipt)
            device_occupancy.write(path.parent / "occupancy.npz", keys, request.ordinal_start,
                                   host["output"], self._sector_names)
        else:
            save_sampling(path.parent, receipt)
            _write(path, keys, request.ordinal_start, host["output"])
        self.source.commit(request)
        self.state = state
        return {"events": request.count, "response_s": response_s, "capacity_usage": usage,
                "transfer_and_write_s": time.perf_counter() - started, "terminal_device_get_calls": 1,
                "model_kind": self.metadata["model_kind"]}


def _rows(table, events):
    mask = np.asarray(table["mask"])
    if mask.ndim != 1 or mask.dtype.kind != "b":
        raise ValueError("output mask must be a boolean vector")
    rows = {name: np.asarray(values)[mask] for name, values in table.items() if name != "mask"}
    ev = rows["ev"]
    if ev.dtype.kind not in "iu" or np.any(ev < 0) or np.any(ev >= events):
        raise ValueError("output event index outside requested census")
    if any(not np.all(np.isfinite(values)) for values in rows.values()):
        raise ValueError("nonfinite terminal detector output")
    return rows


def _write(path, keys, ordinal, output):
    """Final-output compaction only: no detector calculation or intermediate upload."""
    n = len(keys)
    crv = tuple(_rows(output[name], n) for name in ("digis", "pulses", "clusters"))
    tracker = _rows(output["tracker"], n)
    order = np.argsort(tracker["ev"], kind="stable")
    counts = np.bincount(tracker["ev"], minlength=n)
    columns = dict(zip(IDS, keys.T))
    for name, values in tracker.items():
        if name != "ev":
            values = values[order]
            columns["d_" + name] = ak.unflatten(values.reshape(-1), counts * values.shape[1]) if values.ndim == 2 else ak.unflatten(values, counts)
    timing = SimpleNamespace(**{name: output["timing"][name][:n] for name in ("mc_pbtime", "reco_pbtime", "event_length")})
    write_output(path, keys, timing, crv, columns)
    with uproot.update(path) as root:
        root.mktree("benchmark/events", {**dict(zip(IDS, keys.T)),
                    "realization": np.arange(n, dtype=np.uint64) + np.uint64(ordinal)})


def _overflow_detail(flag, usage, capacities):
    """'flag (demand D, capacity C)' where the flag maps to a sized buffer; demands read from a saturated buffer show '>='."""
    name = flag.split("/")[-1]
    if flag == "row_capacity":
        parts = [f"{k[5:]} {v}/{capacities['rows'][k[5:]]}" for k, v in usage.items()
                 if k.startswith("rows/") and v > capacities["rows"][k[5:]]]
        return f"{flag} ({', '.join(parts) or 'see rows'})"
    if flag == "donor_capacity":
        return f"{flag} (demand {usage.get('max_donors')}, capacity {capacities['max_donors']})"
    detector = flag.split("/")[1] if flag.startswith("response/") and flag.count("/") >= 2 else None
    if detector:
        key = name[4:] if detector == "crv" and name.startswith("crv_") else name
        demand = usage.get(f"response/{detector}/{key}")
        limit = capacities.get("response", {}).get(detector, {}).get(key)
        if demand is not None and limit is not None:
            relation = ">=" if demand >= limit else "="
            return f"{flag} (demand {relation} {demand}, capacity {limit})"
    return flag
