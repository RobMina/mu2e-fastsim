"""Persistent GPU-backed response adapter; host packing and ROOT writing remain explicit."""
from operator import index
from pathlib import Path
import time

import numpy as np
import uproot

from .frames import IDS, _event_index, _table_event_ids, _validate_event_ids


def _nonnegative(value, name):
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if isinstance(value, (bool, np.bool_)) or result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _batch_seed(seed, batch_index):
    return int(np.random.SeedSequence([seed, batch_index]).generate_state(1, dtype=np.uint32)[0])


def _aligned_steps(steps, census, name):
    keys = _table_event_ids(steps)
    if np.any(census.get_indexer(_event_index(keys)) < 0):
        raise ValueError(f"{name} step event absent from batch full-ID census")
    # Uniform integer columns prevent DataFrame.to_numpy from promoting mixed signed/unsigned IDs.
    if any(steps[field].dtype != np.dtype(np.uint64) for field in IDS):
        steps = steps.copy(deep=False)
        for field in IDS:
            steps[field] = steps[field].to_numpy(dtype=np.uint64)
    return steps


class Backend:
    """Load assets/timing once, then run the unchanged chunked detector consumers per batch."""

    def __init__(self, args, source):
        self.chunk = _nonnegative(args.chunk, "chunk")
        if not self.chunk:
            raise ValueError("chunk must be positive")
        self.seed = _nonnegative(args.seed, "seed")
        self.tracker_seed = _nonnegative(args.tracker_seed, "tracker_seed")
        if args.tracker_conditions is None:
            raise ValueError("tracker_conditions are required for the combined backend")

        from mu2efastsim import precision
        import jax
        devices = jax.devices()
        if not any(device.platform == "gpu" for device in devices):
            raise RuntimeError("this streaming backend requires a GPU")
        from crvfastsim import onspill, runconfig
        from crvfastsim.response import photons_jax, sipm
        from trkfastsim.digi import digicond, fastdigi
        from . import crv_inputs, fast_detectors

        self._jax, self._onspill = jax, onspill
        self._fastdigi, self._driver = fastdigi, fast_detectors
        keys = np.array(_validate_event_ids(source.signal.event_ids), dtype=np.uint64, copy=True)
        self._timing_index = _event_index(keys)
        if not len(keys) or not self._timing_index.is_unique:
            raise ValueError("missing or duplicate source full event IDs")
        mc, reco, length = crv_inputs.read_timing(keys, args.truth, args.stock)
        self._timing = onspill.EventTiming(keys, mc, reco, length)
        rc = runconfig.get("run1a_v01")
        self.geom, self.counters = rc.geometry(), rc.counters_cfg()
        self.cond = crv_inputs.load_conditions(args.truth, args.photon_csv, self.geom, self.counters)
        expanded = digicond.parse_fhicl_dump(Path(args.expanded).read_text())
        self.cfg = crv_inputs.build_config(expanded, jax=True)
        from crvfastsim.response import lookup   # the optical assets pinned by the run configuration
        with lookup.pinned(rc.asset_pin_map() if hasattr(rc, "asset_pin_map") else None):
            self.tabs = photons_jax.load_device_tables(self.cfg.response)
            self.pmap = sipm.photon_map()
        self.tracker_cond = digicond.Conditions.load(args.tracker_conditions)
        self.metadata = {
            "name": "mixing.stream_backend:Backend",
            "devices": [str(device) for device in devices],
            "precision": {"crv": precision.NAME, "tracker": "float64"},
            "support": ["tracker_digis", "crv_reco"],
            "residency": "GPU-backed response with host packing, thermal attempts, transfers and ROOT writing; not fully device-resident",
            "reconstruction": "not implemented",
            "scope": "tracker digis/ADCs and reduced CRV digis/pulses/clusters; no tracker track reconstruction, calorimeter, trigger or complete art products",
            "rng": "SeedSequence([detector_seed, batch_index]) uint32; existing per-chunk folding retained",
            "chunk": self.chunk,
            "tracker_fallback": "existing digitizer may use NumPy on chunk capacity overflow; recorded in stage stats",
        }

    def sync(self):
        """Finish optional JAX effects; detector outputs and writer inputs are already host-materialized."""
        barrier = getattr(self._jax, "effects_barrier", None)
        if barrier is not None:
            barrier()

    def run(self, batch, outdir):
        """Write fast.root only into a parent-created batch directory; return measured stages."""
        started = time.perf_counter()
        path = Path(outdir) / "fast.root"
        if not path.parent.is_dir():
            raise FileNotFoundError("batch output directory must already exist")
        if path.exists():
            raise FileExistsError(path)
        keys = _validate_event_ids(batch.keys)
        census = _event_index(keys)
        if not len(keys) or not census.is_unique:
            raise ValueError("missing or duplicate batch full event IDs")
        rows = self._timing_index.get_indexer(census)
        if np.any(rows < 0):
            raise ValueError("batch event absent from explicit timing full-ID census")
        ordinal = _nonnegative(batch.ordinal_start, "ordinal_start")
        batch_index = _nonnegative(batch.batch_index, "batch_index")
        if ordinal + len(keys) - 1 > np.iinfo(np.uint64).max:
            raise ValueError("realization ordinal exceeds uint64")
        tracker_steps = _aligned_steps(batch.tracker, census, "tracker")
        crv_steps = _aligned_steps(batch.crv, census, "CRV")
        pdg = np.asarray(batch.pdg)
        if pdg.shape != (len(crv_steps),) or pdg.dtype.kind not in "iu":
            raise ValueError("pdg must be an integer array aligned with CRV steps")
        timing = self._onspill.EventTiming(keys, self._timing.mc_pbtime[rows],
            self._timing.reco_pbtime[rows], self._timing.event_length[rows], self._timing.period)
        response_seed = _batch_seed(self.seed, batch_index)
        tracker_seed = _batch_seed(self.tracker_seed, batch_index)
        alignment_s = time.perf_counter() - started

        started = time.perf_counter()
        crv, crv_metrics = self._driver.crv_pass(crv_steps, pdg, timing, self.geom, self.counters,
            self.cond, self.cfg, self.tabs, self.pmap, response_seed, self.chunk)
        tracker_start = time.perf_counter()
        stats = {}
        out_keys, digi, nsteps = self._fastdigi.digitize_steps(tracker_steps, self.tracker_cond,
            self._jax.random.key(tracker_seed), keys=keys, chunk=self.chunk, stats=stats, backend="jax",
            event_timing={"pbtime": timing.mc_pbtime, "event_length": timing.event_length})
        returned = np.asarray(out_keys)
        # The tracker uses signed int64 internally; compare integer bit patterns, never mixed-sign floats.
        if (returned.shape != keys.shape or returned.dtype.kind not in "iu"
                or not np.array_equal(returned.astype(np.uint64, copy=False), keys.astype(np.uint64, copy=False))):
            raise ValueError("tracker changed event identities")
        tracker_metrics = {**stats, "seconds": time.perf_counter() - tracker_start, "steps": nsteps}
        self.sync()
        response_s = time.perf_counter() - started

        started = time.perf_counter()
        columns = self._driver.digi_columns(keys, digi, self.tracker_cond.nADCSamples)
        columns.update({name: keys[:, i] for i, name in enumerate(IDS)})
        self._driver.write_output(path, keys, timing, crv, columns)
        with uproot.update(path) as root:
            identity = {name: keys[:, i] for i, name in enumerate(IDS)}
            identity["realization"] = np.arange(len(keys), dtype=np.uint64) + np.uint64(ordinal)
            root.mktree("benchmark/events", identity)
        write_s = time.perf_counter() - started
        return {"events": len(keys), "response_s": response_s, "write_s": write_s,
                "stages": {"alignment_s": alignment_s, "crv": crv_metrics, "tracker": tracker_metrics},
                "response_seed": response_seed, "tracker_seed": tracker_seed}
