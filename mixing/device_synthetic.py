"""Explicitly nonphysical device integration model and reproducible fake frame pools.

These arithmetic operators exercise grouping, random response, masks, ADCs and
empty-inclusive serialization. They do not approximate tracker or CRV physics.
"""
from types import SimpleNamespace

import awkward as ak
import numpy as np

from .frames import FrameBatch
from .sampling import BunchLaw, STREAMS


def model(args, source, layout):
    import jax
    import jax.numpy as jnp

    channels = 8
    def reduce(collection, field, channel):
        rows = collection["columns"]
        label = collection["event"] * channels + (rows[channel] % channels).astype(jnp.int64)
        value = jnp.where(collection["mask"], rows[field], 0.)
        return jnp.zeros(layout.batch_events * channels, jnp.float64).at[label].add(value)

    def apply(batch, assets, crv_key, tracker_key):
        label = jnp.arange(layout.batch_events * channels)
        ev, channel = label // channels, label % channels
        active = batch["event_mask"][ev]
        tracker = reduce(batch["collections"]["tracker"], "eion", "sid")
        crv = reduce(batch["collections"]["crv"], "edep", "bar")
        trk_mask, crv_mask = active & (tracker > 0), active & (crv > 0)
        tq = tracker * assets["gain"] + jax.random.uniform(tracker_key, tracker.shape, dtype=jnp.float64)
        cq = crv * assets["gain"] + jax.random.uniform(crv_key, crv.shape, dtype=jnp.float64)
        adc = lambda charge, width: jnp.clip(jnp.rint(charge[:, None] + jnp.arange(width)), 0, 4095).astype(jnp.int16)
        output = {
            "tracker": {"ev": ev, "mask": trk_mask, "sid": channel, "adc": adc(tq, 15)},
            "digis": {"ev": ev, "mask": crv_mask, "bar": channel, "sipm": jnp.zeros_like(ev),
                      "starttdc": jnp.full_like(ev, 40), "adc": adc(cq, 12)},
            "pulses": {"ev": ev, "mask": crv_mask, "pe": cq, "time": jnp.full_like(cq, 500.)},
            "clusters": {"ev": ev, "mask": crv_mask & (cq > 2), "pe": cq, "time": jnp.full_like(cq, 500.)},
            "timing": {"mc_pbtime": jnp.zeros(layout.batch_events), "reco_pbtime": jnp.zeros(layout.batch_events),
                       "event_length": jnp.full(layout.batch_events, 1700.)}}
        return output, {"nonfinite": jnp.any(~jnp.isfinite(tq)) | jnp.any(~jnp.isfinite(cq))}

    return SimpleNamespace(kind="synthetic", support=["synthetic_tracker", "synthetic_crv"],
                           metadata={"name": "arithmetic integration fixture", "physics_validated": False,
                                     "scope": "synthetic integration only; not detector simulation"},
                           assets={"gain": np.array(10., np.float64)}, apply=apply)


def source(manifest=None, *, seed=0, max_selected_rows):
    """Build five tiny sparse pools, including empty events and IDs above 2**63."""
    from .device_backend import Source
    sizes = (5, 7, 3, 6, 4)
    pools = []
    for pool_index, size in enumerate(sizes):
        counts = (np.arange(size) % 3).astype(np.int64)
        ev = np.repeat(np.arange(size), counts)
        offsets = np.r_[0, np.cumsum(counts)]
        local = np.arange(counts.sum()) - offsets[ev]
        key = local * 10 + 3
        jagged = lambda columns: ak.unflatten(ak.zip(columns), counts)
        collections = {
            "particles": jagged({"key": key, "parent": np.where(local > 0, 3, -1),
                                  "genidx": local, "pdg": np.full(len(ev), 13, np.int64)}),
            "gen": jagged({"idx": local, "pdg": np.full(len(ev), 13, np.int64)}),
            "tracker": jagged({"sp": key, "sid": (ev + pool_index) % 8,
                                "eion": np.full(len(ev), 0.1 * (pool_index + 1)), "t": ev * 1. + 400.}),
            "crv": jagged({"sp": key, "bar": (ev + 2 * pool_index) % 8,
                            "edep": np.full(len(ev), 0.2 * (pool_index + 1)), "t0": ev * 1. + 400., "t1": ev * 1. + 401.})}
        ids = np.column_stack([np.full(size, 2**63 + pool_index, np.uint64),
                               np.ones(size, np.uint64), np.arange(size, dtype=np.uint64) + 2**53])
        pools.append(FrameBatch(ids, collections))
    result = Source.__new__(Source)
    result._signal, result._pools = pools[0], dict(zip(STREAMS, pools[1:]))
    result._seed, result._max_selected_rows = seed, max_selected_rows
    result._ordinal, result._batch_index = 0, 0
    result._law = BunchLaw.from_sdf(20., .8, 120.)
    result._rates = dict.fromkeys(STREAMS, 0.04)
    result._campaign = {"max_skip": dict.fromkeys(STREAMS, 3), "mode": "sequential"}
    result.metadata = {"synthetic": True, "signal_events_full": 5, "pool_event_counts": dict(zip(STREAMS, sizes[1:]))}
    return result


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    from .device_backend import Backend
    from .stream_benchmark import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args(argv)
    capacities = args.out.parent / (args.out.name + "_capacities.json")
    with capacities.open("x") as stream:
        json.dump({"max_donors": 16, "rows": dict.fromkeys(("particles", "gen", "tracker", "crv"), 1024)}, stream)
    config = SimpleNamespace(out=args.out, frames_manifest=None, device_platform=args.platform,
                             device_kernel="mixing.device_synthetic:model", device_capacities=capacities,
                             synthetic=True, backend="mixing.device_backend:Backend", events=13, batch_events=4,
                             max_selected_rows=4096, chunk=4, warmup_batches=1, seed=17, tracker_seed=19,
                             sampling_seed=23, require_stage=[])
    report = run(config, source, Backend)
    print(json.dumps({key: report[key] for key in ("events", "warmup_events", "measured_batches", "backend")}))


if __name__ == "__main__":
    main()
