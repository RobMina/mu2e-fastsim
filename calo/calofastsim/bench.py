"""Shared-timer entry points for the calorimeter device chain (``python3 -m mu2efastsim.bench``).

Boundary: stock CaloShowerSteps -> calorimeter digis, hits and clusters (:func:`device_chain.make_steps_chain`).

Inputs come from environment variables, so no path is built into the code:

- ``CALO_BENCH_TABLES``: colon-separated ``CaloLibraryDumper`` export tables (``dump/events``, ``dump/showers``);
  every stored event is one bench event, with all its steps;
- ``CALO_BENCH_CONDITIONS``: a directory with ``Sim_crystal.txt``, ``CsI-waveform-2020-02-12.root``,
  ``ReadoutNoise.root``, ``CsI-propag-2020-07-29.root`` and ``calo_geometry_run1430.npz``.

``bench_prepare(n, seed=None)`` loads the steps into host memory and packs ``n`` events into batches. With a seed
(the timer's ``--fresh-inputs``), the events start at an offset drawn from the seed and wrap around the tables,
and the per-event bunch times are drawn from the seed, so every pass sees other events. ``bench_steps`` runs the
batches through the chain (built once per process and configuration) and returns the device outputs.

Example::

    python3 -m mu2efastsim.bench calofastsim.bench:bench_steps --prepare calofastsim.bench:bench_prepare \\
        --events 40960 --fresh-inputs --dtype float32 --kwargs '{"dtype": "float32"}' --workflow W7 \\
        --boundary "CE CaloShowerSteps -> calo digis + hits + clusters" --out bench.json
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# capacities per event of a batch (averages over the batch), from CE events with margin; the shapes per event or
# per digi are absolute
PER_EVENT = dict(steps=56, photoelectrons=5000, rows=26, digis=16, hits=8)
FIXED = dict(digi_length=96, event_hits=32, rounds=8)
BATCH = 4096
MBTIME, EVENT_LENGTH, MARKER, CRYSTAL_LENGTH = 1695.0, 1700.0, 200.0, 200.0
PBTIMES = (-220.0, -215.0, -210.0, -205.0, -200.0)      # MC proton-bunch times of the on-spill production samples
_STEPS = {}
_CHAINS = {}


def caps(batch=BATCH, **override):
    """:class:`device_chain.Caps` for ``batch`` events: the per-event averages times the batch, the fixed shapes."""
    from .device_chain import Caps
    per = {k: v * batch for k, v in PER_EVENT.items()}
    per.update(FIXED)
    per.update(override)
    return Caps(events=batch, **per)


def load_steps(tables):
    """Per-event step arrays of every stored event of the export ``tables`` (host memory, in file order)."""
    import uproot
    events = []
    for path in tables:
        with uproot.open(path) as f:
            ev = f["dump/events"].arrays(["run", "subrun", "event"], library="np")
            sh = f["dump/showers"].arrays(["run", "subrun", "event", "crystal", "time", "energy_g4", "energy_birks",
                                           "position_z"], library="np")

        def key(d):
            return (d["run"].astype(np.int64) << 40) | (d["subrun"].astype(np.int64) << 20) | d["event"].astype(
                np.int64)
        ids = key(ev)
        order = np.argsort(ids, kind="stable")
        pos = np.searchsorted(ids[order], key(sh))
        if np.any(pos >= len(ids)) or not np.array_equal(ids[order][np.minimum(pos, len(ids) - 1)], key(sh)):
            raise ValueError(f"{path}: showers of events missing from the event table")
        row = order[pos]
        srt = np.argsort(row, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(row, minlength=len(ids)))]
        for e in range(len(ids)):
            s = srt[bounds[e]:bounds[e + 1]]
            events.append({k: sh[k][s] for k in ("crystal", "time", "energy_g4", "energy_birks", "position_z")})
    if not events:
        raise ValueError("no events in the bench tables")
    return events


def _steps():
    tables = tuple(os.environ["CALO_BENCH_TABLES"].split(":"))
    if tables not in _STEPS:
        _STEPS[tables] = load_steps(tables)
    return _STEPS[tables]


def _timing(m, rng):
    return {"pbtime": rng.choice(np.asarray(PBTIMES, np.float32), m), "mbtime": np.full(m, MBTIME, np.float32),
            "event_length": np.full(m, EVENT_LENGTH), "onspill": np.ones(m, bool),
            "marker": np.full(m, MARKER, np.float32), "crystal_length": np.full(m, CRYSTAL_LENGTH, np.float32)}


def pack_batches(events, n, batch=BATCH, seed=None):
    """``n`` events (wrapping around ``events``) as padded host batches ``(steps, timing)`` of ``batch`` events."""
    from .device_chain import entries_batch, steps_batch
    rng = np.random.default_rng(seed)
    start = 0 if seed is None else int(rng.integers(len(events)))
    idx = (start + np.arange(n)) % len(events)
    cap = PER_EVENT["steps"] * batch
    empty = {k: np.zeros((0, 3) if k in ("position", "direction") else 0) for k in
             ("energy", "cosine", "radius", "disk", "time", "position", "direction")}
    out = []
    for b0 in range(0, n, batch):
        parts = [events[i] for i in idx[b0:b0 + batch]]
        cnt = np.array([len(p["crystal"]) for p in parts])
        st = {"event": np.repeat(np.arange(len(parts)), cnt),
              "crystal": np.concatenate([p["crystal"] for p in parts]).astype(np.int64),
              "time": np.concatenate([p["time"] for p in parts]),
              "energy_g4": np.concatenate([p["energy_g4"] for p in parts]),
              "energy_birks": np.concatenate([p["energy_birks"] for p in parts]),
              "z_local": np.concatenate([p["position_z"] for p in parts])}
        if len(st["event"]) > cap:
            raise ValueError(f"batch of {len(parts)} events has {len(st['event'])} steps, above the {cap} capacity")
        t = entries_batch(empty, _timing(len(parts), rng), np.zeros(0, int), batch)[1]
        out.append((steps_batch(st, cap), t))
    return out


def bench_prepare(n, seed=None, batch=BATCH):
    """mu2efastsim.bench --prepare: ``n`` events of the bench tables as host batches (seeded with ``seed``)."""
    return {"batches": pack_batches(_steps(), int(n), batch, seed), "seed": seed, "batch": batch}


def conditions(dtype="float64", directory=None, fit_hessian=None):
    """The calorimeter conditions of ``CALO_BENCH_CONDITIONS`` as a :class:`device_chain.DeviceResponse`."""
    from . import stockio
    from .cluster import Geometry
    from .device_chain import pack_response
    d = Path(directory or os.environ["CALO_BENCH_CONDITIONS"])
    sim = stockio.load_sim_crystal(d / "Sim_crystal.txt")
    return pack_response(sim, stockio.load_pulse(d / "CsI-waveform-2020-02-12.root"),
                         stockio.load_noise(d / "ReadoutNoise.root"),
                         stockio.load_propagation(d / "CsI-propag-2020-07-29.root"),
                         Geometry.load(d / "calo_geometry_run1430.npz"), adc2mev=stockio.adc_to_mev,
                         dtype=np.dtype(dtype), fit_hessian=fit_hessian)


def chain(dtype="float64", batch=BATCH, digi_response="waveform", fit_hessian=None, **cap_override):
    """The jitted steps chain for this configuration, built once per process."""
    key = (np.dtype(dtype).name, batch, digi_response, fit_hessian, tuple(sorted(cap_override.items())))
    if key not in _CHAINS:
        import jax
        jax.config.update("jax_enable_x64", True)
        from .device_chain import make_steps_chain
        _CHAINS[key] = make_steps_chain(conditions(dtype, fit_hessian=fit_hessian), caps(batch, **cap_override),
                                        digi_response=digi_response)
    return _CHAINS[key]


def bench_steps(prepared, dtype="float64", digi_response="waveform", seed=1, fit_hessian=None, **cap_override):
    """mu2efastsim.bench entry point: every batch of ``prepared`` through the steps chain; returns the per-batch
    clusters and overflow counts (device arrays, which the timer waits on)."""
    import jax
    import jax.numpy as jnp
    run = chain(dtype, prepared["batch"], digi_response, fit_hessian, **cap_override)
    key = jax.random.PRNGKey(prepared["seed"] if prepared["seed"] is not None else seed)
    outs = []
    for i, (s, t) in enumerate(prepared["batches"]):
        r = run({k: jnp.asarray(v) for k, v in s.items()}, {k: jnp.asarray(v) for k, v in t.items()},
                jax.random.fold_in(key, i))
        outs.append({"clusters": r["clusters"], "overflow": r["overflow"]})
    return outs


def overflow(outs):
    """Summed overflow counts of :func:`bench_steps` outputs (host dict); all must be zero for a valid timing."""
    from .device_chain import OVERFLOW_NAMES
    total = np.sum([np.asarray(o["overflow"]) for o in outs], axis=0)
    return dict(zip(OVERFLOW_NAMES, total.tolist()))
