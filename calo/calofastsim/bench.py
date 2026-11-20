"""Shared-timer entry points for the calorimeter device chain (``python3 -m mu2efastsim.bench``).

W7's default (since 2026-11-20): ``bench_entrants`` with ``--prepare calofastsim.bench:bench_prepare_entrants``,
calorimeter entrants (front and inner-edge entry states of the bench tables' events) -> library draw -> hits and
clusters through :mod:`calofastsim.entrant_chain` with :data:`entrant_chain.W7_DEFAULT` and the per-entrant library
``CALO_BENCH_LIBRARY`` (W7's: content hash :data:`entrant_chain.W7_LIBRARY_HASH`).

Stock-steps entry (the response alone): ``bench_steps``, stock CaloShowerSteps -> calorimeter hits and clusters
(:func:`device_chain.make_steps_chain`). Both use the default crystal response (``digi_response="parametric"``,
hits drawn per crystal, no digis); ``digi_response="waveform"`` runs photoelectrons, digis and the template fit.

Inputs come from environment variables, so no path is built into the code:

- ``CALO_BENCH_TABLES``: colon-separated ``CaloLibraryDumper`` export tables (``dump/events``, ``dump/showers``);
  every stored event is one bench event, with all its steps;
- ``CALO_BENCH_CONDITIONS``: a directory with ``Sim_crystal.txt``, ``CsI-waveform-2020-02-12.root``,
  ``ReadoutNoise.root``, ``CsI-propag-2020-07-29.root`` and ``calo_geometry_run1430.npz``.

``bench_prepare(n, seed=None)`` loads the steps into host memory and packs ``n`` events into batches. With a seed
(the timer's ``--fresh-inputs``), the events start at an offset drawn from the seed and wrap around the tables,
and the per-event bunch times are drawn from the seed, so every pass sees other events. ``bench_steps`` runs the
batches through the chain (built once per process and configuration) and returns the device outputs.

- ``CALO_BENCH_LIBRARY`` (entrant entry points): the per-entrant library npz.

W7 default, example::

    python3 -m mu2efastsim.bench calofastsim.bench:bench_entrants --prepare calofastsim.bench:bench_prepare_entrants \\
        --events 163840 --fresh-inputs --dtype float32 --kwargs '{"dtype": "float32"}' --workflow W7 \\
        --boundary "CE calorimeter entrants -> calo hits + clusters" --out bench.json

The stock-steps entry: ``calofastsim.bench:bench_steps --prepare calofastsim.bench:bench_prepare``, boundary
"CE CaloShowerSteps -> calo hits + clusters".

Batches: ``BATCH`` (65,536 events) for the default parametric response, which runs fastest at large batches;
the waveform response runs fastest at ``WAVEFORM_BATCH`` (4,096): ``--prepare calofastsim.bench:
bench_prepare_waveform`` with ``--kwargs '{"digi_response": "waveform", ...}'``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# capacities per event of a batch (averages over the batch), from CE events with margin; the shapes per event or
# per digi are absolute
PER_EVENT = dict(steps=56, photoelectrons=5000, rows=26, digis=16, hits=8)
FIXED = dict(digi_length=96, event_hits=32, rounds=8)
BATCH = 65536                   # events per device call, parametric response (one GPU: 4,096 -> 65,536 = 2.1x)
WAVEFORM_BATCH = 4096           # the waveform response (16,384 and 65,536 are slower)
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


def bench_prepare_waveform(n, seed=None):
    """bench_prepare in batches of WAVEFORM_BATCH, for ``digi_response="waveform"``."""
    return bench_prepare(n, seed=seed, batch=WAVEFORM_BATCH)


def conditions(dtype="float64", directory=None, fit_hessian="analytic"):
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


def chain(dtype="float64", batch=BATCH, digi_response="parametric", fit_hessian="analytic", **cap_override):
    """The jitted steps chain for this configuration, built once per process."""
    key = (np.dtype(dtype).name, batch, digi_response, fit_hessian, tuple(sorted(cap_override.items())))
    if key not in _CHAINS:
        import jax
        jax.config.update("jax_enable_x64", True)
        from .device_chain import make_steps_chain
        _CHAINS[key] = make_steps_chain(conditions(dtype, fit_hessian=fit_hessian), caps(batch, **cap_override),
                                        digi_response=digi_response)
    return _CHAINS[key]


def bench_steps(prepared, dtype="float64", digi_response="parametric", seed=1, fit_hessian="analytic",
                **cap_override):
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


# ------------------------------------------------- entrant library (opt-in shower source: entrants -> clusters)
# Boundary: calorimeter entrants (front and inner-edge entry states in the disk frame, every entrant of the bench
# events, calofastsim.showers.entrants.read of the CALO_BENCH_TABLES exports) -> library draw -> placement -> hits ->
# clusters (calofastsim.entrant_chain). The library is ``CALO_BENCH_LIBRARY`` (an entrants LIBSPEC npz).
ENTRANTS_PER_EVENT = 2          # entrant rows per batch event (CE: 1.0 per stored event, at most 2 in 14,218)
ENTRANT_STEPS = 64              # drawn steps per event (CE: 41 per entrant on average, at most 93)
_ENTRANTS = {}
_ENTRANT_CHAINS = {}


def load_entrants(tables):
    """Per-event entrant states of every stored event of the export ``tables`` (host memory, in file order)."""
    from .showers import entrants as E
    events = []
    for path in tables:
        ent, _, _, cen = E.read(path)
        order = np.argsort(ent["event_row"], kind="stable")
        bounds = np.searchsorted(ent["event_row"][order], np.arange(cen["events"] + 1))
        for e in range(cen["events"]):
            s = order[bounds[e]:bounds[e + 1]]
            events.append({k: ent[k][s] for k in ("pdg", "face", "disk", "time", "local_position", "local_momentum")})
    if not events:
        raise ValueError("no events in the bench tables")
    return events


def pack_entrant_batches(events, n, batch=BATCH, seed=None):
    """``n`` events (wrapping around ``events``) as padded host batches ``(q, timing)`` for the entrant chain."""
    from .entrant_chain import entrants_batch
    rng = np.random.default_rng(seed)
    start = 0 if seed is None else int(rng.integers(len(events)))
    idx = (start + np.arange(n)) % len(events)
    out = []
    for b0 in range(0, n, batch):
        parts = [events[i] for i in idx[b0:b0 + batch]]
        cnt = np.array([len(p["face"]) for p in parts])
        e = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        e["event"] = np.repeat(np.arange(len(parts)), cnt)
        out.append(entrants_batch(e, _timing(len(parts), rng), ENTRANTS_PER_EVENT * batch, batch))
    return out


def bench_prepare_entrants(n, seed=None, batch=BATCH):
    """mu2efastsim.bench --prepare for :func:`bench_entrants`: ``n`` events of the bench tables' entrants."""
    tables = tuple(os.environ["CALO_BENCH_TABLES"].split(":"))
    if tables not in _ENTRANTS:
        _ENTRANTS[tables] = load_entrants(tables)
    return {"batches": pack_entrant_batches(_ENTRANTS[tables], int(n), batch, seed), "seed": seed, "batch": batch}


def entrant_chain(dtype="float64", batch=BATCH, digi_response="parametric", key_neighbors=8, placement="crystal",
                  fit_hessian="analytic", *, selection="cell", far="g4", nearest_k=1, max_entry_distance=20.0,
                  **cap_override):
    """The jitted entrant chain on ``CALO_BENCH_LIBRARY`` for this configuration, built once per process and library
    path (a library swapped within one process gets its own chain)."""
    library_path = os.path.abspath(os.environ["CALO_BENCH_LIBRARY"])
    key = (library_path, np.dtype(dtype).name, batch, digi_response, key_neighbors, placement, fit_hessian,
           selection, far, nearest_k, max_entry_distance, tuple(sorted(cap_override.items())))
    if key not in _ENTRANT_CHAINS:
        import jax
        jax.config.update("jax_enable_x64", True)
        from .entrant_chain import make_entrant_chain, pack_entrants
        from .showers.entrants import load
        library, _ = load(library_path)
        lib = pack_entrants(library, placement=placement, key_neighbors=key_neighbors, selection=selection, far=far,
                            nearest_k=nearest_k, max_entry_distance=max_entry_distance, dtype=np.dtype(dtype))
        over = dict(steps=ENTRANT_STEPS * batch)
        over.update(cap_override)
        _ENTRANT_CHAINS[key] = make_entrant_chain(lib, conditions(dtype, fit_hessian=fit_hessian), caps(batch, **over),
                                                  digi_response=digi_response)
    return _ENTRANT_CHAINS[key]


def bench_entrants(prepared, dtype="float64", digi_response="parametric", key_neighbors=8, placement="crystal",
                   seed=1, fit_hessian="analytic", selection="cell", far="g4", nearest_k=1, max_entry_distance=20.0,
                   **cap_override):
    """mu2efastsim.bench entry point: every batch of ``prepared`` through the entrant chain (entrants -> clusters);
    returns the per-batch clusters and overflow counts (device arrays). ``selection``/``far``/``nearest_k``/
    ``max_entry_distance`` as :func:`entrant_chain.pack_entrants` (opt-in nearest-donor rule)."""
    import jax
    import jax.numpy as jnp
    run = entrant_chain(dtype, prepared["batch"], digi_response, key_neighbors, placement, fit_hessian,
                        selection=selection, far=far, nearest_k=nearest_k, max_entry_distance=max_entry_distance,
                        **cap_override)
    key = jax.random.PRNGKey(prepared["seed"] if prepared["seed"] is not None else seed)
    outs = []
    for i, (q, t) in enumerate(prepared["batches"]):
        r = run({k: jnp.asarray(v) for k, v in q.items()}, {k: jnp.asarray(v) for k, v in t.items()},
                jax.random.fold_in(key, i))
        outs.append({"clusters": r["clusters"], "overflow": r["overflow"]})
    return outs


def overflow(outs):
    """Summed overflow counts of :func:`bench_steps` outputs (host dict); all must be zero for a valid timing."""
    from .device_chain import OVERFLOW_NAMES
    total = np.sum([np.asarray(o["overflow"]) for o in outs], axis=0)
    return dict(zip(OVERFLOW_NAMES, total.tolist()))
