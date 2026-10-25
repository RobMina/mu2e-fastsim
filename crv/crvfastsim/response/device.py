"""The CRV response on the device without host round trips: CrvSteps -> photons -> SiPM charges in one pass.

Host work happens once, before anything goes to the device: the per-(step, SiPM) rows and their photon numbers
(photons.iter_rows, cheap numpy) and the SiPM slot table. On the device the photons of every (table, reflector, SiPM)
block (photons_jax.device_photons) are concatenated, masked by the SiPM status, mapped to SiPM slots and fed to
sipm_jax.device_sipm. Static buffers come from expected counts; residual overflow after SiPM sizing retries raises
before output is consumed. Outputs stay on the device (the waveform stage takes them as they are).
"""
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from . import photons, photons_jax, sipm_jax
from .sipm import NPIX
from .contract import require_complete, validate_sipm

CHAN_MULT = 32768  # = chain.CHAN_MULT (chain imports response, not the reverse)


@jax.jit
def _mask_label(step, chan, t, valid, status, ev_key):
    """Status and time masks and the SiPM label in one kernel (eager gathers over the photon buffer abort the oneAPI
    compiler: 7-wide vector in the index conversion)."""
    valid = valid & ((status[chan] & 0b101) == 0) & (t <= 1e5)   # stock: no charges on status bits 0 / 2
    return valid, ev_key[step] * CHAN_MULT + chan


@partial(jax.jit, static_argnames=("NB",))
def _compact_slots(valid, step, t, label, nv, NB):
    """Compact to the first NB valid photons and give each SiPM label a dense slot (rank among the valid labels)."""
    keep = jnp.nonzero(valid, size=NB, fill_value=0)[0]
    vc = jnp.arange(NB) < nv
    step, t, label = step[keep], t[keep], label[keep]
    lab = jnp.where(vc, label, jnp.iinfo(jnp.int64).max)
    order = jnp.argsort(lab)
    ls = lab[order]
    newl = jnp.concatenate([jnp.array([True]), ls[1:] != ls[:-1]])
    rank_sorted = jnp.cumsum(newl) - 1
    slot = jnp.zeros_like(rank_sorted).at[order].set(rank_sorted)
    return step, t, label, vc, slot


@jax.jit
def _take_root(root, label, step, n):
    root = jnp.minimum(root, n - 1)
    return label[root], step[root]


def response(steps, pdg, geom, rcfg, scfg, device_tables, deviation, status, ev_key, rng, key, pmap):
    """steps (DataFrame), pdg [nsteps], ev_key [nsteps] int64 event key per step. Returns a dict of device arrays per
    hit (fired mask, label, time, charge, pe, step) and an overflow flag."""
    validate_sipm(scfg)
    k_ph, k_si = jax.random.split(key)
    parts = []
    for L, rows in photons.iter_rows(steps, pdg, geom, rcfg, {k: v.L for k, v in device_tables.items()}, deviation, rng):
        k_ph, sub = jax.random.split(k_ph)
        parts.append(photons_jax.device_photons(device_tables[L.name.replace("LookupTable_", "")], rows, sub))
    step = jnp.concatenate([p["step"] for p in parts])
    chan = jnp.concatenate([p["chan"] for p in parts])
    t = jnp.concatenate([p["time"] for p in parts])
    valid = jnp.concatenate([p["valid"] for p in parts])
    over = jnp.any(jnp.stack([p["overflow"] for p in parts]))
    valid, label = _mask_label(step, chan, t, valid, jnp.asarray(status), jnp.asarray(ev_key))
    # compact to the valid photons: the arrival buffers are padded 3-5x, and the SiPM stage's memory scales with its input
    nv = int(valid.sum())                                       # one scalar sync per batch
    NB, Ms, Ntot, q_ct = sipm_jax.device_buckets(nv, scfg)
    step, t, label, vc, slot = _compact_slots(valid, step, t, label, nv, NB)
    n = max(nv, 1)
    dead = np.zeros(NPIX * NPIX, bool)
    for x, y in scfg.inactive:
        dead[x * NPIX + y] = True
    args = (slot, t, vc, k_si,
            jnp.asarray(np.cumsum(pmap.ravel()) / pmap.sum()), pmap.shape[1], jnp.asarray(dead),
            q_ct, scfg.overvoltage, scfg.time_constant, scfg.p1, scfg.p2, scfg.capacitance, NB, *Ms, Ntot)
    r = sipm_jax.run_device_sipm(args, scfg)   # same key on the retry, so the result is the larger table's
    require_complete({'overflow_photons': over, 'overflow_sipm': r['overflow']})
    lab_r, step_r = _take_root(r["root"], label, step, n)
    return {"fired": r["fired"], "label": lab_r, "time": r["time"], "charge": r["charge"], "pe": r["pe"],
            "step": step_r, "overflow": over | r["overflow"], "overflow_photons": over,
            "overflow_sipm": r["overflow"], "n_long_streams": r["n_long_streams"],
            "max_long_stream": r["max_long_stream"], "max_stream": r["max_stream"]}


def to_host(r):
    """Fired hits as host arrays sorted by (label, time), like sipm.simulate's output (chan = label)."""
    require_complete(r)
    f = np.asarray(r["fired"])
    out = {k: np.asarray(r[k])[f] for k in ("label", "time", "charge", "pe", "step")}
    o = np.lexsort((out["time"], out["label"]))
    out = {k: v[o] for k, v in out.items()}
    out["chan"], out["photon"] = out.pop("label"), out.pop("step")
    return out


BATCH_STEPS = 10000  # fallback steps per batch when the device reports no memory (CPU); ~7 GiB on synthetic muons
SIPM_BYTES_PER_SLOT = 920   # device_sipm memory per photon slot (compiled memory_analysis, 1-8 M photons)
PHOTONS_PER_YIELD_MEV = 0.0030  # photons at the SiPMs per (yield x visible MeV), G4 shard 11 and synthetic alike
MEM_FRACTION = 0.6          # share of the free device memory a batch may plan for (photon buffers, tables, headroom)
HOLD_BYTES_PER_PHOTON = 100  # kept outputs (keep=True): ~47 B per slot, slots padded up to 2x the photons
MIN_PHOTONS, MAX_PHOTONS = 1 << 18, 1 << 24
BUDGET_MARGIN = 0.85        # plan for 85 % of the power-of-two slot count: a batch slightly above it would double NB


def photon_budget(device=None, fraction=MEM_FRACTION, reserve=0):
    """Photons per batch that fit the device's free memory less `reserve` bytes, or None when the device reports no
    memory (CPU: use BATCH_STEPS). device_sipm rounds its slots up to a power of two, so the budget is the largest
    power of two that fits."""
    d = device or jax.devices()[0]
    ms = d.memory_stats() if hasattr(d, "memory_stats") else None
    if not ms or "bytes_limit" not in ms:
        return None
    free = ms["bytes_limit"] - ms.get("bytes_in_use", 0) - reserve
    n = int(fraction * free / SIPM_BYTES_PER_SLOT)
    return int(BUDGET_MARGIN * min(MAX_PHOTONS, max(MIN_PHOTONS, 1 << max(n, 1).bit_length() - 1)))


def batch_bounds(ev_key, est, budget=None, batch_steps=BATCH_STEPS):
    """Step-index cuts [a, b) of whole events (ev_key sorted): at most batch_steps steps and, when budget is given, at
    most budget estimated photons per batch (the memory cap). An event over either limit gets a batch of its own.
    The step limit stays because larger batches run slower per photon (EAF 1g.10gb, 2026-10-21)."""
    starts = np.flatnonzero(np.r_[True, ev_key[1:] != ev_key[:-1]])
    ends = np.r_[starts[1:], len(ev_key)]
    if not len(starts):
        return []
    cum_st = ends.astype(float)                               # steps up to the end of each event
    cum_ph = np.cumsum(np.add.reduceat(est, starts)) if budget is not None else None
    out, i = [], 0
    while i < len(starts):
        j = int(np.searchsorted(cum_st, starts[i] + batch_steps, side="right"))
        if cum_ph is not None:
            base = cum_ph[i - 1] if i else 0.0
            j = min(j, int(np.searchsorted(cum_ph, base + budget, side="right")))
        j = max(j, i + 1)
        out.append((int(starts[i]), int(ends[j - 1])))
        i = j
    return out


def photon_estimate(steps, geom, rcfg):
    """Expected photons at the SiPMs per step from the step's visible energy and its sector's yield (host, cheap)."""
    Y = np.array([rcfg.sectors[n].yield_per_mev for n in geom.sector_names])
    return PHOTONS_PER_YIELD_MEV * Y[geom.sector[steps.bar.to_numpy().astype(np.int64)]] * steps.edep.to_numpy()


def response_batched(steps, pdg, geom, rcfg, scfg, device_tables, deviation, status, ev_key, rng, key, pmap,
                     batch_steps=BATCH_STEPS, keep=True, budget="auto"):
    """response() over batches of whole events (steps sorted by event key). Returns a list of device dicts; keep=False
    waits for each batch and keeps only its overflow flag (timing without holding every batch on the device).
    budget: photons per batch; "auto" sizes it from the device's free memory (photon_budget), None cuts by batch_steps."""
    validate_sipm(scfg)
    if not len(ev_key):
        return []
    o = np.argsort(ev_key, kind="stable")
    steps, pdg, ev_key = steps.iloc[o].reset_index(drop=True), np.asarray(pdg)[o], np.asarray(ev_key)[o]
    est = photon_estimate(steps, geom, rcfg)
    if budget == "auto":   # outputs kept on the device for the caller take memory from every later batch
        budget = photon_budget(reserve=int(keep) * HOLD_BYTES_PER_PHOTON * float(est.sum()))
    bounds = batch_bounds(ev_key, est, budget, batch_steps)
    outs = []
    for a, b in bounds:
        key, sub = jax.random.split(key)
        r = response(steps.iloc[a:b].reset_index(drop=True), pdg[a:b], geom, rcfg, scfg, device_tables, deviation,
                     status, ev_key[a:b], rng, sub, pmap)
        require_complete(r)
        if not keep:
            jax.block_until_ready(r["charge"])
            outs.append({"overflow": r["overflow"]})
            del r
            continue
        r["step"] = r["step"] + a          # step rows of the event-sorted table
        r["order"] = o                     # host map back to the caller's step rows
        r["budget"] = budget
        outs.append(r)
    return outs


def to_host_batches(rs):
    """to_host over a list of batch outputs; step ids mapped back to the caller's row order."""
    if not rs:
        return {k: np.zeros(0, np.int64 if k in ('chan', 'photon') else float)
                for k in ('chan', 'photon', 'time', 'charge', 'pe')}
    parts = [to_host(r) for r in rs]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["photon"] = rs[0]["order"][out["photon"]]
    o = np.lexsort((out["time"], out["chan"]))
    return {k: v[o] for k, v in out.items()}
