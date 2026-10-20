"""The CRV response on the device without host round trips: CrvSteps -> photons -> SiPM charges in one pass.

Host work happens once, before anything goes to the device: the per-(step, SiPM) rows and their photon numbers
(photons.iter_rows, cheap numpy) and the SiPM slot table. On the device the photons of every (table, reflector, SiPM)
block (photons_jax.device_photons) are concatenated, masked by the SiPM status, mapped to SiPM slots and fed to
sipm_jax.device_sipm. Static buffers come from expected counts; any overflow flag makes the caller redo the event
batch on the host path. Outputs stay on the device (the waveform stage takes them as they are).
"""
import numpy as np

import jax
import jax.numpy as jnp

from . import photons, photons_jax, sipm_jax
from .sipm import NPIX


def response(steps, pdg, geom, rcfg, scfg, device_tables, deviation, status, ev_key, rng, key, pmap):
    """steps (DataFrame), pdg [nsteps], ev_key [nsteps] int64 event key per step. Returns a dict of device arrays per
    hit (fired mask, label, time, charge, pe, step) and an overflow flag."""
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
    status_d = jnp.asarray(status)
    valid &= (status_d[chan] & 0b101) == 0                     # stock: no charges on status bits 0 / 2
    valid &= t <= 1e5
    label = jnp.asarray(ev_key)[step] * 4096 + chan
    # compact to the valid photons: the arrival buffers are padded 3-5x, and the SiPM stage's memory scales with its input
    nv = int(valid.sum())                                       # one scalar sync per batch
    NB, Ms, Ntot, q_ct = sipm_jax.device_buckets(nv, scfg)
    keep = jnp.nonzero(valid, size=NB, fill_value=0)[0]
    vc = jnp.arange(NB) < nv
    step, t, label = step[keep], t[keep], label[keep]
    # SiPM slots on device: dense rank of the label among valid photons
    big = jnp.iinfo(jnp.int64).max
    lab = jnp.where(vc, label, big)
    order = jnp.argsort(lab)
    ls = lab[order]
    newl = jnp.concatenate([jnp.array([True]), ls[1:] != ls[:-1]])
    rank_sorted = jnp.cumsum(newl) - 1
    slot = jnp.zeros_like(rank_sorted).at[order].set(rank_sorted)
    n = max(nv, 1)
    dead = np.zeros(NPIX * NPIX, bool)
    for x, y in scfg.inactive:
        dead[x * NPIX + y] = True
    args = (slot, t, vc, k_si,
            jnp.asarray(np.cumsum(pmap.ravel()) / pmap.sum()), pmap.shape[1], jnp.asarray(dead),
            q_ct, scfg.overvoltage, scfg.time_constant, scfg.p1, scfg.p2, scfg.capacitance, NB, *Ms, Ntot)
    r = sipm_jax.run_device_sipm(args, scfg)   # same key on the retry, so the result is the larger table's
    root = jnp.minimum(r["root"], n - 1)
    return {"fired": r["fired"], "label": label[root], "time": r["time"], "charge": r["charge"], "pe": r["pe"],
            "step": step[root], "overflow": over | r["overflow"], "overflow_photons": over,
            "overflow_sipm": r["overflow"], "n_long_streams": r["n_long_streams"],
            "max_long_stream": r["max_long_stream"], "max_stream": r["max_stream"]}


def to_host(r):
    """Fired hits as host arrays sorted by (label, time), like sipm.simulate's output (chan = label)."""
    f = np.asarray(r["fired"])
    out = {k: np.asarray(r[k])[f] for k in ("label", "time", "charge", "pe", "step")}
    o = np.lexsort((out["time"], out["label"]))
    out = {k: v[o] for k, v in out.items()}
    out["chan"], out["photon"] = out.pop("label"), out.pop("step")
    return out


BATCH_STEPS = 10000  # steps per device batch (whole events); ~4 M photons on synthetic muons, ~7 GiB in device_sipm


def response_batched(steps, pdg, geom, rcfg, scfg, device_tables, deviation, status, ev_key, rng, key, pmap,
                     batch_steps=BATCH_STEPS, keep=True):
    """response() over batches of whole events (steps sorted by event key). Returns a list of device dicts; keep=False
    waits for each batch and keeps only its overflow flag (timing without holding every batch on the device)."""
    o = np.argsort(ev_key, kind="stable")
    steps, pdg, ev_key = steps.iloc[o].reset_index(drop=True), np.asarray(pdg)[o], np.asarray(ev_key)[o]
    starts = np.flatnonzero(np.r_[True, ev_key[1:] != ev_key[:-1]])
    outs, a = [], 0
    while a < len(ev_key):
        j = int(np.searchsorted(starts, a + batch_steps, side="right")) - 1
        b = int(starts[j]) if j < len(starts) and starts[j] > a else len(ev_key)
        key, sub = jax.random.split(key)
        r = response(steps.iloc[a:b].reset_index(drop=True), pdg[a:b], geom, rcfg, scfg, device_tables, deviation,
                     status, ev_key[a:b], rng, sub, pmap)
        if not keep:
            jax.block_until_ready(r["charge"])
            outs.append({"overflow": r["overflow"]})
            del r
            a = b
            continue
        r["step"] = r["step"] + a          # step rows of the event-sorted table
        r["order"] = o                     # host map back to the caller's step rows
        outs.append(r)
        a = b
    return outs


def to_host_batches(rs):
    """to_host over a list of batch outputs; step ids mapped back to the caller's row order."""
    parts = [to_host(r) for r in rs]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["photon"] = rs[0]["order"][out["photon"]]
    o = np.lexsort((out["time"], out["chan"]))
    return {k: v[o] for k, v in out.items()}
