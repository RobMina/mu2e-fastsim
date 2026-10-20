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
    # SiPM slots on device: dense rank of the label among valid photons
    big = jnp.iinfo(jnp.int64).max
    lab = jnp.where(valid, label, big)
    order = jnp.argsort(lab)
    ls = lab[order]
    newl = jnp.concatenate([jnp.array([True]), ls[1:] != ls[:-1]])
    rank_sorted = jnp.cumsum(newl) - 1
    slot = jnp.zeros_like(rank_sorted).at[order].set(rank_sorted)
    n = valid.shape[0]
    NB, Ms, Ntot, q_ct = sipm_jax.device_buckets(n, scfg)
    pad = NB - n
    dead = np.zeros(NPIX * NPIX, bool)
    for x, y in scfg.inactive:
        dead[x * NPIX + y] = True
    args = (jnp.pad(slot, (0, pad)), jnp.pad(t, (0, pad)), jnp.pad(valid, (0, pad)), k_si,
            jnp.asarray(np.cumsum(pmap.ravel()) / pmap.sum()), pmap.shape[1], jnp.asarray(dead),
            q_ct, scfg.overvoltage, scfg.time_constant, scfg.p1, scfg.p2, scfg.capacitance, NB, *Ms, Ntot)
    L = sipm_jax.STREAM_L
    r = sipm_jax.device_sipm(*args, L, scfg.n_pixels_x, scfg.n_pixels_y)
    while bool(r["stream_overflow"]) and L < 1024:   # one scalar sync; same key, so the result is the larger-L one
        L *= 2
        r = sipm_jax.device_sipm(*args, L, scfg.n_pixels_x, scfg.n_pixels_y)
    root = jnp.minimum(r["root"], n - 1)
    lab_p = jnp.pad(label, (0, pad))
    step_p = jnp.pad(step, (0, pad))
    return {"fired": r["fired"], "label": lab_p[root], "time": r["time"], "charge": r["charge"], "pe": r["pe"],
            "step": step_p[root], "overflow": over | r["overflow"], "overflow_photons": over,
            "overflow_sipm": r["overflow"]}


def to_host(r):
    """Fired hits as host arrays sorted by (label, time), like sipm.simulate's output (chan = label)."""
    f = np.asarray(r["fired"])
    out = {k: np.asarray(r[k])[f] for k in ("label", "time", "charge", "pe", "step")}
    o = np.lexsort((out["time"], out["label"]))
    out = {k: v[o] for k, v in out.items()}
    out["chan"], out["photon"] = out.pop("label"), out.pop("step")
    return out
