"""Device-side twin of reco.waveform_jax.pack: SiPM charges (device arrays, a fired mask) -> the padded [C, Q] cluster
matrices that reco.waveform_jax.synth_dense consumes, without bringing the charges to the host.

Same rules as the host packer (reco/waveform.py charge_clusters and waveform_jax.pack): charges sorted by (label,
time); a new cluster where the label changes or the gap exceeds singlePEWaveformMaxTime + 4 periods; the cluster start
is the first FEB tick (TDC0 + k * 12.5 ns) after first charge - 12.5 ns; keep = n_charges * max(1PE template) >=
minVoltage. Static shapes C (clusters) and Q (charges per cluster) are buffers; overflow is flagged. TDC0 per label
comes from a device array indexed by FEB slot (one uniform draw per (event, FEB), as chain.tdc0_per_feb).
"""
from functools import partial

import jax
import jax.numpy as jnp

PERIOD = 12.5


@partial(jax.jit, static_argnames=("C", "Q"))
def pack(label, time, charge, fired, tdc0_of_label, tmpl_max, ref_charge, max_time, min_voltage, C, Q):
    """label/time/charge/fired: per hit (device). tdc0_of_label: callable-free array aligned with hits (TDC0 of each
    hit's FEB). Returns T, W [C, Q], n, t_start, keep, chan, est [C] and an overflow flag."""
    big = jnp.iinfo(jnp.int64).max
    lab = jnp.where(fired, label, big)
    o = jnp.lexsort((time, lab))
    lab, t, q, td = lab[o], time[o], charge[o], tdc0_of_label[o]
    ok = lab != big
    gap = max_time + 4 * PERIOD
    new = jnp.concatenate([jnp.array([True]), (lab[1:] != lab[:-1]) | (t[1:] - t[:-1] > gap)]) & ok
    cid = jnp.cumsum(new) - 1
    first = jax.lax.cummax(jnp.where(new, jnp.arange(lab.shape[0]), 0), axis=0)
    pos = jnp.arange(lab.shape[0]) - first
    nclus = new.sum()
    over = (nclus > C) | jnp.any(ok & (pos >= Q))
    ci, pi = jnp.where(ok, jnp.minimum(cid, C - 1), C - 1), jnp.minimum(pos, Q - 1)
    sink = ~ok | (pos >= Q)
    T = jnp.zeros((C, Q)).at[ci, pi].set(jnp.where(sink, 0.0, t))
    W = jnp.zeros((C, Q)).at[ci, pi].set(jnp.where(sink, 0.0, q / ref_charge))
    n = jnp.zeros(C, jnp.int64).at[ci].add(jnp.where(sink, 0, 1))
    tf = jnp.zeros(C).at[jnp.where(new, ci, C - 1)].set(jnp.where(new, t, 0.0))
    tdf = jnp.zeros(C).at[jnp.where(new, ci, C - 1)].set(jnp.where(new, td, 0.0))
    chan = jnp.full(C, -1, jnp.int64).at[jnp.where(new, ci, C - 1)].set(jnp.where(new, lab, -1))
    span = jnp.zeros(C).at[ci].max(jnp.where(sink, 0.0, t - t[first]))
    t_start = jnp.ceil((tf - PERIOD - tdf) / PERIOD) * PERIOD + tdf
    keep = (n * tmpl_max >= min_voltage) & (jnp.arange(C) < nclus)
    est = jnp.floor((span + max_time) / PERIOD).astype(jnp.int64)
    return {"T": T, "W": W, "n": n, "t_start": t_start, "tdc0": tdf, "keep": keep, "chan": chan, "est": est,
            "nclus": nclus, "overflow": over}
