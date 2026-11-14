"""SimpleTimeCluster (CosmicReco) on the device, for the cosmic-seed trigger paths (cst_*).

The stock module sorts the event's ComboHits by time and walks them once. At the settings the trigger uses
(useTimeStep true, useTimeWindow false, TestFlag false, no non-noise bits, minNPanels <= 1, one panel/plane off) the
walk reduces to:

- a run is a maximal sequence of time-sorted hits whose consecutive gaps are <= TimeStep;
- a run is a cluster when its time span is <= TimeWindow and its straw-hit count (the sum of nStrawHits) is
  >= minNStrawHits;
- the walk restarts at the last hit of every run of two or more hits, so that hit alone is a second cluster when its
  nStrawHits is >= minNStrawHits (its span is 0).

Every cluster's nStrawHits is its straw-hit count, which is all a TimeClusterFilter without plane cuts reads. Runs
need no loop: start flags, a cumulative maximum for the run start, and cumulative sums for the counts. Hits with equal
times keep their input order here (stock's std::sort may order them either way); only the second cluster's count can
depend on that order, and only when the last two hits of a run have the same time.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def settings(m):
    """SimpleTimeCluster module settings -> kernel scalars; raises for settings the device does not reduce."""
    bad = []
    if m.get("TestFlag", False):
        bad.append("TestFlag true")
    if not m.get("useTimeStep", False) or m.get("useTimeWindow", False):
        bad.append("useTimeStep false or useTimeWindow true")
    if m.get("UseOnlyOnePanel", False) or m.get("UseOnlyOnePlane", False):
        bad.append("UseOnlyOnePanel/Plane")
    if m.get("HitNonNoiseSelectionBits", []):
        bad.append("HitNonNoiseSelectionBits")
    if int(m.get("minNPanels", 0)) > 1:
        bad.append("minNPanels > 1")
    if "maxNStrawHits" in m:
        bad.append("maxNStrawHits")
    if bad:
        raise ValueError("device SimpleTimeCluster does not support " + ", ".join(bad))
    return {"tstep": float(m["TimeStep"]), "twin": float(m["TimeWindow"]), "minnsh": int(m["minNStrawHits"])}


def clusters(ph, s):
    """Panel ComboHits [E, H] -> time clusters [E, 2H] ({"valid", "nsh"}): the run clusters, then the last-hit
    clusters, each in the slot of its run's last hit."""
    v = ph["valid"]
    E, H = v.shape
    order = jnp.argsort(jnp.where(v, ph["time"], jnp.inf), axis=1, stable=True)
    t = jnp.take_along_axis(ph["time"], order, axis=1).astype(jnp.float64)   # float times: exact differences in x64
    vs = jnp.take_along_axis(v, order, axis=1)
    n = jnp.where(vs, jnp.take_along_axis(ph["ncombo"], order, axis=1), 0)
    jpos = jnp.broadcast_to(jnp.arange(H)[None, :], (E, H))
    gap = jnp.concatenate([jnp.full((E, 1), jnp.inf, t.dtype), t[:, 1:] - t[:, :-1]], axis=1)
    start = vs & (gap > s["tstep"])
    nxt_start = jnp.concatenate([start[:, 1:] | ~vs[:, 1:], jnp.ones((E, 1), bool)], axis=1)
    end = vs & nxt_start
    rs = jax.lax.cummax(jnp.where(start, jpos, 0), axis=1)
    cs = jnp.cumsum(n, axis=1)
    count = cs - jnp.take_along_axis(cs - n, rs, axis=1)
    span = t - jnp.take_along_axis(t, rs, axis=1)
    run_ok = end & (span <= s["twin"]) & (count >= s["minnsh"])
    last_ok = end & (jpos > rs) & (n >= s["minnsh"])
    return {"valid": jnp.concatenate([run_ok, last_ok], axis=1), "nsh": jnp.concatenate([count, n], axis=1)}
