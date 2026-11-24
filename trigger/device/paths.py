"""Per-path decisions from the device time clusters and helices, with the menu's filter settings.

TimeClusterFilter and HelixFilter are evaluated as the stock filters do. KalSeedFilter needs the stock seed fit,
which the device does not run (decision D1: a reco response layer): its cuts are applied to the helix quantities
(momentum, |tanDip|, d0, straw hits) and its remaining loss (fit failure, momentum error, chi2) enters as a pass
probability per path measured from the stock filter's pass counts on development shards. Its draw is keyed by the
event number, so a run is reproducible. HelixFilter's phi-dependent prescale (`prescaleUsingD0Phi`) counts its own
invocations; that counter is kept on the host, in event order, exactly as the stock filter does.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _cuts(c, key, default):
    return float(c.get(key, default))


def tc_pass(tc, cfg):
    """TimeClusterFilter: a cluster with minNStrawHits straw hits (and, with requireCaloCluster, a calo cluster: only
    clusters that carry their calo cluster index, `calo`, as the TZClusterFinder port's; other device clusters
    carry none and fail). The plane requirements are not on the device (fail)."""
    if int(cfg.get("minNPlanes", 0)) > 0 or int(cfg.get("minPlaneSpan", 0)) > 0:
        return jnp.zeros(tc["valid"].shape[0], bool)
    ok = tc["valid"] & (tc["nsh"] >= int(cfg["minNStrawHits"]))
    if cfg.get("requireCaloCluster", False):
        if "calo" not in tc:
            return jnp.zeros(tc["valid"].shape[0], bool)
        ok = ok & (tc["calo"] >= 0)
    return jnp.any(ok, axis=1)


def _check_helix(h, cut, helicity):
    """HelixCutsTool::checkHelix for the helices of one helicity (configured=false passes)."""
    if not cut.get("configured", False):
        return h["ok"]
    ok = h["ok"]
    if cut.get("requireCaloCluster", False):
        return jnp.zeros_like(ok)
    if cut.get("doHelicityCheck", False):
        ok &= h["helicity"] == helicity
    g = lambda k, d: _cuts(cut, k, d)
    ok &= (h["nsh"] >= g("minNStrawHits", 0)) & (h["pt"] >= g("minPt", 0)) & \
        (h["chi2xy"] <= g("maxChi2XY", 1e30)) & (h["chi2zphi"] <= g("maxChi2PhiZ", 1e30)) & \
        (h["d0"] <= g("maxD0", 1e30)) & (h["d0"] >= g("minD0", -1e30)) & \
        (jnp.abs(h["lambda"]) <= g("maxAbsLambda", 1e30)) & (jnp.abs(h["lambda"]) >= g("minAbsLambda", 0)) & \
        (h["nloops"] <= g("maxNLoops", 1e30)) & (h["nloops"] >= g("minNLoops", 0)) & \
        (h["p"] >= g("minMomentum", 0)) & (h["p"] <= g("maxMomentum", 1e30)) & (h["hitratio"] >= g("minHitRatio", 0))
    return ok


def helix_pass(h, cfg):
    """Per helix [E, NTC]: passes HelixFilter's helicity-matched cuts (before the D0-phi prescale).

    HelixFilter::checkHelixFromHelicity sends a helix to the cut set whose helicity it has; a cut set records its
    helicity only when it is configured, so the helices of an unconfigured helicity match neither set and fail
    (they do not reach checkHelix's `configured: false` pass). Example: the De helix paths configure only the
    positive helicity, and the finder also fits negative-helicity helices (an RPC e+ going downstream)."""
    pc, nc = cfg["posHelicitySelection"], cfg["negHelicitySelection"]
    pos = _check_helix(h, pc, 1.0) & bool(pc.get("configured", False))
    neg = _check_helix(h, nc, -1.0) & bool(nc.get("configured", False))
    return jnp.where(h["helicity"] > 0, pos, jnp.where(h["helicity"] < 0, neg, False))


SPILL = {"OffSpill": 0, "OnSpill": 1}           # EventWindowMarker::SpillType


def prescale_pass(events, spill, factors):
    """PrescaleEvent: in the event's mode (the EventWindowMarker spill type), pass when the event number is a multiple
    of the mode's factor; a factor <= 0, or a mode the module does not configure, rejects (stateless in stock)."""
    ok = jnp.zeros(events.shape, bool)
    for mode, f in factors.items():
        if f > 0:
            ok = ok | ((spill == SPILL[mode]) & (events % f == 0))
    return ok


def digi_pass(nsd, cfg):
    """DigiFilter on the straw digi count (useStrawDigi only; useCaloDigi is not on the device)."""
    if cfg.get("useCaloDigi", False) or not cfg.get("useStrawDigi", False):
        raise ValueError("device DigiFilter supports useStrawDigi true, useCaloDigi false")
    return (nsd >= int(cfg["minNStrawDigi"])) & (nsd <= int(cfg["maxNStrawDigi"]))


def lumi_stream(reached, run, subrun, cfg):
    """Host: LumiStreamFilter over the events of one job in processing order. It counts the events that reach it,
    passes when the count reaches a multiple of eventFreq (then restarts the count) and, with passFirst, the first
    event of every subrun; the count and the first-event flag restart at each subrun's end. Data are not read."""
    reached = np.asarray(reached, bool)
    n = reached.size
    out = np.zeros(n, bool)
    if n == 0:
        return out
    freq, first = int(cfg.get("eventFreq", 0)), bool(cfg.get("passFirst", False))
    if int(cfg.get("duplications", 0)) != 0:
        raise ValueError("device LumiStreamFilter supports duplications 0")
    run, subrun = np.asarray(run), np.asarray(subrun)
    newsub = np.ones(n, bool)
    newsub[1:] = (run[1:] != run[:-1]) | (subrun[1:] != subrun[:-1])
    seg = np.cumsum(newsub) - 1
    count = np.zeros(n, np.int64)                  # events that reached the filter in the subrun, up to this one
    for s in np.unique(seg):                       # a few subruns per job
        m = seg == s
        count[m] = np.cumsum(reached[m])
    if freq > 0:                                   # the count restarts at every pass: a pass at each multiple
        out |= reached & (count % freq == 0)
    if first:
        firsts = reached & (count == 1)
        out |= firsts
    return out


def draw(events, salt):
    """Reproducible uniform draw per event number."""
    key = jax.random.PRNGKey(salt)
    return jax.vmap(lambda e: jax.random.uniform(jax.random.fold_in(key, e)))(events)


# (helicity, fit direction) -> fit particle: a positive-helicity helix is a downstream e- or an upstream e+
PARTICLE = {(1.0, "downstream"): 11, (-1.0, "downstream"): -11, (1.0, "upstream"): -11, (-1.0, "upstream"): 11}


def kalseed_pass(h, cfg, helix_mask_mom, eff, events, salt):
    """Response layer for KalSeedFilter: each cut set (fit particle, direction) on the helix quantities of the
    helices that hypothesis describes, times a per-path pass probability `eff` drawn per event."""
    out = jnp.zeros(h["ok"].shape, bool)
    for c in cfg["KalSeedCuts"]:
        dirn = c.get("fitdirection", "downstream")
        hels = [hel for (hel, d), part in PARTICLE.items() if d == dirn and part == int(c["fitparticle"])]
        if not hels:
            continue
        sign = 1.0 if dirn == "downstream" else -1.0
        ok = h["ok"] & (h["helicity"] == hels[0]) & (h["p"] >= helix_mask_mom)
        ok &= (h["p"] > float(c["minMomentum"])) & (h["p"] < float(c["maxMomentum"]))
        td = sign * h["tandip"]
        ok &= (td > float(c["minTanDip"])) & (td < float(c["maxTanDip"]))
        ok &= (h["d0"] > float(c["minD0"])) & (h["d0"] < float(c["maxD0"]))
        ok &= h["nsh"] >= int(c["minNStrawHits"])
        out = out | ok
    hit = jnp.any(out, axis=1)
    return hit & (draw(events, salt) < eff) if eff < 1.0 else hit     # a uniform draw in [0, 1) is always < 1


def ipa_prescale(n_calls, fcent, par):
    """HelixFilter::evalIPAPresc and the `_nevt % prescaler` test (host; n_calls counts filter invocations)."""
    presc = np.trunc(par["amplitude"] - (par["amplitude"] - 1) * np.sin(par["frequency"] * fcent + par["phase"])).astype(np.int64)
    return (n_calls[:, None] % np.maximum(presc, 1)) == 0
