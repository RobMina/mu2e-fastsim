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
    if cfg.get("requireCaloCluster", False) or int(cfg.get("minNPlanes", 0)) > 0 or int(cfg.get("minPlaneSpan", 0)) > 0:
        return jnp.zeros(tc["valid"].shape[0], bool)      # device clusters carry no calorimeter cluster / plane cuts
    return jnp.any(tc["valid"] & (tc["nsh"] >= int(cfg["minNStrawHits"])), axis=1)


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
    """Per helix [E, NTC]: passes HelixFilter's helicity-matched cuts (before the D0-phi prescale)."""
    pos = _check_helix(h, cfg["posHelicitySelection"], 1.0)
    neg = _check_helix(h, cfg["negHelicitySelection"], -1.0)
    return jnp.where(h["helicity"] > 0, pos, jnp.where(h["helicity"] < 0, neg, False))


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
    return jnp.any(out, axis=1) & (draw(events, salt) < eff)


def ipa_prescale(n_calls, fcent, par):
    """HelixFilter::evalIPAPresc and the `_nevt % prescaler` test (host; n_calls counts filter invocations)."""
    presc = np.trunc(par["amplitude"] - (par["amplitude"] - 1) * np.sin(par["frequency"] * fcent + par["phase"])).astype(np.int64)
    return (n_calls[:, None] % np.maximum(presc, 1)) == 0
